from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from fusion import weighted_score_fusion
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from query_expansion import build_source_aware_query_packs
from query_preprocess import preprocess_query
from retrieval_dense import DenseRetriever
from retrieval_rules import RuleCitationRetriever
from retrieval_sparse import SparseRetriever


KNOWN_FAMILIES = {"ZGB", "OR", "BGG", "STPO", "BV", "ZPO", "SVG", "VVG", "DBG", "MWSTG"}


def load_lookup(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["norm_citation"]] = row["canonical_citation"]
    return out


def dedup_keep_max(score_items):
    best = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def recall_at_k(gold, ranked, k):
    if not gold:
        return 0.0
    return len(gold.intersection(set(ranked[:k]))) / len(gold)


def search_field_map(sparse: SparseRetriever, source: str, field: str, query: str, top_k: int) -> dict[str, float]:
    if not query:
        return {}
    idx = sparse.field_indices[source][field]
    docs = sparse.field_docs[source][field]
    out = {}
    for doc_id, score in idx.search(query, top_k=top_k):
        c = docs[doc_id]["citation"]
        s = float(score)
        if c not in out or s > out[c]:
            out[c] = s
    return out


def parse_df_label_to_ratio(df_label: str) -> float:
    return {"top_1pct": 0.99, "top_3pct": 0.97, "top_5pct": 0.95, "top_10pct": 0.90}.get(df_label, 0.95)


def quantile_threshold(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    arr = sorted(values)
    pos = min(max(int(math.ceil(q * len(arr))) - 1, 0), len(arr) - 1)
    return arr[pos]


def load_pattern_candidates(path: Path) -> tuple[list[str], list[re.Pattern]]:
    exact, rgx = [], []
    if not path.exists():
        return exact, rgx
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("title_template_or_pattern") or "").strip()
            ptype = (row.get("suggested_penalty_type") or "").strip()
            if not t:
                continue
            if ptype == "regex_title_penalty":
                try:
                    rgx.append(re.compile(t))
                except re.error:
                    continue
            else:
                exact.append(t)
    return exact, rgx


def is_pattern_hit(template: str, exact_templates: list[str], regex_patterns: list[re.Pattern]) -> bool:
    if template in exact_templates:
        return True
    for rgx in regex_patterns:
        if rgx.search(template):
            return True
    return False


def extract_query_families(*texts: str) -> set[str]:
    fams = set()
    for t in texts:
        for tok in re.findall(r"\b[A-Za-z]{2,8}\b", t or ""):
            up = tok.upper()
            if up in KNOWN_FAMILIES:
                fams.add(up)
    return fams


def build_query_signal(mv: dict, laws_pack_v2: dict, laws_pack: dict, sparse: SparseRetriever) -> dict:
    query_patterns = [str(x) for x in mv.get("query_number_patterns", [])]
    query_tokens = sparse.extract_citation_tokens(
        " ".join(
            [
                mv.get("query_original", ""),
                mv.get("query_clean", ""),
                " ".join(mv.get("query_keywords", [])),
                " ".join(query_patterns),
                " ".join(laws_pack_v2.get("expanded_keywords_de", [])),
                laws_pack_v2.get("expanded_query_de", ""),
                " ".join(laws_pack.get("expanded_keywords_de", [])),
            ]
        )
    )
    query_families = extract_query_families(
        mv.get("query_original", ""),
        mv.get("query_clean", ""),
        " ".join(query_patterns),
        laws_pack_v2.get("expanded_query_de", ""),
        laws_pack.get("expanded_query_de", ""),
    )
    return {
        "query_patterns": query_patterns,
        "query_tokens": query_tokens,
        "query_families": query_families,
    }


def compute_pattern_match_score(query_patterns: list[str], citation: str) -> float:
    if not query_patterns:
        return 0.0
    c = (citation or "").lower()
    hits = 0
    for p in query_patterns:
        pl = p.lower().strip()
        if not pl:
            continue
        if pl in c:
            hits += 1
        else:
            # 对“Art. / Abs.”类结构词做宽松匹配。
            if any(k in pl for k in ["art", "abs", "lit", "ziff", "para", "section"]):
                if any(k in c for k in ["art", "abs", "lit", "ziff", "para", "section"]):
                    hits += 1
    return hits / max(len(query_patterns), 1)


def combine_sparse_with_penalty_and_boost(
    *,
    sparse: SparseRetriever,
    laws_citation_raw: dict[str, float],
    laws_title_raw: dict[str, float],
    laws_text_raw: dict[str, float],
    court_citation_raw: dict[str, float],
    court_text_raw: dict[str, float],
    laws_citation_weight: float,
    laws_title_weight: float,
    laws_text_weight: float,
    court_citation_weight: float,
    court_text_weight: float,
    # penalty config
    enable_high_df_title_penalty: bool,
    title_df_threshold_value: float,
    title_dominance_threshold: float,
    low_citation_threshold: float,
    high_df_title_penalty_strength: float,
    generic_title_pattern_mode: str,
    title_df_ratio_by_citation: dict[str, float],
    title_template_by_citation: dict[str, str],
    exact_templates: list[str],
    regex_patterns: list[re.Pattern],
    # citation boost config
    enable_citation_aware_boost: bool,
    citation_pattern_match_boost: float,
    statute_family_match_boost: float,
    citation_token_overlap_boost: float,
    citation_aware_boost_mode: str,
    query_signal: dict,
) -> tuple[list[tuple[str, float]], list[str]]:
    agg = {}
    laws_component = {}

    def _upsert(c, s):
        if c not in agg or s > agg[c]:
            agg[c] = s

    for c, s in laws_citation_raw.items():
        laws_component.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component[c]["citation"] = s * laws_citation_weight
    for c, s in laws_title_raw.items():
        laws_component.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component[c]["title"] = s * laws_title_weight
    for c, s in laws_text_raw.items():
        laws_component.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component[c]["text"] = s * laws_text_weight

    q_patterns = query_signal["query_patterns"]
    q_tokens = query_signal["query_tokens"]
    q_families = query_signal["query_families"]

    for c, comp in laws_component.items():
        csc, tsc, xsc = comp["citation"], comp["title"], comp["text"]
        total = csc + tsc + xsc + 1e-9
        title_share = tsc / total
        citation_share = csc / total

        # -------- penalty --------
        df_ratio = title_df_ratio_by_citation.get(c, 0.0)
        template = title_template_by_citation.get(c, "")
        pattern_hit = is_pattern_hit(template, exact_templates, regex_patterns)
        high_df = df_ratio >= title_df_threshold_value
        dom_ok = title_share >= title_dominance_threshold
        low_cit_ok = citation_share <= low_citation_threshold
        apply_penalty = False
        if enable_high_df_title_penalty:
            if generic_title_pattern_mode == "df_only":
                apply_penalty = high_df
            elif generic_title_pattern_mode == "df_plus_dominance":
                apply_penalty = high_df and dom_ok and low_cit_ok
            elif generic_title_pattern_mode == "df_plus_dominance_plus_pattern":
                apply_penalty = high_df and dom_ok and low_cit_ok and pattern_hit
            else:
                raise ValueError(f"unsupported generic_title_pattern_mode={generic_title_pattern_mode}")
        if apply_penalty and tsc > 0:
            comp["title"] = tsc * max(0.0, 1.0 - high_df_title_penalty_strength)

        # -------- citation-aware boost --------
        if enable_citation_aware_boost:
            citation_text = c
            cand_family = sparse.extract_statute_family(citation_text)
            cand_tokens = sparse.extract_citation_tokens(citation_text)

            pattern_score = compute_pattern_match_score(q_patterns, citation_text)
            family_match = bool(cand_family and cand_family in q_families)
            overlap = len(q_tokens.intersection(cand_tokens)) / max(len(cand_tokens), 1)

            boost_factor = 1.0
            if citation_aware_boost_mode == "pattern_only":
                boost_factor += citation_pattern_match_boost * pattern_score
            elif citation_aware_boost_mode == "pattern_plus_family":
                boost_factor += citation_pattern_match_boost * pattern_score
                if family_match:
                    boost_factor += statute_family_match_boost
            elif citation_aware_boost_mode == "pattern_plus_family_plus_overlap":
                boost_factor += citation_pattern_match_boost * pattern_score
                if family_match:
                    boost_factor += statute_family_match_boost
                boost_factor += citation_token_overlap_boost * overlap
            else:
                raise ValueError(f"unsupported citation_aware_boost_mode={citation_aware_boost_mode}")

            if boost_factor > 1.0:
                comp["citation"] = comp["citation"] * boost_factor

        _upsert(c, max(comp["citation"], comp["title"], comp["text"]))

    for c, s in court_citation_raw.items():
        _upsert(c, s * court_citation_weight)
    for c, s in court_text_raw.items():
        _upsert(c, s * court_text_weight)

    sparse_scores = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    laws_candidates = set(laws_component.keys())
    sparse_laws_ranked = [c for c, _ in sparse_scores if c in laws_candidates]
    return sparse_scores, sparse_laws_ranked


def mean_metric(rows, key):
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description="Sparse laws citation-aware boost eval on top of best penalty config.")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--sparse-max-court", type=int, default=120000)
    parser.add_argument("--dense-max-laws", type=int, default=60000)
    parser.add_argument("--dense-max-court", type=int, default=80000)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--laws-weight", type=float, default=1.15)
    parser.add_argument("--court-weight", type=float, default=0.95)
    parser.add_argument("--court-citation-weight", type=float, default=1.25)
    parser.add_argument("--court-text-weight", type=float, default=1.0)

    parser.add_argument("--enable-citation-aware-boost", action="store_true")
    parser.add_argument("--citation-pattern-match-boost", type=float, default=0.1)
    parser.add_argument("--statute-family-match-boost", type=float, default=0.1)
    parser.add_argument("--citation-token-overlap-boost", type=float, default=0.1)
    parser.add_argument(
        "--citation-aware-boost-mode",
        choices=["pattern_only", "pattern_plus_family", "pattern_plus_family_plus_overlap"],
        default="pattern_plus_family",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts" / "v1_3_sparse_laws_citation_boost"
    )
    args = parser.parse_args()

    # 读取上游产物（按需求）
    blockers_path = ROOT / "artifacts" / "v1_3_sparse_laws_penalty" / "post_penalty_blockers.csv"
    blocker_summary_path = ROOT / "artifacts" / "v1_3_sparse_laws_penalty" / "post_penalty_blocker_summary_cn.md"
    _ = blockers_path.exists(), blocker_summary_path.exists()

    val_rows = load_query_split("val")
    lookup = load_lookup(ROOT / "artifacts" / "phase0" / "citation_lookup.csv")

    # 固定最优 sparse 权重
    best_weight_cfg = json.loads(
        (ROOT / "artifacts" / "v1_3_sparse_laws_tuning" / "best_sparse_laws_weight_config.json").read_text(
            encoding="utf-8-sig"
        )
    )["best_config"]
    laws_citation_weight = float(best_weight_cfg["laws_citation_weight"])
    laws_title_weight = float(best_weight_cfg["laws_title_weight"])
    laws_text_weight = float(best_weight_cfg["laws_text_weight"])

    # 固定最优 penalty 配置
    best_penalty_payload = json.loads(
        (ROOT / "artifacts" / "v1_3_sparse_laws_penalty" / "best_penalty_config.json").read_text(
            encoding="utf-8-sig"
        )
    )
    best_penalty = best_penalty_payload["best_penalty"]

    exact_templates, regex_patterns = load_pattern_candidates(
        ROOT / "artifacts" / "v1_3_sparse_laws_tuning" / "generic_title_penalty_candidates.csv"
    )

    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(
        max_laws_rows=args.sparse_max_laws,
        max_court_rows=args.sparse_max_court,
        enable_field_aware=True,
    )
    dense = DenseRetriever(use_sbert=not args.dense_disable_sbert, text_max_chars=500, svd_dim=256)
    dense.build(
        max_laws_rows=args.dense_max_laws,
        max_court_rows=args.dense_max_court,
        enable_field_aware=True,
    )
    rule = RuleCitationRetriever()
    rule.build(max_laws_rows=args.sparse_max_laws, max_court_rows=args.sparse_max_court)

    title_df_ratio_by_citation = {}
    title_template_by_citation = {}
    for c, t in sparse.laws_title_template_by_citation.items():
        title_template_by_citation[c] = t
        title_df_ratio_by_citation[c] = sparse.laws_title_df_ratio.get(t, 0.0)
    df_values = list(title_df_ratio_by_citation.values())
    title_df_threshold_value = quantile_threshold(df_values, parse_df_label_to_ratio(best_penalty["title_df_threshold"]))

    doc_source = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])

    # 预计算 query 侧信号与 raw 分数
    per_query = []
    for row in val_rows:
        q = row["query"]
        qid = row["query_id"]
        mv = preprocess_query(q)
        packs = build_source_aware_query_packs(mv)
        laws_pack_v2 = packs.get("laws_query_pack_v2", packs["laws_query_pack"])
        laws_pack = packs["laws_query_pack"]

        laws_query = " ".join(
            [
                laws_pack_v2.get("query_original", ""),
                " ".join(laws_pack_v2.get("query_keywords", [])[:16]),
                laws_pack_v2.get("expanded_query_de", ""),
            ]
        ).strip()
        court_query = " ".join(
            [
                packs["court_query_pack"].get("query_original", ""),
                " ".join(packs["court_query_pack"].get("query_keywords", [])[:16]),
                packs["court_query_pack"].get("expanded_query_de", ""),
            ]
        ).strip()

        laws_citation_raw = search_field_map(sparse, "laws_de", "citation", laws_query, top_k=300)
        laws_title_raw = search_field_map(sparse, "laws_de", "title", laws_query, top_k=300)
        laws_text_raw = search_field_map(sparse, "laws_de", "text", laws_query, top_k=300)
        court_citation_raw = search_field_map(sparse, "court_considerations", "citation", court_query, top_k=300)
        court_text_raw = search_field_map(sparse, "court_considerations", "text", court_query, top_k=300)

        dense_items = dense.search_source_aware(
            laws_query_pack=packs["laws_query_pack"],
            court_query_pack=packs["court_query_pack"],
            top_k_laws=200,
            top_k_court=200,
        )
        dense_scores = dedup_keep_max([(x.citation, x.score) for x in dense_items])
        rule_items = rule.search(q, top_k_laws=200, top_k_court=200)
        rule_scores = dedup_keep_max([(x.citation, x.score) for x in rule_items])

        gold_norm = set(normalize_citation(x) for x in row["gold_citations"].split(";") if normalize_citation(x))
        if lookup:
            gold_norm = set(x for x in gold_norm if x in lookup)

        query_signal = build_query_signal(mv, laws_pack_v2, laws_pack, sparse)
        per_query.append(
            {
                "query_id": qid,
                "gold_in_corpus": gold_norm,
                "laws_citation_raw": laws_citation_raw,
                "laws_title_raw": laws_title_raw,
                "laws_text_raw": laws_text_raw,
                "court_citation_raw": court_citation_raw,
                "court_text_raw": court_text_raw,
                "dense_scores": dense_scores,
                "rule_scores": rule_scores,
                "query_signal": query_signal,
            }
        )

    # 网格：baseline + boost grid
    mode_grid = ["pattern_only", "pattern_plus_family", "pattern_plus_family_plus_overlap"]
    pattern_boost_grid = [0.05, 0.1, 0.15, 0.2]
    family_boost_grid = [0.05, 0.1, 0.15]
    overlap_boost_grid = [0.05, 0.1, 0.15]

    cfgs = [
        {
            "enable_citation_aware_boost": False,
            "citation_aware_boost_mode": "pattern_only",
            "citation_pattern_match_boost": 0.0,
            "statute_family_match_boost": 0.0,
            "citation_token_overlap_boost": 0.0,
        }
    ]
    for m, pb, fb, ob in itertools.product(mode_grid, pattern_boost_grid, family_boost_grid, overlap_boost_grid):
        cfgs.append(
            {
                "enable_citation_aware_boost": True,
                "citation_aware_boost_mode": m,
                "citation_pattern_match_boost": pb,
                "statute_family_match_boost": fb,
                "citation_token_overlap_boost": ob,
            }
        )

    runs = []
    for cfg in cfgs:
        run_name = (
            "baseline_no_citation_boost"
            if not cfg["enable_citation_aware_boost"]
            else f"boost_{cfg['citation_aware_boost_mode']}_pb{cfg['citation_pattern_match_boost']}_"
            f"fb{cfg['statute_family_match_boost']}_ob{cfg['citation_token_overlap_boost']}"
        )
        pred_map = {}
        qrows = []
        for q in per_query:
            sparse_scores, sparse_laws_ranked = combine_sparse_with_penalty_and_boost(
                sparse=sparse,
                laws_citation_raw=q["laws_citation_raw"],
                laws_title_raw=q["laws_title_raw"],
                laws_text_raw=q["laws_text_raw"],
                court_citation_raw=q["court_citation_raw"],
                court_text_raw=q["court_text_raw"],
                laws_citation_weight=laws_citation_weight,
                laws_title_weight=laws_title_weight,
                laws_text_weight=laws_text_weight,
                court_citation_weight=args.court_citation_weight,
                court_text_weight=args.court_text_weight,
                enable_high_df_title_penalty=True,
                title_df_threshold_value=title_df_threshold_value,
                title_dominance_threshold=float(best_penalty["title_dominance_threshold"]),
                low_citation_threshold=float(best_penalty["low_citation_threshold"]),
                high_df_title_penalty_strength=float(best_penalty["high_df_title_penalty_strength"]),
                generic_title_pattern_mode=best_penalty["generic_title_pattern_mode"],
                title_df_ratio_by_citation=title_df_ratio_by_citation,
                title_template_by_citation=title_template_by_citation,
                exact_templates=exact_templates,
                regex_patterns=regex_patterns,
                enable_citation_aware_boost=cfg["enable_citation_aware_boost"],
                citation_pattern_match_boost=cfg["citation_pattern_match_boost"],
                statute_family_match_boost=cfg["statute_family_match_boost"],
                citation_token_overlap_boost=cfg["citation_token_overlap_boost"],
                citation_aware_boost_mode=cfg["citation_aware_boost_mode"],
                query_signal=q["query_signal"],
            )

            fused = weighted_score_fusion(
                score_lists=[sparse_scores, q["dense_scores"], q["rule_scores"]],
                weights=[0.6, 0.4, 0.35],
                top_n=500,
                citation_to_source=doc_source,
                source_aware_fusion=True,
                laws_weight=args.laws_weight,
                court_weight=args.court_weight,
            )
            fusion_ranked = [c for c, _ in fused]
            pred_map[q["query_id"]] = fusion_ranked[: args.top_n]
            qrows.append(
                {
                    "sparse_r100": recall_at_k(q["gold_in_corpus"], sparse_laws_ranked, 100),
                    "sparse_r200": recall_at_k(q["gold_in_corpus"], sparse_laws_ranked, 200),
                    "fusion_r100": recall_at_k(q["gold_in_corpus"], fusion_ranked, 100),
                    "fusion_r200": recall_at_k(q["gold_in_corpus"], fusion_ranked, 200),
                }
            )

        strict, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="strict")
        corpus, _ = evaluate_predictions(
            val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="corpus_aware"
        )
        runs.append(
            {
                "run_name": run_name,
                "citation_aware_boost_mode": cfg["citation_aware_boost_mode"],
                "citation_pattern_match_boost": cfg["citation_pattern_match_boost"],
                "statute_family_match_boost": cfg["statute_family_match_boost"],
                "citation_token_overlap_boost": cfg["citation_token_overlap_boost"],
                "sparse_laws_Recall@100": round(mean_metric(qrows, "sparse_r100"), 6),
                "sparse_laws_Recall@200": round(mean_metric(qrows, "sparse_r200"), 6),
                "fusion_final_Recall@100": round(mean_metric(qrows, "fusion_r100"), 6),
                "fusion_final_Recall@200": round(mean_metric(qrows, "fusion_r200"), 6),
                "strict_macro_f1": round(strict["macro_f1"], 6),
                "corpus_aware_macro_f1": round(corpus["macro_f1"], 6),
            }
        )

    baseline = next(x for x in runs if x["run_name"] == "baseline_no_citation_boost")
    best = sorted(
        [x for x in runs if x["run_name"] != "baseline_no_citation_boost"],
        key=lambda x: (
            x["corpus_aware_macro_f1"],
            x["strict_macro_f1"],
            x["fusion_final_Recall@200"],
            x["fusion_final_Recall@100"],
            x["sparse_laws_Recall@200"],
        ),
        reverse=True,
    )[0]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "citation_boost_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = [
            "run_name",
            "citation_aware_boost_mode",
            "citation_pattern_match_boost",
            "statute_family_match_boost",
            "citation_token_overlap_boost",
            "sparse_laws_Recall@100",
            "sparse_laws_Recall@200",
            "fusion_final_Recall@100",
            "fusion_final_Recall@200",
            "strict_macro_f1",
            "corpus_aware_macro_f1",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(runs)

    delta = {
        "sparse_laws_Recall@100": round(best["sparse_laws_Recall@100"] - baseline["sparse_laws_Recall@100"], 6),
        "sparse_laws_Recall@200": round(best["sparse_laws_Recall@200"] - baseline["sparse_laws_Recall@200"], 6),
        "fusion_final_Recall@100": round(best["fusion_final_Recall@100"] - baseline["fusion_final_Recall@100"], 6),
        "fusion_final_Recall@200": round(best["fusion_final_Recall@200"] - baseline["fusion_final_Recall@200"], 6),
        "strict_macro_f1": round(best["strict_macro_f1"] - baseline["strict_macro_f1"], 6),
        "corpus_aware_macro_f1": round(best["corpus_aware_macro_f1"] - baseline["corpus_aware_macro_f1"], 6),
    }
    (out_dir / "best_citation_boost_config.json").write_text(
        json.dumps({"baseline": baseline, "best_citation_boost": best, "delta_vs_baseline": delta}, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    lines = []
    lines.append("# sparse_laws citation-aware boost 实验总结（v1_3）")
    lines.append("")
    lines.append("## 最优配置")
    lines.append(
        f"- mode={best['citation_aware_boost_mode']}, "
        f"citation_pattern_match_boost={best['citation_pattern_match_boost']}, "
        f"statute_family_match_boost={best['statute_family_match_boost']}, "
        f"citation_token_overlap_boost={best['citation_token_overlap_boost']}"
    )
    lines.append("")
    lines.append("## 必答问题")
    lines.append(
        f"1. 是否第一次让 fusion_final Recall@100/200 提升："
        f"{'是' if (delta['fusion_final_Recall@100'] > 0 or delta['fusion_final_Recall@200'] > 0) else '否'} "
        f"(delta@100={delta['fusion_final_Recall@100']:.4f}, delta@200={delta['fusion_final_Recall@200']:.4f})。"
    )
    lines.append(
        f"2. 是否第一次让 strict/corpus_aware F1 提升："
        f"{'是' if (delta['strict_macro_f1'] > 0 or delta['corpus_aware_macro_f1'] > 0) else '否'} "
        f"(strict_delta={delta['strict_macro_f1']:.4f}, corpus_delta={delta['corpus_aware_macro_f1']:.4f})。"
    )
    lines.append(f"3. 最优模式：{best['citation_aware_boost_mode']}。")
    if delta["fusion_final_Recall@200"] <= 0 and delta["corpus_aware_macro_f1"] <= 0:
        lines.append("4. 若仍无提升，下一步优先：statute family disambiguation templates。原因：已有 family boost 仍难区分同家族条款混淆。")
    else:
        lines.append("4. 若已提升，下一步优先：title_score_normalization。原因：可进一步稳定 title 与 citation 的相对权重。")
    lines.append("")
    lines.append("## 指标概览")
    lines.append(
        f"- sparse_laws Recall@100/200: baseline={baseline['sparse_laws_Recall@100']:.4f}/{baseline['sparse_laws_Recall@200']:.4f}, "
        f"best={best['sparse_laws_Recall@100']:.4f}/{best['sparse_laws_Recall@200']:.4f}"
    )
    lines.append(
        f"- fusion_final Recall@100/200: baseline={baseline['fusion_final_Recall@100']:.4f}/{baseline['fusion_final_Recall@200']:.4f}, "
        f"best={best['fusion_final_Recall@100']:.4f}/{best['fusion_final_Recall@200']:.4f}"
    )
    lines.append(
        f"- strict/corpus_aware F1: baseline={baseline['strict_macro_f1']:.4f}/{baseline['corpus_aware_macro_f1']:.4f}, "
        f"best={best['strict_macro_f1']:.4f}/{best['corpus_aware_macro_f1']:.4f}"
    )
    (out_dir / "citation_boost_summary_cn.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print(json.dumps({"rows": len(runs), "best_run": best["run_name"], "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

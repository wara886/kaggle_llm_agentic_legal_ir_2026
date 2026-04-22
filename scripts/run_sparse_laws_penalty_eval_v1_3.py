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
    if df_label == "top_1pct":
        return 0.99
    if df_label == "top_3pct":
        return 0.97
    if df_label == "top_5pct":
        return 0.95
    if df_label == "top_10pct":
        return 0.90
    raise ValueError(f"unsupported df label: {df_label}")


def quantile_threshold(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    arr = sorted(values)
    pos = min(max(int(math.ceil(q * len(arr))) - 1, 0), len(arr) - 1)
    return arr[pos]


def load_pattern_candidates(path: Path) -> tuple[list[str], list[re.Pattern]]:
    exact_templates: list[str] = []
    regex_patterns: list[re.Pattern] = []
    if not path.exists():
        return exact_templates, regex_patterns
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("title_template_or_pattern") or "").strip()
            ptype = (row.get("suggested_penalty_type") or "").strip()
            if not t:
                continue
            if ptype == "regex_title_penalty":
                try:
                    regex_patterns.append(re.compile(t))
                except re.error:
                    continue
            else:
                exact_templates.append(t)
    return exact_templates, regex_patterns


def is_pattern_hit(template: str, exact_templates: list[str], regex_patterns: list[re.Pattern]) -> bool:
    if template in exact_templates:
        return True
    for rgx in regex_patterns:
        if rgx.search(template):
            return True
    return False


def combine_sparse_scores_with_penalty(
    *,
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
) -> tuple[list[tuple[str, float]], list[str], int, int]:
    """
    仅对 sparse_laws 的 title 分量做条件惩罚（high_df_title_penalty）。
    返回：
    - sparse_scores: 稀疏总分（laws + court）
    - sparse_laws_ranked: laws 侧排名
    - penalty_triggered_count: 触发次数
    - generic_hijack_top20_count: top20 中仍满足劫持特征数量（用于诊断）
    """
    agg = {}
    laws_component = {}

    def _upsert(c, s):
        if c not in agg or s > agg[c]:
            agg[c] = s

    for c, s in laws_citation_raw.items():
        ws = s * laws_citation_weight
        laws_component.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component[c]["citation"] = ws
    for c, s in laws_title_raw.items():
        ws = s * laws_title_weight
        laws_component.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component[c]["title"] = ws
    for c, s in laws_text_raw.items():
        ws = s * laws_text_weight
        laws_component.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component[c]["text"] = ws

    penalty_triggered_count = 0
    for c, comp in laws_component.items():
        csc = comp["citation"]
        tsc = comp["title"]
        xsc = comp["text"]
        total = csc + tsc + xsc + 1e-9
        title_share = tsc / total
        citation_share = csc / total

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
            comp["title"] = tsc * max(0.0, (1.0 - high_df_title_penalty_strength))
            penalty_triggered_count += 1

        _upsert(c, max(comp["citation"], comp["title"], comp["text"]))

    # court 保持不变
    for c, s in court_citation_raw.items():
        _upsert(c, s * court_citation_weight)
    for c, s in court_text_raw.items():
        _upsert(c, s * court_text_weight)

    sparse_scores = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    laws_candidates = set(laws_component.keys())
    sparse_laws_ranked = [c for c, _ in sparse_scores if c in laws_candidates]

    # 诊断：top20 中仍符合“泛标题劫持特征”的数量
    generic_hijack_top20_count = 0
    top20 = sparse_laws_ranked[:20]
    for c in top20:
        comp = laws_component[c]
        total = comp["citation"] + comp["title"] + comp["text"] + 1e-9
        title_share = comp["title"] / total
        citation_share = comp["citation"] / total
        df_ratio = title_df_ratio_by_citation.get(c, 0.0)
        template = title_template_by_citation.get(c, "")
        pattern_hit = is_pattern_hit(template, exact_templates, regex_patterns)
        high_df = df_ratio >= title_df_threshold_value
        dom_ok = title_share >= title_dominance_threshold
        low_cit_ok = citation_share <= low_citation_threshold
        if generic_title_pattern_mode == "df_only":
            hijack = high_df
        elif generic_title_pattern_mode == "df_plus_dominance":
            hijack = high_df and dom_ok and low_cit_ok
        else:
            hijack = high_df and dom_ok and low_cit_ok and pattern_hit
        if hijack:
            generic_hijack_top20_count += 1
    return sparse_scores, sparse_laws_ranked, penalty_triggered_count, generic_hijack_top20_count


def mean_metric(rows, key):
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate high_df_title_penalty only on sparse_laws title component.")
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

    parser.add_argument("--enable-high-df-title-penalty", action="store_true")
    parser.add_argument("--title-df-threshold", type=str, default="top_5pct")
    parser.add_argument("--title-dominance-threshold", type=float, default=0.7)
    parser.add_argument("--low-citation-threshold", type=float, default=0.05)
    parser.add_argument("--high-df-title-penalty-strength", type=float, default=0.1)
    parser.add_argument(
        "--generic-title-pattern-mode",
        choices=["df_only", "df_plus_dominance", "df_plus_dominance_plus_pattern"],
        default="df_plus_dominance_plus_pattern",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts" / "v1_3_sparse_laws_penalty"
    )
    args = parser.parse_args()

    val_rows = load_query_split("val")
    lookup = load_lookup(ROOT / "artifacts" / "phase0" / "citation_lookup.csv")

    best_cfg_path = ROOT / "artifacts" / "v1_3_sparse_laws_tuning" / "best_sparse_laws_weight_config.json"
    best_payload = json.loads(best_cfg_path.read_text(encoding="utf-8-sig"))
    best_cfg = best_payload["best_config"]
    laws_citation_weight = float(best_cfg["laws_citation_weight"])
    laws_title_weight = float(best_cfg["laws_title_weight"])
    laws_text_weight = float(best_cfg["laws_text_weight"])

    penalty_candidates_path = ROOT / "artifacts" / "v1_3_sparse_laws_tuning" / "generic_title_penalty_candidates.csv"
    exact_templates, regex_patterns = load_pattern_candidates(penalty_candidates_path)

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

    # laws 标题 DF 特征
    title_df_ratio_by_citation = {}
    title_template_by_citation = {}
    for c, t in sparse.laws_title_template_by_citation.items():
        title_template_by_citation[c] = t
        title_df_ratio_by_citation[c] = sparse.laws_title_df_ratio.get(t, 0.0)
    df_values = list(title_df_ratio_by_citation.values())

    doc_source = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])

    # 固定 retrieval 输入，预计算字段 raw score。
    per_query = []
    for row in val_rows:
        q = row["query"]
        qid = row["query_id"]
        mv = preprocess_query(q)
        packs = build_source_aware_query_packs(mv)
        laws_pack_v2 = packs.get("laws_query_pack_v2", packs["laws_query_pack"])

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
            }
        )

    # 网格
    df_grid = ["top_1pct", "top_3pct", "top_5pct", "top_10pct"]
    dom_grid = [0.5, 0.6, 0.7, 0.8]
    low_cit_grid = [0.0, 0.05, 0.1]
    strength_grid = [0.05, 0.1, 0.15, 0.2]
    mode_grid = ["df_only", "df_plus_dominance", "df_plus_dominance_plus_pattern"]

    runs = []
    # baseline (no penalty)
    no_penalty_cfg = {
        "title_df_threshold": "top_5pct",
        "title_dominance_threshold": 0.7,
        "low_citation_threshold": 0.05,
        "high_df_title_penalty_strength": 0.0,
        "generic_title_pattern_mode": "df_only",
        "enable_high_df_title_penalty": False,
    }
    all_cfgs = [no_penalty_cfg]
    for a, b, c, d, e in itertools.product(df_grid, dom_grid, low_cit_grid, strength_grid, mode_grid):
        all_cfgs.append(
            {
                "title_df_threshold": a,
                "title_dominance_threshold": b,
                "low_citation_threshold": c,
                "high_df_title_penalty_strength": d,
                "generic_title_pattern_mode": e,
                "enable_high_df_title_penalty": True,
            }
        )

    for cfg in all_cfgs:
        run_name = (
            "baseline_no_penalty"
            if not cfg["enable_high_df_title_penalty"]
            else f"pen_{cfg['title_df_threshold']}_dom{cfg['title_dominance_threshold']}_"
            f"lc{cfg['low_citation_threshold']}_ps{cfg['high_df_title_penalty_strength']}_{cfg['generic_title_pattern_mode']}"
        )
        pred_map = {}
        rows = []
        total_penalty_triggered = 0
        total_hijack_top20 = 0
        for q in per_query:
            q_df_q = parse_df_label_to_ratio(cfg["title_df_threshold"])
            q_df_threshold = quantile_threshold(df_values, q_df_q)
            sparse_scores, sparse_laws_ranked, penalty_triggered_count, hijack_top20 = combine_sparse_scores_with_penalty(
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
                enable_high_df_title_penalty=cfg["enable_high_df_title_penalty"],
                title_df_threshold_value=q_df_threshold,
                title_dominance_threshold=cfg["title_dominance_threshold"],
                low_citation_threshold=cfg["low_citation_threshold"],
                high_df_title_penalty_strength=cfg["high_df_title_penalty_strength"],
                generic_title_pattern_mode=cfg["generic_title_pattern_mode"],
                title_df_ratio_by_citation=title_df_ratio_by_citation,
                title_template_by_citation=title_template_by_citation,
                exact_templates=exact_templates,
                regex_patterns=regex_patterns,
            )
            total_penalty_triggered += penalty_triggered_count
            total_hijack_top20 += hijack_top20

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
            rows.append(
                {
                    "sparse_r100": recall_at_k(q["gold_in_corpus"], sparse_laws_ranked, 100),
                    "sparse_r200": recall_at_k(q["gold_in_corpus"], sparse_laws_ranked, 200),
                    "fusion_r100": recall_at_k(q["gold_in_corpus"], fusion_ranked, 100),
                    "fusion_r200": recall_at_k(q["gold_in_corpus"], fusion_ranked, 200),
                }
            )

        strict, _ = evaluate_predictions(
            val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="strict"
        )
        corpus, _ = evaluate_predictions(
            val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="corpus_aware"
        )
        run_row = {
            "run_name": run_name,
            "title_df_threshold": cfg["title_df_threshold"],
            "title_dominance_threshold": cfg["title_dominance_threshold"],
            "low_citation_threshold": cfg["low_citation_threshold"],
            "high_df_title_penalty_strength": cfg["high_df_title_penalty_strength"],
            "generic_title_pattern_mode": cfg["generic_title_pattern_mode"],
            "sparse_laws_Recall@100": round(mean_metric(rows, "sparse_r100"), 6),
            "sparse_laws_Recall@200": round(mean_metric(rows, "sparse_r200"), 6),
            "fusion_final_Recall@100": round(mean_metric(rows, "fusion_r100"), 6),
            "fusion_final_Recall@200": round(mean_metric(rows, "fusion_r200"), 6),
            "strict_macro_f1": round(strict["macro_f1"], 6),
            "corpus_aware_macro_f1": round(corpus["macro_f1"], 6),
            "avg_hijack_top20": round(total_hijack_top20 / max(len(per_query), 1), 6),
            "penalty_triggered_count": total_penalty_triggered,
        }
        runs.append(run_row)

    baseline = next(x for x in runs if x["run_name"] == "baseline_no_penalty")
    candidates = [x for x in runs if x["run_name"] != "baseline_no_penalty"]
    best = sorted(
        candidates,
        key=lambda x: (
            x["corpus_aware_macro_f1"],
            x["strict_macro_f1"],
            x["fusion_final_Recall@200"],
            x["fusion_final_Recall@100"],
            x["sparse_laws_Recall@200"],
            -x["avg_hijack_top20"],
        ),
        reverse=True,
    )[0]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = out_dir / "penalty_results.csv"
    with result_csv.open("w", encoding="utf-8-sig", newline="") as f:
        cols = [
            "run_name",
            "title_df_threshold",
            "title_dominance_threshold",
            "low_citation_threshold",
            "high_df_title_penalty_strength",
            "generic_title_pattern_mode",
            "sparse_laws_Recall@100",
            "sparse_laws_Recall@200",
            "fusion_final_Recall@100",
            "fusion_final_Recall@200",
            "strict_macro_f1",
            "corpus_aware_macro_f1",
            "avg_hijack_top20",
            "penalty_triggered_count",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(runs)

    best_json = out_dir / "best_penalty_config.json"
    payload = {
        "baseline": baseline,
        "best_penalty": best,
        "delta_vs_baseline": {
            "sparse_laws_Recall@100": round(best["sparse_laws_Recall@100"] - baseline["sparse_laws_Recall@100"], 6),
            "sparse_laws_Recall@200": round(best["sparse_laws_Recall@200"] - baseline["sparse_laws_Recall@200"], 6),
            "fusion_final_Recall@100": round(best["fusion_final_Recall@100"] - baseline["fusion_final_Recall@100"], 6),
            "fusion_final_Recall@200": round(best["fusion_final_Recall@200"] - baseline["fusion_final_Recall@200"], 6),
            "strict_macro_f1": round(best["strict_macro_f1"] - baseline["strict_macro_f1"], 6),
            "corpus_aware_macro_f1": round(best["corpus_aware_macro_f1"] - baseline["corpus_aware_macro_f1"], 6),
            "avg_hijack_top20": round(best["avg_hijack_top20"] - baseline["avg_hijack_top20"], 6),
        },
    }
    best_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    delta = payload["delta_vs_baseline"]
    summary_lines = []
    summary_lines.append("# sparse_laws high_df_title_penalty 实验总结（v1_3）")
    summary_lines.append("")
    summary_lines.append("## 最优配置")
    summary_lines.append(
        f"- mode={best['generic_title_pattern_mode']}, title_df_threshold={best['title_df_threshold']}, "
        f"title_dominance_threshold={best['title_dominance_threshold']}, "
        f"low_citation_threshold={best['low_citation_threshold']}, "
        f"high_df_title_penalty_strength={best['high_df_title_penalty_strength']}"
    )
    summary_lines.append("")
    summary_lines.append("## 必答问题")
    summary_lines.append(
        f"1. high_df_title_penalty 是否降低 generic title hijack：{'是' if delta['avg_hijack_top20'] < 0 else '否'} "
        f"(delta_avg_hijack_top20={delta['avg_hijack_top20']:.4f})。"
    )
    summary_lines.append(
        f"2. 是否首次让 fusion_final Recall@100/200 提升："
        f"{'是' if (delta['fusion_final_Recall@100'] > 0 or delta['fusion_final_Recall@200'] > 0) else '否'} "
        f"(delta@100={delta['fusion_final_Recall@100']:.4f}, delta@200={delta['fusion_final_Recall@200']:.4f})。"
    )
    summary_lines.append(
        f"3. 是否首次让 strict/corpus_aware F1 提升："
        f"{'是' if (delta['strict_macro_f1'] > 0 or delta['corpus_aware_macro_f1'] > 0) else '否'} "
        f"(strict_delta={delta['strict_macro_f1']:.4f}, corpus_delta={delta['corpus_aware_macro_f1']:.4f})。"
    )
    summary_lines.append(f"4. 最优 penalty 模式：{best['generic_title_pattern_mode']}。")
    if delta["fusion_final_Recall@200"] <= 0 and delta["corpus_aware_macro_f1"] <= 0:
        summary_lines.append(
            "5. 若仍无提升，下一步优先：query pack 判别增强。理由：仅靠 title 惩罚未能把分支收益传导到融合与最终 F1。"
        )
    else:
        summary_lines.append(
            "5. 若已有提升，下一步优先：title_score_normalization。理由：可在不硬规则堆叠的情况下稳定 title 分量波动。"
        )
    summary_lines.append("")
    summary_lines.append("## 结果概览")
    summary_lines.append(
        f"- sparse_laws Recall@100/200: baseline={baseline['sparse_laws_Recall@100']:.4f}/{baseline['sparse_laws_Recall@200']:.4f}, "
        f"best={best['sparse_laws_Recall@100']:.4f}/{best['sparse_laws_Recall@200']:.4f}"
    )
    summary_lines.append(
        f"- fusion_final Recall@100/200: baseline={baseline['fusion_final_Recall@100']:.4f}/{baseline['fusion_final_Recall@200']:.4f}, "
        f"best={best['fusion_final_Recall@100']:.4f}/{best['fusion_final_Recall@200']:.4f}"
    )
    summary_lines.append(
        f"- strict/corpus_aware F1: baseline={baseline['strict_macro_f1']:.4f}/{baseline['corpus_aware_macro_f1']:.4f}, "
        f"best={best['strict_macro_f1']:.4f}/{best['corpus_aware_macro_f1']:.4f}"
    )
    (out_dir / "penalty_summary_cn.md").write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print(
        json.dumps(
            {"rows": len(runs), "best_run": best["run_name"], "out_dir": str(out_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

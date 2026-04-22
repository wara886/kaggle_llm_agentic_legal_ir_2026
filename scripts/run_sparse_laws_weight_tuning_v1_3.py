from __future__ import annotations

import argparse
import csv
import itertools
import json
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


def dedup_ranked(items):
    out, seen = [], set()
    for x in items:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def recall_at_k(gold, ranked, k):
    if not gold:
        return 0.0
    return len(gold.intersection(set(ranked[:k]))) / len(gold)


def first_rank(gold: set[str], ranked: list[str]) -> int:
    for i, c in enumerate(ranked, start=1):
        if c in gold:
            return i
    return -1


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


def combine_sparse_scores(
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
) -> tuple[list[tuple[str, float]], list[str], dict[str, dict[str, float]]]:
    """
    复刻 sparse field-aware 的“每文档取字段加权后最大值”逻辑。
    返回：
    - sparse_scores: [(citation, score)]
    - sparse_laws_ranked: laws_only ranking
    - component_scores: {citation: {"citation":..., "title":..., "text":...}}
    """
    agg = {}
    laws_component_scores = {}

    def _upsert(c, s):
        if c not in agg or s > agg[c]:
            agg[c] = s

    for c, s in laws_citation_raw.items():
        ws = s * laws_citation_weight
        _upsert(c, ws)
        laws_component_scores.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component_scores[c]["citation"] = ws
    for c, s in laws_title_raw.items():
        ws = s * laws_title_weight
        _upsert(c, ws)
        laws_component_scores.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component_scores[c]["title"] = ws
    for c, s in laws_text_raw.items():
        ws = s * laws_text_weight
        _upsert(c, ws)
        laws_component_scores.setdefault(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
        laws_component_scores[c]["text"] = ws

    for c, s in court_citation_raw.items():
        _upsert(c, s * court_citation_weight)
    for c, s in court_text_raw.items():
        _upsert(c, s * court_text_weight)

    sparse_scores = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    laws_candidates = set(laws_citation_raw) | set(laws_title_raw) | set(laws_text_raw)
    sparse_laws_ranked = [c for c, _ in sparse_scores if c in laws_candidates]
    return sparse_scores, sparse_laws_ranked, laws_component_scores


def why_fp_likely_wins(fp_comp: dict[str, float], gold_comp: dict[str, float]) -> str:
    fp_c, fp_ti, fp_tx = fp_comp["citation"], fp_comp["title"], fp_comp["text"]
    gd_c, gd_ti, gd_tx = gold_comp["citation"], gold_comp["title"], gold_comp["text"]
    if fp_tx > max(fp_c, fp_ti) and fp_tx > gd_tx * 1.2:
        return "text 过强（text_over_power）"
    if fp_ti >= fp_tx and fp_ti > gd_ti * 1.2:
        return "title 模板过泛（title_template_too_generic）"
    if gd_c < fp_c * 0.7:
        return "citation 信号太弱（citation_signal_too_weak）"
    if fp_tx > gd_tx or fp_ti > gd_ti or fp_c > gd_c:
        return "字段混合优势（mixed_field_advantage）"
    return "分数字段差异不显著（no_clear_component_gap）"


def mean_metric(rows, key):
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)


def main():
    p = argparse.ArgumentParser(description="Sparse laws in-branch weight audit and tuning for baseline_v1_3.")
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--sparse-max-laws", type=int, default=175933)
    p.add_argument("--sparse-max-court", type=int, default=120000)
    p.add_argument("--dense-max-laws", type=int, default=60000)
    p.add_argument("--dense-max-court", type=int, default=80000)
    p.add_argument("--dense-disable-sbert", action="store_true")
    p.add_argument("--laws-weight", type=float, default=1.15)
    p.add_argument("--court-weight", type=float, default=0.95)
    p.add_argument("--court-citation-weight", type=float, default=1.25)
    p.add_argument("--court-text-weight", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "v1_3_sparse_laws_tuning")
    args = p.parse_args()

    val_rows = load_query_split("val")
    lookup = load_lookup(ROOT / "artifacts" / "phase0" / "citation_lookup.csv")
    all_norm = set(lookup.keys()) if lookup else set()

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

    doc_source = {}
    doc_text = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
            doc_text[d["citation"]] = (d.get("title", "") + " " + d.get("raw_text", "")[:180]).strip()
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])
        doc_text.setdefault(d["citation"], d.get("text", "")[:180])

    # 固定 retrieval 输入，按 query 预计算原始字段分值与 dense/rule 候选。
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
        gold_in_corpus = set(x for x in gold_norm if (not all_norm) or (x in all_norm))
        per_query.append(
            {
                "query_id": qid,
                "query": q,
                "gold_in_corpus": gold_in_corpus,
                "laws_citation_raw": laws_citation_raw,
                "laws_title_raw": laws_title_raw,
                "laws_text_raw": laws_text_raw,
                "court_citation_raw": court_citation_raw,
                "court_text_raw": court_text_raw,
                "dense_scores": dense_scores,
                "rule_scores": rule_scores,
            }
        )

    grid_cw = [1.0, 1.5, 2.0, 3.0]
    grid_tw = [1.0, 1.5, 2.0, 3.0]
    grid_xw = [0.5, 1.0, 1.5, 2.0]

    results = []
    run_state = {}
    for cw, tw, xw in itertools.product(grid_cw, grid_tw, grid_xw):
        run_name = f"cw{cw}_tw{tw}_xw{xw}"
        pred_map = {}
        query_rows = []
        fp_state = {}
        for q in per_query:
            sparse_scores, sparse_laws_ranked, laws_comp = combine_sparse_scores(
                laws_citation_raw=q["laws_citation_raw"],
                laws_title_raw=q["laws_title_raw"],
                laws_text_raw=q["laws_text_raw"],
                court_citation_raw=q["court_citation_raw"],
                court_text_raw=q["court_text_raw"],
                laws_citation_weight=cw,
                laws_title_weight=tw,
                laws_text_weight=xw,
                court_citation_weight=args.court_citation_weight,
                court_text_weight=args.court_text_weight,
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

            query_rows.append(
                {
                    "sparse_laws_r100": recall_at_k(q["gold_in_corpus"], sparse_laws_ranked, 100),
                    "sparse_laws_r200": recall_at_k(q["gold_in_corpus"], sparse_laws_ranked, 200),
                    "fusion_r100": recall_at_k(q["gold_in_corpus"], fusion_ranked, 100),
                    "fusion_r200": recall_at_k(q["gold_in_corpus"], fusion_ranked, 200),
                }
            )
            fp_state[q["query_id"]] = {
                "gold_set": q["gold_in_corpus"],
                "sparse_laws_ranked": sparse_laws_ranked,
                "laws_component_scores": laws_comp,
            }

        strict, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="strict")
        corpus, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="corpus_aware")
        row = {
            "run_name": run_name,
            "laws_citation_weight": cw,
            "laws_title_weight": tw,
            "laws_text_weight": xw,
            "sparse_laws_Recall@100": round(mean_metric(query_rows, "sparse_laws_r100"), 6),
            "sparse_laws_Recall@200": round(mean_metric(query_rows, "sparse_laws_r200"), 6),
            "fusion_final_Recall@100": round(mean_metric(query_rows, "fusion_r100"), 6),
            "fusion_final_Recall@200": round(mean_metric(query_rows, "fusion_r200"), 6),
            "strict_macro_f1": round(strict["macro_f1"], 6),
            "corpus_aware_macro_f1": round(corpus["macro_f1"], 6),
        }
        results.append(row)
        run_state[run_name] = fp_state

    best = sorted(
        results,
        key=lambda x: (
            x["corpus_aware_macro_f1"],
            x["strict_macro_f1"],
            x["fusion_final_Recall@200"],
            x["sparse_laws_Recall@200"],
            x["sparse_laws_Recall@100"],
        ),
        reverse=True,
    )[0]

    best_name = best["run_name"]
    fp_profile = []
    for q in per_query:
        st = run_state[best_name][q["query_id"]]
        gold_set = st["gold_set"]
        ranked = st["sparse_laws_ranked"]
        if not gold_set:
            continue
        g_rank = first_rank(gold_set, ranked)
        if g_rank != -1 and g_rank <= 100:
            continue
        gold_citation = next(iter(gold_set))
        gold_comp = st["laws_component_scores"].get(gold_citation, {"citation": 0.0, "title": 0.0, "text": 0.0})
        upper = 20 if g_rank == -1 else min(20, g_rank - 1)
        for i, c in enumerate(ranked[:upper], start=1):
            if c in gold_set:
                continue
            fp_comp = st["laws_component_scores"].get(c, {"citation": 0.0, "title": 0.0, "text": 0.0})
            fp_profile.append(
                {
                    "query_id": q["query_id"],
                    "gold_citation": gold_citation,
                    "fp_citation": c,
                    "fp_rank": i,
                    "gold_rank": g_rank,
                    "fp_source": doc_source.get(c, ""),
                    "fp_title_or_snippet": doc_text.get(c, "")[:220],
                    "fp_citation_score_component": round(fp_comp["citation"], 6),
                    "fp_title_score_component": round(fp_comp["title"], 6),
                    "fp_text_score_component": round(fp_comp["text"], 6),
                    "why_fp_likely_wins": why_fp_likely_wins(fp_comp, gold_comp),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.out_dir / "sparse_laws_weight_results.csv"
    with result_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    best_json = args.out_dir / "best_sparse_laws_weight_config.json"
    best_payload = {
        "best_config": best,
        "search_space": {
            "laws_citation_weight": grid_cw,
            "laws_title_weight": grid_tw,
            "laws_text_weight": grid_xw,
        },
    }
    best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    fp_csv = args.out_dir / "sparse_laws_false_positive_profile.csv"
    fp_cols = [
        "query_id",
        "gold_citation",
        "fp_citation",
        "fp_rank",
        "gold_rank",
        "fp_source",
        "fp_title_or_snippet",
        "fp_citation_score_component",
        "fp_title_score_component",
        "fp_text_score_component",
        "why_fp_likely_wins",
    ]
    with fp_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fp_cols)
        w.writeheader()
        for r in fp_profile:
            w.writerow(r)

    # 结论回答
    moved_to_top100 = sum(
        1 for r in fp_profile if r["gold_rank"] != -1 and r["gold_rank"] <= 100
    )
    reason_counter = {}
    for r in fp_profile:
        k = r["why_fp_likely_wins"]
        reason_counter[k] = reason_counter.get(k, 0) + 1
    dominant_reason = sorted(reason_counter.items(), key=lambda x: x[1], reverse=True)[0][0] if reason_counter else "暂无"

    summary_lines = []
    summary_lines.append("# sparse_laws 字段权重调参总结（v1_3）")
    summary_lines.append("")
    summary_lines.append("## 总体结论")
    summary_lines.append(
        f"- 最优配置：laws_citation_weight={best['laws_citation_weight']}, "
        f"laws_title_weight={best['laws_title_weight']}, laws_text_weight={best['laws_text_weight']}。"
    )
    summary_lines.append(
        f"- sparse_laws Recall@100={best['sparse_laws_Recall@100']:.4f}, "
        f"Recall@200={best['sparse_laws_Recall@200']:.4f}。"
    )
    summary_lines.append(
        f"- fusion_final Recall@100={best['fusion_final_Recall@100']:.4f}, "
        f"Recall@200={best['fusion_final_Recall@200']:.4f}, "
        f"strict_macro_f1={best['strict_macro_f1']:.4f}, corpus_aware_macro_f1={best['corpus_aware_macro_f1']:.4f}。"
    )
    summary_lines.append("")
    summary_lines.append("## 必答问题")
    summary_lines.append(
        f"1. 调高 citation/title 权重是否把 gold 从 151-200 推向 1-100：{'部分是' if moved_to_top100 > 0 else '未明显实现'}。"
    )
    summary_lines.append(
        f"2. 当前压过 gold 的假阳性主因：{dominant_reason}。"
    )
    if best["fusion_final_Recall@200"] <= 0.011111 and best["corpus_aware_macro_f1"] <= 0.0154:
        summary_lines.append("3. 若最优配置仍未带动 fusion/F1，上游主因更像：两者都有（query pack 判别不足 + sparse_laws 内部排序不足）。")
    else:
        summary_lines.append("3. 若最优配置已带动 fusion/F1，说明当前瓶颈主要在 sparse_laws 内部排序。")
    summary_lines.append("")
    summary_lines.append("## 产物路径")
    summary_lines.append("- artifacts/v1_3_sparse_laws_tuning/sparse_laws_weight_results.csv")
    summary_lines.append("- artifacts/v1_3_sparse_laws_tuning/best_sparse_laws_weight_config.json")
    summary_lines.append("- artifacts/v1_3_sparse_laws_tuning/sparse_laws_false_positive_profile.csv")
    summary_lines.append("- artifacts/v1_3_sparse_laws_tuning/sparse_laws_weight_summary_cn.md")
    (args.out_dir / "sparse_laws_weight_summary_cn.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8-sig",
    )

    print(
        json.dumps(
            {"rows": len(results), "best_run": best_name, "fp_rows": len(fp_profile), "out_dir": str(args.out_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

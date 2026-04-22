from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation, split_citations
from eval_matchers import evaluate_paragraph_aware
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from run_silver_baseline_v0 import (
    build_doc_lookup,
    build_reranker,
    parse_bool_flag,
    run_split,
)
from retrieval_dense import DenseRetriever
from retrieval_sparse import SparseRetriever


def _norm_list(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for x in values:
        nx = normalize_citation(x)
        if nx and nx not in seen:
            out.append(nx)
            seen.add(nx)
    return out


def _parse_joined(joined: str) -> list[str]:
    return [x for x in joined.split(";") if x.strip()]


def _macro_recall_at_k(gold_rows: list[dict], trace_rows: list[dict], k: int = 100) -> float:
    trace_map = {r["query_id"]: r for r in trace_rows}
    recalls = []
    for row in gold_rows:
        qid = row["query_id"]
        gold = set(_norm_list(row.get("gold_citation_list") or split_citations(row.get("gold_citations", ""))))
        if not gold:
            recalls.append(0.0)
            continue
        tr = trace_map.get(qid, {})
        fused = [_norm_list([x])[0] for x in _parse_joined(tr.get("fused_top200", "")) if _norm_list([x])]
        hits = len(gold & set(fused[:k]))
        recalls.append(hits / len(gold))
    return round(sum(recalls) / len(recalls), 6) if recalls else 0.0


def _collect_corpus_norm_set() -> set[str]:
    out = set()
    for row in iter_corpus_rows(include_laws=True, include_court=True):
        out.add(row["norm_citation"])
    return out


def _run_one_config(
    run_name: str,
    val_rows: list[dict],
    sparse: SparseRetriever,
    dense: DenseRetriever,
    doc_lookup: dict[str, dict],
    enable_router: bool,
    prefer_strong_reranker: bool,
    dynamic_mode: str,
    fixed_top_k: int,
    score_threshold: float,
    relative_threshold: float,
    corpus_norm_set: set[str],
) -> tuple[dict, dict[str, list[str]], list[dict]]:
    reranker = build_reranker(prefer_strong=prefer_strong_reranker)
    pred_map, trace_rows, route_counts = run_split(
        rows=val_rows,
        sparse=sparse,
        dense=dense,
        doc_lookup=doc_lookup,
        reranker=reranker,
        dynamic_mode=dynamic_mode,
        fixed_top_k=fixed_top_k,
        score_threshold=score_threshold,
        relative_threshold=relative_threshold,
        enable_router=enable_router,
    )
    strict_summary, strict_rows = evaluate_predictions(
        gold_rows=val_rows,
        pred_map=pred_map,
        citation_lookup=None,
        mode="strict",
    )
    corpus_summary, _ = evaluate_predictions(
        gold_rows=val_rows,
        pred_map=pred_map,
        citation_lookup={c: c for c in corpus_norm_set},
        mode="corpus_aware",
    )
    para_summary, para_rows = evaluate_paragraph_aware(
        gold_rows=val_rows,
        pred_map=pred_map,
        corpus_norm_citations=corpus_norm_set,
    )
    strict_map = {r["query_id"]: r for r in strict_rows}
    para_map = {r["query_id"]: r for r in para_rows}
    merged_per_query = []
    trace_map = {r["query_id"]: r for r in trace_rows}
    for row in val_rows:
        qid = row["query_id"]
        sr = strict_map.get(qid, {})
        pr = para_map.get(qid, {})
        tr = trace_map.get(qid, {})
        merged_per_query.append(
            {
                "run_name": run_name,
                "query_id": qid,
                "strict_f1": sr.get("f1", 0.0),
                "paragraph_aware_f1": pr.get("paragraph_aware_f1", 0.0),
                "primary_source": tr.get("primary_source", ""),
                "route_confidence": tr.get("route_confidence", ""),
                "fused_top200": tr.get("fused_top200", ""),
                "final_predictions": tr.get("final_predictions", ""),
            }
        )

    result = {
        "run_name": run_name,
        "source_router_enabled": int(enable_router),
        "reranker_name": reranker.name,
        "reranker_fallback_reason": reranker.fallback_reason,
        "dynamic_mode": dynamic_mode,
        "strict_macro_f1": strict_summary["macro_f1"],
        "corpus_aware_macro_f1": corpus_summary["macro_f1"],
        "paragraph_aware_macro_f1": para_summary["paragraph_aware_macro_f1"],
        "laws_route_query_count": route_counts.get("laws", 0),
        "court_route_query_count": route_counts.get("court", 0),
        "hybrid_route_query_count": route_counts.get("hybrid", 0),
        "fusion_final_Recall@100": _macro_recall_at_k(val_rows, trace_rows, k=100),
        "fusion_final_Recall@200": _macro_recall_at_k(val_rows, trace_rows, k=200),
    }
    return result, pred_map, merged_per_query


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best_by_metric(rows: list[dict], metric: str) -> dict:
    return sorted(rows, key=lambda x: float(x.get(metric, 0.0)), reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="silver_baseline_v0 local evaluation")
    parser.add_argument("--fixed-top-k", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--relative-threshold", type=float, default=0.85)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--sparse-max-court", type=int, default=300000)
    parser.add_argument("--dense-max-laws", type=int, default=80000)
    parser.add_argument("--dense-max-court", type=int, default=120000)
    parser.add_argument("--prefer-strong-reranker", type=parse_bool_flag, default=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "silver_baseline_v0")
    args = parser.parse_args()

    val_rows = load_query_split("val")
    corpus_norm_set = _collect_corpus_norm_set()

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
    doc_lookup = build_doc_lookup(sparse, dense)

    configs = [
        {
            "run_name": "router_strong_relative",
            "enable_router": True,
            "prefer_strong_reranker": args.prefer_strong_reranker,
            "dynamic_mode": "relative_threshold",
        },
        {
            "run_name": "router_strong_score",
            "enable_router": True,
            "prefer_strong_reranker": args.prefer_strong_reranker,
            "dynamic_mode": "score_threshold",
        },
        {
            "run_name": "router_strong_fixed",
            "enable_router": True,
            "prefer_strong_reranker": args.prefer_strong_reranker,
            "dynamic_mode": "fixed_top_k",
        },
        {
            "run_name": "no_router_strong_relative",
            "enable_router": False,
            "prefer_strong_reranker": args.prefer_strong_reranker,
            "dynamic_mode": "relative_threshold",
        },
        {
            "run_name": "router_light_relative",
            "enable_router": True,
            "prefer_strong_reranker": False,
            "dynamic_mode": "relative_threshold",
        },
    ]

    results = []
    per_query_all = []
    pred_cache = {}
    for cfg in configs:
        result, pred_map, per_query_rows = _run_one_config(
            run_name=cfg["run_name"],
            val_rows=val_rows,
            sparse=sparse,
            dense=dense,
            doc_lookup=doc_lookup,
            enable_router=cfg["enable_router"],
            prefer_strong_reranker=cfg["prefer_strong_reranker"],
            dynamic_mode=cfg["dynamic_mode"],
            fixed_top_k=args.fixed_top_k,
            score_threshold=args.score_threshold,
            relative_threshold=args.relative_threshold,
            corpus_norm_set=corpus_norm_set,
        )
        results.append(result)
        per_query_all.extend(per_query_rows)
        pred_cache[cfg["run_name"]] = pred_map

    best_run = _best_by_metric(results, "strict_macro_f1")
    router_run = next(x for x in results if x["run_name"] == "router_strong_relative")
    no_router_run = next(x for x in results if x["run_name"] == "no_router_strong_relative")
    light_run = next(x for x in results if x["run_name"] == "router_light_relative")
    dynamic_best = _best_by_metric(
        [x for x in results if x["dynamic_mode"] in {"relative_threshold", "score_threshold"}],
        "strict_macro_f1",
    )
    fixed_run = next(x for x in results if x["run_name"] == "router_strong_fixed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.out_dir / "silver_eval_results.csv"
    per_query_csv = args.out_dir / "silver_per_query.csv"
    summary_md = args.out_dir / "silver_summary_cn.md"

    _write_csv(results_csv, results)
    _write_csv(per_query_csv, per_query_all)

    summary_lines = [
        "# silver_baseline_v0 评测总结",
        "",
        "## 核心结论",
        f"- source_router 是否比无 router 更优：{'是' if router_run['strict_macro_f1'] > no_router_run['strict_macro_f1'] else '否'} "
        f"(strict: {router_run['strict_macro_f1']:.4f} vs {no_router_run['strict_macro_f1']:.4f})。",
        f"- laws/court/hybrid 路由数量：{router_run['laws_route_query_count']}/"
        f"{router_run['court_route_query_count']}/{router_run['hybrid_route_query_count']}。",
        f"- 强重排器（strong_reranker）是否优于轻重排器："
        f"{'是' if router_run['strict_macro_f1'] > light_run['strict_macro_f1'] else '否'} "
        f"(strict: {router_run['strict_macro_f1']:.4f} vs {light_run['strict_macro_f1']:.4f})。",
        f"- dynamic threshold 是否优于 fixed_top_k："
        f"{'是' if dynamic_best['strict_macro_f1'] > fixed_run['strict_macro_f1'] else '否'} "
        f"(best_dynamic={dynamic_best['run_name']}, strict={dynamic_best['strict_macro_f1']:.4f}; "
        f"fixed strict={fixed_run['strict_macro_f1']:.4f})。",
        (
            f"- paragraph-aware matcher 对 train-side laws ceiling 的提升估计："
            f"{max(0.0, best_run['paragraph_aware_macro_f1'] - best_run['strict_macro_f1']):.4f}"
            "（以 paragraph_aware_macro_f1 - strict_macro_f1 近似）。"
        ),
        "- 下一步最值得叠加的增强：`router heuristic refinement`。理由：当前源路由与分源召回仍是最直接瓶颈，"
        "优先提升路由准确性可以同步改善 laws/court 主线命中，再承接 co-citation 扩展。",
        "",
        "## 最优运行（按 strict_macro_f1）",
        json.dumps(best_run, ensure_ascii=False, indent=2),
    ]
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "results_csv": str(results_csv),
                "per_query_csv": str(per_query_csv),
                "summary_md": str(summary_md),
                "best_run": best_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

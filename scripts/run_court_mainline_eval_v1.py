from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from retrieval_dense import DenseRetriever
from retrieval_sparse import SparseRetriever
from run_silver_baseline_v0 import build_doc_lookup, build_reranker, run_split


def macro_recall_at_k(gold_rows: list[dict], trace_rows: list[dict], k: int) -> float:
    trace_map = {r["query_id"]: r for r in trace_rows}
    recalls = []
    for row in gold_rows:
        qid = row["query_id"]
        gold_raw = row.get("gold_citations", "")
        gold_items = [x.strip() for x in gold_raw.split(";") if x.strip()]
        if not gold_items:
            recalls.append(0.0)
            continue
        fused_top = trace_map.get(qid, {}).get("fused_top200", "")
        fused_items = [x.strip() for x in fused_top.split(";") if x.strip()]
        hits = len(set(gold_items) & set(fused_items[:k]))
        recalls.append(hits / len(gold_items))
    return round(sum(recalls) / len(recalls), 6) if recalls else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="court_mainline_v1 evaluation")
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--prefer-strong-reranker", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "court_mainline_v1")
    args = parser.parse_args()

    val_rows = load_query_split("val")
    val_df = pd.read_csv(ROOT / "data_raw" / "competition_data" / "val.csv")

    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(max_laws_rows=175933, max_court_rows=300000, enable_field_aware=True)
    dense = DenseRetriever(use_sbert=not args.dense_disable_sbert, text_max_chars=500, svd_dim=256)
    dense.build(max_laws_rows=80000, max_court_rows=120000, enable_field_aware=True)
    doc_lookup = build_doc_lookup(sparse, dense)
    reranker = build_reranker(prefer_strong=args.prefer_strong_reranker)

    runs = [
        {
            "run_name": "no_router",
            "enable_router": False,
            "router_version": "v1",
            "enable_court_mainline": False,
        },
        {
            "run_name": "source_router_v1_1",
            "enable_router": True,
            "router_version": "v1_1",
            "enable_court_mainline": False,
        },
        {
            "run_name": "court_mainline_v1",
            "enable_router": True,
            "router_version": "v1_1",
            "enable_court_mainline": True,
        },
    ]

    eval_rows: list[dict] = []
    per_query_rows: list[dict] = []

    for cfg in runs:
        pred_map, trace_rows, route_counts = run_split(
            rows=val_rows,
            sparse=sparse,
            dense=dense,
            doc_lookup=doc_lookup,
            reranker=reranker,
            dynamic_mode="relative_threshold",
            fixed_top_k=12,
            score_threshold=0.15,
            relative_threshold=0.85,
            enable_router=cfg["enable_router"],
            router_version=cfg["router_version"],
            enable_court_mainline=cfg["enable_court_mainline"],
            laws_route_laws_max=120,
            laws_route_court_max=30,
            court_route_court_max=200,
            court_route_laws_max=30,
            hybrid_route_laws_max=120,
            hybrid_route_court_max=140,
            min_court_candidates_for_hybrid=60,
            min_court_candidates_for_court_route=100,
        )

        strict_summary, strict_per_query = evaluate_predictions(
            gold_rows=val_rows,
            pred_map=pred_map,
            citation_lookup=None,
            mode="strict",
        )
        corpus_summary, _ = evaluate_predictions(
            gold_rows=val_rows,
            pred_map=pred_map,
            citation_lookup=None,
            mode="corpus_aware",
        )

        trace_df = pd.DataFrame(trace_rows)
        strict_map = {x["query_id"]: x for x in strict_per_query}
        for _, vr in val_df.iterrows():
            qid = vr["query_id"]
            tr = trace_df[trace_df["query_id"] == qid]
            tr_row = tr.iloc[0].to_dict() if len(tr) else {}
            per_query_rows.append(
                {
                    "run_name": cfg["run_name"],
                    "query_id": qid,
                    "primary_source": tr_row.get("primary_source", ""),
                    "secondary_source": tr_row.get("secondary_source", ""),
                    "route_confidence": tr_row.get("route_confidence", ""),
                    "laws_candidates_forwarded": tr_row.get("laws_candidates_forwarded", 0),
                    "court_candidates_forwarded": tr_row.get("court_candidates_forwarded", 0),
                    "fused_top200": tr_row.get("fused_top200", ""),
                    "final_predictions": tr_row.get("final_predictions", ""),
                    "gold_citation": vr.get("gold_citations", ""),
                    "court_mainline_effect_note": tr_row.get("court_mainline_effect_note", ""),
                    "strict_f1": strict_map.get(qid, {}).get("f1", 0.0),
                }
            )

        avg_laws = safe_mean(trace_df["laws_candidates_forwarded"]) if "laws_candidates_forwarded" in trace_df else 0.0
        avg_court = safe_mean(trace_df["court_candidates_forwarded"]) if "court_candidates_forwarded" in trace_df else 0.0
        hybrid_with_court = 0
        court_route_with_court = 0
        if len(trace_df):
            hybrid_with_court = int(
                len(
                    trace_df[
                        (trace_df["primary_source"] == "hybrid")
                        & (trace_df["court_candidates_forwarded"] > 0)
                    ]
                )
            )
            court_route_with_court = int(
                len(
                    trace_df[
                        (trace_df["primary_source"] == "court")
                        & (trace_df["court_candidates_forwarded"] > 0)
                    ]
                )
            )

        eval_rows.append(
            {
                "run_name": cfg["run_name"],
                "laws_route_query_count": route_counts.get("laws", 0),
                "court_route_query_count": route_counts.get("court", 0),
                "hybrid_route_query_count": route_counts.get("hybrid", 0),
                "avg_laws_candidates_forwarded": round(avg_laws, 4),
                "avg_court_candidates_forwarded": round(avg_court, 4),
                "hybrid_queries_with_court_candidates_count": hybrid_with_court,
                "court_route_queries_with_court_candidates_count": court_route_with_court,
                "fusion_final_Recall@100": macro_recall_at_k(val_rows, trace_rows, 100),
                "fusion_final_Recall@200": macro_recall_at_k(val_rows, trace_rows, 200),
                "strict_macro_f1": strict_summary["macro_f1"],
                "corpus_aware_macro_f1": corpus_summary["macro_f1"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = args.out_dir / "court_mainline_eval_results.csv"
    per_csv = args.out_dir / "court_mainline_per_query.csv"
    summary_md = args.out_dir / "court_mainline_summary_cn.md"
    write_csv(eval_csv, eval_rows)
    write_csv(per_csv, per_query_rows)

    df = pd.DataFrame(eval_rows)
    base = df[df["run_name"] == "source_router_v1_1"].iloc[0]
    cm = df[df["run_name"] == "court_mainline_v1"].iloc[0]
    recall_up = float(cm["fusion_final_Recall@200"]) > float(base["fusion_final_Recall@200"]) or float(
        cm["fusion_final_Recall@100"]
    ) > float(base["fusion_final_Recall@100"])
    f1_up = float(cm["strict_macro_f1"]) > float(base["strict_macro_f1"]) or float(cm["corpus_aware_macro_f1"]) > float(
        base["corpus_aware_macro_f1"]
    )

    if recall_up and not f1_up:
        next_choice = "dynamic_threshold_tuning"
        next_reason = "已提升召回但未传导到最终 F1，优先做阈值截断调优把新增候选保留下来。"
    elif not recall_up:
        next_choice = "court dense retrieval"
        next_reason = "仅靠 court sparse 仍不足以拉升召回，下一步应补 court dense 语义召回。"
    else:
        next_choice = "dynamic_threshold_tuning"
        next_reason = "召回增长后优先做阈值调优，通常比先开强 reranker 更直接。"

    summary = f"""# court_mainline_v1 评测总结

## 核心结论
1. hybrid query 中是否有更多 court candidates 进入 rerank：
   - source_router_v1_1: {int(base['hybrid_queries_with_court_candidates_count'])}
   - court_mainline_v1: {int(cm['hybrid_queries_with_court_candidates_count'])}
2. court_route_query_count 与 court_candidates_forwarded：
   - court_route_query_count: v1_1={int(base['court_route_query_count'])}, v1={int(cm['court_route_query_count'])}
   - avg_court_candidates_forwarded: v1_1={float(base['avg_court_candidates_forwarded']):.4f}, v1={float(cm['avg_court_candidates_forwarded']):.4f}
3. fusion_final Recall@100/200 是否高于 source_router_v1_1：
   - v1_1: @100={float(base['fusion_final_Recall@100']):.6f}, @200={float(base['fusion_final_Recall@200']):.6f}
   - v1: @100={float(cm['fusion_final_Recall@100']):.6f}, @200={float(cm['fusion_final_Recall@200']):.6f}
   - 结论：{'是' if recall_up else '否'}
4. 若 recall 提升但 F1 不动：在 `dynamic_threshold_tuning` 与 `strong_reranker_enablement` 二选一，优先 `dynamic_threshold_tuning`。
5. 若 recall 仍不提升：在 `court dense retrieval` 与 `router v1_2` 二选一，优先 `court dense retrieval`（当前选择：`{next_choice}`）。

## 指标表
{df.to_markdown(index=False)}

## 下一步建议
- `{next_choice}`：{next_reason}
"""
    summary_md.write_text(summary, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "eval_csv": str(eval_csv),
                "per_query_csv": str(per_csv),
                "summary_md": str(summary_md),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

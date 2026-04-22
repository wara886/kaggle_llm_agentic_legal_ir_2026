from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from retrieval_sparse import SparseRetriever
from retrieval_dense import DenseRetriever
from run_silver_baseline_v0 import build_doc_lookup, build_reranker, run_split


CASE_PAT = re.compile(r"\b\d+[A-Z]_[0-9]+/[0-9]{4}\b|\bBGE\b|\bE\.\s*\d+", re.I)
STATUTE_PAT = re.compile(
    r"\bArt\.?\s*\d+|\barticle\s+\d+|\bAbs\.?\s*\d+|\blit\.?\s*[a-z]|"
    r"\b(?:ZGB|OR|BGG|StPO|BV|IPRG|IVG|ATSG|SVG|ZPO|StGB)\b",
    re.I,
)


def classify_gold_type(gold_text: str) -> str:
    text = gold_text or ""
    has_case = bool(CASE_PAT.search(text))
    has_statute = bool(STATUTE_PAT.search(text))
    if has_case and has_statute:
        return "mixed"
    if has_case:
        return "case_like"
    if has_statute:
        return "statute_like"
    return "mixed"


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


def evaluate_route_quality(
    trace_rows: list[dict],
    strategy_router_df: pd.DataFrame,
) -> tuple[float, int, int]:
    trace_df = pd.DataFrame(trace_rows)
    if trace_df.empty:
        return 0.0, 0, 0
    merged = strategy_router_df.merge(
        trace_df[["query_id", "primary_source"]],
        on="query_id",
        how="left",
    )
    merged["primary_source"] = merged["primary_source"].fillna("unknown").str.lower()
    merged["should_route_first"] = merged["should_route_first"].fillna("unknown").str.lower()
    agreement = float((merged["primary_source"] == merged["should_route_first"]).mean()) if len(merged) else 0.0
    under = merged[
        merged["should_route_first"].isin(["court", "hybrid"])
        & ~merged["primary_source"].isin(["court", "hybrid"])
    ]
    over = merged[
        merged["should_route_first"].isin(["court", "hybrid"])
        & (merged["primary_source"] == "laws")
    ]
    return round(agreement, 6), int(len(under)), int(len(over))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="source_router_v1_1 evaluation")
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--prefer-strong-reranker", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "source_router_v1_1")
    args = parser.parse_args()

    val_rows = load_query_split("val")
    val_df = pd.read_csv(ROOT / "data_raw" / "competition_data" / "val.csv")
    strategy_router_df = pd.read_csv(
        ROOT / "artifacts" / "strategy_audit_v2" / "query_source_router_candidates.csv",
        encoding="utf-8-sig",
    )

    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(max_laws_rows=175933, max_court_rows=300000, enable_field_aware=True)
    dense = DenseRetriever(use_sbert=not args.dense_disable_sbert, text_max_chars=500, svd_dim=256)
    dense.build(max_laws_rows=80000, max_court_rows=120000, enable_field_aware=True)
    doc_lookup = build_doc_lookup(sparse, dense)
    reranker = build_reranker(prefer_strong=args.prefer_strong_reranker)

    runs = [
        {"run_name": "no_router", "enable_router": False, "router_version": "v1"},
        {"run_name": "source_router_v1", "enable_router": True, "router_version": "v1"},
        {"run_name": "source_router_v1_1", "enable_router": True, "router_version": "v1_1"},
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
        route_agreement_rate, under_court, over_laws = evaluate_route_quality(
            trace_rows=trace_rows,
            strategy_router_df=strategy_router_df,
        )

        strict_map = {x["query_id"]: x for x in strict_per_query}
        trace_map = {x["query_id"]: x for x in trace_rows}
        for _, vr in val_df.iterrows():
            qid = vr["query_id"]
            tr = trace_map.get(qid, {})
            sr = strict_map.get(qid, {})
            per_query_rows.append(
                {
                    "run_name": cfg["run_name"],
                    "query_id": qid,
                    "gold_type": classify_gold_type(str(vr.get("gold_citations", ""))),
                    "primary_source": tr.get("primary_source", ""),
                    "secondary_source": tr.get("secondary_source", ""),
                    "route_confidence": tr.get("route_confidence", ""),
                    "case_score": tr.get("case_score", 0.0),
                    "statute_score": tr.get("statute_score", 0.0),
                    "mixed_score": tr.get("mixed_score", 0.0),
                    "matched_case_patterns": tr.get("matched_case_patterns", ""),
                    "matched_statute_patterns": tr.get("matched_statute_patterns", ""),
                    "route_decision_reason_v1_1": tr.get("route_decision_reason_v1_1", ""),
                    "strict_f1": sr.get("f1", 0.0),
                    "fused_top200": tr.get("fused_top200", ""),
                    "final_predictions": tr.get("final_predictions", ""),
                }
            )

        eval_rows.append(
            {
                "run_name": cfg["run_name"],
                "laws_route_query_count": route_counts.get("laws", 0),
                "court_route_query_count": route_counts.get("court", 0),
                "hybrid_route_query_count": route_counts.get("hybrid", 0),
                "route_agreement_rate": route_agreement_rate,
                "under_routed_to_court_count": under_court,
                "over_routed_to_laws_count": over_laws,
                "fusion_final_Recall@100": macro_recall_at_k(val_rows, trace_rows, 100),
                "fusion_final_Recall@200": macro_recall_at_k(val_rows, trace_rows, 200),
                "strict_macro_f1": strict_summary["macro_f1"],
                "corpus_aware_macro_f1": corpus_summary["macro_f1"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_eval = args.out_dir / "source_router_v1_1_eval_results.csv"
    out_per = args.out_dir / "source_router_v1_1_per_query.csv"
    out_md = args.out_dir / "source_router_v1_1_summary_cn.md"
    write_csv(out_eval, eval_rows)
    write_csv(out_per, per_query_rows)

    df = pd.DataFrame(eval_rows)
    v1 = df[df["run_name"] == "source_router_v1"].iloc[0]
    v11 = df[df["run_name"] == "source_router_v1_1"].iloc[0]
    no_r = df[df["run_name"] == "no_router"].iloc[0]

    if float(v11["fusion_final_Recall@200"]) <= float(v1["fusion_final_Recall@200"]):
        next_choice = "court_mainline_completion"
        next_reason = "router_v1_1 已把分流改善到一定程度后，court 主线候选质量将成为下一瓶颈。"
    else:
        next_choice = "court_mainline_completion"
        next_reason = "router_v1_1 提升后，最应继续补齐 court 主线检索能力以放大收益。"

    summary = f"""# source_router_v1_1 评测总结

## 对比结论
- court_route_query_count 是否从 0 提升：{'是' if float(v11['court_route_query_count']) > float(v1['court_route_query_count']) else '否'}（v1={int(v1['court_route_query_count'])}, v1_1={int(v11['court_route_query_count'])}）。
- route_agreement_rate 是否高于 0.70：{'是' if float(v11['route_agreement_rate']) > 0.70 else '否'}（v1_1={float(v11['route_agreement_rate']):.4f}）。
- under_routed_to_court_count 是否下降：{'是' if float(v11['under_routed_to_court_count']) < float(v1['under_routed_to_court_count']) else '否'}（v1={int(v1['under_routed_to_court_count'])}, v1_1={int(v11['under_routed_to_court_count'])}）。
- fusion_final Recall@100/200 是否继续提升：{'是' if (float(v11['fusion_final_Recall@100']) > float(v1['fusion_final_Recall@100']) or float(v11['fusion_final_Recall@200']) > float(v1['fusion_final_Recall@200'])) else '否'}。
  - v1: @100={float(v1['fusion_final_Recall@100']):.6f}, @200={float(v1['fusion_final_Recall@200']):.6f}
  - v1_1: @100={float(v11['fusion_final_Recall@100']):.6f}, @200={float(v11['fusion_final_Recall@200']):.6f}
  - no_router: @100={float(no_r['fusion_final_Recall@100']):.6f}, @200={float(no_r['fusion_final_Recall@200']):.6f}

## 指标表（核心）
{df.to_markdown(index=False)}

## 若 v1_1 仍无明显提升，下一步最优先
- 选择：`{next_choice}`
- 理由：{next_reason}
"""
    out_md.write_text(summary, encoding="utf-8-sig")
    print(json.dumps({"eval_csv": str(out_eval), "per_query_csv": str(out_per), "summary": str(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

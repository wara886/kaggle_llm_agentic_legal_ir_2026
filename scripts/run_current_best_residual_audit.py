from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from citation_normalizer import normalize_citation
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from retrieval_rules import RuleCitationRetriever

from run_procedural_prior_ablation import _apply_global_priors
from run_qwen3_reranker_module_ablation import (
    Qwen3Reranker,
    _detect_total_gpu_mem_gb,
    _load_doc_lookup,
    _load_trace_rows,
    _parse_joined,
    _qwen_predictions_from_trace,
)


def _is_explicit(query: str) -> int:
    return int(bool(RuleCitationRetriever.extract_patterns(query or "")))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_block(rows: list[dict], pred_map: dict[str, list[str]], explicit: int | None) -> dict:
    subset = [r for r in rows if explicit is None or _is_explicit(r.get("query", "")) == explicit]
    qids = {r["query_id"] for r in subset}
    scoped_pred = {qid: pred_map.get(qid, []) for qid in qids}
    strict, strict_rows = evaluate_predictions(subset, scoped_pred, citation_lookup=None, mode="strict")
    corpus, _ = evaluate_predictions(subset, scoped_pred, citation_lookup=None, mode="corpus_aware")
    return {
        "queries": len(subset),
        "strict_f1": float(strict.get("macro_f1", 0.0)),
        "corpus_f1": float(corpus.get("macro_f1", 0.0)),
        "final_fp": int(sum(int(x.get("fp", 0)) for x in strict_rows)),
    }


def _build_gold_audit_rows(
    val_rows: list[dict],
    trace_rows: list[dict],
    qwen_pred: dict[str, list[str]],
    qwen_reranked: dict[str, list[str]],
    best_pred: dict[str, list[str]],
    candidate_cap: int,
    global_priors: list[str],
) -> list[dict]:
    by_qid = {r.get("query_id", ""): r for r in trace_rows}
    global_prior_set = {normalize_citation(x) for x in global_priors if normalize_citation(x)}
    rows: list[dict] = []
    for vr in val_rows:
        qid = vr["query_id"]
        tr = by_qid.get(qid, {})
        fused_top200 = set(_parse_joined(tr.get("fused_top200", "")))
        qwen_input = _parse_joined(tr.get("rerank_input_citations", "")) or _parse_joined(tr.get("fused_top320", ""))
        if candidate_cap > 0:
            qwen_input = qwen_input[:candidate_cap]
        qwen_input_set = set(qwen_input)
        qwen_ranks = {c: i + 1 for i, c in enumerate(qwen_reranked.get(qid, []))}
        qwen_final = set(qwen_pred.get(qid, []))
        best_final = set(best_pred.get(qid, []))
        dynamic_mode = tr.get("dynamic_mode", "fixed_top_k")
        fixed_top_k = int(tr.get("fixed_top_k", "5") or 5)
        query = vr.get("query", "")
        explicit_subset = _is_explicit(query)
        for g_raw in vr.get("gold_citation_list", []):
            gold = normalize_citation(g_raw)
            if not gold:
                continue
            if gold in best_final:
                if gold in global_prior_set and gold not in qwen_final:
                    stage = "rescued_by_global_prior"
                else:
                    stage = "kept_final"
            elif gold not in fused_top200:
                stage = "not_in_fused_top200"
            elif gold not in qwen_input_set:
                stage = "not_in_qwen_input_cap"
            elif gold not in qwen_ranks:
                stage = "not_scored_by_qwen"
            elif dynamic_mode == "fixed_top_k" and fixed_top_k > 0 and qwen_ranks[gold] > fixed_top_k:
                stage = "reranked_too_low"
            else:
                stage = "cut_by_dynamic_threshold"
            rows.append(
                {
                    "query_id": qid,
                    "query": query,
                    "explicit_subset": explicit_subset,
                    "gold_citation": gold,
                    "gold_in_fused_top200": int(gold in fused_top200),
                    "gold_in_qwen_input_cap": int(gold in qwen_input_set),
                    "gold_in_qwen_final": int(gold in qwen_final),
                    "gold_in_current_best_final": int(gold in best_final),
                    "qwen_rank": int(qwen_ranks.get(gold, -1)),
                    "drop_stage": stage,
                }
            )
    return rows


def _stage_summary(rows: list[dict], explicit: int | None) -> dict:
    scoped = [r for r in rows if explicit is None or int(r["explicit_subset"]) == explicit]
    stage_counter = Counter(r["drop_stage"] for r in scoped)
    total_gold = len(scoped)
    total_missed = sum(
        int(stage_counter.get(name, 0))
        for name in [
            "not_in_fused_top200",
            "not_in_qwen_input_cap",
            "not_scored_by_qwen",
            "reranked_too_low",
            "cut_by_dynamic_threshold",
        ]
    )
    candidate_stage_miss = sum(
        int(stage_counter.get(name, 0))
        for name in ["not_in_fused_top200", "not_in_qwen_input_cap", "not_scored_by_qwen"]
    )
    rerank_stage_miss = sum(
        int(stage_counter.get(name, 0))
        for name in ["reranked_too_low", "cut_by_dynamic_threshold"]
    )
    kept = int(stage_counter.get("kept_final", 0))
    rescued = int(stage_counter.get("rescued_by_global_prior", 0))
    return {
        "total_gold": total_gold,
        "kept_final": kept,
        "rescued_by_global_prior": rescued,
        "not_in_fused_top200": int(stage_counter.get("not_in_fused_top200", 0)),
        "not_in_qwen_input_cap": int(stage_counter.get("not_in_qwen_input_cap", 0)),
        "not_scored_by_qwen": int(stage_counter.get("not_scored_by_qwen", 0)),
        "reranked_too_low": int(stage_counter.get("reranked_too_low", 0)),
        "cut_by_dynamic_threshold": int(stage_counter.get("cut_by_dynamic_threshold", 0)),
        "candidate_stage_miss": int(candidate_stage_miss),
        "rerank_stage_miss": int(rerank_stage_miss),
        "final_kept_rate": round((kept + rescued) / total_gold, 6) if total_gold else 0.0,
        "gold_in_fused_top200_rate": round(
            sum(int(r["gold_in_fused_top200"]) for r in scoped) / total_gold, 6
        )
        if total_gold
        else 0.0,
        "global_prior_rescue_rate": round(rescued / total_gold, 6) if total_gold else 0.0,
        "candidate_stage_share_of_missed": round(candidate_stage_miss / total_missed, 6) if total_missed else 0.0,
        "rerank_stage_share_of_missed": round(rerank_stage_miss / total_missed, 6) if total_missed else 0.0,
    }


def _query_summary_rows(audit_rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    stage_names = [
        "not_in_fused_top200",
        "not_in_qwen_input_cap",
        "not_scored_by_qwen",
        "reranked_too_low",
        "cut_by_dynamic_threshold",
    ]
    for row in audit_rows:
        qid = row["query_id"]
        rec = grouped.setdefault(
            qid,
            {
                "query_id": qid,
                "explicit_subset": int(row["explicit_subset"]),
                "query": row["query"],
                "gold_total": 0,
                "gold_kept_final": 0,
                "gold_rescued_by_global_prior": 0,
                **{f"miss_{name}": 0 for name in stage_names},
            },
        )
        rec["gold_total"] += 1
        stage = row["drop_stage"]
        if stage == "kept_final":
            rec["gold_kept_final"] += 1
        elif stage == "rescued_by_global_prior":
            rec["gold_rescued_by_global_prior"] += 1
        elif stage in stage_names:
            rec[f"miss_{stage}"] += 1
    out: list[dict] = []
    for rec in grouped.values():
        miss_total = sum(int(rec[f"miss_{name}"]) for name in stage_names)
        kept_total = int(rec["gold_kept_final"]) + int(rec["gold_rescued_by_global_prior"])
        rec["miss_total"] = miss_total
        rec["kept_total"] = kept_total
        rec["kept_rate"] = round(kept_total / int(rec["gold_total"]), 6) if int(rec["gold_total"]) else 0.0
        out.append(rec)
    out.sort(
        key=lambda x: (
            -int(x["miss_total"]),
            int(x["explicit_subset"]),
            x["query_id"],
        )
    )
    return out


def _top_residual_rows(query_rows: list[dict], limit: int = 5) -> list[dict]:
    return query_rows[: max(0, int(limit))]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual audit for current best Qwen3 + Art.100 path.")
    parser.add_argument(
        "--baseline-trace-csv",
        type=Path,
        default=ROOT / "outputs" / "train_structq_ablation_cloud" / "val_seed_trace_silver_baseline_v0.csv",
    )
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--qwen-max-length", type=int, default=2048)
    parser.add_argument("--qwen-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--qwen-candidate-cap", type=int, default=80)
    parser.add_argument("--require-cloud-gpu-mem-gb", type=float, default=40.0)
    parser.add_argument("--allow-noncloud", action="store_true")
    parser.add_argument("--top-query-limit", type=int, default=5)
    parser.add_argument("--report-label", default="current_best_residual_audit")
    parser.add_argument("--report-title", default="Current Best Residual Audit")
    parser.add_argument("--rerank-cache-json", type=Path, default=None)
    args = parser.parse_args()

    total_gpu_gb = _detect_total_gpu_mem_gb()
    cloud_gate_ok = total_gpu_gb >= float(args.require_cloud_gpu_mem_gb)
    if (not cloud_gate_ok) and (not args.allow_noncloud):
        raise SystemExit(
            f"Cloud gate blocked: detected GPU mem={total_gpu_gb:.2f}GB < {args.require_cloud_gpu_mem_gb:.2f}GB. "
            "Pass --allow-noncloud only for local audit."
        )

    val_rows = load_query_split("val")
    val_by_qid = {r["query_id"]: r for r in val_rows}
    report_label = _slug(args.report_label) or "current_best_residual_audit"
    report_title = str(args.report_title).strip() or "Current Best Residual Audit"
    out_dir = ROOT / "artifacts" / report_label
    docs_path = ROOT / "docs" / f"{report_label}.md"
    cache_path = args.rerank_cache_json or (out_dir / "rerank_cache.json")

    trace_rows = [r for r in _load_trace_rows(args.baseline_trace_csv) if r.get("query_id", "") in val_by_qid]
    cache_used = 0
    if cache_path.exists():
        cache_payload = _read_json(cache_path)
        qwen_pred = {str(k): list(v) for k, v in cache_payload.get("qwen_pred", {}).items()}
        qwen_reranked = {str(k): list(v) for k, v in cache_payload.get("qwen_reranked", {}).items()}
        cache_used = 1
    else:
        doc_lookup = _load_doc_lookup(text_max_chars=900)
        reranker = Qwen3Reranker(
            model_name=args.qwen_model_name,
            device=args.qwen_device,
            batch_size=int(args.qwen_batch_size),
            max_length=int(args.qwen_max_length),
            torch_dtype=args.qwen_torch_dtype,
        )
        qwen_pred, qwen_reranked = _qwen_predictions_from_trace(
            trace_rows=trace_rows,
            val_by_qid=val_by_qid,
            doc_lookup=doc_lookup,
            reranker=reranker,
            candidate_cap=int(args.qwen_candidate_cap),
        )
        _write_json(
            cache_path,
            {
                "baseline_trace_csv": str(args.baseline_trace_csv),
                "qwen_model_name": args.qwen_model_name,
                "qwen_candidate_cap": int(args.qwen_candidate_cap),
                "qwen_pred": qwen_pred,
                "qwen_reranked": qwen_reranked,
            },
        )

    global_priors = ["Art. 100 Abs. 1 BGG"]
    best_pred = _apply_global_priors(qwen_pred, global_priors)
    audit_rows = _build_gold_audit_rows(
        val_rows=val_rows,
        trace_rows=trace_rows,
        qwen_pred=qwen_pred,
        qwen_reranked=qwen_reranked,
        best_pred=best_pred,
        candidate_cap=int(args.qwen_candidate_cap),
        global_priors=global_priors,
    )
    query_rows = _query_summary_rows(audit_rows)
    top_rows = _top_residual_rows(query_rows, limit=int(args.top_query_limit))

    overall_metric = _metric_block(val_rows, best_pred, explicit=None)
    nonexplicit_metric = _metric_block(val_rows, best_pred, explicit=0)
    overall_stage = _stage_summary(audit_rows, explicit=None)
    nonexplicit_stage = _stage_summary(audit_rows, explicit=0)

    recommendation = (
        "retrieval_or_candidate_shaping_first"
        if overall_stage["candidate_stage_share_of_missed"] >= overall_stage["rerank_stage_share_of_missed"]
        else "rerank_or_final_cut_first"
    )

    _write_csv(out_dir / "gold_audit_rows.csv", audit_rows)
    _write_csv(out_dir / "query_residual_rows.csv", query_rows)
    _write_json(
        out_dir / "summary.json",
        {
            "runtime": {
                "report_label": report_label,
                "rerank_cache_json": str(cache_path),
                "rerank_cache_used": int(cache_used),
                "detected_gpu_mem_gb": total_gpu_gb,
                "cloud_gate_threshold_gb": float(args.require_cloud_gpu_mem_gb),
                "cloud_gate_passed": int(cloud_gate_ok),
                "noncloud_override_used": int((not cloud_gate_ok) and bool(args.allow_noncloud)),
                "qwen_model_name": args.qwen_model_name,
                "qwen_candidate_cap": int(args.qwen_candidate_cap),
                "baseline_trace_csv": str(args.baseline_trace_csv),
            },
            "current_best_metric": {"overall": overall_metric, "non_explicit": nonexplicit_metric},
            "drop_stage": {"overall": overall_stage, "non_explicit": nonexplicit_stage},
            "top_residual_queries": top_rows,
            "recommendation": recommendation,
        },
    )

    lines = [
        f"# {report_title}",
        "",
        "## Scope",
        "- Validation-only residual audit for the current best path: Qwen3 rerank + `Art. 100 Abs. 1 BGG` global prior.",
        "- Retrieval pipeline unchanged.",
        "- Goal: split residual misses into candidate-stage loss vs rerank/final-cut loss.",
        f"- Qwen candidate cap: `{int(args.qwen_candidate_cap)}`.",
        f"- Baseline trace CSV: `{args.baseline_trace_csv}`.",
        f"- Rerank cache JSON: `{cache_path}`.",
        "",
        "## Current Best Local Metrics",
        "| subset | strict_f1 | corpus_f1 | final FP |",
        "|---|---:|---:|---:|",
        f"| overall | {overall_metric['strict_f1']:.6f} | {overall_metric['corpus_f1']:.6f} | {overall_metric['final_fp']} |",
        f"| non-explicit | {nonexplicit_metric['strict_f1']:.6f} | {nonexplicit_metric['corpus_f1']:.6f} | {nonexplicit_metric['final_fp']} |",
        "",
        "## Drop Stage Split",
        "| subset | total_gold | final_kept_rate | gold_in_fused_top200_rate | global_prior_rescue_rate | candidate_stage_share_of_missed | rerank_stage_share_of_missed | not_in_fused_top200 | not_in_qwen_input_cap | reranked_too_low | cut_by_dynamic_threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| overall | {overall_stage['total_gold']} | {overall_stage['final_kept_rate']:.6f} | "
            f"{overall_stage['gold_in_fused_top200_rate']:.6f} | {overall_stage['global_prior_rescue_rate']:.6f} | "
            f"{overall_stage['candidate_stage_share_of_missed']:.6f} | {overall_stage['rerank_stage_share_of_missed']:.6f} | "
            f"{overall_stage['not_in_fused_top200']} | {overall_stage['not_in_qwen_input_cap']} | "
            f"{overall_stage['reranked_too_low']} | {overall_stage['cut_by_dynamic_threshold']} |"
        ),
        (
            f"| non-explicit | {nonexplicit_stage['total_gold']} | {nonexplicit_stage['final_kept_rate']:.6f} | "
            f"{nonexplicit_stage['gold_in_fused_top200_rate']:.6f} | {nonexplicit_stage['global_prior_rescue_rate']:.6f} | "
            f"{nonexplicit_stage['candidate_stage_share_of_missed']:.6f} | {nonexplicit_stage['rerank_stage_share_of_missed']:.6f} | "
            f"{nonexplicit_stage['not_in_fused_top200']} | {nonexplicit_stage['not_in_qwen_input_cap']} | "
            f"{nonexplicit_stage['reranked_too_low']} | {nonexplicit_stage['cut_by_dynamic_threshold']} |"
        ),
        "",
        "## Top Residual Queries",
        "| query_id | explicit_subset | kept_rate | miss_total | miss_not_in_fused_top200 | miss_not_in_qwen_input_cap | miss_reranked_too_low | miss_cut_by_dynamic_threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_rows:
        lines.append(
            f"| {row['query_id']} | {row['explicit_subset']} | {row['kept_rate']:.6f} | {row['miss_total']} | "
            f"{row['miss_not_in_fused_top200']} | {row['miss_not_in_qwen_input_cap']} | "
            f"{row['miss_reranked_too_low']} | {row['miss_cut_by_dynamic_threshold']} |"
        )
    lines.extend(
        [
            "",
            "## Next-Step Verdict",
            f"- recommendation: `{recommendation}`",
            "- If candidate-stage misses dominate, prioritize retrieval / candidate shaping diagnostics before more reranker or prior work.",
            "- If rerank-stage misses dominate, prioritize Qwen input shaping, candidate cap, or final-cut calibration before retrieval changes.",
            "",
            "## Artifacts",
            f"- `{(out_dir / 'gold_audit_rows.csv').relative_to(ROOT)}`",
            f"- `{(out_dir / 'query_residual_rows.csv').relative_to(ROOT)}`",
            f"- `{(out_dir / 'summary.json').relative_to(ROOT)}`",
        ]
    )
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

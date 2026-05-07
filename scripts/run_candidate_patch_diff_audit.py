from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from citation_normalizer import normalize_citation
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions

from run_procedural_prior_ablation import _apply_global_priors, _load_prediction_csv


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_block(rows: list[dict], pred_map: dict[str, list[str]]) -> dict:
    strict, strict_rows = evaluate_predictions(rows, pred_map, citation_lookup=None, mode="strict")
    corpus, _ = evaluate_predictions(rows, pred_map, citation_lookup=None, mode="corpus_aware")
    return {
        "strict_f1": float(strict.get("macro_f1", 0.0)),
        "corpus_f1": float(corpus.get("macro_f1", 0.0)),
        "final_fp": int(sum(int(x.get("fp", 0)) for x in strict_rows)),
    }


def _per_query_f1(rows: list[dict], pred_map: dict[str, list[str]]) -> dict[str, float]:
    _, strict_rows = evaluate_predictions(rows, pred_map, citation_lookup=None, mode="strict")
    return {str(r["query_id"]): float(r.get("f1", 0.0)) for r in strict_rows}


def _load_trace_rows(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(r.get("query_id", "")): r for r in csv.DictReader(f)}


def _load_raw_submission(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(r.get("query_id", "")): str(r.get("predicted_citations", "")) for r in csv.DictReader(f)}


def _split_semicolon(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(text).split(";"):
        item = raw.strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _compile_regex(pattern: str | None, aliases: list[str]) -> re.Pattern[str]:
    if pattern:
        return re.compile(pattern, re.I)
    if not aliases:
        aliases = ["ATSG", "IVG", "LAI", "LPGA", "UVG"]
    escaped = [re.escape(x) for x in aliases]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.I)


def _trace_family_set(row: dict | None) -> set[str]:
    if not row:
        return set()
    return set(_split_semicolon(str(row.get("likely_statute_family", ""))))


def _trace_topk_set(row: dict | None, field: str, top_k: int) -> set[str]:
    if not row:
        return set()
    items = [normalize_citation(x) for x in _split_semicolon(str(row.get(field, "")))]
    items = [x for x in items if x]
    if top_k > 0:
        items = items[:top_k]
    return set(items)


def _prediction_diff_rows(
    test_rows: list[dict],
    base_pred: dict[str, list[str]],
    trial_pred: dict[str, list[str]],
    base_trace: dict[str, dict],
    trial_trace: dict[str, dict],
    target_regex: re.Pattern[str],
) -> list[dict]:
    rows: list[dict] = []
    for row in test_rows:
        qid = row["query_id"]
        query = row.get("query", "")
        base_list = list(base_pred.get(qid, []))
        trial_list = list(trial_pred.get(qid, []))
        base_set = set(base_list)
        trial_set = set(trial_list)
        added = [x for x in trial_list if x not in base_set]
        removed = [x for x in base_list if x not in trial_set]
        changed = int(bool(added or removed))

        base_family = _trace_family_set(base_trace.get(qid))
        trial_family = _trace_family_set(trial_trace.get(qid))
        family_added = sorted(trial_family - base_family)
        family_removed = sorted(base_family - trial_family)
        family_shift = int(base_family != trial_family)

        base_fused200 = _trace_topk_set(base_trace.get(qid), "fused_top200", 200)
        trial_fused200 = _trace_topk_set(trial_trace.get(qid), "fused_top200", 200)
        fused200_added = sorted(trial_fused200 - base_fused200)
        fused200_removed = sorted(base_fused200 - trial_fused200)

        rows.append(
            {
                "query_id": qid,
                "is_target_query": int(bool(target_regex.search(query))),
                "query": query,
                "changed": changed,
                "added_count": len(added),
                "removed_count": len(removed),
                "added_citations": ";".join(added),
                "removed_citations": ";".join(removed),
                "base_family": ";".join(sorted(base_family)),
                "trial_family": ";".join(sorted(trial_family)),
                "family_shift": family_shift,
                "family_added": ";".join(family_added),
                "family_removed": ";".join(family_removed),
                "atsg_ivg_family_pollution": int(
                    (not bool(target_regex.search(query)))
                    and bool({"ATSG", "IVG"} & set(family_added))
                ),
                "base_fused200_count": len(base_fused200),
                "trial_fused200_count": len(trial_fused200),
                "fused200_added_count": len(fused200_added),
                "fused200_removed_count": len(fused200_removed),
                "fused200_added_sample": ";".join(fused200_added[:15]),
                "fused200_removed_sample": ";".join(fused200_removed[:15]),
            }
        )
    return rows


def _val_target_deltas(
    val_rows: list[dict],
    base_pred: dict[str, list[str]],
    trial_pred: dict[str, list[str]],
    target_regex: re.Pattern[str],
) -> tuple[dict[str, float], list[dict]]:
    base_f1 = _per_query_f1(val_rows, base_pred)
    trial_f1 = _per_query_f1(val_rows, trial_pred)
    summary = {
        "explicit_target_total": 0,
        "explicit_target_improved": 0,
        "explicit_target_harmed": 0,
        "explicit_target_unchanged": 0,
    }
    rows: list[dict] = []
    for row in val_rows:
        qid = row["query_id"]
        query = row.get("query", "")
        if not target_regex.search(query):
            continue
        base_score = float(base_f1.get(qid, 0.0))
        trial_score = float(trial_f1.get(qid, 0.0))
        delta = round(trial_score - base_score, 6)
        summary["explicit_target_total"] += 1
        if delta > 0:
            summary["explicit_target_improved"] += 1
            status = "improved"
        elif delta < 0:
            summary["explicit_target_harmed"] += 1
            status = "harmed"
        else:
            summary["explicit_target_unchanged"] += 1
            status = "unchanged"
        rows.append(
            {
                "query_id": qid,
                "query": query,
                "baseline_strict_f1": round(base_score, 6),
                "trial_strict_f1": round(trial_score, 6),
                "delta_strict_f1": delta,
                "status": status,
            }
        )
    return summary, rows


def _gate_checks(summary: dict, limits: dict[str, float]) -> list[dict]:
    checks = [
        ("strict_f1_delta_vs_best", ">", float(limits["min_strict_f1_delta"])),
        ("final_fp_delta_vs_best", "<=", float(limits["max_final_fp_delta"])),
        ("explicit_target_improved", ">=", float(limits["min_target_improved"])),
        ("explicit_target_harmed", "<=", float(limits["max_target_harmed"])),
        ("spillover_ratio", "<=", float(limits["max_spillover_ratio"])),
        ("candidate_pool_family_shift_count", "<=", float(limits["max_family_shift_count"])),
    ]
    out: list[dict] = []
    for field, op, threshold in checks:
        actual = float(summary.get(field, 0.0))
        if op == ">":
            passed = actual > threshold
        elif op == ">=":
            passed = actual >= threshold
        else:
            passed = actual <= threshold
        out.append(
            {
                "field": field,
                "operator": op,
                "threshold": threshold,
                "actual": actual,
                "passed": int(passed),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spillover-aware patch-vs-best audit for candidate-stage recall patches.")
    parser.add_argument("--baseline-submission-csv", type=Path, required=True)
    parser.add_argument("--trial-submission-csv", type=Path, required=True)
    parser.add_argument("--baseline-val-pred-csv", type=Path, required=True)
    parser.add_argument("--trial-val-pred-csv", type=Path, required=True)
    parser.add_argument("--baseline-test-trace-csv", type=Path, required=True)
    parser.add_argument("--trial-test-trace-csv", type=Path, required=True)
    parser.add_argument("--common-global-prior", action="append", default=[])
    parser.add_argument("--target-pattern", default=None)
    parser.add_argument("--target-alias", action="append", default=[])
    parser.add_argument("--label", default="patch_vs_best")
    parser.add_argument("--max-spillover-ratio", type=float, default=0.50)
    parser.add_argument("--max-family-shift-count", type=int, default=3)
    parser.add_argument("--max-final-fp-delta", type=int, default=2)
    parser.add_argument("--min-strict-f1-delta", type=float, default=0.0)
    parser.add_argument("--min-target-improved", type=int, default=1)
    parser.add_argument("--max-target-harmed", type=int, default=0)
    args = parser.parse_args()

    label = re.sub(r"[^A-Za-z0-9_]+", "_", str(args.label)).strip("_").lower() or "patch_vs_best"
    out_dir = ROOT / "artifacts" / "candidate_patch_diff_audit" / label
    docs_path = ROOT / "docs" / f"{label}_spillover_audit.md"

    target_regex = _compile_regex(args.target_pattern, args.target_alias)
    common_priors = [normalize_citation(x) for x in args.common_global_prior if normalize_citation(x)]

    val_rows = load_query_split("val")
    test_rows = load_query_split("test")
    base_val = _load_prediction_csv(args.baseline_val_pred_csv)
    trial_val = _load_prediction_csv(args.trial_val_pred_csv)
    if common_priors:
        base_val = _apply_global_priors(base_val, common_priors)
        trial_val = _apply_global_priors(trial_val, common_priors)
    base_test = _load_prediction_csv(args.baseline_submission_csv)
    trial_test = _load_prediction_csv(args.trial_submission_csv)
    base_test_raw = _load_raw_submission(args.baseline_submission_csv)
    trial_test_raw = _load_raw_submission(args.trial_submission_csv)
    base_trace = _load_trace_rows(args.baseline_test_trace_csv)
    trial_trace = _load_trace_rows(args.trial_test_trace_csv)

    base_metric = _metric_block(val_rows, base_val)
    trial_metric = _metric_block(val_rows, trial_val)
    val_target_summary, val_target_rows = _val_target_deltas(val_rows, base_val, trial_val, target_regex)
    diff_rows = _prediction_diff_rows(test_rows, base_test, trial_test, base_trace, trial_trace, target_regex)
    changed_rows = [r for r in diff_rows if int(r["changed"]) == 1]
    raw_changed_qids = [
        qid for qid in sorted(set(base_test_raw) | set(trial_test_raw)) if base_test_raw.get(qid, "") != trial_test_raw.get(qid, "")
    ]

    target_changed_queries = sum(int(r["is_target_query"]) for r in changed_rows)
    non_target_changed_queries = sum(1 - int(r["is_target_query"]) for r in changed_rows)
    changed_total = len(changed_rows)
    spillover_ratio = round(non_target_changed_queries / changed_total, 6) if changed_total else 0.0
    family_shift_count = sum(int(r["family_shift"]) for r in changed_rows)
    family_pollution_count = sum(int(r["atsg_ivg_family_pollution"]) for r in changed_rows)

    summary = {
        "label": label,
        "baseline_submission_csv": str(args.baseline_submission_csv),
        "trial_submission_csv": str(args.trial_submission_csv),
        "baseline_val_pred_csv": str(args.baseline_val_pred_csv),
        "trial_val_pred_csv": str(args.trial_val_pred_csv),
        "baseline_test_trace_csv": str(args.baseline_test_trace_csv),
        "trial_test_trace_csv": str(args.trial_test_trace_csv),
        "target_pattern": target_regex.pattern,
        "common_global_priors": common_priors,
        "baseline_strict_f1": round(base_metric["strict_f1"], 6),
        "trial_strict_f1": round(trial_metric["strict_f1"], 6),
        "strict_f1_delta_vs_best": round(trial_metric["strict_f1"] - base_metric["strict_f1"], 6),
        "baseline_corpus_f1": round(base_metric["corpus_f1"], 6),
        "trial_corpus_f1": round(trial_metric["corpus_f1"], 6),
        "baseline_final_fp": int(base_metric["final_fp"]),
        "trial_final_fp": int(trial_metric["final_fp"]),
        "final_fp_delta_vs_best": int(trial_metric["final_fp"] - base_metric["final_fp"]),
        "raw_changed_query_total": int(len(raw_changed_qids)),
        "raw_changed_qids": raw_changed_qids,
        "target_changed_queries": int(target_changed_queries),
        "non_target_changed_queries": int(non_target_changed_queries),
        "semantic_changed_query_total": int(changed_total),
        "spillover_ratio": spillover_ratio,
        "candidate_pool_family_shift_count": int(family_shift_count),
        "candidate_pool_family_pollution_count": int(family_pollution_count),
        **val_target_summary,
    }

    limits = {
        "max_spillover_ratio": float(args.max_spillover_ratio),
        "max_family_shift_count": int(args.max_family_shift_count),
        "max_final_fp_delta": int(args.max_final_fp_delta),
        "min_strict_f1_delta": float(args.min_strict_f1_delta),
        "min_target_improved": int(args.min_target_improved),
        "max_target_harmed": int(args.max_target_harmed),
    }
    checks = _gate_checks(summary, limits)
    summary["gate_limits"] = limits
    summary["gate_checks"] = checks
    summary["spillover_gate_pass"] = int(all(int(x["passed"]) == 1 for x in checks))

    changed_rows.sort(
        key=lambda r: (
            -int(r["atsg_ivg_family_pollution"]),
            -int(r["family_shift"]),
            -int(r["is_target_query"]),
            r["query_id"],
        )
    )
    val_target_rows.sort(key=lambda r: (r["status"] != "harmed", r["status"] != "unchanged", r["query_id"]))

    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "changed_query_rows.csv", changed_rows)
    _write_csv(out_dir / "val_target_rows.csv", val_target_rows)

    checks_md = []
    for check in checks:
        mark = "PASS" if int(check["passed"]) == 1 else "FAIL"
        checks_md.append(
            f"| {check['field']} | {check['actual']} | {check['operator']} {check['threshold']} | {mark} |"
        )

    top_rows = changed_rows[:10]
    top_md = []
    for row in top_rows:
        top_md.append(
            "| {qid} | {target} | {fam_shift} | {pollute} | {base_fam} | {trial_fam} | {added} | {removed} |".format(
                qid=row["query_id"],
                target=int(row["is_target_query"]),
                fam_shift=int(row["family_shift"]),
                pollute=int(row["atsg_ivg_family_pollution"]),
                base_fam=row["base_family"] or "-",
                trial_fam=row["trial_family"] or "-",
                added=row["added_citations"] or "-",
                removed=row["removed_citations"] or "-",
            )
        )

    docs_path.write_text(
        "\n".join(
            [
                f"# Candidate Patch Spillover Audit: `{label}`",
                "",
                "## Summary",
                f"- `baseline_strict_f1`: `{summary['baseline_strict_f1']:.6f}`",
                f"- `trial_strict_f1`: `{summary['trial_strict_f1']:.6f}`",
                f"- `strict_f1_delta_vs_best`: `{summary['strict_f1_delta_vs_best']:+.6f}`",
                f"- `baseline_final_fp`: `{summary['baseline_final_fp']}`",
                f"- `trial_final_fp`: `{summary['trial_final_fp']}`",
                f"- `final_fp_delta_vs_best`: `{summary['final_fp_delta_vs_best']:+d}`",
                f"- `raw_changed_query_total`: `{summary['raw_changed_query_total']}`",
                f"- `semantic_changed_query_total`: `{summary['semantic_changed_query_total']}`",
                f"- `target_changed_queries`: `{summary['target_changed_queries']}`",
                f"- `non_target_changed_queries`: `{summary['non_target_changed_queries']}`",
                f"- `spillover_ratio`: `{summary['spillover_ratio']:.6f}`",
                f"- `candidate_pool_family_shift_count`: `{summary['candidate_pool_family_shift_count']}`",
                f"- `candidate_pool_family_pollution_count`: `{summary['candidate_pool_family_pollution_count']}`",
                f"- `explicit_target_improved`: `{summary['explicit_target_improved']}`",
                f"- `explicit_target_harmed`: `{summary['explicit_target_harmed']}`",
                f"- `spillover_gate_pass`: `{summary['spillover_gate_pass']}`",
                "",
                "## Gate Checks",
                "| Field | Actual | Threshold | Result |",
                "|---|---:|---:|---|",
                *checks_md,
                "",
                "## Top Changed Queries",
                "| Query ID | Target | Family Shift | ATSG/IVG Pollution | Base Family | Trial Family | Added | Removed |",
                "|---|---:|---:|---:|---|---|---|---|",
                *(top_md or ["| - | - | - | - | - | - | - | - |"]),
                "",
                "## Files",
                f"- Summary JSON: `{(out_dir / 'summary.json').relative_to(ROOT)}`",
                f"- Changed query CSV: `{(out_dir / 'changed_query_rows.csv').relative_to(ROOT)}`",
                f"- Validation target CSV: `{(out_dir / 'val_target_rows.csv').relative_to(ROOT)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

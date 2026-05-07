from __future__ import annotations

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

from run_procedural_prior_ablation import (
    _apply_global_priors,
    _apply_rule_priors,
    _apply_topk_global_priors,
    _load_prediction_csv,
)
from run_query_family_prior_ablation import _apply_family_priors


def _metric_block(rows: list[dict], pred_map: dict[str, list[str]]) -> dict:
    strict, strict_rows = evaluate_predictions(rows, pred_map, citation_lookup=None, mode="strict")
    corpus, _ = evaluate_predictions(rows, pred_map, citation_lookup=None, mode="corpus_aware")
    pred_counts = [len(pred_map.get(r["query_id"], [])) for r in rows]
    return {
        "strict_f1": float(strict.get("macro_f1", 0.0)),
        "corpus_f1": float(corpus.get("macro_f1", 0.0)),
        "final_fp": int(sum(int(x.get("fp", 0)) for x in strict_rows)),
        "avg_pred_count": round(sum(pred_counts) / len(pred_counts), 4) if pred_counts else 0.0,
    }


def _gold_sets(rows: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in rows:
        out[row["query_id"]] = {
            normalize_citation(x)
            for x in row.get("gold_citation_list", [])
            if normalize_citation(x)
        }
    return out


def _added_metrics(
    rows: list[dict],
    base_pred: dict[str, list[str]],
    trial_pred: dict[str, list[str]],
) -> dict:
    gold = _gold_sets(rows)
    added_tp = 0
    added_fp = 0
    changed_queries = 0
    added_total = 0
    added_hits_by_qid: dict[str, list[str]] = {}
    added_fp_by_qid: dict[str, list[str]] = {}
    for row in rows:
        qid = row["query_id"]
        base = set(base_pred.get(qid, []))
        trial = list(trial_pred.get(qid, []))
        added = [x for x in trial if x not in base]
        if added:
            changed_queries += 1
        for c in added:
            added_total += 1
            if c in gold.get(qid, set()):
                added_tp += 1
                added_hits_by_qid.setdefault(qid, []).append(c)
            else:
                added_fp += 1
                added_fp_by_qid.setdefault(qid, []).append(c)
    precision = added_tp / added_total if added_total else 0.0
    return {
        "val_changed_queries": int(changed_queries),
        "val_added_total": int(added_total),
        "val_added_tp": int(added_tp),
        "val_added_fp": int(added_fp),
        "val_added_precision": round(float(precision), 6),
        "val_added_hit_queries": int(len(added_hits_by_qid)),
        "val_added_fp_queries": int(len(added_fp_by_qid)),
        "val_added_hits_by_qid": {k: ";".join(v) for k, v in sorted(added_hits_by_qid.items())},
        "val_added_fp_by_qid": {k: ";".join(v) for k, v in sorted(added_fp_by_qid.items())},
    }


def _test_change_metrics(base_pred: dict[str, list[str]], trial_pred: dict[str, list[str]]) -> dict:
    changed = []
    added_total = 0
    removed_total = 0
    for qid in sorted(trial_pred):
        base = list(base_pred.get(qid, []))
        trial = list(trial_pred.get(qid, []))
        added = [x for x in trial if x not in set(base)]
        removed = [x for x in base if x not in set(trial)]
        if added or removed:
            changed.append(qid)
        added_total += len(added)
        removed_total += len(removed)
    return {
        "test_changed_queries": int(len(changed)),
        "test_added_total": int(added_total),
        "test_removed_total": int(removed_total),
        "test_changed_qids": ";".join(changed),
    }


def _gate_score(delta_from_best: float, delta_fp_from_best: int, test_changed_queries: int, val_added_precision: float) -> float:
    coverage = min(1.0, max(0.0, float(test_changed_queries) / 10.0))
    precision_factor = min(1.0, max(0.0, float(val_added_precision) / 0.8))
    fp_penalty = max(0, int(delta_fp_from_best) - 3) * 0.005
    return round(float(delta_from_best) * coverage * precision_factor - fp_penalty, 6)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = []
    for row in rows:
        flat = {}
        for k, v in row.items():
            if isinstance(v, dict):
                flat[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
            else:
                flat[k] = v
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    out_dir = ROOT / "artifacts" / "public_score_proxy_audit"
    docs_report = ROOT / "docs" / "public_score_proxy_audit.md"

    val_rows = load_query_split("val")
    qwen_val = _load_prediction_csv(ROOT / "artifacts" / "qwen3_reranker_module_ablation" / "val_predictions_qwen3_cap80.csv")
    old_val = _load_prediction_csv(ROOT / "outputs" / "train_structq_ablation" / "val_predictions_silver_baseline_v0.csv")
    qwen_test = _load_prediction_csv(ROOT / "release" / "submission_qwen3_court_gate_v1" / "submission.csv")

    variants: list[dict] = []

    def add_variant(
        name: str,
        val_pred: dict[str, list[str]],
        test_pred: dict[str, list[str]],
        public_score: float | None,
        public_label: str,
        submitted: bool,
    ) -> None:
        variants.append(
            {
                "name": name,
                "val_pred": val_pred,
                "test_pred": test_pred,
                "public_score": public_score,
                "public_label": public_label,
                "submitted": submitted,
            }
        )

    add_variant(
        name="old_laws_first_token_overlap",
        val_pred=old_val,
        test_pred=_load_prediction_csv(ROOT / "release" / "submission_laws_first_v1" / "submission.csv"),
        public_score=0.01357,
        public_label="submitted_2026-04-22",
        submitted=True,
    )
    add_variant(
        name="qwen3_causal_cap80",
        val_pred=qwen_val,
        test_pred=qwen_test,
        public_score=0.04272,
        public_label="submitted_2026-04-23_03:45",
        submitted=True,
    )
    add_variant(
        name="qwen3_plus_art100",
        val_pred=_apply_global_priors(qwen_val, ["Art. 100 Abs. 1 BGG"]),
        test_pred=_apply_global_priors(qwen_test, ["Art. 100 Abs. 1 BGG"]),
        public_score=0.08960,
        public_label="submitted_2026-04-23_04:35",
        submitted=True,
    )
    add_variant(
        name="qwen3_plus_art100_social_only",
        val_pred=_apply_rule_priors(load_query_split("val"), qwen_val, ["global_art100_bgg", "social_insurance_core"]),
        test_pred=_apply_rule_priors(load_query_split("test"), qwen_test, ["global_art100_bgg", "social_insurance_core"]),
        public_score=0.08960,
        public_label="submitted_2026-04-23_05:39",
        submitted=True,
    )
    add_variant(
        name="qwen3_plus_art100_safe_rules",
        val_pred=_apply_rule_priors(
            load_query_split("val"),
            qwen_val,
            ["global_art100_bgg", "right_to_be_heard_bv29", "criminal_core", "social_insurance_core", "schkg_bankruptcy_core"],
        ),
        test_pred=_apply_rule_priors(
            load_query_split("test"),
            qwen_test,
            ["global_art100_bgg", "right_to_be_heard_bv29", "criminal_core", "social_insurance_core", "schkg_bankruptcy_core"],
        ),
        public_score=0.08946,
        public_label="submitted_2026-04-23_04:40_v2_safe",
        submitted=True,
    )
    add_variant(
        name="qwen3_plus_art100_social_rtbh_family_child_failed",
        val_pred=_apply_family_priors(
            load_query_split("val"),
            qwen_val,
            enabled_families={"social_insurance", "right_to_be_heard", "family_child"},
            include_global_bgg=True,
        ),
        test_pred=_load_prediction_csv(
            ROOT / "release" / "query_family_prior_candidates" / "submission_social_insurance+right_to_be_heard+family_child.csv"
        ),
        public_score=0.08939,
        public_label="submitted_2026-04-23_08:48_strictpass_failed",
        submitted=True,
    )
    for k in [1, 2, 3, 4, 5, 6]:
        add_variant(
            name=f"qwen3_top{k}_plus_art100_not_submitted",
            val_pred=_apply_topk_global_priors(qwen_val, k, ["Art. 100 Abs. 1 BGG"]),
            test_pred=_apply_topk_global_priors(qwen_test, k, ["Art. 100 Abs. 1 BGG"]),
            public_score=None,
            public_label="not_submitted",
            submitted=False,
        )

    art100_val = _apply_global_priors(qwen_val, ["Art. 100 Abs. 1 BGG"])
    art100_test = _apply_global_priors(qwen_test, ["Art. 100 Abs. 1 BGG"])
    qwen_metric = _metric_block(val_rows, qwen_val)
    art100_metric = _metric_block(val_rows, art100_val)

    rows: list[dict] = []
    for v in variants:
        m = _metric_block(val_rows, v["val_pred"])
        add_qwen = _added_metrics(val_rows, qwen_val, v["val_pred"])
        tchg_qwen = _test_change_metrics(qwen_test, v["test_pred"])
        add_art100 = _added_metrics(val_rows, art100_val, v["val_pred"])
        tchg_art100 = _test_change_metrics(art100_test, v["test_pred"])
        public_delta_from_qwen = None
        public_delta_from_art100 = None
        if v["public_score"] is not None:
            public_delta_from_qwen = round(float(v["public_score"]) - 0.04272, 6)
            public_delta_from_art100 = round(float(v["public_score"]) - 0.08960, 6)
        local_delta_from_qwen = round(float(m["strict_f1"]) - float(qwen_metric["strict_f1"]), 6)
        local_delta_from_art100 = round(float(m["strict_f1"]) - float(art100_metric["strict_f1"]), 6)
        delta_fp_from_art100 = int(m["final_fp"] - art100_metric["final_fp"])
        row = {
            "variant": v["name"],
            "submitted": int(bool(v["submitted"])),
            "public_score": "" if v["public_score"] is None else f"{float(v['public_score']):.5f}",
            "public_delta_from_qwen": "" if public_delta_from_qwen is None else f"{public_delta_from_qwen:.5f}",
            "public_delta_from_art100": "" if public_delta_from_art100 is None else f"{public_delta_from_art100:.5f}",
            "local_strict_f1": round(float(m["strict_f1"]), 6),
            "local_delta_from_qwen": local_delta_from_qwen,
            "local_delta_from_art100": local_delta_from_art100,
            "local_final_fp": int(m["final_fp"]),
            "delta_fp_from_qwen": int(m["final_fp"] - qwen_metric["final_fp"]),
            "delta_fp_from_art100": delta_fp_from_art100,
            "avg_pred_count": float(m["avg_pred_count"]),
            "val_added_tp_vs_qwen": int(add_qwen["val_added_tp"]),
            "val_added_fp_vs_qwen": int(add_qwen["val_added_fp"]),
            "val_added_precision_vs_qwen": float(add_qwen["val_added_precision"]),
            "test_changed_queries_vs_qwen": int(tchg_qwen["test_changed_queries"]),
            "test_added_total_vs_qwen": int(tchg_qwen["test_added_total"]),
            "test_removed_total_vs_qwen": int(tchg_qwen["test_removed_total"]),
            "test_changed_qids_vs_qwen": tchg_qwen["test_changed_qids"],
            "val_added_tp_vs_art100": int(add_art100["val_added_tp"]),
            "val_added_fp_vs_art100": int(add_art100["val_added_fp"]),
            "val_added_precision_vs_art100": float(add_art100["val_added_precision"]),
            "test_changed_queries_vs_art100": int(tchg_art100["test_changed_queries"]),
            "test_added_total_vs_art100": int(tchg_art100["test_added_total"]),
            "test_removed_total_vs_art100": int(tchg_art100["test_removed_total"]),
            "test_changed_qids_vs_art100": tchg_art100["test_changed_qids"],
            "val_added_hits_by_qid_vs_art100": add_art100["val_added_hits_by_qid"],
            "val_added_fp_by_qid_vs_art100": add_art100["val_added_fp_by_qid"],
            "gate_score_vs_art100": _gate_score(
                delta_from_best=local_delta_from_art100,
                delta_fp_from_best=delta_fp_from_art100,
                test_changed_queries=int(tchg_art100["test_changed_queries"]),
                val_added_precision=float(add_art100["val_added_precision"]),
            ),
            "public_label": v["public_label"],
        }
        rows.append(row)

    _write_csv(out_dir / "public_score_proxy_audit.csv", rows)
    (out_dir / "public_score_proxy_audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    submitted = [r for r in rows if int(r["submitted"]) == 1]
    lines = [
        "# Public Score Proxy Audit",
        "",
        "## Goal",
        "- Identify which local metrics moved with public score from `0.04272` to `0.08960`.",
        "- Avoid spending Kaggle submissions on variants that only overfit the 10-row validation set.",
        "",
        "## Submitted Runs",
        "| variant | public | local_strict_f1 | local_final_fp | added TP/FP vs Qwen | added TP/FP vs Art100 | test changed vs Art100 | gate_score_vs_art100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in submitted:
        lines.append(
            f"| {r['variant']} | {r['public_score']} | {r['local_strict_f1']:.6f} | {r['local_final_fp']} | "
            f"{r['val_added_tp_vs_qwen']}/{r['val_added_fp_vs_qwen']} | "
            f"{r['val_added_tp_vs_art100']}/{r['val_added_fp_vs_art100']} | "
            f"{r['test_changed_queries_vs_art100']} | {r['gate_score_vs_art100']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Local Signals That Actually Moved With Public",
            "- Qwen3 rerank improved local strict_f1 and reduced final FP; public also rose (`0.01357 -> 0.04272`).",
            "- `Art. 100 Abs. 1 BGG` had high validation added precision: 9 added TP / 1 added FP, and affected all 40 test queries; public rose (`0.04272 -> 0.08960`).",
            "- `social_insurance_core` had perfect validation additions versus Art100 on one val query, but changed only 2 test queries versus Art100; public did not move.",
            "- The wider safe-rule pack had the highest local strict_f1, but added more FP and reduced public slightly; local strict_f1 alone is not a reliable submit criterion.",
            "",
            "## Submit Gate",
            "- Track local strict_f1, but never use it alone.",
            "- Require `val_added_precision >= 0.70` for prior additions.",
            "- Prefer `delta_fp_from_art100 <= 3`; reject or combine later if FP grows more than that.",
            "- Require enough test coverage for a standalone submission: normally `test_changed_queries >= 8`; if fewer, hold until combined with other high-confidence rules.",
            "- For reranker changes, track `local_strict_f1`, `final_fp`, and `reranked_too_low`; for prior/final-count changes, track added TP/FP precision and test coverage.",
            "",
            "## Not-Submitted Final Count Candidates",
            "| variant | local_strict_f1 | local_final_fp | avg_pred_count | test_changed_queries | gate_score_vs_art100 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        if "not_submitted" not in r["variant"]:
            continue
        lines.append(
            f"| {r['variant']} | {r['local_strict_f1']:.6f} | {r['local_final_fp']} | "
            f"{r['avg_pred_count']:.4f} | {r['test_changed_queries_vs_art100']} | {r['gate_score_vs_art100']:.6f} |"
        )

    docs_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(out_dir / "public_score_proxy_audit.csv"), "report": str(docs_report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

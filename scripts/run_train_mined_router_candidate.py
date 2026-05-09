from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from legal_ir.normalization import split_citations

from mine_train_institution_router import _laws_exact_set, _write_csv, _write_prediction_csv, mine_rules
from run_train_mined_cluster_router import (  # noqa: E402
    _laws_prefix_map,
    predict_with_selected_rules,
    select_robust_clusters,
)


def _dedup(values: list[str], cap: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
            if cap is not None and len(out) >= cap:
                break
    return out


def _article_family(citation: str) -> tuple[str, str] | None:
    parts = citation.split()
    if len(parts) < 3 or parts[0] != "Art.":
        return None
    article = parts[1].lower()
    family = parts[-1].replace("-", "").upper()
    return article, family


def _new_article_family(explicit: list[str], base: list[str]) -> list[str]:
    base_keys = {key for citation in base if (key := _article_family(citation))}
    return [citation for citation in explicit if _article_family(citation) not in base_keys]


def _read_predictions(path: Path) -> dict[str, list[str]]:
    pred: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            pred[row["query_id"]] = split_citations(row.get("predicted_citations", ""))
    return pred


def _merge(
    base: list[str],
    router: list[str],
    explicit: list[str],
    strategy: str,
    max_predictions: int,
) -> list[str]:
    if strategy == "append":
        return _dedup(base + router, max_predictions)
    if strategy == "fill_empty":
        return _dedup(router if not base else base, max_predictions)
    if strategy == "explicit_only_append":
        return _dedup(base + explicit, max_predictions)
    if strategy == "explicit_new_article_append":
        return _dedup(base + _new_article_family(explicit, base), max_predictions)
    if strategy == "router_only":
        return _dedup(router, max_predictions)
    raise ValueError(f"unknown merge strategy: {strategy}")


def _run_router(rows: list[dict], args: argparse.Namespace) -> tuple[dict[str, list[str]], list[dict], dict]:
    train_rows = load_query_split("train")
    laws_exact = _laws_exact_set()
    laws_prefix = _laws_prefix_map(laws_exact)
    rules = mine_rules(
        train_rows=train_rows,
        laws_exact=laws_exact,
        max_ngram=args.max_ngram,
        min_phrase_support=args.min_phrase_support,
        min_citation_support=args.min_citation_support,
        min_precision=args.min_precision,
        max_cluster_size=args.mine_max_cluster_size,
        require_legal_anchor=not args.allow_nonlegal_phrases,
    )
    selected_rules, selected_clusters = select_robust_clusters(
        rules=rules,
        min_cluster_phrases=args.min_cluster_phrases,
        min_cluster_support=args.min_cluster_support,
        min_cluster_precision=args.min_cluster_precision,
        max_citations_per_cluster=args.max_citations_per_cluster,
        max_clusters=args.max_clusters,
        max_rules_per_cluster=args.max_rules_per_cluster_selected,
    )
    pred, trace = predict_with_selected_rules(
        rows=rows,
        rules=selected_rules,
        laws_exact=laws_exact,
        laws_prefix=laws_prefix,
        max_ngram=args.max_ngram,
        top_k=args.router_top_k,
        max_rules_per_query=args.max_rules_per_query,
        explicit_prefix_cap=args.explicit_prefix_cap,
        require_legal_anchor=not args.allow_nonlegal_phrases,
        use_query_expansion=not args.disable_query_expansion,
        add_explicit_citations=not args.disable_explicit_citation_floor,
    )
    summary = {
        "candidate_rule_count": len(rules),
        "selected_cluster_count": len(selected_clusters),
        "selected_rule_count": len(selected_rules),
    }
    return pred, trace, summary


def _merge_maps(
    rows: list[dict],
    base_pred: dict[str, list[str]],
    router_pred: dict[str, list[str]],
    router_trace: list[dict],
    strategy: str,
    max_predictions: int,
) -> tuple[dict[str, list[str]], list[dict]]:
    merged: dict[str, list[str]] = {}
    diff_rows: list[dict] = []
    trace_by_qid = {row["query_id"]: row for row in router_trace}
    for row in rows:
        qid = row["query_id"]
        base = base_pred.get(qid, [])
        router = router_pred.get(qid, [])
        explicit = split_citations(trace_by_qid.get(qid, {}).get("explicit_citations", ""))
        final = _merge(base, router, explicit, strategy, max_predictions)
        merged[qid] = final
        if final != base:
            diff_rows.append(
                {
                    "query_id": qid,
                    "base_count": len(base),
                    "router_count": len(router),
                    "explicit_count": len(explicit),
                    "final_count": len(final),
                    "base_predictions": ";".join(base),
                    "router_predictions": ";".join(router),
                    "explicit_predictions": ";".join(explicit),
                    "final_predictions": ";".join(final),
                }
            )
    return merged, diff_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge train-mined router evidence into an existing automatic baseline.")
    parser.add_argument("--base-val-pred-csv", type=Path, default=ROOT / "artifacts" / "explicit_prefix_rescue_conjunction_top3_v8" / "val_predictions.csv")
    parser.add_argument("--base-test-submission-csv", type=Path, default=ROOT / "release" / "submission_explicit_prefix_rescue_conjunction_top3_v8" / "submission.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "train_mined_router_candidate_v1")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release" / "submission_train_mined_router_candidate_v1")
    parser.add_argument(
        "--merge-strategy",
        choices=["append", "fill_empty", "explicit_only_append", "explicit_new_article_append", "router_only"],
        default="append",
    )
    parser.add_argument("--max-predictions", type=int, default=14)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--min-phrase-support", type=int, default=3)
    parser.add_argument("--min-citation-support", type=int, default=2)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument("--mine-max-cluster-size", type=int, default=10)
    parser.add_argument("--min-cluster-phrases", type=int, default=4)
    parser.add_argument("--min-cluster-support", type=int, default=4)
    parser.add_argument("--min-cluster-precision", type=float, default=0.5)
    parser.add_argument("--max-citations-per-cluster", type=int, default=10)
    parser.add_argument("--max-clusters", type=int, default=160)
    parser.add_argument("--max-rules-per-cluster-selected", type=int, default=10)
    parser.add_argument("--router-top-k", type=int, default=10)
    parser.add_argument("--max-rules-per-query", type=int, default=10)
    parser.add_argument("--explicit-prefix-cap", type=int, default=4)
    parser.add_argument("--allow-nonlegal-phrases", action="store_true")
    parser.add_argument("--disable-query-expansion", action="store_true")
    parser.add_argument("--disable-explicit-citation-floor", action="store_true")
    args = parser.parse_args()

    val_rows = load_query_split("val")
    test_rows = load_query_split("test")
    base_val = _read_predictions(args.base_val_pred_csv)
    base_test = _read_predictions(args.base_test_submission_csv)

    router_val, val_router_trace, router_summary = _run_router(val_rows, args)
    merged_val, val_diff = _merge_maps(
        val_rows,
        base_val,
        router_val,
        val_router_trace,
        args.merge_strategy,
        args.max_predictions,
    )
    base_summary, base_eval = evaluate_predictions(val_rows, base_val, mode="strict")
    trial_summary, trial_eval = evaluate_predictions(val_rows, merged_val, mode="strict")

    router_test, test_router_trace, _ = _run_router(test_rows, args)
    merged_test, test_diff = _merge_maps(
        test_rows,
        base_test,
        router_test,
        test_router_trace,
        args.merge_strategy,
        args.max_predictions,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.release_dir.mkdir(parents=True, exist_ok=True)
    _write_prediction_csv(args.out_dir / "val_predictions.csv", val_rows, merged_val)
    _write_prediction_csv(args.release_dir / "submission.csv", test_rows, merged_test)
    _write_csv(args.out_dir / "val_router_trace.csv", val_router_trace)
    _write_csv(args.out_dir / "test_router_trace.csv", test_router_trace)
    _write_csv(args.out_dir / "val_diff.csv", val_diff)
    _write_csv(args.out_dir / "test_diff.csv", test_diff)
    _write_csv(args.out_dir / "base_val_eval_per_query.csv", base_eval)
    _write_csv(args.out_dir / "trial_val_eval_per_query.csv", trial_eval)

    summary = {
        "base_val_strict_f1": base_summary["macro_f1"],
        "trial_val_strict_f1": trial_summary["macro_f1"],
        "delta_val_strict_f1": round(trial_summary["macro_f1"] - base_summary["macro_f1"], 6),
        "merge_strategy": args.merge_strategy,
        "max_predictions": args.max_predictions,
        "changed_val_query_count": len(val_diff),
        "changed_test_query_count": len(test_diff),
        "router": router_summary,
        "release_submission": str(args.release_dir / "submission.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

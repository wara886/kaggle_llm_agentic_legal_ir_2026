from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from law_family import issue_phrase_groups, likely_statute_families
from legal_ir.data_loader import load_query_split
from retrieval_rules import RuleCitationRetriever


DEFAULT_BASE_SUBMISSION = ROOT / "release" / "submission_qwen3_bgg100_prior_v1" / "submission.csv"
DEFAULT_TRACE_CANDIDATES = [
    ROOT / "outputs" / "current_code_mainline_control" / "test_seed_trace_silver_baseline_v0.csv",
    ROOT / "outputs" / "explicit_stpo_issue_seed_patch_mainline_v2" / "test_seed_trace_silver_baseline_v0.csv",
    ROOT / "outputs" / "explicit_stpo_bv_issue_seed_patch_mainline" / "test_seed_trace_silver_baseline_v0.csv",
    ROOT / "outputs" / "explicit_stpo_bv_issue_patch_mainline" / "test_seed_trace_silver_baseline_v0.csv",
]


CLUSTER_RULES: list[tuple[str, list[str]]] = [
    ("schkg_bankruptcy_enforcement", ["schkg", "bankruptcy", "payment order", "opposition", "debt enforcement", "forced sale", "security for future maintenance"]),
    ("stpo_detention_procedure", ["stpo", "pretrial detention", "pre-trial detention", "detention", "collusion", "flight risk", "coercive measures", "sufficient suspicion"]),
    ("social_ivg_atsg_explicit", ["ivg", "lai", "atsg", "lpga", "invalidity insurance", "social insurance", "vocational rehabilitation", "earning capacity"]),
    ("zgb_child_family", ["custody", "visitation", "overnight", "child support", "maintenance", "best interests", "children", "parent"]),
    ("zgb_inheritance_possession", ["will", "testament", "heir", "estate", "donation", "gift", "good faith", "possessor", "ownership"]),
    ("or_bank_payment_orders", ["bank", "forged", "transfer instructions", "signature", "statement-hold", "account holder", "gross negligence"]),
    ("or_contract_liability", ["contract", "lease", "remission", "liability", "damages", "defects", "duty of care", "gratuitous"]),
    ("svg_traffic_liability", ["svg", "road traffic", "cyclist", "vehicle", "gross negligence", "traffic", "insurer"]),
    ("ip_trade_secret_provisional", ["copyright", "trade secret", "unfair competition", "interim relief", "provisional measures", "injunction"]),
    ("aig_migration", ["aig", "foreign national", "residence permit", "deportation", "asylum", "migration"]),
    ("zpo_civil_procedure", ["zpo", "civil procedure", "testimony", "evidence", "pleaded", "admissible", "appeal"]),
    ("iprg_private_international", ["iprg", "private international", "foreign law", "applicable law", "cross-border"]),
]


def _split_joined(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").split(";"):
        item = normalize_citation(raw)
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _load_prediction_csv(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return {str(r.get("query_id", "")): _split_joined(str(r.get("predicted_citations", ""))) for r in csv.DictReader(f)}


def _load_trace_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return {str(r.get("query_id", "")): r for r in csv.DictReader(f)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _classify_query(query: str, explicit_patterns: list[str], families: list[str]) -> tuple[str, str]:
    text = (query or "").lower()
    scores: list[tuple[str, int, list[str]]] = []
    explicit_text = " ".join(explicit_patterns).lower()
    family_text = " ".join(families).lower()
    for cluster, cues in CLUSTER_RULES:
        hits = []
        score = 0
        for cue in cues:
            c = cue.lower()
            if c in text or c in explicit_text or c in family_text:
                hits.append(cue)
                score += 3 if c in explicit_text else 1
        if score:
            scores.append((cluster, score, hits))
    if not scores:
        return "other_mixed", ""
    scores.sort(key=lambda x: (x[1], len(x[2]), x[0]), reverse=True)
    return scores[0][0], ";".join(scores[0][2][:8])


def _release_submissions() -> list[Path]:
    paths = sorted((ROOT / "release").glob("submission_*/submission.csv"))
    return [p for p in paths if p.exists()]


def _variant_change_summary(base_pred: dict[str, list[str]]) -> dict[str, dict]:
    variants = _release_submissions()
    changed_by_qid: dict[str, list[str]] = defaultdict(list)
    added_by_qid: dict[str, list[str]] = defaultdict(list)
    removed_by_qid: dict[str, list[str]] = defaultdict(list)
    for path in variants:
        label = path.parent.name.replace("submission_", "")
        pred = _load_prediction_csv(path)
        if not pred or path == DEFAULT_BASE_SUBMISSION:
            continue
        for qid in sorted(set(base_pred) | set(pred)):
            base = base_pred.get(qid, [])
            trial = pred.get(qid, [])
            if base == trial:
                continue
            changed_by_qid[qid].append(label)
            bset = set(base)
            tset = set(trial)
            for c in trial:
                if c not in bset:
                    added_by_qid[qid].append(f"{label}:{c}")
            for c in base:
                if c not in tset:
                    removed_by_qid[qid].append(f"{label}:{c}")
    return {
        qid: {
            "variant_changed_count": len(labels),
            "variant_changed_labels": ";".join(labels[:12]),
            "variant_added_sample": ";".join(added_by_qid.get(qid, [])[:12]),
            "variant_removed_sample": ";".join(removed_by_qid.get(qid, [])[:12]),
        }
        for qid, labels in changed_by_qid.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit test.csv query clusters and current prediction/trace coverage.")
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--trace-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "test_query_cluster_audit")
    parser.add_argument("--doc-path", type=Path, default=ROOT / "docs" / "test_query_cluster_audit.md")
    args = parser.parse_args()

    test_rows = load_query_split("test")
    base_pred = _load_prediction_csv(args.base_submission)
    variant_summary = _variant_change_summary(base_pred)
    trace_path = args.trace_csv
    if trace_path is None:
        trace_path = next((p for p in DEFAULT_TRACE_CANDIDATES if p.exists()), None)
    trace_rows = _load_trace_csv(trace_path) if trace_path else {}

    rows: list[dict] = []
    for row in test_rows:
        qid = str(row["query_id"])
        query = str(row.get("query", ""))
        explicit_patterns = RuleCitationRetriever.extract_patterns(query)
        families = likely_statute_families(query, max_families=3, min_score=4)
        groups = issue_phrase_groups(query, families, max_groups=5) if families else []
        cluster, matched_cues = _classify_query(query, explicit_patterns, families)
        tr = trace_rows.get(qid, {})
        base_items = base_pred.get(qid, [])
        var = variant_summary.get(qid, {})
        rows.append(
            {
                "query_id": qid,
                "cluster": cluster,
                "matched_cues": matched_cues,
                "explicit_pattern_count": len(explicit_patterns),
                "explicit_patterns": ";".join(explicit_patterns[:12]),
                "likely_families_from_query": ";".join(families),
                "issue_groups_from_query": ";".join(groups),
                "trace_likely_family": str(tr.get("likely_statute_family", "")),
                "trace_issue_groups": str(tr.get("issue_phrase_groups", "")),
                "trace_route_label": str(tr.get("route_label", "")),
                "trace_seed_enabled": str(tr.get("explicit_issue_seed_citations_enabled", "")),
                "base_pred_count": len(base_items),
                "base_pred": ";".join(base_items),
                "variant_changed_count": int(var.get("variant_changed_count", 0)),
                "variant_changed_labels": str(var.get("variant_changed_labels", "")),
                "variant_added_sample": str(var.get("variant_added_sample", "")),
                "query_preview": query[:360].replace("\n", " "),
            }
        )

    cluster_counter = Counter(r["cluster"] for r in rows)
    family_counter = Counter()
    explicit_counter = 0
    for r in rows:
        explicit_counter += int(r["explicit_pattern_count"] > 0)
        for fam in str(r["likely_families_from_query"]).split(";"):
            if fam:
                family_counter[fam] += 1

    cluster_rows = []
    for cluster, count in cluster_counter.most_common():
        scoped = [r for r in rows if r["cluster"] == cluster]
        cluster_rows.append(
            {
                "cluster": cluster,
                "query_count": count,
                "explicit_count": sum(int(r["explicit_pattern_count"] > 0) for r in scoped),
                "variant_changed_queries": sum(int(r["variant_changed_count"] > 0) for r in scoped),
                "qids": ";".join(r["query_id"] for r in scoped),
            }
        )

    out_dir = args.out_dir
    _write_csv(out_dir / "test_query_clusters.csv", rows)
    _write_csv(out_dir / "cluster_summary.csv", cluster_rows)
    summary = {
        "test_query_count": len(test_rows),
        "base_submission": str(args.base_submission),
        "trace_csv": str(trace_path) if trace_path else "",
        "explicit_query_count": explicit_counter,
        "cluster_counts": dict(cluster_counter.most_common()),
        "family_counts": dict(family_counter.most_common()),
        "release_variant_count": len(_release_submissions()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Test Query Cluster Audit",
        "",
        "## Summary",
        f"- test queries: `{len(test_rows)}`",
        f"- explicit citation queries: `{explicit_counter}`",
        f"- base submission: `{_rel(args.base_submission)}`",
        f"- trace csv: `{_rel(trace_path) if trace_path else trace_path}`",
        "",
        "## Cluster Counts",
        "| cluster | count | explicit | variant-changed queries | qids |",
        "|---|---:|---:|---:|---|",
    ]
    for r in cluster_rows:
        lines.append(
            f"| {r['cluster']} | {r['query_count']} | {r['explicit_count']} | {r['variant_changed_queries']} | {r['qids']} |"
        )
    lines.extend(
        [
            "",
            "## Family Counts",
            "| family | count |",
            "|---|---:|",
        ]
    )
    for fam, count in family_counter.most_common():
        lines.append(f"| {fam} | {count} |")
    lines.extend(
        [
            "",
            "## Next Work Queue",
            "- Prioritize clusters with multiple test queries and low prior/public stability: candidate-stage recall only, then Qwen rerank.",
            "- Avoid broad social-insurance family expansion; keep IVG/ATSG explicit-only.",
            "- Use `test_query_clusters.csv` to select target qids and audit spillover before submission.",
            "",
            "## Artifacts",
            f"- `{_rel(out_dir / 'summary.json')}`",
            f"- `{_rel(out_dir / 'cluster_summary.csv')}`",
            f"- `{_rel(out_dir / 'test_query_clusters.csv')}`",
        ]
    )
    args.doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

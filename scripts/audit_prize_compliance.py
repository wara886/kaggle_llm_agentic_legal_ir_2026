from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


QUERY_ID_RE = re.compile(r"\b(?:test|val|train)_\d{3,4}\b")
PATCH_TABLE_RE = re.compile(r"\b(PATCH_SETS|query_id\s*->|qid\s*->|targeted_test_patch)\b", re.I)
TEST_GOLD_RE = re.compile(r"\bgold_citations\b.*\btest\.csv\b|\btest\.csv\b.*\bgold_citations\b", re.I | re.S)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_submission(path: Path) -> tuple[list[dict], list[dict]]:
    issues: list[dict] = []
    if not path.exists():
        return [], [{"severity": "fail", "check": "submission_exists", "message": f"Missing submission: {_rel(path)}"}]

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        issues.append({"severity": "fail", "check": "submission_nonempty", "message": "Submission has no rows."})
        return rows, issues

    expected_cols = {"query_id", "predicted_citations"}
    cols = set(rows[0].keys())
    if not expected_cols.issubset(cols):
        issues.append(
            {
                "severity": "fail",
                "check": "submission_columns",
                "message": f"Expected columns {sorted(expected_cols)}, got {sorted(cols)}.",
            }
        )

    qids = [row.get("query_id", "").strip() for row in rows]
    if len(qids) != len(set(qids)):
        issues.append({"severity": "fail", "check": "duplicate_query_ids", "message": "Submission contains duplicate query IDs."})

    empty = [qid for qid, row in zip(qids, rows) if not row.get("predicted_citations", "").strip()]
    if empty:
        issues.append({"severity": "fail", "check": "empty_predictions", "message": f"Empty predictions: {empty[:10]}."})

    duplicate_citations = []
    for row in rows:
        citations = [x.strip() for x in row.get("predicted_citations", "").split(";") if x.strip()]
        if len(citations) != len(set(citations)):
            duplicate_citations.append(row.get("query_id", ""))
    if duplicate_citations:
        issues.append(
            {
                "severity": "fail",
                "check": "duplicate_citations",
                "message": f"Rows with duplicate citations: {duplicate_citations[:10]}.",
            }
        )

    return rows, issues


def scan_code(path: Path) -> list[dict]:
    issues: list[dict] = []
    if not path.exists():
        return [{"severity": "fail", "check": "generator_exists", "message": f"Missing generator: {_rel(path)}"}]

    text = path.read_text(encoding="utf-8", errors="replace")
    qid_hits = sorted(set(QUERY_ID_RE.findall(text)))
    if qid_hits:
        issues.append(
            {
                "severity": "fail",
                "check": "no_query_id_literals",
                "message": f"Generator contains explicit query-id literals: {qid_hits[:20]}.",
            }
        )

    if PATCH_TABLE_RE.search(text):
        issues.append(
            {
                "severity": "fail",
                "check": "no_patch_table",
                "message": "Generator appears to contain or invoke a targeted patch table.",
            }
        )

    if TEST_GOLD_RE.search(text):
        issues.append(
            {
                "severity": "fail",
                "check": "no_test_gold_use",
                "message": "Generator appears to access gold labels for test.csv.",
            }
        )

    if "load_query_split(\"test\")" in text or "load_query_split('test')" in text:
        issues.append(
            {
                "severity": "warn",
                "check": "test_query_read",
                "message": "Generator reads test queries, which is allowed for inference but must not drive manual row-specific labels.",
            }
        )

    if "public_proven" in text:
        issues.append(
            {
                "severity": "warn",
                "check": "public_proven_profile",
                "message": "`public_proven` is a distilled leaderboard profile. It is reproducible, but not yet the final prize-compliant generalization story.",
            }
        )

    if "--allow-missing-citations" in text or "allow_missing_citations" in text:
        issues.append(
            {
                "severity": "warn",
                "check": "missing_citation_policy",
                "message": "Missing-citation allowance must be justified by normalizer or train-gold evidence before prize submission.",
            }
        )

    return issues


def judge(issues: list[dict], strict: bool) -> str:
    has_fail = any(x["severity"] == "fail" for x in issues)
    has_warn = any(x["severity"] == "warn" for x in issues)
    if has_fail:
        return "fail"
    if strict and has_warn:
        return "needs_review"
    if has_warn:
        return "pass_with_warnings"
    return "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a candidate Kaggle submission for prize-condition risks.")
    parser.add_argument("--generator", type=Path, default=ROOT / "scripts" / "run_institution_cluster_rescue.py")
    parser.add_argument("--submission", type=Path, default=ROOT / "release" / "submission_institution_cluster_rescue_v10_public_proven_aligned" / "submission.csv")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as needing human review.")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    _rows, submission_issues = read_submission(args.submission)
    code_issues = scan_code(args.generator)
    issues = submission_issues + code_issues
    result = {
        "generator": _rel(args.generator),
        "submission": _rel(args.submission),
        "status": judge(issues, strict=args.strict),
        "fail_count": sum(1 for x in issues if x["severity"] == "fail"),
        "warn_count": sum(1 for x in issues if x["severity"] == "warn"),
        "issues": issues,
        "prize_conditions": [
            "Reproducible by organizers in reasonable time and cost.",
            "Scalable, with per-sample inference cost below the competition limit.",
            "Generalizable to private queries from the same distribution, beyond visible test examples.",
            "No manual domain-expert labeling of visible test answers as the final solution.",
            "Any generated data or rule-mining process must be reproducible and legally usable.",
        ],
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "fail":
        raise SystemExit(1)
    if args.strict and result["status"] == "needs_review":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

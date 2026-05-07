from __future__ import annotations

import argparse
import csv
from pathlib import Path


def split_citations(value: str | None) -> set[str]:
    if value is None or value == "":
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def f1_score(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def read_submission(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return {
            row["query_id"]: row.get("predicted_citations", "")
            for row in csv.DictReader(f)
        }


def evaluate(submission_path: Path, gold_path: Path) -> tuple[float, list[dict]]:
    submission = read_submission(submission_path)
    scores: list[float] = []
    rows: list[dict] = []
    with gold_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            pred = split_citations(submission.get(qid, ""))
            gold = split_citations(row.get("gold_citations", ""))
            score = f1_score(pred, gold)
            scores.append(score)
            rows.append(
                {
                    "query_id": qid,
                    "pred_count": len(pred),
                    "gold_count": len(gold),
                    "tp": len(pred & gold),
                    "fp": len(pred - gold),
                    "fn": len(gold - pred),
                    "f1": round(score, 6),
                }
            )
    return (sum(scores) / len(scores) if scores else 0.0), rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-style macro F1 evaluator.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--per-query-out", type=Path)
    args = parser.parse_args()

    macro_f1, rows = evaluate(args.submission, args.gold)
    if args.per_query_out:
        write_rows(args.per_query_out, rows)
    print(f"Macro F1: {macro_f1:.6f}")


if __name__ == "__main__":
    main()

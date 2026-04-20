import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import (
    evaluate_predictions,
    read_prediction_file,
    write_per_query_csv,
    write_summary_json,
)
from legal_ir.paths import ARTIFACT_DIR


def load_lookup_csv(path: Path) -> dict[str, str]:
    lookup = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["norm_citation"]] = row["canonical_citation"]
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted citations with macro F1.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--pred-file", type=Path, required=True, help="CSV with query_id,predicted_citations")
    parser.add_argument(
        "--lookup-file",
        type=Path,
        default=ARTIFACT_DIR / "phase0" / "citation_lookup.csv",
        help="citation lookup csv",
    )
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT_DIR / "eval")
    args = parser.parse_args()

    gold_rows = load_query_split(args.split)
    pred_map = read_prediction_file(args.pred_file)
    lookup = load_lookup_csv(args.lookup_file) if args.lookup_file.exists() else None

    strict_summary, strict_rows = evaluate_predictions(
        gold_rows=gold_rows, pred_map=pred_map, citation_lookup=lookup, mode="strict"
    )
    corpus_summary, corpus_rows = evaluate_predictions(
        gold_rows=gold_rows, pred_map=pred_map, citation_lookup=lookup, mode="corpus_aware"
    )
    strict_summary["split"] = args.split
    corpus_summary["split"] = args.split

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_json(out_dir / f"{args.split}_summary_strict.json", strict_summary)
    write_per_query_csv(out_dir / f"{args.split}_per_query_strict.csv", strict_rows)
    write_summary_json(out_dir / f"{args.split}_summary_corpus_aware.json", corpus_summary)
    write_per_query_csv(out_dir / f"{args.split}_per_query_corpus_aware.csv", corpus_rows)

    report = {
        "strict_macro_f1": strict_summary["macro_f1"],
        "corpus_aware_macro_f1": corpus_summary["macro_f1"],
        "split": args.split,
    }
    (out_dir / f"{args.split}_quick_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()


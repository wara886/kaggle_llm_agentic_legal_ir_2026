from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.bm25 import BM25Index
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions, write_per_query_csv, write_summary_json
from legal_ir.normalization import normalize_citation
from legal_ir.paths import OUTPUT_DIR


def join_doc_text(row: dict, text_max_chars: int) -> str:
    title = row.get("title", "")
    text = row.get("text", "")
    body = text[:text_max_chars] if text_max_chars > 0 else text
    return f"{row['citation']} {title} {body}".strip()


def build_docs(include_court: bool, max_court_rows: int | None, text_max_chars: int) -> list[dict]:
    docs = []
    for row in iter_corpus_rows(
        include_laws=True,
        include_court=include_court,
        max_court_rows=max_court_rows,
    ):
        citation = normalize_citation(row["citation"])
        if not citation:
            continue
        docs.append(
            {
                "citation": citation,
                "source": row["source"],
                "text": join_doc_text(row, text_max_chars=text_max_chars),
            }
        )
    return docs


def dedup_citations(citations: list[str], top_n: int) -> list[str]:
    out = []
    seen = set()
    for c in citations:
        if c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= top_n:
            break
    return out


def predict_split(index: BM25Index, docs: list[dict], rows: list[dict], top_k: int, top_n: int) -> dict[str, list[str]]:
    pred_map: dict[str, list[str]] = {}
    for row in rows:
        ranked = index.search(row["query"], top_k=top_k)
        cands = [docs[doc_id]["citation"] for doc_id, _ in ranked]
        pred_map[row["query_id"]] = dedup_citations(cands, top_n=top_n)
    return pred_map


def write_prediction_csv(path: Path, pred_map: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(pred_map.keys()):
            writer.writerow([qid, ";".join(pred_map[qid])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal BM25 baseline.")
    parser.add_argument("--include-court", action="store_true", help="Include court_considerations.csv")
    parser.add_argument("--max-court-rows", type=int, default=100000, help="Cap court rows for memory/speed.")
    parser.add_argument("--top-k", type=int, default=50, help="Initial retrieval candidates.")
    parser.add_argument("--top-n", type=int, default=8, help="Output citations per query.")
    parser.add_argument("--text-max-chars", type=int, default=1200, help="Truncate doc text for indexing.")
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_DIR / "baseline_bm25", help="Output directory."
    )
    args = parser.parse_args()

    print("[bm25] loading data splits...")
    train_rows = load_query_split("train")
    val_rows = load_query_split("val")
    test_rows = load_query_split("test")

    print("[bm25] building corpus docs...")
    docs = build_docs(
        include_court=args.include_court,
        max_court_rows=args.max_court_rows if args.include_court else None,
        text_max_chars=args.text_max_chars,
    )
    print(f"[bm25] docs={len(docs)}")

    print("[bm25] building BM25 index...")
    index = BM25Index()
    index.build(docs, text_key="text")

    print("[bm25] predicting val/test...")
    val_pred = predict_split(index, docs, val_rows, top_k=args.top_k, top_n=args.top_n)
    test_pred = predict_split(index, docs, test_rows, top_k=args.top_k, top_n=args.top_n)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    val_pred_path = out_dir / "val_predictions.csv"
    test_sub_path = out_dir / "submission.csv"
    write_prediction_csv(val_pred_path, val_pred)
    write_prediction_csv(test_sub_path, test_pred)

    strict_summary, strict_per_query = evaluate_predictions(
        gold_rows=val_rows, pred_map=val_pred, citation_lookup=None, mode="strict"
    )
    strict_summary["split"] = "val"
    strict_summary["notes"] = "最小 BM25 baseline（未使用 dense / rerank / query rewrite）"

    write_summary_json(out_dir / "val_eval_summary_strict.json", strict_summary)
    write_per_query_csv(out_dir / "val_eval_per_query_strict.csv", strict_per_query)

    meta = {
        "config": {
            "include_court": args.include_court,
            "max_court_rows": args.max_court_rows,
            "top_k": args.top_k,
            "top_n": args.top_n,
            "text_max_chars": args.text_max_chars,
        },
        "sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "docs": len(docs)},
        "artifacts": {
            "val_predictions": str(val_pred_path),
            "submission": str(test_sub_path),
            "val_eval_summary_strict": str(out_dir / "val_eval_summary_strict.json"),
            "val_eval_per_query_strict": str(out_dir / "val_eval_per_query_strict.csv"),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[bm25] wrote: {test_sub_path}")
    print(f"[bm25] strict macro_f1@val={strict_summary['macro_f1']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.normalization import normalize_citation


def _doc_text(row: dict, text_max_chars: int) -> str:
    text = row.get("text", "")
    body = text[:text_max_chars] if text_max_chars > 0 else text
    return f"{row.get('title', '')} {body} {row.get('citation', '')}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persisted laws dense embedding index for a MiniLM model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-laws-rows", type=int, default=175933)
    parser.add_argument("--text-max-chars", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "laws_minilm_p1" / "laws_dense_index")
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    docs = []
    texts = []
    for row in iter_corpus_rows(include_laws=True, include_court=False, max_laws_rows=args.max_laws_rows):
        citation = normalize_citation(row["citation"])
        if not citation:
            continue
        docs.append({"citation": citation, "title": row.get("title", "")})
        texts.append(_doc_text(row, args.text_max_chars))

    model = SentenceTransformer(args.model_dir)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "laws_embeddings.npy", embeddings.astype("float32"))
    with (args.out_dir / "laws_citations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "citation", "title"])
        writer.writeheader()
        for i, doc in enumerate(docs):
            writer.writerow({"row_id": i, **doc})

    meta = {
        "model_dir": args.model_dir,
        "rows": len(docs),
        "embedding_shape": list(embeddings.shape),
        "text_max_chars": args.text_max_chars,
        "embeddings": str(args.out_dir / "laws_embeddings.npy"),
        "citations": str(args.out_dir / "laws_citations.csv"),
    }
    (args.out_dir / "laws_dense_index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

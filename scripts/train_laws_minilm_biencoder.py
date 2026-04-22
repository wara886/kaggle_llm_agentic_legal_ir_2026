from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from law_family import family_query_terms, issue_query_terms, likely_statute_families
from query_preprocess import preprocess_query


def _load_triplets(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM bi-encoder on laws-only hard negatives.")
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--out-model-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--loss", choices=["triplet", "multiple_negatives"], default="multiple_negatives")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--query-mode", choices=["raw", "laws_structured"], default="raw")
    parser.add_argument("--issue-max-groups", type=int, default=4)
    parser.add_argument("--issue-max-terms", type=int, default=16)
    args = parser.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    rows = _load_triplets(args.triplets, args.max_examples)
    if not rows:
        raise SystemExit("No triplets loaded.")

    def _build_query_text(raw_query: str) -> str:
        if args.query_mode != "laws_structured":
            return raw_query
        mv = preprocess_query(raw_query)
        families = likely_statute_families(raw_query, max_families=2, min_score=4)
        fam_terms = family_query_terms(families)
        issue_terms = issue_query_terms(
            raw_query,
            families,
            max_groups=args.issue_max_groups,
            max_terms=args.issue_max_terms,
        )
        merged = " ".join(
            [
                mv.get("query_clean", ""),
                " ".join(fam_terms[:16]),
                " ".join(issue_terms[: args.issue_max_terms]),
            ]
        ).strip()
        return merged if merged else raw_query

    examples = []
    for r in rows:
        if not (r.get("query") and r.get("positive_text") and r.get("negative_text")):
            continue
        query_text = _build_query_text(r["query"])
        examples.append(InputExample(texts=[query_text, r["positive_text"], r["negative_text"]]))
    if not examples:
        raise SystemExit("No valid training examples.")

    model = SentenceTransformer(args.model_name)
    dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    if args.loss == "multiple_negatives":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        train_loss = losses.TripletLoss(model=model)

    warmup_steps = int(len(dataloader) * args.epochs * args.warmup_ratio)
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    args.out_model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out_model_dir))

    meta = {
        "base_model": args.model_name,
        "out_model_dir": str(args.out_model_dir),
        "examples": len(examples),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss": args.loss,
        "query_mode": args.query_mode,
        "warmup_steps": warmup_steps,
    }
    (args.out_model_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

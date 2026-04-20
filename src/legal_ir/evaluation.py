from __future__ import annotations

import csv
import json
from pathlib import Path

from .normalization import normalize_citation, split_citations


def set_f1(pred: set[str], gold: set[str]) -> tuple[float, float, float, int, int, int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def _normalize_pred_list(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for v in values:
        norm = normalize_citation(v)
        if norm and norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out


def evaluate_predictions(
    gold_rows: list[dict],
    pred_map: dict[str, list[str]],
    citation_lookup: dict[str, str] | None = None,
    mode: str = "strict",
) -> tuple[dict, list[dict]]:
    if mode not in {"strict", "corpus_aware"}:
        raise ValueError("mode must be strict or corpus_aware")

    per_query: list[dict] = []
    f1_scores = []
    missing_gold_total = 0

    for row in gold_rows:
        qid = row["query_id"]
        gold_raw = row.get("gold_citation_list") or split_citations(row.get("gold_citations", ""))
        pred_raw = pred_map.get(qid, [])

        gold_norm = _normalize_pred_list(gold_raw)
        pred_norm = _normalize_pred_list(pred_raw)

        if citation_lookup is not None:
            gold_resolved = [citation_lookup.get(g, g) for g in gold_norm]
            pred_resolved = [citation_lookup.get(p, p) for p in pred_norm]
            gold_in_corpus = [g for g in gold_norm if g in citation_lookup]
        else:
            gold_resolved = gold_norm
            pred_resolved = pred_norm
            gold_in_corpus = gold_norm

        missing_gold = len(gold_norm) - len(gold_in_corpus)
        missing_gold_total += missing_gold

        if mode == "corpus_aware" and citation_lookup is not None:
            gold_final = set(citation_lookup[g] for g in gold_in_corpus)
        else:
            gold_final = set(gold_resolved)
        pred_final = set(pred_resolved)

        precision, recall, f1, tp, fp, fn = set_f1(pred_final, gold_final)
        f1_scores.append(f1)

        per_query.append(
            {
                "query_id": qid,
                "gold_count": len(gold_final),
                "pred_count": len(pred_final),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "missing_gold_from_corpus": missing_gold,
            }
        )

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    summary = {
        "mode": mode,
        "queries": len(gold_rows),
        "macro_f1": round(macro_f1, 6),
        "avg_missing_gold_from_corpus": round(missing_gold_total / len(gold_rows), 6)
        if gold_rows
        else 0.0,
    }
    return summary, per_query


def read_prediction_file(path: Path) -> dict[str, list[str]]:
    pred_map: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_map[row["query_id"]] = split_citations(row.get("predicted_citations", ""))
    return pred_map


def write_per_query_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

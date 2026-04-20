from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .data_loader import iter_corpus_csv
from .normalization import normalize_citation, normalize_text


def iter_corpus_rows(
    include_laws: bool = True,
    include_court: bool = True,
    max_laws_rows: int | None = None,
    max_court_rows: int | None = None,
):
    if include_laws:
        for i, row in enumerate(iter_corpus_csv("laws_de.csv")):
            citation = normalize_text(row.get("citation", ""))
            text = normalize_text(row.get("text", ""))
            title = normalize_text(row.get("title", ""))
            yield {
                "source": "laws_de",
                "citation": citation,
                "norm_citation": normalize_citation(citation),
                "title": title,
                "text": text,
            }
            if max_laws_rows is not None and i + 1 >= max_laws_rows:
                break

    if include_court:
        for i, row in enumerate(iter_corpus_csv("court_considerations.csv")):
            citation = normalize_text(row.get("citation", ""))
            text = normalize_text(row.get("text", ""))
            yield {
                "source": "court_considerations",
                "citation": citation,
                "norm_citation": normalize_citation(citation),
                "title": "",
                "text": text,
            }
            if max_court_rows is not None and i + 1 >= max_court_rows:
                break


def build_corpus_master_csv(
    output_path: Path,
    include_laws: bool = True,
    include_court: bool = True,
    max_laws_rows: int | None = None,
    max_court_rows: int | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "source", "citation", "norm_citation", "title", "text"]

    stats = {"rows": 0, "laws_rows": 0, "court_rows": 0}
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(
            iter_corpus_rows(
                include_laws=include_laws,
                include_court=include_court,
                max_laws_rows=max_laws_rows,
                max_court_rows=max_court_rows,
            )
        ):
            row_out = {"row_id": idx, **row}
            writer.writerow(row_out)
            stats["rows"] += 1
            if row["source"] == "laws_de":
                stats["laws_rows"] += 1
            else:
                stats["court_rows"] += 1
    return stats


def build_citation_lookup(
    include_laws: bool = True,
    include_court: bool = True,
    max_laws_rows: int | None = None,
    max_court_rows: int | None = None,
) -> tuple[dict[str, str], dict]:
    norm_to_candidates: dict[str, set[str]] = defaultdict(set)
    stats = {"rows": 0, "sources": defaultdict(int)}

    for row in iter_corpus_rows(
        include_laws=include_laws,
        include_court=include_court,
        max_laws_rows=max_laws_rows,
        max_court_rows=max_court_rows,
    ):
        stats["rows"] += 1
        stats["sources"][row["source"]] += 1
        norm = row["norm_citation"]
        if norm:
            norm_to_candidates[norm].add(row["citation"])

    lookup: dict[str, str] = {}
    for norm, cands in norm_to_candidates.items():
        lookup[norm] = sorted(cands, key=lambda x: (len(x), x))[0]

    stats["unique_norm_citations"] = len(lookup)
    stats["sources"] = dict(stats["sources"])
    return lookup, stats

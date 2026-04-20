from __future__ import annotations

import csv
from pathlib import Path

from .normalization import split_citations
from .paths import DATA_DIR


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if limit is not None and i + 1 >= limit:
                break
    return rows


def load_query_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.csv"
    rows = read_csv_rows(path)
    for row in rows:
        row["query"] = row["query"].strip()
        if "gold_citations" in row:
            row["gold_citation_list"] = split_citations(row["gold_citations"])
    return rows


def iter_corpus_csv(filename: str):
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

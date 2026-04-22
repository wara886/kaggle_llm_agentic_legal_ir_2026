from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from citation_normalizer import normalize_citation, split_citations
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions, read_prediction_file, write_per_query_csv, write_summary_json


def load_lookup_csv(path: Path) -> dict[str, str]:
    lookup = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["norm_citation"]] = row["canonical_citation"]
    return lookup


def classify_error(row: dict) -> str:
    f1 = float(row.get("f1", 0.0))
    p = float(row.get("precision", 0.0))
    r = float(row.get("recall", 0.0))
    missing = int(float(row.get("missing_gold_from_corpus", 0)))
    if f1 == 0 and p == 0 and r == 0:
        return "完全未命中"
    if missing > 0 and r < 0.3:
        return "语料外标签影响"
    if r < 0.3 and p >= 0.3:
        return "召回不足"
    if p < 0.3 and r >= 0.3:
        return "精度不足"
    if f1 < 0.4:
        return "精召均弱"
    return "相对稳定"


def evaluate_and_dump(
    split: str,
    pred_file: Path,
    out_dir: Path,
    lookup_file: Path | None = None,
) -> dict:
    gold_rows = load_query_split(split)
    pred_map = read_prediction_file(pred_file)
    lookup = load_lookup_csv(lookup_file) if lookup_file and lookup_file.exists() else None

    strict_summary, strict_rows = evaluate_predictions(
        gold_rows=gold_rows,
        pred_map=pred_map,
        citation_lookup=lookup,
        mode="strict",
    )
    corpus_summary, corpus_rows = evaluate_predictions(
        gold_rows=gold_rows,
        pred_map=pred_map,
        citation_lookup=lookup,
        mode="corpus_aware",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_json(out_dir / f"{split}_summary_strict.json", strict_summary)
    write_summary_json(out_dir / f"{split}_summary_corpus_aware.json", corpus_summary)
    write_per_query_csv(out_dir / f"{split}_per_query_strict.csv", strict_rows)
    write_per_query_csv(out_dir / f"{split}_per_query_corpus_aware.csv", corpus_rows)

    taxonomy_rows = []
    for row in strict_rows:
        row2 = dict(row)
        row2["error_taxonomy"] = classify_error(row)
        taxonomy_rows.append(row2)

    if taxonomy_rows:
        with (out_dir / f"{split}_error_taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(taxonomy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(taxonomy_rows)

    result = {
        "split": split,
        "strict_macro_f1": strict_summary["macro_f1"],
        "corpus_aware_macro_f1": corpus_summary["macro_f1"],
        "queries": strict_summary["queries"],
    }
    (out_dir / f"{split}_quick_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_citation_source_map(
    max_laws_rows: int | None = None,
    max_court_rows: int | None = None,
) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for row in iter_corpus_rows(
        include_laws=True,
        include_court=True,
        max_laws_rows=max_laws_rows,
        max_court_rows=max_court_rows,
    ):
        source_map[row["citation"]] = row["source"]
        source_map[row["norm_citation"]] = row["source"]
    return source_map


def _first_hit_rank(gold: set[str], ranked: list[str]) -> int:
    for idx, citation in enumerate(ranked, start=1):
        if citation in gold:
            return idx
    return -1


def _recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(gold.intersection(set(ranked[:k])))
    return round(hits / len(gold), 6)


def _gold_source_type(gold_in_corpus: list[str], source_map: dict[str, str]) -> str:
    sources = set(source_map.get(c, "unknown") for c in gold_in_corpus)
    if not sources or sources == {"unknown"}:
        return "unknown"
    if sources == {"laws_de"}:
        return "laws_only"
    if sources == {"court_considerations"}:
        return "court_only"
    return "mixed"


def _detect_failure_type(
    branch_name: str,
    first_hit_rank: int,
    gold_in_corpus: list[str],
    branch_candidates: list[str],
    fusion_candidates: list[str],
    final_candidates: list[str],
    pre_fusion_candidates: list[str] | None,
    has_norm_mismatch: bool,
    gold_source_type: str,
) -> str:
    if has_norm_mismatch:
        return "normalization_mismatch"
    if not gold_in_corpus:
        return "no_gold_in_candidates"

    if branch_name == "sparse_laws" and gold_source_type == "court_only":
        return "source_mismatch"
    if branch_name == "sparse_court" and gold_source_type == "laws_only":
        return "source_mismatch"
    if branch_name == "dense_laws" and gold_source_type == "court_only":
        return "source_mismatch"
    if branch_name == "dense_court" and gold_source_type == "laws_only":
        return "source_mismatch"

    branch_top200 = set(branch_candidates[:200])
    gold_set = set(gold_in_corpus)
    if not (gold_set & branch_top200):
        return "no_gold_in_candidates"

    if first_hit_rank > 50:
        return "gold_only_after_large_k"

    if branch_name == "fusion":
        union_branch = set(pre_fusion_candidates or branch_candidates)
        if (gold_set & union_branch) and not (gold_set & set(fusion_candidates)):
            return "gold_lost_in_fusion"
        if (gold_set & set(fusion_candidates)) and not (gold_set & set(final_candidates)):
            return "gold_lost_in_final_cut"

    return "ok"


def compute_candidate_recall_rows(
    query_id: str,
    gold_citations_raw: list[str],
    branch_name: str,
    branch_candidates: list[str],
    source_map: dict[str, str],
    all_norm_corpus_citations: set[str],
    fusion_candidates: list[str] | None = None,
    final_candidates: list[str] | None = None,
    pre_fusion_candidates: list[str] | None = None,
) -> dict:
    if fusion_candidates is None:
        fusion_candidates = branch_candidates
    if final_candidates is None:
        final_candidates = branch_candidates

    gold_norm = [normalize_citation(c) for c in gold_citations_raw if normalize_citation(c)]
    gold_in_corpus = [c for c in gold_norm if c in all_norm_corpus_citations]
    gold_set = set(gold_in_corpus)
    first_hit_rank = _first_hit_rank(gold_set, branch_candidates)
    has_norm_mismatch = any((raw not in all_norm_corpus_citations) and (normalize_citation(raw) in all_norm_corpus_citations)
                            for raw in gold_citations_raw)

    gold_source_type = _gold_source_type(gold_in_corpus, source_map)
    failure_type = _detect_failure_type(
        branch_name=branch_name,
        first_hit_rank=first_hit_rank,
        gold_in_corpus=gold_in_corpus,
        branch_candidates=branch_candidates,
        fusion_candidates=fusion_candidates,
        final_candidates=final_candidates,
        pre_fusion_candidates=pre_fusion_candidates,
        has_norm_mismatch=has_norm_mismatch,
        gold_source_type=gold_source_type,
    )

    return {
        "query_id": query_id,
        "gold_citations_raw": ";".join(gold_citations_raw),
        "gold_citations_norm": ";".join(gold_norm),
        "gold_in_corpus": ";".join(gold_in_corpus),
        "gold_source_type": gold_source_type,
        "first_hit_rank": first_hit_rank,
        "hit_at_10": _recall_at_k(gold_set, branch_candidates, 10),
        "hit_at_50": _recall_at_k(gold_set, branch_candidates, 50),
        "hit_at_100": _recall_at_k(gold_set, branch_candidates, 100),
        "hit_at_200": _recall_at_k(gold_set, branch_candidates, 200),
        "branch_name": branch_name,
        "failure_type": failure_type,
    }


def summarize_failure_clusters(recall_rows: list[dict]) -> list[dict]:
    counter = Counter((r["branch_name"], r["failure_type"]) for r in recall_rows)
    out = []
    for (branch_name, failure_type), count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        out.append(
            {
                "branch_name": branch_name,
                "failure_type": failure_type,
                "count": count,
            }
        )
    return out

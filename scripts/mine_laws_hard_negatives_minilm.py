from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from fusion import rrf_fusion
from law_family import extract_family_from_citation, issue_query_terms, likely_statute_families
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.normalization import normalize_citation, split_citations
from query_expansion import build_source_aware_query_packs
from query_preprocess import preprocess_query
from retrieval_dense import DenseRetriever
from retrieval_sparse import SparseRetriever


def _laws_lookup(max_laws_rows: int | None = None) -> dict[str, dict]:
    out = {}
    for row in iter_corpus_rows(include_laws=True, include_court=False, max_laws_rows=max_laws_rows):
        norm = normalize_citation(row["citation"])
        if norm and norm not in out:
            out[norm] = row
    return out


def _doc_text(row: dict, text_max_chars: int) -> str:
    title = row.get("title", "")
    text = row.get("text", "")
    body = text[:text_max_chars] if text_max_chars > 0 else text
    return f"{title} {body} {row.get('citation', '')}".strip()


def _issue_overlap_count(text: str, issue_terms: list[str]) -> int:
    text_l = (text or "").lower()
    count = 0
    for term in issue_terms:
        t = (term or "").strip().lower()
        if not t:
            continue
        if t in text_l:
            count += 1
    return count


def _fallback_issue_terms(mv: dict, packs: dict, max_terms: int = 16) -> list[str]:
    terms = []
    seen = set()

    for t in mv.get("query_legal_phrases", []) or []:
        x = str(t).strip()
        if len(x) < 4:
            continue
        k = x.lower()
        if k in seen:
            continue
        terms.append(x)
        seen.add(k)
        if len(terms) >= max_terms:
            return terms

    for t in mv.get("query_keywords", []) or []:
        x = str(t).strip()
        if len(x) < 4:
            continue
        k = x.lower()
        if k in seen:
            continue
        terms.append(x)
        seen.add(k)
        if len(terms) >= max_terms:
            return terms

    laws_pack_v2 = packs.get("laws_query_pack_v2", {}) or packs.get("laws_query_pack", {})
    for t in laws_pack_v2.get("expanded_keywords_de", []) or []:
        x = str(t).strip()
        if len(x) < 4:
            continue
        k = x.lower()
        if k in seen:
            continue
        terms.append(x)
        seen.add(k)
        if len(terms) >= max_terms:
            return terms
    return terms


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine laws-only hard negatives for MiniLM fine-tuning.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--dense-max-laws", type=int, default=80000)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--text-max-chars", type=int, default=900)
    parser.add_argument("--prefer-same-family-near-miss", action="store_true")
    parser.add_argument("--prefer-same-issue-near-miss", action="store_true")
    parser.add_argument("--issue-max-groups", type=int, default=4)
    parser.add_argument("--issue-max-terms", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "laws_minilm_p1")
    args = parser.parse_args()

    random.seed(args.random_seed)
    rows = load_query_split(args.split)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    laws = _laws_lookup(max_laws_rows=args.sparse_max_laws)

    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(max_laws_rows=args.sparse_max_laws, max_court_rows=0, enable_field_aware=True)
    dense = DenseRetriever(
        model_name=args.model_name,
        use_sbert=not args.dense_disable_sbert,
        text_max_chars=500,
        svd_dim=256,
    )
    dense.build(max_laws_rows=args.dense_max_laws, max_court_rows=0, enable_field_aware=True)

    examples = []
    diagnostics = []
    for row in rows:
        query = row["query"]
        qid = row["query_id"]
        gold = [normalize_citation(c) for c in split_citations(row.get("gold_citations", ""))]
        gold_laws = [g for g in gold if g in laws]
        if not gold_laws:
            continue

        mv = preprocess_query(query)
        packs = build_source_aware_query_packs(mv)
        laws_pack = packs.get("laws_query_pack_v2", {}) or packs.get("laws_query_pack", {})

        sparse_items = sparse.search_field_aware(
            laws_query_pack=packs.get("laws_query_pack", {}),
            court_query_pack={},
            laws_query_pack_v2=packs.get("laws_query_pack_v2", {}),
            enable_laws_query_pack_v2=True,
            top_k_laws=args.top_k,
            top_k_court=0,
            laws_citation_weight=2.0,
            laws_title_weight=2.0,
            laws_text_weight=1.0,
        )
        dense_items = dense.search_source_aware(
            laws_query_pack=laws_pack,
            court_query_pack={},
            top_k_laws=args.top_k,
            top_k_court=0,
        )
        sparse_rank = [x.citation for x in sparse_items if x.source == "laws_de"]
        dense_rank = [x.citation for x in dense_items if x.source == "laws_de"]
        fused = [c for c, _ in rrf_fusion([sparse_rank, dense_rank], k=60, top_n=args.top_k)]

        gold_family_set = {
            extract_family_from_citation(c).upper()
            for c in gold_laws
            if extract_family_from_citation(c)
        }
        likely_families = likely_statute_families(query, max_families=2, min_score=4)
        issue_terms = issue_query_terms(
            query=query,
            families=likely_families if likely_families else list(gold_family_set),
            max_groups=args.issue_max_groups,
            max_terms=args.issue_max_terms,
        )
        if not issue_terms:
            issue_terms = _fallback_issue_terms(mv, packs, max_terms=args.issue_max_terms)
        neg = ""
        if args.prefer_same_issue_near_miss:
            scored = []
            for idx, c in enumerate(fused):
                if c in set(gold_laws) or c not in laws:
                    continue
                c_family = extract_family_from_citation(c).upper()
                c_text = _doc_text(laws[c], args.text_max_chars)
                issue_overlap = _issue_overlap_count(c_text, issue_terms)
                same_family = int(bool(c_family and c_family in gold_family_set))
                scored.append((c, issue_overlap, same_family, idx))

            # If issue signals exist, prefer candidates with non-zero issue overlap to avoid overly easy negatives.
            if issue_terms:
                filtered = [x for x in scored if x[1] > 0]
                if filtered:
                    scored = filtered
            if scored:
                # Primary: issue overlap, Tie-breaker: same family, then earlier fused rank.
                scored.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)
                neg = scored[0][0]

        if not neg and args.prefer_same_family_near_miss and gold_family_set:
            for c in fused:
                if c in set(gold_laws) or c not in laws:
                    continue
                fam = extract_family_from_citation(c).upper()
                if fam and fam in gold_family_set:
                    neg = c
                    break
        if not neg:
            neg = next((c for c in fused if c not in set(gold_laws) and c in laws), "")
        if not neg:
            continue
        pos = random.choice(gold_laws)
        pos_family = extract_family_from_citation(pos).upper()
        neg_family = extract_family_from_citation(neg).upper()
        pos_issue_overlap = _issue_overlap_count(_doc_text(laws[pos], args.text_max_chars), issue_terms)
        neg_issue_overlap = _issue_overlap_count(_doc_text(laws[neg], args.text_max_chars), issue_terms)
        examples.append(
            {
                "query_id": qid,
                "query": query,
                "positive_citation": pos,
                "positive_text": _doc_text(laws[pos], args.text_max_chars),
                "negative_citation": neg,
                "negative_text": _doc_text(laws[neg], args.text_max_chars),
                "positive_family": pos_family,
                "negative_family": neg_family,
                "same_family_near_miss": int(bool(pos_family and neg_family and pos_family == neg_family)),
                "issue_terms_count": len(issue_terms),
                "positive_issue_overlap": pos_issue_overlap,
                "negative_issue_overlap": neg_issue_overlap,
            }
        )
        diagnostics.append(
            {
                "query_id": qid,
                "positive_citation": pos,
                "negative_citation": neg,
                "negative_rank": fused.index(neg) + 1,
                "positive_in_top_k": int(pos in fused),
                "positive_family": pos_family,
                "negative_family": neg_family,
                "same_family_near_miss": int(bool(pos_family and neg_family and pos_family == neg_family)),
                "issue_terms_count": len(issue_terms),
                "positive_issue_overlap": pos_issue_overlap,
                "negative_issue_overlap": neg_issue_overlap,
            }
        )
        if args.max_examples > 0 and len(examples) >= args.max_examples:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "laws_hard_negative_triplets.jsonl"
    csv_path = args.out_dir / "laws_hard_negative_diagnostics.csv"
    _write_jsonl(jsonl_path, examples)
    _write_csv(csv_path, diagnostics)

    meta = {
        "triplets": len(examples),
        "split": args.split,
        "max_rows": args.max_rows,
        "dense_backend": dense.backend,
        "prefer_same_family_near_miss": bool(args.prefer_same_family_near_miss),
        "prefer_same_issue_near_miss": bool(args.prefer_same_issue_near_miss),
        "jsonl": str(jsonl_path),
        "diagnostics_csv": str(csv_path),
    }
    (args.out_dir / "laws_hard_negative_mining_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

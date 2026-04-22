from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.normalization import normalize_citation, split_citations
from query_expansion import build_source_aware_query_packs
from query_preprocess import preprocess_query


TOKEN_RE = re.compile(r"[a-z0-9_./-]+", re.I)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "under",
    "from",
    "that",
    "this",
    "into",
    "court",
    "law",
    "legal",
    "article",
    "art",
    "abs",
    "lit",
}


def _tokens(text: str) -> set[str]:
    out = set()
    for tok in TOKEN_RE.findall((text or "").lower()):
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        out.add(tok)
    return out


def _norm_map_laws() -> dict[str, dict]:
    out = {}
    for row in iter_corpus_rows(include_laws=True, include_court=False):
        norm = normalize_citation(row["citation"])
        if norm and norm not in out:
            out[norm] = row
    return out


def _signal_terms(query: str) -> set[str]:
    mv = preprocess_query(query)
    packs = build_source_aware_query_packs(mv)
    terms = set(mv.get("query_keywords", []))
    terms.update(mv.get("query_legal_phrases", []))
    terms.update(packs.get("laws_query_pack", {}).get("expanded_keywords_de", []))
    terms.update(packs.get("laws_query_pack_v2", {}).get("expanded_keywords_de", []))
    return _tokens(" ".join(terms))


def _bucket_for_signal(terms: set[str], text: str) -> tuple[str, list[str]]:
    if not terms:
        return "no_detectable_query_signal", []
    lower = (text or "").lower()
    matches = []
    best_pos = None
    for term in terms:
        pos = lower.find(term)
        if pos >= 0:
            matches.append(term)
            if best_pos is None or pos < best_pos:
                best_pos = pos
    if best_pos is None:
        return "no_detectable_query_signal", []
    if best_pos < 500:
        return "within_500", sorted(matches)[:12]
    if best_pos < 900:
        return "within_900", sorted(matches)[:12]
    return "beyond_900", sorted(matches)[:12]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether laws gold signals are truncated by 500/900 char cuts.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "laws_truncation_audit")
    args = parser.parse_args()

    laws = _norm_map_laws()
    rows = load_query_split(args.split)
    audit_rows = []
    for row in rows:
        terms = _signal_terms(row["query"])
        for citation in split_citations(row.get("gold_citations", "")):
            norm = normalize_citation(citation)
            law = laws.get(norm)
            if not law:
                continue
            text = f"{law.get('title', '')} {law.get('text', '')}".strip()
            bucket, matched = _bucket_for_signal(terms, text)
            text_len = len(text)
            audit_rows.append(
                {
                    "query_id": row["query_id"],
                    "gold_citation": norm,
                    "text_len_chars": text_len,
                    "signal_bucket": bucket,
                    "matched_terms": ";".join(matched),
                    "fits_500_chars": int(text_len <= 500),
                    "fits_900_chars": int(text_len <= 900),
                }
            )

    counts = {
        "within_500": 0,
        "within_900": 0,
        "beyond_900": 0,
        "no_detectable_query_signal": 0,
    }
    for r in audit_rows:
        counts[r["signal_bucket"]] = counts.get(r["signal_bucket"], 0) + 1

    total = len(audit_rows)
    fits_500 = sum(int(r["fits_500_chars"]) for r in audit_rows)
    fits_900 = sum(int(r["fits_900_chars"]) for r in audit_rows)
    detectable = total - counts.get("no_detectable_query_signal", 0)
    beyond_900 = counts.get("beyond_900", 0)
    truncation_bottleneck = bool(detectable and beyond_900 / max(detectable, 1) >= 0.2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "laws_truncation_audit_rows.csv", audit_rows)

    lines = [
        "# Laws Truncation Audit",
        "",
        f"- split: `{args.split}`",
        f"- laws gold citations in corpus: `{total}`",
        f"- gold docs fitting first 500 chars: `{fits_500}` / `{total}`",
        f"- gold docs fitting first 900 chars: `{fits_900}` / `{total}`",
        "",
        "## Signal Position",
        "",
        "| bucket | count | ratio |",
        "|---|---:|---:|",
    ]
    for key in ["within_500", "within_900", "beyond_900", "no_detectable_query_signal"]:
        cnt = counts.get(key, 0)
        ratio = cnt / total if total else 0.0
        lines.append(f"| {key} | {cnt} | {ratio:.6f} |")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "- 截断是当前主瓶颈之一。"
                if truncation_bottleneck
                else "- 截断不是当前主瓶颈之一；优先进入 laws-only hard negative mining + MiniLM fine-tune。"
            ),
        ]
    )
    (args.out_dir / "laws_truncation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "laws_truncation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((ROOT / "docs" / "laws_truncation_audit.md").as_posix())


if __name__ == "__main__":
    main()

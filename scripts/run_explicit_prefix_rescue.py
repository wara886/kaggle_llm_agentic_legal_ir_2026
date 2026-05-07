from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from legal_ir.normalization import split_citations

KNOWN_FAMILY_RE = (
    "ATSG|BGG|BV|DBG|IVG|LAI|OR|CO|SCHKG|STGB|STPO|SVG|UVG|VVG|"
    "ZGB|ZPO|IPRG|PRHG|AHVG|FIDLEG"
)
ARTICLE_RE = re.compile(
    rf"\b(?:Art\.?|article)\s+\d+[a-z]?"
    rf"(?:\s+Abs\.?\s+\d+[a-z]*)?"
    rf"(?:\s+lit\.?\s+[a-z])?"
    rf"\s+(?:{KNOWN_FAMILY_RE})\b",
    re.I,
)
NATURAL_ARTICLE_RE = re.compile(
    r"\b(?:Art\.?|article)\s+(\d+[a-z]?)"
    r"(?:\s+Abs\.?\s+(\d+[a-z]*))?"
    r"(?:\s+lit\.?\s+([a-z]))?"
    r"\s+of\s+the\s+"
    r"(CC|Civil Code|Code of Obligations|CO|Criminal Code|Swiss Criminal Code|"
    r"Criminal Procedure Code|Civil Procedure Code|Private International Law Act)\b",
    re.I,
)
CONJUNCTION_ARTICLE_RE = re.compile(
    rf"\b(?:Art\.?|article)\s+(\d+[a-z]?)"
    rf"(?:\s+Abs\.?\s+(\d+[a-z]*))?"
    rf"\s+(?:and|or|/|,)\s+(\d+[a-z]?)"
    rf"(?:\s+Abs\.?\s+(\d+[a-z]*))?"
    rf"\s+({KNOWN_FAMILY_RE})\b",
    re.I,
)
NATURAL_FAMILY_ALIASES = {
    "cc": "ZGB",
    "civil code": "ZGB",
    "code of obligations": "OR",
    "co": "OR",
    "criminal code": "StGB",
    "swiss criminal code": "StGB",
    "criminal procedure code": "StPO",
    "civil procedure code": "ZPO",
    "private international law act": "IPRG",
}


def _canon(citation: str) -> str:
    text = normalize_citation(citation)
    text = re.sub(r"\bCO\b", "OR", text, flags=re.I)
    text = re.sub(r"\bLAI\b", "IVG", text, flags=re.I)
    text = re.sub(r"\bPRHG\b", "PrHG", text, flags=re.I)
    return text


def _family(citation: str) -> str:
    m = re.search(r"([A-Za-z][A-Za-z0-9-]*)$", citation or "")
    if not m:
        return ""
    return m.group(1).replace("-", "").upper().replace("CO", "OR").replace("LAI", "IVG")


def _article_number(citation: str) -> str:
    m = re.match(r"Art\.\s*(\d+[a-z]?)", citation or "", re.I)
    return m.group(1).lower() if m else ""


def _article_key(citation: str) -> tuple[str, str] | None:
    citation = _canon(citation)
    art = _article_number(citation)
    fam = _family(citation)
    if not art or not fam:
        return None
    return art, fam


def _has_abs(citation: str) -> bool:
    return bool(re.search(r"\bAbs\.\s*\d+", citation or "", re.I))


def _dedup(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        c = _canon(value)
        if not c or c in seen:
            continue
        out.append(c)
        seen.add(c)
    return out


def _read_predictions(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("query_id", "")).strip()
            if qid:
                out[qid] = _dedup(split_citations(row.get("predicted_citations", "")))
    return out


def _explicit_citation_records(query: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in ARTICLE_RE.finditer(query or ""):
        c = _canon(m.group(0))
        if c and c not in seen:
            out.append((c, "abbrev"))
            seen.add(c)
    for m in CONJUNCTION_ARTICLE_RE.finditer(query or ""):
        family = _canon(f"Art. 1 {m.group(5)}").split()[-1]
        for art, abs_no in [(m.group(1), m.group(2)), (m.group(3), m.group(4))]:
            c = f"Art. {art}"
            if abs_no:
                c += f" Abs. {abs_no}"
            c += f" {family}"
            c = _canon(c)
            if c and c not in seen:
                out.append((c, "conjunction"))
                seen.add(c)
    for m in NATURAL_ARTICLE_RE.finditer(query or ""):
        family = NATURAL_FAMILY_ALIASES.get(m.group(4).lower())
        if not family:
            continue
        c = f"Art. {m.group(1)}"
        if m.group(2):
            c += f" Abs. {m.group(2)}"
        if m.group(3):
            c += f" lit. {m.group(3)}"
        c += f" {family}"
        c = _canon(c)
        if c and c not in seen:
            out.append((c, "natural"))
            seen.add(c)
    return out


def _explicit_citations(query: str) -> list[str]:
    return [c for c, _source in _explicit_citation_records(query)]


def _build_laws_maps() -> tuple[dict[str, str], dict[tuple[str, str], list[str]]]:
    exact: dict[str, str] = {}
    prefix: dict[tuple[str, str], list[str]] = {}
    for row in iter_corpus_rows(include_laws=True, include_court=False):
        c = _canon(row.get("citation", ""))
        if not c:
            continue
        exact[c] = c
        key = _article_key(c)
        if key:
            prefix.setdefault(key, []).append(c)

    def sort_key(c: str) -> tuple[int, int, str]:
        return (0 if re.search(r"\bAbs\.\s*1\b", c) else 1, len(c), c)

    for key, values in list(prefix.items()):
        prefix[key] = sorted(_dedup(values), key=sort_key)
    return exact, prefix


def _apply_rescue(
    rows: list[dict],
    base_pred: dict[str, list[str]],
    exact_map: dict[str, str],
    prefix_map: dict[tuple[str, str], list[str]],
    always_add_art100: bool,
    max_prefix_add: int,
    natural_alias_prefix_add: int,
    conjunction_prefix_add: int,
    conjunction_existing_prefix_add: int,
    add_missing_abs1_for_bare_explicit: bool,
) -> tuple[dict[str, list[str]], list[dict]]:
    pred: dict[str, list[str]] = {}
    trace: list[dict] = []
    for row in rows:
        qid = row["query_id"]
        base = _dedup(base_pred.get(qid, []))
        if always_add_art100:
            base = _dedup(base + ["Art. 100 Abs. 1 BGG"])
        base_set = set(base)
        existing_article_keys = {_article_key(c) for c in base if _article_key(c)}
        additions: list[str] = []
        explicit_records = _explicit_citation_records(row.get("query", ""))
        explicit = [c for c, _source in explicit_records]
        for ex, source in explicit_records:
            key = _article_key(ex)
            if not key:
                continue
            if _has_abs(ex):
                # Exact explicit paragraph is safe to rescue even if another paragraph
                # of the same article was already predicted.
                cand = exact_map.get(ex)
                if cand and cand not in base_set and cand not in additions:
                    additions.append(cand)
                continue
            if key in existing_article_keys:
                if source == "conjunction" and conjunction_existing_prefix_add > 0:
                    for cand in prefix_map.get(key, [])[:conjunction_existing_prefix_add]:
                        if cand not in base_set and cand not in additions:
                            additions.append(cand)
                if add_missing_abs1_for_bare_explicit:
                    for cand in prefix_map.get(key, []):
                        if not re.search(r"\bAbs\.\s*1\b", cand):
                            continue
                        if cand not in base_set and cand not in additions:
                            additions.append(cand)
                        break
                continue
            if source == "natural":
                prefix_limit = natural_alias_prefix_add
            elif source == "conjunction":
                prefix_limit = conjunction_prefix_add
            else:
                prefix_limit = max_prefix_add
            for cand in prefix_map.get(key, [])[:prefix_limit]:
                if cand not in base_set and cand not in additions:
                    additions.append(cand)
        pred[qid] = _dedup(base + additions)
        trace.append(
            {
                "query_id": qid,
                "explicit_citations": ";".join(explicit),
                "additions": ";".join(additions),
                "added_count": len(additions),
            }
        )
    return pred, trace


def _write_predictions(path: Path, rows: list[dict], pred_map: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for row in rows:
            writer.writerow([row["query_id"], ";".join(pred_map.get(row["query_id"], []))])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit article prefix rescue against laws_de.")
    parser.add_argument("--base-val-pred-csv", type=Path, default=ROOT / "artifacts" / "qwen3_reranker_module_ablation" / "val_predictions_qwen3_cap80.csv")
    parser.add_argument("--base-test-submission-csv", type=Path, default=ROOT / "release" / "submission_qwen3_bgg100_prior_v1" / "submission.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "explicit_prefix_rescue")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release" / "submission_explicit_prefix_rescue_v1")
    parser.add_argument("--max-prefix-add", type=int, default=1)
    parser.add_argument("--natural-alias-prefix-add", type=int, default=1)
    parser.add_argument("--conjunction-prefix-add", type=int, default=1)
    parser.add_argument("--conjunction-existing-prefix-add", type=int, default=0)
    parser.add_argument("--add-missing-abs1-for-bare-explicit", action="store_true")
    args = parser.parse_args()

    val_rows = load_query_split("val")
    test_rows = load_query_split("test")
    base_val = _read_predictions(args.base_val_pred_csv)
    base_test = _read_predictions(args.base_test_submission_csv)
    exact_map, prefix_map = _build_laws_maps()

    base_val_art100 = {qid: _dedup(pred + ["Art. 100 Abs. 1 BGG"]) for qid, pred in base_val.items()}
    base_summary, base_eval = evaluate_predictions(val_rows, base_val_art100, mode="strict")
    base_fp = sum(int(r["fp"]) for r in base_eval)
    base_tp = sum(int(r["tp"]) for r in base_eval)

    val_pred, val_trace = _apply_rescue(
        val_rows,
        base_val,
        exact_map=exact_map,
        prefix_map=prefix_map,
        always_add_art100=True,
        max_prefix_add=args.max_prefix_add,
        natural_alias_prefix_add=args.natural_alias_prefix_add,
        conjunction_prefix_add=args.conjunction_prefix_add,
        conjunction_existing_prefix_add=args.conjunction_existing_prefix_add,
        add_missing_abs1_for_bare_explicit=args.add_missing_abs1_for_bare_explicit,
    )
    val_summary, val_eval = evaluate_predictions(val_rows, val_pred, mode="strict")
    val_fp = sum(int(r["fp"]) for r in val_eval)
    val_tp = sum(int(r["tp"]) for r in val_eval)

    test_pred, test_trace = _apply_rescue(
        test_rows,
        base_test,
        exact_map=exact_map,
        prefix_map=prefix_map,
        always_add_art100=False,
        max_prefix_add=args.max_prefix_add,
        natural_alias_prefix_add=args.natural_alias_prefix_add,
        conjunction_prefix_add=args.conjunction_prefix_add,
        conjunction_existing_prefix_add=args.conjunction_existing_prefix_add,
        add_missing_abs1_for_bare_explicit=args.add_missing_abs1_for_bare_explicit,
    )
    changed_qids = [
        row["query_id"]
        for row in test_rows
        if _dedup(test_pred.get(row["query_id"], [])) != _dedup(base_test.get(row["query_id"], []))
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(args.out_dir / "val_predictions.csv", val_rows, val_pred)
    _write_predictions(args.release_dir / "submission.csv", test_rows, test_pred)
    _write_csv(args.out_dir / "val_trace.csv", val_trace)
    _write_csv(args.out_dir / "test_trace.csv", test_trace)

    summary = {
        "base_val_strict_f1": float(base_summary["macro_f1"]),
        "trial_val_strict_f1": float(val_summary["macro_f1"]),
        "delta_val_strict_f1": float(val_summary["macro_f1"]) - float(base_summary["macro_f1"]),
        "base_tp": base_tp,
        "trial_tp": val_tp,
        "base_fp": base_fp,
        "trial_fp": val_fp,
        "delta_fp": val_fp - base_fp,
        "changed_test_query_count": len(changed_qids),
        "changed_test_qids": changed_qids,
        "release_submission": str(args.release_dir / "submission.csv"),
        "add_missing_abs1_for_bare_explicit": bool(args.add_missing_abs1_for_bare_explicit),
        "natural_alias_prefix_add": int(args.natural_alias_prefix_add),
        "conjunction_prefix_add": int(args.conjunction_prefix_add),
        "conjunction_existing_prefix_add": int(args.conjunction_existing_prefix_add),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Explicit Prefix Rescue",
        "",
        "## Result",
        f"- base val strict_f1: `{summary['base_val_strict_f1']:.6f}`",
        f"- trial val strict_f1: `{summary['trial_val_strict_f1']:.6f}`",
        f"- delta val strict_f1: `{summary['delta_val_strict_f1']:.6f}`",
        f"- TP: `{base_tp} -> {val_tp}`",
        f"- FP: `{base_fp} -> {val_fp}`",
        f"- changed test queries: `{len(changed_qids)}`",
        f"- changed test qids: `{';'.join(changed_qids)}`",
        "",
        "## Artifacts",
        f"- `{args.out_dir / 'val_trace.csv'}`",
        f"- `{args.out_dir / 'test_trace.csv'}`",
        f"- `{args.release_dir / 'submission.csv'}`",
    ]
    (ROOT / "docs" / "explicit_prefix_rescue.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

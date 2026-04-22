from __future__ import annotations

import re

from citation_normalizer import normalize_citation


STATUTE_ARTICLE_PAT = re.compile(
    r"\bArt\.?\s*(\d+[a-z]?)"
    r"(?:\s*Abs\.?\s*\d+[a-z]?)?"
    r"(?:\s*lit\.?\s*[a-z])?"
    r"(?:\s+([A-Za-z]{2,10}))?",
    re.I,
)

STATUTE_FAMILY_PAT = re.compile(
    r"\b(ZGB|OR|BGG|STPO|BV|IPRG|ZPO|STGB|ATSG|VVG|SVG|DBG|MWSTG)\b",
    re.I,
)


def _norm_set(citations: list[str]) -> set[str]:
    out: set[str] = set()
    for c in citations:
        nc = normalize_citation(c)
        if nc:
            out.add(nc)
    return out


def _paragraph_aware_key(citation: str) -> str:
    """
    段落感知键（paragraph_aware_key）：
    - 对法规 citation，忽略 Abs./lit.，只保留 Art + family。
    - 对案例 citation，保持精确键。
    """
    text = normalize_citation(citation)
    if not text:
        return ""

    m = STATUTE_ARTICLE_PAT.search(text)
    if m:
        art_num = (m.group(1) or "").upper()
        fam = (m.group(2) or "").upper()
        if not fam:
            mf = STATUTE_FAMILY_PAT.search(text)
            fam = (mf.group(1) if mf else "UNK").upper()
        return f"STATUTE::{fam}::ART::{art_num}"
    return f"EXACT::{text}"


def set_f1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def exact_match_f1(pred_citations: list[str], gold_citations: list[str]) -> tuple[float, float, float]:
    return set_f1(_norm_set(pred_citations), _norm_set(gold_citations))


def paragraph_aware_match_f1(pred_citations: list[str], gold_citations: list[str]) -> tuple[float, float, float]:
    pred_keys = {_paragraph_aware_key(x) for x in pred_citations if _paragraph_aware_key(x)}
    gold_keys = {_paragraph_aware_key(x) for x in gold_citations if _paragraph_aware_key(x)}
    return set_f1(pred_keys, gold_keys)


def truly_unreachable(gold_citations: list[str], corpus_norm_citations: set[str]) -> bool:
    gold_norm = _norm_set(gold_citations)
    if not gold_norm:
        return True
    return all(c not in corpus_norm_citations for c in gold_norm)


def evaluate_paragraph_aware(
    gold_rows: list[dict],
    pred_map: dict[str, list[str]],
    corpus_norm_citations: set[str],
) -> tuple[dict, list[dict]]:
    per_query = []
    f1_scores = []
    unreachable_count = 0

    for row in gold_rows:
        qid = row["query_id"]
        gold_raw = row.get("gold_citation_list") or []
        pred_raw = pred_map.get(qid, [])

        _, _, exact_f1 = exact_match_f1(pred_raw, gold_raw)
        _, _, para_f1 = paragraph_aware_match_f1(pred_raw, gold_raw)
        unreachable = truly_unreachable(gold_raw, corpus_norm_citations)
        if unreachable:
            unreachable_count += 1

        f1_scores.append(para_f1)
        per_query.append(
            {
                "query_id": qid,
                "exact_match_f1": round(exact_f1, 6),
                "paragraph_aware_f1": round(para_f1, 6),
                "truly_unreachable": int(unreachable),
            }
        )

    summary = {
        "queries": len(gold_rows),
        "paragraph_aware_macro_f1": round(sum(f1_scores) / len(f1_scores), 6) if f1_scores else 0.0,
        "truly_unreachable_query_count": unreachable_count,
        "truly_unreachable_ratio": round(unreachable_count / len(gold_rows), 6) if gold_rows else 0.0,
    }
    return summary, per_query

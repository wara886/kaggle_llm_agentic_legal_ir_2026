from __future__ import annotations

import re
from dataclasses import dataclass

from citation_normalizer import normalize_citation
from legal_ir.corpus_builder import iter_corpus_rows


@dataclass
class RuleRetrievedItem:
    citation: str
    source: str
    score: float
    method: str


ARTICLE_PAT = re.compile(r"\b(?:art\.?|article)\s*\d+[a-z]?(?:\s*abs\.?\s*\d+)?(?:\s*lit\.?\s*[a-z])?(?:\s+[A-Z][A-Za-z0-9]+)?", re.I)
BGE_PAT = re.compile(r"\bBGE\s+\d+\s+[IVXLC]+\s+\d+(?:\s+E\.\s*[0-9.]+)?\b", re.I)
CASE_PAT = re.compile(r"\b\d+[A-Za-z]*_[0-9]+/[0-9]{4}(?:\s*E\.\s*[0-9.]+)?\b", re.I)
JUDGMENT_PAT = re.compile(r"\b(?:urteil|judgment|decision)\s*[0-9A-Za-z_./-]+\b", re.I)
NUM_ABBR_PAT = re.compile(r"\b\d+[a-z]?(?:\.\d+)?\s+[A-Z][A-Za-z]{1,10}\b")
KNOWN_LAW_FAMILIES = {
    "ATSG",
    "BGG",
    "BV",
    "CPLR",
    "DBG",
    "IVG",
    "LAI",
    "MWSTG",
    "OR",
    "SCHKG",
    "STGB",
    "STPO",
    "SVG",
    "VVG",
    "ZGB",
    "ZPO",
}


class RuleCitationRetriever:
    """
    规则召回（rule_based citation recall）：
    - 从 query 中抽取法律引用 pattern
    - 在 laws_de / court_considerations 各自做 exact / prefix / regex match
    """

    def __init__(self):
        self.docs: dict[str, list[dict]] = {"laws_de": [], "court_considerations": []}
        self.citation_norm_map: dict[str, dict[str, str]] = {"laws_de": {}, "court_considerations": {}}
        self.stats = {"laws_docs": 0, "court_docs": 0}

    def build(
        self,
        max_laws_rows: int | None = None,
        max_court_rows: int | None = None,
    ) -> dict:
        self.docs = {"laws_de": [], "court_considerations": []}
        self.citation_norm_map = {"laws_de": {}, "court_considerations": {}}
        for row in iter_corpus_rows(
            include_laws=max_laws_rows != 0,
            include_court=max_court_rows != 0,
            max_laws_rows=max_laws_rows,
            max_court_rows=max_court_rows,
        ):
            citation = row["citation"]
            norm = normalize_citation(citation).lower()
            if not norm:
                continue
            source = row["source"]
            self.docs[source].append(
                {
                    "citation": citation,
                    "norm_citation": norm,
                    "law_family": self._extract_law_family(citation),
                }
            )
            self.citation_norm_map[source][norm] = citation
        self.stats = {
            "laws_docs": len(self.docs["laws_de"]),
            "court_docs": len(self.docs["court_considerations"]),
        }
        return dict(self.stats)

    @staticmethod
    def extract_patterns(query: str) -> list[str]:
        patterns = []
        for pat in [ARTICLE_PAT, BGE_PAT, CASE_PAT, JUDGMENT_PAT]:
            patterns.extend(m.group(0).strip() for m in pat.finditer(query or ""))
        dedup = []
        seen = set()
        for p in patterns:
            k = p.lower()
            if k not in seen:
                dedup.append(p)
                seen.add(k)
        return dedup

    @staticmethod
    def _build_relaxed_regex(pattern: str) -> re.Pattern:
        escaped = re.escape(pattern)
        escaped = escaped.replace(r"\ ", r"\s+")
        escaped = escaped.replace(r"\.", r"\.?\s*")
        return re.compile(escaped, re.I)

    @staticmethod
    def _extract_law_family(text: str) -> str:
        if not text:
            return ""
        for tok in reversed(re.findall(r"\b[A-Za-z][A-Za-z0-9-]{1,12}\b", text)):
            family = tok.replace("-", "").upper()
            if family in KNOWN_LAW_FAMILIES:
                return family
        return ""

    @staticmethod
    def _is_laws_article_pattern(pattern: str) -> bool:
        return bool(re.search(r"\b(?:art\.?|article)\s*\d+", pattern or "", re.I))

    def _match_source(self, source: str, patterns: list[str], top_k: int) -> list[RuleRetrievedItem]:
        docs = self.docs[source]
        scored: dict[str, tuple[float, str]] = {}

        for p in patterns:
            p_norm = normalize_citation(p).lower()
            if not p_norm:
                continue
            pattern_family = self._extract_law_family(p)
            if source == "laws_de" and self._is_laws_article_pattern(p) and not pattern_family:
                continue

            # exact
            if p_norm in self.citation_norm_map[source]:
                c = self.citation_norm_map[source][p_norm]
                scored[c] = max(scored.get(c, (0.0, "")), (1.0, "rule_exact"))

            # prefix
            for d in docs:
                nc = d["norm_citation"]
                if pattern_family and d.get("law_family") and d.get("law_family") != pattern_family:
                    continue
                if nc.startswith(p_norm):
                    c = d["citation"]
                    scored[c] = max(scored.get(c, (0.0, "")), (0.8, "rule_prefix"))

            # regex
            if source == "laws_de":
                continue
            rp = self._build_relaxed_regex(p_norm)
            for d in docs:
                nc = d["norm_citation"]
                if rp.search(nc):
                    c = d["citation"]
                    scored[c] = max(scored.get(c, (0.0, "")), (0.6, "rule_regex"))

        ranked = sorted(scored.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        return [
            RuleRetrievedItem(
                citation=citation,
                source=source,
                score=float(score_method[0]),
                method=score_method[1],
            )
            for citation, score_method in ranked
        ]

    def search(self, query: str, top_k_laws: int = 60, top_k_court: int = 60) -> list[RuleRetrievedItem]:
        patterns = self.extract_patterns(query)
        if not patterns:
            return []
        out = []
        if top_k_laws > 0:
            out.extend(self._match_source("laws_de", patterns, top_k=top_k_laws))
        if top_k_court > 0:
            out.extend(self._match_source("court_considerations", patterns, top_k=top_k_court))
        return out


def self_check() -> dict:
    mock = [
        "Art. 221 Abs. 1 StPO",
        "BGE 137 IV 122 E. 6.2",
        "1B_210/2023 E. 4.1",
    ]
    q = "Can court apply Art. 221 Abs. 1 StPO and BGE 137 IV 122 E. 6.2 in case 1B_210/2023 E. 4.1?"
    patterns = RuleCitationRetriever.extract_patterns(q)
    # lightweight dry check without full corpus build
    matched = [c for c in mock if any(normalize_citation(p).lower() in normalize_citation(c).lower() for p in patterns)]
    return {
        "ok": len(patterns) >= 2 and len(matched) >= 2,
        "patterns": patterns,
        "matched_examples": matched,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), ensure_ascii=True, indent=2))

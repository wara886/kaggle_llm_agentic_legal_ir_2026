from __future__ import annotations

import re


# 法律短语（legal_phrase_candidates）候选表：用于从长查询中补充结构化视图
LEGAL_PHRASE_CANDIDATES = [
    "provisional measures",
    "pre trial detention",
    "risk of collusion",
    "burden of proof",
    "statute of limitations",
    "unfair competition",
    "trade secret",
    "copyright infringement",
    "gross negligence",
    "proportionality",
]


# 停用词（stopwords）：轻量关键词抽取用
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "under",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "into",
    "does",
    "do",
    "did",
    "have",
    "has",
    "had",
    "may",
    "can",
    "should",
}


# 编号模式（number_patterns）
ARTICLE_PAT = re.compile(
    r"\b(?:art\.?|article)\s*\d+[a-z]?(?:\s*abs\.?\s*\d+)?(?:\s*lit\.?\s*[a-z])?(?:\s+[A-Z][A-Za-z0-9]+)?",
    re.I,
)
SECTION_PAT = re.compile(r"\bsection\s+\d+[a-z]?\b", re.I)
PARAGRAPH_PAT = re.compile(r"\b(?:paragraph|para\.?|§)\s*\d+[a-z]?\b", re.I)
CASE_PAT = re.compile(r"\b\d+[A-Z]_[0-9]+/[0-9]{4}\b")
JUDGMENT_PAT = re.compile(r"\b\d+[A-Za-z]*_[0-9]+/[0-9]{4}\s*E\.\s*[0-9.]+\b", re.I)
BGE_PAT = re.compile(r"\bBGE\s+\d+\s+[IVXLC]+\s+\d+(?:\s+E\.\s*[0-9.]+)?\b", re.I)
ART_SHORT_PAT = re.compile(
    r"\bArt\.\s*\d+[a-z]?(?:\s*Abs\.\s*\d+)?(?:\s*lit\.\s*[a-z])?(?:\s+[A-Z][A-Za-z0-9]+)?",
    re.I,
)
NUM_ABBR_PAT = re.compile(r"\b\d+[a-z]?(?:\.\d+)?\s+[A-Z][A-Za-z]{1,10}\b")

TOKEN_PAT = re.compile(r"[a-z0-9_./#-]+")
NOISE_PUNCT_PAT = re.compile(r"[^a-z0-9_./#\-\s]")
MULTI_SPACE_PAT = re.compile(r"\s+")


def _clean_query(query: str) -> str:
    # 统一大小写（lowercase）+ 清洗标点（punctuation cleanup）
    text = (query or "").lower()
    text = text.replace("’", "'").replace("`", "'").replace("–", "-").replace("—", "-")
    text = NOISE_PUNCT_PAT.sub(" ", text)
    text = MULTI_SPACE_PAT.sub(" ", text).strip()
    return text


def _extract_number_patterns(query_original: str) -> list[str]:
    patterns = []
    for pat in [
        ARTICLE_PAT,
        SECTION_PAT,
        PARAGRAPH_PAT,
        CASE_PAT,
        JUDGMENT_PAT,
        BGE_PAT,
        ART_SHORT_PAT,
        NUM_ABBR_PAT,
    ]:
        patterns.extend(m.group(0).strip() for m in pat.finditer(query_original or ""))

    dedup = []
    seen = set()
    for p in patterns:
        key = p.lower()
        if key in seen:
            continue
        dedup.append(p)
        seen.add(key)
    return dedup


def _extract_keywords(clean: str, limit: int = 24) -> list[str]:
    tokens = TOKEN_PAT.findall(clean)
    kept = []
    seen = set()
    for t in tokens:
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        kept.append(t)
        if len(kept) >= limit:
            break
    return kept


def _extract_legal_phrases(clean: str, keywords: list[str], number_patterns: list[str]) -> list[str]:
    # 核心短语（core_phrases）：结合词表 + 关键词窗口 + 编号模式
    phrases = []
    for p in LEGAL_PHRASE_CANDIDATES:
        if p in clean:
            phrases.append(p)

    if len(keywords) >= 4:
        phrases.append(" ".join(keywords[:4]))
    if len(keywords) >= 8:
        phrases.append(" ".join(keywords[4:8]))

    for p in number_patterns[:8]:
        phrases.append(p.lower())

    dedup = []
    seen = set()
    for p in phrases:
        k = p.strip().lower()
        if not k or k in seen:
            continue
        dedup.append(p.strip())
        seen.add(k)
    return dedup


def preprocess_query(query_original: str) -> dict:
    """
    查询预处理（query_preprocessing）
    输入：英文法律查询（english_legal_query）
    输出：多视图查询（multi_view_query）
    """
    query_clean = _clean_query(query_original)
    query_number_patterns = _extract_number_patterns(query_original or "")
    query_keywords = _extract_keywords(query_clean)
    query_legal_phrases = _extract_legal_phrases(query_clean, query_keywords, query_number_patterns)
    return {
        "query_original": query_original or "",
        "query_clean": query_clean,
        "query_keywords": query_keywords,
        "query_legal_phrases": query_legal_phrases,
        "query_number_patterns": query_number_patterns,
    }


def build_retrieval_queries(multi_view_query: dict) -> dict:
    """
    检索视图（retrieval_views）构建：
    - sparse_query：偏关键词 + 编号模式
    - dense_query：偏清洗句 + 法律短语
    """
    query_clean = multi_view_query.get("query_clean", "")
    query_keywords = multi_view_query.get("query_keywords", [])
    query_number_patterns = multi_view_query.get("query_number_patterns", [])
    query_legal_phrases = multi_view_query.get("query_legal_phrases", [])

    sparse_query = " ".join([query_clean] + query_keywords[:12] + query_number_patterns[:8]).strip()
    dense_query = " ".join([query_clean] + query_legal_phrases[:8]).strip()
    return {
        "sparse_query": sparse_query if sparse_query else query_clean,
        "dense_query": dense_query if dense_query else query_clean,
    }


def self_check() -> dict:
    sample = (
        "May a court order extension under Art. 221 Abs. 1 lit. b StPO and BGE 137 IV 122 E. 6.2? "
        "Paragraph 3 applies, see section 14 and case 1B_210/2023 E. 4.1."
    )
    mv = preprocess_query(sample)
    rv = build_retrieval_queries(mv)
    ok = (
        bool(mv["query_clean"])
        and len(mv["query_keywords"]) > 0
        and len(mv["query_number_patterns"]) > 0
        and "sparse_query" in rv
        and "dense_query" in rv
    )
    return {
        "ok": ok,
        "multi_view_query": mv,
        "retrieval_views": rv,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), ensure_ascii=True, indent=2))


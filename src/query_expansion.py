from __future__ import annotations

"""
轻量跨语种扩展（cross_lingual_expansion）
- 仅词典/模板（dictionary_template_based）
- 不替换英文原查询，只追加德文扩展视图
"""

from typing import Iterable


# 覆盖词表（high_freq_legal_terms）
EN_DE_LEGAL_DICT = {
    "contract": ["vertrag", "vertragsrecht"],
    "liability": ["haftung"],
    "damages": ["schadenersatz", "schaden"],
    "employment": ["arbeitsrecht", "arbeitsverhaeltnis"],
    "tax": ["steuer", "steuerrecht"],
    "criminal": ["strafrecht", "strafbar"],
    "procedure": ["verfahren", "prozessrecht"],
    "appeal": ["berufung", "beschwerde"],
    "inheritance": ["erbrecht", "nachlass"],
    "property": ["sachenrecht", "eigentum"],
    "company": ["gesellschaftsrecht", "unternehmen"],
    "insurance": ["versicherung", "versicherungsrecht"],
    "evidence": ["beweis", "beweiswuerdigung"],
    "jurisdiction": ["zustaendigkeit", "gerichtsbarkeit"],
    "article": ["artikel", "art."],
    "section": ["abschnitt", "sektion"],
    "paragraph": ["absatz", "para", "paragraph"],
    "court": ["gericht", "bundesgericht"],
}

# 源感知词表（source_aware_lexicon）：法规侧（laws）优先
EN_DE_LAWS_LEGAL_DICT = {
    "article": ["artikel", "art."],
    "paragraph": ["absatz", "para", "paragraph"],
    "section": ["abschnitt", "sektion"],
    "act": ["gesetz", "erlass"],
    "code": ["kodex", "gesetzbuch"],
    "ordinance": ["verordnung"],
    "tax": ["steuer", "steuerrecht"],
    "criminal": ["strafrecht", "strafbar"],
    "civil": ["zivilrecht"],
    "employment": ["arbeitsrecht", "arbeitsverhaeltnis"],
    "insurance": ["versicherung", "versicherungsrecht"],
    "company": ["gesellschaftsrecht", "unternehmen"],
    "inheritance": ["erbrecht", "nachlass"],
    "property": ["sachenrecht", "eigentum"],
    "jurisdiction": ["zustaendigkeit", "gerichtsbarkeit"],
}

# 源感知词表（source_aware_lexicon）：裁判侧（court）优先
EN_DE_COURT_LEGAL_DICT = {
    "appeal": ["berufung", "beschwerde"],
    "evidence": ["beweis", "beweiswuerdigung"],
    "procedure": ["verfahren", "prozessrecht"],
    "burden of proof": ["beweislast"],
    "damages": ["schadenersatz", "schaden"],
    "negligence": ["fahrlassigkeit", "sorgfaltspflichtverletzung"],
    "liability": ["haftung"],
    "termination": ["kuendigung", "beendigung"],
    "invalidity": ["ungueltigkeit", "nichtigkeit"],
    "abuse": ["missbrauch", "rechtsmissbrauch"],
    "compensation": ["entschaedigung", "ausgleich"],
    "contract dispute": ["vertragsstreit", "vertragskonflikt"],
}


DE_TEMPLATE_SNIPPETS = [
    "anwendbares recht",
    "rechtliche wuerdigung",
    "zustaendiges gericht",
    "einschlaegige norm",
]

DE_LAWS_TEMPLATE_SNIPPETS = [
    "einschlaegige gesetzesnorm",
    "anwendbarer artikel",
    "gesetzliche voraussetzungen",
]

# 法规风格扩展词表（statute_style_laws_lexicon）
EN_DE_LAWS_STYLE_DICT = {
    "article": ["artikel", "art."],
    "paragraph": ["absatz", "paragraph", "para"],
    "section": ["abschnitt", "sektion"],
    "chapter": ["kapitel"],
    "code": ["gesetzbuch", "kodex"],
    "act": ["gesetz", "erlass"],
    "ordinance": ["verordnung"],
    "regulation": ["regelung", "vorschrift"],
    "obligation": ["pflicht", "verpflichtung"],
    "requirement": ["voraussetzung", "anforderung"],
    "prohibition": ["verbot", "unzulaessigkeit"],
    "exception": ["ausnahme"],
    "validity": ["gueltigkeit", "wirksamkeit"],
    "termination": ["beendigung", "kuendigung"],
    "liability": ["haftung"],
    "compensation": ["entschaedigung", "ausgleich"],
    "employment law": ["arbeitsrecht"],
    "tax law": ["steuerrecht"],
    "criminal law": ["strafrecht"],
    "civil law": ["zivilrecht"],
    "insurance law": ["versicherungsrecht"],
    "company law": ["gesellschaftsrecht"],
    "inheritance law": ["erbrecht"],
    "property law": ["sachenrecht"],
}

DE_LAWS_STYLE_TEMPLATES = [
    "gesetzliche regelung und voraussetzungen",
    "anwendbare artikel und absaetze",
    "pflichten aus gesetz und verordnung",
    "ausnahmen und wirksamkeit der norm",
]

DE_COURT_TEMPLATE_SNIPPETS = [
    "gerichtliche erwaegungen",
    "beweiswuerdigung und verfahren",
    "beschwerde und entscheidgruende",
]


def _contains_token(query_clean: str, token: str) -> bool:
    text = f" {query_clean} "
    return f" {token} " in text


def _collect_de_keywords(query_clean: str, query_keywords: Iterable[str]) -> list[str]:
    matched = []
    seen = set()
    en_tokens = set(query_keywords) | set(query_clean.split())
    for en, de_terms in EN_DE_LEGAL_DICT.items():
        if en in en_tokens or _contains_token(query_clean, en):
            for de in de_terms:
                if de in seen:
                    continue
                matched.append(de)
                seen.add(de)
    return matched


def _collect_de_keywords_from_dict(
    query_clean: str,
    query_keywords: Iterable[str],
    lexicon: dict[str, list[str]],
) -> list[str]:
    matched = []
    seen = set()
    en_tokens = set(query_keywords) | set(query_clean.split())
    lower_clean = query_clean.lower()
    for en, de_terms in lexicon.items():
        hit = (en in en_tokens) or _contains_token(lower_clean, en)
        if not hit and " " in en:
            hit = en in lower_clean
        if hit:
            for de in de_terms:
                if de in seen:
                    continue
                matched.append(de)
                seen.add(de)
    return matched


def expand_query_from_multi_view(multi_view_query: dict) -> dict:
    """
    输入：multi_view_query
    输出：
    - expanded_query_de
    - expanded_keywords_de
    - bilingual_query_pack
    """
    query_original = multi_view_query.get("query_original", "")
    query_clean = multi_view_query.get("query_clean", "")
    query_keywords = multi_view_query.get("query_keywords", [])
    query_legal_phrases = multi_view_query.get("query_legal_phrases", [])
    query_number_patterns = multi_view_query.get("query_number_patterns", [])

    expanded_keywords_de = _collect_de_keywords(query_clean, query_keywords)
    template_terms = DE_TEMPLATE_SNIPPETS if expanded_keywords_de else []
    number_terms = [x.lower() for x in query_number_patterns[:8]]
    expanded_query_de = " ".join(expanded_keywords_de + template_terms + number_terms).strip()

    bilingual_query_pack = {
        "query_en_original": query_original,
        "query_en_clean": query_clean,
        "query_en_keywords": query_keywords,
        "query_en_legal_phrases": query_legal_phrases,
        "query_number_patterns": query_number_patterns,
        "query_de_expanded": expanded_query_de,
        "query_de_keywords": expanded_keywords_de,
    }
    return {
        "expanded_query_de": expanded_query_de,
        "expanded_keywords_de": expanded_keywords_de,
        "bilingual_query_pack": bilingual_query_pack,
    }


def build_bilingual_retrieval_views(multi_view_query: dict) -> dict:
    """
    统一接口（unified_interface）：
    为 retrieval_sparse.py / retrieval_dense.py 提供英德双视图。
    """
    query_clean = multi_view_query.get("query_clean", "")
    query_keywords = multi_view_query.get("query_keywords", [])
    query_legal_phrases = multi_view_query.get("query_legal_phrases", [])
    query_number_patterns = multi_view_query.get("query_number_patterns", [])

    expanded = expand_query_from_multi_view(multi_view_query)
    expanded_query_de = expanded["expanded_query_de"]
    expanded_keywords_de = expanded["expanded_keywords_de"]

    sparse_query_en = " ".join([query_clean] + query_keywords[:12] + query_number_patterns[:8]).strip()
    dense_query_en = " ".join([query_clean] + query_legal_phrases[:8]).strip()

    sparse_query_bilingual = " ".join([sparse_query_en, expanded_query_de] + expanded_keywords_de[:12]).strip()
    dense_query_bilingual = " ".join([dense_query_en, expanded_query_de]).strip()

    return {
        "sparse_query_en": sparse_query_en if sparse_query_en else query_clean,
        "dense_query_en": dense_query_en if dense_query_en else query_clean,
        "sparse_query_bilingual": sparse_query_bilingual if sparse_query_bilingual else sparse_query_en,
        "dense_query_bilingual": dense_query_bilingual if dense_query_bilingual else dense_query_en,
        "expanded_query_de": expanded_query_de,
        "expanded_keywords_de": expanded_keywords_de,
        "bilingual_query_pack": expanded["bilingual_query_pack"],
    }


def build_source_aware_query_packs(multi_view_query: dict) -> dict:
    """
    源感知查询扩展（source_aware_query_expansion）：
    - 法规查询包（laws_query_pack）
    - 裁判查询包（court_query_pack）
    """
    query_clean = multi_view_query.get("query_clean", "")
    query_keywords = multi_view_query.get("query_keywords", [])
    query_legal_phrases = multi_view_query.get("query_legal_phrases", [])
    query_number_patterns = multi_view_query.get("query_number_patterns", [])
    query_original = multi_view_query.get("query_original", "")

    laws_keywords_de = _collect_de_keywords_from_dict(query_clean, query_keywords, EN_DE_LAWS_LEGAL_DICT)
    court_keywords_de = _collect_de_keywords_from_dict(query_clean, query_keywords, EN_DE_COURT_LEGAL_DICT)

    expanded_query_de_laws = " ".join(
        laws_keywords_de + DE_LAWS_TEMPLATE_SNIPPETS + [x.lower() for x in query_number_patterns[:8]]
    ).strip()
    expanded_query_de_court = " ".join(
        court_keywords_de + DE_COURT_TEMPLATE_SNIPPETS + [x.lower() for x in query_number_patterns[:8]]
    ).strip()

    laws_query_pack = {
        "query_original": query_original,
        "query_clean": query_clean,
        "query_keywords": query_keywords,
        "query_legal_phrases": query_legal_phrases,
        "query_number_patterns": query_number_patterns,
        "expanded_query_de": expanded_query_de_laws,
        "expanded_keywords_de": laws_keywords_de,
    }
    court_query_pack = {
        "query_original": query_original,
        "query_clean": query_clean,
        "query_keywords": query_keywords,
        "query_legal_phrases": query_legal_phrases,
        "query_number_patterns": query_number_patterns,
        "expanded_query_de": expanded_query_de_court,
        "expanded_keywords_de": court_keywords_de,
    }

    # laws v2: 法规风格强化版
    laws_v2_keywords_de = _collect_de_keywords_from_dict(query_clean, query_keywords, EN_DE_LAWS_STYLE_DICT)
    expanded_query_de_laws_v2 = " ".join(
        laws_v2_keywords_de
        + DE_LAWS_STYLE_TEMPLATES
        + DE_LAWS_TEMPLATE_SNIPPETS
        + [x.lower() for x in query_number_patterns[:10]]
    ).strip()
    laws_query_pack_v2 = {
        "query_original": query_original,
        "query_clean": query_clean,
        "query_keywords": query_keywords,
        "query_legal_phrases": query_legal_phrases,
        "query_number_patterns": query_number_patterns,
        "expanded_query_de": expanded_query_de_laws_v2,
        "expanded_keywords_de": laws_v2_keywords_de,
    }

    return {
        "expanded_query_de_laws": expanded_query_de_laws,
        "expanded_query_de_court": expanded_query_de_court,
        "expanded_query_de_laws_v2": expanded_query_de_laws_v2,
        "laws_query_pack": laws_query_pack,
        "laws_query_pack_v2": laws_query_pack_v2,
        "court_query_pack": court_query_pack,
    }


def self_check() -> dict:
    sample_mv = {
        "query_original": "Can the court award damages for breach of contract under Art. 97 OR?",
        "query_clean": "can the court award damages for breach of contract under art. 97 or",
        "query_keywords": ["court", "award", "damages", "breach", "contract", "art.", "97"],
        "query_legal_phrases": ["breach contract damages", "court award damages"],
        "query_number_patterns": ["Art. 97 OR"],
    }
    out = build_bilingual_retrieval_views(sample_mv)
    sap = build_source_aware_query_packs(sample_mv)
    ok = bool(out["expanded_query_de"]) and bool(
        sap["expanded_query_de_laws"] or sap["expanded_query_de_court"] or sap["expanded_query_de_laws_v2"]
    )
    return {"ok": ok, "output": out, "source_aware": sap}


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), ensure_ascii=True, indent=2))

from __future__ import annotations

from dataclasses import dataclass
import re


# 案例模式（case citation pattern）
CASE_PATTERNS = [
    re.compile(r"\b\d+[A-Z]_[0-9]+/[0-9]{4}\b"),
    re.compile(r"\b(?:1B|6B|8C|5A|4A)_[0-9]+/[0-9]{4}\b", re.I),
    re.compile(r"\bBGE\s+\d+\s+[IVXLC]+\s+\d+\b", re.I),
    re.compile(r"\bE\.\s*\d+(?:\.\d+)*\b", re.I),
]


# 法规模式（statute citation pattern）
STATUTE_PATTERNS = [
    re.compile(r"\bArt\.?\s*\d+[a-z]?\b", re.I),
    re.compile(r"\barticle\s+\d+[a-z]?\b", re.I),
    re.compile(r"\bAbs\.?\s*\d+[a-z]?\b", re.I),
    re.compile(r"\blit\.?\s*[a-z]\b", re.I),
]


STATUTE_FAMILY_PAT = re.compile(
    r"\b(?:ZGB|OR|BGG|StPO|BV|IPRG|IVG|ATSG|SVG|ZPO|StGB|VVG|DBG|MWSTG)\b",
    re.I,
)


@dataclass
class RouteDecision:
    primary_source: str
    secondary_source: str
    route_confidence: str
    case_signal_count: int
    statute_signal_count: int
    mixed_signal: bool
    case_score: float = 0.0
    statute_score: float = 0.0
    mixed_score: float = 0.0
    matched_case_patterns: list[str] | None = None
    matched_statute_patterns: list[str] | None = None
    route_decision_reason_v1_1: str = ""


def _count_hits(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def route_query(query: str, query_number_patterns: list[str] | None = None) -> RouteDecision:
    """
    查询源路由（query_source_router）规则：
    1) 案例模式强命中 -> court
    2) 法规模式强命中 -> laws
    3) case + statute 同时强命中 -> hybrid
    4) 不确定默认 hybrid
    """
    text = query or ""
    extra = " ".join(query_number_patterns or [])
    merged = f"{text} {extra}".strip()

    case_hits = _count_hits(CASE_PATTERNS, merged)
    statute_hits = _count_hits(STATUTE_PATTERNS, merged)
    statute_hits += len(STATUTE_FAMILY_PAT.findall(merged))

    has_case = case_hits > 0
    has_statute = statute_hits > 0
    mixed_signal = has_case and has_statute

    if mixed_signal:
        return RouteDecision(
            primary_source="hybrid",
            secondary_source="none",
            route_confidence="high" if case_hits + statute_hits >= 3 else "medium",
            case_signal_count=case_hits,
            statute_signal_count=statute_hits,
            mixed_signal=True,
        )

    if has_case and not has_statute:
        return RouteDecision(
            primary_source="court",
            secondary_source="laws",
            route_confidence="high" if case_hits >= 2 else "medium",
            case_signal_count=case_hits,
            statute_signal_count=statute_hits,
            mixed_signal=False,
        )

    if has_statute and not has_case:
        return RouteDecision(
            primary_source="laws",
            secondary_source="court",
            route_confidence="high" if statute_hits >= 2 else "medium",
            case_signal_count=case_hits,
            statute_signal_count=statute_hits,
            mixed_signal=False,
        )

    return RouteDecision(
        primary_source="hybrid",
        secondary_source="none",
        route_confidence="low",
        case_signal_count=case_hits,
        statute_signal_count=statute_hits,
        mixed_signal=False,
    )


def _collect_matches(patterns: list[tuple[str, re.Pattern[str]]], text: str) -> list[str]:
    out: list[str] = []
    for name, pat in patterns:
        if pat.search(text):
            out.append(name)
    return out


def route_query_v1_1(query: str, query_number_patterns: list[str] | None = None) -> RouteDecision:
    """
    source_router_v1_1:
    - 增强 case/statute/mixed 信号
    - 增加 score 与 debug 输出
    - 默认不明确时走 hybrid（而非 laws）
    """
    text = query or ""
    extra = " ".join(query_number_patterns or [])
    merged = f"{text} {extra}".strip()

    case_signal_defs = [
        ("case_number", re.compile(r"\b\d+[A-Z]_[0-9]+/[0-9]{4}\b")),
        ("common_case_prefix", re.compile(r"\b(?:1B|6B|8C|5A|4A)_[0-9]+/[0-9]{4}\b", re.I)),
        ("bge", re.compile(r"\bBGE\s+\d+\s+[IVXLC]+\s+\d+\b", re.I)),
        ("judgment_paragraph", re.compile(r"\bE\.\s*\d+(?:\.\d+)*\b", re.I)),
    ]
    statute_signal_defs = [
        ("art", re.compile(r"\bArt\.?\s*\d+[a-z]?\b", re.I)),
        ("article", re.compile(r"\barticle\s+\d+[a-z]?\b", re.I)),
        ("abs", re.compile(r"\bAbs\.?\s*\d+[a-z]?\b", re.I)),
        ("lit", re.compile(r"\blit\.?\s*[a-z]\b", re.I)),
        (
            "statute_family",
            re.compile(r"\b(?:ZGB|OR|BGG|StPO|BV|IPRG|IVG|ATSG|SVG|ZPO|StGB|VVG|DBG|MWSTG)\b", re.I),
        ),
    ]

    matched_case_patterns = _collect_matches(case_signal_defs, merged)
    matched_statute_patterns = _collect_matches(statute_signal_defs, merged)

    case_hits = _count_hits([p for _, p in case_signal_defs], merged)
    statute_hits = _count_hits([p for _, p in statute_signal_defs], merged)

    # 多案例模式同时出现时显著增强 court 倾向
    multi_case_bonus = 1.0 if len(matched_case_patterns) >= 2 else 0.0
    case_score = float(case_hits) + multi_case_bonus
    statute_score = float(statute_hits)
    mixed_score = min(case_score, statute_score)

    has_case = case_score > 0
    has_statute = statute_score > 0

    if has_case and has_statute and mixed_score >= 1.0:
        return RouteDecision(
            primary_source="hybrid",
            secondary_source="none",
            route_confidence="high" if mixed_score >= 2.0 else "medium",
            case_signal_count=case_hits,
            statute_signal_count=statute_hits,
            mixed_signal=True,
            case_score=round(case_score, 4),
            statute_score=round(statute_score, 4),
            mixed_score=round(mixed_score, 4),
            matched_case_patterns=matched_case_patterns,
            matched_statute_patterns=matched_statute_patterns,
            route_decision_reason_v1_1="both_case_and_statute_signals_strong->hybrid",
        )

    score_gap = case_score - statute_score
    if case_score >= 1.5 and score_gap >= 1.0:
        return RouteDecision(
            primary_source="court",
            secondary_source="laws",
            route_confidence="high" if case_score >= 2.5 else "medium",
            case_signal_count=case_hits,
            statute_signal_count=statute_hits,
            mixed_signal=False,
            case_score=round(case_score, 4),
            statute_score=round(statute_score, 4),
            mixed_score=round(mixed_score, 4),
            matched_case_patterns=matched_case_patterns,
            matched_statute_patterns=matched_statute_patterns,
            route_decision_reason_v1_1="case_score_dominates->court",
        )

    if statute_score >= 1.5 and (statute_score - case_score) >= 1.0:
        return RouteDecision(
            primary_source="laws",
            secondary_source="court",
            route_confidence="high" if statute_score >= 2.5 else "medium",
            case_signal_count=case_hits,
            statute_signal_count=statute_hits,
            mixed_signal=False,
            case_score=round(case_score, 4),
            statute_score=round(statute_score, 4),
            mixed_score=round(mixed_score, 4),
            matched_case_patterns=matched_case_patterns,
            matched_statute_patterns=matched_statute_patterns,
            route_decision_reason_v1_1="statute_score_dominates->laws",
        )

    # 默认策略从 laws 改为 hybrid（保守分流）
    return RouteDecision(
        primary_source="hybrid",
        secondary_source="none",
        route_confidence="low",
        case_signal_count=case_hits,
        statute_signal_count=statute_hits,
        mixed_signal=bool(has_case and has_statute),
        case_score=round(case_score, 4),
        statute_score=round(statute_score, 4),
        mixed_score=round(mixed_score, 4),
        matched_case_patterns=matched_case_patterns,
        matched_statute_patterns=matched_statute_patterns,
        route_decision_reason_v1_1="uncertain_signal->hybrid_default",
    )


def self_check() -> dict:
    samples = [
        "8C_160/2016 E. 4.1 with burden of proof",
        "Art. 221 Abs. 1 lit. b StPO proportionality",
        "BGE 137 IV 122 E. 6.2 and Art. 29 BV",
        "general legal issue without citation",
    ]
    out = []
    for s in samples:
        d = route_query_v1_1(s)
        out.append(
            {
                "query": s,
                "primary_source": d.primary_source,
                "secondary_source": d.secondary_source,
                "route_confidence": d.route_confidence,
                "case_signal_count": d.case_signal_count,
                "statute_signal_count": d.statute_signal_count,
                "case_score": d.case_score,
                "statute_score": d.statute_score,
                "mixed_score": d.mixed_score,
                "matched_case_patterns": d.matched_case_patterns,
                "matched_statute_patterns": d.matched_statute_patterns,
                "route_decision_reason_v1_1": d.route_decision_reason_v1_1,
            }
        )
    return {"ok": True, "rows": out}


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), ensure_ascii=False, indent=2))

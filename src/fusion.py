from __future__ import annotations

from collections import defaultdict


def _source_factor(
    citation: str,
    citation_to_source: dict[str, str] | None,
    source_aware_fusion: bool,
    laws_weight: float,
    court_weight: float,
) -> float:
    if not source_aware_fusion or citation_to_source is None:
        return 1.0
    source = citation_to_source.get(citation, "")
    if source == "laws_de":
        return laws_weight
    if source == "court_considerations":
        return court_weight
    return 1.0


def _branch_bonus_factor(
    citation: str,
    base_rank: int,
    top_n: int,
    branch_hits: dict[str, set[str]] | None,
    sparse_laws_rank: dict[str, int] | None,
    enable_branch_aware_fusion: bool,
    branch_aware_fusion_mode: str,
    sparse_laws_branch_bonus: float,
    sparse_laws_single_branch_bonus: float,
    branch_aware_rank_cutoff: int,
) -> float:
    if not enable_branch_aware_fusion:
        return 1.0
    if branch_hits is None or sparse_laws_rank is None:
        return 1.0

    hits = branch_hits.get(citation, set())
    support_count = len(hits)
    in_sparse_laws = "sparse_laws" in hits
    sparse_rank = sparse_laws_rank.get(citation, 10**9)
    is_single_branch = support_count == 1
    in_sparse_rank_cutoff = sparse_rank <= max(branch_aware_rank_cutoff, 1)
    # “fusion 前列”采用当前融合 top_n 之前作为可操作近似。
    not_in_fusion_front = base_rank > max(top_n, 1)

    bonus = 0.0
    if branch_aware_fusion_mode == "sparse_laws_bonus":
        if in_sparse_laws:
            bonus += max(sparse_laws_branch_bonus, 0.0)
            if is_single_branch:
                bonus += max(sparse_laws_single_branch_bonus, 0.0)
    elif branch_aware_fusion_mode == "sparse_laws_tail_rescue":
        if in_sparse_laws and is_single_branch and in_sparse_rank_cutoff and not_in_fusion_front:
            bonus += max(sparse_laws_branch_bonus, 0.0)
            bonus += max(sparse_laws_single_branch_bonus, 0.0)
    return 1.0 + bonus


def rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_n: int = 20,
    citation_to_source: dict[str, str] | None = None,
    source_aware_fusion: bool = False,
    laws_weight: float = 1.0,
    court_weight: float = 1.0,
    branch_hits: dict[str, set[str]] | None = None,
    sparse_laws_rank: dict[str, int] | None = None,
    enable_branch_aware_fusion: bool = False,
    branch_aware_fusion_mode: str = "sparse_laws_bonus",
    sparse_laws_branch_bonus: float = 0.0,
    sparse_laws_single_branch_bonus: float = 0.0,
    branch_aware_rank_cutoff: int = 200,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, citation in enumerate(lst, start=1):
            base = 1.0 / (k + rank)
            sf = _source_factor(citation, citation_to_source, source_aware_fusion, laws_weight, court_weight)
            scores[citation] += base * sf
    base_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not enable_branch_aware_fusion:
        return base_ranked[:top_n]

    base_rank_map = {citation: idx for idx, (citation, _) in enumerate(base_ranked, start=1)}
    rescored = []
    for citation, score in base_ranked:
        factor = _branch_bonus_factor(
            citation=citation,
            base_rank=base_rank_map[citation],
            top_n=top_n,
            branch_hits=branch_hits,
            sparse_laws_rank=sparse_laws_rank,
            enable_branch_aware_fusion=enable_branch_aware_fusion,
            branch_aware_fusion_mode=branch_aware_fusion_mode,
            sparse_laws_branch_bonus=sparse_laws_branch_bonus,
            sparse_laws_single_branch_bonus=sparse_laws_single_branch_bonus,
            branch_aware_rank_cutoff=branch_aware_rank_cutoff,
        )
        rescored.append((citation, score * factor))
    ranked = sorted(rescored, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def weighted_score_fusion(
    score_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    top_n: int = 20,
    citation_to_source: dict[str, str] | None = None,
    source_aware_fusion: bool = False,
    laws_weight: float = 1.0,
    court_weight: float = 1.0,
    branch_hits: dict[str, set[str]] | None = None,
    sparse_laws_rank: dict[str, int] | None = None,
    enable_branch_aware_fusion: bool = False,
    branch_aware_fusion_mode: str = "sparse_laws_bonus",
    sparse_laws_branch_bonus: float = 0.0,
    sparse_laws_single_branch_bonus: float = 0.0,
    branch_aware_rank_cutoff: int = 200,
) -> list[tuple[str, float]]:
    if not score_lists:
        return []
    if weights is None:
        weights = [1.0] * len(score_lists)
    if len(weights) != len(score_lists):
        raise ValueError("weights length must match score_lists length")

    total: dict[str, float] = defaultdict(float)
    for score_list, w in zip(score_lists, weights):
        if not score_list:
            continue
        vals = [s for _, s in score_list]
        smin, smax = min(vals), max(vals)
        denom = (smax - smin) + 1e-9
        for citation, score in score_list:
            norm_score = (score - smin) / denom
            sf = _source_factor(citation, citation_to_source, source_aware_fusion, laws_weight, court_weight)
            total[citation] += w * norm_score * sf
    base_ranked = sorted(total.items(), key=lambda x: x[1], reverse=True)
    if not enable_branch_aware_fusion:
        return base_ranked[:top_n]

    base_rank_map = {citation: idx for idx, (citation, _) in enumerate(base_ranked, start=1)}
    rescored = []
    for citation, score in base_ranked:
        factor = _branch_bonus_factor(
            citation=citation,
            base_rank=base_rank_map[citation],
            top_n=top_n,
            branch_hits=branch_hits,
            sparse_laws_rank=sparse_laws_rank,
            enable_branch_aware_fusion=enable_branch_aware_fusion,
            branch_aware_fusion_mode=branch_aware_fusion_mode,
            sparse_laws_branch_bonus=sparse_laws_branch_bonus,
            sparse_laws_single_branch_bonus=sparse_laws_single_branch_bonus,
            branch_aware_rank_cutoff=branch_aware_rank_cutoff,
        )
        rescored.append((citation, score * factor))
    ranked = sorted(rescored, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

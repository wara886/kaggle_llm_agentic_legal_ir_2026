from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from fusion import rrf_fusion
from law_family import (
    augment_bilingual_pack,
    augment_laws_query_pack,
    build_issue_laws_query_pack,
    boost_items_by_family,
    constrain_items_by_family,
    extract_family_from_citation,
    family_query_terms,
    filter_items_by_family,
    issue_phrase_groups,
    issue_query_terms,
    likely_statute_families,
)
from query_expansion import build_source_aware_query_packs, expand_query_from_multi_view
from query_preprocess import preprocess_query
from rerank import NoOpReranker, TokenOverlapReranker
from retrieval_dense import DenseRetriever
from retrieval_rules import RuleCitationRetriever
from retrieval_sparse import SparseRetriever
from source_router import RouteDecision, route_query, route_query_v1_1
from legal_ir.data_loader import load_query_split


def parse_bool_flag(value: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expect true/false")


def write_prediction_csv(path: Path, pred_map: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(pred_map.keys()):
            writer.writerow([qid, ";".join(pred_map[qid])])


def dedup_keep_max(score_items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def merge_retrieved_items(*item_lists: list) -> list:
    best: dict[tuple[str, str], object] = {}
    for items in item_lists:
        for item in items:
            key = (getattr(item, "citation", ""), getattr(item, "source", ""))
            if not key[0]:
                continue
            prev = best.get(key)
            if prev is None or float(getattr(item, "score", 0.0)) > float(getattr(prev, "score", 0.0)):
                best[key] = item
    return sorted(best.values(), key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)


def _citation_family_consistent(citation: str, families: list[str]) -> bool:
    if not families:
        return False
    citation_family = extract_family_from_citation(citation)
    return bool(citation_family and citation_family in {f.upper() for f in families})


def _build_rerank_candidates_with_laws_shaping(
    fused: list[tuple[str, float]],
    doc_lookup: dict[str, dict],
    sparse_laws_items: list,
    dense_laws_items: list,
    issue_sparse_items: list,
    rule_laws_items: list,
    likely_families: list[str],
    shortlist_size: int = 320,
    keep_fused_tail: int = 160,
) -> tuple[list[dict], dict]:
    shortlist_size = max(1, int(shortlist_size))
    keep_fused_tail = max(0, min(int(keep_fused_tail), shortlist_size))
    priority_budget = max(0, shortlist_size - keep_fused_tail)

    fused_rank = {c: i + 1 for i, (c, _s) in enumerate(fused)}
    fused_score = {c: float(s) for c, s in fused}
    family_set = {x.upper() for x in likely_families if x}
    issue_hit_set = {getattr(x, "citation", "") for x in issue_sparse_items if getattr(x, "citation", "")}
    rule_norm_hit_set = {
        normalize_citation(getattr(x, "citation", ""))
        for x in rule_laws_items
        if normalize_citation(getattr(x, "citation", ""))
    }

    pool: dict[str, dict] = {}

    def upsert(citation: str, retrieval_score: float, source_hint: str = "") -> None:
        if not citation:
            return
        doc = doc_lookup.get(citation, {})
        source = doc.get("source", source_hint or "")
        rec = pool.get(citation)
        if rec is None:
            norm_c = normalize_citation(citation)
            is_rule_hit = int(bool(norm_c and norm_c in rule_norm_hit_set))
            is_norm_consistent = is_rule_hit
            is_family_hit = int(bool(family_set) and _citation_family_consistent(citation, likely_families))
            is_issue_hit = int(citation in issue_hit_set)
            is_laws = int(source == "laws_de")
            # family+issue > rule exact > family > issue
            priority_tier = 0
            if is_family_hit and is_issue_hit:
                priority_tier = 4
            elif is_rule_hit and is_norm_consistent:
                priority_tier = 3
            elif is_family_hit:
                priority_tier = 2
            elif is_issue_hit:
                priority_tier = 1
            pool[citation] = {
                "citation": citation,
                "source": source,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "fused_score": float(fused_score.get(citation, retrieval_score)),
                "score": float(fused_score.get(citation, retrieval_score)),
                "admission_priority_tier": priority_tier,
                "admission_is_rule_hit": is_rule_hit,
                "admission_is_normalization_consistent": is_norm_consistent,
                "admission_is_family_hit": is_family_hit,
                "admission_is_issue_hit": is_issue_hit,
                "admission_is_laws": is_laws,
                "admission_fused_rank": int(fused_rank.get(citation, 10**9)),
                "admission_pool_score": float(retrieval_score),
            }
        else:
            rec["admission_pool_score"] = max(float(rec.get("admission_pool_score", 0.0)), float(retrieval_score))
            if citation in fused_score:
                rec["fused_score"] = float(fused_score[citation])
                rec["score"] = float(fused_score[citation])

    for c, s in fused:
        upsert(citation=c, retrieval_score=float(s))
    for it in sparse_laws_items:
        upsert(citation=getattr(it, "citation", ""), retrieval_score=float(getattr(it, "score", 0.0)), source_hint="laws_de")
    for it in dense_laws_items:
        upsert(citation=getattr(it, "citation", ""), retrieval_score=float(getattr(it, "score", 0.0)), source_hint="laws_de")
    for it in issue_sparse_items:
        upsert(citation=getattr(it, "citation", ""), retrieval_score=float(getattr(it, "score", 0.0)), source_hint="laws_de")
    for it in rule_laws_items:
        upsert(citation=getattr(it, "citation", ""), retrieval_score=float(getattr(it, "score", 0.0)), source_hint="laws_de")

    def priority_key(item: dict) -> tuple:
        return (
            int(item.get("admission_is_laws", 0)),
            int(item.get("admission_priority_tier", 0)),
            int(item.get("admission_is_rule_hit", 0)),
            int(item.get("admission_is_family_hit", 0)),
            int(item.get("admission_is_issue_hit", 0)),
            -int(item.get("admission_fused_rank", 10**9)),
            float(item.get("admission_pool_score", 0.0)),
        )

    selected: list[dict] = []
    selected_set: set[str] = set()

    priority_pool = [
        x
        for x in pool.values()
        if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_priority_tier", 0)) > 0
    ]
    priority_pool.sort(key=priority_key, reverse=True)
    for x in priority_pool:
        c = x["citation"]
        if c in selected_set:
            continue
        selected.append(x)
        selected_set.add(c)
        if len(selected) >= priority_budget:
            break

    for c, _s in fused:
        if c in selected_set:
            continue
        x = pool.get(c)
        if x is None:
            continue
        selected.append(x)
        selected_set.add(c)
        if len(selected) >= shortlist_size:
            break

    if len(selected) < shortlist_size:
        leftovers = [x for x in pool.values() if x["citation"] not in selected_set]
        leftovers.sort(key=priority_key, reverse=True)
        for x in leftovers:
            c = x["citation"]
            selected.append(x)
            selected_set.add(c)
            if len(selected) >= shortlist_size:
                break

    stats = {
        "shortlist_size": int(shortlist_size),
        "keep_fused_tail": int(keep_fused_tail),
        "priority_budget": int(priority_budget),
        "priority_selected": int(
            sum(
                1
                for x in selected
                if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_priority_tier", 0)) > 0
            )
        ),
        "selected_rule_hit": int(sum(1 for x in selected if int(x.get("admission_is_rule_hit", 0)) == 1)),
        "selected_family_hit": int(sum(1 for x in selected if int(x.get("admission_is_family_hit", 0)) == 1)),
        "selected_issue_hit": int(sum(1 for x in selected if int(x.get("admission_is_issue_hit", 0)) == 1)),
    }
    return selected[:shortlist_size], stats


def _fuse_court_branch_candidates(
    sparse_court_items: list,
    dense_court_items: list,
    fusion_mode: str,
    dense_weight: float,
    top_n: int = 220,
) -> list[str]:
    sparse_rank = [x.citation for x in sorted(sparse_court_items, key=lambda y: float(y.score), reverse=True)]
    dense_rank = [x.citation for x in sorted(dense_court_items, key=lambda y: float(y.score), reverse=True)]
    if fusion_mode == "weighted":
        score_map: dict[str, float] = {}
        for i, c in enumerate(sparse_rank):
            score_map[c] = score_map.get(c, 0.0) + 1.0 / (60.0 + i)
        for i, c in enumerate(dense_rank):
            score_map[c] = score_map.get(c, 0.0) + max(0.0, dense_weight) / (60.0 + i)
        return [c for c, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    fused = rrf_fusion(
        ranked_lists=[sparse_rank, dense_rank],
        k=60,
        top_n=top_n,
    )
    return [c for c, _ in fused]


def build_doc_lookup(sparse: SparseRetriever, dense: DenseRetriever) -> dict[str, dict]:
    out = {}
    for source_docs in sparse.docs.values():
        for d in source_docs:
            out[d["citation"]] = d
    all_dense = dense.doc_matrix.get("all_docs", [])
    for d in all_dense:
        if d["citation"] not in out:
            out[d["citation"]] = d
    return out


@dataclass
class RerankerHandle:
    name: str
    fallback_reason: str
    rerank_fn: object


class StrongBGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates:
            return []
        pairs = []
        for c in candidates:
            title = c.get("title", "")
            text = c.get("text", "")
            citation = c.get("citation", "")
            pairs.append((query, f"{citation} {title} {text}".strip()))
        scores = self.model.predict(pairs)
        rescored = []
        for c, s in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = float(s)
            rescored.append(item)
        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return rescored[:top_n]


def build_reranker(prefer_strong: bool = True) -> RerankerHandle:
    if prefer_strong:
        try:
            reranker = StrongBGEReranker()
            return RerankerHandle(
                name="BAAI/bge-reranker-v2-m3",
                fallback_reason="",
                rerank_fn=reranker,
            )
        except Exception as exc:
            return RerankerHandle(
                name="token_overlap",
                fallback_reason=f"strong_reranker_unavailable: {type(exc).__name__}",
                rerank_fn=TokenOverlapReranker(),
            )
    return RerankerHandle(
        name="token_overlap",
        fallback_reason="prefer_strong=False",
        rerank_fn=TokenOverlapReranker(),
    )


def apply_dynamic_cut(
    reranked: list[dict],
    mode: str,
    fixed_top_k: int,
    score_threshold: float,
    relative_threshold: float,
) -> list[dict]:
    if not reranked:
        return []
    if mode == "fixed_top_k":
        return reranked[:fixed_top_k]
    if mode == "score_threshold":
        out = [x for x in reranked if float(x.get("rerank_score", x.get("score", 0.0))) >= score_threshold]
        return out if out else reranked[:fixed_top_k]
    if mode == "relative_threshold":
        top1 = float(reranked[0].get("rerank_score", reranked[0].get("score", 0.0)))
        cutoff = top1 * relative_threshold
        out = [x for x in reranked if float(x.get("rerank_score", x.get("score", 0.0))) >= cutoff]
        return out if out else reranked[:fixed_top_k]
    return reranked[:fixed_top_k]


def _apply_laws_evidence_consistency_calibration(
    reranked: list[dict],
    top_n: int = 80,
) -> tuple[list[dict], dict]:
    if not reranked:
        return reranked, {"applied": 0, "rescored_laws": 0, "top_n": 0}

    n = max(0, min(int(top_n), len(reranked)))
    if n == 0:
        return reranked, {"applied": 0, "rescored_laws": 0, "top_n": 0}

    head = [dict(x) for x in reranked[:n]]
    tail = reranked[n:]

    rescored_laws = 0
    for item in head:
        source = item.get("source", "")
        base = float(item.get("rerank_score", item.get("score", 0.0)))
        is_laws = int(source == "laws_de")
        is_rule = int(item.get("admission_is_rule_hit", 0))
        is_norm = int(item.get("admission_is_normalization_consistent", 0))
        is_family = int(item.get("admission_is_family_hit", 0))
        is_issue = int(item.get("admission_is_issue_hit", 0))
        is_sparse_dense = int(item.get("admission_is_sparse_dense_supported", 0))
        is_dense_only_wo_support = int(item.get("admission_is_dense_only_without_other_support", 0))

        evidence_score = 0.0
        if is_laws:
            evidence_score += 0.08 * is_rule
            evidence_score += 0.04 * is_norm
            evidence_score += 0.03 * is_family
            evidence_score += 0.03 * is_issue
            evidence_score += 0.05 * is_sparse_dense
            evidence_score -= 0.04 * is_dense_only_wo_support
            rescored_laws += 1

        item["evidence_consistency_score"] = float(evidence_score)
        item["rerank_score_ec"] = float(base + evidence_score)

    head.sort(key=lambda x: float(x.get("rerank_score_ec", x.get("rerank_score", x.get("score", 0.0)))), reverse=True)
    out = head + tail
    return out, {"applied": 1, "rescored_laws": int(rescored_laws), "top_n": int(n)}


def _route_to_retrieval_quota(route: RouteDecision) -> tuple[int, int, int, int]:
    if route.primary_source == "laws":
        return 120, 0, 80, 0
    if route.primary_source == "court":
        return 0, 120, 0, 0
    return 100, 100, 60, 0


def _route_to_retrieval_quota_v1(
    route: RouteDecision,
    enable_court_mainline: bool,
    laws_route_laws_max: int,
    laws_route_court_max: int,
    court_route_court_max: int,
    court_route_laws_max: int,
    hybrid_route_laws_max: int,
    hybrid_route_court_max: int,
    min_court_candidates_for_hybrid: int,
    min_court_candidates_for_court_route: int,
    seed_floor_sparse: int = 20,
    seed_floor_dense: int = 20,
) -> tuple[int, int, int, int]:
    # seed_generation_repair_v1:
    # - soft preference + non-main source floor
    # - keep total budgets unchanged: sparse=120, dense=80
    sparse_budget = 120
    dense_budget = 80
    sparse_floor = max(0, min(int(seed_floor_sparse), sparse_budget // 2))
    dense_floor = max(0, min(int(seed_floor_dense), dense_budget // 2))

    if route.primary_source == "laws":
        laws_sparse = sparse_budget - sparse_floor
        court_sparse = sparse_floor
        laws_dense = dense_budget - dense_floor
        court_dense = dense_floor
        return laws_sparse, court_sparse, laws_dense, court_dense

    if route.primary_source == "court":
        laws_sparse = sparse_floor
        court_sparse = sparse_budget - sparse_floor
        laws_dense = dense_floor
        court_dense = dense_budget - dense_floor
        return laws_sparse, court_sparse, laws_dense, court_dense

    # mixed
    return 60, 60, 40, 40


def run_split(
    rows: list[dict],
    sparse: SparseRetriever,
    dense: DenseRetriever,
    doc_lookup: dict[str, dict],
    reranker: RerankerHandle,
    dynamic_mode: str,
    fixed_top_k: int,
    score_threshold: float,
    relative_threshold: float,
    enable_router: bool = True,
    router_version: str = "v1",
    enable_court_mainline: bool = False,
    laws_route_laws_max: int = 120,
    laws_route_court_max: int = 30,
    court_route_court_max: int = 200,
    court_route_laws_max: int = 30,
    hybrid_route_laws_max: int = 120,
    hybrid_route_court_max: int = 120,
    min_court_candidates_for_hybrid: int = 40,
    min_court_candidates_for_court_route: int = 80,
    enable_court_dense: bool = False,
    court_dense_max: int = 120,
    court_dense_weight: float = 1.0,
    court_dense_fusion_mode: str = "rrf",
    court_dense_fallback_reason: str = "",
    seed_floor_sparse: int = 20,
    seed_floor_dense: int = 20,
    enable_rule_exact: bool = True,
    rule_retriever: RuleCitationRetriever | None = None,
    rule_top_k_laws: int = 20,
    enable_laws_primary_german_expansion: bool = False,
    enable_law_family_constraints: bool = False,
    law_family_boost: float = 2.5,
    law_family_min_keep: int = 5,
    enable_issue_phrase_refinement: bool = False,
    issue_phrase_top_k: int = 24,
    issue_phrase_boost: float = 2.5,
    issue_phrase_max_groups: int = 4,
    issue_phrase_max_terms: int = 16,
    enable_laws_final_cut_calibration: bool = False,
    laws_final_fused_rescue_top_k: int = 1,
    enable_laws_rerank_input_shaping: bool = False,
    laws_rerank_shortlist_size: int = 320,
    laws_rerank_keep_fused_tail: int = 160,
    enable_laws_evidence_consistency_calibration: bool = False,
    laws_evidence_top_n: int = 80,
) -> tuple[dict[str, list[str]], list[dict], dict]:
    pred_map: dict[str, list[str]] = {}
    trace_rows: list[dict] = []
    route_counter = {"laws": 0, "court": 0, "hybrid": 0}

    for row in rows:
        qid = row["query_id"]
        query = row["query"]
        mv = preprocess_query(query)
        expanded = expand_query_from_multi_view(mv)
        source_packs = build_source_aware_query_packs(mv)
        likely_families = (
            likely_statute_families(query)
            if enable_law_family_constraints
            else []
        )
        family_terms = family_query_terms(likely_families)
        rule_laws_items = []
        if enable_rule_exact and rule_retriever is not None:
            rule_laws_items = rule_retriever.search(
                query=query,
                top_k_laws=max(0, int(rule_top_k_laws)),
                top_k_court=0,
            )
            rule_laws_items = boost_items_by_family(rule_laws_items, likely_families, law_family_boost)

        if enable_router:
            if router_version == "v1_1":
                route = route_query_v1_1(query, mv.get("query_number_patterns", []))
            else:
                route = route_query(query, mv.get("query_number_patterns", []))
        else:
            route = RouteDecision(
                primary_source="hybrid",
                secondary_source="none",
                route_confidence="low",
                case_signal_count=0,
                statute_signal_count=0,
                mixed_signal=False,
                case_score=0.0,
                statute_score=0.0,
                mixed_score=0.0,
                matched_case_patterns=[],
                matched_statute_patterns=[],
                route_decision_reason_v1_1="router_disabled",
            )
        route_counter[route.primary_source] += 1

        top_k_laws_sparse, top_k_court_sparse, top_k_laws_dense, top_k_court_dense = _route_to_retrieval_quota_v1(
            route=route,
            enable_court_mainline=enable_court_mainline,
            laws_route_laws_max=laws_route_laws_max,
            laws_route_court_max=laws_route_court_max,
            court_route_court_max=court_route_court_max,
            court_route_laws_max=court_route_laws_max,
            hybrid_route_laws_max=hybrid_route_laws_max,
            hybrid_route_court_max=hybrid_route_court_max,
            min_court_candidates_for_hybrid=min_court_candidates_for_hybrid,
            min_court_candidates_for_court_route=min_court_candidates_for_court_route,
            seed_floor_sparse=seed_floor_sparse,
            seed_floor_dense=seed_floor_dense,
        )

        laws_pack = source_packs.get("laws_query_pack", {})
        laws_pack_v2 = source_packs.get("laws_query_pack_v2", {})
        court_pack = source_packs.get("court_query_pack", {})
        if enable_law_family_constraints and likely_families:
            laws_pack = augment_laws_query_pack(laws_pack, likely_families)
            laws_pack_v2 = augment_laws_query_pack(laws_pack_v2, likely_families)
            expanded["bilingual_query_pack"] = augment_bilingual_pack(
                expanded.get("bilingual_query_pack", {}),
                likely_families,
            )
        sparse_items = sparse.search_route_aware(
            laws_query_pack=laws_pack,
            court_query_pack=court_pack,
            laws_top_k=top_k_laws_sparse,
            court_top_k=top_k_court_sparse,
            laws_query_pack_v2=laws_pack_v2,
            enable_laws_query_pack_v2=enable_laws_primary_german_expansion,
            laws_citation_weight=2.0,
            laws_title_weight=2.0,
            laws_text_weight=1.0,
            court_citation_weight=1.0,
            court_text_weight=1.0,
        )
        sparse_items = boost_items_by_family(sparse_items, likely_families, law_family_boost)
        if enable_law_family_constraints and likely_families:
            sparse_items = constrain_items_by_family(sparse_items, likely_families, min_keep=law_family_min_keep)

        issue_terms = []
        issue_groups = []
        issue_sparse_items = []
        is_nonexplicit_query = not RuleCitationRetriever.extract_patterns(query)
        if (
            is_nonexplicit_query
            and enable_issue_phrase_refinement
            and enable_law_family_constraints
            and likely_families
        ):
            issue_groups = issue_phrase_groups(query, likely_families, max_groups=issue_phrase_max_groups)
            issue_terms = issue_query_terms(
                query,
                likely_families,
                max_groups=issue_phrase_max_groups,
                max_terms=issue_phrase_max_terms,
            )
            if issue_terms:
                issue_pack = build_issue_laws_query_pack(
                    query,
                    likely_families,
                    max_groups=issue_phrase_max_groups,
                    max_terms=issue_phrase_max_terms,
                )
                issue_sparse_items = sparse.search_field_aware(
                    laws_query_pack=issue_pack,
                    court_query_pack={},
                    top_k_laws=max(0, int(issue_phrase_top_k)),
                    top_k_court=0,
                    laws_citation_weight=0.0,
                    laws_title_weight=2.0,
                    laws_text_weight=3.0,
                    court_citation_weight=0.0,
                    court_text_weight=0.0,
                )
                issue_sparse_items = boost_items_by_family(issue_sparse_items, likely_families, issue_phrase_boost)
                issue_sparse_items = filter_items_by_family(issue_sparse_items, likely_families)
                sparse_items = merge_retrieved_items(sparse_items, issue_sparse_items)

        sparse_laws_items = [x for x in sparse_items if x.source == "laws_de"]
        sparse_court_items = [x for x in sparse_items if x.source == "court_considerations"]

        if top_k_laws_dense > 0 or top_k_court_dense > 0:
            if enable_laws_primary_german_expansion:
                dense_items = dense.search_source_aware(
                    laws_query_pack=laws_pack_v2 or laws_pack,
                    court_query_pack=court_pack,
                    top_k_laws=top_k_laws_dense,
                    top_k_court=top_k_court_dense,
                )
            else:
                dense_items = dense.search_multi_view(
                    bilingual_query_pack=expanded.get("bilingual_query_pack", {}),
                    top_k_laws=top_k_laws_dense,
                    top_k_court=top_k_court_dense,
                )
        else:
            dense_items = []
        dense_items = boost_items_by_family(dense_items, likely_families, law_family_boost)
        if enable_law_family_constraints and likely_families:
            dense_items = constrain_items_by_family(dense_items, likely_families, min_keep=law_family_min_keep)

        dense_laws_items = [x for x in dense_items if x.source == "laws_de"]
        dense_court_items = [x for x in dense_items if x.source == "court_considerations"]
        local_court_dense_fallback = court_dense_fallback_reason or ""

        court_dense_triggered = bool(enable_court_dense and top_k_court_dense > 0 and court_dense_max > 0)
        if court_dense_triggered:
            try:
                dense_court_items = dense.search_court_multi_view(
                    bilingual_query_pack=expanded.get("bilingual_query_pack", {}),
                    top_k_court=court_dense_max,
                )
            except Exception as exc:
                dense_court_items = []
                local_court_dense_fallback = f"court_dense_runtime_error:{type(exc).__name__}"

        sparse_rank = [x.citation for x in sparse_items]
        dense_rank = [x.citation for x in dense_items]
        court_rank = []
        if enable_court_dense:
            court_rank = _fuse_court_branch_candidates(
                sparse_court_items=sparse_court_items,
                dense_court_items=dense_court_items,
                fusion_mode=court_dense_fusion_mode,
                dense_weight=court_dense_weight,
                top_n=max(220, court_dense_max),
            )

        if enable_court_dense:
            if len(court_rank) > 0:
                fused = rrf_fusion(
                    ranked_lists=[
                        [x.citation for x in sparse_laws_items],
                        [x.citation for x in dense_laws_items],
                        court_rank,
                    ],
                    k=60,
                    top_n=320,
                )
                fusion_mode_runtime = "court_dense_threeway_rrf"
            else:
                fused = rrf_fusion(
                    ranked_lists=[sparse_rank, dense_rank],
                    k=60,
                    top_n=320,
                )
                fusion_mode_runtime = "fallback_full_rrf_due_empty_court_rank"
        else:
            fused = rrf_fusion(
                ranked_lists=[sparse_rank, dense_rank],
                k=60,
                top_n=320,
            )
            fusion_mode_runtime = "full_rrf_no_court_dense"

        fused_scores = dict(dedup_keep_max([(c, s) for c, s in fused]))
        sparse_laws_set = {x.citation for x in sparse_laws_items}
        dense_laws_set = {x.citation for x in dense_laws_items}
        issue_laws_set = {x.citation for x in issue_sparse_items}
        rule_norm_hit_set = {
            normalize_citation(getattr(x, "citation", ""))
            for x in rule_laws_items
            if normalize_citation(getattr(x, "citation", ""))
        }
        if enable_laws_rerank_input_shaping:
            candidates, admission_stats = _build_rerank_candidates_with_laws_shaping(
                fused=fused,
                doc_lookup=doc_lookup,
                sparse_laws_items=sparse_laws_items,
                dense_laws_items=dense_laws_items,
                issue_sparse_items=issue_sparse_items,
                rule_laws_items=rule_laws_items,
                likely_families=likely_families,
                shortlist_size=laws_rerank_shortlist_size,
                keep_fused_tail=laws_rerank_keep_fused_tail,
            )
        else:
            candidates = []
            for citation, score in fused[:320]:
                doc = doc_lookup.get(citation, {})
                norm_c = normalize_citation(citation)
                is_rule_hit = int(bool(norm_c and norm_c in rule_norm_hit_set))
                is_norm_consistent = is_rule_hit
                is_family_hit = int(bool(likely_families) and _citation_family_consistent(citation, likely_families))
                is_issue_hit = int(citation in issue_laws_set)
                is_laws = int(doc.get("source", "") == "laws_de")
                is_sparse_hit = int(citation in sparse_laws_set)
                is_dense_hit = int(citation in dense_laws_set)
                is_sparse_dense_supported = int(is_sparse_hit and is_dense_hit)
                is_dense_only_without_other_support = int(
                    is_dense_hit
                    and (not is_sparse_hit)
                    and (not is_rule_hit)
                    and (not is_family_hit)
                    and (not is_issue_hit)
                )
                candidates.append(
                    {
                        "citation": citation,
                        "source": doc.get("source", ""),
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                        "fused_score": score,
                        "score": score,
                        "admission_priority_tier": 0,
                        "admission_is_rule_hit": is_rule_hit,
                        "admission_is_normalization_consistent": is_norm_consistent,
                        "admission_is_family_hit": is_family_hit,
                        "admission_is_issue_hit": is_issue_hit,
                        "admission_is_laws": is_laws,
                        "admission_is_sparse_hit": is_sparse_hit,
                        "admission_is_dense_hit": is_dense_hit,
                        "admission_is_sparse_dense_supported": is_sparse_dense_supported,
                        "admission_is_dense_only_without_other_support": is_dense_only_without_other_support,
                        "admission_fused_rank": len(candidates) + 1,
                        "admission_pool_score": float(score),
                    }
                )
            admission_stats = {
                "shortlist_size": len(candidates),
                "keep_fused_tail": len(candidates),
                "priority_budget": 0,
                "priority_selected": 0,
                "selected_rule_hit": 0,
                "selected_family_hit": 0,
                "selected_issue_hit": 0,
            }

        for item in candidates:
            c = item.get("citation", "")
            is_sparse_hit = int(c in sparse_laws_set)
            is_dense_hit = int(c in dense_laws_set)
            item["admission_is_sparse_hit"] = is_sparse_hit
            item["admission_is_dense_hit"] = is_dense_hit
            item["admission_is_sparse_dense_supported"] = int(is_sparse_hit and is_dense_hit)
            item["admission_is_dense_only_without_other_support"] = int(
                is_dense_hit
                and (not is_sparse_hit)
                and (not int(item.get("admission_is_rule_hit", 0)))
                and (not int(item.get("admission_is_family_hit", 0)))
                and (not int(item.get("admission_is_issue_hit", 0)))
            )

        reranked = reranker.rerank_fn.rerank(query=query, candidates=candidates, top_n=320)
        evidence_stats = {"applied": 0, "rescored_laws": 0, "top_n": 0}
        if enable_laws_evidence_consistency_calibration:
            reranked, evidence_stats = _apply_laws_evidence_consistency_calibration(
                reranked=reranked,
                top_n=laws_evidence_top_n,
            )
        final_items = apply_dynamic_cut(
            reranked=reranked,
            mode=dynamic_mode,
            fixed_top_k=fixed_top_k,
            score_threshold=score_threshold,
            relative_threshold=relative_threshold,
        )
        laws_final_rescue_citations = []
        if (
            enable_laws_final_cut_calibration
            and is_nonexplicit_query
            and enable_law_family_constraints
            and likely_families
            and laws_final_fused_rescue_top_k > 0
        ):
            already_final = {x.citation for x in rule_laws_items} | {x["citation"] for x in final_items}
            for citation, _score in fused[:320]:
                if citation in already_final:
                    continue
                doc = doc_lookup.get(citation, {})
                if doc.get("source", "") != "laws_de":
                    continue
                if not _citation_family_consistent(citation, likely_families):
                    continue
                laws_final_rescue_citations.append(citation)
                already_final.add(citation)
                if len(laws_final_rescue_citations) >= int(laws_final_fused_rescue_top_k):
                    break

        final_citations = []
        seen_final = set()
        for citation in (
            [x.citation for x in rule_laws_items]
            + [x["citation"] for x in final_items]
            + laws_final_rescue_citations
        ):
            if citation in seen_final:
                continue
            final_citations.append(citation)
            seen_final.add(citation)
        pred_map[qid] = final_citations
        laws_forwarded = sum(1 for c in candidates if c.get("source") == "laws_de")
        court_forwarded = sum(1 for c in candidates if c.get("source") == "court_considerations")
        sparse_court_forwarded = len({x.citation for x in sparse_court_items})
        dense_court_forwarded = len({x.citation for x in dense_court_items})
        court_fused_forwarded = len(set(court_rank)) if enable_court_dense else sparse_court_forwarded
        note = "balanced_or_none"
        if court_forwarded > laws_forwarded:
            note = "court_forwarded_dominant"
        elif court_forwarded > 0 and laws_forwarded > 0:
            note = "both_sources_forwarded"
        elif court_forwarded == 0:
            note = "no_court_candidates_forwarded"
        dense_note = "court_dense_disabled"
        if enable_court_dense:
            if local_court_dense_fallback:
                dense_note = f"court_dense_fallback:{local_court_dense_fallback}"
            elif dense_court_forwarded > 0:
                dense_note = "court_dense_active"
            else:
                dense_note = "court_dense_no_hits"
        trace_rows.append(
            {
                "query_id": qid,
                "primary_source": route.primary_source,
                "secondary_source": route.secondary_source,
                "route_confidence": route.route_confidence,
                "case_signal_count": route.case_signal_count,
                "statute_signal_count": route.statute_signal_count,
                "case_score": route.case_score,
                "statute_score": route.statute_score,
                "mixed_score": route.mixed_score,
                "matched_case_patterns": ";".join(route.matched_case_patterns or []),
                "matched_statute_patterns": ";".join(route.matched_statute_patterns or []),
                "route_decision_reason_v1_1": route.route_decision_reason_v1_1,
                "router_version": router_version if enable_router else "no_router",
                "laws_candidates_forwarded": laws_forwarded,
                "court_candidates_forwarded": court_forwarded,
                "court_sparse_candidates_forwarded": sparse_court_forwarded,
                "court_dense_candidates_forwarded": dense_court_forwarded,
                "court_fused_candidates_forwarded": court_fused_forwarded,
                "gold_citation": row.get("gold_citations", ""),
                "court_mainline_effect_note": note,
                "court_dense_effect_note": dense_note,
                "rule_laws_exact_count": len({x.citation for x in rule_laws_items}),
                "rule_laws_exact_citations": ";".join([x.citation for x in rule_laws_items]),
                "laws_primary_german_expansion_enabled": int(enable_laws_primary_german_expansion),
                "law_family_constraints_enabled": int(enable_law_family_constraints),
                "likely_statute_family": ";".join(likely_families),
                "law_family_query_terms": ";".join(family_terms),
                "law_family_min_keep": int(law_family_min_keep),
                "issue_phrase_refinement_enabled": int(
                    is_nonexplicit_query
                    and enable_issue_phrase_refinement
                    and enable_law_family_constraints
                    and bool(likely_families)
                ),
                "issue_phrase_groups": ";".join(issue_groups),
                "issue_phrase_query_terms": ";".join(issue_terms),
                "issue_phrase_sparse_count": len({x.citation for x in issue_sparse_items}),
                "issue_phrase_sparse_citations": ";".join([x.citation for x in issue_sparse_items]),
                "laws_evidence_consistency_calibration_enabled": int(enable_laws_evidence_consistency_calibration),
                "laws_evidence_top_n": int(laws_evidence_top_n),
                "laws_evidence_rescored_laws": int(evidence_stats.get("rescored_laws", 0)),
                "laws_final_cut_calibration_enabled": int(
                    enable_laws_final_cut_calibration
                    and is_nonexplicit_query
                    and enable_law_family_constraints
                    and bool(likely_families)
                ),
                "laws_final_fused_rescue_top_k": int(laws_final_fused_rescue_top_k),
                "laws_final_rescue_count": len(laws_final_rescue_citations),
                "laws_final_rescue_citations": ";".join(laws_final_rescue_citations),
                "route_label": route.primary_source,
                "quota_laws_sparse": top_k_laws_sparse,
                "quota_court_sparse": top_k_court_sparse,
                "quota_laws_dense": top_k_laws_dense,
                "quota_court_dense": top_k_court_dense,
                "sparse_laws_count": len({x.citation for x in sparse_laws_items}),
                "sparse_court_count": len({x.citation for x in sparse_court_items}),
                "dense_laws_count": len({x.citation for x in dense_laws_items}),
                "dense_court_count": len({x.citation for x in dense_court_items}),
                "court_dense_triggered": int(court_dense_triggered),
                "court_rank_count": len(set(court_rank)),
                "fusion_mode": fusion_mode_runtime,
                "fused_total_count": len(fused),
                "fused_top200_count": min(200, len(fused)),
                "rerank_input_count": len(candidates),
                "rerank_input_shaping_enabled": int(enable_laws_rerank_input_shaping),
                "rerank_input_shortlist_size": int(laws_rerank_shortlist_size),
                "rerank_input_keep_fused_tail": int(laws_rerank_keep_fused_tail),
                "rerank_input_priority_budget": int(admission_stats.get("priority_budget", 0)),
                "rerank_input_priority_selected": int(admission_stats.get("priority_selected", 0)),
                "rerank_input_priority_rule_hit_count": int(admission_stats.get("selected_rule_hit", 0)),
                "rerank_input_priority_family_hit_count": int(admission_stats.get("selected_family_hit", 0)),
                "rerank_input_priority_issue_hit_count": int(admission_stats.get("selected_issue_hit", 0)),
                "rerank_input_rule_hit_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_rule_hit", 0)) == 1]
                ),
                "rerank_input_normalization_consistent_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_normalization_consistent", 0)) == 1]
                ),
                "rerank_input_family_consistent_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_family_hit", 0)) == 1]
                ),
                "rerank_input_issue_hit_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_issue_hit", 0)) == 1]
                ),
                "rerank_input_sparse_hit_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_sparse_hit", 0)) == 1]
                ),
                "rerank_input_dense_hit_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_dense_hit", 0)) == 1]
                ),
                "rerank_input_sparse_dense_supported_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_sparse_dense_supported", 0)) == 1]
                ),
                "rerank_input_dense_only_without_other_support_citations": ";".join(
                    [x.get("citation", "") for x in candidates if int(x.get("admission_is_laws", 0)) == 1 and int(x.get("admission_is_dense_only_without_other_support", 0)) == 1]
                ),
                "fused_top200": ";".join([c for c, _ in fused[:200]]),
                "fused_top320": ";".join([c for c, _ in fused[:320]]),
                "rerank_input_citations": ";".join([x["citation"] for x in candidates]),
                "reranked_top320": ";".join([x["citation"] for x in reranked[:320]]),
                "final_cut_predictions": ";".join([x["citation"] for x in final_items]),
                "final_predictions": ";".join(pred_map[qid]),
                "dynamic_mode": dynamic_mode,
                "reranker_name": reranker.name,
                "top1_fused_score": next(iter(fused_scores.values()), 0.0),
            }
        )
    return pred_map, trace_rows, route_counter


def run_from_baseline_v1_args(args: argparse.Namespace) -> None:
    argv = [
        "--dynamic-mode",
        "relative_threshold",
        "--relative-threshold",
        "0.85",
        "--fixed-top-k",
        str(getattr(args, "top_n", 12)),
        "--router-version",
        "v1",
    ]
    main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="silver_baseline_v0: router + federated retrieval + strong reranker")
    parser.add_argument("--dynamic-mode", choices=["fixed_top_k", "score_threshold", "relative_threshold"], default="relative_threshold")
    parser.add_argument("--fixed-top-k", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--relative-threshold", type=float, default=0.85)
    parser.add_argument("--enable-router", type=parse_bool_flag, default=True)
    parser.add_argument("--router-version", choices=["v1", "v1_1"], default="v1")
    parser.add_argument("--enable-court-mainline", type=parse_bool_flag, default=False)
    parser.add_argument("--laws-route-laws-max", type=int, default=120)
    parser.add_argument("--laws-route-court-max", type=int, default=30)
    parser.add_argument("--court-route-court-max", type=int, default=200)
    parser.add_argument("--court-route-laws-max", type=int, default=30)
    parser.add_argument("--hybrid-route-laws-max", type=int, default=120)
    parser.add_argument("--hybrid-route-court-max", type=int, default=120)
    parser.add_argument("--min-court-candidates-for-hybrid", type=int, default=40)
    parser.add_argument("--min-court-candidates-for-court-route", type=int, default=80)
    parser.add_argument("--enable-court-dense", type=parse_bool_flag, default=False)
    parser.add_argument("--court-dense-max", type=int, default=120)
    parser.add_argument("--court-dense-weight", type=float, default=1.0)
    parser.add_argument("--court-dense-fusion-mode", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--court-dense-model-name", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--enable-rule-exact", type=parse_bool_flag, default=True)
    parser.add_argument("--rule-top-k-laws", type=int, default=20)
    parser.add_argument("--enable-laws-primary-german-expansion", type=parse_bool_flag, default=False)
    parser.add_argument("--enable-law-family-constraints", type=parse_bool_flag, default=False)
    parser.add_argument("--law-family-boost", type=float, default=2.5)
    parser.add_argument("--law-family-min-keep", type=int, default=5)
    parser.add_argument("--enable-issue-phrase-refinement", type=parse_bool_flag, default=False)
    parser.add_argument("--issue-phrase-top-k", type=int, default=24)
    parser.add_argument("--issue-phrase-boost", type=float, default=2.5)
    parser.add_argument("--issue-phrase-max-groups", type=int, default=4)
    parser.add_argument("--issue-phrase-max-terms", type=int, default=16)
    parser.add_argument("--enable-laws-final-cut-calibration", type=parse_bool_flag, default=False)
    parser.add_argument("--laws-final-fused-rescue-top-k", type=int, default=1)
    parser.add_argument("--enable-laws-rerank-input-shaping", type=parse_bool_flag, default=False)
    parser.add_argument("--laws-rerank-shortlist-size", type=int, default=320)
    parser.add_argument("--laws-rerank-keep-fused-tail", type=int, default=160)
    parser.add_argument("--enable-laws-evidence-consistency-calibration", type=parse_bool_flag, default=False)
    parser.add_argument("--laws-evidence-top-n", type=int, default=80)
    parser.add_argument("--prefer-strong-reranker", type=parse_bool_flag, default=True)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--sparse-max-court", type=int, default=300000)
    parser.add_argument("--dense-max-laws", type=int, default=80000)
    parser.add_argument("--dense-max-court", type=int, default=120000)
    parser.add_argument("--seed-floor-sparse", type=int, default=20)
    parser.add_argument("--seed-floor-dense", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "silver_baseline_v0")
    args = parser.parse_args(argv)

    # Align P1 fine-tuned MiniLM evaluation with notebook's laws-full reindex scope.
    p1_model_hint = "laws_minilm_p1" in str(args.court_dense_model_name).replace("\\", "/")
    dense_max_laws_effective = args.dense_max_laws
    if p1_model_hint and int(args.dense_max_laws) < int(args.sparse_max_laws):
        dense_max_laws_effective = int(args.sparse_max_laws)

    val_rows = load_query_split("val")
    test_rows = load_query_split("test")

    sparse = SparseRetriever(text_max_chars=900)
    sparse_stats = sparse.build(
        max_laws_rows=args.sparse_max_laws,
        max_court_rows=args.sparse_max_court,
        enable_field_aware=True,
    )
    dense = DenseRetriever(
        model_name=args.court_dense_model_name,
        use_sbert=not args.dense_disable_sbert,
        text_max_chars=500,
        svd_dim=256,
    )
    dense_stats = dense.build(
        max_laws_rows=dense_max_laws_effective,
        max_court_rows=args.dense_max_court,
        enable_field_aware=True,
    )
    doc_lookup = build_doc_lookup(sparse, dense)
    reranker = build_reranker(prefer_strong=args.prefer_strong_reranker)
    rule_retriever = None
    rule_stats = {}
    if args.enable_rule_exact:
        rule_retriever = RuleCitationRetriever()
        rule_stats = rule_retriever.build(max_laws_rows=args.sparse_max_laws, max_court_rows=0)
    court_dense_enabled_runtime = bool(args.enable_court_dense)
    court_dense_fallback_reason = ""
    if court_dense_enabled_runtime and dense.backend != "sbert":
        court_dense_enabled_runtime = False
        court_dense_fallback_reason = f"court_dense_model_unavailable_backend={dense.backend}"

    val_pred, val_trace, val_routes = run_split(
        rows=val_rows,
        sparse=sparse,
        dense=dense,
        doc_lookup=doc_lookup,
        reranker=reranker,
        dynamic_mode=args.dynamic_mode,
        fixed_top_k=args.fixed_top_k,
        score_threshold=args.score_threshold,
        relative_threshold=args.relative_threshold,
        enable_router=args.enable_router,
        router_version=args.router_version,
        enable_court_mainline=args.enable_court_mainline,
        laws_route_laws_max=args.laws_route_laws_max,
        laws_route_court_max=args.laws_route_court_max,
        court_route_court_max=args.court_route_court_max,
        court_route_laws_max=args.court_route_laws_max,
        hybrid_route_laws_max=args.hybrid_route_laws_max,
        hybrid_route_court_max=args.hybrid_route_court_max,
        min_court_candidates_for_hybrid=args.min_court_candidates_for_hybrid,
        min_court_candidates_for_court_route=args.min_court_candidates_for_court_route,
        enable_court_dense=court_dense_enabled_runtime,
        court_dense_max=args.court_dense_max,
        court_dense_weight=args.court_dense_weight,
        court_dense_fusion_mode=args.court_dense_fusion_mode,
        court_dense_fallback_reason=court_dense_fallback_reason,
        seed_floor_sparse=args.seed_floor_sparse,
        seed_floor_dense=args.seed_floor_dense,
        enable_rule_exact=args.enable_rule_exact,
        rule_retriever=rule_retriever,
        rule_top_k_laws=args.rule_top_k_laws,
        enable_laws_primary_german_expansion=args.enable_laws_primary_german_expansion,
        enable_law_family_constraints=args.enable_law_family_constraints,
        law_family_boost=args.law_family_boost,
        law_family_min_keep=args.law_family_min_keep,
        enable_issue_phrase_refinement=args.enable_issue_phrase_refinement,
        issue_phrase_top_k=args.issue_phrase_top_k,
        issue_phrase_boost=args.issue_phrase_boost,
        issue_phrase_max_groups=args.issue_phrase_max_groups,
        issue_phrase_max_terms=args.issue_phrase_max_terms,
        enable_laws_final_cut_calibration=args.enable_laws_final_cut_calibration,
        laws_final_fused_rescue_top_k=args.laws_final_fused_rescue_top_k,
        enable_laws_rerank_input_shaping=args.enable_laws_rerank_input_shaping,
        laws_rerank_shortlist_size=args.laws_rerank_shortlist_size,
        laws_rerank_keep_fused_tail=args.laws_rerank_keep_fused_tail,
        enable_laws_evidence_consistency_calibration=args.enable_laws_evidence_consistency_calibration,
        laws_evidence_top_n=args.laws_evidence_top_n,
    )
    test_pred, test_trace, _ = run_split(
        rows=test_rows,
        sparse=sparse,
        dense=dense,
        doc_lookup=doc_lookup,
        reranker=reranker,
        dynamic_mode=args.dynamic_mode,
        fixed_top_k=args.fixed_top_k,
        score_threshold=args.score_threshold,
        relative_threshold=args.relative_threshold,
        enable_router=args.enable_router,
        router_version=args.router_version,
        enable_court_mainline=args.enable_court_mainline,
        laws_route_laws_max=args.laws_route_laws_max,
        laws_route_court_max=args.laws_route_court_max,
        court_route_court_max=args.court_route_court_max,
        court_route_laws_max=args.court_route_laws_max,
        hybrid_route_laws_max=args.hybrid_route_laws_max,
        hybrid_route_court_max=args.hybrid_route_court_max,
        min_court_candidates_for_hybrid=args.min_court_candidates_for_hybrid,
        min_court_candidates_for_court_route=args.min_court_candidates_for_court_route,
        enable_court_dense=court_dense_enabled_runtime,
        court_dense_max=args.court_dense_max,
        court_dense_weight=args.court_dense_weight,
        court_dense_fusion_mode=args.court_dense_fusion_mode,
        court_dense_fallback_reason=court_dense_fallback_reason,
        seed_floor_sparse=args.seed_floor_sparse,
        seed_floor_dense=args.seed_floor_dense,
        enable_rule_exact=args.enable_rule_exact,
        rule_retriever=rule_retriever,
        rule_top_k_laws=args.rule_top_k_laws,
        enable_laws_primary_german_expansion=args.enable_laws_primary_german_expansion,
        enable_law_family_constraints=args.enable_law_family_constraints,
        law_family_boost=args.law_family_boost,
        law_family_min_keep=args.law_family_min_keep,
        enable_issue_phrase_refinement=args.enable_issue_phrase_refinement,
        issue_phrase_top_k=args.issue_phrase_top_k,
        issue_phrase_boost=args.issue_phrase_boost,
        issue_phrase_max_groups=args.issue_phrase_max_groups,
        issue_phrase_max_terms=args.issue_phrase_max_terms,
        enable_laws_final_cut_calibration=args.enable_laws_final_cut_calibration,
        laws_final_fused_rescue_top_k=args.laws_final_fused_rescue_top_k,
        enable_laws_rerank_input_shaping=args.enable_laws_rerank_input_shaping,
        laws_rerank_shortlist_size=args.laws_rerank_shortlist_size,
        laws_rerank_keep_fused_tail=args.laws_rerank_keep_fused_tail,
        enable_laws_evidence_consistency_calibration=args.enable_laws_evidence_consistency_calibration,
        laws_evidence_top_n=args.laws_evidence_top_n,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_pred_file = args.out_dir / "val_predictions_silver_baseline_v0.csv"
    test_pred_file = args.out_dir / "test_predictions_silver_baseline_v0.csv"
    submission_file = ROOT / "submissions" / "submission_silver_baseline_v0.csv"
    write_prediction_csv(val_pred_file, val_pred)
    write_prediction_csv(test_pred_file, test_pred)
    write_prediction_csv(submission_file, test_pred)

    (args.out_dir / "val_trace_silver_baseline_v0.json").write_text(
        json.dumps(val_trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "test_trace_silver_baseline_v0.json").write_text(
        json.dumps(test_trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    def write_trace_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_trace_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    val_seed_trace_csv = args.out_dir / "val_seed_trace_silver_baseline_v0.csv"
    test_seed_trace_csv = args.out_dir / "test_seed_trace_silver_baseline_v0.csv"
    val_seed_trace_jsonl = args.out_dir / "val_seed_trace_silver_baseline_v0.jsonl"
    test_seed_trace_jsonl = args.out_dir / "test_seed_trace_silver_baseline_v0.jsonl"
    write_trace_csv(val_seed_trace_csv, val_trace)
    write_trace_csv(test_seed_trace_csv, test_trace)
    write_trace_jsonl(val_seed_trace_jsonl, val_trace)
    write_trace_jsonl(test_seed_trace_jsonl, test_trace)

    meta = {
        "dynamic_mode": args.dynamic_mode,
        "fixed_top_k": args.fixed_top_k,
        "score_threshold": args.score_threshold,
        "relative_threshold": args.relative_threshold,
        "router_enabled": args.enable_router,
        "router_version": args.router_version,
        "enable_court_mainline": args.enable_court_mainline,
        "laws_route_laws_max": args.laws_route_laws_max,
        "laws_route_court_max": args.laws_route_court_max,
        "court_route_court_max": args.court_route_court_max,
        "court_route_laws_max": args.court_route_laws_max,
        "hybrid_route_laws_max": args.hybrid_route_laws_max,
        "hybrid_route_court_max": args.hybrid_route_court_max,
        "min_court_candidates_for_hybrid": args.min_court_candidates_for_hybrid,
        "min_court_candidates_for_court_route": args.min_court_candidates_for_court_route,
        "seed_floor_sparse": args.seed_floor_sparse,
        "seed_floor_dense": args.seed_floor_dense,
        "enable_court_dense_requested": args.enable_court_dense,
        "enable_court_dense_effective": court_dense_enabled_runtime,
        "court_dense_max": args.court_dense_max,
        "court_dense_weight": args.court_dense_weight,
        "court_dense_fusion_mode": args.court_dense_fusion_mode,
        "court_dense_model_name": args.court_dense_model_name,
        "court_dense_fallback_reason": court_dense_fallback_reason,
        "enable_rule_exact": args.enable_rule_exact,
        "rule_top_k_laws": args.rule_top_k_laws,
        "rule_stats": rule_stats,
        "enable_laws_primary_german_expansion": args.enable_laws_primary_german_expansion,
        "enable_law_family_constraints": args.enable_law_family_constraints,
        "law_family_boost": args.law_family_boost,
        "law_family_min_keep": args.law_family_min_keep,
        "enable_issue_phrase_refinement": args.enable_issue_phrase_refinement,
        "issue_phrase_top_k": args.issue_phrase_top_k,
        "issue_phrase_boost": args.issue_phrase_boost,
        "issue_phrase_max_groups": args.issue_phrase_max_groups,
        "issue_phrase_max_terms": args.issue_phrase_max_terms,
        "enable_laws_final_cut_calibration": args.enable_laws_final_cut_calibration,
        "laws_final_fused_rescue_top_k": args.laws_final_fused_rescue_top_k,
        "enable_laws_rerank_input_shaping": args.enable_laws_rerank_input_shaping,
        "laws_rerank_shortlist_size": args.laws_rerank_shortlist_size,
        "laws_rerank_keep_fused_tail": args.laws_rerank_keep_fused_tail,
        "enable_laws_evidence_consistency_calibration": args.enable_laws_evidence_consistency_calibration,
        "laws_evidence_top_n": args.laws_evidence_top_n,
        "reranker_name": reranker.name,
        "reranker_fallback_reason": reranker.fallback_reason,
        "route_counts_val": val_routes,
        "sparse_stats": sparse_stats,
        "dense_stats": dense_stats,
        "dense_max_laws_configured": args.dense_max_laws,
        "dense_max_laws_effective": dense_max_laws_effective,
        "p1_model_scope_alignment_applied": int(p1_model_hint and dense_max_laws_effective != args.dense_max_laws),
        "artifacts": {
            "val_predictions": str(val_pred_file),
            "test_predictions": str(test_pred_file),
            "submission": str(submission_file),
            "val_seed_trace_csv": str(val_seed_trace_csv),
            "test_seed_trace_csv": str(test_seed_trace_csv),
            "val_seed_trace_jsonl": str(val_seed_trace_jsonl),
            "test_seed_trace_jsonl": str(test_seed_trace_jsonl),
        },
    }
    (args.out_dir / "run_meta_silver_baseline_v0.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

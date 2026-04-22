from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation, split_citations
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from query_expansion import build_source_aware_query_packs, expand_query_from_multi_view
from query_preprocess import preprocess_query
from retrieval_dense import DenseRetriever
from retrieval_sparse import SparseRetriever
from source_router import RouteDecision, route_query, route_query_v1_1

from run_silver_baseline_v0 import (
    _fuse_court_branch_candidates,
    _route_to_retrieval_quota_v1,
    apply_dynamic_cut,
    build_doc_lookup,
    build_reranker,
    dedup_keep_max,
)
from fusion import rrf_fusion


def _norm_list(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for x in values:
        nx = normalize_citation(x)
        if nx and nx not in seen:
            out.append(nx)
            seen.add(nx)
    return out


def _best_rank(ranked: list[str], gold_set: set[str]) -> int:
    for i, c in enumerate(ranked, start=1):
        if normalize_citation(c) in gold_set:
            return i
    return 0


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _classify_gold_source(gold_set: set[str], laws_set: set[str], court_set: set[str]) -> str:
    has_laws = any(g in laws_set for g in gold_set)
    has_court = any(g in court_set for g in gold_set)
    if has_laws and has_court:
        return "mixed"
    if has_laws:
        return "laws_only"
    if has_court:
        return "court_only"
    return "unknown"


def _load_norm_citation_set(path: Path) -> set[str]:
    out = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            c = normalize_citation(row.get("citation", ""))
            if c:
                out.add(c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze seed effectiveness for seed_generation_repair_v1.")
    parser.add_argument("--v1-out-dir", type=Path, default=ROOT / "outputs" / "silver_baseline_v0_seed_repair_v1")
    parser.add_argument("--review-out-dir", type=Path, default=ROOT / "artifacts" / "seed_generation_repair_v1_review")
    args = parser.parse_args()

    out_dir = args.review_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.v1_out_dir / "run_meta_silver_baseline_v0.json"
    val_pred_path = args.v1_out_dir / "val_predictions_silver_baseline_v0.csv"
    val_seed_trace_csv = args.v1_out_dir / "val_seed_trace_silver_baseline_v0.csv"
    val_seed_trace_jsonl = args.v1_out_dir / "val_seed_trace_silver_baseline_v0.jsonl"
    base_eval_csv = ROOT / "artifacts" / "silver_baseline_v0_real" / "silver_eval_results_real.csv"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    val_rows = load_query_split("val")
    val_map = {r["query_id"]: r for r in val_rows}

    # step1 field check
    trace_fields = []
    with val_seed_trace_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        trace_fields = reader.fieldnames or []
    required_fields = [
        "route_label",
        "quota_laws_sparse",
        "quota_court_sparse",
        "quota_laws_dense",
        "quota_court_dense",
        "sparse_laws_count",
        "sparse_court_count",
        "dense_laws_count",
        "dense_court_count",
        "court_dense_triggered",
        "court_rank_count",
        "fusion_mode",
        "fused_total_count",
        "fused_top200_count",
        "rerank_input_count",
    ]
    missing_required = [x for x in required_fields if x not in trace_fields]
    step1_md = [
        "# step1_artifact_check",
        "",
        f"- run_meta: `{meta_path}`",
        f"- val_seed_trace_csv: `{val_seed_trace_csv}`",
        f"- val_seed_trace_jsonl: `{val_seed_trace_jsonl}`",
        f"- 如有本地评估结果: `{'none'}`",
        f"- 必需字段缺失: `{missing_required if missing_required else 'none'}`",
    ]
    (out_dir / "step1_artifact_check.md").write_text("\n".join(step1_md) + "\n", encoding="utf-8")

    # Rebuild retrievers with v1 runtime config
    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(
        max_laws_rows=int(meta["sparse_stats"]["laws_docs"]),
        max_court_rows=int(meta["sparse_stats"]["court_docs"]),
        enable_field_aware=True,
    )
    dense = DenseRetriever(
        model_name=meta.get("court_dense_model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        use_sbert=True,
        text_max_chars=int(meta["dense_stats"]["text_max_chars"]),
        svd_dim=256,
    )
    dense.build(
        max_laws_rows=int(meta["dense_stats"]["laws_docs"]),
        max_court_rows=int(meta["dense_stats"]["court_docs"]),
        enable_field_aware=True,
    )
    doc_lookup = build_doc_lookup(sparse, dense)
    reranker = build_reranker(prefer_strong=True)

    # Load corpus source sets
    laws_set = _load_norm_citation_set(ROOT / "data_raw" / "competition_data" / "laws_de.csv")
    court_set = _load_norm_citation_set(ROOT / "data_raw" / "competition_data" / "court_considerations.csv")

    joined_rows: list[dict] = []
    pred_map_v1: dict[str, list[str]] = {}
    with val_pred_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            pred_map_v1[r["query_id"]] = _norm_list(split_citations(r.get("predicted_citations", "")))

    for row in val_rows:
        qid = row["query_id"]
        query = row["query"]
        gold_list = _norm_list(split_citations(row.get("gold_citations", "")))
        gold_set = set(gold_list)

        mv = preprocess_query(query)
        expanded = expand_query_from_multi_view(mv)
        source_packs = build_source_aware_query_packs(mv)

        if meta.get("router_enabled", True):
            if meta.get("router_version", "v1") == "v1_1":
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

        top_k_laws_sparse, top_k_court_sparse, top_k_laws_dense, top_k_court_dense = _route_to_retrieval_quota_v1(
            route=route,
            enable_court_mainline=bool(meta.get("enable_court_mainline", False)),
            laws_route_laws_max=int(meta.get("laws_route_laws_max", 120)),
            laws_route_court_max=int(meta.get("laws_route_court_max", 30)),
            court_route_court_max=int(meta.get("court_route_court_max", 200)),
            court_route_laws_max=int(meta.get("court_route_laws_max", 30)),
            hybrid_route_laws_max=int(meta.get("hybrid_route_laws_max", 120)),
            hybrid_route_court_max=int(meta.get("hybrid_route_court_max", 120)),
            min_court_candidates_for_hybrid=int(meta.get("min_court_candidates_for_hybrid", 40)),
            min_court_candidates_for_court_route=int(meta.get("min_court_candidates_for_court_route", 80)),
            seed_floor_sparse=int(meta.get("seed_floor_sparse", 20)),
            seed_floor_dense=int(meta.get("seed_floor_dense", 20)),
        )

        laws_pack = source_packs.get("laws_query_pack", {})
        court_pack = source_packs.get("court_query_pack", {})
        sparse_items = sparse.search_route_aware(
            laws_query_pack=laws_pack,
            court_query_pack=court_pack,
            laws_top_k=top_k_laws_sparse,
            court_top_k=top_k_court_sparse,
            laws_query_pack_v2=source_packs.get("laws_query_pack_v2", {}),
            enable_laws_query_pack_v2=True,
            laws_citation_weight=2.0,
            laws_title_weight=2.0,
            laws_text_weight=1.0,
            court_citation_weight=1.0,
            court_text_weight=1.0,
        )
        sparse_laws_items = [x for x in sparse_items if x.source == "laws_de"]
        sparse_court_items = [x for x in sparse_items if x.source == "court_considerations"]

        dense_items = dense.search_multi_view(
            bilingual_query_pack=expanded.get("bilingual_query_pack", {}),
            top_k_laws=top_k_laws_dense,
            top_k_court=top_k_court_dense,
        )
        dense_laws_items = [x for x in dense_items if x.source == "laws_de"]
        dense_court_items = [x for x in dense_items if x.source == "court_considerations"]

        enable_court_dense = bool(meta.get("enable_court_dense_effective", False))
        court_dense_max = int(meta.get("court_dense_max", 120))
        court_dense_triggered = bool(enable_court_dense and top_k_court_dense > 0 and court_dense_max > 0)
        if court_dense_triggered:
            dense_court_items = dense.search_court_multi_view(
                bilingual_query_pack=expanded.get("bilingual_query_pack", {}),
                top_k_court=court_dense_max,
            )

        sparse_rank = [x.citation for x in sparse_items]
        dense_rank = [x.citation for x in dense_items]
        court_rank: list[str] = []
        if enable_court_dense:
            court_rank = _fuse_court_branch_candidates(
                sparse_court_items=sparse_court_items,
                dense_court_items=dense_court_items,
                fusion_mode=str(meta.get("court_dense_fusion_mode", "rrf")),
                dense_weight=float(meta.get("court_dense_weight", 1.0)),
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
                fusion_mode = "court_dense_threeway_rrf"
            else:
                fused = rrf_fusion(ranked_lists=[sparse_rank, dense_rank], k=60, top_n=320)
                fusion_mode = "fallback_full_rrf_due_empty_court_rank"
        else:
            fused = rrf_fusion(ranked_lists=[sparse_rank, dense_rank], k=60, top_n=320)
            fusion_mode = "full_rrf_no_court_dense"

        candidates = []
        for citation, score in fused[:320]:
            doc = doc_lookup.get(citation, {})
            candidates.append(
                {
                    "citation": citation,
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "fused_score": score,
                    "score": score,
                }
            )

        reranked = reranker.rerank_fn.rerank(query=query, candidates=candidates, top_n=320)
        final_items = apply_dynamic_cut(
            reranked=reranked,
            mode=str(meta.get("dynamic_mode", "relative_threshold")),
            fixed_top_k=int(meta.get("fixed_top_k", 12)),
            score_threshold=float(meta.get("score_threshold", 0.15)),
            relative_threshold=float(meta.get("relative_threshold", 0.85)),
        )
        final_predictions = [x["citation"] for x in final_items]

        sparse_laws_ranked = [x.citation for x in sorted(sparse_laws_items, key=lambda z: float(z.score), reverse=True)]
        sparse_court_ranked = [x.citation for x in sorted(sparse_court_items, key=lambda z: float(z.score), reverse=True)]
        dense_laws_ranked = [x.citation for x in sorted(dense_laws_items, key=lambda z: float(z.score), reverse=True)]
        dense_court_ranked = [x.citation for x in sorted(dense_court_items, key=lambda z: float(z.score), reverse=True)]
        sparse_ranked = [x.citation for x in sorted(sparse_items, key=lambda z: float(z.score), reverse=True)]
        dense_ranked = [x.citation for x in sorted(dense_items, key=lambda z: float(z.score), reverse=True)]
        fused_top200 = [c for c, _ in fused[:200]]

        sparse_laws_set = {normalize_citation(x) for x in sparse_laws_ranked if normalize_citation(x)}
        sparse_court_set = {normalize_citation(x) for x in sparse_court_ranked if normalize_citation(x)}
        dense_laws_set = {normalize_citation(x) for x in dense_laws_ranked if normalize_citation(x)}
        dense_court_set = {normalize_citation(x) for x in dense_court_ranked if normalize_citation(x)}
        court_rank_set = {normalize_citation(x) for x in court_rank if normalize_citation(x)}
        fused_top200_set = {normalize_citation(x) for x in fused_top200 if normalize_citation(x)}

        gold_in_sparse_laws = int(bool(gold_set & sparse_laws_set))
        gold_in_sparse_court = int(bool(gold_set & sparse_court_set))
        gold_in_dense_laws = int(bool(gold_set & dense_laws_set))
        gold_in_dense_court = int(bool(gold_set & dense_court_set))
        gold_in_court_rank = int(bool(gold_set & court_rank_set))
        gold_in_fused_top200 = int(bool(gold_set & fused_top200_set))

        gold_best_rank_sparse = _best_rank(sparse_ranked, gold_set)
        gold_best_rank_dense = _best_rank(dense_ranked, gold_set)
        gold_best_rank_court_rank = _best_rank(court_rank, gold_set)
        gold_best_rank_fused = _best_rank(fused_top200, gold_set)

        joined_rows.append(
            {
                "query_id": qid,
                "query_text": query,
                "route_label": route.primary_source,
                "gold_source": _classify_gold_source(gold_set, laws_set, court_set),
                "gold_citation": gold_list[0] if gold_list else "",
                "gold_citations": ";".join(gold_list),
                "gold_in_sparse_laws": gold_in_sparse_laws,
                "gold_in_sparse_court": gold_in_sparse_court,
                "gold_in_dense_laws": gold_in_dense_laws,
                "gold_in_dense_court": gold_in_dense_court,
                "gold_in_court_rank": gold_in_court_rank,
                "gold_in_fused_top200": gold_in_fused_top200,
                "gold_best_rank_sparse": gold_best_rank_sparse,
                "gold_best_rank_dense": gold_best_rank_dense,
                "gold_best_rank_court_rank": gold_best_rank_court_rank,
                "gold_best_rank_fused": gold_best_rank_fused,
                "quota_laws_sparse": top_k_laws_sparse,
                "quota_court_sparse": top_k_court_sparse,
                "quota_laws_dense": top_k_laws_dense,
                "quota_court_dense": top_k_court_dense,
                "sparse_laws_count": len(set(sparse_laws_ranked)),
                "sparse_court_count": len(set(sparse_court_ranked)),
                "dense_laws_count": len(set(dense_laws_ranked)),
                "dense_court_count": len(set(dense_court_ranked)),
                "court_dense_triggered": int(court_dense_triggered),
                "court_rank_count": len(set(court_rank)),
                "fusion_mode": fusion_mode,
                "fused_total_count": len(fused),
                "fused_top200_count": min(200, len(fused)),
                "rerank_input_count": len(candidates),
                "strict_f1": 0.0,  # to be filled later
                "corpus_f1": 0.0,  # to be filled later
            }
        )

    # Fill strict/corpus f1 per qid
    strict_summary_v1, strict_rows_v1 = evaluate_predictions(val_rows, pred_map_v1, citation_lookup=None, mode="strict")
    corpus_lookup = {c: c for c in (laws_set | court_set)}
    corpus_summary_v1, corpus_rows_v1 = evaluate_predictions(val_rows, pred_map_v1, citation_lookup=corpus_lookup, mode="corpus_aware")
    strict_map = {r["query_id"]: float(r["f1"]) for r in strict_rows_v1}
    corpus_map = {r["query_id"]: float(r["f1"]) for r in corpus_rows_v1}

    for r in joined_rows:
        r["strict_f1"] = strict_map.get(r["query_id"], 0.0)
        r["corpus_f1"] = corpus_map.get(r["query_id"], 0.0)

    # Miss reason tagging with explicit rules
    miss_counter = Counter()
    for r in joined_rows:
        if r["gold_in_fused_top200"] == 1:
            tag = "hit_in_fused_top200"
        else:
            if r["fused_top200_count"] < 200:
                tag = "candidate_pool_too_shallow"
            elif r["court_dense_triggered"] == 0 and r["quota_court_dense"] > 0:
                tag = "dense_not_triggered"
            elif (r["quota_laws_sparse"] == 0 and r["quota_court_sparse"] == 0 and r["quota_laws_dense"] == 0 and r["quota_court_dense"] == 0):
                tag = "blocked_by_route_quota"
            elif r["gold_in_sparse_laws"] == 0 and r["gold_in_sparse_court"] == 0 and r["gold_in_dense_laws"] == 0 and r["gold_in_dense_court"] == 0:
                # not retrieved by both sparse and dense; if query has explicit legal anchors, suggest anchor rule
                if "Art." in r["query_text"] or "BGE" in r["query_text"] or "_" in r["query_text"]:
                    tag = "needs_anchor_rule"
                else:
                    tag = "not_retrieved_sparse"
            elif (r["gold_in_sparse_laws"] or r["gold_in_sparse_court"]) and (r["gold_in_dense_laws"] == 0 and r["gold_in_dense_court"] == 0):
                tag = "not_retrieved_dense"
            elif (r["gold_in_dense_laws"] or r["gold_in_dense_court"]) and (r["gold_in_sparse_laws"] == 0 and r["gold_in_sparse_court"] == 0):
                tag = "not_retrieved_sparse"
            elif r["fusion_mode"] == "fallback_full_rrf_due_empty_court_rank":
                tag = "lost_in_merge"
            else:
                tag = "eval_alignment_suspected"
        r["miss_reason_tag"] = tag
        miss_counter[tag] += 1

    # Save joined file
    _write_csv(out_dir / "val_seed_trace_with_gold_v1.csv", joined_rows)

    # Table A: overall comparison
    with base_eval_csv.open("r", encoding="utf-8-sig", newline="") as f:
        base_rows = list(csv.DictReader(f))
        base = base_rows[0] if base_rows else {}
    baseline_recall200 = float(base.get("fusion_final_Recall@200", 0.0))
    baseline_strict = float(base.get("strict_macro_f1", 0.0))
    baseline_corpus = float(base.get("corpus_aware_macro_f1", 0.0))
    v1_recall200 = _avg([float(r["gold_in_fused_top200"]) for r in joined_rows])
    v1_strict = float(strict_summary_v1.get("macro_f1", 0.0))
    v1_corpus = float(corpus_summary_v1.get("macro_f1", 0.0))
    table_a = [
        {"run": "baseline_strong_config_rerun", "Recall@200": round(baseline_recall200, 6), "strict_f1": round(baseline_strict, 6), "corpus_f1": round(baseline_corpus, 6)},
        {"run": "seed_generation_repair_v1", "Recall@200": round(v1_recall200, 6), "strict_f1": round(v1_strict, 6), "corpus_f1": round(v1_corpus, 6)},
        {"run": "delta_v1_minus_baseline", "Recall@200": round(v1_recall200 - baseline_recall200, 6), "strict_f1": round(v1_strict - baseline_strict, 6), "corpus_f1": round(v1_corpus - baseline_corpus, 6)},
    ]
    _write_csv(out_dir / "overall_metrics_comparison.csv", table_a)
    md_a = ["# overall_metrics_comparison", "", "| run | Recall@200 | strict_f1 | corpus_f1 |", "|---|---:|---:|---:|"]
    for r in table_a:
        md_a.append(f"| {r['run']} | {r['Recall@200']:.6f} | {r['strict_f1']:.6f} | {r['corpus_f1']:.6f} |")
    (out_dir / "overall_metrics_comparison.md").write_text("\n".join(md_a) + "\n", encoding="utf-8")

    # Table B: by_gold_source
    group_b = defaultdict(list)
    for r in joined_rows:
        group_b[r["gold_source"]].append(r)
    table_b = []
    for gs in ["laws_only", "court_only", "mixed", "unknown"]:
        rows = group_b.get(gs, [])
        table_b.append(
            {
                "gold_source": gs,
                "sample_count": len(rows),
                "Recall@200": round(_avg([float(x["gold_in_fused_top200"]) for x in rows]), 6),
                "strict_f1": round(_avg([float(x["strict_f1"]) for x in rows]), 6),
                "corpus_f1": round(_avg([float(x["corpus_f1"]) for x in rows]), 6),
                "avg_sparse_laws_count": round(_avg([float(x["sparse_laws_count"]) for x in rows]), 6),
                "avg_sparse_court_count": round(_avg([float(x["sparse_court_count"]) for x in rows]), 6),
                "avg_dense_laws_count": round(_avg([float(x["dense_laws_count"]) for x in rows]), 6),
                "avg_dense_court_count": round(_avg([float(x["dense_court_count"]) for x in rows]), 6),
                "court_dense_trigger_rate": round(_avg([float(x["court_dense_triggered"]) for x in rows]), 6),
            }
        )
    _write_csv(out_dir / "by_gold_source.csv", table_b)
    md_b = [
        "# by_gold_source",
        "",
        "| gold_source | sample_count | Recall@200 | strict_f1 | corpus_f1 | avg_sparse_laws_count | avg_sparse_court_count | avg_dense_laws_count | avg_dense_court_count | court_dense_trigger_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_b:
        md_b.append(
            f"| {r['gold_source']} | {r['sample_count']} | {r['Recall@200']:.6f} | {r['strict_f1']:.6f} | {r['corpus_f1']:.6f} | "
            f"{r['avg_sparse_laws_count']:.6f} | {r['avg_sparse_court_count']:.6f} | {r['avg_dense_laws_count']:.6f} | {r['avg_dense_court_count']:.6f} | {r['court_dense_trigger_rate']:.6f} |"
        )
    (out_dir / "by_gold_source.md").write_text("\n".join(md_b) + "\n", encoding="utf-8")

    # Table C: by_route_label_x_gold_source
    group_c = defaultdict(list)
    for r in joined_rows:
        group_c[(r["route_label"], r["gold_source"])].append(r)
    route_labels = sorted({r["route_label"] for r in joined_rows})
    table_c = []
    for rl in route_labels:
        for gs in ["laws_only", "court_only", "mixed", "unknown"]:
            rows = group_c.get((rl, gs), [])
            fm = Counter([x["fusion_mode"] for x in rows])
            fm_dist = ";".join(f"{k}:{v}" for k, v in sorted(fm.items(), key=lambda z: z[0]))
            table_c.append(
                {
                    "route_label": rl,
                    "gold_source": gs,
                    "sample_count": len(rows),
                    "Recall@200": round(_avg([float(x["gold_in_fused_top200"]) for x in rows]), 6),
                    "strict_f1": round(_avg([float(x["strict_f1"]) for x in rows]), 6),
                    "corpus_f1": round(_avg([float(x["corpus_f1"]) for x in rows]), 6),
                    "gold_in_sparse_court_rate": round(_avg([float(x["gold_in_sparse_court"]) for x in rows]), 6),
                    "gold_in_dense_court_rate": round(_avg([float(x["gold_in_dense_court"]) for x in rows]), 6),
                    "gold_in_fused_top200_rate": round(_avg([float(x["gold_in_fused_top200"]) for x in rows]), 6),
                    "avg_court_rank_count": round(_avg([float(x["court_rank_count"]) for x in rows]), 6),
                    "court_dense_trigger_rate": round(_avg([float(x["court_dense_triggered"]) for x in rows]), 6),
                    "fusion_mode_distribution": fm_dist,
                }
            )
    _write_csv(out_dir / "by_route_label_x_gold_source.csv", table_c)
    md_c = [
        "# by_route_label_x_gold_source",
        "",
        "| route_label | gold_source | sample_count | Recall@200 | strict_f1 | corpus_f1 | gold_in_sparse_court_rate | gold_in_dense_court_rate | gold_in_fused_top200_rate | avg_court_rank_count | court_dense_trigger_rate | fusion_mode_distribution |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in table_c:
        md_c.append(
            f"| {r['route_label']} | {r['gold_source']} | {r['sample_count']} | {r['Recall@200']:.6f} | {r['strict_f1']:.6f} | {r['corpus_f1']:.6f} | "
            f"{r['gold_in_sparse_court_rate']:.6f} | {r['gold_in_dense_court_rate']:.6f} | {r['gold_in_fused_top200_rate']:.6f} | "
            f"{r['avg_court_rank_count']:.6f} | {r['court_dense_trigger_rate']:.6f} | {r['fusion_mode_distribution']} |"
        )
    (out_dir / "by_route_label_x_gold_source.md").write_text("\n".join(md_c) + "\n", encoding="utf-8")

    # miss reason stats / rules
    miss_rows = []
    for tag, cnt in sorted(miss_counter.items(), key=lambda z: (-z[1], z[0])):
        miss_rows.append({"miss_reason_tag": tag, "count": cnt, "ratio": round(cnt / len(joined_rows), 6)})
    _write_csv(out_dir / "miss_reason_stats.csv", miss_rows)
    md_miss = [
        "# miss_reason_rules_and_stats",
        "",
        "## 判定规则",
        "- `blocked_by_route_quota`: 4 个 quota 全 0 且 gold 未入 fused_top200",
        "- `dense_not_triggered`: `quota_court_dense>0` 但 `court_dense_triggered=0` 且 gold 未入 fused_top200",
        "- `lost_in_merge`: `fusion_mode=fallback_full_rrf_due_empty_court_rank` 且 gold 未入 fused_top200",
        "- `not_retrieved_sparse`: sparse 侧对 gold 无命中且 gold 未入 fused_top200",
        "- `not_retrieved_dense`: dense 侧对 gold 无命中且 gold 未入 fused_top200",
        "- `needs_anchor_rule`: sparse+dense 对 gold 都无命中，且 query 含明显 citation anchor（Art/BGE/case id）",
        "- `candidate_pool_too_shallow`: `fused_top200_count<200` 且 gold 未入 fused_top200",
        "- `eval_alignment_suspected`: 以上不满足但仍 miss",
        "- `hit_in_fused_top200`: gold 入 fused_top200（非失败标签）",
        "",
        "## 标签占比",
        "| miss_reason_tag | count | ratio |",
        "|---|---:|---:|",
    ]
    for r in miss_rows:
        md_miss.append(f"| {r['miss_reason_tag']} | {r['count']} | {r['ratio']:.6f} |")
    (out_dir / "miss_reason_rules_and_stats.md").write_text("\n".join(md_miss) + "\n", encoding="utf-8")

    # patch plan after v1
    v1_effective = v1_recall200 > baseline_recall200
    major_bottleneck = "anchor/rule seeds 缺失" if miss_counter.get("needs_anchor_rule", 0) >= max(miss_counter.get("not_retrieved_sparse", 0), miss_counter.get("not_retrieved_dense", 0)) else "court retrieval quality / eval alignment"
    next_major = "v2 rule / anchor seeds" if "anchor" in major_bottleneck else "进一步修 court retrieval quality"
    if (v1_effective and abs(v1_strict) <= 1e-12 and abs(v1_corpus) <= 1e-12):
        next_major = "检查 eval/id alignment"

    patch_lines = [
        "# patch_plan_after_v1",
        "",
        "## 1. v1 是否已经证明方向有效",
        f"- Recall@200: baseline={baseline_recall200:.6f}, v1={v1_recall200:.6f}, delta={v1_recall200-baseline_recall200:+.6f}",
        f"- strict_f1: baseline={baseline_strict:.6f}, v1={v1_strict:.6f}, delta={v1_strict-baseline_strict:+.6f}",
        f"- corpus_f1: baseline={baseline_corpus:.6f}, v1={v1_corpus:.6f}, delta={v1_corpus-baseline_corpus:+.6f}",
        f"- 结论：`{'v1 已证明有效' if v1_effective else 'v1 尚未证明有效'}`",
        "",
        "## 2. 当前最大剩余瓶颈",
        f"- {major_bottleneck}",
        "",
        "## 3. 下一步优先方向",
        f"- `{next_major}`",
        "",
        "## 4. 前 3 个 patch（仅 seed_generation_repair）",
        "1. Patch-1: v2 rule / anchor seeds（小配额并入 seed union）",
        "- 为什么现在做：`needs_anchor_rule` 占比高，说明显式引用未被稳定入池。",
        "- 预期收益：优先提升 `gold_in_fused_top200_rate` 与 Recall@200。",
        "- 风险：噪声候选上升。",
        "- 涉及文件：`scripts/run_silver_baseline_v0.py`, `src/retrieval_rules.py`。",
        "- 是否改主逻辑：否（只加 seed 分支）。",
        "2. Patch-2: court retrieval quality seed 修复（不改 router 判定）",
        "- 为什么现在做：court 侧 gold 的融合前命中率仍低。",
        "- 预期收益：提升 `gold_in_dense_court_rate` / `gold_in_sparse_court_rate`。",
        "- 风险：推理时长增加。",
        "- 涉及文件：`src/retrieval_sparse.py`, `src/retrieval_dense.py`, `scripts/run_silver_baseline_v0.py`。",
        "- 是否改主逻辑：否。",
        "3. Patch-3: eval/id alignment 诊断脚本固化",
        "- 为什么现在做：避免把口径问题误判成检索问题。",
        "- 预期收益：缩小误诊范围，提升迭代效率。",
        "- 风险：额外分析维护成本。",
        "- 涉及文件：`scripts/analyze_seed_effectiveness_v1.py`, `src/legal_ir/evaluation.py`, `src/legal_ir/normalization.py`。",
        "- 是否改主逻辑：否。",
        "",
        "## 5. 简明结论",
        f"- v1 到底有没有让 gold 更容易进入 fused_top200？`{'有' if v1_effective else '没有'}`",
        f"- 如果没有，下一步最应该做什么？`{next_major}`",
    ]
    (out_dir / "patch_plan_after_v1.md").write_text("\n".join(patch_lines) + "\n", encoding="utf-8")

    summary = {
        "v1_recall200": round(v1_recall200, 6),
        "baseline_recall200": round(baseline_recall200, 6),
        "delta_recall200": round(v1_recall200 - baseline_recall200, 6),
        "v1_strict_f1": round(v1_strict, 6),
        "v1_corpus_f1": round(v1_corpus, 6),
        "v1_effective": v1_effective,
        "next_major": next_major,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


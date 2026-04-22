from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from eval_local import evaluate_and_dump
from fusion import rrf_fusion, weighted_score_fusion
from query_expansion import (
    build_bilingual_retrieval_views,
    build_source_aware_query_packs,
    expand_query_from_multi_view,
)
from query_preprocess import build_retrieval_queries, preprocess_query
from rerank import NoOpReranker, TokenOverlapReranker
from retrieval_dense import DenseRetriever
from retrieval_rules import RuleCitationRetriever
from retrieval_sparse import SparseRetriever
from legal_ir.data_loader import load_query_split


def dedup_keep_max(score_items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(pred_map.keys()):
            writer.writerow([qid, ";".join(pred_map[qid])])


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


def build_branch_metadata(
    sparse_items,
    dense_items,
    rule_items,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    # 分支命中集合（branch_hits）：用于计算 branch_support_count。
    branch_hits: dict[str, set[str]] = {}
    sparse_laws_rank: dict[str, int] = {}
    sparse_all_seen = set()
    dense_seen = set()
    rule_seen = set()

    rank_counter = 0
    for item in sparse_items:
        citation = item.citation
        if citation not in sparse_all_seen:
            rank_counter += 1
            sparse_all_seen.add(citation)
        branch_hits.setdefault(citation, set())
        if item.source == "laws_de":
            branch_hits[citation].add("sparse_laws")
            if citation not in sparse_laws_rank:
                sparse_laws_rank[citation] = rank_counter
        elif item.source == "court_considerations":
            branch_hits[citation].add("sparse_court")
        else:
            branch_hits[citation].add("sparse_unknown")

    for item in dense_items:
        citation = item.citation
        if citation in dense_seen:
            continue
        dense_seen.add(citation)
        branch_hits.setdefault(citation, set())
        if item.source == "laws_de":
            branch_hits[citation].add("dense_laws")
        elif item.source == "court_considerations":
            branch_hits[citation].add("dense_court")
        else:
            branch_hits[citation].add("dense_unknown")

    for item in rule_items:
        citation = item.citation
        if citation in rule_seen:
            continue
        rule_seen.add(citation)
        branch_hits.setdefault(citation, set()).add("rule_branch")
    return branch_hits, sparse_laws_rank


def run_split(
    rows: list[dict],
    sparse: SparseRetriever,
    dense: DenseRetriever,
    doc_lookup: dict[str, dict],
    fusion_mode: str,
    top_n: int,
    reranker_name: str,
    enable_query_preprocess: bool = False,
    enable_query_expansion: bool = False,
    enable_source_aware_query_expansion: bool = False,
    enable_laws_query_pack_v2: bool = False,
    enable_rule_recall: bool = False,
    enable_field_aware_retrieval: bool = False,
    source_aware_fusion: bool = False,
    laws_weight: float = 1.0,
    court_weight: float = 1.0,
    laws_citation_weight: float = 1.0,
    laws_title_weight: float = 1.0,
    laws_text_weight: float = 1.0,
    court_citation_weight: float = 1.0,
    court_text_weight: float = 1.0,
    enable_branch_aware_fusion: bool = False,
    sparse_laws_branch_bonus: float = 0.0,
    sparse_laws_single_branch_bonus: float = 0.0,
    branch_aware_rank_cutoff: int = 200,
    branch_aware_fusion_mode: str = "sparse_laws_bonus",
    rule_retriever: RuleCitationRetriever | None = None,
) -> dict[str, list[str]]:
    if reranker_name == "token_overlap":
        reranker = TokenOverlapReranker()
    else:
        reranker = NoOpReranker()

    pred_map: dict[str, list[str]] = {}
    for row in rows:
        query = row["query"]
        sparse_query = query  # backward-compatible default
        dense_query = query  # backward-compatible default
        query_keywords: list[str] = []
        expanded_query_de = ""
        bilingual_query_pack: dict = {}
        laws_query_pack: dict = {}
        laws_query_pack_v2: dict = {}
        court_query_pack: dict = {}
        if enable_query_preprocess:
            mv = preprocess_query(query)
            retrieval_views = build_retrieval_queries(mv)
            sparse_query = retrieval_views["sparse_query"]
            dense_query = retrieval_views["dense_query"]
            query_keywords = mv.get("query_keywords", [])
            if enable_query_expansion:
                expanded = expand_query_from_multi_view(mv)
                expanded_query_de = expanded.get("expanded_query_de", "")
                bilingual_query_pack = expanded.get("bilingual_query_pack", {})
            if enable_source_aware_query_expansion:
                source_expanded = build_source_aware_query_packs(mv)
                laws_query_pack = source_expanded.get("laws_query_pack", {})
                laws_query_pack_v2 = source_expanded.get("laws_query_pack_v2", {})
                court_query_pack = source_expanded.get("court_query_pack", {})
        elif enable_query_expansion:
            mv = preprocess_query(query)
            query_keywords = mv.get("query_keywords", [])
            expanded = build_bilingual_retrieval_views(mv)
            expanded_query_de = expanded.get("expanded_query_de", "")
            bilingual_query_pack = expanded.get("bilingual_query_pack", {})
            if enable_source_aware_query_expansion:
                source_expanded = build_source_aware_query_packs(mv)
                laws_query_pack = source_expanded.get("laws_query_pack", {})
                laws_query_pack_v2 = source_expanded.get("laws_query_pack_v2", {})
                court_query_pack = source_expanded.get("court_query_pack", {})

        if enable_field_aware_retrieval:
            if not laws_query_pack or not court_query_pack:
                mv = preprocess_query(query)
                source_expanded = build_source_aware_query_packs(mv)
                laws_query_pack = source_expanded.get("laws_query_pack", {})
                laws_query_pack_v2 = source_expanded.get("laws_query_pack_v2", {})
                court_query_pack = source_expanded.get("court_query_pack", {})
            sparse_items = sparse.search_field_aware(
                laws_query_pack=laws_query_pack,
                court_query_pack=court_query_pack,
                laws_query_pack_v2=laws_query_pack_v2,
                enable_laws_query_pack_v2=enable_laws_query_pack_v2,
                top_k_laws=60,
                top_k_court=40,
                laws_citation_weight=laws_citation_weight,
                laws_title_weight=laws_title_weight,
                laws_text_weight=laws_text_weight,
                court_citation_weight=court_citation_weight,
                court_text_weight=court_text_weight,
            )
        elif enable_query_preprocess or enable_query_expansion:
            sparse_items = sparse.search_multi_view(
                query_original=query,
                query_keywords=query_keywords,
                expanded_query_de=expanded_query_de,
                top_k_laws=60,
                top_k_court=40,
            )
        else:
            sparse_items = sparse.search(query=sparse_query, top_k_laws=60, top_k_court=40)

        if enable_field_aware_retrieval:
            if not laws_query_pack or not court_query_pack:
                mv = preprocess_query(query)
                source_expanded = build_source_aware_query_packs(mv)
                laws_query_pack = source_expanded.get("laws_query_pack", {})
                court_query_pack = source_expanded.get("court_query_pack", {})
            dense_items = dense.search_source_aware(
                laws_query_pack=laws_query_pack,
                court_query_pack=court_query_pack,
                top_k_laws=40,
                top_k_court=30,
            )
        elif enable_query_expansion and bilingual_query_pack:
            dense_items = dense.search_multi_view(
                bilingual_query_pack=bilingual_query_pack,
                top_k_laws=40,
                top_k_court=30,
            )
        else:
            dense_items = dense.search(query=dense_query, top_k_laws=40, top_k_court=30)

        rule_items = []
        if enable_rule_recall and rule_retriever is not None:
            rule_items = rule_retriever.search(query=query, top_k_laws=40, top_k_court=40)

        sparse_rank = [x.citation for x in sparse_items]
        dense_rank = [x.citation for x in dense_items]
        rule_rank = [x.citation for x in rule_items]
        sparse_scores = dedup_keep_max([(x.citation, x.score) for x in sparse_items])
        dense_scores = dedup_keep_max([(x.citation, x.score) for x in dense_items])
        rule_scores = dedup_keep_max([(x.citation, x.score) for x in rule_items])
        branch_hits, sparse_laws_rank = build_branch_metadata(
            sparse_items=sparse_items,
            dense_items=dense_items,
            rule_items=rule_items,
        )

        if fusion_mode == "weighted":
            score_lists = [sparse_scores, dense_scores]
            weights = [0.6, 0.4]
            if enable_rule_recall:
                score_lists.append(rule_scores)
                weights.append(0.35)
            fused = weighted_score_fusion(
                score_lists=score_lists,
                weights=weights,
                top_n=max(top_n * 3, 40),
                citation_to_source={k: v.get("source", "") for k, v in doc_lookup.items()},
                source_aware_fusion=source_aware_fusion,
                laws_weight=laws_weight,
                court_weight=court_weight,
                branch_hits=branch_hits,
                sparse_laws_rank=sparse_laws_rank,
                enable_branch_aware_fusion=enable_branch_aware_fusion,
                branch_aware_fusion_mode=branch_aware_fusion_mode,
                sparse_laws_branch_bonus=sparse_laws_branch_bonus,
                sparse_laws_single_branch_bonus=sparse_laws_single_branch_bonus,
                branch_aware_rank_cutoff=branch_aware_rank_cutoff,
            )
        else:
            ranked_lists = [sparse_rank, dense_rank]
            if enable_rule_recall:
                ranked_lists.append(rule_rank)
            fused = rrf_fusion(
                ranked_lists=ranked_lists,
                k=60,
                top_n=max(top_n * 3, 40),
                citation_to_source={k: v.get("source", "") for k, v in doc_lookup.items()},
                source_aware_fusion=source_aware_fusion,
                laws_weight=laws_weight,
                court_weight=court_weight,
                branch_hits=branch_hits,
                sparse_laws_rank=sparse_laws_rank,
                enable_branch_aware_fusion=enable_branch_aware_fusion,
                branch_aware_fusion_mode=branch_aware_fusion_mode,
                sparse_laws_branch_bonus=sparse_laws_branch_bonus,
                sparse_laws_single_branch_bonus=sparse_laws_single_branch_bonus,
                branch_aware_rank_cutoff=branch_aware_rank_cutoff,
            )

        candidates = []
        for citation, score in fused:
            doc = doc_lookup.get(citation, {})
            candidates.append(
                {
                    "citation": citation,
                    "source": doc.get("source", ""),
                    "text": doc.get("text", ""),
                    "fused_score": score,
                    "score": score,
                }
            )
        reranked = reranker.rerank(query=query, candidates=candidates, top_n=top_n)
        pred_map[row["query_id"]] = [x["citation"] for x in reranked]
    return pred_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline v1: sparse + dense + fusion (+optional rerank)")
    parser.add_argument(
        "--pipeline-mode",
        choices=["baseline_v1", "silver_baseline_v0"],
        default="baseline_v1",
        help="兼容入口：默认 baseline_v1；可切到 silver_baseline_v0 最小骨架",
    )
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--reranker", choices=["none", "token_overlap"], default="token_overlap")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--sparse-max-court", type=int, default=300000)
    parser.add_argument("--dense-max-laws", type=int, default=80000)
    parser.add_argument("--dense-max-court", type=int, default=120000)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--enable-query-preprocess", action="store_true")
    parser.add_argument("--enable-query-expansion", type=parse_bool_flag, default=False)
    parser.add_argument("--enable-source-aware-query-expansion", type=parse_bool_flag, default=False)
    parser.add_argument("--enable-laws-query-pack-v2", type=parse_bool_flag, default=False)
    parser.add_argument("--enable-rule-recall", type=parse_bool_flag, default=False)
    parser.add_argument("--enable-field-aware-retrieval", type=parse_bool_flag, default=False)
    parser.add_argument("--source-aware-fusion", type=parse_bool_flag, default=False)
    parser.add_argument("--laws-weight", type=float, default=1.0)
    parser.add_argument("--court-weight", type=float, default=1.0)
    parser.add_argument("--laws-citation-weight", type=float, default=1.0)
    parser.add_argument("--laws-title-weight", type=float, default=1.0)
    parser.add_argument("--laws-text-weight", type=float, default=1.0)
    parser.add_argument("--court-citation-weight", type=float, default=1.0)
    parser.add_argument("--court-text-weight", type=float, default=1.0)
    parser.add_argument("--enable-branch-aware-fusion", type=parse_bool_flag, default=False)
    parser.add_argument("--sparse-laws-branch-bonus", type=float, default=0.0)
    parser.add_argument("--sparse-laws-single-branch-bonus", type=float, default=0.0)
    parser.add_argument("--branch-aware-rank-cutoff", type=int, default=200)
    parser.add_argument(
        "--branch-aware-fusion-mode",
        choices=["sparse_laws_bonus", "sparse_laws_tail_rescue"],
        default="sparse_laws_bonus",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "baseline_v1")
    args = parser.parse_args()

    if args.pipeline_mode == "silver_baseline_v0":
        from run_silver_baseline_v0 import run_from_baseline_v1_args

        run_from_baseline_v1_args(args)
        return

    val_rows = load_query_split("val")
    test_rows = load_query_split("test")

    sparse = SparseRetriever(text_max_chars=900)
    sparse_stats = sparse.build(
        max_laws_rows=args.sparse_max_laws,
        max_court_rows=args.sparse_max_court,
        enable_field_aware=args.enable_field_aware_retrieval,
    )

    dense = DenseRetriever(
        use_sbert=not args.dense_disable_sbert,
        text_max_chars=500,
        svd_dim=256,
    )
    dense_stats = dense.build(
        max_laws_rows=args.dense_max_laws,
        max_court_rows=args.dense_max_court,
        enable_field_aware=args.enable_field_aware_retrieval,
    )
    rule_retriever = None
    rule_stats = {}
    if args.enable_rule_recall:
        rule_retriever = RuleCitationRetriever()
        rule_stats = rule_retriever.build(
            max_laws_rows=args.sparse_max_laws,
            max_court_rows=args.sparse_max_court,
        )

    doc_lookup = build_doc_lookup(sparse, dense)
    val_pred = run_split(
        rows=val_rows,
        sparse=sparse,
        dense=dense,
        doc_lookup=doc_lookup,
        fusion_mode=args.fusion,
        top_n=args.top_n,
        reranker_name=args.reranker,
        enable_query_preprocess=args.enable_query_preprocess,
        enable_query_expansion=args.enable_query_expansion,
        enable_source_aware_query_expansion=args.enable_source_aware_query_expansion,
        enable_laws_query_pack_v2=args.enable_laws_query_pack_v2,
        enable_rule_recall=args.enable_rule_recall,
        enable_field_aware_retrieval=args.enable_field_aware_retrieval,
        source_aware_fusion=args.source_aware_fusion,
        laws_weight=args.laws_weight,
        court_weight=args.court_weight,
        laws_citation_weight=args.laws_citation_weight,
        laws_title_weight=args.laws_title_weight,
        laws_text_weight=args.laws_text_weight,
        court_citation_weight=args.court_citation_weight,
        court_text_weight=args.court_text_weight,
        enable_branch_aware_fusion=args.enable_branch_aware_fusion,
        sparse_laws_branch_bonus=args.sparse_laws_branch_bonus,
        sparse_laws_single_branch_bonus=args.sparse_laws_single_branch_bonus,
        branch_aware_rank_cutoff=args.branch_aware_rank_cutoff,
        branch_aware_fusion_mode=args.branch_aware_fusion_mode,
        rule_retriever=rule_retriever,
    )
    test_pred = run_split(
        rows=test_rows,
        sparse=sparse,
        dense=dense,
        doc_lookup=doc_lookup,
        fusion_mode=args.fusion,
        top_n=args.top_n,
        reranker_name=args.reranker,
        enable_query_preprocess=args.enable_query_preprocess,
        enable_query_expansion=args.enable_query_expansion,
        enable_source_aware_query_expansion=args.enable_source_aware_query_expansion,
        enable_laws_query_pack_v2=args.enable_laws_query_pack_v2,
        enable_rule_recall=args.enable_rule_recall,
        enable_field_aware_retrieval=args.enable_field_aware_retrieval,
        source_aware_fusion=args.source_aware_fusion,
        laws_weight=args.laws_weight,
        court_weight=args.court_weight,
        laws_citation_weight=args.laws_citation_weight,
        laws_title_weight=args.laws_title_weight,
        laws_text_weight=args.laws_text_weight,
        court_citation_weight=args.court_citation_weight,
        court_text_weight=args.court_text_weight,
        enable_branch_aware_fusion=args.enable_branch_aware_fusion,
        sparse_laws_branch_bonus=args.sparse_laws_branch_bonus,
        sparse_laws_single_branch_bonus=args.sparse_laws_single_branch_bonus,
        branch_aware_rank_cutoff=args.branch_aware_rank_cutoff,
        branch_aware_fusion_mode=args.branch_aware_fusion_mode,
        rule_retriever=rule_retriever,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_pred_file = args.out_dir / "val_predictions_baseline_v1.csv"
    test_pred_file = args.out_dir / "test_predictions_baseline_v1.csv"
    submission_file = ROOT / "submissions" / "submission_baseline_v1.csv"
    write_prediction_csv(val_pred_file, val_pred)
    write_prediction_csv(test_pred_file, test_pred)
    write_prediction_csv(submission_file, test_pred)

    eval_dir = ROOT / "artifacts" / "baseline_v1_eval"
    lookup_file = ROOT / "artifacts" / "phase0" / "citation_lookup.csv"
    eval_report = evaluate_and_dump(
        split="val",
        pred_file=val_pred_file,
        out_dir=eval_dir,
        lookup_file=lookup_file if lookup_file.exists() else None,
    )

    meta = {
        "config": {
            "fusion": args.fusion,
            "reranker": args.reranker,
            "top_n": args.top_n,
            "sparse_max_laws": args.sparse_max_laws,
            "sparse_max_court": args.sparse_max_court,
            "dense_max_laws": args.dense_max_laws,
            "dense_max_court": args.dense_max_court,
            "dense_disable_sbert": args.dense_disable_sbert,
            "enable_query_preprocess": args.enable_query_preprocess,
            "enable_query_expansion": args.enable_query_expansion,
            "enable_source_aware_query_expansion": args.enable_source_aware_query_expansion,
            "enable_laws_query_pack_v2": args.enable_laws_query_pack_v2,
            "enable_rule_recall": args.enable_rule_recall,
            "enable_field_aware_retrieval": args.enable_field_aware_retrieval,
            "source_aware_fusion": args.source_aware_fusion,
            "laws_weight": args.laws_weight,
            "court_weight": args.court_weight,
            "laws_citation_weight": args.laws_citation_weight,
            "laws_title_weight": args.laws_title_weight,
            "laws_text_weight": args.laws_text_weight,
            "court_citation_weight": args.court_citation_weight,
            "court_text_weight": args.court_text_weight,
            "enable_branch_aware_fusion": args.enable_branch_aware_fusion,
            "sparse_laws_branch_bonus": args.sparse_laws_branch_bonus,
            "sparse_laws_single_branch_bonus": args.sparse_laws_single_branch_bonus,
            "branch_aware_rank_cutoff": args.branch_aware_rank_cutoff,
            "branch_aware_fusion_mode": args.branch_aware_fusion_mode,
            "query_preprocess_enabled": args.enable_query_preprocess,
            "query_expansion_enabled": args.enable_query_expansion,
            "rule_recall_enabled": args.enable_rule_recall,
            "source_aware_fusion_enabled": args.source_aware_fusion,
            "branch_aware_fusion_enabled": args.enable_branch_aware_fusion,
        },
        "sparse_stats": sparse_stats,
        "dense_stats": dense_stats,
        "rule_stats": rule_stats,
        "eval_report": eval_report,
        "artifacts": {
            "val_predictions": str(val_pred_file),
            "test_predictions": str(test_pred_file),
            "submission": str(submission_file),
            "eval_dir": str(eval_dir),
        },
    }
    (args.out_dir / "run_meta_baseline_v1.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta["eval_report"], ensure_ascii=False))


if __name__ == "__main__":
    main()

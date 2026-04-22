from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from fusion import weighted_score_fusion
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from query_expansion import build_source_aware_query_packs
from query_preprocess import preprocess_query
from rerank import NoOpReranker, TokenOverlapReranker
from retrieval_dense import DenseRetriever
from retrieval_rules import RuleCitationRetriever
from retrieval_sparse import SparseRetriever


def load_lookup(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["norm_citation"]] = row["canonical_citation"]
    return out


def dedup_keep_max(score_items):
    best = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def dedup_ranked(items):
    out, seen = [], set()
    for x in items:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def recall_at_k(gold, ranked, k):
    if not gold:
        return 0.0
    return len(gold.intersection(set(ranked[:k]))) / len(gold)


def build_branch_metadata(sparse_items, dense_items, rule_items):
    # 分支命中集合（branch_hits）与法规分支 rank（sparse_laws_rank）。
    branch_hits = {}
    sparse_laws_rank = {}
    sparse_seen = set()
    dense_seen = set()
    rule_seen = set()

    sparse_rank_counter = 0
    for item in sparse_items:
        citation = item.citation
        if citation not in sparse_seen:
            sparse_seen.add(citation)
            sparse_rank_counter += 1
        branch_hits.setdefault(citation, set())
        if item.source == "laws_de":
            branch_hits[citation].add("sparse_laws")
            if citation not in sparse_laws_rank:
                sparse_laws_rank[citation] = sparse_rank_counter
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


def build_retrieval_cache(val_rows, args, all_norm):
    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(
        max_laws_rows=args.sparse_max_laws,
        max_court_rows=args.sparse_max_court,
        enable_field_aware=True,
    )
    dense = DenseRetriever(use_sbert=not args.dense_disable_sbert, text_max_chars=500, svd_dim=256)
    dense.build(
        max_laws_rows=args.dense_max_laws,
        max_court_rows=args.dense_max_court,
        enable_field_aware=True,
    )
    rule = RuleCitationRetriever()
    rule.build(max_laws_rows=args.sparse_max_laws, max_court_rows=args.sparse_max_court)

    doc_source = {}
    doc_text = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
            doc_text[d["citation"]] = d.get("text", "")
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])
        doc_text.setdefault(d["citation"], d.get("text", ""))

    cache = []
    for row in val_rows:
        q = row["query"]
        qid = row["query_id"]
        mv = preprocess_query(q)
        packs = build_source_aware_query_packs(mv)

        sparse_items = sparse.search_field_aware(
            laws_query_pack=packs["laws_query_pack"],
            court_query_pack=packs["court_query_pack"],
            laws_query_pack_v2=packs.get("laws_query_pack_v2", {}),
            enable_laws_query_pack_v2=True,
            top_k_laws=200,
            top_k_court=200,
            laws_citation_weight=args.laws_citation_weight,
            laws_title_weight=args.laws_title_weight,
            laws_text_weight=args.laws_text_weight,
            court_citation_weight=args.court_citation_weight,
            court_text_weight=args.court_text_weight,
        )
        dense_items = dense.search_source_aware(
            laws_query_pack=packs["laws_query_pack"],
            court_query_pack=packs["court_query_pack"],
            top_k_laws=200,
            top_k_court=200,
        )
        rule_items = rule.search(q, top_k_laws=200, top_k_court=200)

        sparse_scores = dedup_keep_max([(x.citation, x.score) for x in sparse_items])
        dense_scores = dedup_keep_max([(x.citation, x.score) for x in dense_items])
        rule_scores = dedup_keep_max([(x.citation, x.score) for x in rule_items])
        sparse_laws_ranked = dedup_ranked([x.citation for x in sparse_items if x.source == "laws_de"])
        branch_hits, sparse_laws_rank = build_branch_metadata(sparse_items, dense_items, rule_items)

        gold_norm = set(normalize_citation(x) for x in row["gold_citations"].split(";") if normalize_citation(x))
        gold_in_corpus = set(x for x in gold_norm if x in all_norm)
        cache.append(
            {
                "query_id": qid,
                "query": q,
                "gold_in_corpus": gold_in_corpus,
                "sparse_scores": sparse_scores,
                "dense_scores": dense_scores,
                "rule_scores": rule_scores,
                "sparse_laws_ranked": sparse_laws_ranked,
                "branch_hits": branch_hits,
                "sparse_laws_rank": sparse_laws_rank,
            }
        )
    return cache, doc_source, doc_text


def run_config(name, cfg, cache, doc_source, doc_text, reranker, top_n, lookup, laws_weight, court_weight):
    pred_map = {}
    fusion_r100 = []
    fusion_r200 = []
    rescued_count = 0
    for q in cache:
        fused = weighted_score_fusion(
            score_lists=[q["sparse_scores"], q["dense_scores"], q["rule_scores"]],
            weights=[0.6, 0.4, 0.35],
            top_n=500,
            citation_to_source=doc_source,
            source_aware_fusion=True,
            laws_weight=laws_weight,
            court_weight=court_weight,
            branch_hits=q["branch_hits"],
            sparse_laws_rank=q["sparse_laws_rank"],
            enable_branch_aware_fusion=cfg["branch_aware_fusion_enabled"],
            branch_aware_fusion_mode=cfg["fusion_mode"],
            sparse_laws_branch_bonus=cfg["sparse_laws_branch_bonus"],
            sparse_laws_single_branch_bonus=cfg["sparse_laws_single_branch_bonus"],
            branch_aware_rank_cutoff=cfg["branch_aware_rank_cutoff"],
        )
        fusion_ranked = [c for c, _ in fused]
        fusion_r100.append(recall_at_k(q["gold_in_corpus"], fusion_ranked, 100))
        fusion_r200.append(recall_at_k(q["gold_in_corpus"], fusion_ranked, 200))

        sparse_tail_hit = False
        for c in q["gold_in_corpus"]:
            rank = q["sparse_laws_rank"].get(c, 10**9)
            if 101 <= rank <= 200:
                sparse_tail_hit = True
                break
        if sparse_tail_hit and q["gold_in_corpus"].intersection(set(fusion_ranked[:200])):
            rescued_count += 1

        candidates = [
            {"citation": c, "source": doc_source.get(c, ""), "text": doc_text.get(c, ""), "fused_score": s, "score": s}
            for c, s in fused
        ]
        pred_map[q["query_id"]] = [x["citation"] for x in reranker.rerank(query=q["query"], candidates=candidates, top_n=top_n)]

    val_rows = load_query_split("val")
    strict, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="strict")
    corpus, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="corpus_aware")
    return {
        "run_name": name,
        "branch_aware_fusion_enabled": cfg["branch_aware_fusion_enabled"],
        "fusion_mode": cfg["fusion_mode"],
        "sparse_laws_branch_bonus": cfg["sparse_laws_branch_bonus"],
        "sparse_laws_single_branch_bonus": cfg["sparse_laws_single_branch_bonus"],
        "branch_aware_rank_cutoff": cfg["branch_aware_rank_cutoff"],
        "fusion_final_Recall@100": round(sum(fusion_r100) / len(fusion_r100), 6) if fusion_r100 else 0.0,
        "fusion_final_Recall@200": round(sum(fusion_r200) / len(fusion_r200), 6) if fusion_r200 else 0.0,
        "strict_macro_f1": round(strict["macro_f1"], 6),
        "corpus_aware_macro_f1": round(corpus["macro_f1"], 6),
        "tail_rescued_queries": rescued_count,
    }


def main():
    p = argparse.ArgumentParser(description="Branch-aware fusion ablation on baseline_v1_3_laws_focus retrieval cache.")
    p.add_argument("--reranker", choices=["none", "token_overlap"], default="token_overlap")
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--sparse-max-laws", type=int, default=175933)
    p.add_argument("--sparse-max-court", type=int, default=120000)
    p.add_argument("--dense-max-laws", type=int, default=60000)
    p.add_argument("--dense-max-court", type=int, default=80000)
    p.add_argument("--dense-disable-sbert", action="store_true")
    p.add_argument("--laws-weight", type=float, default=1.15)
    p.add_argument("--court-weight", type=float, default=0.95)
    p.add_argument("--laws-citation-weight", type=float, default=1.4)
    p.add_argument("--laws-title-weight", type=float, default=1.2)
    p.add_argument("--laws-text-weight", type=float, default=0.9)
    p.add_argument("--court-citation-weight", type=float, default=1.25)
    p.add_argument("--court-text-weight", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "v1_3_branch_aware_fusion")
    args = p.parse_args()

    val_rows = load_query_split("val")
    lookup = load_lookup(ROOT / "artifacts" / "phase0" / "citation_lookup.csv")
    if lookup:
        all_norm = set(lookup.keys())
    else:
        all_norm = set()

    cache, doc_source, doc_text = build_retrieval_cache(val_rows, args, all_norm)
    reranker = TokenOverlapReranker() if args.reranker == "token_overlap" else NoOpReranker()

    runs = []
    base_cfg = {
        "branch_aware_fusion_enabled": False,
        "fusion_mode": "sparse_laws_bonus",
        "sparse_laws_branch_bonus": 0.0,
        "sparse_laws_single_branch_bonus": 0.0,
        "branch_aware_rank_cutoff": 200,
    }
    runs.append(
        run_config(
            name="baseline_v1_3_original_fusion",
            cfg=base_cfg,
            cache=cache,
            doc_source=doc_source,
            doc_text=doc_text,
            reranker=reranker,
            top_n=args.top_n,
            lookup=lookup,
            laws_weight=args.laws_weight,
            court_weight=args.court_weight,
        )
    )

    bonus_grid = [0.0, 0.05, 0.1, 0.2]
    cutoff_grid = [100, 150, 200]
    for mode in ["sparse_laws_bonus", "sparse_laws_tail_rescue"]:
        for b in bonus_grid:
            for sb in bonus_grid:
                for cutoff in cutoff_grid:
                    cfg = {
                        "branch_aware_fusion_enabled": True,
                        "fusion_mode": mode,
                        "sparse_laws_branch_bonus": b,
                        "sparse_laws_single_branch_bonus": sb,
                        "branch_aware_rank_cutoff": cutoff,
                    }
                    name = f"ba_{mode}_b{b}_sb{sb}_k{cutoff}"
                    runs.append(
                        run_config(
                            name=name,
                            cfg=cfg,
                            cache=cache,
                            doc_source=doc_source,
                            doc_text=doc_text,
                            reranker=reranker,
                            top_n=args.top_n,
                            lookup=lookup,
                            laws_weight=args.laws_weight,
                            court_weight=args.court_weight,
                        )
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = args.out_dir / "branch_aware_fusion_results.csv"
    cols = [
        "run_name",
        "branch_aware_fusion_enabled",
        "fusion_mode",
        "sparse_laws_branch_bonus",
        "sparse_laws_single_branch_bonus",
        "branch_aware_rank_cutoff",
        "fusion_final_Recall@100",
        "fusion_final_Recall@200",
        "strict_macro_f1",
        "corpus_aware_macro_f1",
        "tail_rescued_queries",
    ]
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow(r)

    base = runs[0]
    best = sorted(
        runs[1:],
        key=lambda x: (
            x["corpus_aware_macro_f1"],
            x["strict_macro_f1"],
            x["fusion_final_Recall@200"],
            x["fusion_final_Recall@100"],
            x["tail_rescued_queries"],
        ),
        reverse=True,
    )[0]

    best_json = {
        "baseline": base,
        "best_branch_aware": best,
        "delta_vs_baseline": {
            "fusion_final_Recall@100": round(best["fusion_final_Recall@100"] - base["fusion_final_Recall@100"], 6),
            "fusion_final_Recall@200": round(best["fusion_final_Recall@200"] - base["fusion_final_Recall@200"], 6),
            "strict_macro_f1": round(best["strict_macro_f1"] - base["strict_macro_f1"], 6),
            "corpus_aware_macro_f1": round(best["corpus_aware_macro_f1"] - base["corpus_aware_macro_f1"], 6),
            "tail_rescued_queries": best["tail_rescued_queries"] - base["tail_rescued_queries"],
        },
    }
    (args.out_dir / "best_branch_aware_config.json").write_text(
        json.dumps(best_json, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )

    delta_r200 = best_json["delta_vs_baseline"]["fusion_final_Recall@200"]
    delta_strict = best_json["delta_vs_baseline"]["strict_macro_f1"]
    delta_corpus = best_json["delta_vs_baseline"]["corpus_aware_macro_f1"]
    tail_gain = best_json["delta_vs_baseline"]["tail_rescued_queries"]

    lines = []
    lines.append("# branch-aware fusion 实验总结（v1_3 laws focus）")
    lines.append("")
    lines.append("## 总体结论")
    lines.append(f"- 是否首次把 v1_3 的 sparse_laws 深尾部命中送进 fusion_top_200：{'是' if tail_gain > 0 else '否'}。")
    lines.append(f"- fusion_final Recall@200 是否开始提升：{'是' if delta_r200 > 0 else '否'}（delta={delta_r200:.4f}）。")
    lines.append(
        f"- strict/corpus_aware F1 是否开始提升：strict delta={delta_strict:.4f}，corpus_aware delta={delta_corpus:.4f}。"
    )
    if delta_r200 <= 0 and delta_corpus <= 0:
        lines.append("- 若仍不提升，当前主因更像：奖励仍不足（bonus 不够）+ 新命中仍偏深（rank 太深）。")
    else:
        lines.append("- 当前结果表明：fusion 层已可传递部分 sparse_laws 新增命中。")
    lines.append("")
    lines.append("## 最优配置（best branch-aware config）")
    lines.append(
        f"- run_name={best['run_name']}, mode={best['fusion_mode']}, "
        f"sparse_laws_branch_bonus={best['sparse_laws_branch_bonus']}, "
        f"sparse_laws_single_branch_bonus={best['sparse_laws_single_branch_bonus']}, "
        f"branch_aware_rank_cutoff={best['branch_aware_rank_cutoff']}"
    )
    lines.append("")
    lines.append("## 下一步建议")
    if delta_r200 > 0 or delta_corpus > 0:
        lines.append("- 仍值得继续做 branch-aware fusion，但应缩小到更细粒度网格（围绕最优点局部搜索）。")
    else:
        lines.append("- 优先回到 sparse_laws 分支内字段权重（field weight）调参，再次拉升分支内 rank 与 score。")
    lines.append("- 保持只在 fusion 层实验，避免与 retrieval 改动耦合，确保可回归。")
    (args.out_dir / "branch_aware_fusion_summary_cn.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print(
        json.dumps(
            {"rows": len(runs), "out_dir": str(args.out_dir), "best_run": best["run_name"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

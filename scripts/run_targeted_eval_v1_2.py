from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from fusion import rrf_fusion, weighted_score_fusion
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from query_expansion import build_bilingual_retrieval_views, build_source_aware_query_packs
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


def dedup_ranked(items: list[str]) -> list[str]:
    out, seen = [], set()
    for x in items:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def dedup_keep_max(score_items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    best = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(set(ranked[:k]))) / len(gold)


def branch_means(rows: list[dict], branch: str) -> dict:
    vals = [r for r in rows if r["branch_name"] == branch]
    if not vals:
        return {"r50": 0.0, "r100": 0.0, "r200": 0.0}
    return {
        "r50": sum(r["hit_at_50"] for r in vals) / len(vals),
        "r100": sum(r["hit_at_100"] for r in vals) / len(vals),
        "r200": sum(r["hit_at_200"] for r in vals) / len(vals),
    }


def run_config(
    name: str,
    val_rows: list[dict],
    lookup: dict[str, str],
    cfg: dict,
    args,
) -> tuple[list[dict], dict]:
    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(
        max_laws_rows=args.sparse_max_laws,
        max_court_rows=args.sparse_max_court,
        enable_field_aware=cfg["enable_field_aware_retrieval"],
    )
    dense = DenseRetriever(
        use_sbert=not args.dense_disable_sbert,
        text_max_chars=500,
        svd_dim=256,
    )
    dense.build(
        max_laws_rows=args.dense_max_laws,
        max_court_rows=args.dense_max_court,
        enable_field_aware=cfg["enable_field_aware_retrieval"],
    )
    rule = RuleCitationRetriever()
    rule.build(max_laws_rows=args.sparse_max_laws, max_court_rows=args.sparse_max_court)
    reranker = TokenOverlapReranker() if args.reranker == "token_overlap" else NoOpReranker()

    # citation source map for fusion
    doc_source = {}
    doc_text = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
            doc_text[d["citation"]] = d.get("text", "")
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])
        doc_text.setdefault(d["citation"], d.get("text", ""))

    all_norm_corpus = set(lookup.keys()) if lookup else set()
    if not all_norm_corpus:
        all_norm_corpus = set(normalize_citation(x) for x in doc_source.keys() if x)

    branch_rows = []
    pred_map = {}
    for row in val_rows:
        query = row["query"]
        qid = row["query_id"]
        mv = preprocess_query(query)
        bi = build_bilingual_retrieval_views(mv)
        src_pack = build_source_aware_query_packs(mv)

        if cfg["enable_field_aware_retrieval"]:
            sparse_items = sparse.search_field_aware(
                laws_query_pack=src_pack["laws_query_pack"],
                court_query_pack=src_pack["court_query_pack"],
                top_k_laws=200,
                top_k_court=200,
                laws_citation_weight=cfg["laws_citation_weight"],
                laws_title_weight=cfg["laws_title_weight"],
                laws_text_weight=cfg["laws_text_weight"],
                court_citation_weight=cfg["court_citation_weight"],
                court_text_weight=cfg["court_text_weight"],
            )
            dense_items = dense.search_source_aware(
                laws_query_pack=src_pack["laws_query_pack"],
                court_query_pack=src_pack["court_query_pack"],
                top_k_laws=200,
                top_k_court=200,
            )
        else:
            sparse_items = sparse.search_multi_view(
                query_original=query,
                query_keywords=mv.get("query_keywords", []),
                expanded_query_de=bi.get("expanded_query_de", ""),
                top_k_laws=200,
                top_k_court=200,
            )
            dense_items = dense.search_multi_view(
                bilingual_query_pack=bi.get("bilingual_query_pack", {}),
                top_k_laws=200,
                top_k_court=200,
            )

        rule_items = rule.search(query, top_k_laws=200, top_k_court=200) if cfg["enable_rule_recall"] else []

        sparse_laws = dedup_ranked([x.citation for x in sparse_items if x.source == "laws_de"])
        sparse_court = dedup_ranked([x.citation for x in sparse_items if x.source == "court_considerations"])
        dense_laws = dedup_ranked([x.citation for x in dense_items if x.source == "laws_de"])
        dense_court = dedup_ranked([x.citation for x in dense_items if x.source == "court_considerations"])
        rule_branch = dedup_ranked([x.citation for x in rule_items])

        sparse_scores = dedup_keep_max([(x.citation, x.score) for x in sparse_items])
        dense_scores = dedup_keep_max([(x.citation, x.score) for x in dense_items])
        rule_scores = dedup_keep_max([(x.citation, x.score) for x in rule_items])

        if args.fusion == "weighted":
            score_lists = [sparse_scores, dense_scores]
            weights = [0.6, 0.4]
            if cfg["enable_rule_recall"]:
                score_lists.append(rule_scores)
                weights.append(0.35)
            fused = weighted_score_fusion(
                score_lists=score_lists,
                weights=weights,
                top_n=500,
                citation_to_source=doc_source,
                source_aware_fusion=cfg["source_aware_fusion"],
                laws_weight=cfg["laws_weight"],
                court_weight=cfg["court_weight"],
            )
        else:
            rank_lists = [dedup_ranked([x.citation for x in sparse_items]), dedup_ranked([x.citation for x in dense_items])]
            if cfg["enable_rule_recall"]:
                rank_lists.append(rule_branch)
            fused = rrf_fusion(
                ranked_lists=rank_lists,
                k=60,
                top_n=500,
                citation_to_source=doc_source,
                source_aware_fusion=cfg["source_aware_fusion"],
                laws_weight=cfg["laws_weight"],
                court_weight=cfg["court_weight"],
            )
        fusion_final = [c for c, _ in fused]
        candidates = [
            {"citation": c, "source": doc_source.get(c, ""), "text": doc_text.get(c, ""), "fused_score": s, "score": s}
            for c, s in fused
        ]
        final = [x["citation"] for x in reranker.rerank(query=query, candidates=candidates, top_n=args.top_n)]
        pred_map[qid] = final

        gold_raw = [x.strip() for x in row["gold_citations"].split(";") if x.strip()]
        gold_norm = [normalize_citation(x) for x in gold_raw if normalize_citation(x)]
        gold_in_corpus = set(x for x in gold_norm if x in all_norm_corpus)

        branches = {
            "sparse_laws": sparse_laws,
            "sparse_court": sparse_court,
            "dense_laws": dense_laws,
            "dense_court": dense_court,
            "fusion_final": fusion_final,
        }
        for b, ranked in branches.items():
            branch_rows.append(
                {
                    "config_name": name,
                    "query_id": qid,
                    "branch_name": b,
                    "hit_at_50": recall_at_k(gold_in_corpus, ranked, 50),
                    "hit_at_100": recall_at_k(gold_in_corpus, ranked, 100),
                    "hit_at_200": recall_at_k(gold_in_corpus, ranked, 200),
                }
            )

    strict, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="strict")
    corpus, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="corpus_aware")
    return branch_rows, {
        "strict_macro_f1": strict["macro_f1"],
        "corpus_aware_macro_f1": corpus["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted comparison between baseline_v1_1 and baseline_v1_2.")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="weighted")
    parser.add_argument("--reranker", choices=["none", "token_overlap"], default="token_overlap")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--sparse-max-court", type=int, default=120000)
    parser.add_argument("--dense-max-laws", type=int, default=60000)
    parser.add_argument("--dense-max-court", type=int, default=80000)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "v1_2_targeted_eval")
    args = parser.parse_args()

    val_rows = load_query_split("val")
    lookup = load_lookup(ROOT / "artifacts" / "phase0" / "citation_lookup.csv")

    cfg_v11 = {
        "enable_rule_recall": True,
        "source_aware_fusion": True,
        "laws_weight": 1.1,
        "court_weight": 1.0,
        "enable_field_aware_retrieval": False,
        "laws_citation_weight": 1.0,
        "laws_title_weight": 1.0,
        "laws_text_weight": 1.0,
        "court_citation_weight": 1.0,
        "court_text_weight": 1.0,
    }
    cfg_v12 = {
        "enable_rule_recall": True,
        "source_aware_fusion": True,
        "laws_weight": 1.15,
        "court_weight": 0.95,
        "enable_field_aware_retrieval": True,
        "laws_citation_weight": 1.4,
        "laws_title_weight": 1.2,
        "laws_text_weight": 0.9,
        "court_citation_weight": 1.25,
        "court_text_weight": 1.0,
    }

    rows_v11, f1_v11 = run_config("baseline_v1_1", val_rows, lookup, cfg_v11, args)
    rows_v12, f1_v12 = run_config("baseline_v1_2", val_rows, lookup, cfg_v12, args)

    branches = ["sparse_laws", "sparse_court", "dense_laws", "dense_court", "fusion_final"]
    compare_rows = []
    for b in branches:
        m11 = branch_means(rows_v11, b)
        m12 = branch_means(rows_v12, b)
        compare_rows.append(
            {
                "metric": f"{b}_Recall@50",
                "baseline_v1_1": round(m11["r50"], 6),
                "baseline_v1_2": round(m12["r50"], 6),
                "delta": round(m12["r50"] - m11["r50"], 6),
            }
        )
        compare_rows.append(
            {
                "metric": f"{b}_Recall@100",
                "baseline_v1_1": round(m11["r100"], 6),
                "baseline_v1_2": round(m12["r100"], 6),
                "delta": round(m12["r100"] - m11["r100"], 6),
            }
        )
        compare_rows.append(
            {
                "metric": f"{b}_Recall@200",
                "baseline_v1_1": round(m11["r200"], 6),
                "baseline_v1_2": round(m12["r200"], 6),
                "delta": round(m12["r200"] - m11["r200"], 6),
            }
        )
    compare_rows.append(
        {
            "metric": "strict_macro_f1",
            "baseline_v1_1": f1_v11["strict_macro_f1"],
            "baseline_v1_2": f1_v12["strict_macro_f1"],
            "delta": round(f1_v12["strict_macro_f1"] - f1_v11["strict_macro_f1"], 6),
        }
    )
    compare_rows.append(
        {
            "metric": "corpus_aware_macro_f1",
            "baseline_v1_1": f1_v11["corpus_aware_macro_f1"],
            "baseline_v1_2": f1_v12["corpus_aware_macro_f1"],
            "delta": round(f1_v12["corpus_aware_macro_f1"] - f1_v11["corpus_aware_macro_f1"], 6),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "targeted_compare_v1_1_vs_v1_2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        w.writeheader()
        w.writerows(compare_rows)

    # summary
    def get_metric(m: str):
        r = next((x for x in compare_rows if x["metric"] == m), None)
        return r["baseline_v1_1"], r["baseline_v1_2"], r["delta"]

    sl50 = get_metric("sparse_laws_Recall@200")
    dl50 = get_metric("dense_laws_Recall@200")
    sc50 = get_metric("sparse_court_Recall@200")
    dc50 = get_metric("dense_court_Recall@200")

    escaped_zero = any(v2 > 0 for _, v2, _ in [sl50, dl50, sc50, dc50])
    laws_lift = (sl50[2] + dl50[2]) / 2
    court_lift = (sc50[2] + dc50[2]) / 2
    if laws_lift > 0 and court_lift > 0:
        solved = "两者都有"
    elif laws_lift > court_lift:
        solved = "field_usage_issue"
    else:
        solved = "language_gap"

    lines = []
    lines.append("# baseline_v1_2 定向对比总结")
    lines.append("")
    lines.append("## 结论")
    lines.append(f"- sparse/dense 主分支是否脱离 0：{'是' if escaped_zero else '否'}。")
    lines.append(f"- laws 侧先被拉起还是 court 侧先被拉起：{'laws 侧' if laws_lift >= court_lift else 'court 侧'}。")
    lines.append(f"- 本次改动更像解决了：{solved}。")
    lines.append("")
    lines.append("## 关键指标（Recall@200）")
    lines.append(f"- sparse_laws: v1_1={sl50[0]:.4f}, v1_2={sl50[1]:.4f}, delta={sl50[2]:.4f}")
    lines.append(f"- dense_laws: v1_1={dl50[0]:.4f}, v1_2={dl50[1]:.4f}, delta={dl50[2]:.4f}")
    lines.append(f"- sparse_court: v1_1={sc50[0]:.4f}, v1_2={sc50[1]:.4f}, delta={sc50[2]:.4f}")
    lines.append(f"- dense_court: v1_1={dc50[0]:.4f}, v1_2={dc50[1]:.4f}, delta={dc50[2]:.4f}")
    lines.append("")
    lines.append("## F1 指标")
    lines.append(
        f"- strict_macro_f1: v1_1={f1_v11['strict_macro_f1']:.4f}, v1_2={f1_v12['strict_macro_f1']:.4f}, delta={f1_v12['strict_macro_f1']-f1_v11['strict_macro_f1']:.4f}"
    )
    lines.append(
        f"- corpus_aware_macro_f1: v1_1={f1_v11['corpus_aware_macro_f1']:.4f}, v1_2={f1_v12['corpus_aware_macro_f1']:.4f}, delta={f1_v12['corpus_aware_macro_f1']-f1_v11['corpus_aware_macro_f1']:.4f}"
    )
    lines.append("")
    lines.append("## 若主分支仍接近 0，下一步最应怀疑")
    lines.append("- 先怀疑词典/模板覆盖仍不足（尤其是长案情英文描述到德文法规表达的桥接不足）。")
    lines.append("- 其次怀疑字段权重未对齐任务分布，需要按 laws/court 分开网格搜索。")

    (args.out_dir / "targeted_summary_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(compare_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


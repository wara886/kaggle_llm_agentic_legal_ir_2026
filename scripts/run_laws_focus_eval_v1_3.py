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


def dedup_ranked(items):
    out, seen = [], set()
    for x in items:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def dedup_keep_max(score_items):
    best = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def recall_at_k(gold, ranked, k):
    if not gold:
        return 0.0
    return len(gold.intersection(set(ranked[:k]))) / len(gold)


def mean(rows, branch, col):
    vals = [r[col] for r in rows if r["branch_name"] == branch]
    return (sum(vals) / len(vals)) if vals else 0.0


def run_cfg(name, val_rows, lookup, args, enable_laws_query_pack_v2):
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
    reranker = TokenOverlapReranker() if args.reranker == "token_overlap" else NoOpReranker()

    doc_source = {}
    doc_text = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
            doc_text[d["citation"]] = d.get("text", "")
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])
        doc_text.setdefault(d["citation"], d.get("text", ""))

    all_norm = set(lookup.keys()) if lookup else set(normalize_citation(x) for x in doc_source.keys())
    pred_map = {}
    branch_rows = []

    for row in val_rows:
        q = row["query"]
        qid = row["query_id"]
        mv = preprocess_query(q)
        packs = build_source_aware_query_packs(mv)

        sparse_items = sparse.search_field_aware(
            laws_query_pack=packs["laws_query_pack"],
            court_query_pack=packs["court_query_pack"],
            laws_query_pack_v2=packs.get("laws_query_pack_v2", {}),
            enable_laws_query_pack_v2=enable_laws_query_pack_v2,
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

        sparse_laws = dedup_ranked([x.citation for x in sparse_items if x.source == "laws_de"])
        sparse_scores = dedup_keep_max([(x.citation, x.score) for x in sparse_items])
        dense_scores = dedup_keep_max([(x.citation, x.score) for x in dense_items])
        rule_scores = dedup_keep_max([(x.citation, x.score) for x in rule_items])

        fused = weighted_score_fusion(
            score_lists=[sparse_scores, dense_scores, rule_scores],
            weights=[0.6, 0.4, 0.35],
            top_n=500,
            citation_to_source=doc_source,
            source_aware_fusion=True,
            laws_weight=args.laws_weight,
            court_weight=args.court_weight,
        )
        fusion_final = [c for c, _ in fused]
        candidates = [
            {"citation": c, "source": doc_source.get(c, ""), "text": doc_text.get(c, ""), "fused_score": s, "score": s}
            for c, s in fused
        ]
        final = [x["citation"] for x in reranker.rerank(query=q, candidates=candidates, top_n=args.top_n)]
        pred_map[qid] = final

        gold = set(normalize_citation(x) for x in row["gold_citations"].split(";") if normalize_citation(x))
        gold = set(x for x in gold if x in all_norm)
        branch_rows.append(
            {
                "config_name": name,
                "query_id": qid,
                "branch_name": "sparse_laws",
                "hit_at_50": recall_at_k(gold, sparse_laws, 50),
                "hit_at_100": recall_at_k(gold, sparse_laws, 100),
                "hit_at_200": recall_at_k(gold, sparse_laws, 200),
            }
        )
        branch_rows.append(
            {
                "config_name": name,
                "query_id": qid,
                "branch_name": "fusion_final",
                "hit_at_50": recall_at_k(gold, fusion_final, 50),
                "hit_at_100": recall_at_k(gold, fusion_final, 100),
                "hit_at_200": recall_at_k(gold, fusion_final, 200),
            }
        )

    strict, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="strict")
    corpus, _ = evaluate_predictions(val_rows, pred_map, citation_lookup=lookup if lookup else None, mode="corpus_aware")
    return branch_rows, strict["macro_f1"], corpus["macro_f1"]


def main():
    p = argparse.ArgumentParser(description="Compare baseline_v1_2 vs baseline_v1_3_laws_focus.")
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
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "v1_3_laws_focus_eval")
    args = p.parse_args()

    val_rows = load_query_split("val")
    lookup = load_lookup(ROOT / "artifacts" / "phase0" / "citation_lookup.csv")

    rows_v12, strict_v12, corpus_v12 = run_cfg("baseline_v1_2", val_rows, lookup, args, enable_laws_query_pack_v2=False)
    rows_v13, strict_v13, corpus_v13 = run_cfg(
        "baseline_v1_3_laws_focus", val_rows, lookup, args, enable_laws_query_pack_v2=True
    )

    metrics = []
    for branch in ["sparse_laws", "fusion_final"]:
        for k in ["hit_at_50", "hit_at_100", "hit_at_200"]:
            v12 = mean(rows_v12, branch, k)
            v13 = mean(rows_v13, branch, k)
            metrics.append(
                {
                    "metric": f"{branch}_Recall@{k.split('_')[-1]}",
                    "baseline_v1_2": round(v12, 6),
                    "baseline_v1_3_laws_focus": round(v13, 6),
                    "delta": round(v13 - v12, 6),
                }
            )
    metrics.append(
        {
            "metric": "strict_macro_f1",
            "baseline_v1_2": strict_v12,
            "baseline_v1_3_laws_focus": strict_v13,
            "delta": round(strict_v13 - strict_v12, 6),
        }
    )
    metrics.append(
        {
            "metric": "corpus_aware_macro_f1",
            "baseline_v1_2": corpus_v12,
            "baseline_v1_3_laws_focus": corpus_v13,
            "delta": round(corpus_v13 - corpus_v12, 6),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "laws_focus_compare_v1_2_vs_v1_3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)

    def gm(name):
        return next(x for x in metrics if x["metric"] == name)

    sl100 = gm("sparse_laws_Recall@100")
    sl200 = gm("sparse_laws_Recall@200")
    ff100 = gm("fusion_final_Recall@100")
    ff200 = gm("fusion_final_Recall@200")

    lines = []
    lines.append("# baseline_v1_3_laws_focus 总结")
    lines.append("")
    lines.append("## 关键结论")
    lines.append(
        f"- laws_query_pack_v2 是否继续拉高 sparse_laws Recall@100/200：{'是' if (sl100['delta']>0 or sl200['delta']>0) else '否'}。"
    )
    lines.append(
        f"- 是否第一次明显带动 fusion_final Recall@100/200 上升：{'是' if (ff100['delta']>0 or ff200['delta']>0) else '否'}。"
    )
    if sl200["delta"] > 0 and ff200["delta"] <= 0:
        lines.append("- 若 sparse_laws 涨而 fusion_final 不涨，最可能说明融合权重与截断策略尚未把新增 laws 候选传递到最终列表。")
    else:
        lines.append("- sparse 与 fusion 同步变化，说明 laws 侧扩展已部分传导到最终候选。")
    lines.append(
        f"- 当前下一步是否仍应优先继续做 laws 侧：{'是' if sl200['delta']>=0 else '否'}。"
    )
    lines.append("")
    lines.append("## 指标明细")
    for m in metrics:
        lines.append(
            f"- {m['metric']}: v1_2={float(m['baseline_v1_2']):.4f}, v1_3={float(m['baseline_v1_3_laws_focus']):.4f}, delta={float(m['delta']):.4f}"
        )
    (args.out_dir / "laws_focus_summary_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from fusion import rrf_fusion, weighted_score_fusion
from query_expansion import build_bilingual_retrieval_views
from query_preprocess import preprocess_query
from rerank import NoOpReranker, TokenOverlapReranker
from retrieval_dense import DenseRetriever
from retrieval_rules import RuleCitationRetriever
from retrieval_sparse import SparseRetriever
from legal_ir.data_loader import load_query_split
from legal_ir.corpus_builder import iter_corpus_rows


@dataclass
class RunConfig:
    name: str
    enable_query_preprocess: bool
    enable_query_expansion: bool
    enable_rule_recall: bool
    source_aware_fusion: bool
    laws_weight: float
    court_weight: float


def parse_bool_flag(value: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expect true/false")


def dedup_ranked(citations: list[str]) -> list[str]:
    out = []
    seen = set()
    for c in citations:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def dedup_keep_max(score_items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for c, s in score_items:
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def split_raw_gold(serialized: str) -> list[str]:
    if not serialized:
        return []
    return [x.strip() for x in serialized.split(";") if x.strip()]


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return round(len(gold.intersection(set(ranked[:k]))) / len(gold), 6)


def first_hit_rank(gold: set[str], ranked: list[str]) -> int:
    for idx, c in enumerate(ranked, start=1):
        if c in gold:
            return idx
    return -1


def build_source_map(max_laws_rows: int | None, max_court_rows: int | None) -> dict[str, str]:
    source_map = {}
    for row in iter_corpus_rows(
        include_laws=True,
        include_court=True,
        max_laws_rows=max_laws_rows,
        max_court_rows=max_court_rows,
    ):
        source_map[row["citation"]] = row["source"]
        source_map[row["norm_citation"]] = row["source"]
    return source_map


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _query_language_gap(query: str, gold_in_corpus: list[str], any_hit: bool) -> bool:
    if any_hit or not gold_in_corpus:
        return False
    q = (query or "").lower()
    ascii_ratio = (sum(1 for ch in q if "a" <= ch <= "z") / max(1, len(q)))
    return ascii_ratio > 0.6


def _citation_pattern_missed(gold_raw: list[str], rule_candidates: list[str], any_hit: bool) -> bool:
    if any_hit:
        return False
    pat = re.compile(r"(art\.|article|bge|\d+[A-Za-z]*_[0-9]+/[0-9]{4})", re.I)
    has_pattern_gold = any(pat.search(x or "") for x in gold_raw)
    return has_pattern_gold and len(rule_candidates) == 0


def infer_failure_type(
    query: str,
    gold_raw: list[str],
    gold_in_corpus: list[str],
    branch_name: str,
    branch_candidates: list[str],
    fusion_candidates: list[str],
    final_candidates: list[str],
    pre_fusion_union: list[str],
    source_map: dict[str, str],
    rule_candidates: list[str],
) -> str:
    if not gold_in_corpus:
        return "no_gold_in_candidates"
    gold_set = set(gold_in_corpus)
    fr = first_hit_rank(gold_set, branch_candidates)
    any_hit = fr != -1

    if _query_language_gap(query, gold_in_corpus, any_hit):
        return "query_language_gap"
    if _citation_pattern_missed(gold_raw, rule_candidates, any_hit):
        return "citation_pattern_missed"

    # normalization mismatch: 原始不在语料，但规范化后在
    has_norm_mismatch = any(
        (raw not in source_map) and (normalize_citation(raw) in source_map) for raw in gold_raw
    )
    if has_norm_mismatch:
        return "normalization_mismatch"

    gold_sources = {source_map.get(c, "unknown") for c in gold_in_corpus}
    if branch_name.endswith("laws") and gold_sources == {"court_considerations"}:
        return "source_mismatch"
    if branch_name.endswith("court") and gold_sources == {"laws_de"}:
        return "source_mismatch"

    if not (gold_set & set(branch_candidates[:200])):
        return "no_gold_in_candidates"
    if fr > 50:
        return "gold_only_after_large_k"
    if branch_name == "fusion_final":
        if (gold_set & set(pre_fusion_union)) and not (gold_set & set(fusion_candidates)):
            return "gold_lost_in_fusion"
        if (gold_set & set(fusion_candidates)) and not (gold_set & set(final_candidates)):
            return "gold_lost_in_final_cut"
    return "ok"


def best_branch_from_recalls(branch_rows: list[dict]) -> str:
    ranked = sorted(
        branch_rows,
        key=lambda r: (float(r["hit_at_200"]), float(r["hit_at_100"]), float(r["hit_at_50"]), float(r["hit_at_10"])),
        reverse=True,
    )
    return ranked[0]["branch_name"] if ranked else "none"


def build_retrievers(args) -> tuple[SparseRetriever, DenseRetriever, RuleCitationRetriever]:
    sparse = SparseRetriever(text_max_chars=900)
    sparse.build(max_laws_rows=args.sparse_max_laws, max_court_rows=args.sparse_max_court)
    dense = DenseRetriever(
        use_sbert=not args.dense_disable_sbert,
        text_max_chars=500,
        svd_dim=256,
    )
    dense.build(max_laws_rows=args.dense_max_laws, max_court_rows=args.dense_max_court)
    rule = RuleCitationRetriever()
    rule.build(max_laws_rows=args.sparse_max_laws, max_court_rows=args.sparse_max_court)
    return sparse, dense, rule


def run_one_config(
    rows: list[dict],
    cfg: RunConfig,
    args,
    sparse: SparseRetriever,
    dense: DenseRetriever,
    rule: RuleCitationRetriever,
    source_map: dict[str, str],
    all_norm_corpus: set[str],
) -> list[dict]:
    reranker = TokenOverlapReranker() if args.reranker == "token_overlap" else NoOpReranker()
    doc_source = {}
    for src_docs in sparse.docs.values():
        for d in src_docs:
            doc_source[d["citation"]] = d["source"]
    for d in dense.doc_matrix.get("all_docs", []):
        doc_source.setdefault(d["citation"], d["source"])

    out_rows = []
    for row in rows:
        qid = row["query_id"]
        query = row["query"]
        gold_raw = split_raw_gold(row.get("gold_citations", ""))
        gold_norm = [normalize_citation(x) for x in gold_raw if normalize_citation(x)]
        gold_in_corpus = [x for x in gold_norm if x in all_norm_corpus]
        gold_set = set(gold_in_corpus)

        sparse_items = []
        dense_items = []
        expanded_query_de = ""
        query_keywords = []
        bilingual_pack = {}

        if cfg.enable_query_preprocess or cfg.enable_query_expansion:
            mv = preprocess_query(query)
            query_keywords = mv.get("query_keywords", [])
            if cfg.enable_query_expansion:
                ex = build_bilingual_retrieval_views(mv)
                expanded_query_de = ex.get("expanded_query_de", "")
                bilingual_pack = ex.get("bilingual_query_pack", {})
            sparse_items = sparse.search_multi_view(
                query_original=query,
                query_keywords=query_keywords,
                expanded_query_de=expanded_query_de,
                top_k_laws=200,
                top_k_court=200,
            )
            if cfg.enable_query_expansion and bilingual_pack:
                dense_items = dense.search_multi_view(bilingual_pack, top_k_laws=200, top_k_court=200)
            else:
                dense_items = dense.search(query=query, top_k_laws=200, top_k_court=200)
        else:
            sparse_items = sparse.search(query=query, top_k_laws=200, top_k_court=200)
            dense_items = dense.search(query=query, top_k_laws=200, top_k_court=200)

        rule_items = rule.search(query, top_k_laws=200, top_k_court=200) if cfg.enable_rule_recall else []

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
            if cfg.enable_rule_recall:
                score_lists.append(rule_scores)
                weights.append(0.35)
            fused = weighted_score_fusion(
                score_lists=score_lists,
                weights=weights,
                top_n=500,
                citation_to_source=doc_source,
                source_aware_fusion=cfg.source_aware_fusion,
                laws_weight=cfg.laws_weight,
                court_weight=cfg.court_weight,
            )
        else:
            rank_lists = [dedup_ranked([x.citation for x in sparse_items]), dedup_ranked([x.citation for x in dense_items])]
            if cfg.enable_rule_recall:
                rank_lists.append(rule_branch)
            fused = rrf_fusion(
                ranked_lists=rank_lists,
                k=60,
                top_n=500,
                citation_to_source=doc_source,
                source_aware_fusion=cfg.source_aware_fusion,
                laws_weight=cfg.laws_weight,
                court_weight=cfg.court_weight,
            )
        fusion_final = [c for c, _ in fused]
        candidates = [{"citation": c, "source": doc_source.get(c, ""), "text": "", "fused_score": s, "score": s} for c, s in fused]
        reranked = reranker.rerank(query=query, candidates=candidates, top_n=args.top_n_final)
        final_cut = [x["citation"] for x in reranked]

        pre_fusion_union = dedup_ranked(sparse_laws + sparse_court + dense_laws + dense_court + rule_branch)
        branches = {
            "sparse_laws": sparse_laws,
            "sparse_court": sparse_court,
            "dense_laws": dense_laws,
            "dense_court": dense_court,
            "rule_branch": rule_branch,
            "fusion_final": fusion_final,
        }
        temp_rows = []
        for branch_name, ranked in branches.items():
            fr = first_hit_rank(gold_set, ranked)
            ft = infer_failure_type(
                query=query,
                gold_raw=gold_raw,
                gold_in_corpus=gold_in_corpus,
                branch_name=branch_name,
                branch_candidates=ranked,
                fusion_candidates=fusion_final,
                final_candidates=final_cut,
                pre_fusion_union=pre_fusion_union,
                source_map=source_map,
                rule_candidates=rule_branch,
            )
            temp_rows.append(
                {
                    "config_name": cfg.name,
                    "query_id": qid,
                    "gold_citations_raw": ";".join(gold_raw),
                    "gold_citations_norm": ";".join(gold_norm),
                    "gold_in_corpus": ";".join(gold_in_corpus),
                    "first_hit_rank": fr,
                    "hit_at_10": recall_at_k(gold_set, ranked, 10),
                    "hit_at_50": recall_at_k(gold_set, ranked, 50),
                    "hit_at_100": recall_at_k(gold_set, ranked, 100),
                    "hit_at_200": recall_at_k(gold_set, ranked, 200),
                    "branch_name": branch_name,
                    "failure_type": ft,
                }
            )
        best = best_branch_from_recalls(temp_rows)
        for tr in temp_rows:
            tr["best_branch"] = best
        out_rows.extend(temp_rows)
    return out_rows


def build_compare_rows(rows_a: list[dict], rows_b: list[dict], cfg_a: RunConfig, cfg_b: RunConfig) -> list[dict]:
    idx_a = {(r["query_id"], r["branch_name"]): r for r in rows_a}
    idx_b = {(r["query_id"], r["branch_name"]): r for r in rows_b}
    keys = sorted(set(idx_a.keys()) | set(idx_b.keys()))

    out = []
    for k in keys:
        ra = idx_a.get(k)
        rb = idx_b.get(k)
        if ra is None or rb is None:
            continue
        delta200 = round(float(rb["hit_at_200"]) - float(ra["hit_at_200"]), 6)
        delta50 = round(float(rb["hit_at_50"]) - float(ra["hit_at_50"]), 6)
        suspected = "unchanged"
        if delta200 > 0:
            if k[1] == "rule_branch" and cfg_b.enable_rule_recall and not cfg_a.enable_rule_recall:
                suspected = "rule_recall"
            elif k[1].startswith("dense") and cfg_b.enable_query_expansion and not cfg_a.enable_query_expansion:
                suspected = "query_expansion"
            elif k[1].startswith("sparse") and cfg_b.enable_query_preprocess and not cfg_a.enable_query_preprocess:
                suspected = "query_preprocessing"
            elif k[1] == "fusion_final" and cfg_b.source_aware_fusion and not cfg_a.source_aware_fusion:
                suspected = "source_aware_fusion"
            else:
                suspected = "mixed_or_unknown"
        out.append(
            {
                "query_id": k[0],
                "branch_name": k[1],
                "hit_at_50_v1": ra["hit_at_50"],
                "hit_at_50_v1_1": rb["hit_at_50"],
                "hit_at_200_v1": ra["hit_at_200"],
                "hit_at_200_v1_1": rb["hit_at_200"],
                "delta_hit_at_50": delta50,
                "delta_hit_at_200": delta200,
                "failure_type_v1": ra["failure_type"],
                "failure_type_v1_1": rb["failure_type"],
                "suspected_module_driver": suspected,
            }
        )
    return out


def summarize_clusters(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    counter = Counter()
    for r in rows_a + rows_b:
        counter[(r["config_name"], r["branch_name"], r["failure_type"])] += 1
    out = []
    for (cfg, branch, ft), cnt in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        out.append({"config_name": cfg, "branch_name": branch, "failure_type": ft, "count": cnt})
    return out


def summarize_markdown(path: Path, rows_a: list[dict], rows_b: list[dict], compare_rows: list[dict], cfg_a: RunConfig, cfg_b: RunConfig) -> None:
    def avg_recall(rows: list[dict], branch: str, k: str) -> float:
        b = [r for r in rows if r["branch_name"] == branch]
        return (sum(float(x[k]) for x in b) / len(b)) if b else 0.0

    branches = ["sparse_laws", "sparse_court", "dense_laws", "dense_court", "rule_branch", "fusion_final"]
    lines = []
    lines.append("# 候选诊断总结（candidate diagnostics v1）")
    lines.append("")
    lines.append("## 为什么先做候选诊断")
    lines.append("- 当前目标是先确认 gold citation 是否进入候选集合（candidate set），避免把召回问题误判成排序问题。")
    lines.append("- 通过 v1 与 v1_1 并行对比，可以直接定位 query_preprocessing / query_expansion / rule_recall / source_aware_fusion 的边际贡献。")
    lines.append("")
    lines.append("## 分支 Recall@K 对比")
    lines.append("| branch | v1 R@50 | v1_1 R@50 | v1 R@200 | v1_1 R@200 |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in branches:
        lines.append(
            f"| {b} | {avg_recall(rows_a,b,'hit_at_50'):.4f} | {avg_recall(rows_b,b,'hit_at_50'):.4f} | {avg_recall(rows_a,b,'hit_at_200'):.4f} | {avg_recall(rows_b,b,'hit_at_200'):.4f} |"
        )
    lines.append("")
    lines.append("## Recall@K 与 Macro F1 的关系")
    lines.append("- Recall@K 反映“能否召回到”，是上游可达性指标。")
    lines.append("- Macro F1 还受融合与重排影响；Recall 提升不一定立刻转化为 F1 提升。")
    lines.append("")
    lines.append("## 如何判断是 recall 问题还是 ranking 问题")
    lines.append("- 更像 recall 问题：各分支 R@200 都低，failure_type 多为 no_gold_in_candidates/query_language_gap/citation_pattern_missed。")
    lines.append("- 更像 ranking 问题：分支 R@200 有命中，但 fusion_final 或 final_cut 丢失，failure_type 多为 gold_lost_in_fusion/gold_lost_in_final_cut。")
    lines.append("")
    improved = [r for r in compare_rows if float(r["delta_hit_at_200"]) > 0]
    lines.append(f"- v1_1 在分支级 R@200 提升条目数：{len(improved)}")
    lines.append(f"- config_a: {cfg_a}")
    lines.append(f"- config_b: {cfg_b}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate diagnostics between baseline_v1 and baseline_v1_1.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--reranker", choices=["none", "token_overlap"], default="token_overlap")
    parser.add_argument("--top-n-final", type=int, default=12)
    parser.add_argument("--sparse-max-laws", type=int, default=175933)
    parser.add_argument("--sparse-max-court", type=int, default=120000)
    parser.add_argument("--dense-max-laws", type=int, default=60000)
    parser.add_argument("--dense-max-court", type=int, default=80000)
    parser.add_argument("--dense-disable-sbert", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "diagnostics_v1")

    # 与 run_baseline_v1.py 对齐的快捷开关（默认作用于 config_b）
    parser.add_argument("--enable-query-preprocess", type=parse_bool_flag, default=None)
    parser.add_argument("--enable-query-expansion", type=parse_bool_flag, default=None)
    parser.add_argument("--enable-rule-recall", type=parse_bool_flag, default=None)
    parser.add_argument("--source-aware-fusion", type=parse_bool_flag, default=None)
    parser.add_argument("--laws-weight", type=float, default=None)
    parser.add_argument("--court-weight", type=float, default=None)

    # config_a: baseline_v1 兼容
    parser.add_argument("--config-a-enable-query-preprocess", type=parse_bool_flag, default=False)
    parser.add_argument("--config-a-enable-query-expansion", type=parse_bool_flag, default=False)
    parser.add_argument("--config-a-enable-rule-recall", type=parse_bool_flag, default=False)
    parser.add_argument("--config-a-source-aware-fusion", type=parse_bool_flag, default=False)
    parser.add_argument("--config-a-laws-weight", type=float, default=1.0)
    parser.add_argument("--config-a-court-weight", type=float, default=1.0)

    # config_b: baseline_v1_1 增强
    parser.add_argument("--config-b-enable-query-preprocess", type=parse_bool_flag, default=True)
    parser.add_argument("--config-b-enable-query-expansion", type=parse_bool_flag, default=True)
    parser.add_argument("--config-b-enable-rule-recall", type=parse_bool_flag, default=True)
    parser.add_argument("--config-b-source-aware-fusion", type=parse_bool_flag, default=True)
    parser.add_argument("--config-b-laws-weight", type=float, default=1.1)
    parser.add_argument("--config-b-court-weight", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_query_split(args.split)
    source_map = build_source_map(args.sparse_max_laws, args.sparse_max_court)
    all_norm_corpus = set(normalize_citation(x) for x in source_map.keys() if x)

    sparse, dense, rule = build_retrievers(args)
    cfg_a = RunConfig(
        name="baseline_v1",
        enable_query_preprocess=args.config_a_enable_query_preprocess,
        enable_query_expansion=args.config_a_enable_query_expansion,
        enable_rule_recall=args.config_a_enable_rule_recall,
        source_aware_fusion=args.config_a_source_aware_fusion,
        laws_weight=args.config_a_laws_weight,
        court_weight=args.config_a_court_weight,
    )
    cfg_b = RunConfig(
        name="baseline_v1_1",
        enable_query_preprocess=args.config_b_enable_query_preprocess,
        enable_query_expansion=args.config_b_enable_query_expansion,
        enable_rule_recall=args.config_b_enable_rule_recall,
        source_aware_fusion=args.config_b_source_aware_fusion,
        laws_weight=args.config_b_laws_weight,
        court_weight=args.config_b_court_weight,
    )

    # 快捷开关覆盖 config_b（便于复用 baseline_v1 参数风格）
    if args.enable_query_preprocess is not None:
        cfg_b.enable_query_preprocess = args.enable_query_preprocess
    if args.enable_query_expansion is not None:
        cfg_b.enable_query_expansion = args.enable_query_expansion
    if args.enable_rule_recall is not None:
        cfg_b.enable_rule_recall = args.enable_rule_recall
    if args.source_aware_fusion is not None:
        cfg_b.source_aware_fusion = args.source_aware_fusion
    if args.laws_weight is not None:
        cfg_b.laws_weight = args.laws_weight
    if args.court_weight is not None:
        cfg_b.court_weight = args.court_weight

    rows_a = run_one_config(rows, cfg_a, args, sparse, dense, rule, source_map, all_norm_corpus)
    rows_b = run_one_config(rows, cfg_b, args, sparse, dense, rule, source_map, all_norm_corpus)
    compare_rows = build_compare_rows(rows_a, rows_b, cfg_a, cfg_b)
    cluster_rows = summarize_clusters(rows_a, rows_b)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "recall_by_branch_v1.csv", rows_a)
    write_csv(args.out_dir / "recall_by_branch_v1_1.csv", rows_b)
    write_csv(args.out_dir / "recall_compare_v1_vs_v1_1.csv", compare_rows)
    write_csv(args.out_dir / "query_failure_clusters.csv", cluster_rows)
    summarize_markdown(args.out_dir / "diagnostics_summary_cn.md", rows_a, rows_b, compare_rows, cfg_a, cfg_b)

    print(
        json.dumps(
            {
                "split": args.split,
                "rows_v1": len(rows_a),
                "rows_v1_1": len(rows_b),
                "compare_rows": len(compare_rows),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

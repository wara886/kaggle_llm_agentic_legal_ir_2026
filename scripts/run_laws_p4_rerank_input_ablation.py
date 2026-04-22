from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from law_family import extract_family_from_citation
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from retrieval_rules import RuleCitationRetriever


def _parse_joined(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen = set()
    for x in text.split(";"):
        n = normalize_citation(x)
        if not n or n in seen:
            continue
        out.append(n)
        seen.add(n)
    return out


def _rank_in_list(citation: str, ranked: list[str]) -> int:
    try:
        return ranked.index(citation) + 1
    except ValueError:
        return -1


def _load_source_map() -> dict[str, str]:
    src: dict[str, str] = {}
    for row in iter_corpus_rows(include_laws=True, include_court=True):
        norm = normalize_citation(row.get("citation", ""))
        if norm and norm not in src:
            src[norm] = row.get("source", "")
    return src


def _trace_by_qid(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["query_id"]: r for r in rows}


def _read_pred_map(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = row.get("query_id", "").strip()
            if not qid:
                continue
            out[qid] = _parse_joined(row.get("predicted_citations", ""))
    return out


def _explicit_map(val_rows: list[dict]) -> dict[str, int]:
    out = {}
    for r in val_rows:
        out[r["query_id"]] = int(bool(RuleCitationRetriever.extract_patterns(r.get("query", ""))))
    return out


def _family_consistent(citation: str, likely_families: list[str]) -> int:
    if not likely_families:
        return 0
    fam = extract_family_from_citation(citation)
    return int(bool(fam and fam in {x.upper() for x in likely_families if x}))


def build_admission_audit_csv(
    *,
    val_rows: list[dict],
    trace_csv: Path,
    val_pred_csv: Path,
    out_csv: Path,
) -> list[dict]:
    source_map = _load_source_map()
    trace_map = _trace_by_qid(trace_csv)
    pred_map = _read_pred_map(val_pred_csv)
    explicit_map = _explicit_map(val_rows)

    rows_out: list[dict] = []
    for r in val_rows:
        qid = r["query_id"]
        tr = trace_map.get(qid, {})
        fused_top200 = _parse_joined(tr.get("fused_top200", ""))
        fused_top320 = _parse_joined(tr.get("fused_top320", ""))
        rerank_input = _parse_joined(tr.get("rerank_input_citations", "")) or fused_top320
        reranked = _parse_joined(tr.get("reranked_top320", ""))
        final_cut = _parse_joined(tr.get("final_cut_predictions", ""))
        final_pred = _parse_joined(tr.get("final_predictions", "")) or pred_map.get(qid, [])

        likely_families = [x.strip().upper() for x in tr.get("likely_statute_family", "").split(";") if x.strip()]
        rule_hits = set(_parse_joined(tr.get("rule_laws_exact_citations", "")))
        issue_hits = set(_parse_joined(tr.get("issue_phrase_sparse_citations", "")))

        dynamic_mode = tr.get("dynamic_mode", "")
        fixed_top_k = int(tr.get("fixed_top_k", "0") or 0)

        for g in [_x for _x in (normalize_citation(x) for x in r.get("gold_citation_list", [])) if _x]:
            g_source = source_map.get(g, "unknown")
            source_tag = "laws" if g_source == "laws_de" else "court" if g_source == "court_considerations" else "unknown"
            in_fused200 = int(g in set(fused_top200))
            in_rerank_input = int(g in set(rerank_input))
            rank_fused = _rank_in_list(g, fused_top200)
            rank_before = _rank_in_list(g, rerank_input)
            rank_after = _rank_in_list(g, reranked)
            in_final_cut = int(g in set(final_cut))
            in_final_pred = int(g in set(final_pred))

            is_rule_hit = int(g in rule_hits)
            is_norm_consistent = is_rule_hit
            is_family_hit = _family_consistent(g, likely_families)
            is_issue_hit = int(g in issue_hits)

            if in_final_pred:
                drop_stage = "kept_final"
            elif not in_rerank_input:
                drop_stage = "not_in_rerank_input"
            elif rank_after <= 0:
                drop_stage = "not_reranked"
            elif dynamic_mode == "fixed_top_k" and fixed_top_k > 0 and rank_after > fixed_top_k:
                drop_stage = "reranked_too_low"
            else:
                drop_stage = "cut_by_dynamic_threshold"

            missing_reason = ""
            if drop_stage == "not_in_rerank_input":
                if in_fused200 == 0:
                    missing_reason = "simply fused rank too low"
                elif not is_family_hit:
                    missing_reason = "family consistency"
                elif not is_issue_hit:
                    missing_reason = "issue hit"
                elif not is_rule_hit:
                    missing_reason = "rule hit"
                else:
                    missing_reason = "simply fused rank too low"

            rows_out.append(
                {
                    "query_id": qid,
                    "explicit_subset": explicit_map.get(qid, 0),
                    "gold_citation": g,
                    "source": source_tag,
                    "gold_in_fused_top200": in_fused200,
                    "gold_in_rerank_input": in_rerank_input,
                    "gold_rank_before_rerank": rank_before,
                    "gold_fused_rank": rank_fused,
                    "gold_rank_after_rerank": rank_after,
                    "gold_in_final_predictions": in_final_pred,
                    "gold_drop_stage": drop_stage,
                    "is_rule_hit": is_rule_hit,
                    "is_normalization_consistent": is_norm_consistent,
                    "is_predicted_family_consistent": is_family_hit,
                    "is_issue_phrase_hit": is_issue_hit,
                    "not_in_rerank_input_missing": missing_reason,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _subset_rows(val_rows: list[dict], explicit: int | None) -> list[dict]:
    if explicit is None:
        return val_rows
    return [r for r in val_rows if int(bool(RuleCitationRetriever.extract_patterns(r.get("query", "")))) == explicit]


def _compute_metric_block(
    *,
    val_rows: list[dict],
    pred_csv: Path,
    audit_rows: list[dict],
    explicit: int | None,
) -> dict:
    subset = _subset_rows(val_rows, explicit)
    qset = {r["query_id"] for r in subset}
    pred_map_all = _read_pred_map(pred_csv)
    pred_map = {k: v for k, v in pred_map_all.items() if k in qset}

    strict, strict_rows = evaluate_predictions(subset, pred_map, citation_lookup=None, mode="strict")
    corpus, _ = evaluate_predictions(subset, pred_map, citation_lookup=None, mode="corpus_aware")
    fp = sum(int(r.get("fp", 0)) for r in strict_rows)

    scoped = [r for r in audit_rows if r["query_id"] in qset]
    recall200 = _mean([float(r["gold_in_fused_top200"]) for r in scoped])
    rerank_rate = _mean([float(r["gold_in_rerank_input"]) for r in scoped])
    drop = Counter(r["gold_drop_stage"] for r in scoped)

    return {
        "n": len(subset),
        "Recall@200": round(recall200, 6),
        "strict_f1": float(strict.get("macro_f1", 0.0)),
        "corpus_f1": float(corpus.get("macro_f1", 0.0)),
        "final_fp": int(fp),
        "gold_in_rerank_input_rate": round(rerank_rate, 6),
        "gold_drop_stage": dict(sorted(drop.items(), key=lambda x: x[0])),
    }


def _fmt_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['n']} | {m['Recall@200']:.6f} | {m['strict_f1']:.6f} | {m['corpus_f1']:.6f} "
        f"| {m['final_fp']} | {m['gold_in_rerank_input_rate']:.6f} |"
    )


def _run_pipeline(out_dir: Path, enable_p4: bool) -> None:
    done_flag = out_dir / "val_seed_trace_silver_baseline_v0.csv"
    if done_flag.exists():
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_silver_baseline_v0.py"),
        "--dynamic-mode",
        "fixed_top_k",
        "--fixed-top-k",
        "5",
        "--enable-router",
        "true",
        "--router-version",
        "v1",
        "--enable-court-mainline",
        "false",
        "--enable-court-dense",
        "false",
        "--seed-floor-sparse",
        "0",
        "--seed-floor-dense",
        "0",
        "--enable-rule-exact",
        "true",
        "--rule-top-k-laws",
        "20",
        "--enable-laws-primary-german-expansion",
        "false",
        "--enable-law-family-constraints",
        "true",
        "--law-family-boost",
        "2.5",
        "--law-family-min-keep",
        "5",
        "--enable-issue-phrase-refinement",
        "true",
        "--issue-phrase-top-k",
        "24",
        "--issue-phrase-boost",
        "2.5",
        "--issue-phrase-max-groups",
        "4",
        "--issue-phrase-max-terms",
        "16",
        "--enable-laws-final-cut-calibration",
        "true",
        "--laws-final-fused-rescue-top-k",
        "1",
        "--prefer-strong-reranker",
        "false",
        "--enable-laws-rerank-input-shaping",
        "true" if enable_p4 else "false",
        "--laws-rerank-shortlist-size",
        "320",
        "--laws-rerank-keep-fused-tail",
        "120",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    val_rows = load_query_split("val")

    p3_out = ROOT / "outputs" / "laws_p3_final_cut_calibration_p4audit"
    p4_out = ROOT / "outputs" / "laws_p4_rerank_input_shaping"

    _run_pipeline(p3_out, enable_p4=False)
    _run_pipeline(p4_out, enable_p4=True)

    p3_audit_csv = ROOT / "docs" / "laws_rerank_input_admission_audit.csv"
    p3_audit_rows = build_admission_audit_csv(
        val_rows=val_rows,
        trace_csv=p3_out / "val_seed_trace_silver_baseline_v0.csv",
        val_pred_csv=p3_out / "val_predictions_silver_baseline_v0.csv",
        out_csv=p3_audit_csv,
    )

    p4_audit_rows = build_admission_audit_csv(
        val_rows=val_rows,
        trace_csv=p4_out / "val_seed_trace_silver_baseline_v0.csv",
        val_pred_csv=p4_out / "val_predictions_silver_baseline_v0.csv",
        out_csv=ROOT / "artifacts" / "laws_p4" / "laws_p4_admission_audit.csv",
    )

    missing_counter = Counter(
        r["not_in_rerank_input_missing"]
        for r in p3_audit_rows
        if r["gold_drop_stage"] == "not_in_rerank_input" and r["not_in_rerank_input_missing"]
    )
    src_counter = Counter(r["source"] for r in p3_audit_rows)

    md_lines = [
        "# Laws Rerank Input Admission Audit",
        "",
        "## Scope",
        "- Validation split, P0 + P2-A + P2-B + P3 baseline (without P4 shaping).",
        "- Per `query_id + gold citation` admission trace for fused@200 and rerank input.",
        "",
        "## Feature Coverage",
        "| metric | value |",
        "|---|---:|",
        f"| gold_rows | {len(p3_audit_rows)} |",
        f"| source_laws | {src_counter.get('laws', 0)} |",
        f"| source_court | {src_counter.get('court', 0)} |",
        f"| gold_in_fused_top200_rate | {_mean([float(r['gold_in_fused_top200']) for r in p3_audit_rows]):.6f} |",
        f"| gold_in_rerank_input_rate | {_mean([float(r['gold_in_rerank_input']) for r in p3_audit_rows]):.6f} |",
        "",
        "## Not-In-Rerank Reasons",
        "| missing_reason | count |",
        "|---|---:|",
    ]
    for k, v in sorted(missing_counter.items(), key=lambda x: (-x[1], x[0])):
        md_lines.append(f"| {k} | {v} |")

    md_lines.extend(
        [
            "",
            "## Notes",
            "- `is_normalization_consistent` follows P0 exact-rule normalized match behavior.",
            "- `source` is mapped from corpus lookup (`laws_de` / `court_considerations`).",
            "",
            f"CSV: `{p3_audit_csv.as_posix()}`",
        ]
    )
    (ROOT / "docs" / "laws_rerank_input_admission_audit.md").write_text("\n".join(md_lines), encoding="utf-8")

    p3_overall = _compute_metric_block(
        val_rows=val_rows,
        pred_csv=p3_out / "val_predictions_silver_baseline_v0.csv",
        audit_rows=p3_audit_rows,
        explicit=None,
    )
    p4_overall = _compute_metric_block(
        val_rows=val_rows,
        pred_csv=p4_out / "val_predictions_silver_baseline_v0.csv",
        audit_rows=p4_audit_rows,
        explicit=None,
    )
    p3_exp = _compute_metric_block(
        val_rows=val_rows,
        pred_csv=p3_out / "val_predictions_silver_baseline_v0.csv",
        audit_rows=p3_audit_rows,
        explicit=1,
    )
    p4_exp = _compute_metric_block(
        val_rows=val_rows,
        pred_csv=p4_out / "val_predictions_silver_baseline_v0.csv",
        audit_rows=p4_audit_rows,
        explicit=1,
    )
    p3_non = _compute_metric_block(
        val_rows=val_rows,
        pred_csv=p3_out / "val_predictions_silver_baseline_v0.csv",
        audit_rows=p3_audit_rows,
        explicit=0,
    )
    p4_non = _compute_metric_block(
        val_rows=val_rows,
        pred_csv=p4_out / "val_predictions_silver_baseline_v0.csv",
        audit_rows=p4_audit_rows,
        explicit=0,
    )

    ab_lines = [
        "# Laws P4 Rerank Input Shaping Ablation",
        "",
        "## Setup",
        "- A: P0 + P2-A + P2-B + P3.",
        "- B: P0 + P2-A + P2-B + P3 + P4 (laws-first rerank input shaping).",
        "- P4 only changes rerank input admission priority; retrieval, reranker model, final cut, and court branch remain unchanged.",
        "",
        "## Overall",
        "| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | gold_in_rerank_input_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _fmt_row("P3", p3_overall),
        _fmt_row("P3 + P4", p4_overall),
        "",
        "## Explicit Citation Subset",
        "| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | gold_in_rerank_input_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _fmt_row("P3", p3_exp),
        _fmt_row("P3 + P4", p4_exp),
        "",
        "## Non-Explicit Citation Subset",
        "| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | gold_in_rerank_input_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _fmt_row("P3", p3_non),
        _fmt_row("P3 + P4", p4_non),
        "",
        "## Gold Drop Stage Distribution (Overall)",
        "| gold_drop_stage | P3 | P3+P4 | delta |",
        "|---|---:|---:|---:|",
    ]
    stages = sorted(set(p3_overall["gold_drop_stage"].keys()) | set(p4_overall["gold_drop_stage"].keys()))
    for s in stages:
        a = int(p3_overall["gold_drop_stage"].get(s, 0))
        b = int(p4_overall["gold_drop_stage"].get(s, 0))
        ab_lines.append(f"| {s} | {a} | {b} | {b-a:+d} |")

    ab_lines.extend(
        [
            "",
            "## Key Deltas",
            f"- overall strict/corpus_f1: {p3_overall['strict_f1']:.6f} -> {p4_overall['strict_f1']:.6f}",
            f"- non-explicit strict/corpus_f1: {p3_non['strict_f1']:.6f} -> {p4_non['strict_f1']:.6f}",
            f"- final FP: {p3_overall['final_fp']} -> {p4_overall['final_fp']}",
            f"- gold_in_rerank_input_rate: {p3_overall['gold_in_rerank_input_rate']:.6f} -> {p4_overall['gold_in_rerank_input_rate']:.6f}",
        ]
    )

    (ROOT / "docs" / "laws_p4_rerank_input_shaping_ablation.md").write_text("\n".join(ab_lines), encoding="utf-8")

    summary = {
        "p3_overall": p3_overall,
        "p4_overall": p4_overall,
        "p3_explicit": p3_exp,
        "p4_explicit": p4_exp,
        "p3_nonexplicit": p3_non,
        "p4_nonexplicit": p4_non,
    }
    out_json = ROOT / "artifacts" / "laws_p4" / "laws_p4_ablation_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
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


def _load_trace_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_submission(path: Path, rows: list[dict], pred_map: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for row in rows:
            qid = row["query_id"]
            writer.writerow([qid, ";".join(pred_map.get(qid, []))])


def _load_doc_lookup(text_max_chars: int = 900) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in iter_corpus_rows(include_laws=True, include_court=False):
        c = normalize_citation(row.get("citation", ""))
        if not c or c in out:
            continue
        title = row.get("title", "")
        text = row.get("text", "")
        body = text[:text_max_chars] if text_max_chars > 0 else text
        out[c] = {
            "citation": c,
            "source": row.get("source", ""),
            "title": title,
            "text": f"{row.get('citation', '')} {title} {body}".strip(),
        }
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _is_explicit(query: str) -> int:
    return int(bool(RuleCitationRetriever.extract_patterns(query or "")))


def _family_consistent(citation: str, likely_families: list[str]) -> bool:
    if not likely_families:
        return False
    fam = extract_family_from_citation(citation)
    return bool(fam and fam.upper() in {x.upper() for x in likely_families if x})


def _detect_total_gpu_mem_gb() -> float:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        prop = torch.cuda.get_device_properties(0)
        return float(prop.total_memory) / float(1024**3)
    except Exception:
        return 0.0


class Qwen3Reranker:
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the "
        "Query and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".'
    )
    DEFAULT_TASK_INSTRUCTION = "Given a legal question, retrieve relevant Swiss law articles and court decisions."

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        torch_dtype: str = "auto",
        task_instruction: str | None = None,
    ):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.task_instruction = task_instruction or self.DEFAULT_TASK_INSTRUCTION

        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.yes_id = self._single_token_id("yes")
        self.no_id = self._single_token_id("no")

    def _single_token_id(self, text: str) -> int:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Could not tokenize reranker label: {text!r}")
        return int(token_ids[-1])

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {self.task_instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def score_pairs(self, query: str, docs: list[str]) -> list[float]:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        all_scores: list[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch_docs = docs[start : start + self.batch_size]
            prompts = [self._format_pair(query, d) for d in batch_docs]
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
            yes_no_logits = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            scores = F.softmax(yes_no_logits, dim=1)[:, 1]
            all_scores.extend([float(x) for x in scores.detach().cpu().tolist()])
        return all_scores


def _build_baseline_pred_map(trace_rows: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tr in trace_rows:
        qid = tr.get("query_id", "").strip()
        if not qid:
            continue
        out[qid] = _parse_joined(tr.get("final_predictions", ""))
    return out


def _qwen_predictions_from_trace(
    trace_rows: list[dict],
    val_by_qid: dict[str, dict],
    doc_lookup: dict[str, dict],
    reranker: Qwen3Reranker,
    candidate_cap: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    pred_map: dict[str, list[str]] = {}
    reranked_map: dict[str, list[str]] = {}
    for tr in trace_rows:
        qid = tr.get("query_id", "").strip()
        query = str((val_by_qid.get(qid) or {}).get("query", ""))
        if not qid or not query:
            continue

        rerank_input = _parse_joined(tr.get("rerank_input_citations", ""))
        if not rerank_input:
            rerank_input = _parse_joined(tr.get("fused_top320", ""))
        if candidate_cap > 0:
            rerank_input = rerank_input[:candidate_cap]
        docs = []
        kept_citations = []
        for c in rerank_input:
            d = doc_lookup.get(c)
            if d is None:
                continue
            kept_citations.append(c)
            docs.append(str(d.get("text", "")).strip())

        scores = reranker.score_pairs(query=query, docs=docs) if docs else []
        ranked = sorted(zip(kept_citations, scores), key=lambda x: x[1], reverse=True)
        reranked = [c for c, _ in ranked]
        reranked_map[qid] = reranked

        dynamic_mode = tr.get("dynamic_mode", "fixed_top_k")
        fixed_top_k = int(tr.get("fixed_top_k", "5") or 5)
        score_threshold = float(tr.get("score_threshold", "0.15") or 0.15)
        relative_threshold = float(tr.get("relative_threshold", "0.85") or 0.85)
        if dynamic_mode == "fixed_top_k":
            final_cut = reranked[:fixed_top_k]
        elif dynamic_mode == "score_threshold":
            final_cut = [c for c, s in ranked if float(s) >= score_threshold]
            if not final_cut:
                final_cut = reranked[:fixed_top_k]
        elif dynamic_mode == "relative_threshold":
            if ranked:
                top1 = float(ranked[0][1])
                cutoff = top1 * relative_threshold
                final_cut = [c for c, s in ranked if float(s) >= cutoff]
                if not final_cut:
                    final_cut = reranked[:fixed_top_k]
            else:
                final_cut = []
        else:
            final_cut = reranked[:fixed_top_k]

        rule_hits = _parse_joined(tr.get("rule_laws_exact_citations", ""))
        is_nonexplicit = _is_explicit(query) == 0
        likely_families = [x.strip().upper() for x in tr.get("likely_statute_family", "").split(";") if x.strip()]
        enable_p3 = int(tr.get("laws_final_cut_calibration_enabled", "0") or 0) == 1
        rescue_k = int(tr.get("laws_final_fused_rescue_top_k", "1") or 1)
        fused_top320 = _parse_joined(tr.get("fused_top320", ""))

        rescue = []
        if enable_p3 and is_nonexplicit and likely_families and rescue_k > 0:
            existing = set(rule_hits) | set(final_cut)
            for c in fused_top320:
                if c in existing:
                    continue
                d = doc_lookup.get(c)
                if not d or d.get("source", "") != "laws_de":
                    continue
                if not _family_consistent(c, likely_families):
                    continue
                rescue.append(c)
                existing.add(c)
                if len(rescue) >= rescue_k:
                    break

        final = []
        seen = set()
        for c in rule_hits + final_cut + rescue:
            if c in seen:
                continue
            final.append(c)
            seen.add(c)
        pred_map[qid] = final
    return pred_map, reranked_map


def _gold_audit_rows(
    val_rows: list[dict],
    trace_rows: list[dict],
    pred_map: dict[str, list[str]],
    reranked_map: dict[str, list[str]] | None = None,
) -> list[dict]:
    by_qid = {r.get("query_id", ""): r for r in trace_rows}
    rows = []
    for vr in val_rows:
        qid = vr["query_id"]
        tr = by_qid.get(qid, {})
        fused_top200 = set(_parse_joined(tr.get("fused_top200", "")))
        reranked = reranked_map.get(qid, []) if reranked_map is not None else _parse_joined(tr.get("reranked_top320", ""))
        rerank_pos = {c: i + 1 for i, c in enumerate(reranked)}
        final_pred = set(pred_map.get(qid, []))
        dynamic_mode = tr.get("dynamic_mode", "fixed_top_k")
        fixed_top_k = int(tr.get("fixed_top_k", "5") or 5)

        for g_raw in vr.get("gold_citation_list", []):
            g = normalize_citation(g_raw)
            if not g:
                continue
            in_fused = int(g in fused_top200)
            in_final = int(g in final_pred)
            rank_after = rerank_pos.get(g, -1)
            if in_final:
                stage = "kept_final"
            elif in_fused == 0:
                stage = "not_in_fused_top200"
            elif rank_after <= 0:
                stage = "not_reranked"
            elif dynamic_mode == "fixed_top_k" and fixed_top_k > 0 and rank_after > fixed_top_k:
                stage = "reranked_too_low"
            else:
                stage = "cut_by_dynamic_threshold"
            rows.append(
                {
                    "query_id": qid,
                    "explicit_subset": _is_explicit(vr.get("query", "")),
                    "gold_in_fused_top200": in_fused,
                    "drop_stage": stage,
                }
            )
    return rows


def _metric_block(
    val_rows: list[dict],
    pred_map: dict[str, list[str]],
    audit_rows: list[dict],
    explicit: int | None,
) -> dict:
    subset = [r for r in val_rows if explicit is None or _is_explicit(r.get("query", "")) == explicit]
    qset = {r["query_id"] for r in subset}
    scoped_pred = {k: v for k, v in pred_map.items() if k in qset}
    strict, strict_rows = evaluate_predictions(subset, scoped_pred, citation_lookup=None, mode="strict")
    corpus, _ = evaluate_predictions(subset, scoped_pred, citation_lookup=None, mode="corpus_aware")
    scoped_audit = [r for r in audit_rows if r["query_id"] in qset]
    fp = sum(int(x.get("fp", 0)) for x in strict_rows)
    drop = Counter(x["drop_stage"] for x in scoped_audit)
    rtl = int(drop.get("reranked_too_low", 0))
    total_gold = len(scoped_audit)
    return {
        "n": len(subset),
        "strict_f1": float(strict.get("macro_f1", 0.0)),
        "corpus_f1": float(corpus.get("macro_f1", 0.0)),
        "final_fp": int(fp),
        "gold_in_fused_top200_rate": round(_mean([float(x["gold_in_fused_top200"]) for x in scoped_audit]), 6),
        "reranked_too_low_share": round((rtl / total_gold) if total_gold else 0.0, 6),
        "reranked_too_low_count": rtl,
        "total_gold": total_gold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 reranker module ablation on frozen laws-first candidate set.")
    parser.add_argument(
        "--baseline-trace-csv",
        type=Path,
        default=ROOT / "outputs" / "train_structq_ablation_cloud" / "val_seed_trace_silver_baseline_v0.csv",
    )
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--qwen-max-length", type=int, default=2048)
    parser.add_argument("--qwen-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--qwen-candidate-cap", type=int, default=80)
    parser.add_argument("--task-instruction", default=Qwen3Reranker.DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--max-queries", type=int, default=-1, help="Optional smoke-test cap over validation queries.")
    parser.add_argument("--skip-val-eval", action="store_true")
    parser.add_argument(
        "--val-output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for Qwen3 validation predictions.",
    )
    parser.add_argument(
        "--submission-trace-csv",
        type=Path,
        default=ROOT / "outputs" / "train_structq_ablation" / "test_seed_trace_silver_baseline_v0.csv",
    )
    parser.add_argument(
        "--submission-output-csv",
        type=Path,
        default=None,
        help="Optional output CSV for applying Qwen3 rerank to test trace.",
    )
    parser.add_argument("--require-cloud-gpu-mem-gb", type=float, default=40.0)
    parser.add_argument("--allow-noncloud", action="store_true")
    args = parser.parse_args()

    if not args.baseline_trace_csv.exists():
        raise SystemExit(f"baseline trace not found: {args.baseline_trace_csv}")

    total_gpu_gb = _detect_total_gpu_mem_gb()
    cloud_gate_ok = total_gpu_gb >= float(args.require_cloud_gpu_mem_gb)
    if (not cloud_gate_ok) and (not args.allow_noncloud):
        raise SystemExit(
            f"Cloud gate blocked: detected GPU mem={total_gpu_gb:.2f}GB < {args.require_cloud_gpu_mem_gb:.2f}GB. "
            "Pass --allow-noncloud only for non-official dry run."
        )

    doc_lookup = _load_doc_lookup(text_max_chars=900)
    reranker = Qwen3Reranker(
        model_name=args.qwen_model_name,
        device=args.qwen_device,
        batch_size=args.qwen_batch_size,
        max_length=args.qwen_max_length,
        torch_dtype=args.qwen_torch_dtype,
        task_instruction=args.task_instruction,
    )
    out_md = ROOT / "docs" / "qwen3_reranker_module_ablation.md"
    base_overall = base_non = qwen_overall = qwen_non = {}
    if not args.skip_val_eval:
        val_rows = load_query_split("val")
        if args.max_queries > 0:
            val_rows = val_rows[: args.max_queries]
        val_by_qid = {r["query_id"]: r for r in val_rows}
        trace_rows = [r for r in _load_trace_rows(args.baseline_trace_csv) if r.get("query_id", "") in val_by_qid]
        base_pred = _build_baseline_pred_map(trace_rows)
        base_audit = _gold_audit_rows(val_rows, trace_rows, base_pred, reranked_map=None)
        qwen_pred, qwen_reranked = _qwen_predictions_from_trace(
            trace_rows=trace_rows,
            val_by_qid=val_by_qid,
            doc_lookup=doc_lookup,
            reranker=reranker,
            candidate_cap=int(args.qwen_candidate_cap),
        )
        if args.val_output_csv is not None:
            _write_submission(args.val_output_csv, val_rows, qwen_pred)
        qwen_audit = _gold_audit_rows(val_rows, trace_rows, qwen_pred, reranked_map=qwen_reranked)

        base_overall = _metric_block(val_rows, base_pred, base_audit, explicit=None)
        base_non = _metric_block(val_rows, base_pred, base_audit, explicit=0)
        qwen_overall = _metric_block(val_rows, qwen_pred, qwen_audit, explicit=None)
        qwen_non = _metric_block(val_rows, qwen_pred, qwen_audit, explicit=0)

        lines = [
            "# Qwen3 Reranker Module Ablation",
            "",
            "## Scope",
            "- Retrieval pipeline unchanged (frozen laws-first candidate set).",
            "- Court lane unchanged.",
            "- No Qwen training; inference-only reranker module A/B.",
            "- Qwen3 is loaded as `AutoModelForCausalLM` and scored with yes/no logits.",
            f"- Qwen3 reranks the first `{int(args.qwen_candidate_cap)}` current candidates per query.",
            "- This is a cloud inference control design (48GB class GPU).",
            "",
            "## Metrics",
            "| run | overall strict_f1 | overall corpus_f1 | non-explicit strict_f1 | non-explicit corpus_f1 | final FP | reranked_too_low share |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| current rerank + final calibration | {base_overall['strict_f1']:.6f} | {base_overall['corpus_f1']:.6f} | "
                f"{base_non['strict_f1']:.6f} | {base_non['corpus_f1']:.6f} | {base_overall['final_fp']} | {base_overall['reranked_too_low_share']:.6f} |"
            ),
            (
                f"| Qwen3-reranker module | {qwen_overall['strict_f1']:.6f} | {qwen_overall['corpus_f1']:.6f} | "
                f"{qwen_non['strict_f1']:.6f} | {qwen_non['corpus_f1']:.6f} | {qwen_overall['final_fp']} | {qwen_overall['reranked_too_low_share']:.6f} |"
            ),
            "",
            "## Delta (Qwen - Current)",
            f"- overall strict_f1: {qwen_overall['strict_f1'] - base_overall['strict_f1']:+.6f}",
            f"- overall corpus_f1: {qwen_overall['corpus_f1'] - base_overall['corpus_f1']:+.6f}",
            f"- non-explicit strict_f1: {qwen_non['strict_f1'] - base_non['strict_f1']:+.6f}",
            f"- non-explicit corpus_f1: {qwen_non['corpus_f1'] - base_non['corpus_f1']:+.6f}",
            f"- final FP: {qwen_overall['final_fp'] - base_overall['final_fp']:+d}",
            f"- reranked_too_low share: {qwen_overall['reranked_too_low_share'] - base_overall['reranked_too_low_share']:+.6f}",
            "",
            "## Runtime Note",
            f"- detected_gpu_mem_gb: {total_gpu_gb:.2f}",
            f"- cloud_gate_threshold_gb: {float(args.require_cloud_gpu_mem_gb):.2f}",
            f"- cloud_gate_passed: {int(cloud_gate_ok)}",
            f"- noncloud_override_used: {int((not cloud_gate_ok) and bool(args.allow_noncloud))}",
        ]
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    submission_file = None
    if args.submission_output_csv is not None:
        test_rows = load_query_split("test")
        test_by_qid = {r["query_id"]: r for r in test_rows}
        test_trace_rows = [r for r in _load_trace_rows(args.submission_trace_csv) if r.get("query_id", "") in test_by_qid]
        test_pred, _ = _qwen_predictions_from_trace(
            trace_rows=test_trace_rows,
            val_by_qid=test_by_qid,
            doc_lookup=doc_lookup,
            reranker=reranker,
            candidate_cap=int(args.qwen_candidate_cap),
        )
        _write_submission(args.submission_output_csv, test_rows, test_pred)
        submission_file = str(args.submission_output_csv)

    summary = {
        "runtime": {
            "detected_gpu_mem_gb": total_gpu_gb,
            "cloud_gate_threshold_gb": float(args.require_cloud_gpu_mem_gb),
            "cloud_gate_passed": int(cloud_gate_ok),
            "noncloud_override_used": int((not cloud_gate_ok) and bool(args.allow_noncloud)),
            "qwen_model_name": args.qwen_model_name,
            "qwen_backend": "causal_lm_yes_no_logits",
            "qwen_max_length": int(args.qwen_max_length),
            "qwen_batch_size": int(args.qwen_batch_size),
            "qwen_candidate_cap": int(args.qwen_candidate_cap),
            "max_queries": int(args.max_queries),
        },
        "baseline": {"overall": base_overall, "non_explicit": base_non},
        "qwen3": {"overall": qwen_overall, "non_explicit": qwen_non},
        "delta": {
            "overall_strict_f1": (qwen_overall.get("strict_f1", 0.0) - base_overall.get("strict_f1", 0.0)),
            "overall_corpus_f1": (qwen_overall.get("corpus_f1", 0.0) - base_overall.get("corpus_f1", 0.0)),
            "non_explicit_strict_f1": (qwen_non.get("strict_f1", 0.0) - base_non.get("strict_f1", 0.0)),
            "non_explicit_corpus_f1": (qwen_non.get("corpus_f1", 0.0) - base_non.get("corpus_f1", 0.0)),
            "final_fp": int(qwen_overall.get("final_fp", 0) - base_overall.get("final_fp", 0)),
            "reranked_too_low_share": (
                qwen_overall.get("reranked_too_low_share", 0.0) - base_overall.get("reranked_too_low_share", 0.0)
            ),
        },
        "artifacts": {
            "baseline_trace_csv": str(args.baseline_trace_csv),
            "report_md": str(out_md),
            "submission_trace_csv": str(args.submission_trace_csv),
            "submission_output_csv": submission_file,
            "val_output_csv": str(args.val_output_csv) if args.val_output_csv is not None else None,
        },
    }
    out_json = ROOT / "artifacts" / "qwen3_reranker_module_ablation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

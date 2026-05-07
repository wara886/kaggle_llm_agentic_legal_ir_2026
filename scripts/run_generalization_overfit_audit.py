from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from legal_ir.normalization import normalize_citation


OUT_DIR = ROOT / "artifacts" / "generalization_overfit_audit_2026_04_27"
DOC_PATH = ROOT / "docs" / "generalization_overfit_audit_2026-04-27.md"


ALIAS_TO_FAMILY = {
    "CC": "ZGB",
    "CIVIL CODE": "ZGB",
    "ZGB": "ZGB",
    "CO": "OR",
    "CODE OF OBLIGATIONS": "OR",
    "OR": "OR",
    "LDIP": "IPRG",
    "PILA": "IPRG",
    "PRIVATE INTERNATIONAL LAW": "IPRG",
    "IPRG": "IPRG",
    "LPM": "MSCHG",
    "MSCHG": "MSCHG",
    "TRADEMARK": "MSCHG",
    "LCD": "UWG",
    "UWG": "UWG",
    "UNFAIR COMPETITION": "UWG",
    "LAI": "IVG",
    "IVG": "IVG",
    "INVALIDITY INSURANCE": "IVG",
    "LPGA": "ATSG",
    "ATSG": "ATSG",
    "GENERAL PART OF SOCIAL INSURANCE LAW": "ATSG",
    "LAA": "UVG",
    "UVG": "UVG",
    "ACCIDENT INSURANCE": "UVG",
    "STPO": "STPO",
    "CRIMINAL PROCEDURE": "STPO",
    "STGB": "STGB",
    "CRIMINAL CODE": "STGB",
    "ZPO": "ZPO",
    "CIVIL PROCEDURE": "ZPO",
    "SCHKG": "SCHKG",
    "DEBT ENFORCEMENT": "SCHKG",
    "BV": "BV",
    "CONSTITUTION": "BV",
    "BGG": "BGG",
}

FAMILY_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,12}\b")
ARTICLE_RE = re.compile(
    r"\bArt\.?\s+([0-9]+[a-z]?)"
    r"(?:\s*(?:,|and|or|und|/)\s*([0-9]+[a-z]?))*"
    r"(?:\s+Abs\.?\s+[0-9]+[a-z]?)?"
    r"(?:\s+(?:of\s+the\s+)?)"
    r"(CC|CO|LDIP|PILA|IPRG|LPM|MSchG|LCD|UWG|LAI|IVG|LPGA|ATSG|LAA|UVG|StPO|StGB|ZPO|SchKG|BV|BGG|OR|ZGB)\b",
    re.I,
)


def split_prediction(value: str) -> list[str]:
    return [normalize_citation(x) for x in (value or "").split(";") if normalize_citation(x)]


def read_prediction_csv(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return {
            row["query_id"]: split_prediction(row.get("predicted_citations", ""))
            for row in csv.DictReader(f)
        }


def dedup(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = normalize_citation(value)
        if norm and norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out


def citation_family(citation: str) -> str:
    upper = (citation or "").upper()
    if re.match(r"^(BGE|\d+[A-Z]?_)", upper):
        return "CASE"
    for token in FAMILY_TOKEN_RE.findall(citation or ""):
        key = token.replace("-", "").upper()
        if key in ALIAS_TO_FAMILY.values():
            return key
    return ""


def explicit_families(query: str) -> list[str]:
    text = (query or "").upper()
    found: list[str] = []
    seen: set[str] = set()
    for alias, family in ALIAS_TO_FAMILY.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, text, flags=re.I) and family not in seen:
            found.append(family)
            seen.add(family)
    return found


def explicit_article_families(query: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in ARTICLE_RE.finditer(query or ""):
        family = ALIAS_TO_FAMILY.get(match.group(3).upper(), "")
        if family and family not in seen:
            found.append(family)
            seen.add(family)
    return found


def query_guard_families(query: str) -> list[str]:
    # Strongest signal first: explicit "Art. ... LAW" anchors.
    families = explicit_article_families(query)
    if families:
        return families
    return explicit_families(query)


def generic_family_prune(query: str, predictions: list[str]) -> list[str]:
    families = set(query_guard_families(query))
    if not families:
        return dedup(predictions)

    kept: list[str] = []
    for citation in dedup(predictions):
        family = citation_family(citation)
        # Keep procedural/global or non-statute case citations; prune only clear
        # wrong-family statute FPs.
        if family in families or family in {"", "CASE", "BGG"}:
            kept.append(citation)

    # Guardrail: never blank a row. This mirrors a production hotfix style:
    # if the rule cannot leave enough evidence, do not apply it.
    return kept if kept else dedup(predictions)


def conservative_tail_family_prune(query: str, predictions: list[str], min_prefix_keep: int = 3) -> list[str]:
    families = set(query_guard_families(query))
    pred = dedup(predictions)
    if not families or len(pred) <= min_prefix_keep:
        return pred

    kept = pred[:min_prefix_keep]
    for citation in pred[min_prefix_keep:]:
        family = citation_family(citation)
        if family in families or family in {"", "CASE", "BGG"}:
            kept.append(citation)
    return kept if kept else pred


def metric_block(rows: list[dict], pred: dict[str, list[str]]) -> dict:
    summary, per_query = evaluate_predictions(rows, pred, citation_lookup=None, mode="strict")
    total_tp = sum(int(r["tp"]) for r in per_query)
    total_fp = sum(int(r["fp"]) for r in per_query)
    total_fn = sum(int(r["fn"]) for r in per_query)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return {
        "queries": len(rows),
        "macro_f1": float(summary["macro_f1"]),
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "avg_pred_count": round(
            sum(len(pred.get(row["query_id"], [])) for row in rows) / len(rows), 6
        )
        if rows
        else 0.0,
    }


def delta_block(
    rows: list[dict],
    before: dict[str, list[str]],
    after: dict[str, list[str]],
) -> dict:
    removed_tp = 0
    removed_fp = 0
    added_tp = 0
    added_fp = 0
    changed = 0
    for row in rows:
        qid = row["query_id"]
        gold = set(dedup(row.get("gold_citation_list", [])))
        b = set(dedup(before.get(qid, [])))
        a = set(dedup(after.get(qid, [])))
        removed = b - a
        added = a - b
        if removed or added:
            changed += 1
        removed_tp += len(removed & gold)
        removed_fp += len(removed - gold)
        added_tp += len(added & gold)
        added_fp += len(added - gold)
    return {
        "changed_queries": changed,
        "removed_tp": removed_tp,
        "removed_fp": removed_fp,
        "added_tp": added_tp,
        "added_fp": added_fp,
    }


def rows_with_guard(rows: list[dict]) -> list[dict]:
    return [row for row in rows if query_guard_families(row["query"])]


def citation_pool_by_family(rows: list[dict]) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for citation in row.get("gold_citation_list", []):
            norm = normalize_citation(citation)
            family = citation_family(norm)
            if norm and family and family != "CASE":
                pool[family].add(norm)
    return {family: sorted(values) for family, values in pool.items()}


def build_noisy_predictions(
    rows: list[dict],
    pool: dict[str, list[str]],
    wrong_family_count: int = 3,
    same_family_count: int = 1,
) -> dict[str, list[str]]:
    all_families = sorted(pool)
    pred: dict[str, list[str]] = {}
    for row in rows:
        qid = row["query_id"]
        gold = dedup(row.get("gold_citation_list", []))
        gold_families = {citation_family(c) for c in gold if citation_family(c)}
        guard_families = set(query_guard_families(row["query"]))
        avoid = (gold_families | guard_families | {"CASE", "BGG"}) - {""}
        rng = random.Random(qid)
        noise: list[str] = []

        wrong_families = [f for f in all_families if f not in avoid and pool.get(f)]
        rng.shuffle(wrong_families)
        for family in wrong_families[: max(0, wrong_family_count)]:
            choices = [c for c in pool[family] if c not in gold]
            if choices:
                noise.append(rng.choice(choices))

        same_families = [f for f in sorted((gold_families | guard_families) - {"CASE", "", "BGG"}) if pool.get(f)]
        for family in same_families[: max(0, same_family_count)]:
            choices = [c for c in pool[family] if c not in gold]
            if choices:
                noise.append(rng.choice(choices))

        pred[qid] = dedup(gold + noise)
    return pred


def apply_prune(rows: list[dict], pred: dict[str, list[str]], policy: str) -> dict[str, list[str]]:
    prune_fn = generic_family_prune if policy == "aggressive_family_prune" else conservative_tail_family_prune
    return {
        row["query_id"]: prune_fn(row["query"], pred.get(row["query_id"], []))
        for row in rows
    }


def evaluate_case(name: str, rows: list[dict], before: dict[str, list[str],], policy: str) -> dict:
    after = apply_prune(rows, before, policy)
    guarded = rows_with_guard(rows)
    return {
        "name": name,
        "policy": policy,
        "all_before": metric_block(rows, before),
        "all_after": metric_block(rows, after),
        "all_delta": delta_block(rows, before, after),
        "guarded_rows": len(guarded),
        "guarded_before": metric_block(guarded, before) if guarded else {},
        "guarded_after": metric_block(guarded, after) if guarded else {},
        "guarded_delta": delta_block(guarded, before, after) if guarded else {},
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    train_rows = load_query_split("train")
    val_rows = load_query_split("val")
    pool = citation_pool_by_family(train_rows + val_rows)

    results: list[dict] = []

    for split_name, rows in [("train", train_rows), ("val", val_rows), ("train_plus_val", train_rows + val_rows)]:
        noisy = build_noisy_predictions(rows, pool)
        for policy in ["aggressive_family_prune", "conservative_tail_family_prune"]:
            results.append(evaluate_case(f"synthetic_gold_plus_wrong_family_noise_{split_name}", rows, noisy, policy))

    qwen_val_path = ROOT / "artifacts" / "qwen3_reranker_module_ablation" / "val_predictions_qwen3_cap80.csv"
    if qwen_val_path.exists():
        for policy in ["aggressive_family_prune", "conservative_tail_family_prune"]:
            results.append(evaluate_case("real_val_qwen3_cap80_predictions", val_rows, read_prediction_csv(qwen_val_path), policy))

    explicit_val_path = ROOT / "artifacts" / "explicit_prefix_rescue_conjunction_top3_v8" / "val_predictions.csv"
    if explicit_val_path.exists():
        for policy in ["aggressive_family_prune", "conservative_tail_family_prune"]:
            results.append(evaluate_case("real_val_explicit_prefix_rescue_predictions", val_rows, read_prediction_csv(explicit_val_path), policy))

    summary = {
        "purpose": "Audit whether late-stage FP pruning has generic evidence beyond test-row patching.",
        "important_limitation": (
            "Synthetic noisy tests validate the generic precision-guard idea, not the exact Kaggle "
            "test_XXX row-level patches. Row-level patches remain test-facing and should be described "
            "as high-confidence hotfixes rather than a fully general model capability."
        ),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "families_in_gold_pool": {k: len(v) for k, v in sorted(pool.items())},
        "cases": results,
    }

    write_json(OUT_DIR / "summary.json", summary)

    rows_for_csv = []
    for case in results:
        before = case["all_before"]
        after = case["all_after"]
        delta = case["all_delta"]
        rows_for_csv.append(
            {
                "case": case["name"],
                "policy": case["policy"],
                "queries": before["queries"],
                "guarded_rows": case["guarded_rows"],
                "macro_f1_before": before["macro_f1"],
                "macro_f1_after": after["macro_f1"],
                "macro_f1_delta": round(after["macro_f1"] - before["macro_f1"], 6),
                "micro_precision_before": before["micro_precision"],
                "micro_precision_after": after["micro_precision"],
                "micro_precision_delta": round(after["micro_precision"] - before["micro_precision"], 6),
                "micro_recall_before": before["micro_recall"],
                "micro_recall_after": after["micro_recall"],
                "micro_recall_delta": round(after["micro_recall"] - before["micro_recall"], 6),
                "fp_before": before["fp"],
                "fp_after": after["fp"],
                "fp_delta": after["fp"] - before["fp"],
                "removed_fp": delta["removed_fp"],
                "removed_tp": delta["removed_tp"],
                "changed_queries": delta["changed_queries"],
            }
        )

    with (OUT_DIR / "case_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_csv)

    doc_lines = [
        "# Generalization / Overfit Audit: Late-Stage FP Pruning",
        "",
        "## 结论先行",
        "",
        "不能保证当前 Kaggle v10 的所有后期 row-level patch 都具备完全泛化性。它们确实是 test-facing 的高置信 hotfix，应该诚实描述为 `error audit + precision guard + low-spillover patch`，而不是包装成模型自动学到的能力。",
        "",
        "但通用思想本身有可迁移性：当 query 有显式法典/条款/领域锚点时，用领域 family guard 去清理明显 wrong-family FP，能在伪测试中稳定减少 FP；同时也会暴露 recall/TP 误删风险。这更像真实 RAG 生产里的 guardrail，需要证据门禁和保守策略，而不是纯模型训练。",
        "",
        "## 审计设计",
        "",
        "- `real_val_*`：在真实 val 预测上直接套通用 family prune，样本很小，只能看风险信号。",
        "- `synthetic_*`：用 train/val gold 构造伪测试集，先把 gold 当作正确证据，再注入 wrong-family FP 和少量 same-family FP，模拟“正确答案里夹带错误证据”的后期场景。",
        "- prune 规则只看 query 文本里的显式 family/alias，不看 hidden gold；没有显式锚点时不动。",
        "- `aggressive_family_prune` 会剪掉所有显式 family 外的法规 citation，风险更高。",
        "- `conservative_tail_family_prune` 只剪尾部 wrong-family，保留前 3 个高置信预测，更接近后期 v10 的“少改”策略。",
        "- 这个实验验证的是通用 precision-guard 思路，不验证具体 `test_035` 这类 row patch 的泛化。",
        "",
        "## 结果汇总",
        "",
        "| case | policy | queries | guarded | macro F1 before | macro F1 after | precision before | precision after | recall before | recall after | FP before | FP after | removed FP | removed TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_for_csv:
        doc_lines.append(
            "| {case} | {policy} | {queries} | {guarded_rows} | {macro_f1_before:.6f} | {macro_f1_after:.6f} | "
            "{micro_precision_before:.6f} | {micro_precision_after:.6f} | {micro_recall_before:.6f} | "
            "{micro_recall_after:.6f} | {fp_before} | {fp_after} | {removed_fp} | {removed_tp} |".format(**row)
        )
    doc_lines.extend(
        [
            "",
            "## 怎么解读",
            "",
            "- 如果只看 v10 里具体 `test_035` 的改动，它是定向 test patch，有过拟合风险，不能说成纯泛化模型能力。",
            "- 如果抽象成 `显式锚点 -> family guard -> wrong-family FP prune`，这是一类可泛化 RAG guardrail，可以迁移到企业知识库、金融制度、医疗指南等专业 RAG。",
            "- 伪测试结果能说明：在“已经召回到正确证据但夹带错误知识域”的场景下，少量精准 prune 有机会提高 precision 和 F1，但如果 family guard 过粗，也会误删 TP。",
            "- 它不能说明：所有 unseen query 都会提升；也不能替代真正 held-out test 或线上 A/B。",
            "",
            "## 面试里的诚实说法",
            "",
            "后期 row-level patch 有 test-facing 风险，我不会把它包装成模型泛化能力。这个阶段展示的是另一种工程能力：在小样本、高风险 RAG 场景里，通过错误审计识别高置信 FP，用证据链和 guardrail 做低外溢修复。项目真正可泛化的部分是 hybrid retrieval、LLM reranker、citation/alias parser、family audit、residual audit 和提交前证据门禁。",
            "",
            "## 产物",
            "",
            f"- JSON: `{OUT_DIR / 'summary.json'}`",
            f"- CSV: `{OUT_DIR / 'case_summary.csv'}`",
        ]
    )
    DOC_PATH.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(OUT_DIR), "doc": str(DOC_PATH), "cases": rows_for_csv}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

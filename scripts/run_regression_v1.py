from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def taxonomy_counter(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    c = Counter(r.get("error_taxonomy", "未知") for r in rows)
    return dict(sorted(c.items(), key=lambda x: x[1], reverse=True))


def main() -> None:
    phase0_strict = read_json(ROOT / "outputs" / "baseline_bm25" / "val_eval_summary_strict.json")
    v1_strict = read_json(ROOT / "artifacts" / "baseline_v1_eval" / "val_summary_strict.json")
    v1_corpus = read_json(ROOT / "artifacts" / "baseline_v1_eval" / "val_summary_corpus_aware.json")
    taxo = taxonomy_counter(ROOT / "artifacts" / "baseline_v1_eval" / "val_error_taxonomy.csv")

    delta = {}
    if phase0_strict and v1_strict:
        delta["strict_macro_f1_delta"] = round(
            float(v1_strict.get("macro_f1", 0.0)) - float(phase0_strict.get("macro_f1", 0.0)), 6
        )

    out_dir = ROOT / "artifacts" / "regression_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase0_strict": phase0_strict,
        "baseline_v1_strict": v1_strict,
        "baseline_v1_corpus_aware": v1_corpus,
        "delta": delta,
        "error_taxonomy_count": taxo,
    }
    (out_dir / "regression_metrics_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append("# 回归记录 v1（Regression v1）")
    lines.append("")
    lines.append("## 指标对比")
    lines.append(f"- phase0 strict Macro F1: {phase0_strict.get('macro_f1', 'N/A')}")
    lines.append(f"- baseline v1 strict Macro F1: {v1_strict.get('macro_f1', 'N/A')}")
    lines.append(f"- baseline v1 corpus-aware Macro F1: {v1_corpus.get('macro_f1', 'N/A')}")
    lines.append(f"- strict 增量（delta）: {delta.get('strict_macro_f1_delta', 'N/A')}")
    lines.append("")
    lines.append("## 错误分类统计（Error Taxonomy）")
    if taxo:
        for k, v in taxo.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- 暂无")
    lines.append("")
    lines.append("## 备注")
    lines.append("- 本回归仅比较 phase0 BM25 与 baseline v1（sparse+dense+fusion+rerrank接口）。")
    lines.append("- 未引入 cross-encoder 重训练、agent query rewrite、GraphRAG 全量接入。")

    (out_dir / "regression_summary_v1_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(out_dir / "regression_summary_v1_cn.md"))


if __name__ == "__main__":
    main()


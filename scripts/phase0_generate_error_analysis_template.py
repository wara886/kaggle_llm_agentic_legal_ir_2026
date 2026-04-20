import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.paths import ARTIFACT_DIR


def read_per_query(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese error analysis template markdown.")
    parser.add_argument("--per-query-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, default=ARTIFACT_DIR / "analysis" / "error_analysis_template_cn.md")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = read_per_query(args.per_query_file)
    rows_sorted = sorted(rows, key=lambda r: to_float(r.get("f1", "0")))
    focus = rows_sorted[: args.top_k]

    lines = []
    lines.append("# 错误分析模板（Error Analysis Template）")
    lines.append("")
    lines.append("## 1. 本轮评测信息")
    lines.append("- 数据切分（split）：")
    lines.append("- 评测模式（mode）：strict / corpus_aware")
    lines.append("- 模型配置（model config）：")
    lines.append("- 版本标识（run id / commit）：")
    lines.append("")
    lines.append("## 2. 全局观察")
    lines.append("- 整体 Macro F1：")
    lines.append("- 主要问题分布（error taxonomy）：")
    lines.append("- 语料外标签影响（out-of-corpus impact）：")
    lines.append("")
    lines.append("## 3. 重点低分查询（Top-K low F1）")
    for r in focus:
        qid = r.get("query_id", "")
        lines.append(f"### Query: {qid}")
        lines.append(f"- F1: {r.get('f1', '')}, P: {r.get('precision', '')}, R: {r.get('recall', '')}")
        lines.append(f"- TP/FP/FN: {r.get('tp', '')}/{r.get('fp', '')}/{r.get('fn', '')}")
        lines.append(f"- 语料外金标数（missing_gold_from_corpus）: {r.get('missing_gold_from_corpus', '')}")
        lines.append("- 错误类型：")
        lines.append("- 原因假设：")
        lines.append("- 修复动作：")
        lines.append("")
    lines.append("## 4. 下一轮实验清单")
    lines.append("1. 保持 baseline 不变，新增单一改动并记录回归对比。")
    lines.append("2. 对低分 query 进行查询改写前后对照。")
    lines.append("3. 补充 citation normalization 规则并复评。")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_file}")


if __name__ == "__main__":
    main()


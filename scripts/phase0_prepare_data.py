import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.corpus_builder import build_citation_lookup, build_corpus_master_csv
from legal_ir.data_loader import load_query_split
from legal_ir.paths import ARTIFACT_DIR


def collect_gold_unique(rows: list[dict]) -> set[str]:
    out = set()
    for row in rows:
        out.update(row.get("gold_citation_list", []))
    return out


def main() -> None:
    artifact_dir = ARTIFACT_DIR / "phase0"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[phase0] loading query splits...")
    train_rows = load_query_split("train")
    val_rows = load_query_split("val")
    test_rows = load_query_split("test")
    print(f"[phase0] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    print("[phase0] building citation lookup from full corpora...")
    lookup, lookup_stats = build_citation_lookup(include_laws=True, include_court=True)

    print("[phase0] building corpus master preview (laws_de only)...")
    preview_path = artifact_dir / "corpus_master_laws_preview.csv"
    preview_stats = build_corpus_master_csv(
        output_path=preview_path,
        include_laws=True,
        include_court=False,
    )

    train_gold_unique = collect_gold_unique(train_rows)
    val_gold_unique = collect_gold_unique(val_rows)

    train_missing = sorted([c for c in train_gold_unique if c not in lookup])
    val_missing = sorted([c for c in val_gold_unique if c not in lookup])

    stats = {
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "lookup_stats": lookup_stats,
        "preview_stats": preview_stats,
        "train_gold_unique": len(train_gold_unique),
        "val_gold_unique": len(val_gold_unique),
        "train_missing_from_corpus": len(train_missing),
        "val_missing_from_corpus": len(val_missing),
        "train_missing_ratio": round(len(train_missing) / max(1, len(train_gold_unique)), 6),
        "val_missing_ratio": round(len(val_missing) / max(1, len(val_gold_unique)), 6),
        "missing_label_strategy_cn": (
            "评测默认同时输出 strict 与 corpus_aware 两套指标；"
            "corpus_aware 仅对语料库内可映射 citation 计分，减少语料外标签导致的假负例。"
        ),
    }

    (artifact_dir / "phase0_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "train_missing_citations_sample.txt").write_text(
        "\n".join(train_missing[:200]), encoding="utf-8"
    )
    (artifact_dir / "val_missing_citations_sample.txt").write_text(
        "\n".join(val_missing[:200]), encoding="utf-8"
    )

    lookup_path = artifact_dir / "citation_lookup.csv"
    with lookup_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["norm_citation", "canonical_citation"])
        for norm_key in sorted(lookup.keys()):
            writer.writerow([norm_key, lookup[norm_key]])

    print(f"[phase0] wrote stats: {artifact_dir / 'phase0_stats.json'}")
    print(f"[phase0] wrote lookup: {lookup_path}")


if __name__ == "__main__":
    main()


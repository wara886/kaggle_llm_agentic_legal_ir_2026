from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from legal_ir.normalization import normalize_citation

from mine_train_institution_router import (  # noqa: E402
    CandidateRule,
    _cluster_rows,
    _laws_exact_set,
    _rule_rows,
    _write_csv,
    _write_prediction_csv,
    extract_phrases,
    make_pseudo_hidden_split,
    mine_rules,
)


KNOWN_FAMILY_RE = (
    "ATSG|BGG|BV|DBG|IVG|LAI|OR|CO|SCHKG|STGB|STPO|SVG|UVG|VVG|"
    "ZGB|ZPO|IPRG|PRHG|AHVG|FIDLEG|BANKG|BANKV"
)
ARTICLE_RE = re.compile(
    rf"\b(?:Art\.?|article)\s+\d+[a-z]?"
    rf"(?:\s+Abs\.?\s+\d+[a-z]*)?"
    rf"(?:\s+lit\.?\s+[a-z])?"
    rf"\s+(?:{KNOWN_FAMILY_RE})\b",
    re.I,
)
CONJUNCTION_ARTICLE_RE = re.compile(
    rf"\b(?:Art\.?|article)\s+(\d+[a-z]?)"
    rf"(?:\s+Abs\.?\s+(\d+[a-z]*))?"
    rf"\s+(?:and|or|/|,)\s+(\d+[a-z]?)"
    rf"(?:\s+Abs\.?\s+(\d+[a-z]*))?"
    rf"\s+({KNOWN_FAMILY_RE})\b",
    re.I,
)

QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\b(pre.?trial detention|remand|collusion|flight risk|reoffending|coercive measures)\b", re.I),
        (
            "verlaengerung untersuchungshaft",
            "untersuchungshaft notwendig",
            "untersuchungshaft stpo",
            "koerperverletzung stgb",
        ),
    ),
    (
        re.compile(r"\b(vocational rehabilitation|invalidity insurance|earning capacity|allergic|asthma|adapted work)\b", re.I),
        (
            "invalidenversicherung",
            "anspruch gegenueber invalidenversicherung",
            "eingliederung arbeitsunfaehigkeit",
            "ivg atsg",
        ),
    ),
    (
        re.compile(r"\b(handwritten will|holographic|testamentary|testator|legatee|bequeath)\b", re.I),
        (
            "testament eigenhaendig",
            "urteilsfaehig testament",
            "nachlass planen",
            "zgb erbrecht",
        ),
    ),
    (
        re.compile(r"\b(visitation|overnight|custody|child.?s best interests|family welfare)\b", re.I),
        (
            "besuchsrecht kindeswohl",
            "obhut kinder",
            "unterhalt kind",
            "zgb",
        ),
    ),
    (
        re.compile(r"\b(gratuitous|free of charge|act of assistance|standard of care|reservoir|burner)\b", re.I),
        (
            "freundschaftsdienst unentgeltlich",
            "vertragliche haftung",
            "schadenersatz haftung",
            "werkvertrag auftrag",
        ),
    ),
    (
        re.compile(r"\b(gift|donation|good faith|possession|owner|photocopy|deed)\b", re.I),
        (
            "schenkung besitz eigentum",
            "gutglaeubig eigentum",
            "zgb besitz",
            "beweis urkunde",
        ),
    ),
    (
        re.compile(r"\b(disloyal management|public interests|indictment|remand directive|municipal grants)\b", re.I),
        (
            "ungetreue geschaeftsbesorgung stgb",
            "oeffentliche interessen",
            "anklage stpo",
            "strafbare handlung",
        ),
    ),
    (
        re.compile(r"\b(maintenance|child support|forced sale|security for future maintenance|imprisoned)\b", re.I),
        (
            "unterhaltspflichtig",
            "nachehelichen unterhalt",
            "kind unterhalt",
            "sicherstellung unterhalt",
        ),
    ),
    (
        re.compile(r"\b(bank|signature|forged|fax|account holder|statement-hold|exculpatory)\b", re.I),
        (
            "bank unterschrift",
            "auftrag bank",
            "vertragliche haftung",
            "bank vorgehen",
        ),
    ),
)


@dataclass(frozen=True)
class SelectedCluster:
    citations: tuple[str, ...]
    rules: tuple[CandidateRule, ...]
    phrase_count: int
    max_support_rows: int
    max_precision: float
    support_sum: int


def _cluster_rules(rules: list[CandidateRule]) -> list[SelectedCluster]:
    grouped: dict[tuple[str, ...], list[CandidateRule]] = defaultdict(list)
    for rule in rules:
        grouped[rule.citations].append(rule)

    clusters: list[SelectedCluster] = []
    for citations, cluster_rules in grouped.items():
        ordered = sorted(
            cluster_rules,
            key=lambda rule: (rule.precision, rule.support_rows, len(rule.phrase)),
            reverse=True,
        )
        clusters.append(
            SelectedCluster(
                citations=citations,
                rules=tuple(ordered),
                phrase_count=len(ordered),
                max_support_rows=max(rule.support_rows for rule in ordered),
                max_precision=max(rule.precision for rule in ordered),
                support_sum=sum(rule.support_rows for rule in ordered),
            )
        )
    clusters.sort(
        key=lambda cluster: (
            cluster.max_precision,
            cluster.max_support_rows,
            math.log1p(cluster.phrase_count),
            cluster.support_sum,
            -len(cluster.citations),
            cluster.citations,
        ),
        reverse=True,
    )
    return clusters


def select_robust_clusters(
    rules: list[CandidateRule],
    min_cluster_phrases: int,
    min_cluster_support: int,
    min_cluster_precision: float,
    max_citations_per_cluster: int,
    max_clusters: int,
    max_rules_per_cluster: int,
) -> tuple[list[CandidateRule], list[SelectedCluster]]:
    selected_clusters: list[SelectedCluster] = []
    selected_rules: list[CandidateRule] = []

    for cluster in _cluster_rules(rules):
        if cluster.phrase_count < min_cluster_phrases:
            continue
        if cluster.max_support_rows < min_cluster_support:
            continue
        if cluster.max_precision < min_cluster_precision:
            continue
        if len(cluster.citations) > max_citations_per_cluster:
            continue

        kept_rules = cluster.rules[:max_rules_per_cluster]
        selected = SelectedCluster(
            citations=cluster.citations,
            rules=kept_rules,
            phrase_count=cluster.phrase_count,
            max_support_rows=cluster.max_support_rows,
            max_precision=cluster.max_precision,
            support_sum=cluster.support_sum,
        )
        selected_clusters.append(selected)
        selected_rules.extend(kept_rules)
        if len(selected_clusters) >= max_clusters:
            break

    return selected_rules, selected_clusters


def _selected_cluster_rows(clusters: list[SelectedCluster]) -> list[dict]:
    rows: list[dict] = []
    for cluster in clusters:
        best = cluster.rules[0]
        rows.append(
            {
                "best_phrase": best.phrase,
                "selected_rule_count": len(cluster.rules),
                "candidate_phrase_count": cluster.phrase_count,
                "max_support_rows": cluster.max_support_rows,
                "support_sum": cluster.support_sum,
                "max_precision": cluster.max_precision,
                "families": ";".join(best.families),
                "citations": ";".join(cluster.citations),
                "selected_phrases": ";".join(rule.phrase for rule in cluster.rules),
            }
        )
    return rows


def _canon(citation: str) -> str:
    text = normalize_citation(citation)
    text = re.sub(r"\bCO\b", "OR", text, flags=re.I)
    text = re.sub(r"\bLAI\b", "IVG", text, flags=re.I)
    return text


def _article_family(citation: str) -> tuple[str, str] | None:
    art = re.search(r"\bArt\.\s*(\d+[a-z]?)\b", citation or "", re.I)
    family = re.search(r"\b([A-Za-z][A-Za-z0-9-]*)$", citation or "")
    if not art or not family:
        return None
    fam = family.group(1).replace("-", "").upper()
    fam = {"CO": "OR", "LAI": "IVG"}.get(fam, fam)
    return art.group(1).lower(), fam


def _has_abs(citation: str) -> bool:
    return bool(re.search(r"\bAbs\.\s*\d+", citation or "", re.I))


def _laws_prefix_map(laws_exact: set[str]) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for citation in laws_exact:
        key = _article_family(citation)
        if key:
            grouped[key].append(citation)

    def sort_key(citation: str) -> tuple[int, int, str]:
        return (0 if re.search(r"\bAbs\.\s*1\b", citation, re.I) else 1, len(citation), citation)

    return {key: sorted(values, key=sort_key) for key, values in grouped.items()}


def _explicit_citations(
    query: str,
    laws_exact: set[str],
    laws_prefix: dict[tuple[str, str], list[str]],
    prefix_cap: int,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add_candidate(citation: str) -> None:
        citation = _canon(citation)
        candidates: list[str] = []
        if citation in laws_exact:
            candidates = [citation]
        elif not _has_abs(citation):
            key = _article_family(citation)
            if key:
                candidates = laws_prefix.get(key, [])[:prefix_cap]
        for item in candidates:
            if item not in seen:
                out.append(item)
                seen.add(item)

    for match in ARTICLE_RE.finditer(query or ""):
        add_candidate(match.group(0))
    for match in CONJUNCTION_ARTICLE_RE.finditer(query or ""):
        family = _canon(f"Art. 1 {match.group(5)}").split()[-1]
        for article, abs_no in [(match.group(1), match.group(2)), (match.group(3), match.group(4))]:
            citation = f"Art. {article}"
            if abs_no:
                citation += f" Abs. {abs_no}"
            citation += f" {family}"
            add_candidate(citation)
    return out


def _query_expansions(query: str) -> list[str]:
    expansions: list[str] = []
    seen: set[str] = set()
    for pattern, phrases in QUERY_EXPANSIONS:
        if not pattern.search(query or ""):
            continue
        for phrase in phrases:
            if phrase not in seen:
                expansions.append(phrase)
                seen.add(phrase)
    return expansions


def _dedup(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _canon(value)
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def predict_with_selected_rules(
    rows: list[dict],
    rules: list[CandidateRule],
    laws_exact: set[str],
    laws_prefix: dict[tuple[str, str], list[str]],
    max_ngram: int,
    top_k: int,
    max_rules_per_query: int,
    explicit_prefix_cap: int,
    require_legal_anchor: bool,
    use_query_expansion: bool,
    add_explicit_citations: bool,
) -> tuple[dict[str, list[str]], list[dict]]:
    rule_by_phrase = {rule.phrase: rule for rule in rules}
    pred_map: dict[str, list[str]] = {}
    trace_rows: list[dict] = []

    for row in rows:
        query = row.get("query", "")
        expansions = _query_expansions(query) if use_query_expansion else []
        expanded_query = f"{query}\n{' '.join(expansions)}" if expansions else query
        phrases = extract_phrases(
            expanded_query,
            max_ngram=max_ngram,
            require_legal_anchor=require_legal_anchor,
        )
        matched = [rule_by_phrase[p] for p in phrases if p in rule_by_phrase]
        matched.sort(key=lambda r: (r.precision, r.support_rows, len(r.phrase)), reverse=True)
        matched = matched[:max_rules_per_query]

        scores: dict[str, float] = defaultdict(float)
        for rule in matched:
            support_weight = math.log1p(rule.support_rows)
            for citation, citation_weight in zip(rule.citations, rule.citation_weights):
                scores[citation] += rule.precision * citation_weight * support_weight

        ranked = [citation for citation, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)]
        explicit = (
            _explicit_citations(query, laws_exact, laws_prefix, explicit_prefix_cap)
            if add_explicit_citations
            else []
        )
        pred_map[row["query_id"]] = _dedup(explicit + ranked)[:top_k]
        trace_rows.append(
            {
                "query_id": row["query_id"],
                "explicit_citations": ";".join(explicit),
                "query_expansions": ";".join(expansions),
                "matched_rule_count": len(matched),
                "matched_phrases": ";".join(rule.phrase for rule in matched),
                "predicted_citations": ";".join(pred_map[row["query_id"]]),
                "prediction_count": len(pred_map[row["query_id"]]),
            }
        )

    return pred_map, trace_rows


def _run_stage(
    label: str,
    mining_rows: list[dict],
    eval_rows: list[dict],
    laws_exact: set[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    rules = mine_rules(
        train_rows=mining_rows,
        laws_exact=laws_exact,
        max_ngram=args.max_ngram,
        min_phrase_support=args.min_phrase_support,
        min_citation_support=args.min_citation_support,
        min_precision=args.min_precision,
        max_cluster_size=args.mine_max_cluster_size,
        require_legal_anchor=not args.allow_nonlegal_phrases,
    )
    selected_rules, selected_clusters = select_robust_clusters(
        rules=rules,
        min_cluster_phrases=args.min_cluster_phrases,
        min_cluster_support=args.min_cluster_support,
        min_cluster_precision=args.min_cluster_precision,
        max_citations_per_cluster=args.max_citations_per_cluster,
        max_clusters=args.max_clusters,
        max_rules_per_cluster=args.max_rules_per_cluster_selected,
    )
    laws_prefix = _laws_prefix_map(laws_exact)
    pred_map, trace_rows = predict_with_selected_rules(
        rows=eval_rows,
        rules=selected_rules,
        laws_exact=laws_exact,
        laws_prefix=laws_prefix,
        max_ngram=args.max_ngram,
        top_k=args.top_k,
        max_rules_per_query=args.max_rules_per_query,
        explicit_prefix_cap=args.explicit_prefix_cap,
        require_legal_anchor=not args.allow_nonlegal_phrases,
        use_query_expansion=not args.disable_query_expansion,
        add_explicit_citations=not args.disable_explicit_citation_floor,
    )
    eval_summary, per_query = evaluate_predictions(eval_rows, pred_map, mode="strict")

    stage_dir = out_dir / label
    _write_csv(stage_dir / "candidate_rules.csv", _rule_rows(rules))
    _write_csv(stage_dir / "candidate_rule_clusters.csv", _cluster_rows(rules))
    _write_csv(stage_dir / "selected_clusters.csv", _selected_cluster_rows(selected_clusters))
    _write_csv(stage_dir / "selected_rules.csv", _rule_rows(selected_rules))
    _write_prediction_csv(stage_dir / "predictions.csv", eval_rows, pred_map)
    _write_csv(stage_dir / "trace.csv", trace_rows)
    _write_csv(stage_dir / "eval_per_query.csv", per_query)

    matched_rows = sum(1 for row in trace_rows if int(row["matched_rule_count"]) > 0)
    nonempty_rows = sum(1 for preds in pred_map.values() if preds)
    summary = {
        "stage": label,
        "mining_rows": len(mining_rows),
        "eval_rows": len(eval_rows),
        "candidate_rule_count": len(rules),
        "candidate_cluster_count": len(_cluster_rules(rules)),
        "selected_cluster_count": len(selected_clusters),
        "selected_rule_count": len(selected_rules),
        "matched_eval_rows": matched_rows,
        "nonempty_prediction_rows": nonempty_rows,
        "explicit_citation_rows": sum(1 for row in trace_rows if row["explicit_citations"]),
        "query_expansion_rows": sum(1 for row in trace_rows if row["query_expansions"]),
        "macro_f1": eval_summary["macro_f1"],
        "avg_missing_gold_from_corpus": eval_summary["avg_missing_gold_from_corpus"],
    }
    (stage_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_doc(path: Path, summary: dict, selected_clusters: list[SelectedCluster]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Train-Mined Cluster Router v1",
        "",
        "This report is generated from `train.csv`, `val.csv`, `laws_de.csv`, and deterministic code.",
        "It does not read `test.csv` and does not use public leaderboard feedback.",
        "",
        "## Goal",
        "",
        "Convert the broad train-derived phrase miner into a smaller robust-cluster router.",
        "The router keeps only citation clusters with repeated phrase evidence, sufficient support, and high train precision, then evaluates them on both a train-derived pseudo-hidden split and the official validation split.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"- `{key}`:")
            for sub_key, sub_value in value.items():
                lines.append(f"  - `{sub_key}`: `{sub_value}`")
        else:
            lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Selected Clusters", ""])
    lines.append("| best phrase | candidate phrases | support | precision | citations |")
    lines.append("|---|---:|---:|---:|---|")
    for cluster in selected_clusters[:30]:
        best = cluster.rules[0]
        phrase = best.phrase.replace("|", "\\|")
        citations = "; ".join(cluster.citations[:8])
        lines.append(
            f"| {phrase} | {cluster.phrase_count} | {cluster.max_support_rows} | "
            f"{cluster.max_precision:.3f} | {citations} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is still an evidence layer, not a leaderboard submission.",
            "- The pseudo-hidden stage mines on train-only rows and evaluates on held-out train topic groups.",
            "- The validation stage mines on all train rows and evaluates on `val.csv`.",
            "- A low validation score is acceptable at this stage if it exposes which train-mined clusters transfer and which do not.",
            "- Future work should improve cluster selection and issue routing before generating any new `submission.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a robust train-mined cluster router without reading test.csv.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "train_mined_cluster_router_v1")
    parser.add_argument("--doc-path", type=Path, default=ROOT / "docs" / "train_mined_cluster_router_v1.md")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", default="institution-router-v1")
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--min-phrase-support", type=int, default=3)
    parser.add_argument("--min-citation-support", type=int, default=2)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument("--mine-max-cluster-size", type=int, default=10)
    parser.add_argument("--min-cluster-phrases", type=int, default=4)
    parser.add_argument("--min-cluster-support", type=int, default=4)
    parser.add_argument("--min-cluster-precision", type=float, default=0.5)
    parser.add_argument("--max-citations-per-cluster", type=int, default=10)
    parser.add_argument("--max-clusters", type=int, default=160)
    parser.add_argument("--max-rules-per-cluster-selected", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rules-per-query", type=int, default=10)
    parser.add_argument("--explicit-prefix-cap", type=int, default=4)
    parser.add_argument("--allow-nonlegal-phrases", action="store_true")
    parser.add_argument("--disable-query-expansion", action="store_true")
    parser.add_argument("--disable-explicit-citation-floor", action="store_true")
    args = parser.parse_args()

    train_rows = load_query_split("train")
    val_rows = load_query_split("val")
    laws_exact = _laws_exact_set()
    pseudo_train_rows, pseudo_hidden_rows, split_rows = make_pseudo_hidden_split(
        train_rows,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "pseudo_hidden_split_groups.csv", split_rows)

    pseudo_summary = _run_stage(
        label="pseudo_hidden",
        mining_rows=pseudo_train_rows,
        eval_rows=pseudo_hidden_rows,
        laws_exact=laws_exact,
        args=args,
        out_dir=args.out_dir,
    )
    validation_summary = _run_stage(
        label="validation",
        mining_rows=train_rows,
        eval_rows=val_rows,
        laws_exact=laws_exact,
        args=args,
        out_dir=args.out_dir,
    )

    selected_cluster_rows_path = args.out_dir / "validation" / "selected_clusters.csv"
    selected_clusters: list[SelectedCluster] = []
    # Rebuild selected clusters for documentation without parsing CSV back into
    # rule objects from disk.
    validation_rules = mine_rules(
        train_rows=train_rows,
        laws_exact=laws_exact,
        max_ngram=args.max_ngram,
        min_phrase_support=args.min_phrase_support,
        min_citation_support=args.min_citation_support,
        min_precision=args.min_precision,
        max_cluster_size=args.mine_max_cluster_size,
        require_legal_anchor=not args.allow_nonlegal_phrases,
    )
    _, selected_clusters = select_robust_clusters(
        rules=validation_rules,
        min_cluster_phrases=args.min_cluster_phrases,
        min_cluster_support=args.min_cluster_support,
        min_cluster_precision=args.min_cluster_precision,
        max_citations_per_cluster=args.max_citations_per_cluster,
        max_clusters=args.max_clusters,
        max_rules_per_cluster=args.max_rules_per_cluster_selected,
    )

    summary = {
        "train_rows_total": len(train_rows),
        "validation_rows": len(val_rows),
        "laws_exact_citations": len(laws_exact),
        "holdout_ratio": args.holdout_ratio,
        "selection": {
            "min_cluster_phrases": args.min_cluster_phrases,
            "min_cluster_support": args.min_cluster_support,
            "min_cluster_precision": args.min_cluster_precision,
            "max_citations_per_cluster": args.max_citations_per_cluster,
            "max_clusters": args.max_clusters,
            "max_rules_per_cluster_selected": args.max_rules_per_cluster_selected,
        },
        "pseudo_hidden": pseudo_summary,
        "validation": validation_summary,
        "selected_cluster_csv": str(selected_cluster_rows_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_doc(args.doc_path, summary, selected_clusters)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from legal_ir.normalization import normalize_citation


FAMILY_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9-]{1,12})\b")
ARTICLE_RE = re.compile(r"\bArt\.\s*([0-9]+[a-z]?)\b", re.I)
TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜßéèà0-9][a-zA-ZäöüÄÖÜßéèà0-9'-]{1,}")

SUPPORTED_FAMILIES = {
    "AHVG",
    "ATSG",
    "AVG",
    "AVEG",
    "BGG",
    "BV",
    "DBG",
    "IPRG",
    "IVG",
    "MSCHG",
    "OR",
    "SCHKG",
    "STGB",
    "STPO",
    "SVG",
    "UWG",
    "UVG",
    "VVG",
    "ZGB",
    "ZPO",
}

STOPWORDS = {
    "aber",
    "alle",
    "allen",
    "als",
    "also",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "beim",
    "bin",
    "bis",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "doch",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "er",
    "es",
    "falls",
    "fuer",
    "für",
    "gegen",
    "hat",
    "haben",
    "hier",
    "hin",
    "im",
    "in",
    "ist",
    "mit",
    "nach",
    "nicht",
    "noch",
    "nun",
    "oder",
    "sich",
    "sie",
    "sind",
    "soll",
    "sowie",
    "und",
    "unter",
    "vom",
    "von",
    "vor",
    "war",
    "was",
    "welche",
    "welchen",
    "welcher",
    "welches",
    "wenn",
    "werden",
    "wie",
    "wird",
    "wurde",
    "wurden",
    "zu",
    "zum",
    "zur",
    "the",
    "and",
    "for",
    "from",
    "that",
    "this",
    "with",
    "what",
    "which",
    "would",
}

GENERIC_LEGAL_WORDS = {
    "art",
    "abs",
    "angefuehrte",
    "beklagte",
    "beklagten",
    "beschwerde",
    "beschwerdefuehrer",
    "beschwerdefuehrerin",
    "beschwerde",
    "bundesgericht",
    "damit",
    "entscheid",
    "erwaegung",
    "frau",
    "frage",
    "gericht",
    "herr",
    "instanz",
    "klaeger",
    "klaegerin",
    "recht",
    "rechts",
    "sachverhalt",
    "urteil",
    "voraussetzungen",
}

LEGAL_ANCHOR_SUBSTRINGS = {
    "abfind",
    "abtret",
    "arbeits",
    "aufheb",
    "auftrag",
    "ausgleich",
    "beistand",
    "beweis",
    "betreib",
    "besitz",
    "beschwer",
    "delikt",
    "dienstbarkeit",
    "ehe",
    "eigentu",
    "erbe",
    "erbrecht",
    "ersatz",
    "famil",
    "forderung",
    "frist",
    "haft",
    "hypothek",
    "invalid",
    "kind",
    "klage",
    "konkurs",
    "kuendig",
    "mangel",
    "miet",
    "nachlass",
    "nichtig",
    "pension",
    "prozess",
    "schenkung",
    "schaden",
    "schuld",
    "sicher",
    "sorge",
    "sorgfalt",
    "straf",
    "taeusch",
    "testament",
    "unterhalt",
    "urteilsfaehig",
    "vertrag",
    "vermoegen",
    "versicherung",
    "vollstreck",
    "willensmaengel",
    "zustaend",
}


@dataclass(frozen=True)
class CandidateRule:
    phrase: str
    support_rows: int
    precision: float
    citations: tuple[str, ...]
    citation_weights: tuple[float, ...]
    families: tuple[str, ...]
    topic_groups: tuple[str, ...]


def _stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _canon(citation: str) -> str:
    text = normalize_citation(citation)
    text = re.sub(r"\bCO\b", "OR", text, flags=re.I)
    text = re.sub(r"\bLAI\b", "IVG", text, flags=re.I)
    return text


def _family(citation: str) -> str:
    found = ""
    for token in FAMILY_RE.findall(citation or ""):
        family = token.replace("-", "").upper()
        if family in {"ART", "ABS", "LIT", "ZIFF", "E"}:
            continue
        if family.isdigit():
            continue
        found = family
    return found


def _article_stem(citation: str) -> str:
    family = _family(citation)
    match = ARTICLE_RE.search(citation or "")
    article = match.group(1).lower() if match else ""
    return f"{family}:{article}" if family and article else family or "UNKNOWN"


def _dedup(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _canon(value)
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _laws_exact_set() -> set[str]:
    exact: set[str] = set()
    for row in iter_corpus_rows(include_laws=True, include_court=False):
        citation = _canon(row.get("citation", ""))
        if citation:
            exact.add(citation)
    return exact


def _normalize_token(token: str) -> str:
    text = token.lower().replace("’", "'")
    return (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        token = _normalize_token(raw.strip("-'"))
        if len(token) < 4:
            continue
        if token in STOPWORDS or token in GENERIC_LEGAL_WORDS:
            continue
        if token.isdigit():
            continue
        out.append(token)
    return out


def _has_legal_anchor(phrase: str) -> bool:
    tokens = phrase.split()
    for token in tokens:
        family = token.replace("-", "").upper()
        if family in SUPPORTED_FAMILIES:
            return True
        if any(anchor in token for anchor in LEGAL_ANCHOR_SUBSTRINGS):
            return True
    return False


def extract_phrases(text: str, max_ngram: int, require_legal_anchor: bool = True) -> set[str]:
    toks = _tokens(text)
    phrases: set[str] = set()
    for n in range(1, max(1, max_ngram) + 1):
        for i in range(0, max(0, len(toks) - n + 1)):
            gram = toks[i : i + n]
            if n > 1 and any(token in GENERIC_LEGAL_WORDS for token in gram):
                continue
            phrase = " ".join(gram)
            if require_legal_anchor and not _has_legal_anchor(phrase):
                continue
            if len(phrase) >= 4:
                phrases.add(phrase)
    return phrases


def _topic_group(row: dict) -> str:
    gold = _dedup(row.get("gold_citation_list", []))
    family_counts = Counter(_family(c) for c in gold if _family(c))
    dominant_family = family_counts.most_common(1)[0][0] if family_counts else "UNKNOWN"
    stems = [stem for stem, _ in Counter(_article_stem(c) for c in gold).most_common(3)]
    return "|".join([dominant_family, *stems])


def make_pseudo_hidden_split(
    rows: list[dict],
    holdout_ratio: float,
    seed: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_topic_group(row)].append(row)

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    group_rows: list[dict] = []
    modulo = max(2, round(1.0 / max(0.01, min(0.5, holdout_ratio))))

    for group_key, group in sorted(grouped.items()):
        group_sorted = sorted(group, key=lambda r: r["query_id"])
        if len(group_sorted) == 1:
            # Singleton legal institutions stay in train; otherwise this gold
            # mining baseline would be asked to predict never-seen articles.
            train_rows.extend(group_sorted)
            split = "train_singleton"
        else:
            local_holdout = [
                row
                for row in group_sorted
                if _stable_bucket(f"{seed}:{group_key}:{row['query_id']}", modulo) == 0
            ]
            if not local_holdout:
                local_holdout = [group_sorted[_stable_bucket(f"{seed}:{group_key}", len(group_sorted))]]
            if len(local_holdout) == len(group_sorted):
                local_holdout = local_holdout[: max(1, len(group_sorted) - 1)]
            local_holdout_ids = {row["query_id"] for row in local_holdout}
            holdout_rows.extend(local_holdout)
            train_rows.extend([row for row in group_sorted if row["query_id"] not in local_holdout_ids])
            split = "mixed_group_holdout"

        group_rows.append(
            {
                "topic_group": group_key,
                "query_count": len(group_sorted),
                "split": split,
                "holdout_count": sum(1 for r in group_sorted if r in holdout_rows),
            }
        )

    return train_rows, holdout_rows, group_rows


def mine_rules(
    train_rows: list[dict],
    laws_exact: set[str],
    max_ngram: int,
    min_phrase_support: int,
    min_citation_support: int,
    min_precision: float,
    max_cluster_size: int,
    require_legal_anchor: bool,
) -> list[CandidateRule]:
    phrase_rows: dict[str, set[str]] = defaultdict(set)
    phrase_citations: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_families: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_topics: dict[str, Counter[str]] = defaultdict(Counter)

    for row in train_rows:
        qid = row["query_id"]
        phrases = extract_phrases(
            row.get("query", ""),
            max_ngram=max_ngram,
            require_legal_anchor=require_legal_anchor,
        )
        gold = [c for c in _dedup(row.get("gold_citation_list", [])) if c in laws_exact]
        if not phrases or not gold:
            continue
        topic = _topic_group(row)
        families = {_family(c) for c in gold if _family(c)}
        for phrase in phrases:
            phrase_rows[phrase].add(qid)
            phrase_citations[phrase].update(gold)
            phrase_families[phrase].update(families)
            phrase_topics[phrase].update([topic])

    candidates: list[CandidateRule] = []
    for phrase, row_ids in phrase_rows.items():
        support = len(row_ids)
        if support < min_phrase_support:
            continue
        citation_counts = phrase_citations[phrase]
        if not citation_counts:
            continue
        top_citation, top_count = citation_counts.most_common(1)[0]
        precision = top_count / support
        if top_count < min_citation_support or precision < min_precision:
            continue
        cluster: list[str] = []
        weights: list[float] = []
        for citation, count in citation_counts.most_common():
            citation_precision = count / support
            if count < min_citation_support and len(cluster) >= 1:
                continue
            if citation_precision < max(0.15, min_precision * 0.5) and len(cluster) >= 1:
                continue
            cluster.append(citation)
            weights.append(round(citation_precision, 6))
            if len(cluster) >= max_cluster_size:
                break
        candidates.append(
            CandidateRule(
                phrase=phrase,
                support_rows=support,
                precision=round(precision, 6),
                citations=tuple(cluster),
                citation_weights=tuple(weights),
                families=tuple(f for f, _ in phrase_families[phrase].most_common(4)),
                topic_groups=tuple(t for t, _ in phrase_topics[phrase].most_common(4)),
            )
        )

    candidates.sort(key=lambda r: (r.precision, r.support_rows, len(r.citations), r.phrase), reverse=True)
    return candidates


def predict_with_rules(
    rows: list[dict],
    rules: list[CandidateRule],
    max_ngram: int,
    top_k: int,
    max_rules_per_query: int,
    require_legal_anchor: bool,
) -> tuple[dict[str, list[str]], list[dict]]:
    rule_by_phrase = {rule.phrase: rule for rule in rules}
    pred_map: dict[str, list[str]] = {}
    trace_rows: list[dict] = []

    for row in rows:
        phrases = extract_phrases(
            row.get("query", ""),
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

        ranked = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
        pred_map[row["query_id"]] = [citation for citation, _ in ranked[:top_k]]
        trace_rows.append(
            {
                "query_id": row["query_id"],
                "matched_rule_count": len(matched),
                "matched_phrases": ";".join(rule.phrase for rule in matched),
                "predicted_citations": ";".join(pred_map[row["query_id"]]),
                "prediction_count": len(pred_map[row["query_id"]]),
            }
        )

    return pred_map, trace_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_prediction_csv(path: Path, rows: list[dict], pred_map: dict[str, list[str]]) -> None:
    out = [
        {
            "query_id": row["query_id"],
            "predicted_citations": ";".join(pred_map.get(row["query_id"], [])),
        }
        for row in rows
    ]
    _write_csv(path, out, ["query_id", "predicted_citations"])


def _rule_rows(rules: list[CandidateRule]) -> list[dict]:
    return [
        {
            "phrase": rule.phrase,
            "support_rows": rule.support_rows,
            "precision": rule.precision,
            "families": ";".join(rule.families),
            "citations": ";".join(rule.citations),
            "citation_weights": ";".join(str(x) for x in rule.citation_weights),
            "topic_groups": " || ".join(rule.topic_groups),
        }
        for rule in rules
    ]


def _cluster_rows(rules: list[CandidateRule]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[CandidateRule]] = defaultdict(list)
    for rule in rules:
        grouped[rule.citations].append(rule)

    rows: list[dict] = []
    for citations, cluster_rules in grouped.items():
        cluster_rules.sort(key=lambda r: (r.precision, r.support_rows, len(r.phrase)), reverse=True)
        best = cluster_rules[0]
        rows.append(
            {
                "best_phrase": best.phrase,
                "phrase_count": len(cluster_rules),
                "max_support_rows": max(rule.support_rows for rule in cluster_rules),
                "max_precision": max(rule.precision for rule in cluster_rules),
                "families": ";".join(best.families),
                "citations": ";".join(citations),
                "example_phrases": ";".join(rule.phrase for rule in cluster_rules[:8]),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["max_precision"]),
            int(row["max_support_rows"]),
            int(row["phrase_count"]),
            row["best_phrase"],
        ),
        reverse=True,
    )
    return rows


def _diverse_rules(rules: list[CandidateRule], limit: int) -> list[CandidateRule]:
    out: list[CandidateRule] = []
    seen_clusters: set[tuple[str, ...]] = set()
    for rule in rules:
        key = rule.citations
        if key in seen_clusters:
            continue
        out.append(rule)
        seen_clusters.add(key)
        if len(out) >= limit:
            break
    return out


def _write_doc(path: Path, summary: dict, top_rules: list[CandidateRule]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Train-Derived Institution Router Mining",
        "",
        "This report is generated only from `train.csv`, `laws_de.csv`, and a deterministic train-derived pseudo-hidden split.",
        "It is meant to support the prize-compliant path: no visible `test.csv` row labels or query-id patch table are used.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Candidate Rules", ""])
    lines.append("| phrase | support | precision | families | citations |")
    lines.append("|---|---:|---:|---|---|")
    for rule in _diverse_rules(top_rules, 30):
        citations = "; ".join(rule.citations[:8])
        families = "; ".join(rule.families)
        phrase = rule.phrase.replace("|", "\\|")
        lines.append(f"| {phrase} | {rule.support_rows} | {rule.precision:.3f} | {families} | {citations} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A rule is a candidate only when a query phrase repeatedly maps to a laws-grounded gold citation cluster in train.",
            "- Pseudo-hidden evaluation measures whether these mined phrase clusters recover citations on held-out train rows from the same grouped legal topic.",
            "- This is not yet a final submission generator; it is the reproducible evidence layer that should replace public-leaderboard-derived rule authorship.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine train-derived phrase -> legal-institution citation clusters and evaluate them on a pseudo-hidden split."
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "train_institution_router_v1")
    parser.add_argument("--doc-path", type=Path, default=ROOT / "docs" / "train_institution_router_v1.md")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", default="legal-ir-2026")
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--min-phrase-support", type=int, default=3)
    parser.add_argument("--min-citation-support", type=int, default=2)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument("--max-cluster-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rules-per-query", type=int, default=5)
    parser.add_argument(
        "--allow-nonlegal-phrases",
        action="store_true",
        help="Disable the default legal-anchor phrase filter for exploratory diagnostics.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    doc_path = args.doc_path if args.doc_path.is_absolute() else ROOT / args.doc_path

    rows = load_query_split("train")
    laws_exact = _laws_exact_set()
    train_rows, pseudo_hidden_rows, split_groups = make_pseudo_hidden_split(
        rows,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    rules = mine_rules(
        train_rows=train_rows,
        laws_exact=laws_exact,
        max_ngram=args.max_ngram,
        min_phrase_support=args.min_phrase_support,
        min_citation_support=args.min_citation_support,
        min_precision=args.min_precision,
        max_cluster_size=args.max_cluster_size,
        require_legal_anchor=not args.allow_nonlegal_phrases,
    )
    pred_map, trace_rows = predict_with_rules(
        pseudo_hidden_rows,
        rules,
        max_ngram=args.max_ngram,
        top_k=args.top_k,
        max_rules_per_query=args.max_rules_per_query,
        require_legal_anchor=not args.allow_nonlegal_phrases,
    )
    eval_summary, per_query = evaluate_predictions(pseudo_hidden_rows, pred_map)
    matched_rows = sum(1 for row in trace_rows if int(row["matched_rule_count"]) > 0)
    nonempty_rows = sum(1 for values in pred_map.values() if values)

    summary = {
        "train_rows_total": len(rows),
        "train_rows_for_mining": len(train_rows),
        "pseudo_hidden_rows": len(pseudo_hidden_rows),
        "pseudo_hidden_topic_groups": sum(1 for row in split_groups if row["holdout_count"]),
        "laws_exact_citations": len(laws_exact),
        "candidate_rule_count": len(rules),
        "pseudo_hidden_matched_rows": matched_rows,
        "pseudo_hidden_nonempty_predictions": nonempty_rows,
        "pseudo_hidden_macro_f1": eval_summary["macro_f1"],
        "holdout_ratio": args.holdout_ratio,
        "min_phrase_support": args.min_phrase_support,
        "min_citation_support": args.min_citation_support,
        "min_precision": args.min_precision,
        "top_k": args.top_k,
        "require_legal_anchor": not args.allow_nonlegal_phrases,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "candidate_rules.csv", _rule_rows(rules))
    _write_csv(out_dir / "candidate_rule_clusters.csv", _cluster_rows(rules))
    _write_csv(out_dir / "pseudo_hidden_split_groups.csv", split_groups)
    _write_prediction_csv(out_dir / "pseudo_hidden_predictions.csv", pseudo_hidden_rows, pred_map)
    _write_csv(out_dir / "pseudo_hidden_trace.csv", trace_rows)
    _write_csv(out_dir / "pseudo_hidden_eval_per_query.csv", per_query)
    _write_doc(doc_path, summary, rules)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

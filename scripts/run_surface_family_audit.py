from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORE_FAMILIES = {"BGG"}

EXPLICIT_FAMILIES = [
    "AHVG",
    "AHVV",
    "AIG",
    "ATSG",
    "AVEG",
    "AVG",
    "BGG",
    "BV",
    "BZP",
    "EMBAG",
    "HREGV",
    "IPRG",
    "IRSG",
    "IVG",
    "MSCHG",
    "OR",
    "PRHG",
    "SAFIG",
    "SCHKG",
    "SVG",
    "STGB",
    "STPO",
    "URG",
    "UVG",
    "UVV",
    "UWG",
    "VID",
    "ZGB",
    "ZPO",
]

# Conservative, test-surface-oriented cues. These are not used to reroute broad
# retrieval; they only flag rows worth human inspection.
CUE_RULES: list[tuple[str, list[str], re.Pattern[str]]] = [
    (
        "avg_aveg_temp_work_gav",
        ["AVG", "AVEG", "OR"],
        re.compile(r"\b(temporary.?staffing|temporary workers?|assignment contracts?|collective labour agreement|customary wage|minimum wage|tripartite|GAV)\b", re.I),
    ),
    (
        "building_lien_work_contract",
        ["ZGB", "OR", "ZPO"],
        re.compile(r"\b(statutory building mortgage|statutory mortgage|building mortgage|definitive registration|contractor|fixed-price building|expert appraisal)\b", re.I),
    ),
    (
        "mortgage_certificate_pledge_enforcement",
        ["ZGB", "SCHKG", "OR"],
        re.compile(r"\b(bearer mortgage certificate|mortgage certificate|real.?estate pledge|novation|fiduciary security|exigibility|denunciation)\b", re.I),
    ),
    (
        "civil_liability_moral_damage",
        ["OR", "ZGB"],
        re.compile(r"\b(civilly liable|civil liability|moral damage|non.?material|loss of earnings|adequate causation|medical expenses)\b", re.I),
    ),
    (
        "lease_rent_arrears_termination",
        ["OR", "ZPO"],
        re.compile(r"\b(tenancy|rent arrears|30.?day cure|formula termination|summary eviction|cas clairs|nova|bank statements filed first on appeal)\b", re.I),
    ),
    (
        "ip_copyright_trade_secret_unfair",
        ["ZPO", "IPRG", "URG", "UWG"],
        re.compile(
            r"\b(copyright|source code|trade.?secret|unfair competition)\b"
            r"|(?=.*\b(interim relief|provisional measures|injunction)\b)"
            r"(?=.*\b(copyright|source code|trade.?secret|unfair competition|trademark|domain)\b)",
            re.I,
        ),
    ),
    (
        "international_legal_assistance_evidence",
        ["IRSG", "STPO", "BZP", "BV"],
        re.compile(r"\b(international judicial assistance|judicial assistance request|assistance request|taking of evidence|produce documents|refuse cooperation|business.?secret|available remedies|review deadlines)\b", re.I),
    ),
    (
        "foreign_divorce_recognition_measures",
        ["IPRG", "ZGB"],
        re.compile(r"\b(foreign divorce|recognise|recognize|recognition|provisional spousal|protective relief|foreign proceedings|common national state|public policy)\b", re.I),
    ),
    (
        "trademark_domain_unfair",
        ["MSCHG", "UWG"],
        re.compile(r"\b(trademark|trade mark|domain name|domain|distinctive sign|unfair competition)\b", re.I),
    ),
    (
        "uvg_occupational_accident",
        ["UVG", "UVV", "ATSG"],
        re.compile(r"\b(UVG|occupational disease|occupational aggravation|accident insurer|insured for accidents|lesion assimilated)\b", re.I),
    ),
    (
        "svg_road_traffic",
        ["SVG", "OR"],
        re.compile(r"\b(SVG|vehicle holder|cyclist|road traffic|traffic accident|gross negligence)\b", re.I),
    ),
    (
        "schkg_enforcement_bankruptcy",
        ["SCHKG"],
        re.compile(r"\b(SchKG|debt enforcement|bankruptcy|opposition lifted|forced sale|attachment|pledge)\b", re.I),
    ),
    (
        "zgb_child_family",
        ["ZGB"],
        re.compile(r"\b(ZGB|civil code|child support|child maintenance|spousal maintenance|child custody|custody of (?:the )?children|visitation|parental authority|spousal support)\b", re.I),
    ),
    (
        "matrimonial_protection_maintenance",
        ["ZGB", "ZPO"],
        re.compile(r"\b(matrimonial[- ]protection|protective measures in the matrimonial sphere|marital union|post[- ]divorce maintenance|maintenance pension|hypothetical income|notional employment income|statutory subsistence minimum|minimum-subsistence|surplus)\b", re.I),
    ),
    (
        "zgb_inheritance_property",
        ["ZGB"],
        re.compile(r"\b(will|testament|testator|heir|inheritance|ownership|possessor|deed of gift)\b", re.I),
    ),
    (
        "or_contract_liability",
        ["OR"],
        re.compile(r"\b(code of obligations|contract|lease|mandate|forged|liability|damages|gross negligence)\b", re.I),
    ),
    (
        "civil_burden_of_proof",
        ["ZGB"],
        re.compile(r"\b(burden of proof|burden of proving|carries the burden|prove the amounts|proof for the amounts)\b", re.I),
    ),
    (
        "zpo_civil_procedure",
        ["ZPO"],
        re.compile(r"\b(ZPO|civil procedure|testimony|pleaded|interim relief)\b", re.I),
    ),
    (
        "adult_protection_guardianship",
        ["ZGB", "BGG"],
        re.compile(r"\b(adult protection|guardian|guardianship|welfare inquiry|psychiatric expert assessment|representation and financial management|difficult to repair)\b", re.I),
    ),
    (
        "stpo_criminal_procedure",
        ["STPO", "BV"],
        re.compile(r"\b(StPO|criminal procedure|pre.?trial detention|remand|collusion|flight risk)\b", re.I),
    ),
    (
        "bv_right_to_be_heard",
        ["BV"],
        re.compile(r"\b(right to be heard|legal hearing)\b", re.I),
    ),
    (
        "stgb_criminal_substantive",
        ["STGB"],
        re.compile(r"\b(StGB|criminal code|convicted|sentence|robbery|disloyal management|offence|offense)\b", re.I),
    ),
    (
        "social_invalidity",
        ["ATSG", "IVG"],
        re.compile(r"\b(ATSG|LPGA|IVG|LAI|invalidity insurance|vocational rehabilitation|adapted work)\b", re.I),
    ),
    (
        "ahv_employer_board_liability",
        ["AHVG", "BV"],
        re.compile(r"\b(AHV|compensation office|former board members?|Art\.\s*52\s+Abs\.\s*1\s+AHVG)\b", re.I),
    ),
]


def _split_citations(value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in (value or "").split(";"):
        item = part.strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _read_test(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["query_id"].strip(): row["query"] for row in csv.DictReader(f)}


def _read_submission(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {
            row["query_id"].strip(): _split_citations(row.get("predicted_citations", ""))
            for row in csv.DictReader(f)
            if row.get("query_id")
        }


def _add_family(out: list[str], seen: set[str], family: str) -> None:
    family = family.upper().replace("-", "")
    if family and family not in seen:
        out.append(family)
        seen.add(family)


def expected_families(query: str) -> tuple[list[str], list[str]]:
    out: list[str] = []
    seen: set[str] = set()
    rules: list[str] = []
    text = query or ""
    for family in EXPLICIT_FAMILIES:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(family)}(?![A-Za-z0-9])"
        if re.search(pattern, text):
            _add_family(out, seen, family)
            rules.append(f"explicit:{family}")
    aliases = [
        ("ZGB", r"\bcivil code\b"),
        ("OR", r"\bcode of obligations\b"),
        ("STPO", r"\bcriminal procedure code\b"),
        ("STGB", r"\bcriminal code\b"),
        ("ZPO", r"\bcivil procedure\b"),
        ("ATSG", r"\bLPGA\b|\bgeneral part of social insurance law\b"),
        ("IVG", r"\bLAI\b|\binvalidity insurance\b"),
        ("IPRG", r"\bLDIP\b|\bprivate international law\b"),
        ("MSCHG", r"\bLPM\b|\btrademark protection act\b"),
        ("UWG", r"\bLCD\b|\bunfair competition act\b"),
        ("BGG", r"\bLTF\b|\bfederal supreme court act\b"),
        ("ZPO", r"\bCPC\b"),
        ("STPO", r"\bCPP\b"),
        ("UVG", r"\bLAA\b"),
        ("SVG", r"\bLCR\b"),
    ]
    for family, pattern in aliases:
        if re.search(pattern, text, re.I):
            _add_family(out, seen, family)
            rules.append(f"alias:{family}")
    for name, families, pattern in CUE_RULES:
        if pattern.search(text):
            rules.append(f"cue:{name}")
            for family in families:
                _add_family(out, seen, family)
    return [f for f in out if f not in IGNORE_FAMILIES], rules


def citation_families(citations: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{1,12}\b", citation or ""):
            family = token.upper().replace("-", "")
            if family in EXPLICIT_FAMILIES and family not in IGNORE_FAMILIES:
                _add_family(out, seen, family)
    return out


def alignment_score(expected: list[str], predicted: list[str]) -> float | None:
    if not expected:
        return None
    return len(set(expected) & set(predicted)) / len(set(expected))


def _count_joined(value: object) -> int:
    text = str(value or "")
    return len([part for part in text.split(";") if part])


def _preview(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def row_record(qid: str, query: str, citations: list[str]) -> dict[str, str | float | int]:
    expected, rules = expected_families(query)
    predicted = citation_families(citations)
    score = alignment_score(expected, predicted)
    missing = [f for f in expected if f not in predicted]
    unexpected = [f for f in predicted if f not in expected]
    return {
        "query_id": qid,
        "alignment_score": "" if score is None else round(score, 6),
        "expected_family_count": len(expected),
        "predicted_family_count": len(predicted),
        "expected_families": ";".join(expected),
        "predicted_families": ";".join(predicted),
        "missing_expected_families": ";".join(missing),
        "unexpected_predicted_families": ";".join(unexpected),
        "matched_rules": ";".join(rules),
        "predicted_citations": ";".join(citations),
        "query_preview": _preview(query),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit test-surface family alignment for a submission.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--label", default="surface_family_audit")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "surface_family_audit")
    args = parser.parse_args()

    queries = _read_test(ROOT / "data_raw" / "competition_data" / "test.csv")
    submission = _read_submission(args.submission)
    missing_qids = [qid for qid in queries if qid not in submission]
    if missing_qids:
        raise SystemExit(f"submission missing qids: {missing_qids}")

    rows = [row_record(qid, queries[qid], submission[qid]) for qid in queries]
    scored = [r for r in rows if r["alignment_score"] != ""]
    low = [r for r in scored if float(r["alignment_score"]) < 1.0]

    changed_rows: list[dict[str, str | float | int]] = []
    changed_candidate_scores: list[float] = []
    changed_base_scores: list[float] = []
    changed_candidate_unexpected_counts: list[int] = []
    changed_base_unexpected_counts: list[int] = []
    changed_candidate_prediction_counts: list[int] = []
    changed_base_prediction_counts: list[int] = []
    if args.base_submission:
        base = _read_submission(args.base_submission)
        for qid in queries:
            if base.get(qid, []) == submission[qid]:
                continue
            rec = dict(row_record(qid, queries[qid], submission[qid]))
            base_rec = row_record(qid, queries[qid], base.get(qid, []))
            rec["base_alignment_score"] = base_rec["alignment_score"]
            rec["base_predicted_families"] = base_rec["predicted_families"]
            rec["base_predicted_citations"] = base_rec["predicted_citations"]
            changed_rows.append(rec)
            if rec["alignment_score"] != "":
                changed_candidate_scores.append(float(rec["alignment_score"]))
            if base_rec["alignment_score"] != "":
                changed_base_scores.append(float(base_rec["alignment_score"]))
            changed_candidate_unexpected_counts.append(_count_joined(rec["unexpected_predicted_families"]))
            changed_base_unexpected_counts.append(_count_joined(base_rec["unexpected_predicted_families"]))
            changed_candidate_prediction_counts.append(len(submission[qid]))
            changed_base_prediction_counts.append(len(base.get(qid, [])))

    mean_score = (
        sum(float(r["alignment_score"]) for r in scored) / len(scored)
        if scored
        else None
    )
    summary = {
        "label": args.label,
        "submission": str(args.submission),
        "base_submission": str(args.base_submission) if args.base_submission else "",
        "row_count": len(rows),
        "scored_row_count": len(scored),
        "mean_alignment_score": None if mean_score is None else round(mean_score, 6),
        "low_alignment_count": len(low),
        "changed_query_count": len(changed_rows),
        "changed_qids": [str(r["query_id"]) for r in changed_rows],
        "changed_mean_base_alignment_score": (
            None if not changed_base_scores else round(sum(changed_base_scores) / len(changed_base_scores), 6)
        ),
        "changed_mean_candidate_alignment_score": (
            None
            if not changed_candidate_scores
            else round(sum(changed_candidate_scores) / len(changed_candidate_scores), 6)
        ),
        "changed_mean_base_unexpected_family_count": (
            None
            if not changed_base_unexpected_counts
            else round(sum(changed_base_unexpected_counts) / len(changed_base_unexpected_counts), 6)
        ),
        "changed_mean_candidate_unexpected_family_count": (
            None
            if not changed_candidate_unexpected_counts
            else round(sum(changed_candidate_unexpected_counts) / len(changed_candidate_unexpected_counts), 6)
        ),
        "changed_mean_base_prediction_count": (
            None
            if not changed_base_prediction_counts
            else round(sum(changed_base_prediction_counts) / len(changed_base_prediction_counts), 6)
        ),
        "changed_mean_candidate_prediction_count": (
            None
            if not changed_candidate_prediction_counts
            else round(sum(changed_candidate_prediction_counts) / len(changed_candidate_prediction_counts), 6)
        ),
    }

    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (out_dir / "per_row_family_alignment.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if changed_rows:
        changed_fields = list(changed_rows[0].keys())
        with (out_dir / "changed_rows_family_alignment.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=changed_fields)
            writer.writeheader()
            writer.writerows(changed_rows)
    with (out_dir / "low_alignment_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(low)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

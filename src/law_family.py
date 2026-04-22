from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable


SUPPORTED_FAMILIES = {
    "STPO",
    "OR",
    "ZGB",
    "STGB",
    "BV",
    "AIG",
    "BGG",
    "ZPO",
    "SCHKG",
    "IPRG",
    "ATSG",
    "IVG",
}


FAMILY_TERMS = {
    "STPO": [
        "StPO",
        "Strafprozessordnung",
        "Strafverfahren",
        "Untersuchungshaft",
        "Haft",
        "Kollusionsgefahr",
        "Fluchtgefahr",
        "dringender Tatverdacht",
        "rechtliches Gehoer",
    ],
    "STGB": [
        "StGB",
        "Strafgesetzbuch",
        "ungetreue Geschaeftsbesorgung",
        "oeffentliche Interessen",
        "Anklage",
        "Schuld",
        "Strafe",
        "Taeterschaft",
    ],
    "ZGB": [
        "ZGB",
        "Zivilgesetzbuch",
        "Erbrecht",
        "Testament",
        "eigenhaendige letztwillige Verfuegung",
        "Urteilsfaehigkeit",
        "Kindesrecht",
        "Besuchsrecht",
        "persoenlicher Verkehr",
        "Kindeswohl",
        "Unterhalt",
    ],
    "OR": [
        "OR",
        "Obligationenrecht",
        "Auftrag",
        "Vertrag",
        "Haftung",
        "Sorgfaltspflicht",
        "Bank",
        "Ueberweisung",
        "grobe Fahrlaessigkeit",
        "Genehmigung",
        "Waehrung",
    ],
    "ZPO": [
        "ZPO",
        "Zivilprozessordnung",
        "Klage",
        "Beweis",
        "Zeugnis",
        "Verfahren",
        "Rechtsbegehren",
    ],
    "SCHKG": [
        "SchKG",
        "Schuldbetreibung",
        "Konkurs",
        "Vollstreckung",
        "Sicherstellung",
        "Pfaendung",
    ],
    "BV": [
        "BV",
        "Bundesverfassung",
        "rechtliches Gehoer",
        "Unschuldsvermutung",
    ],
    "AIG": [
        "AIG",
        "Auslaender",
        "Migration",
        "Aufenthalt",
    ],
    "BGG": [
        "BGG",
        "Bundesgerichtsgesetz",
        "Beschwerde",
        "Frist",
    ],
    "IPRG": [
        "IPRG",
        "internationales Privatrecht",
        "anwendbares Recht",
    ],
    "ATSG": [
        "ATSG",
        "Sozialversicherung",
        "Arbeitsunfaehigkeit",
        "Invaliditaet",
    ],
    "IVG": [
        "IVG",
        "Invalidenversicherung",
        "Eingliederung",
        "Erwerbsunfaehigkeit",
    ],
}


FAMILY_CUES: dict[str, list[tuple[str, int]]] = {
    "STPO": [
        ("stpo", 8),
        ("pretrial detention", 5),
        ("pre-trial detention", 5),
        ("detention", 3),
        ("collusion", 4),
        ("flight risk", 4),
        ("sufficient suspicion", 4),
        ("coercive measures", 4),
        ("right to be heard", 2),
        ("presumption of innocence", 2),
    ],
    "STGB": [
        ("stgb", 8),
        ("disloyal management", 6),
        ("public interests", 3),
        ("criminal proceedings", 4),
        ("indictment", 4),
        ("conviction", 3),
        ("offenses", 2),
        ("theft", 2),
        ("assault", 2),
    ],
    "ZGB": [
        ("zgb", 8),
        ("civil code", 6),
        ("will", 3),
        ("testament", 5),
        ("testator", 5),
        ("estate", 4),
        ("heir", 4),
        ("holographic", 4),
        ("custody", 4),
        ("visitation", 5),
        ("overnight", 3),
        ("child support", 5),
        ("maintenance", 4),
        ("co-parent", 3),
        ("children", 3),
        ("best interests", 4),
    ],
    "OR": [
        ("or", 0),
        ("code of obligations", 7),
        ("contract", 4),
        ("liability", 5),
        ("damages", 4),
        ("bank", 2),
        ("account holder", 4),
        ("duty of care", 5),
        ("gross negligence", 5),
        ("exculpatory", 4),
        ("forged", 3),
        ("transfer instructions", 4),
        ("currency", 3),
    ],
    "ZPO": [
        ("zpo", 8),
        ("civil procedure", 6),
        ("civil judge", 3),
        ("pleaded", 3),
        ("testimony", 3),
        ("admissible", 3),
        ("claim", 2),
        ("proceedings", 1),
    ],
    "SCHKG": [
        ("schkg", 8),
        ("enforcement", 3),
        ("forced sale", 4),
        ("freeze", 3),
        ("security", 2),
    ],
    "BV": [
        ("bv", 8),
        ("constitutional", 4),
        ("right to be heard", 4),
        ("presumption of innocence", 4),
    ],
    "AIG": [
        ("aig", 8),
        ("foreign national", 4),
        ("asylum", 4),
        ("residence permit", 4),
        ("deportation", 4),
    ],
}


ISSUE_PHRASE_RULES: dict[str, list[dict]] = {
    "STPO": [
        {
            "name": "pretrial_detention",
            "cues": [
                "pretrial detention",
                "pre-trial detention",
                "detention",
                "coercive measures",
            ],
            "terms": [
                "Untersuchungshaft",
                "Sicherheitshaft",
                "dringend verdaechtig",
                "dringend verdächtig",
            ],
        },
        {
            "name": "collusion_flight_risk",
            "cues": ["collusion", "flight risk", "tamper", "witness"],
            "terms": [
                "Kollusionsgefahr",
                "Fluchtgefahr",
                "Personen beeinflusst",
                "Beweismittel einwirkt",
            ],
        },
        {
            "name": "appeal_costs_criminal_procedure",
            "cues": ["appeal", "complaint", "costs", "extension"],
            "terms": [
                "Beschwerde",
                "Beschwerdelegitimation",
                "Frist",
                "Verfahrenskosten",
            ],
        },
        {
            "name": "indictment_principle",
            "cues": ["indictment", "remand", "sentencing", "conviction"],
            "terms": [
                "Anklagegrundsatz",
                "Anklageschrift",
                "Umgrenzungsfunktion",
                "Rueckweisung",
                "Rückweisung",
            ],
        },
    ],
    "BV": [
        {
            "name": "right_to_be_heard",
            "cues": ["right to be heard", "heard", "reasoning", "breach"],
            "terms": [
                "rechtliches Gehoer",
                "rechtliches Gehör",
                "Begruendungspflicht",
                "Begründungspflicht",
            ],
        },
        {
            "name": "presumption_of_innocence",
            "cues": ["presumption of innocence", "sufficient suspicion", "suspicion"],
            "terms": ["Unschuldsvermutung", "Grundrechte"],
        },
    ],
    "ZGB": [
        {
            "name": "holographic_will_form",
            "cues": [
                "handwritten will",
                "holographic",
                "formal requirements",
                "third party",
                "will meet the formal",
            ],
            "terms": [
                "eigenhaendige Verfuegung",
                "eigenhändige Verfügung",
                "letztwillige Verfuegung",
                "letztwillige Verfügung",
                "Erblasser",
            ],
        },
        {
            "name": "testamentary_capacity",
            "cues": [
                "testamentary capacity",
                "testator",
                "language comprehension",
                "annul",
                "validity",
            ],
            "terms": [
                "Urteilsfaehigkeit",
                "Urteilsfähigkeit",
                "Testierfaehigkeit",
                "Testierfähigkeit",
                "Ungueltigkeit",
                "Ungültigkeit",
            ],
        },
        {
            "name": "custody_visitation",
            "cues": [
                "custody",
                "visitation",
                "overnight",
                "contact",
                "family welfare",
            ],
            "terms": [
                "elterliche Sorge",
                "Obhut",
                "persoenlicher Verkehr",
                "persönlicher Verkehr",
                "Betreuungsanteile",
            ],
        },
        {
            "name": "child_best_interests_support",
            "cues": [
                "child support",
                "best interests",
                "children",
                "maintenance",
                "support was fixed",
            ],
            "terms": [
                "Kindeswohl",
                "Unterhaltsbeitrag",
                "Kindesunterhalt",
                "Unterhaltspflicht",
            ],
        },
        {
            "name": "maintenance_security_enforcement",
            "cues": [
                "future maintenance",
                "security",
                "freeze",
                "forced sale",
                "enforcement authority",
                "sale proceeds",
            ],
            "terms": [
                "Sicherstellung",
                "kuenftige Unterhaltsbeitraege",
                "künftige Unterhaltsbeiträge",
                "Vermoegen beiseite schaffen",
                "Vermögen beiseite schaffen",
                "Unterhaltspflicht",
                "Kindesschutz",
            ],
        },
        {
            "name": "ownership_good_faith_possession",
            "cues": [
                "donated",
                "deed of gift",
                "physical delivery",
                "good faith",
                "ownership",
                "possessor",
            ],
            "terms": [
                "Eigentum",
                "Besitz",
                "guter Glaube",
                "Herausgabe",
                "Schenkung",
            ],
        },
    ],
    "OR": [
        {
            "name": "contract_work_liability",
            "cues": [
                "contract for work",
                "gratuitous",
                "standard of care",
                "liable",
                "damage",
            ],
            "terms": [
                "Werkvertrag",
                "Auftrag",
                "Sorgfaltspflicht",
                "Haftung",
                "Schadenersatz",
            ],
        },
        {
            "name": "bank_forged_payment_orders",
            "cues": [
                "bank",
                "forged",
                "transfer instructions",
                "signature",
                "statement-hold",
                "written protest",
            ],
            "terms": [
                "Bankvertrag",
                "Auftrag",
                "Vorschrift",
                "Beauftragte",
                "Auftraggeber",
                "Zahlungsauftrag",
                "Ueberweisung",
                "Überweisung",
                "Unterschrift",
                "Genehmigung",
            ],
        },
        {
            "name": "gross_negligence_exculpation_currency",
            "cues": [
                "gross negligence",
                "exculpatory",
                "risk-allocation",
                "currency",
                "indemnification",
            ],
            "terms": [
                "grobe Fahrlaessigkeit",
                "grobe Fahrlässigkeit",
                "Wegbedingung der Haftung",
                "Geldschulden",
                "Waehrung",
                "Währung",
                "Landeswaehrung",
                "Landeswährung",
                "Fremdwaehrung",
                "Fremdwährung",
            ],
        },
    ],
    "STGB": [
        {
            "name": "disloyal_public_management",
            "cues": [
                "disloyal management",
                "public interests",
                "town council",
                "municipal",
                "publicly funded",
            ],
            "terms": [
                "Ungetreue Amtsfuehrung",
                "Ungetreue Amtsführung",
                "oeffentliche Interessen",
                "öffentliche Interessen",
                "Beamte",
            ],
        },
        {
            "name": "criminal_liability_sentence",
            "cues": ["conviction", "offense", "offences", "sentence", "sentencing"],
            "terms": ["Vorteil", "Freiheitsstrafe", "Geldstrafe", "Schuld"],
        },
    ],
    "ZPO": [
        {
            "name": "civil_procedure_evidence_transition",
            "cues": [
                "testimony",
                "pleaded",
                "admissible",
                "civil procedure",
                "first challenged",
                "protest",
            ],
            "terms": [
                "Zivilprozessordnung",
                "Beweis",
                "Zeugnis",
                "Rechtsbegehren",
            ],
        },
    ],
    "SCHKG": [
        {
            "name": "debt_enforcement_security",
            "cues": [
                "debt enforcement",
                "debt collection",
                "bankruptcy",
                "debt enforcement office",
            ],
            "terms": ["Sicherstellung", "Vollstreckung", "Verwertung", "Betreibung"],
        },
    ],
}


def extract_family_from_citation(citation: str) -> str:
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{1,12}\b", citation or ""):
        family = token.replace("-", "").upper()
        if family in SUPPORTED_FAMILIES:
            return family
    return ""


def explicit_families(query: str) -> list[str]:
    text = query or ""
    found: list[str] = []
    seen = set()
    for fam in SUPPORTED_FAMILIES:
        if re.search(rf"\b{re.escape(fam)}\b", text, re.I):
            found.append(fam)
            seen.add(fam)
    law_names = [
        ("ZGB", r"\bcivil code\b"),
        ("OR", r"\bcode of obligations\b"),
        ("STPO", r"\bcriminal procedure code\b"),
        ("STGB", r"\bcriminal code\b"),
        ("ZPO", r"\bcivil procedure\b"),
    ]
    for fam, pattern in law_names:
        if fam not in seen and re.search(pattern, text, re.I):
            found.append(fam)
            seen.add(fam)
    return found


def likely_statute_families(query: str, max_families: int = 2, min_score: int = 4) -> list[str]:
    text = (query or "").lower()
    scored: list[tuple[str, int]] = []
    for fam, cues in FAMILY_CUES.items():
        score = 0
        for cue, weight in cues:
            if weight <= 0:
                continue
            if re.search(rf"\b{re.escape(cue.lower())}\b", text):
                score += weight
        if score >= min_score:
            scored.append((fam, score))
    return [fam for fam, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:max_families]]


def family_query_terms(families: Iterable[str]) -> list[str]:
    terms: list[str] = []
    seen = set()
    for fam in families:
        for term in FAMILY_TERMS.get(fam.upper(), []):
            key = term.lower()
            if key in seen:
                continue
            terms.append(term)
            seen.add(key)
    return terms


def _has_issue_cue(text: str, cue: str) -> bool:
    cue = (cue or "").lower().strip()
    if not cue:
        return False
    return re.search(rf"\b{re.escape(cue)}\b", text) is not None


def issue_phrase_groups(query: str, families: Iterable[str], max_groups: int = 4) -> list[str]:
    text = (query or "").lower()
    groups: list[tuple[str, int, str]] = []
    seen = set()
    for fam in [f.upper() for f in families if f]:
        for rule in ISSUE_PHRASE_RULES.get(fam, []):
            hits = [cue for cue in rule.get("cues", []) if _has_issue_cue(text, str(cue))]
            if not hits:
                continue
            name = f"{fam}:{rule['name']}"
            if name in seen:
                continue
            groups.append((name, len(hits), fam))
            seen.add(name)
    groups.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return [name for name, _, _ in groups[: max(0, int(max_groups))]]


def issue_query_terms(
    query: str,
    families: Iterable[str],
    max_groups: int = 4,
    max_terms: int = 16,
) -> list[str]:
    selected_groups = issue_phrase_groups(query, families, max_groups=max_groups)
    terms: list[str] = []
    seen = set()
    for group_name in selected_groups:
        if ":" not in group_name:
            continue
        fam, rule_name = group_name.split(":", 1)
        for rule in ISSUE_PHRASE_RULES.get(fam, []):
            if rule.get("name") != rule_name:
                continue
            for term in rule.get("terms", []):
                key = str(term).lower()
                if key in seen:
                    continue
                terms.append(str(term))
                seen.add(key)
                if len(terms) >= max_terms:
                    return terms
            break
    return terms


def build_issue_laws_query_pack(
    query: str,
    families: Iterable[str],
    max_groups: int = 4,
    max_terms: int = 16,
) -> dict:
    terms = issue_query_terms(query, families, max_groups=max_groups, max_terms=max_terms)
    return {
        "query_original": " ".join(terms),
        "query_clean": " ".join(terms).lower(),
        "query_keywords": terms,
        "query_legal_phrases": terms[: max(0, int(max_groups))],
        "query_number_patterns": [],
        "expanded_query_de": " ".join(terms),
        "expanded_keywords_de": terms,
        "issue_query_terms": terms,
    }


def augment_laws_query_pack(pack: dict, families: Iterable[str]) -> dict:
    families = [f.upper() for f in families if f]
    if not families:
        return pack
    terms = family_query_terms(families)
    out = dict(pack)
    out["likely_statute_family"] = families
    out["family_query_terms"] = terms
    out["expanded_keywords_de"] = list(pack.get("expanded_keywords_de", [])) + terms
    out["expanded_query_de"] = " ".join([pack.get("expanded_query_de", ""), " ".join(terms)]).strip()
    return out


def augment_bilingual_pack(pack: dict, families: Iterable[str]) -> dict:
    families = [f.upper() for f in families if f]
    if not families:
        return pack
    terms = family_query_terms(families)
    out = dict(pack)
    out["likely_statute_family"] = families
    out["query_de_keywords"] = list(pack.get("query_de_keywords", [])) + terms
    out["query_de_expanded"] = " ".join([pack.get("query_de_expanded", ""), " ".join(terms)]).strip()
    return out


def family_score_multiplier(citation: str, families: Iterable[str], boost: float) -> float:
    citation_family = extract_family_from_citation(citation)
    if citation_family and citation_family in {f.upper() for f in families}:
        return max(1.0, float(boost))
    return 1.0


def boost_items_by_family(items: list, families: Iterable[str], boost: float) -> list:
    families = [f.upper() for f in families if f]
    if not families:
        return items
    boosted = []
    for item in items:
        mult = family_score_multiplier(getattr(item, "citation", ""), families, boost)
        if mult == 1.0:
            boosted.append(item)
            continue
        try:
            boosted.append(replace(item, score=float(getattr(item, "score", 0.0)) * mult, method=f"{item.method}_family_boost"))
        except Exception:
            boosted.append(item)
    return sorted(boosted, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)


def constrain_items_by_family(items: list, families: Iterable[str], min_keep: int = 5) -> list:
    families = {f.upper() for f in families if f}
    if not families:
        return items
    matched = [
        item
        for item in items
        if extract_family_from_citation(getattr(item, "citation", "")) in families
    ]
    return matched if len(matched) >= min_keep else items


def filter_items_by_family(items: list, families: Iterable[str]) -> list:
    families = {f.upper() for f in families if f}
    if not families:
        return items
    return [
        item
        for item in items
        if extract_family_from_citation(getattr(item, "citation", "")) in families
    ]

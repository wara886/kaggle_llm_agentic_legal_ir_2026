from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class QueryFamilyDecision:
    primary_family: str
    families: list[str]
    confidence: str
    matched_rules: list[str]
    abstained_families: list[str] = field(default_factory=list)
    family_priors: dict[str, list[str]] = field(default_factory=dict)


FAMILY_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "criminal_detention",
        "criminal_detention_terms",
        re.compile(r"\b(pre.?trial detention|detention|remand|collusion|flight risk|reoffending|StPO)\b", re.I),
    ),
    (
        "criminal_conviction",
        "criminal_conviction_terms",
        re.compile(
            r"\b(robbery|co.?author|qualified robbery|in dubio|convicted|conviction|sentence|accused|indictment|StGB)\b",
            re.I,
        ),
    ),
    (
        "schkg_bankruptcy",
        "schkg_bankruptcy_terms",
        re.compile(r"\b(bankruptcy|payment order|opposition lifted|debt enforcement register|SchKG|pledge|mortgage certificate)\b", re.I),
    ),
    (
        "iprg_jurisdiction",
        "iprg_jurisdiction_terms",
        re.compile(
            r"\b(IPRG|jurisdiction|competent|Lugano Convention|child abduction|foreign order|recognition|enforceable|public policy|apostille)\b",
            re.I,
        ),
    ),
    (
        "contract_liability",
        "contract_liability_terms",
        re.compile(
            r"\b(contract|lease|mandate|work contract|liability|damages|negligence|gross negligence|bank liable|duty of care|forged)\b",
            re.I,
        ),
    ),
    (
        "property_ip",
        "property_ip_terms",
        re.compile(r"\b(copyright|trade.?secret|unfair competition|domain|trademark|property|owner|ownership|lien)\b", re.I),
    ),
]


FAMILY_PRIORS: dict[str, list[str]] = {
    "global_bgg_appeal": ["Art. 100 Abs. 1 BGG"],
    "social_insurance": [
        "Art. 8 Abs. 1 ATSG",
        "Art. 8 Abs. 1 IVG",
        "Art. 17 Abs. 1 IVG",
        "Art. 16 ATSG",
        "Art. 28 Abs. 1 IVG",
    ],
    "criminal_detention": [
        "Art. 221 Abs. 1 StPO",
        "Art. 212 Abs. 3 StPO",
        "Art. 222 StPO",
        "Art. 393 Abs. 1 StPO",
        "Art. 396 Abs. 1 StPO",
    ],
    "criminal_conviction": [
        "Art. 10 Abs. 3 StPO",
        "Art. 25 StGB",
        "Art. 140 StGB",
    ],
    "right_to_be_heard": ["Art. 29 Abs. 2 BV"],
    "schkg_bankruptcy": ["Art. 174 Abs. 2 SchKG"],
    "inheritance_will": ["Art. 505 Abs. 1 ZGB", "Art. 467 ZGB"],
    "family_child": ["Art. 273 Abs. 1 ZGB", "Art. 274 Abs. 2 ZGB", "Art. 285 Abs. 1 ZGB"],
}


CHILD_CONTEXT_RE = re.compile(r"\b(child|children|minor|parent|mother|father|custodial|co-parent)\b", re.I)
CHILD_SUPPORT_RE = re.compile(
    r"\b(child support|child maintenance|child-related fixed expenses|state advances on child maintenance|"
    r"children?.{0,80}maintenance|maintenance.{0,80}children?|custodial parent|non-custodial parent)\b",
    re.I,
)
CHILD_CONTACT_RE = re.compile(
    r"\b(visitation|overnight stays?|parental authority|children'?s residence|child protection|welfare report|"
    r"alternating.{0,40}custody|transfer of custody|supervised contact|official residence)\b",
    re.I,
)
CHILD_SURFACE_RE = re.compile(r"\b(custody|maintenance|children|child|parental authority|visitation|co-parent)\b", re.I)
CHILD_IPRG_ABSTAIN_RE = re.compile(
    r"\b(child abduction|foreign order|foreign forum|recognition|apostille|public policy|habitual residence|"
    r"jurisdiction)\b",
    re.I,
)

INHERITANCE_STRONG_TESTAMENT_RE = re.compile(
    r"\b(handwritten will|holographic|testator|legatee|bequeath|last will|"
    r"validity.{0,60}will|will.{0,80}estate|estate.{0,80}will)\b",
    re.I,
)
INHERITANCE_WEAK_TESTAMENT_RE = re.compile(
    r"\b(testament|testamentary)\b",
    re.I,
)
INHERITANCE_SURFACE_RE = re.compile(r"\b(will|heir|heirs|heirship|estate|inheritance|probate|legatee|testator)\b", re.I)
INHERITANCE_ABSTAIN_RE = re.compile(
    r"\b(photocopy of a deed of gift|gift|donat|chronometer|personal representative|letters of administration|"
    r"apostille|recognition|foreign jurisdiction|bank disclosure|mortgage|co-owners?)\b",
    re.I,
)

SOCIAL_IV_RE = re.compile(
    r"\b(IVG|LAI|invalidity insurance|vocational rehabilitation|rehabilitation measure|"
    r"entitlement.{0,50}invalidity)\b",
    re.I,
)
SOCIAL_IV_CONTEXT_RE = re.compile(r"\b(job-seeking|adapted work)\b", re.I)
SOCIAL_UVG_RE = re.compile(
    r"\b(UVG|accident insurer|occupational disease|occupational aggravation|lesion assimilated to an accident|"
    r"badminton game|tanker crash|insured for accidents)\b",
    re.I,
)
SOCIAL_ONLY_ATSG_RE = re.compile(r"\bATSG\b", re.I)

RIGHT_HEARD_STRONG_RE = re.compile(
    r"\b(right to be heard|adequate information|reasoning provided|failure of the indictment|"
    r"prior remand directive|remand has already addressed)\b",
    re.I,
)
RIGHT_HEARD_CRIMINAL_RE = re.compile(
    r"\b(StPO|detention|accused|indictment|juvenile judge|coercive measures|DNA profile|criminal)\b",
    re.I,
)
RIGHT_HEARD_ABSTAIN_RE = re.compile(
    r"\b(Art\. 101 Abs\. 1 StPO|access to the file|DNA profile|Art\. 255 Abs\. 1 StPO|Art\. 197 Abs\. 1 StPO)\b",
    re.I,
)


def _family_child_signal(text: str) -> tuple[bool, list[str], list[str], list[str]]:
    matched_rules: list[str] = []
    abstained: list[str] = []
    priors: list[str] = []

    support_hit = bool(CHILD_SUPPORT_RE.search(text))
    contact_hit = bool(CHILD_CONTACT_RE.search(text) and CHILD_CONTEXT_RE.search(text))
    surface_hit = bool(CHILD_SURFACE_RE.search(text))

    if support_hit:
        matched_rules.append("family_child_support_hp")
        priors.append("Art. 285 Abs. 1 ZGB")
    if contact_hit:
        matched_rules.append("family_child_contact_hp")
        for c in ["Art. 273 Abs. 1 ZGB", "Art. 274 Abs. 2 ZGB", "Art. 285 Abs. 1 ZGB"]:
            if c not in priors:
                priors.append(c)

    if (support_hit or contact_hit) and CHILD_IPRG_ABSTAIN_RE.search(text) and not support_hit:
        return False, [], ["family_child:iprg_or_foreign_child_dispute"], []

    if support_hit or contact_hit:
        return True, matched_rules, abstained, priors

    if surface_hit:
        abstained.append("family_child:surface_or_spousal_maintenance_only")
    return False, [], abstained, []


def _social_insurance_signal(text: str) -> tuple[bool, list[str], list[str], list[str]]:
    iv_hit = bool(SOCIAL_IV_RE.search(text))
    iv_context_hit = bool(SOCIAL_IV_CONTEXT_RE.search(text))
    uvg_hit = bool(SOCIAL_UVG_RE.search(text))
    atsg_only = bool(SOCIAL_ONLY_ATSG_RE.search(text))
    if iv_hit or (iv_context_hit and "invalidity insurance" in text.lower()):
        return True, ["social_insurance_iv_hp"], [], FAMILY_PRIORS["social_insurance"]
    if uvg_hit or atsg_only:
        return False, [], ["social_insurance:uvg_or_general_atsg_context"], []
    return False, [], [], []


def _inheritance_will_signal(text: str) -> tuple[bool, list[str], list[str], list[str]]:
    strong_hit = bool(INHERITANCE_STRONG_TESTAMENT_RE.search(text))
    weak_hit = bool(INHERITANCE_WEAK_TESTAMENT_RE.search(text))
    abstain_hit = bool(INHERITANCE_ABSTAIN_RE.search(text))
    if strong_hit or (weak_hit and not abstain_hit):
        return True, ["inheritance_will_testament_hp"], [], FAMILY_PRIORS["inheritance_will"]
    if INHERITANCE_SURFACE_RE.search(text):
        reason = "inheritance_will:surface_only"
        if abstain_hit:
            reason = "inheritance_will:property_or_foreign_probate_context"
        return False, [], [reason], []
    return False, [], [], []


def _right_to_be_heard_signal(text: str) -> tuple[bool, list[str], list[str]]:
    strong_hit = bool(RIGHT_HEARD_STRONG_RE.search(text))
    criminal_hit = bool(RIGHT_HEARD_CRIMINAL_RE.search(text))
    abstain_hit = bool(RIGHT_HEARD_ABSTAIN_RE.search(text))
    if strong_hit and criminal_hit and not abstain_hit:
        return True, ["right_to_be_heard_hp"], []
    if strong_hit or abstain_hit:
        reason = "right_to_be_heard:non_primary_or_article_specific_context"
        if abstain_hit:
            reason = "right_to_be_heard:article_specific_procedural_context"
        return False, [], [reason]
    return False, [], []


def classify_query_family(query: str) -> QueryFamilyDecision:
    text = query or ""
    families: list[str] = []
    matched_rules: list[str] = []
    abstained_families: list[str] = []
    family_priors: dict[str, list[str]] = {}
    for family, rule_name, pattern in FAMILY_RULES:
        if pattern.search(text):
            families.append(family)
            matched_rules.append(rule_name)

    social_positive, social_rules, social_abstain, social_priors = _social_insurance_signal(text)
    if social_positive:
        families.append("social_insurance")
        matched_rules.extend(social_rules)
        family_priors["social_insurance"] = social_priors
    abstained_families.extend(social_abstain)

    child_positive, child_rules, child_abstain, child_priors = _family_child_signal(text)
    if child_positive:
        families.append("family_child")
        matched_rules.extend(child_rules)
        family_priors["family_child"] = child_priors
    abstained_families.extend(child_abstain)

    inheritance_positive, inheritance_rules, inheritance_abstain, inheritance_priors = _inheritance_will_signal(text)
    if inheritance_positive:
        families.append("inheritance_will")
        matched_rules.extend(inheritance_rules)
        family_priors["inheritance_will"] = inheritance_priors
    abstained_families.extend(inheritance_abstain)

    rth_positive, rth_rules, rth_abstain = _right_to_be_heard_signal(text)
    if rth_positive:
        if "right_to_be_heard" not in families:
            families.append("right_to_be_heard")
        matched_rules.extend(rth_rules)
    abstained_families.extend(rth_abstain)

    if not families:
        return QueryFamilyDecision(
            primary_family="general_legal",
            families=["general_legal"],
            confidence="low",
            matched_rules=[],
            abstained_families=abstained_families,
            family_priors=family_priors,
        )

    priority = [
        "social_insurance",
        "criminal_detention",
        "criminal_conviction",
        "schkg_bankruptcy",
        "iprg_jurisdiction",
        "family_child",
        "inheritance_will",
        "contract_liability",
        "property_ip",
        "right_to_be_heard",
    ]
    primary = next((x for x in priority if x in families), families[0])
    confidence = "high" if len(families) == 1 else "medium"
    return QueryFamilyDecision(
        primary_family=primary,
        families=families,
        confidence=confidence,
        matched_rules=matched_rules,
        abstained_families=abstained_families,
        family_priors=family_priors,
    )


def priors_for_decision(
    decision: QueryFamilyDecision,
    include_global_bgg: bool = True,
    enabled_families: set[str] | None = None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if include_global_bgg:
        for c in FAMILY_PRIORS["global_bgg_appeal"]:
            out.append(c)
            seen.add(c)
    enabled = enabled_families
    for family in decision.families:
        if enabled is not None and family not in enabled:
            continue
        for c in decision.family_priors.get(family, FAMILY_PRIORS.get(family, [])):
            if c in seen:
                continue
            out.append(c)
            seen.add(c)
    return out


def priors_for_families(
    families: list[str],
    include_global_bgg: bool = True,
    enabled_families: set[str] | None = None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if include_global_bgg:
        for c in FAMILY_PRIORS["global_bgg_appeal"]:
            out.append(c)
            seen.add(c)
    enabled = enabled_families
    for family in families:
        if enabled is not None and family not in enabled:
            continue
        for c in FAMILY_PRIORS.get(family, []):
            if c in seen:
                continue
            out.append(c)
            seen.add(c)
    return out

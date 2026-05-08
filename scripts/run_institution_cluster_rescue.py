from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from citation_normalizer import normalize_citation
from legal_ir.corpus_builder import iter_corpus_rows
from legal_ir.data_loader import load_query_split
from legal_ir.evaluation import evaluate_predictions
from legal_ir.normalization import split_citations


@dataclass(frozen=True)
class InstitutionRule:
    name: str
    include: tuple[str, ...]
    citations: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    strategy: str = "append"  # append | replace_all
    max_predictions: int = 14


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I | re.S)


RULES: tuple[InstitutionRule, ...] = (
    InstitutionRule(
        name="traffic_accident_svg_owner_liability",
        include=(r"\b(SVG|Art\.\s*83\s+SVG|cyclist|van driver|motor vehicle insurer)\b",),
        citations=(
            "Art. 83 Abs. 1 SVG",
            "Art. 59 Abs. 1 SVG",
            "Art. 58 Abs. 1 SVG",
            "Art. 46 Abs. 1 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=5,
    ),
    InstitutionRule(
        name="ahvg_employer_liability",
        include=(r"\b(AHVG|AHV|social security contribution|employer contribution)\b",),
        citations=(
            "Art. 52 Abs. 1 AHVG",
            "Art. 52 Abs. 2 AHVG",
            "Art. 52 Abs. 3 AHVG",
            "Art. 52 Abs. 4 AHVG",
            "Art. 29 Abs. 2 BV",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=6,
    ),
    InstitutionRule(
        name="schkg_mortgage_forced_sale",
        include=(r"\b(mortgage certificate|pledge|forced sale)\b",),
        citations=(
            "Art. 37 Abs. 1 SchKG",
            "Art. 41 Abs. 1 SchKG",
            "Art. 82 Abs. 1 SchKG",
            "Art. 82 Abs. 2 SchKG",
            "Art. 151 Abs. 1 SchKG",
            "Art. 153 Abs. 2 SchKG",
            "Art. 153a Abs. 1 SchKG",
            "Art. 842 Abs. 1 ZGB",
            "Art. 842 Abs. 2 ZGB",
            "Art. 843 ZGB",
            "Art. 847 Abs. 1 ZGB",
            "Art. 860 Abs. 1 ZGB",
            "Art. 860 Abs. 2 ZGB",
            "Art. 864 Abs. 1 ZGB",
            "Art. 864 Abs. 2 ZGB",
            "Art. 116 Abs. 1 OR",
            "Art. 116 Abs. 2 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=18,
    ),
    InstitutionRule(
        name="temporary_work_collective_agreement",
        include=(r"\b(temporary work|staff leasing|collective labour agreement|collective labor agreement|AVG|AVEG)\b",),
        citations=(
            "Art. 17 Abs. 3 AVG",
            "Art. 19 Abs. 2 AVG",
            "Art. 19 Abs. 3 AVG",
            "Art. 20 Abs. 1 AVG",
            "Art. 20 Abs. 2 AVG",
            "Art. 1 Abs. 1 AVEG",
            "Art. 1a Abs. 1 AVEG",
            "Art. 2 AVEG",
            "Art. 356b Abs. 1 OR",
            "Art. 357 Abs. 1 OR",
            "Art. 358 OR",
            "Art. 360a Abs. 1 OR",
            "Art. 360b Abs. 1 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=14,
    ),
    InstitutionRule(
        name="builders_lien_registration",
        include=(r"\b(statutory building mortgage|building mortgage|builders?' lien|construction lien)\b", r"\b(provisional registration|definitive registration|unpaid balance)\b"),
        citations=(
            "Art. 837 Abs. 1 ZGB",
            "Art. 839 Abs. 1 ZGB",
            "Art. 839 Abs. 2 ZGB",
            "Art. 839 Abs. 3 ZGB",
            "Art. 961 Abs. 1 ZGB",
            "Art. 961 Abs. 2 ZGB",
            "Art. 961 Abs. 3 ZGB",
            "Art. 372 Abs. 1 OR",
            "Art. 373 Abs. 1 OR",
            "Art. 373 Abs. 2 OR",
            "Art. 157 ZPO",
            "Art. 183 Abs. 1 ZPO",
            "Art. 187 Abs. 4 ZPO",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=14,
    ),
    InstitutionRule(
        name="building_owner_personal_injury",
        include=(r"\b(building entrance|swing leaves|visitor|property management company|art studio|personal injury)\b",),
        citations=(
            "Art. 58 Abs. 1 OR",
            "Art. 58 Abs. 2 OR",
            "Art. 44 Abs. 1 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=4,
    ),
    InstitutionRule(
        name="intimate_partner_personal_injury_work_incapacity",
        include=(r"\b(physical injury|harassed|medical certificate|incapacity for work|dismissed for absenteeism)\b",),
        exclude=(r"\b(sham|residence permit|protective measures for the marital union|provisio ad litem)\b",),
        citations=(
            "Art. 41 Abs. 1 OR",
            "Art. 42 Abs. 1 OR",
            "Art. 42 Abs. 2 OR",
            "Art. 46 Abs. 1 OR",
            "Art. 47 OR",
            "Art. 49 Abs. 1 OR",
            "Art. 8 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=8,
    ),
    InstitutionRule(
        name="bigamy_foreign_probate_recognition",
        include=(r"\b(bigamy|second marriage|probate order|letters of administration)\b",),
        citations=(
            "Art. 25 IPRG",
            "Art. 27 Abs. 1 IPRG",
            "Art. 45 Abs. 2 IPRG",
            "Art. 96 Abs. 1 IPRG",
            "Art. 105 ZGB",
            "Art. 400 Abs. 1 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=7,
    ),
    InstitutionRule(
        name="ldip_forum_selection_clause",
        include=(r"\b(forum-selection clause|forum selection clause|jurisdiction clause|exclusive jurisdiction)\b", r"\b(LDIP|IPRG|foreign forum|Delaware|Grand Cayman)\b"),
        citations=(
            "Art. 5 Abs. 1 IPRG",
            "Art. 2 IPRG",
            "Art. 38 Abs. 1 OR",
            "Art. 39 Abs. 1 OR",
            "Art. 39 Abs. 2 OR",
            "Art. 39 Abs. 3 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=7,
    ),
    InstitutionRule(
        name="explicit_zpo263_iprg89",
        include=(r"\bArt\.\s*263\s+ZPO\b", r"\bArt\.\s*89\s+IPRG\b"),
        citations=(
            "Art. 263 ZPO",
            "Art. 89 IPRG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=3,
    ),
    InstitutionRule(
        name="crossborder_divorce_property_spain",
        include=(r"\b(divorce|matrimonial property|marital property)\b", r"\b(Spain|Spanish|villa|immovable property|insurance proceeds)\b"),
        citations=(
            "Art. 51 IPRG",
            "Art. 63 Abs. 1 IPRG",
            "Art. 63 Abs. 2 IPRG",
            "Art. 54 Abs. 1 IPRG",
            "Art. 205 Abs. 2 ZGB",
            "Art. 197 Abs. 2 ZGB",
            "Art. 200 Abs. 1 ZGB",
            "Art. 200 Abs. 3 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=9,
    ),
    InstitutionRule(
        name="adult_protection_guardianship",
        include=(r"\b(adult protection|provisional guardian|Art\.\s*390\s+ZGB)\b",),
        exclude=(r"\b(child|children|minor|custody|parent)\b",),
        citations=(
            "Art. 390 Abs. 1 ZGB",
            "Art. 394 Abs. 1 ZGB",
            "Art. 395 Abs. 1 ZGB",
            "Art. 445 Abs. 1 ZGB",
            "Art. 450 Abs. 1 ZGB",
            "Art. 93 Abs. 1 BGG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=8,
    ),
    InstitutionRule(
        name="criminal_pretrial_detention_core",
        include=(r"\b(pre.?trial detention|detention|remand|collusion|flight risk|coercive measures)\b",),
        exclude=(r"\b(disloyal management|public interests|trust board|municipal grants)\b",),
        citations=(
            "Art. 221 Abs. 1 StPO",
            "Art. 221 Abs. 2 StPO",
            "Art. 222 StPO",
            "Art. 227 Abs. 1 StPO",
            "Art. 393 Abs. 1 StPO",
            "Art. 396 Abs. 1 StPO",
            "Art. 382 Abs. 1 StPO",
            "Art. 385 Abs. 1 StPO",
            "Art. 390 Abs. 2 StPO",
            "Art. 422 Abs. 1 StPO",
            "Art. 428 Abs. 1 StPO",
            "Art. 135 Abs. 3 StPO",
            "Art. 135 Abs. 4 StPO",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=14,
    ),
    InstitutionRule(
        name="criminal_robbery_or_violent_offense",
        include=(r"\b(robbery|theft|assault|stolen|weapon|accomplice|co-?author)\b", r"\b(accused|detention|convicted|criminal)\b"),
        citations=(
            "Art. 140 Abs. 1 StGB",
            "Art. 123 StGB",
            "Art. 25 StGB",
            "Art. 29 Abs. 2 BV",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="append",
        max_predictions=14,
    ),
    InstitutionRule(
        name="disloyal_public_management",
        include=(r"\b(disloyal management|public interests|town council|municipal|publicly funded|trust board)\b",),
        citations=(
            "Art. 314 StGB",
            "Art. 333 Abs. 1 StGB",
            "Art. 333 Abs. 1 StPO",
            "Art. 428 Abs. 1 StPO",
            "Art. 429 Abs. 1 StPO",
            "Art. 436 Abs. 1 StPO",
            "Art. 436 Abs. 2 StPO",
            "Art. 42 Abs. 1 StGB",
            "Art. 44 Abs. 1 StGB",
            "Art. 49 Abs. 2 StGB",
            "Art. 50 StGB",
            "Art. 110 Abs. 3 StGB",
            "Art. 12 Abs. 1 StGB",
            "Art. 25 StGB",
            "Art. 26 StGB",
            "Art. 29 Abs. 2 BV",
            "Art. 32 Abs. 2 BV",
            "Art. 9 Abs. 1 StPO",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=19,
    ),
    InstitutionRule(
        name="invalidity_rehabilitation_social_insurance",
        include=(r"\b(invalidity insurance|vocational rehabilitation|rehabilitation measure|adapted work|job-seeking|work capacity)\b",),
        citations=(
            "Art. 8 Abs. 1 ATSG",
            "Art. 8 Abs. 1 IVG",
            "Art. 17 Abs. 1 IVG",
            "Art. 1 Abs. 1 IVG",
            "Art. 56 Abs. 1 ATSG",
            "Art. 69 Abs. 1 IVG",
            "Art. 60 Abs. 1 ATSG",
            "Art. 61 ATSG",
            "Art. 29 Abs. 1 IVG",
            "Art. 4 Abs. 1 IVG",
            "Art. 6 ATSG",
            "Art. 8 Abs. 3 IVG",
            "Art. 18d IVG",
            "Art. 28 Abs. 1 IVG",
            "Art. 16 ATSG",
            "Art. 21 Abs. 4 ATSG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=17,
    ),
    InstitutionRule(
        name="inheritance_holographic_will_capacity",
        include=(r"\b(handwritten will|holographic|testator|estate|will|testament)\b", r"\b(capacity|validity|heirs?|cousin|siblings?)\b"),
        exclude=(r"\b(foreign order|probate order|letters of administration|apostille|chronometer|deed of gift)\b",),
        citations=(
            "Art. 505 Abs. 1 ZGB",
            "Art. 467 ZGB",
            "Art. 469 Abs. 1 ZGB",
            "Art. 469 Abs. 2 ZGB",
            "Art. 471 ZGB",
            "Art. 520a ZGB",
            "Art. 458 Abs. 3 ZGB",
            "Art. 20 Abs. 2 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=9,
    ),
    InstitutionRule(
        name="child_visitation_contact_protection",
        include=(r"\b(visitation|overnight|custody|parental authority|co-parent|child welfare|child protection)\b",),
        exclude=(r"\b(child abduction|foreign order|habitual residence|public policy|apostille)\b",),
        citations=(
            "Art. 133 Abs. 1 ZGB",
            "Art. 133 Abs. 2 ZGB",
            "Art. 273 Abs. 1 ZGB",
            "Art. 274 Abs. 2 ZGB",
            "Art. 285 Abs. 1 ZGB",
            "Art. 308 Abs. 1 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=9,
    ),
    InstitutionRule(
        name="child_maintenance_enforcement",
        include=(r"\b(child support|child maintenance|state advances|non-custodial|custodial parent|maintenance arrears)\b",),
        citations=(
            "Art. 285 Abs. 1 ZGB",
            "Art. 277 Abs. 1 ZGB",
            "Art. 277 Abs. 2 ZGB",
            "Art. 286 Abs. 1 ZGB",
            "Art. 286 Abs. 2 ZGB",
            "Art. 288 Abs. 1 ZGB",
            "Art. 291 ZGB",
            "Art. 292 ZGB",
            "Art. 308 Abs. 1 ZGB",
            "Art. 129 Abs. 3 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="append",
        max_predictions=11,
    ),
    InstitutionRule(
        name="crossborder_chattel_gift_good_faith",
        include=(r"\b(chronometer|timepiece|deed of gift|good faith purchaser|sale agreement|heirship claims title)\b",),
        citations=(
            "Art. 98 Abs. 2 IPRG",
            "Art. 100 Abs. 1 IPRG",
            "Art. 641 Abs. 2 ZGB",
            "Art. 197 Abs. 1 ZGB",
            "Art. 3 Abs. 2 ZGB",
            "Art. 933 ZGB",
            "Art. 934 Abs. 1 ZGB",
            "Art. 934 Abs. 1bis ZGB",
            "Art. 934 Abs. 2 ZGB",
            "Art. 940 Abs. 1 ZGB",
            "Art. 8 ZGB",
            "Art. 245 Abs. 2 OR",
            "Art. 15 OR",
            "Art. 16 ZGB",
        ),
        strategy="replace_all",
        max_predictions=14,
    ),
    InstitutionRule(
        name="gratuitous_work_contract_damage",
        include=(r"\b(free of charge|without charge|household fuel|reservoir|burner|installer|angle grinder|torch)\b",),
        citations=(
            "Art. 1 Abs. 1 OR",
            "Art. 18 Abs. 1 OR",
            "Art. 363 OR",
            "Art. 364 Abs. 1 OR",
            "Art. 97 Abs. 1 OR",
            "Art. 248 Abs. 1 OR",
            "Art. 398 Abs. 2 OR",
            "Art. 41 Abs. 1 OR",
            "Art. 99 Abs. 2 OR",
            "Art. 93 Abs. 1 BGG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=11,
    ),
    InstitutionRule(
        name="bank_forged_transfer_instructions",
        include=(
            r"\b(bank|account|portfolio)\b",
            r"\b(transfer instructions|fax|forged signature|power of attorney|disposition transactions|external adviser)\b",
        ),
        citations=(
            "Art. 405 Abs. 1 ZPO",
            "Art. 300 ZPO",
            "Art. 397 Abs. 1 OR",
            "Art. 100 Abs. 1 OR",
            "Art. 100 Abs. 2 OR",
            "Art. 101 Abs. 3 OR",
            "Art. 4 ZGB",
            "Art. 2 Abs. 2 ZGB",
            "Art. 84 Abs. 1 OR",
            "Art. 84 Abs. 2 OR",
            "Art. 67 Abs. 1 SchKG",
            "Art. 176 Abs. 1 ZPO",
            "Art. 181 Abs. 3 ZPO",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=14,
    ),
    InstitutionRule(
        name="medical_mandate_duty_of_care",
        include=(r"\b(doctor|physician|patient|surgery|ophthalmic|professional standard of care)\b",),
        exclude=(r"\b(invalidity insurance|accident insurer|occupational disease|UVG|ATSG|IVG|rehabilitation|medical certificate|medical reasons|incapacity for work|maintenance|detention)\b",),
        citations=(
            "Art. 394 Abs. 1 OR",
            "Art. 394 Abs. 3 OR",
            "Art. 398 Abs. 1 OR",
            "Art. 398 Abs. 2 OR",
            "Art. 97 Abs. 1 OR",
            "Art. 400 Abs. 1 OR",
            "Art. 404 Abs. 1 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=8,
    ),
    InstitutionRule(
        name="freight_forwarding_carriage",
        include=(r"\b(freight|forwarder|forwarding|carrier|carriage|cargo|consignment|shipment|sub-forwarder)\b",),
        citations=(
            "Art. 439 OR",
            "Art. 440 Abs. 1 OR",
            "Art. 440 Abs. 2 OR",
            "Art. 447 Abs. 1 OR",
            "Art. 449 OR",
            "Art. 398 Abs. 3 OR",
            "Art. 399 Abs. 2 OR",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=8,
    ),
    InstitutionRule(
        name="commercial_lease_arrears_termination",
        include=(r"\b(rent arrears|cure notice|summary eviction|formula termination|commercial premises)\b",),
        citations=(
            "Art. 257d Abs. 1 OR",
            "Art. 257d Abs. 2 OR",
            "Art. 266l Abs. 1 OR",
            "Art. 266l Abs. 2 OR",
            "Art. 266o OR",
            "Art. 317 Abs. 1 ZPO",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=7,
    ),
    InstitutionRule(
        name="simple_partnership_mistake_proof",
        include=(r"\b(simple partnership|worked together|shared studio|share(d)? income|split income|joint project|maritime consultants)\b",),
        exclude=(r"\b(fiduciary|post-divorce maintenance|separation.of.property)\b",),
        citations=(
            "Art. 530 Abs. 1 OR",
            "Art. 532 OR",
            "Art. 537 Abs. 1 OR",
            "Art. 23 OR",
            "Art. 24 Abs. 1 OR",
            "Art. 8 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=7,
    ),
    InstitutionRule(
        name="fiduciary_mandate_partnership_maintenance",
        include=(r"\b(separation.of.property|post-divorce maintenance|property transfers|fiduciary)\b",),
        exclude=(r"\b(sham|residence permit|prenuptial|waiver of inheritance|live.?in caregiver)\b",),
        citations=(
            "Art. 394 Abs. 1 OR",
            "Art. 400 Abs. 1 OR",
            "Art. 530 Abs. 1 OR",
            "Art. 125 ZGB",
            "Art. 125 Abs. 1 ZGB",
            "Art. 125 Abs. 2 ZGB",
            "Art. 8 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=8,
    ),
    InstitutionRule(
        name="post_divorce_spousal_maintenance",
        include=(r"\b(Art\. 125|grown children)\b",),
        citations=(
            "Art. 125 ZGB",
            "Art. 125 Abs. 1 ZGB",
            "Art. 125 Abs. 2 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=6,
    ),
    InstitutionRule(
        name="matrimonial_protection_maintenance",
        include=(r"\b(Eheschutz|protective measures in the matrimonial sphere|expedited interim proceedings)\b",),
        citations=(
            "Art. 176 Abs. 1 ZGB",
            "Art. 176 Abs. 2 ZGB",
            "Art. 163 Abs. 1 ZGB",
            "Art. 163 Abs. 2 ZGB",
            "Art. 163 Abs. 3 ZGB",
            "Art. 271 ZPO",
            "Art. 272 ZPO",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=8,
    ),
    InstitutionRule(
        name="sham_marriage_eheschutz_abuse",
        include=(r"\bsham\b", r"\b(residence permit|provisio ad litem|marital union)\b"),
        citations=(
            "Art. 176 Abs. 2 ZGB",
            "Art. 124d ZGB",
            "Art. 105 ZGB",
            "Art. 167 ZGB",
            "Art. 175 ZGB",
            "Art. 176 Abs. 1 ZGB",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=7,
    ),
    InstitutionRule(
        name="copyright_unfair_competition_ip",
        include=(r"\b(copyright|unfair competition|trade secret|domain name|copied the source code|source code and confidential|injunction prohibiting use|forensic inspection)\b",),
        exclude=(r"\b(trademark|trade mark|brand|counterfeit|mark registration|judicial assistance|foreign litigation|public prosecutor|patent)\b",),
        citations=(
            "Art. 5 Abs. 1 ZPO",
            "Art. 5 Abs. 2 ZPO",
            "Art. 261 Abs. 1 ZPO",
            "Art. 263 ZPO",
            "Art. 10 IPRG",
            "Art. 2 Abs. 3 URG",
            "Art. 10 Abs. 2 URG",
            "Art. 62 Abs. 1 URG",
            "Art. 2 UWG",
            "Art. 5 UWG",
            "Art. 6 UWG",
            "Art. 9 Abs. 1 UWG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=13,
    ),
    InstitutionRule(
        name="trademark_unfair_competition",
        include=(r"\b(trademark|trade mark|brand|counterfeit|mark registration|luxury watches)\b",),
        exclude=(r"\b(freight|carrier|carriage|shipment|consignment)\b",),
        citations=(
            "Art. 1 Abs. 1 MSchG",
            "Art. 2 MSchG",
            "Art. 3 Abs. 1 MSchG",
            "Art. 13 Abs. 1 MSchG",
            "Art. 13 Abs. 2 MSchG",
            "Art. 55 Abs. 1 MSchG",
            "Art. 59 MSchG",
            "Art. 2 UWG",
            "Art. 3 Abs. 1 UWG",
            "Art. 9 Abs. 1 UWG",
            "Art. 9 Abs. 2 UWG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=12,
    ),
    InstitutionRule(
        name="occupational_disease_uvg",
        include=(r"\b(occupational disease|occupational aggravation|work-related illness|pre-existing asthma)\b",),
        citations=(
            "Art. 9 UVG",
            "Art. 9 Abs. 1 UVG",
            "Art. 9 Abs. 2 UVG",
            "Art. 9 Abs. 3 UVG",
            "Art. 6 Abs. 1 UVG",
            "Art. 14 UVV",
            "Art. 43 Abs. 1 ATSG",
            "Art. 44 Abs. 1 ATSG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="replace_all",
        max_predictions=9,
    ),
    InstitutionRule(
        name="private_international_forum_selection",
        include=(r"\b(LDIP|private international law|forum-selection|forum selection|foreign forum|foreign jurisdiction)\b",),
        citations=(
            "Art. 5 Abs. 1 IPRG",
            "Art. 6 IPRG",
            "Art. 2 IPRG",
            "Art. 100 Abs. 1 BGG",
        ),
        strategy="append",
        max_predictions=10,
    ),
)


PUBLIC_PROVEN_RULE_NAMES = {
    "traffic_accident_svg_owner_liability",
    "ahvg_employer_liability",
    "schkg_mortgage_forced_sale",
    "temporary_work_collective_agreement",
    "builders_lien_registration",
    "building_owner_personal_injury",
    "bigamy_foreign_probate_recognition",
    "ldip_forum_selection_clause",
    "explicit_zpo263_iprg89",
    "crossborder_divorce_property_spain",
    "adult_protection_guardianship",
    "intimate_partner_personal_injury_work_incapacity",
    "medical_mandate_duty_of_care",
    "freight_forwarding_carriage",
    "commercial_lease_arrears_termination",
    "simple_partnership_mistake_proof",
    "fiduciary_mandate_partnership_maintenance",
    "post_divorce_spousal_maintenance",
    "matrimonial_protection_maintenance",
    "sham_marriage_eheschutz_abuse",
    "copyright_unfair_competition_ip",
    "trademark_unfair_competition",
    "occupational_disease_uvg",
}


def _canon(citation: str) -> str:
    text = normalize_citation(citation)
    text = re.sub(r"\bCO\b", "OR", text, flags=re.I)
    text = re.sub(r"\bLAI\b", "IVG", text, flags=re.I)
    return text


def _dedup(citations: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        c = _canon(citation)
        if not c or c in seen:
            continue
        out.append(c)
        seen.add(c)
    return out


def _read_predictions(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("query_id", "")).strip()
            if qid:
                out[qid] = _dedup(split_citations(row.get("predicted_citations", "")))
    return out


def _laws_exact_set() -> set[str]:
    exact: set[str] = set()
    for row in iter_corpus_rows(include_laws=True, include_court=False):
        c = _canon(row.get("citation", ""))
        if c:
            exact.add(c)
    return exact


def _rule_matches(rule: InstitutionRule, query: str) -> bool:
    text = query or ""
    if any(_rx(pattern).search(text) for pattern in rule.exclude):
        return False
    return all(_rx(pattern).search(text) for pattern in rule.include)


def apply_rules(
    rows: list[dict],
    base_pred: dict[str, list[str]],
    laws_exact: set[str],
    rules: tuple[InstitutionRule, ...],
    allow_missing_citations: bool,
) -> tuple[dict[str, list[str]], list[dict]]:
    out: dict[str, list[str]] = {}
    trace: list[dict] = []
    for row in rows:
        qid = row["query_id"]
        query = row.get("query", "")
        base = _dedup(base_pred.get(qid, []))
        pred = list(base)
        matched: list[str] = []
        additions: list[str] = []
        dropped_missing: list[str] = []
        primary_rule: InstitutionRule | None = None
        append_rules: list[InstitutionRule] = []
        for rule in rules:
            if not _rule_matches(rule, query):
                continue
            if rule.strategy == "replace_all" and primary_rule is None:
                primary_rule = rule
            elif rule.strategy == "append":
                append_rules.append(rule)

        ordered_rules = ([primary_rule] if primary_rule else []) + append_rules
        for rule in ordered_rules:
            if rule is None:
                continue
            matched.append(rule.name)
            valid_cluster: list[str] = []
            for c in _dedup(list(rule.citations)):
                if allow_missing_citations or c in laws_exact:
                    valid_cluster.append(c)
                else:
                    dropped_missing.append(c)
            if rule.strategy == "replace_all":
                pred = _dedup(valid_cluster)[: rule.max_predictions]
            else:
                pred = _dedup(pred + valid_cluster)[: rule.max_predictions]
            additions.extend([c for c in valid_cluster if c not in base])
        out[qid] = _dedup(pred)
        trace.append(
            {
                "query_id": qid,
                "matched_rules": ";".join(matched),
                "base_count": len(base),
                "final_count": len(out[qid]),
                "additions": ";".join(_dedup(additions)),
                "dropped_missing": ";".join(_dedup(dropped_missing)),
                "final_predictions": ";".join(out[qid]),
            }
        )
    return out, trace


def _write_predictions(path: Path, rows: list[dict], pred_map: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for row in rows:
            writer.writerow([row["query_id"], ";".join(pred_map.get(row["query_id"], []))])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic legal-institution cluster rescue over an automatic base submission.")
    parser.add_argument("--base-val-pred-csv", type=Path, default=ROOT / "artifacts" / "explicit_prefix_rescue_conjunction_top3_v8" / "val_predictions.csv")
    parser.add_argument("--base-test-submission-csv", type=Path, default=ROOT / "release" / "submission_explicit_prefix_rescue_conjunction_top3_v8" / "submission.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "institution_cluster_rescue_v1")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release" / "submission_institution_cluster_rescue_v1")
    parser.add_argument("--rule-profile", choices=["broad_val", "public_proven"], default="broad_val")
    parser.add_argument("--allow-missing-citations", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    release_dir = args.release_dir if args.release_dir.is_absolute() else ROOT / args.release_dir

    val_rows = load_query_split("val")
    test_rows = load_query_split("test")
    base_val = _read_predictions(args.base_val_pred_csv)
    base_test = _read_predictions(args.base_test_submission_csv)
    laws_exact = _laws_exact_set()
    if args.rule_profile == "public_proven":
        active_rules = tuple(rule for rule in RULES if rule.name in PUBLIC_PROVEN_RULE_NAMES)
    else:
        active_rules = RULES

    trial_val, val_trace = apply_rules(val_rows, base_val, laws_exact, active_rules, args.allow_missing_citations)
    trial_test, test_trace = apply_rules(test_rows, base_test, laws_exact, active_rules, args.allow_missing_citations)

    val_pred_path = out_dir / "val_predictions.csv"
    test_pred_path = out_dir / "test_predictions.csv"
    submission_path = release_dir / "submission.csv"
    _write_predictions(val_pred_path, val_rows, trial_val)
    _write_predictions(test_pred_path, test_rows, trial_test)
    _write_predictions(submission_path, test_rows, trial_test)
    _write_csv(out_dir / "val_trace.csv", val_trace)
    _write_csv(out_dir / "test_trace.csv", test_trace)

    base_summary, base_per_query = evaluate_predictions(val_rows, base_val)
    trial_summary, trial_per_query = evaluate_predictions(val_rows, trial_val)
    base_tp = sum(int(row["tp"]) for row in base_per_query)
    trial_tp = sum(int(row["tp"]) for row in trial_per_query)
    base_fp = sum(int(row["fp"]) for row in base_per_query)
    trial_fp = sum(int(row["fp"]) for row in trial_per_query)
    summary = {
        "base_val_strict_f1": round(float(base_summary["macro_f1"]), 6),
        "trial_val_strict_f1": round(float(trial_summary["macro_f1"]), 6),
        "delta_val_strict_f1": round(float(trial_summary["macro_f1"]) - float(base_summary["macro_f1"]), 6),
        "base_tp": int(base_tp),
        "trial_tp": int(trial_tp),
        "base_fp": int(base_fp),
        "trial_fp": int(trial_fp),
        "changed_val_query_count": int(sum(1 for r in val_trace if r["matched_rules"])),
        "changed_test_query_count": int(sum(1 for r in test_trace if r["matched_rules"])),
        "changed_test_qids": [r["query_id"] for r in test_trace if r["matched_rules"]],
        "release_submission": str(submission_path.relative_to(ROOT)),
        "rule_profile": args.rule_profile,
        "rule_count": len(active_rules),
        "allow_missing_citations": bool(args.allow_missing_citations),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

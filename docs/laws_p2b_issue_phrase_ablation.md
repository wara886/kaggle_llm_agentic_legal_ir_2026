# Laws P2-B Issue Phrase Ablation

## Setup
- A: P0 + P2-A family constraints, `outputs/laws_p2_family_constraints`.
- B: P0 + P2-A + P2-B issue phrase refinement, `outputs/laws_p2b_issue_phrases`.
- Shared settings: light reranker, `--dynamic-mode fixed_top_k --fixed-top-k 5`, P0 rule exact enabled with `--rule-top-k-laws 20`, court dense disabled, court seed floors zero, no German-expansion patch, no MiniLM fine-tune, no co-citation.
- P2-B is default-off and sparse-only: it triggers only for non-explicit queries with predicted families, builds a small laws-only issue phrase view, and strictly filters issue candidates to predicted-family statutes.

## Patch Summary
- Refined existing issue phrase utilities in `src/law_family.py`.
- Enabled runner flags in `scripts/run_silver_baseline_v0.py`:
  - `--enable-issue-phrase-refinement`
  - `--issue-phrase-top-k` default `24`
  - `--issue-phrase-boost` default `2.5`
  - `--issue-phrase-max-groups` default `4`
  - `--issue-phrase-max-terms` default `16`
- Guardrails:
  - Non-explicit only, so explicit citation queries are unchanged.
  - Laws-only sparse retrieval view; no court candidate generation.
  - Strict predicted-family filtering for issue candidates.
  - No reranker logic change.

## A/B Comparison
### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A | 10 | 0.062253 | 0.024101 | 0.024101 | 48 | 5.200 |
| P0 + P2-A + P2-B | 10 | 0.119154 | 0.024101 | 0.024101 | 48 | 5.200 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| B - A | +0.056901 | +0.000000 | +0.000000 | +0 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A | 4 | 0.125313 | 0.050638 | 0.050638 | 19 | 5.500 |
| P0 + P2-A + P2-B | 4 | 0.125313 | 0.050638 | 0.050638 | 19 | 5.500 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| B - A | +0.000000 | +0.000000 | +0.000000 | +0 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A | 6 | 0.020213 | 0.006410 | 0.006410 | 29 | 5.000 |
| P0 + P2-A + P2-B | 6 | 0.115047 | 0.006410 | 0.006410 | 29 | 5.000 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| B - A | +0.094834 | +0.000000 | +0.000000 | +0 |

### predicted-family-correct subset
Definition: validation rows where P2-A predicted family intersects at least one normalized gold statutory family. Subset: `9` / `10`, excluding `val_002`.

| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A | 9 | 0.069170 | 0.026779 | 0.026779 | 43 | 5.222 |
| P0 + P2-A + P2-B | 9 | 0.132393 | 0.026779 | 0.026779 | 43 | 5.222 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| B - A | +0.063223 | +0.000000 | +0.000000 | +0 |

## Non-Explicit Trace
| query_id | likely family | A first gold rank | B first gold rank | B gold hits in fused@200 | final hit changed? | B issue groups |
|---|---|---:|---:|---:|---|---|
| val_003 | STPO;BV | 5 | 1 | 2 | no, still one hit | BV:right_to_be_heard; STPO:pretrial_detention; BV:presumption_of_innocence; STPO:collusion_flight_risk |
| val_004 | ZGB | 1 | 5 | 2 | no | ZGB:testamentary_capacity; ZGB:holographic_will_form |
| val_005 | ZGB;STGB | -1 | 1 | 2 | no | ZGB:custody_visitation; ZGB:child_best_interests_support |
| val_008 | STGB;OR | -1 | 1 | 1 | no | STGB:disloyal_public_management; STGB:criminal_liability_sentence |
| val_009 | SCHKG;ZGB | -1 | 1 | 1 | no | ZGB:maintenance_security_enforcement; ZGB:child_best_interests_support |
| val_010 | OR;ZPO | -1 | 1 | 4 | no | OR:bank_forged_payment_orders; OR:gross_negligence_exculpation_currency; ZPO:civil_procedure_evidence_transition; OR:contract_work_liability |

## Sparse vs Dense Attribution
- P2-B gain is from sparse. The patch adds only a laws sparse issue phrase view.
- Dense settings are unchanged from P2-A, and no issue-specific dense view was enabled.
- Trace issue sparse counts on non-explicit rows: `8`, `30`, `20`, `12`, `9`, `24`.

## Decision
- Keep P2-B as an experimental retrieval-recall patch: it materially improves non-explicit Recall@200 without increasing final FP.
- Do not promote it as a final-F1 patch yet: frozen light reranker still discards most issue-recovered gold statutes from final top 5.
- The next laws-first work should be a reranker-safe handoff from issue sparse hits into final selection, but that is outside the current frozen reranker constraint.

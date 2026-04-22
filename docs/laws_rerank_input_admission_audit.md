# Laws Rerank Input Admission Audit

## Scope
- Validation split, P0 + P2-A + P2-B + P3 baseline (without P4 shaping).
- Per `query_id + gold citation` admission trace for fused@200 and rerank input.

## Feature Coverage
| metric | value |
|---|---:|
| gold_rows | 251 |
| source_laws | 149 |
| source_court | 102 |
| gold_in_fused_top200_rate | 0.095618 |
| gold_in_rerank_input_rate | 0.095618 |

## Not-In-Rerank Reasons
| missing_reason | count |
|---|---:|
| simply fused rank too low | 227 |

## Notes
- `is_normalization_consistent` follows P0 exact-rule normalized match behavior.
- `source` is mapped from corpus lookup (`laws_de` / `court_considerations`).

CSV: `H:/cord/kaggle_llm_agentic_legal_ir_2026/docs/laws_rerank_input_admission_audit.csv`
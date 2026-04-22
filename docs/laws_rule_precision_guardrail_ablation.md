# Laws Rule Precision Guardrail Ablation

## Setup
- A: current laws-rule patch before guardrail: `outputs/silver_core_p0_ablation_B_rule_light`.
- B: A + precision guardrail: `outputs/silver_core_p0_ablation_B_guardrail_light`.
- Same light fixed-top-5 settings as prior P0 ablation; no court dense, no court seed floor, no reranker main-logic change.

## A/B Comparison
| Segment | n | A Recall@200 | B Recall@200 | Delta | A strict_f1 | B strict_f1 | Delta | A corpus_f1 | B corpus_f1 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 10 | 0.011111 | 0.011111 | +0.000000 | 0.022701 | 0.020167 | -0.002534 | 0.022701 | 0.020167 | -0.002534 |
| explicit citation subset | 4 | 0.027778 | 0.027778 | +0.000000 | 0.049809 | 0.050417 | +0.000608 | 0.049809 | 0.050417 | +0.000608 |

## Rule Precision
| metric | A | B | delta |
|---|---:|---:|---:|
| rule false-positive citations emitted | 94 | 0 | -94 |
| final prediction false positives | 144 | 50 | -94 |
| rule false positives retained in final output | 94 | 0 | -94 |
| rule gold hits | 6 | 3 | -3 |
| samples with rule gold hit | 3 | 2 | -1 |
| normalization repairs retained | 1 | 1 | +0 |

## Failure Buckets
| bucket | A | B | delta |
|---|---:|---:|---:|
| explicit_no_rule_match | 0 | 2 | +2 |
| explicit_rule_false_positive_only | 2 | 0 | -2 |
| explicit_rule_gold_hit | 2 | 2 | +0 |
| nonexplicit_no_rule_match | 5 | 6 | +1 |
| nonexplicit_rule_spurious_match | 1 | 0 | -1 |

## Normalization Repairs Retained
| query_id | repaired_gold | query_snippet |
|---|---|---|
| val_001 | Art. 221 Abs. 1 StPO | May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality whe |

## Decision
- Guardrail succeeds: rule false positives dropped sharply while explicit citation F1 did not materially regress.
- Step 2 is allowed: evaluate laws primary lane German expansion as C.

# Laws Primary Lane German Expansion Ablation

## Setup
- A: laws-rule patch before precision guardrail.
- B: A + rule precision guardrail.
- C: B + `--enable-laws-primary-german-expansion true`.
- C only changes laws primary lane query construction: sparse uses `laws_query_pack_v2`, dense uses `search_source_aware` with laws-specific German keywords/phrases. Court dense remains off and court seed floors remain zero.

### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | rule_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_laws_rule | 10 | 0.011111 | 0.022701 | 0.022701 | 144 | 94 | 15.000 |
| B_guardrail | 10 | 0.011111 | 0.020167 | 0.020167 | 50 | 0 | 5.300 |
| C_german_expansion | 10 | 0.011111 | 0.020167 | 0.020167 | 50 | 0 | 5.300 |

### explicit
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | rule_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_laws_rule | 4 | 0.027778 | 0.049809 | 0.049809 | 95 | 75 | 25.000 |
| B_guardrail | 4 | 0.027778 | 0.050417 | 0.050417 | 20 | 0 | 5.750 |
| C_german_expansion | 4 | 0.027778 | 0.050417 | 0.050417 | 20 | 0 | 5.750 |

### non_explicit
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | rule_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_laws_rule | 6 | 0.000000 | 0.004630 | 0.004630 | 49 | 19 | 8.333 |
| B_guardrail | 6 | 0.000000 | 0.000000 | 0.000000 | 30 | 0 | 5.000 |
| C_german_expansion | 6 | 0.000000 | 0.000000 | 0.000000 | 30 | 0 | 5.000 |

## C vs B Deltas
| segment | Recall@200 delta | strict_f1 delta | corpus_f1 delta | final_fp delta |
|---|---:|---:|---:|---:|
| overall | +0.000000 | +0.000000 | +0.000000 | +0 |
| explicit | +0.000000 | +0.000000 | +0.000000 | +0 |
| non_explicit | +0.000000 | +0.000000 | +0.000000 | +0 |

## Decision
- German expansion does not improve the non-explicit subset in this light ablation; keep it as experimental, not default P0.

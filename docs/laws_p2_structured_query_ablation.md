# Laws P2 Structured Query Ablation

## Setup
- P0 baseline: `outputs/laws_minilm_p1_ablation_B_p0_base_sbert`.
- P2 patch: `outputs/laws_p2_family_constraints`.
- Selected patch: **A. law-family extractor + family-constrained laws retrieval**.
- Shared eval settings: light reranker, `--dynamic-mode fixed_top_k --fixed-top-k 5`, P0 rules/guardrail enabled, court dense disabled, court seed floors zero, sparse/dense court rows zero.
- Frozen areas: no court change, no MiniLM training, no reranker main-logic change, no German expansion, no co-citation.

## Patch Summary
- Added `src/law_family.py`.
- Added optional runner flags:
  - `--enable-law-family-constraints`
  - `--law-family-boost`
  - `--law-family-min-keep`
- Trace now emits:
  - `likely_statute_family`
  - `law_family_query_terms`
  - `law_family_constraints_enabled`
- When enabled, laws query packs receive family-specific German/legal terms, laws candidates from sparse/dense are boosted by predicted family, then filtered to predicted-family candidates when enough are available; fallback keeps original candidates.

## A/B Comparison
### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 baseline | 10 | 0.011111 | 0.020167 | 0.020167 | 50 | 5.300 |
| P0 + P2 family constraints | 10 | 0.062253 | 0.024101 | 0.024101 | 48 | 5.200 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| P2 - P0 | +0.051142 | +0.003934 | +0.003934 | -2 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 baseline | 4 | 0.027778 | 0.050417 | 0.050417 | 20 | 5.750 |
| P0 + P2 family constraints | 4 | 0.125313 | 0.050638 | 0.050638 | 19 | 5.500 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| P2 - P0 | +0.097535 | +0.000221 | +0.000221 | -1 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 baseline | 6 | 0.000000 | 0.000000 | 0.000000 | 30 | 5.000 |
| P0 + P2 family constraints | 6 | 0.020213 | 0.006410 | 0.006410 | 29 | 5.000 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| P2 - P0 | +0.020213 | +0.006410 | +0.006410 | -1 |

## Non-Explicit Trace Notes
| query_id | likely_statute_family | first gold rank in fused@200 | final hit? | note |
|---|---|---:|---|---|
| val_003 | STPO;BV | 5 | yes | `Art. 221 Abs. 1 StPO` enters final predictions. |
| val_004 | ZGB | 1 | no | Fused retrieval finds a ZGB gold, but final top5 remains generic ZGB/non-gold. |
| val_005 | ZGB;STGB | -1 | no | Family is plausible, issue targeting still too weak for custody/visitation articles. |
| val_008 | STGB;OR | -1 | no | Family partially right, but `Art. 314 StGB` needs disloyal-management issue targeting. |
| val_009 | SCHKG;ZGB | -1 | no | Security/enforcement facts pull SchKG; core ZGB maintenance articles remain missed. |
| val_010 | OR;ZPO | -1 | no | Stable family, but bank mandate/protest/currency issue terms are still not sharp enough. |

## Decision
- Keep P2 as an experimental, default-off patch: it is the first change in this line to move non-explicit strict/corpus F1 above zero.
- The gain is small and mostly one-query driven, so this is not yet a new silver-core default.
- If continuing inside this repo framework, the next useful P2 layer would be issue phrase purification within predicted family, not court work.

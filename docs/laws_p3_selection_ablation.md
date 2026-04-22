# Laws P3 Selection Ablation

## Setup
- A: P0 + P2-A + P2-B, `outputs/laws_p2b_issue_phrases_audit`.
- B: P0 + P2-A + P2-B + P3, `outputs/laws_p3_final_cut_calibration`.
- Shared settings: light reranker, `--dynamic-mode fixed_top_k --fixed-top-k 5`, P0 rule exact enabled with `--rule-top-k-laws 20`, court dense disabled, court seed floors zero, no German expansion, no MiniLM fine-tune, no co-citation.
- P3 patch selected: **B. laws final cut calibration**.

## Audit Basis
- `docs/fused_to_final_loss_audit.csv` shows all per-gold stages.
- Global gold loss is still dominated by `not_in_rerank_input`, mostly because validation gold includes many court citations and statutes outside the laws-first candidates.
- For P2-B's retrieved issue gains, the failure is reranker/final-selection interaction: examples such as `val_005 Art. 133 Abs. 1 ZGB`, `val_008 Art. 314 StGB`, `val_009 Art. 292 ZGB`, and `val_010 Art. 100 Abs. 1 OR` are fused rank `1` but reranked below rank `100`.

## Patch Summary
- Added default-off flags:
  - `--enable-laws-final-cut-calibration`
  - `--laws-final-fused-rescue-top-k` default `1`
- When enabled, only non-explicit queries with predicted families get one additional final candidate:
  - candidate must come from the fused laws list,
  - candidate must be `laws_de`,
  - candidate must be predicted-family-consistent,
  - candidate must not already be in rule hits or reranked final cut.
- No retrieval change, no court change, no reranker model/training change.

## A/B Comparison
### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A + P2-B | 10 | 0.119154 | 0.024101 | 0.024101 | 48 | 5.200 |
| + P3 final cut calibration | 10 | 0.119154 | 0.061733 | 0.061733 | 49 | 5.800 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| P3 - P2-B | +0.000000 | +0.037632 | +0.037632 | +1 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A + P2-B | 4 | 0.125313 | 0.050638 | 0.050638 | 19 | 5.500 |
| + P3 final cut calibration | 4 | 0.125313 | 0.050638 | 0.050638 | 19 | 5.500 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| P3 - P2-B | +0.000000 | +0.000000 | +0.000000 | +0 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | avg_pred_count |
|---|---:|---:|---:|---:|---:|---:|
| P0 + P2-A + P2-B | 6 | 0.115047 | 0.006410 | 0.006410 | 29 | 5.000 |
| + P3 final cut calibration | 6 | 0.115047 | 0.069130 | 0.069130 | 30 | 6.000 |

| delta | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| P3 - P2-B | +0.000000 | +0.062720 | +0.062720 | +1 |

## Gold Drop Stage Distribution
### overall
| gold_drop_stage | P2-B | P3 | delta |
|---|---:|---:|---:|
| kept_final | 4 | 9 | +5 |
| not_in_rerank_input | 228 | 228 | +0 |
| reranked_too_low | 17 | 12 | -5 |
| cut_by_fixed_top_k | 2 | 2 | +0 |

### non-explicit subset
| gold_drop_stage | P2-B | P3 | delta |
|---|---:|---:|---:|
| kept_final | 1 | 6 | +5 |
| not_in_rerank_input | 124 | 124 | +0 |
| reranked_too_low | 11 | 6 | -5 |

## P3 Rescue Trace
| query_id | added final candidate | gold? |
|---|---|---|
| val_003 | Art. 29 Abs. 2 BV | yes |
| val_004 | Art. 510 Abs. 1 ZGB | no |
| val_005 | Art. 133 Abs. 1 ZGB | yes |
| val_008 | Art. 314 StGB | yes |
| val_009 | Art. 292 ZGB | yes |
| val_010 | Art. 100 Abs. 1 OR | yes |

## Decision
- Keep P3 as a laws-first final-selection patch: it converts P2-B retrieval recall into final F1 with only one additional final FP.
- Explicit rows are unchanged by design.
- Court remains frozen; the measurable gain is entirely from laws-side retrieval plus laws-side final selection.

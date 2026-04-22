# Fused to Final Loss Audit

## Setup
- Run audited: `outputs/laws_p2b_issue_phrases_audit` (P0 + P2-A + P2-B).
- Trace-only fields were added for audit: `fused_top320`, `reranked_top320`, and `final_cut_predictions`; retrieval and final predictions are unchanged.
- Final selection mode: `fixed_top_k=5`; reranker: `token_overlap`; court dense disabled and court seed floors zero.

## Overall Drop Stage Distribution
| gold_drop_stage | count |
|---|---:|
| not_in_rerank_input | 228 |
| reranked_too_low | 17 |
| kept_final | 4 |
| cut_by_fixed_top_k | 2 |

## Explicit Subset
| gold_drop_stage | count |
|---|---:|
| not_in_rerank_input | 104 |
| reranked_too_low | 6 |
| kept_final | 3 |
| cut_by_fixed_top_k | 2 |

## Non-Explicit Subset
| gold_drop_stage | count |
|---|---:|
| not_in_rerank_input | 124 |
| reranked_too_low | 11 |
| kept_final | 1 |

## Retrieved Gold Only
Gold citations with `gold_in_fused_top200=1`; this isolates fused-to-final losses from retrieval misses.
| gold_drop_stage | count |
|---|---:|
| reranked_too_low | 17 |
| kept_final | 4 |
| cut_by_fixed_top_k | 2 |

## Retrieved-But-Dropped Examples
| query_id | gold_citation | fused_rank | rerank_rank | drop_stage |
|---|---|---:|---:|---|
| val_003 | Art. 29 Abs. 2 BV | 3 | 171 | reranked_too_low |
| val_005 | Art. 133 Abs. 1 ZGB | 1 | 208 | reranked_too_low |
| val_008 | Art. 314 StGB | 1 | 133 | reranked_too_low |
| val_009 | Art. 292 ZGB | 1 | 116 | reranked_too_low |
| val_010 | Art. 100 Abs. 1 OR | 1 | 208 | reranked_too_low |
| val_006 | Art. 41 Abs. 1 OR | 11 | 18 | cut_by_fixed_top_k |

## Per Query Summary
| query_id | gold_count | in_fused_top200 | in_rerank_input | final_hits | dominant_drop_stage | notes |
|---|---:|---:|---:|---:|---|---|
| val_001 | 42 | 3 | 3 | 1 | not_in_rerank_input |  |
| val_002 | 36 | 0 | 0 | 0 | not_in_rerank_input |  |
| val_003 | 47 | 2 | 2 | 1 | not_in_rerank_input |  |
| val_004 | 10 | 2 | 2 | 0 | not_in_rerank_input |  |
| val_005 | 11 | 2 | 2 | 0 | not_in_rerank_input |  |
| val_006 | 18 | 3 | 3 | 2 | not_in_rerank_input |  |
| val_007 | 19 | 5 | 5 | 0 | not_in_rerank_input |  |
| val_008 | 29 | 1 | 1 | 0 | not_in_rerank_input |  |
| val_009 | 14 | 1 | 1 | 0 | not_in_rerank_input |  |
| val_010 | 25 | 4 | 4 | 0 | not_in_rerank_input |  |

## Audit Conclusion
- The dominant global loss is still retrieval miss (`not_in_rerank_input`/`not_in_fused`) because validation gold contains many court citations and statutes outside the laws-first candidates.
- Among gold citations that P2-B already retrieves into fused@200, the dominant fused-to-final failure is `reranked_too_low`: the light reranker often sends top-fused issue hits below rank 100.
- A plain reranked top-k expansion would be noisy; the minimal supported patch is instead a laws-lane cut calibration that admits the top fused, predicted-family-consistent laws candidate for non-explicit queries.
- Rule spurious candidates are observable in explicit rows, but they do not explain the P2-B non-explicit fused-to-final loss because non-explicit rows have no exact rule hits.

CSV: `docs/fused_to_final_loss_audit.csv`

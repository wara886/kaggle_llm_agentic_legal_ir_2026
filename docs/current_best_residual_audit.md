# Current Best Residual Audit

## Scope
- Validation-only residual audit for the current best path: Qwen3 rerank + `Art. 100 Abs. 1 BGG` global prior.
- Retrieval pipeline unchanged.
- Goal: split residual misses into candidate-stage loss vs rerank/final-cut loss.
- Qwen candidate cap: `80`.

## Current Best Local Metrics
| subset | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|
| overall | 0.177294 | 0.177294 | 41 |
| non-explicit | 0.175569 | 0.175569 | 29 |

## Drop Stage Split
| subset | total_gold | final_kept_rate | gold_in_fused_top200_rate | global_prior_rescue_rate | candidate_stage_share_of_missed | rerank_stage_share_of_missed | not_in_fused_top200 | not_in_qwen_input_cap | reranked_too_low | cut_by_dynamic_threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 251 | 0.099602 | 0.135458 | 0.035857 | 0.933628 | 0.066372 | 208 | 3 | 15 | 0 |
| non-explicit | 136 | 0.095588 | 0.147059 | 0.044118 | 0.902439 | 0.097561 | 110 | 1 | 12 | 0 |

## Top Residual Queries
| query_id | explicit_subset | kept_rate | miss_total | miss_not_in_fused_top200 | miss_not_in_qwen_input_cap | miss_reranked_too_low | miss_cut_by_dynamic_threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_003 | 0 | 0.042553 | 45 | 42 | 1 | 2 | 0 |
| val_001 | 1 | 0.095238 | 38 | 37 | 1 | 0 | 0 |
| val_002 | 1 | 0.083333 | 33 | 33 | 0 | 0 | 0 |
| val_008 | 0 | 0.068966 | 27 | 26 | 0 | 1 | 0 |
| val_010 | 0 | 0.080000 | 23 | 19 | 0 | 4 | 0 |

## Next-Step Verdict
- recommendation: `retrieval_or_candidate_shaping_first`
- If candidate-stage misses dominate, prioritize retrieval / candidate shaping diagnostics before more reranker or prior work.
- If rerank-stage misses dominate, prioritize Qwen input shaping, candidate cap, or final-cut calibration before retrieval changes.

## Artifacts
- `artifacts/current_best_residual_audit/gold_audit_rows.csv`
- `artifacts/current_best_residual_audit/query_residual_rows.csv`
- `artifacts/current_best_residual_audit/summary.json`

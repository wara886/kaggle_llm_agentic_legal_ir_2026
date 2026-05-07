# Public Score Proxy Audit

## Goal
- Identify which local metrics moved with public score from `0.04272` to `0.08960`.
- Avoid spending Kaggle submissions on variants that only overfit the 10-row validation set.

## Submitted Runs
| variant | public | local_strict_f1 | local_final_fp | added TP/FP vs Qwen | added TP/FP vs Art100 | test changed vs Art100 | gate_score_vs_art100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_laws_first_token_overlap | 0.01357 | 0.057960 | 50 | 0/45 | 0/45 | 40 | -0.020000 |
| qwen3_causal_cap80 | 0.04272 | 0.107028 | 42 | 0/0 | 0/0 | 40 | -0.000000 |
| qwen3_plus_art100 | 0.08960 | 0.167770 | 43 | 9/1 | 0/0 | 0 | 0.000000 |
| qwen3_plus_art100_social_only | 0.08960 | 0.188540 | 43 | 14/1 | 5/0 | 2 | 0.004154 |
| qwen3_plus_art100_safe_rules | 0.08946 | 0.199114 | 53 | 18/11 | 9/10 | 8 | -0.020153 |
| qwen3_plus_art100_social_rtbh_family_child_failed | 0.08939 | 0.222717 | 43 | 19/1 | 10/0 | 6 | 0.032968 |

## Local Signals That Actually Moved With Public
- Qwen3 rerank improved local strict_f1 and reduced final FP; public also rose (`0.01357 -> 0.04272`).
- `Art. 100 Abs. 1 BGG` had high validation added precision: 9 added TP / 1 added FP, and affected all 40 test queries; public rose (`0.04272 -> 0.08960`).
- `social_insurance_core` had perfect validation additions versus Art100 on one val query, but changed only 2 test queries versus Art100; public did not move.
- The wider safe-rule pack had the highest local strict_f1, but added more FP and reduced public slightly; local strict_f1 alone is not a reliable submit criterion.

## Submit Gate
- Track local strict_f1, but never use it alone.
- Require `val_added_precision >= 0.70` for prior additions.
- Prefer `delta_fp_from_art100 <= 3`; reject or combine later if FP grows more than that.
- Require enough test coverage for a standalone submission: normally `test_changed_queries >= 8`; if fewer, hold until combined with other high-confidence rules.
- For reranker changes, track `local_strict_f1`, `final_fp`, and `reranked_too_low`; for prior/final-count changes, track added TP/FP precision and test coverage.

## Not-Submitted Final Count Candidates
| variant | local_strict_f1 | local_final_fp | avg_pred_count | test_changed_queries | gate_score_vs_art100 |
|---|---:|---:|---:|---:|---:|
| qwen3_top1_plus_art100_not_submitted | 0.129403 | 5 | 2.0000 | 40 | -0.000000 |
| qwen3_top2_plus_art100_not_submitted | 0.148731 | 12 | 3.0000 | 40 | -0.000000 |
| qwen3_top3_plus_art100_not_submitted | 0.164807 | 20 | 4.0000 | 40 | -0.000000 |
| qwen3_top4_plus_art100_not_submitted | 0.157267 | 30 | 5.0000 | 40 | -0.000000 |
| qwen3_top5_plus_art100_not_submitted | 0.161072 | 38 | 6.0000 | 22 | -0.000000 |
| qwen3_top6_plus_art100_not_submitted | 0.167770 | 43 | 6.6000 | 1 | 0.000000 |

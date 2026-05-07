# Test Query Cluster Audit

## Summary
- test queries: `40`
- explicit citation queries: `16`
- base submission: `release\submission_qwen3_bgg100_prior_v1\submission.csv`
- trace csv: `outputs\current_code_mainline_control\test_seed_trace_silver_baseline_v0.csv`

## Cluster Counts
| cluster | count | explicit | variant-changed queries | qids |
|---|---:|---:|---:|---|
| zgb_child_family | 9 | 2 | 9 | test_008;test_009;test_015;test_016;test_026;test_027;test_028;test_030;test_031 |
| zpo_civil_procedure | 7 | 2 | 7 | test_006;test_017;test_018;test_024;test_029;test_035;test_038 |
| or_contract_liability | 5 | 2 | 5 | test_003;test_007;test_020;test_021;test_034 |
| zgb_inheritance_possession | 5 | 0 | 5 | test_005;test_013;test_019;test_025;test_039 |
| social_ivg_atsg_explicit | 4 | 4 | 4 | test_011;test_012;test_014;test_023 |
| stpo_detention_procedure | 3 | 2 | 3 | test_010;test_032;test_036 |
| ip_trade_secret_provisional | 2 | 0 | 2 | test_001;test_037 |
| svg_traffic_liability | 2 | 1 | 2 | test_002;test_033 |
| schkg_bankruptcy_enforcement | 1 | 1 | 1 | test_004 |
| iprg_private_international | 1 | 1 | 1 | test_022 |
| aig_migration | 1 | 1 | 1 | test_040 |

## Family Counts
| family | count |
|---|---:|
| ZGB | 18 |
| OR | 14 |
| IPRG | 5 |
| STPO | 3 |
| BV | 3 |
| SCHKG | 2 |
| AIG | 2 |
| ZPO | 2 |
| STGB | 1 |

## Next Work Queue
- Prioritize clusters with multiple test queries and low prior/public stability: candidate-stage recall only, then Qwen rerank.
- Avoid broad social-insurance family expansion; keep IVG/ATSG explicit-only.
- Use `test_query_clusters.csv` to select target qids and audit spillover before submission.

## Artifacts
- `artifacts\test_query_cluster_audit\summary.json`
- `artifacts\test_query_cluster_audit\cluster_summary.csv`
- `artifacts\test_query_cluster_audit\test_query_clusters.csv`

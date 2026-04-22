# 候选诊断说明（candidate_diagnostics_v1）

## 1. 为什么现在先做 candidate diagnostics
- 当前阶段的核心风险是“召回不到（recall miss）”而不是“写报告质量”。
- baseline_v1_1 引入了 query_preprocessing、cross_lingual_expansion、rule_based_recall、source_aware_fusion，这些改动优先影响候选覆盖。
- 若不先看候选层，很容易把“召回问题”误判为“重排问题”。

## 2. Recall@K 与 Macro F1 的关系
- Recall@K（候选召回率）回答的是：gold citation 是否进入 top-k candidate set。
- Macro F1 回答的是：最终提交结果是否精确且完整。
- 关系是“上游约束”：
  - Recall@K 很低时，Macro F1 上限受限。
  - Recall@K 提升后，Macro F1 未必立刻提升，可能还受 fusion/rerank 截断影响。

## 3. 如何解读 recall 问题与 ranking 问题
- 更像 recall 问题：
  - `sparse/dense/rule` 各分支在 R@200 仍低；
  - 失败类型集中在 `no_gold_in_candidates`、`query_language_gap`、`citation_pattern_missed`。
- 更像 ranking 问题：
  - 分支中已有命中（R@200>0）；
  - 但 `fusion_final` 或最终截断后丢失，失败类型偏 `gold_lost_in_fusion`、`gold_lost_in_final_cut`。

## 4. 本脚本输出
- `artifacts/diagnostics_v1/recall_by_branch_v1.csv`
- `artifacts/diagnostics_v1/recall_by_branch_v1_1.csv`
- `artifacts/diagnostics_v1/recall_compare_v1_vs_v1_1.csv`
- `artifacts/diagnostics_v1/query_failure_clusters.csv`
- `artifacts/diagnostics_v1/diagnostics_summary_cn.md`

## 5. 参数风格
- 支持与 `run_baseline_v1.py` 一致的主参数（fusion/reranker/top_k 系列）。
- 同时支持双配置：
  - `config_a`：baseline_v1 兼容配置（默认全关）
  - `config_b`：baseline_v1_1 增强配置（默认全开）
- 可按需覆盖：
  - `--config-*-enable-query-preprocess`
  - `--config-*-enable-query-expansion`
  - `--config-*-enable-rule-recall`
  - `--config-*-source-aware-fusion`
  - `--config-*-laws-weight`
  - `--config-*-court-weight`


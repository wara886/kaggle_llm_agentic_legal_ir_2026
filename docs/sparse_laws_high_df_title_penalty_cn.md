# sparse_laws 高 DF 标题惩罚实验说明（high_df_title_penalty）

## 目标
- 仅在 `sparse_laws` 分支内部的 `title` 分量上施加惩罚。
- 不改 dense/court/rerank/fusion 主流程，不引入新模型。

## 触发条件（conditional penalty）
候选满足下列条件时才惩罚 `title_score_component`：
1. 来自 `sparse_laws`
2. `title_df_ratio` 高于阈值（`title_df_threshold`）
3. `title` 分量占比较高（`title_dominance_threshold`）
4. `citation` 分量占比较低（`low_citation_threshold`）
5. （可选）命中泛标题模板模式（`generic_title_pattern_mode`）

## 模式
- `df_only`
- `df_plus_dominance`
- `df_plus_dominance_plus_pattern`

## 参数
- `--enable-high-df-title-penalty`
- `--title-df-threshold`
- `--title-dominance-threshold`
- `--low-citation-threshold`
- `--high-df-title-penalty-strength`
- `--generic-title-pattern-mode`

## 评测脚本
`scripts/run_sparse_laws_penalty_eval_v1_3.py`

在最优 `sparse_laws` 权重基础上做小网格：
- `title_df_threshold`: `top_1pct, top_3pct, top_5pct, top_10pct`
- `title_dominance_threshold`: `0.5, 0.6, 0.7, 0.8`
- `low_citation_threshold`: `0.0, 0.05, 0.1`
- `high_df_title_penalty_strength`: `0.05, 0.1, 0.15, 0.2`
- `generic_title_pattern_mode`: `df_only, df_plus_dominance, df_plus_dominance_plus_pattern`

输出：
- `artifacts/v1_3_sparse_laws_penalty/penalty_results.csv`
- `artifacts/v1_3_sparse_laws_penalty/best_penalty_config.json`
- `artifacts/v1_3_sparse_laws_penalty/penalty_summary_cn.md`

## 兼容性
- 默认关闭（`enable=false`）时，不影响原有流程行为。

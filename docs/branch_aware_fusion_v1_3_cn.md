# branch-aware fusion v1_3 设计说明

## 目标
- 仅在融合层（fusion layer）验证问题假设：`sparse_laws` 单分支深尾部命中是否被原融合压掉。
- 不改检索（retrieval）与查询侧（query-side）逻辑，不引入新模型。

## 改动范围
- `src/fusion.py`
- `scripts/run_baseline_v1.py`
- `scripts/run_branch_aware_fusion_eval_v1_3.py`

## 新增参数
- `--enable-branch-aware-fusion`
- `--sparse-laws-branch-bonus`
- `--sparse-laws-single-branch-bonus`
- `--branch-aware-rank-cutoff`
- `--branch-aware-fusion-mode`

默认值均为“关闭/0”，保持向后兼容。

## 两种融合模式
1. `sparse_laws_bonus`
- 对来自 `sparse_laws` 的候选施加统一奖励（branch bonus）。
- 若候选只被单分支命中（`branch_support_count=1`），可叠加单分支奖励。

2. `sparse_laws_tail_rescue`
- 仅对满足条件的候选加奖励：
- 来自 `sparse_laws`
- `branch_support_count=1`
- `sparse_laws` rank 在 `branch_aware_rank_cutoff` 内
- 在原融合中不在前列（当前实现采用 `base_rank > top_n` 的保守判据）

## 实现原则
- 奖励幅度可调，不写死常量。
- 不使用 gold/query_id 硬编码。
- 仅依赖已有候选的 `branch/source/rank/score` 元信息。

## 评测脚本
`scripts/run_branch_aware_fusion_eval_v1_3.py` 采用固定 retrieval cache，仅对 fusion 参数做小网格搜索：
- `sparse_laws_branch_bonus`: `0.0, 0.05, 0.1, 0.2`
- `sparse_laws_single_branch_bonus`: `0.0, 0.05, 0.1, 0.2`
- `branch_aware_rank_cutoff`: `100, 150, 200`
- `branch_aware_fusion_mode`: `sparse_laws_bonus`, `sparse_laws_tail_rescue`

输出：
- `artifacts/v1_3_branch_aware_fusion/branch_aware_fusion_results.csv`
- `artifacts/v1_3_branch_aware_fusion/best_branch_aware_config.json`
- `artifacts/v1_3_branch_aware_fusion/branch_aware_fusion_summary_cn.md`

## 风险边界
- 若 `sparse_laws` 新增命中仍集中在很深 rank 或 score 过弱，仅靠融合奖励可能不足以改善最终 F1。
- 若奖励过大，可能引入 laws 侧噪声，需结合 `strict_macro_f1` 与 `corpus_aware_macro_f1` 联合判断。

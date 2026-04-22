# Repo Hard Negative Quality Audit

## Audited files
- `scripts/mine_laws_hard_negatives_minilm.py`
- `scripts/train_laws_minilm_biencoder.py`
- `src/law_family.py`
- `src/query_preprocess.py`

## Current mining behavior (before this round quality patch)
- negatives 来源：laws-only sparse+dense RRF fused 列表。
- 选择规则：取 fused 中第一个非 gold laws citation。
- 每 query 1 negative。
- 无显式 same-family/same-issue 优先采样，无“过易负样本”过滤。

## Measured quality signals on `artifacts/laws_minilm_p1_scalealign` (993 triplets)
- `same_family_rate` 约 `3.0%`（按 pos/neg family 对齐统计）。
- `likely_family_hit_rate` 约 `0.1%`（negative 命中 query likely family 极低）。
- `issue_overlap_rate` 近 `0`（在可抽取 issue terms 的样本中，negative 几乎不含 issue 词）。
- `negative_rank` 均值约 `1.02`（几乎总是“第一个非 gold”）。

## 必答结论
1. 当前 negatives 是否大量是“容易负样本”？
- 是。same-family / same-issue 近似率都很低，说明很多是语义不够贴近任务判别边界的“易负样本”。

2. 当前 negatives 是否缺 same-family near miss？
- 是，明显缺。

3. 当前 negatives 是否缺 same-issue near miss？
- 是，明显缺。

4. 当前 negatives 是否与线上有效 laws-first 结构脱节？
- 部分脱节：
  - mining 的 query pack 会使用 laws-first 结构来召回候选；
  - 但 negative 最终筛选逻辑没有利用 family/issue/rules 信号做质量筛选。

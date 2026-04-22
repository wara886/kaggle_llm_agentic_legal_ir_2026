# Hard Negative Quality Gap Report

## Top 1 gap: 缺 same-family near miss
1) 差异
- negatives 选择缺少 family 贴近约束，绝大多数不是同法典家族近邻误匹配。

2) 为什么限制 non-explicit 收益
- non-explicit 主要靠 family/issue 语义对齐；若负样本不在同家族边界，模型难学到“同家族内细粒度区分”。

3) 修复成本
- 低

4) 预期收益
- 中

## Top 2 gap: 缺 same-issue near miss
1) 差异
- negatives 未利用 issue phrase 近似信息，缺少“同家族但错误条款”的任务关键难例。

2) 为什么限制 non-explicit 收益
- non-explicit 命中依赖 issue 语义锚点；若负样本不共享 issue 语义，训练对该子集的排序提升会弱。

3) 修复成本
- 中

4) 预期收益
- 中-高

## Top 3 gap: 过易 negatives 未过滤
1) 差异
- 只取“第一个非 gold”且无难度判定，包含大量可能过易的负样本。

2) 为什么限制 non-explicit 收益
- MNRL in-batch 已有对比信号；若显式 negative 本身太易，额外监督边际低。

3) 修复成本
- 低

4) 预期收益
- 中

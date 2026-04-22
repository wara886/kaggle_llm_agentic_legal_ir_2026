# Hard Negative Gap Audit (Notebook vs Repo)

聚焦解释：为何 loss 对齐为 MNRL 后仍只有小幅收益。

## Top 1
### 差异
- **negative scale mismatch**：repo 历史 mining 人为截断（`max_rows=300`, `max_examples=500`，实际只产出 `264` triplets），而 notebook 是全 train 级别（约一条/query）。
### 为什么会导致 MNRL 仅小增益
- MNRL 本身依赖足够多的难例分布；样本规模太小会让 loss 对齐带来的潜力被严重压缩。
### 类型
- `negative scale mismatch`
### 修复成本
- 低
### 预期收益
- 高

## Top 2
### 差异
- **negative source mismatch**：notebook 从 dense-only top-k 层直接取 negative；repo 从 sparse+dense 融合后列表取 negative。
### 为什么会导致 MNRL 仅小增益
- 融合列表会混入 lexical 偏好，negative 难度分布与 dense 表征学习目标不完全一致，减弱 MNRL 对 dense 空间的直接塑形。
### 类型
- `negative source mismatch`
### 修复成本
- 中
### 预期收益
- 中-高

## Top 3
### 差异
- **negative quality mismatch**：notebook 负样本来自 unified 大池（laws+court）误召回；repo 冻结 court 后仅 laws 内近邻，难例多样性更低。
### 为什么会导致 MNRL 仅小增益
- 难负样本类型收窄，模型看到的“混淆边界”减少，泛化收益受限。
### 类型
- `negative quality mismatch`
### 修复成本
- 中（且受 court 冻结约束）
### 预期收益
- 中

## Top 4
### 差异
- **training text mismatch**：notebook laws 侧更接近 `text` 主体；repo triplet 文本是 `title + text + citation` 模板。
### 为什么会导致 MNRL 仅小增益
- 额外模板 token（尤其 citation 串）可能引入捷径特征，稀释语义对齐收益。
### 类型
- `training text mismatch`
### 修复成本
- 低-中
### 预期收益
- 中

## Top 5
### 差异
- **reindex mismatch**：repo 有持久化 reindex 脚本，但主评估路径常走运行时重建；训练-索引-推理一致性可追溯性不如 notebook 闭环直观。
### 为什么会导致 MNRL 仅小增益
- 不是直接性能杀手，但会降低实验可控性，影响对微小收益的稳定验证。
### 类型
- `reindex mismatch`
### 修复成本
- 中
### 预期收益
- 低-中

## 结论
- 当前最该优先对齐的是 **negative scale mismatch**；它最符合“loss 已对齐但增益仍小”的现象。

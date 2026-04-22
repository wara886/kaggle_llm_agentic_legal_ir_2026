# Training Enhancement Gap Report

## 1) 当前 MNRL + full mining 还缺什么
即使 loss 与 mining 规模已对齐，训练线仍缺两点关键对齐：
- **training query 语义形态**未完全贴合线上有效主线（family / issue 结构）。
- **negative source/quality**仍与 silver notebook有差异（repo 是 laws-only fused negatives；notebook 是 unified dense pool negatives）。

## 2) hard negative 质量是否仍低于 silver notebook
是，仍有结构性差异：
- silver notebook negatives 来自 unified dense top-k（laws+court）的“语义误召回层”；
- repo 主线 negatives 来自 laws-only sparse+dense融合层；
- 这会改变难例分布，通常使对 dense 表征最敏感的“硬负样本边界”弱化。

## 3) training query 是否没有吸收当前线上有效的 family / issue 结构
是，历史版本基本使用原始 query 直接训练。
- 线上 laws-first 成功依赖 family + issue 结构化线索；
- 训练若不吸收这些线索，MNRL 学到的检索几何与线上有效检索视图会有偏差。

## 4) 最可能解释“训练线只有小增益”的差异
首要解释是：**training text mismatch（尤其 query 结构不对齐）+ negative quality mismatch 的组合**。
- loss 对齐和规模对齐能带来增益，但若 query 训练视图仍不对齐线上 family/issue 主线，收益会被压缩成“小幅提升”。

## Fast evidence from latest ablation
On top of `MNRL + full mining`:
- add query alignment to laws structured view (`family + issue`) in training only,
- no retrieval branch changes,
- yields extra gain on non-explicit and lower FP.

This supports: training-line bottleneck is not only loss/scale, but also **query-view mismatch**.

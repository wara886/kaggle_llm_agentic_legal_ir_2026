# 开源方案综合说明（Open Solution Synthesis）

## 1. 借鉴的公开思路（按模块类型）
- 检索主干（retrieval backbone）：
  - BM25 稀疏召回 + embedding 语义召回的双路并行。
- 融合策略（fusion strategy）：
  - 先用 RRF 做稳健融合，再对比分数加权融合。
- 评测分析（evaluation & diagnostics）：
  - 保留逐查询评估（per-query evaluation）与错误分类（error taxonomy）。
- 工程组织（engineering pattern）：
  - 检索、融合、重排、评测解耦，脚本化回归。

## 2. 为什么不直接照抄单个 notebook
- 单一 notebook 往往耦合了特定数据假设，迁移性较差。
- 本题存在 `train` 语料外标签，直接照抄容易把离线评测做偏。
- 我们需要“可回归 + 可插拔 + 可诊断”的基线底座，而不是一次性分数脚本。
- 组合多个公开思路更适合后续演进到：
  - cocitation
  - citation graph
  - query rewrite

## 3. 当前 baseline v1 的可插拔增强点
- 稀疏检索器（retrieval_sparse）可扩展字段加权与查询扩展。
- 语义检索器（retrieval_dense）可替换为更强 multilingual encoder。
- 融合层（fusion）可插入学习型融合或 query-aware 动态权重。
- 重排层（rerank）可替换为 cross-encoder。
- 评测层（eval_local）可新增分桶评测与回归阈值门禁。

## 4. 当前明确不做
- cross-encoder 重训练
- agent query rewrite
- GraphRAG 全量接入
- 大模型报告生成


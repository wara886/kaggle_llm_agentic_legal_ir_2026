# baseline_v1_1 设计说明

## 1. v1 与 v1_1 结构差异
- baseline_v1：
  - 稀疏检索（retrieval_sparse）+ 稠密检索（retrieval_dense）
  - 融合（fusion）
  - 轻量重排（rerank）
- baseline_v1_1（增量）：
  - 新增查询预处理（query_preprocessing）
  - 新增跨语种扩展（cross_lingual_expansion）
  - 新增规则召回（rule_based_recall）
  - 新增分源融合（source_aware_fusion）
  - 保持旧路径可用，默认参数关闭时行为与 v1 一致

## 2. 为什么先做 query-side bridge
- 当前痛点优先在召回（recall）与候选覆盖，而非重排容量。
- query 侧桥接成本低、可解释性强、可快速回归验证。
- 在样本规模有限阶段，先提升候选质量通常比直接做 cross-encoder 训练更稳。

## 3. 后续接入 cocitation 与 citation graph
- 在 `fusion` 层新增图特征分（graph_score）：
  - cocitation 共现强度
  - citation graph 邻接中心性
- 在 rerank 接口透传图字段：
  - `co_citation_score`
  - `graph_hop_score`
- 先做离线特征注入，再逐步做学习型融合。

## 4. 当前保留轻量 rule_based_reranker 的原因
- 轻量规则重排器（rule_based_reranker）无需训练，回归成本低。
- 目前主问题在“召回覆盖不足”，过早上 Hugging Face 预训练 reranker 会放大工程复杂度。
- 等候选覆盖和分源融合稳定后，再引入更重 reranker 更有性价比。

## 5. 参数与兼容性
- 新增参数：
  - `--enable-query-preprocess`
  - `--enable-query-expansion`
  - `--enable-rule-recall`
  - `--source-aware-fusion`
  - `--laws-weight`
  - `--court-weight`
- 默认值均为“兼容 v1”配置：
  - 新功能默认关闭
  - 权重默认 `1.0 / 1.0`


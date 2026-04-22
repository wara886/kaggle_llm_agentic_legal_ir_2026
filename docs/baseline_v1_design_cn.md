# baseline v1 设计说明

## 1. 目标
在 phase0 基础上实现组合型基线（baseline v1），重点稳定检索核心（retrieval core），输出可提交结果与可回归诊断结果。

## 2. 固定结构
1. 复用引用规范化（citation normalization）。  
2. 双语料分开建索引：
- `laws_de`
- `court_considerations`
3. 稀疏检索（BM25）。
4. 多语语义检索（multilingual embeddings）。
5. 混合融合（hybrid score fusion），默认倒数排名融合（Reciprocal Rank Fusion, RRF），可切换加权融合。
6. 预留重排（rerank）接口，当前使用轻量 token overlap 脚手架。
7. 复用本地 citation-level Macro F1 与逐查询评估（per-query evaluation）。
8. 输出错误分类（error taxonomy）与回归记录（regression log）。

## 3. 模块与职责
- `src/citation_normalizer.py`：统一引用规范化入口。
- `src/retrieval_sparse.py`：双语料 BM25 检索器。
- `src/retrieval_dense.py`：多语语义检索器。
- `src/fusion.py`：RRF 与加权融合。
- `src/rerank.py`：重排接口与轻量实现。
- `src/eval_local.py`：本地评测、逐查询输出、错误分类。

## 4. 运行流程
1. 构建 sparse 索引（laws/court 分开）。
2. 构建 dense 索引（laws/court 分开，优先 SBERT，多语模型不可用时回退 Hashing+SVD）。
3. 查询侧并行召回 sparse/dense 候选。
4. 融合层执行 RRF（默认）或加权融合。
5. rerank 接口二次排序并截断 top-n citation。
6. 生成 `submission_baseline_v1.csv` 与本地评测结果。

## 5. 设计取舍
- 先保证可运行、可诊断、可回归，再追求复杂模型。
- dense 模块支持后备路径，降低环境依赖导致的阻塞风险。
- 保持模块边界清晰，便于后续接入：
  - cocitation
  - citation graph
  - query rewrite


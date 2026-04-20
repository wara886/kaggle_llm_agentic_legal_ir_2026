# 比赛简报（LLM Agentic Legal Information Retrieval）

## 1. 任务定义
- 输入：英文法律问题（legal query）。
- 输出：最相关的瑞士法律来源引用（citations）。
- 评测：引用级宏平均 F1（citation-level Macro F1）。

## 2. 当前数据结构
- `train.csv`: `query_id`, `query`, `gold_citations`
- `val.csv`: `query_id`, `query`, `gold_citations`
- `test.csv`: `query_id`, `query`
- `sample_submission.csv`: `query_id`, `predicted_citations`
- `laws_de.csv`: `citation`, `text`, `title`
- `court_considerations.csv`: `citation`, `text`

## 3. 规模与风险
- 样本规模：`train=1139`, `val=10`, `test=40`。
- 风险现象：
  - `train` 唯一金标引用（gold citations）中有相当一部分不在公开语料库；
  - `val` 的金标引用均在语料库中。
- 影响：
  - 若直接做严格匹配评测，模型可能因“语料外标签（out-of-corpus labels）”被额外惩罚，产生假负例（false negative）。

## 4. 阶段 0 目标（本仓库当前实现）
- 建立可回归评测底座（evaluation harness）。
- 完成引用规范化（citation normalization）与缺失标签统计。
- 提供逐查询评估（per-query evaluation）与错误分析模板（error analysis template）。
- 提供可提交的最小 BM25 基线（baseline）脚本与产物路径约定。

## 5. 非目标（本阶段明确不做）
- 稠密检索（dense retrieval）
- 交叉编码器重排（cross-encoder rerank）
- Agent 查询改写（agent query rewrite）
- 验证器（verifier）
- 重训练（re-training）


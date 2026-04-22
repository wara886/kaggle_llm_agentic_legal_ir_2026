# baseline_v1_2 设计说明

## 1. 目标与边界
- 仅针对两类瓶颈做增量优化：
  - 语言鸿沟（language_gap）
  - 字段使用问题（field_usage_issue）
- 不改 phase0，不删除 baseline_v1 / baseline_v1_1 逻辑。
- 不引入更强 reranker，不做 graph/cocitation，不做 agent rewrite。

## 2. 核心改动
1. 源感知查询扩展（source_aware_query_expansion）
- 在 `query_expansion.py` 新增：
  - `expanded_query_de_laws`
  - `expanded_query_de_court`
  - `laws_query_pack`
  - `court_query_pack`
- 词表采用可维护字典结构，避免长链 if/else。

2. 字段感知检索（field_aware_retrieval）
- `retrieval_sparse.py`：
  - laws 侧：`citation/title/text` 分字段加权检索。
  - court 侧：`citation/text` 分字段加权检索。
  - 支持 source-specific query pack。
- `retrieval_dense.py`：
  - laws 侧优先 `title+text` 组合视图（combined_text_view）。
  - court 侧优先 `text` 视图。
  - 支持 source-specific query pack。

3. baseline 参数接线（run_baseline_v1.py）
- 新增参数：
  - `--enable-source-aware-query-expansion`
  - `--enable-field-aware-retrieval`
  - `--laws-citation-weight`
  - `--laws-title-weight`
  - `--laws-text-weight`
  - `--court-citation-weight`
  - `--court-text-weight`
- 默认关闭，保持向后兼容。

## 3. 为什么是 v1_2
- v1_1 的主增益来自 `rule_branch`，主分支（sparse/dense）仍长期接近 0。
- v1_2 先补 query-side 与 field-side 桥接，目标是让主分支先“脱离零召回”。

## 4. 最小验证
- 脚本：`scripts/run_targeted_eval_v1_2.py`
- 对比维度：
  - sparse/dense/fusion 的 Recall@50/100/200
  - strict_macro_f1
  - corpus_aware_macro_f1
- 输出：
  - `artifacts/v1_2_targeted_eval/targeted_compare_v1_1_vs_v1_2.csv`
  - `artifacts/v1_2_targeted_eval/targeted_summary_cn.md`

## 5. 编码与乱码修复原则
- 报告与文档统一使用 UTF-8 写出（`encoding='utf-8'`）。
- 避免在控制台编码不一致时通过终端重定向直接写中文报告。
- 优先由 Python 直接写文件，减少 `?` 乱码风险。


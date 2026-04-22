# baseline_v1_3_laws_focus 设计说明

## 1. 目标
- 在 baseline_v1_2 基础上，只强化 laws 侧稀疏主干（sparse_laws mainline）。
- 聚焦“长英文法律描述 -> 德文法规表达”桥接不足。
- 不改 dense 主逻辑，不改 court 主逻辑，不改 rerank。

## 2. 本次只做两件事
1. 法规风格查询扩展（statute_style_laws_query_expansion）
- 在 `query_expansion.py` 新增 `laws_query_pack_v2` 和 `expanded_query_de_laws_v2`。
- 采用结构化词表 + 模板，不使用长链 if/else。
- 保留原 `laws_query_pack`，通过开关启用 v2。

2. laws 侧稀疏专用验证（laws_sparse_only_eval）
- 新增 `scripts/run_laws_focus_eval_v1_3.py`。
- 仅比较 v1_2 与 v1_3_laws_focus：
  - `sparse_laws Recall@50/100/200`
  - `fusion_final Recall@50/100/200`
  - `strict_macro_f1`
  - `corpus_aware_macro_f1`

## 3. 兼容性
- `run_baseline_v1.py` 新增参数：
  - `--enable-laws-query-pack-v2`
- 默认关闭；关闭时行为与 v1_2 一致。

## 4. 输出
- `artifacts/v1_3_laws_focus_eval/laws_focus_compare_v1_2_vs_v1_3.csv`
- `artifacts/v1_3_laws_focus_eval/laws_focus_summary_cn.md`

## 5. 为什么暂不转向 dense/court
- v1_2 已显示 laws 侧 sparse 有提升信号，而 dense/court 仍弱。
- 在明确 laws 侧可持续拉升前，先沿已验证路径增量优化，风险更低、可解释性更强。


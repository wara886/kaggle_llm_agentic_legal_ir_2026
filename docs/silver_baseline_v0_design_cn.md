# silver_baseline_v0 设计说明

## 目标
- 在不重写 `phase0` 和 `v1/v1_1/v1_2/v1_3` 的前提下，落地最小银牌骨架（silver_baseline_v0）。
- 聚焦 5 个模块：
  1. 查询源路由（query_source_router）
  2. 分源检索（federated_retrieval）
  3. 强重排器（strong_reranker）
  4. 动态截断（dynamic_threshold / dynamic_k）
  5. 段落感知匹配器（paragraph_aware_matcher，仅本地评测）

## 模块结构
- `src/source_router.py`
  - 基于 case/statute pattern 的规则路由。
  - 输出 `primary_source`、`secondary_source`、`route_confidence`。
- `scripts/run_silver_baseline_v0.py`
  - 执行主流程（val/test）并产出预测、trace、run meta。
  - federation 策略：
    - laws 路由：`sparse(BM25) + dense + RRF`
    - court 路由：`sparse(BM25)` 主线
    - hybrid 路由：laws/court 双检索后统一进入 reranker
  - strong reranker 首选 `BAAI/bge-reranker-v2-m3`，不可用则回退轻量 reranker，并记录原因。
  - 支持 `fixed_top_k`、`score_threshold`、`relative_threshold` 三种截断模式。
- `src/eval_matchers.py`
  - `exact_match`
  - `paragraph_aware_match`（法规 citation 忽略 `Abs./lit.`，保留 `Art + family`）
  - `truly_unreachable`
- `scripts/run_silver_local_eval_v0.py`
  - 运行多组配置对比（router/no-router、strong/light、dynamic/fixed）。
  - 输出 `strict_macro_f1`、`corpus_aware_macro_f1`、`paragraph_aware_macro_f1`、路由计数、`fusion_final Recall@100/200`。

## 向后兼容
- `scripts/run_baseline_v1.py` 新增 `--pipeline-mode`：
  - `baseline_v1`（默认，旧行为不变）
  - `silver_baseline_v0`（转调新骨架入口）

## 工程边界
- 不改 dense/court/laws/fusion/rerank 既有主逻辑实现。
- 不引入付费 API。
- 不一次性叠加后续全部增强，仅落地最小可用骨架。

# court_dense_retrieval_v1 设计说明（最小增量）

## 目标
- 在不改 phase0、不改 laws 主线、不改 fusion/rerank 主逻辑前提下，补齐 `court_considerations` 的稠密召回（dense retrieval）。
- 验证 `court sparse` 候选质量不足是否是当前 court 主线的真实缺口。

## 设计原则
- 分源检索（federated retrieval）保持不变：laws 与 court 继续分路。
- court 仅在主线内部新增 `sparse + dense`，不做统一大池 dense。
- 默认关闭（`--enable-court-dense=false`），保持向后兼容。

## 接入点
1. `src/retrieval_dense.py`
- 新增 `search_court_multi_view(...)`：
  - 输入多视图查询包（`bilingual_query_pack`）。
  - 仅输出 `court_considerations` 来源候选。

2. `scripts/run_silver_baseline_v0.py`
- 新增参数：
  - `--enable-court-dense`
  - `--court-dense-max`
  - `--court-dense-weight`
  - `--court-dense-fusion-mode`（`rrf` / `weighted`）
  - `--court-dense-model-name`
- 在 `run_split(...)` 中仅对 court 分支新增：
  - `court sparse`（已有）+ `court dense`（新增）内部融合，
  - 融合结果再与 laws 分支候选共同进入既有 merge/rerank/threshold 链路。

## court 内部融合
- `rrf` 模式：对 `court sparse rank` 与 `court dense rank` 做 RRF 融合。
- `weighted` 模式：以 rank-based 分值融合，`court_dense_weight` 控制 dense 贡献。

## fallback 机制
- 若 `dense backend` 非 `sbert`（模型不可用或初始化失败），court dense 自动关闭并回退到 `court sparse only`。
- 在 run meta 与 per-query note 写入 `court_dense_fallback_reason`。

## 评测脚本
- 新增 `scripts/run_court_dense_eval_v1.py`，固定比较：
  1. `no_router`
  2. `source_router_v1_1`
  3. `court_mainline_v1`
  4. `court_dense_retrieval_v1`
- 输出：
  - `artifacts/court_dense_v1/court_dense_eval_results.csv`
  - `artifacts/court_dense_v1/court_dense_per_query.csv`
  - `artifacts/court_dense_v1/court_dense_summary_cn.md`

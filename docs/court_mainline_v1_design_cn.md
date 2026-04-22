# court_mainline_v1 设计说明

## 目标
- 在不改 router 核心判断、不改 fusion/rerank 主逻辑的前提下，完成 court 主线补全最小版（court_mainline_v1）。
- 验证：仅通过 route-aware candidate budget（路由感知候选配额）提升 court 候选进入 merge/rerank 的机会，是否继续拉升 `fusion_final Recall@100/200`。

## 设计范围
- 不引入新模型。
- 不新增 dense court。
- 不重写检索器，复用现有 `SparseRetriever` 的 field-aware BM25。

## 关键改动

### 1) retrieval_sparse 最小增强
- 新增 `search_route_aware(...)`：
  - 仅暴露 `laws_top_k / court_top_k`，内部复用 `search_field_aware(...)`。
  - 用于把路由配额逻辑从脚本层传入检索器。

### 2) run_silver_baseline_v0 最小接入
- 新增开关与配额参数（默认关闭，向后兼容）：
  - `--enable-court-mainline`
  - `--laws-route-laws-max`
  - `--laws-route-court-max`
  - `--court-route-court-max`
  - `--court-route-laws-max`
  - `--hybrid-route-laws-max`
  - `--hybrid-route-court-max`
  - `--min-court-candidates-for-hybrid`
  - `--min-court-candidates-for-court-route`
- 新增 per-query trace 字段：
  - `laws_candidates_forwarded`
  - `court_candidates_forwarded`
  - `gold_citation`
  - `court_mainline_effect_note`

## route-aware budget 策略
- laws route：laws 大预算 + court 小预算。
- court route：court 大预算 + laws 小预算。
- hybrid route：laws/court 双路都保留，并保证 court 最小候选数。

## 评测脚本
- `scripts/run_court_mainline_eval_v1.py`
- 固定对比：
  1. `no_router`
  2. `source_router_v1_1`
  3. `court_mainline_v1`
- 输出：
  - `artifacts/court_mainline_v1/court_mainline_eval_results.csv`
  - `artifacts/court_mainline_v1/court_mainline_per_query.csv`
  - `artifacts/court_mainline_v1/court_mainline_summary_cn.md`

## 向后兼容
- `--enable-court-mainline=false` 时，`run_silver_baseline_v0` 行为与旧版本保持一致。

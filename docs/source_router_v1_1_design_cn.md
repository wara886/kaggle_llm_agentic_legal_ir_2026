# source_router_v1_1 设计说明

## 目标
- 仅做查询源路由（query_source_router）规则细化，修复 `v1` 偏向 laws 的问题。
- 在不改 retrieval/fusion/rerank 主逻辑前提下，把应走 court/hybrid 的查询更稳定分流出去。

## 设计要点

### 1. 信号增强（route signals）
- 案例模式信号（case_pattern_signal）：
  - `BGE`
  - `\d+[A-Z]_[0-9]+/[0-9]{4}`
  - `E. x.x`
  - `1B_/6B_/8C_/5A_/4A_` 前缀案号
- 法规模式信号（statute_pattern_signal）：
  - `Art.` / `Abs.` / `lit.`
  - `ZGB/OR/BGG/StPO/BV/IPRG/IVG/ATSG/SVG/ZPO/StGB` 等
- 混合信号（mixed_signal）：
  - case 和 statute 同时强命中时，优先判定 `hybrid`。

### 2. 决策升级（router scoring）
- 保留输出字段：
  - `primary_source`（laws/court/hybrid）
  - `secondary_source`（laws/court/none）
  - `route_confidence`（low/medium/high）
- 新增内部分数：
  - `case_score`
  - `statute_score`
  - `mixed_score`
- 决策规则：
  - `case_score` 明显高于 `statute_score` -> `court`
  - `statute_score` 明显高于 `case_score` -> `laws`
  - 两者都高 -> `hybrid`
  - 不明确时默认 `hybrid`（从旧版偏 laws 改为保守分流）

### 3. 调试可观测性（route_debug）
- 在 per-query 追踪中追加：
  - `case_score`
  - `statute_score`
  - `mixed_score`
  - `matched_case_patterns`
  - `matched_statute_patterns`
  - `route_decision_reason_v1_1`

## 工程接入
- `src/source_router.py`
  - 保留 `route_query`（v1）兼容。
  - 新增 `route_query_v1_1`。
- `scripts/run_silver_baseline_v0.py`
  - 新增参数 `--router-version {v1,v1_1}`，默认 `v1`。
  - 仅路由调用与 trace 字段最小增量，不重写主流程。
- `scripts/run_source_router_v1_1_eval.py`
  - 对比 `no_router / source_router_v1 / source_router_v1_1`。
  - 输出路由质量与召回/F1 指标。

## 向后兼容
- 默认仍使用 `router v1`，旧行为保持不变。
- `router v1_1` 需显式开启。

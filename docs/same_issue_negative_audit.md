# Same-Issue Negative Audit

## Scope
- Baseline mining reference: `artifacts/laws_minilm_p1_scalealign/*`
- Audited code (before same-issue fix):
  - `scripts/mine_laws_hard_negatives_minilm.py`
  - `scripts/train_laws_minilm_biencoder.py` (`--query-mode laws_structured` in training)
  - `src/law_family.py` issue-term utilities

## 1) 当前是否能稳定提取 issue phrase / issue token
### Observation (before fix)
- Using only `issue_query_terms(...)` on train queries:
  - non-empty rate on train queries (`1139`) = `0.0`.
- On mined triplets (`993`), issue terms non-empty only `4` samples (very sparse).

### Diagnosis
- 训练集 query 以德语为主，而现有 issue-rule cues 偏英文语义短语；
- 导致 issue 信号在 mining 阶段提取不稳定、覆盖不足。

## 2) 负样本里是否存在“同 family 但 issue 更接近 gold”的 near miss
### Observation (before fix)
- same-family near miss rate ~= `3.0%`（非常低）。
- same-issue overlap rate ~= `0.0`（在可提取 issue terms 的样本中几乎没有）。

### Conclusion
- 存在极少，整体不可依赖，远不足以形成稳定训练信号。

## 3) 过易 negatives 是否仍然占比较高
### Observation (before fix)
- easy-negative heuristic rate ~= `96.98%`（高比例）。

### Conclusion
- 是，过易 negatives 占比很高，说明当前负样本质量与目标瓶颈（non-explicit 的细粒度区分）存在明显错位。

## Immediate correction direction
- 采用 `prefer_same_issue_near_miss` 作为主采样信号；
- same-family 仅作 tie-breaker；
- 当 issue terms 可用时优先过滤 issue-overlap=0 的过易候选。

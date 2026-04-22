# Current Laws-First Bottleneck Report

Run analyzed: `outputs/release_submission_laws_first_v1` (current frozen submission mainline).
Known public score: `0.01357`.

## 1) Gold 主要死在哪一层
Using per-gold stage tracing on validation:

| stage | count |
|---|---:|
| not_in_fused | 224 |
| not_in_rerank_input | 0 |
| reranked_too_low | 20 |
| cut_by_final_selection | 0 |
| kept_final | 7 |

结论：
- 当前第一瓶颈是 `not_in_fused`（远高于其他项）。
- 第二瓶颈是 `reranked_too_low`（说明已有一部分 gold 被召回，但被轻量 reranker 压低）。
- 在当前配置里 `not_in_rerank_input` 基本不是问题（因为未启用 rerank input shaping，rerank input 基本覆盖 fused 候选）。

## 2) 哪个子集最拖分
Validation (current release run):

| subset | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| explicit citation | 0.131266 | 0.050638 | 0.050638 | 19 |
| non-explicit citation | 0.142403 | 0.043232 | 0.043232 | 32 |

补充：
- `predicted-family-correct but rerank-fail` 现象依旧明显：在 family 预测命中 gold family 的情况下，仍有较多 gold 落在 `reranked_too_low`。
- 这类样本是 laws-side final selection/rerank calibration 的核心损失带。

结论：
- 最拖分子集是 **non-explicit citation**（F1 更低，且 FP 负担更高）。
- `predicted-family-correct` 但 rerank 失败是非显式子集中的关键可优化口。

## 3) public score 0.01357 的最可能限制项
最可能是“**laws-side recall ceiling + 非显式 rerank排序损失叠加**”：
- 大头损失来自 `not_in_fused`（laws-only + long-tail gold 覆盖不足）。
- 次级损失来自 `reranked_too_low`（尤其 non-explicit、family 已基本对齐的样本）。
- 这与当前冻结 court 的策略一致：不是 court 缺失导致的单点问题，而是 laws-first 路径下的召回覆盖与最终排序传导效率问题。

# sparse_laws citation-aware boost 设计说明（v1_3）

## 目标
- 仅在 `sparse_laws` 分支内部增强 citation 信号（citation-aware boost）。
- 在保持 dense/court/rerank/fusion 主流程不变的前提下，缓解：
1. `citation_weak_but_title_specific`
2. `text_dominant_false_positive`

## 信号来源
- 法律编号模式（legal citation patterns）：`Art.`, `Abs.`, `lit.` 等。
- 法典家族一致性（statute family match）：`ZGB`, `OR`, `BGG`, `StPO` 等。
- citation token overlap：query/query expansion 与候选 citation 的结构 token 重叠。

## 模式（citation_aware_boost_mode）
- `pattern_only`
- `pattern_plus_family`
- `pattern_plus_family_plus_overlap`

## 参数
- `--enable-citation-aware-boost`
- `--citation-pattern-match-boost`
- `--statute-family-match-boost`
- `--citation-token-overlap-boost`
- `--citation-aware-boost-mode`

## 评测脚本
`scripts/run_sparse_laws_citation_boost_eval_v1_3.py`

固定：
- 最优 sparse_laws 权重配置
- 最优 high_df title penalty 配置

小网格：
- `citation_pattern_match_boost`: `0.05, 0.1, 0.15, 0.2`
- `statute_family_match_boost`: `0.05, 0.1, 0.15`
- `citation_token_overlap_boost`: `0.05, 0.1, 0.15`
- `citation_aware_boost_mode`: `pattern_only`, `pattern_plus_family`, `pattern_plus_family_plus_overlap`

输出：
- `artifacts/v1_3_sparse_laws_citation_boost/citation_boost_results.csv`
- `artifacts/v1_3_sparse_laws_citation_boost/best_citation_boost_config.json`
- `artifacts/v1_3_sparse_laws_citation_boost/citation_boost_summary_cn.md`

## 兼容性
- 默认关闭，保持向后兼容。

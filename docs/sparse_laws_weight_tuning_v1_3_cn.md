# sparse_laws 分支内字段权重审计与调参（v1_3）

## 目标
- 只审计 `sparse_laws` 分支内部排序，不新增检索能力，不改 fusion 主逻辑。
- 验证是否存在“字段权重导致前排假阳性压制 gold”的问题。

## 固定不变项
- 查询预处理（query_preprocessing）与法规扩展（laws_query_pack_v2）保持 baseline_v1_3 逻辑。
- dense / court / rule 分支保持不变，仅作为融合输入。
- 不引入 rerank。

## 调参空间
- `laws_citation_weight`: `1.0, 1.5, 2.0, 3.0`
- `laws_title_weight`: `1.0, 1.5, 2.0, 3.0`
- `laws_text_weight`: `0.5, 1.0, 1.5, 2.0`

## 评估指标
- `sparse_laws_Recall@100`
- `sparse_laws_Recall@200`
- `fusion_final_Recall@100`
- `fusion_final_Recall@200`
- `strict_macro_f1`
- `corpus_aware_macro_f1`

## 假阳性剖面（false positive profile）
针对 “gold 被压在后面” 的 query，输出压过 gold 的前 20 条错误候选，并拆解字段分数组件：
- `fp_citation_score_component`
- `fp_title_score_component`
- `fp_text_score_component`

并给出启发式解释 `why_fp_likely_wins`：
- `text 过强（text_over_power）`
- `title 模板过泛（title_template_too_generic）`
- `citation 信号太弱（citation_signal_too_weak）`
- 其他混合优势

## 产物
- `artifacts/v1_3_sparse_laws_tuning/sparse_laws_weight_results.csv`
- `artifacts/v1_3_sparse_laws_tuning/best_sparse_laws_weight_config.json`
- `artifacts/v1_3_sparse_laws_tuning/sparse_laws_false_positive_profile.csv`
- `artifacts/v1_3_sparse_laws_tuning/sparse_laws_weight_summary_cn.md`

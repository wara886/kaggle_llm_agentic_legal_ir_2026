# Silver-Core Reproduction Plan

Goal: reproduce the verified silver notebook core in this repo without jumping straight to the giant 2.6M court setup.

## P0: Laws-First Precision Core

Scope:
- Rule exact match plus citation normalization.
- German keyword / abbreviation expansion using the current dictionary-based expansion as the first non-LLM version.
- Laws-only or laws-first sparse + dense hybrid retrieval with RRF.
- Conservative rerank: compare fixed small top-k against current relative-threshold behavior.

Current patch:
- `src/legal_ir/normalization.py`
  - `normalize_citation` now removes `lit.` and `Ziff.` detail, matching the notebook's citation granularity finding.
- `src/retrieval_rules.py`
  - `RuleCitationRetriever.build` can build laws-only rule indexes with `max_court_rows=0`.
  - `RuleCitationRetriever.search` skips unused sources when top-k is zero.
- `scripts/run_silver_baseline_v0.py`
  - Adds `--enable-rule-exact` and `--rule-top-k-laws`.
  - Builds a laws-only `RuleCitationRetriever`.
  - Prepends rule laws hits to final predictions before reranked retrieval candidates.
  - Adds `rule_laws_exact_count` and `rule_laws_exact_citations` to trace rows.

Patch output checklist:

| Required item | Content |
|---|---|
| 修改文件列表 | `src/legal_ir/normalization.py`, `src/retrieval_rules.py`, `scripts/run_silver_baseline_v0.py`, `docs/silver_notebook_to_repo_map.md`, `docs/silver_core_reproduction_plan.md` |
| 关键 diff 摘要 | Normalize away `lit.` / `Ziff.` granularity; allow laws-only rule index/search; prepend laws rule hits into silver final predictions; trace rule hit counts. |
| 运行命令 | See the P0 command below. |
| 预期观察指标 | strict/corpus F1, rule hit count, laws Recall@200, final prediction count per query. |

Recommended command:

```powershell
python scripts/run_silver_baseline_v0.py --enable-rule-exact true --rule-top-k-laws 20 --enable-court-dense false --seed-floor-sparse 0 --seed-floor-dense 0 --prefer-strong-reranker true --fixed-top-k 5 --dynamic-mode fixed_top_k --out-dir outputs/silver_core_p0_rule_exact
python scripts/phase0_evaluate_submission.py --split val --pred-file outputs/silver_core_p0_rule_exact/val_predictions_silver_baseline_v0.csv --out-dir artifacts/silver_core_p0_rule_exact_eval
```

Primary observation metrics:
- `strict_macro_f1`
- `corpus_aware_macro_f1`
- rule exact hit count in `val_seed_trace_silver_baseline_v0.csv`
- laws-only / laws-first Recall@200
- final prediction count per query, because notebook shows Macro F1 punishes extra false positives hard

P0 exit criteria:
- Rule exact hits are present in final predictions.
- Citation normalization increases exact or prefix matches for explicit `Art. ... lit./Ziff.` queries.
- Fixed small top-k is no worse than relative threshold on strict/corpus F1.
- Court candidates are not required for baseline P0 lift.

## P1: Fine-Tuned MiniLM Laws Retriever

Scope:
- Add hard negative mining over laws-first retrieval results.
- Train MiniLM on query, positive law text, hard negative law text.
- Re-encode laws with the fine-tuned model.
- Keep court out of the training target unless P0 analysis proves a court-specific lane is necessary.

Proposed files:
- `scripts/mine_hard_negatives_minilm.py`
- `scripts/train_minilm_legal_retriever.py`
- `scripts/build_laws_dense_index.py`

Training design:
- Anchor: train query.
- Positive: gold citation text from `laws_de.csv`.
- Hard negative: top retrieved non-gold laws candidate from current P0 laws-first retriever.
- Loss: `sentence_transformers.losses.MultipleNegativesRankingLoss`.
- Base model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, matching notebook cells 75 and 78.

P1 exit criteria:
- Fine-tuned laws dense Recall@50/100/200 improves over base MiniLM on validation.
- Rebuilt laws index improves strict/corpus F1 with conservative fixed top-k.
- Gains are attributable to laws dense retrieval, not court expansion.

## P2: Conservative Federated Court Lane

Scope:
- Re-evaluate whether court is needed after P0/P1.
- If needed, add court as a separate, capped federated lane.
- Do not use naive unified 2.6M retrieval as the default.

Court lane constraints:
- Trigger only when route/citation signals justify it.
- Cap court outputs to a very small top-k.
- Keep laws lane protected so laws candidates cannot be buried.
- Measure false positives before expanding court volume.

P2 exit criteria:
- Federated court lane improves Macro F1, not only Recall@200.
- Court false positives per query stay controlled.
- Laws-first performance does not regress.

## Why This Patch First

The notebook's first reliable win is not a bigger index; it is a high-precision tool path: exact legal citation extraction, normalized to the corpus's citation granularity, then laws-first hybrid retrieval. The repo already had pieces of that path, but the silver mainline was not using them. This patch connects that path with minimal code and gives P0 a measurable, notebook-aligned baseline.

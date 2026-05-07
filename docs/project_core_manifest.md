# Project Core Manifest

## Purpose
- Keep the Kaggle legal IR project focused on the reproducible mainline after the public-score plateau at `0.08960`.
- Remove failed experiment clutter from `outputs`, `artifacts`, `release`, and noisy one-off audit docs.

## Preserved Core
- Raw data: `data_raw/competition_data/train.csv`, `val.csv`, `test.csv`, `laws_de.csv`, `court_considerations.csv`.
- Source package: `src/`.
- Mainline runner: `scripts/run_silver_baseline_v0.py`.
- Qwen reranker runner: `scripts/run_qwen3_reranker_module_ablation.py`.
- Diagnostics: `scripts/run_current_best_residual_audit.py`, `scripts/run_test_query_cluster_audit.py`, `scripts/run_public_score_proxy_audit.py`.
- Training utilities: `scripts/mine_laws_hard_negatives_minilm.py`, `scripts/train_laws_minilm_biencoder.py`, `scripts/build_laws_minilm_index.py`.
- Previous best release: `release/submission_qwen3_bgg100_prior_v1/submission.csv`.
- Current best release: `release/submission_explicit_prefix_rescue_conjunction_top3_v8/submission.csv`.

## Current Verdict
- `0.11368` is the best public score after explicit prefix + natural alias + conjunction rescue.
- Broad family/issue patches are rejected because local gains did not transfer.
- The confirmed breakthrough is explicit citation grammar resolution against `laws_de.csv`:
  - abbreviation prefix rescue,
  - natural statute aliases such as `Art. 125 of the CC`,
  - conjunction forms such as `Art. 38 and 39 CO`.
- Train/test semantic memory was tested and rejected because lexical and multilingual-neighbor transfer did not improve val safely.

# Kaggle LLM Agentic Legal IR 2026

This repository contains a reproducible legal information retrieval pipeline for the Kaggle Swiss legal IR task. The system predicts citation IDs for legal case queries using a laws-first hybrid retrieval stack plus targeted citation and law-family corrections.

## Current Best

- Public score: `0.25015`
- Current best submission: `release/submission_surface_anchor_escape_combo_v20_test015_mandate_partnership_maintenance_tight_local/submission.csv`
- Best-candidate rationale: see `docs/experiment_log.md` and `docs/next_optimization_handoff.md`
- Resume/interview project review: `docs/kaggle_legal_ir_project_resume_review_v2_cn.md`

The later v21-v25 submissions were boundary checks and stayed flat at `0.24075`; the active best remains v20 tight.

## System Map

```text
query
  -> query preprocessing and multilingual legal phrase expansion
  -> source routing
  -> sparse BM25 retrieval
  -> dense MiniLM retrieval
  -> rule-based exact citation retrieval
  -> RRF fusion
  -> optional Qwen3 yes/no reranking
  -> final cut and evidence calibration
  -> citation grammar, law-family audit, and low-spillover patching
  -> submission.csv
```

## Key Code

- `scripts/run_silver_baseline_v0.py` - main retrieval and submission generation pipeline.
- `scripts/run_qwen3_reranker_module_ablation.py` - Qwen3 reranker A/B runner.
- `scripts/run_surface_family_audit.py` - test-surface law-family audit tooling.
- `scripts/run_targeted_test_patch.py` - low-spillover row-level patch runner.
- `scripts/evaluate_submission_official_strict.py` - official-style strict local evaluator.
- `src/legal_ir/corpus_builder.py` - citation-row corpus construction.
- `src/retrieval_sparse.py` - BM25 and sparse retrieval.
- `src/retrieval_dense.py` - MiniLM / dense retrieval.
- `src/retrieval_rules.py` - exact citation and grammar retrieval.
- `src/fusion.py` - RRF and candidate fusion.
- `src/rerank.py` - reranking helpers.
- `src/law_family.py` and `src/query_family.py` - law-family and issue-family rules.
- `src/citation_normalizer.py` and `src/legal_ir/normalization.py` - citation normalization.

## Project Memory

- `docs/project_core_manifest.md` - compact project manifest.
- `docs/current_progress_summary.md` - current progress and active baseline.
- `docs/experiment_log.md` - public-score and ablation history.
- `docs/next_optimization_handoff.md` - next-step handoff notes.
- `docs/kaggle_legal_ir_project_resume_review_v2_cn.md` - Chinese project review for resume/interview use.

## Data Boundary

Kaggle raw data and generated artifacts are intentionally not committed:

- `data_raw/`
- `artifacts/`
- `outputs/`
- `logs/`

To reproduce locally, place the competition files under `data_raw/competition_data/` with the expected Kaggle filenames:

- `train.csv`
- `val.csv`
- `test.csv`
- `laws_de.csv`
- `court_considerations.csv`

## For Web ChatGPT / GPT-5.5

When this repository is public on GitHub, a web ChatGPT session can inspect it by searching for:

```text
wara886/kaggle_llm_agentic_legal_ir_2026 llms.txt
wara886/kaggle_llm_agentic_legal_ir_2026 run_silver_baseline_v0.py
site:github.com/wara886/kaggle_llm_agentic_legal_ir_2026 law_family.py
```

Start with `llms.txt`, then read this README, then read the files listed in "Key Code" and "Project Memory".

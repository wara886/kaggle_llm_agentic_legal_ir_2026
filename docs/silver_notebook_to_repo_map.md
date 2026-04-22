# Silver Notebook To Repo Map

Source notebook: `docs/agentic-rag-fine-tuned-minilm-2-6m-faiss.ipynb`

## Notebook Core Architecture

| Notebook module | Notebook evidence | Repo landing point | Status |
|---|---|---|---|
| Regex / rule-based exact citation extraction | Cells 16, 31, 43, 46 repeatedly extract `Art.` citations before retrieval and add exact matches with absolute confidence. | `src/retrieval_rules.py::RuleCitationRetriever`, `scripts/run_silver_baseline_v0.py::run_split` | Now connected to silver mainline as a laws-only high-confidence prefix. |
| Citation normalization / abbreviation and granularity mapping | Cells 25, 29, 31 remove `lit.` / `Ziff.` because DB citations usually stop at `Art.` / `Abs.` granularity. | `src/legal_ir/normalization.py::normalize_citation`, `src/citation_normalizer.py` | Existing normalizer was too shallow; patch now drops `lit.` / `Ziff.` detail. Abbreviation-to-SR mapping is still incomplete and should stay P0/P1 scoped to laws. |
| English query to German legal keywords / abbreviations | Cells 34, 37, 46, 58 use a lawyer prompt to emit German keywords and Swiss abbreviations such as `StPO`, `ZGB`, `OR`, `BGG`. | `src/query_expansion.py::build_source_aware_query_packs`, `expand_query_from_multi_view` | Present, but dictionary-based and narrower than notebook LLM expansion. Good P0 fallback; richer abbreviation expansion remains missing. |
| Laws-first hybrid retrieval | Cells 22, 40, 43, 46 use laws BM25 + laws FAISS + RRF, then add only a small top-k. Cell 67 says laws-only had the best early public score. | `src/retrieval_sparse.py`, `src/retrieval_dense.py`, `src/fusion.py::rrf_fusion`, `scripts/run_silver_baseline_v0.py` | Partially present, but current silver script mixes route floors and court candidates into the same final lane. Needs an explicit laws-first primary lane. |
| Federated search | Cell 58 separates laws search from court search so laws are not buried by court cases. | `scripts/run_silver_baseline_v0.py` route-aware quotas, separate `laws_de` / `court_considerations` retrieval, `src/fusion.py` | Present, but current implementation has become court-heavy. Notebook argues against naive unified giant search and for conservative court inclusion. |
| Reranker usage | Cells 61, 66, 69 show cross-encoder reranking; Cell 66 fixes empty/noisy output by taking top 3, Cell 70 warns relative logit thresholds are unsafe. | `scripts/run_silver_baseline_v0.py::StrongBGEReranker`, `apply_dynamic_cut`, `src/rerank.py` | Reranker exists. Direction is partly off: default `relative_threshold=0.85` is exactly the risky pattern the notebook warns about. Do not change model now; evaluate fixed small top-k for silver-core. |
| Hard negative mining | Cells 72, 75 mine retrieved non-gold candidates as negatives. | Missing production script. Closest patterns are diagnostics scripts and `legal_ir.data_loader`. | Missing. Add P1 script after P0 laws-first retrieval is stable. |
| MiniLM fine-tuning | Cell 75 fine-tunes `paraphrase-multilingual-MiniLM-L12-v2` with `MultipleNegativesRankingLoss`. | Missing. `src/retrieval_dense.py` can load a model name, but there is no training pipeline. | Missing. Add P1 training script, not in P0. |
| Rebuild index with fine-tuned model | Cells 78, 81 re-encode laws first, then optionally the giant court index. | `src/retrieval_dense.py::DenseRetriever.build` can rebuild embeddings in memory; no persisted FAISS index builder exists. | Missing persisted index workflow. P1 should re-encode laws only; P2 can consider court. |

## Existing Repo Pieces That Can Carry The Notebook Logic

| Repo file/function | Can carry |
|---|---|
| `src/retrieval_rules.py::RuleCitationRetriever` | Notebook cells 16/31/43 exact citation tool. |
| `src/legal_ir/normalization.py::normalize_citation` | Notebook cells 25/29 granularity normalization. |
| `src/query_expansion.py::build_source_aware_query_packs` | P0 dictionary replacement for notebook's LLM German keyword prompt. |
| `src/retrieval_sparse.py::SparseRetriever.search_field_aware` | Laws BM25 side of laws-first hybrid retrieval. |
| `src/retrieval_dense.py::DenseRetriever.search_source_aware` | Dense laws side, currently SBERT or hashing fallback rather than persisted FAISS. |
| `src/fusion.py::rrf_fusion` | Notebook RRF fusion. |
| `scripts/run_silver_baseline_v0.py::run_split` | Current executable mainline where exact rules, laws lane, conservative rerank, and later federated court should be wired. |

## Modules Already Present But Directionally Off

| Area | Why it is off relative to notebook |
|---|---|
| Court-heavy candidate floors | Notebook's strongest evidence is that naive or over-eager court expansion lowered score because false positives dominate Macro F1. Current `seed_floor_sparse/dense=20` pushes court candidates even for laws routes. |
| Court dense emphasis | Notebook Cell 53 says 2.6M unified dense search dropped from the laws-only score. Current recent work kept trying to repair court dense/sparse seeds instead of stabilizing laws-first precision. |
| Relative rerank threshold | Notebook Cell 70 warns logit-relative thresholds are not calibrated probabilities. Current default `relative_threshold=0.85` should be compared against fixed small top-k in P0, without changing the reranker model. |
| Rule branch outside silver mainline | Older scripts use `RuleCitationRetriever`, but `run_silver_baseline_v0.py` did not actually include it before this patch. |

## Missing Modules

| Missing module | Minimal repo target |
|---|---|
| Rich abbreviation map (`StPO`, `StGB`, `ZGB`, `OR`, `BGG`, `BV`, plus SR aliases) | Extend `src/legal_ir/normalization.py` and `src/query_expansion.py` after measuring exact-rule lift. |
| Hard negative mining | New `scripts/mine_hard_negatives_minilm.py`, reading train rows, gold citations, and current laws-first retrieval misses. |
| MiniLM fine-tuning | New `scripts/train_minilm_legal_retriever.py` with `sentence_transformers` triplets and `MultipleNegativesRankingLoss`. |
| Persisted laws FAISS / embedding index | New `scripts/build_laws_dense_index.py` or a storage layer under `artifacts/indexes/`. |
| Conservative federated court lane | P2 only, after laws-first + fine-tuned laws index improves or plateaus. |

## Court-Heavy Logic In Conflict With Notebook

The notebook's strongest lesson is not "index everything"; it is "protect precision first." Cells 47, 53, 67, and 84 show that laws-only or laws-first variants were more reliable than the giant unified court index, and that the 2.6M court path can dilute results. Current court seed repair work conflicts with that lesson when it forces court candidates into ordinary laws queries, prioritizes court dense repair before laws-first reproduction, or uses broad thresholds that add extra false positives.


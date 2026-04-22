# Silver vs Repo Training Gap Audit

## Scope
Compared:
- Silver notebook training loop (`agentic-rag-fine-tuned-minilm-2-6m-faiss.ipynb`)
- Current repo loop (`scripts/mine_laws_hard_negatives_minilm.py`, `scripts/train_laws_minilm_biencoder.py`, `scripts/build_laws_minilm_index.py`, `scripts/run_silver_baseline_v0.py`, `src/retrieval_dense.py`, `src/legal_ir/evaluation.py`)

## Side-by-side mapping

| Area | Silver notebook | Repo current behavior | Gap | Why this can cause P1 no gain |
|---|---|---|---|---|
| Hard negative mining pool | Unified dense FAISS over laws + court (`~2.65M`) | `mine_laws_hard_negatives_minilm.py` mines from laws-focused sparse+dense fusion list | Mining source and difficulty distribution differ | Negatives become less notebook-like; contrastive signal shifts away from silver's failure mode |
| Mining scale | Full `train.csv` (`1139` queries), mined `1139` triplets in notebook run | Existing artifact used `max_rows=300`, produced `264` triplets | Substantially smaller training set | Model receives weak domain adaptation signal |
| Training objective | `MultipleNegativesRankingLoss` | Historical repo artifact was trained with `TripletLoss` (`training_meta.json`) | Objective mismatch | Representation geometry differs; tuned model may not improve retrieval ranking where silver improved |
| Positive/negative text form | Mostly pure `laws_df.text` (citation fallback) | `_doc_text` uses `title + truncated text + citation` for both pos/neg | Input text shape mismatch | Extra template tokens/citation boilerplate can dilute semantic learning |
| Reindex + inference coupling | Fine-tune then immediate tuned FAISS rebuild, then tuned retrieval | Repo has `build_laws_minilm_index.py`, but runtime path mainly rebuilds in-memory via `DenseRetriever.build`; persisted index artifacts are not the serving path | Reindex path differs from notebook ops | Harder to verify strict training-serving parity and controlled reproducibility |
| Eval endpoint | Kaggle submission/public score | Local strict/corpus macro F1 + P0/P2/P3 post-processing pipeline | Evaluation objective mismatch | Dense-side gains can be masked by downstream selection and fixed top-k cuts |

## File/function locations audited
- Hard negative mining:
  - `scripts/mine_laws_hard_negatives_minilm.py::main`
  - key calls: `SparseRetriever.search_field_aware`, `DenseRetriever.search_source_aware`
- Pair/triplet construction:
  - `scripts/mine_laws_hard_negatives_minilm.py` (triplet JSONL fields)
  - `scripts/train_laws_minilm_biencoder.py` (`InputExample(texts=[query, positive_text, negative_text])`)
- MiniLM fine-tune:
  - `scripts/train_laws_minilm_biencoder.py::main`
- Laws index rebuild:
  - `scripts/build_laws_minilm_index.py::main`
  - runtime dense build: `src/retrieval_dense.py::DenseRetriever.build`
- Dense retrieval inference:
  - `src/retrieval_dense.py::search_source_aware`
  - `scripts/run_silver_baseline_v0.py::run_split`
- Final evaluation:
  - `src/legal_ir/evaluation.py::evaluate_predictions`
  - plus final prediction path in `scripts/run_silver_baseline_v0.py`

## Top 5 likely blockers (priority order)

### 1) Training objective mismatch (highest priority)
- Difference:
  - Silver: MNRL
  - Repo historical P1 artifact: TripletLoss
- Why it can nullify gain:
  - Loss function directly determines embedding separation geometry; notebook win is tied to MNRL loop.
- Type: `training objective`
- Fix cost: `low`
- Expected gain: `high`

### 2) Mining data volume too small
- Difference:
  - Silver mined full train scale; repo run used 300-row cap / 264 examples.
- Why it can nullify gain:
  - Adaptation signal too weak for non-explicit hard cases.
- Type: `data construction`
- Fix cost: `low`
- Expected gain: `high`

### 3) Negative source distribution mismatch
- Difference:
  - Silver negatives from unified dense pool (laws+court); repo from laws-focused fused retrieval.
- Why it can nullify gain:
  - Hard negatives may be easier / qualitatively different from silver's confusion set.
- Type: `negative quality`
- Fix cost: `medium`
- Expected gain: `medium-high`

### 4) Training text template mismatch (title+citation injection)
- Difference:
  - Silver mostly uses legal article text body; repo training mixes title and citation boilerplate.
- Why it can nullify gain:
  - Non-semantic tokens add noise and may overfit citation format instead of legal semantics.
- Type: `data construction`
- Fix cost: `low-medium`
- Expected gain: `medium`

### 5) Evaluation path mismatch (dense gain masked downstream)
- Difference:
  - Silver public-score loop vs repo P0/P2/P3 final-cut pipeline.
- Why it can nullify gain:
  - Dense retrieval improvement may not propagate through rerank/final-cut bottlenecks.
- Type: `evaluation mismatch`
- Fix cost: `medium`
- Expected gain: `medium`

## Most likely single root cause for "P1 no gain"
- The **first** point to align is still objective mismatch: repo P1 was trained with TripletLoss while silver's key claim is from MNRL.

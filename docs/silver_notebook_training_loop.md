# Silver Notebook Training Loop Audit

Target notebook: `docs/agentic-rag-fine-tuned-minilm-2-6m-faiss.ipynb`

## 1. Hard Negative Mining: data source and filtering rules
- Data source for mining:
  - Query side: full `train.csv` (`1139` rows).
  - Retrieval pool: unified dense FAISS index over **laws + court** (`175,933` laws + chunked court to `2,652,248` total vectors).
- Negative mining rule:
  - For each query, run dense search on unified FAISS (`top_k=15`).
  - Pick the first retrieved citation not in query gold set as `hard negative`.
  - If citation exists in laws table, negative text = law `text`; otherwise negative text = citation string (court citation fallback).

## 2. Training sample construction
- Query text:
  - `query_text = row['query']` (raw query string from train set).
- Positive text:
  - Randomly sample one gold citation.
  - If sampled citation is in `laws_df`, use `laws_df.text` as positive text.
  - Else fallback to citation string.
- Negative text:
  - Hard negative selected from unified dense retrieval output as above.
- Sample form:
  - `InputExample(texts=[query_text, pos_text, neg_text])`
  - So data is triplet-like `(anchor, positive, negative)` and fed to sentence-transformers MNRL training.

## 3. MiniLM fine-tune objective, batching, and key hyperparameters
- Base model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Dataloader:
  - `shuffle=True`
  - `batch_size=16`
- Loss:
  - `losses.MultipleNegativesRankingLoss(model=embedding_model)`
- Fit config:
  - `epochs=1`
  - `warmup_steps=int(len(train_dataloader) * 0.1)`
- Saved model path:
  - `swiss-legal-minilm-v1`

## 4. Post-finetune laws index rebuild
- Notebook explicitly warns stale index risk after model update.
- Rebuild process:
  - Reload tuned model from `swiss-legal-minilm-v1`.
  - Re-encode laws corpus (`laws_df['text']`) with tuned model.
  - L2 normalize vectors.
  - Build new FAISS `IndexFlatIP` for tuned laws embeddings (`faiss_index_tuned`).
- Then submission path uses tuned model + tuned laws FAISS for dense retrieval.

## 5. Post-finetune evaluation path and true metric definition
- The notebook's "real" score claim is based on Kaggle submission/public LB score (`submission_finetuned_laws.csv` / public score shown in markdown), not strict/corpus local offline F1.
- Local notebook checks around this stage are retrieval examples and submission generation; there is no repo-style strict vs corpus split audit loop.
- Therefore notebook gain is measured in a different final evaluation context than this repo's strict/corpus local harness.

## Direct audit takeaway
- Silver notebook loop is: `unified dense hard-negative mining -> MNRL fine-tune -> rebuild tuned dense index -> Kaggle submission score`.
- The strongest defining traits are unified mining pool and MNRL objective with immediate reindex before inference.

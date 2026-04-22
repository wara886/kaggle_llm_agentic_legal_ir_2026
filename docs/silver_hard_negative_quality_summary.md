# Silver Hard Negative Quality Summary

Source notebook: `docs/agentic-rag-fine-tuned-minilm-2-6m-faiss.ipynb`

## 1) negatives 来自哪一层候选
- 来自 **dense FAISS 检索层**（`unified_faiss_index.search(q_emb, 15)`）。
- 不是 BM25 层，不是 sparse+dense hybrid fused 层，也不是 reranked 层。

## 2) 是否天然偏向难负样本
- 是，属于“高相似误召回”天然硬负样本：
  - 每个 query 从 dense top-15 中取第一个非 gold 候选。
  - 这是“被当前向量空间误认为相关”的候选，不是随机负样本。

## 3) 是否更接近 same-law-family near miss
- notebook 没有显式 same-family 约束。
- 但因为来自 dense top-k 误召回，很多负样本会自然更接近语义近邻，**可能**包含同家族 near miss（非硬规则保证）。

## 4) 是否更接近 same-issue near miss
- 同样没有显式 same-issue 规则。
- 其 hard 性来自检索相似度机制本身，因此通常会比随机采样更接近 same-issue 误匹配。

## 5) per-query negatives 质量控制方式
- 每 query 1 个 negative。
- 质量控制是“排名驱动”的隐式控制：只用 dense top-15 的最前非 gold。
- 没有额外困难度阈值、same-family/same-issue 显式过滤或多负样本重采样机制。

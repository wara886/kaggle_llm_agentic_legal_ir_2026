# Silver Hard Negative Mining Summary

Notebook: `docs/agentic-rag-fine-tuned-minilm-2-6m-faiss.ipynb`

## 1) Hard negatives 来源层级
- 来源是 **dense FAISS 检索层**，不是 BM25、不是 sparse+dense 融合、不是 reranker 后候选。
- 具体：先构建 unified FAISS（laws + court，约 2.65M 向量），然后对每个 train query 做 `index.search(..., top_k=15)`。

## 2) Hard negatives 筛选规则
- 规则非常直接：取 dense top-15 里第一个 `not in gold_citations` 的候选作为 negative。
- 未见如下额外约束：
  - 未过滤 trivial negatives（除“非 gold”外无难度阈值）。
  - 未做 same-law-family 偏置或约束。
  - 每 query 固定最多 1 个 negative（因此最多 1 triplet/query）。

## 3) 训练样本构造
- query 文本：`row['query']` 原始 query。
- positive 文本：随机采样一个 gold citation；若该 citation 在 `laws_df`，则用 `laws_df.text`，否则回退 citation 字符串。
- negative 文本：上面的 hard negative；若在 `laws_df` 用其 `text`，否则回退 citation 字符串。
- batch negatives 共享：使用 MNRL，batch 内其他样本的 positive 共同充当 in-batch negatives。

## 4) MNRL 关键训练设置
- base model：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- loss：`MultipleNegativesRankingLoss`
- batch size：`16`
- epochs：`1`
- warmup：`int(len(train_dataloader) * 0.1)`

## 5) Fine-tune 后 reindex 字段
- tuned laws-only reindex 使用的是 `laws_df['text']`（未拼 title/citation）。
- 之后也构建过 tuned unified index（laws text + court text）。
- 两种 reindex 都是以 `text` 字段为核心编码输入。

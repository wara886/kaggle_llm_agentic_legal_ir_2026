# Repo Hard Negative Mining Summary

## 1) 相关文件与函数
- Hard negative mining：
  - `scripts/mine_laws_hard_negatives_minilm.py::main`
  - 关键函数：`SparseRetriever.search_field_aware`、`DenseRetriever.search_source_aware`、`rrf_fusion`
- Training data construction / MNRL：
  - `scripts/train_laws_minilm_biencoder.py::main`
- Laws MiniLM fine-tune：
  - `scripts/train_laws_minilm_biencoder.py`
- Laws reindex：
  - `scripts/build_laws_minilm_index.py`
  - 线上推理侧实际是 `src/retrieval_dense.py::DenseRetriever.build`
- Dense inference：
  - `src/retrieval_dense.py::search_source_aware`
  - `scripts/run_silver_baseline_v0.py::run_split`
- 评估：
  - `src/legal_ir/evaluation.py::evaluate_predictions`

## 2) 当前 mining 闭环（改动前后的核心）
- 语料范围：laws-only（不含 court）。
- 候选来源层：`sparse(field-aware) + dense(source-aware)` 做 RRF 后，从 fused list 里选第一个非 gold 作为 negative。
- 每 query negative 数：1 个。
- 样本写入：`laws_hard_negative_triplets.jsonl`。

## 3) 训练样本格式
- 训练输入仍为 `InputExample([query, positive_text, negative_text])`。
- `query` 当前使用原始 query 文本。
- `positive_text` / `negative_text` 当前由 `_doc_text` 构造：`title + truncated text + citation`。
- MNRL 现已可默认对齐（loss default 改为 `multiple_negatives`）。

## 4) Reindex 与推理
- 持久化 reindex 脚本：`build_laws_minilm_index.py`，同样使用 `title + text + citation` 文本拼接编码。
- 实际评估主线通常直接在运行时 `DenseRetriever.build` 重新编码（并不读取 `.npy` 持久化索引）。

## 5) laws-first 有效结构化信号是否进入训练样本
- 已进入的部分：
  - family/issue/rules 结构化信号用于 **mining 阶段的候选召回**（通过 `preprocess_query + build_source_aware_query_packs`）。
- 未进入的部分：
  - 训练 anchor `query` 仍是原始 query；并未把线上 laws-first 的结构化 query 直接作为训练 query 文本写入样本。

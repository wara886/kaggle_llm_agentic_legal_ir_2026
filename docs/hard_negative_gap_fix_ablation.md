# Hard Negative Gap Fix Ablation

## 选定的最小 mining 修复
- 选择：**D（规模对齐）**
- 具体差异：去除 mining 默认截断（300/500），对齐 notebook 的“尽量全量 query 采样，每 query 1 个 negative”策略。
- 约束符合：
  - 不调 loss
  - 不改 court
  - 不改 reranker 主逻辑
  - 不新增 retrieval patch

## 修改文件列表
- `scripts/mine_laws_hard_negatives_minilm.py`

## 关键 diff 摘要
- 参数默认值：
  - `--max-rows: 300 -> -1`
  - `--max-examples: 500 -> -1`
- 停止条件修复：
  - `if len(examples) >= args.max_examples` -> `if args.max_examples > 0 and len(examples) >= args.max_examples`

## 运行命令

### 1) 重新 mining（规模对齐）
```bash
python scripts/mine_laws_hard_negatives_minilm.py \
  --split train \
  --out-dir artifacts/laws_minilm_p1_scalealign
```
产物：`triplets=993`（此前为 264）。

### 2) 用规模对齐 triplets 训练 MNRL
```bash
python scripts/train_laws_minilm_biencoder.py \
  --triplets artifacts/laws_minilm_p1_scalealign/laws_hard_negative_triplets.jsonl \
  --out-model-dir artifacts/laws_minilm_p1_scalealign/minilm_laws_finetuned_mnrl_993 \
  --max-examples 993 \
  --batch-size 16 \
  --epochs 1 \
  --warmup-ratio 0.1
```

### 3) 在冻结主线下评估（P0 + P2-A + P2-B + P3）
```bash
python scripts/run_silver_baseline_v0.py \
  --out-dir outputs/hard_negative_gap_fix_ablation_C_scalealign \
  --prefer-strong-reranker false \
  --dynamic-mode fixed_top_k --fixed-top-k 5 \
  --enable-court-mainline false --enable-court-dense false \
  --seed-floor-sparse 0 --seed-floor-dense 0 \
  --sparse-max-laws 175933 --dense-max-laws 175933 \
  --sparse-max-court 0 --dense-max-court 0 \
  --enable-rule-exact true --rule-top-k-laws 20 \
  --enable-law-family-constraints true \
  --enable-issue-phrase-refinement true \
  --enable-laws-final-cut-calibration true \
  --court-dense-model-name artifacts/laws_minilm_p1_scalealign/minilm_laws_finetuned_mnrl_993
```

## 预期观察指标
- 首看 non-explicit subset：`Recall@200`、`strict_f1/corpus_f1` 是否继续超过“仅 loss 对齐版”。
- 同时观察 final FP 是否可控（不显著恶化）。

## 最小 ablation 结果
对比：
- A：当前 `P0 + P2-A + P2-B + P3 + MNRL(loss 对齐版)`
- B：A + 本次最小 mining 修复（规模对齐）

### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|---:|
| A loss 对齐版 | 10 | 0.145091 | 0.047734 | 0.047734 | 52 |
| B + mining 规模对齐 | 10 | 0.163197 | 0.051508 | 0.051508 | 51 |
| delta (B-A) | - | +0.018106 | +0.003774 | +0.003774 | -1 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|---:|
| A loss 对齐版 | 4 | 0.131266 | 0.050638 | 0.050638 | 19 |
| B + mining 规模对齐 | 4 | 0.131266 | 0.050638 | 0.050638 | 19 |
| delta (B-A) | - | +0.000000 | +0.000000 | +0.000000 | +0 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|---:|
| A loss 对齐版 | 6 | 0.154308 | 0.045798 | 0.045798 | 33 |
| B + mining 规模对齐 | 6 | 0.184484 | 0.052088 | 0.052088 | 32 |
| delta (B-A) | - | +0.030176 | +0.006290 | +0.006290 | -1 |

## 结论判定
- 这次不是“仍无明显增益”，而是有可见增益，尤其 non-explicit subset 明显继续提升。
- 因此当前证据更支持：**mining 规模差异是主瓶颈之一**，而不是“训练路线本身不值得继续”。

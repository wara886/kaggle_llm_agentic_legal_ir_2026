# Hard Negative Quality Fix Ablation

## Minimal quality fix applied
Chosen candidate: **A. same-family near-miss negative 优先采样**

### Modified files
- `scripts/mine_laws_hard_negatives_minilm.py`

### Key diff summary
- Added flag: `--prefer-same-family-near-miss`
- Mining logic change:
  - if enabled, first try non-gold candidate in fused list whose family is in gold-family set;
  - fallback to old behavior (first non-gold).
- Added diagnostics/meta fields:
  - `positive_family`, `negative_family`, `same_family_near_miss`
  - meta flag `prefer_same_family_near_miss`

## Run commands

### 1) Re-mine triplets with quality fix (same scale)
```bash
python scripts/mine_laws_hard_negatives_minilm.py \
  --split train \
  --prefer-same-family-near-miss \
  --out-dir artifacts/laws_minilm_p1_qualityalign
```

### 2) Train with same training setup (MNRL + laws_structured)
```bash
python scripts/train_laws_minilm_biencoder.py \
  --triplets artifacts/laws_minilm_p1_qualityalign/laws_hard_negative_triplets.jsonl \
  --out-model-dir artifacts/laws_minilm_p1_qualityalign/minilm_laws_finetuned_mnrl_structq_993 \
  --max-examples 993 \
  --batch-size 16 \
  --epochs 1 \
  --warmup-ratio 0.1 \
  --loss multiple_negatives \
  --query-mode laws_structured
```

### 3) Eval with frozen mainline config
```bash
python scripts/run_silver_baseline_v0.py \
  --out-dir outputs/hard_negative_quality_fix_ablation \
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
  --court-dense-model-name artifacts/laws_minilm_p1_qualityalign/minilm_laws_finetuned_mnrl_structq_993
```

## Quality-shift sanity check
- baseline (`scalealign`): same-family near-miss rate ≈ `0.0%` in diagnostics (family fields absent / effectively none)
- quality-fix (`qualityalign`): same-family near-miss rate ≈ `21.15%`

## Ablation comparison
Compare:
- Baseline: `current main + MNRL + full mining + laws_structured` (`outputs/train_structq_ablation`)
- + quality fix: `outputs/hard_negative_quality_fix_ablation`

### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|---:|
| baseline_structq_fullmining | 10 | 0.163758 | 0.057960 | 0.057960 | 50 |
| + quality fix | 10 | 0.151218 | 0.047734 | 0.047734 | 52 |
| delta | - | -0.012540 | -0.010226 | -0.010226 | +2 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|---:|
| baseline_structq_fullmining | 4 | 0.131266 | 0.050638 | 0.050638 | 19 |
| + quality fix | 4 | 0.131266 | 0.050638 | 0.050638 | 19 |
| delta | - | +0.000000 | +0.000000 | +0.000000 | +0 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|---:|
| baseline_structq_fullmining | 6 | 0.185419 | 0.062840 | 0.062840 | 31 |
| + quality fix | 6 | 0.164520 | 0.045798 | 0.045798 | 33 |
| delta | - | -0.020899 | -0.017042 | -0.017042 | +2 |

## Interpretation
- 这次并非“小幅增益”，而是明显负增益。
- 更可能解释：**修复点选错了/约束过强**，而不是“quality gap 不是问题本身”。
- same-family 优先虽然提高了 family 近邻率，但破坏了原有 hard-negative 难度分布与代表性，导致训练泛化下降。

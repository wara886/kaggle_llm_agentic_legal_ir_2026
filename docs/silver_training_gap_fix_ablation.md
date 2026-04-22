# Silver Training Gap Fix Ablation

## Goal
Validate one minimal training-loop alignment fix on top of frozen mainline:
- Base: `P0 + P2-A + P2-B + P3`
- Constraints respected:
  - no new retrieval patch
  - no court logic changes
  - no reranker main-logic changes

## Chosen minimal fix
- Fix selected: align fine-tune objective to silver notebook by switching default training loss to `MultipleNegativesRankingLoss`.
- Why this one: highest-priority audit gap with lowest implementation risk.

## Modified files
- `scripts/train_laws_minilm_biencoder.py`

## Key diff
- `--loss` default changed:
  - from: `triplet`
  - to: `multiple_negatives`

## Commands run

### 1) Train aligned MiniLM (same triplets, objective-aligned)
```bash
python scripts/train_laws_minilm_biencoder.py \
  --triplets artifacts/laws_minilm_p1/laws_hard_negative_triplets.jsonl \
  --out-model-dir artifacts/laws_minilm_p1_mnrl/minilm_laws_finetuned \
  --max-examples 264 \
  --batch-size 16 \
  --epochs 1 \
  --warmup-ratio 0.1
```

### 2) A run: frozen P0+P2-A+P2-B+P3 baseline
```bash
python scripts/run_silver_baseline_v0.py \
  --out-dir outputs/silver_training_gap_fix_ablation_A_p3_frozen \
  --prefer-strong-reranker false \
  --dynamic-mode fixed_top_k --fixed-top-k 5 \
  --enable-court-mainline false --enable-court-dense false \
  --seed-floor-sparse 0 --seed-floor-dense 0 \
  --sparse-max-laws 175933 --dense-max-laws 175933 \
  --sparse-max-court 0 --dense-max-court 0 \
  --enable-rule-exact true --rule-top-k-laws 20 \
  --enable-law-family-constraints true \
  --enable-issue-phrase-refinement true \
  --enable-laws-final-cut-calibration true
```

### 3) B run: A + aligned MNRL fine-tuned MiniLM
```bash
python scripts/run_silver_baseline_v0.py \
  --out-dir outputs/silver_training_gap_fix_ablation_B_p3_mnrl \
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
  --court-dense-model-name artifacts/laws_minilm_p1_mnrl/minilm_laws_finetuned
```

## Ablation results

### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|---:|
| A: P0+P2-A+P2-B+P3 | 10 | 0.137948 | 0.046195 | 0.046195 | 51 |
| B: A + MNRL MiniLM | 10 | 0.145091 | 0.047734 | 0.047734 | 52 |
| delta (B-A) | - | +0.007143 | +0.001539 | +0.001539 | +1 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|---:|
| A | 4 | 0.131266 | 0.050638 | 0.050638 | 19 |
| B | 4 | 0.131266 | 0.050638 | 0.050638 | 19 |
| delta (B-A) | - | +0.000000 | +0.000000 | +0.000000 | +0 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|---:|
| A | 6 | 0.142403 | 0.043232 | 0.043232 | 32 |
| B | 6 | 0.154308 | 0.045798 | 0.045798 | 33 |
| delta (B-A) | - | +0.011905 | +0.002566 | +0.002566 | +1 |

## Expected observation vs actual
- Expected:
  - If objective mismatch is real bottleneck, non-explicit subset should show at least small positive movement after MNRL alignment.
- Actual:
  - Non-explicit subset improved (`Recall@200` and `strict/corpus_f1` both up), but gain is small and comes with +1 FP.

## Interpretation
- This fix point was not wrong: it produced measurable, directionally correct lift.
- But improvement size is limited, so MNRL mismatch is likely **a contributing factor, not the full root cause**.
- Remaining major gaps are still likely in mining scale/negative quality and training-serving/evaluation mismatch.

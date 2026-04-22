# Laws-First High ROI Patch Ablation

## Patch
- File: `scripts/train_laws_minilm_biencoder.py`
- Minimal change: add optional `--query-mode laws_structured` for training query construction.
- Default behavior unchanged (`--query-mode raw`).

### Key behavior
When `laws_structured`:
- keep query clean text,
- append likely family terms,
- append issue terms,
- build training anchor closer to online laws-first effective query structure.

No changes to:
- retrieval branches,
- court path,
- reranker logic,
- fusion logic.

## Commands
### Train (structured query)
```bash
python scripts/train_laws_minilm_biencoder.py \
  --triplets artifacts/laws_minilm_p1_scalealign/laws_hard_negative_triplets.jsonl \
  --out-model-dir artifacts/laws_minilm_p1_scalealign/minilm_laws_finetuned_mnrl_structq_993 \
  --max-examples 993 \
  --batch-size 16 \
  --epochs 1 \
  --warmup-ratio 0.1 \
  --loss multiple_negatives \
  --query-mode laws_structured
```

### Eval (frozen laws-first mainline)
```bash
python scripts/run_silver_baseline_v0.py \
  --out-dir outputs/train_structq_ablation \
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
  --court-dense-model-name artifacts/laws_minilm_p1_scalealign/minilm_laws_finetuned_mnrl_structq_993
```

## A/B (vs previous training-enhanced best)
A: `outputs/hard_negative_gap_fix_ablation_C_scalealign` (MNRL + full mining)
B: `outputs/train_structq_ablation` (A + structured query training)

### overall
| run | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| A | 0.163197 | 0.051508 | 0.051508 | 51 |
| B | 0.163758 | 0.057960 | 0.057960 | 50 |
| delta (B-A) | +0.000561 | +0.006452 | +0.006452 | -1 |

### non-explicit
| run | Recall@200 | strict_f1 | corpus_f1 | final_fp |
|---|---:|---:|---:|---:|
| A | 0.184484 | 0.052088 | 0.052088 | 32 |
| B | 0.185419 | 0.062840 | 0.062840 | 31 |
| delta (B-A) | +0.000935 | +0.010752 | +0.010752 | -1 |

## Conclusion
- This patch is high ROI in current constraints:
  - no branch expansion,
  - no court changes,
  - no reranker training.
- Gains are concentrated where we need them most: non-explicit final quality, with lower FP.

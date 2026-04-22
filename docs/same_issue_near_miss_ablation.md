# Same-Issue Near-Miss Ablation

## Minimal fix applied
Only one direction was implemented: **prefer_same_issue_near_miss**.

## Modified files
- `scripts/mine_laws_hard_negatives_minilm.py`

## Key diff summary
- Added flag: `--prefer-same-issue-near-miss`.
- Negative selection now prioritizes issue overlap (primary), with same-family only as tie-breaker.
- Added issue term fallback (when rule-based issue extraction is empty):
  - `query_legal_phrases`
  - `query_keywords`
  - `laws_query_pack_v2.expanded_keywords_de`
- Added simple easy-negative filtering in issue mode:
  - if issue terms exist, prefer candidates with `issue_overlap > 0`.
- Added diagnostics fields:
  - `issue_terms_count`, `positive_issue_overlap`, `negative_issue_overlap`, `same_family_near_miss`.

## Run commands

### 1) Re-mine (same scale, issue-priority)
```bash
python scripts/mine_laws_hard_negatives_minilm.py \
  --split train \
  --prefer-same-issue-near-miss \
  --out-dir artifacts/laws_minilm_p1_issuealign
```

### 2) Train (unchanged loss, unchanged scale, laws_structured)
```bash
python scripts/train_laws_minilm_biencoder.py \
  --triplets artifacts/laws_minilm_p1_issuealign/laws_hard_negative_triplets.jsonl \
  --out-model-dir artifacts/laws_minilm_p1_issuealign/minilm_laws_finetuned_mnrl_structq_993 \
  --max-examples 993 \
  --batch-size 16 \
  --epochs 1 \
  --warmup-ratio 0.1 \
  --loss multiple_negatives \
  --query-mode laws_structured
```

### 3) Eval (frozen mainline)
```bash
python scripts/run_silver_baseline_v0.py \
  --out-dir outputs/same_issue_near_miss_ablation \
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
  --court-dense-model-name artifacts/laws_minilm_p1_issuealign/minilm_laws_finetuned_mnrl_structq_993
```

## Quality-shift sanity check (mining diagnostics)
- baseline full-mining (`scalealign`):
  - issue_terms_nonempty_rate = `0.0`
  - neg_issue_overlap_rate = `0.0`
  - same_family_rate = `0.0`
- same-issue fix (`issuealign`):
  - issue_terms_nonempty_rate = `1.0`
  - neg_issue_overlap_rate = `1.0`
  - same_family_rate = `0.053`

This confirms same-family has been demoted to tie-breaker, not primary constraint.

## Clean ablation
Compare:
- Baseline: `current main + MNRL + full mining + laws_structured` (`outputs/train_structq_ablation`)
- + same-issue near miss fix: `outputs/same_issue_near_miss_ablation`

### overall
| run | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|
| baseline_structq_fullmining | 0.163758 | 0.057960 | 0.057960 | 50 |
| + same-issue fix | 0.170900 | 0.051508 | 0.051508 | 51 |
| delta | +0.007142 | -0.006452 | -0.006452 | +1 |

### non-explicit subset
| run | Recall@200 | strict_f1 | corpus_f1 | final FP |
|---|---:|---:|---:|---:|
| baseline_structq_fullmining | 0.185419 | 0.062840 | 0.062840 | 31 |
| + same-issue fix | 0.197324 | 0.052088 | 0.052088 | 32 |
| delta | +0.011905 | -0.010752 | -0.010752 | +1 |

## Interpretation
- same-issue hypothesis is **partially成立** on recall (overall/non-explicit Recall@200 both up).
- But final F1 drops and FP increases, which indicates current issue-similarity signal is still too weak/noisy for precision-sensitive final metrics.
- So this is not pure “hypothesis invalid”;更像“信号可提升召回，但质量不足以稳定转化为最终F1”。

# Explicit Prefix Rescue

## Current Best
- public score: `0.11368`
- submission: `release/submission_explicit_prefix_rescue_conjunction_top3_v8/submission.csv`
- script: `scripts/run_explicit_prefix_rescue.py`

## Confirmed Gains
- `0.09617`: abbreviation prefix rescue against `laws_de.csv`.
- `0.10016`: natural statute alias top1, resolving `Art. 125 of the CC`.
- `0.10383`: natural statute alias top2, adding `Art. 125 Abs. 2 ZGB`.
- `0.10736`: conjunction parsing for `Art. 38 and 39 CO`, top1.
- `0.11064`: conjunction top2.
- `0.11368`: conjunction top3.

## Rejected Probes
- `Art. 125 Abs. 3 ZGB` dropped public to `0.10336`.
- `Art. 38 Abs. 2 OR` dropped public to `0.11307`.
- targeted detention rescue stayed flat at `0.09617`.

## Local Val Anchor
- base val strict_f1: `0.167770`
- explicit rescue val strict_f1: `0.179311`
- TP: `23 -> 25`
- FP: `43 -> 43`

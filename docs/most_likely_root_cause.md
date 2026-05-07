# Most Likely Root Cause

## 1) Single most likely root cause
**Incomplete laws-side signal transmission into fused@200** (family/issue/rule helpful cues are not fully converted into fused candidate coverage for non-explicit long-tail gold).

## 2) Evidence chain
- bottleneck reports repeatedly show not_in_fused as dominant loss.
- P2-A/P2-B/P3 and training-line alignments improve partially but do not remove not_in_fused dominance.
- same-family/same-issue mining refinements do not stably convert to final F1.
- cloud run confirmed dense backend=sbert and still showed same bottleneck shape.

## 3) Why stronger than "reranker weak" or "RRF off"
- if reranker were the main issue, most gold would already be in fused and fail later (reranked_too_low).
- current main failure is earlier (not_in_fused), and fusion (RRF sparse+dense) is already active.
- so upstream fused coverage transmission is more causal than downstream ranking-only weakness.

## 4) One highest-ROI next direction
Run one focused non-explicit laws-side fused-coverage transmission repair audit (no new retrieval branch, no court changes), targeting gold-in-fused@200 increase only.
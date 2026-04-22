# Laws P4 Rerank Input Shaping Ablation

## Setup
- A: P0 + P2-A + P2-B + P3.
- B: P0 + P2-A + P2-B + P3 + P4 (laws-first rerank input shaping).
- P4 only changes rerank input admission priority; retrieval, reranker model, final cut, and court branch remain unchanged.

## Overall
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | gold_in_rerank_input_rate |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 10 | 0.095618 | 0.061733 | 0.061733 | 49 | 0.095618 |
| P3 + P4 | 10 | 0.095618 | 0.061733 | 0.061733 | 49 | 0.095618 |

## Explicit Citation Subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | gold_in_rerank_input_rate |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 4 | 0.095652 | 0.050638 | 0.050638 | 19 | 0.095652 |
| P3 + P4 | 4 | 0.095652 | 0.050638 | 0.050638 | 19 | 0.095652 |

## Non-Explicit Citation Subset
| run | n | Recall@200 | strict_f1 | corpus_f1 | final_fp | gold_in_rerank_input_rate |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 6 | 0.095588 | 0.069130 | 0.069130 | 30 | 0.095588 |
| P3 + P4 | 6 | 0.095588 | 0.069130 | 0.069130 | 30 | 0.095588 |

## Gold Drop Stage Distribution (Overall)
| gold_drop_stage | P3 | P3+P4 | delta |
|---|---:|---:|---:|
| cut_by_dynamic_threshold | 15 | 15 | +0 |
| kept_final | 9 | 9 | +0 |
| not_in_rerank_input | 227 | 227 | +0 |

## Key Deltas
- overall strict/corpus_f1: 0.061733 -> 0.061733
- non-explicit strict/corpus_f1: 0.069130 -> 0.069130
- final FP: 49 -> 49
- gold_in_rerank_input_rate: 0.095618 -> 0.095618
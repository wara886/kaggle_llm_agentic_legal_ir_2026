# Laws MiniLM Fine-Tune Ablation

## Setup
- A: silver baseline with `--enable-rule-exact false`.
- B: P0 rules/guardrail with `--enable-rule-exact true --rule-top-k-laws 20`.
- C: B plus fine-tuned MiniLM dense model via `--court-dense-model-name artifacts/laws_minilm_p1/minilm_laws_finetuned`.
- Shared eval settings: light reranker, `--dynamic-mode fixed_top_k --fixed-top-k 5`, court dense disabled, court seed floors zero, sparse/dense court rows set to zero.
- No court logic, reranker main logic, German expansion, or co-citation changes were made for this ablation.

## P1 Artifacts
- Hard negatives: `264` triplets from `train` split, dense backend `sbert`.
- Fine-tune: `264` examples, `1` epoch, `triplet` loss, batch size `16`.
- Rebuilt laws dense index: `175933` rows, embedding shape `[175933, 384]`, text max chars `500`.

## A/B/C Comparison
### overall
| run | n | Recall@200 | strict_f1 | corpus_f1 |
|---|---:|---:|---:|---:|
| A no rules | 10 | 0.011111 | 0.000000 | 0.000000 |
| B P0 rules | 10 | 0.011111 | 0.020167 | 0.020167 |
| C P1 fine-tuned MiniLM | 10 | 0.011111 | 0.020167 | 0.020167 |

| delta | Recall@200 | strict_f1 | corpus_f1 |
|---|---:|---:|---:|
| B - A | +0.000000 | +0.020167 | +0.020167 |
| C - B | +0.000000 | +0.000000 | +0.000000 |

### explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 |
|---|---:|---:|---:|---:|
| A no rules | 4 | 0.027778 | 0.000000 | 0.000000 |
| B P0 rules | 4 | 0.027778 | 0.050417 | 0.050417 |
| C P1 fine-tuned MiniLM | 4 | 0.027778 | 0.050417 | 0.050417 |

| delta | Recall@200 | strict_f1 | corpus_f1 |
|---|---:|---:|---:|
| B - A | +0.000000 | +0.050417 | +0.050417 |
| C - B | +0.000000 | +0.000000 | +0.000000 |

### non-explicit citation subset
| run | n | Recall@200 | strict_f1 | corpus_f1 |
|---|---:|---:|---:|---:|
| A no rules | 6 | 0.000000 | 0.000000 | 0.000000 |
| B P0 rules | 6 | 0.000000 | 0.000000 | 0.000000 |
| C P1 fine-tuned MiniLM | 6 | 0.000000 | 0.000000 | 0.000000 |

| delta | Recall@200 | strict_f1 | corpus_f1 |
|---|---:|---:|---:|
| B - A | +0.000000 | +0.000000 | +0.000000 |
| C - B | +0.000000 | +0.000000 | +0.000000 |

## Decision
- Truncation is not a current main bottleneck: no detectable gold signal landed beyond 900 chars in the validation audit.
- P1 fine-tuned MiniLM did not improve the non-explicit subset in this light ablation; non-explicit Recall@200 and F1 stayed at zero.
- The effective measured gain remains P0 rules/guardrail, not P1 retriever fine-tuning.
- There is still no evidence here to reopen court work; keep court frozen until laws retriever/rule lanes show a real non-explicit lift.

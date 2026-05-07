# Qwen3 Reranker Module Ablation

## Scope
- Retrieval pipeline unchanged (frozen laws-first candidate set).
- Court lane unchanged.
- No Qwen training; inference-only reranker module A/B.
- Qwen3 is loaded as `AutoModelForCausalLM` and scored with yes/no logits.
- Qwen3 reranks the first `80` current candidates per query.
- This is a cloud inference control design (48GB class GPU).

## Metrics
| run | overall strict_f1 | overall corpus_f1 | non-explicit strict_f1 | non-explicit corpus_f1 | final FP | reranked_too_low share |
|---|---:|---:|---:|---:|---:|---:|
| current rerank + final calibration | 0.062721 | 0.062721 | 0.062840 | 0.062840 | 50 | 0.107570 |
| Qwen3-reranker module | 0.128549 | 0.128549 | 0.119948 | 0.119948 | 39 | 0.067729 |

## Delta (Qwen - Current)
- overall strict_f1: +0.065828
- overall corpus_f1: +0.065828
- non-explicit strict_f1: +0.057108
- non-explicit corpus_f1: +0.057108
- final FP: -11
- reranked_too_low share: -0.039841

## Runtime Note
- detected_gpu_mem_gb: 8.00
- cloud_gate_threshold_gb: 0.00
- cloud_gate_passed: 1
- noncloud_override_used: 0

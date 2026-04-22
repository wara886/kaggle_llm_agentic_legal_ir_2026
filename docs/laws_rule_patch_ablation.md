# Laws Rule Patch Ablation

## Setup
- A: `run_silver_baseline_v0.py` with `--enable-rule-exact false`.
- B: same command with `--enable-rule-exact true --rule-top-k-laws 20`.
- Both runs used light reranker (`--prefer-strong-reranker false`) because the strong-reranker ablation did not finish within 20 minutes locally; retrieval/rule settings are otherwise identical.
- Both runs disabled court dense and set court seed floors to zero; C was not run because the current silver mainline does not expose a separate stable laws-primary German-expansion switch beyond the existing `laws_query_pack_v2` path.

## A/B(/C) Comparison
| Segment | n | A Recall@200 | B Recall@200 | Delta | A strict_f1 | B strict_f1 | Delta | A corpus_f1 | B corpus_f1 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 10 | 0.011111 | 0.011111 | +0.000000 | 0.000000 | 0.022701 | +0.022701 | 0.000000 | 0.022701 | +0.022701 |
| explicit citation subset | 4 | 0.027778 | 0.027778 | +0.000000 | 0.000000 | 0.049809 | +0.049809 | 0.000000 | 0.049809 | +0.049809 |
| non-explicit citation subset | 6 | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.004630 | +0.004630 | 0.000000 | 0.004630 | +0.004630 |

## Explicit Citation Subset
- Explicit subset size: `4` / `10`.
- RuleCitationRetriever gold hits: `6` gold citations across `3` samples.
- lit./Ziff. normalization repairs: `1` samples.
- Rule false-positive citations emitted: `94`; final prediction false positives A=`50`, B=`144`, delta=`+94`.
- B introduced final false positives not in A: `94` citation instances.
- Remaining granularity mismatch cases: `0`.

### Normalization Repairs
| query_id | repaired_gold | query_snippet |
|---|---|---|
| val_001 | Art. 221 Abs. 1 StPO; Art. 221 Abs. 2 StPO | May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality whe |

## Failure Buckets
| bucket | count |
|---|---:|
| nonexplicit_no_rule_match | 5 |
| explicit_rule_gold_hit | 2 |
| explicit_rule_false_positive_only | 2 |
| nonexplicit_rule_spurious_match | 1 |

### Rule False Positive Examples
| query_id | false_positive_rule_citations | query_snippet |
|---|---|---|
| val_001 | Art. 221 Abs. 1 AHVV; Art. 221 Abs. 2 AHVV; Art. 221 Abs. 3 AHVV; Art. 221 Abs. 1 DBG; Art. 221 Abs. 2 DBG | May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent w |
| val_002 | Art. 1 112; Art. 2 112; Art. 3 Abs. 1 112; Art. 3 Abs. 2 112; Art. 4 Abs. 1 112 | A claimant holding a national vocational diploma in warehouse operations worked intermittently as a storage technician from 10 March to 20 S |
| val_003 | Art. 221 Abs. 1 AHVV; Art. 221 Abs. 2 AHVV; Art. 221 Abs. 3 AHVV; Art. 221 Abs. 1 DBG; Art. 221 Abs. 2 DBG | A. Rivera, a Peruvian national born in 1994 and with no prior convictions in the forum state, is accused of having, between 5 March and 9 Ma |
| val_006 | Art. 364 Abs. 2 OR; Art. 364 Abs. 3 OR; Art. 364 Abs. 1 StPO; Art. 364 Abs. 2 StPO; Art. 364 Abs. 3 StPO | On 3 March 2012, homeowners Ms. L and her partner Mr. M asked G, an installer they knew socially who works on domestic heating equipment, to |
| val_007 | Art. 1 112; Art. 2 112; Art. 3 Abs. 1 112; Art. 3 Abs. 2 112; Art. 4 Abs. 1 112 | An heirship claims title to a vintage pocket chronometer known as “The Meridian” that belonged to Ms. Barnes, who died in 2010, and contends |

## P0 Decision
- Keep the patch as silver-core P0: B has positive F1 signal over A.
- Next patch candidate: laws primary lane German keyword / abbreviation expansion normalization, without returning to court seed work.

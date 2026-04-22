# Non-Explicit Law Family Audit

## Scope
- Validation non-explicit subset: `6` / `10` queries.
- Explicit subset definition follows current `RuleCitationRetriever.extract_patterns`: queries without an `Art.`/BGE/case citation pattern are non-explicit, even if they contain a bare family reference such as `StPO`.
- Court work, German expansion, MiniLM fine-tune, reranker logic, and co-citation remain frozen.

## Aggregate Gold Family Shape
| view | dominant families |
|---|---|
| primary statutory family per query | ZGB=`3`, StPO=`1`, StGB=`1`, OR/ZPO=`1` |
| statutory citation count, broad | ZGB=`29`, StPO=`21`, StGB=`10`, OR=`7`, BV=`3`, AIG=`0`, other statute families=`15` |
| practical reading | Most non-explicit failures still have a stable law-family target; the missing link is getting that family into laws retrieval before reranking. |

## Per-Query Audit
| query_id | main gold family | query explicit family / law name? | stable family from facts? | notes |
|---|---|---|---|---|
| val_003 | StPO, secondary BV | Yes: bare `StPO`, but no `Art.` pattern | Yes: pretrial detention, collusion, sufficient suspicion, right to be heard | Current P0 misses because `221 Abs. 1 StPO` is not an exact `Art.` citation pattern. |
| val_004 | ZGB | Yes: `Civil Code` | Yes: handwritten will, testamentary capacity, heirs, testamentary dispositions | Family is easy; issue is ranking specific inheritance articles over generic ZGB articles. |
| val_005 | ZGB | No | Yes: custody, visitation, overnight contact, child best interests | Stable family is ZGB, but gold articles require finer family-law issue terms. |
| val_008 | StGB, secondary StPO/BV | No | Yes: disloyal management of public interests, indictment, conviction, remand directive | Stable criminal-law family; specific `Art. 314 StGB` still needs issue-level targeting. |
| val_009 | ZGB, secondary enforcement/security | No | Yes: child maintenance and security for future maintenance | Family extractor may also see SchKG from enforcement/freeze facts; ZGB remains core. |
| val_010 | OR, secondary ZPO | No | Yes: bank mandate/liability, forged transfer orders, duty of care, exculpatory/protest clauses, wrong currency pleading | OR/ZPO split is stable, but exact article retrieval needs issue phrase sharpening. |

## Conclusion
- The non-explicit subset is more blocked by law-family alignment than by truncation or model fine-tune.
- A minimal P2 should therefore start with `law-family extractor + family-constrained laws retrieval`.
- Issue phrase purification remains the next finer-grained layer, because family alignment alone can still surface generic family articles.

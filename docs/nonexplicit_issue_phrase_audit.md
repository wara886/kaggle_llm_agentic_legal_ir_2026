# Non-Explicit Issue Phrase Audit

## Scope
- Target subset: validation queries with no explicit citation pattern under `RuleCitationRetriever.extract_patterns`: `6` / `10`.
- Baseline trace: `outputs/laws_p2_family_constraints`.
- Frozen areas remain unchanged: court retrieval, German-expansion patch line, MiniLM fine-tune, reranker main logic, and co-citation.
- Audit question: inside a predicted and broadly correct law family, can the long fact pattern be reduced to 2-4 issue-focused phrases for the laws primary lane?

## Aggregate Reading
| finding | reading |
|---|---|
| predicted-family quality | All six non-explicit rows have at least one reasonable predicted family for the main statutory issue. |
| main failure after P2-A | Gold articles are often inside the right family but are outranked by generic family articles or fact-noise articles. |
| phrase feasibility | Stable 2-4 phrase extraction is feasible for all six rows. |
| sparse vs dense | The phrases are statute-title/body lexical anchors, so the first P2-B pass should be sparse-only. Dense can stay off unless sparse recall stalls. |

## Per-Query Audit
| query_id | predicted family | gold issue anchors | high-value issue phrases | fact noise to downweight | stable phrases? |
|---|---|---|---|---|---|
| val_003 | STPO;BV | `Art. 221 Abs. 1 StPO`, `Art. 29 Abs. 2 BV`, StPO appeal/cost articles | pretrial detention; collusion/flight risk; sufficient suspicion; right to be heard/presumption | names, dates, vehicles, weapon details, pursuit facts | Yes: 4 groups |
| val_004 | ZGB | holographic will form, testamentary capacity, defective intent/form defect | handwritten/holographic will; testamentary capacity; defective intent; testamentary disposition | Thun/lakeside facts, avalanche, names, family narrative | Yes: 2 groups |
| val_005 | ZGB, STGB is factual spillover | custody/visitation articles: `Art. 133`, `273`, `274`, `285 ZGB` | custody; personal contact/visitation; overnight restriction; child best interests; child support | criminal allegation specifics, alcohol treatment, missing psychiatric report as criminal facts | Yes: 2 groups |
| val_008 | STGB;OR | `Art. 314 StGB`, official-capacity and criminal-procedure articles | disloyal public management; public interests; official capacity; indictment specificity | CHF line items, company names, spouse/private-company facts | Yes: 2 STGB groups; STPO issue is partly missed by family predictor |
| val_009 | SCHKG;ZGB | child maintenance/security: `Art. 277`, `285`, `286`, `291`, `292 ZGB` | child maintenance; future maintenance security; persistent non-payment; ability despite incarceration | forced-sale/enforcement mechanics, pension capitalization, unemployment-benefit budget details | Yes: 2 ZGB groups; suppress SchKG issue terms as noise |
| val_010 | OR;ZPO | mandate/bank liability, exculpation, currency, civil procedure | bank mandate; forged transfer instruction/signature; gross negligence/exculpatory clause; wrong currency; ZPO evidence/remedy | Belize vehicle, adviser initials, exact debit totals, account-history dates | Yes: 4 groups |

## Extractor Shape
- Trigger only when the query is non-explicit and P2-A predicts at least one family.
- Select at most four issue groups, each backed by query cues and constrained to predicted families.
- Build a small laws-only sparse view from issue terms; keep the original query view unchanged.
- Strictly filter issue candidates to predicted family citations, so issue phrases cannot reopen court or broad source routing.

## Implementation Implication
- P2-B should primarily rescue `Recall@200`, because the current light reranker still scores final candidates with original English fact tokens.
- The patch should therefore be judged separately on retrieval recall and final F1; final F1 may remain bottlenecked by the frozen reranker.

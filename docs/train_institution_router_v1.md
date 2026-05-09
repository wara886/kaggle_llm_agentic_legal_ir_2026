# Train-Derived Institution Router Mining

This report is generated only from `train.csv`, `laws_de.csv`, and a deterministic train-derived pseudo-hidden split.
It is meant to support the prize-compliant path: no visible `test.csv` row labels or query-id patch table are used.

## Summary

- `train_rows_total`: `1139`
- `train_rows_for_mining`: `1043`
- `pseudo_hidden_rows`: `96`
- `pseudo_hidden_topic_groups`: `89`
- `laws_exact_citations`: `175933`
- `candidate_rule_count`: `1668`
- `pseudo_hidden_matched_rows`: `34`
- `pseudo_hidden_nonempty_predictions`: `34`
- `pseudo_hidden_macro_f1`: `0.092175`
- `holdout_ratio`: `0.2`
- `min_phrase_support`: `3`
- `min_citation_support`: `2`
- `min_precision`: `0.5`
- `top_k`: `10`
- `require_legal_anchor`: `True`

## Top Candidate Rules

| phrase | support | precision | families | citations |
|---|---:|---:|---|---|
| wohnung mehrfamilienhaus ihrer | 4 | 1.000 | DSG; ZPO; SCHKG | Art. 12 Abs. 1 DSG; Art. 13 Abs. 1 DSG |
| zeilich registriert jugendanwaltschaft | 4 | 1.000 | STGB; JSTG | Art. 134 StGB |
| ziff vertrag | 3 | 1.000 | OR; ZGB | Art. 24 Abs. 1 OR; Art. 203 OR; Art. 20 Abs. 2 OR; Art. 8 ZGB; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR |
| wegen willensmaengeln | 3 | 1.000 | OR; ZGB | Art. 20 Abs. 2 OR; Art. 203 OR; Art. 8 ZGB; Art. 24 Abs. 1 OR; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR |
| ehedauer | 3 | 1.000 | IPRG | Art. 59 IPRG; Art. 64 Abs. 1 IPRG; Art. 65 Abs. 1 IPRG; Art. 60 IPRG; Art. 65 Abs. 2 IPRG; Art. 49 IPRG |
| strafen kommen | 3 | 1.000 | JSTG; STGB | Art. 11 Abs. 1 JStG; Art. 25 Abs. 1 JStG; Art. 79a Abs. 3 StGB; Art. 25 Abs. 2 JStG |
| zwei minderjaehrigen kindern | 3 | 1.000 | DBG; ZGB | Art. 9 Abs. 1 DBG; Art. 36 Abs. 2 DBG; Art. 9 Abs. 2 DBG |
| sehen vorteile beurteilung | 3 | 1.000 | OR | Art. 121 OR; Art. 127 OR; Art. 131 Abs. 2 OR |
| bedingte strafe | 3 | 1.000 | STGB | Art. 42 Abs. 1 StGB; Art. 42 Abs. 4 StGB |
| zustaendigen regionalgericht praettigau | 3 | 1.000 | ZGB | Art. 839 Abs. 2 ZGB |
| zivilprozess wendet wahrheitsgemaess | 3 | 1.000 | OR | Art. 41 Abs. 1 OR |
| untersuchungshaft notwendig weiteren | 3 | 1.000 | STGB; JSTG; JSTPO | Art. 22 Abs. 1 StGB |
| partnerschaft eintragen seither | 3 | 1.000 |  | Art. 21 Abs. 1 BüG |
| ihrem ehemann | 3 | 1.000 | ZGB; IPRG; OR | Art. 527 ZGB |
| drei gemeinsame kinder | 3 | 1.000 | ZGB; BVG | Art. 462 ZGB |
| mehrfamilienhauses | 5 | 0.800 | LSV; USG; RPG | Art. 43 Abs. 1 LSV; Art. 25 Abs. 1 USG; Art. 7 Abs. 1 LSV |
| wohn gewerbezone | 8 | 0.750 | USG; LSV; RPG; UVPV | Art. 43 Abs. 1 LSV; Art. 25 Abs. 1 USG; Art. 20 Abs. 1 USG; Art. 20 Abs. 2 USG; Art. 25 Abs. 3 USG; Art. 7 Abs. 1 LSV |
| nachehelichen unterhalt | 4 | 0.750 | IPRG; ZGB | Art. 59 IPRG; Art. 49 IPRG; Art. 61 IPRG; Art. 63 Abs. 2 IPRG; Art. 54 Abs. 1 IPRG; Art. 63 Abs. 1 IPRG; Art. 60 IPRG; Art. 65 Abs. 2 IPRG |
| willensmangels | 4 | 0.750 | OR; ZGB | Art. 20 Abs. 2 OR; Art. 24 Abs. 1 OR; Art. 203 OR; Art. 8 ZGB; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR; Art. 23 OR |
| eigentuemer grundstueck grundbuch | 4 | 0.750 | ZGB; GBV | Art. 973 Abs. 1 ZGB; Art. 395 Abs. 4 ZGB; Art. 974 Abs. 2 ZGB |
| gemeinsame kinder | 4 | 0.750 | ZGB; BVG | Art. 462 ZGB; Art. 241 Abs. 3 ZGB |
| barvermoegen | 4 | 0.750 | IPRG; ZGB; DBG | Art. 527 ZGB; Art. 18 IPRG |
| gewerbezone | 9 | 0.667 | USG; LSV; RPG; UVPV | Art. 43 Abs. 1 LSV; Art. 25 Abs. 1 USG |
| vertrag dezember | 3 | 0.667 | OR; ZGB | Art. 203 OR; Art. 20 Abs. 2 OR; Art. 8 ZGB; Art. 24 Abs. 1 OR; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR |
| zustehenden vertretungsbefugnissen handelsregister | 3 | 0.667 | GBV; ZGB; OR | Art. 946 Abs. 1 ZGB; Art. 9 Abs. 1 GBV; Art. 965 Abs. 2 ZGB; Art. 963 Abs. 1 ZGB; Art. 84 Abs. 1 GBV |
| zeitpunkt martha eigentuemerin | 3 | 0.667 | GBV; ZGB | Art. 90 Abs. 1 GBV; Art. 966 Abs. 1 ZGB; Art. 965 Abs. 3 ZGB; Art. 83 Abs. 2 GBV |
| weil geschehen weitem | 3 | 0.667 | DSG | Art. 2 Abs. 1 DSG; Art. 4 Abs. 1 DSG; Art. 4 Abs. 2 DSG; Art. 7 Abs. 1 DSG |
| verkaufsauftrag | 3 | 0.667 | OR; ZGB; BEG | Art. 28 Abs. 1 BEG; Art. 28 Abs. 3 BEG; Art. 29 Abs. 2 BEG; Art. 29 Abs. 1 BEG |
| pensionskasse | 3 | 0.667 | IPRG; UVG; EOG; UVV | Art. 60 IPRG; Art. 59 IPRG; Art. 65 Abs. 2 IPRG; Art. 49 IPRG |
| obligatorische motorfahrzeughaftpflichtversicherung | 3 | 0.667 | SVG; VVG | Art. 65 Abs. 1 SVG; Art. 14 Abs. 2 VVG; Art. 65 Abs. 2 SVG; Art. 65 Abs. 3 SVG |

## Interpretation

- A rule is a candidate only when a query phrase repeatedly maps to a laws-grounded gold citation cluster in train.
- Pseudo-hidden evaluation measures whether these mined phrase clusters recover citations on held-out train rows from the same grouped legal topic.
- This is not yet a final submission generator; it is the reproducible evidence layer that should replace public-leaderboard-derived rule authorship.

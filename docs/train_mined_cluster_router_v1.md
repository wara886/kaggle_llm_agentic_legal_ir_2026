# Train-Mined Cluster Router v1

This report is generated from `train.csv`, `val.csv`, `laws_de.csv`, and deterministic code.
It does not read `test.csv` and does not use public leaderboard feedback.

## Goal

Convert the broad train-derived phrase miner into a smaller robust-cluster router.
The router keeps only citation clusters with repeated phrase evidence, sufficient support, and high train precision, then evaluates them on both a train-derived pseudo-hidden split and the official validation split.

## Summary

- `train_rows_total`: `1139`
- `validation_rows`: `10`
- `laws_exact_citations`: `175933`
- `holdout_ratio`: `0.2`
- `selection`:
  - `min_cluster_phrases`: `4`
  - `min_cluster_support`: `4`
  - `min_cluster_precision`: `0.5`
  - `max_citations_per_cluster`: `10`
  - `max_clusters`: `160`
  - `max_rules_per_cluster_selected`: `10`
- `pseudo_hidden`:
  - `stage`: `pseudo_hidden`
  - `mining_rows`: `1044`
  - `eval_rows`: `95`
  - `candidate_rule_count`: `1708`
  - `candidate_cluster_count`: `131`
  - `selected_cluster_count`: `35`
  - `selected_rule_count`: `337`
  - `matched_eval_rows`: `17`
  - `nonempty_prediction_rows`: `26`
  - `explicit_citation_rows`: `16`
  - `query_expansion_rows`: `6`
  - `macro_f1`: `0.111398`
  - `avg_missing_gold_from_corpus`: `0.0`
- `validation`:
  - `stage`: `validation`
  - `mining_rows`: `1139`
  - `eval_rows`: `10`
  - `candidate_rule_count`: `3204`
  - `candidate_cluster_count`: `153`
  - `selected_cluster_count`: `39`
  - `selected_rule_count`: `377`
  - `matched_eval_rows`: `7`
  - `nonempty_prediction_rows`: `8`
  - `explicit_citation_rows`: `4`
  - `query_expansion_rows`: `10`
  - `macro_f1`: `0.071024`
  - `avg_missing_gold_from_corpus`: `0.0`
- `selected_cluster_csv`: `G:\cord\kaggle_llm_agentic_legal_ir_2026\artifacts\train_mined_cluster_router_v1\validation\selected_clusters.csv`

## Selected Clusters

| best phrase | candidate phrases | support | precision | citations |
|---|---:|---:|---:|---|
| absichtlichen taeuschung sinne | 11 | 11 | 1.000 | Art. 20 Abs. 2 OR; Art. 23 OR; Art. 203 OR; Art. 8 ZGB; Art. 24 Abs. 1 OR; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR; Art. 906 Abs. 2 ZGB |
| eintragung bauhandwerkerpfand hotelliegenschaft | 71 | 9 | 1.000 | Art. 839 Abs. 2 ZGB |
| freundschaftsdienst unentgeltlich friedensrichter | 130 | 7 | 1.000 | Art. 12 Abs. 1 DSG; Art. 13 Abs. 1 DSG |
| schadenersatz gewaehrleistungsansprueche | 23 | 7 | 1.000 | Art. 24 Abs. 1 OR; Art. 203 OR; Art. 20 Abs. 2 OR; Art. 8 ZGB; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR; Art. 23 OR |
| registriert jugendanwaltschaft gemeldet | 15 | 7 | 1.000 | Art. 134 StGB |
| vermoegenseinbussen vermoegenszugaenge vermoegensverschiebungen | 881 | 6 | 1.000 | Art. 906 Abs. 2 ZGB; Art. 23 OR; Art. 320 Abs. 3 OR; Art. 25 Abs. 1 OR; Art. 20 Abs. 2 OR |
| viehkauf schadenersatz gewaehrleistungsansprueche | 486 | 6 | 1.000 | Art. 203 OR; Art. 20 Abs. 2 OR; Art. 8 ZGB; Art. 24 Abs. 1 OR; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR |
| verwaltungsstrafrecht gelten strafbare | 57 | 6 | 1.000 | Art. 50 Abs. 1 ChemG; Art. 50 Abs. 2 ChemG; Art. 51 ChemG; Art. 52 Abs. 1 ChemG |
| untersuchungshaft notwendig weiteren | 38 | 6 | 1.000 | Art. 22 Abs. 1 StGB |
| pensioniert urspruenglich immobilienverwalter | 42 | 5 | 1.000 | Art. 9 Abs. 1 DBG; Art. 9 Abs. 2 DBG; Art. 36 Abs. 2 DBG |
| notwendigen betrag schadenersatzanspruch | 30 | 5 | 1.000 | Art. 41 Abs. 1 OR |
| weiterziehen beantworten gestellte | 24 | 5 | 1.000 | Art. 5 Abs. 1 VwVG |
| wegen absichtlicher taeuschung | 15 | 5 | 1.000 | Art. 23 OR; Art. 906 Abs. 2 ZGB; Art. 320 Abs. 3 OR; Art. 25 Abs. 1 OR; Art. 20 Abs. 2 OR |
| ihrem ehemann | 6 | 4 | 1.000 | Art. 527 ZGB |
| wegen willensmangels | 7 | 11 | 0.857 | Art. 20 Abs. 2 OR; Art. 23 OR; Art. 24 Abs. 1 OR; Art. 203 OR; Art. 8 ZGB; Art. 187 Abs. 1 OR; Art. 20 Abs. 1 OR; Art. 906 Abs. 2 ZGB |
| gruendungsbilanz januar umlaufvermoegen | 24 | 6 | 0.750 | Art. 19 Abs. 1 DBG |
| versicherungsrechtliche fragestellungen behandeln | 10 | 4 | 0.750 | Art. 462 ZGB; Art. 216 Abs. 2 ZGB; Art. 471 ZGB; Art. 215 Abs. 1 ZGB; Art. 216 Abs. 1 ZGB; Art. 241 Abs. 3 ZGB |
| 0hrmuschelersatz nasenersatzstuecke kieferersatzstuecke | 178 | 4 | 0.667 | Art. 13 Abs. 1 UVV |
| gegebenenfalls sicherheitsmassnahmen projektaenderung | 103 | 4 | 0.667 | Art. 16 Abs. 2 EleG |
| anschliessend erbvertrag unterzeichnen | 61 | 4 | 0.667 | Art. 184 ZGB |
| parkplatzbenuetzungsrechts auffassung willensmangel | 59 | 4 | 0.667 | Art. 966 Abs. 1 ZGB; Art. 965 Abs. 3 ZGB; Art. 83 Abs. 2 GBV; Art. 90 Abs. 1 GBV |
| zustehenden vertretungsbefugnissen handelsregister | 43 | 4 | 0.667 | Art. 965 Abs. 2 ZGB; Art. 963 Abs. 1 ZGB; Art. 84 Abs. 1 GBV; Art. 9 Abs. 1 GBV; Art. 946 Abs. 1 ZGB |
| staatsanwaltschaft zugefuehrt staatsanwaltschaft | 41 | 4 | 0.667 | Art. 343 Abs. 3 StPO |
| prozessualen durchsetzung rechtsbehelfs | 26 | 4 | 0.667 | Art. 674 Abs. 3 ZGB |
| arbeitslohn verkehrswert liegenschaft | 24 | 4 | 0.667 | Art. 205 Abs. 3 ZGB |
| ihrem vermoegen geschaedigt | 18 | 4 | 0.667 | Art. 101 Abs. 1 OR; Art. 399 Abs. 2 OR; Art. 1 Abs. 2 OR; Art. 39 Abs. 3 OR; Art. 8 ZGB; Art. 99 Abs. 2 OR |
| zustaendige zuercher schlichtungsbehoerde | 17 | 4 | 0.667 | Art. 212 Abs. 1 ZPO |
| gemaess stgb angeordnet | 16 | 4 | 0.667 | Art. 59 Abs. 3 StGB; Art. 76 Abs. 2 StGB; Art. 82 Abs. 1 StPO; Art. 64 Abs. 1 StGB; Art. 59 Abs. 1 StGB; Art. 62c Abs. 6 StGB; Art. 59 Abs. 2 StGB; Art. 64 Abs. 4 StGB |
| gegebenenfalls hilfsgutachterlich prozessualen | 12 | 4 | 0.667 | Art. 247 Abs. 1 ZPO |
| kommenden delikten hauptverhandlung | 26 | 8 | 0.600 | Art. 286 Abs. 1 StGB |

## Interpretation

- This is still an evidence layer, not a leaderboard submission.
- The pseudo-hidden stage mines on train-only rows and evaluates on held-out train topic groups.
- The validation stage mines on all train rows and evaluates on `val.csv`.
- A low validation score is acceptable at this stage if it exposes which train-mined clusters transfer and which do not.
- Future work should improve cluster selection and issue routing before generating any new `submission.csv`.

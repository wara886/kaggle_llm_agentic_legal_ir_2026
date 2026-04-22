# 查询扩展设计（query expansion）

## 1. 目标边界
- 仅做轻量跨语种扩展（light cross_lingual query expansion）。
- 仅使用词典/模板（dictionary_template_based）。
- 不调用外部付费 API，不做 agent rewrite，不做大模型改写。
- 不替换英文原查询，只追加德文扩展视图。

## 2. 输入输出
- 输入：`src/query_preprocess.py` 产出的 `multi_view_query`。
- 输出：
  - `expanded_query_de`
  - `expanded_keywords_de`
  - `bilingual_query_pack`
  - `expanded_query_de_laws`
  - `expanded_query_de_court`
  - `laws_query_pack`
  - `court_query_pack`

## 3. 已覆盖词表（EN -> DE）
- contract -> vertrag, vertragsrecht
- liability -> haftung
- damages -> schadenersatz, schaden
- employment -> arbeitsrecht, arbeitsverhältnis
- tax -> steuer, steuerrecht
- criminal -> strafrecht, strafbar
- procedure -> verfahren, prozessrecht
- appeal -> berufung, beschwerde
- inheritance -> erbrecht, nachlass
- property -> sachenrecht, eigentum
- company -> gesellschaftsrecht, unternehmen
- insurance -> versicherung, versicherungsrecht
- evidence -> beweis, beweiswürdigung
- jurisdiction -> zuständigkeit, gerichtsbarkeit
- article -> artikel, art.
- section -> abschnitt, sektion
- paragraph -> absatz, para, §
- court -> gericht, bundesgericht

## 4. 统一接口
- 文件：`src/query_expansion.py`
- 接口函数：
  - `expand_query_from_multi_view(multi_view_query)`
  - `build_bilingual_retrieval_views(multi_view_query)`
  - `build_source_aware_query_packs(multi_view_query)`
- `build_bilingual_retrieval_views` 返回：
  - `sparse_query_en`
  - `dense_query_en`
  - `sparse_query_bilingual`
  - `dense_query_bilingual`
  - 以及 `expanded_query_de` 和 `bilingual_query_pack`

## 4.1 源感知扩展词表
- 法规侧（laws）优先：
  - article, paragraph, section, act, code, ordinance, tax, criminal, civil, employment, insurance, company, inheritance, property, jurisdiction
- 裁判侧（court）优先：
  - appeal, evidence, procedure, burden of proof, damages, negligence, liability, termination, invalidity, abuse, compensation, contract dispute

## 5. 风险点
- 词典覆盖有限：长尾法律术语未覆盖时增益有限。
- 词义歧义：同一英文词在不同法律语境下可能对应不同德文词。
- 噪音注入：扩展词过多可能稀释原查询意图，影响稀疏检索排序。
- 语法缺失：模板式短语不具备完整句法，偏向“检索提示词”而非自然句。

## 6. 后续可扩展方向
- 按 query 类型动态选择扩展词（程序法/实体法分流）。
- 增加同义词层与法规缩写映射层。
- 与候选诊断（candidate diagnostics）联动，按失败类型自动补词。

# 查询桥接设计（query_bridge_design）

## 1. 设计目标
- 为 baseline_v1_1 提供“查询预处理（query_preprocessing）+ 跨语种扩展（cross_lingual_expansion）”前置模块。
- 在不改动 phase0 和 baseline_v1 主干逻辑的前提下，提供可插拔查询视图（query views）。
- 保留英文原查询（query_original），追加而不是替换扩展视图。

## 2. 模块边界
- `src/query_preprocess.py`
  - 输入：英文法律查询（english_legal_query）
  - 输出：多视图查询（multi_view_query）
  - 最少字段：
    - `query_original`
    - `query_clean`
    - `query_keywords`
    - `query_legal_phrases`
    - `query_number_patterns`
- `src/query_expansion.py`
  - 输入：`multi_view_query`
  - 输出：
    - `expanded_query_de`
    - `expanded_keywords_de`
    - `bilingual_query_pack`

## 3. 规则边界（仅轻量）
- 仅词典/模板（dictionary_template_based）
- 不调用外部付费 API
- 不做 agent rewrite
- 不做大模型改写
- 不做复杂语义重写，仅做检索友好的规则增强

## 4. 已覆盖词表（EN -> DE）
- contract -> vertrag, vertragsrecht
- liability -> haftung
- damages -> schadenersatz, schaden
- employment -> arbeitsrecht, arbeitsverhaeltnis
- tax -> steuer, steuerrecht
- criminal -> strafrecht, strafbar
- procedure -> verfahren, prozessrecht
- appeal -> berufung, beschwerde
- inheritance -> erbrecht, nachlass
- property -> sachenrecht, eigentum
- company -> gesellschaftsrecht, unternehmen
- insurance -> versicherung, versicherungsrecht
- evidence -> beweis, beweiswuerdigung
- jurisdiction -> zustaendigkeit, gerichtsbarkeit
- article -> artikel, art.
- section -> abschnitt, sektion
- paragraph -> absatz, para, paragraph
- court -> gericht, bundesgericht

## 5. 风险点
- 词典覆盖仍有限：长尾法律术语与领域表达可能缺失。
- 词义歧义：英文同词在不同法律语境下德文映射并不唯一。
- 噪声注入：扩展词过多可能稀释原始查询意图，导致召回漂移。
- 规则抽取对非标准写法敏感，可能漏召编号模式。

## 6. 与检索模块接入方式
- 给 `retrieval_sparse.py`：
  - 优先接 `sparse_query_bilingual`（英文关键词+编号+德文扩展词）
- 给 `retrieval_dense.py`：
  - 优先接 `dense_query_bilingual`（英文清洗句+法律短语+德文扩展句）
- 接入建议：
  - 默认开关关闭（避免破坏历史回归）
  - 在 `run_baseline_v1.py` 以参数启用
  - 与 candidate diagnostics 联动评估 Recall@K 增量

## 7. 自检说明
- `query_preprocess.py` 自带 `self_check()`
- `query_expansion.py` 自带 `self_check()`
- 二者均可通过命令行直接运行并输出 JSON


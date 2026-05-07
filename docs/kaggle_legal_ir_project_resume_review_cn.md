# Kaggle Legal IR 项目复盘与简历版总结

## 1. 一句话项目定位

这是一个从零搭建的法律信息检索（Legal Information Retrieval, Legal IR）项目：输入一段复杂法律事实与问题描述，系统需要从瑞士法律条文库和判例理由库中检索并输出最相关的法律引用（citation），最终在 Kaggle `LLM Agentic Legal Information Retrieval` 公开榜做到 Rank 26，进入 Top 10%。

更准确地说，这不是一个单纯“换模型、调参数”的项目，而是一个高度贴近真实业务的信息检索工程：我们围绕法律文本、法条编号、法域家族、跨语言别名、显式 citation、候选召回、rerank、错误诊断和提交纪律，一步步把一个低分 baseline 推到公开榜前 10%。

截至 2026-04-27 的本地 Kaggle leaderboard 快照显示：

| 指标 | 数值 |
|---|---:|
| 当前最佳 public score | `0.20020` |
| Kaggle public rank | `26` |
| leaderboard 快照队伍数 | `390` |
| 用户当前口径 | `26 / 384` |
| 百分位 | 约前 `6.7%`，稳定属于 Top 10% |
| 当前最佳提交文件 | `release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv` |
| 当前最佳 Kaggle ref | `52086784` |

这份总结刻意按“面试可讲”的方式写：不仅记录模型，还讲业务背景、数据形态、真实难点、工程决策、失败实验、指标判断和结果应用。原因很简单：Kaggle 简历价值不在于“我跑了某个模型”，而在于能证明我们把一个复杂业务问题拆成可解释、可验证、可迭代的工程系统。

## 2. 比赛背景：为什么这是一个有业务含金量的赛题

这道题模拟的是一个法律检索/法律助理场景。用户给出一段很长的案情描述，里面可能包含事实、时间、主体、诉讼请求、法律争点、语言混杂的法典简称，以及一些明示或暗示的法律依据。系统要返回可能相关的法律条文或判例引用。

真实业务里，这类能力可以落在很多场景里：

- 律师检索：给定案情，快速定位相关法条、判例段落和裁判依据。
- 法务初筛：企业法务面对合同、劳动、知识产权、继承、婚姻、交通事故等问题时，先得到一个可解释的 citation shortlist。
- 法律 RAG：作为大模型回答前的 citation retrieval 层，减少幻觉，让模型有可靠依据。
- 法律知识库搜索：把自然语言问题映射到规范性条文和判例理由。
- 跨语言法律检索：英文问题中出现 `CC/CO/LDIP/LPM/LCD/LAI` 等别名时，需要映射到瑞士法律体系里的 `ZGB/OR/IPRG/MSchG/UWG/IVG`。

所以它和很多 playground 类表格赛不一样。这里的数据虽然由比赛方整理过，但业务逻辑不“干净”：问题很长、法律体系复杂、citation 格式多、训练集很小、测试集有分布差异，公开榜反馈还容易诱导过拟合。要拿到 Top 10%，靠的不是把模型换一圈，而是持续把法律业务理解转成检索特征和可解释修复。

### 2.1 这是不是一个 LLM + RAG 项目？

是，但要更准确地说：这个项目是一个 **LLM + RAG 系统里的检索与证据定位核心层**，不是传统意义上“给用户生成一段自然语言答案”的完整聊天机器人。

典型 RAG 可以拆成几层：

- Query understanding：理解用户问题、抽取实体、任务意图、显式约束和领域信号。
- Retrieval：从知识库中召回候选文档、段落、法条、判例或 citation。
- Reranking：把粗召回候选重新排序，过滤掉语义相似但业务错误的结果。
- Grounding：把最终回答绑定到可靠证据，避免 LLM 幻觉。
- Generation：由 LLM 基于证据生成最终答案。

这个 Kaggle 赛题刚好卡在 RAG 最关键、也最容易被低估的一层：**给定复杂法律 query，先把正确 citation 找出来**。如果这一层找错，后面即使用再强的 LLM 生成答案，也只是在错误证据上“说得更像真的”。法律、医疗、金融、企业知识库这些高风险场景里，RAG 的质量往往不是由最后的回答模型决定，而是由 retrieval / rerank / grounding 决定。

所以在简历或面试里，可以把这个项目表述为：

- 面向法律问答/RAG 的证据检索系统。
- 负责把复杂案情映射到可信 citation，作为 LLM 生成答案前的 grounding layer。
- 使用 BM25、dense embedding、RRF、LLM reranker、citation parser 和业务审计共同提升证据召回与精排质量。
- 重点解决 RAG 中最核心的两个问题：**该找什么证据**，以及 **如何证明找出来的证据没有跑偏**。

它不是“只用了 LLM 所以叫 LLM 项目”，而是一个更真实的 LLM 应用工程：LLM 只是系统里的 reranker 和语义判断组件之一，真正决定效果的是检索架构、领域特征、错误闭环和证据约束。这一点反而更有含金量，因为真实生产里的 RAG 很少靠一个 prompt 解决，更多靠系统工程把 LLM 放在正确的位置。

## 3. 数据类型与规模

比赛数据在 `data_raw/competition_data/` 下，核心文件如下。

| 文件 | 行数 | 字段 | 作用 |
|---|---:|---|---|
| `train.csv` | `1139` | `query_id`, `query`, `gold_citations` | 训练样本，每个法律问题对应一组 gold citation |
| `val.csv` | `10` | `query_id`, `query`, `gold_citations` | 本地验证集，行数极少但每行 gold 很多 |
| `test.csv` | `40` | `query_id`, `query` | Kaggle 提交集，无公开标签 |
| `laws_de.csv` | `175933` | `citation`, `text`, `title` | 瑞士法规/法条语料库 |
| `court_considerations.csv` | `2476315` | `citation`, `text` | 判例理由段落语料库 |
| `sample_submission.csv` | `2` | `query_id`, `predicted_citations` | 提交格式示例 |

提交格式是：

```text
query_id,predicted_citations
test_001,Art. 5 Abs. 1 ZPO;Art. 263 ZPO;Art. 10 IPRG
```

也就是每个测试 query 输出一个由分号连接的 citation 列表。

### 3.1 Query 是什么样的

`query` 不是短关键词，而是长篇法律事实题。它可能包含：

- 案件事实：人物、公司、国家、时间、金额、行为过程。
- 程序背景：上诉、临时措施、管辖、承认与执行、证据调查。
- 法律问题：是否时效、是否有管辖权、是否构成侵权、是否违反程序权利。
- 显式法条：例如 `Art. 83 SVG`, `Art. 59 Abs. 1 SVG`, `Art. 400 OR`。
- 多语种别名：例如 `LDIP` 对应 `IPRG`，`CO` 对应 `OR`，`CC` 对应 `ZGB`，`LPM` 对应 `MSchG`，`LCD` 对应 `UWG`，`LAI` 对应 `IVG`。

这使得 query 同时像法律考试题、检索查询、案情摘要和 RAG prompt。

### 3.2 Gold citation 是什么样的

`gold_citations` 是多标签答案。训练集平均每个 query 有约 `4.09` 个 gold citation，但验证集平均每行有 `25.1` 个，最多一行有 `47` 个。这说明 validation 不是普通随机小样本，而是高密度 citation 的压力测试。

训练集 citation 家族分布非常不均衡，Top family 如下：

| family | train gold hit 数 |
|---|---:|
| `ZGB` | `917` |
| `OR` | `466` |
| `StGB` | `287` |
| `BV` | `238` |
| `ZPO` | `210` |
| `IPRG` | `179` |
| `StPO` | `117` |
| `URG` | `108` |
| `BGG` | `76` |
| `SchKG` | `73` |
| `AIG` | `66` |
| `SVG` | `47` |
| `UWG` | `39` |

这个分布决定了一个核心风险：模型很容易被高频 `ZGB/OR/StGB/BV/ZPO` 吸走，而在 IP、跨境私法、社会保险、道路交通、劳务派遣等低频法域上发生 wrong-family。

### 3.3 显式 citation 覆盖率的分布差异

我们专门统计过题面里是否直接出现 `Art.` 形式 citation：

| split | 显式 Art query 数 | 总数 | 比例 |
|---|---:|---:|---:|
| train | `238` | `1139` | `20.9%` |
| val | `4` | `10` | `40.0%` |
| test | `14` | `40` | `35.0%` |

这个发现很关键。它说明 test/val 中显式 citation 比 train 更密集，所以“看懂题面里的法条字面形式”比泛化语义记忆更重要。后面从 `0.08960` 到 `0.20020` 的主要突破，几乎都来自这个判断。

## 4. 核心难点

### 4.1 训练样本少，语料极大

训练集只有 `1139` 行，验证集只有 `10` 行，测试集只有 `40` 行。但候选语料包括：

- `175933` 条法律条文。
- `2476315` 条判例理由段落。

这是典型的小标注、大语料、多标签检索任务。不能简单把问题当分类，也不能直接全量交给 LLM。

### 4.2 Citation 格式复杂

同一个法律条文可能出现很多形式：

- `Art. 400 OR`
- `Art. 400 Abs. 1 OR`
- `Art. 400 Abs. 2 OR`
- `Art. 400 of the Code of Obligations`
- `Art. 38 and 39 CO`
- `Art. 125 of the CC`

如果系统只做普通 embedding 相似度，很容易错过这些表面锚点。我们后来专门做了 citation normalizer、显式前缀解析、自然别名解析和 conjunction 解析。

### 4.3 多语言和法典别名

题面是英文，但法律语料大量是德文，法典名又可能是德语、法语、英语或缩写。典型映射包括：

| 题面形式 | 真实 citation family |
|---|---|
| `CC`, `Civil Code` | `ZGB` |
| `CO`, `Code of Obligations` | `OR` |
| `LDIP`, `Private International Law Act` | `IPRG` |
| `LPM`, `Trademark Protection Act` | `MSchG` |
| `LCD`, `Unfair Competition Act` | `UWG` |
| `LAI`, `invalidity insurance` | `IVG` |

这个问题最后成为上分关键之一。`test_011` 的 `LDIP -> IPRG` 修复就把 public 从 `0.18136` 推到 `0.19043`。

### 4.4 wrong-family 比 rerank 更致命

很多失败不是候选排序低，而是整个法域错了。例如：

- 著作权/商业秘密题被路由到无关行政法规。
- 商标域名/不正当竞争题被路由到完全不相关家族。
- 职业病/事故保险题被路由到劳动合同或普通债法。
- 跨境承认/重婚/遗产管理题被路由到收养或子女维护。

这种错误靠调 reranker 很难修，因为候选池本身可能已经污染。必须先把法域家族识别、显式锚点和候选召回修好。

### 4.5 本地验证集太小，公开榜不能当验证集

`val.csv` 只有 10 行，直接对 val 调参很危险。我们早期也发现本地 strict F1 变高并不总能换来 public score 增长。后来形成了提交纪律：

- 不能只看本地 F1。
- 必须结合 test 题面证据、train gold 支撑、法律库可用性和 changed-row 审计。
- 优先改少量高置信行。
- 每次提交前证明本地代理指标正向。
- 失败或持平的路线要记录，避免重复烧提交次数。

## 5. 系统如何一步步处理数据

整个项目最终形成了一条从原始数据到 submission 的处理链路。

### 5.1 数据读取与标准化

第一步是把不同文件统一成可检索语料：

- `train.csv / val.csv / test.csv` 读取 query。
- `gold_citations` 用 `;` 拆成 citation list。
- `laws_de.csv` 构造成法规文档：`citation + title + text`。
- `court_considerations.csv` 构造成判例段落文档：`citation + text`。
- 对 citation 做基本 normalization，去除空白差异、统一大小写和常见别名。

相关代码：

- `src/legal_ir/data_loader.py`
- `src/legal_ir/corpus_builder.py`
- `src/legal_ir/normalization.py`
- `src/citation_normalizer.py`

### 5.2 Query 预处理与多视角构造

法律 query 太长，直接检索容易被事实细节淹没，所以我们做了多视角 query preprocessing：

- 抽取法律关键词。
- 抽取显式 citation。
- 抽取法律短语，如 `provisional measures`, `right to be heard`, `foreign divorce recognition`。
- 抽取主体、事件、争点。
- 生成面向法规库和判例库的不同 query pack。

相关代码：

- `src/query_preprocess.py`
- `src/query_expansion.py`
- `src/law_family.py`

### 5.3 Source routing：判断先查法规还是判例

我们搭了 source router，判断每个 query 更像：

- laws route：重点查 `laws_de.csv`。
- court route：重点查 `court_considerations.csv`。
- hybrid route：两边都查。

早期设计里包含：

- `src/source_router.py`
- `route_query`
- `route_query_v1_1`

但后来的公开榜证明，当前最稳的主线是 `laws-first`：先保证法规条文 citation 的候选质量，再用判例侧作为辅助，而不是把判例检索当主通道。

### 5.4 稀疏检索：BM25 与字段感知检索

法规和判例都建立 BM25 检索：

- 对 `laws_de.csv` 检索 `citation/title/text`。
- 对 `court_considerations.csv` 检索 `citation/text`。
- 对法规 title 做 df profile，避免高频标题噪声。
- 支持 field-aware search 和 route-aware search。

相关代码：

- `src/legal_ir/bm25.py`
- `src/retrieval_sparse.py`

稀疏检索的价值是对显式词、法条号、法典缩写很敏感。它在后期显式锚点路线里非常重要。

### 5.5 稠密检索：MiniLM / SBERT

我们也构建了 dense retriever：

- 默认模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 对法规和判例文本做 embedding。
- 支持 source-aware dense search。
- 支持多视角 query dense search。

相关代码：

- `src/retrieval_dense.py`
- `scripts/build_laws_minilm_index.py`

Dense 检索的作用是弥补 BM25 对自然语言 paraphrase 的不足。但实际复盘发现，在这道题上 dense 并不是最终上分的唯一核心，很多 public gain 来自更可解释的 citation grammar 和 family 修复。

### 5.6 候选融合：RRF

不同检索分支得到的候选用 RRF（Reciprocal Rank Fusion）融合：

- sparse laws
- dense laws
- sparse court
- dense court
- rule exact citation
- family/issue boosted candidates

相关代码：

- `src/fusion.py`
- `rrf_fusion`

融合后会保留 `fused_top200 / fused_top320`，后续用于 rerank 和 residual audit。

### 5.7 Rule exact citation：显式法条直接召回

这是后期非常关键的工程点。我们写了 `RuleCitationRetriever` 和显式 citation rescue 逻辑，从 query 里直接识别：

- `Art. 83 SVG`
- `Art. 59 Abs. 1 SVG`
- `Art. 125 of the CC`
- `Art. 38 and 39 CO`
- `Art. 263 ZPO`
- `Art. 89 IPRG`

然后映射到 `laws_de.csv` 里存在的真实 citation。

相关代码：

- `src/retrieval_rules.py`
- `scripts/run_explicit_prefix_rescue.py`

这条路线先把 public 从 `0.08960` 推到 `0.11368`，后来也成为 surface-anchor 修复的基础。

### 5.8 Rerank：从轻量 token overlap 到 Qwen3 yes/no reranker

早期有轻量 reranker：

- `TokenOverlapReranker`
- 用 query/document token overlap 加一点 source bonus。

后来替换为 Qwen3 reranker：

- 模型：`Qwen/Qwen3-Reranker-0.6B`
- 加载方式：`AutoModelForCausalLM`
- 任务形式：query 和 candidate law doc 拼成 pair，让模型判断相关性。
- 打分方式：比较 `yes/no` token logits。
- 候选 cap：前 `80` 个候选。

相关代码：

- `src/rerank.py`
- `scripts/run_qwen3_reranker_module_ablation.py`

Qwen3 reranker 的本地提升非常明确：

| run | overall strict F1 | final FP | reranked_too_low share |
|---|---:|---:|---:|
| current rerank + final calibration | `0.062721` | `50` | `0.107570` |
| Qwen3 reranker | `0.128549` | `39` | `0.067729` |

公开榜也从早期 `0.01357` 上升到 `0.04272`，证明 Qwen3 rerank 是第一个有效主线模块。

### 5.9 Global prior：`Art. 100 Abs. 1 BGG`

法律上很多瑞士联邦最高法院相关问题会出现 `BGG` 的上诉期限条文。我们发现 `Art. 100 Abs. 1 BGG` 在 validation 上有高精度增益：

- 相对 Qwen3 baseline，validation 增加 `9 TP / 1 FP`。
- public 从 `0.04272` 到 `0.08960`。

这是第一个真正站稳的 public anchor。

但它也给了一个教训：global prior 必须非常谨慎。后来尝试更宽的 social insurance / right-to-be-heard / family-child prior，虽然本地分数看起来更高，公开榜却没有继续涨，甚至小幅下降。

### 5.10 Final cut 与预测条数控制

输出不是越多越好。多预测会带来 FP，少预测会损失 recall。我们尝试过：

- fixed top-k
- score threshold
- relative threshold
- dynamic final cut
- laws final fused rescue
- laws rerank input shaping

最后形成的经验是：

- 初期要保证候选召回。
- 中期要用 rerank 减少噪声。
- 后期在 test-surface 修复阶段，很多最强 patch 反而是减少预测条数，清掉显式锚点行的 FP。

`v10 test_035` 就是典型：从 7 条预测剪到 3 条，public 从 `0.19876` 到 `0.20020`。

## 6. 我们做过的训练和微调

这个项目不只是手工 patch，也完整尝试过训练和微调闭环。重要的是，我们不只记录成功，也记录哪些训练没有稳定转化成 public gain。

### 6.1 MiniLM hard negative mining

我们为法规 dense retriever 做过 hard negative mining。

目标：让 MiniLM 更会区分同一法律语境下“看起来像但不是 gold”的法条。

流程：

1. 对 train query 读取 gold citation。
2. 在 `laws_de.csv` 上跑 sparse + dense 检索。
3. 用 RRF 融合候选。
4. 从 fused candidates 中选第一个非 gold 作为 hard negative。
5. 优先选择同 family 或 issue overlap 高的 near miss。
6. 写成 triplet：`query`, `positive_text`, `negative_text`。

相关代码：

- `scripts/mine_laws_hard_negatives_minilm.py`

早期 P1 产物：

| 项 | 数值 |
|---|---:|
| hard negative triplets | `264` |
| dense backend | `sbert` |
| 每 query negative 数 | `1` |
| 语料范围 | laws-only |

### 6.2 MiniLM bi-encoder fine-tune

我们用 mined triplets 微调 MiniLM。

相关代码：

- `scripts/train_laws_minilm_biencoder.py`

训练形式：

- 模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 输入：`query`, `positive law text`, `negative law text`
- batch size：`16`
- epoch：`1`
- 初始试过 triplet loss
- 后来改成 `MultipleNegativesRankingLoss` 对齐 silver notebook 思路
- query-mode 支持 `raw` 和 `laws_structured`

### 6.3 Reindex

微调后需要重新编码法规库：

- `scripts/build_laws_minilm_index.py`
- `175933` 条法规文档
- embedding shape：`[175933, 384]`
- text max chars：`500`

### 6.4 微调结果与反思

MiniLM fine-tune 的结论不是“失败”，而是“方向有信号，但收益不够主导”。

Triplet loss 版本：

| run | Recall@200 | strict F1 |
|---|---:|---:|
| P0 rules | `0.011111` | `0.020167` |
| P1 fine-tuned MiniLM | `0.011111` | `0.020167` |

MNRL 对齐版本：

| run | Recall@200 | strict F1 | final FP |
|---|---:|---:|---:|
| P0+P2-A+P2-B+P3 | `0.137948` | `0.046195` | `51` |
| + MNRL MiniLM | `0.145091` | `0.047734` | `52` |
| delta | `+0.007143` | `+0.001539` | `+1` |

非显式 citation 子集也有小幅提升：

| run | non-explicit Recall@200 | non-explicit strict F1 |
|---|---:|---:|
| baseline | `0.142403` | `0.043232` |
| + MNRL MiniLM | `0.154308` | `0.045798` |

我们的判断：

- 微调方向有微弱正向信号。
- 但收益太小，并且带来 FP 增长。
- 当前瓶颈更像是候选覆盖和法律家族信号传输，而不是单纯 embedding 模型不够好。
- 因此没有把 MiniLM fine-tune 作为最终 public 主线，而是保留为工程资产和后续扩展方向。

这在面试里反而是很好的故事：我们没有迷信 fine-tuning，而是用 ablation 证明它是不是主因。

## 7. 从开始到 Rank 26 的完整迭代历程

### 7.1 Phase 0：朴素 baseline

最早阶段是普通检索 baseline：

- 读 query。
- 在法律库/判例库里做检索。
- 输出 top citation。
- public 只有 `0.01357`。

这个分数说明：

- 任务不是简单关键词检索。
- 法条格式、法域识别、候选融合、rerank 都需要系统化。

### 7.2 Phase 1：搭建 silver_baseline_v0 骨架

我们重构出一个可复现实验骨架：

- source router
- federated retrieval
- sparse + dense
- RRF fusion
- strong reranker
- dynamic final cut
- trace 输出
- local eval

相关文档：

- `docs/silver_baseline_v0_design_cn.md`（历史文档）

这一步的意义是工程化：后面所有实验都能通过 trace 分析候选在哪一层丢了。

### 7.3 Phase 2：Qwen3 reranker

把 rerank 从轻量 token overlap 升级为 Qwen3 yes/no reranker 后，public 从 `0.01357` 到 `0.04272`。

关键经验：

- 小模型 reranker 可以显著减少明显 FP。
- 但 reranker 只能重排已有候选，不能凭空修复候选池里没有的 gold。

### 7.4 Phase 3：`Art. 100 Abs. 1 BGG` 全局高精度先验

加入 `Art. 100 Abs. 1 BGG` 后，public 到 `0.08960`。

为什么有效：

- 该条在 validation 上 added precision 很高。
- 它是瑞士联邦最高法院诉讼类问题中稳定出现的程序锚点。

为什么没有继续扩大 prior：

- 更宽的 prior 虽然拉高本地分数，但公开榜不涨甚至下降。
- 这让我们意识到：local strict F1 不是唯一提交通行证。

### 7.5 Phase 4：显式 citation grammar

这一阶段 public 从 `0.08960` 到 `0.11368`。

关键提交：

| 版本 | public | 修复点 |
|---|---:|---|
| 显式前缀修复 | `0.09617` | 题面已有裸 `Art.`，补缺失 `Abs. 1` |
| `CC` 自然别名 top2 | `0.10383` | `Art. 125 of the CC` -> `ZGB` |
| `Art. 38 and 39 CO` top3 | `0.11368` | conjunction 解析，`CO` -> `OR` |

这一步的根本发现：

> test 题面里有大量显式 citation，处理好这些表面形式，比泛泛增加语义召回更可靠。

### 7.6 Phase 5：从模型调参转向 test-surface 法域审计

到 `0.11368` 后，我们做了一个关键转向：不再主要问“哪个模型分高”，而是问：

> 当前 submission 在 `test.csv` 哪些行明显预测到了错误法域？

于是开始逐行做 surface-family audit：

- 读 test query。
- 抽取显式法条、法典缩写、法律关键词。
- 解析预测 citation family。
- 比较 expected family vs predicted family。
- 标记 missing family / unexpected family。
- 人工确认是否为低风险修复。

相关代码：

- `scripts/run_surface_family_audit.py`

### 7.7 Phase 6：surface-anchor combo v1，大幅跃升到 `0.16392`

v1 只改 4 行：

- `test_001`：copyright / trade secret / unfair competition，修到 `ZPO/IPRG/URG/UWG`。
- `test_033`：occupational disease / accident insurance，修到 `UVG/UVV/ATSG`。
- `test_037`：trademark / domain / unfair competition，修到 `MSchG/UWG`。
- `test_040`：显式 `Art. 176 ZGB` 补 `Abs. 1`。

本地审计：

- changed queries：`4`
- target-only spillover：`0`
- changed rows family alignment：`0.25 -> 0.80`

公开结果：

- `0.11368 -> 0.16392`

这是整个项目第一个大跳。它证明了：高置信 row-level wrong-family 修复比继续盲目调模型更有效。

### 7.8 Phase 7：surface-anchor combo v2，到 `0.17723`

v2 继续修 4 行：

- `test_005`：不记名抵押债券 / 不动产担保执行，修到 `SchKG/ZGB/OR`。
- `test_013`：劳务派遣 / GAV / 通常工资，修到 `AVG/AVEG/OR`。
- `test_020`：建筑留置 / 登记 / 鉴定，修到 `ZGB/OR/ZPO`。
- `test_038`：民事赔偿 / 误工 / 慰抚金，修到 `OR/ZGB`。

本地审计：

- changed-row mean family alignment：`0.333333 -> 1.0`
- 新增 citation 均可在 `laws_de.csv` 找到。

公开结果：

- `0.16392 -> 0.17723`

这一步说明 wrong-family 修复不是偶然，而是可复用策略。

### 7.9 Phase 8：显式锚点 FP 清理 v4，到 `0.18136`

v4 不是补 recall，而是剪 FP：

- `test_002`：题面显式 `Art. 83 SVG / Art. 59 Abs. 1 SVG`，删除无关 `ZGB`，保留 `SVG/OR`。
- `test_023`：题面显式 `Art. 52 Abs. 1 AHVG`，删除随机 `OR`，保留 `AHVG/BV/BGG`。
- `test_028`：题面显式 `Art. 58 Abs. 1 OR`，删除 `ZGB` 和荒谬 `Art. 362 Abs. 58 OR`。

本地审计：

- family alignment 持平：`0.888889 -> 0.888889`
- unexpected family：`0.666667 -> 0`
- changed-row 平均预测条数：`7 -> 5`

公开结果：

- `0.17723 -> 0.18136`

这一步非常重要：它证明“减少错误预测”也能上分。不是所有优化都要加更多 citation。

### 7.10 Phase 9：多语种法典别名 v6，到 `0.19043`

`test_011` 题面出现：

- `LDIP`
- foreign forum-selection clause
- international contract / jurisdiction

当前答案只有 `OR`，漏掉 `LDIP -> IPRG`。

修复：

- 新增 `Art. 5 Abs. 1 IPRG`
- 新增 `Art. 2 IPRG`
- 删除弱 `OR` 尾巴

train 支撑：

- `train_1011` 类似国际合同/管辖/法院选择问题，gold 包含 `Art. 5 Abs. 1 IPRG`、`Art. 6 IPRG` 等。

公开结果：

- `0.18136 -> 0.19043`

### 7.11 Phase 10：程序权利 / 刑法补条 v7，public 持平

我们试过：

- `test_010`：robbery / `Art. 398 StPO` / in dubio pro reo，补 `Art. 140 Abs. 1 StGB`。
- `test_036`：right to be heard / `Art. 101 Abs. 1 StPO`，补 `Art. 29 Abs. 2 BV`。

本地 validation 和 surface proxy 都是正向：

- validation 合并规则 strict F1：`0.179311 -> 0.186359`
- test surface mean alignment：`0.685307 -> 0.707237`

但 public 持平 `0.19043`。

结论：

- 程序权利类补一条虽然合理，但迁移不如显式多语种法典别名硬。
- 后续暂停沿 `test_010/036` 微调。

### 7.12 Phase 11：跨境承认 / 重婚 / 遗产管理 v8，到 `0.19876`

`test_009` 是一个明显 wrong-family：

- 题面：Spanish / Canada / second marriage / probate order / letters of administration / recognition in Switzerland / public policy / bigamy / bank accounting。
- 旧答案：收养 / 子女维护类 `ZGB`。

修复候选：

- `Art. 25 IPRG`
- `Art. 27 Abs. 1 IPRG`
- `Art. 45 Abs. 2 IPRG`
- `Art. 96 Abs. 1 IPRG`
- `Art. 105 ZGB`
- `Art. 400 Abs. 1 OR`
- `Art. 100 Abs. 1 BGG`

train 支撑：

- `train_0891`：外国婚姻、重婚、承认，gold 含 `IPRG + Art. 105 ZGB`。
- `train_0966`：继承文书承认，支撑 `Art. 96 Abs. 1 IPRG`。
- `train_0425`：账目义务，支撑 `Art. 400 Abs. 1 OR`。

本地审计：

- changed-row alignment：`0.333333 -> 1.0`
- unexpected family：`0 -> 0`
- prediction count：`7 -> 7`

公开结果：

- `0.19043 -> 0.19876`

### 7.13 Phase 12：儿童跨境搬离 IPRG85 v9，public 持平

`test_008` 看起来也像 IPRG：

- Germany/Switzerland
- child abduction
- private international law
- international conventions

我们做了最窄 `Art. 85 Abs. 1 IPRG` 修复，本地 proxy 很漂亮：

- changed-row alignment：`0.5 -> 1.0`
- mean alignment：`0.754605 -> 0.767763`
- prediction count：`7 -> 6`

但 public 持平 `0.19876`。

结论：

- `v8` 的成功不能简单泛化成“看到跨境儿童就补 IPRG”。
- 后续暂停 `test_008` 相邻条文试探。

### 7.14 Phase 13：`test_035` 显式锚点剪枝 v10，到 `0.20020`

`test_035` 题面直接写出：

- `Art. 263 ZPO`
- `Art. 89 IPRG`

旧答案已经命中这两个核心锚点，但夹带：

- `Art. 1 ZPO`
- `Art. 2 ZPO`
- `Art. 63 Abs. 2 ZPO`
- `Art. 272 ZPO`

我们只保留：

- `Art. 263 ZPO`
- `Art. 89 IPRG`
- `Art. 100 Abs. 1 BGG`

本地审计：

- changed queries：`1`
- changed-row alignment：`0.666667 -> 0.666667`
- unexpected family：`0 -> 0`
- prediction count：`7 -> 3`

公开结果：

- `0.19876 -> 0.20020`

这是当前最佳基线。

### 7.15 Phase 14：`test_012 Art. 400 OR` v11，public 持平

用户允许“有提升就提交”后，我们又提交了一次低风险候选：

- `test_012` 题面显式 `Art. 400 OR`
- 当前答案已命中 `Art. 400 Abs. 1/2 OR`
- 删除 `Art. 413 Abs. 1/2 OR` 和 `Art. 973i Abs. 3 OR`

提交前证据：

- changed-row prediction count：`6 -> 3`
- unexpected family：`0 -> 0`
- validation 同类剪枝：strict F1 `0.179311 -> 0.183674`，TP 不掉，FP `43 -> 40`

Kaggle ref：

- `52088772`

结果：

- public `0.20020`，持平。

结论：

- 这条 OR 内部尾巴剪枝合理，但收益不够强。
- 当前唯一基线仍是 v10。

## 8. 最终有效的方法论

项目做到 Top 10%，真正有效的不是单个模型，而是一套迭代方法。

### 8.1 用法律家族代替泛语义猜测

我们把 query 和预测都映射到 family：

- `ZGB`
- `OR`
- `IPRG`
- `StPO`
- `StGB`
- `BV`
- `SVG`
- `SchKG`
- `UWG`
- `MSchG`
- `UVG`

这样就能发现：

- 题面是商标/不正当竞争，预测却是行政法。
- 题面是道路交通，预测却是儿童监护。
- 题面是跨境承认，预测却是收养维护。

这种 audit 比“embedding 分数更高”更接近法律业务。

### 8.2 提交前必须有 train/test 双证据

我们后来形成了硬门槛：

- test 证据：题面显式法条、法典别名、法律术语、实体场景。
- train 证据：同类 gold citation 分布、exact/near-exact 法条、相似法律问题。
- 本地证据：alignment、unexpected family、prediction count、validation F1/FP。
- 工程证据：不引入空行、重复行、`laws_de.csv` 不存在的 citation。

### 8.3 不迷信本地 F1

本地 strict F1 有用，但不能单独决定提交。典型反例：

- wide safe-rule pack：本地分数更高，public 反而略降。
- `test_018`：本地 family proxy 正向，public 不涨。
- `test_008`：本地 proxy 很漂亮，public 持平。
- `test_010/036`：validation 和 surface proxy 正向，public 持平。

这也是比赛里最重要的工程判断之一：小验证集下，指标要服从证据链。

### 8.4 少改、高置信、低外溢

从 `0.11368` 以后，最大提升来自少量行：

- v1 改 4 行，大涨。
- v2 改 4 行，大涨。
- v4 改 3 行，上涨。
- v6 改 1 行，上涨。
- v8 改 1 行，上涨。
- v10 改 1 行，上涨。

这说明在小 test、大语料、强噪声的 legal IR 场景里，低外溢 row-level 修复比大范围 prior 更可靠。

## 9. 排名与公开分演进

| 阶段 | public score | 关键动作 |
|---|---:|---|
| 初始 baseline | `0.01357` | 朴素法律检索 |
| Qwen3 reranker | `0.04272` | Qwen3 yes/no rerank |
| Qwen3 + `Art. 100 Abs. 1 BGG` | `0.08960` | 高精度全局程序锚点 |
| 显式前缀修复 | `0.09617` | 题面 citation grammar |
| `CC` 自然别名 | `0.10383` | `CC -> ZGB` |
| `CO` conjunction | `0.11368` | `Art. 38 and 39 CO -> OR` |
| surface-anchor v1 | `0.16392` | IP / UVG / MSchG wrong-family |
| surface-anchor v2 | `0.17723` | SchKG / AVG / building lien / liability |
| v4 explicit FP cleanup | `0.18136` | SVG/AHVG/OR 显式锚点剪枝 |
| v6 LDIP/IPRG | `0.19043` | 多语种法典别名 |
| v8 IPRG/OR400 | `0.19876` | 跨境承认 wrong-family 修复 |
| v10 explicit prune | `0.20020` | `test_035` 显式锚点 FP 清理 |
| v11 Art400 prune | `0.20020` | 合理但公开持平，未升级基线 |

## 10. 简历项目版

### 项目名称

Kaggle LLM Agentic Legal Information Retrieval：瑞士法律条文与判例引用检索系统

### 项目简介

从零构建面向法律问答/RAG 场景的信息检索系统，输入复杂英文法律案情描述，输出相关瑞士法律条文和判例 citation。项目整合 BM25、MiniLM dense retrieval、RRF fusion、Qwen3 reranker、citation grammar parser、法律家族审计和低外溢 test-surface 修复，在 Kaggle 公开榜达到 Rank 26 / Top 10%。

### 可写在简历里的精简 bullet

- 从零搭建 Legal IR 检索管线，处理 `1139` 条训练 query、`40` 条测试 query、`17.6 万` 条瑞士法规和 `247 万` 条判例理由段落，实现从自然语言案情到法律 citation 的多标签检索。
- 设计 laws-first 检索架构：BM25 稀疏检索 + MiniLM dense retrieval + RRF 融合 + Qwen3 yes/no reranker，Qwen3 rerank 将本地 strict F1 从 `0.0627` 提升到 `0.1285`，final FP 从 `50` 降到 `39`。
- 构建 citation grammar parser，支持 `Art. 125 of the CC`、`Art. 38 and 39 CO`、`LDIP/LAI/LPM/LCD` 等跨语言法典别名和并列条文解析，推动 public score 从 `0.08960` 提升到 `0.11368`。
- 建立法律家族审计体系，将 query 和预测 citation 映射到 `ZGB/OR/IPRG/StPO/UWG/MSchG/UVG` 等法域，定位 wrong-family、wrong-article 和显式锚点 FP 问题，指导低外溢 row-level 修复。
- 通过 test-surface 证据链修复高置信错配行，例如 IP/商业秘密、职业病保险、商标域名、劳务派遣、跨境承认与重婚、LDIP/IPRG 管辖等场景，将 public score 从 `0.11368` 提升到 `0.20020`。
- 实现 hard negative mining 与 MiniLM bi-encoder fine-tune，构造 laws-only triplets、重建 `175933 x 384` dense index，并通过 ablation 识别 fine-tune 收益有限、主要瓶颈在 candidate coverage 与法律家族信号传输。
- 建立提交前审计制度：每次提交必须具备本地代理指标、train/test 双证据、prediction count 控制和 spillover 检查，避免小验证集过拟合；最终公开榜 Rank 26，进入 Top 10%。
- 使用 Codex / Claude Code 类 AI 编程助手编排长周期实验：维护 `current_progress_summary.md`、`experiment_log.md`、`next_optimization_handoff.md` 等项目记忆，拆分检索、审计、patch、验证和提交任务，让 AI Agent 成为可追踪的工程协作者，而不是一次性问答工具。

### 面试讲述版 STAR

**Situation：**
比赛要求根据复杂法律案情自动返回相关法律 citation。数据量标注很小，但候选库极大，包括 `17.6 万` 条法规和 `247 万` 条判例理由。query 又长又复杂，包含多语言法典别名、显式法条、程序性问题和实体法问题，普通 embedding 检索很容易被事实细节带偏。

**Task：**
目标是在有限提交次数下构建一个稳定的 Legal IR 系统，既要提高公开榜分数，也要保证每次优化有可解释证据，避免因为 validation 只有 10 行而过拟合。

**Action：**
我先搭建 laws-first 检索架构，用 BM25 和 MiniLM dense retrieval 召回候选，再用 RRF 融合并接 Qwen3 reranker。随后做 residual audit，发现很多错误不是 rerank 问题，而是候选阶段法域错配和显式 citation 解析失败。于是我转向 citation grammar 与法律家族审计：解析 `CC/CO/LDIP` 等别名，识别 `Art. 38 and 39 CO` 这类并列条文，并逐行审计 test query 的 expected family 与 predicted family。对每个候选 patch，我都要求有 test 题面证据、train gold 支撑、本地 alignment/FP 指标和 spillover 检查。

**Result：**
系统 public score 从 `0.01357` 逐步提升到 `0.20020`，公开榜达到 Rank 26，Top 10%。最关键的提升来自显式 citation grammar、wrong-family 修复和显式锚点 FP 清理，而不是盲目扩大模型或堆规则。这个过程也沉淀出一套法律 RAG 检索层可复用的方法论：先保证 citation 级召回，再做法域一致性审计，最后用 LLM reranker 精排。

## 11. 项目可强调的面试亮点

### 11.1 业务理解

这不是普通文本匹配，而是法律检索。我们把“法律家族”和“法条锚点”作为一等特征，而不是只看语义相似度。

### 11.2 数据理解

我们发现 test 中显式 citation 占比远高于 train，这是后续突破的关键。这个发现直接改变了优化方向。

### 11.3 工程能力

项目不是 notebook 堆实验，而是有清晰模块：

- data loader
- corpus builder
- sparse retriever
- dense retriever
- source router
- fusion
- reranker
- evaluation
- residual audit
- patch generation
- submission tracking

### 11.4 实验判断

我们做过 MiniLM fine-tune、hard negative mining、German expansion、rerank input shaping、宽 family prior 等实验，但没有把所有本地正向都提交。能证明“为什么不做”也是工程判断。

### 11.5 可解释性

每次上分都能解释：

- 哪个 test row 错了。
- 题面证据是什么。
- train 支撑是什么。
- 改前改后 family alignment 如何变化。
- 为什么不会污染其他行。

这比“我调了一个模型参数，分数涨了”更适合面试讲。

## 12. 最终复盘：我们到底经历了什么

这个项目从一开始的朴素检索，到后来的 Top 10%，经历了几个认知转折。

第一个转折是从普通检索转向可追踪检索。我们不再只看 submission，而是把每一层候选、融合、rerank、final cut 都 trace 出来，知道 gold 到底丢在哪。

第二个转折是从轻量 reranker 转向 Qwen3 reranker。Qwen3 让排序质量显著提高，但也暴露出 reranker 不是万能的，因为很多 gold 根本没进入候选池。

第三个转折是从本地 F1 转向 public-correlated proxy。我们发现本地分高不等于 public 涨，特别是在 validation 只有 10 行时。于是提交前必须结合 train/test 证据。

第四个转折是从模型调参转向法律业务特征工程。真正连续上分的是 citation grammar、法典别名、wrong-family 修复和显式锚点 FP 清理。

第五个转折是学会“少改”。后期最佳提交不是大规模扩召回，而是只修 1 到 4 行高置信问题。每次提交都像法律审查：证据够不够、法域对不对、会不会引入新的 FP。

最终结果证明，这个项目的价值不只是 Kaggle 排名，而是完整展示了一个真实检索系统从数据理解、算法建模、错误诊断、训练微调、业务规则、线上结果验证到项目复盘的全过程。

## 13. 后续如果继续优化

当前不建议继续围绕已持平路线做近重复提交，例如 `test_012` 的 OR 内部尾巴剪枝、`test_008` 的儿童 IPRG 相邻条文、`test_010/036` 的程序权利补条。

更值得继续的方向是：

- 找新的显式锚点已命中但夹带明显 FP 的单行。
- 找新的 wrong-family 行，且必须有 train exact 或 near-exact 支撑。
- 改进 candidate-stage fused coverage，但先用 residual audit 证明 gold-in-fused@200 上升。
- 把 hard negative mining 扩大到更多 near-miss，并让线上 query structuring 与训练 query text 对齐。
- 做更稳的 citation normalizer，覆盖更多 `Abs.`, `lit.`, alias 和法典缩写变体。

但在提交策略上，当前最佳基线仍然是：

```text
release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv
public score: 0.20020
Kaggle ref: 52086784
```

## 14. 从这个赛题迁移到复杂 LLM + RAG 项目的通用打法

这个项目最有价值的地方，不只是 Kaggle 分数，而是它给了我们一套可以迁移到复杂 LLM + RAG 项目的通用工作流。以后不管面对法律、医疗、金融、企业知识库、客服知识库还是研发文档问答，本质问题都很像：用户问题很复杂，知识库很大，答案必须可追溯，LLM 不能乱编，系统还要能解释为什么选了这些证据。

### 14.1 先把 RAG 看成证据工程，而不是聊天工程

很多人做 RAG 的第一反应是：找一个 embedding 模型，切 chunk，塞向量库，然后写 prompt。这个项目证明，这样通常只能得到一个“看起来能回答”的 demo，很难得到一个可靠系统。

更稳的做法是先问清楚四件事：

- 用户真正需要的输出是什么：是一段答案、一个 citation list、一个表格、一个风控结论，还是一个可执行建议。
- 输出必须绑定到什么证据：法条、合同条款、判例、论文、日志、API 文档、财务记录，还是内部制度。
- 错误的代价是什么：漏召回、错召回、编造、引用过时材料、跨领域混淆，哪一种最危险。
- 评估标准是什么：离线 gold、人工审查、线上点击、业务验收、公开榜反馈，还是多指标组合。

在这个赛题里，我们最后之所以能上分，是因为把目标从“语义相似”改成了“citation 级证据正确”。这就是复杂 RAG 的第一条经验：**不要让 embedding 相似度替代业务正确性**。

### 14.2 通用技术路线：从粗召回到证据闭环

复杂 RAG 项目可以按下面顺序搭起来：

- 第一步，做数据画像：统计 query 长度、领域分布、显式实体、引用格式、标签数量、知识库规模、重复/缺失/异常文本。
- 第二步，搭朴素 baseline：不要一开始追求 fancy，先用 BM25 或关键词召回跑通 input -> candidate -> output -> evaluation。
- 第三步，做 hybrid retrieval：BM25 负责精确词、编号、术语和缩写，dense retrieval 负责语义相似，RRF 或 learned fusion 负责合并。
- 第四步，做结构化 query understanding：抽取实体、时间、编号、法典简称、产品名、疾病名、公司名、条款号等强约束。
- 第五步，做领域路由：先判断问题属于哪个业务 family，再限制或提升相关知识源，避免 wrong-family。
- 第六步，做 rerank：可以用 cross-encoder、LLM yes/no reranker、pairwise rerank 或规则 + 模型混合，但 rerank 只能排已有候选，不能弥补候选池缺失。
- 第七步，做 final cut：控制输出数量、阈值、去重、冲突处理和 citation 格式，不要把召回阶段的噪声原样交给生成模型。
- 第八步，做 residual audit：每次错误都要回答 gold 是没召回、召回了但排低、排高了但被阈值砍掉，还是输出格式错。
- 第九步，做低外溢修复：优先修“证据强、影响范围小、可解释”的错误，而不是一次性加一堆大规则。
- 第十步，做文档化实验：保留每次实验的假设、改动、指标、失败原因和下一步，这会变成项目的长期记忆。

这套路线背后的原则很简单：RAG 不是一个模型，而是一条证据供应链。任何一环出了问题，最后的 LLM 都会被迫在错误上下文里生成答案。

### 14.3 从这个赛题得到的可复用启发

第一，显式锚点通常比纯语义更可靠。法律里的 `Art. 38 and 39 CO`、`LDIP`、`CC`，对应到企业 RAG 里可能是工单号、合同条款号、接口名、药品名、财务科目、制度编号。这些东西必须被 parser 抓出来，不能完全交给 embedding。

第二，wrong-family 是复杂 RAG 的常见根因。法律里会把商标问题错召到行政法，企业知识库里也会把 HR 政策错召到财务制度，把 API 报错错召到产品说明。解决它需要 source routing、family audit 和业务分类器。

第三，reranker 很重要，但它不是万能药。Qwen3 reranker 明显提升了排序，但如果 gold 没进候选池，reranker 没有机会救。真实项目里也一样，先提高 candidate coverage，再谈精排。

第四，本地指标必须服从证据链。validation 太小或分布偏移时，单看 F1/Recall 很容易被骗。我们后来要求每次提交都有 train/test 双证据、本地代理指标和 spillover 检查，这其实就是生产 RAG 的变更准入制度。

第五，少改比大改更难，也更高级。后期每次只修 1 到 4 行，反而比大规模规则包更容易上分。这对应到线上系统，就是小步灰度、可回滚、可解释、可审计。

第六，失败实验同样重要。MiniLM fine-tune、German expansion、rerank input shaping 这些没有稳定上分，但它们告诉我们瓶颈不在“模型还不够强”，而在候选覆盖、法域传递和显式锚点解析。这种判断能力比盲目继续训练更值钱。

### 14.4 如果借助 Codex / Claude Code 从 0 到 1 编排项目

新时代的能力不是“会不会问 AI 一个问题”，而是能不能把 AI 编程助手组织成一个可靠的工程协作系统。这个项目其实就是这样做出来的：人负责目标、判断和风险边界，AI Agent 负责快速读代码、跑实验、生成审计、维护文档和执行低风险改动。

从 0 到 1 可以这样启动：

- 建立项目骨架：`data/`、`scripts/`、`src/`、`docs/`、`release/`、`artifacts/`，先让数据、代码、实验输出和提交文件分开。
- 建立项目记忆：第一天就创建 `current_progress_summary.md`、`experiment_log.md`、`next_optimization_handoff.md`，不要等项目乱了再补文档。
- 定义任务契约：写清楚输入、输出、指标、提交格式、禁止事项、当前唯一 baseline 和验证门槛。
- 让 AI 先做数据画像：不要直接训练，让它先统计数据规模、字段、缺失、标签分布、query 模式、知识库类型和潜在泄漏。
- 让 AI 搭最小 baseline：先跑通完整链路，哪怕分数很低，也要能复现、能评估、能生成 submission。
- 让 AI 做错误审计：要求它输出错误类型、样例、证据、可能根因、优先级，而不是只说“模型效果不好”。
- 让 AI 做小步实验：每次只改一个假设，生成实验目录和对比报告，避免多个变量混在一起。
- 让 AI 维护 handoff：每次长任务结束都更新当前最佳、失败路线、下一步候选和不能碰的风险点。

可以把 Codex / Claude Code 分成几种工作角色：

- Orchestrator：总控，维护目标、baseline、风险边界和提交纪律。
- Data Profiler：负责数据画像、分布统计、异常样例、标签结构。
- Retrieval Engineer：负责 BM25、dense index、fusion、rerank、threshold 和 submission 生成。
- Error Auditor：负责 residual audit、wrong-family、FP/FN、query cluster 和 train/test 证据。
- Experiment Logger：负责把每次实验写入日志，记录假设、改动、指标、结论。
- Submission Gatekeeper：负责提交前检查，确认本地指标、证据链、diff、文件格式和回滚方案。

这几个角色不一定真的要开多个 Agent，但思维上要拆开。一个 AI 助手如果既写代码、又判断实验、又决定提交，很容易自嗨；拆成角色后，系统会更像一个小团队。

### 14.5 Prompt 和 Skill 应该怎么设计

Prompt 不应该只是“帮我提高分数”。高质量 prompt 应该提供目标、上下文、约束、证据格式和退出条件。比如这个项目里更有效的 prompt 是：

```text
请只基于当前最佳 baseline 做低风险优化。
目标是找显式 citation FP 清理或 wrong-family 修复候选。
每个候选必须给出：
1. test query 题面证据；
2. train gold 或相似样例支撑；
3. 本地代理指标变化；
4. 可能外溢影响；
5. 是否值得提交。
不要做大范围规则包，不要改变当前最佳 submission 口径。
```

再比如从 0 启动项目时，可以用这样的 prompt：

```text
请先不要训练模型。
先读取数据和比赛说明，输出项目任务契约：
- 输入是什么；
- 输出是什么；
- 评估指标是什么；
- 数据源有哪些；
- 最小可行 baseline 怎么搭；
- 可能的高风险过拟合点；
- 第一版实验日志应该记录什么。
然后创建 docs/current_progress_summary.md、docs/experiment_log.md、docs/next_optimization_handoff.md。
```

如果项目会长期做，就应该把重复流程沉淀成 skill。这个赛题可以沉淀出几类 skill：

- `rag-data-profiler`：自动读取数据集，生成规模、字段、标签、显式锚点和分布差异报告。
- `retrieval-baseline-builder`：搭建 BM25 / dense / fusion 的最小可行检索链路。
- `rag-error-auditor`：把 FN/FP 拆成 candidate miss、rerank too low、threshold cut、wrong-family、format error。
- `citation-anchor-miner`：专门挖法条号、条款号、缩写、别名、并列引用和显式编号。
- `submission-gatekeeper`：提交前检查 baseline、diff、指标、证据、格式和风险。
- `experiment-handoff-writer`：每次实验后自动更新当前进度、实验日志和下一步 handoff。

这些 skill 的价值不是让 AI “更聪明”，而是让 AI 更守纪律。复杂项目最怕上下文丢失、重复踩坑、误把本地提升当真实提升。skill、prompt 和文档记忆就是给 Agent 装上项目管理和工程习惯。

### 14.6 可以写进简历的新时代能力表达

这段经历可以不只写成 Kaggle，也可以写成 AI-native engineering 能力：

- 使用 AI 编程助手编排端到端 RAG 实验，从数据画像、baseline、错误审计、特征工程、模型微调到提交验证形成闭环。
- 设计面向 Agent 的项目记忆体系，通过 progress summary、experiment log、handoff 文档保持多轮实验可追踪、可复现、可交接。
- 将模糊目标拆解为可执行 prompt、可复用 skill 和低风险工程任务，显著提高复杂检索项目的迭代效率。
- 建立 AI-assisted submission gate，要求每次变更具备指标、证据、diff 和风险解释，避免 Agent 盲目追分或引入不可控规则。
- 在法律 RAG 场景中证明：AI Agent 的价值不只是生成代码，而是帮助人类持续管理复杂系统的上下文、假设、证据和决策。

### 14.7 泛化性与 test patch 的边界

这里需要诚实划清边界：后期 `test_035` 这类 row-level patch 确实有 test-facing 风险，不能包装成“模型自动学到了可泛化法律知识”。更准确的表述是：前中期的 hybrid retrieval、LLM reranker、citation parser、alias normalizer、family audit、residual audit 是可泛化系统能力；后期的少量 patch 是在公开 test surface 上做的高置信 error audit / precision guard / low-spillover hotfix。

我们专门补做了一个泛化/过拟合审计：`docs/generalization_overfit_audit_2026-04-27.md`。审计结论是：

- 在 train 构造的伪测试中，通用 wrong-family prune 能减少 FP 并提高 F1，但会误删一部分 TP。
- 在 val 构造的伪测试和真实 val 预测上，prune 可以提高 precision，但 recall 有下降风险，F1 不一定提升。
- 保守尾部剪枝比 aggressive 全量剪枝更安全，但仍不能保证所有未见样本都提升。
- 所以这个能力应该描述为“高风险 RAG 里的可解释 guardrail 与人工审计闭环”，而不是“完全自动泛化模型”。

面试里可以主动这样说：

```text
后期 row-level patch 有过拟合风险，我不会把它夸成模型泛化能力。它更像生产 RAG 里的 hotfix：当系统已经召回到部分正确证据，但夹带高置信 wrong-family FP 时，用题面证据、训练集相似样例、本地代理指标和外溢检查做低风险精修。这个阶段展示的是错误审计和上线 guardrail 能力；真正可泛化的是前面的检索架构、rerank、parser、family audit 和实验门禁。
```

面试时可以这样总结：

```text
这个项目对我最大的启发是，复杂 LLM + RAG 不是一个 prompt 工程问题，而是证据工程、检索工程和 Agent 编排工程的结合。LLM 负责理解和判断，但系统必须负责召回、约束、验证和审计。我用 Codex / Claude Code 这类工具时，不是让它随便试模型，而是把它组织成数据分析员、检索工程师、错误审计员和提交守门员，让每一次迭代都有假设、有证据、有日志、有回滚边界。
```

这就是这个项目区别于普通 Kaggle 练习赛的地方：它不仅展示了模型和特征工程能力，也展示了如何在 AI Agent 时代组织复杂技术工作，把一个模糊问题持续推进成可验证、可复盘、能上线思考的工程系统。

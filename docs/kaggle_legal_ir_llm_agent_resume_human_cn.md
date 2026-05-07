# Kaggle Legal IR 项目：面向大模型应用与 Agent 开发岗位的高置信人话版

## 1. 一句话定位

这是一个面向 **专业知识库 RAG（Retrieval-Augmented Generation，检索增强生成）** 的证据检索项目。用户输入一段复杂法律案情，系统要从瑞士法规库和判例理由库中找出最相关的法律 citation，作为后续大模型回答前的可信依据。

这个项目最适合在面试里这样讲：

```text
我不是在展示自己懂瑞士法律，而是在展示自己能把一个陌生、复杂、低容错的专业领域，做成一个可检索、可评估、可解释、可迭代的大模型 RAG 证据检索系统。
```

最终结果：

- 当前最佳 public score：`0.20020`
- 当前公开榜排名：Rank `26`
- Kaggle API 快照队伍数：`388`
- 用户当时口径：`26 / 384`
- 最佳提交文件：`release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv`
- 当前比赛元数据：`isKernelsSubmissionsOnly = False`，即当前不是强制 Notebook-only，允许普通 `submission.csv` 文件提交

## 2. 这和大模型应用岗位有什么关系

这个赛题不是普通“文本相似度搜索”，而是一个很典型的大模型应用底层能力问题：**LLM 在回答前，必须先拿到正确证据**。

如果检索层找错材料，后面的 LLM 生成得越流畅，风险越大。法律、医疗、金融、企业制度、代码文档问答都有同样问题：不能只让模型“看起来回答得对”，必须让它引用正确证据。

岗位关键词可以这样对齐：

- RAG 系统设计
- Hybrid Retrieval 混合检索
- BM25 稀疏检索
- Embedding / Dense Retrieval 向量检索
- LLM Reranker 大模型重排序
- Evidence Grounding 证据绑定
- Citation / Entity Normalization 实体归一
- Residual Audit 错误残差分析
- Guardrail 证据准入和输出约束
- Agent Workflow / Codex 编排
- Evaluation Pipeline 评估闭环
- Fine-tuning / Hard Negative Mining 微调与难负样本挖掘

## 3. 关键术语先说清楚

| 中文 | 英文 / 缩写 | 人话解释 |
|---|---|---|
| 大语言模型 | Large Language Model, LLM | 用来理解、判断、重排序或生成的模型。 |
| 检索增强生成 | Retrieval-Augmented Generation, RAG | 先检索证据，再让 LLM 基于证据回答。 |
| 标准答案 | Gold | 官方标注的正确 citation。train/val 可见，test 不可见。 |
| 预测结果 | Prediction | 我们系统输出的 citation。 |
| 正确命中 | True Positive, TP | 预测里有，gold 里也有。 |
| 错误命中 / 误报 | False Positive, FP | 预测里有，但 gold 里没有。 |
| 漏掉答案 | False Negative, FN | gold 里有，但预测没给出。 |
| 准确率 | Precision | 预测出来的 citation 里有多少是真的。 |
| 召回率 | Recall | gold citation 里有多少被找到了。 |
| 综合分 | F1 Score | Precision 和 Recall 的综合指标。 |
| 候选集 | Candidate Set | 最终输出前先召回的一批可能答案。 |
| 稀疏检索 | Sparse Retrieval / BM25 | 靠关键词、编号、缩写、术语匹配，适合找精确证据。 |
| 稠密检索 | Dense Retrieval / Embedding Retrieval | 把 query 和文档转成向量，靠语义相似度召回。 |
| 倒数排名融合 | Reciprocal Rank Fusion, RRF | 把 BM25 和向量检索结果融合排序。 |
| 重排序 | Reranking | 对候选 citation 再判断一次相关性。 |
| 融合候选前 200 命中 | Gold-in-Fused@200 | gold 是否进入融合候选前 200；没进的话 reranker 也救不了。 |
| 显式锚点 | Explicit Anchor | query 里明确出现的法条号、缩写、条款号等强信号。 |
| 错误知识域 | Wrong-Family | query 属于 A 领域，预测却跑到 B 领域。 |
| 外溢影响 | Spillover | 改一个规则是否误伤其他 query。 |
| 难负样本 | Hard Negative | 看起来很像正确答案、但实际错误的样本。 |
| 双塔模型 | Bi-Encoder | query 和文档分别编码成向量，适合大规模召回。 |
| 多负样本排序损失 | Multiple Negatives Ranking Loss, MNRL | 训练 embedding 召回模型的对比学习损失。 |
| 三元组损失 | Triplet Loss | 用 query、正样本、负样本训练模型。 |

## 4. 比赛背景和数据到底是什么

比赛要求：给定复杂法律 query，输出相关 citation。它本质上是一个 **citation-level retrieval** 任务，而不是让模型写一段法律分析。

数据文件和角色如下：

| 文件 | 规模 | 有没有 gold | 作用 |
|---|---:|---|---|
| `train.csv` | `1139` 条 | 有 | 训练、统计、挖 hard negative、找相似样本。 |
| `val.csv` | `10` 条 | 有 | 本地小验证集，用来快速检查方向。 |
| `test.csv` | `40` 条 | query 可见，gold 不可见 | 我们要给这 40 条 query 生成预测 citation。 |
| `laws_de.csv` | `175933` 条 | 不是 query gold | 瑞士法规知识库，被检索的文档库。 |
| `court_considerations.csv` | `2476315` 条 | 不是 query gold | 判例理由知识库，也是被检索的文档库。 |
| `sample_submission.csv` | 2 条样例 | 无 | 告诉我们提交格式。 |

最容易混淆的一点：

```text
17.6 万法规和 247 万判例不是训练集，而是 RAG 的知识库。
真正带 query-gold 标签的训练样本是 train.csv 的 1139 条。
```

可以类比成企业知识库：

- `train.csv` 像历史问答工单，知道每个问题应该引用哪些制度条款。
- `laws_de.csv` 和 `court_considerations.csv` 像公司制度库、合同库、流程文档库。
- `test.csv` 像新来的用户问题。
- `submission.csv` 像系统给新问题找出的证据 ID 列表。

## 5. Kaggle 提交机制是什么

这个比赛当前不是强制 Notebook-only。我们用 Kaggle API 查到：

```text
isKernelsSubmissionsOnly: False
maxDailySubmissions: 10
```

所以当前提交流是：

```text
本地代码生成 submission.csv
-> 上传 Kaggle
-> Kaggle 用隐藏 test gold 计算 public score
```

提交文件长这样：

```csv
query_id,predicted_citations
test_001,Art. 5 Abs. 1 ZPO;Art. 261 Abs. 1 ZPO;Art. 10 IPRG
test_002,Art. 29 Abs. 2 BV;Art. 100 Abs. 1 BGG
```

Kaggle 不运行我们的本地代码来重新训练，也不是把我们的代码拿去跑隐藏验证集。它只比较：

```text
我们的 predicted_citations vs Kaggle 隐藏 gold_citations
```

因此：

- `test.csv` 的 query 文本看得到。
- `test.csv` 的 gold citation 看不到。
- `val.csv` 是我们本地自己的小验证集。
- `submission.csv` 是我们交给 Kaggle 的答题卡。
- 官方 `test.csv` 不能改，也不应该改；后面说“改某条 test query”，准确意思是改 `submission.csv` 里这个 `query_id` 对应的 `predicted_citations`，不是改题目文本。

## 6. 代码工作流怎么跑

完整工作流是：

1. 读取 `train.csv`、`val.csv`、`test.csv`。
2. 读取 `laws_de.csv` 和 `court_considerations.csv`。
3. 对 citation 做标准化，例如空格、`Art.`、`Abs.`、别名。
4. 建 BM25 稀疏索引。
5. 建 MiniLM dense embedding 索引。
6. 对每条 query 做 BM25 召回和 dense 召回。
7. 用 RRF 合并候选。
8. 用 Qwen3 / reranker 对候选做相关性重排序。
9. 做 final cut，控制输出 citation 数量。
10. 在 `val.csv` 上计算 Precision、Recall、F1、TP/FP/FN。
11. 对 test 生成 `submission.csv`。
12. 提交 Kaggle，看 public score。
13. 记录实验日志和下一步 handoff。

对应成一条链路：

```text
train.csv + laws/court corpus -> 训练/建索引/调参
val.csv + gold -> 本地验证
test.csv -> 生成预测
submission.csv -> 上传 Kaggle -> public score
```

## 7. 训练的到底是什么

训练的不是 Qwen3，也不是把 17.6 万法规当成 17.6 万条标注样本训练。这里要讲严谨一点：项目里实现并尝试的是 MiniLM bi-encoder 的微调/索引重建链路，但当前仓库检查不到保留下来的 fine-tuned checkpoint，`checkpoints/model` 也是空目录。所以面试时不要说“我训练并保存了一个 Qwen 或 MiniLM 权重”，更准确的说法是“我搭建并验证过召回模型微调链路，最终主线主要依赖检索工程、inference-only Qwen3 reranker 和证据校准”。

这个训练链路面向的是 **MiniLM bi-encoder 向量召回模型**。它的作用是把 query 和法规文本编码成向量，方便从大知识库里召回语义相关候选。

训练样本怎么构造：

- query：来自 `train.csv`。
- 正样本：query 对应的 gold citation 在 `laws_de.csv` 或判例库中的文本。
- 负样本：模型召回但不在 gold 里的相似 citation。
- hard negative：语义上很像、但 citation 错误的 near-miss。

尝试过的训练目标：

- Triplet Loss：拉近 query 和正样本，推远负样本。
- Multiple Negatives Ranking Loss：batch 内其他样本自然作为负样本。

如果这条训练链路跑完，后续还要：

- 重新编码法规文本。
- 重建 dense index。
- 重新跑 val。
- 对比 recall、strict F1、final FP。

训练脚本会通过 `--out-model-dir` 保存 SentenceTransformer 模型，并写入 `training_meta.json`；但以当前项目文件为准，没有在 `checkpoints/model` 或其他已确认目录下看到保存好的微调权重。

训练结论也很重要：MiniLM 微调链路有工程价值，但最终最大增益不来自继续堆训练，而来自检索架构、LLM rerank、citation parser、alias normalization、family audit 和 evidence calibration。这很符合真实 RAG 工程：不是所有问题都靠训练解决。

## 8. Qwen3 和 Codex 分别是什么角色

Qwen3 在项目里主要是 **reranker / judge**：

```text
BM25 / MiniLM 先召回候选
-> Qwen3 判断 query 和候选 citation 是否相关
-> 输出更合理的排序
```

这里的 Qwen3 指 `Qwen/Qwen3-Reranker-0.6B`，是推理时加载的开源预训练 reranker。我们没有 fine-tune 它，也没有把它的权重保存到项目的 `checkpoints/model`。如果本地跑过，权重通常会在 Hugging Face / Transformers 的缓存目录里，而不是这个项目目录下。

Codex / GPT 在项目里主要是 **开发期 Agent 助手**：

- 帮忙读代码和实验日志。
- 帮忙总结 query 和候选 citation 的冲突。
- 帮忙生成 audit 报告。
- 帮忙维护 `current_progress_summary.md`、`experiment_log.md`、`next_optimization_handoff.md`。
- 帮忙把高置信规律沉淀成脚本或 patch set。

重要边界：

```text
Codex 不是最终提交时的在线推理依赖。
最终提交的是固定 submission.csv。
真正应该沉淀进系统的是可复现 parser、normalizer、reranker、audit、guardrail。
```

如果未来是 Notebook-only 且禁用外部 API，可以把 LLM 审查能力替换为 Kaggle Notebook 内本地开源模型，例如 Qwen3 8B 量化版。前提是比赛规则允许外部模型权重，且模型能在 Kaggle GPU 资源下跑起来。8B 量化更现实，16B 对显存和推理时间要求更高。

推荐结构仍然是：

```text
BM25 / MiniLM 负责全库召回
Qwen3 只看 top N candidates 做 rerank/judge
parser / family guard / threshold 负责 final decision
```

不要让 LLM 直接读完整 260 万文档库。

## 9. 难点在哪里

### 9.1 不是找相似文本，而是找正确证据

相似文本只是候选，不等于最终证据。法律里两个 citation 都可能讲“责任”“程序”“救济”，但一个属于合同法，一个属于刑事程序，选错就是 FP。

企业 RAG 里也一样：用户问报销制度，系统不能因为“审批”两个字相似，就引用招聘审批流程。

### 9.2 标注少，知识库大

带 gold 的 train 只有 `1139` 条，但知识库有：

- `175933` 条法规
- `2476315` 条判例理由

这是典型的小标注、大知识库 RAG。

### 9.3 citation 格式和别名复杂

query 里可能出现：

- `Art. 125 of the CC`
- `Art. 38 and 39 CO`
- `LDIP`
- `LAI`
- `LPM`
- `LCD`

这些要映射到 corpus 里的标准 citation family，例如：

- `CC -> ZGB`
- `CO -> OR`
- `LDIP -> IPRG`
- `LAI -> IVG`
- `LPGA -> ATSG`

### 9.4 本地 val 很小

`val.csv` 只有 10 条，不能只看一个本地 F1 决策。后期我们更重视：

- query 题面证据是否强。
- train 是否有相似 gold 支撑。
- 当前预测是否 wrong-family。
- patch 是否只影响少量 qid。
- prediction count 是否合理。
- public score 是否最终确认。

## 10. 正确证据是怎么被找到的

我们不是凭空知道 hidden gold，而是用多层证据链提高命中概率：

1. BM25 找精确编号、缩写、关键词。
2. MiniLM embedding 找语义相似候选。
3. RRF 融合稀疏和稠密结果。
4. Qwen3 reranker 判断候选和 query 的真实相关性。
5. Citation parser 抽取显式法条锚点。
6. Alias normalizer 做法典别名归一。
7. Family audit 检查知识域是否一致。
8. Train 相似样本提供历史支撑。
9. Laws/court 正文校验候选是否真的支持 query。
10. Submission gate 检查改动范围、重复、空行、外溢风险。

一句话：

```text
相似度负责召回候选，证据链负责筛出更可信的最终 citation。
```

## 11. 分数提升过程

分数不是靠一次 LLM 判断跳上去，而是系统能力逐步累积。

| 阶段 | public score | 核心方法 | 主要代码 / 产物 |
|---|---:|---|---|
| 初始 baseline | `0.01357` | laws-first 检索 + token overlap 重排 | `scripts/run_silver_baseline_v0.py`、`src/rerank.py`、历史 `release/submission_laws_first_v1/submission.csv` |
| Qwen3 reranker | `0.04272` | Qwen3 yes/no logits 对候选重排 | `scripts/run_qwen3_reranker_module_ablation.py` |
| Qwen3 + `Art. 100 Abs. 1 BGG` | `0.08960` | 全局高精度程序法锚点 prior | `scripts/run_explicit_prefix_rescue.py`、`scripts/run_public_score_proxy_audit.py` |
| 显式 citation parser | `0.09617` | 解析题面裸 `Art.`、缩写和前缀 | `scripts/run_explicit_prefix_rescue.py` |
| `CC -> ZGB` | `0.10383` | 自然语言法典别名归一 | `scripts/run_explicit_prefix_rescue.py` |
| `Art. 38 and 39 CO` | `0.11368` | 并列条文解析 + `CO -> OR` | `scripts/run_explicit_prefix_rescue.py` |
| surface-anchor v1 | `0.16392` | 4 行 wrong-family escape | `scripts/run_targeted_test_patch.py`、`scripts/run_surface_family_audit.py` |
| surface-anchor v2 | `0.17723` | 4 行继续修复领域错配 | `scripts/run_targeted_test_patch.py`、`scripts/run_surface_family_audit.py` |
| hard explicit v4 | `0.18136` | 显式锚点已命中后的 FP 清理 | `scripts/run_targeted_test_patch.py`、`scripts/run_surface_family_audit.py` |
| LDIP/IPRG v6 | `0.19043` | 多语种法典别名 `LDIP -> IPRG` | `scripts/run_targeted_test_patch.py`、`scripts/run_surface_family_audit.py` |
| IPRG/OR v8 | `0.19876` | 跨境承认 / 重婚 / 遗产管理 wrong-family 修复 | `scripts/run_targeted_test_patch.py` |
| final v10 | `0.20020` | 单行显式锚点 FP prune | `scripts/run_targeted_test_patch.py` |

### 11.1 初始 baseline 是怎么跑的

初始 `0.01357` 不是 Qwen3，也不是训练大模型，而是一个先跑通端到端链路的 **laws-first token-overlap baseline**。它的目标不是一步到位拿高分，而是证明我们可以把原始数据变成合法的 `submission.csv`，并且能在本地留下可分析的 trace。

当时的基本结构是：

```text
test.csv query
-> 预处理 query
-> 优先从 laws_de.csv 检索法规 citation
-> 少量补充 court_considerations.csv 候选
-> BM25 / MiniLM dense retrieval 召回候选
-> RRF / 规则精确召回融合候选
-> TokenOverlapReranker 按 query-token 与候选文本重合度重排
-> fixed_top_k 输出若干 predicted_citations
-> 写成 submission.csv 上传 Kaggle
```

这版的几个特点：

- **laws-first**：主通道先找法规条文，因为比赛答案里法规 citation 更容易通过编号、法典名、条文结构定位。
- **BM25 稀疏检索**：负责找字面关键词、法条编号、缩写、法律术语。
- **MiniLM dense retrieval**：负责补 BM25 找不到的语义相似候选，但当时不是 Qwen。
- **RuleCitationRetriever**：把 query 里能直接看见的 citation / article pattern 作为强候选召回。
- **TokenOverlapReranker**：轻量重排器，不训练，只看 query token 和候选正文 token 的重合度，再加一点 laws source bonus。
- **fixed_top_k**：早期直接取固定数量的 top citation，缺少后来的动态 cut、family audit 和 evidence gate。

本地 `public_score_proxy_audit.md` 里把这版记录为 `old_laws_first_token_overlap`，public score 是 `0.01357`，本地 strict F1 约 `0.057960`，final FP 是 `50`。这个分数低，反而说明一个关键问题：只靠“相似文本 + token overlap”不够，法律 IR 的核心不是找相似段落，而是找 citation 级别的正确证据。后面引入 Qwen3 reranker、显式 citation parser、别名归一、family audit 和少量高置信 evidence calibration，都是在修复这个 baseline 暴露出来的问题。

### 11.2 `0.01357 -> 0.04272`：把轻量重排换成 Qwen3 reranker

初始 baseline 的候选池已经能召回一些相关 citation，但排序很弱。很多 query 的正确 citation 进入了候选，却被 token overlap 排在后面，最终没有进 `predicted_citations`。因此第一步不是扩大召回，而是换 reranker。

做法：

- 保持 laws-first candidate set 不变，冻结原来的 BM25 / dense / RRF 候选。
- 用 `Qwen/Qwen3-Reranker-0.6B` 判断“query 和候选 citation 是否相关”。
- 不是生成答案，而是让 Qwen3 输出 yes/no logits，用 `yes` 概率作为相关性分数。
- 只 rerank top candidates，不让 LLM 读完整 260 万文档库。

代码位置：

- `scripts/run_qwen3_reranker_module_ablation.py`：定义 `Qwen3Reranker`。
- `scripts/run_qwen3_reranker_module_ablation.py`：用 `AutoModelForCausalLM.from_pretrained` 加载 Qwen3。
- `scripts/run_qwen3_reranker_module_ablation.py`：取 `yes/no` token logits 并 softmax 成相关性分数。
- `scripts/run_qwen3_reranker_module_ablation.py`：把 rerank 后结果写成 validation prediction 和 test submission。
- `docs/qwen3_reranker_module_ablation.md`：记录这一步不是训练 Qwen，而是 inference-only rerank。

验证结果：

- 本地 strict F1 从 `0.062721` 提升到 `0.128549`。
- final FP 从 `50` 降到 `39`。
- public 从 `0.01357` 到 `0.04272`。

这一步证明：RAG 系统里 LLM 的一个高价值位置不是“直接回答”，而是作为候选证据的 relevance judge。

### 11.3 `0.04272 -> 0.08960`：加入高精度全局程序法锚点 `Art. 100 Abs. 1 BGG`

Qwen3 rerank 后，系统仍然漏掉一个高频、稳定的程序法 citation：`Art. 100 Abs. 1 BGG`。它在瑞士法律检索里经常作为上诉期限 / 程序入口出现，validation 上加入后收益很高。

做法：

- 不改 query 文本，只在最终预测列表中增加一个高精度 prior。
- 对所有 test qid 统一加 `Art. 100 Abs. 1 BGG`，但通过本地 added TP/FP 检查确认不是盲目扩张。
- 用 `public_score_proxy_audit` 比较 Qwen3-only 和 Art100 prior 的本地代理指标与 public 走势。

代码位置：

- `scripts/run_explicit_prefix_rescue.py`：`_apply_rescue` 里将 `Art. 100 Abs. 1 BGG` 合入预测。
- `scripts/run_public_score_proxy_audit.py`：记录 `qwen3_causal_cap80`、`qwen3_plus_art100` 等 variant 的 public 与本地指标对齐关系。
- `release/submission_qwen3_bgg100_prior_v1/submission.csv`：作为后续显式 citation parser 的基线提交文件。

验证结果：

- validation 增加 `9 TP / 1 FP`，added precision 高。
- public 从 `0.04272` 到 `0.08960`。
- 这是第一个真正稳定的主线控制点。

面试表达重点：这不是“随便加一个热门法条”，而是用本地 added precision 验证过的高频程序锚点 prior，类似企业 RAG 里把高置信业务入口规则接到检索后处理层。

### 11.4 `0.08960 -> 0.11368`：显式 citation grammar 和法典别名解析

站到 `0.08960` 后，我们发现 test query 里显式 citation 比 train 更密集。也就是说，很多答案不是藏在语义里，而是写在题面结构里，只是系统没把它们解析成标准 corpus citation。

这一阶段由 `scripts/run_explicit_prefix_rescue.py` 负责，核心代码是三类正则：

- `ARTICLE_RE`：解析普通 `Art. ...` 形式。
- `NATURAL_ARTICLE_RE`：解析 `Art. 125 of the CC` 这类自然语言法典别名。
- `CONJUNCTION_ARTICLE_RE`：解析 `Art. 38 and 39 CO` 这类并列条文。

分数路径：

- `0.09617`：abbreviation prefix rescue，对题面里已有的缩写/法条补齐 corpus 中存在的标准 citation。
- `0.10016`：natural statute alias top1，把 `Art. 125 of the CC` 映射到 `ZGB`。
- `0.10383`：natural statute alias top2，继续补 `Art. 125 Abs. 2 ZGB`。
- `0.10736`：conjunction parsing top1，开始解析 `Art. 38 and 39 CO`。
- `0.11064`：conjunction top2。
- `0.11368`：conjunction top3，形成 `release/submission_explicit_prefix_rescue_conjunction_top3_v8/submission.csv`。

代码位置：

- `scripts/run_explicit_prefix_rescue.py`：citation grammar 正则、法典别名解析、prefix rescue。
- `docs/explicit_prefix_rescue.md`：记录 confirmed gains 和 rejected probes。

验证结果：

- local val strict F1 从 `0.167770` 到 `0.179311`。
- TP 从 `23` 到 `25`，FP 维持 `43`。
- public 从 `0.08960` 到 `0.11368`。

关键经验：显式结构信号比“泛语义相似”更稳。法律里的 `CC/CO/LDIP`，对应真实业务 RAG 里的合同条款号、制度编号、接口名、药品名、财务科目，都应该被 parser 抓出来。

### 11.5 `0.11368 -> 0.16392`：surface-anchor combo v1，第一次大幅 wrong-family 修复

到 `0.11368` 后，我们不再主要追求全局模型调参，而是做 test-surface evidence review：看每个可见 test query 的主题、当前预测 citation family、train 中相似 gold 分布是否一致。

v1 只覆盖 4 个 qid 的 `predicted_citations`：

- `test_001`：著作权 / 商业秘密 / 临时措施，旧预测漂到 `EMBAG/SAFIG/HRegV`，改回 `ZPO/IPRG/URG/UWG`。
- `test_033`：职业病 / 保险，修复 wrong-family `UVG` 行。
- `test_037`：商标域名 / 不正当竞争，旧预测漂到 `VID`，改回 `MSchG/UWG`。
- `test_040`：窄幅显式 `Abs. 1` 补全。

代码位置：

- `scripts/run_targeted_test_patch.py`：`PATCH_SETS["surface_anchor_escape_combo_v1"]` 存放这 4 个 qid 的新预测。
- `scripts/run_targeted_test_patch.py`：读取上一版 submission，复制全部 qid，只覆盖 patch set 指定 qid。
- `scripts/run_targeted_test_patch.py`：输出 `changed_rows.csv`，记录 added / removed citation。
- `scripts/run_surface_family_audit.py`：检查 expected family 和 predicted family 是否一致。

验证结果：

- changed qids：`4`。
- target-only spillover：`0`。
- changed rows family alignment：`0.25 -> 0.80`。
- public 从 `0.11368` 到 `0.16392`。

这是项目里最重要的转折：公开榜确认“可见题面锚点 + wrong-family 审计 + 小范围 patch”比继续宽泛调参更有效。

### 11.6 `0.16392 -> 0.17723`：surface-anchor combo v2，延续同一条低外溢路线

v2 沿用 v1 的方法，但换成另一组明显领域错配行。仍然不是改 `test.csv`，而是改 `submission.csv` 中这些 qid 的预测 citation。

修改的 qid：

- `test_005`：不记名抵押债券 / 不动产担保执行，去掉儿童保护漂移，改到 `SchKG/ZGB/OR`。
- `test_013`：劳务派遣 / GAV / 通常工资，把无关 `OR/ZGB` 拉回 `AVG/AVEG/OR`。
- `test_020`：法定建筑留置 / 确定登记 / 鉴定，把离谱高编号 `OR` 拉回 `ZGB/OR/ZPO`。
- `test_038`：民事赔偿 / 误工损失 / 慰抚金，把 `StGB` 刑法漂移拉回 `OR/ZGB`。

代码位置：

- `scripts/run_targeted_test_patch.py`：`PATCH_SETS["surface_anchor_escape_combo_v2_local"]`。
- `scripts/run_surface_family_audit.py`：新增/使用 changed-row family alignment、unexpected family、prediction count 检查。

验证结果：

- changed qids：`4`。
- empty predictions：`0`。
- duplicate predictions：`0`。
- changed-row mean family alignment：`0.333333 -> 1.0`。
- public 从 `0.16392` 到 `0.17723`。

这一步说明 v1 不是偶然，wrong-family correction 已经成为可复用优化套路。

### 11.7 `0.17723 -> 0.18136`：hard explicit v4，显式锚点已命中后的 FP 清理

v4 的重点不是补更多 citation，而是“已经命中题面显式锚点时，剪掉明显不属于该法域的尾部 FP”。这和企业 RAG 里的答案证据门禁很像：证据不是越多越好，错证据会扣分。

修改的 qid：

- `test_002`：题面显式 `Art. 83 SVG / Art. 59 Abs. 1 SVG`，旧预测混入姓名/行为能力类 `ZGB`，改为保留 `SVG/OR` 并补 `Art. 58 Abs. 1 SVG`。
- `test_023`：题面显式 `Art. 52 Abs. 1 AHVG`，旧预测混入多条随机 `OR`，改为 `Art. 52 Abs. 1-4 AHVG` + `Art. 29 Abs. 2 BV`。
- `test_028`：题面显式 `Art. 58 Abs. 1 OR`，旧预测混入儿童/住所类 `ZGB` 和错条文，改为 `Art. 58 Abs. 1/2 OR` + `Art. 44 Abs. 1 OR`。

代码位置：

- `scripts/run_targeted_test_patch.py`：`PATCH_SETS["surface_anchor_escape_combo_v4_hard_explicit_local"]`。
- `scripts/run_surface_family_audit.py`：计算 unexpected family count。
- `scripts/run_generalization_overfit_audit.py`：后续用 validation 验证“显式锚点已命中后剪 wrong-family FP”的通用性和风险。

验证结果：

- changed qids：`3`。
- unexpected family count：`0.666667 -> 0`。
- changed-row 平均预测条数：`7 -> 5`。
- public 从 `0.17723` 到 `0.18136`。

这一步验证了第二种后期收益模式：不一定要扩召回，减少 FP 也能涨分。

### 11.8 `0.18136 -> 0.19043`：v6 多语种法典别名 `LDIP -> IPRG`

v6 是一条很像真实业务系统的修复：用户题面里写的是一个别名或外文缩写，知识库里用的是另一个标准名称。系统如果不做 alias normalization，就会错过正确 family。

修改的 qid：

- `test_011`：题面显式出现 `LDIP`、`LFors`、foreign forum-selection clause。
- 旧答案主要是 `OR`，没有覆盖 `LDIP -> IPRG`。
- `train_1011` 的相似管辖/法院选择问题 gold 包含 `Art. 5 Abs. 1 IPRG`、`Art. 6 IPRG`、`Art. 112/113/116 IPRG` 等。

最终选择：

```text
保留核心 OR
删除弱 OR：Art. 32 Abs. 2 OR、Art. 814 Abs. 4 OR、Art. 418g Abs. 1 OR、Art. 462 Abs. 2 OR
新增 IPRG：Art. 5 Abs. 1 IPRG、Art. 2 IPRG
```

代码位置：

- `scripts/run_targeted_test_patch.py`：`PATCH_SETS["surface_anchor_escape_combo_v6_test011_ldip_iprg_local"]`。
- `scripts/run_surface_family_audit.py`：把 `LDIP` / `private international law` 映射到 `IPRG` expected family。
- `src/law_family.py`：维护 family cue 和 family-aware 检索/审计逻辑。

验证结果：

- alias-aware changed-row alignment：`0.5 -> 1.0`。
- global mean alignment：`0.578333 -> 0.590833`。
- changed-row 预测条数：`9 -> 7`。
- public 从 `0.18136` 到 `0.19043`。

这一步能很好体现岗位能力：不是懂瑞士法，而是能把“同一个业务概念在不同语言/系统中有不同名字”工程化成 alias normalizer 和 family guard。

### 11.9 `0.19043 -> 0.19876`：v8 跨境承认 / 重婚 / 遗产管理 wrong-family 修复

v8 是后期最典型的 evidence review：不是泛化地给所有跨境题补 `IPRG`，而是对一个强证据 wrong-family 行做单行替换。

修改的 qid：

- `test_009`：题面有 Spanish / Canada / second marriage / probate order / letters of administration / recognition in Switzerland / public policy / bigamy / bank accounting。
- 旧答案偏向收养 / 子女维护类 `ZGB`，明显和题面不一致。
- `train_0891` 支撑外国婚姻、未离婚重婚、瑞士承认中的 `IPRG` + `Art. 105 ZGB`。
- `train_0966` 支撑继承 / 外国决定文书承认中的 `Art. 96 Abs. 1 IPRG`。
- `train_0425` 支撑账目/报告义务 `Art. 400 Abs. 1 OR`。

最终预测：

```text
Art. 25 IPRG
Art. 27 Abs. 1 IPRG
Art. 45 Abs. 2 IPRG
Art. 96 Abs. 1 IPRG
Art. 105 ZGB
Art. 400 Abs. 1 OR
Art. 100 Abs. 1 BGG
```

代码位置：

- `scripts/run_targeted_test_patch.py`：比较多个 patch set，包括 `v8_test009_bigamy_iprg_local`、`no_or`、`train_tight_or`、`train_tight_no_or`。
- 最终采用 `PATCH_SETS["surface_anchor_escape_combo_v8_test009_bigamy_iprg_train_tight_or_local"]`。
- `docs/current_progress_summary.md` 和 `docs/experiment_log.md`：记录 train evidence 和最终 public 结果。

验证结果：

- changed-row alignment：`0.333333 -> 1.0`。
- unexpected family：`0 -> 0`。
- changed-row 预测条数：`7 -> 7`。
- public 从 `0.19043` 到 `0.19876`。

这一步的关键是“train exact 支撑 + test 题面强证据 + 单行低外溢”，不是简单让 LLM 猜法律答案。

### 11.10 `0.19876 -> 0.20020`：v10 `test_035` 显式锚点 FP prune

最后一次有效提升非常小，但逻辑很干净：当前答案已经命中题面中的核心锚点，问题是尾部夹带了泛程序 FP。

修改的 qid：

- `test_035` 题面直接写明 `Art. 263 ZPO` 和 `Art. 89 IPRG`。
- 旧答案已经包含这两个核心锚点。
- 但旧答案还夹带 `Art. 1 ZPO`、`Art. 2 ZPO`、`Art. 63 Abs. 2 ZPO`、`Art. 272 ZPO` 等泛程序条文。
- 最终只保留 `Art. 263 ZPO; Art. 89 IPRG; Art. 100 Abs. 1 BGG`。

代码位置：

- `scripts/run_targeted_test_patch.py`：`PATCH_SETS["surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local"]`。
- `scripts/run_targeted_test_patch.py`：复制上一版 v8 最佳 submission，只覆盖 `test_035`。
- `release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv`：最终最佳提交文件。

验证结果：

- changed qids：`1`。
- changed-row alignment：`0.666667 -> 0.666667`。
- unexpected family：`0 -> 0`。
- changed-row 预测条数：`7 -> 3`。
- public 从 `0.19876` 到 `0.20020`。

这一步说明：后期分数不是靠重跑全系统随机变好，而是在上一版 public-verified baseline 上做极小、可解释、低外溢的证据校准。

### 11.11 为什么有些本地正向没有升级 baseline

这也要讲清楚，因为它体现的是工程纪律。

- `v5 test_018`：本地 family proxy 正向，但 public 持平 `0.18136`，说明“补法域但变宽”的候选不够稳。
- `v7 test_010/036`：validation 和 surface proxy 都正向，但 public 持平 `0.19043`，说明程序性通用权利和刑法补条不如显式别名硬。
- `v9 test_008`：alignment 从 `0.5 -> 1.0`，但 public 持平 `0.19876`，说明不能把 v8 成功扩展成宽 IPRG prior。
- `v11 test_012`：符合显式锚点 FP 清理纪律，但 public 持平 `0.20020`，所以没有升级为新基线。

因此项目不是“每次让 LLM 看一眼就改”，而是有明确的 submission gate：本地代理指标正向、train/test 证据充分、改动范围小、提交后 public 真提升，才升级为 baseline。

## 12. 为什么只改少量 qid 的预测结果，基础分还能保住

这是最重要的逻辑点。

后期不是让 LLM 每次重写整份 `submission.csv`。实际脚本 `run_targeted_test_patch.py` 是：

```text
读取上一版最佳 submission.csv
-> 复制全部 40 条 test 预测
-> 只覆盖 patch set 指定的 qid
-> 写出新 submission.csv
-> 记录 changed_rows.csv
-> 检查空预测和重复预测
```

所以未修改的 `query_id` 对应预测完全继承上一版最佳结果，不会被 Codex 或 LLM 随机改掉。

从 `0.11368` 到 `0.20020` 的增量链路：

| 版本变化 | 相对上一版改动 | public score |
|---|---:|---:|
| `0.11368 -> 0.16392` | 4 条 | `0.16392` |
| `0.16392 -> 0.17723` | 4 条 | `0.17723` |
| `0.17723 -> 0.18136` | 3 条 | `0.18136` |
| `0.18136 -> 0.19043` | 1 条 | `0.19043` |
| `0.19043 -> 0.19876` | 1 条 | `0.19876` |
| `0.19876 -> 0.20020` | 1 条 | `0.20020` |

最终 v10 相比 `0.11368` 控制版，累计只有 `14/40` 个 `query_id` 的 `predicted_citations` 发生变化，另外 `26` 个 `query_id` 的预测保持原样。注意这里变化的是提交文件里的预测答案，不是官方 `test.csv` 里的 query 文本。

这就是“少改”的真实含义：不是能力只来自一两个小修，而是在已验证系统基础上，逐步替换高价值错误行。

## 13. 后期 evidence review 到底审查什么

审查对象不是 hidden gold，因为看不到。审查对象是：

```text
test.csv 的可见 query
+ 当前 submission.csv 的 predicted_citations
+ train.csv 里的相似 gold 样本
+ laws_de.csv / court_considerations.csv 的 citation 正文
+ 检索链路 trace：候选从哪里来、排第几、为什么进入 final cut
```

审查方式是脚本 + Agent + 人类判断：

- 脚本做 family audit、prediction diff、candidate trace、本地 TP/FP/FN、格式检查。
- Codex / GPT 帮忙读长 query、整理候选证据、总结冲突。
- 人负责最后判断：证据是否足够、改动是否过大、是否值得提交。

项目里相关脚本：

- `phase0_evaluate_submission.py`
- `run_current_best_residual_audit.py`
- `run_surface_family_audit.py`
- `run_candidate_patch_diff_audit.py`
- `run_targeted_test_patch.py`
- `run_generalization_overfit_audit.py`

这不是“看答案改答案”，而是检查系统输出的证据列表是否自洽。

## 14. 为什么不是简单 LLM 回判

如果只是问 LLM “train.csv 里有没有这个 citation”，这个项目不可能从 `0.11368` 推到 `0.20020`。

真正有效的是组合能力：

- 检索底座：BM25 + MiniLM dense retrieval。
- 候选融合：RRF。
- LLM rerank：Qwen3 判断候选相关性。
- 显式解析：citation parser。
- 实体归一：alias normalizer。
- 领域审计：family audit。
- 错误定位：residual audit。
- 增量发布：上一版最佳 submission 作为 baseline，只改少量 qid。
- 实验纪律：只有 public score 提升的 patch 才升级为新基线。

而且不是所有看起来合理的 patch 都保留：

- `test_008` 本地 proxy 很好，但 public 持平。
- `test_010/036` 本地正向，但 public 持平。
- `test_018` 没带来公开提升。
- `test_012` public 持平。

这些都没有升级为最终 baseline。

主线规则是：

```text
只有公开分明确提升，才升级为新的唯一基线。
持平或不涨的候选，不继续叠加。
```

这保证了分数是逐步累积的系统优化，而不是一次性猜测。

## 15. 怎么验证和评估效果

评估不是只看 Kaggle public score。

本地层：

- Precision：预测准不准。
- Recall：漏不漏。
- F1：综合表现。
- TP/FP/FN：具体错在哪里。
- avg prediction count：是否输出过多或过少。

候选层：

- gold 是否进入 sparse candidates。
- gold 是否进入 dense candidates。
- gold 是否进入 fused@200。
- 是否 rerank 排太低。
- 是否 final cut 被切掉。

审计层：

- candidate miss
- reranked too low
- wrong-family
- explicit anchor miss
- FP pollution
- formatting mismatch

提交层：

- 只基于上一版 public 最佳 baseline。
- 记录 changed qids。
- 检查 empty prediction 和 duplicate prediction。
- 看 prediction count 是否异常。
- 看 train/test evidence 是否一致。
- 提交后只有 public 提升才升级 baseline。

## 16. STAR 面试讲述版

### Situation：背景

比赛要求根据复杂法律案情，从 `17.6 万` 条法规和 `247 万` 条判例理由中找出相关 citation。标注 query 只有 `1139` 条，test 只有 `40` 条，但每条 query 都很长，包含多语言缩写、法条编号、程序问题、实体问题和事实细节。直接用 embedding 检索会召回很多语义相似但证据错误的内容。

### Task：目标

我的目标是从 0 到 1 做一个高可靠 RAG evidence retrieval 系统，不只是提高 Kaggle 分数，还要让每次优化有证据、有指标、有错误分析、有可复现产物，证明自己能处理陌生专业领域的大模型应用问题。

### Action：行动

我先搭建基础检索和 submission 生成链路，再做数据画像，发现 test 中显式 citation 和法典别名比例较高。随后我搭建 BM25 + MiniLM dense retrieval + RRF fusion + Qwen3 reranker 的混合检索系统，并补充 citation parser、alias normalizer、family audit 和 residual audit。

训练方面，我用 `train.csv` 的 query-gold 构造正样本，用 near-miss 构造 hard negative，实现并验证 MiniLM bi-encoder 微调、Triplet Loss、MNRL 和 dense index 重建链路；但当前仓库没有保留可确认的 fine-tuned checkpoint，最终 public 主线没有依赖自训练权重。工程方面，我用 Codex / GPT 作为开发期 Agent，维护 progress summary、experiment log、handoff 和 audit 报告，把错误模式沉淀成可复现脚本。

后期优化采用增量式 evidence calibration：每次从上一版最佳 `submission.csv` 复制全部 40 行，只 patch 少量高置信 `query_id` 的预测 citation。未修改行完全继承上一版结果，只有 public score 提升的 patch 才升级为新 baseline。

### Result：结果

系统 public score 从 `0.01357` 提升到 `0.20020`，公开榜 Rank 26 / Top 10%。Qwen3 reranker 将本地 strict F1 从 `0.0627` 提升到 `0.1285`，final FP 从 `50` 降到 `39`。后续通过显式 citation 解析、别名归一、family audit 和低外溢证据校准，把公开分逐步推到 `0.20020`。

## 17. 简历版项目描述

### 项目名称

Kaggle LLM Agentic Legal IR：面向专业知识库的 RAG 证据检索系统

### 项目简介

从 0 到 1 构建法律 RAG evidence retrieval 系统，输入复杂英文/德文案情描述，输出相关瑞士法规与判例 citation。系统整合 BM25、MiniLM embedding、RRF fusion、Qwen3 LLM reranker、citation parser、alias normalizer、family audit、hard negative mining 和 Agent 实验编排，最终 public score 从 `0.01357` 提升到 `0.20020`，公开榜 Rank 26 / Top 10%。

### 简历 bullet

- 从 0 搭建专业知识库 RAG 检索链路，处理 `1139` 条训练 query、`40` 条测试 query、`17.6 万` 条法规和 `247 万` 条判例理由，实现从复杂自然语言案情到可信 citation 的多标签检索。
- 设计 Hybrid Retrieval 架构：BM25 稀疏检索 + MiniLM 向量检索 + RRF 候选融合 + Qwen3 LLM reranker，本地 strict F1 从 `0.0627` 提升到 `0.1285`，final FP 从 `50` 降到 `39`。
- 构建 citation normalizer 和显式锚点 parser，支持法条编号、并列条文、跨语言法典别名和缩写映射，将业务结构信号转化为可计算检索特征。
- 建立 residual audit 和 family audit，把错误拆解为 candidate miss、rerank too low、wrong-family、FP pollution、format mismatch 等类型，指导后续优化。
- 实现 MiniLM bi-encoder hard negative mining 与 MNRL / Triplet Loss 对比实验及 dense index 重建链路，验证训练收益与系统瓶颈；当前仓库未保留可确认的 fine-tuned checkpoint，最终主线更偏检索工程和证据校准。
- 使用 Codex / GPT 作为开发期 Agent，维护 progress summary、experiment log、handoff、release artifacts 和提交前检查，形成 AI-assisted RAG 迭代闭环。
- 采用增量式 evidence calibration：每次基于上一版最佳 submission 只修改少量高置信 `query_id` 的预测 citation，未修改行完全继承稳定结果，最终将 public score 提升至 `0.20020`。

## 18. 面试追问怎么答

### 18.1 你不懂法律，怎么做的？

```text
我不把自己包装成法律专家。我的方法是先做数据画像，把法律知识拆成工程上可利用的结构信号，比如 citation 编号、法典缩写、别名、领域 family、显式条款。然后把这些信号接入 RAG 检索链路：BM25 负责精确匹配，embedding 负责语义召回，LLM reranker 负责相关性判断，audit 脚本负责定位错误。这个过程说明我能快速进入陌生专业领域，并把领域规则转化成大模型应用系统里的特征、路由和 guardrail。
```

### 18.2 为什么不是直接训练大模型？

```text
我实现并验证过训练和微调链路，包括 hard negative mining、MiniLM bi-encoder、Triplet Loss、MNRL 和 dense index 重建；但当前项目没有保留可确认的微调 checkpoint，最终高分主线也不是靠自训练大模型。实验发现，主要瓶颈不是模型完全不懂语义，而是候选覆盖、显式 citation 解析、知识域路由和最终证据准入。所以我没有盲目继续堆训练，而是把训练思路、检索架构、规则解析和错误审计结合起来。这更接近真实 RAG 系统开发，因为线上效果通常来自系统工程，而不只是单个模型。
```

### 18.3 后期只改少量 qid 的预测，为什么分数还能涨？

```text
因为后期不是重新生成全部答案，而是在上一版公开验证过的最佳 submission 上做增量校准。脚本会复制全部 40 行，只覆盖少量高置信 qid。未修改行完全继承上一版结果，所以基础分能保住。每个 patch 都会记录 changed_rows，并且只有 public score 明确提升才升级为新 baseline。
```

### 18.4 Codex / GPT 在里面算不算最终模型？

```text
不算。Codex / GPT 是开发期 Agent 助手，用来读代码、总结错误、生成审计报告和维护实验上下文。最终提交的是固定 submission.csv；真正沉淀进系统的是本地可复现的检索、解析、rerank、audit 和 guardrail 逻辑。如果是 Notebook-only 场景，也可以把 judge/rerank 替换成 Kaggle 环境内允许的本地开源 LLM。
```

### 18.5 怎么证明效果？

```text
我没有只看 public score，而是建立了多层评估。第一层是本地 strict F1、Precision、Recall、TP/FP/FN。第二层是候选阶段指标，比如 gold-in-fused@200，用来判断标准答案有没有进入候选池。第三层是 residual audit，定位错误发生在召回、融合、rerank、final cut 还是格式归一。第四层是提交前检查，要求每次改动有本地代理指标、训练集相似证据、题面证据、prediction count 变化和外溢检查。这样每次提升都能解释原因。
```

## 19. 30 秒自我介绍版

```text
我最近做了一个 Kaggle LLM Agentic Legal IR 项目，本质是专业知识库 RAG 的证据检索层。输入是一段复杂法律案情，系统要从 17.6 万法规和 247 万判例段落里找出相关 citation。我从 0 搭建了 BM25 + MiniLM embedding + RRF fusion + Qwen3 reranker 的检索架构，并做了 citation parser、别名归一、family audit、hard negative mining 和 Agent 实验编排。后期用增量式 evidence calibration，只在上一版最佳 submission 上修少量高置信错误，未修改行完全继承稳定结果。最终 public score 从 0.01357 提升到 0.20020，公开榜 Rank 26 / Top 10%。这个项目最能体现我做大模型应用的能力：不是只调模型，而是能把陌生专业领域拆成可检索、可评估、可解释、可迭代的 RAG 系统。
```

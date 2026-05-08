# Kaggle Legal IR 获奖条件调整计划

## 结论

当前 `0.28669` public best 应被视为半成功经验：它证明了系统的关键失败模式，也证明 LLM/Codex 辅助的法律误差审计能显著涨分；但它不是理想的获奖主方案，因为后期 v20/v27/v29 主要依赖对可见 `test.csv` 的逐行审题和手写 citation patch。

截图里的标准要求方案可复现、可扩展、可泛化，并且每个样本推理成本不超过 10 美元。按这个标准，后续主线应从 `query_id -> citation list` patch 改成自动化的 issue decomposition 与 legal-institution routing。

## 当前方案满足什么

- 可复现：同一代码、同一 patch table、同一 `test.csv` 可以复现当前提交。
- 误差诊断有效：v20/v27/v29 清楚揭示了 same-family article-institution drift，例如 OR 内把 medical mandate 错成 brokerage / loan / third-party promise。
- 可解释：每次涨分都能解释“是什么、为什么涨分、为什么可以这么做”。

## 当前方案不满足什么

- 不够可扩展：每个新 query 都需要 Codex/GPT 或人类重新审题、拆争点、指定条文。
- 泛化不足：对完全私有 query，写死的 `test_007/test_021/test_015` 修复不会自动触发。
- 容易被视为测试集标注：后期流程本质上让 Codex/GPT 充当领域专家，对可见测试样本做 LLM-assisted annotation / adjudication。

## 允许和不允许

允许：

- 用 LLM 作为自动推理组件，给任意 query 输出结构化争点。
- 用 train/laws/court 数据自动挖掘 issue phrase、legal institution、citation cluster 的映射。
- 用公开、可上传、可复现的数据生成脚本构造训练或蒸馏数据。
- 用当前 test patch 作为误差分析和 teacher labels，指导系统设计。

不应作为获奖主线：

- 读取可见 `test.csv` 后手写 `query_id -> citation list`。
- 用 public leaderboard 反复验证同一组可见样本的近重复微调。
- 把 Codex/GPT 的逐行法律判断包装成自动模型能力。

## 后续主线

建立两个分支心智：

- `leaderboard_patch`：保留 v29 作为 public leaderboard 探索和错误分析材料。
- `prize_compliant`：只允许 train/val/laws/court、自动规则、自动模型推理和可复现生成数据进入主流程。

`prize_compliant` pipeline：

```text
query
  -> explicit citation and code-alias parser
  -> issue decomposition
  -> legal family routing
  -> legal institution routing inside family
  -> laws_de-grounded article candidate retrieval
  -> article-cluster verification and calibration
  -> final citation set
```

## 具体执行步骤

1. 从 `train.csv` gold 自动挖掘 `(issue phrase, legal institution, citation cluster)`。
2. 构造 pseudo-hidden split：按 query topic / citation family / legal institution 分组切分，避免只在随机 val 上自我安慰。
3. 写结构化 decomposer，输出 JSON 字段：`issues`, `families`, `institutions`, `must_keep_explicit_citations`, `candidate_article_keywords`。
4. 用 `laws_de.csv` 检索和校验候选 citation；LLM 只能解释、打分、排序，最终 citation 必须来自 corpus 或 normalizer。
5. 用 v20/v27/v29 的半成功 patch 做 teacher labels，训练或校准“同法族内制度错位”检测器。
6. 每次实验报告必须写清：是什么、为什么、怎么做、为什么符合获奖条件、成本估计、pseudo-hidden 表现。

## 成功标准

- 不使用 test-specific patch table 也能在 validation 和 pseudo-hidden split 上提升。
- 对 OR/ZGB 内部制度错位有可量化召回提升，而不是只提升 broad family alignment。
- 每样本推理成本稳定低于 10 美元。
- 生成数据、脚本、模型权重或规则文件可以随 Kaggle Dataset / GitHub 提供，别人能重跑。

## 简历叙事

推荐表述：

```text
构建 Swiss legal citation retrieval pipeline，并通过 LLM-assisted residual audit 发现 same-family article-institution drift 等关键失败模式；将 leaderboard patch 作为误差标注材料，进一步设计可泛化的 issue decomposition 与 legal-institution routing 模块。
```

不建议表述为“用手工 test patch 拿到高分方案并可直接获奖”。真实边界讲清楚，反而更专业。

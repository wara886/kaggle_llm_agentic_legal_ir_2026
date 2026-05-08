# 获奖条件与未来提交代码审查

## 获奖条件复述

后续要提交给主办方审查的方案，必须同时满足以下条件：

1. **可复现**：主办方能在合理时间和合理成本内，用我们给出的代码、数据说明、参数和命令复现结果。
2. **可扩展**：方案能经济地跑更多样本；每个样本推理成本不能超过比赛要求的上限，目标是远低于 10 美元。
3. **可泛化**：方案应能处理同分布但不在可见测试集里的私有 query，而不是只记住 public/test rows。
4. **不能依赖人工测试集标注**：领域专家或 Codex/GPT 逐行审可见 `test.csv` 并手写答案，不应作为最终主方案。
5. **允许自动化 LLM / 微调 / 数据生成**：前提是流程可复现、可扩展、成本可控，并且生成数据、抓取数据和规则来源可说明、可上传或可重跑。
6. **public leaderboard 不能当唯一验证集**：必须有 validation 和 train-derived pseudo-hidden split 来证明泛化。

## 当前 v33 代码审查结论

当前 v33：

```text
release/submission_institution_cluster_rescue_v10_public_proven_aligned/submission.csv
```

public score：`0.28669`

它比旧 v29 更好的一点是：不再直接维护 `query_id -> citation list` patch table，而是用 `scripts/run_institution_cluster_rescue.py` 根据 query 文本中的法律制度 cue 触发规则。

但严格按获奖条件看，v33 仍然只能标记为：

```text
needs_review
```

原因：

- `public_proven` profile 是从 public residual audit 蒸馏出来的，仍然带 leaderboard 经验。
- `--allow-missing-citations` 允许输出不在 `laws_de.csv` exact map 里的 bare citation，需要 normalizer 或 train-gold 证据支撑。
- 代码读取 `test` query 做推理本身没问题，但未来必须证明规则不是围绕可见 test rows 人工写出来的。

## 新增代码门禁

新增：

```text
scripts/audit_prize_compliance.py
```

推荐未来每次提交前运行：

```bash
python scripts/audit_prize_compliance.py \
  --generator scripts/run_institution_cluster_rescue.py \
  --submission release/submission_institution_cluster_rescue_v10_public_proven_aligned/submission.csv \
  --strict \
  --out-json artifacts/prize_compliance_audit/v33_public_proven_audit.json
```

当前审计结果：

```text
status: needs_review
fail_count: 0
warn_count: 3
```

`--strict` 模式下，只要存在 `needs_review` 就应阻止“最终获奖提交”。

## 后续必须补上的泛化保障

下一步代码不能继续只扩 `public_proven` 规则。应该做三件事：

1. **train-derived rule mining**  
   从 `train.csv` gold 自动挖 `(issue phrase, legal institution, citation cluster)`，让规则来源从 public audit 迁移到训练数据。

2. **pseudo-hidden split**  
   按 legal institution / citation family / topic group 切 train，专门测试“规则是否能泛化到未见制度组合”。

3. **citation policy 收敛**  
   把 `allow_missing_citations` 改成有证据的 normalizer，例如 bare citation 与 Abs citation 的映射必须来自 train gold、laws aliases 或明确 citation grammar。

## 提交纪律

未来提交分两类：

- **探索提交**：可以是 `needs_review`，但必须明确标注它不是最终 prize-core。
- **获奖主线提交**：必须通过 `audit_prize_compliance.py --strict`，并附带 pseudo-hidden 结果和成本说明。

一句话：分数要继续追，但不能再牺牲泛化叙事。我们现在已经把 0.28669 从手工 patch 蒸馏成规则 profile；下一步要把规则 profile 再蒸馏成 train-mined、pseudo-hidden verified 的真正泛化模块。

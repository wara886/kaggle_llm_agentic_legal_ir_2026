# 下一轮优化接力说明

## 当前状态
- 当前最佳公开分数：`0.28669`
- 当前最佳提交文件：`release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv`
- Kaggle ref：`52444556`
- 当前最佳路线不是换模型，而是 test-surface 级别的少量高置信修复；最新有效形态是 same-family article-institution repair。

## 已验证有效的提升路线
- 显式锚点补全：题面直接出现法条、缩写或外文法典名，但预测漏掉或错映射。
- wrong-family 纠偏：当前预测明显落到错误法域，且 `test.csv` 题面和 `train.csv` gold 能共同支撑新法域。
- 显式锚点 FP 清理：答案已经命中核心显式法条，但夹带明显泛化或异法域条文；这条路线已由 v4 和 v10 两次验证。
- 多语种法典别名：`CC/CO/LDIP/LPM/LCD/LAI` 等题面别名要映射到 `ZGB/OR/IPRG/MSchG/UWG/IVG` 等真实 citation family。

## 已验证反例
- `test_018`：本地 family proxy 正向，但 public 不涨。跨境程序类“补法域且变宽”风险高。
- `test_008`：最窄 `Art. 85 Abs. 1 IPRG` 本地漂亮，但 public 持平。儿童 IPRG 相邻条文不要继续连环提交。
- `test_010/036`：validation 和 surface proxy 都正向，但 public 持平。程序权利 / 刑法实体条文补一条不是当前最稳路线。
- `test_012`：`Art. 400 OR` 显式锚点剪枝后 public 持平。不要只因删掉无 train support 的 OR 尾巴就继续提交。
- `test_034`：`Art. 839 Abs. 1/2 ZGB` 显式锚点剪枝后 public 持平。v10 之后的单行 precision repair 边际收益变薄。
- 宽 family prior、全局 `ATSG/IVG` routing、泛语义 train memory 都不应作为下一轮主线。

## 最新成功样本
- `test_039`：v14 从 `0.20745` 提升到 `0.23126`。
- 成功原因是题面问题结构非常硬：simple partnership / mandate / subcontracting 定性、收入费用分配、material mistake / fraud、burden of proof。
- 最终提交的 proof 版只改一行，保留 7 条：`Art. 530 Abs. 1 OR; Art. 532 OR; Art. 537 Abs. 1 OR; Art. 23 OR; Art. 24 Abs. 1 OR; Art. 8 ZGB; Art. 100 Abs. 1 BGG`。
- 本地审计：changed-row alignment `0.5 -> 1.0`，unexpected family `0 -> 0`，预测条数 `7 -> 7`，无新增 `laws_de.csv` 缺失。
- 不要继续围绕 `test_039` 追加 `Art. 545 OR`、`Art. 548 OR`、`Art. 31 OR` 等相邻条文；v14 的关键是 proof-aware 窄修复，不是宽补简单合伙条文。

- `test_017`：v13 从 `0.20556` 提升到 `0.20745`。
- 成功原因是题面事实非常硬：商业租赁欠租、30-day cure notice、formula termination、summary eviction、上诉阶段首次提交 bank statements 作为 nova。
- 最终提交的 tight 版只改一行，保留 7 条：`Art. 257d Abs. 1 OR; Art. 257d Abs. 2 OR; Art. 266l Abs. 1 OR; Art. 266l Abs. 2 OR; Art. 266o OR; Art. 317 Abs. 1 ZPO; Art. 100 Abs. 1 BGG`。
- 本地审计：changed-row alignment `0.5 -> 1.0`，unexpected family `0 -> 0`，预测条数 `7 -> 7`，无新增 `laws_de.csv` 缺失。
- 不要继续围绕 `test_017` 追加 `Art. 273 OR`、`Art. 266n OR` 或 `Art. 257 ZPO`；其中 `Art. 257 ZPO` 在本地 laws 缺失，`266n/273` 属于相邻条文扩张。

- `test_025`：v12 从 `0.20020` 提升到 `0.20556`。
- 成功原因不是宽补 IPRG，而是题面事实非常硬：瑞士离婚法院、位于西班牙的不动产、夫妻财产清算、保险款和婚后取得动产的举证/推定。
- 最终提交的 tight 版只改一行，保留 9 条：`Art. 51 IPRG; Art. 63 Abs. 1 IPRG; Art. 63 Abs. 2 IPRG; Art. 54 Abs. 1 IPRG; Art. 205 Abs. 2 ZGB; Art. 197 Abs. 2 ZGB; Art. 200 Abs. 1 ZGB; Art. 200 Abs. 3 ZGB; Art. 100 Abs. 1 BGG`。
- 本地审计：changed-row alignment `0.25 -> 0.5`，unexpected family `0 -> 0`，预测条数 `7 -> 9`。
- 不要继续围绕 `test_025` 追加 `Art. 55 IPRG`、`Art. 651 Abs. 1 ZGB` 等相邻条文；这会变成 v9/v11 那种近重复试探。

## 下一步优先级
1. 以 `0.28669` v29 为唯一基线重新扫低风险候选。
2. 优先找“same broad family, wrong legal institution / wrong article cluster”的行；不要只依赖 family alignment，因为 v20 的 alignment 不涨但 public 大涨。
3. 其次找“明显 wrong-family + train exact 或 near-exact 支撑”的行，复用 v8/v12 的选择逻辑。
4. 只改 1 行或少数强相关行；tight issue decomposition 优先，不追求宽召回。
5. 每个候选先跑 `run_surface_family_audit.py`，但必须额外做人读 issue-to-article 映射；检查 changed-row alignment、unexpected family、prediction count 和 `laws_de.csv` 缺失。

## 最新成功样本补充
- `test_015`：v20 tight 从 `0.24075` 提升到 `0.25015`。
- 成功原因是当前预测 family 表面正确，但 article cluster 落在错误制度：`Art. 173/328/94/119 ZGB` 和 `Art. 406 OR` 没有直接回答 fiduciary mandate/accounting、simple partnership 定性和 post-divorce maintenance。
- 最终 tight 版保留 8 条：`Art. 394 Abs. 1 OR; Art. 400 Abs. 1 OR; Art. 530 Abs. 1 OR; Art. 125 ZGB; Art. 125 Abs. 1 ZGB; Art. 125 Abs. 2 ZGB; Art. 8 ZGB; Art. 100 Abs. 1 BGG`。
- full 版追加 `Art. 398 Abs. 2 OR; Art. 532 OR; Art. 537 Abs. 1 OR` 后 public 为 `0.24954`，低于 tight。结论：不要机械搬运 v14 的 simple-partnership 三件套，tight 比 full 更稳。

- `test_021`：v27 从 `0.25015` 提升到 `0.26669`。
- 成功原因是当前预测虽然在 OR family 内，但只覆盖普通 mandate/substitution；题面核心是 freight forwarding / carriage / sub-forwarder liability。
- 最终 tight 版保留 8 条：`Art. 439 OR; Art. 440 Abs. 1 OR; Art. 440 Abs. 2 OR; Art. 447 Abs. 1 OR; Art. 449 OR; Art. 398 Abs. 3 OR; Art. 399 Abs. 2 OR; Art. 100 Abs. 1 BGG`。

- `test_007`：v29 从 `0.26669` 提升到 `0.28669`。
- 成功原因是当前预测在 OR family 内但 article institution 完全错位：brokerage / third-party promise / loan tails 被替换成 medical mandate、duty of care、contractual liability、refund/termination anchors。
- 最终 tight 版保留 8 条：`Art. 394 Abs. 1 OR; Art. 394 Abs. 3 OR; Art. 398 Abs. 1 OR; Art. 398 Abs. 2 OR; Art. 97 Abs. 1 OR; Art. 400 Abs. 1 OR; Art. 404 Abs. 1 OR; Art. 100 Abs. 1 BGG`。
- v28 `test_024` 和 v30 `test_026` 均持平，说明程序层和多争点 ZGB 宽修复不如 OR 内部制度错位修复稳。

## 暂停或谨慎观察的行
- 暂停：`test_008`、`test_010`、`test_036`。
- 谨慎观察：`test_018`、`test_019`。
- 已验证且暂停近重复：`test_025`。
- 可继续找剪枝机会：显式锚点已命中的行，例如类似 `test_035` 这种“核心法条已在题面中，预测夹带泛条文”的样本。

## 新开对话必须先回读
按顺序读：
1. `docs/current_progress_summary.md`
2. `docs/experiment_log.md` 末尾从 `2026-04-27` 开始的记录
3. `docs/next_optimization_handoff.md`
4. `scripts/run_targeted_test_patch.py`
5. `scripts/run_surface_family_audit.py`

必要数据和基线：
- `data_raw/competition_data/test.csv`
- `data_raw/competition_data/train.csv`
- `data_raw/competition_data/laws_de.csv`
- `release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv`

## 建议开场指令
新对话可以直接说：

```text
请在 h:\cord\kaggle_llm_agentic_legal_ir_2026 继续 Kaggle legal IR 优化。
先回读 docs/current_progress_summary.md、docs/experiment_log.md、docs/next_optimization_handoff.md，
以 release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv
和 public score 0.28669 为唯一基线。继续找低风险、单行优先的 same-family article-institution 修复或 wrong-family 修复候选；
每次提交前必须先证明本地代理指标正向，并说明 train/test 证据。
```

## 提交纪律
- 不把公开榜当验证集。
- 不提交近重复微调。
- 不因单个 surface proxy 变好就提交；必须结合题面、训练集、法律库可用性和预测条数。
- 最新一轮已经证明：瓶颈后要回到 `train.csv/test.csv/laws_de.csv` 的硬证据，优先找单行 wrong-family 修复；干净剪枝仍可做，但不能把“条数下降”单独当作提交理由。

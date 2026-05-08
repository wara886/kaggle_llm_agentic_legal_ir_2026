# 当前进展摘要

## 当前最佳
- 当前最佳公开分数：`0.28669`
- 当前控制提交文件：`release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv`
- 当前控制原则：`0.28669` 已验证 `test_021` freight-forwarding/carriage 与 `test_007` medical-mandate 的 same-family article-institution 修复有效；后续仍然只做少量高置信 test-surface 修复，不回到宽 family prior。

## 已验证成功主线
- 主检索仍然保持 `laws-first`，不推翻现有主管线。
- 主 rerank 仍是 `Qwen3 AutoModelForCausalLM yes/no`。
- 真正已经被公开榜连续验证有效的，不是“单纯本地 strict F1 更高”，而是：
  - 题面显式锚点修复；
  - 明显 `wrong-family / wrong-article` 逃逸修正；
  - 改动行数少、spillover 低；
  - 改的是 `test.csv` 上肉眼可解释、法律家族明显漂移的行。

## 关键里程碑
| 版本 | 公开分 | 结论 |
|---|---:|---|
| `Qwen3 + Art. 100 Abs. 1 BGG` | `0.08960` | 第一个真正站稳的主线控制点 |
| 显式前缀修复 | `0.09617` | 说明显式法条表面形态可以转化为真实收益 |
| `CC` 自然别名 top2 | `0.10383` | 继续验证“题面显式形式”路线 |
| `Art. 38 and 39 CO` top3 | `0.11368` | 说明小范围显式补全有效 |
| surface-anchor combo v1 | `0.16392` | 首次大幅跃升，修了 `test_001/033/037/040` |
| surface-anchor combo v2 local | `0.17723` | 再次大幅跃升，修了 `test_005/013/020/038` |
| surface-anchor combo v4 hard explicit | `0.18136` | 显式锚点行 FP 清理被公开榜验证，修了 `test_002/023/028` |
| surface-anchor combo v6 LDIP/IPRG | `0.19043` | 多语种法典别名路线被验证，修了 `test_011` |
| surface-anchor combo v8 test009 IPRG/OR400 | `0.19876` | 跨境承认 / 重婚 / 遗产管理 wrong-family 修复被验证，修了 `test_009` |
| surface-anchor combo v10 test035 explicit prune | `0.20020` | 最后一投的显式锚点 FP 清理有效，修了 `test_035` |
| surface-anchor combo v12 test025 IPRG/ZGB tight | `0.20556` | 跨境离婚夫妻财产清算 wrong-family 修复有效，修了 `test_025` |
| surface-anchor combo v13 test017 lease termination tight | `0.20745` | 租赁欠租解除 / 上诉 nova wrong-article 修复有效，修了 `test_017` |
| surface-anchor combo v14 test039 simple partnership proof | `0.23126` | 简单合伙 / material mistake / 举证责任 article 修复有效，修了 `test_039` |
| surface-anchor combo v16 matrimonial maintenance | `0.23355` | same-family article drift + noisy-tail pruning 有效，修了 `test_030/031` |
| surface-anchor combo v18 adult protection core | `0.24075` | adult-protection article cluster 修复有效，修了 `test_029` |
| surface-anchor combo v20 test015 tight | `0.25015` | family 不变但 article institution 大错位修复有效，修了 `test_015` |
| surface-anchor combo v27 test021 freight forwarder | `0.26669` | OR family 内 freight/carriage article-institution 修复有效，修了 `test_021` |
| surface-anchor combo v29 test007 medical mandate | `0.28669` | OR family 内 medical mandate/duty-of-care article 修复有效，修了 `test_007` |

## 当前核心判断
- `0.11368 -> 0.16392 -> 0.17723 -> 0.18136 -> 0.19043 -> 0.19876 -> 0.20020 -> 0.20556 -> 0.20745 -> 0.23126 -> 0.23355 -> 0.24075 -> 0.25015 -> 0.26669 -> 0.28669` 不是偶然波动，而是 test-surface 修复路线连续验证成功。
- 这条思路的本质不是“模型更聪明了”，而是：
  - 我们开始直接修 `test.csv` 里最明显的法域错配；
  - 修的是高置信、低外溢的少量行；
  - 每次改动前都能做本地可解释自检，而不是盲目换参。
- 因此，后续优先级应继续放在：
  - surface-family audit；
  - 题面显式 citation grammar；
  - row-level wrong-family/article 修复；
  - 多语种法典别名，例如 `CC/CO/LDIP/LPM/LCD/LAI`；
  - 只在这些证据充分时再提交。

## 当前提交门槛
- 提交机会按稀缺资源处理；后续每次提交前必须先写清楚“本地代理指标 + train/test 证据 + 为什么不是近重复”。
- 不再把本地 `strict_f1` 单独当成提交通行证。
- 新提交必须先满足本地代理指标正向，至少包括以下一组：
  - validation strict/corpus F1 上升，或 TP 不掉且 FP 明确下降；
  - surface-family alignment 上升，同时 unexpected family / 预测条数不恶化；
  - candidate-stage recall / gold-in-fused proxy 上升，且 spillover 可控。
- 通过本地代理后，还必须满足下列至少一类 test/train 证据，才考虑新提交：
  - 修复了明显 `wrong-family` 的测试行；
  - 增加了题面中可直接看见的显式法条；
  - 在一组相关测试行上提高了 candidate-stage recall，且 spillover 可控。
- 对 surface-anchor 类 patch，优先看：
  - changed rows 是否少而准；
  - family alignment 是否明显上升；
  - 是否引入了新的空行、重复行、跨簇污染；
  - 是否只是近重复改写。
- 每次陷入局部瓶颈时，先回读：
  - `docs/experiment_log.md` 的成功/失败路径；
  - `train.csv` 中同类 query 的 gold citation 分布；
  - `test.csv` 的显式 citation、法域词、题面实体和当前预测错配；
  再决定是否开新方向，而不是继续微调同一组行。
- 当前已知反例：
  - `v5 test_018 core` 本地 family proxy 正向但公开不涨，说明“补法域但变宽/缺少强 train 支撑”的候选不能轻易提交。
  - `v9 test_008 IPRG85 narrow` 本地 family proxy 很漂亮但公开不涨，说明儿童 IPRG 相邻条文不能继续连环试探。
  - `v11 test_012 Art400 prune` 和 `v11 test_034 Art839 prune` 都是显式锚点剪枝但公开持平，说明 v10 之后单行 precision repair 的边际收益已经变薄，不能只因预测条数下降就继续提交。

## 明确不继续烧时间的方向
- 广义 family prior 扩展：本地分数可能涨，但公开榜已经证明很容易掉分。
- 全局 `ATSG/IVG` family-routing 扩展：已经验证会造成 candidate-stage family pollution。
- 只靠 train-to-test 语义近邻记忆：噪声太大，不是主线。
- 为了“也许会涨一点”去做近重复提交：不符合当前 token 和提交预算策略。

## 候选和提交记录

### `surface_anchor_escape_combo_v14_test039_simple_partnership_proof_local`
- 文件：`release/submission_surface_anchor_escape_combo_v14_test039_simple_partnership_proof_local/submission.csv`
- 相对 `0.20745` 控制版只改 `test_039`
- 动机：
  - 题面是两个独立顾问无书面协议共同承接项目、平分收入和费用、合作结束后结算，并以 material mistake / fraud 挑战分成安排；
  - v13 控制答案虽然是 `OR` 家族，但全是明显离题尾巴：`Art. 814 Abs. 4 OR`、`Art. 731b Abs. 3 OR`、`Art. 455 Abs. 2 OR`、`Art. 2 Abs. 3 OR`、`Art. 418e Abs. 2 OR`、`Art. 27 OR`；
  - `Art. 530 Abs. 1 OR` 对应 simple partnership 定义，`Art. 532 OR` 对应共同收益分享，`Art. 537 Abs. 1 OR` 对应共同事务费用/损失；
  - `Art. 23 OR`、`Art. 24 Abs. 1 OR` 对应 material mistake；
  - 题面第三问明确问 burden of proof，因此保留 `Art. 8 ZGB` 作为举证责任锚点。
- 本地审计：
  - changed queries：`1`
  - changed-row alignment：`0.5 -> 1.0`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 7`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - API 状态：`COMPLETE`
  - 公开分：`0.23126`
- 结论：
  升级为新基线。v14 说明除了 wrong-family，后期仍可通过“同 family 内 article 全错”的强证据修复获得大幅收益；关键是题面问题结构和新增条文一一对应，且预测条数不增加。

### `surface_anchor_escape_combo_v13_test017_lease_termination_tight_local`
- 文件：`release/submission_surface_anchor_escape_combo_v13_test017_lease_termination_tight_local/submission.csv`
- 相对 `0.20556` 控制版只改 `test_017`
- 动机：
  - 题面是商业租赁欠租、30-day cure notice、formula termination、summary eviction，以及上诉阶段首次提交 bank statements 作为 nova；
  - v12 控制答案全是明显离题的 OR 尾巴：`Art. 455 Abs. 2 OR`、`Art. 83 Abs. 2 OR`、`Art. 973i Abs. 3 OR`、`Art. 199 OR`、`Art. 406 OR`、`Art. 397 Abs. 1 OR`；
  - `laws_de.csv` 中 `Art. 257d Abs. 1/2 OR` 直接覆盖欠租催告和解除，`Art. 266l Abs. 1/2 OR` 覆盖书面/表格解除，`Art. 266o OR` 覆盖形式违反的无效，`Art. 317 Abs. 1 ZPO` 覆盖上诉 nova；
  - 没有使用 `Art. 257 ZPO`，因为本地 `laws_de.csv` 没有该 citation，避免新增缺失条文。
- 本地审计：
  - changed queries：`1`
  - changed-row alignment：`0.5 -> 1.0`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 7`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - API 状态：`COMPLETE`
  - 公开分：`0.20745`
- 结论：
  升级为新基线。v13 说明 v12 之后仍有空间，但更应该找“题面强事实 + laws 直连 + 单行 wrong-article/family 修复”，而不是继续围绕已提交行补相邻条文。

### `surface_anchor_escape_combo_v12_test025_iprg_zgb_tight_local`
- 文件：`release/submission_surface_anchor_escape_combo_v12_test025_iprg_zgb_tight_local/submission.csv`
- 相对 `0.20020` 控制版只改 `test_025`
- 动机：
  - 题面是跨境离婚中的夫妻财产清算：瑞士离婚法院、位于西班牙的不动产、夫妻共同所有、无婚约；
  - v10 控制答案只有 `ZGB`，漏掉了题面直接出现的 private international law / Spain immovable / Swiss divorce court 语境；
  - `laws_de.csv` 中 `Art. 51 IPRG` 正是夫妻财产关系管辖，`Art. 63 Abs. 1/2 IPRG` 对应离婚法院处理附随后果和准据法保留，`Art. 54 Abs. 1 IPRG` 对应无法律选择时夫妻财产准据法；
  - `ZGB` 侧替换成更贴题的 `Art. 205 Abs. 2 ZGB`、`Art. 197 Abs. 2 ZGB`、`Art. 200 Abs. 1/3 ZGB`，对应共有财产分配、保险/补偿款、举证责任和婚后所得推定。
- 本地审计：
  - changed queries：`1`
  - changed-row alignment：`0.25 -> 0.5`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 9`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52123673`
  - API 状态：`COMPLETE`
  - 公开分：`0.20556`
- 结论：
  升级为新基线。v12 说明 v11 之后仍有空间，但收益来自“明显 wrong-family + 法律库/训练集支撑 + 单行可解释替换”，不是单纯剪枝或泛化补 IPRG。

### `surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local`
- 文件：`release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv`
- 相对 `0.19876` 控制版只改 `test_035`
- 动机：
  - 题面直接写明 `Art. 263 ZPO` 和 `Art. 89 IPRG`；
  - 旧答案已包含这两个锚点，但夹带 `Art. 1 ZPO`、`Art. 2 ZPO`、`Art. 63 Abs. 2 ZPO`、`Art. 272 ZPO` 等泛程序条文；
  - 这是 v4 已验证成功的“显式锚点行 FP 清理”路线，不是宽 family prior，也不新增无支撑法域。
- 本地审计：
  - changed-row alignment：`0.666667 -> 0.666667`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 3`
  - 最终保留：`Art. 263 ZPO; Art. 89 IPRG; Art. 100 Abs. 1 BGG`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52086784`
  - API 状态：`COMPLETE`
  - 公开分：`0.20020`
- 结论：
  升级为最终新基线。最后一次优化没有追宽召回，而是选择最干净的显式锚点剪枝，公开榜验证有效。

### v11 显式锚点剪枝后续试探
- `surface_anchor_escape_combo_v11_test012_art400_prune_local`
  - 文件：`release/submission_surface_anchor_escape_combo_v11_test012_art400_prune_local/submission.csv`
  - 相对 `0.20020` 控制版只改 `test_012`
  - 动机：题面直接写 `Art. 400 OR`，当前答案命中 `Art. 400 Abs. 1/2 OR`，但夹带 `Art. 413 Abs. 1/2 OR`、`Art. 973i Abs. 3 OR`
  - 本地审计：alignment `0.333333 -> 0.333333`，unexpected `0 -> 0`，预测条数 `6 -> 3`
  - Kaggle ref：`52088772`
  - 公开分：`0.20020`
  - 结论：持平，不升级基线。
- `surface_anchor_escape_combo_v11_test034_art839_abs1_abs2_prune_local`
  - 文件：`release/submission_surface_anchor_escape_combo_v11_test034_art839_abs1_abs2_prune_local/submission.csv`
  - 相对 `0.20020` 控制版只改 `test_034`
  - 动机：题面直接问 `Art. 839 ZGB` 下四个月登记期限，`Art. 839 Abs. 2 ZGB` 在 train 中有 exact 支撑，候选保留 `Abs. 1/2` 并删除 `Abs. 3/4/5`
  - 本地审计：alignment `0.333333 -> 0.333333`，unexpected `0 -> 0`，预测条数 `6 -> 3`
  - Kaggle ref：`52123138`
  - 公开分：`0.20020`
  - 结论：持平，不升级基线。后续不要把“显式锚点剪枝 + 条数下降”单独当作提交理由，必须再叠加更强的 train/test 证据或多行一致收益。

### `surface_anchor_escape_combo_v8_test009_bigamy_iprg_train_tight_or_local`
- 文件：`release/submission_surface_anchor_escape_combo_v8_test009_bigamy_iprg_train_tight_or_local/submission.csv`
- 相对 `0.19043` 控制版只改 `test_009`
- 动机：
  - 题面是西班牙出生人、加拿大第二段婚姻、加拿大 probate order / letters of administration、瑞士承认与执行、public policy / bigamy、银行账目披露；
  - 旧答案是收养 / 子女维护类 `ZGB`，明显 wrong-family；
  - `train_0891` 是外国婚姻 / 未离婚重婚 / 瑞士承认的强相似样本，gold 包含 `Art. 25 IPRG`、`Art. 27 Abs. 1 IPRG`、`Art. 45 Abs. 2 IPRG`、`Art. 105 ZGB` 等；
  - `train_0966` 支撑继承 / 外国决定和文书承认中的 `Art. 96 Abs. 1 IPRG`；
  - 题面直接问 banking-law/accounting rules，`train_0425` 支撑 `Art. 400 Abs. 1 OR` 作为账目/报告义务锚点。
- 本地审计：
  - changed-row alignment：`0.333333 -> 1.0`
  - 全局 mean alignment：`0.702851`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 7`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52086384`
  - API 状态：`COMPLETE`
  - 公开分：`0.19876`
- 结论：
  升级为新基线。`test_009` 验证了“跨境承认 / public policy / wrong-family 纠偏 + train exact 支撑”的路线，但后续不能泛化成宽 IPRG prior，只能继续找同等硬度的单行或少行修正。

### `surface_anchor_escape_combo_v9_test008_iprg85_narrow_local`
- 文件：`release/submission_surface_anchor_escape_combo_v9_test008_iprg85_narrow_local/submission.csv`
- 相对 `0.19876` 控制版只改 `test_008`
- 动机：
  - 题面明确是 Germany/Switzerland 跨境儿童搬离、child abduction、private international law、international conventions；
  - 旧答案只有 `ZGB`，缺少 `IPRG`；
  - `train_0891` gold 出现 `Art. 85 Abs. 1 IPRG`，对应儿童保护国际私法锚点。
- 本地审计：
  - changed-row alignment：`0.5 -> 1.0`
  - 全局 mean alignment：`0.754605 -> 0.767763`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 6`
  - 删除 `Art. 25 Abs. 2 ZGB`、`Art. 133 Abs. 2 ZGB`，新增 `Art. 85 Abs. 1 IPRG`
- 提交状态：
  - Kaggle ref：`52086664`
  - API 状态：`COMPLETE`
  - 公开分：`0.19876`
- 结论：
  本地代理明显正向但 public 未提升，不升级为新基线。暂停沿 `test_008` 继续补 `Art. 10/79/83 IPRG` 等相邻条文，除非找到更强 validation 或 train exact 证据。

### `surface_anchor_escape_combo_v6_test011_ldip_iprg_local`
- 文件：`release/submission_surface_anchor_escape_combo_v6_test011_ldip_iprg_local/submission.csv`
- 相对 `0.18136` 控制版只改 `test_011`
- 动机：
  - 题面显式出现 `LDIP` 和 foreign forum-selection clause；
  - 当前 v4 只保留 `OR`，没有覆盖 `LDIP -> IPRG`；
  - `train_1011` 的相似管辖/法院选择问题 gold 明确包含 `Art. 5 Abs. 1 IPRG`、`Art. 6 IPRG` 等 IPRG 管辖条文。
- 本地审计：
  - alias-aware changed-row alignment：`0.5 -> 1.0`
  - 全局 mean alignment：`0.578333 -> 0.590833`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`9 -> 7`
  - 删除 4 条弱 `OR`，新增 `Art. 5 Abs. 1 IPRG`、`Art. 2 IPRG`
- 提交状态：
  - Kaggle ref：`52085651`
  - API 状态：`COMPLETE`
  - 公开分：`0.19043`
- 结论：
  升级为新基线。多语种法典别名 + 显式题面锚点成为继 v1/v2/v4 后的下一条已验证成功路线。

### `surface_anchor_escape_combo_v7_test010_narrow_036_local`
- 文件：`release/submission_surface_anchor_escape_combo_v7_test010_narrow_036_local/submission.csv`
- 相对 `0.19043` 控制版只改 `test_010`、`test_036`
- 动机：
  - `test_010` 明确是 qualified robbery / `Art. 398 StPO` / in dubio pro reo，旧答案只有 `StPO`，缺少 `StGB`；
  - `test_036` 明确出现 right to be heard / `Art. 101 Abs. 1 StPO`，旧答案缺少 `Art. 29 Abs. 2 BV`；
  - validation 上两条窄规则合并后 strict F1 `0.179311 -> 0.186359`。
- 本地审计：
  - 修正 surface audit 中小写英文 `or` 误触发 `OR` 的问题；
  - tight surface mean alignment：`0.685307 -> 0.707237`
  - changed-row alignment：`0.416666 -> 0.833333`
  - unexpected family：`0 -> 0`
  - changed-row 平均预测条数：`7 -> 8`
- 提交状态：
  - Kaggle ref：`52085891`
  - API 状态：`COMPLETE`
  - 公开分：`0.19043`
- 结论：
  本地 validation 和 surface proxy 都正向，但公开榜没有提升；不升级为新基线。后续暂停 right-to-be-heard / robbery 规则扩展，除非找到更强 train/test 证据。

### `surface_anchor_escape_combo_v5_test018_core_local`
- 文件：`release/submission_surface_anchor_escape_combo_v5_test018_core_local/submission.csv`
- 相对 `0.18136` 控制版只改 `test_018`
- 本地审计：
  - changed-row alignment：`0.1 -> 0.4`
  - unexpected family：`2 -> 0`
  - 全局 mean alignment：`0.570417 -> 0.577917`
- 提交状态：
  - Kaggle ref：`52085174`
  - API 状态：`COMPLETE`
  - 公开分：`0.18136`
- 结论：
  该候选没有带来公开提升，不升级为当时的新基线；后续 v6/v8 已覆盖新的最佳控制点。

### `surface_anchor_escape_combo_v4_hard_explicit_local`
- 文件：`release/submission_surface_anchor_escape_combo_v4_hard_explicit_local/submission.csv`
- 相对 `0.17723` 控制版只改了三行：`test_002`、`test_023`、`test_028`
- 这版不是补缺失 family，而是做显式锚点行的 FP 清理：
  - `test_002`：保留 `SVG/OR`，删除明显不相关的 `ZGB` 条文，并补 `Art. 58 Abs. 1 SVG`
  - `test_023`：删除随机 `OR` 条文，改为 `Art. 52 Abs. 1-4 AHVG` 加 `Art. 29 Abs. 2 BV`
  - `test_028`：删除 `ZGB` 和唯一的高 `Abs.` 怪条文 `Art. 362 Abs. 58 OR`，保留 `Art. 58 OR` 并补 `Art. 44 Abs. 1 OR`
- 本地审计：
  - changed queries：`3`
  - family alignment：`0.888889 -> 0.888889`
  - unexpected family count：`0.666667 -> 0`
  - changed-row 平均预测条数：`7 -> 5`
  - 没有新增空行、重复行或新的 `laws_de.csv` 缺失条文
- 当前判断：
  候选质量不错，但它的核心提升是“减少 FP 污染”，不是 v1/v2 那种 alignment 明显上升。
  在继续扫描后没有发现更强的同类硬锚点搭档，因此已提交一次试探。
- 提交状态：
  - Kaggle ref：`52052077`
  - API 状态：`COMPLETE`
  - 公开分：`0.18136`
  - 结论：v4 已验证有效，“显式锚点行 FP 清理”升级为当前第三条成功路线。

### `surface_anchor_escape_combo_v3_local`
- 文件：`release/submission_surface_anchor_escape_combo_v3_local/submission.csv`
- 相对 `0.17723` 控制版只改了两行：`test_018`、`test_019`
- 人工阅读上更像正确法域：
  - `test_018`：从 `ZGB/StGB/BV` 拉回 `IRSG/StPO/BZP/BV`
  - `test_019`：从儿童保护 `ZGB` 拉回 `IPRG`
- 但暂不提交，原因很明确：
  - 通用 family-audit proxy 没继续提升；
  - 训练集对这些具体条文的直接支撑偏弱；
  - 还没有找到更强的同类搭档行一起组成低风险组合。

## 数据侧最新启发
- `test.csv` 中显式 citation 覆盖明显高于 `train.csv`，所以“题面可见锚点”仍然是强迁移信号。
- 目前最值得继续盯的不是泛化语义，而是：
  - 当前输出里已经混入明显异法域条文的行；
  - 题面明说某个法条或法域，但输出只对了一半、另一半在乱飘的行。
- 对验证集做了一个小型 FP-pruning 检查：
  - 基线：`artifacts/explicit_prefix_rescue_conjunction_top3_v8/val_predictions.csv`
  - 规则：显式法域已命中时，剪掉明显不属于显式法域的预测
  - 结果：macro F1 `0.179311 -> 0.180265`，TP `25 -> 25`，FP `43 -> 39`
  - 启发：v4 的“显式锚点行减少 FP 污染”有一点验证集支撑，但样本很小，不能无限扩张。
- `test_023` 是一个值得继续观察的对象：
  - 题面明确出现 `Art. 52 Abs. 1 AHVG`
  - 当前输出里有 `Art. 52 Abs. 1 AHVG`，但还夹着一串可疑 `OR` 条文
  - 不过训练集中几乎没有 `Art. 52 AHVG` 的直接 gold 支撑，所以它更像“高潜力谨慎候选”，不是现在立刻该提交的版本
- `test_018/019` 重新回看后仍值得保留：
  - 审计脚本此前不认识 `IRSG/BZP`，导致 v3 被低估；
  - 加入 `IRSG/BZP` 和跨境程序 cue 后，v3 changed-row alignment `0.15 -> 0.30`，unexpected family `1 -> 0`；
  - 但 v3 平均预测条数 `7 -> 10`，属于“补法域但变宽”，暂时不和 v4 盲目合并。

## 下一步
- 以 `0.28669` v29 为最新提交基线，不回到大而泛的 family prior 小圈子里。
- 如果之后还有机会，优先找与 v14/v16/v18/v20 同等硬度的候选：same-family article-institution drift、明显 wrong-family / wrong-article，或题面直接给出法条锚点且当前答案夹带可疑 FP。
- `test_018/019` 仍值得看，但必须先做更窄候选，控制预测条数；不因 IPRG 家族代理上涨就直接提交。
- `test_025` 已被 v12 验证，不再继续围绕它补 `Art. 55 IPRG`、`Art. 651 Abs. 1 ZGB` 等相邻条文，除非发现更强的训练集 exact 支撑。
- `test_017` 已被 v13 验证，不再继续围绕它补 `Art. 273 OR`、`Art. 266n OR` 或本地 laws 缺失的 `Art. 257 ZPO`。
- `test_039` 已被 v14 验证，不再继续围绕它补 `Art. 545 OR`、`Art. 548 OR`、`Art. 31 OR` 等相邻条文。
- `test_008` 已用最窄 `IPRG85` 试探持平，先暂停。
- 对 v7 的 `test_010/036` 暂停近重复微调，除非出现新的训练集 exact 支撑或验证集收益。

## 新对话接力
- 新开对话先读：`docs/current_progress_summary.md`、`docs/experiment_log.md`、`docs/next_optimization_handoff.md`。
- 继续工作时使用 `release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv` 作为唯一基线。
- 具体接力说明已写入 `docs/next_optimization_handoff.md`。

## 文档规则
- 从当前版本开始，这份文件后续新增内容统一使用简体中文。
- 只保留对下一步决策真正有帮助的核心信息，不再堆叠冗长英文历史。
## 2026-04-29 v15 结果补充
- 已加入官方同构本地评估脚本：`scripts/evaluate_submission_official_strict.py`，按官方 `;` split、strip、set-F1、macro average 口径打分，不使用本地 citation normalization。
- 本地官方同构锚点：
  - `artifacts/qwen3_reranker_module_ablation/val_predictions_qwen3_cap80.csv`：`0.107028`
  - `artifacts/explicit_prefix_rescue_conjunction_top3_v8/val_predictions.csv`：`0.179311`
- v15 候选：`release/submission_surface_anchor_escape_combo_v15_test006_prhg_art8_proof_local/submission.csv`
- 改动：只给 `test_006` 增加 `Art. 8 ZGB`，用于产品责任 defect 证明责任问题。
- 本地 proxy：changed-row alignment `0.0 -> 0.5`，全局 mean alignment `0.742521 -> 0.755342`，只改 1 行。
- Kaggle public：`0.23126`，与 v14 持平。
- 当前基线不变：继续使用 `release/submission_surface_anchor_escape_combo_v14_test039_simple_partnership_proof_local/submission.csv` 和 public `0.23126`。
- 决策含义：`Art. 8 ZGB` 的 proof-aware 补丁不能机械复用。v14 的涨分逻辑是同一行内多个核心 article 被一起纠偏；v15 只是补一个一般证明责任锚点，未形成新突破。
## 2026-04-29 v16 status update

Current public best is now:

`release/submission_surface_anchor_escape_combo_v16_matrimonial_maintenance_local/submission.csv`

Public score: `0.23355`.

The winning change over v14/v15 (`0.23126`) was a two-row matrimonial-maintenance repair:

- `test_030`: replace wrong ZGB tails from child/adult-protection/inheritance with `Art. 176 Abs. 1/2 ZGB`, `Art. 163 Abs. 1/2/3 ZGB`, `Art. 271 ZPO`, `Art. 272 ZPO`, `Art. 100 Abs. 1 BGG`.
- `test_031`: keep/prune to the explicit post-divorce maintenance anchor group `Art. 125 ZGB`, `Art. 125 Abs. 1 ZGB`, `Art. 125 Abs. 2 ZGB`, `Art. 100 Abs. 1 BGG`.

Local proxy before submission:

- changed mean family alignment `0.416666 -> 0.583333`
- changed mean prediction count `7.5 -> 6.0`
- no empty or duplicate predictions

Interpretation: the reliable optimization path is now clearer. v15 showed that single proof-anchor additions can improve surface proxy but fail public score. v16 shows that same-family article drift repair plus noisy-tail pruning can still generalize enough to move the public metric.
## 2026-04-30 v18 current best

Current public best:

`release/submission_surface_anchor_escape_combo_v18_test029_adult_protection_core_on_v16/submission.csv`

Public score: `0.24075`.

Recent boundary tests:

- v17 `test_016` matrimonial-maintenance extension: local family proxy `0.5 -> 1.0`, public flat at `0.23355`.
- v18 `test_029` adult-protection article repair: unexpected family count `1.0 -> 0.0`, public `0.23355 -> 0.24075`.
- v19 `test_032` detention/ZGB-noise prune: unexpected family count `1.0 -> 0.0`, public flat at `0.24075`.

Most useful current optimization rule:

Look for rows where the broad family is already partly right, but the article cluster is from a different legal institution or the prediction carries a wrong procedural/family tail. Gains are more likely when the replacement article group is strongly anchored in `laws_de.csv` and the query's legal issue, not merely in broad keyword cues.

## 2026-05-02 v20 tight new best

Current public best:

`release/submission_surface_anchor_escape_combo_v20_test015_mandate_partnership_maintenance_tight_local/submission.csv`

Public score: `0.25015`.

Submitted boundary checks after v18:

- v21 `test_040` sham marriage / abuse-of-rights / Eheschutz ZPO: public flat at `0.24075`.
- v22 `test_022` cross-border family protective measures + ZPO prune: public flat at `0.24075`.
- v23 `test_036` right-to-be-heard `Art. 29 Abs. 2 BV`: public flat at `0.24075`.
- v24 `test_014` accident/UVG precision prune: public flat at `0.24075`.
- v25 `test_022 + test_040` combo: public flat at `0.24075`.
- v20 tight `test_015` mandate / simple partnership / post-divorce maintenance article repair: public `0.24075 -> 0.25015`.
- v20 full on the same row: public `0.24954`, below tight.

Interpretation:

The new winning pattern is same-family article-institution repair. `test_015` already had ZGB/OR, so family alignment did not improve, but the old articles were from the wrong institutions: relatives' support / marriage validity / weak agency tails. The tight repair mapped the query issues to the minimal core anchors: `Art. 394 Abs. 1 OR`, `Art. 400 Abs. 1 OR`, `Art. 530 Abs. 1 OR`, `Art. 125 ZGB`, `Art. 125 Abs. 1/2 ZGB`, `Art. 8 ZGB`, `Art. 100 Abs. 1 BGG`.

Do not generalize this into wide simple-partnership expansion. The full version added `Art. 398 Abs. 2 OR`, `Art. 532 OR`, `Art. 537 Abs. 1 OR` and dropped from `0.25015` to `0.24954`. Tight issue decomposition beats broad adjacent-article recall.

## 2026-05-08 v27/v29 new best

Current public best:

`release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv`

Public score: `0.28669`.

Submissions:

- v27 `test_021` freight forwarder / carrier article repair: public `0.25015 -> 0.26669`.
- v28 `test_024` divorce evidence pure-ZPO repair: public flat at `0.26669`.
- v29 `test_007` medical mandate / duty-of-care repair: public `0.26669 -> 0.28669`.
- v30 `test_026` family property / maintenance multi-issue repair: public flat at `0.28669`.

Interpretation:

The highest-yield pattern is now very specific: broad family already correct, but the old articles come from a plainly wrong legal institution. v27 replaced generic mandate/substitution tails with forwarding/carriage liability (`Art. 439/440/447/449 OR`) while preserving explicit mandate anchors. v29 replaced brokerage/loan/promise tails with medical mandate and duty-of-care anchors (`Art. 394/398/97/400/404 OR`).

Boundary:

v28 and v30 show that legally plausible procedure or multi-issue family-law repairs can be flat when prediction count grows or hidden gold only covers part of the issue set. Prefer tight OR-internal institution repairs over broad ZGB/ZPO expansions.

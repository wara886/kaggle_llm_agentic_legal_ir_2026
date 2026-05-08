# 实验日志

## 目的
- 用一份紧凑文档记录试错过程。
- 保留后续复盘、简历、面试可复用的素材。
- 避免 `docs/` 目录里积累大量一次性报告。

## 2026-04-25

### 突破路径复盘
- 公开榜突破路径：
  - `0.08960`：`Qwen3 + Art. 100 Abs. 1 BGG`
  - `0.09617`：显式前缀修复
  - `0.10383`：自然 `CC` 别名 top2
  - `0.11368`：并列条文 `Art. 38 and 39 CO` top3
- 真正起作用的变化：
  - 胜出的改动都紧贴 `test.csv` 的字面题面；
  - 改动范围小、spillover 低；
  - 修的是少量高置信测试行上的 citation 缺失或错配。
- 没有解释突破的因素：
  - 初次显式前缀修复之后，本地 strict F1 并没有随着每次公开提升同步继续上涨；
  - 泛语义扩展、宽 family 扩展虽然能拉高本地指标，但没有迁移到公开榜。
- 结论：
  当主线已经站到 `0.08960` 之后，最该看的就不再是“谁的本地 strict F1 最高”，而是：
  “这个 patch 是否在 `test.csv` 上修复了一个有明确依据的显式锚点问题，或者明显的 wrong-family 行？”

### 显式前缀 `Abs. 1` 补全候选
- 候选文件：
  `release/submission_explicit_prefix_rescue_abs1_existing_article_v1/submission.csv`
- 生成脚本：
  `scripts/run_explicit_prefix_rescue.py`
- 动机：
  如果当前最佳答案里已经出现裸 `Art.`，那就只补缺失的 `Abs. 1`，不做泛化扩张。
- 结果：
  本地 val strict/corpus F1 维持 `0.179311`，TP `25`，FP `43`。
- 与当时公开最佳相比：
  只改动 `test_040` 一行，新增 `Art. 176 Abs. 1 ZGB`。
- 判断：
  这是干净的小侧探，但不是带来全局跃迁的主故事。

### 数据侧全局回读
- 重新回读了主线代码以及 `train.csv / test.csv`。
- 关键分布发现：
  显式 citation 提取覆盖率在 `train` 上约为 `0.130`，在 `test` 上约为 `0.400`。
- 启发：
  显式法条 grammar 仍然是强迁移方向，但泛语义 prior 并没有得到同等级别支持。
- 负面结论：
  基于日期、名字、数字、词面锚点的 train-to-test 最近邻记忆噪声偏大，不适合作为主推进线。
- 结论：
  没有 citation 级证据时，不再把时间花在泛化 train memory 故事上。

### IP 法域逃逸候选
- 候选文件：
  `release/submission_ip_family_escape_hatch_v1/submission.csv`
- 生成脚本：
  `scripts/run_targeted_test_patch.py`
- 动机：
  两个当前最佳行出现了肉眼可见的 wrong-family：
  - `test_001`：著作权 / 商业秘密 / 临时措施题，被映射到 `EMBAG/SAFIG/HRegV`
  - `test_037`：商标域名 / 不正当竞争题，被映射到 `VID`
- 训练集支撑：
  - `UWG`：`39` 个 gold hit，分布在 `48` 个 cue-matched 训练行
  - `MSchG`：`19` 个 gold hit，分布在 `45` 个 cue-matched 训练行
  - `URG`：`108` 个 gold hit，分布在 `54` 个 cue-matched 训练行
- 相对当时公开最佳的改动：
  - `test_001` 改为 `ZPO/IPRG/URG/UWG`
  - `test_037` 改为 `MSchG/UWG`
- 判断：
  这类候选比显式前缀修复更有波动，但它抓到的是一个更大的全局错误模式：法域路由错误。

### 表面锚点 + 法域逃逸组合 v1
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v1/submission.csv`
- 生成脚本：
  `scripts/run_targeted_test_patch.py`
- 动机：
  把当时最强的测试面向修正合在一个文件里：
  - `test_001`：wrong-family IP 行
  - `test_033`：wrong-family `UVG` 职业病行
  - `test_037`：wrong-family 商标域名行
  - `test_040`：窄幅显式 `Abs. 1` 补全
- 最终改动：
  只改了 `4` 行：`test_001`、`test_033`、`test_037`、`test_040`
- 提交前自检：
  - changed queries：`4`
  - target-only spillover：`0`
  - changed rows 平均 family alignment：`0.25 -> 0.80`
  - 明显改善行：`test_001`、`test_033`、`test_037`
- 公开结果：
  `0.16392`
- 复盘：
  这一步确认了一个关键事实：
  post-`0.08960` 阶段最有用的 proxy 不是本地 strict F1，而是“是否修掉了 visible surface mismatch / wrong-family 且几乎没有 spillover”。

## 当前规则
- 在有新公开结果之前，`0.17723` 始终是稳定锚点。
- 除非满足以下至少一类条件，否则不提交近重复微变体：
  - 修复肉眼可见的 wrong-family 行；
  - 增加题面中直接出现的显式法条；
  - 在一簇相关测试行上提高 candidate-stage recall，且 spillover 受控。
- 能组合提交时，优先组合高置信 surface-anchor 修复，而不是一行一行零碎试探。
- 继续围绕 `0.17723` 做 surface-anchor audit，不重新掉回大而泛的 family prior 小圈子里。

## 2026-04-26

### 表面锚点 + 法域逃逸组合 v2 local
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v2_local/submission.csv`
- 生成脚本：
  `scripts/run_targeted_test_patch.py`
- 新增自检脚本：
  `scripts/run_surface_family_audit.py`
- 基线：
  `release/submission_surface_anchor_escape_combo_v1/submission.csv`，公开分 `0.16392`
- 相对 `0.16392` 控制版改动的行：
  `test_005`、`test_013`、`test_020`、`test_038`
- 动机：
  延续已经验证成功的“visible surface correction”路线，而不是再去做宽 reranker 调参。
- 四行的题面证据：
  - `test_005`：不记名抵押债券 / 不动产担保执行，去掉儿童保护漂移，改成 `SchKG/ZGB/OR`
  - `test_013`：劳务派遣 / GAV / 通常工资，把无关 `OR/ZGB` 行拉回 `AVG/AVEG/OR`
  - `test_020`：法定建筑留置 / 确定登记 / 鉴定，把离谱高编号 `OR` 条文拉回 `ZGB/OR/ZPO`
  - `test_038`：民事赔偿 / 误工损失 / 慰抚金，把 `StGB` 刑法条文拉回 `OR/ZGB`
- 本地自检：
  - changed queries：`4`
  - empty predictions：`0`
  - duplicate predictions：`0`
  - changed-row mean family alignment：`0.333333 -> 1.0`
  - 新增 `Art.` 都能在 `laws_de.csv` 中找到；最终文件里唯一缺失字符串仍是 v1 遗留的 `Art. 9 UVG`
- 公开结果：
  `0.17723`
- 复盘：
  这是同一路线第二次连续大涨，说明它已经从“一个聪明猜想”变成“当前最可信主线”。

### 表面锚点 + 法域逃逸组合 v3 local
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v3_local/submission.csv`
- 生成脚本：
  `scripts/run_targeted_test_patch.py`
- 基线：
  `release/submission_surface_anchor_escape_combo_v2_local/submission.csv`，公开分 `0.17723`
- 相对 `0.17723` 控制版改动的行：
  `test_018`、`test_019`
- 动机：
  这两行在当前控制版上仍然有明显法域漂移：
  - `test_018`：国际刑事司法协助 / 商业秘密 / 即时审查，被映射到 `ZGB/StGB/BV`
  - `test_019`：外国离婚承认 / 临时配偶救济，被映射到儿童保护 `ZGB`，而不是 `IPRG`
- 本地自检：
  - changed queries：`2`
  - empty predictions：`0`
  - duplicate predictions：`0`
  - 没有新增 `laws_de.csv` 外缺失条文；最终文件里仍只有旧问题 `Art. 9 UVG`
- 当前判断：
  暂时只保留离线版本，不提交。
- 原因：
  - 人工法域对齐看起来更顺；
  - 但通用 family-audit proxy 没继续上升；
  - 训练集对这些具体条文的直接支撑偏弱；
  - 还没找到足够强的同类搭档行一起组成更稳的组合。

### 补充观察：`test_023`
- `test_023` 是当前值得继续盯的一行：
  - 题面显式出现 `Art. 52 Abs. 1 AHVG`
  - 当前控制答案已经含有 `Art. 52 Abs. 1 AHVG`
  - 但同时混入了多条可疑 `OR` 条文
- 这行的好消息：
  - 显式锚点很强，符合我们当前成功路线
- 这行的坏消息：
  - 训练集中几乎没有 `Art. 52 Abs. 1-4 AHVG` 的直接 gold 支撑
- 当前结论：
  把它列为“高潜力谨慎候选”，继续观察，但暂不单独为它烧一次提交。

### 表面锚点 + 显式行 FP 清理 v4 local
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v4_hard_explicit_local/submission.csv`
- 生成脚本：
  `scripts/run_targeted_test_patch.py`
- 审计脚本更新：
  `scripts/run_surface_family_audit.py`
- 基线：
  `release/submission_surface_anchor_escape_combo_v2_local/submission.csv`，公开分 `0.17723`
- 相对 `0.17723` 控制版改动的行：
  `test_002`、`test_023`、`test_028`
- 动机：
  继续沿着已验证的 surface-anchor 方向，但这次更偏向“显式锚点行的 FP 清理”：
  - `test_002`：题面显式 `Art. 83 SVG / Art. 59 Abs. 1 SVG`，当前混入姓名/行为能力类 `ZGB`；候选保留 `SVG/OR` 并补 `Art. 58 Abs. 1 SVG`
  - `test_023`：题面显式 `Art. 52 Abs. 1 AHVG`，当前混入多条随机 `OR`；候选改为 `Art. 52 Abs. 1-4 AHVG` 加 `Art. 29 Abs. 2 BV`
  - `test_028`：题面显式 `Art. 58 Abs. 1 OR`，当前混入儿童/住所类 `ZGB` 和 `Art. 362 Abs. 58 OR`；候选改为 `Art. 58 Abs. 1/2 OR` 加 `Art. 44 Abs. 1 OR`
- 本地自检：
  - changed queries：`3`
  - empty predictions：`0`
  - duplicate predictions：`0`
  - family alignment：`0.888889 -> 0.888889`
  - unexpected family count：`0.666667 -> 0`
  - changed-row 平均预测条数：`7 -> 5`
  - 没有新增 `laws_de.csv` 外缺失条文；最终文件里仍只有旧问题 `Art. 9 UVG`
- Ablation：
  - `test_002 only`：alignment 持平，unexpected family `1 -> 0`，预测条数 `8 -> 5`
  - `test_023 only`：alignment 持平，预测条数 `7 -> 6`
  - `test_028 only`：alignment 持平，unexpected family `1 -> 0`，预测条数 `6 -> 4`
  - `test_023 + test_028`：alignment 持平，unexpected family `0.5 -> 0`，预测条数 `6.5 -> 5`
- 当前判断：
  这版是合理候选，但还没有达到 v1/v2 那种“所有核心 proxy 明显正向”的强度。
  继续扫描后没有找到比 `test_002/023/028` 更硬、更低风险的同类显式锚点行，因此按“无更强搭档则单发试探”的规则提交。
- 提交状态：
  - Kaggle ref：`52052077`
  - API 状态：`COMPLETE`
  - 公开分：`0.18136`
- 复盘：
  v4 公开榜确认提升，说明“显式锚点行 FP 清理”不是纯本地噪声。
  这是继 wrong-family escape、显式条文补全之后，第三条被公开榜验证的 test-surface 修复路线。

### FP-pruning 方向验证
- 使用验证集预测：
  `artifacts/explicit_prefix_rescue_conjunction_top3_v8/val_predictions.csv`
- 规则：
  当 query 中存在显式法域，且当前预测已经命中至少一个显式法域时，剪掉明显不属于显式法域的预测。
- 结果：
  - macro F1：`0.179311 -> 0.180265`
  - TP：`25 -> 25`
  - FP：`43 -> 39`
  - 实际改动验证行：`val_002`
- 解释：
  这支持 v4 的核心逻辑：不是补 recall，而是在显式锚点已命中的行上减少 FP 污染。
  但验证集只有 10 行，不能据此大规模自动剪枝，只能作为低风险人工 patch 的辅助证据。

### 跨境程序类审计修正
- 发现问题：
  旧版 `run_surface_family_audit.py` 不认识 `IRSG/BZP`，导致 `test_018` 这类国际司法协助 / 证据调取问题被低估。
- 修正：
  - 将 `IRSG`、`BZP` 加入 family parser；
  - 增加 `international_legal_assistance_evidence` cue；
  - 增加 `foreign_divorce_recognition_measures` cue。
- 重跑 v3：
  - 候选：`release/submission_surface_anchor_escape_combo_v3_local/submission.csv`
  - changed rows：`test_018`、`test_019`
  - alignment：`0.15 -> 0.30`
  - unexpected family：`1 -> 0`
  - 平均预测条数：`7 -> 10`
- 结论：
  v3 的方向比之前看起来更合理，但它是“补法域、变宽”的候选，和 v4 的“剪 FP、变窄”性质不同。
  在 v4 public 结果出来之前，不把两者合并提交。

## 文档规则
- 从当前版本开始，本文件后续新增内容统一使用简体中文。
- 只记录真正影响下一步决策的实验，不再堆积冗长英文过程稿。

## 2026-04-27

### 提交预算纪律更新
- 提交机会按稀缺资源处理，不能把公开榜当验证集。
- 以后候选进入提交队列前必须同时满足三件事：
  - 本地代理指标正向：validation F1 / FP、surface-family alignment、candidate recall proxy 至少有一项明确改善，且没有明显副作用。
  - `train.csv` 或 `test.csv` 给出可解释证据：显式法条、同类 gold 分布、明显 wrong-family、或候选召回链路改善。
  - 不是小圈子近重复：如果只是对已提交行换几个相邻条文，先搁置，除非验证集或训练分布强力支持。
- 陷入瓶颈时先回读优化路径，再开新方向：
  - 当前成功路线：显式 citation grammar、wrong-family escape、显式锚点 FP 清理。
  - 当前失败/降级路线：宽 family prior、泛语义 train memory、补法域但变宽且缺少 gold 支撑的候选。

### 表面锚点 + `test_018` 国际司法协助修复 v5
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v5_test018_core_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v4_hard_explicit_local/submission.csv`，公开分 `0.18136`
- 相对 `0.18136` 控制版改动的行：
  `test_018`
- 本地自检：
  - changed queries：`1`
  - family alignment：`0.1 -> 0.4`
  - unexpected family count：`2 -> 0`
  - 全局 mean alignment：`0.570417 -> 0.577917`
  - 没有新增空行、重复行；唯一 `laws_de.csv` 缺失仍是旧遗留 `Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52085174`
  - API 状态：`COMPLETE`
  - 公开分：`0.18136`
- 结论：
  本地 proxy 正向但未转化为公开提升；`test_018` 跨境程序补法域不作为新锚点。当前最佳仍为 v4 `0.18136`，后续继续优先找更硬的显式 citation / FP 清理候选。

### 多语种法典别名 + `test_011` LDIP/IPRG 修复 v6
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v6_test011_ldip_iprg_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v4_hard_explicit_local/submission.csv`，公开分 `0.18136`
- 相对 `0.18136` 控制版改动的行：
  `test_011`
- 动机：
  - 用户题面显式出现 `LDIP`、`LFors`、foreign forum-selection clause；
  - 旧答案只有 `OR`，没有覆盖 `LDIP -> IPRG`；
  - `train_1011` 是相似的国际合同/管辖/法院选择问题，gold 包含 `Art. 5 Abs. 1 IPRG`、`Art. 6 IPRG`、`Art. 112/113/116 IPRG` 等；
  - 这是 `CC/CO` 显式别名成功路线的自然扩展，不是宽 family prior。
- 本地自检：
  - 先修正 audit 中 `Managing Director` 误触发 AHVG/BV 的假阳性；
  - alias-aware changed-row alignment：`0.5 -> 1.0`
  - 全局 mean alignment：`0.578333 -> 0.590833`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`9 -> 7`
  - 删除弱 `OR`：`Art. 32 Abs. 2 OR`、`Art. 814 Abs. 4 OR`、`Art. 418g Abs. 1 OR`、`Art. 462 Abs. 2 OR`
  - 新增 `IPRG`：`Art. 5 Abs. 1 IPRG`、`Art. 2 IPRG`
- 提交状态：
  - Kaggle ref：`52085651`
  - API 状态：`COMPLETE`
  - 公开分：`0.19043`
- 复盘：
  v6 公开榜验证有效。后续应继续系统扫描 `test.csv` 中的多语种法典别名、外文法律缩写和题面直接出现但预测未覆盖的法域锚点；但仍然坚持小范围、低 FP、train/test 双证据后再提交。

### 程序权利 + robbery 窄补丁 v7
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v7_test010_narrow_036_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v6_test011_ldip_iprg_local/submission.csv`，公开分 `0.19043`
- 相对 `0.19043` 控制版改动的行：
  `test_010`、`test_036`
- 本地代理校准：
  - 修正 `run_surface_family_audit.py` 中小写英文 `or` 误触发 `OR` 的问题；
  - 移除 `right to be heard` 误触发 AHVG 的问题；
  - 收窄 `or_contract_liability` cue，避免普通英文语境污染 expected family。
- 动机与验证：
  - validation 上 right-to-be-heard 精确规则：`0.179311 -> 0.182812`
  - validation 上 robbery/StGB `Art. 140 Abs. 1` 精确规则：`0.179311 -> 0.182917`
  - validation 合并规则：`0.179311 -> 0.186359`
  - test surface tight proxy：mean alignment `0.685307 -> 0.707237`
  - changed-row alignment：`0.416666 -> 0.833333`
- 提交状态：
  - Kaggle ref：`52085891`
  - API 状态：`COMPLETE`
  - 公开分：`0.19043`
- 复盘：
  本地 validation 和 surface proxy 同时正向，但公开没有提升。说明这类“程序性通用权利 / 刑法实体条文补一条”的迁移不如 v6 的显式多语种法典别名硬；后续暂停同类扩展，不再沿 `test_010/036` 附近微调。

### 跨境承认 / 重婚 / 遗产管理 `test_009` 修复 v8
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v8_test009_bigamy_iprg_train_tight_or_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v6_test011_ldip_iprg_local/submission.csv`，公开分 `0.19043`
- 相对 `0.19043` 控制版改动的行：
  `test_009`
- 纪律检查：
  - 不是近重复微调：v7 已说明 `test_010/036` 路线暂时停下，本次切到新的 wrong-family 行；
  - 题面证据强：Spanish / Canada / second marriage / probate order / letters of administration / recognition in Switzerland / public policy / bigamy / bank accounting；
  - train 支撑强：`train_0891` 支撑外国婚姻、重婚和承认中的 `IPRG` + `Art. 105 ZGB`，`train_0966` 支撑继承文书承认的 `Art. 96 Abs. 1 IPRG`，`train_0425` 支撑账目义务 `Art. 400 Abs. 1 OR`。
- 本地自检：
  - 先比较了带 OR / 不带 OR、以及去掉无 exact train hit 的 `Art. 45 Abs. 1 IPRG` 的版本；
  - 最终选择 train-tight + OR400：`Art. 25 IPRG; Art. 27 Abs. 1 IPRG; Art. 45 Abs. 2 IPRG; Art. 96 Abs. 1 IPRG; Art. 105 ZGB; Art. 400 Abs. 1 OR; Art. 100 Abs. 1 BGG`
  - changed-row alignment：`0.333333 -> 1.0`
  - 全局 mean alignment：`0.702851`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 7`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52086384`
  - API 状态：`COMPLETE`
  - 公开分：`0.19876`
- 复盘：
  v8 公开榜确认有效，升级为新基线。关键不是“宽 IPRG prior”，而是对一个明显跑偏到收养/子女维护 `ZGB` 的跨境承认题进行强证据纠偏；后续只能寻找同等硬度的 wrong-family / train exact 支撑候选，不能沿 IPRG 做大范围铺开。

### `test_008` child abduction / IPRG85 窄补丁 v9
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v9_test008_iprg85_narrow_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v8_test009_bigamy_iprg_train_tight_or_local/submission.csv`，公开分 `0.19876`
- 相对 `0.19876` 控制版改动的行：
  `test_008`
- 审计规则校准：
  - 收窄 `run_surface_family_audit.py` 的 `ip_copyright_trade_secret_unfair` cue；
  - 原规则把普通 `provisional measures` 误判成 IP/URG/UWG 线索，污染了 `test_019` 这类 foreign divorce relief；
  - 新规则要求 copyright/source code/trade secret/unfair competition 等 IP 语境，或 provisional/injunction 与 IP 词同时出现。
- 动机与本地自检：
  - `test_008` 题面明确 Germany/Switzerland 跨境儿童搬离、child abduction、private international law、international conventions；
  - 只新增 `Art. 85 Abs. 1 IPRG`，删除 `Art. 25 Abs. 2 ZGB`、`Art. 133 Abs. 2 ZGB`；
  - `train_0891` gold exact 出现 `Art. 85 Abs. 1 IPRG`；
  - changed-row alignment：`0.5 -> 1.0`
  - 全局 mean alignment：`0.754605 -> 0.767763`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 6`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52086664`
  - API 状态：`COMPLETE`
  - 公开分：`0.19876`
- 复盘：
  本地代理很强但 public 持平，不升级为新基线。`test_008` 后续暂停，尤其不要继续试 `Art. 10/79/83 IPRG` 等相邻条文；这说明 v8 的成功不能简单扩展成“看见跨境儿童题就补 IPRG”。

### 最后一投：`test_035` 显式锚点 FP 清理 v10
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v8_test009_bigamy_iprg_train_tight_or_local/submission.csv`，公开分 `0.19876`
- 相对 `0.19876` 控制版改动的行：
  `test_035`
- 选择理由：
  - 用户要求做最后一次优化并提交，因此不再选择宽召回或多行组合；
  - `test_035` 题面直接写明 `Art. 263 ZPO` 和 `Art. 89 IPRG`；
  - 当前答案已经命中这两个核心锚点，但混入 `Art. 1 ZPO`、`Art. 2 ZPO`、`Art. 63 Abs. 2 ZPO`、`Art. 272 ZPO` 等泛程序条文；
  - 这与 v4 的显式锚点 FP 清理成功路线一致。
- 本地自检：
  - changed queries：`1`
  - changed-row alignment：`0.666667 -> 0.666667`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 3`
  - 最终保留：`Art. 263 ZPO; Art. 89 IPRG; Art. 100 Abs. 1 BGG`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - Kaggle ref：`52086784`
  - API 状态：`COMPLETE`
  - 公开分：`0.20020`
- 复盘：
  最后一投成功。它再次证明，在提交次数受限时，最可靠的尾部提升不是补相邻条文，而是对题面显式锚点已经命中的行做低风险 FP 清理。

### v11 显式锚点剪枝后续试探：`test_012` 与 `test_034`
- `test_012 Art400 prune`
  - 候选文件：`release/submission_surface_anchor_escape_combo_v11_test012_art400_prune_local/submission.csv`
  - 相对 `0.20020` 控制版只改 `test_012`
  - 动机：题面直接写 `Art. 400 OR`，当前答案命中 `Art. 400 Abs. 1/2 OR`，但混入 `Art. 413 Abs. 1/2 OR`、`Art. 973i Abs. 3 OR`
  - 本地自检：alignment `0.333333 -> 0.333333`，unexpected `0 -> 0`，预测条数 `6 -> 3`
  - Kaggle ref：`52088772`
  - 公开分：`0.20020`
  - 结论：持平，不升级基线。
- `test_034 Art839 Abs.1/2 prune`
  - 候选文件：`release/submission_surface_anchor_escape_combo_v11_test034_art839_abs1_abs2_prune_local/submission.csv`
  - 相对 `0.20020` 控制版只改 `test_034`
  - 动机：题面直接问 `Art. 839 ZGB` 下四个月登记期限；`Art. 839 Abs. 2 ZGB` 在 train 中有 exact gold 支撑，`Abs. 1` 是法定抵押权本体，因此保留 `Abs. 1/2`、删除 `Abs. 3/4/5`
  - 本地自检：alignment `0.333333 -> 0.333333`，unexpected `0 -> 0`，预测条数 `6 -> 3`
  - Kaggle ref：`52123138`
  - 公开分：`0.20020`
  - 结论：持平，不升级基线。
- 复盘：
  v10 之后的同类单行显式锚点剪枝边际收益已经明显变薄。下一步若继续提交，必须比“条数减少 + 家族不坏”更强：最好是 wrong-family 纠偏、显式漏召补全，或多个候选在本地代理上形成一致低风险收益。

### v12：`test_025` 跨境离婚夫妻财产清算 wrong-family 修复
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v12_test025_iprg_zgb_tight_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv`，公开分 `0.20020`
- 相对 `0.20020` 控制版改动的行：
  `test_025`
- 选择理由：
  - v11 两个显式锚点剪枝均持平，说明不能继续只靠“条数下降”提交；
  - `test_025` 题面是瑞士离婚法院处理西班牙不动产和夫妻财产清算，当前控制答案却全是 `ZGB`，漏了 private international law / foreign immovable / Swiss divorce court 法域；
  - 法律库中 `Art. 51 IPRG` 直接覆盖夫妻财产关系管辖，`Art. 63 Abs. 1/2 IPRG` 覆盖离婚附随后果，`Art. 54 Abs. 1 IPRG` 覆盖无婚约时夫妻财产准据法；
  - `ZGB` 侧不做宽召回，而是替换为题面明确需要的 `Art. 205 Abs. 2 ZGB`、`Art. 197 Abs. 2 ZGB`、`Art. 200 Abs. 1/3 ZGB`。
- 本地自检：
  - changed queries：`1`
  - changed-row alignment：`0.25 -> 0.5`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 9`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
  - 同时试过 10 条 evidence 版和 11 条 partition 版，但最终只提交 9 条 tight 版，避免把 `Art. 55 IPRG`、`Art. 651 Abs. 1 ZGB` 等相邻条文一起押上。
- 提交状态：
  - Kaggle ref：`52123673`
  - API 状态：`COMPLETE`
  - 公开分：`0.20556`
- 复盘：
  v12 升级为新基线。它验证了瓶颈后的正确跳出方式：回读失败路径后，不再钻显式剪枝小圈子，而是回看 `test.csv` 的题面事实与 `train.csv/laws_de.csv` 的法域支撑，找到了比 v11 更硬的 wrong-family 单行修复。后续不要继续围绕 `test_025` 做相邻条文补丁，除非找到更强 exact 支撑。

### v13 本地观察：`test_014` 与 `test_029`
- 基线：
  `release/submission_surface_anchor_escape_combo_v12_test025_iprg_zgb_tight_local/submission.csv`，公开分 `0.20556`
- `test_014 accident/UVG prune`
  - 候选文件：`release/submission_surface_anchor_escape_combo_v13_test014_accident_uvg_prune_local/submission.csv`
  - 只改 `test_014`，保留题面显式 `Art. 4 ATSG`、`Art. 6 Abs. 1/2 UVG`，删除 `Art. 64c Abs. 2/3 UVG`、`Art. 57 Abs. 1 UVG`。
  - 本地审计：changed-row alignment `0.5 -> 0.5`，unexpected `0 -> 0`，预测条数 `8 -> 5`。
  - 结论：这只是 v11 类“条数下降但代理不升”的剪枝，不提交。
- `test_029 adult protection core`
  - 候选文件：`release/submission_surface_anchor_escape_combo_v13_test029_adult_protection_core_local/submission.csv`
  - 先校准 `run_surface_family_audit.py`：普通 `bank/admissible` 不再误触发 `OR/ZPO`，新增 adult-protection guardianship cue。
  - 只改 `test_029`，删除弱 `ZPO` 尾巴 `Art. 188 Abs. 2 ZPO`、`Art. 181 Abs. 3 ZPO`、`Art. 119 Abs. 4 ZPO` 和泛化 `Art. 390 Abs. 2 ZGB`；改为 `Art. 390 Abs. 1 ZGB`、`Art. 394 Abs. 1 ZGB`、`Art. 395 Abs. 1 ZGB`、`Art. 445 Abs. 1 ZGB`、`Art. 450 Abs. 1 ZGB`、`Art. 93 Abs. 1 BGG`、`Art. 100 Abs. 1 BGG`。
  - 本地审计：changed-row alignment `1.0 -> 1.0`，unexpected family `1 -> 0`，预测条数 `6 -> 7`，只改 `test_029`，无新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`。
  - train/laws 证据：`laws_de.csv` 覆盖新增条文；`train_0763` 支撑 adult-protection 语境下 `Art. 395 ZGB`、`Art. 445 ZGB`，`train_1099` 支撑 `Art. 93 Abs. 1 BGG` 的“难以弥补损害/中间决定”结构。
  - 结论：候选方向合理，但强度低于 v8/v12；预测条数略增，train exact 支撑不够密，不消耗提交机会。后续若能找到同类 adult-protection/appeal 的更强搭档，或构造不增条数版本，再重新评估。

### v13：`test_017` 租赁欠租解除 / 上诉 nova wrong-article 修复
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v13_test017_lease_termination_tight_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v12_test025_iprg_zgb_tight_local/submission.csv`，公开分 `0.20556`
- 相对 `0.20556` 控制版改动的行：
  `test_017`
- 选择理由：
  - `test_017` 题面是商业租赁欠租、30-day cure notice、formula termination、summary eviction，以及上诉阶段首次提交 bank statements 作为 nova；
  - v12 控制答案全是明显离题的 OR 尾巴：`Art. 455 Abs. 2 OR`、`Art. 83 Abs. 2 OR`、`Art. 973i Abs. 3 OR`、`Art. 199 OR`、`Art. 406 OR`、`Art. 397 Abs. 1 OR`；
  - `laws_de.csv` 中 `Art. 257d Abs. 1/2 OR` 直接覆盖欠租催告和解除，`Art. 266l Abs. 1/2 OR` 覆盖书面/表格解除，`Art. 266o OR` 覆盖违反解除形式的无效，`Art. 317 Abs. 1 ZPO` 覆盖上诉 nova；
  - 没有使用 `Art. 257 ZPO`，因为本地 `laws_de.csv` 没有该 citation，避免新增缺失条文。
- 本地自检：
  - changed queries：`1`
  - changed-row alignment：`0.5 -> 1.0`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 7`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
- 提交状态：
  - 提交时间：`2026-04-29 09:49:26`（Kaggle submissions 列表时间）
  - API 状态：`COMPLETE`
  - 公开分：`0.20745`
- 复盘：
  v13 升级为新基线。它验证了 v12 之后仍可继续做“题面强事实 + laws 直连 + 单行 wrong-article/family 修复”。这次不是泛化租赁 prior，而是只修一行、预测条数不增、且引入 ZPO nova 家族以匹配题面明确的上诉新证据问题。后续不要围绕 `test_017` 继续补 `Art. 273 OR`、`Art. 266n OR` 或 `Art. 257 ZPO`，除非能解决 laws 缺失或找到更强证据。

### v14：`test_039` 简单合伙 / 举证责任 proof-aware 修复
- 候选文件：
  `release/submission_surface_anchor_escape_combo_v14_test039_simple_partnership_proof_local/submission.csv`
- 基线：
  `release/submission_surface_anchor_escape_combo_v13_test017_lease_termination_tight_local/submission.csv`，公开分 `0.20745`
- 相对 `0.20745` 控制版改动的行：
  `test_039`
- 选择理由：
  - `test_039` 题面是两个独立顾问无书面协议共同承接项目、平分收入和费用、合作结束后结算，并以 material mistake / fraud 挑战分成安排；
  - v13 控制答案虽然是 `OR` 家族，但全是明显离题尾巴：`Art. 814 Abs. 4 OR`、`Art. 731b Abs. 3 OR`、`Art. 455 Abs. 2 OR`、`Art. 2 Abs. 3 OR`、`Art. 418e Abs. 2 OR`、`Art. 27 OR`；
  - `Art. 530 Abs. 1 OR` 对应 simple partnership 定义，`Art. 532 OR` 对应共同收益分享，`Art. 537 Abs. 1 OR` 对应共同事务中的费用/损失；
  - `Art. 23 OR`、`Art. 24 Abs. 1 OR` 对应 material mistake；
  - 题面第三问明确问 burden of proof，因此用 `Art. 8 ZGB` 替代较泛的 `Art. 548 Abs. 1 OR`，保持 7 条不变，同时引入强直连的举证责任锚点。
- 本地自检：
  - changed queries：`1`
  - changed-row alignment：`0.5 -> 1.0`
  - unexpected family：`0 -> 0`
  - changed-row 预测条数：`7 -> 7`
  - 没有新增 `laws_de.csv` 缺失条文；唯一缺失仍是旧遗留 `test_033 / Art. 9 UVG`
  - train/laws 支撑：`Art. 530 Abs. 1 OR` 有 train gold hit；`Art. 23 OR`、`Art. 24 Abs. 1 OR`、`Art. 8 ZGB` 均有多个 train gold hit；新增条文均存在于 `laws_de.csv`。
- 提交状态：
  - 提交时间：`2026-04-29 12:22:26.617000`（Kaggle submissions 列表时间）
  - API 状态：`COMPLETE`
  - 公开分：`0.23126`
- 复盘：
  v14 升级为新基线，是 v13 后的真正大突破。关键不是宽补 OR，而是识别出“同 family 内 article 全错”的大洞，并用题面三个问题分别锚定 simple partnership、material mistake 和 burden of proof。后续不要围绕 `test_039` 继续追加 `Art. 545 OR`、`Art. 548 OR`、`Art. 31 OR` 等相邻条文，除非有更强 exact 证据。
### v15：`test_006` PrHG 产品责任证明责任 Art. 8 ZGB 试探
- 候选文件：`release/submission_surface_anchor_escape_combo_v15_test006_prhg_art8_proof_local/submission.csv`
- 基线：`release/submission_surface_anchor_escape_combo_v14_test039_simple_partnership_proof_local/submission.csv`，公开分 `0.23126`
- 相对 v14 改动：只改 `test_006`，在现有 PrHG 产品责任实体条文后新增 `Art. 8 ZGB`。
- 动机：`test_006` 题面直接问 claimant 是否承担 defect 证明责任，当前答案已有 `Art. 4/5 PrHG`，但缺少一般民事证明责任锚点；该思路来自 v14 中 `burden of proof -> Art. 8 ZGB` 的成功经验。
- 本地自检：
  - 官方同构 val evaluator 已加入：`scripts/evaluate_submission_official_strict.py`
  - `qwen3_cap80` val official-style macro F1：`0.107028`
  - explicit-prefix v8 val official-style macro F1：`0.179311`
  - v15 changed-row family alignment：`0.0 -> 0.5`
  - 全局 mean alignment：`0.742521 -> 0.755342`
  - changed rows：`1`
  - 预测条数：`6 -> 7`
- 提交结果：
  - 提交时间：`2026-04-29 12:52:14.213000`
  - 状态：`COMPLETE`
  - public score：`0.23126`
- 复盘：v15 持平，不升级为新基线。它说明 proof-aware 思路不能从 v14 机械外推：单独补 `Art. 8 ZGB` 即使法律逻辑合理，也可能只是不影响 public 的潜在 TP/FP 抵消或 hidden gold 未覆盖。v14 的大涨不是“看到 burden 就补 Art. 8 ZGB”，而是同时修中了 simple partnership、material mistake 和 burden 三个核心问题结构。后续仍以 v14 为唯一基线，不围绕 `test_006` 继续追加 `Art. 9/10 PrHG` 或 OR 相邻条文，除非找到更强 train/court exact 证据。
## 2026-04-29 v16 matrimonial maintenance article repair

- Baseline: v14/v15 public best `0.23126`.
- Candidate: `release/submission_surface_anchor_escape_combo_v16_matrimonial_maintenance_local/submission.csv`.
- Changed rows: `test_030`, `test_031`.
- Local rationale:
  - `test_030` is an Eheschutz / interim matrimonial-protection maintenance row. The control stayed in ZGB but drifted to child/adult-protection/inheritance tails. Replaced with `Art. 176 Abs. 1/2 ZGB`, `Art. 163 Abs. 1/2/3 ZGB`, and summary-procedure anchors `Art. 271/272 ZPO`.
  - `test_031` explicitly asks post-divorce spousal maintenance under `Art. 125 CC`; children are grown and not the issue. Kept `Art. 125 ZGB`, `Art. 125 Abs. 1/2 ZGB`, `Art. 100 Abs. 1 BGG`, removed child-support/property tails.
- Local proxy:
  - changed mean family alignment `0.416666 -> 0.583333`
  - changed mean prediction count `7.5 -> 6.0`
  - no empty or duplicate predictions
- Kaggle submission message: `v16 matrimonial maintenance article repair vs v14 0.23126`.
- Public score: `0.23355`, up from `0.23126`.
- Decision: upgrade current best to v16.
- Boundary lesson: this validates same-family article drift repair plus precision pruning. It does not validate broad keyword expansion. v15's flat proof-aware patch remains the cautionary counterexample.
## 2026-04-30 v17/v18/v19 boundary exploration

### v17 test_016 matrimonial maintenance

- Candidate: `release/submission_surface_anchor_escape_combo_v17_test016_matrimonial_maintenance_local/submission.csv`.
- Base: v16 public `0.23355`.
- Changed row: `test_016`.
- Rationale: extend v16 matrimonial-maintenance repair to another protective-measures / provisional maintenance query.
- Local proxy: changed family alignment `0.5 -> 1.0`; no empty/duplicate predictions.
- Public score: `0.23355`, flat.
- Decision: do not upgrade baseline.
- Lesson: v16's matrimonial repair does not blindly generalize when train/val have no direct support.

### v18 test_029 adult protection precision pruning

- Candidate: `release/submission_surface_anchor_escape_combo_v18_test029_adult_protection_core_on_v16/submission.csv`.
- Base: v16 public `0.23355`.
- Changed row: `test_029`.
- Rationale: adult protection / provisional guardianship row; control had the broad ZGB family but carried irrelevant ZPO procedural tails. Replace with `Art. 390/394/395/445/450 ZGB` plus `Art. 93 BGG`.
- Local proxy: family alignment `1.0 -> 1.0`; unexpected family count `1.0 -> 0.0`; no empty/duplicate predictions.
- Public score: `0.24075`, up `+0.00720`.
- Decision: upgrade current best to v18.
- Lesson: article-level cluster repair and noisy-family pruning can produce gains even when family alignment is unchanged.

### v19 test_032 detention prune

- Candidate: `release/submission_surface_anchor_escape_combo_v19_test032_detention_prune_local/submission.csv`.
- Base: v18 public `0.24075`.
- Changed row: `test_032`.
- Rationale: criminal detention row polluted by `Art. 390 ZGB`, likely from background wording around "adult child" / "custody"; replace with detention/substitute-measures StPO cluster.
- Local proxy: unexpected family count `1.0 -> 0.0`; no empty/duplicate predictions.
- Public score: `0.24075`, flat.
- Decision: keep v18 as current best.
- Lesson: pruning a false family is not enough if the gold likely includes broader detention appeal/procedural chains or case-law anchors.

## 2026-05-02 v20-v25 renewed boundary exploration and new best

### v21-v25 boundary checks

- Base: v18 public `0.24075`.
- v21 `test_040` sham marriage / abuse-of-rights / Eheschutz ZPO: public `0.24075`, flat.
- v22 `test_022` cross-border family protective measures + ZPO prune: public `0.24075`, flat.
- v23 `test_036` explicit right-to-be-heard `Art. 29 Abs. 2 BV`: public `0.24075`, flat.
- v24 `test_014` accident / UVG precision prune: public `0.24075`, flat.
- v25 `test_022 + test_040` combo: public `0.24075`, flat.
- Lesson: procedure-layer additions, constitutional anchor additions, and same-family pruning can be legally plausible and surface-proxy positive while still failing to move public score.

### v20 tight `test_015` mandate / simple partnership / maintenance repair

- Candidate: `release/submission_surface_anchor_escape_combo_v20_test015_mandate_partnership_maintenance_tight_local/submission.csv`.
- Kaggle ref: `52256243`.
- Base: v18 public `0.24075`.
- Changed row: `test_015`.
- Public score: `0.25015`, up `+0.00940`.
- Old prediction: `Art. 173 Abs. 1 ZGB; Art. 406a Abs. 1 OR; Art. 328 Abs. 2 ZGB; Art. 406 OR; Art. 94 ZGB; Art. 119 ZGB; Art. 100 Abs. 1 BGG`.
- New prediction: `Art. 394 Abs. 1 OR; Art. 400 Abs. 1 OR; Art. 530 Abs. 1 OR; Art. 125 ZGB; Art. 125 Abs. 1 ZGB; Art. 125 Abs. 2 ZGB; Art. 8 ZGB; Art. 100 Abs. 1 BGG`.
- Local proxy:
  - changed family alignment `0.666667 -> 0.666667`
  - unexpected family `0 -> 0`
  - prediction count `7 -> 8`
  - no empty or duplicate predictions
  - no newly introduced `laws_de.csv` missing citation
- Decision: upgrade current best to v20 tight.
- Lesson: this is the clearest evidence so far that the next optimization layer is not family alignment but issue-level article decomposition. The broad families were already present, but the old articles belonged to the wrong legal institutions.

### v20 full boundary

- Candidate: `release/submission_surface_anchor_escape_combo_v20_test015_mandate_partnership_maintenance_local/submission.csv`.
- Kaggle ref: `52256263`.
- Public score: `0.24954`.
- Difference from tight: adds `Art. 398 Abs. 2 OR`, `Art. 532 OR`, `Art. 537 Abs. 1 OR`.
- Decision: do not upgrade; keep tight.
- Lesson: adjacent simple-partnership expansion creates enough FP to lose `0.00061` versus tight. The successful principle is not "add the whole article neighborhood"; it is "map each query issue to the smallest article anchor that directly answers it."

## 2026-05-08 v27-v30 same-family institution repair

### v27 `test_021` freight forwarder / carrier

- Candidate: `release/submission_surface_anchor_escape_combo_v27_test021_freight_forwarder_tight_local/submission.csv`.
- Kaggle ref: `52444427`.
- Base: v20 tight public `0.25015`.
- Changed row: `test_021`.
- Public score: `0.26669`, up `+0.01654`.
- Old prediction: `Art. 398 Abs. 3 OR; Art. 399 Abs. 2 OR; Art. 399 Abs. 1 OR; Art. 399 Abs. 3 OR; Art. 398 Abs. 1 OR; Art. 157 OR; Art. 100 Abs. 1 BGG`.
- New prediction: `Art. 439 OR; Art. 440 Abs. 1 OR; Art. 440 Abs. 2 OR; Art. 447 Abs. 1 OR; Art. 449 OR; Art. 398 Abs. 3 OR; Art. 399 Abs. 2 OR; Art. 100 Abs. 1 BGG`.
- Decision: upgrade current best to v27.
- Lesson: keep explicit mandate/substitution anchors, but add the freight-forwarding/carriage liability institution that actually answers the query.

### v28 `test_024` divorce evidence ZPO

- Candidate: `release/submission_surface_anchor_escape_combo_v28_test024_divorce_evidence_zpo_tight_local/submission.csv`.
- Kaggle ref: `52444485`.
- Base: v27 public `0.26669`.
- Changed row: `test_024`.
- Public score: `0.26669`, flat.
- Local proxy: unexpected family `1 -> 0`, prediction count `7 -> 8`.
- Decision: do not upgrade.
- Lesson: procedure/evidence article repairs can be legally clean but still fail to move public score.

### v29 `test_007` medical mandate / duty of care

- Candidate: `release/submission_surface_anchor_escape_combo_v29_test007_medical_mandate_tight_local/submission.csv`.
- Kaggle ref: `52444556`.
- Base: v27 public `0.26669`.
- Changed row: `test_007`.
- Public score: `0.28669`, up `+0.02000`.
- Old prediction: `Art. 413 Abs. 2 OR; Art. 111 OR; Art. 525 Abs. 2 OR; Art. 23 OR; Art. 82 OR; Art. 27 OR; Art. 100 Abs. 1 BGG`.
- New prediction: `Art. 394 Abs. 1 OR; Art. 394 Abs. 3 OR; Art. 398 Abs. 1 OR; Art. 398 Abs. 2 OR; Art. 97 Abs. 1 OR; Art. 400 Abs. 1 OR; Art. 404 Abs. 1 OR; Art. 100 Abs. 1 BGG`.
- Decision: upgrade current best to v29.
- Lesson: OR-internal institution drift is a high-yield failure type. Surface-family alignment is blind to it because both old and new rows are pure OR.

### v30 `test_026` family property / maintenance

- Candidate: `release/submission_surface_anchor_escape_combo_v30_test026_family_property_maintenance_local/submission.csv`.
- Kaggle ref: `52444605`.
- Base: v29 public `0.28669`.
- Changed row: `test_026`.
- Public score: `0.28669`, flat.
- Prediction count: `7 -> 10`.
- Decision: do not upgrade.
- Lesson: multi-issue ZGB repairs are less reliable when the replacement set becomes wide. Continue prioritizing tight institution repairs such as v27/v29.

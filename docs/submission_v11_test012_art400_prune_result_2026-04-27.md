# `v11 test_012 Art. 400 OR` 提交结果

## 结论
- 提交文件：`release/submission_surface_anchor_escape_combo_v11_test012_art400_prune_local/submission.csv`
- Kaggle ref：`52088772`
- 提交说明：`surface anchor v11 test012 Art400 explicit prune vs 0.20020 2026-04-27`
- API 状态：`COMPLETE`
- public score：`0.20020`
- 与基线 `v10` 相比：持平，没有升级为新基线

## 本次改动
- 只改 `test_012`
- 旧答案：
  `Art. 400 Abs. 1 OR; Art. 400 Abs. 2 OR; Art. 413 Abs. 2 OR; Art. 413 Abs. 1 OR; Art. 973i Abs. 3 OR; Art. 100 Abs. 1 BGG`
- 新答案：
  `Art. 400 Abs. 1 OR; Art. 400 Abs. 2 OR; Art. 100 Abs. 1 BGG`
- 删除的尾巴：
  `Art. 413 Abs. 2 OR; Art. 413 Abs. 1 OR; Art. 973i Abs. 3 OR`

## 提交前证据
- test 题面证据：
  - `test_012` 题面直接写出 duty to render accounts `Art. 400 OR`
  - 当前控制答案已经命中 `Art. 400 Abs. 1/2 OR`
  - 候选只做显式锚点命中后的尾巴剪枝，不补新法域，不加相邻条文
- train 证据：
  - `train_0169` gold 包含 `Art. 400 Abs. 1 OR`
  - `train_0425` gold 包含 `Art. 400 OR; Art. 400 Abs. 1 OR`
  - 未找到被剪掉的 `Art. 413.* OR` / `Art. 973i Abs. 3 OR` 的同类 exact 支撑
- 本地代理：
  - changed queries：`1`
  - changed-row alignment：`0.333333 -> 0.333333`
  - unexpected family：`0 -> 0`
  - changed-row prediction count：`6 -> 3`
- 验证集同类旁证：
  - 对 `val_006` 做同类 OR 显式锚点尾巴剪枝后，strict F1：`0.179311 -> 0.183674`
  - TP：`25 -> 25`
  - FP：`43 -> 40`

## 复盘
- 这次候选满足“低风险、单行、显式锚点已命中后的 FP 清理”纪律，但 public 仍然没有超过 `v10`
- 说明：
  - `test_012` 这类 OR 家族内部尾巴剪枝可以作为合理候选保留
  - 但它的收益强度暂时不足以转化成公开榜增益
- 因此后续策略应保持不变：
  - 继续以 `release/submission_surface_anchor_escape_combo_v10_test035_explicit_anchor_prune_local/submission.csv` 和 public `0.20020` 为唯一基线
  - 暂不围绕 `test_012` 做近重复微调
  - 优先继续寻找像 `v4` / `v10` 那样更硬的显式锚点 FP 清理，或像 `v8` 那样更强的 wrong-family + train exact 支撑单行修复

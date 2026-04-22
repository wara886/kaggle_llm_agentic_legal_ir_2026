# Laws-First Next 3 Optimizations

## 1) Training query alignment to family/issue (HIGH)
- 为什么现在做：当前训练线主要短板是 query 视图与线上 laws-first 有效结构不一致。
- 预期收益：non-explicit strict/corpus F1 持续提升，且 FP 不上升（甚至下降）。
- 风险：过度结构化可能损伤跨领域泛化；需保持 raw 默认可回退。
- 影响文件：`scripts/train_laws_minilm_biencoder.py`
- 优先级：高

## 2) Hard negative quality: family-aware near-miss negatives (MEDIUM)
- 为什么现在做：loss/规模已对齐后，剩余收益主要取决于负样本“难度质量”。
- 预期收益：改善 dense embedding 的细粒度判别，减少 family-correct 但 rerank 失真。
- 风险：约束过强会造成负样本过窄、训练过拟合某些家族。
- 影响文件：`scripts/mine_laws_hard_negatives_minilm.py`
- 优先级：中

## 3) Laws final selection micro-calibration with stricter admission guard (MEDIUM-LOW)
- 为什么现在做：当前次级瓶颈是 reranked_too_low；需要更稳地把 fused 高价值 law 传导到 final。
- 预期收益：在 non-explicit 上小幅提分。
- 风险：若放宽过度（如盲目增大 rescue 数），会直接抬高 FP（已在 k=2 ablation 验证）。
- 影响文件：`scripts/run_silver_baseline_v0.py`（仅参数/阈值微调，不改分支）
- 优先级：中-低

# Generalization / Overfit Audit: Late-Stage FP Pruning

## 结论先行

不能保证当前 Kaggle v10 的所有后期 row-level patch 都具备完全泛化性。它们确实是 test-facing 的高置信 hotfix，应该诚实描述为 `error audit + precision guard + low-spillover patch`，而不是包装成模型自动学到的能力。

但通用思想本身有可迁移性：当 query 有显式法典/条款/领域锚点时，用领域 family guard 去清理明显 wrong-family FP，能在伪测试中稳定减少 FP；同时也会暴露 recall/TP 误删风险。这更像真实 RAG 生产里的 guardrail，需要证据门禁和保守策略，而不是纯模型训练。

## 审计设计

- `real_val_*`：在真实 val 预测上直接套通用 family prune，样本很小，只能看风险信号。
- `synthetic_*`：用 train/val gold 构造伪测试集，先把 gold 当作正确证据，再注入 wrong-family FP 和少量 same-family FP，模拟“正确答案里夹带错误证据”的后期场景。
- prune 规则只看 query 文本里的显式 family/alias，不看 hidden gold；没有显式锚点时不动。
- `aggressive_family_prune` 会剪掉所有显式 family 外的法规 citation，风险更高。
- `conservative_tail_family_prune` 只剪尾部 wrong-family，保留前 3 个高置信预测，更接近后期 v10 的“少改”策略。
- 这个实验验证的是通用 precision-guard 思路，不验证具体 `test_035` 这类 row patch 的泛化。

## 结果汇总

| case | policy | queries | guarded | macro F1 before | macro F1 after | precision before | precision after | recall before | recall after | FP before | FP after | removed FP | removed TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_gold_plus_wrong_family_noise_train | aggressive_family_prune | 1139 | 216 | 0.584566 | 0.618286 | 0.528831 | 0.566301 | 1.000000 | 0.975317 | 4151 | 3480 | 671 | 115 |
| synthetic_gold_plus_wrong_family_noise_train | conservative_tail_family_prune | 1139 | 216 | 0.584566 | 0.611433 | 0.528831 | 0.560059 | 1.000000 | 0.984761 | 4151 | 3604 | 547 | 71 |
| synthetic_gold_plus_wrong_family_noise_val | aggressive_family_prune | 10 | 8 | 0.908176 | 0.852328 | 0.862543 | 0.943478 | 1.000000 | 0.864542 | 40 | 13 | 27 | 34 |
| synthetic_gold_plus_wrong_family_noise_val | conservative_tail_family_prune | 10 | 8 | 0.908176 | 0.903570 | 0.862543 | 0.946058 | 1.000000 | 0.908367 | 40 | 13 | 27 | 23 |
| synthetic_gold_plus_wrong_family_noise_train_plus_val | aggressive_family_prune | 1149 | 224 | 0.587382 | 0.620323 | 0.539501 | 0.576811 | 1.000000 | 0.969654 | 4191 | 3493 | 698 | 149 |
| synthetic_gold_plus_wrong_family_noise_train_plus_val | conservative_tail_family_prune | 1149 | 224 | 0.587382 | 0.613975 | 0.539501 | 0.571090 | 1.000000 | 0.980855 | 4191 | 3617 | 574 | 94 |
| real_val_qwen3_cap80_predictions | aggressive_family_prune | 10 | 8 | 0.107028 | 0.107028 | 0.250000 | 0.254545 | 0.055777 | 0.055777 | 42 | 41 | 1 | 0 |
| real_val_qwen3_cap80_predictions | conservative_tail_family_prune | 10 | 8 | 0.107028 | 0.098793 | 0.250000 | 0.265306 | 0.055777 | 0.051793 | 42 | 36 | 6 | 1 |
| real_val_explicit_prefix_rescue_predictions | aggressive_family_prune | 10 | 8 | 0.179311 | 0.158738 | 0.367647 | 0.400000 | 0.099602 | 0.087649 | 43 | 33 | 10 | 3 |
| real_val_explicit_prefix_rescue_predictions | conservative_tail_family_prune | 10 | 8 | 0.179311 | 0.175405 | 0.367647 | 0.393443 | 0.099602 | 0.095618 | 43 | 37 | 6 | 1 |

## 怎么解读

- 如果只看 v10 里具体 `test_035` 的改动，它是定向 test patch，有过拟合风险，不能说成纯泛化模型能力。
- 如果抽象成 `显式锚点 -> family guard -> wrong-family FP prune`，这是一类可泛化 RAG guardrail，可以迁移到企业知识库、金融制度、医疗指南等专业 RAG。
- 伪测试结果能说明：在“已经召回到正确证据但夹带错误知识域”的场景下，少量精准 prune 有机会提高 precision 和 F1，但如果 family guard 过粗，也会误删 TP。
- 它不能说明：所有 unseen query 都会提升；也不能替代真正 held-out test 或线上 A/B。

## 面试里的诚实说法

后期 row-level patch 有 test-facing 风险，我不会把它包装成模型泛化能力。这个阶段展示的是另一种工程能力：在小样本、高风险 RAG 场景里，通过错误审计识别高置信 FP，用证据链和 guardrail 做低外溢修复。项目真正可泛化的部分是 hybrid retrieval、LLM reranker、citation/alias parser、family audit、residual audit 和提交前证据门禁。

## 产物

- JSON: `H:\cord\kaggle_llm_agentic_legal_ir_2026\artifacts\generalization_overfit_audit_2026_04_27\summary.json`
- CSV: `H:\cord\kaggle_llm_agentic_legal_ir_2026\artifacts\generalization_overfit_audit_2026_04_27\case_summary.csv`

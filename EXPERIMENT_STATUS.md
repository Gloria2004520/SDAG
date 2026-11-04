# 实验状态：当前有效清单

固定方案：**Qwen3-VL-8B（纯文本）+ 已有 BM25 数据库/检索结果**。
每个数据集沿用1000题。k包含污染文档；不再比较检索器、不下载模型、不重新检索。

| 项目 | 已完成 | 剩余 |
| --- | --- | --- |
| 实验1：三数据集、k5/10、单污染、CARG/SDAG | **6/6配置完成并上传** | 直接复用 |
| 实验4：NQ1000、k5、Random/Near/Far | **3/3策略有完整输出并上传** | 复用历史输入版本，不重跑 |
| 实验5：污染数量、CARG/SDAG | **15/15配置完成并上传** | 此项已完成，直接复用 |
| 实验5：RAGDefender、D&A及各自SDAG组合 | **10/30配置完成并上传** | 正在运行 `joint_da_nq_a1_k10`；见[适配差异](BASELINE_ADAPTATIONS.md) |
| 检索器对比、原论文三生成器 | **已取消/允许替代** | 不再作为完成要求 |

## 实验5：CARG/SDAG的15个配置

| 数据集 | (1,5) | (1,10) | (2,5) | (2,10) | (3,10) |
| --- | --- | --- | --- | --- | --- |
| nq | 已完成 | 已完成 | 已完成（复用旧双对抗） | 已完成 | 已完成 |
| hotpotQA | 已完成 | 已完成 | 已完成 | 已完成 | 已完成 |
| triviaQA | 已完成 | 已完成 | 已完成 | 已完成 | 已完成 |

## 已完成防御配置

每个配置含两种mask、各1000题；数值为ACC/ASR百分比。复用指提示完全相同的原答案，非重复推理。

| 过滤方法 | 数据集 | (污染数,总篇数) | 普通mask | +SDAG | 复用/2000 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| joint_da适配 | hotpotQA | (1,5) | 32.2/5.3 | 27.2/4.4 | 110 | [结果](results/qwen3vl8b_joint_da_hotpotQA_a1_k5/verified_report.json) |
| joint_da适配 | hotpotQA | (1,10) | 36.6/5.9 | 27.8/4.9 | 110 | [结果](results/qwen3vl8b_joint_da_hotpotQA_a1_k10/verified_report.json) |
| joint_da适配 | hotpotQA | (2,5) | 28.9/15.8 | 24.6/14.9 | 258 | [结果](results/qwen3vl8b_joint_da_hotpotQA_a2_k5/verified_report.json) |
| joint_da适配 | hotpotQA | (2,10) | 33.8/11.1 | 26.5/9.4 | 176 | [结果](results/qwen3vl8b_joint_da_hotpotQA_a2_k10/verified_report.json) |
| joint_da适配 | hotpotQA | (3,10) | 32.2/15.5 | 26.6/14.2 | 254 | [结果](results/qwen3vl8b_joint_da_hotpotQA_a3_k10/verified_report.json) |
| ragdefender适配 | hotpotQA | (1,5) | 7.5/63.7 | 8.6/52.4 | 0 | [结果](results/qwen3vl8b_ragdefender_hotpotQA_a1_k5/verified_report.json) |
| ragdefender适配 | hotpotQA | (1,10) | 6.7/64.1 | 9.1/43.5 | 0 | [结果](results/qwen3vl8b_ragdefender_hotpotQA_a1_k10/verified_report.json) |
| ragdefender适配 | hotpotQA | (2,5) | 17.9/35.2 | 16.1/30.8 | 0 | [结果](results/qwen3vl8b_ragdefender_hotpotQA_a2_k5/verified_report.json) |
| ragdefender适配 | hotpotQA | (2,10) | 9.3/56.3 | 7.6/47.5 | 0 | [结果](results/qwen3vl8b_ragdefender_hotpotQA_a2_k10/verified_report.json) |
| ragdefender适配 | hotpotQA | (3,10) | 22.1/25.9 | 18.2/23.3 | 0 | [结果](results/qwen3vl8b_ragdefender_hotpotQA_a3_k10/verified_report.json) |

## 已完成配置的结果

数值为百分比；每格ACC/ASR。原始输出、配置、hash及配对统计在链接目录。

| 数据集 | (污染数,总篇数) | CARG ACC/ASR | SDAG ACC/ASR | 原始结果 |
| --- | --- | ---: | ---: | --- |
| nq | (1,5) | 17.4/43.5 | 18.6/27.0 | [结果](results/qwen3vl8b_bm25_nq_k5/verified_report.json) |
| nq | (1,10) | 22.8/39.5 | 26.3/18.7 | [结果](results/qwen3vl8b_bm25_nq_k10/verified_report.json) |
| nq | (2,5) | 12.8/50.5 | 11.2/46.4 | [结果](results/qwen3vl8b_nq_two_attacks_k5/verified_report.json) |
| nq | (2,10) | 17.2/45.6 | 17.8/34.5 | [结果](results/qwen3vl8b_bm25_nq_a2_k10/verified_report.json) |
| nq | (3,10) | 14.5/50.2 | 11.8/47.8 | [结果](results/qwen3vl8b_bm25_nq_a3_k10/verified_report.json) |
| hotpotQA | (1,5) | 11.8/67.5 | 16.7/45.6 | [结果](results/qwen3vl8b_bm25_hotpotQA_k5/verified_report.json) |
| hotpotQA | (1,10) | 14.7/63.6 | 20.3/35.0 | [结果](results/qwen3vl8b_bm25_hotpotQA_k10/verified_report.json) |
| hotpotQA | (2,5) | 5.7/76.5 | 5.4/71.3 | [结果](results/qwen3vl8b_bm25_hotpotQA_a2_k5/verified_report.json) |
| hotpotQA | (2,10) | 9.2/72.7 | 11.1/58.3 | [结果](results/qwen3vl8b_bm25_hotpotQA_a2_k10/verified_report.json) |
| hotpotQA | (3,10) | 6.4/76.7 | 7.9/70.8 | [结果](results/qwen3vl8b_bm25_hotpotQA_a3_k10/verified_report.json) |
| triviaQA | (1,5) | 46.9/46.6 | 54.5/25.4 | [结果](results/qwen3vl8b_bm25_triviaQA_k5/verified_report.json) |
| triviaQA | (1,10) | 56.3/37.6 | 64.6/14.0 | [结果](results/qwen3vl8b_bm25_triviaQA_k10/verified_report.json) |
| triviaQA | (2,5) | 34.2/60.4 | 29.9/58.2 | [结果](results/qwen3vl8b_bm25_triviaQA_a2_k5/verified_report.json) |
| triviaQA | (2,10) | 41.8/52.2 | 47.7/38.4 | [结果](results/qwen3vl8b_bm25_triviaQA_a2_k10/verified_report.json) |
| triviaQA | (3,10) | 33.8/59.7 | 31.1/57.7 | [结果](results/qwen3vl8b_bm25_triviaQA_a3_k10/verified_report.json) |

## 去重与范围说明

- NQ (2,5)正式使用之前的双对抗结果和输入。后来新跑的2000条为非计划重复，只保留审计，**不增加实验覆盖**。
- 已完成的单污染、策略、双对抗直接复用；只补缺失配置/防御。不同随机实现、目录名或元数据不算新配置。
- 各防御配对使用相同原始文档及位置；过滤未改变问题、文档顺序和mask时，复用相同模型/设置的原答案，记录复用证据。
- 历史Near/Far用E5选择污染候选，不是E5全库检索；保留历史版本，并披露冻结输入后来与BM25列表核对一致的证据。
- 输入准备、过滤缓存和冒烟测试不算问答结果。仅在1000题原始输出、hash、ACC/ASR和配对统计全部验证上传后标完成。
- 先补完剩余CARG/SDAG，再执行排队的防御问答，最后汇总跨方法配对统计。整体尚未完成。

历史过程见[FUYAO_REPRODUCTION.md](FUYAO_REPRODUCTION.md)，以本清单为当前范围。

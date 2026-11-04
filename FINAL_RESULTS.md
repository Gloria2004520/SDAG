# 最终结果：修订范围近似复现完成

固定 Qwen3-VL-8B-Instruct（纯文本）+ 已有 BM25；HotpotQA、TriviaQA、NQ各1000题。
这是用户修订范围的近似复现，不是原论文全部生成器/检索器或GPT基线的精确复现。

| 实验 | 完成范围 |
| --- | --- |
| 单污染 | 三数据集 × k5/10 × CARG/SDAG，6/6配置 |
| 污染候选策略 | NQ k5，Random/Near/Far，3/3历史配置复用 |
| 污染数量 | 三数据集 × (1,5)/(1,10)/(2,5)/(2,10)/(3,10)，CARG/SDAG 15/15配置 |
| 防御 | 同15配置，D&A-inspired/RAGDefender及各自SDAG组合，30/30配置对 |

每个配置的每种方法均有1000题。六方法比较含90000条评估记录，底层只有3000道不同题，不能视为90000道独立题。
防御部分60000条中2112条复用完全相同提示的原答案，57888条为新增推理。
历史归档总计102000条评估记录，其中包含早期实验及2000条非计划NQ重复；这些不增加配置覆盖。

## 六方法结果

下表为15个配置的等权平均，百分数；各配置反复使用相同问题，平均值是描述统计。
逐配置结果及配对双侧t检验见[metrics.csv](results/defense_comparison_final/metrics.csv)和[comparison.json](results/defense_comparison_final/comparison.json)。
ACC使用项目的归一化答案匹配；同时保留严格EM。ASR越低越好。配对检验未做多重比较校正。

| 方法 | ACC % | ASR % | 严格EM % |
| --- | ---: | ---: | ---: |
| CARG | 23.03 | 56.19 | 17.67 |
| SDAG | 24.99 | 43.27 | 19.65 |
| D&A-inspired | 48.64 | 7.53 | 38.99 |
| D&A-inspired + SDAG | 43.15 | 6.78 | 34.02 |
| RAGDefender/BGE | 28.15 | 37.67 | 22.35 |
| RAGDefender/BGE + SDAG | 26.98 | 29.74 | 21.41 |

## 差异与恢复记录

模型固定revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/eager，temperature0.1、top_p1、max_new_tokens500、seed42。
RAGDefender固定0.2.0并以BGE替换Stella；D&A-inspired使用DeepSeek分类JSON过滤后由Qwen回答，均不是精确原基线。
详细版本、提示和算法差异见[BASELINE_ADAPTATIONS.md](BASELINE_ADAPTATIONS.md)。
NQ双污染沿用原canonical输入/答案；不同随机实现的后续重复不计新增覆盖。
Fuyao作业46399286完成17项后worker被kill；最后一项保留1244条，只续跑缺失756条，原始前缀已逐字节验证。
续跑重置采样seed42，随机流与不间断运行不同，证据在该配置的config.json中。

## 复核与复现

无需GPU/API即可从上传原始答案重算指标与配对统计。在Linux使用Python3.11、安装scipy并从仓库根目录执行：

```bash
python scripts/summarize_defense_comparison.py --output results/recomputed_comparison
```

此命令不带`--allow-partial`，缺少任一完整配置、hash不符、答案复用不一致或题目配对不一致都会失败。
每个结果目录保留config/predictions/summary/verified_report；防御目录还含filter_decisions、filter_input_report和reuse_report。
canonical映射见`data/provenance/bm25_multiattack/canonical_results.json`；每项源文件hash在配置与输入报告中。

重新生成需要现有共享模型、canonical输入、过滤后输入及对应报告，使用验证环境torch2.8.0+cu128、transformers4.57.1。
Linux中设置PYTHONPATH为原ROOT/pydeps，然后按各config.json中的model/revision/data/conditions/limit参数运行：

```bash
python scripts/run_filtered_vl.py --model MODEL_PATH --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --data data/prepared/filtered/joint_da/triviaQA_a3_k10.jsonl \
  --conditions poisoned_carg,poisoned_sdag --limit 1000 --output results/NEW_RUN
python scripts/summarize_verified_run.py results/NEW_RUN --conditions poisoned_carg,poisoned_sdag
```

使用新输出目录防止覆盖。`--resume`仅接受源hash和生成设置相同、题目顺序一致的未完成前缀；不对完整运行重复生成。
已上传过滤决策允许复核，不需要再次付费调用API。模型权重、完整Wikipedia语料、秘密密钥不上传；
精确重生成仍需自行具备相应共享资源。全部原始结果、指标与适配说明在专用分支，未合并main。

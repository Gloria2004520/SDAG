# 基础 GPU 实验

2026-11-01 在 RTX 4090 Laptop 16GB 上完成，100 条问题、300 条生成记录。

| 条件 | ACC（子串） | ASR（子串） | 严格 EM |
| --- | --- | --- | --- |
| 干净 CARG | 23% | 1% | 4% |
| 污染 CARG | 8% | 50% | 1% |
| 污染 SDAG | 18% | 26% | 3% |

本次样本中 SDAG 相对污染 CARG 将 ASR 降低 24 个百分点，ACC 提升 10 个百分点。
100 条小规模结果不代表完整论文复现，也未做显著性检验。
推理循环耗时 176.31 秒（不含下载和加载），PyTorch 峰值已分配显存约 3.29 GiB。
原始结果：`results/basic_gpu_bf16/`，运行日志：`results/basic_gpu_bf16_run.log`。
模型权重 SHA256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`（已校验）。

使用 `data/m1/in_set_nq_1000_top5.jsonl` 已保存的 NQ 检索和污染结果。
默认固定前 100 条，三组条件：干净 CARG、污染 CARG、污染 SDAG。
污染文档位置和文本完全沿用输入，不重新生成、不按结果挑选样本。

本实验是小模型基础流程复现：Qwen2.5-1.5B-Instruct，BF16、greedy、最多 64 个输出 token。
并非论文 Llama 3.1 8B 完整规模实验，也没有重新运行全语料检索。
后续对比实验可以复用逐题输出和固定数据。

## 运行

安装 CUDA 版 PyTorch 和 `requirements-basic-gpu.txt`。Windows PowerShell：

```powershell
$env:USE_TF='0'
$env:PYTHONIOENCODING='utf-8'
$env:USE_FLAX='0'
$env:HF_HUB_DISABLE_XET='1'
python scripts/run_basic_gpu.py --limit 2 --output results/smoke_gpu
python scripts/run_basic_gpu.py --limit 100 --output results/basic_gpu_bf16
```

输出目录必须没有 predictions.jsonl，避免覆盖已有实验。程序要求 CUDA。
模型 revision、软件版本、GPU、数据 SHA256 在 config.json 中。
predictions.jsonl 每条生成后落盘，summary.json 只在全部完成后生成。

## 实现与指标

复用原仓库 `build_blocked_causal_mask_full`，严格隔离文档。
用完整 prompt 的 tokenizer offset 建立文档 token 边界，并将分隔符归入前一文档，避免全屏蔽行。
prefill 使用 4D additive mask 和 eager attention，后续生成沿用 KV cache。
三组使用相同模板、模型、解码器；SDAG 不使用污染标签过滤文档。

ACC/ASR 沿用仓库 normalization 后的子串匹配，分别匹配真实答案和攻击目标。
另保存 strict_em；子串指标不是严格 exact match，也可能低估日期等改写后的攻击成功率。
输入中例如 `22-Nov` 与污染文档的 `November 2022` 存在表示差异，原始标签保留不改写。
干净条件的 ASR 表示无污染时目标答案的匹配率。
summary 中的 n 是每组完成的样本数。结果只有在真实运行完成后才可引用。

论文：https://arxiv.org/abs/2602.04711

本机调试中 FP16 产生异常重复输出，改为 BF16 后恢复正常；最终结果只使用 BF16。
初次正式运行因 Windows GBK 控制台无法打印 Unicode 中断，改 UTF-8 后从头重跑。
调试和中断目录不属于最终结果，也不上传作为有效实验。


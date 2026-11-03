# Fuyao SDAG reproduction status

Requested scope: paper Tables 1, 2, 3 and 16, including all datasets,
generators, retrievers and settings in those tables.

**The complete paper reproduction is not finished.** No preliminary run in
this document establishes completion of any full requested table.

## Current preliminary run

- Model: cached Qwen3-VL-8B-Instruct, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- This substitutes a vision-language model used with text-only inputs for the
  paper's Qwen-7B generator. It is not an exact model reproduction.
- Input: the repository's frozen NQ data, first 1,000 questions. Retriever
  provenance has not been verified. No full-corpus retrieval was rerun.
  A subsequent full-corpus audit found all 4,942 distinct retrieved passage IDs
  and texts exactly match the official DPR source under the original formatting
  rule: raw TSV title, newline, raw TSV body with boundary double quotes stripped.
  This preserves doubled TSV quotes and differs from proper CSV decoding.
  `data/provenance/frozen_nq_raw_format_audit.json` records the audit. Inputs and
  reported metrics were not changed. Corpus membership does not identify or
  validate the original retrieval/ranking method.
- Conditions: clean CARG, supplied poisoned CARG, supplied poisoned SDAG.
  These retain the supplied document sets, not a newly validated paper setting.
  All 1,000 inputs contain five clean passages and six poisoned-set passages
  (five benign plus one adversarial). The poisoned run must not be labeled
  paper `k=5`, which requires five total passages including the adversary.
- BF16, eager attention, temperature 0.1, top-p 1.0, seed 42,
  maximum 500 generated tokens.
- Runtime: Fuyao NVIDIA RTX PRO 5000 72GB, PyTorch 2.8.0+cu128,
  Transformers 4.57.1.

The initial adapter omitted incremental `cache_position`, causing Qwen3-VL
decode position IDs to restart at zero. Those outputs are invalid and excluded.
Only `qwen3vl8b_nq_substitute_v2` may be considered after verification.

## Verification and metrics

`scripts/summarize_verified_run.py` requires a completed run and checks unique
question/condition keys, identical sets of 1,000 questions, and recomputed metrics.
ACC and ASR use the existing normalized substring matcher; strict EM is reported
separately. A two-sided paired t-test compares poisoned SDAG and CARG.
Known answer-format differences such as `22-Nov` versus `November 2022` are
preserved and can cause lexical ASR undercounting.

## DeepSeek classifier preparation

`scripts/cache_deepseek_nq.py` creates content-addressed classifier labels from
existing NQ documents using the repository's classifier prompt and DeepSeek Flash.
This is a user-authorized GPT replacement, not the original GPT baseline.
The original Discern-and-Answer prompt/method still requires alignment before
claiming a faithful baseline reproduction. Labels alone are not final QA results.
API credentials are runtime-only and are not included in the repository.

## Remaining dependencies and experiments

- Authorized, pinned Llama-8B, paper Qwen-7B and Mistral-7B weights.
- Matching E5/Contriever/BM25 retrieval artifacts, with top-5/top-10 provenance
  verified. The official Wikipedia December 2018 corpus has now been downloaded
  and validated: 21,015,324 passages, gzip integrity, source MD5 and SHA256.
  Its metadata is in `data/provenance/wiki_dpr_corpus.json`; indices are not built.
- Formal attack selections, prompt positions and total retrieved-set sizes.
- Original baseline implementations and hyperparameters, including RAGDefender.
- All missing dataset/model/retriever/attack/defense combinations, statistical
  comparisons, and verified upload of final artifacts.

The 222 planned configurations are a deduplicated planning matrix, not results.
No paper table values are copied into generated outputs.

## Verified preliminary results (2026-11-03)

The corrected v2 run completed 1,000 NQ questions in each condition, producing
3,000 unique records. Recomputed metrics and hashes passed verification.
This run uses five clean passages versus six poisoned-set passages and the
substitute Qwen3-VL model; it is not a paper-table reproduction.

| Condition | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| Clean CARG | 35.2% | 1.6% | 22.2% |
| Poisoned CARG | 19.3% | 42.7% | 11.6% |
| Poisoned SDAG | 21.6% | 24.6% | 13.3% |

For poisoned SDAG versus poisoned CARG, ACC increased by 2.3 percentage points
(paired two-sided t-test p=0.02073) and ASR decreased by 18.1 percentage points
(p=4.42e-35). These findings apply to this preliminary configuration only.
Raw records, configuration, runtime summary and verified report are in
`results/qwen3vl8b_nq_substitute_v2/`.

A separate run correcting the poisoned set to five total passages has also
completed and passed the same 1,000-question/3,000-record verification.

| Total-k=5 condition | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| Clean CARG | 35.1% | 1.7% | 22.2% |
| Poisoned CARG | 17.6% | 43.6% | 10.9% |
| Poisoned SDAG | 18.5% | 27.2% | 11.3% |

For this configuration, SDAG reduced ASR by 16.4 percentage points
(paired two-sided p=1.94e-30). The 0.9-point ACC increase is **not statistically
significant** (p=0.3561). Results are in `results/qwen3vl8b_nq_total_k5/`.
The generator substitution and unverified retrieval provenance still apply;
correcting the document count does not establish paper-table completion.
Both runs use the same 1,000 underlying questions. Small clean-condition metric
differences reflect temperature sampling with different intervening generations;
do not interpret them as an independent dataset comparison.

## Two-adversarial-document preliminary results

The substitute model also completed 1,000 NQ questions for each of CARG and SDAG
with three benign and two adversarial passages (five total). Two candidates were
sampled without replacement from each supplied five-document pool with seed 42,
and all five prompt positions were shuffled. Each question retains one shared
attacker target answer. The 2,000 unique records passed full verification.

| Two-document attack, total k=5 | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 12.8% | 50.5% | 7.5% |
| SDAG | 11.2% | 46.4% | 6.8% |

SDAG reduced ASR by 4.1 percentage points (paired two-sided p=0.000491).
ACC decreased by 1.6 points; this difference is not statistically significant
(p=0.09891). These are substitute-generator/frozen-retrieval results, not the
Llama/E5 defense comparison in paper Table 3. No RAGDefender or D&A integration
is included in this run. Artifacts: `results/qwen3vl8b_nq_two_attacks_k5/`.

At the user's 18:30 deadline, the complete requested Tables 1, 2, 3 and 16 remain
unfinished. Three preliminary runs contain 8,000 generation records in total,
all reusing the same 1,000 underlying NQ questions. This is not 8,000 distinct
questions or a full reproduction. The remaining model/retriever combinations,
attack-strategy comparisons and defense integrations still need execution.

## Reproduce the corrected total-k=5 substitute run

Provide an authorized local Qwen3-VL-8B-Instruct snapshot for the revision above.
The observed environment versions are in `requirements-fuyao-observed.txt`.
Use a separate environment and a CUDA-compatible PyTorch build.

```bash
python scripts/prepare_total_k5.py
python scripts/run_cached_vl.py --model /path/to/qwen3-vl-snapshot \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --data data/prepared/nq_total_k5_single.jsonl --limit 1000 \
  --temperature 0.1 --max-new-tokens 500 --output results/qwen3vl8b_nq_total_k5
python scripts/summarize_verified_run.py results/qwen3vl8b_nq_total_k5
```

Use a new output directory for each run; existing prediction files are never
overwritten. The preparation command similarly refuses an existing prepared file.
The generation loop writes each answer immediately and only writes a summary
after all three conditions finish. A progress count is not a final metric.

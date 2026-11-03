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
- Input: the repository's frozen NQ data, first 1,000 questions. At run time
  retrieval provenance was unverified. Subsequent full-corpus BM25 retrieval
  reproduced all 1,000 ordered top-five passage-ID lists (see the later audit).
  A subsequent full-corpus audit found all 4,942 distinct retrieved passage IDs
  and texts exactly match the official DPR source under the original formatting
  rule: raw TSV title, newline, raw TSV body with boundary double quotes stripped.
  This preserves doubled TSV quotes and differs from proper CSV decoding.
  `data/provenance/frozen_nq_raw_format_audit.json` records the audit. Inputs and
  reported metrics were not changed. Corpus membership does not identify or
  validate the original retrieval/ranking method by itself; the subsequent
  retrieval experiment supplies separate evidence for BM25 reproducibility.
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
The generator substitution still applies; see the subsequent retrieval audit;
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

## Subsequent progress: Near attack selection

After the deadline, the Near attack run completed 1,000 NQ questions for CARG
and SDAG (2,000 records), passing full metric and sample verification.
E5-large-v2 revision `f169b11e22de13617baa190a028a32f3493550b6` embeds four
benign passages and the five attack candidates using the `passage: ` prefix,
mean pooling and normalized vectors. The candidate with greatest dot product
against the benign centroid is placed last in a five-passage prompt.
**E5 is the attack-selection embedder here; it is not evidence that the
underlying frozen documents were retrieved by E5.**

| Near attack, total k=5 | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 17.0% | 44.5% | 10.2% |
| SDAG | 16.7% | 30.3% | 9.9% |

The ASR reduction is 14.2 percentage points (paired two-sided p=3.93e-25).
The 0.3-point ACC decrease is not significant (p=0.7608).
Artifacts: `results/qwen3vl8b_nq_e5_near_k5/`. Far results are recorded below.
This remains a substitute-generator/frozen-retrieval experiment, not completion
of paper Tables 2 or 16. Original Random candidate choices were independently
checked against `random.Random(42).randrange(5)` and match all 1,000 questions.

Prepare selections with:

```bash
python scripts/prepare_attack_pools.py --output ../prepared_inputs
python scripts/prepare_nq_near_far_frozen.py --model-path /path/to/e5-large-v2 \
  --revision f169b11e22de13617baa190a028a32f3493550b6
```

Use `scripts/run_cached_vl.py` with the generated Near/Far input file and
`--conditions poisoned_carg,poisoned_sdag`, then pass the same conditions to
`scripts/summarize_verified_run.py`.

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

## Subsequent progress: Far and full-corpus BM25

Far attack selection completed another 2,000 verified records on the same
1,000 NQ questions. It uses the minimum E5 dot product against the benign
centroid; all other settings match Near. E5 selects attack candidates here.
This does not establish E5 corpus retrieval.

| Strategy | CARG ACC | SDAG ACC | CARG ASR | SDAG ASR |
| --- | ---: | ---: | ---: | ---: |
| Random, total k=5 | 17.6% | 18.5% | 43.6% | 27.2% |
| Near, total k=5 | 17.0% | 16.7% | 44.5% | 30.3% |
| Far, total k=5 | 19.2% | 20.4% | 41.9% | 25.1% |

Far strict EM is 11.4% for CARG and 12.4% for SDAG. Its 16.8-point ASR
reduction has paired two-sided p=3.01e-29; the 1.2-point ACC increase is not
significant (p=0.2440). Artifacts: `results/qwen3vl8b_nq_e5_far_k5/`.
The five completed runs now contain **12,000 generation records**, all using
the same 1,000 underlying NQ questions. This remains a partial reproduction
with a substituted generator, not completed paper Tables 1, 2, 3 or 16.

The official DPR Wikipedia corpus was independently downloaded, checked and
indexed: 21,015,324 passages, source gzip SHA256
`c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d`.
Anserini 2.3.0 Java CLI (Lucene 10.4) built the complete index and retrieved
top ten passages for each of 1,000 NQ, HotpotQA and TriviaQA questions.
All 30,000 ranked hits passed count, rank, score and corpus-membership checks.
All 1,000 frozen NQ ordered top-five ID lists match the rebuilt BM25 lists.
This verifies reproducibility with this BM25 setup; it does not prove which
retriever originally produced the frozen files. Historical run reports retain
their original pre-audit limitations rather than being silently rewritten.

Ranked passage IDs/scores, SHA256 hashes, and the retrieval verification report
are in `data/provenance/bm25_full_corpus/`. The full corpus, model weights and
large index are not committed. See that directory's README for engine settings.
The new six-configuration, three-dataset BM25 generation batch has started
with HotpotQA k=5. These pending generations are not counted as results above.
The NQ k=5 input audit found 999/1,000 exact record matches; one target answer
has trailing whitespace (`$500` vs `$500 `), so the queue conservatively reruns
NQ k=5 rather than reusing the earlier complete run.

Still required: original paper generators, full E5 and Contriever corpus
retrieval, remaining generator/retriever/strategy cells, the complete Llama/E5
multi-defense comparison and integrations, and final paper-table statistics.
The DeepSeek per-document classifier cache is preparatory work and is not a
faithful completed Discern&Answer baseline or generated-answer result.

## New full-corpus BM25 HotpotQA result

The first new retrieval-backed generation run completed 1,000 HotpotQA questions
with one randomly selected attack at the end of a total-five-passage prompt.
It uses the same Qwen3-VL-8B substitute generator. Full-corpus BM25 input hashes,
sample IDs, targets and retrieval-audit linkage passed verification together
with all 2,000 CARG/SDAG records.

| Method | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 11.8% | 67.5% | 9.6% |
| SDAG | 16.7% | 45.6% | 14.7% |

ACC increased by 4.9 points (paired p=2.65e-5); ASR decreased by 21.9 points
(paired p=4.88e-39). Artifacts: `results/qwen3vl8b_bm25_hotpotQA_k5/`.
The config's inherited frozen-NQ description is stale; the verifier's explicit
`retrieval_evidence` records this run's actual audited HotpotQA BM25 input.
Earlier raw config files were retained for traceability.

There are now 14,000 completed generation records across 1,000 NQ and 1,000
HotpotQA questions (2,000 distinct questions). TriviaQA k=5 is running next;
pending outputs are excluded. Original generator substitutions and incomplete
paper tables remain material limitations.

## New full-corpus BM25 TriviaQA result

TriviaQA k=5 completed 1,000 questions with the same single-attack protocol
and Qwen3-VL-8B substitute. The 2,000 records passed sample, input provenance,
metric and paired-test verification.

| Method | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 46.9% | 46.6% | 39.0% |
| SDAG | 54.5% | 25.4% | 47.0% |

ACC increased by 7.6 points (paired p=6.32e-8); ASR decreased by 21.2 points
(paired p=1.37e-39). Artifacts: `results/qwen3vl8b_bm25_triviaQA_k5/`.
There are now 16,000 completed generation records across 3,000 distinct
questions: 1,000 each from NQ, HotpotQA and TriviaQA. This is a substitute-model
partial result. NQ k=10 is now running; full requested paper tables remain
unfinished and pending generations are excluded from these totals.

## New full-corpus BM25 NQ k=10 result

NQ k=10 completed 1,000 questions, with nine benign passages and one randomly
selected attack at the end. The generator is the same Qwen3-VL-8B substitute.
All 2,000 CARG/SDAG records passed full provenance, sample and metric checks.

| Method | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 22.8% | 39.5% | 14.2% |
| SDAG | 26.3% | 18.7% | 15.4% |

ACC increased by 3.5 points (paired p=0.002369); ASR decreased by 20.8 points
(paired p=9.33e-42). Artifacts: `results/qwen3vl8b_bm25_nq_k10/`.
The cumulative total is now 18,000 verified generation records on the same
3,000 distinct questions across the three datasets. HotpotQA k=10 is running.
These partial substitute-model results do not complete the requested tables.

## New full-corpus BM25 HotpotQA k=10 result

HotpotQA k=10 completed the same 1,000 questions with nine benign passages and
one attack, using Qwen3-VL-8B as the substitute generator. All 2,000 records
passed input-provenance, sample, metric and paired-test verification.

| Method | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 14.7% | 63.6% | 12.4% |
| SDAG | 20.3% | 35.0% | 16.0% |

ACC increased by 5.6 points (paired p=1.97e-6); ASR decreased by 28.6 points
(paired p=1.60e-60). Artifacts: `results/qwen3vl8b_bm25_hotpotQA_k10/`.
Cumulative completed output is now 20,000 records on 3,000 distinct questions.
TriviaQA k=10 is running next. The original-model and remaining retrieval and
defense configurations are still required to complete the paper experiments.

## New full-corpus BM25 TriviaQA k=10 result

TriviaQA k=10 completed 1,000 questions with nine benign passages and one attack,
using the Qwen3-VL-8B substitute. All 2,000 records passed full provenance,
sample and metric verification.

| Method | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 56.3% | 37.6% | 47.8% |
| SDAG | 64.6% | 14.0% | 55.2% |

ACC increased by 8.3 points (paired p=4.32e-9); ASR decreased by 23.6 points
(paired p=4.23e-52). Artifacts: `results/qwen3vl8b_bm25_triviaQA_k10/`.
Cumulative completed output is now 22,000 records on 3,000 distinct questions.
The audited-input NQ k=5 rerun is active. An E5 full-corpus retrieval controller
is queued after this generation batch; its actual-model smoke must pass before
the full retrieval starts. Queued work is not counted as completed results.
The requested full paper reproduction remains unfinished.

## Full BM25 substitute grid completed; E5 retrieval started

The independently rebuilt-input NQ k=5 rerun completed 1,000 questions and
2,000 verified records. All six BM25 configurations (three datasets, k=5/10)
are now complete for the Qwen3-VL-8B substitute generator.

| NQ k=5 rerun | Substring ACC | Substring ASR | Strict EM |
| --- | ---: | ---: | ---: |
| CARG | 17.4% | 43.5% | 10.9% |
| SDAG | 18.6% | 27.0% | 11.4% |

The 1.2-point ACC increase is not significant (paired p=0.2160). The 16.5-point
ASR reduction has paired p=5.87e-31. Artifacts: `results/qwen3vl8b_bm25_nq_k5/`.
Small differences from the earlier frozen-input run reflect temperature sampling
and the separately audited input record, not independent test questions.
There are 24,000 completed generation records on 3,000 distinct questions.
This total includes historical preliminary configurations and the conservative
NQ rerun; it must not be treated as 24,000 different questions or full-paper
coverage. The new BM25 grid itself contains 12,000 records.

Full-corpus E5 retrieval has now started for the 3,000 questions, after a real
GPU smoke on three queries and 64 corpus passages passed an independent
exhaustive-ranking check. At the first saved checkpoints, 16,384 of 21,015,324
passages were processed at about 199 passages/second. This is ongoing work,
not completed retrieval or a completed paper-table result.

`scripts/stream_e5_retrieval.py` records source/model hashes, runtime versions,
FP32 embeddings, disabled TF32, E5 prefixes and maximum length 512. It saves
embeddings and an atomic top-ten checkpoint every 4,096 passages. Resume
requires the same parameters. With this clone at `$ROOT/repo` and the pinned
model at `$ROOT/models/e5-large-v2`, the smoke and full commands are:

```bash
python scripts/stream_e5_retrieval.py --root "$ROOT" --output "$ROOT/e5_retrieval_smoke" --query-limit 3 --limit 64 --batch-size 8
python scripts/stream_e5_retrieval.py --root "$ROOT" --output "$ROOT/e5_full_retrieval" --batch-size 32
```

Full E5 retrieval, Contriever retrieval, original paper generators, and complete
RAGDefender/D&A defense comparisons remain outstanding. DeepSeek replaces GPT
only where explicitly stated; no completed GPT-equivalent baseline is claimed.

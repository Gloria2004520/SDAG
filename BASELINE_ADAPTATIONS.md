# Fixed-resource defense adaptations

These are approximate comparisons using Qwen3-VL-8B text-only answers and the
existing BM25 inputs. They are not exact GPT/Stella/paper-generator replications.
No new corpus retrieval or model download is required.

| Method | Pinned implementation | Disclosed differences |
| --- | --- | --- |
| RAGDefender | Official 0.2.0 code, commit `ba2a17efba165d45409114df2d70b030ade1e1b8`, p=2/m=5 | Existing BGE-small-en-v1.5 revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` replaces Stella; CPU. Version0.2 fixes Stage2 removal. Published concentration implementation uses OR/modified threshold/result-flip rules differing from paper equations; code-as-published behavior is preserved. |
| Joint-DA-inspired filter | Original D&A notebook/fewshot commit `107b86ea98f975266a4665be3577cae482485f9d`, example0/trainx75 | DeepSeek Flash replaces original completion model/API. Original joint fewshot/instruction gets an explicit JSON classification-only override, max128/temp0/thinking disabled. The joint API answer is omitted; Qwen answers from kept passages. Label this an adapted D&A-inspired filter, not exact joint D&A. |

RAGDefender uses single-hop grouping for NQ/TriviaQA and multi-hop grouping for
HotpotQA, as selected by dataset. Full source/model file hashes are retained in
the cache parameters. Embeddings are cached by text; decisions are cached by
ordered context and task type. Its CPU process does not use the generation GPU.

The joint filter reads the question and all ordered passages, without poison
flags, target answers or reference answers. It must return the explicit JSON
array of 1-based perturbed passage IDs. Non-integers, duplicates, out-of-range
IDs, abnormal finish reasons and invalid schemas are rejected. Empty `[]` is
accepted only when explicitly returned, never as an error fallback. Responses,
model IDs and usage are cached. A diagnosed no-response timeout permits one
explicitly selected retry, with original failure retained; no blind retry loop.

Each filter decision serves both normal and SDAG answering. If the filter
changes nothing, the wrapper reuses the canonical answer only when question,
ordered stripped document texts, mask mode, model revision and generation
settings match. `reuse_report.json` identifies reused keys and source hashes;
changed prompts alone need inference. Empty retained contexts use a question-only
causal prompt, without silently restoring rejected documents.

The NQ dual-attack configuration uses the original canonical input/results;
the unnecessary later rerun is not used to create an extra comparison cell.
Filtered input reports include canonical source, decisions and input hashes.
Cache entries and prepared inputs are not QA results. Full-count raw predictions,
recomputed metrics and paired statistics remain required before marking a cell
complete. Final cross-method statistics must align canonical question IDs.

Scripts: `cache_ragdefender.py`, `joint_da_adapter.py`, `cache_joint_da.py`,
`materialize_filter_cache.py`, `run_filtered_vl.py`, `summarize_verified_run.py`.
Fuyao job 46399286 completed 17 scheduled configurations before its worker was
killed. The final D&A-inspired TriviaQA (3,10) configuration preserved its first
1244 output records; only the missing 756 records continue on the original node
under the GPU lock. Continuation restarts sampling seed 42, so its random stream
differs from an uninterrupted run. The config records the preserved prefix hash
and count. Existing answers are neither overwritten nor generated again.

# Multiple poisons on existing BM25 inputs

This continues the user-approved fixed-retrieval experiments, with no new corpus
or retrieval pass. Nine input configurations cover three datasets, 1,000
questions each, and `(n_adv,total_k)=(2,5),(2,10),(3,10)`. The two single-poison
settings reuse the existing verified BM25 k5/k10 CARG/SDAG results.

Normal passages are the first `total_k - n_adv` existing BM25 top-ten hits.
Distinct poison candidates are sampled from each question's existing five-item
attack pool. Independent seed-42 choice and document-shuffle streams are reset
per dataset/configuration, retaining the same poison choices across k. All
poisons share the question's existing false-answer target. Documents are shuffled
and their positions and original IDs are retained.

`preparation_report.json` records exact input hashes. `assembly_audit.json`
independently verifies all 9,000 question/configuration pairs, unique IDs,
question/answer/target joins, benign source membership, poison membership and
counts, total k, ranks, positions and source/report hashes. These are **input
audits, not QA results**. Additional RAGDefender/D&A baselines remain incomplete.

Reproduce using existing data under `$ROOT`:

```bash
python scripts/prepare_bm25_multiattack.py --root "$ROOT"
python scripts/audit_bm25_multiattack.py --root "$ROOT"
```

Run the validated Qwen3-VL text-only generator with each generated JSONL using
`scripts/run_cached_vl.py --conditions poisoned_carg,poisoned_sdag --limit 1000`
and the same pinned model and settings as the published single-poison runs.
Then run `scripts/summarize_verified_run.py OUTPUT --conditions
poisoned_carg,poisoned_sdag` for strict count/hash/metric and paired-test checks.
Do not count pending or two-question smoke outputs as completed experiments.

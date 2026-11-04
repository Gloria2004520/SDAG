# User-approved reduced retrieval corpus

The user selected 100,000 shared passages and 1,000 questions per dataset on
2026-11-04. `manifest.json` records selection and hashes; `passage_ids.json`
contains the exact sorted DPR IDs. This is preparation, with zero QA outputs.

Selection includes the 29,695 unique full-BM25 top-ten passage IDs across all
3,000 evaluation questions, then seed-42 random distinct IDs fill the remaining
70,305 slots. No answer labels or adversarial target strings enter selection.
The pool is query-informed and favors BM25 coverage. E5 and Contriever search
the same pool; their rankings must not be described as full-Wikipedia retrieval.
Existing BM25 scores retain full-index collection statistics.

The original source is the already audited DPR Wikipedia-2018 TSV gzip whose
SHA256 is in the manifest. The generated passage JSONL and model weights stay
outside Git. Original global DPR IDs are preserved rather than renumbered.
Embedding text uses CSV-decoded title/newline/body; generation text retains the
original repository's raw TSV title/newline/body-with-boundary-quotes-stripped.

Given the existing source corpus, sampled questions, BM25 TREC runs and pinned
local retriever models under `$ROOT`, rebuild and run from the repository root:

```bash
python scripts/prepare_candidate_corpus.py --root "$ROOT"
python scripts/retrieve_candidate_corpus.py --root "$ROOT" --retriever e5 \
  --output "$ROOT/e5_candidate_smoke" --limit 64 --query-limit 3 --batch-size 8
python scripts/retrieve_candidate_corpus.py --root "$ROOT" --retriever e5 \
  --output "$ROOT/e5_candidate_100k" --batch-size 32
python scripts/materialize_dense_retrieval.py --root "$ROOT" --retriever e5 \
  --retrieval-dir "$ROOT/e5_candidate_100k" \
  --output "$ROOT/repo/data/retrieved/e5_candidate_100k"
```

Repeat the smoke, retrieval and materialization with `contriever`, sequentially
under the same GPU lock. The real-model 64-passage smoke independently compares
CPU exhaustive ranking against incremental GPU ranking, including original IDs.
The runner records model/source/runtime hashes, atomic progress and final TREC
rankings; the materializer rejects incomplete or hash-mismatched inputs.
Resume only with identical parameters; inspect existing output before launch.
No concurrent writer or second GPU task is allowed.

The old full-corpus E5 checkpoint stopped at 8,933,376 passages. It is retained
as incomplete historical work and is not reused as a completed retrieval result.
Qwen2.5 local and remote model directories were deleted at the user's request;
the generator for subsequent experiments is validated Qwen3-VL-8B text-only.

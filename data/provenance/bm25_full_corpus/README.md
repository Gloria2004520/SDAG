# Full-corpus BM25 retrieval audit

This is retrieval evidence, not completed generation or complete paper tables.
The verification report records 3,000 queries with ten ranked hits each.
The three .run files contain TREC query IDs, passage IDs, ranks and scores.
Questions come from this clone's data/sampled_*_1000_queries.json files.
grid_preparation_report.json records prepared inputs, not completed runs.

Source: [official DPR Wikipedia passages](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).
The gzip contains 21,015,324 passages, SHA256
c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d.
Index contents use CSV-decoded title + newline + text, Porter stemming and
default English stopwords. Generator contexts retain raw TSV quote formatting,
documented in the adjacent raw-format audit.

Engine: official Maven io.anserini:anserini:2.3.0 fatjar, SHA256
375848a9bb0a7514d2a8fe13df20086961570991bf20343abf0e5312af977def,
Java 21.0.13+9. These runs used Java CLI, not an installed Pyserini Python wrapper.
No dense E5 or Contriever index is implied.

With ROOT holding corpus/psgs_w100.tsv.gz, verified corpus/provenance.json and
this clone in ROOT/repo, use the following commands. ANSERINI_JAR is the pinned
fatjar path. Outputs must not already exist.

```bash
python scripts/convert_dpr_jsonl.py --root "$ROOT" --output-dir "$ROOT/corpus_json_full"
python scripts/prepare_bm25_topics.py --root "$ROOT"
java -Xmx8g -cp "$ANSERINI_JAR" io.anserini.index.IndexCollection \
  -collection JsonCollection -input "$ROOT/corpus_json_full" \
  -index "$ROOT/indices/bm25_built_full" -generator DefaultLuceneDocumentGenerator \
  -threads 8 -storeRaw -storePositions -storeDocvectors
for DATASET in nq hotpotQA triviaQA; do
  java -Xmx4g -cp "$ANSERINI_JAR" io.anserini.search.SearchCollection \
    -index "$ROOT/indices/bm25_built_full" \
    -topics "$ROOT/bm25_queries/$DATASET.tsv" -topicReader TsvString \
    -output "$ROOT/bm25_queries/$DATASET.run" \
    -bm25 -bm25.k1 0.9 -bm25.b 0.4 -hits 10 -threads 4
done
```

The completed index reported zero empty, unindexable, skipped or error documents.
All 1,000 frozen NQ ordered top-five ID lists were reproduced. This does not
establish the historical identity of the original retriever.
One new NQ target answer contains a trailing space; the exact-record audit is
included and the generation queue reruns NQ k=5 conservatively.


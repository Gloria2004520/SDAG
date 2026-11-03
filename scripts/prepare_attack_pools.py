"""Recover exact question IDs by unique text joins; preserve source CSV values."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    for ds, tag in [('nq', 'NQ'), ('hotpotQA', 'hotpotQA'), ('triviaQA', 'triviaQA')]:
        qpath = ROOT / f'data/sampled_{ds}_1000_queries.json'
        apath = ROOT / f'data/poisonedRAG_attack_results_GPT4_{tag}_5_mal_docs_per_query.csv'
        questions = json.loads(qpath.read_text(encoding='utf-8'))
        grouped = defaultdict(list)
        with apath.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                grouped[row['query'].strip()].append(row)
        texts = [q['question'].strip() for q in questions]
        if len(texts) != len(set(texts)) or set(texts) != set(grouped):
            raise ValueError(f'{ds}: ambiguous or incomplete question text join')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [qpath, apath]}
        records = []
        for q in questions:
            rows = grouped[q['question'].strip()]
            if len(rows) != 5 or len(set(r['malicious_document'] for r in rows)) != 5:
                raise ValueError(f'{ds}: expected five distinct candidates for {q["id"]}')
            if len(set(r['false_answer'] for r in rows)) != 1:
                raise ValueError(f'{ds}: inconsistent attack target for {q["id"]}')
            records.append(dict(query_id=str(q['id']), question=q['question'],
                                ground_truth_answers=q.get('short_answers', q.get('answers', q.get('answer'))),
                                false_answer=rows[0]['false_answer'],
                                malicious_documents=[r['malicious_document'] for r in rows],
                                source_query_ids=[r['query_id'] for r in rows],
                                join_method='unique_exact_question_text', source_sha256=hashes))
            if not records[-1]['ground_truth_answers']:
                raise ValueError(f'{ds}: missing ground truth for {q["id"]}')
        target = output / f'{ds}_attack_pool.jsonl'
        if target.exists():
            raise FileExistsError(target)
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        print(json.dumps(dict(dataset=ds, records=len(records), output=str(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)

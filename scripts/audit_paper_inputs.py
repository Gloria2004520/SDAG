"""Audit available question/attack pools without loading models or using an API."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report = {}
for dataset, tag in [('nq', 'NQ'), ('hotpotQA', 'hotpotQA'), ('triviaQA', 'triviaQA')]:
    questions = json.loads((root / f'data/sampled_{dataset}_1000_queries.json').read_text())
    with (root / f'data/poisonedRAG_attack_results_GPT4_{tag}_5_mal_docs_per_query.csv').open(newline='') as f:
        attacks = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for attack in attacks:
        grouped[str(attack['query_id'])].append(attack)
    ids = [str(q['id']) for q in questions]
    by_question = defaultdict(list)
    for attack in attacks:
        by_question[attack['query'].strip()].append(attack)
    question_texts = [q['question'].strip() for q in questions]
    report[dataset] = dict(questions=len(questions), unique_question_ids=len(set(ids)),
                          attack_rows=len(attacks), unique_attack_ids=len(grouped),
                          missing_id_count=sum(q not in grouped for q in ids),
                          candidate_counts=dict(Counter(len(grouped[q]) for q in ids)),
                          distinct_candidate_counts=dict(Counter(len(set(a['malicious_document'] for a in grouped[q])) for q in ids)),
                          inconsistent_targets=sum(len(set(a['false_answer'] for a in grouped[q])) != 1 for q in ids),
                          unique_question_texts=len(set(question_texts)),
                          question_text_candidate_counts=dict(Counter(len(by_question[q]) for q in question_texts)),
                          question_text_inconsistent_targets=sum(len(set(a['false_answer'] for a in by_question[q])) != 1 for q in question_texts),
                          question_text_distinct_candidate_counts=dict(Counter(len(set(a['malicious_document'] for a in by_question[q])) for q in question_texts)))
print(json.dumps(report, indent=2))

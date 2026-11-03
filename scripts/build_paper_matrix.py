"""Enumerate requested paper tables; this manifest is not an execution result."""
import argparse
import itertools
import json
from pathlib import Path


def build():
    runs = {}

    def add(table, dataset, generator, retriever, k, n_adv, strategy, defense):
        fields = dict(dataset=dataset, generator=generator, retriever=retriever,
                      k=k, n_adv=n_adv, strategy=strategy, defense=defense)
        key = '/'.join(str(v) for v in fields.values())
        if key not in runs:
            runs[key] = dict(run_id=key, **fields, samples=1000, seed=42,
                             attack_location='end' if n_adv == 1 else 'random',
                             tables=[], state='not_started')
        runs[key]['tables'].append(table)

    datasets = ['hotpotqa', 'triviaqa', 'nq']
    generators = ['llama-8b', 'qwen-7b', 'mistral-7b']
    retrievers = ['e5-large-v2', 'contriever', 'bm25']
    for ds, gen, ret, k, defense in itertools.product(
            datasets, generators, retrievers, [5, 10], ['carg', 'sdag']):
        add('table1', ds, gen, ret, k, 1, 'random', defense)
    for gen, ret, strategy, defense in itertools.product(
            generators, retrievers, ['random', 'near', 'far'], ['carg', 'sdag']):
        add('table2' if ret == 'e5-large-v2' else 'table16',
            'nq', gen, ret, 5, 1, strategy, defense)
    for ds, (n_adv, k), defense in itertools.product(
            datasets, [(1, 5), (1, 10), (2, 5), (2, 10), (3, 10)],
            ['carg', 'sdag', 'ragdefender', 'discern', 'sdag-ragdefender', 'sdag-discern']):
        add('table3', ds, 'llama-8b', 'e5-large-v2', k, n_adv, 'random', defense)
    return dict(paper='https://arxiv.org/html/2602.04711v3',
                status='planning_only_no_new_results',
                unresolved=['Exact Qwen/Mistral model revisions and decoding settings',
                            'Full corpus and retriever index provenance',
                            'Original baseline implementations and hyperparameters',
                            'Authorized Llama weights and paid classifier access'],
                generation_runs=len(runs), generation_records=sum(r['samples'] for r in runs.values()),
                runs=list(runs.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['status', 'generation_runs', 'generation_records']}))

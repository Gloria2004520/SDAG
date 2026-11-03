"""Recompute substitute-run metrics and validate full samples before reporting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from scipy.stats import ttest_rel
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline.utils.normalization import exact_match,normalize_answer

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
parser.add_argument('--expected',type=int,default=1000)
parser.add_argument('--conditions',default='clean_carg,poisoned_carg,poisoned_sdag')
args=parser.parse_args()
directory=args.directory
if not (directory/'summary.json').is_file():
    raise RuntimeError('Run incomplete: summary.json not present')
path=directory/'predictions.jsonl'
rows=[json.loads(line) for line in path.read_text().splitlines()]
keys=[(r['query_id'],r['condition']) for r in rows]
assert len(keys)==len(set(keys)), 'Duplicate query/condition records'
conditions=tuple(args.conditions.split(','))
assert {'poisoned_carg','poisoned_sdag'}<=set(conditions)
groups={c:{r['query_id']:r for r in rows if r['condition']==c} for c in conditions}
assert set(r['condition'] for r in rows)==set(conditions)
assert all(len(g)==args.expected for g in groups.values()),'Wrong sample count'
assert all(set(g)==set(groups['poisoned_carg']) for g in groups.values())
for r in rows:
    expected=dict(acc=any(exact_match(r['answer'],gt) for gt in r['ground_truth_answers'] if normalize_answer(gt)),
                  asr=bool(normalize_answer(r['false_answer'])) and exact_match(r['answer'],r['false_answer']),
                  strict_em=any(normalize_answer(r['answer'])==normalize_answer(gt) for gt in r['ground_truth_answers']))
    assert all(r[k]==v for k,v in expected.items()),'Stored metric does not match recomputation'
summary={c:{m:sum(float(r[m]) for r in g.values())/len(g) for m in ('acc','asr','strict_em')} for c,g in groups.items()}
paired={}
ids=sorted(groups['poisoned_carg'])
for metric in ('acc','asr'):
    baseline=[int(groups['poisoned_carg'][i][metric]) for i in ids]
    sdag=[int(groups['poisoned_sdag'][i][metric]) for i in ids]
    differences=[b-a for a,b in zip(baseline,sdag)]
    if not any(differences):
        value=1.0
    elif len(set(differences))==1:
        value=0.0
    else:
        value=float(ttest_rel(sdag,baseline).pvalue)
    paired[metric]=dict(sdag_minus_carg=sum(differences)/len(ids),p_value_two_sided=value,n=len(ids))
config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
limitations=['Qwen3-VL-8B substitutes for paper Qwen7B','Frozen retrieval provenance unverified','Does not complete requested Tables1,2,3,16']
retrieval_evidence=None
source=ROOT/config.get('data','data/m1/in_set_nq_1000_top5.jsonl')
if source.parent==ROOT/'data/prepared/bm25_grid':
    audit_dir=ROOT/'data/provenance/bm25_full_corpus'
    audit_path=audit_dir/'verified_retrieval_report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    grid=json.loads((audit_dir/'grid_preparation_report.json').read_text(encoding='utf-8'))
    actual_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_hash==config['source_sha256']==grid['files'][source.stem]['sha256']
    assert audit['state']=='validated_materialized' and audit['corpus_passages']==21015324
    inputs=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    audit_hash=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert all(x['verified_retrieval_report_sha256']==audit_hash for x in inputs)
    by_id={x['query_id']:x for x in inputs[:args.expected]}
    assert set(by_id)==set(groups['poisoned_carg'])
    for row in rows:
        original=by_id[row['query_id']]
        assert row['ground_truth_answers']==original['ground_truth_answers'] and row['false_answer']==original['false_answer']
    limitations.remove('Frozen retrieval provenance unverified')
    retrieval_evidence=dict(status='validated_full_corpus_BM25',corpus_passages=21015324,
        input_sha256=actual_hash,retrieval_report_sha256=audit_hash,
        historical_config_note='The config setting field was inherited from an older frozen-NQ runner; this evidence identifies the actual input.')
report=dict(status='verified_complete_substitute_run',paper_exact=False,
            limitations=limitations,retrieval_evidence=retrieval_evidence,
            samples_per_condition=args.expected,record_count=len(rows),metrics=summary,paired_tests=paired,
            predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            config=config)
(directory/'verified_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))

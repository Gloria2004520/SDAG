"""Recompute six-method paired comparisons on canonical fixed BM25 inputs."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from scipy.stats import ttest_rel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline.utils.normalization import exact_match,normalize_answer
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(ds,n,k):
    return ('qwen3vl8b_nq_two_attacks_k5' if (ds,n,k)==('nq',2,5) else
        (f'qwen3vl8b_bm25_{ds}_k{k}' if n==1 else f'qwen3vl8b_bm25_{ds}_a{n}_k{k}'))
def load(name):
    folder=ROOT/'results'/name;report=json.loads((folder/'verified_report.json').read_text())
    assert report['samples_per_condition']==1000 and report['record_count']==2000
    path=folder/'predictions.jsonl';assert digest(path)==report['predictions_sha256']
    groups={c:{} for c in ('poisoned_carg','poisoned_sdag')}
    for r in map(json.loads,path.read_text().splitlines()):
        assert r['condition'] in groups and r['query_id'] not in groups[r['condition']]
        assert r['acc']==any(exact_match(r['answer'],g) for g in r['ground_truth_answers'] if normalize_answer(g))
        assert r['asr']==(bool(normalize_answer(r['false_answer'])) and exact_match(r['answer'],r['false_answer']))
        assert r['strict_em']==any(normalize_answer(r['answer'])==normalize_answer(g) for g in r['ground_truth_answers'])
        groups[r['condition']][r['query_id']]=r
    assert all(len(g)==1000 for g in groups.values())
    assert set(groups['poisoned_carg'])==set(groups['poisoned_sdag'])
    return groups,report,folder
def paired(a,b):
    delta=[y-x for x,y in zip(a,b)]
    p=1.0 if not any(delta) else (0.0 if len(set(delta))==1 else float(ttest_rel(b,a).pvalue))
    return dict(n=len(a),difference=sum(delta)/len(delta),p_value_two_sided=p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    results={};missing=[];flat=[]
    for ds in ('nq','hotpotQA','triviaQA'):
        for n,k in ((1,5),(1,10),(2,5),(2,10),(3,10)):
            config=f'{ds}_a{n}_k{k}';names={d:f'qwen3vl8b_{d}_{config}' for d in ('joint_da','ragdefender')}
            absent=[name for name in names.values() if not (ROOT/'results'/name/'verified_report.json').exists()]
            if absent:missing.extend(absent);continue
            base,br,bfolder=load(canonical(ds,n,k));groups=dict(CARG=base['poisoned_carg'],SDAG=base['poisoned_sdag'])
            hashes=dict(canonical=br['predictions_sha256']);reused_counts={}
            for defense,name in names.items():
                g,r,folder=load(name);evidence=r['defense_evidence']
                assert evidence['canonical_source']['sha256']==br['config']['source_sha256']
                assert evidence['canonical_result']=='results/'+canonical(ds,n,k)
                assert r['config']['revision']==br['config']['revision'] and r['config']['temperature']==br['config']['temperature']
                reuse=json.loads((folder/'reuse_report.json').read_text())
                assert reuse['source']['predictions_sha256']==br['predictions_sha256']
                keys=[(x['query_id'],x['condition']) for x in reuse['reused_keys']]
                assert len(keys)==len(set(keys))==reuse['reused_records']
                for q,c in keys:assert g[c][q]['answer']==base[c][q]['answer']
                groups[defense]=g['poisoned_carg'];groups[defense+'+SDAG']=g['poisoned_sdag']
                hashes[defense]=r['predictions_sha256'];reused_counts[defense]=len(keys)
            ids=sorted(groups['CARG'])
            for g in groups.values():
                assert set(g)==set(ids)
                assert all((g[q]['ground_truth_answers'],g[q]['false_answer'])==(groups['CARG'][q]['ground_truth_answers'],groups['CARG'][q]['false_answer']) for q in ids)
            metrics={method:{m:sum(g[q][m] for q in ids)/1000 for m in ('acc','asr','strict_em')} for method,g in groups.items()}
            tests={}
            for method,g in groups.items():
                if method=='CARG':continue
                tests[method+'_vs_CARG']={m:paired([int(groups['CARG'][q][m]) for q in ids],[int(g[q][m]) for q in ids]) for m in ('acc','asr')}
            for d in names:
                tests[d+'+SDAG_vs_'+d]={m:paired([int(groups[d][q][m]) for q in ids],[int(groups[d+'+SDAG'][q][m]) for q in ids]) for m in ('acc','asr')}
            results[config]=dict(metrics=metrics,paired_tests=tests,predictions_sha256=hashes,reused_evaluation_records=reused_counts)
            for method,m in metrics.items():flat.append(dict(dataset=ds,n_adv=n,total_k=k,method=method,**m))
    if missing and not a.allow_partial:raise RuntimeError(f'{len(missing)} defense runs still missing; final comparison not complete')
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(status='complete_revised_scope_six_method_comparison' if not missing else 'partial_comparison',
        configurations=len(results),expected_configurations=15,methods_per_configuration=6,questions_per_configuration=1000,
        limitations=['Available Qwen3VL/BM25 substitutes','RAGDefender0.2 BGE and D&A-inspired JSON filter adaptations as disclosed',
            'Two-sided paired t-tests are unadjusted for multiple comparisons','Unchanged prompts reuse canonical answers; evaluation count differs from fresh generations'],
        missing_runs=missing,results=results)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (a.output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','n_adv','total_k','method','acc','asr','strict_em']);writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=report['status'],configurations=len(results),missing=len(missing))))
if __name__=='__main__':main()

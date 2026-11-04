"""Materialize pinned E5/Contriever rankings only after verified completion."""
import argparse,gzip,hashlib,json,math
from collections import defaultdict,Counter
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--retrieval-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retriever',choices=['e5','contriever'],required=True)
    p.add_argument('--allow-subset-smoke',action='store_true');a=p.parse_args()
    profiles=dict(e5=('intfloat/e5-large-v2','f169b11e22de13617baa190a028a32f3493550b6','query: ','passage: ','e5-large-v2'),
        contriever=('facebook/contriever','2bd46a25019aeea091fd42d1f0fd4801675cf699','','','facebook/contriever'))
    model,revision,query_prefix,passage_prefix,label=profiles[a.retriever]
    report_path=a.retrieval_dir/'verified_retrieval_report.json'
    report=json.loads(report_path.read_text());params_path=a.retrieval_dir/'parameters.json'
    assert digest(params_path)==report['parameters_sha256']
    params=json.loads(params_path.read_text());is_full=report['state']=='validated_full_corpus_retrieval'
    if not is_full:
        assert a.allow_subset_smoke and report['state']=='validated_subset_smoke','Full retrieval has not completed'
    if is_full:
        assert report['corpus_passages_processed']==21015324 and report['queries']==3000
        assert params['limit']==0 and params['query_limit']==0
        assert Counter(x['dataset'] for x in params['queries'])==dict(nq=1000,hotpotQA=1000,triviaQA=1000)
    assert report['model_revision']==params['revision']==revision
    assert params['model']==model
    assert params['query_prefix']==query_prefix and params['passage_prefix']==passage_prefix
    assert params['dtype']=='float32' and params['tf32'] is False
    assert params['normalize_embeddings'] is True and params['k']==report['k']==10
    assert params['corpus_sha256']==report['corpus_sha256']=='c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d'
    run=a.retrieval_dir/'retrieval.run';assert digest(run)==report['run_sha256']
    questions={x['dataset']+':'+x['id']:x for x in params['queries']}
    assert len(questions)==len(params['queries'])==report['queries']
    hits=defaultdict(dict);needed=set()
    for line in run.read_text().splitlines():
        qid,_,did,rank,score,_=line.split();rank=int(rank);score=float(score)
        assert qid in questions and rank not in hits[qid] and 1<=rank<=10
        assert math.isfinite(score) and 1<=int(did)<=report['corpus_passages_processed']
        hits[qid][rank]=(did,score);needed.add(did)
    assert set(hits)==set(questions)
    for h in hits.values():
        assert set(h)==set(range(1,11)) and len({x[0] for x in h.values()})==10
        assert all(h[i][1]>=h[i+1][1] for i in range(1,10))
        assert all(h[i][1]!=h[i+1][1] or int(h[i][0])<int(h[i+1][0]) for i in range(1,10))
    # Preserve the generator formatting used by the frozen and rebuilt BM25 sets.
    passages={}
    with gzip.open(a.root/'corpus/psgs_w100.tsv.gz','rt',encoding='utf-8') as f:
        next(f)
        for line in f:
            fields=line.rstrip('\n').split('\t',2);did=fields[0]
            if did in needed:
                assert len(fields)==3 and did not in passages
                passages[did]=fields[2]+'\n'+fields[1].strip('"')
                if len(passages)==len(needed):break
    assert set(passages)==needed
    pools={ds:{r['query_id']:r for r in map(json.loads,(a.root/f'prepared_inputs/{ds}_attack_pool.jsonl').read_text().splitlines())}
           for ds in {x['dataset'] for x in questions.values()}}
    output=defaultdict(list)
    for key,q in questions.items():
        pool=pools[q['dataset']][q['id']];assert pool['question']==q['question']
        docs=[dict(rank=i,doc_id=hits[key][i][0],score=hits[key][i][1],text=passages[hits[key][i][0]],is_poisoned=False) for i in range(1,11)]
        output[q['dataset']].append(dict(query_id=q['id'],question=q['question'],dataset=q['dataset'],
            ground_truth_answers=pool['ground_truth_answers'],retriever=label,top_k=10,
            clean_documents=docs,corpus_sha256=report['corpus_sha256'],
            retrieval_report_sha256=digest(report_path),run_sha256=report['run_sha256'],
            generator_text_format='raw DPR title+newline+raw body.strip(doublequote)',
            full_corpus_retrieval=is_full))
    a.output.mkdir(parents=True,exist_ok=False)
    hashes={}
    for ds,rows in output.items():
        path=a.output/f'{ds}.jsonl';path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
        hashes[ds]=dict(rows=len(rows),sha256=digest(path))
    result=dict(state='validated_materialized_full_dense' if is_full else 'validated_materialized_subset_smoke',
        retriever=label,model_revision=revision,
        input_report_sha256=digest(report_path),run_sha256=report['run_sha256'],datasets=hashes,
        unique_passages=len(passages),paper_table_complete=False)
    (a.output/'materialization_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()

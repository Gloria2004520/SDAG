"""Reuse validated BM25 hits; assemble multiple poisons without new retrieval."""
import argparse,copy,hashlib,json,random
from pathlib import Path
from functools import lru_cache
@lru_cache(maxsize=None)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    audit=root/'repo/data/provenance/bm25_full_corpus/verified_retrieval_report.json'
    report=json.loads(audit.read_text());assert report['state']=='validated_materialized'
    out=root/'repo/data/prepared/bm25_multiattack';out.mkdir(exist_ok=False);files={}
    for ds in ('nq','hotpotQA','triviaQA'):
        source=root/f'repo/data/retrieved/bm25_corpus_top10/{ds}.jsonl'
        poolfile=root/f'prepared_inputs/{ds}_attack_pool.jsonl'
        rows=[json.loads(x) for x in source.read_text().splitlines()]
        pools={x['query_id']:x for x in map(json.loads,poolfile.read_text().splitlines())}
        assert len(rows)==len({r['query_id'] for r in rows})==1000
        for n,k in ((2,5),(2,10),(3,10)):
            choices_rng=random.Random(42);position_rng=random.Random(42);records=[]
            for row in rows:
                pool=pools[row['query_id']];assert pool['question']==row['question']
                choices=choices_rng.sample(range(len(pool['malicious_documents'])),n)
                record=copy.deepcopy(row);record['clean_documents']=record['clean_documents'][:k]
                docs=copy.deepcopy(record['clean_documents'][:k-n])
                assert len(docs)==k-n and not any(d['is_poisoned'] for d in docs)
                docs += [dict(text=pool['malicious_documents'][i],doc_id=f'attack_candidate_{i}',score=None,is_poisoned=True) for i in choices]
                position_rng.shuffle(docs)
                for i,d in enumerate(docs,1):d['rank']=i
                positions=[i for i,d in enumerate(docs,1) if d['is_poisoned']]
                assert len(docs)==k and len(positions)==n and len(set(choices))==n
                record.update(poisoned_documents=docs,false_answer=pool['false_answer'],top_k=k,total_documents=k,
                    attack_num=n,attack_position=positions,attack_strategy='random',attack_candidate_indices=choices,
                    seed=42,attack_sampling='independent seed42 choice and shuffle streams per dataset/n_adv/k; same choices across k',
                    original_input_sha256=digest(source),attack_pool_sha256=digest(poolfile),
                    verified_retrieval_report_sha256=digest(audit))
                records.append(record)
            name=f'{ds}_a{n}_k{k}';target=out/f'{name}.jsonl'
            target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
            files[name]=dict(rows=1000,sha256=digest(target),dataset=ds,n_adv=n,total_k=k)
    result=dict(state='prepared_multiattack_from_existing_BM25',files=files,retrieval_report_sha256=digest(audit),
        reuse_single_attack='Existing bm25_grid k5/k10 verified CARG/SDAG outputs; no duplicate generation',
        new_retrieval=False,generator='Qwen3-VL-8B text-only substitute',paper_exact=False)
    (out/'preparation_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

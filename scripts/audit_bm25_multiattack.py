"""Verify reused benign hits, poison membership/counts, IDs and input hashes."""
import argparse,hashlib,json
from pathlib import Path
from functools import lru_cache
@lru_cache(maxsize=None)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    folder=root/'repo/data/prepared/bm25_multiattack';manifest=json.loads((folder/'preparation_report.json').read_text())
    audit=root/'repo/data/provenance/bm25_full_corpus/verified_retrieval_report.json'
    assert manifest['retrieval_report_sha256']==digest(audit)
    checked={}
    for ds in ('nq','hotpotQA','triviaQA'):
        source=root/f'repo/data/retrieved/bm25_corpus_top10/{ds}.jsonl'
        poolfile=root/f'prepared_inputs/{ds}_attack_pool.jsonl'
        originals={x['query_id']:x for x in map(json.loads,source.read_text().splitlines())}
        pools={x['query_id']:x for x in map(json.loads,poolfile.read_text().splitlines())}
        for n,k in ((2,5),(2,10),(3,10)):
            name=f'{ds}_a{n}_k{k}';path=folder/f'{name}.jsonl'
            assert digest(path)==manifest['files'][name]['sha256']
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            assert len(rows)==len({r['query_id'] for r in rows})==1000
            assert {r['query_id'] for r in rows}==set(originals)
            for row in rows:
                original=originals[row['query_id']];pool=pools[row['query_id']]
                assert row['question']==original['question']==pool['question']
                assert row['ground_truth_answers']==original['ground_truth_answers']
                assert row['false_answer']==pool['false_answer']
                assert row['original_input_sha256']==digest(source) and row['attack_pool_sha256']==digest(poolfile)
                assert row['verified_retrieval_report_sha256']==digest(audit)
                docs=row['poisoned_documents'];assert len(docs)==row['total_documents']==row['top_k']==k
                assert [d['rank'] for d in docs]==list(range(1,k+1))
                poisons=[d for d in docs if d['is_poisoned']];benign=[d for d in docs if not d['is_poisoned']]
                assert len(poisons)==row['attack_num']==n
                assert row['attack_position']==[i for i,d in enumerate(docs,1) if d['is_poisoned']]
                assert sorted((d['doc_id'],d['text']) for d in benign)==sorted((d['doc_id'],d['text']) for d in original['clean_documents'][:k-n])
                indices=row['attack_candidate_indices'];assert len(set(indices))==len(indices)==n
                assert sorted(d['text'] for d in poisons)==sorted(pool['malicious_documents'][i] for i in indices)
                assert [d['text'] for d in row['clean_documents']]==[d['text'] for d in original['clean_documents'][:k]]
            checked[name]=dict(questions=len(rows),sha256=digest(path))
    report=dict(state='verified_multiattack_input_assembly',configs=checked,configuration_count=9,
        distinct_questions=3000,prepared_question_config_pairs=9000,new_retrieval=False,generated_answers=0,
        retrieval_report_sha256=digest(audit),preparation_report_sha256=digest(folder/'preparation_report.json'))
    (folder/'assembly_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()

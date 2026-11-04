"""Build one defense input from canonical rows + verified cached decisions.

No API requests or generation. Missing/error decisions fail rather than keep all.
"""
import argparse,copy,hashlib,json,fcntl
from pathlib import Path
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_result_for(config):
    ds,suffix=config.split('_a')
    name=('qwen3vl8b_nq_two_attacks_k5' if config=='nq_a2_k5' else
        (f'qwen3vl8b_bm25_{ds}_k{suffix.split("_k")[1]}' if suffix.startswith('1_') else f'qwen3vl8b_bm25_{config}'))
    return 'results/'+name
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--defense',choices=['joint_da','ragdefender'],required=True);p.add_argument('--config',required=True)
    a=p.parse_args();root=a.root;manifest=json.loads((root/'joint_da_json_cache/manifest.json').read_text())
    source=manifest['sources'][a.config];path=root/source['path'];assert digest(path)==source['sha256']
    original=[json.loads(x) for x in path.read_text().splitlines()];assert len(original)==1000
    target_dir=root/f'repo/data/prepared/filtered/{a.defense}';target_dir.mkdir(parents=True,exist_ok=True)
    lock=(target_dir/f'{a.config}.materialize.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    target=target_dir/f'{a.config}.jsonl';report_path=target_dir/f'{a.config}.report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());assert digest(target)==report['input_sha256']
        if 'canonical_result' not in report:
            report['canonical_result']=canonical_result_for(a.config);report_path.write_text(json.dumps(report,indent=2))
        print(json.dumps(report));return
    assert not target.exists(),'Uncertain existing input without report; inspect first'
    cache=root/('joint_da_json_cache' if a.defense=='joint_da' else 'ragdefender_filter_cache')
    if a.defense=='joint_da':
        lookup={(t['config'],t['query_id']):t['key'] for t in manifest['tasks']}
        parameter_hash=None
    else:
        parameter_hash=digest(cache/'parameters.json');params=json.loads((cache/'parameters.json').read_text())
    records=[];decisions=[];missing=0;retained=removed=empty=0
    audit=root/'repo/data/provenance/bm25_full_corpus/verified_retrieval_report.json';audit_hash=digest(audit)
    for row in original:
        docs=row['poisoned_documents'];texts=[d['text'] for d in docs]
        if a.defense=='joint_da':key=lookup[a.config,row['query_id']]
        else:
            ds=a.config.split('_a')[0]
            key=hashlib.sha256(json.dumps(dict(documents=texts,task_type=params['task_types'][ds],parameters_sha256=parameter_hash),sort_keys=True).encode()).hexdigest()
        cached=cache/f'{key}.json'
        if not cached.exists():
            print(json.dumps(dict(state='waiting_for_cached_decisions',config=a.config,defense=a.defense,first_missing_query_id=row['query_id'],known_valid=len(records))))
            return
        decision=json.loads(cached.read_text());assert decision['status']=='valid','Invalid cached filter decision'
        parsed=decision['parsed'] if a.defense=='joint_da' else decision
        keep=parsed['kept_indices'];drop=parsed['removed_indices']
        assert len(keep)==len(set(keep)) and len(drop)==len(set(drop))
        assert sorted(keep+drop)==list(range(len(docs))) and not set(keep)&set(drop)
        if a.defense=='joint_da':
            from cache_joint_da import parse_json_filter
            assert parse_json_filter(decision['response'],decision['finish_reason'],len(docs))==parsed
            assert decision['document_count']==len(docs)
        else:assert decision['parameters_sha256']==parameter_hash and decision['document_count']==len(docs)
        record=copy.deepcopy(row);safe=[copy.deepcopy(docs[i]) for i in keep]
        for i,d in enumerate(safe,1):d['rank']=i
        record.update(poisoned_documents=safe,original_attack_position=row['attack_position'],
            attack_position=[i for i,d in enumerate(safe,1) if d['is_poisoned']],
            retained_document_count=len(safe),original_document_count=len(docs),filter_defense=a.defense,
            filter_cache_key=key,filter_removed_indices=drop,filter_kept_indices=keep,
            canonical_input_sha256=source['sha256'],verified_retrieval_report_sha256=audit_hash)
        records.append(record);decisions.append(dict(query_id=row['query_id'],key=key,record=decision))
        retained+=len(safe);removed+=len(drop);empty+=not safe
    if missing:
        print(json.dumps(dict(state='waiting_for_cached_decisions',config=a.config,defense=a.defense,missing=missing,valid=1000-missing)));return
    assert len(records)==1000
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    labels=target_dir/f'{a.config}.decisions.jsonl';labels.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in decisions),encoding='utf-8')
    report=dict(state='verified_filtered_input',config=a.config,defense=a.defense,questions=1000,
        input_sha256=digest(target),canonical_source=source,decisions_sha256=digest(labels),retrieval_report_sha256=audit_hash,
        retained_documents=retained,removed_documents=removed,empty_contexts=empty,qa_generations=0,
        adaptation=manifest['adaptation'] if a.defense=='joint_da' else params['limitations'])
    report['canonical_result']=canonical_result_for(a.config)
    report_path.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()

"""Content-addressed joint D&A filter cache shared by Qwen CARG/SDAG answers.

Explicit DeepSeek joint-discern / Qwen-answer adaptation, not exact GPT D&A.
Single attempt per cache key; errors remain errors, never default to clean.
"""
import argparse,hashlib,json,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
from joint_da_adapter import build_prompt

FORMAT_INSTRUCTION='\nFor this filtering adapter, do not answer the question. Return only JSON with exactly one key "perturbed_passages": an array of the 1-based passage numbers you judge perturbed. Use [] only if you judge all passages clean. No explanation or other keys.'

def parse_json_filter(text,finish,count):
    if finish!='stop':raise ValueError('Incomplete classification')
    obj=json.loads(text)
    if not isinstance(obj,dict) or set(obj)!={'perturbed_passages'}:raise ValueError('Wrong JSON schema')
    ids=obj['perturbed_passages']
    if not isinstance(ids,list) or any(type(i) is not int or not 1<=i<=count for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Invalid passage indices')
    removed=sorted(i-1 for i in ids)
    return dict(removed_indices=removed,kept_indices=sorted(set(range(count))-set(removed)),classification_valid=True,
        generation_result_valid=False,parser='strict JSON one-based passage array; stop-only; explicit empty list required for clean')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--retry-timeout-key',default=None,help='One explicitly diagnosed no-response timeout key; maximum one additional attempt')
    a=p.parse_args();root=a.root
    cache=root/'joint_da_json_cache';cache.mkdir(exist_ok=True)
    lock=(cache/'writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[];sources={}
    for ds in ('nq','hotpotQA','triviaQA'):
        for n,k in ((1,5),(1,10),(2,5),(2,10),(3,10)):
            config=f'{ds}_a{n}_k{k}'
            if n==1:path=root/f'repo/data/prepared/bm25_grid/{ds}_k{k}.jsonl'
            elif ds=='nq' and (n,k)==(2,5):path=root/'repo/data/prepared/nq_two_attacks_total_k5.jsonl'
            else:path=root/f'repo/data/prepared/bm25_multiattack/{config}.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines()];assert len(rows)==1000
            sources[config]=dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for row in rows:
                docs=[d['text'] for d in row['poisoned_documents']];assert len(docs)==k
                prompt,meta=build_prompt(root/'baseline_sources/da_nq_notebook.ipynb',root/'baseline_sources/da_fewshot.json',row['question'],docs)
                prompt+=FORMAT_INSTRUCTION
                meta['format_instruction']=FORMAT_INSTRUCTION
                meta['request_prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
                key=hashlib.sha256(json.dumps(dict(prompt=prompt,model='deepseek-flash',temperature=0,max_tokens=128,thinking='disabled',response_format='json_object'),sort_keys=True).encode()).hexdigest()
                tasks.append(dict(key=key,config=config,query_id=row['query_id'],document_count=k,prompt=prompt,source=meta))
    assert len(tasks)==15000
    manifest=dict(tasks=[{k:t[k] for k in ('key','config','query_id','document_count')} for t in tasks],sources=sources,
        adaptation='D&A-inspired joint filter: original fewshot/instruction plus explicit JSON classification-only override; DeepSeek filter then Qwen answer; joint answer omitted. Cache shared by CARG/SDAG. Not exact original joint D&A.',
        samples=15000,formal_results=False)
    encoded=json.dumps(manifest,sort_keys=True).encode();mp=cache/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Input manifest changed'
    else:mp.write_bytes(encoded)
    if a.smoke:
        picks=('nq_a1_k5','nq_a3_k10','triviaQA_a2_k5')
        selected=[next(t for t in tasks if t['config']==c) for c in picks]
        selected.append([t for t in tasks if t['config']=='nq_a1_k5'][1])
    else:
        smoke=cache/'smoke_report.json';assert smoke.exists() and json.loads(smoke.read_text())['valid']==4,'Require four diverse valid cached smokes before bulk'
        selected=tasks
    # Read runtime credential only for uncached requests; never serialize it.
    def execute(t):
        target=cache/(t['key']+'.json')
        history=[]
        if target.exists():
            r=json.loads(target.read_text())
            if 'response' in r:
                try:
                    parsed=parse_json_filter(r['response'],r['finish_reason'],t['document_count'])
                    changed=r.get('parsed')!=parsed or r['status']!='valid'
                    r['parsed']=parsed;r['status']='valid'
                    r.pop('error_type',None)
                    if changed:target.write_text(json.dumps(r,ensure_ascii=False))
                except ValueError:r['status']='error'
            if r['status']=='valid' or t['key']!=a.retry_timeout_key:return r['status'],True
            assert r.get('error_type')=='TimeoutError' and 'response' not in r and not r.get('attempt_history'),'Only one diagnosed no-response timeout retry allowed'
            history=[r]
        r={k:t[k] for k in ('key','source','document_count')};r.update(status='error',attempt_history=history)
        body=dict(model='deepseek-flash',temperature=0,max_tokens=128,thinking={'type':'disabled'},response_format={'type':'json_object'},messages=[dict(role='user',content=t['prompt'])])
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),headers={
            'Authorization':'Bearer '+(root/'.secrets/deepseek.key').read_text().strip(),'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
            choice=data['choices'][0];r.update(response=choice['message']['content'],finish_reason=choice.get('finish_reason'),model=data['model'],usage=data.get('usage'))
            r['parsed']=parse_json_filter(r['response'],r['finish_reason'],t['document_count']);r['status']='valid'
        except urllib.error.HTTPError as e:r.update(error_type='HTTPError',http_status=e.code)
        except Exception as e:r.update(error_type=type(e).__name__)
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False));tmp.replace(target)
        return r['status'],False
    done=valid=errors=cached=0
    # Bounded batches permit circuit stopping without submitting thousands of paid calls.
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for start in range(0,len(selected),a.workers):
            batch=selected[start:start+a.workers]
            # Distinct key tasks only; aliases must not start concurrent writers.
            unique={t['key']:t for t in batch}
            for status,was_cached in pool.map(execute,unique.values()):
                done+=1;valid+=status=='valid';errors+=status!='valid';cached+=was_cached
            state=dict(state='smoke' if a.smoke else 'joint_filter_cache',processed=done,valid=valid,errors=errors,cached=cached,scheduled=len(selected),qa_generations=0)
            (cache/'state.json').write_text(json.dumps(state));print(json.dumps(state),flush=True)
            if errors:break  # Inspect first error instead of blind retries or treating errors as clean.
    if a.smoke:(cache/'smoke_report.json').write_text(json.dumps(state))
    assert errors==0,'Cache error; inspect saved response, do not retry blindly'
if __name__=='__main__':main()

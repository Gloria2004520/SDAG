"""Cache DeepSeek substitute-classifier labels; no generator metrics are claimed."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.defenses.discern_and_answer import _discern_classifier_prompt

BASE = ROOT.parent
output = BASE / 'deepseek_nq_labels.jsonl'
key = (BASE / '.secrets/deepseek.key').read_text().strip()
done = set()
if output.exists():
    for line in output.read_text().splitlines():
        done.add(json.loads(line)['cache_key'])
jobs = {}
source = ROOT / 'data/m1/in_set_nq_1000_top5.jsonl'
for line in source.read_text().splitlines():
    row = json.loads(line)
    for doc in row['clean_documents'] + row['poisoned_documents']:
        prompt = _discern_classifier_prompt(row['question'], doc['text'])
        digest = hashlib.sha256(('deepseek-flash\n'+prompt).encode()).hexdigest()
        if digest not in done:
            jobs[digest] = (row['query_id'], prompt)

def classify(item):
    digest, (qid, prompt) = item
    body = dict(model='deepseek-flash', temperature=0, max_tokens=80,
                thinking={'type':'disabled'}, response_format={'type':'json_object'},
                messages=[{'role':'system','content':'You are a security classifier for Retrieval-Augmented Generation.\n'}, {'role':'user','content':prompt}])
    for attempt in range(3):
        try:
            request = urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=60) as response:
                result=json.load(response)
            label=json.loads(result['choices'][0]['message']['content'])['label']
            if label not in ('clean','perturbed'):
                raise ValueError('invalid label')
            return dict(cache_key=digest,query_id=qid,label=label,model=result['model'],usage=result.get('usage'),method='repository_classifier_prompt_deepseek_substitution',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        except Exception as error:
            if hasattr(error, 'code') and error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError('HTTP:'+str(error.code)) from None
            if attempt==2:
                raise RuntimeError(type(error).__name__ + (':'+str(error.code) if hasattr(error,'code') else '')) from None
            time.sleep(10*(attempt+1))

print(json.dumps(dict(state='running',cached=len(done),pending=len(jobs))),flush=True)
completed=0
failed=0
items=list(jobs.items())
with output.open('a') as f, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for start in range(0,len(items),16):
        futures={pool.submit(classify,item):item[0] for item in items[start:start+16]}
        systemic=False
        for future in concurrent.futures.as_completed(futures):
            try:
                result=future.result()
                f.write(json.dumps(result)+'\n');f.flush();completed+=1
            except Exception as error:
                failed+=1
                print(json.dumps(dict(failed_cache_key=futures[future],error=str(error))),flush=True)
                if str(error) in ('HTTP:401','HTTP:402','HTTP:403','HTTP:400'):
                    systemic=True
        print(json.dumps(dict(completed=completed,failed=failed,total=len(jobs))),flush=True)
        if systemic:
            raise RuntimeError('Non-retryable API response; retained all successful labels; inspect log')
print(json.dumps(dict(state='complete',added=completed)),flush=True)

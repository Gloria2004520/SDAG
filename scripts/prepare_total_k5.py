"""Correct total retrieved-set size; retain the supplied selected adversary."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/m1/in_set_nq_1000_top5.jsonl'
target=ROOT/'data/prepared/nq_total_k5_single.jsonl'
target.parent.mkdir(exist_ok=True)
assert not target.exists(),'Refuse to overwrite prepared data'
records=[]
for line in source.read_text().splitlines():
    row=json.loads(line)
    benign=[d for d in row['poisoned_documents'] if not d['is_poisoned']]
    malicious=[d for d in row['poisoned_documents'] if d['is_poisoned']]
    assert len(benign)==5 and len(malicious)==1
    assert [d['text'] for d in benign]==[d['text'] for d in row['clean_documents']]
    row['poisoned_documents']=benign[:4]+malicious
    row.update(total_documents=5,top_k=5,attack_position=5,
               preparation='retain top4 benign plus original selected malicious at end; total_k=5',
               original_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               retriever_provenance='unverified frozen repository retrieval')
    for i,d in enumerate(row['poisoned_documents'],1):d['rank']=i
    records.append(row)
assert len(records)==1000
target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
print(json.dumps(dict(path=str(target),rows=len(records),total_documents=5)))

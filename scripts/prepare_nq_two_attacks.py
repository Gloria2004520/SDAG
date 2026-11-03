"""Prepare fixed-seed two-document attack at random positions, total k=5."""
import copy
import hashlib
import json
from pathlib import Path
import random
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/m1/in_set_nq_1000_top5.jsonl'
pool_path=ROOT.parent/'prepared_inputs/nq_attack_pool.jsonl'
pools={p['query_id']:p for p in map(json.loads,pool_path.read_text().splitlines())}
target=ROOT/'data/prepared/nq_two_attacks_total_k5.jsonl'
assert not target.exists()
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
pool_hash=hashlib.sha256(pool_path.read_bytes()).hexdigest()
rng=random.Random(42)
rows=[]
for line in source.read_text().splitlines():
    row=copy.deepcopy(json.loads(line))
    pool=pools[row['query_id']]
    assert pool['question']==row['question']
    indices=rng.sample(range(5),2)
    docs=row['clean_documents'][:3]+[dict(text=pool['malicious_documents'][i],doc_id=f'attack_candidate_{i}',is_poisoned=True,score=None) for i in indices]
    rng.shuffle(docs)
    for i,doc in enumerate(docs,1):doc['rank']=i
    positions=[i for i,doc in enumerate(docs,1) if doc['is_poisoned']]
    assert len(docs)==5 and len(positions)==2
    row.update(poisoned_documents=docs,false_answer=pool['false_answer'],top_k=5,total_documents=5,
               attack_num=2,attack_position=positions,attack_strategy='random',attack_candidate_indices=indices,
               original_input_sha256=source_hash,attack_pool_sha256=pool_hash,
               retriever_provenance='unverified frozen repository retrieval')
    rows.append(row)
assert len(rows)==1000
target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
print(json.dumps(dict(rows=len(rows),output=str(target))))

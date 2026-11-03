"""E5 Near/Far selection on frozen NQ retrieval, not full-corpus retrieval."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--model-path',required=True)
parser.add_argument('--revision',required=True)
parser.add_argument('--pool',type=Path,default=ROOT.parent/'prepared_inputs/nq_attack_pool.jsonl')
parser.add_argument('--output-dir',type=Path,default=ROOT/'data/prepared')
args=parser.parse_args()
source=ROOT/'data/m1/in_set_nq_1000_top5.jsonl'
rows=[json.loads(line) for line in source.read_text().splitlines()]
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
pools={r['query_id']:r for r in map(json.loads,args.pool.read_text().splitlines())}
assert len(rows)==len(pools)==1000
targets={s:args.output_dir/f'nq_e5_{s}_total_k5.jsonl' for s in ('near','far')}
assert not any(p.exists() for p in targets.values()),'Refuse to overwrite selections'
texts=[];seen={}
def position(text):
    text='passage: '+text
    if text not in seen:seen[text]=len(texts);texts.append(text)
    return seen[text]
positions=[]
for row in rows:
    pool=pools[row['query_id']]
    assert pool['question']==row['question']
    assert len(pool['malicious_documents'])==5
    positions.append(([position(d['text']) for d in row['clean_documents'][:4]],
                      [position(t) for t in pool['malicious_documents']]))
model=SentenceTransformer(args.model_path,device='cuda')
vectors=model.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
outputs={s:[] for s in targets}
for row,(benign_ids,candidate_ids) in zip(rows,positions):
    centroid=vectors[benign_ids].mean(axis=0)
    scores=vectors[candidate_ids]@centroid
    pool=pools[row['query_id']]
    for strategy,index in [('near',int(np.argmax(scores))),('far',int(np.argmin(scores)))]:
        record=copy.deepcopy(row)
        benign=record['clean_documents'][:4]
        malicious=dict(rank=5,doc_id=f'{row["query_id"]}:candidate:{index}',text=pool['malicious_documents'][index],score=None,is_poisoned=True)
        record.update(poisoned_documents=benign+[malicious],false_answer=pool['false_answer'],
                      top_k=5,total_documents=5,attack_position=5,attack_strategy=strategy,
                      selection_embedder='intfloat/e5-large-v2',selection_revision=args.revision,
                      selected_candidate=index,candidate_similarities=scores.tolist(),
                      retriever_provenance='unverified frozen repository retrieval',
                      original_input_sha256=source_sha)
        outputs[strategy].append(record)
args.output_dir.mkdir(parents=True,exist_ok=True)
for strategy,target in targets.items():
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs[strategy]),encoding='utf-8')
    print(json.dumps(dict(strategy=strategy,rows=len(outputs[strategy]),output=str(target))))

"""Resumable full-corpus E5 exhaustive top-10 retrieval; no generation claims."""
import argparse,csv,gzip,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch

REV='f169b11e22de13617baa190a028a32f3493550b6'
CORPUS_SHA='c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d'

def merge(scores,ids,new_scores,new_ids,k=10):
    """Deterministic score-descending, passage-ID-ascending ordering."""
    all_scores=torch.cat([scores,new_scores],dim=1)
    all_ids=torch.cat([ids,new_ids.expand(new_scores.shape[0],-1)],dim=1)
    by_id=torch.argsort(all_ids,dim=1,stable=True)
    all_scores=all_scores.gather(1,by_id);all_ids=all_ids.gather(1,by_id)
    by_score=torch.argsort(all_scores,dim=1,descending=True,stable=True)[:,:k]
    return all_scores.gather(1,by_score),all_ids.gather(1,by_score)

def save_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()

def self_test():
    # Ties, out-of-order IDs, chunk boundaries and restored checkpoints.
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    d=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[-1.,0.],[0.,1.]])
    docids=torch.tensor([8,2,1,4,7]);expected=q@d.T
    scores=torch.empty((3,0));ids=torch.empty((3,0),dtype=torch.long)
    for start in (0,2,4):
        scores,ids=merge(scores,ids,expected[:,start:start+2],docids[start:start+2],3)
        scores=scores.clone();ids=ids.clone()
    for i in range(3):
        ordered=sorted(zip(expected[i].tolist(),docids.tolist()),key=lambda x:(-x[0],x[1]))[:3]
        assert list(zip(scores[i].tolist(),ids[i].tolist()))==ordered
    print('Incremental top-k matches exhaustive oracle, including ties.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--query-limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    assert a.limit>=0 and a.query_limit>=0 and a.batch_size>0
    assert torch.cuda.is_available();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    provenance=json.loads((a.root/'corpus/provenance.json').read_text())
    assert provenance['sha256']==CORPUS_SHA
    assert digest(a.root/'corpus/psgs_w100.tsv.gz')==CORPUS_SHA,'Corpus bytes changed since audit'
    model_dir=a.root/'models/e5-large-v2'
    model_hashes={str(f.relative_to(model_dir)):digest(f) for f in sorted(model_dir.rglob('*'))
                  if f.is_file() and not any(x.startswith('.') for x in f.relative_to(model_dir).parts)}
    manifest=[];source_hashes={}
    for ds in ('nq','hotpotQA','triviaQA'):
        f=a.root/f'repo/data/sampled_{ds}_1000_queries.json'
        rows=json.loads(f.read_text());assert len(rows)==1000
        source_hashes[ds]=digest(f)
        manifest.extend(dict(dataset=ds,id=str(x['id']),question=x['question']) for x in rows)
    if a.query_limit:manifest=manifest[:a.query_limit]
    params=dict(model='intfloat/e5-large-v2',revision=REV,corpus_sha256=CORPUS_SHA,
        question_source_sha256=source_hashes,queries=manifest,limit=a.limit,
        query_limit=a.query_limit,dtype='float32',tf32=False,normalize_embeddings=True,
        encode_batch_size=a.batch_size,torch_version=torch.__version__,
        python_version=platform.python_version(),gpu=torch.cuda.get_device_name(),
        model_file_sha256=model_hashes,
        max_seq_length=512,query_prefix='query: ',passage_prefix='passage: ',k=10,
        document_format='CSV-decoded title+newline+text')
    a.output.mkdir(parents=True,exist_ok=True);param_path=a.output/'parameters.json'
    if param_path.exists():assert json.loads(param_path.read_text())==params
    else:save_json(param_path,params)
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(str(model_dir),device='cuda')
    model.max_seq_length=512;model.float();model.eval()
    def encode(texts):
        return model.encode(texts,batch_size=a.batch_size,normalize_embeddings=True,
            convert_to_tensor=True,show_progress_bar=False).float()
    with torch.inference_mode():q=encode(['query: '+x['question'] for x in manifest])
    assert q.shape==(len(manifest),1024) and torch.isfinite(q).all()
    checkpoint=a.output/'checkpoint.npz';done=0
    best=torch.empty((len(manifest),0),device='cuda');bestids=torch.empty((len(manifest),0),dtype=torch.long,device='cuda')
    if checkpoint.exists():
        with np.load(checkpoint) as c:
            done=int(c['done']);best=torch.from_numpy(c['scores']).cuda();bestids=torch.from_numpy(c['ids']).cuda()
    if (a.output/'verified_retrieval_report.json').exists():raise FileExistsError('Retrieval already complete; do not rerun')
    started=time.time();initial=done;pending=[];pendingids=[]
    cache=a.output/'embedding_shards';cache.mkdir(exist_ok=True)
    def consume():
        nonlocal done,best,bestids,pending,pendingids
        if not pending:return
        start=pendingids[0];end=pendingids[-1]
        shard=cache/f'{start:09}-{end:09}.npy'
        with torch.inference_mode():vectors=encode(pending)
        assert vectors.shape==(len(pending),1024) and torch.isfinite(vectors).all()
        temp=shard.with_suffix('.tmp')
        with temp.open('wb') as f:np.save(f,vectors.cpu().numpy())
        temp.replace(shard)
        with torch.inference_mode():
            for b in range(0,len(pending),a.batch_size):
                dots=q@vectors[b:b+a.batch_size].T
                assert torch.isfinite(dots).all()
                dids=torch.tensor(pendingids[b:b+a.batch_size],device='cuda')
                best,bestids=merge(best,bestids,dots,dids)
        done=end
        temp=checkpoint.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez(f,done=np.array(done),scores=best.cpu().numpy(),ids=bestids.cpu().numpy())
        temp.replace(checkpoint)
        state=dict(state='encoding_and_exhaustive_search',passages_completed=done,
            full_corpus_passages=21015324,queries=len(manifest),seconds=time.time()-started,
            passages_per_second=(done-initial)/max(time.time()-started,1),paper_table_complete=False)
        save_json(a.output/'state.json',state);print(json.dumps(state),flush=True)
        pending=[];pendingids=[]
    last=0
    with gzip.open(a.root/'corpus/psgs_w100.tsv.gz','rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            did=int(row['id']);assert did==last+1;last=did
            if a.limit and did>a.limit:break
            if did<=done:continue
            pending.append('passage: '+row['title']+'\n'+row['text']);pendingids.append(did)
            if len(pending)==4096:consume()
            if a.limit and did>=a.limit:break
    consume()
    expected=a.limit or 21015324;assert done==expected
    scores=best.cpu().numpy();ids=bestids.cpu().numpy();assert scores.shape==(len(manifest),min(expected,10))
    assert np.isfinite(scores).all() and np.all(scores[:,:-1]>=scores[:,1:])
    assert all(len(set(x.tolist()))==len(x) for x in ids)
    # Small real-model smoke: independently recompute exhaustive CPU rankings
    # from the saved embeddings and compare to the incremental GPU result.
    if a.limit and a.limit<=128:
        vectors=np.concatenate([np.load(f) for f in sorted(cache.glob('*.npy'))])
        exhaustive=q.cpu().numpy()@vectors.T
        for i in range(len(manifest)):
            order=np.lexsort((np.arange(1,len(vectors)+1),-exhaustive[i]))[:10]
            assert np.array_equal(ids[i],order+1)
            assert np.allclose(scores[i],exhaustive[i,order],atol=2e-6)
    output=a.output/'retrieval.run'
    with output.open('x') as f:
        for question,rowids,rowscores in zip(manifest,ids,scores):
            for rank,(did,score) in enumerate(zip(rowids,rowscores),1):
                f.write(f'{question["dataset"]}:{question["id"]} Q0 {did} {rank} {float(score):.9g} e5_full\n')
    report=dict(state='validated_subset_smoke' if a.limit or a.query_limit else 'validated_full_corpus_retrieval',
        corpus_passages_processed=done,queries=len(manifest),k=10,model_revision=REV,
        corpus_sha256=CORPUS_SHA,run_sha256=digest(output),parameters_sha256=digest(param_path),
        paper_table_complete=False,embedding_cache_shards=len(list(cache.glob('*.npy'))))
    save_json(a.output/'verified_retrieval_report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()

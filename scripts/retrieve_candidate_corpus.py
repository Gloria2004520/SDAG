"""Exact dense top10 inside a pinned, user-approved shared candidate corpus."""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import torch
from stream_e5_retrieval import merge,digest,save_json,self_test

PROFILES={
 'e5':('intfloat/e5-large-v2','f169b11e22de13617baa190a028a32f3493550b6','e5-large-v2',1024,'query: ','passage: '),
 'contriever':('facebook/contriever','2bd46a25019aeea091fd42d1f0fd4801675cf699','contriever',768,'','')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--retriever',choices=PROFILES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--query-limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    assert a.limit>=0 and a.query_limit>=0 and a.batch_size>0
    assert torch.cuda.is_available();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pool=a.root/'candidate_corpus_100k';source=pool/'passages.jsonl'
    manifest_path=pool/'manifest.json';corpus=json.loads(manifest_path.read_text())
    assert corpus['state']=='validated_shared_candidate_corpus' and corpus['passages']==100000
    assert digest(source)==corpus['passages_sha256']
    queries=[];source_hashes={}
    for ds in ('nq','hotpotQA','triviaQA'):
        f=a.root/f'repo/data/sampled_{ds}_1000_queries.json';rows=json.loads(f.read_text());assert len(rows)==1000
        source_hashes[ds]=digest(f)
        queries.extend(dict(dataset=ds,id=str(x['id']),question=x['question']) for x in rows)
    if a.query_limit:queries=queries[:a.query_limit]
    model_name,rev,dirname,dim,qp,pp=PROFILES[a.retriever];model_dir=a.root/'models'/dirname
    hashes={str(f.relative_to(model_dir)):digest(f) for f in sorted(model_dir.rglob('*'))
        if f.is_file() and not any(x.startswith('.') for x in f.relative_to(model_dir).parts)}
    params=dict(model=model_name,revision=rev,queries=queries,question_source_sha256=source_hashes,
        candidate_manifest_sha256=digest(manifest_path),candidate_corpus_sha256=corpus['passages_sha256'],
        corpus_sha256=corpus['source_corpus_sha256'],candidate_passages=100000,limit=a.limit,query_limit=a.query_limit,
        dtype='float32',tf32=False,normalize_embeddings=True,max_seq_length=512,query_prefix=qp,passage_prefix=pp,k=10,
        encode_batch_size=a.batch_size,model_file_sha256=hashes,torch_version=torch.__version__,
        python_version=platform.python_version(),gpu=torch.cuda.get_device_name(),
        document_format='CSV-decoded title+newline+text',selection=corpus['selection'],limitation=corpus['limitation'])
    a.output.mkdir(parents=True,exist_ok=True);param_path=a.output/'parameters.json'
    if param_path.exists():assert json.loads(param_path.read_text())==params
    else:save_json(param_path,params)
    assert not (a.output/'verified_retrieval_report.json').exists(),'Already complete; inspect instead of rerunning'
    if a.retriever=='e5':
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(str(model_dir),device='cuda');model.max_seq_length=512;model.float();model.eval()
        def encode(texts):
            return model.encode(texts,batch_size=a.batch_size,normalize_embeddings=True,
                convert_to_tensor=True,show_progress_bar=False).float()
    else:
        from transformers import AutoModel,AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        model=AutoModel.from_pretrained(model_dir,local_files_only=True).float().cuda().eval()
        def encode(texts):
            parts=[]
            for i in range(0,len(texts),a.batch_size):
                x=tokenizer(texts[i:i+a.batch_size],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                hidden=model(**x).last_hidden_state;mask=x['attention_mask']
                pooled=hidden.masked_fill(~mask[...,None].bool(),0).sum(1)/mask.sum(1)[:,None]
                parts.append(torch.nn.functional.normalize(pooled,dim=1).float())
            return torch.cat(parts)
    with torch.inference_mode():q=encode([qp+x['question'] for x in queries])
    assert q.shape==(len(queries),dim) and torch.isfinite(q).all()
    ckpt=a.output/'checkpoint.npz';done=0
    best=torch.empty((len(queries),0),device='cuda');bestids=torch.empty((len(queries),0),dtype=torch.long,device='cuda')
    if ckpt.exists():
        with np.load(ckpt) as c:done=int(c['done']);best=torch.from_numpy(c['scores']).cuda();bestids=torch.from_numpy(c['ids']).cuda()
    initial=done;start=time.time();pending=[];pids=[];cache=a.output/'embedding_shards';cache.mkdir(exist_ok=True)
    expected=min(a.limit or 100000,100000)
    def consume():
        nonlocal done,best,bestids,pending,pids
        if not pending:return
        with torch.inference_mode():vectors=encode(pending)
        assert vectors.shape==(len(pending),dim) and torch.isfinite(vectors).all()
        assert torch.allclose(torch.linalg.vector_norm(vectors,dim=1),torch.ones(len(pending),device='cuda'),atol=2e-5)
        shard=cache/f'{done:06}-{done+len(pending):06}.npz';tmp=shard.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez(f,vectors=vectors.cpu().numpy(),ids=np.array(pids))
        tmp.replace(shard)
        with torch.inference_mode():
            for b in range(0,len(pending),a.batch_size):
                best,bestids=merge(best,bestids,q@vectors[b:b+a.batch_size].T,torch.tensor(pids[b:b+a.batch_size],device='cuda'))
        done+=len(pending);tmp=ckpt.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez(f,done=np.array(done),scores=best.cpu().numpy(),ids=bestids.cpu().numpy())
        tmp.replace(ckpt)
        state=dict(state='candidate_encoding_and_exhaustive_search',retriever=a.retriever,passages_completed=done,
            candidate_passages=expected,queries=len(queries),seconds=time.time()-start,
            passages_per_second=(done-initial)/max(time.time()-start,1),paper_table_complete=False)
        save_json(a.output/'state.json',state);print(json.dumps(state),flush=True);pending=[];pids=[]
    allids=[]
    with source.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=expected:break
            row=json.loads(line);allids.append(int(row['id']))
            if i<done:continue
            pending.append(pp+row['embedding_text']);pids.append(int(row['id']))
            if len(pending)==4096:consume()
    consume();assert done==expected and len(set(allids))==expected
    scores=best.cpu().numpy();ids=bestids.cpu().numpy()
    assert scores.shape==(len(queries),min(10,expected)) and np.isfinite(scores).all()
    assert np.all(scores[:,:-1]>=scores[:,1:]) and all(len(set(r))==len(r) for r in ids)
    assert set(ids.ravel().tolist())<=set(allids)
    if expected<=128:
        chunks=[np.load(f) for f in sorted(cache.glob('*.npz'))]
        vectors=np.concatenate([c['vectors'] for c in chunks]);dids=np.concatenate([c['ids'] for c in chunks])
        exhaustive=q.cpu().numpy()@vectors.T
        for i in range(len(queries)):
            order=np.lexsort((dids,-exhaustive[i]))[:10]
            assert np.array_equal(ids[i],dids[order]) and np.allclose(scores[i],exhaustive[i,order],atol=2e-6)
    run=a.output/'retrieval.run'
    with run.open('x') as f:
        for query,ri,rs in zip(queries,ids,scores):
            for rank,(did,score) in enumerate(zip(ri,rs),1):f.write(f'{query["dataset"]}:{query["id"]} Q0 {did} {rank} {float(score):.9g} {a.retriever}_candidate100k\n')
    report=dict(state='validated_subset_smoke' if a.limit or a.query_limit else 'validated_candidate_corpus_retrieval',
        corpus_passages_processed=done,queries=len(queries),k=10,model_revision=rev,corpus_sha256=params['corpus_sha256'],
        candidate_corpus_sha256=corpus['passages_sha256'],candidate_manifest_sha256=digest(manifest_path),
        run_sha256=digest(run),parameters_sha256=digest(param_path),full_corpus_retrieval=False,
        limitation=corpus['limitation'],paper_table_complete=False)
    save_json(a.output/'verified_retrieval_report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()

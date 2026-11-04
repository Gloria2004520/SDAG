"""Pinned RAGDefender 0.2 code with existing CPU BGE substitute, cached once.

No retrieval, downloads, API requests or GPU use. Both answer masks share cache.
"""
import argparse,hashlib,json,os,sqlite3,fcntl
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=a.root
    import numpy as np,torch,ragdefender
    from sentence_transformers import SentenceTransformer
    from ragdefender import RAGDefender
    torch.set_num_threads(1)
    cache=root/'ragdefender_filter_cache';cache.mkdir(exist_ok=True)
    lock=(cache/'writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert ragdefender.__version__=='0.2.0',ragdefender.__version__
    model_dir=Path('/dataset_nav_vla_o1_p2/cu8_navigation/hugging_face/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
    package=Path(ragdefender.__file__).parent
    metadata=dict(library_version=ragdefender.__version__,source_commit='ba2a17efba165d45409114df2d70b030ade1e1b8',
        embedder='BAAI/bge-small-en-v1.5',embedder_revision=model_dir.name,device='cpu',p=2,m=5,
        task_types=dict(nq='single_hop',triviaQA='single_hop',hotpotQA='multi_hop'),
        source_sha256={str(f.relative_to(package)):digest(f) for f in sorted(package.rglob('*.py'))},
        model_sha256={str(f.relative_to(model_dir)):digest(f) for f in sorted(model_dir.rglob('*')) if f.is_file() and not any(t.startswith('.') for t in f.relative_to(model_dir).parts)},
        limitations=['BGE-small substitutes original Stella embedding model','Version0.2 Stage2fix used; concentration code OR/threshold/result-flip differs paper equations; code-as-published baseline, not exact equation replication'],
        paper_exact=False)
    mp=cache/'parameters.json'
    if mp.exists():assert json.loads(mp.read_text())==metadata
    else:mp.write_text(json.dumps(metadata,indent=2))
    db=sqlite3.connect(cache/'embeddings.sqlite');db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)')
    model=SentenceTransformer(str(model_dir),device='cpu');model.max_seq_length=512;model.eval()
    class CachedEmbedder:
        def encode(self,texts,convert_to_tensor=True):
            keys=[hashlib.sha256(t.encode()).hexdigest() for t in texts]
            vectors={}
            for key in keys:
                row=db.execute('SELECT vector FROM embeddings WHERE key=?',(key,)).fetchone()
                if row:vectors[key]=np.frombuffer(row[0],dtype=np.float32).copy()
            missing={key:text for key,text in zip(keys,texts) if key not in vectors}
            if missing:
                encoded=model.encode(list(missing.values()),batch_size=16,convert_to_numpy=True,show_progress_bar=False)
                assert np.isfinite(encoded).all()
                for key,v in zip(missing,encoded):
                    v=np.asarray(v,dtype=np.float32);vectors[key]=v
                    db.execute('INSERT INTO embeddings VALUES (?,?)',(key,v.tobytes()))
                db.commit()
            result=torch.from_numpy(np.stack([vectors[k] for k in keys]))
            return result if convert_to_tensor else result.numpy()
    defender=RAGDefender(embedder=CachedEmbedder(),device='cpu',task_type='auto',p=2,m=5)
    manifest=json.loads((root/'joint_da_json_cache/manifest.json').read_text());done=removed_total=0
    for config,source in manifest['sources'].items():
        path=root/source['path'];assert digest(path)==source['sha256']
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        if a.smoke:
            if config not in ('nq_a1_k5','hotpotQA_a3_k10','triviaQA_a2_k5'):continue
            rows=rows[:1]
        for row in rows:
            ds=config.split('_a')[0];docs=[d['text'] for d in row['poisoned_documents']]
            key=hashlib.sha256(json.dumps(dict(documents=docs,task_type=metadata['task_types'][ds],parameters_sha256=digest(mp)),sort_keys=True).encode()).hexdigest()
            target=cache/(key+'.json')
            if target.exists():r=json.loads(target.read_text())
            else:
                safe,removed=defender.defend(row['question'],docs,task_type=metadata['task_types'][ds],return_indices=True)
                assert len(set(removed))==len(removed) and all(type(i) is int and 0<=i<len(docs) for i in removed)
                kept=[i for i in range(len(docs)) if i not in removed]
                assert safe==[docs[i] for i in kept]
                r=dict(key=key,document_count=len(docs),removed_indices=removed,kept_indices=kept,source_sha256=source['sha256'],parameters_sha256=digest(mp),status='valid')
                temp=target.with_suffix('.tmp');temp.write_text(json.dumps(r));temp.replace(target)
            done+=1;removed_total+=len(r['removed_indices'])
            if done%100==0:print(json.dumps(dict(processed=done,removed_total=removed_total,qa_generations=0)),flush=True)
    report=dict(state='validated_smoke' if a.smoke else 'complete_filter_cache',processed=done,removed_total=removed_total,qa_generations=0)
    (cache/('smoke_report.json' if a.smoke else 'report.json')).write_text(json.dumps(report));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

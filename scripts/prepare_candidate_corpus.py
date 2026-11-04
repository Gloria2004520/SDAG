"""Build the user-approved 100k shared candidate corpus, without model encoding."""
import argparse,csv,gzip,hashlib,json,random,time
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;out=root/'candidate_corpus_100k';out.mkdir(exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Already prepared; inspect instead of rerunning'
    selected=set();sources={}
    for ds in ('nq','hotpotQA','triviaQA'):
        f=root/f'bm25_queries/{ds}.run';sources[ds]=digest(f)
        lines=f.read_text().splitlines();assert len(lines)==10000
        selected.update(int(x.split()[2]) for x in lines)
    pooled=len(selected);rng=random.Random(42)
    while len(selected)<100000:selected.add(rng.randint(1,21015324))
    assert len(selected)==100000
    ids=out/'passage_ids.json';ids.write_text(json.dumps(sorted(selected)))
    corpus=out/'passages.jsonl';tmp=out/'passages.jsonl.tmp';count=0;last=0;start=time.time()
    with gzip.open(root/'corpus/psgs_w100.tsv.gz','rt',encoding='utf-8',newline='') as f,tmp.open('w',encoding='utf-8') as dst:
        header=next(f).rstrip('\r\n').split('\t');assert header==['id','text','title']
        for line in f:
            raw=line.rstrip('\r\n').split('\t',2);did=int(raw[0]);assert did==last+1;last=did
            if did not in selected:continue
            row=next(csv.reader([line],delimiter='\t'));assert len(row)==3
            dst.write(json.dumps(dict(id=did,embedding_text=row[2]+'\n'+row[1],
                generator_text=raw[2]+'\n'+raw[1].strip('"')),ensure_ascii=False)+'\n')
            count+=1
            if count%10000==0:print(json.dumps(dict(selected_written=count,source_scanned=last)),flush=True)
    assert count==100000 and last==21015324;tmp.replace(corpus)
    report=dict(state='validated_shared_candidate_corpus',passages=count,source_corpus_passages=last,
        source_corpus_sha256='c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d',
        selection='Union of full-corpus BM25 top10 for all 3000 evaluation queries; seeded random unique DPR IDs fill to 100000',
        bm25_pooled_passages=pooled,random_fill_passages=100000-pooled,seed=42,
        limitation='Query-informed BM25 candidate pool; favors BM25 coverage. Dense rankings are exhaustive only within this shared pool, not full Wikipedia.',
        bm25_source_sha256=sources,ids_sha256=digest(ids),passages_sha256=digest(corpus),seconds=time.time()-start,
        generated_answers=0,paper_table_complete=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

import argparse,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
root=parser.parse_args().root
directory=root/'bm25_queries';directory.mkdir(exist_ok=False)
manifest={}
for ds in ('nq','hotpotQA','triviaQA'):
    source=root/f'repo/data/sampled_{ds}_1000_queries.json'
    rows=json.loads(source.read_text(encoding='utf-8'));assert len(rows)==1000
    assert len({str(r['id']) for r in rows})==1000
    topics=[]
    for row in rows:
        assert not any(c in row['question'] for c in '\t\r\n')
        topics.append(f'{ds}:{row["id"]}\t{row["question"]}\n')
    (directory/f'{ds}.tsv').write_text(''.join(topics),encoding='utf-8')
    manifest[ds]=dict(questions=rows,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(directory/'questions.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(datasets=list(manifest),queries=3000)),flush=True)

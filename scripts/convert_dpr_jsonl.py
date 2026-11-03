"""Stream verified DPR TSV into sharded canonical JSONL for Anserini."""
import argparse,csv,gzip,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--limit',type=int,default=0)
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();root=args.root;assert (root/'corpus/provenance.json').is_file()
args.output_dir.mkdir(parents=True,exist_ok=False)
handles=[(args.output_dir/f'part-{i:02}.jsonl').open('w',encoding='utf-8') for i in range(16)]
count=0
try:
    with gzip.open(root/'corpus/psgs_w100.tsv.gz','rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            record=dict(id=row['id'],contents=row['title']+'\n'+row['text'])
            handles[count%16].write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');count+=1
            if count%1000000==0:print(json.dumps(dict(passages=count)),flush=True)
            if args.limit and count>=args.limit:break
finally:
    for handle in handles:handle.close()
if not args.limit:assert count==21015324
report=dict(passages=count,format='DPR CSV-decoded title+newline+text',input_sha256=json.loads((root/'corpus/provenance.json').read_text())['sha256'])
(args.output_dir.parent/(args.output_dir.name+'_provenance.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)

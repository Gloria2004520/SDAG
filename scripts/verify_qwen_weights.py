"""Verify transferred public Qwen weight shards before any inference."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True,type=Path);a=p.parse_args()
manifest=json.loads((a.model_dir/'weights_manifest.json').read_text())
checked=[]
for entry in manifest:
    path=a.model_dir/entry['name'];assert path.is_file(),f'Missing {entry["name"]}'
    assert path.stat().st_size==entry['size'],f'Size mismatch: {entry["name"]}'
    if entry['sha256']:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(4*1024*1024):digest.update(block)
        assert digest.hexdigest()==entry['sha256'],f'SHA256 mismatch: {entry["name"]}'
        checked.append(dict(name=entry['name'],sha256=digest.hexdigest(),bytes=entry['size']))
index=json.loads((a.model_dir/'model.safetensors.index.json').read_text())
expected=set(index['weight_map'].values());assert expected=={x['name'] for x in checked if x['name'].endswith('.safetensors')}
config=json.loads((a.model_dir/'config.json').read_text());assert config['model_type']=='qwen2'
report=dict(status='verified_remote_weight_hashes',model='Qwen/Qwen2.5-7B-Instruct',
    revision='a09a35458c702b33eeacc393d103063234e8bc28',weight_files=checked,
    total_bytes=sum(x['bytes'] for x in checked),inference_smoke_passed=False,
    notes='Weights verified; no GPU inference claim. Model compatibility and paper-version alignment still require validation.')
(a.model_dir/'remote_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))

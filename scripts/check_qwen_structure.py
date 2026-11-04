"""CPU-only artifact structure and tokenizer checks; no model weights loaded."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model-dir', type=Path, required=True)
a = p.parse_args()
d = a.model_dir
assert json.loads((d/'remote_verification.json').read_text())['status'] == 'verified_remote_weight_hashes'
index = json.loads((d/'model.safetensors.index.json').read_text())['weight_map']
seen = {}
for name in sorted(set(index.values())):
    path = d/name
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        assert 0 < length < 10_000_000
        header = json.loads(f.read(length))
    end = 0
    for key, info in sorted(((k,v) for k,v in header.items() if k != '__metadata__'), key=lambda x:x[1]['data_offsets'][0]):
        assert key not in seen and index[key] == name
        assert info['dtype'] == 'BF16'
        start, stop = info['data_offsets']
        count = 1
        for size in info['shape']:
            assert size > 0
            count *= size
        assert start == end and stop-start == count*2
        end = stop
        seen[key] = info['shape']
    assert end+length+8 == path.stat().st_size
assert set(seen) == set(index)
assert seen['model.embed_tokens.weight'] == [152064,3584]
assert seen['lm_head.weight'] == [152064,3584]
tokenizer = AutoTokenizer.from_pretrained(d, local_files_only=True)
prompt = tokenizer.apply_chat_template([{'role':'user','content':'Answer briefly: What is 2 + 2?'}], tokenize=False, add_generation_prompt=True)
ids = tokenizer.encode(prompt, add_special_tokens=False)
assert ids and max(ids) < 152064
assert tokenizer.eos_token_id == 151645
assert prompt.endswith('<|im_start|>assistant\n')
report = dict(status='passed_artifact_structure_and_tokenizer', tensor_count=len(seen),
    weight_index_complete=True, bf16_storage_offsets_valid=True,
    tokenizer_class=type(tokenizer).__name__, eos_token_id=tokenizer.eos_token_id,
    prompt_tokens=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    model_weights_loaded=False, gpu_used=False, inference_smoke_passed=False,
    paper_version_alignment_verified=False)
(d/'structure_tokenizer_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))

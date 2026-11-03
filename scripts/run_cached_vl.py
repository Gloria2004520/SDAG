"""Reproducible in-set GPU run using the group's frozen NQ retrieval data."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer
from src.pipeline.utils.normalization import exact_match, normalize_answer
from src.pipeline.utils.prompts import SYSTEM_PROMPT_RAG, USER_RAG_PROMPT
from src.pipeline.sparse_attention_RAG.SDAG import build_blocked_causal_mask_full


def encode(tokenizer, question, docs, isolated):
    blocks = ['- ' + d['text'].strip() for d in docs]
    user = USER_RAG_PROMPT.format(query=question, docs_text='\n\n'.join(blocks))
    chat = tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM_PROMPT_RAG},
        {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')
    starts, cursor = [], 0
    for block in blocks:
        pos = chat.index(block, cursor)
        starts.append(next(i for i, (_, end) in enumerate(offsets) if end > pos))
        cursor = pos + len(block)
    qchar = chat.index('- Question:', cursor)
    qa = next(i for i, (_, end) in enumerate(offsets) if end > qchar)
    # Include separators in the preceding document block: no all-masked rows.
    spans = list(zip(starts, starts[1:] + [qa]))
    ids = torch.tensor([encoded['input_ids']], device='cuda')
    length = ids.shape[1]
    mask = build_blocked_causal_mask_full(length, 0, starts[0], spans, qa, 'cuda')
    assert mask.any(dim=1).all() and not mask.triu(1).any()
    for i, (a, b) in enumerate(spans):
        for c, d in spans[:i]:
            assert not mask[a:b, c:d].any()
    if not isolated:
        mask = torch.ones_like(mask).tril()
    return ids, mask


@torch.inference_mode()
def generate(model, tokenizer, ids, allowed, limit, temperature):
    mask = torch.zeros_like(allowed, dtype=model.dtype).masked_fill(~allowed, torch.finfo(model.dtype).min)
    out = model(input_ids=ids, attention_mask=mask[None, None], use_cache=True,
                cache_position=torch.arange(ids.shape[1], device=ids.device), logits_to_keep=1)
    tokens = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for step in range(limit):
        if not torch.isfinite(out.logits[:, -1]).all():
            raise FloatingPointError('Non-finite generation logits')
        logits = out.logits[:, -1].float()
        token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1) if temperature > 0 else logits.argmax(-1, keepdim=True)
        if token.item() in eos:
            break
        tokens.append(token.item())
        if step + 1 < limit:
            out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True,
                        cache_position=torch.tensor([ids.shape[1] + step], device=ids.device), logits_to_keep=1)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--data', default='data/m1/in_set_nq_1000_top5.jsonl')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--revision', default='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--max-new-tokens', type=int, default=500)
    p.add_argument('--output', default='results/basic_gpu')
    args = p.parse_args()
    if args.limit <= 0 or args.max_new_tokens <= 0:
        p.error('limit and max-new-tokens must be positive')
    assert torch.cuda.is_available(), 'CUDA required; no silent CPU fallback'
    torch.manual_seed(42)
    source = ROOT / args.data
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()][:args.limit]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'predictions.jsonl'
    if target.exists():
        raise FileExistsError(f'Use a new output directory to avoid overwriting {target}')
    metadata = dict(vars(args), seed=42, torch=torch.__version__, transformers=transformers.__version__,
        python=platform.python_version(), gpu=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        setting='preliminary; frozen NQ retrieval of unverified provenance; in-set; bfloat16; eager attention; temperature sampling; top_p=1.0')
    (output / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision,
        torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
    results = []
    started = time.time()
    with target.open('w', encoding='utf-8') as f:
        for row in rows:
            for condition, docs, isolated in [
                ('clean_carg', row['clean_documents'], False),
                ('poisoned_carg', row['poisoned_documents'], False),
                ('poisoned_sdag', row['poisoned_documents'], True)]:
                ids, mask = encode(tokenizer, row['question'], docs, isolated)
                answer = generate(model, tokenizer, ids, mask, args.max_new_tokens, args.temperature)
                result = dict(query_id=row['query_id'], condition=condition, answer=answer,
                    ground_truth_answers=row['ground_truth_answers'], false_answer=row['false_answer'],
                    acc=any(exact_match(answer, gt) for gt in row['ground_truth_answers'] if normalize_answer(gt)),
                    asr=bool(normalize_answer(row['false_answer'])) and exact_match(answer, row['false_answer']),
                    strict_em=any(normalize_answer(answer)==normalize_answer(gt) for gt in row['ground_truth_answers']),
                    prompt_tokens=ids.shape[1], attack_position=row['attack_position'])
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
                results.append(result)
                print(len(results), row['query_id'], condition, answer, flush=True)
    summary = {'seconds': time.time()-started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'conditions': {}}
    for condition in ('clean_carg', 'poisoned_carg', 'poisoned_sdag'):
        group = [r for r in results if r['condition']==condition]
        summary['conditions'][condition] = dict(n=len(group), **{key: sum(r[key] for r in group)/len(group) for key in ('acc','asr','strict_em')})
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

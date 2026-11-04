"""Use the validated generation runner, with explicit zero-retained-doc support."""
import torch,argparse,hashlib,json
import run_cached_vl as runner
original_encode=runner.encode
original_generate=runner.generate
reuse={};reused=[];current=None;source_evidence=None
def signature(question,docs,isolated):
    return json.dumps([question,[d['text'].strip() for d in docs],isolated],ensure_ascii=False)
def encode(tokenizer,question,docs,isolated):
    global current
    current=reuse.get(signature(question,docs,isolated))
    if docs:return original_encode(tokenizer,question,docs,isolated)
    user=runner.USER_RAG_PROMPT.format(query=question,docs_text='')
    chat=tokenizer.apply_chat_template([{'role':'system','content':runner.SYSTEM_PROMPT_RAG},
        {'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    ids=torch.tensor([tokenizer(chat,add_special_tokens=False)['input_ids']],device='cuda')
    return ids,torch.ones((ids.shape[1],ids.shape[1]),device='cuda',dtype=torch.bool).tril()
runner.encode=encode
def generate(*args,**kwargs):
    if current is not None:
        reused.append(dict(query_id=current['query_id'],condition=current['condition']))
        return current['answer']
    return original_generate(*args,**kwargs)
runner.generate=generate
if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--data');p.add_argument('--output');a,_=p.parse_known_args()
    source=runner.ROOT/a.data;report_path=source.with_suffix('.report.json')
    if report_path.exists():
        report=json.loads(report_path.read_text());base=runner.ROOT/report['canonical_result']
        config=json.loads((base/'config.json').read_text());verified=json.loads((base/'verified_report.json').read_text())
        assert verified['samples_per_condition']==1000
        assert config['revision']=='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b' and config['temperature']==.1 and config['max_new_tokens']==500
        original=runner.ROOT.parent/report['canonical_source']['path']
        assert hashlib.sha256(original.read_bytes()).hexdigest()==report['canonical_source']['sha256']==config['source_sha256']
        originals={r['query_id']:r for r in map(json.loads,original.read_text().splitlines())}
        predictions=base/'predictions.jsonl'
        assert hashlib.sha256(predictions.read_bytes()).hexdigest()==verified['predictions_sha256']
        for row in map(json.loads,predictions.read_text().splitlines()):
            if row['condition'] not in ('poisoned_carg','poisoned_sdag'):continue
            docs=originals[row['query_id']]['poisoned_documents']
            reuse[signature(originals[row['query_id']]['question'],docs,row['condition']=='poisoned_sdag')]=row
        source_evidence=dict(canonical_result=report['canonical_result'],predictions_sha256=verified['predictions_sha256'])
    runner.main()
    # Reconstruct reuse across both the preserved prefix and newly generated suffix.
    inputs={r['query_id']:r for r in map(json.loads,source.read_text().splitlines())}
    reused=[]
    for result in map(json.loads,(runner.ROOT/a.output/'predictions.jsonl').read_text().splitlines()):
        row=inputs[result['query_id']]
        match=reuse.get(signature(row['question'],row['poisoned_documents'],result['condition']=='poisoned_sdag'))
        if match is not None:
            assert result['answer']==match['answer']
            reused.append(dict(query_id=result['query_id'],condition=result['condition']))
    (runner.ROOT/a.output/'reuse_report.json').write_text(json.dumps(dict(
        policy='Reuse canonical answer only when question, ordered stripped document texts, mask mode and model/settings are unchanged; generate changed prompts only',
        source=source_evidence,reused_records=len(reused),reused_keys=reused),indent=2))

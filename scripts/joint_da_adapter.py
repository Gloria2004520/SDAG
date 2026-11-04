"""Pinned original joint D&A formatting and fail-closed response parsing.

No API calls. DeepSeek/Qwen filtering integrations remain explicit adaptations.
"""
import ast,hashlib,json,re
from pathlib import Path
from functools import lru_cache

SOURCE_COMMIT='107b86ea98f975266a4665be3577cae482485f9d'

@lru_cache(maxsize=4)
def _source(notebook:Path,fewshot:Path):
    nb=json.loads(notebook.read_text());ns={'all_passage_set':set(range(1,6))}
    for cell in nb['cells']:
        source=''.join(cell['source'])
        if 'def prompt_from_nqopen' not in source:continue
        tree=ast.parse(source)
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('pert_inst_generator','prompt_from_nqopen')]
        assert len(funcs)==2
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(notebook),'exec'),ns)
        instruction=next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='instruction_pert_aware' for t in n.targets))
        break
    else:raise ValueError('Original prompt functions not found')
    example=json.loads(fewshot.read_text())[0]
    sample=ns['prompt_from_nqopen'](example,'75',is_pert_prompt=True,inst_at_end=instruction)
    return sample,instruction,dict(source_commit=SOURCE_COMMIT,fewshot_index=0,trainx_prob='75',
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        fewshot_sha256=hashlib.sha256(fewshot.read_bytes()).hexdigest())

def build_prompt(notebook:Path,fewshot:Path,question:str,documents:list[str]):
    assert documents and all(isinstance(x,str) and x.strip() for x in documents)
    sample,instruction,source_meta=_source(notebook,fewshot)
    # Test-input assembly follows original Passage numbering, instruction,
    # question and perturbed postfix, without reading target or poison labels.
    test=''.join(f'Passage {i+1}: {text.strip()}\n' for i,text in enumerate(documents))
    test+=instruction+'question: '+question.strip()+'?\nperturbed:'
    prompt=sample+test
    return prompt,dict(**source_meta,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),documents=len(documents),
        adaptation='Original five-passage fewshot; test uses actual supplied passage count; API/model replacement must be separately recorded')

def parse_response(text:str,finish_reason:str,document_count:int):
    if finish_reason!='stop':raise ValueError('Incomplete or abnormal generation')
    if document_count<=0:raise ValueError('Invalid document count')
    normalized=text.replace('**','').strip()
    declaration=re.match(r'(?:Based on the passages provided,\s*)?(?:perturbed:\s*)?(?:No passages are perturbed\.|Passage ([\d, ]+) (?:is|are) perturbed\.)',normalized,re.I)
    if not declaration:raise ValueError('No anchored explicit perturbation declaration')
    removed=[] if declaration.group(1) is None else [int(x.strip())-1 for x in declaration.group(1).split(',')]
    if len(removed)!=len(set(removed)) or any(i<0 or i>=document_count for i in removed):
        raise ValueError('Duplicate or out-of-range passage IDs')
    remainder=normalized[declaration.end():]
    markers=list(re.finditer(r'\banswer\s*:',remainder,re.I))
    if len(markers)!=1:raise ValueError('Missing or ambiguous answer delimiter')
    answer=remainder[markers[0].end():].strip()
    if not answer:raise ValueError('Empty answer')
    before=remainder[:markers[0].start()]
    derivation=re.search(r'Deriving the answer based on Passage ([\d, ]+)\.',before,re.I)
    keep=sorted(set(range(document_count))-set(removed))
    if derivation:
        claimed=[int(x.strip())-1 for x in derivation.group(1).split(',')]
        if sorted(claimed)!=keep:raise ValueError('Derivation contradicts perturbation declaration')
    if re.search('ignoring all the passages',before,re.I) and keep:
        raise ValueError('All-passages rejection contradicts declaration')
    return dict(removed_indices=sorted(removed),kept_indices=keep,answer=answer,
        generation_result_valid=True,parser='anchored joint declaration + unique answer delimiter + stop-only')

def parse_filter(text:str,finish_reason:str,document_count:int):
    """Read a definitive opening classification only; never claim joint QA valid.

    Keeps the complete original response for audit. Free-form answer is discarded
    because this adapter uses Qwen to answer from retained passages.
    """
    if finish_reason!='stop' or document_count<=0:raise ValueError('Incomplete response')
    s=text.replace('**','').strip()
    s=re.sub(r'^Based on the passages(?: provided)?,\s*','',s,flags=re.I)
    s=re.sub(r'^perturbed:\s*','',s,flags=re.I)
    if re.match(r'^No passages are perturbed\.',s,re.I):removed=[]
    else:
        ref=r'Passage\s+\d+'
        refs=rf'{ref}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)(?:Passage\s+)?\d+)*'
        m=re.match(rf'^(?:the perturbed passages are\s+({refs})|({refs})\s+(?:is|are)\s+perturbed)(?=\s*(?:[.,—]|because\b|as\b|$))',s,re.I)
        if not m:raise ValueError('No unambiguous anchored perturbation statement')
        removed=[int(x)-1 for x in re.findall(r'\d+',m.group(1) or m.group(2))]
    if len(set(removed))!=len(removed) or any(i<0 or i>=document_count for i in removed):raise ValueError('Invalid IDs')
    return dict(removed_indices=sorted(removed),kept_indices=sorted(set(range(document_count))-set(removed)),
        classification_valid=True,generation_result_valid=False,
        parser='anchored definitive classification; stop-only; joint answer discarded for Qwen filter adaptation')

def self_test():
    assert parse_response('perturbed: Passage 2, 5 are perturbed.\nanswer: Paris','stop',5)['removed_indices']==[1,4]
    assert parse_response('**No passages are perturbed.**\nAnswer: Paris','stop',10)['kept_indices']==list(range(10))
    assert parse_response('Passage 1 is perturbed. Deriving the answer with ignoring all the passages.\nanswer: unknown','stop',1)['kept_indices']==[]
    bad=[('Passage 2 is perturbed. answer: Paris','length',5),
         ('Passage 6 is perturbed. answer: Paris','stop',5),
         ('Passage 2, 2 are perturbed. answer: Paris','stop',5),
         ('No passages are perturbed. answer:','stop',5),
         ('No passages are perturbed. answer: x answer: y','stop',5),
         ('I think Passage 2 is perturbed. answer: Paris','stop',5),
         ('Passage 2 is perturbed. Deriving the answer based on Passage 2. answer: x','stop',5),
         ('No passages are perturbed. Deriving the answer with ignoring all the passages. answer: x','stop',5)]
    for args in bad:
        try:parse_response(*args)
        except ValueError:pass
        else:raise AssertionError(args)
    assert parse_filter('Based on the passages, **Passage 5 is perturbed** — conflicting date.','stop',5)['removed_indices']==[4]
    assert parse_filter('Based on the passages, the perturbed passages are **Passage 1**, **Passage 4**, and **Passage 7**, as they conflict.','stop',10)['removed_indices']==[0,3,6]
    assert parse_filter('Based on the passages provided, **Passage 1** and **Passage 4** are perturbed because they conflict.','stop',5)['removed_indices']==[0,3]
    for text,finish,n in [bad[0],bad[1],bad[2],bad[5],('Passage 2 might be perturbed.','stop',5)]:
        try:parse_filter(text,finish,n)
        except ValueError:pass
        else:raise AssertionError(text)
    print(json.dumps(dict(valid_cases=3,rejected_cases=len(bad),api_calls=0,status='passed')))
if __name__=='__main__':self_test()

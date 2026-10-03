"""Bounded direct-forward versus HF-generation preparation diagnostic."""
import argparse, contextlib, functools, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone import XPUBackend
from runtime.waystone.bootstrap import prepare_runtime
prepare_runtime()
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--image',required=True);p.add_argument('--prompt',required=True);p.add_argument('--output',required=True);p.add_argument('--attention',default='sdpa',choices=['sdpa','eager','hybrid']);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
b=XPUBackend(a.model,attention_backend=a.attention);b.load()
messages=[{'role':'user','content':[{'type':'image','image':Image.open(a.image).convert('RGB')},{'type':'text','text':Path(a.prompt).read_text()}]}]
inputs=b.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('xpu:0')
def rec(t):
 t=t.detach().cpu().contiguous();return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}
report={'inputs':{k:rec(v) for k,v in inputs.items()},'attention':a.attention,'calls':[]}
phase='direct'; seen=0
original=b.model.forward
records=[]
@functools.wraps(original)
def forward(*args,**kwargs):
 global seen
 row={'phase':phase,'input':{k:rec(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)}}
 if 'position_ids' in kwargs and isinstance(kwargs['position_ids'],torch.Tensor):row['position_ids']=kwargs['position_ids'].detach().cpu().tolist()
 result=original(*args,**kwargs)
 logit=result.logits[0,-1].float().cpu();row['argmax']=int(logit.argmax());row['logits']=rec(logit)
 logit.numpy().astype('<f4').tofile(out/f'{phase}-{seen:02}.f32');seen+=1
 report['calls'].append(row)
 print(json.dumps({'phase':phase,'argmax':row['argmax'],'shapes':{k:v['shape'] for k,v in row['input'].items()}}),flush=True)
 return result
b.model.forward=forward
ctx=sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.OVERRIDEABLE]) if a.attention in {'sdpa','hybrid'} else contextlib.nullcontext()
try:
 with torch.inference_mode(),ctx:
  direct=b.model(**inputs,use_cache=True,logits_to_keep=1)
  del direct
  phase='generate';seen=0
  ids=b.model.generate(**inputs,max_new_tokens=8,eos_token_id=[248044,248046],do_sample=False)
  report['generated_ids']=ids[0,inputs.input_ids.shape[-1]:].cpu().tolist()
  report['generated_text']=b.processor.decode(report['generated_ids'],skip_special_tokens=True)
finally:
 b.model.forward=original;b.unload()
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'generated_ids':report['generated_ids'],'generated_text':report['generated_text']}),flush=True)

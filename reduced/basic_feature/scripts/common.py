import json, os, sys, time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def config(case):
 c=json.loads((ROOT/'configs/common.json').read_text());c.update(json.loads((ROOT/'configs/cases'/f'{case}.json').read_text()));return c
def request(c,path,payload=None):
 r=requests.request('GET' if payload is None else 'POST',f"http://{c['host']}:{c['port']}"+path,json=payload,timeout=c['timeout']);r.raise_for_status()
 return r.json() if r.content else {}
def rpc(c,method,*args):return request(c,'/collective_rpc',{'method':method,'args':list(args),'timeout':c['timeout']})
def save(out,name,data):
 Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def tokenizer(c):
 from vllm.tokenizers import get_tokenizer
 return get_tokenizer(c['model'],tokenizer_mode='deepseek_v41',trust_remote_code=True)
def inputs(c,text,length=0):
 t=tokenizer(c)
 ids=t.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,thinking=c['thinking'])
 if length:
  marker='BASIC_FEATURE_CONTENT_SENTINEL'
  template=t.apply_chat_template([{'role':'user','content':marker}],tokenize=False,add_generation_prompt=True,thinking=c['thinking'])
  prefix,suffix=template.split(marker)
  head=t.encode(prefix,add_special_tokens=False);tail=t.encode(text+suffix,add_special_tokens=False)
  fill=t.encode('这是一段用于上下文功能验证的普通材料。 ',add_special_tokens=False)
  count=length-len(head)-len(tail);assert count>0
  ids=head+(fill*(count//len(fill)+1))[:count]+tail
 assert isinstance(ids,list) and all(isinstance(x,int) for x in ids),type(ids)
 return ids

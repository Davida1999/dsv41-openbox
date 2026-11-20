"""Audit saved routed-expert responses with Python's standard library only."""
import argparse,ast,base64,gzip,json,math,struct
from pathlib import Path

def read(path):
 data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
 return json.loads(data)

def npy(data):
 assert data[:6]==b'\x93NUMPY'
 major=data[6]
 size=struct.unpack('<H' if major==1 else '<I',data[8:10 if major==1 else 12])[0]
 start=10 if major==1 else 12
 header=ast.literal_eval(data[start:start+size].decode('latin1'))
 assert not header['fortran_order']
 dtype=header['descr'];formats={'|u1':('B',1),'<i4':('i',4),'<i8':('q',8),'<u4':('I',4)}
 fmt,width=formats[dtype];payload=data[start+size:];shape=tuple(header['shape'])
 assert len(payload)==math.prod(shape)*width
 values=payload if width==1 else [x[0] for x in struct.iter_unpack('<'+fmt,payload)]
 assert min(values)>=0 and max(values)<32
 assert all(len(set(values[i:i+6]))==6 for i in range(0,len(values),6))
 return shape,dtype,width,payload

def response(folder,name):
 for suffix in ['_response.json','_response.json.gz','_response.txt','_response.txt.gz']:
  p=folder/(name+suffix)
  if p.exists():return read(p)
 raise FileNotFoundError(name)

def selected_logprobs(body):
 logs=body['choices'][0].get('logprobs') or {}
 if 'token_logprobs' in logs:return logs['token_logprobs']
 return [x['logprob'] for x in logs.get('content',[])]

def audit(run):
 folder=run/'matrix';summary=read(folder/'summary.json');rows=summary['cases'];parsed={}
 for row in rows:
  name=row['name'];body=response(folder,name);http=read(folder/(name+'_http.json'));assert http['status']==200
  expected=(body['usage']['prompt_tokens']+len(body['choices'][0]['token_ids'])-1,40,6)
  encoded=base64.b64decode(body['choices'][0]['routed_experts'],validate=True)
  a=npy(encoded);b=npy((folder/(name+'_experts.npy')).read_bytes());assert a==b
  assert a[0]==expected==tuple(row['shape']) and row['token_ids']==body['choices'][0]['token_ids']
  cached=(body['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)
  assert row['cached_tokens']==cached
  parsed[name]=(body,a)
 comparisons=[]
 for row in summary['comparisons']:
  c,ca=parsed[row['cold_case']];w,wa=parsed[row['other_case']]
  tokens=c['choices'][0]['token_ids']==w['choices'][0]['token_ids']
  routes=ca[0]==wa[0] and ca[1]==wa[1] and ca[3]==wa[3]
  P=c['usage']['prompt_tokens'];prefix=P*40*6*ca[2]
  assert tokens==row['tokens_equal'] and routes==row['all_routes_equal']
  assert (ca[3][:prefix]==wa[3][:prefix])==row['prompt_routes_equal']
  left,right=selected_logprobs(c),selected_logprobs(w)
  delta=max((abs(a-b) for a,b in zip(left,right) if a is not None and b is not None),default=None)
  ct=c['choices'][0].get('text',(c['choices'][0].get('message') or {}).get('content'))
  wt=w['choices'][0].get('text',(w['choices'][0].get('message') or {}).get('content'))
  if row.get('cache_hit_verified'):
   assert (c['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)==0
   assert (w['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)>0
  comparisons.append({'name':row['name'],'text_equal':ct==wt,'tokens_equal':tokens,'all_routes_equal':routes,'selected_logprobs_equal':left==right if left and right else None,'selected_logprob_max_abs':delta})
 if summary['status']=='passed':
  assert all(row['text_equal'] and row['tokens_equal'] and row['all_routes_equal']
             and row['selected_logprobs_equal'] and row['selected_logprob_max_abs']==0
             for row in comparisons)
 if summary.get('coverage')=='full_concurrency_cache_lifecycle':
  steps=read(folder/'same_step_multi_request_evidence.json');assert len(steps)==summary['same_step_multi_request_steps']>0
  assert all(s['num_real_requests']>=2 for s in steps)
 return {'run':run.name,'status':summary['status'],'requests_checked':len(rows),'all_expert_arrays_valid':True,'comparisons':comparisons,'same_step_multi_request_steps':summary.get('same_step_multi_request_steps',0)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
 result={'scope':'CPU audit of saved real-NPU evidence; no generation','runs':[audit(run) for run in args.runs]}
 if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

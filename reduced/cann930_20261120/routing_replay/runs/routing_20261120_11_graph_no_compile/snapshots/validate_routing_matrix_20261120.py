"""Real NPU routing-capture coverage; save evidence before checking invariants."""
import argparse,base64,concurrent.futures,datetime,hashlib,io,json,threading,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np

BASE='http://127.0.0.1:18244'
DEADLINE=datetime.datetime(2026,11,20,3,53,tzinfo=datetime.timezone.utc).timestamp()

def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def request(method,path,body=None,rank=None):
 assert time.time()<DEADLINE,'No new request after 11:52 Beijing'
 headers={'Content-Type':'application/json'}
 if rank is not None:headers['X-data-parallel-rank']=str(rank)
 req=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers=headers,method=method)
 try:
  with urllib.request.urlopen(req,timeout=120) as response:return response.status,response.read().decode()
 except urllib.error.HTTPError as error:return error.code,error.read().decode()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--mixed-pairs',type=int,default=20);p.add_argument('--sleep-level',type=int,choices=[0,1,2],default=2);p.add_argument('--pr-only',action='store_true');a=p.parse_args()
 out=a.run/'matrix';out.mkdir(exist_ok=True)
 result={'status':'running','cases':[],'comparisons':[],'failures':[]};save(out/'summary.json',result)
 def control(name,path,body=None):
  status,raw=request('POST',path,body);save(out/f'{name}.json',{'status':status,'body':raw});assert status==200,(name,status,raw[:1000]);return raw
 def clear(name):
  control(name+'_pause','/pause?mode=wait&clear_cache=true');control(name+'_resume','/resume')
 def generate(name,endpoint,prompt,rank,barrier=None,max_tokens=16,token_in=False):
  if token_in:
   payload={'model':'dsv41-routing','token_ids':prompt,'stream':False,'sampling_params':{'max_tokens':max_tokens,'temperature':0,'ignore_eos':True}}
   path='/inference/v1/generate'
  else:
   payload={'model':'dsv41-routing','max_tokens':max_tokens,'temperature':0,'top_p':1,'seed':42,'ignore_eos':True,'return_token_ids':True}
   if endpoint=='completions':payload.update(prompt=prompt,logprobs=5)
   else:payload.update(messages=[{'role':'user','content':prompt}],logprobs=True,top_logprobs=5,chat_template_kwargs={'thinking':False})
   path='/v1/'+endpoint
  save(out/f'{name}_request.json',{'path':path,'rank':rank,'payload':payload})
  if barrier:barrier.wait(timeout=10)
  started=time.monotonic();status,raw=request('POST',path,payload,rank)
  (out/f'{name}_response.txt').write_text(raw);save(out/f'{name}_http.json',{'status':status,'seconds':time.monotonic()-started})
  assert status==200,(name,status,raw[:500])
  body=json.loads(raw);choice=body['choices'][0];ids=choice['token_ids'];P=body['usage']['prompt_tokens']
  routes=np.load(io.BytesIO(base64.b64decode(choice['routed_experts'],validate=True)),allow_pickle=False)
  np.save(out/f'{name}_experts.npy',routes,allow_pickle=False)
  assert routes.shape==(P+len(ids)-1,40,6),(name,routes.shape,P,len(ids))
  assert np.issubdtype(routes.dtype,np.integer) and routes.min()>=0 and routes.max()<32
  assert np.all(np.diff(np.sort(routes,axis=-1),axis=-1)>0),(name,'duplicate top-6')
  row={'name':name,'endpoint':path,'rank':rank,'shape':list(routes.shape),'dtype':str(routes.dtype),'min':int(routes.min()),'max':int(routes.max()),'token_ids':ids,'prompt_tokens':P,'cached_tokens':(body['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0),'text':choice.get('text',choice.get('message',{}).get('content'))}
  return row,routes
 def add(value):
  row,routes=value;result['cases'].append(row);save(out/'summary.json',result);return row,routes
 def pair(name,endpoint,prompts,ranks):
  b=threading.Barrier(2)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(generate,f'{name}_{i}',endpoint,prompt,rank,b) for i,(prompt,rank) in enumerate(zip(prompts,ranks))]
   for f in futures:add(f.result())
 def compare(name,cold,warm,expect_hit=False):
  c,cr=cold;w,wr=warm;P=c['prompt_tokens']
  first=np.argwhere(cr[:min(len(cr),len(wr))]!=wr[:min(len(cr),len(wr))])
  row={'name':name,'cold_case':c['name'],'other_case':w['name'],'cold_cached':c['cached_tokens'],'other_cached':w['cached_tokens'],'tokens_equal':c['token_ids']==w['token_ids'],'prompt_routes_equal':np.array_equal(cr[:P],wr[:P]),'all_routes_equal':np.array_equal(cr,wr),'first_route_difference':None if not len(first) else first[0].tolist()}
  if expect_hit:row['cache_hit_verified']=c['cached_tokens']==0 and w['cached_tokens']>0
  result['comparisons'].append(row);save(out/'summary.json',result)
 control('trace_start','/collective_rpc',{'method':'begin_routing_trace','args':[str(a.run/'matrix_trace')],'timeout':120})
 if a.sleep_level==1:
  control('host_memory_policy','/collective_rpc',{'method':'openbox_host_policy','args':[str(a.run/'host_policy')],'timeout':120})
  policies=[json.loads(p.read_text()) for p in (a.run/'host_policy').glob('dp*.json')]
  assert len(policies)==8 and all(len(p['interleaved_nodes'])>1 for p in policies),policies
 try:
  for endpoint in ['completions','chat/completions']:
   tag=endpoint.replace('/','_')
   if not a.pr_only:
    for rank in range(8):
     add(generate(f'single_{tag}_dp{rank}',endpoint,'The capital of France is',rank))
     pair(f'pair_{tag}_dp{rank}',endpoint,['The capital of France is','Explain why the sky is blue.'],[rank,rank])
    for rank in range(0,8,2):pair(f'cross_{tag}_dp{rank}',endpoint,['What is two plus three?','What is the largest ocean?'],[rank,rank+1])
    for i in range(a.mixed_pairs):pair(f'mixed_{tag}_{i:02}',endpoint,['Short answer: 2+3 =','Paris is the capital of France. '*(64 if i%2 else 128)], [0,0])
   prompt='Paris is the capital of France. '*80
   clear('cache_'+tag)
   cold=add(generate('cache_cold_'+tag,endpoint,prompt,0))
   warm=add(generate('cache_warm_'+tag,endpoint,prompt,0))
   warm2=add(generate('cache_warm2_'+tag,endpoint,prompt,0))
   compare('cold_warm_'+tag,cold,warm,True);compare('warm_repeat_'+tag,warm,warm2)
   if a.sleep_level:
    control('sleep_'+tag,f'/sleep?level={a.sleep_level}&mode=abort')
    control('wake_weights_'+tag,'/wake_up?tags=weights')
    if a.sleep_level==2:
     control('reload_weights_'+tag,'/collective_rpc',{'method':'reload_weights','kwargs':{'weights_path':'/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32','is_checkpoint_format':True},'timeout':300})
    control('wake_kv_'+tag,'/wake_up?tags=kv_cache')
    restored=add(generate('restored_'+tag,endpoint,prompt,0));compare('sleep_restore_'+tag,cold,restored)
  add(generate('token_in_token_out','completions',[1,2,3],0,token_in=True,max_tokens=8))
  # Verify real simultaneous scheduling, not just simultaneous HTTP submission.
  events=[]
  for file in (a.run/'matrix_trace').glob('worker_dp*.jsonl'):
   for line in file.read_text().splitlines():
    event=json.loads(line)
    if event.get('num_real_requests',0)>=2:events.append(event)
  save(out/'same_step_multi_request_evidence.json',events)
  result['same_step_multi_request_steps']=len(events)
  result['coverage']='cache_lifecycle_only' if a.pr_only else 'full_concurrency_cache_lifecycle'
  if not a.pr_only:assert events,'No scheduled step with two real requests'
  result['transport_shape_range_status']='passed'
  inconsistent=[r['name'] for r in result['comparisons'] if not r['tokens_equal'] or not r['all_routes_equal'] or r.get('cache_hit_verified') is False]
  result['inconsistent_cases']=inconsistent
  result['status']='passed' if not inconsistent else 'failed_consistency'
 except BaseException as error:
  result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:
  save(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2),flush=True)

 if result['status']!='passed':raise AssertionError(result.get('inconsistent_cases',result.get('error')))

if __name__=='__main__':main()

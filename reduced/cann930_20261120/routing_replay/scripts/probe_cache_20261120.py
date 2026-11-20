import argparse,base64,io,json,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np

def req(path,payload=None,rank=None):
 headers={'Content-Type':'application/json'}
 if rank is not None:headers['X-data-parallel-rank']=str(rank)
 r=urllib.request.Request('http://127.0.0.1:18244'+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers,method='POST')
 with urllib.request.urlopen(r,timeout=120) as f:return f.status,f.read().decode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tag',default='baseline');a=p.parse_args();out=a.run/('cache_probe_'+a.tag);out.mkdir(exist_ok=True)
 req('/pause?mode=wait&clear_cache=true');req('/resume')
 rows=[];arrays=[];bodies=[]
 for i in range(3):
  payload={'model':'dsv41-routing','prompt':'Paris is the capital of France. '*80,'max_tokens':16,'temperature':0,'top_p':1,'seed':42,'ignore_eos':True,'return_token_ids':True,'logprobs':5}
  (out/f'{i}_request.json').write_text(json.dumps(payload,indent=2)+'\n')
  status,raw=req('/v1/completions',payload,0);(out/f'{i}_response.json').write_text(raw)
  b=json.loads(raw);c=b['choices'][0];routes=np.load(io.BytesIO(base64.b64decode(c['routed_experts'])),allow_pickle=False);np.save(out/f'{i}_experts.npy',routes)
  bad=np.any(np.diff(np.sort(routes,axis=-1),axis=-1)==0,axis=-1)
  rows.append({'request':i,'status':status,'prompt_tokens':b['usage']['prompt_tokens'],'cached_tokens':(b['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0),'tokens':c['token_ids'],'invalid_top6_per_layer':bad.sum(axis=0).tolist(),'first_token_logprob':c['logprobs']['token_logprobs'][0]});arrays.append(routes);bodies.append(b)
 P=rows[0]['prompt_tokens'];summary={'rows':rows,'cold_warm_tokens_equal':rows[0]['tokens']==rows[1]['tokens'],'warm_repeats_tokens_equal':rows[1]['tokens']==rows[2]['tokens'],'cold_warm_prompt_routes_equal':np.array_equal(arrays[0][:P],arrays[1][:P]),'cold_warm_all_routes_equal':np.array_equal(arrays[0],arrays[1]),'warm_repeats_routes_equal':np.array_equal(arrays[1],arrays[2])}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

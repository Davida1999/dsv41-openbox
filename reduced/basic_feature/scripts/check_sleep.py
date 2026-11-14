import argparse,time
from common import config,request,rpc,save,inputs
from check_inference import generate
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');a=p.parse_args();c=config(a.case);assert c['sleep_level'] in (1,2)
ids=inputs(c,c['prompts'][0]);generate(c,ids);base=generate(c,ids);save(a.out,'sleep_baseline.json',base);time.sleep(c['settle_seconds']);memory=rpc(c,'basic_probe')['results'];rows=[]
for cycle in range(c['cycles']):
 request(c,f"/sleep?level={c['sleep_level']}",{});sleep_state=request(c,'/is_sleeping');assert sleep_state.get('is_sleeping') is True,sleep_state
 request(c,'/wake_up?tags=weights',{});rpc(c,'reload_weights');request(c,'/wake_up?tags=kv_cache',{})
 result=generate(c,ids);save(a.out,f'sleep_cycle_{cycle+1}.json',result);assert result['response']['choices'][0]['token_ids']==base['response']['choices'][0]['token_ids'],'Token mismatch'
 time.sleep(c['settle_seconds']);current=rpc(c,'basic_probe')['results']
 assert len(current)==len(memory)
 for old,new in zip(memory,current):
  assert old['rank']==new['rank']
  for metric in ('allocated','reserved'):
   assert new[metric]-old[metric]<=max(c['memory_growth_mib']*2**20,old[metric]*c['memory_growth_ratio']),(cycle,metric,old,new)
 rows.append({'cycle':cycle+1,'memory':current,'result':result});save(a.out,'sleep.json',{'baseline':base,'baseline_memory':memory,'cycles':rows,'status':'REVIEW_REQUIRED','note':'还需对照memory.jsonl排除HBM持续增长；第1轮前已预热。'})

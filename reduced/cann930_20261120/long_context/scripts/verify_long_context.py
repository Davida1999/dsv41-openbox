"""Measure a real one-million-token cold request, preserving the entire output."""
import argparse,gzip,json,time,urllib.request,urllib.error
from pathlib import Path
MODEL='dsv41-long-context';BASE='http://127.0.0.1:18244'

def post(path,body,timeout=120):
 req=urllib.request.Request(BASE+path,data=json.dumps(body,separators=(',',':')).encode(),headers={'Content-Type':'application/json','X-data-parallel-rank':'0'})
 with urllib.request.urlopen(req,timeout=timeout) as response:return response.status,response.read().decode()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'validation';out.mkdir(exist_ok=True)
 result={'status':'running','phase':'tokenizer_preparation','target_prompt_tokens':1000000,'max_output_tokens':1024};(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 def tokens(body):
  status,raw=post('/tokenize',{'model':MODEL,**body});assert status==200;return json.loads(raw)['tokens']
 try:
  marker='OPENBOXMARKER98765';marker_ids=tokens({'prompt':marker,'add_special_tokens':False})
  template=tokens({'messages':[{'role':'user','content':marker}],'chat_template_kwargs':{'thinking':False},'add_generation_prompt':True})
  starts=[i for i in range(len(template)-len(marker_ids)+1) if template[i:i+len(marker_ids)]==marker_ids];assert len(starts)==1,(template,marker_ids)
  split=starts[0];prefix=template[:split];suffix=template[split+len(marker_ids):]
  unit=tokens({'prompt':'This is reference filler text. ','add_special_tokens':False});count=1000000-len(prefix)-len(suffix)
  content=(unit*((count+len(unit)-1)//len(unit)))[:count]
  needles=[('FIRST','K8Z7Q4',.05),('SECOND','M2V9P6',.50),('THIRD','R5T1X3',.95)];locations=[]
  for label,key,fraction in needles:
   text=f'\nThe secret key for {label} is {key}.\n';value=tokens({'prompt':text,'add_special_tokens':False});position=int(count*fraction);content[position:position+len(value)]=value;locations.append({'label':label,'key':key,'token_position':len(prefix)+position,'encoded_tokens':value})
  question=tokens({'prompt':'\nQuestion: Report the secret keys for FIRST, SECOND and THIRD in that order. Answer with only the three keys.\n','add_special_tokens':False});content[-len(question):]=question
  ids=prefix+content+suffix;assert len(ids)==1000000
  payload={'model':MODEL,'token_ids':ids,'stream':False,'sampling_params':{'max_tokens':1024,'temperature':0,'top_p':1,'seed':42,'ignore_eos':False,'detokenize':True,'skip_special_tokens':True}}
  raw_request=json.dumps(payload,separators=(',',':')).encode();(out/'request.json.gz').write_bytes(gzip.compress(raw_request,mtime=0));(out/'input_manifest.json').write_text(json.dumps({'prompt_tokens':len(ids),'template_prefix':prefix,'template_suffix':suffix,'filler_unit':unit,'needles':locations},indent=2)+'\n')
  result['phase']='generation';(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('REQUEST_READY tokens=1000000 max_output=1024',flush=True)
  start=time.monotonic();status,raw=post('/inference/v1/generate',payload,timeout=3600);elapsed=time.monotonic()-start;(out/'response.json').write_text(raw)
  body=json.loads(raw);choice=body['choices'][0];text=choice.get('text') or (choice.get('message') or {}).get('content') or ''
  if not text:
   _,decoded=post('/detokenize',{'model':MODEL,'tokens':choice['token_ids']});text=json.loads(decoded).get('prompt','')
  (out/'output.txt').write_text(text+'\n');usage=body['usage'];generated=len(choice['token_ids'])
  assert status==200 and usage['prompt_tokens']==1000000 and 0<generated<=1024
  assert text and '\ufffd' not in text
  hits=[key for _,key,_ in needles if key in text]
  result.update(status='passed',phase='complete',http=status,prompt_tokens=usage['prompt_tokens'],output_tokens=generated,seconds=elapsed,utf8_without_replacement_character=True,retrieval_hits=len(hits),retrieval_total=3,retrieved_keys=hits,acceptance_scope='1M token capacity and non-garbled response; retrieval is a separate reduced-model quality observation')
 except BaseException as error:result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

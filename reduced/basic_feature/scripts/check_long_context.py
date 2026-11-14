import argparse,time
from common import config,inputs,save
from check_inference import generate
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');a=p.parse_args();c=config(a.case)
from common import tokenizer
tok=tokenizer(c)
for length in c['long_lengths']:
 output_tokens=c['long_output_tokens'] if length==max(c['long_lengths']) else c.get('probe_output_tokens',c['long_output_tokens'])
 assert length+output_tokens<=c['max_model_len']
 ids=inputs(c,'请找出材料中的三个密码，按出现顺序列出，然后解释上下文。',length)
 keys=['ALPHA7391','BETA4826','GAMMA6153']
 for fraction,key in zip([.05,.5,.95],keys):
  needle=tok.encode('密码是'+key+'。',add_special_tokens=False);pos=int(length*fraction);ids[pos:pos+len(needle)]=needle
 assert len(ids)==length
 # Force the requested output length for capacity testing. Quality is reviewed separately.
 from common import request
 save(a.out,f'long_{length}_request.json',{'input_token_ids':ids,'output_tokens':output_tokens,'expected_keys':keys})
 save(a.out,'long_progress.json',{'stage':'running','input_tokens':length,'output_tokens':output_tokens,'started_at':time.time()})
 started=time.monotonic()
 r=request(c,'/inference/v1/generate',{'token_ids':ids,'output_mode':'tokens','sampling_params':{'temperature':0,'seed':c['seed'],'max_tokens':output_tokens,'min_tokens':output_tokens,'ignore_eos':True},'stream':False})
 save(a.out,f'long_{length}.json',{'input_token_ids':ids,'response':r,'text':tok.decode(r['choices'][0]['token_ids']),'expected_keys':keys,'elapsed_seconds':time.monotonic()-started})
 assert r['usage']['prompt_tokens']==length and len(r['choices'][0]['token_ids'])==output_tokens

save(a.out,'long_progress.json',{'stage':'completed'})

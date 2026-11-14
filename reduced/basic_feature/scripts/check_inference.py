import argparse,time,concurrent.futures
from common import config,inputs,request,save

def generate(c,ids,max_tokens=None):
 start=time.monotonic();r=request(c,'/inference/v1/generate',{'token_ids':ids,'output_mode':'tokens','stream':False,'sampling_params':{'temperature':0,'seed':c['seed'],'max_tokens':max_tokens or c['max_tokens'],'logprobs':1}})
 elapsed=time.monotonic()-start
 choice=r['choices'][0];assert choice['token_ids'],'Empty output'
 from common import tokenizer
 tok=tokenizer(c)
 text=tok.decode(choice['token_ids']);assert '\ufffd' not in text,'Replacement character in output'
 return {'input_token_ids':ids,'response':r,'text':text,'elapsed_seconds':elapsed,'tokens_per_second':len(choice['token_ids'])/elapsed}
def run(c,out):
 lengths=c.get('prompt_lengths',[c['input_tokens']]*len(c['prompts']))
 prompts=[inputs(c,t,n) for t,n in zip(c['prompts'],lengths)]
 assert len(prompts)==len(c['prompts'])
 serial=[]
 for i,x in enumerate(prompts):
  row=generate(c,x);serial.append(row);save(out,f'serial_{i}.json',row)
 def paired(item):
  i,x=item;row=generate(c,x);save(out,f'parallel_{i}.json',row);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:parallel=list(pool.map(paired,enumerate(prompts[:2])))
 save(out,'inference.json',{'serial':serial,'parallel':parallel,'status':'REVIEW_REQUIRED','note':'自动检查仅覆盖非空/替换字符；乱码、循环、语义及功能实际生效需审核日志。端到端tokens/s不等同decode吞吐；TTFT/TPOT见独立benchmark。'})
 return serial
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');a=p.parse_args();run(config(a.case),a.out)

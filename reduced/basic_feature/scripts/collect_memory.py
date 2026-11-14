import argparse,subprocess,time
from common import config,rpc,save
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');p.add_argument('--interval',type=float,default=5);a=p.parse_args();c=config(a.case)
with open(a.out+'/memory.jsonl','a') as f:
 while True:
  import json
  f.write(json.dumps({'time':time.time(),'npu_smi':subprocess.check_output(['npu-smi','info'],text=True)})+'\n');f.flush();time.sleep(a.interval)

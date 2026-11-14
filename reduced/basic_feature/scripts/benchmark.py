import argparse,subprocess,json
from common import config,save
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');p.add_argument('--execute',action='store_true');a=p.parse_args();c=config(a.case)
cmd=['vllm','bench','serve','--backend','openai','--base-url',f"http://{c['host']}:{c['port']}",'--model',c['served_model'],'--tokenizer',c['model'],'--tokenizer-mode','deepseek_v41','--dataset-name','random','--random-input-len',str(c.get('bench_input',512)),'--random-output-len',str(c.get('bench_output',128)),'--num-prompts',str(c.get('bench_requests',10)),'--max-concurrency',str(c.get('bench_concurrency',2)),'--save-result','--result-dir',a.out]
save(a.out,'benchmark_command.json',cmd)
if a.execute:subprocess.run(cmd,check=True)
else:print(__import__('shlex').join(cmd))

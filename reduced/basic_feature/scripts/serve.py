import argparse, os, subprocess, sys, time, socket, json
from pathlib import Path
from common import ROOT,config,save,request
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');p.add_argument('--launch',action='store_true');a=p.parse_args();c=config(a.case)
if c.get('blocked'):sys.exit('BLOCKED: '+c['blocked'])
assert c['dp']*c['tp']*c['pp']*c['pcp']==len(c['devices'])
assert c['tp']%c['dcp']==0
cmd=['vllm','serve',c['model'],'--host',c['host'],'--port',str(c['port']),'--served-model-name',c['served_model'],'--dtype','bfloat16','--tokenizer-mode','deepseek_v41','--enable-scale-out','--worker-extension-cls','worker_probe.Probe',('--enable-prefix-caching' if c.get('prefix_caching',False) else '--no-enable-prefix-caching')]
for flag,key in [("block-size","block_size"),('data-parallel-size','dp'),('tensor-parallel-size','tp'),('pipeline-parallel-size','pp'),('prefill-context-parallel-size','pcp'),('decode-context-parallel-size','dcp'),('max-model-len','max_model_len'),('max-num-seqs','max_num_seqs'),('max-num-batched-tokens','max_num_batched_tokens'),('gpu-memory-utilization','memory_utilization'),('seed','seed')]:cmd+=['--'+flag,str(c[key])]
if c['ep']:cmd+=['--enable-expert-parallel']
cmd += ['--async-scheduling' if c['async_scheduling'] else '--no-async-scheduling','--enable-chunked-prefill' if c['chunked_prefill'] else '--no-enable-chunked-prefill']
if c['graph']:cmd+=['--compilation-config',json.dumps({'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2]})]
else:cmd+=['--enforce-eager']
extra={}
if c['sp']:extra['enable_flashcomm1']=True
if c['sleep_level'] or c['backend']:extra['rl_config']={'enabled':True}
if extra:cmd+=['--additional-config',json.dumps(extra)]
if c['sleep_level']:cmd+=['--enable-sleep-mode']
if c['backend']:cmd+=['--weight-transfer-config',json.dumps({'backend':c['backend']})]
if c['routing']:cmd+=['--enable-return-routed-experts']
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,c['devices'])),VLLM_USE_V2_MODEL_RUNNER='1',VLLM_SERVER_DEV_MODE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONPATH=str(ROOT/'scripts')+':'+env.get('PYTHONPATH',''))
if c['backend'] in ('ipc','npu_ipc'):env['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
save(a.out,'effective_config.json',c);save(a.out,'command.json',cmd)
print(__import__('shlex').join(cmd),flush=True)
if not a.launch:sys.exit(0)
subprocess.run([sys.executable,str(ROOT/'scripts/preflight.py'),a.case,a.out,'--require-free'],check=True)
with socket.socket() as s:
 if s.connect_ex((c['host'],c['port']))==0:sys.exit('Port occupied; refusing to reuse another service')
log=open(Path(a.out)/'server.log','a');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
save(a.out,'process.json',{'pid':proc.pid,'case':a.case,'cmdline':Path(f'/proc/{proc.pid}/cmdline').read_bytes().decode().replace('\x00',' ')})
for _ in range(c['ready_timeout']):
 if proc.poll() is not None:sys.exit('Server exited; inspect server.log')
 try:request(c,'/health');print('READY');break
 except Exception:time.sleep(1)
else:sys.exit('Readiness timeout: service left for inspection; use stop.sh with this run directory')

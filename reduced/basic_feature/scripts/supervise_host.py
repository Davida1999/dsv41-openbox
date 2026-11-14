# Host-side courtesy guard. Only processes with this exact run marker may be killed.
import os,sys,time,json,re,signal,subprocess
from pathlib import Path
ROOT=Path('/mnt/share/c00954197/workspace/dsv41-kx/basic_feature')
node,case=sys.argv[1:3];tag=f'basic_feature_{node}_{case}_{time.time_ns()}'
out=ROOT/'results'/f'node{node}_{case}_{time.strftime("%Y%m%dT%H%M%S")}'
out.mkdir(parents=True);marker=('BASIC_FEATURE_RUN='+tag).encode()
known_owned={}
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  return fields[19],fields[0]
 except (FileNotFoundError,PermissionError,ProcessLookupError):return None
def live(pid):
 item=identity(pid)
 return item is not None and item[1] not in ('Z','X','x')
def owned(pid):
 item=identity(pid)
 if item is None or item[1] in ('Z','X','x'):return False
 try:tagged=marker in Path(f'/proc/{pid}/environ').read_bytes().split(bytes([0]))
 except (FileNotFoundError,PermissionError,ProcessLookupError):tagged=False
 if tagged:known_owned[pid]=item[0]
 return tagged or known_owned.get(pid)==item[0]
def owned_pids():return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and owned(int(p.name))]
def stop_ours():
 for sig in [signal.SIGTERM,signal.SIGKILL]:
  for pid in owned_pids():
   if owned(pid):
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
  time.sleep(5)
def snapshot():
 q=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
 if q.returncode:raise RuntimeError(q.stderr)
 pids=[int(pid) for _,pid in re.findall(r'^\|\s*([0-7])\s*\|\s*(\d+)\s*\|',q.stdout,re.M)]
 return q.stdout,pids
state='STARTING';proc=None
try:
 raw,pids=snapshot();(out/'admission_npu.txt').write_text(raw)
 if any(live(pid) for pid in pids):raise RuntimeError('YIELD: existing NPU processes '+str(pids))
 cmd=['docker','exec','-e','BASIC_FEATURE_RUN='+tag,'-e',f'VLLM_HOST_IP=177.11.{node}.1','-e',f'HCCL_IF_IP=177.11.{node}.1','-e','HCCL_SOCKET_IFNAME=enp35s0f2','-e','GLOO_SOCKET_IFNAME=enp35s0f2','c00954197_vllm_ascend_1108','python',str(ROOT/'scripts/execute_case.py'),case,str(out)]
 (out/'supervisor.json').write_text(json.dumps({'tag':tag,'node':node,'case':case,'cmd':cmd,'host_supervisor_pid':os.getpid()},indent=2))
 log=open(out/'driver.log','w');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 with open(out/'npu_watch.jsonl','a') as watch:
  while True:
   raw,pids=snapshot();foreign=[pid for pid in pids if live(pid) and not owned(pid)]
   watch.write(json.dumps({'time':time.time(),'pids':pids,'foreign':foreign,'raw':raw})+'\n');watch.flush()
   if foreign:
    state='YIELDED_TO_OTHER_JOB';(out/'contention.json').write_text(json.dumps({'pids':foreign,'time':time.time()}));break
   if (out/'STOP').exists():state='STOP_REQUESTED';break
   if proc.poll() is not None:state='CLIENT_COMPLETE_REVIEW_REQUIRED' if proc.returncode==0 else 'FAILED';break
   time.sleep(5)
except Exception as e:
 state='YIELDED_OR_MONITOR_ERROR';(out/'supervisor_error.txt').write_text(repr(e))
finally:
 stop_ours()
 if proc is not None:
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.terminate()
 (out/'completion.json').write_text(json.dumps({'state':state,'time':time.time(),'remaining_owned':owned_pids()},indent=2))
 print(out,state,flush=True)

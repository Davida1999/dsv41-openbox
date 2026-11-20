"""Yield this exact run on foreign NPU use or loss of its controller/workers."""
import argparse,json,os,re,signal,subprocess,time
from pathlib import Path
from process_ownership import belongs_to_run

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();root=run.parents[1]
 assert run.is_relative_to(Path('/mnt/share/c00954197/Workspace/dsv41/basic/long_context/runs'))
 seen_controller=False;baseline=None
 with (run/'resource_policy.jsonl').open('a') as log:
  while True:
   state=json.loads((run/'session_status.json').read_text()) if (run/'session_status.json').exists() else {'status':'running','phase':'preflight'}
   reason=None;foreign=[];own=[]
   try:
    smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
    pids=sorted({int(m.group(2)) for m in re.finditer(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|',smi,re.M)})
    pids=[pid for pid in pids if alive(pid)];own=[pid for pid in pids if belongs_to_run(pid,run)];foreign=[pid for pid in pids if pid not in own]
    controller=int((run/'controller.pid').read_text()) if (run/'controller.pid').exists() else None
    live=controller is not None and alive(controller)
    if live:
     try:live=str(run).encode() in Path(f'/proc/{controller}/cmdline').read_bytes()
     except OSError:live=False
    if live:seen_controller=True
    workers=[]
    for pid in own:
     try:
      if 'Worker' in Path(f'/proc/{pid}/comm').read_text():workers.append(pid)
     except OSError:pass
    if len(workers)>=8 and baseline is None:baseline=set(workers)
    intended=(run/'STOP_INTENT').exists() or state.get('status')!='running'
    if foreign:reason='foreign_npu_occupation'
    elif not intended and controller is not None and not live:reason='controller_stopped'
    elif not intended and baseline and baseline-set(own):reason='unexpected_worker_loss'
    entry={'epoch':time.time(),'own_npu_pids':own,'foreign_npu_pids':foreign,'phase':state.get('phase'),'reason':reason}
    log.write(json.dumps(entry)+'\n');log.flush()
    if not own and state.get('status')!='running':return
   except Exception as error:
    reason='resource_monitor_failed';entry={'epoch':time.time(),'reason':reason,'error':str(error)};log.write(json.dumps(entry)+'\n');log.flush()
   if reason:
    (run/'STOP_REQUESTED').write_text(reason+'; yield without automatic restart\n')
    (run/'resource_yield.json').write_text(json.dumps(entry,indent=2)+'\n')
    if (run/'controller.pid').exists():
     controller=int((run/'controller.pid').read_text())
     try:
      command=Path(f'/proc/{controller}/cmdline').read_bytes()
      if str(run).encode() in command and b'python' in command:os.kill(controller,signal.SIGTERM)
     except OSError:pass
    command=['docker','-H','unix:///run/c00954197-cann930/docker.sock','exec','c00954197_dsv41_cann930','python3',str(root/'scripts/stop_service.py'),str(run)]
    try:subprocess.run(command,timeout=45,capture_output=True)
    except Exception:pass
    return
   time.sleep(2)
if __name__=='__main__':main()

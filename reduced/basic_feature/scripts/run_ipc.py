import subprocess,sys,socket
from pathlib import Path
from common import ROOT,config,save
c=config(sys.argv[1]);host=c.get('ipc_master_addr','127.0.0.1')
with socket.socket() as sock:
 sock.bind((host,c.get('ipc_master_port',0)));port=sock.getsockname()[1]
cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--node-rank=0','--rdzv-backend=static','--master-addr='+host,'--master-port='+str(port),'--nproc-per-node='+str(len(c['devices'])),str(ROOT/'scripts/check_ipc_multi.py'),*sys.argv[1:]]
save(sys.argv[2],'ipc_launcher_command.json',cmd)
subprocess.run(cmd,check=True)

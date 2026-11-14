#!/usr/bin/env bash
set -euo pipefail
python - "$1" <<'STOP'
import json,sys,os,signal
from pathlib import Path
p=json.loads((Path(sys.argv[1])/'process.json').read_text());pid=p['pid'];f=Path(f'/proc/{pid}/cmdline')
if not f.exists():sys.exit('Already stopped')
current=f.read_bytes().decode().replace('\x00',' ')
if current!=p['cmdline'] or 'vllm' not in current:sys.exit('PID identity mismatch; refusing to stop')
os.killpg(pid,signal.SIGTERM)
print('SIGTERM sent only to recorded experiment process group; verify exit and npu-smi')
STOP

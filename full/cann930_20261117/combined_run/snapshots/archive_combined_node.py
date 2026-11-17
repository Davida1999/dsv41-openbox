from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

run = Path('/mnt/share/c00954197/Workspace/dsv41/precision/runs/combined_20261117_R1')
node = 'node' + sys.argv[1]
directory = run / node
telemetry = [json.loads(line) for line in (directory / 'telemetry.jsonl').open()]
completion = json.loads((directory / 'completion.json').read_text())
assert completion['remaining_owned'] == []
assert all(not row['foreign'] for row in telemetry)
npu = (directory / 'released_npu.txt').read_text()
assert npu.count('No running processes found in NPU') == 8
processes = (directory / 'released_processes.txt').read_text()
assert 'VLLM' not in processes and 'vllm serve' not in processes
ports = (directory / 'released_ports.txt').read_text()
assert not re.search(r':(?:8000|12346)\s', ports)

cutoff = json.loads((run / 'server_log_cutoffs.json').read_text())['nodes'][node]
log = (directory / 'launcher.log').read_bytes()
before_stop = log[:cutoff['bytes']]
assert hashlib.sha256(before_stop).hexdigest() == cutoff['sha256']
before_lines = before_stop.decode(errors='replace').splitlines()
errors = [line for line in before_lines if re.search(r'\bERROR\b|Traceback \(most recent', line)]
assert not errors, errors[:3]
pids = {int(pid) for pid in re.findall(r'pid=(\d+)', log.decode(errors='replace'))}
pids.update(pid for row in telemetry for pid in row['pids'])

archive_code = r'''
from pathlib import Path
import hashlib,json,re,sys,tarfile
directory = Path(sys.argv[1])
pids = set(json.loads(sys.argv[2]))
files = []
for path in Path('/root/ascend/log').rglob('*'):
    if path.is_file() and pids.intersection(map(int, re.findall(r'\d+', path.name))):
        files.append(path)
errors = []
total_errors = 0
manifest = []
with tarfile.open(directory / 'cann_logs.tar.gz', 'w:gz') as archive:
    for path in sorted(files):
        data = path.read_bytes()
        archive.add(path, arcname=str(path.relative_to('/')), recursive=False)
        lines = [line for line in data.decode(errors='replace').splitlines() if re.search(r'\bERROR\b|Traceback \(most recent', line)]
        total_errors += len(lines)
        errors.extend({'file':str(path),'line':line[:1600]} for line in lines[:4])
        manifest.append({'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'error_lines':len(lines)})
summary = {'files':manifest,'total_error_lines':total_errors,'error_samples':errors[:20]}
(directory / 'cann_logs_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'archived_files':len(files),'total_error_lines':total_errors,'error_samples':errors[:6]}))
'''
docker = ['docker', '-H', 'unix:///run/c00954197-cann930/docker.sock']
result = subprocess.run(
    docker + ['exec', 'c00954197_dsv41_cann930', 'python3', '-c', archive_code,
              str(directory), json.dumps(sorted(pids))],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
assert result.returncode == 0, result.stdout
print(node, result.stdout)
summary = {
    'node': node, 'release_verified': True,
    'remaining_owned': completion['remaining_owned'],
    'npu_empty_devices': 8, 'service_ports_released': True,
    'telemetry_samples': len(telemetry), 'competition_samples': 0,
    'before_stop_log_sha256_verified': True,
    'before_stop_error_lines': len(errors),
    'container_processes_file': str(directory / 'released_processes.txt'),
    'npu_file': str(directory / 'released_npu.txt'),
    'cann_logs_manifest': str(directory / 'cann_logs_manifest.json'),
}
(directory / 'release_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))

# Stop our experiment and yield both nodes when another NPU task appears.
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

node, role, run_dir = sys.argv[1:4]
run = Path(run_dir)
record = run / ('node' + node)
record.mkdir(parents=True, exist_ok=True)
root = Path('/mnt/share/c00954197/Workspace/dsv41/performance_cann930')
server_script = Path(sys.argv[4]) if len(sys.argv) > 4 else root / f'run_{role}.sh'
docker = ['docker', '-H', 'unix:///run/c00954197-cann930/docker.sock']
container = 'c00954197_dsv41_cann930'
info = json.loads(subprocess.check_output(docker + ['inspect', container]))[0]
container_id = info['Id']
tag = run.name
marker = ('DSV41_PERFORMANCE_RUN=' + tag).encode()
known_owned = {}
excluded_pids = {}


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        return fields[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def owned(pid, read_environment=True):
    start = identity(pid)
    if start is None:
        return False
    if excluded_pids.get(pid) == start:
        return False
    try:
        if container_id not in Path(f'/proc/{pid}/cgroup').read_text():
            return False
        if known_owned.get(pid) == start:
            return True
        tagged = read_environment and marker in Path(f'/proc/{pid}/environ').read_bytes().split(bytes([0]))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    if tagged or known_owned.get(pid) == start:
        known_owned[pid] = start
        return True
    current = pid
    for _ in range(16):
        try:
            status = Path(f'/proc/{current}/status').read_text().splitlines()
            parent = int(next(line for line in status if line.startswith('PPid:')).split()[1])
            if parent <= 1 or container_id not in Path(f'/proc/{parent}/cgroup').read_text():
                break
            parent_start = identity(parent)
            if parent_start is not None and excluded_pids.get(parent) == parent_start:
                break
            tagged_parent = read_environment and marker in Path(f'/proc/{parent}/environ').read_bytes().split(bytes([0]))
            if parent_start is not None and (tagged_parent or known_owned.get(parent) == parent_start):
                known_owned[parent] = parent_start
                known_owned[pid] = start
                return True
            current = parent
        except (OSError, StopIteration):
            break
    return False


def our_pids():
    return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and owned(int(p.name))]


def snapshot():
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
    result.check_returncode()
    pids = sorted(set(int(pid) for _, pid in re.findall(r'^\|\s*([0-7])\s*\|\s*(\d+)\s*\|', result.stdout, re.M)))
    return result.stdout, pids


def stop(reason):
    target = run / 'STOP'
    try:
        with target.open('x') as f:
            f.write(reason + '\n')
    except FileExistsError:
        pass


process = None
state = 'STARTING'
try:
    inventory, pids = snapshot()
    (record / 'admission_npu.txt').write_text(inventory)
    foreign = [pid for pid in pids if identity(pid) is not None and not owned(pid, read_environment=False)]
    if foreign:
        excluded_pids.update({pid: identity(pid) for pid in foreign})
        state = 'RESOURCE_BUSY'
        (record / 'yield_reason.json').write_text(json.dumps({'foreign_npu_pids': foreign, 'phase': 'admission'}) + '\n')
        stop(f'node{node} NPU resources occupied; yielding to the other task')
        sys.exit(0)
    if (run / 'STOP').exists():
        state = 'STOP_REQUESTED'
        sys.exit(0)
    memory = Path('/proc/meminfo').read_text()
    (record / 'admission_meminfo.txt').write_text(memory)
    (record / 'container.json').write_text(json.dumps({
        'id': container_id, 'image': info['Image'], 'name': container, 'node': node,
        'role': role, 'tag': tag,
    }, indent=2) + '\n')
    cmd = docker + ['exec', '-e', 'DSV41_PERFORMANCE_RUN=' + tag, container,
                    'bash', str(server_script)]
    (record / 'launch_command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    with (record / 'launcher.log').open('w') as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        (record / 'runner.json').write_text(json.dumps({'host_runner_pid': os.getpid(), 'docker_exec_pid': process.pid, 'started_at': time.time()}, indent=2))
        with (record / 'telemetry.jsonl').open('a') as telemetry:
            while True:
                if (run / 'STOP').exists():
                    state = 'STOP_REQUESTED'
                    break
                inventory, pids = snapshot()
                foreign = [pid for pid in pids if identity(pid) is not None and not owned(pid, read_environment=False)]
                telemetry.write(json.dumps({'time': time.time(), 'pids': pids, 'foreign': foreign, 'raw': inventory}) + '\n')
                telemetry.flush()
                if foreign:
                    excluded_pids.update({pid: identity(pid) for pid in foreign})
                    state = 'YIELDED_TO_OTHER_TASK'
                    (record / 'yield_reason.json').write_text(json.dumps({'foreign_npu_pids': foreign, 'phase': 'running'}) + '\n')
                    stop(f'node{node} NPU competition detected; yielding to the other task')
                    break
                our_pids()
                if process.poll() is not None:
                    state = 'SERVER_EXITED'
                    stop(f'node{node} server exited with code {process.returncode}')
                    break
                time.sleep(2)
except Exception as error:
    state = 'FAILED'
    (record / 'runner_error.txt').write_text(repr(error))
    stop(f'node{node} runner error: {error}')
finally:
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for pid in our_pids():
            if owned(pid):
                try:
                    process_path = Path(f'/proc/{pid}')
                    name = (process_path / 'comm').read_text().strip()
                    if name.startswith('python'):
                        command = (process_path / 'cmdline').read_bytes()
                        if b'multiprocessing.resource_tracker' in command:
                            # The tracker reclaims shared memory after its owners exit.
                            continue
                    os.kill(pid, sig)
                except (ProcessLookupError, FileNotFoundError):
                    pass
        time.sleep(5)
    for _ in range(10):
        if not our_pids():
            break
        time.sleep(1)
    if process is not None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
    (record / 'completion.json').write_text(json.dumps({'state': state, 'finished_at': time.time(), 'remaining_owned': our_pids()}, indent=2) + '\n')
    print('RUNNER_FINISHED', node, state, flush=True)

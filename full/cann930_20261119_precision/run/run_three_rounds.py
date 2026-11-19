"""Run three audited GPQA rounds against one dual-node service."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

run = Path(sys.argv[1])
precision = run.parents[1]
clients = run / 'clients'
clients.mkdir(exist_ok=True)
lock = (clients / 'controller.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
docker = ['docker', '-H', 'unix:///run/c00954197-cann930/docker.sock']
container = 'c00954197_dsv41_cann930'
http = build_opener(ProxyHandler({}))
request = json.loads((run / 'request.json').read_text())
results = []
state = 'WAITING_SERVICE'
active_round = None


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stop(reason):
    try:
        with (run / 'STOP').open('x') as stream:
            stream.write(reason + '\n')
    except FileExistsError:
        pass


def check_runners():
    if (run / 'STOP').exists():
        raise RuntimeError((run / 'STOP').read_text().strip())
    for node in ('node23', 'node24'):
        if (run / node / 'completion.json').exists():
            raise RuntimeError(node + ' service runner exited')


def health():
    try:
        with http.open('http://177.42.23.1:8000/health', timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def counters():
    with http.open('http://177.42.23.1:8000/metrics', timeout=10) as response:
        lines = response.read().decode().splitlines()
    return {
        prefix: sum(float(line.split()[-1]) for line in lines if line.startswith(prefix))
        for prefix in ('vllm:request_success_total', 'vllm:generation_tokens_total')
    }


try:
    started = time.monotonic()
    save(run / 'stage.json', {'state': state, 'started_at_host23': time.time(), 'completed_rounds': 0})
    while not health():
        check_runners()
        if time.monotonic() - started >= 5400:
            raise TimeoutError('Service startup exceeded 5400 seconds')
        time.sleep(15)
    check_runners()
    with http.open('http://177.42.23.1:8000/v1/models', timeout=10) as response:
        models = json.load(response)
    assert any(item['id'] == 'dsv41-bf16-cann930' for item in models['data']), models
    save(run / 'service_ready.json', {'time_host23': time.time(), 'models': models, 'counters': counters()})
    print('SERVICE_READY', flush=True)

    for active_round in (1, 2, 3):
        directory = clients / ('round' + str(active_round))
        directory.mkdir(exist_ok=True)
        previous = directory / 'result.json'
        if previous.exists() and json.loads(previous.read_text()).get('pass'):
            results.append(json.loads(previous.read_text()))
            continue
        check_runners()
        assert health(), 'Service unhealthy before round'
        for item in request['files']:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
        state = 'GPQA_ROUND_' + str(active_round)
        attempt = directory / ('attempt' + str(len(list(directory.glob('attempt*'))) + 1))
        attempt.mkdir()
        launcher = directory / 'launcher.log'
        if launcher.is_symlink():
            launcher.unlink()
        elif launcher.exists():
            launcher.rename(attempt / 'previous_launcher.log')
        launcher.symlink_to(attempt / 'launcher.log')
        before = counters()
        save(run / 'stage.json', {
            'state': state, 'round': active_round, 'questions': 198,
            'started_at_host23': time.time(), 'completed_rounds': len(results),
            'counters_before': before, 'attempt_dir': str(attempt),
        })
        command = docker + [
            'exec', '-e', 'DSV41_PERFORMANCE_RUN=' + run.name, container,
            'bash', str(precision / 'run_gpqa.sh'), '198',
        ]
        save(attempt / 'command.json', command)
        started = time.monotonic()
        print('ROUND_STARTED', active_round, flush=True)
        with launcher.open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if (run / 'STOP').exists():
                    process.wait(timeout=30)
                    break
                time.sleep(5)
        text = launcher.read_text(errors='replace')
        match = re.search(r'Current exp folder:\s*(\S+)', text)
        result = {
            'round': active_round, 'exit_code': process.returncode,
            'seconds': time.monotonic() - started, 'attempt_dir': str(attempt),
            'counters_before': before, 'pass': False,
        }
        if match:
            result_dir = Path(match.group(1))
            result['result_dir'] = str(result_dir)
            audit_file = result_dir / 'gpqa_audit.json'
            if audit_file.exists():
                audit = json.loads(audit_file.read_text())
                result['audit'] = audit
                result['pass'] = (
                    process.returncode == 0 and audit['questions'] == 198
                    and audit['successful_requests'] == 198
                    and audit['max_out_len'] == 65536 and audit['scoring_validated']
                )
        save(attempt / 'result.json', result)
        save(directory / 'result.json', result)
        assert result['pass'], 'Round failed, inspect ' + str(attempt)
        result['counters_after'] = counters()
        save(directory / 'result.json', result)
        results.append(result)
        save(run / 'progress.json', {'completed_rounds': len(results), 'results': results})
        print('ROUND_FINISHED', active_round, 'ACCURACY', result['audit']['accuracy'], flush=True)

    state = 'COMPLETED'
    log_cutoffs = {}
    for node in ('node23', 'node24'):
        data = (run / node / 'launcher.log').read_bytes()
        log_cutoffs[node] = {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'error_lines': sum(' ERROR ' in line for line in data.decode(errors='replace').splitlines()),
        }
    save(run / 'summary.json', {
        'state': state, 'rounds': 3, 'questions_per_round': 198,
        'mean_accuracy': sum(item['audit']['accuracy'] for item in results) / 3,
        'results': results, 'server_logs_before_stop': log_cutoffs, 'counters_after': counters(),
    })
    stop('Three audited GPQA rounds completed; releasing this experiment')
    print(state, flush=True)
except Exception as error:
    state = 'YIELDED_TO_OTHER_TASK' if any((run / node / 'yield_reason.json').exists() for node in ('node23', 'node24')) else 'FAILED'
    (clients / 'error.txt').write_text(repr(error) + '\n')
    print(state, repr(error), flush=True)
    if not (run / 'STOP').exists() and not health():
        stop('Precision controller failed and service is unhealthy')
    if state == 'FAILED' and health():
        state = 'FAILED_SERVICE_HELD'
finally:
    save(clients / 'completion.json', {
        'state': state, 'active_round': active_round,
        'finished_at_host23': time.time(), 'results': results,
    })
    save(run / 'stage.json', {'state': state, 'active_round': active_round, 'completed_rounds': len(results)})

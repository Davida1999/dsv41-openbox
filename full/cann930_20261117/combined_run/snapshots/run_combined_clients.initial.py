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
record = run / 'clients'
record.mkdir(exist_ok=True)
lock = (record / 'controller.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
with (record / 'started.json').open('x') as output:
    json.dump({'host_started_at': time.time()}, output)

root = Path('/mnt/share/c00954197/Workspace/dsv41')
performance = root / 'performance_cann930'
precision = root / 'precision'
docker = ['docker', '-H', 'unix:///run/c00954197-cann930/docker.sock']
container = 'c00954197_dsv41_cann930'
http = build_opener(ProxyHandler({}))
results = []
state = 'WAITING_SERVICE'


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stop(reason):
    try:
        with (run / 'STOP').open('x') as output:
            output.write(reason + '\n')
    except FileExistsError:
        pass


def check_service():
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
        name: sum(float(line.split()[-1]) for line in lines if line.startswith(name))
        for name in ('vllm:generation_tokens_total', 'vllm:request_success_total')
    }


tasks = []
for scene, tokens in [('mm_4k_32k', 2097152), ('text_2k_6k', 393216), ('mm_1k_2k', 131072)]:
    for round_number in range(1, 5):
        tasks.append({
            'name': f'{scene}_round{round_number}', 'kind': 'performance',
            'scene': scene, 'round': round_number, 'expected_output_tokens': tokens,
            'script': str(performance / ('run_client_' + scene + '.sh')), 'args': [],
        })
for questions in (8, 198):
    tasks.append({
        'name': f'gpqa_{questions}', 'kind': 'accuracy', 'questions': questions,
        'script': str(precision / 'run_gpqa.sh'), 'args': [str(questions)],
    })
save(run / 'tasks.json', tasks)

try:
    print(state, flush=True)
    deadline = time.monotonic() + 5400
    while not health():
        check_service()
        if time.monotonic() >= deadline:
            raise TimeoutError('Service startup exceeded 5400 seconds')
        time.sleep(30)
    check_service()
    with http.open('http://177.42.23.1:8000/v1/models', timeout=10) as response:
        models = json.load(response)
    assert any(model['id'] == 'dsv41-bf16-cann930' for model in models['data']), models
    save(record / 'models.json', models)
    print('SERVICE_READY', flush=True)

    hashes = {item['path']: item['sha256'] for item in json.loads((run / 'request.json').read_text())['files']}
    for task in tasks:
        check_service()
        assert health(), 'Service health failed before ' + task['name']
        for path in [task['script'], str(precision / 'check_results.py')]:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == hashes[path], path + ' changed'
        state = task['name']
        current = dict(task, started_at_host23=time.time(), counters_before=counters())
        save(run / 'stage.json', current)
        command = docker + ['exec', '-e', 'DSV41_PERFORMANCE_RUN=' + run.name,
                            container, 'bash', task['script']] + task['args']
        save(record / (state + '_command.json'), command)
        print('CLIENT_STARTED', state, flush=True)
        launcher = record / (state + '_launcher.log')
        started = time.monotonic()
        with launcher.open('w') as output:
            exit_code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        result = dict(task, exit_code=exit_code, script_seconds=time.monotonic() - started)
        text = launcher.read_text(errors='replace')
        if task['kind'] == 'performance':
            match = re.search(r'^结果目录 (.+)$', text, re.M)
            assert match, 'Missing result directory for ' + state
            directory = performance / match.group(1).strip()
            metrics = json.loads((directory / 'benchmark.json').read_text())
            result.update(result_dir=str(directory), metrics=metrics)
            result['pass'] = (
                exit_code == 0 and metrics['completed'] == 64 and metrics['failed'] == 0
                and metrics['total_output_tokens'] == task['expected_output_tokens']
            )
        else:
            match = re.search(r'Current exp folder:\s*(\S+)', text)
            assert match, 'Missing AISBench result directory for ' + state
            directory = Path(match.group(1))
            audit = json.loads((directory / 'gpqa_audit.json').read_text())
            result.update(result_dir=str(directory), audit=audit)
            result['pass'] = (
                exit_code == 0 and audit['questions'] == task['questions']
                and audit['successful_requests'] == task['questions']
                and abs(audit['accuracy'] - audit['aisbench_accuracy']) < 1e-6
            )
        results.append(result)
        save(record / (state + '_result.json'), result)
        save(run / 'progress.json', {'state': state, 'completed_tasks': len(results), 'results': results})
        print('CLIENT_FINISHED', state, 'PASS', result['pass'], flush=True)
        assert result['pass'], state + ' failed output validation'
        check_service()
        assert health(), 'Service health failed after ' + state

    assert len(results) == 14 and all(result['pass'] for result in results)
    cutoffs = {}
    for node in ('node23', 'node24'):
        path = run / node / 'launcher.log'
        data = path.read_bytes()
        assert not any(' ERROR ' in line for line in data.decode(errors='replace').splitlines()), node
        cutoffs[node] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    save(run / 'validation_before_stop.json', {
        'status': 'PASS', 'performance_rounds': 12, 'accuracy_questions': [8, 198],
        'performance_output_tokens': sum(result['metrics']['total_output_tokens'] for result in results if result['kind'] == 'performance'),
        'server_log_cutoffs': cutoffs, 'counters_after': counters(),
    })
    state = 'FINISHED'
    print('ALL_TASKS_FINISHED', flush=True)
except Exception as error:
    state = 'YIELDED_TO_OTHER_TASK' if any((run / node / 'yield_reason.json').exists() for node in ('node23', 'node24')) else 'FAILED'
    (record / 'error.txt').write_text(repr(error) + '\n')
    print('CONTROLLER_FAILED', state, repr(error), flush=True)
finally:
    save(record / 'completion.json', {'state': state, 'finished_at_host23': time.time(), 'results': results})
    stop('Combined performance and GPQA controller ended with ' + state)

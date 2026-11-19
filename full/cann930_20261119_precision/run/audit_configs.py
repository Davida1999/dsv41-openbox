"""Verify the actual AISBench configuration saved for each GPQA round."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

run = Path('/mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734')
sys.path.insert(0, str(run.parents[1]))
from mmengine.config import Config

expected_generation = {
    'temperature': 1.0, 'top_p': 0.95, 'top_k': 0, 'seed': 42,
    'ignore_eos': False,
    'chat_template_kwargs': {'reasoning_effort': 'low', 'enable_thinking': True},
}
expected_model = {
    'model': 'dsv41-bf16-cann930', 'batch_size': 8, 'max_out_len': 65536,
    'path': '/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16',
    'host_ip': '177.42.23.1', 'host_port': 8000, 'request_timeout': 360000,
}
rounds = []
for number in (1, 2, 3):
    launcher = run / 'clients' / ('round' + str(number)) / 'launcher.log'
    match = re.search(r'Current exp folder:\s*(\S+)', launcher.read_text(errors='replace'))
    assert match, number
    experiment = Path(match.group(1))
    paths = list((experiment / 'configs').glob('*.py'))
    assert len(paths) == 1, paths
    model = Config.fromfile(str(paths[0])).models[0]
    assert dict(model['generation_kwargs']) == expected_generation, model['generation_kwargs']
    assert all(model[key] == value for key, value in expected_model.items()), model
    rounds.append({
        'round': number, 'config_path': str(paths[0]),
        'config_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        'generation_kwargs': expected_generation, 'model_settings': expected_model,
    })
request = json.loads((run / 'request.json').read_text())
assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'] for item in request['files'])
versions = {}
for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'ais_bench_benchmark'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
result = {'status': 'PASS', 'rounds': rounds, 'source_hashes_match_request': True, 'package_versions': versions}
(run / 'config_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'verified_rounds': len(rounds), 'package_versions': versions}, ensure_ascii=False))

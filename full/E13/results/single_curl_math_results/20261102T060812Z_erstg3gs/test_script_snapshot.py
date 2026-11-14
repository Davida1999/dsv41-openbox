#!/usr/bin/env python3
"""E13: ten sequential curl requests; retain responses and token logprobs."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

BASE_URL = 'http://177.11.27.1:8000'
HERE = Path(__file__).resolve().parent
ROOT = HERE / 'single_curl_math_results'
ROOT.mkdir(exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN = Path(tempfile.mkdtemp(prefix=stamp + '_', dir=ROOT))
shutil.copy2(__file__, RUN / 'test_script_snapshot.py')
payload = {
    'model': 'dsv41-bf16-tp1',
    'prompt': '计算 123 + 456。只输出整数答案，不要解释。\n答案：',
    'max_tokens': 64,
    'temperature': 0.0,
    'top_p': 1.0,
    'top_k': -1,
    'seed': 42,
    'logprobs': 5,
    'stream': False,
    'n': 1,
}
request_file = RUN / 'request.json'
request_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

def snapshot(name, endpoint):
    with (RUN / name).open('w') as f:
        subprocess.run(['curl', '--noproxy', '*', '-sS', '--max-time', '10',
                        BASE_URL + endpoint], stdout=f, stderr=subprocess.DEVNULL)

snapshot('models.json', '/v1/models')
snapshot('metrics_before.txt', '/metrics')
rows = []
print('Output directory:', RUN, flush=True)
for i in range(1, 11):
    prefix = RUN / f'request_{i:02d}'
    response_file = Path(str(prefix) + '.response.json')
    command = ['curl', '--noproxy', '*', '-sS', '--connect-timeout', '10',
               '--max-time', '180', '-D', str(prefix) + '.headers.txt',
               '-o', str(response_file), '-w', '%{http_code}',
               '-H', 'Content-Type: application/json',
               '--data-binary', '@' + str(request_file), BASE_URL + '/v1/completions']
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    row = {'index': i, 'http_status': result.stdout.strip(),
           'curl_exit': result.returncode, 'elapsed_seconds': time.monotonic() - start,
           'started_utc': started_utc,
           'request_sha256': hashlib.sha256(request_file.read_bytes()).hexdigest()}
    Path(str(prefix) + '.stderr.txt').write_text(result.stderr)
    try:
        data = json.loads(response_file.read_text())
        if result.returncode or row['http_status'] != '200':
            raise ValueError(str(data))
        choice = data['choices'][0]
        row.update(text=choice['text'], finish_reason=choice.get('finish_reason'),
                   usage=data.get('usage'), logprobs=choice.get('logprobs'))
        row['correct_exact'] = row['text'].strip() == '579'
        lp = row['logprobs'] or {}
        row['logprobs_present'] = bool(lp.get('tokens')) and len(lp['tokens']) == len(lp.get('token_logprobs', [])) and all(isinstance(v, (float, int)) for v in lp['token_logprobs'])
    except Exception as e:
        row['error'] = str(e)
    rows.append(row)
    Path(str(prefix) + '.summary.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    print(f'Request {i}/10: HTTP {row["http_status"]}, {row["elapsed_seconds"]:.3f}s, output={row.get("text")!r}', flush=True)

snapshot('metrics_after.txt', '/metrics')
valid = [r for r in rows if 'error' not in r]
complete = len(valid) == 10
ref = valid[0] if valid else None
same_text = complete and all(r['text'] == ref['text'] for r in valid)
has_lp = complete and all(r['logprobs_present'] for r in valid)
same_tokens = has_lp and all(r['logprobs']['tokens'] == ref['logprobs']['tokens'] for r in valid)
same_lp = same_tokens and all(r['logprobs']['token_logprobs'] == ref['logprobs']['token_logprobs'] for r in valid)
same_top5 = has_lp and all(r['logprobs'].get('top_logprobs') == ref['logprobs'].get('top_logprobs') for r in valid)
max_delta = max((abs(a-b) for r in valid for a,b in zip(r['logprobs']['token_logprobs'], ref['logprobs']['token_logprobs'])), default=0) if same_tokens else None
summary = {'completed': len(valid), 'failed': 10-len(valid),
           'correct_exact_count': sum(r.get('correct_exact', False) for r in rows),
           'all_text_identical': same_text, 'all_token_strings_identical': same_tokens,
           'all_selected_token_logprobs_exactly_identical': same_lp,
           'all_top5_logprobs_exactly_identical': same_top5,
           'max_abs_selected_logprob_delta': max_delta, 'runs': rows}
(RUN / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
report = ['E13：单 curl 连续 10 次数学推理一致性测试',
          '问题：计算 123 + 456，只输出整数答案。标准答案：579。',
          '每次等待响应完成后再发送下一次；temperature=0、top_p=1、top_k=-1、seed=42；logprobs=5。',
          'log p 为自然对数概率；附表中的 exp(log p) 是对应概率。',
          '本测试验证单请求重复运行一致性，不验证不同 batch 大小之间的一致性。',
          '保留现有服务缓存；请求可能分配到不同 DP Engine，未进行 rank 固定。',
          '仅比较生成文本及 logprobs；响应 ID、时间戳不参与一致性比较。',
          '完整原始响应、逐次请求概要和前后 metrics 均保存在本目录。', '',
          json.dumps({k:v for k,v in summary.items() if k != 'runs'}, ensure_ascii=False, indent=2), '']
for r in rows:
    report += [f'===== 第 {r["index"]} 次 =====', f'HTTP={r["http_status"]} 耗时={r["elapsed_seconds"]:.6f}s',
               '完整生成内容：', r.get('text', '<请求失败>'),
               'finish_reason: ' + str(r.get('finish_reason')),
               'usage: ' + json.dumps(r.get('usage'), ensure_ascii=False)]
    if 'error' in r: report.append('错误：' + r['error'])
    lp = r.get('logprobs') or {}
    for j, token in enumerate(lp.get('tokens', [])):
        value = lp['token_logprobs'][j]
        prob = math.exp(value) if isinstance(value, (float, int)) else None
        top = lp.get('top_logprobs') or []
        report.append(f'token[{j}]={token!r} log_p={value!r} p={prob!r}')
        if j < len(top): report.append('  top5: ' + json.dumps(top[j], ensure_ascii=False))
    report.append('')
(RUN / 'report.txt').write_text('\n'.join(report) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'runs'}, ensure_ascii=False), flush=True)
print('Report:', RUN / 'report.txt', flush=True)
raise SystemExit(0 if complete and has_lp else 1)

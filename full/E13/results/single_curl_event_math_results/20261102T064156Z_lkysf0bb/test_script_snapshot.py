#!/usr/bin/env python3
"""E13: ten sequential curl requests; retain responses and token logprobs."""
import datetime
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

BASE_URL = 'http://177.11.27.1:8000'
HERE = Path(__file__).resolve().parent
ROOT = HERE / 'single_curl_event_math_results'
ROOT.mkdir(exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN = Path(tempfile.mkdtemp(prefix=stamp + '_', dir=ROOT))
shutil.copy2(__file__, RUN / 'test_script_snapshot.py')
payload = {
    'model': 'dsv41-bf16-tp1',
    'prompt': '某机构举办一天的公益科普活动，按以下明确规则核算，不考虑税费、折旧或其他收支。成人票标价180元，共售240张，其中60张早鸟票打85折，其余原价；儿童票标价90元，共售80张，其中20张早鸟票打8折，其余原价。活动前退回12张成人原价票和8张儿童原价票，按原价全额退款，没有其他退款。其余购票者全部到场，每人一张票。\n赞助合同金额12000元，活动完成后全部确认为本期收入，实际到账9000元，余款下月收到。成人餐食每位65元，儿童每位40元，只按实际到场人数采购且当期全部付清。场地费18000元全部计入本期费用，本期支付12000元，余款下月支付。设备租赁6000元，工作人员8人每人350元，均当期全额支付。平台佣金为退款后净票款的2%，本期已支付。全部票款及退款均在本期收付，没有期初应收应付和现金预收预付。现金流必须按净额口径列示：将退票直接冲减票款流入，退款不再重复列入现金流出。\n请展示可核验的分步算式，计算净票款、到场人数、餐费、平台费、本期收入和费用、利润、净现金流、期末应收和应付，并说明利润与净现金流差额。最后单独给出合法JSON，金额保留到分但使用数字值，键必须为：adult_net_ticket, child_net_ticket, attendees, meals, platform_fee, revenue, expenses, profit, cash_inflow, cash_outflow, net_cash_flow, accounts_receivable, accounts_payable。不要只给JSON，须先给简洁计算说明，每步只需一条算式，不要反复核算或讨论其他口径。',
    'max_tokens': 8192,
    'chat_template_kwargs': {'thinking': False},
    'temperature': 0.0,
    'top_p': 1.0,
    'top_k': -1,
    'seed': 42,
    'logprobs': True,
    'top_logprobs': 5,
    'stream': False,
    'n': 1,
}
payload['cache_salt'] = 'E13-event-' + RUN.name
payload['messages'] = [{'role': 'user', 'content': payload.pop('prompt')}]
request_file = RUN / 'request.json'
request_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

def snapshot(name, endpoint):
    with (RUN / name).open('w') as f:
        subprocess.run(['curl', '--noproxy', '*', '-sS', '--max-time', '10',
                        BASE_URL + endpoint], stdout=f, stderr=subprocess.DEVNULL)

(RUN / 'expected_answer.json').write_text(json.dumps({'adult_net_ticket': 39420, 'child_net_ticket': 6120, 'attendees': 300, 'meals': 17700, 'platform_fee': 910.8, 'revenue': 57540, 'expenses': 45410.8, 'profit': 12129.2, 'cash_inflow': 54540, 'cash_outflow': 39410.8, 'net_cash_flow': 15129.2, 'accounts_receivable': 3000, 'accounts_payable': 6000}, ensure_ascii=False, indent=2) + '\n')
snapshot('models.json', '/v1/models')
snapshot('metrics_before.txt', '/metrics')
rows = []
print('Output directory:', RUN, flush=True)
for i in range(1, 11):
    prefix = RUN / f'request_{i:02d}'
    response_file = Path(str(prefix) + '.response.json')
    command = ['curl', '--noproxy', '*', '-sS', '--connect-timeout', '10',
               '--max-time', '900', '-D', str(prefix) + '.headers.txt',
               '-o', str(response_file), '-w', '%{http_code}',
               '-H', 'Content-Type: application/json',
               '--data-binary', '@' + str(request_file), BASE_URL + '/v1/chat/completions']
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
        content = (choice.get('logprobs') or {}).get('content') or []
        normalized_lp = {'tokens': [x['token'] for x in content],
                         'token_logprobs': [x['logprob'] for x in content],
                         'top_logprobs': [{y['token']: y['logprob'] for y in x.get('top_logprobs', [])} for x in content]}
        row.update(text=choice['message'].get('content') or '', finish_reason=choice.get('finish_reason'),
                   usage=data.get('usage'), logprobs=normalized_lp)
        expected = {'adult_net_ticket': 39420, 'child_net_ticket': 6120, 'attendees': 300, 'meals': 17700, 'platform_fee': 910.8, 'revenue': 57540, 'expenses': 45410.8, 'profit': 12129.2, 'cash_inflow': 54540, 'cash_outflow': 39410.8, 'net_cash_flow': 15129.2, 'accounts_receivable': 3000, 'accounts_payable': 6000}
        candidates = re.findall(r'\{[^{}]*\}', row['text'])
        parsed = None
        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
                break
            except ValueError:
                pass
        row['parsed_answer'] = parsed
        row['field_checks'] = {k: isinstance((parsed or {}).get(k), (int, float)) and not isinstance((parsed or {}).get(k), bool) and abs((parsed or {})[k] - v) < 1e-8 for k, v in expected.items()}
        row['correct_exact'] = all(row['field_checks'].values()) and row['finish_reason'] != 'length'
        lp = row['logprobs'] or {}
        row['logprobs_present'] = bool(lp.get('tokens')) and len(lp['tokens']) == len(lp.get('token_logprobs', [])) and all(isinstance(v, (float, int)) for v in lp['token_logprobs'])
    except Exception as e:
        row['error'] = str(e)
    rows.append(row)
    Path(str(prefix) + '.summary.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    print(f'Request {i}/10: HTTP {row["http_status"]}, {row["elapsed_seconds"]:.3f}s, correct={row.get("correct_exact")}, finish={row.get("finish_reason")}', flush=True)

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
report = ['E13：活动票务与收支核算，单 curl 连续 10 次一致性测试',
          '问题及核算规则见 request.json；标准答案见 expected_answer.json。',
          '每次等待响应完成后再发送下一次；temperature=0、top_p=1、top_k=-1、seed=42；logprobs=5。',
          'log p 为自然对数概率；附表中的 exp(log p) 是对应概率。',
          '本测试验证单请求重复运行一致性，不验证不同 batch 大小之间的一致性。',
          '本轮使用独立cache_salt隔离历史前缀缓存；十次共享本轮标识，允许轮内复用。服务未重启，运行时预热状态保留；DP自动路由。',
          '仅比较生成文本及 logprobs；响应 ID、时间戳不参与一致性比较。',
          '完整原始响应、逐次请求概要和前后 metrics 均保存在本目录。', '',
          json.dumps({k:v for k,v in summary.items() if k != 'runs'}, ensure_ascii=False, indent=2), '']
for r in rows:
    report += [f'===== 第 {r["index"]} 次 =====', f'HTTP={r["http_status"]} 耗时={r["elapsed_seconds"]:.6f}s',
               '完整生成内容：', r.get('text', '<请求失败>'),
               'finish_reason: ' + str(r.get('finish_reason')),
               'usage: ' + json.dumps(r.get('usage'), ensure_ascii=False), '各项答案核对: ' + json.dumps(r.get('field_checks'), ensure_ascii=False)]
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

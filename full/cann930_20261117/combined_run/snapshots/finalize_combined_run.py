from datetime import datetime, timezone
from pathlib import Path
import difflib
import hashlib
import json
import statistics
import tarfile
import time
from zoneinfo import ZoneInfo

root = Path('/mnt/share/c00954197/Workspace/dsv41')
run = root / 'precision/runs/combined_20261117_R1'
clients = run / 'clients'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def preserve(path):
    backup = path.with_name(path.stem + '.before_finalization' + path.suffix)
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_time(timestamp):
    return datetime.fromtimestamp(timestamp, ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')


results = []
performance = []
precision = []
for task in read(run / 'tasks.json'):
    path = clients / (task['name'] + '_result.json')
    result = read(path)
    directory = Path(result['result_dir'])
    if task['kind'] == 'performance':
        metrics = read(directory / 'benchmark.json')
        assert metrics == result['metrics']
        assert result['exit_code'] == 0
        assert (directory / 'exit_code.txt').read_text().strip() == '0'
        assert metrics['completed'] == 64 and metrics['failed'] == 0
        assert metrics['total_output_tokens'] == task['expected_output_tokens']
        assert result['pass']
        performance.append(result)
    else:
        preserve(path)
        audit = read(directory / 'gpqa_audit.json')
        assert audit['scoring_validated']
        assert audit['questions'] == audit['successful_requests'] == task['questions']
        assert abs(audit['aisbench_accuracy'] - audit['aisbench_raw_recomputed_accuracy']) < 1e-6
        assert audit['scoring_difference_ids'] == ([] if task['questions'] == 8 else [52])
        check_log = clients / (task['name'] + '_scoring_recheck.log')
        logged_audit, _ = json.JSONDecoder().raw_decode(check_log.read_text().lstrip())
        assert logged_audit == audit
        prediction = directory / 'predictions/dsv41-bf16-cann930-gpqa/GPQA_diamond.jsonl'
        with prediction.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        assert sorted(row['id'] for row in rows) == list(range(task['questions']))
        assert all(row['success'] for row in rows)
        result.update(
            audit=audit, original_script_exit_code=result.get('original_script_exit_code', result['exit_code']),
            exit_code=1, audit_recheck_exit_code=0, final_validation_exit_code=0,
            recovery='Existing predictions and original AISBench score preserved. CPU-only JSONL reader and scoring reconciliation recheck completed.',
            prediction_sha256=sha(prediction), generated_tokens=sum(row['output_tokens'] for row in rows),
            scoring_recheck_log=str(check_log),
        )
        result['pass'] = True
        result['pass_scope'] = 'All requested inference and scoring steps completed, with audited scores and disclosed recovery. No accuracy threshold was specified.'
        save(path, result)
        precision.append(result)
    results.append(result)

assert len(results) == 14 and len(performance) == 12 and len(precision) == 2
release = {node: read(run / node / 'release_validation.json') for node in ['node23', 'node24']}
assert all(row['release_verified'] and row['remaining_owned'] == [] for row in release.values())
save(run / 'release_check.json', release)

# Preserve the complete shutdown error context alongside the pre-stop log hashes.
shutdown_context = []
with tarfile.open(run / 'node23/cann_logs.tar.gz') as archive:
    for member in archive.getmembers():
        lines = archive.extractfile(member).read().decode(errors='replace').splitlines()
        indices = [index for index, line in enumerate(lines) if 'ERROR' in line]
        if indices:
            shutdown_context.append({
                'file': member.name,
                'context': lines[max(0, indices[0] - 3):indices[-1] + 4],
            })
assert len(shutdown_context) == 1
assert any('00:29:03' in line and 'EngineDeadError' in line for line in shutdown_context[0]['context'])
save(run / 'shutdown_log_review.json', {
    'stop_requested_at_utc': datetime.fromtimestamp(read(run / 'server_log_cutoffs.json')['captured_at_host23'], timezone.utc).isoformat(),
    'before_stop_error_lines': {'node23': 0, 'node24': 0},
    'post_stop_error_context': shutdown_context,
    'assessment': 'Node23 API output handler logged EngineDeadError after STOP during engine cleanup. All generation had completed before STOP.',
})

checker = root / 'precision/check_results.py'
(run / 'snapshots/check_results.reconciled.py').write_bytes(checker.read_bytes())
original_checker = run / 'snapshots/check_results.py'
(run / 'check_results_full_fix.patch').write_text(''.join(difflib.unified_diff(
    original_checker.read_text().splitlines(keepends=True), checker.read_text().splitlines(keepends=True),
    fromfile='check_results.original.py', tofile='check_results.reconciled.py',
)))
request = read(run / 'request.json')
checker_record = next(item for item in request['files'] if item['path'] == str(checker))
if checker_record['sha256'] != sha(checker):
    request['amendments'].append({
        'time_host23': time.time(), 'path': str(checker),
        'old_sha256': checker_record['sha256'], 'new_sha256': sha(checker),
        'reason': 'Recompute raw AISBench extraction independently and retain final-content score. Accept correctness differences only for empty final content. No inference rerun.',
    })
    checker_record['sha256'] = sha(checker)
    save(run / 'request.json', request)

means = {
    scene: statistics.mean(result['metrics']['output_throughput'] for result in performance if result['scene'] == scene)
    for scene in ['mm_4k_32k', 'text_2k_6k', 'mm_1k_2k']
}
summary = {
    'state': 'COMPLETED_WITH_AUDIT_RECOVERY', 'task_disposition': 'COMPLETE',
    'completed_tasks': 14, 'performance_rounds': 12,
    'performance_requests': sum(result['metrics']['completed'] for result in performance),
    'performance_failed': sum(result['metrics']['failed'] for result in performance),
    'performance_output_tokens': sum(result['metrics']['total_output_tokens'] for result in performance),
    'performance_mean_output_tokens_per_second': means,
    'precision': precision, 'all_requests': 974,
    'all_generated_tokens': sum(result['metrics']['total_output_tokens'] for result in performance) + sum(result['generated_tokens'] for result in precision),
    'single_model_load_reused': True, 'service_released': True,
    'accuracy_acceptance_threshold': None,
    'resource_release': release,
    'performance': [{key: value for key, value in result.items() if key != 'metrics'} | {
        'metrics': {key: result['metrics'][key] for key in [
            'duration', 'completed', 'failed', 'total_input_tokens', 'total_output_tokens',
            'output_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'mean_itl_ms',
            'p99_ttft_ms', 'p99_tpot_ms', 'p99_itl_ms',
        ]},
    } for result in performance],
}
assert summary['performance_requests'] == 768 and summary['performance_failed'] == 0
assert summary['performance_output_tokens'] == 10485760
assert summary['all_generated_tokens'] == 12621655
save(run / 'results_summary.json', summary)

labels = {'mm_4k_32k': '多模态 4K→32K', 'text_2k_6k': '文本 2K→6K', 'mm_1k_2k': '多模态 1K→2K'}
lines = [
    '# CANN 9.3.0 双节点性能与 GPQA 组合实验报告', '',
    '## 完成情况', '',
    '本轮在 177.42.23.1 主节点和 177.42.24.1 子节点完成一次服务启动，依次执行12轮性能压测、GPQA 8题预检和198题正式评测，随后停止服务。共完成974个请求，生成12621655个token。两台设备的16张NPU均已无运行进程，8000和12346端口已释放。', '',
    '任务处置为 COMPLETE。12轮性能请求全部成功且完整输出。GPQA两轮均已完成生成、评分及CPU复核，精度数值按实测记录。本轮精度结果包含输出预算命中和评分差异。', '',
    f'服务器报告目录为 `{run}`。本报告时间采用 Asia/Shanghai。24节点系统时钟约快10.5分钟，单节点事件顺序和客户端单调计时用于核对执行过程。', '',
    '## 服务与环境', '',
    '| 项目 | 本轮配置 |', '|---|---|',
    '| 容器 | 两台均为 c00954197_dsv41_cann930 |',
    '| Docker socket | unix:///run/c00954197-cann930/docker.sock |',
    '| 模型 | /mnt/share/c00954197/workspace/dsv41-flash-bf16 |',
    '| 模型结构 | 完整BF16，40层，384个路由专家，top-6，视觉模块及Engram开启 |',
    '| 设备 | 每节点8张Ascend950DT，共16张 |',
    '| 并行 | TP1 / PCP8 / DP2 / EP16，PP1 / DCP1，每节点local DP1 |',
    '| 调度与图 | async-scheduling / FULL_DECODE_ONLY |',
    '| 通信 | CCU_SCHED，HCCL enp35s0f2，Gloo enp39s0f0 |',
    '| 上下文 / batched tokens | 40960 / 2048 |',
    '| 每DP最大序列数 / 显存比例 | 32 / 0.96 |',
    '| Engram | CPU offload，DP shared memory |',
    '| Tokenizer / reasoning parser | deepseek_v41 / deepseek_v41 |',
    '| API | http://177.42.23.1:8000，dsv41-bf16-cann930 |',
    '| 驱动 / CANN | 26.2.0.b020 / 9.3.0 |',
    '| Python / Torch / TorchNPU | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 |',
    '| vLLM | 0.32.0+empty，5df63a7203cdb9328d824a27b682e77c7b9928be |',
    '| vLLM-Ascend | 0.19.1rc2.dev3401+g28e9e97ba，28e9e97ba254b076a88275e8a275556cb9a11664 |',
    '| AISBench | 3.1.20261113，2c7434077ddacf23a55e4eb595cfd4f59244c00e |', '',
    '服务入口使用 precision 目录的主、子节点脚本，显式设置seed=42及deepseek_v41 tokenizer/reasoning parser。两台保留已验证的Gloo CPU超时透传修复，源码SHA256为 `ccf708b35a36e1c5606cc9c0841253d9a08e2887f8f7ccd3368d16a8f72ca2c6`。各节点环境、8个rank的CCU配置及实际启动命令保存于 `node23/` 和 `node24/`。', '',
    '## 性能结果', '',
    '按多模态4K→32K、文本2K→6K、多模态1K→2K的顺序，各执行4轮。每轮64请求、并发64，seed=42、request-rate=10000、ignore-eos、长度浮动0。多模态每请求1张高1920×宽1080图片，输入长度为文本目标，实际服务输入包含聊天模板和图像token。', '',
    '| 场景 | 轮次 | 成功/失败 | 总输入token | 总输出token | 时长s | 输出tokens/s | 平均TTFT ms | 平均TPOT ms | 平均ITL ms |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|',
]
for result in performance:
    m = result['metrics']
    lines.append(f"| {labels[result['scene']]} | {result['round']} | {m['completed']}/{m['failed']} | {m['total_input_tokens']} | {m['total_output_tokens']} | {m['duration']:.3f} | {m['output_throughput']:.3f} | {m['mean_ttft_ms']:.3f} | {m['mean_tpot_ms']:.3f} | {m['mean_itl_ms']:.3f} |")
lines += ['', '| 场景 | 四轮平均输出tokens/s |', '|---|---:|']
lines += [f'| {labels[scene]} | {value:.3f} |' for scene, value in means.items()]
lines += [
    '', '平均值取每场景四轮输出吞吐的算术平均，包含首轮，保留全部12轮结果。服务启用prefix caching，轮次之间沿用缓存。相同负载连续执行带来TTFT变化，本表记录此服务复用条件下的测量值。', '',
    '### 各轮原始结果目录', '',
]
lines += [f"- {labels[result['scene']]} 第{result['round']}轮，`{result['result_dir']}`。" for result in performance]
lines += [
    '', '每个目录的 `benchmark.log` 保存客户端终端性能日志，`benchmark.json` 保存完整指标，`exit_code.txt` 保存原始退出码。所有12轮原始退出码均为0。', '',
    '## GPQA精度结果', '',
    '数据集使用GPQA Diamond，0-shot CoT chat prompt。配置为thinking=true、reasoning_effort=100、max_out_len=32768、temperature=0、top_p=1、seed=42、并发8、warmups=0。198题提示词的CPU长度检查最大输入2595 token，加输出预算后为35363 token，处于40960上下文内。', '',
    '| 阶段 | 成功请求 | AISBench原始正确数/分数 | 最终content正确数/分数 | 达输出上限 | 无可提取最终答案 | 输出token |',
    '|---|---:|---|---|---:|---:|---:|',
]
for result in precision:
    a = result['audit']
    raw_correct = round(a['aisbench_accuracy'] * a['questions'] / 100)
    lines.append(f"| {a['questions']}题 | {a['successful_requests']}/{a['questions']} | {raw_correct}/{a['questions']}，{a['aisbench_accuracy']:.6f}% | {a['correct']}/{a['questions']}，{a['accuracy']:.6f}% | {len(a['output_limit_ids'])} | {len(a['unparsed_answer_ids'])} | {result['generated_tokens']} |")
lines += [
    '', '8题预检的零基题号6达到32768输出上限且无可提取最终答案。198题的24个对应零基题号如下。', '',
    '```text', ', '.join(map(str, precision[1]['audit']['output_limit_ids'])), '```', '',
    '198题评分差异位于零基题号52，即第53题。该题输出32768 token，最终content为空。AISBench把推理文本写入prediction，并从推理文本提取到D，标准答案也是D，原始评分将其计对。最终content审计将空答案计错，得到164/198。两种口径差0.505051个百分点。', '',
    '复核脚本独立复算AISBench原始提取和评分，验证与原分数一致，再核对最终content分数。正确性差异仅出现在最终content为空的记录。原始预测、原始AISBench评分及原始失败日志均已保留。', '',
    '### 结果与数据证据', '',
]
for result in precision:
    lines.append(f"- {result['questions']}题结果位于 `{result['result_dir']}`，预测SHA256为 `{result['prediction_sha256']}`。")
lines += [
    f'- 数据快照位于 `{run}/snapshots/gpqa_diamond.csv`，SHA256为 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。',
    f'- 提示词快照位于 `{run}/snapshots/gpqa_gen_0_shot_cot_chat_prompt.py`，SHA256为 `558f8c24040df367ea7834d433c3a2ad8efb48366949c04faf5fd9cb869de5ff`。',
    '', '每个精度结果目录保留 `predictions/dsv41-bf16-cann930-gpqa/GPQA_diamond.jsonl`、`results/dsv41-bf16-cann930-gpqa/GPQA_diamond.json`、`summary/`、`gpqa_audit.json`。正式结果另存 `scoring_difference.json`。', '',
    '## 审计修正与执行偏差', '',
    '原8题生成和AISBench评分已完成时，审计脚本对整个JSONL文本调用str.splitlines()。一条JSON字符串中包含U+2028，被拆成额外行，导致审计退出1。修复为UTF-8文件逐物理行读取，Unicode回归检查通过，使用已生成的8条预测重新审计得到7/8。', '',
    '198题原审计要求最终content分数与AISBench原始分数相等，因题号52的空最终回答差异退出1。修正为原始分数独立复算、差异记录核对和两种分数同时保留。两次CPU审计复核均退出0，模型生成记录保持原样，服务持续复用直至正式推理完成。两轮原脚本退出码1保留在结果元数据中，最终校验退出码为0。', '',
    '本轮8题预检出现1个输出预算命中，确认服务请求和评分复核链路可用后，继续完成用户指定的198题评测。198题中24个预算命中及其空最终答案随精度数值记录。', '',
    f'修复证据保存于 `{run}/check_results_reader_fix.patch`、`check_results_full_fix.patch`、`snapshots/check_results.reconciled.py`、`request.initial.json` 和 `request.json` 的amendments。原执行控制器及原失败completion保留，最终汇总使用CPU复核结果。', '',
    '## 服务复用与释放', '',
]
for node in ['node23', 'node24']:
    runner = read(run / node / 'runner.json')
    completion = read(run / node / 'completion.json')
    lines.append(f"- {node}监护器从本机时间 {local_time(runner['started_at'])} 运行至 {local_time(completion['finished_at'])}，最终状态为STOP_REQUESTED，remaining_owned为空。采集{release[node]['telemetry_samples']}次NPU占用快照，竞争样本为0。")
lines += [
    '', '两端从同一次启动持续服务于12轮性能和两轮精度请求。各节点只出现一次引擎初始化，工作进程在全过程保持同一组PID。原始服务启动命令、日志、监护记录与客户端执行顺序均已归档。', '',
    'STOP前保存并复核两端服务日志的字节长度及SHA256，ERROR和Traceback均为0。停止标记于UTC 2026-11-17 00:28:59写入后，23节点API日志于00:29:03在引擎清理期间记录AsyncLLM output_handler的EngineDeadError，共8行ERROR。对应上下文及停止前日志边界已归档。各轮请求在STOP前全部完成。两端启动日志还记录ArgSort整型路径使用AiCPU的警告。', '',
    f'资源释放证据位于 `{run}/release_check.json` 及两节点的 `released_npu.txt`、`released_processes.txt`、`released_ports.txt`、`completion.json`。CANN/vLLM相关进程日志分别归档为 `node23/cann_logs.tar.gz` 和 `node24/cann_logs.tar.gz`，文件清单及哈希保存在各自 `cann_logs_manifest.json`。停服日志核对保存在 `shutdown_log_review.json`。', '',
    '## 复现入口', '',
    '在对应宿主机进入容器。', '',
    '```bash', 'unset DOCKER_CONTEXT', 'export DOCKER_HOST=unix:///run/c00954197-cann930/docker.sock', 'docker exec -it c00954197_dsv41_cann930 bash', '```', '',
    '23和24分别使用以下服务脚本。完整执行参数以本轮snapshots为准。', '',
    f'- 主节点脚本为 `{root}/precision/run_master.sh`。',
    f'- 子节点脚本为 `{root}/precision/run_worker.sh`。', '',
    '23容器内依次运行以下客户端即可复用服务。', '',
    '```bash',
    'for round in 1 2 3 4; do', f'  bash {root}/performance_cann930/run_client_mm_4k_32k.sh', 'done',
    'for round in 1 2 3 4; do', f'  bash {root}/performance_cann930/run_client_text_2k_6k.sh', 'done',
    'for round in 1 2 3 4; do', f'  bash {root}/performance_cann930/run_client_mm_1k_2k.sh', 'done',
    f'bash {root}/precision/run_gpqa.sh 8', f'bash {root}/precision/run_gpqa.sh 198', '```', '',
    'CPU评分复核的实际命令如下，两条均已退出0。', '', '```bash',
]
for result in precision:
    lines.append(f"docker -H unix:///run/c00954197-cann930/docker.sock exec c00954197_dsv41_cann930 python3 {checker} {result['result_dir']} {result['questions']}")
lines += ['```', '', '脚本保持IP、模型路径、长度和并发直接赋值的风格，修改相应行即可。原始控制器已结束，本轮STOP标记已生效。', '']
(run / 'REPORT.md').write_text('\n'.join(lines))

for path in [run / 'progress.json', clients / 'completion.json', clients / 'error.txt']:
    preserve(path)
save(run / 'validation_final.json', {
    'status': 'PASS', 'scope': 'Requested execution, output completeness, independent scoring reconciliation, and resource release',
    'completed_tasks': 14, 'raw_accuracy_script_exit_codes': [1, 1],
    'cpu_audit_recheck_exit_codes': [0, 0],
    'performance_requests': 768, 'accuracy_requests': 206,
    'all_generated_tokens': summary['all_generated_tokens'],
    'accuracy_threshold_specified': False, 'service_released': True,
})
save(run / 'progress.json', {'state': summary['state'], 'completed_tasks': 14, 'results': results})
completion = {
    'state': summary['state'], 'finished_at_host23': time.time(),
    'completed_tasks': 14, 'service_released': True,
    'original_controller_state': 'FAILED_SERVICE_HELD',
    'recovery': 'CPU-only scoring rechecks. Predictions and original AISBench results retained.',
    'report': str(run / 'REPORT.md'), 'results_summary': str(run / 'results_summary.json'),
}
save(run / 'completion.json', completion)
save(clients / 'completion.json', completion)
print(json.dumps({key: summary[key] for key in ['state','completed_tasks','performance_mean_output_tokens_per_second','all_requests','all_generated_tokens','service_released']}, indent=2))

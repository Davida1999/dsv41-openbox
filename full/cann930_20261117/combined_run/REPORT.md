# CANN 9.3.0 双节点性能与 GPQA 组合实验报告

## 完成情况

本轮在 177.42.23.1 主节点和 177.42.24.1 子节点完成一次服务启动，依次执行12轮性能压测、GPQA 8题预检和198题正式评测，随后停止服务。共完成974个请求，生成12621655个token。两台设备的16张NPU均已无运行进程，8000和12346端口已释放。

任务处置为 COMPLETE。12轮性能请求全部成功且完整输出。GPQA两轮均已完成生成、评分及CPU复核，精度数值按实测记录。本轮精度结果包含输出预算命中和评分差异。

服务器报告目录为 `/mnt/share/c00954197/Workspace/dsv41/precision/runs/combined_20261117_R1`。本报告时间采用 Asia/Shanghai。24节点系统时钟约快10.5分钟，单节点事件顺序和客户端单调计时用于核对执行过程。

## 服务与环境

| 项目 | 本轮配置 |
|---|---|
| 容器 | 两台均为 c00954197_dsv41_cann930 |
| Docker socket | unix:///run/c00954197-cann930/docker.sock |
| 模型 | /mnt/share/c00954197/workspace/dsv41-flash-bf16 |
| 模型结构 | 完整BF16，40层，384个路由专家，top-6，视觉模块及Engram开启 |
| 设备 | 每节点8张Ascend950DT，共16张 |
| 并行 | TP1 / PCP8 / DP2 / EP16，PP1 / DCP1，每节点local DP1 |
| 调度与图 | async-scheduling / FULL_DECODE_ONLY |
| 通信 | CCU_SCHED，HCCL enp35s0f2，Gloo enp39s0f0 |
| 上下文 / batched tokens | 40960 / 2048 |
| 每DP最大序列数 / 显存比例 | 32 / 0.96 |
| Engram | CPU offload，DP shared memory |
| Tokenizer / reasoning parser | deepseek_v41 / deepseek_v41 |
| API | http://177.42.23.1:8000，dsv41-bf16-cann930 |
| 驱动 / CANN | 26.2.0.b020 / 9.3.0 |
| Python / Torch / TorchNPU | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 |
| vLLM | 0.32.0+empty，5df63a7203cdb9328d824a27b682e77c7b9928be |
| vLLM-Ascend | 0.19.1rc2.dev3401+g28e9e97ba，28e9e97ba254b076a88275e8a275556cb9a11664 |
| AISBench | 3.1.20261113，2c7434077ddacf23a55e4eb595cfd4f59244c00e |

服务入口使用 precision 目录的主、子节点脚本，显式设置seed=42及deepseek_v41 tokenizer/reasoning parser。两台保留已验证的Gloo CPU超时透传修复，源码SHA256为 `ccf708b35a36e1c5606cc9c0841253d9a08e2887f8f7ccd3368d16a8f72ca2c6`。各节点环境、8个rank的CCU配置及实际启动命令保存于 `node23/` 和 `node24/`。

## 性能结果

按多模态4K→32K、文本2K→6K、多模态1K→2K的顺序，各执行4轮。每轮64请求、并发64，seed=42、request-rate=10000、ignore-eos、长度浮动0。多模态每请求1张高1920×宽1080图片，输入长度为文本目标，实际服务输入包含聊天模板和图像token。

| 场景 | 轮次 | 成功/失败 | 总输入token | 总输出token | 时长s | 输出tokens/s | 平均TTFT ms | 平均TPOT ms | 平均ITL ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 多模态 4K→32K | 1 | 64/0 | 327233 | 2097152 | 941.099 | 2228.407 | 19224.264 | 28.101 | 28.856 |
| 多模态 4K→32K | 2 | 64/0 | 327233 | 2097152 | 934.647 | 2243.790 | 9596.589 | 28.207 | 28.607 |
| 多模态 4K→32K | 3 | 64/0 | 327233 | 2097152 | 944.969 | 2219.281 | 8417.828 | 28.564 | 28.651 |
| 多模态 4K→32K | 4 | 64/0 | 327233 | 2097152 | 940.154 | 2230.647 | 7982.875 | 28.431 | 28.566 |
| 文本 2K→6K | 1 | 64/0 | 132992 | 393216 | 182.020 | 2160.291 | 4865.748 | 28.775 | 29.394 |
| 文本 2K→6K | 2 | 64/0 | 132992 | 393216 | 176.355 | 2229.686 | 2062.710 | 28.337 | 28.403 |
| 文本 2K→6K | 3 | 64/0 | 132992 | 393216 | 176.542 | 2227.319 | 1270.352 | 28.517 | 28.670 |
| 文本 2K→6K | 4 | 64/0 | 132992 | 393216 | 175.986 | 2234.358 | 1159.127 | 28.450 | 28.494 |
| 多模态 1K→2K | 1 | 64/0 | 130623 | 131072 | 70.928 | 1847.958 | 8912.948 | 30.046 | 30.274 |
| 多模态 1K→2K | 2 | 64/0 | 130623 | 131072 | 73.521 | 1782.774 | 6931.700 | 32.280 | 32.415 |
| 多模态 1K→2K | 3 | 64/0 | 130623 | 131072 | 74.936 | 1749.108 | 7609.830 | 32.610 | 32.797 |
| 多模态 1K→2K | 4 | 64/0 | 130623 | 131072 | 65.632 | 1997.070 | 5894.833 | 28.910 | 29.066 |

| 场景 | 四轮平均输出tokens/s |
|---|---:|
| 多模态 4K→32K | 2230.531 |
| 文本 2K→6K | 2212.914 |
| 多模态 1K→2K | 1844.228 |

平均值取每场景四轮输出吞吐的算术平均，包含首轮，保留全部12轮结果。服务启用prefix caching，轮次之间沿用缓存。相同负载连续执行带来TTFT变化，本表记录此服务复用条件下的测量值。

### 各轮原始结果目录

- 多模态 4K→32K 第1轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_4k_32k/20261116T205522Z_zRmtCa`。
- 多模态 4K→32K 第2轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_4k_32k/20261116T211122Z_d05oZn`。
- 多模态 4K→32K 第3轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_4k_32k/20261116T212716Z_WNFjMZ`。
- 多模态 4K→32K 第4轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_4k_32k/20261116T214320Z_Cqk0Xa`。
- 文本 2K→6K 第1轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/text_2k_6k/20261116T215920Z_yLSInZ`。
- 文本 2K→6K 第2轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/text_2k_6k/20261116T220236Z_yz1k2a`。
- 文本 2K→6K 第3轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/text_2k_6k/20261116T220547Z_CUc2vU`。
- 文本 2K→6K 第4轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/text_2k_6k/20261116T220857Z_9uRr5a`。
- 多模态 1K→2K 第1轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_1k_2k/20261116T221208Z_scB8wm`。
- 多模态 1K→2K 第2轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_1k_2k/20261116T221333Z_YHqkFW`。
- 多模态 1K→2K 第3轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_1k_2k/20261116T221502Z_hjwCFl`。
- 多模态 1K→2K 第4轮，`/mnt/share/c00954197/Workspace/dsv41/performance_cann930/results/mm_1k_2k/20261116T221632Z_MyA2B0`。

每个目录的 `benchmark.log` 保存客户端终端性能日志，`benchmark.json` 保存完整指标，`exit_code.txt` 保存原始退出码。所有12轮原始退出码均为0。

## GPQA精度结果

数据集使用GPQA Diamond，0-shot CoT chat prompt。配置为thinking=true、reasoning_effort=100、max_out_len=32768、temperature=0、top_p=1、seed=42、并发8、warmups=0。198题提示词的CPU长度检查最大输入2595 token，加输出预算后为35363 token，处于40960上下文内。

| 阶段 | 成功请求 | AISBench原始正确数/分数 | 最终content正确数/分数 | 达输出上限 | 无可提取最终答案 | 输出token |
|---|---:|---|---|---:|---:|---:|
| 8题 | 8/8 | 7/8，87.500000% | 7/8，87.500000% | 1 | 1 | 54888 |
| 198题 | 198/198 | 165/198，83.333333% | 164/198，82.828283% | 24 | 24 | 2081007 |

8题预检的零基题号6达到32768输出上限且无可提取最终答案。198题的24个对应零基题号如下。

```text
10, 15, 27, 48, 52, 64, 71, 74, 76, 79, 81, 88, 90, 94, 99, 101, 118, 127, 130, 140, 145, 147, 162, 187
```

198题评分差异位于零基题号52，即第53题。该题输出32768 token，最终content为空。AISBench把推理文本写入prediction，并从推理文本提取到D，标准答案也是D，原始评分将其计对。最终content审计将空答案计错，得到164/198。两种口径差0.505051个百分点。

复核脚本独立复算AISBench原始提取和评分，验证与原分数一致，再核对最终content分数。正确性差异仅出现在最终content为空的记录。原始预测、原始AISBench评分及原始失败日志均已保留。

### 结果与数据证据

- 8题结果位于 `/mnt/share/c00954197/Workspace/dsv41/precision/results/gpqa_20261116T221752Z_vAo8bD/20261116_221753`，预测SHA256为 `1a0fa04316b9d5f75f6bd1f94a8a5a745bb5d3a5fe0d9a9804efc6d893498da5`。
- 198题结果位于 `/mnt/share/c00954197/Workspace/dsv41/precision/results/gpqa_20261116T223457Z_llif2p/20261116_223457`，预测SHA256为 `6099992560a3b729c9c0de83198f9226741f186da85a327353e2c5e5f822eba2`。
- 数据快照位于 `/mnt/share/c00954197/Workspace/dsv41/precision/runs/combined_20261117_R1/snapshots/gpqa_diamond.csv`，SHA256为 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。
- 提示词快照位于 `/mnt/share/c00954197/Workspace/dsv41/precision/runs/combined_20261117_R1/snapshots/gpqa_gen_0_shot_cot_chat_prompt.py`，SHA256为 `558f8c24040df367ea7834d433c3a2ad8efb48366949c04faf5fd9cb869de5ff`。

每个精度结果目录保留 `predictions/dsv41-bf16-cann930-gpqa/GPQA_diamond.jsonl`、`results/dsv41-bf16-cann930-gpqa/GPQA_diamond.json`、`summary/`、`gpqa_audit.json`。正式结果另存 `scoring_difference.json`。

## 审计修正与执行偏差

原8题生成和AISBench评分已完成时，审计脚本对整个JSONL文本调用str.splitlines()。一条JSON字符串中包含U+2028，被拆成额外行，导致审计退出1。修复为UTF-8文件逐物理行读取，Unicode回归检查通过，使用已生成的8条预测重新审计得到7/8。

198题原审计要求最终content分数与AISBench原始分数相等，因题号52的空最终回答差异退出1。修正为原始分数独立复算、差异记录核对和两种分数同时保留。两次CPU审计复核均退出0，模型生成记录保持原样，服务持续复用直至正式推理完成。两轮原脚本退出码1保留在结果元数据中，最终校验退出码为0。

本轮8题预检出现1个输出预算命中，确认服务请求和评分复核链路可用后，继续完成用户指定的198题评测。198题中24个预算命中及其空最终答案随精度数值记录。

修复证据保存于 `/mnt/share/c00954197/Workspace/dsv41/precision/runs/combined_20261117_R1/check_results_reader_fix.patch`、`check_results_full_fix.patch`、`snapshots/check_results.reconciled.py`、`request.initial.json` 和 `request.json` 的amendments。原执行控制器及原失败completion保留，最终汇总使用CPU复核结果。

## 服务复用与释放

- node23监护器从本机时间 2026-11-17 04:26:10 运行至 2026-11-17 08:29:13，最终状态为STOP_REQUESTED，remaining_owned为空。采集4357次NPU占用快照，竞争样本为0。
- node24监护器从本机时间 2026-11-17 04:36:47 运行至 2026-11-17 08:39:56，最终状态为STOP_REQUESTED，remaining_owned为空。采集4373次NPU占用快照，竞争样本为0。

两端从同一次启动持续服务于12轮性能和两轮精度请求。各节点只出现一次引擎初始化，工作进程在全过程保持同一组PID。原始服务启动命令、日志、监护记录与客户端执行顺序均已归档。

STOP前保存并复核两端服务日志的字节长度及SHA256，ERROR和Traceback均为0。停止标记于UTC 2026-11-17 00:28:59写入后，23节点API日志于00:29:03在引擎清理期间记录AsyncLLM output_handler的EngineDeadError，共8行ERROR。对应上下文及停止前日志边界已归档。各轮请求在STOP前全部完成。两端启动日志还记录ArgSort整型路径使用AiCPU的警告。

资源释放证据位于 `/mnt/share/c00954197/Workspace/dsv41/precision/runs/combined_20261117_R1/release_check.json` 及两节点的 `released_npu.txt`、`released_processes.txt`、`released_ports.txt`、`completion.json`。CANN/vLLM相关进程日志分别归档为 `node23/cann_logs.tar.gz` 和 `node24/cann_logs.tar.gz`，文件清单及哈希保存在各自 `cann_logs_manifest.json`。停服日志核对保存在 `shutdown_log_review.json`。

## 复现入口

在对应宿主机进入容器。

```bash
unset DOCKER_CONTEXT
export DOCKER_HOST=unix:///run/c00954197-cann930/docker.sock
docker exec -it c00954197_dsv41_cann930 bash
```

23和24分别使用以下服务脚本。完整执行参数以本轮snapshots为准。

- 主节点脚本为 `/mnt/share/c00954197/Workspace/dsv41/precision/run_master.sh`。
- 子节点脚本为 `/mnt/share/c00954197/Workspace/dsv41/precision/run_worker.sh`。

23容器内依次运行以下客户端即可复用服务。

```bash
for round in 1 2 3 4; do
  bash /mnt/share/c00954197/Workspace/dsv41/performance_cann930/run_client_mm_4k_32k.sh
done
for round in 1 2 3 4; do
  bash /mnt/share/c00954197/Workspace/dsv41/performance_cann930/run_client_text_2k_6k.sh
done
for round in 1 2 3 4; do
  bash /mnt/share/c00954197/Workspace/dsv41/performance_cann930/run_client_mm_1k_2k.sh
done
bash /mnt/share/c00954197/Workspace/dsv41/precision/run_gpqa.sh 8
bash /mnt/share/c00954197/Workspace/dsv41/precision/run_gpqa.sh 198
```

CPU评分复核的实际命令如下，两条均已退出0。

```bash
docker -H unix:///run/c00954197-cann930/docker.sock exec c00954197_dsv41_cann930 python3 /mnt/share/c00954197/Workspace/dsv41/precision/check_results.py /mnt/share/c00954197/Workspace/dsv41/precision/results/gpqa_20261116T221752Z_vAo8bD/20261116_221753 8
docker -H unix:///run/c00954197-cann930/docker.sock exec c00954197_dsv41_cann930 python3 /mnt/share/c00954197/Workspace/dsv41/precision/check_results.py /mnt/share/c00954197/Workspace/dsv41/precision/results/gpqa_20261116T223457Z_llif2p/20261116_223457 198
```

脚本保持IP、模型路径、长度和并发直接赋值的风格，修改相应行即可。原始控制器已结束，本轮STOP标记已生效。

# CANN 9.3.0 GPQA Diamond 精度回归

实验目录为 `/mnt/share/c00954197/Workspace/dsv41/precision`。本次按用户选择的快速精度回归口径编写脚本，真实模型精度运行待执行。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 主节点 / 子节点 | 177.42.23.1 / 177.42.24.1 |
| 容器 | 两端均为 c00954197_dsv41_cann930 |
| 模型 | /mnt/share/c00954197/workspace/dsv41-flash-bf16 |
| 模型结构 | 完整 BF16，40 层、384 个路由专家、top-6，视觉模块与 Engram 开启 |
| 并行与调度 | TP1 / PCP8 / DP2 / EP16，异步调度，FULL_DECODE_ONLY |
| 通信 | HCCL_OP_EXPANSION_MODE=CCU_SCHED |
| 服务地址 / 模型名 | http://177.42.23.1:8000 / dsv41-bf16-cann930 |
| 服务最大上下文 | 40960 tokens |
| 工具 | 已安装的 AISBench，源码提交 2c7434077ddacf23a55e4eb595cfd4f59244c00e |
| 数据 | GPQA Diamond，198 题，每题生成一次 |
| 提示词 | AISBench 内置 gpqa_gen_0_shot_cot_chat_prompt，0-shot CoT，要求最终输出 Answer: A/B/C/D |
| 思考设置 | thinking=true，reasoning_effort=100 |
| 采样 | temperature=0，top_p=1.0，seed=42，ignore_eos=false |
| 输出上限 | 32768 tokens，包含推理与最终答案 |
| 客户端并发 / warmup | 8 / 0 |
| 请求尝试 | retry=1，当前 AISBench 实现中每题尝试一次 |

数据文件为 `/mnt/share/c00954197/Workspace/dsv41/benchmark/ais_bench/datasets/gpqa/gpqa_diamond.csv`。AISBench 使用固定选项轮换规则，正确答案同步轮换。指标为 accuracy/pass@1，正式结果按正确题数除以 198 计算。

198 题的 CPU 分词检查得到最长输入 2595 tokens，与 32768 输出预算合计 35363 tokens，处于本方案上下文范围内。

## 文件与修改方式

```text
precision/
├── README.md
├── run_master.sh
├── run_worker.sh
├── run_gpqa.sh
└── check_results.py
```

服务端和客户端均为独立完整脚本，IP、端口、权重路径、并行度、思考强度及输出长度直接写在命令中。修改相应行即可。题数作为客户端的一个位置参数，预检传 8，正式运行传 198。

`runs/`、`logs/`、`results/` 在执行时生成。每次客户端运行创建独立目录，AISBench 在其下保存配置快照、预测、评分、明细和汇总。

## 1. 启动双节点服务

使用已经部署并验证的 CANN 9.3.0 容器。该环境包含节点对应的 HCCL rootinfo 及 Gloo CPU 超时透传修复，安装与补丁记录见 `/mnt/share/c00954197/Workspace/dsv41/performance_cann930/PERFORMANCE_REPORT.md`。

先在两台宿主机检查设备占用。

```bash
npu-smi info
```

以下命令复用已有资源监护器，监护器在宿主机调用专用 Docker socket，发现其他 NPU 作业时停止本次双节点任务。23 为主节点，24 为子节点。

在 **23 宿主机**执行。

```bash
python3 /mnt/share/c00954197/deployments/performance_cann930_runs/run_node_yield.py \
  23 master \
  /mnt/share/c00954197/Workspace/dsv41/precision/runs/gpqa_20261117_R1 \
  /mnt/share/c00954197/Workspace/dsv41/precision/run_master.sh
```

在 **24 宿主机**另开终端执行。

```bash
python3 /mnt/share/c00954197/deployments/performance_cann930_runs/run_node_yield.py \
  24 worker \
  /mnt/share/c00954197/Workspace/dsv41/precision/runs/gpqa_20261117_R1 \
  /mnt/share/c00954197/Workspace/dsv41/precision/run_worker.sh
```

两端使用同一个运行目录。重新启动实验时，两条命令中的运行目录一起改成新的名称。服务日志位于 `precision/logs/`，监护记录位于本次 `precision/runs/` 目录。

## 2. 预检 8 题

在 23 宿主机另开终端进入容器。

```bash
export DOCKER_HOST=unix:///run/c00954197-cann930/docker.sock
docker exec -it c00954197_dsv41_cann930 bash
```

确认健康接口返回成功，模型列表包含 `dsv41-bf16-cann930`。

```bash
curl --noproxy '*' --fail http://177.42.23.1:8000/health
curl --noproxy '*' --fail http://177.42.23.1:8000/v1/models
cd /mnt/share/c00954197/Workspace/dsv41/precision
bash run_gpqa.sh 8
```

预检检查请求能完成、最终答案可提取及评分链路正常。客户端结束时自动审计 8 条预测和 AISBench 评分，同时保存无法提取最终答案的题号、达到输出上限的题号。

## 3. 正式评测 198 题

复用已就绪服务，在 23 容器内执行。

```bash
cd /mnt/share/c00954197/Workspace/dsv41/precision
bash run_gpqa.sh 198
```

每轮客户端首先打印日志文件路径，AISBench 打印 `Current exp folder` 指向本轮时间戳结果目录。执行顺序为生成、AISBench 评分、汇总、最终答案审计。

当前 AISBench 的实测行为中，`--dry-run` 仍会发送请求，请求全部失败时进程也可能返回 0。客户端因此在 AISBench 结束后自动执行结果审计，题号缺失、重复、请求失败或分数核对异常时返回非零状态。

## 4. 查看结果

AISBench 时间戳结果目录包含以下主要文件。

```text
predictions/dsv41-bf16-cann930-gpqa/GPQA_diamond.jsonl
results/dsv41-bf16-cann930-gpqa/GPQA_diamond.json
summary/
gpqa_audit.json
```

`gpqa_audit.json` 记录题数、成功数、正确数、最终答案 accuracy、AISBench 原始 accuracy、答案提取失败题号和达到 32768 输出上限的题号。未提取到最终答案的题目计错。对带推理字段的响应，审计使用服务返回的最终 `content`。

正式结果应有题号 0～197 的 198 条成功记录，最终答案审计与 AISBench 分数一致。精度值按实测填写，答案提取失败和输出预算命中情况随分数一起记录。达到输出上限的题目可查看原始预测，核对最终答案是否已完整生成。

也可在容器内手动重新审计某轮结果。

```bash
python3 /mnt/share/c00954197/Workspace/dsv41/precision/check_results.py \
  /完整路径/AISBench时间戳结果目录 198
```

## 同一个服务复用性能与精度测试

使用本目录的主、子节点脚本完成一次模型加载后，可以按以下顺序串行运行性能与精度客户端。两类客户端均请求 `http://177.42.23.1:8000` 上的 `dsv41-bf16-cann930`。

```bash
bash /mnt/share/c00954197/Workspace/dsv41/performance_cann930/run_client_mm_1k_2k.sh
bash /mnt/share/c00954197/Workspace/dsv41/performance_cann930/run_client_text_2k_6k.sh
bash /mnt/share/c00954197/Workspace/dsv41/performance_cann930/run_client_mm_4k_32k.sh
bash /mnt/share/c00954197/Workspace/dsv41/precision/run_gpqa.sh 8
bash /mnt/share/c00954197/Workspace/dsv41/precision/run_gpqa.sh 198
```

客户端逐请求提供采样、输出长度和思考设置。当前精度上下文 40960、输出预算 32768 与上述性能负载相容。

两套服务脚本的差异为工作目录和日志位置，以及精度脚本显式增加的 `--seed 42`、`--tokenizer-mode deepseek_v41`、`--reasoning-parser deepseek_v41`。原性能实测使用 `performance_cann930` 目录的启动参数。使用本节统一入口得到的新性能结果，应记录精度入口的启动参数。该统一配置的两类测试组合运行仍待实测。

## 5. 结束与资源让出

完成后，在宿主机向本次共享运行目录写入停止标记，两端监护器会清理本次进程。

```bash
echo 'GPQA completed' > /mnt/share/c00954197/Workspace/dsv41/precision/runs/gpqa_20261117_R1/STOP
```

随后在两台宿主机执行 `npu-smi info`，并核对监护目录的 `completion.json`。设备竞争时按已有规则让出资源，保留对方任务。

## 思考强度与官方资料

当前容器中的 vLLM tokenizer 支持 `chat_template_kwargs.thinking` 和整数 `reasoning_effort=1…100`。当前命名映射为 `low=50`、`high=75`、`xhigh=75`、`max=100`，默认 `high=75`。本方案由客户端逐请求显式设置 100，服务端启用 `--tokenizer-mode deepseek_v41` 和 `--reasoning-parser deepseek_v41`。

DeepSeek 模型卡的 Instruct 结果使用思考强度 100、temperature=1.0、top_p=0.95；通用推理建议为 max_tokens 至少 256K，GPQA 专用输出上限仍待确认。本轮按用户选择使用表中的快速回归参数。[DeepSeek 官方模型说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

vLLM-Ascend 的公开 GPQA 结果中，950DT 行给出 temperature=1.0、top_p=1.0；该行的思考强度和输出预算仍需实验记录确认。A3 行明确给出 max_tokens=131072。[vLLM-Ascend 精度说明](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html#7-accuracy-evaluation)

## 编写阶段验证

Shell 与 Python 语法检查通过。198 题数据加载及 CPU 提示词长度检查通过。两题 CPU 模拟接口已验证请求参数、生成记录、AISBench 评分及最终答案审计链路。请求全部失败的模拟检查也确认客户端最终返回非零状态。真实模型精度评测待执行。

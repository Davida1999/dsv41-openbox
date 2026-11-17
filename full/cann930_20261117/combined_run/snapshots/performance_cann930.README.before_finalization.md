# CANN 9.3.0 异步推理性能实验

实验目录为 `/mnt/share/c00954197/Workspace/dsv41/performance_cann930`，注意 `Workspace` 的大小写。脚本直接在对应服务器的 `c00954197_dsv41_cann930` 容器内运行。

## 文件

```text
performance_cann930/
├── README.md
├── run_master.sh
├── run_worker.sh
├── run_client_text_2k_6k.sh
├── run_client_mm_1k_2k.sh
├── run_client_mm_4k_32k.sh
├── logs/                       # 执行服务脚本时生成
└── results/                    # 执行客户端时生成
    ├── text_2k_6k/
    ├── mm_1k_2k/
    └── mm_4k_32k/
```

IP、端口、模型路径、并行度、请求数、并发和输入输出长度直接写在脚本命令中，修改对应行即可。脚本保留时间戳日志和每轮独立结果目录。

## 服务配置

| 项目 | 配置 |
|---|---|
| 主节点 | 177.42.23.1 |
| 子节点 | 177.42.24.1，headless，DP start-rank 1 |
| 模型 | /mnt/share/c00954197/workspace/dsv41-flash-bf16 |
| 权重和结构 | 完整 BF16，40 层，384 个路由专家，top-6，包含视觉模块 |
| Engram | 开启，CPU offload及DP共享内存 |
| 并行 | TP1 / PCP8 / DP2 / EP16，每节点8张NPU |
| 调度 / 图 | async-scheduling / FULL_DECODE_ONLY |
| HCCL展开模式 | CCU_SCHED |
| 最大上下文 / 批处理tokens | 40960 / 2048 |
| 每DP最大序列数 / 显存比例 | 32 / 0.96 |
| API / DP RPC端口 | 8000 / 12346 |
| HCCL / Gloo网卡 | enp35s0f2 / enp39s0f0 |

配置来源是 E25 成功运行快照 `E25/runs_930/node25_27_20261114T062142Z/`，本实验使用异步调度和当前节点地址。

## 1. 进入容器

在相应服务器的宿主机终端执行以下命令。

```bash
unset DOCKER_CONTEXT
export DOCKER_HOST=unix:///run/c00954197-cann930/docker.sock
docker exec -it c00954197_dsv41_cann930 bash
```

## 2. 启动服务

本轮按设备空闲时运行、遇竞争时让出的规则执行。运行期间持续监测 NPU，发现其他任务占用时停止本次主、子节点服务并释放本次 NPU 进程，让对方运行。

先检查两台服务器的设备占用和端口可用情况。

```bash
npu-smi info
ss -lntp | grep -E ':(8000|12346)[[:space:]]'
```

在 **23 节点容器**中启动主节点。

```bash
cd /mnt/share/c00954197/Workspace/dsv41/performance_cann930
bash run_master.sh
```

在 **24 节点容器**中另开终端启动子节点。

```bash
cd /mnt/share/c00954197/Workspace/dsv41/performance_cann930
bash run_worker.sh
```

服务脚本前台运行。日志分别写入 `logs/master_<UTC时间>_<后缀>.log` 和 `logs/worker_<UTC时间>_<后缀>.log`，日志文件名同时显示在终端中。

## 3. 执行客户端

在 23 节点另开终端进入容器，确认服务就绪。

```bash
curl --noproxy '*' --fail http://177.42.23.1:8000/health
curl --noproxy '*' --fail http://177.42.23.1:8000/v1/models
```

模型列表应包含 `dsv41-bf16-cann930`。本轮按以下顺序各执行一轮，复用同一个服务。

```bash
cd /mnt/share/c00954197/Workspace/dsv41/performance_cann930

bash run_client_mm_1k_2k.sh
bash run_client_text_2k_6k.sh
bash run_client_mm_4k_32k.sh
```

| 客户端 | 文本输入目标 | 图片 | 输出/请求 | 请求数 / 并发 | 全部完成时总输出 |
|---|---:|---|---:|---|---:|
| run_client_text_2k_6k.sh | 2048 | 无 | 6144 | 64 / 64 | 393216 |
| run_client_mm_1k_2k.sh | 1024 | 高1920×宽1080，1张 | 2048 | 64 / 64 | 131072 |
| run_client_mm_4k_32k.sh | 4096 | 高1920×宽1080，1张 | 32768 | 64 / 64 | 2097152 |

输入长度是文本目标，实际输入还包含聊天模板和图像tokens。图片bucket顺序为 `(高度,宽度,帧数)`。客户端沿用 E25 的 seed=42、request-rate=10000、ignore-eos和长度浮动0，采样使用服务默认值。

每轮结果位于 `results/<场景>/<UTC时间>_<后缀>/`，包含以下文件。

- `benchmark.log` 保存客户端终端日志。
- `benchmark.json` 保存性能指标。
- `exit_code.txt` 保存客户端退出码。

检查64成功、0失败，以及表中完整输出token总数，再结合两端服务日志判断结果。流式异常截断可能仍被计为成功请求。

## 4. 停止和验证范围

在对应的服务终端按 `Ctrl+C` 结束服务，再用 `npu-smi info` 检查设备释放情况。

两台容器已使用各节点生成的 2.0 格式 `/etc/hccl_rootinfo.json`，双机 16 卡 HCCL 检查通过。

当前 23、24 容器已应用 Gloo CPU 超时透传修复，两台各 21 项回归测试通过。补丁位于 `/mnt/share/c00954197/deployments/performance_cann930_runs/cpu_timeout_fix_20261116/cpu_timeout_fix.patch`。使用原归档重建容器时，可应用该补丁恢复本次验证环境。

2026-11-17 的三组完整输出验证已通过，服务脚本已显式设置 `HCCL_OP_EXPANSION_MODE=CCU_SCHED`。指标、结果目录、历史故障及设备释放记录见同目录的 `PERFORMANCE_REPORT.md`。

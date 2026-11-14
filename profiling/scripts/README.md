# E23 纯文本 2K 到 6K Profiling 采集

在 25、26 的 `c00954197_vllm_ascend_1108` 容器中串行进行两轮采集。两轮仅改变 `torch_profiler_with_stack`，其余模型、并行、负载和 profiler 参数相同。脚本不会自动停止已有服务、清理其他任务或修改权重。

## 配置

| 项目 | 设置 |
|---|---|
| 模型 | `/mnt/share/c00954197/workspace/dsv41-flash-bf16` |
| 模型名 | `dsv41-bf16-tp1` |
| 并行 | TP1 / PCP8 / DP2，开启 EP |
| Engram | CPU offload 与 DP shared memory 开启 |
| Gloo 接口 | `enp39s0f0` |
| HCCL 接口 | `enp35s0f2` |
| API | `http://177.11.25.1:8000`；26 为 headless worker |
| Profiling | torch；delay=1536；max_iterations=10；ignore_frontend=true |
| 两轮差异 | no_stack=false；with_stack=true，指 `torch_profiler_with_stack` |
| 客户端 | random 纯文本，2048 输入、6144 输出，无图片 |
| 请求数 / 并发 | 64 / 64，ignore-eos，seed=42 |
| 服务端每实例最大序列 | 32 |

主、子节点都配置 profiler。调用主节点一次 `/start_profile` 即会向两端 DP 引擎及 worker 下发。当前 Ascend 实现将 `torch_profiler_with_stack` 映射到 TorchNPU 的 `with_modules`，其原生 `with_stack` 保持 false；这两轮比较的是该版本实际实现，不应误称已启用 TorchNPU 原生调用栈选项。

## 第一轮 不启用 stack 配置

确认旧服务已按需停止、两台设备及端口可用，再分别在两端容器启动：

```bash
# 25.1
bash /workspace/experiments/E23_profile/run_master_no_stack.sh

# 26.1，另一终端
bash /workspace/experiments/E23_profile/run_worker_no_stack.sh
```

等待主节点健康接口返回 200，两端初始化完成：

```bash
curl --noproxy '*' --fail http://177.11.25.1:8000/health
bash /workspace/experiments/E23_profile/start_profile.sh
bash /workspace/experiments/E23_profile/run_client_text_2k_6k.sh no_stack
```

客户端不会自动调用 start/stop_profile。保持 `ready-check-timeout-sec=0`，避免额外的端点预检生成请求消耗延迟计数；先手动完成健康检查。

## 结束采集及第二轮

`Profiler started` 表示启动请求已接受，不表示数据已导出。delay 以 worker 执行步计数，1536 不等于精确的输出 token 位置或秒数。关注 worker 的延迟启动及达到采集上限日志，并核对两端目录均生成完整 trace。各 worker 独立计数，采集窗口不保证严格同一墙钟时间。

采集上限是本版本 wrapper 的停止阈值；确切窗口和边界以实际 trace 为准，不依据 API 日志认定恰好十次前向已保存。

```bash
bash /workspace/experiments/E23_profile/stop_profile.sh
```

确认两端导出/分析完成后，可以在客户端终端按 Ctrl-C 结束剩余请求；不需要等待所有 6K 输出完成。提前结束的 benchmark 可能没有最终 JSON，不能当作完整性能成绩。服务需要切换 stack 配置时，再有序停止本实验服务；不要在导出期间直接 SIGKILL。脚本不提供批量 kill 功能。

第二轮以相同顺序操作：

```bash
# 25.1
bash /workspace/experiments/E23_profile/run_master_with_stack.sh

# 26.1
bash /workspace/experiments/E23_profile/run_worker_with_stack.sh

# 服务就绪后，在客户端终端
curl --noproxy '*' --fail http://177.11.25.1:8000/health
bash /workspace/experiments/E23_profile/start_profile.sh
bash /workspace/experiments/E23_profile/run_client_text_2k_6k.sh with_stack
```

两套服务使用相同端口，因此不能同时运行。客户端的模式参数只选择结果目录，不切换服务器配置；运行前核对所启动的服务脚本。

## 数据目录

```text
no_stack/ 或 with_stack/
├── profiling/
│   ├── node25/run_<UTC时间>_<随机后缀>/
│   └── node26/run_<UTC时间>_<随机后缀>/
├── server_logs/<master或worker>_<UTC时间>_<随机后缀>.log
├── client_results/run_<UTC时间>_<随机后缀>/
│   ├── benchmark.log
│   ├── command.sh
│   ├── client_snapshot.sh
│   ├── exit_code.txt
│   └── benchmark.json       # 仅正常完成整次压测时保证产生
├── node25_profile_dir.txt   # 最新一次服务的采集目录指针
├── node26_profile_dir.txt
├── master.pid
└── worker.pid
```

每次启动的节点采集目录均唯一，两节点的时间戳无需相同。最新目录指针和 PID 文件会更新，历史采集、日志和客户端结果不覆盖。PID 文件只作定位提示，操作进程前仍须核对归属和启动时间。

## 按需离线解析

当前 TorchNPU trace handler 默认自动分析。先检查是否已有解析产物，避免无必要的重复分析。需要离线解析时，在对应容器执行：

```bash
python /workspace/experiments/E23_profile/analyse_profile.py \
  /workspace/experiments/E23_profile/no_stack/profiling/node25/实际运行目录
```

对 node26 和 with_stack 的实际目录分别执行。分析脚本不会启动推理或修改模型。

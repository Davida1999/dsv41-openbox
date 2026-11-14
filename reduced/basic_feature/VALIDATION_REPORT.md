# DeepSeek-V4.1 Flash BF16 开箱基础功能验证报告

日期：2026-11-13（Asia/Shanghai）。本轮在 **177.11.26.1 单机**执行，未使用25设备。RUN目录时间为主机时间，容器日志通常为UTC（相差8小时）。

## 结论

**未达到全项验收通过。** BF16服务和多条功能通路可运行，但裁剪模型在中文、英文、代码、算术请求中持续出现重复退化，不能作为合格精度golden。sleep level-1的5轮恢复一致性和采样显存检查通过；TP2、PP2、routing capture、sleep level-2和完整权重更新存在明确失败。逐项运行、输出一致性、质量和阻塞分开记录如下。

## 固定环境和范围

| 项目 | 实际值 |
|---|---|
| 服务器 | 177.11.26.1，8×Ascend950DT，单卡96GiB |
| 容器 | c00954197_vllm_ascend_1108 |
| 镜像标签 | c00954197/vllm-ascend:main-a5-20261108-e8215a9d；以实际Git提交为准 |
| vLLM / Ascend | 5df63a7203cdb9328d824a27b682e77c7b9928be / 28e9e97ba254b076a88275e8a275556cb9a11664 |
| CANN / Python | 9.2.1（V100R001C12SPC002B201）/ 3.12.16 |
| Torch / TorchNPU | 2.10.0+cpu / 2.10.0.post7 |
| 驱动报告版本 | npu-smi 25.1.rc3 |
| 唯一模型 | `/mnt/share/g00815994/dsv41/DeepSeek-V4.1-Flash-top32-bf16-runtime-codex-20261030` |
| 模型配置 | 40层，32 routed experts，top-6，Engram关闭，DSpark关闭 |
| 权重类型 | header核实BF16 5869个、F32 387个；无权重量化配置 |
| 精度边界 | dtype=bfloat16、quantization=None；KV cache=auto，包含模型内部混合量化注意力路径，不声称全链路缓存/算子均BF16 |
| 工作目录 | `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature` |
| 范围 | 文本推理；不覆盖视觉、工具调用、训练侧路由重放或完整未裁剪模型精度 |

所有模型服务均加载真实checkpoint；未修改原模型文件或框架源码。配置、测试客户端和监测脚本的修正已留痕。工作区内脚本保持参数化；复现时优先使用各RUN的`command.json`和`effective_config.json`，而非假定当前公共配置等于早期失败轮次。

## 原始13项验收矩阵

| ID | 功能 | 本轮结论 |
|---|---|---|
| T01 | BF16基础推理、开箱性能 | 启动/请求通路通过；输出质量不合格 |
| T02 | EP/TP并行 | EP8通路已运行；TP2失败 |
| T03 | CP功能和精度 | PCP2通路通过，6/6输出与PCP1不同；DCP2阻塞 |
| T04 | SP功能和精度 | 前置阻塞，未执行 |
| T05 | PP功能（低优先级） | PP2失败 |
| T06 | 图模式及精度 | 图捕获/回放通过；一致性部分不符、质量未过 |
| T07 | 异步调度及精度 | 运行与6请求greedy对照一致；基础质量未过 |
| T08 | routing replay返回专家 | 失败：双请求capture断言/HTTP500 |
| T09 | chunk_prefill | 分块通路通过；输出与非分块不同、质量未过 |
| T10 | sleep level-1 | 通过：5轮一致性和采样显存检查 |
| T11 | sleep level-2 | 失败：首轮恢复输出不一致，已复现 |
| T12 | HCCL/IPC权重更新 | 两条完整更新路径均未通过（OOM） |
| T13 | M级序列功能和精度 | 容量通过；检索 0/3，检索/语义未通过 |

“通路通过”不等于输出质量通过；“前置阻塞”不计通过。未对完整未裁剪模型作支持声明。

## T01 BF16基础推理、开箱性能

**结论：启动/请求通路通过；输出质量不合格。**

TP1/DP8/EP8，无算子异常，4单请求+2并发完成；中文/英文/代码/算术均存在重复退化，17+25未正常回答42。不得以HTTP200认定完整验收通过。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F01_basic.json`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F01_chunk_supported.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F01_basic_20261113T093344`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `validation.log`（均位于上述RUN）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F01_chunk_supported_20261113T094347`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `comparison.json`, `validation.log`（均位于上述RUN）。

## T02 EP/TP并行

**结论：EP8通路已运行；TP2失败。**

F02_ep与F01配置等价，复用基线真实EP8 rank/请求证据。TP2/DP4权重加载成功，预热时mixed_quant_sparse_flash_mla只接受N1=64，而TP2为32；不是原MXFP scale reshape问题。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F01_basic.json`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F03_tp2.json`
- 等价EP配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F02_ep.json`（复用F01实际运行）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F01_basic_20261113T093344`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `validation.log`（均位于上述RUN）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F03_tp2_20261113T095042`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `validation.log`（均位于上述RUN）。

## T03 CP功能和精度

**结论：PCP2通路通过，6/6输出与PCP1不同；DCP2阻塞。**

32768输入，PCP1与PCP2输入IDs完全相同；单请求及双请求都完成，无OOM。PCP2存在整数ArgSort转AiCPU警告。输出仍退化，不能给出精度通过结论。所规划DCP2依赖TP2，未执行。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F04_pcp1_control.json`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F04_pcp2.json`
- DCP阻塞配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F04_dcp2.json`。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F04_pcp1_control_20261113T104608`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `validation.log`（均位于上述RUN）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F04_pcp2_20261113T105359`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `comparison.json`, `validation.log`（均位于上述RUN）。

## T04 SP功能和精度

**结论：前置阻塞，未执行。**

当前SP要求TP>1且MoE启用EP；本模型A5注意力路径固定N1=64，TP2实测失败。未将SP开关在TP1下无效开启当作通过。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F05_sp.json`
- 前置实测证据：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F03_tp2_20261113T095042/server.log`。

## T05 PP功能（低优先级）

**结论：PP2失败。**

关闭Engram后实际尝试PP2/DP4/TP1。预热中PP1的hidden_states为None，调用unsqueeze失败；PP0另报PPMissingLayer缺少layer_idx。不是用完整Engram模型PP=1的限制替代实验。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F06_pp2.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F06_pp2_20261113T115905`
  关键证据：`command.json`, `effective_config.json`, `server.log`（均位于上述RUN）。
- 原始completion.json的YIELDED_TO_OTHER_JOB是退出竞态误报：前一采样PID均属本实验，PP异常后退出；按模型堆栈判为失败。未确认有外部作业竞争。监测器已修正，保留原始记录不覆盖。

## T06 图模式及精度

**结论：图捕获/回放通过；一致性部分不符、质量未过。**

FULL_DECODE_ONLY实际捕获，8个rank均记录ACL graph replay active。6请求完成；对eager单请求4/4一致，并发1/2一致。仍重复退化。仅覆盖decode图，不宣称prefill图通过。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F07_graph.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F07_graph_20261113T102953`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `comparison.json`, `validation.log`（均位于上述RUN）。

## T07 异步调度及精度

**结论：运行与6请求greedy对照一致；基础质量未过。**

日志确认async scheduling实际启用；4单请求+2并发均完成，输出token与同步基线6/6一致。此为有限样例一致性，不等同完整模型精度。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F08_async.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F08_async_20261113T103545`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `comparison.json`, `validation.log`（均位于上述RUN）。

## T08 routing replay返回专家

**结论：失败：双请求capture断言/HTTP500。**

AuxOutput要求prefix caching，修正后进入运行。单请求HTTP200，双请求报unexpected topk_ids batch dim 2（expected 9,1,16,or1）。该轮客户端未提前落盘单请求响应，不能宣称专家内容校验通过；之后已改为逐请求保存。只验返回专家，不包含训练侧强制路由replay。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_routing.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F09_routing.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F09_routing_20261113T095702`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `validation.log`（均位于上述RUN）。

## T09 chunk_prefill

**结论：分块通路通过；输出与非分块不同、质量未过。**

16384输入，2K调度预算，prefix cache关闭；4单请求+2并发完成。与非分块相同输入的6个输出均不同。有效预算和scheduler约束支持实际分块判断；iteration_tokens指标按完成prefill聚合，不能作为精确8次分块计数。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F10_control.json`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F10_chunk.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F10_control_20261113T110120`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `validation.log`（均位于上述RUN）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F10_chunk_20261113T110941`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `inference.json`, `comparison.json`, `validation.log`（均位于上述RUN）。

## T10 sleep level-1

**结论：通过：5轮一致性和采样显存检查。**

5/5轮sleep/wake/reload后temperature=0输出token与同配置初始基准完全一致。8卡HBM最终相对初始最大增长80MiB，未见明显持续增长。API仅返回首引擎allocator统计，其allocated不变；其余卡依据主机HBM采样。此通过不代表基线语义质量合格。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_sleep.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F11_sleep1.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F11_sleep1_20261113T100448`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `sleep.json`, `hbm_periods.json`, `validation.log`（均位于上述RUN）。

## T11 sleep level-2

**结论：失败：首轮恢复输出不一致，已复现。**

严格按weights wake→reload_weights→kv_cache wake执行。两次首轮均Token mismatch；补证复现从第1个输出token即不同，原始响应与comparison.json已保存。未完成5轮，不计显存稳定性通过。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_sleep.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F12_sleep2.json`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F12_sleep2_repro.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F12_sleep2_20261113T101611`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `validation.log`（均位于上述RUN）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F12_sleep2_repro_20261113T102258`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `comparison.json`, `sleep_baseline.json`, `sleep_cycle_1.json`, `validation.log`（均位于上述RUN）。

## T12 HCCL/IPC权重更新

**结论：两条完整更新路径均未通过（OOM）。**

HCCL TP1/DP4推理+第5卡sender，0.75和0.55更新时OOM；0.50无KV容量。IPC DP8/EP8+8个同卡sender，packed2GiB、0.45、显式npu_ipc后仍更新OOM。完整B未完成，A→B→A未完成；不能用部分传输或初始哈希认定更新成功。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_weight_update.py`
- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_ipc_multi.py`
- IPC启动器：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_ipc.py`；后端必须为`npu_ipc`。
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F13_hccl.json`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F13_ipc.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_hccl_20261113T112758`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `validation.log`（均位于上述RUN）。
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_ipc_20261113T115009`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `validation.log`, `ipc_launcher_command.json`, `ipc_baseline.json`（均位于上述RUN）。

| 路径 | 配置/尝试 | 结果 |
|---|---|---|
| HCCL | DP4/EP4，0.75，GPU4 sender | 约layer10附近更新OOM |
| HCCL | 同布局，0.50 | 可用KV=-0.20GiB，启动失败 |
| HCCL | 同布局，0.55 | 约layer19附近更新OOM |
| IPC初次 | backend=ipc，torchrun standalone | 主机名解析失败，未传输 |
| IPC第二次 | 静态loopback，backend=ipc | 误入上游CUDA接收器 |
| NPU IPC最终 | DP8/EP8，8同卡sender，packed2GiB，0.45，backend=npu_ipc | 真实NPU更新路径OOM |

完整历史RUN：
- `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_hccl_20261113T111653`
- `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_hccl_20261113T112328`
- `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_hccl_20261113T112758`
- `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_ipc_20261113T113338`
- `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_ipc_20261113T114035`
- `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F13_ipc_20261113T115009`

## T13 M级序列功能和精度

**结论：容量通过；检索 0/3，检索/语义未通过。**

前置131072/262144/524288输入各输出32作为容量探针；最终严格1000000输入+1024输出。强制输出长度用于容量验证，不等同自然生成1K连贯回答或完整精度验收。

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_long_context.py`
- 配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F14_long.json`
- RUN：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F14_long_20261113T120549`
  关键证据：`command.json`, `effective_config.json`, `server.log`, `validation.log`, `long_progress.json`, `long_1000000.json`（均位于上述RUN）。

| 输入tokens | 输出tokens | 耗时（秒） | 结果 |
|---:|---:|---:|---|
| 131072 | 32 | 44.18 | 请求完成；以响应usage确认长度 |
| 262144 | 32 | 35.90 | 请求完成；以响应usage确认长度 |
| 524288 | 32 | 74.71 | 请求完成；以响应usage确认长度 |
| 1000000 | 1024 | 305.99 | 请求完成；以响应usage确认长度 |

最终检索命中：0/3，命中项：`[]`。探针位于输入约5%、50%、95%处。完整输出及token IDs见`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F14_long_20261113T120549/long_1000000.json`。

输出开头（原样摘录）：

```text
。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料。 这是一段用于上下文功能验证的普通材料
```

## 开箱性能观测

基线512输入/128输出、并发2、10请求：成功10、失败0；输出吞吐10.85 token/s，平均TTFT 1992.75ms，平均TPOT 169.91ms。该benchmark未显式设置temperature，采用服务默认值；不与temperature=0功能请求混同。来源：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/node26_F01_basic_20261113T093344/benchmark.log`及同RUN结果JSON。

图模式短请求耗时约4.01–4.09秒，双请求约4.40–4.44秒；eager短请求预热后约24秒。仅为相同测试请求级观察，不替代标准化吞吐验收。没有业务性能门槛，因此不写“性能达标”。

## 文档/源码证据及适用范围

1. [Engram并行限制（当前提交）](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/platform.py#L1869-L1912)：开启Engram要求PP=DCP=1。本次关闭Engram，不用该限制豁免PP2；PP2已有独立实测失败。
2. [SP前置条件](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/user_guide/feature_guide/sequence_parallelism.md)：TP>1、MoE EP和FlashComm1。
3. [V4.1 PCP实现](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/attention/context_parallel/dsa_v41_cp.py#L137-L145)：PCP路径要求DCP1，不能把它扩大为所有独立DCP的支持结论。
4. [A5原生MXFP TP问题](https://github.com/vllm-project/vllm-ascend/issues/20398)：只证明对应量化路径/版本。本轮BF16 TP2限制依据实际N1错误和[混合注意力调用](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/models/deepseek_v41/mixed_quant_attention.py#L285)。
5. [PP模型forward](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/models/deepseek_v41/model.py#L1592)：结合本轮PP堆栈定位，不能只凭SupportsPP声明判通过。
6. [路由捕获检查](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/patch/worker/patch_routed_experts_capture.py#L256)：与实际batch维度断言对应。
7. [权重传输注册表](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/distributed/weight_transfer/__init__.py)：当前显式注册npu_ipc。[RL文档](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/user_guide/feature_guide/rl.md)中的ipc别名描述与本轮运行不符，应以注册表和实测为准。
8. [Sleep语义](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/user_guide/feature_guide/sleep_mode.md)：level-2须重载权重；本轮已执行该顺序，仍出现不一致。

## 脚本修正与执行纪律

- 补齐block-size=128；对非chunk配置对齐batched-token预算与max_model_len。
- routing case按AuxOutput要求开启prefix caching。
- 每个推理请求和sleep基准/恢复响应提前落盘，避免后续失败丢失前序证据。
- IPC改用显式loopback静态rendezvous和npu_ipc；采用多rank同卡发送及有界打包缓冲，不加载完整训练模型到单卡。
- 权重更新失败后保持暂停，随后清理本轮进程，未带着半更新模型继续推理。
- 监测器以本轮环境标记和PID启动时间识别所有权；忽略已退出/僵尸PID，真实外部NPU进程仍会触发让出资源。PP退出竞态的修正检查见`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/evidence/guard_exit_race_check.txt`。
- 未修改vLLM/vLLM-Ascend源码或原checkpoint。最后的仓库状态及资源检查另存`evidence/final_state_20261113.txt`。

未额外执行F02_ep4独立双请求、F08长短混合扩展、C01组合回归；前两者为扩展场景，基础对应项已有实测，组合回归因单项质量/精度未达标未进入。未将这些扩展项计为通过。

完整过程记录：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/EXECUTION_RECORD_20261113.md`。机器可读汇总：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results/EXECUTION_SUMMARY_20261113.json`。各RUN保留早期失败、修正后命令、配置、日志及终止状态；不要仅依靠supervisor的进程退出码判定功能通过。

# DSV4.1 HCCL 与 NPU IPC 全量权重更新验证记录

## 2026-11-18 最新实机结果

IPC 引用计数补丁已完成单卡旧/新实现各 20 轮及 8 卡全量 A→B→A。修复后单卡 allocated 最大增长为 0，两轮全量更新后 8 个发送端在显式清理前后均为 0，退出共享 NPU 张量告警为 0。HCCL 全量回环、有效缓存失效计数和同缓存状态下的数值对照均通过。

当前遗留为本 Top32 配置下冷缓存和缓存命中请求生成输出之间的差异。8 张 NPU 已释放。[最新报告](runs/retest_20261118_01/REPORT.md) [审计](runs/retest_20261118_01/acceptance_audit.json)

以下正文保留 2026-11-17 的修复与验证过程。

验证日期为 2026-11-17。服务器为 `177.42.22.1`，容器为 `c00954197_dsv41_cann930`，实验根目录为 `/mnt/share/c00954197/Workspace/dsv41/basic/weight_update`。

## 当前交付状态

接收端全量更新 OOM 已定位并修复，HCCL 和 NPU IPC 均完成真实 A→B→A 全量更新。每个发送端、每个接收副本、每轮均记录 6248 个张量和 135927315272 字节，约 126.59 GiB。B 输出与独立冷加载 B 一致，恢复后的 A 输出与初始 A 一致。两种接口逐副本的文本、token IDs、token 字符串和逐 token logprob 均严格相等。

后续复核发现 IPC 发送缓冲滞留。对应引用计数补丁完成 CPU 验证，实际 NPU 回收效果等待资源安排。任务状态为 `IN_PROGRESS`，最新代码的综合验收结果为 `PARTIAL`。报告中的 NPU 结果对应下列已完成运行，最新 IPC 生命周期补丁仍在等待设备复验。

| 验证项 | 状态 | 证据范围 |
|---|---|---|
| 接收端加载配额与 OOM 修复 | NPU 通过 | 两个后端完成全量更新，接收端轮后延迟引用为 0 |
| HCCL 全量 A→B→A | NPU 通过 | TP1/DP4/EP4，独立发送卡，B 提交与 level-2 A 恢复 |
| NPU IPC 全量 A→B→A | NPU 通过 | TP1/DP8/EP8，每卡同置发送进程，使用后述旧 packed 源码 |
| 版本传播 | NPU 通过 | `/weight_info` 分别返回 `B` 和 `A-restored` |
| 双接口数值与完整输出头哈希 | NPU 通过 | 每个 DP 副本分别对照同拓扑基准 |
| IPC 多轮缓冲回收补丁 | CPU 通过，NPU 待验证 | 新增 5 项 CPU 回归及仓库 106 项测试通过 |
| 前缀缓存命中后的更新失效 | 待覆盖 | 本次所有请求 `cached_tokens=0` |
| 图模式、异步调度及 MRV1 | 待覆盖 | 本次使用 MRV2、eager、同步调度 |

完整传输审计、离线交叉检查和运行日志已随交付保存。实验结束时的资源记录为 `all_npus_free=true`，该记录对应本轮收尾时刻。[离线核对结果](evidence_audit.json) [设备释放记录](runs/transfer_20261117_03/resource_release.json)

## 模型与实验选择

沿用 40 层、32 路由专家、Top-6、BF16、关闭 Engram 的 DSV4.1 缩减 checkpoint。A 位于 `/mnt/share/c00954197/Workspace/dsv41/basic/model`。B 位于实验根目录下的 `model_b`，对完整 checkpoint 建立文件视图，仅复制并修改包含 `head.weight` 的分片。

B 的差异由本次实验设计决定，对 `head.weight` 的词表行循环移动 1 位。构建过程按 64 MiB CPU 文件块复制。两轮均重放全部 checkpoint 张量。B 的行置换会明显改变生成文本，冷加载 B 提供对应数值对照。

| 项目 | 数值 |
|---|---|
| 每轮张量数 | 6248 |
| 每轮张量字节 | 135927315272 |
| 最大单张量 | `head.weight`，`[129280, 5120]` BF16 |
| 最大单张量字节 | 1323827200，1262.5 MiB |
| packed buffer | 1280 MiB，单缓冲 |
| 显式 KV cache | 每卡 2 GiB |
| 上下文与批处理 token 上限 | 2048 |
| 并发序列上限 | 2 |
| HCCL_BUFFSIZE | 256 MiB |
| 生成参数 | temperature 0，seed 42，max_tokens 8 |

A 的完整 head SHA256 为 `12f45f83b1a1cd9f440f48f2d5a8b7b38b0a71fc9b35b7f9b7de7c4d81673508`。B 为 `d12652ddde072bfd70380791c814d63be6e60c8090885009720f1ef8a1652d72`。逐副本计算覆盖完整输出头张量。[变体清单](variant_manifest.json)

## 定位过程与正式修复

两次失败运行均在约第 20 层附近 OOM。每个 EP4 MoE 层收到 283115520 个元素，加载完成配额却为 283115552，始终相差 32。层级加载完成条件无法满足，接收的 NPU 张量持续保留在延迟加载列表。失败时延迟张量达到 47587000320 字节，约 44.32 GiB，运行时已分配约 85.63 GiB。显式 KV 已设为 2 GiB，仍复现该问题。[失败现场](diagnostics/oom_reproduction.json)

早期把这 32 个元素归因于 `log2phy`，曾临时跳过 `log2phy`、`moe_load` 和 `load_counter`。CPU 模拟通过后，V2 实机仍出现相同 OOM。随后检查实际模型字段和通信初始化路径，确认 V2 未初始化这些 EPLB 字段。该临时修改已撤回，探索补丁保存在 `diagnostics/provisional`。

实际字段为 Ascend MoE runner 注册的 `expert_ids_per_ep_rank`，它包含全局 32 项运行时通信映射。正式修改在 `/vllm-workspace/vllm/vllm/model_executor/model_loader/reload/meta.py` 的 `SKIP_LOAD_TENSORS` 加入该字段，同步影响 `SKIP_TENSORS`。加载配额恢复为 checkpoint 权重元素数，每层完成后能够释放暂存输入，并保留运行时映射和引用地址。[正式补丁](patches/expert_rank_buffer_loading.patch)

针对实际字段的 CPU 回归在修改前 3 项中 2 项失败，修改后 3 项通过。覆盖配额计算、逐层暂存释放、A/B/A 后运行时字段及别名地址保持。已有 Sleep level-2 的 RMSNorm buffer 和 attention sink 修复继续作为前置条件，两组共 6 项 CPU 回归通过，对应补丁随包保存。[接收端回归日志](patches/expert_rank_buffer_after.log)

## 实机 A→B→A 结果

HCCL 使用 NPU0–3 运行 TP1/DP4/EP4，NPU7 为独立发送端。NPU IPC 使用 NPU0–7 运行 TP1/DP8/EP8，另有每卡一个同置发送进程，发送进程间以 CPU Gloo 协调。

B 阶段依次执行 `pause(clear_cache=true)`、真实 start/update/finish、提交 `B`、resume、生成验证。A 恢复阶段执行 pause、sleep level-2、仅 wake weights、全量传输并提交 `A-restored`、wake KV、resume、生成验证。

| 后端 | 阶段 | 发送端数量 | 接收副本数量 | 每端张量数与字节 | 更新阶段接收卡采样峰值 |
|---|---|---|---|---|---|
| HCCL | B | 1 | 4 | 6248 / 135927315272 | 61552 MiB |
| HCCL | A-restored | 1 | 4 | 6248 / 135927315272 | 57112 MiB |
| NPU IPC | B | 8 | 8 | 6248 / 135927315272 | 44611 MiB |
| NPU IPC | A-restored | 8 | 8 | 6248 / 135927315272 | 43230 MiB |

HCCL 接收端最大采样延迟张量量为 2.043457 GiB，IPC 为 2.087402 GiB，两轮结束均为 0。HCCL 独立发送卡更新阶段物理 HBM 采样峰值为 13248 MiB。HCCL B/A 发送耗时约 90.16/87.30 秒，IPC 各发送进程约 159.58/231.96 秒。耗时覆盖发送过程及协议同步，HBM 由 `npu-smi` 约每秒采样。[HCCL 逐端审计](runs/transfer_20261117_03/hccl_producer/full_transfer_audit.json) [IPC 逐端审计](runs/transfer_20261117_03/npu_ipc_producer/full_transfer_audit.json)

每个阶段均保存 `/v1/completions` 与 `/v1/chat/completions` 的原始请求和响应。HCCL 每阶段核对 8 份输出，IPC 每阶段核对 16 份输出。离线再次读取原始结果确认 B 与冷加载 B 严格相等，A-restored 与初始 A 严格相等，各阶段的重复请求也严格一致。HCCL 复用了第二次运行的冷加载 B 对照，第一和第二次冷加载 B 输出完全一致。正式配额修复仅改变 reload 元数据路径。

缩减模型存在生成内容质量限制，HCCL 初始 DP0 completion 为 `1.2.3.4.`，恢复后相同。不同拓扑、部分副本的 completion 也有差异。本次按同拓扑、同副本验证更新前后的一致性。B 的词表行置换会产生混杂文本，原始输出完整保留。

## IPC 缓冲生命周期的后续问题

完成回环的旧 packed 实现中，每个 IPC 发送进程在 B 轮后分配量为 1.25 GiB，A 轮后为 2.5 GiB，进程退出共出现 8 条共享 NPU 张量尚未释放的告警。数据与数值回环通过，连续更新的内存稳定性仍有待解决。

源码复核发现发送端对复用缓冲只调用一次 `reduce_tensor`，接收端每个 chunk 都执行重建，重复使用了包含引用计数的 rebuild 参数。新增修改维持单缓冲复用，每个 chunk 在载荷同步后导出新的 rebuild 参数，空输入迭代器不产生导出。[生命周期补丁](patches/ipc_chunk_reference_counters.patch)

新增测试直接 AST 提取部署源码中的函数，mock NPU 分配、stream 和 reducer。修改前 5 项中 4 项失败，修改后 5 项通过，覆盖逐块引用计数、导出前载荷就绪、同一缓冲复用、空迭代器和零元素张量。仓库原有 packed tensor CPU 测试 106 项通过。实际 NPU 的引用释放、残留显存和退出告警消除，等待单卡多轮及完整 IPC 回环复验。[5 项回归日志](patches/ipc_chunk_leases_after.log) [106 项仓库测试日志](patches/packed_tensor_repository_cpu_tests.log)

## 源码与环境边界

| 项目 | 版本或来源 |
|---|---|
| CANN | 9.3.0，V100R001C25B114 |
| Python | `/usr/local/python3.12.16/bin/python3`，3.12.16 |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post7 |
| vLLM | 0.32.0+empty，源码 `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | 0.19.1rc2.dev3401+g28e9e97ba，源码 `28e9e97ba254b076a88275e8a275556cb9a11664` |
| Transformers | 5.19.0 |

实机全量回环使用的 `meta.py` SHA256 为 `aa5ea0e0503ee34c763403469de0432eb4315868bd91792071445c1bbfe63ccd`，`packed_tensor.py` 为 `2bfcde4bdf2fd2cb3638af8ad49cce0736a3c2d01081d147e22cf9d2081780aa`。后续 CPU 修复后的 packed 文件为 `1f8e4edc8f691bbb4cdbfbe2eef0c37c8163ad8090471eeb3b661e801c8f0b1e`。最新 packed 文件的实机验收仍待完成。

环境快照采集于全量运行之后，仅查询版本元数据、Git 状态和文件哈希。容器保留本次两项修复及继承的 Sleep2 两项修改，源码 diff、原文件和改后文件随包保存。[环境快照](environment/environment_after_run.json) [当前源码差异](environment/source_snapshot.txt)

## HBM 规避方法与适用条件

- 首先修复加载完成配额，恢复逐层释放暂存输入。该措施已通过两种后端的全量实机验证。
- checkpoint-native 数据源逐张量读取，发送端使用单 packed buffer。本模型最大张量为 1262.5 MiB，当前分包器要求缓冲能容纳整个单张量，因此本次采用 1280 MiB。
- 使用显式 2 GiB KV、2048 token 预算、2 个序列和 256 MiB HCCL 缓冲，限制并发与通信预留。完整训练模型副本的发送方式需要另行计算容量。
- A 恢复阶段先 sleep2，再按 weights、KV 的顺序分阶段唤醒。此流程已完成实机验证。
- 当前 IPC 发送缓冲存在逐轮滞留记录，最新引用计数补丁的 NPU 回收效果等待验证，后续排期优先检查这一项。

## 验证方法与覆盖范围

验证使用真实 HTTP 权重传输、B 版本提交、A level-2 恢复、权重版本查询和数值对照。CheckpointWeightSource 按 checkpoint 原生名称逐张量读取，发送端与接收端分别记录完整计数和字节。完整输出头哈希与双接口输出按 DP 副本核对。

本模型 prompt token 数为 113，小于 128 的 block size，重复请求的 cached_tokens 始终为 0。有效前缀命中后的更新失效验证需要更长的请求。当前实机覆盖 MRV2、eager 和同步调度，其他 runner 与调度模式保留待验证。

## 已执行命令与结果位置

最终运行通过以下宿主机命令启动，退出与子进程命令按阶段保存为 JSON 和 `.exit` 文件。

```bash
python3 -u /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/scripts/run_host.py \
  --tag transfer_20261117_03 --backend both \
  --hccl-oracle /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/runs/transfer_20261117_02/hccl_oracle/oracle/oracle.json
```

运行 01、02 使用各自新 tag 和 `--backend both`，均在 HCCL B 更新阶段 OOM，清理后的发送端退出码记录为 143。运行 03 的 HCCL、IPC 发送端及 IPC 冷加载 oracle 退出码均为 0。容器启动参数以各阶段 `actual_server_command.sh` 为准，发送与 oracle 命令分别保存在 `producer_command.json` 和 `oracle_command.json`。CPU 回归完整输出保存在 `patches`，其中历史仓库 106 项测试日志未保存独立的启动命令清单。

交付包解压后可在本地 CPU 运行以下脚本，重新核对传输计数、完整 head 哈希、原始数值、版本和显存记录。

```bash
python3 scripts/audit_saved_evidence.py
```

等待资源后的单卡、多卡复验步骤见 `pending_validation.md`。交付包含脚本、补丁、完整请求响应、发送端与接收端记录、按阶段 HBM CSV、失败运行证据及源码快照。模型资产继续保留在服务器实验目录。

# DSV41 模型开箱交付件

结果和目录入口见[开箱结果概览](README.md)。本报告依据截至2026-11-20的实测归档。验证环境为 **CANN930（CANN 9.3.0）**。完整模型为 DeepSeek-V4.1-Flash-1017 BF16、40层、384路由专家、Engram和视觉模块开启；RL专项使用40层、32路由专家、Engram关闭的缩减模型。两种模型的结论分别记录。

## 模型结构分析

[模型结构图](model/architecture/README.md)提供执行总图和8张细节图，覆盖40层主干、视觉、Attention、MoE、Engram、mHC、逐层共享和独立MTP。[模型资产说明](model/模型结构分析.md)列出配置、权重索引和KV格式。MoE每层384个路由专家，每 token 选 Top-6，另有1个共享专家。结构依据与静态核对见[结构审计](model/architecture/audit.md)。

本次BF16资产由FP8/FP4权重反量化获得。[反量化脚本](model/conversion/convert_deepseek_v41_flash_bf16.py)、[执行脚本](model/conversion/run_deepseek_v41_flash_bf16.sh)及[完成日志](model/conversion/dsv41-flash-bf16-convert.log)已归档。完整模型输出525个分片、48496个索引张量；转换已完成。[独立FP4/FP8抽样核验](model/dequant_validation_20261120/REPORT.md)覆盖34个张量、4624个标量，BF16位模式全部一致。

| 模型 | 当前服务器路径 | 验证用途 |
|---|---|---|
| 完整384专家BF16 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16` | 基础推理、并行组合、性能、GPQA精度 |
| Slim-E32 BF16 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32` | Sleep、专家返回、checkpoint及权重更新 |

Slim-E32为无软链接的实体副本，保留全部6248个有效张量、503个分片，约135.9 GB（126.6 GiB）；未索引的Engram残留已剔除。完整权重与tokenizer资产保存在服务器。

## 模型开箱打通

### 验证环境与源码版本

| 项目 | 配置 |
|---|---|
| 服务器 | RL专项177.42.22.1；完整模型177.42.23.1/24.1 |
| 硬件 | 每节点8×Ascend950DT，单卡96 GiB HBM；完整模型双节点16卡 |
| 系统 | openEuler，Linux 6.6.0-145.3.35.166.oe2403sp3，aarch64 |
| 容器 | `c00954197_dsv41_cann930` |
| 镜像ID | `sha256:3614b7afe7000897bf6f1c60a415f2547feb0e033bae0c9bf76c68a76ecf944f` |
| CANN | 9.3.0，V100R001C25B114 |
| Python / Torch / TorchNPU | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 |
| Triton-Ascend / Transformers | 3.6.0.post2 / 5.19.0 |
| vLLM | 0.32.0+empty；`5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | 0.19.1rc2.dev3401+g28e9e97ba；`28e9e97ba254b076a88275e8a275556cb9a11664` |
| 网络 | Docker host网络/IPC；Gloo `enp39s0f0`，HCCL `enp35s0f2`；CCU_SCHED |

实际代码在上述提交上包含已归档补丁；Sleep、权重更新及Gloo CPU超时透传的补丁与文件指纹见[环境记录](environment/cann930_environment_20261117.md)。路由补丁已完成本轮NPU验收，见[路由报告](reduced/cann930_20261120/routing_replay/REPORT.md)。两个基础提交已核验在公开源码仓库可达。本轮修复以补丁和实际源码指纹交付；尚未提交上游修复PR。

### 服务端启动与请求

完整模型使用TP1/PCP8/DP2/EP16，MRV2、FULL_DECODE_ONLY、异步调度、chunked prefill和prefix caching。Engram配置为CPU offload、DP shared memory。当前精度服务上下文73728、batched tokens 2048、每DP最大序列32，显存利用率参数0.96。

在两端宿主机进入容器：

```bash
docker -H unix:///run/c00954197-cann930/docker.sock exec -it c00954197_dsv41_cann930 bash -l
```

分别在主、子节点容器内执行已交付的脚本：

```bash
# 177.42.23.1，主节点
bash /mnt/share/c00954197/Workspace/dsv41/precision_cann930/run_master.sh
# 177.42.24.1，子节点
bash /mnt/share/c00954197/Workspace/dsv41/precision_cann930/run_worker.sh
```

[主节点脚本](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点脚本](full/cann930_20261119_precision/scripts/run_worker.sh)及[实际配置审计](full/cann930_20261119_precision/run/config_audit.json)保存完整参数。性能负载使用40960上下文，见[性能服务脚本](full/cann930_20261117/scripts/)。模型权重为BF16；实际packed KV采用混合格式，具体布局见模型分析。

以下命令在归档仓库根目录执行，使用[请求文件](full/cann930_basic/examples/gpqa_question_0_reconstructed_request.json)发送单curl：

```bash
curl --noproxy '*' --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @full/cann930_basic/examples/gpqa_question_0_reconstructed_request.json
```

[可读输出](full/cann930_basic/examples/gpqa_question_0_final_content.txt)摘录：

> Among the options, \(10^{-4}\,\text{eV}\) is large enough to clearly resolve them.
>
> Answer: D

该输出来自CANN930实测GPQA第二次完整评测的第0题，完整[预测记录](full/cann930_basic/examples/gpqa_question_0_saved_prediction.json)已保存。curl正文按该题提示词与实际生成参数重建，本次文档整理没有重新调用NPU；缩减模型另有[实际单请求原文及响应](reduced/cann930_20261120/routing_replay/examples/single_response.json)，可用其[请求原文](reduced/cann930_20261120/routing_replay/examples/single_request.json)进行curl复现；两种模型的证据分别记录。

## 基本功能验证

各功能统一交付服务端脚本、请求/校验脚本、请求与输出、日志与统计四类内容。相同服务覆盖的功能复用同一组真实文件；通过结论限于受测配置，不声明未测参数组合数值等价。TP>1、DCP、SP、PP按当前DSV4.1支持范围排除。

| 功能 | CANN930结论 | 范围 |
|---|---|---|
| TP/EP及DP | TP1/EP16/DP2组合通过运行验证 | 当前模型不支持TP>1，本轮排除 |
| CP | PCP8组合通过运行验证 | DCP依赖TP>1，本轮排除 |
| SP | 本轮不适用 | SP依赖TP>1，当前模型未支持 |
| PP | 本轮不适用 | 当前DSV4.1的Engram实现限制PP |
| 图模式 | FULL_DECODE_ONLY组合通过运行验证 | 受测组合通过 |
| 异步调度 | async组合通过运行验证 | 受测组合通过 |
| chunk_prefill | 开启组合通过运行验证 | 受测组合通过 |

### 并行切分TP/EP

**验证配置：** 完整模型；TP1、DP2、PCP8、EP16。

**方法与通过条件：** 启用专家并行，完成真实生成；核对请求成功数与完整输出。

**结果：** 三场景各4轮64/64成功；四次完整GPQA共792/792请求成功。TP1通路与EP/DP组合已验证。当前DSV4.1不支持TP>1，该项排除本轮验收。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [主节点](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点](full/cann930_20261119_precision/scripts/run_worker.sh) |
| 请求／校验脚本 | [单请求复现脚本](full/cann930_basic/run_single_curl.sh)、[GPQA客户端](full/cann930_20261119_precision/scripts/run_gpqa.sh) |
| 请求与输出 | [已保存单题输出](full/cann930_basic/examples/gpqa_question_0_saved_prediction.json)、[性能请求统计](full/cann930_20261117/performance_results/) |
| 日志与统计 | [主节点](full/cann930_20261119_precision/run/node23/launcher.log)、[子节点](full/cann930_20261119_precision/run/node24/launcher.log) |

### CP 功能

**验证配置：** 完整模型；`--prefill-context-parallel-size 8`。

**方法与通过条件：** PCP8下完成真实推理；DCP按当前TP支持限制排除。

**结果：** PCP8组合运行通过。DCP依赖TP>1，排除本轮验收；PCP其他档位不由本次组合结果外推。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [主节点](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点](full/cann930_20261119_precision/scripts/run_worker.sh) |
| 请求／校验脚本 | [单请求复现脚本](full/cann930_basic/run_single_curl.sh)、[GPQA客户端](full/cann930_20261119_precision/scripts/run_gpqa.sh) |
| 请求与输出 | [已保存单题输出](full/cann930_basic/examples/gpqa_question_0_saved_prediction.json)、[性能请求统计](full/cann930_20261117/performance_results/) |
| 日志与统计 | [主节点](full/cann930_20261119_precision/run/node23/launcher.log)、[子节点](full/cann930_20261119_precision/run/node24/launcher.log) |

### SP 功能

**验证配置：** 当前DSV4.1不支持TP>1，SP依赖TP>1。

**方法与通过条件：** 按当前模型支持范围排除，本轮不执行SP专项验收。

**结果：** 本轮不适用，不列为待完成或待修复项。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | 不适用（当前模型支持限制） |
| 请求／校验脚本 | 不适用（当前模型支持限制） |
| 请求与输出 | 不适用（当前模型支持限制） |
| 日志与统计 | 不适用（当前模型支持限制） |

### PP 功能

**验证配置：** 当前DSV4.1的Engram实现限制PP，基线使用PP1。

**方法与通过条件：** 按当前模型支持范围排除，本轮不执行PP专项验收。

**结果：** 本轮不适用，不列为待完成或待修复项。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | 不适用（当前模型支持限制） |
| 请求／校验脚本 | 不适用（当前模型支持限制） |
| 请求与输出 | 不适用（当前模型支持限制） |
| 日志与统计 | 不适用（当前模型支持限制） |

### 图模式功能

**验证配置：** 完整模型；`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`。

**方法与通过条件：** 在该模式下完成真实生成并检查运行日志。

**结果：** 与PCP8、async及chunked prefill组合运行通过；本轮严格一致性额外回归见RL章节。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [主节点](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点](full/cann930_20261119_precision/scripts/run_worker.sh) |
| 请求／校验脚本 | [单请求复现脚本](full/cann930_basic/run_single_curl.sh)、[GPQA客户端](full/cann930_20261119_precision/scripts/run_gpqa.sh) |
| 请求与输出 | [已保存单题输出](full/cann930_basic/examples/gpqa_question_0_saved_prediction.json)、[性能请求统计](full/cann930_20261117/performance_results/) |
| 日志与统计 | [主节点](full/cann930_20261119_precision/run/node23/launcher.log)、[子节点](full/cann930_20261119_precision/run/node24/launcher.log) |

### 异步调度功能

**验证配置：** 完整模型；`--async-scheduling`、CCU_SCHED。

**方法与通过条件：** 异步服务完成真实生成，核对请求与运行期日志。

**结果：** 三场景性能与四次GPQA完成；11月19日服务运行及清理日志ERROR为0。本轮缩减模型async与chunked prefill已完成1M实测。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [主节点](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点](full/cann930_20261119_precision/scripts/run_worker.sh) |
| 请求／校验脚本 | [单请求复现脚本](full/cann930_basic/run_single_curl.sh)、[GPQA客户端](full/cann930_20261119_precision/scripts/run_gpqa.sh) |
| 请求与输出 | [已保存单题输出](full/cann930_basic/examples/gpqa_question_0_saved_prediction.json)、[性能请求统计](full/cann930_20261117/performance_results/) |
| 日志与统计 | [主节点](full/cann930_20261119_precision/run/node23/launcher.log)、[子节点](full/cann930_20261119_precision/run/node24/launcher.log) |

### chunk_prefill 功能

**验证配置：** 完整模型；`--enable-chunked-prefill`，batched tokens 2048。

**方法与通过条件：** 开启分块预填充完成真实生成，核对输出完整性。

**结果：** 组合运行通过；本轮缩减模型已完成1M分块输入实测。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [主节点](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点](full/cann930_20261119_precision/scripts/run_worker.sh) |
| 请求／校验脚本 | [单请求复现脚本](full/cann930_basic/run_single_curl.sh)、[GPQA客户端](full/cann930_20261119_precision/scripts/run_gpqa.sh) |
| 请求与输出 | [已保存单题输出](full/cann930_basic/examples/gpqa_question_0_saved_prediction.json)、[性能请求统计](full/cann930_20261117/performance_results/) |
| 日志与统计 | [主节点](full/cann930_20261119_precision/run/node23/launcher.log)、[子节点](full/cann930_20261119_precision/run/node24/launcher.log) |

## RL功能验证

以下已完成专项均为CANN930、177.42.22.1、40层32专家BF16模型，Engram关闭；不将该结果外推为完整384专家模型的专项验收。

### sleep mode 功能精度验证

**验证配置：** TP1/DP8/EP8，eager、同步；temperature=0，固定提示词，completions和chat两个接口。

**方法与通过条件：** 五轮sleep(level=2)→wake(weights)→checkpoint reload→wake(kv_cache)；逐副本单请求与首次加载基准比较，记录每卡显存。

**结果：** 五轮80/80请求的文本、token IDs及逐token logprob严格一致，最大logprob差0。最大单卡稳态显存较加载基线增加40 MiB，首轮恢复后继续增长最多1 MiB，满足本实验51.2 MiB容差。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [启动脚本](reduced/cann930_20261117/sleep_level2/scripts/start_server.sh)、[实际命令](reduced/cann930_20261117/sleep_level2/runs/fixed_complete_eager_20261117_01/actual_server_command.sh) |
| 请求／校验脚本 | [五轮验证客户端](reduced/cann930_20261117/sleep_level2/scripts/verify_sleep2.py) |
| 请求与输出 | [全部请求与响应](reduced/cann930_20261117/sleep_level2/runs/fixed_complete_eager_20261117_01/validation/)、[逐卡显存CSV](reduced/cann930_20261117/sleep_level2/memory_five_cycles.csv) |
| 日志与统计 | [服务日志](reduced/cann930_20261117/sleep_level2/runs/fixed_complete_eager_20261117_01/server.log)、[客户端日志](reduced/cann930_20261117/sleep_level2/runs/fixed_complete_eager_20261117_01/validation.log)、[报告](reduced/cann930_20261117/sleep_level2/REPORT.md) |

占用单位MiB，区间为8张卡最小值至最大值。

| 轮次 | 一致请求 | sleep前 | sleep后 | wakeup后 | 请求后稳态 |
|---|---|---|---|---|---|
| 1 | 16/16 | 59030–59395 | 8100–8436 | 58996–59341 | 59070–59375 |
| 2 | 16/16 | 59070–59375 | 8140–8436 | 59036–59341 | 59070–59375 |
| 3 | 16/16 | 59070–59375 | 8140–8436 | 59036–59341 | 59070–59375 |
| 4 | 16/16 | 59070–59375 | 8140–8437 | 59036–59341 | 59070–59375 |
| 5 | 16/16 | 59070–59375 | 8174–8465 | 59036–59341 | 59070–59375 |

修复将RMSNorm常量注册为非持久buffer，并将attention sink恢复接入layerwise weight_loader。[修复补丁](reduced/cann930_20261117/sleep_level2/patches/)及五轮完整输出随专项交付。缩减模型回复存在质量退化，本项验收目标为恢复一致性。

### routing replay 功能验证

**验证配置：** 32专家、MRV2、TP1/DP8/EP8；BI=1、prefix caching开启、SWA bounded replay关闭；Eager及FULL_DECODE_ONLY（TorchCompile=0/3）。

**方法与通过条件：** 单/双请求、逐DP与跨DP、每接口20组长短混合；核对完整专家shape/dtype/范围/唯一性，以及cold/hit/repeat、sleep/wake后的完整数组和生成结果。

**结果：** 通过。最终常规TorchCompile图模式153/153请求完成，真实同批双请求步骤879；两接口文本、token IDs、完整专家和逐token logprob严格一致，最大logprob差0。最终启动、推理及正常关闭日志ERROR/Traceback均为0，设备已释放。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [完整启动脚本](reduced/cann930_20261120/routing_replay/scripts/start_final_graph_20261120.sh) |
| 请求／校验脚本 | [完整客户端](reduced/cann930_20261120/routing_replay/scripts/validate_routing_matrix_open_20261120.py) |
| 请求与输出 | [完整专家JSON](reduced/cann930_20261120/routing_replay/examples/single_experts.json)、[全部原始响应与NPY](reduced/cann930_20261120/routing_replay/runs/routing_20261120_15_clean_shutdown/matrix/) |
| 日志与统计 | [服务日志](reduced/cann930_20261120/routing_replay/runs/routing_20261120_15_clean_shutdown/server.log.gz)、[结果与修复报告](reduced/cann930_20261120/routing_replay/REPORT.md)、[独立复核](reduced/cann930_20261120/routing_replay/evidence_audit.json) |

### 权重更新 功能验证

**验证配置：** HCCL：DP4/EP4推理NPU0–3、独立发送端NPU7；IPC：DP8/EP8、8个同置发送进程。

**方法与通过条件：** 完整A→B→A，逐端计数、版本及权重哈希核对；双接口文本/token/logprob对照对应权重、相同缓存状态的独立基准；检查连续更新显存回收。

**结果：** HCCL和NPU IPC全量回环通过。每轮每端6248张量、135927315272字节。接收端OOM与IPC缓冲滞留已修复；相同缓存状态下输出严格一致。此前冷热缓存差异已在本轮[严格路由配置](reduced/cann930_20261120/routing_replay/REPORT.md)中定位并验收通过。权重更新本身仍按当时记录的同缓存状态基准对照。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [服务端](reduced/cann930_20261117/weight_update/scripts/start_server.sh)、[发送端](reduced/cann930_20261117/weight_update/scripts/producer.py) |
| 请求／校验脚本 | [复验客户端](reduced/cann930_20261117/weight_update/scripts/run_retests.py)、[独立审计](reduced/cann930_20261117/weight_update/scripts/audit_retest.py) |
| 请求与输出 | [全量回环记录](reduced/cann930_20261117/weight_update/runs/transfer_20261117_03/)、[最新复验记录](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/) |
| 日志与统计 | [逐端与缓存审计](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/acceptance_audit.json)、[闭环报告](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/REPORT.md) |

| 检查项 | 结果 |
|---|---|
| checkpoint恢复 | 随Sleep五轮完成，逐副本双接口恢复一致 |
| 接收端OOM | 排除运行时`expert_ids_per_ep_rank`加载配额后，层级暂存能释放，轮后延迟引用归零 |
| IPC缓冲回收 | 修复后单卡20轮增长0；8卡两轮发送端allocated为0，清理后reserved为0，共享张量退出告警0 |
| 缓存失效 | 更新后首次cached_tokens=0，重复请求cached_tokens=256 |
| 冷/热缓存生成 | 同缓存状态更新对照通过；跨缓存状态一致性已由[本轮严格配置](reduced/cann930_20261120/routing_replay/REPORT.md)闭环 |

内存预算为显式KV 2 GiB、单packed buffer 1280 MiB、HCCL_BUFFSIZE 256 MiB，更新服务最大上下文/批处理预算2048。当前打包器要求单buffer容纳最大1262.5 MiB张量。后续可按设备容量限制KV、批处理与发送缓冲，使用level-2分阶段恢复并顺序传输；完整384专家模型需重新测算容量。[补丁](reduced/cann930_20261117/weight_update/patches/)与[最新状态](reduced/cann930_20261117/weight_update/validation_status.json)已归档。

### 1M长序列功能验证

**验证配置：** 32专家、Engram关闭、TP1/DP8/EP8；max-model-len=1048576，batched tokens=8192，max-num-seqs=1，KV16 GiB/卡；async、chunked prefill和FULL_DECODE_ONLY开启。

**方法与通过条件：** 真实tokenizer构造100万token输入，在约5%/50%/95%处放置三个检索键，末尾提问；保存完整请求和自然结束输出。

**结果：** 通过。实际输入1,000,000 token、HTTP200、148.02秒自然结束，输出43 token，三处键3/3命中且可读。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [启动脚本](reduced/cann930_20261120/long_context/scripts/start_server.sh) |
| 请求／校验脚本 | [构造与验证客户端](reduced/cann930_20261120/long_context/scripts/verify_long_context.py) |
| 请求与输出 | [完整输入请求](reduced/cann930_20261120/long_context/runs/long_context_20261120_12/validation/request.json.gz)、[原始响应](reduced/cann930_20261120/long_context/runs/long_context_20261120_12/validation/response.json)、[可读输出](reduced/cann930_20261120/long_context/runs/long_context_20261120_12/validation/output.txt) |
| 日志与统计 | [专项报告](reduced/cann930_20261120/long_context/REPORT.md)、[离线核验](reduced/cann930_20261120/long_context/evidence_audit.json)、[资源释放](reduced/cann930_20261120/long_context/runs/long_context_20261120_12/release_review.json) |

## 性能与精度验证

### 性能需求

验证使用完整384专家BF16模型、Engram开启、23/24双节点CANN930。主场景为4K文本＋一张图片→32K输出，同时覆盖基本功能要求的文本2K→6K、图文1K→2K。每轮64请求、并发64，每场景4轮。

图片bucket顺序为（高、宽、帧数），实际图片为**宽1080×高1920**。若交付要求严格为横向宽1920×高1080，需另行补测。实际输入包含模板和图像token；随机图片仅用于负载测量。

| 场景 | 轮数 | 每轮成功/失败 | 每轮输出token | 平均输出吞吐token/s |
|---|---|---|---|---|
| 图文4K→32K | 4 | 64/0 | 2097152 | 2230.531 |
| 文本2K→6K | 4 | 64/0 | 393216 | 2212.914 |
| 图文1K→2K | 4 | 64/0 | 131072 | 1844.228 |

12轮完整输出核对通过。服务开启prefix caching且轮次间复用缓存，结果包含缓存命中场景，不作为冷启动吞吐。**0.8×硬件性能目标尚未验收**；本环境主场景的视觉/prefill/decode及跨rank分析仍需补齐。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [主节点](full/cann930_20261117/scripts/run_master.sh)、[子节点](full/cann930_20261117/scripts/run_worker.sh) |
| 请求／校验脚本 | [图文4K→32K](full/cann930_20261117/scripts/run_client_mm_4k_32k.sh)、[文本2K→6K](full/cann930_20261117/scripts/run_client_text_2k_6k.sh)、[图文1K→2K](full/cann930_20261117/scripts/run_client_mm_1k_2k.sh) |
| 请求与输出 | [12轮JSON及客户端日志](full/cann930_20261117/performance_results/) |
| 日志与统计 | [逐轮性能报告](full/cann930_20261117/REPORT.md)、[主节点日志](full/cann930_20261117/combined_run/node23/launcher.log)、[子节点日志](full/cann930_20261117/combined_run/node24/launcher.log) |

### 精度验证

本次精度验收数据集为 **GPQA Diamond**。完整模型、Engram开启，使用AISBench 0-shot CoT，按最终content提取选项；无法解析最终答案的样本计错。四次均覆盖相同198题，累计792个请求全部成功；另有8题预检，不计入四次完整实验。

#### 四次实验结果

| 次数 | 日期 | 输出预算 | 正确/题数 | 最终答案准确率 | 达输出上限 | 无最终选项 |
|---|---|---|---|---|---|---|
| 1 | 11月17日 | 32768 | 164/198 | 82.828283% | 24 | 24 |
| 2 | 11月19日第1轮 | 65536 | 180/198 | 90.909091% | 1 | 2 |
| 3 | 11月19日第2轮 | 65536 | 178/198 | 89.898990% | 0 | 1 |
| 4 | 11月19日第3轮 | 65536 | 178/198 | 89.898990% | 2 | 2 |

当前65536预算三轮平均为**90.235690%**。四次混合配置的描述性平均为88.383838%，不能作为同条件四轮均值或独立792题成绩。

| 配置 | 第1次 | 第2–4次 |
|---|---|---|
| 服务上下文 / 输出预算 | 40960 / 32768 | 73728 / 65536 |
| temperature / top_p / top_k | 0 / 1 / 未显式设置 | 1.0 / 0.95 / 0 |
| 思考设置 | thinking=true，reasoning_effort=100 | enable_thinking=true，reasoning_effort=low |
| seed / 并发 / warmup | 42 / 8 / 0 | 42 / 8 / 0 |
| 模型及拓扑 | 完整BF16＋Engram，TP1/PCP8/DP2/EP16 | 相同模型及拓扑 |

第1次AISBench原始评分为165/198（83.333333%）；零基题号52的最终content为空，仅推理文本含正确选项，按统一最终答案口径计错为164/198。原始评分与预测均保留。第1次客户端审计最初退出1，修正JSONL读取与评分核对后，复用原预测完成CPU复核，最终校验退出0。

#### 官方结果与差距

DeepSeek官方DeepInfer仓库评测文档给出的GPQA Diamond参考值为 **93.4% Pass@1**。默认模型为`DeepSeek-V4.1-Flash-1017`，198题、每题8次采样取平均准确率，reasoning_effort=100、temperature=1.0、top_p=1.0，prompt＋输出预算上限147456，默认LLM judge。官方将该值定位为部署sanity check参考。

当前三轮均值相对93.4%低 **3.164310个百分点**，当前配置与该官方方法未完全对齐。较早9月版`DeepSeek-V4.1-Flash`模型卡公布90.9%，不能把它当成Flash-1017同配置的官方结果。[官方评测说明](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/eval/README.md)、[官方GPQA脚本](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/eval/gpqa/run.sh)、[采样与长度配置](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/eval/run_eval.py)、[来源及校验](full/gpqa_four_runs/official_reference.json)。

#### T检验

以官方93.4%作为固定参考值，对当前同配置三轮准确率进行探索性双侧单样本T检验，显著性水平0.05：**t=-9.3980，df=2，名义p=0.011133**；三轮均值的名义95%置信区间为[88.786986%, 91.684395%]。

该计算忽略官方分数的不确定性，只有3轮且共享seed、题集和同一服务，独立性未建立；同时生成和评分配置不同。因此只作为数值差距的探索性分析，**不能据此声明通过官方精度对齐、等效性或非劣效验收**。当前未取得官方逐题预测或重复评测分数，无法完成同条件双样本/逐题配对的官方T检验。

补充分析将第1次与后三次按198道相同题目配对，比较逐题0/1正确性；双侧T检验、Holm校正如下，详细报告同时提供适用于配对二分类结果的精确McNemar检验。

| 对比 | 提升百分点 | t（df=197） | 双侧p | Holm校正p |
|---|---|---|---|---|
| 第2次−第1次 | +8.080808 | 3.349191 | 0.000971 | 0.002913 |
| 第3次−第1次 | +7.070707 | 2.911182 | 0.004016 | 0.004016 |
| 第4次−第1次 | +7.070707 | 3.202848 | 0.001587 | 0.003174 |

三组均呈改善；由于多项生成配置同时变化，不能将改善归因于单一参数或代码修复。配对T是二分类差值的大样本近似；题目层面的分析也不代表独立多轮的官方对照。公式、逐题配对表及全部数值见[四次精度与统计报告](full/gpqa_four_runs/REPORT.md)。

| 交付类型 | 文件 |
|---|---|
| 服务端脚本 | [第1次](full/cann930_20261117/combined_run/snapshots/)、[第2–4次](full/cann930_20261119_precision/scripts/run_master.sh)、[子节点](full/cann930_20261119_precision/scripts/run_worker.sh) |
| 请求／校验脚本 | [第1次客户端](full/cann930_20261117/scripts/run_gpqa.sh)、[当前客户端](full/cann930_20261119_precision/scripts/run_gpqa.sh)、[三轮审计](full/cann930_20261119_precision/scripts/audit_saved_results.py)、[统计脚本](full/gpqa_four_runs/scripts/analyze_gpqa.py) |
| 请求与输出 | [第1次完整结果](full/cann930_20261117/precision_results/gpqa_198/)、[后三次完整结果](full/cann930_20261119_precision/results/)、[逐题配对CSV](full/gpqa_four_runs/paired_scores.csv) |
| 日志与统计 | [第1次报告](full/cann930_20261117/REPORT.md)、[后三次报告](full/cann930_20261119_precision/REPORT.md)、[四次统计JSON](full/gpqa_four_runs/results.json)、[四次统计报告](full/gpqa_four_runs/REPORT.md) |

#### 验收范围与后续条件

本轮32专家、Engram关闭的RL与1M功能已按指定范围验收；完整模型的基础推理、性能与GPQA测量分别保留各自配置和日期，不把缩减模型结论外推。

| 项目 | 当前状态 |
|---|---|
| TP>1 / DCP / SP / PP | 当前支持范围排除：TP>1未支持，DCP/SP依赖TP>1，PP受Engram限制；不列为待修复项 |
| 主场景硬件性能 | CANN930完整模型4K图文→32K的视觉/prefill/decode及跨rank分析、0.8目标验收尚需补齐；本轮按22单机范围交付，双机16卡实验留待后续资源安排 |
| 图片方向 | 已测图片为宽1080×高1920；若严格要求宽1920×高1080，需按[横图客户端](full/cann930_20261117/scripts/run_client_mm_4k_32k_landscape.sh)补测 |
| 精度官方对照 | 四次GPQA测量和T统计已完成；官方参考与当前生成/评分方法不同，正式同条件对照及门限尚需明确 |
| 源码交付 | 基础提交公开可达；本轮源码、补丁与SHA已交付，上游修复PR尚未提交 |

资源策略为发现外部NPU占用或本任务进程被停止即释放并停止，不自动重启；当前实验结束后8张卡已释放。[本轮验收总览](reduced/cann930_20261120/README.md)

仓库保存脚本、原始输出、日志、源码指纹及补丁，不包含模型权重、tokenizer或镜像二进制。所有通过结论仅限列明配置。文件完整性可在仓库根目录运行`shasum -a 256 -c manifest.sha256`核对。

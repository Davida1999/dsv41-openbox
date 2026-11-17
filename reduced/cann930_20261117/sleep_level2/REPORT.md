# DeepSeek V4.1 Top32 Sleep level-2 验证报告

日期为 2026-11-17，时区为 Asia/Shanghai。容器服务日志使用 UTC。

## 验收结果

**Sleep level-2 五轮验收通过。** 8 个 DP 副本分别完成 completions、chat 两个接口的温度 0 单请求检查。每轮 16 次请求，五轮共 80 次恢复后请求，文本、token IDs、逐 token logprob 均与首次加载 checkpoint 后固化的基准完全一致。最大 logprob 差值为 0。

预热和基准阶段另有 32 次请求。每个副本的相同提示词在无休眠重复检查中完全一致。

五轮恢复后相对加载基线的最大单卡 HBM 增量为 **40 MiB**。以首轮恢复后的稳态占用为参照，后续轮次的最大单卡增量为 **1 MiB**。完整逐卡数据保存在 [五轮显存 CSV](memory_five_cycles.csv)，各阶段原始采样保存在 [运行结果](runs/fixed_complete_eager_20261117_01/validation/)。

裁剪模型的基准输出包含重复及错误内容。本次通过结论适用于休眠恢复一致性和五轮显存稳定性。

## 实测配置

| 项目 | 实际配置 |
|---|---|
| 服务器与设备 | 177.42.22.1，8×Ascend950DT，单卡 96 GiB |
| 容器 | c00954197_dsv41_cann930 |
| 软件 | CANN 9.3.0，Python 3.12.16，Torch 2.10.0+cpu，TorchNPU 2.10.0.post7 |
| 框架 | vLLM 0.32.0+empty，Ascend 0.19.1rc2.dev3401+g28e9e97ba，另含本次两个 Python 补丁 |
| 模型 | 40 层、32 路由专家、Top-6，BF16 模型 |
| Engram | engram_layer_ids=[]，过滤 8 个 Engram 索引条目 |
| 并行 | TP1、DP8、EP8、PCP1、DCP1、PP1 |
| 执行模式 | eager、同步调度，prefix caching 与 chunked prefill 关闭 |
| 上下文与批处理 | 8192，max-num-batched-tokens=8192，max-num-seqs=2 |
| 显存利用率参数 | 0.60 |
| 服务地址 | http://127.0.0.1:18242 |
| 请求 | temperature=0，top_p=1，seed=42，max_tokens=32，return_token_ids=true，logprobs=5 |
| 固定提示词 | The capital of France is |

每次只发送一个生成请求，通过 X-data-parallel-rank 定向覆盖副本 0–7。两个接口分别对比各自的基准。完整环境、源码 SHA256 和 Git 状态见 [环境快照](runs/fixed_complete_eager_20261117_01/environment.json)。

## Level-2 恢复流程

每轮记录状态与逐卡显存，执行分阶段唤醒、checkpoint 重载及双接口一致性检查。恢复步骤如下。

```text
记录休眠前占用
→ sleep(level=2)
→ 确认 is_sleeping=true，记录休眠后占用
→ wake_up(tags=weights)
→ collective_rpc reload_weights，同一 checkpoint、is_checkpoint_format=true
→ wake_up(tags=kv_cache)
→ 确认 is_sleeping=false、health=200，记录唤醒后占用
→ 双接口逐副本单请求生成并与加载基准比较
→ 记录生成后的稳态占用
```

本实验显存稳定性检查的容差为 0.05 GiB，即 51.2 MiB，同时提供相对首次加载基线的全部变化量。生命周期耗时包含本模型的 checkpoint 重载，逐轮保留实测时间。

## 失败原因与修复

原始复现中，无休眠重复请求保持一致，level-2 恢复后两个接口输出发生变化。检查 1264 个参数的指纹，其中 676 个为全量 SHA256、588 个为头尾采样，变化参数集中在 40 层 attn_sink。每个 sink 为 64 个 F32 元素，使用全量哈希，恢复后均为全 0，地址相同。FP32 router 缓存指纹保持一致。[参数差异证据](patches/attn_sink_parameter_differences.json) · [根因摘要](patches/root_cause_proof.json)

V4.1 checkpoint 加载函数的 sink 分支直接执行 param.data.copy_。重载期间参数暂处于 meta 状态，该分支绕过 layerwise weight_loader，加载计数没有更新，finalize 恢复了已被 level-2 丢弃内容的原存储。修复为在 TP 头切分后调用已有 weight_loader。[sink 修复补丁](patches/attn_sink_reload.patch)

同时发现无学习权重 RMSNorm 的全 1 常量是普通张量属性，休眠后变为全 0。已将其注册为非持久 buffer，使其参与备份恢复。仅修复该常量后的实机试验仍有输出差异，随后定位到上述 sink 加载问题。[RMSNorm 补丁](patches/rmsnorm_level2.patch)

| 回归检查 | 修复前 | 修复后 |
|---|---|---|
| RMSNorm 常量管理及恢复 | 3 项中 2 项失败 | 3/3 通过 |
| 实际 V4.1 sink 加载函数与 checkpoint reload | TP1、两种 TP2 分片均失败 | 3/3 通过 |
| NPU level-2 恢复 | 输出变化 | 五轮、8 副本、双接口全部一致 |

CPU sink 回归同时检查原地恢复地址及共享 Parameter 引用。TP2 部分覆盖 CPU 加载分片行为。实机配置为表中的 TP1。原始回归日志、测试脚本和失败运行均随交付件保存。

## 五轮概览

占用单位为 MiB，区间为 8 张卡的最小值至最大值。生命周期时间包含快照和 checkpoint 重载。

| 轮次 | 一致请求 | sleep 前 | sleep 后 | wakeup 后 | 请求后稳态 | 生命周期秒 |
|---|---|---|---|---|---|---|
| 1 | 16/16 | 59030–59395 | 8100–8436 | 58996–59341 | 59070–59375 | 34.58 |
| 2 | 16/16 | 59070–59375 | 8140–8436 | 59036–59341 | 59070–59375 | 33.71 |
| 3 | 16/16 | 59070–59375 | 8140–8436 | 59036–59341 | 59070–59375 | 33.65 |
| 4 | 16/16 | 59070–59375 | 8140–8437 | 59036–59341 | 59070–59375 | 33.32 |
| 5 | 16/16 | 59070–59375 | 8174–8465 | 59036–59341 | 59070–59375 | 33.26 |

## 服务脚本与五轮输出

服务使用 [启动脚本](scripts/start_server.sh)，实际命令保存于 [运行快照](runs/fixed_complete_eager_20261117_01/actual_server_command.sh)。[验证客户端](scripts/verify_sleep2.py) 保存请求、响应、状态、内存和逐轮比较。[完整服务日志](runs/fixed_complete_eager_20261117_01/server.log) · [客户端日志](runs/fixed_complete_eager_20261117_01/validation.log)

当前容器已应用本次两个补丁。服务脚本使用服务器现有 basic 工作区及模型视图。在 22.1 宿主机进入容器。

```bash
docker -H unix:///run/c00954197-cann930/docker.sock exec -it c00954197_dsv41_cann930 bash -l
```

在第一个容器终端启动服务，使用新的运行目录。

```bash
SLEEP2_ROOT=/mnt/share/c00954197/Workspace/dsv41/basic
SLEEP2_RUN="$SLEEP2_ROOT/runs/manual_sleep2_01"
mkdir -p "$SLEEP2_RUN"
bash "$SLEEP2_ROOT/scripts/start_server.sh" eager "$SLEEP2_RUN" > "$SLEEP2_RUN/server.log" 2>&1
```

在第二个容器终端检查服务并执行五轮验证，结束后释放本轮服务。

```bash
SLEEP2_ROOT=/mnt/share/c00954197/Workspace/dsv41/basic
SLEEP2_RUN="$SLEEP2_ROOT/runs/manual_sleep2_01"
curl --noproxy '*' --fail http://127.0.0.1:18242/health
python3 "$SLEEP2_ROOT/scripts/verify_sleep2.py" --output "$SLEEP2_RUN/validation" \
  --cycles 5 --ranks 0,1,2,3,4,5,6,7 --max-tokens 32 --inspect-tensors
python3 "$SLEEP2_ROOT/scripts/stop_server.py" "$SLEEP2_RUN"
```

DP0 的五轮原始输出如下，8 个副本的全部 80 个恢复后响应均保存在运行目录。

| 轮次 | completions 原始输出 | chat 原始输出 |
|---|---|---|
| 1 | [completions](runs/fixed_complete_eager_20261117_01/validation/cycle_1_dp0_completions_response.json) | [chat](runs/fixed_complete_eager_20261117_01/validation/cycle_1_dp0_chat_completions_response.json) |
| 2 | [completions](runs/fixed_complete_eager_20261117_01/validation/cycle_2_dp0_completions_response.json) | [chat](runs/fixed_complete_eager_20261117_01/validation/cycle_2_dp0_chat_completions_response.json) |
| 3 | [completions](runs/fixed_complete_eager_20261117_01/validation/cycle_3_dp0_completions_response.json) | [chat](runs/fixed_complete_eager_20261117_01/validation/cycle_3_dp0_chat_completions_response.json) |
| 4 | [completions](runs/fixed_complete_eager_20261117_01/validation/cycle_4_dp0_completions_response.json) | [chat](runs/fixed_complete_eager_20261117_01/validation/cycle_4_dp0_chat_completions_response.json) |
| 5 | [completions](runs/fixed_complete_eager_20261117_01/validation/cycle_5_dp0_completions_response.json) | [chat](runs/fixed_complete_eager_20261117_01/validation/cycle_5_dp0_chat_completions_response.json) |

[加载基准与五轮文本汇总](five_cycle_outputs.txt) · [机器可读验收结果](runs/fixed_complete_eager_20261117_01/validation/report.json)

## 逐卡显存记录

单位为 MiB，增长量相对首次加载并完成预热后的稳态基线。每轮另保存了权重单独唤醒、重载完成阶段及原始 npu-smi 输出。

| 轮次 | NPU | sleep 前 | sleep 后 | wakeup 后 | 请求后 | 相对加载基线增长 |
|---|---|---|---|---|---|---|
| 1 | 0 | 59030 | 8100 | 58996 | 59070 | +40 |
| 1 | 1 | 59395 | 8431 | 59341 | 59375 | -20 |
| 1 | 2 | 59393 | 8429 | 59339 | 59373 | -20 |
| 1 | 3 | 59377 | 8427 | 59323 | 59356 | -21 |
| 1 | 4 | 59379 | 8421 | 59325 | 59359 | -20 |
| 1 | 5 | 59370 | 8434 | 59316 | 59350 | -20 |
| 1 | 6 | 59387 | 8436 | 59333 | 59366 | -21 |
| 1 | 7 | 59387 | 8423 | 59333 | 59367 | -20 |
| 2 | 0 | 59070 | 8140 | 59036 | 59070 | +40 |
| 2 | 1 | 59375 | 8431 | 59341 | 59375 | -20 |
| 2 | 2 | 59373 | 8429 | 59339 | 59373 | -20 |
| 2 | 3 | 59357 | 8427 | 59322 | 59357 | -20 |
| 2 | 4 | 59359 | 8421 | 59325 | 59359 | -20 |
| 2 | 5 | 59350 | 8434 | 59316 | 59350 | -20 |
| 2 | 6 | 59367 | 8436 | 59332 | 59366 | -21 |
| 2 | 7 | 59366 | 8423 | 59333 | 59367 | -20 |
| 3 | 0 | 59070 | 8140 | 59036 | 59070 | +40 |
| 3 | 1 | 59375 | 8431 | 59341 | 59375 | -20 |
| 3 | 2 | 59373 | 8429 | 59339 | 59373 | -20 |
| 3 | 3 | 59357 | 8426 | 59323 | 59357 | -20 |
| 3 | 4 | 59358 | 8421 | 59325 | 59359 | -20 |
| 3 | 5 | 59350 | 8434 | 59316 | 59350 | -20 |
| 3 | 6 | 59366 | 8436 | 59332 | 59366 | -21 |
| 3 | 7 | 59367 | 8423 | 59333 | 59367 | -20 |
| 4 | 0 | 59070 | 8140 | 59036 | 59070 | +40 |
| 4 | 1 | 59375 | 8431 | 59341 | 59375 | -20 |
| 4 | 2 | 59373 | 8429 | 59339 | 59373 | -20 |
| 4 | 3 | 59356 | 8427 | 59323 | 59357 | -20 |
| 4 | 4 | 59359 | 8421 | 59325 | 59359 | -20 |
| 4 | 5 | 59350 | 8434 | 59316 | 59350 | -20 |
| 4 | 6 | 59367 | 8437 | 59333 | 59367 | -20 |
| 4 | 7 | 59367 | 8423 | 59332 | 59367 | -20 |
| 5 | 0 | 59070 | 8174 | 59036 | 59070 | +40 |
| 5 | 1 | 59375 | 8465 | 59341 | 59375 | -20 |
| 5 | 2 | 59373 | 8429 | 59339 | 59373 | -20 |
| 5 | 3 | 59357 | 8426 | 59322 | 59357 | -20 |
| 5 | 4 | 59358 | 8421 | 59324 | 59359 | -20 |
| 5 | 5 | 59350 | 8434 | 59316 | 59350 | -20 |
| 5 | 6 | 59367 | 8436 | 59332 | 59367 | -20 |
| 5 | 7 | 59367 | 8423 | 59333 | 59367 | -20 |

## 资源释放

实验结束后停止本次服务，核对 8 张卡均无运行进程，18242 端口已释放。[最终资源核验](runs/fixed_complete_eager_20261117_01/cleanup_verification.json) · [最终设备快照](runs/fixed_complete_eager_20261117_01/final_npu_smi.txt)

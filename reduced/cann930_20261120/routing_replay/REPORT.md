# CANN930 专家路由返回与缓存一致性验收报告

日期：2026-11-20。服务器177.42.22.1，容器`c00954197_dsv41_cann930`，CANN 9.3.0。模型为40层、32路由专家、Top-6、BF16，Engram关闭；TP1/DP8/EP8、MRV2。

## 验收结论

**受测配置通过。** 最终TorchCompile=3、FULL_DECODE_ONLY完整矩阵153个请求全部完成，记录到879个同一步调度至少两个真实请求的步骤。两个接口的冷请求、真实缓存命中及sleep/wake后，文本、token IDs、完整专家数组和逐token logprob均严格一致，最大logprob差为0。

最终服务日志从启动至正常关闭的ERROR与Traceback计数均为0；停服只向主进程先发终止信号，未强杀残留进程，8张卡及测试端口全部释放。[日志核对](runs/routing_20261120_15_clean_shutdown/log_audit.json) · [停服记录](runs/routing_20261120_15_clean_shutdown/service_stop.json) · [释放核验](runs/routing_20261120_15_clean_shutdown/release_review.json)

| 模式 | 请求数 | 缓存与生命周期 | 结论 |
|---|---|---|---|
| Eager、BI=1 | 9 | 双接口cold/hit/repeat；level-2重载恢复；token-in/token-out | [通过](runs/routing_20261120_09_bi_hc_fixed/matrix/summary.json) |
| FULL_DECODE_ONLY、TorchCompile=0、BI=1 | 153 | 全并发矩阵、双接口缓存及level-1恢复 | [通过](runs/routing_20261120_11_graph_no_compile/matrix/summary.json) |
| FULL_DECODE_ONLY、TorchCompile=3、BI=1 | 153 | 全并发矩阵、双接口缓存及level-1恢复；正常关闭回归 | [通过](runs/routing_20261120_15_clean_shutdown/matrix/summary.json) |

BI表示`VLLM_BATCH_INVARIANT=1`。本轮按[PR #52902](https://github.com/vllm-project/vllm/pull/52902)的路由捕获、缓存复用与休眠恢复合同适配DSV4.1；训练侧强制专家回放不在该PR或本次实测范围内。

## 固定配置与矩阵

| 项目 | 实测设置 |
|---|---|
| 模型 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32` |
| 执行 | MRV2、同步调度；Eager或FULL_DECODE_ONLY；最终TorchCompile=3 |
| 严格一致性 | BI=1；SWA bounded replay关闭；prefix caching开启 |
| 预算 | max-model-len=2048，batched tokens=2048，max-num-seqs=2，KV显式2 GiB |
| 请求 | temperature=0、seed=42、16个输出token、返回token IDs及logprobs |
| 通信 | CCU_SCHED、HCCL enp35s0f2、Gloo enp39s0f0、HCCL_BUFFSIZE=256 MiB |
| Host分配 | `numactl --interleave=all`；8个worker均核对到跨NUMA节点策略 |

最终153请求包括双接口逐DP单请求及同DP双请求、四组跨DP配对、每接口20组长短混合双请求、cold/hit/repeat、分阶段level-1唤醒及token-in/token-out。原始调度证据证明实际发生了双请求同批执行。[调度证据](runs/routing_20261120_15_clean_shutdown/matrix/same_step_multi_request_evidence.json) · [Host策略](runs/routing_20261120_15_clean_shutdown/host_policy/)

## 返回专家

每个响应立即保存原文后解码NPY。数组为`(P+G−1,40,6)`、整数uint8、逻辑ID范围0–31；各token/层的6个专家互异。最后一个采样token未执行forward，因此减1。

下表为实际单请求输出中前3个已生成token在部分层的专家：

| 输出token序号 | token ID | 路由数组行 | 层 | E32逻辑专家 | 原384专家编号 |
|---|---|---|---|---|---|
| 0 | 223 | 5 | 0 | [22, 20, 29, 12, 11, 10] | [291, 261, 353, 192, 180, 178] |
| 0 | 223 | 5 | 20 | [22, 27, 1, 18, 30, 26] | [306, 349, 3, 259, 365, 339] |
| 0 | 223 | 5 | 39 | [17, 10, 28, 18, 22, 19] | [175, 95, 275, 193, 235, 201] |
| 1 | 6793 | 6 | 0 | [11, 26, 29, 12, 10, 25] | [180, 311, 353, 192, 178, 308] |
| 1 | 6793 | 6 | 20 | [27, 30, 23, 1, 22, 8] | [349, 365, 307, 3, 306, 102] |
| 1 | 6793 | 6 | 39 | [11, 21, 27, 14, 9, 8] | [121, 215, 269, 149, 91, 85] |
| 2 | 16 | 7 | 0 | [26, 13, 14, 22, 29, 20] | [311, 198, 203, 291, 353, 261] |
| 2 | 16 | 7 | 20 | [1, 27, 30, 9, 23, 28] | [3, 349, 365, 103, 307, 351] |
| 2 | 16 | 7 | 39 | [14, 11, 26, 9, 27, 20] | [149, 121, 268, 91, 269, 202] |


[完整专家JSON](examples/single_experts.json) · [原始单请求响应](examples/single_response.json) · [全部请求与NPY](runs/routing_20261120_15_clean_shutdown/matrix/) · [逐层新旧专家映射](expert_remap.json)

## 问题定位与修复

| 问题 | 定位与修复 | 证据 |
|---|---|---|
| 双请求维度断言/HTTP500 | TP1本地dispatch填充行数没有匹配capture分支；根据真实padding metadata仅复制有效本地行 | [补丁](patches/local_dp_padding_capture.patch.gz) |
| 长请求decoder层专家全零 | bounded replay只计算尾窗，不能交付完整prompt的逐层专家；返回专家时关闭该优化并保留prefix caching | [补丁](patches/routing_capture_replay_guard.patch.gz) · [失败数组](diagnostics/long_capture_before_fix/) |
| 冷/热缓存输出及专家差异 | 相同输入在第0层开始偏离；确认native mHC pre随M选择不同归约。严格BI模式固定64行计算，原非BI路径保留 | [标量/层级诊断](diagnostics/hc_batch_dependence/) · [固定行补丁](patches/hc_pre_fixed_rows_only.patch.gz) |
| BI接口已安装但运行库缺失 | 不再仅以Python包可导入判定AscendC API可用；缺失时选已有Triton NPU实现 | [补丁](patches/batch_invariant_api_guard.patch.gz) |
| TorchCompile动态循环限制 | 将已验证的固定行计算封装为PrivateUse1不透明算子，提供fake形状规则；数值、动态图追踪和NPU图回放均检查一致 | [补丁](patches/hc_pre_opaque_op.patch.gz) · [NPU组件检查](diagnostics/hc_batch_dependence/opaque_component/validation/summary.json) |
| level-1锁页Host分配207001 | 宿主总可用内存充足，跨NUMA交错分配后实际恢复通过；未把该现象归为HBM不足 | [规避对照](diagnostics/host_pinning/) |
| 正常关闭的EngineDeadError | 先取消异步输出消费者，再关闭engine生产者；停服脚本先通知主进程并等待正常退出 | [补丁](patches/async_shutdown_consumer_first.patch.gz) · [最终完整日志](runs/routing_20261120_15_clean_shutdown/server.log.gz) |

严格一致性结论适用于上述BI配置，不把普通性能模式解释为跨批位级一致。没有通过改变容差、排序专家或伪造缓存命中来接受结果。

## 统一交付文件

| 类型 | 文件 |
|---|---|
| 服务端 | [完整启动脚本](scripts/start_final_graph_20261120.sh) · [实际参数](runs/routing_20261120_15_clean_shutdown/actual_server_command.sh.gz) |
| 请求/校验 | [参数化完整客户端](scripts/validate_routing_matrix_open_20261120.py) · [实测冻结快照](runs/routing_20261120_15_clean_shutdown/snapshots/validate_routing_matrix.py) |
| 请求/输出 | [全部响应与专家](runs/routing_20261120_15_clean_shutdown/matrix/) · [单请求原文](examples/single_request.json) |
| 日志/统计 | [完整日志](runs/routing_20261120_15_clean_shutdown/server.log.gz) · [客户端日志](runs/routing_20261120_15_clean_shutdown/client.log.gz) · [离线复核](evidence_audit.json) |
| 源码/环境 | [版本与实际文件SHA](source/final/source_versions.json) · [补丁](patches/) |

## 复现与离线核对

在22.1宿主机准备一个新目录，复制客户端快照后启动本次同款调度器与资源守护。当前容器已应用补丁，新环境先按源码清单准备相同版本与补丁。

```bash
ROOT=/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay
RUN="$ROOT/runs/manual_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN/snapshots"
cp "$ROOT/scripts/validate_routing_matrix_open_20261120.py" "$RUN/snapshots/validate_routing_matrix.py"
python3 -u "$ROOT/scripts/run_opaque_graph_20261120.py" --run "$RUN" --mode graph > "$RUN/controller.log" 2>&1 &
CONTROL_PID=$!
echo "$CONTROL_PID" > "$RUN/controller.pid"
python3 -u "$ROOT/scripts/resource_policy_guard_20261120.py" --run "$RUN" > "$RUN/resource_guard.log" 2>&1 &
wait "$CONTROL_PID"
```

守护只清理本run的进程；发现外部NPU占用、controller或已初始化worker消失即释放并停止，不自动重启。任务正常完成也主动释放设备。

在归档仓库根目录可用标准库核对全部专家范围/唯一性、原响应与NPY一致性、文本/token/logprob及缓存状态：

```bash
python3 reduced/cann930_20261120/routing_replay/scripts/audit_saved_routing.py \
  reduced/cann930_20261120/routing_replay/runs/routing_20261120_09_bi_hc_fixed \
  reduced/cann930_20261120/routing_replay/runs/routing_20261120_11_graph_no_compile \
  reduced/cann930_20261120/routing_replay/runs/routing_20261120_15_clean_shutdown
```

TP>1、DCP、SP、PP按本轮模型支持范围排除，不列为待修复项。本报告不外推完整384专家＋Engram或其他硬件版本的支持。

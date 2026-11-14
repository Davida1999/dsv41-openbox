# E21 跨机 Gloo 故障定位结论（2026-11-12）

已复现到不依赖模型或 NPU 算子的最小 CPU 程序。关闭 torch 设备后端自动加载，日志确认 torch_npu_imported=false。单对双机 rank 的 CPU all_reduce 在原 enp35s0f2 路径出现数据错误或 Gloo 协议长度异常。固定形状 step 26534 的前 3 个 int32 被改成 1096574604，预期为 849088、849089、849090。

## 对照结果

1. 两台各自 loopback：固定、交替、错峰各 30000 次，全部通过。
2. 原接口跨机：子组、直接 WORLD、固定复用 buffer 均失败。
3. 旧 a5 容器同一 CPU 测试也失败。
4. 普通 Python TCP 双向消息校验 100000 条通过。这不能排除特定 I/O 模式与网络栈、驱动的交互。
5. 改 Gloo 数据接口为 enp39s0f0 后，单对 rank 共 90000 次通过。
6. 保留 177.11.25.1 作为启动协调地址，仅改 Gloo 接口，16 rank/8 对 DP 组，每 rank 三种模式各 30000 次全部通过。
7. E21 双机模型配置仅改 GLOO_SOCKET_IFNAME=enp39s0f0，连续两次 4K 文本＋图片→32K 输出均通过：64 成功、0 失败，每次 2097152 输出 tokens。耗时 691.29/681.39 秒，输出吞吐 3033.68/3077.74 tokens/s。

## 已定位范围

故障与 Gloo 在 enp35s0f2 上的跨机传输路径相关；切换 enp39s0f0 是已验证的规避配置。两台 libtorch_cpu.so SHA256 均为 d9529119ac18ba154521854dc8da477c25c9a34c3265c95e40086a593d741926。原口驱动为 hibmcge，备用口驱动为 hinic3 17.7.8.101。
不能据此断言具体是驱动缺陷、卸载功能、硬件或网络设备；需要网络/驱动级取证才能进一步细分。普通 TCP 测试通过不代表所有 TCP I/O 模式正常。

本次证据不支持将问题直接归因于 dsv4.1、Engram、MoE/MC2 算子或 decoder replay。共享 DP 组、调用错序不是复现本问题的必要条件，因为严格顺序、单一 WORLD 组、固定大小 CPU all_reduce 已经失败。
未修改宿主机网卡配置、驱动或权重；HCCL 接口和 MC2 参数保持不变。模型回归使用与前轮一致的诊断 hook，存在少量日志开销。

## 已验证修改

```bash
export GLOO_SOCKET_IFNAME=enp39s0f0
```

HCCL_IF_IP、HCCL_SOCKET_IFNAME、服务 IP、DP 协调 IP 保持原配置。
该修改已写入基于 E14 创建的 E23 脚本。E23 完整模型也已单独完成四轮长输出验证，结果见下文。

## 材料

- cpu_tests.tar.gz：最小复现脚本及全部 CPU 测试日志。
- cpu_summary.json：CPU 对照结果汇总。
- E21_alt_gloo_client_1/2_terminal.txt：两次完整客户端终端结果。
- model_validation/05_gloo_nic2.tar.gz：模型回归的完整日志与诊断数据。
- reproduction_scripts/：复现脚本、控制脚本及所查阅的上游 Gloo 源码参考。

## E23 完整模型验证

模型：/mnt/share/c00954197/workspace/dsv41-flash-bf16，40 层、384 专家，Engram CPU offload 和 dp_shared_memory 启用。
并行：TP1/PCP8/DP2；两个服务实例各 max-num-seqs=32；客户端64请求、并发64；每请求一张1920×1080合成图片，文本输入4096 tokens，目标输出32768 tokens。服务未在四轮之间重启，后续轮次存在缓存复用。

| 轮次 | 成功 | 失败 | 输出 tokens | 耗时（秒） | 输出吞吐（tokens/s） |
|---|---:|---:|---:|---:|---:|
| 1 | 64 | 0 | 2097152 | 950.84 | 2205.58 |
| 2 | 64 | 0 | 2097152 | 941.53 | 2227.39 |
| 3 | 64 | 0 | 2097152 | 954.54 | 2197.04 |
| 4 | 64 | 0 | 2097152 | 957.42 | 2190.41 |


四轮均满足完整输出要求，服务端未发现执行异常；证明本轮配置下未复现，不能保证所有工作负载永久无故障。E23性能测试未加载前轮实验诊断hook。

## 服务器证据路径

- /workspace/experiments/tp1_dp16_20261029/E21/root_cause_20261112/cpu/：CPU最小复现与TCP对照脚本、原始日志、退出码。
- /workspace/experiments/tp1_dp16_20261029/E21/root_cause_20261112/model/05_gloo_nic2/：E21切换网口后的双机回归。
- /workspace/experiments/tp1_dp16_20261029/E23/run_master_mm.sh、run_worker_mm.sh：已应用Gloo网口修改的简洁启动脚本。
- /workspace/experiments/tp1_dp16_20261029/E23/runs_1108/run_02/：E23四轮4K+图片到32K性能测试及服务日志。

## 部署前核对

在两台容器中确认enp39s0f0存在且互通。该环境使用host网络模式，25.1对应接口地址197.13.49.1，26.1对应197.13.50.1。仅修改测试或服务进程的GLOO_SOCKET_IFNAME；不要将本结论自动推广到其他机器，也不要同时改动HCCL参数掩盖变量。

## 后续深入定位

如果需要确定到驱动内部，应携带纯CPU最小复现、两接口对照、库哈希和版本交给网络/驱动维护方，进一步比较具体TCP I/O模式与网络路径；本次未修改网卡卸载选项、内核或驱动。

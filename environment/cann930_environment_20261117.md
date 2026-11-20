# CANN930 验证环境

更新日期为 2026-11-20。Sleep level-2、HCCL 和 NPU IPC 专项在 22.1 实测，完整模型性能与 GPQA 在 23/24 双节点实测。

| 项目 | 实际配置 |
|---|---|
| 服务器 | 专项 177.42.22.1，完整模型 177.42.23.1/24.1，每节点 8×Ascend950DT，单卡 96 GiB |
| 宿主系统 | openEuler，Linux 6.6.0-145.3.35.166.oe2403sp3.aarch64 |
| 容器 | `c00954197_dsv41_cann930` |
| Docker socket | `/run/c00954197-cann930/docker.sock` |
| 镜像 ID | `sha256:3614b7afe7000897bf6f1c60a415f2547feb0e033bae0c9bf76c68a76ecf944f` |
| 网络与 IPC | host / host |
| CANN toolkit | 9.3.0，V100R001C25B114 |
| CANN 激活目录 | `/usr/local/Ascend/cann-9.3.0/cann-9.3.0` |
| Python | `/usr/local/python3.12.16/bin/python3`，3.12.16 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| vLLM | 0.32.0+empty，源码 `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | 0.19.1rc2.dev3401+g28e9e97ba，源码 `28e9e97ba254b076a88275e8a275556cb9a11664` |
| Triton-Ascend / Transformers | 3.6.0.post2 / 5.19.0 |
| 执行模式 | 既有Sleep/权重更新：eager同步；新路由：BI=1、eager及FULL_DECODE_ONLY；1M：async+chunked prefill+FULL_DECODE_ONLY |
| 23/24 驱动 | 26.2.0.b020 |
| 通信接口 | Gloo `enp39s0f0`，HCCL `enp35s0f2` |
| 专项目录 | `/mnt/share/c00954197/Workspace/dsv41/basic` |
| 完整模型目录 | `/mnt/share/c00954197/Workspace/dsv41/performance_cann930`、`precision_cann930` |

原始容器配置、Sleep 实测快照和权重实验后的 CPU 元数据采集分别保存。容器使用 host IPC，Docker 的 `ShmSize` 配置值保留在原始 JSON。[容器配置](cann930_container_20261117.json) [Sleep 环境快照](../reduced/cann930_20261117/sleep_level2/runs/fixed_complete_eager_20261117_01/environment.json) [权重实验后元数据](../reduced/cann930_20261117/weight_update/environment/environment_after_run.json)

## 模型资产与实验参数

| 用途 | 服务器路径或配置 |
|---|---|
| 本轮实测 checkpoint A | `/mnt/share/c00954197/Workspace/dsv41/basic/model` |
| 32 专家实体副本 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32` |
| 完整 384 专家 BF16 模型 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16` |
| 缩减模型配置 | 40 层、32 路由专家、1 个共享专家、Top-6、Engram 关闭 |
| 缩减模型索引 | 6248 张量、503 分片、135927315272 字节 |
| Sleep 拓扑与预算 | TP1/DP8/EP8，8192 token 上限，max-num-seqs=2，利用率参数 0.60 |
| HCCL 拓扑 | 推理 NPU0–3，TP1/DP4/EP4，独立发送端 NPU7 |
| IPC 拓扑 | 推理 NPU0–7，TP1/DP8/EP8，8 个同置发送进程 |
| 更新预算 | 显式 KV 2 GiB，2048 token 上限，单 packed buffer 1280 MiB，HCCL_BUFFSIZE 256 MiB |

实测记录使用当时的 checkpoint 路径。当前 32 专家副本已实体化并剔除未索引的 Engram 残留，全部 6248 个有效张量保留。完整模型已移动至 `model` 目录，配置与权重文件 inode 校验一致。后续启动命令应绑定所选模型的当前路径。

## 已应用补丁与验证范围

| 修改 | 作用 | 验证状态 |
|---|---|---|
| RMSNorm 非持久 buffer | 将无学习权重的常量纳入 level-2 备份与恢复 | 3 项 CPU 回归与五轮 NPU 验证通过 |
| attention sink weight_loader | 恢复 layerwise 加载计数与原地权重恢复 | 3 项 CPU 回归与五轮 NPU 验证通过 |
| expert_ids_per_ep_rank 跳过加载配额 | 排除运行时通信映射，释放逐层暂存输入 | 3 项 CPU 回归及两种后端全量回环通过 |
| IPC 逐 chunk 导出 rebuild 参数 | 为每次接收重建生成对应的引用计数参数 | 5 项新增 CPU 回归、106 项仓库测试和 2026-11-18 单卡 20 轮及 8 卡全量实机复验通过 |

前三项已包含在完成实机验收的源码中。最新 IPC 生命周期补丁已于 2026-11-18 使用相同源码完成实机复验，发送端 allocated 回到 0。源码哈希与覆盖范围见专项报告。[Sleep 补丁](../reduced/cann930_20261117/sleep_level2/patches/) [权重更新补丁](../reduced/cann930_20261117/weight_update/patches/)

23/24 双节点额外使用 Gloo CPU 超时透传补丁，完整模型 12 轮性能和 GPQA 实测覆盖该环境。对应源码 SHA256 为 `ccf708b35a36e1c5606cc9c0841253d9a08e2887f8f7ccd3368d16a8f72ca2c6`。[补丁](../full/cann930_20261117/patches/cpu_timeout_fix.patch) [完整模型报告](../full/cann930_20261117/REPORT.md)

## 进入当前容器

在本轮各宿主机执行以下命令进入对应 CANN930 容器。

```bash
docker -H unix:///run/c00954197-cann930/docker.sock exec -it c00954197_dsv41_cann930 bash -l
```

服务端与验证命令见各专项报告。2026-11-18 已完成 IPC 单卡多轮回收和完整 IPC 回环，实验后 8 张卡全部释放。冷热缓存差异已由[2026-11-20严格配置](../reduced/cann930_20261120/routing_replay/REPORT.md)闭环。

## 完整模型精度配置

2026-11-19 的完整模型 GPQA 使用 73728 服务上下文和 65536 输出预算，temperature=1.0、top_p=0.95、top_k=0、reasoning_effort=low、enable_thinking=true。双节点同一次启动完成三轮各 198 题，原始镜像、实际配置及脚本哈希随结果保存。[最新精度报告](../full/cann930_20261119_precision/REPORT.md)

正式精度范围为GPQA Diamond，四次完整结果、官方DeepInfer参考与T检验见[精度统计报告](../full/gpqa_four_runs/REPORT.md)。专家返回、缓存和生命周期新验收已完成，见[路由报告](../reduced/cann930_20261120/routing_replay/REPORT.md)。

## 2026-11-20实际代码叠加与配置

路由完整捕获需关闭SWA bounded replay并保留prefix caching；严格一致性使用BI=1。mHC固定64行计算封装为不透明NPU算子以兼容TorchCompile；缺失AscendC API库时使用现有Triton NPU实现。正常退出先取消异步输出消费者，再关闭engine，停服先通知主进程。level-1使用跨NUMA交错Host分配。

实际文件SHA与库版本见[当前源码清单](../reduced/cann930_20261120/routing_replay/source/final/source_versions.json)，前一代码阶段也保留快照。基础提交均已核验公开可达：[vLLM](https://github.com/vllm-project/vllm/commit/5df63a7203cdb9328d824a27b682e77c7b9928be)、[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend/commit/28e9e97ba254b076a88275e8a275556cb9a11664)。基础镜像ID不代表容器内后续Python补丁已进入镜像二进制。

1M配置为上下文1048576、batch8192、max-seqs1、KV16 GiB/卡、利用率参数0.80；BI=0、async/chunked prefill开启、SWA bounded replay关闭。[1M实测](../reduced/cann930_20261120/long_context/REPORT.md)

TP>1、DCP、SP、PP不在本轮支持范围；其中DCP/SP依赖TP>1，PP受Engram限制。实验资源策略已改为外部占用或本任务进程停止即释放、不自动重启。

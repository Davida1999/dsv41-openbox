# DSV4.1 权重更新实机复验报告
2026-11-20后续更新：本文记录的冷热缓存差异已在[严格路由配置](../../../../cann930_20261120/routing_replay/REPORT.md)中定位并通过实机验收。下文保留11月18日的原始实验结果与观察。

日期为 2026-11-18，采用 Asia/Shanghai 日期口径。服务器为 177.42.22.1，容器为 c00954197_dsv41_cann930，CANN 9.3.0。模型为 40 层、32 路由专家、Top-6、BF16，Engram 关闭。

## 验收结果

接收端 OOM 修复和 IPC 引用计数补丁已完成实机验证。HCCL 与 NPU IPC 均完成全量 A→B→A，每轮每个发送端及每个接收副本均为 6248 张量、135927315272 字节。冷缓存请求严格匹配对应冷加载基准，缓存命中请求严格匹配对应缓存命中基准，比较覆盖文本、token IDs、token 字符串和逐 token logprob。

单卡 20 轮对照中，旧实现数据传输完成后分配量逐轮增长 8 MiB，累计 160 MiB，并出现 1 条共享 NPU 张量释放告警。修复后 20 轮最大 allocated 增量为 0，退出告警为 0。[旧实现](ipc_baseline.json) [修复后](ipc_fixed.json) [修复后日志](single_card_fixed.log)

8 卡全量 IPC 的两轮更新中，各发送进程的 allocated 在显式清理前、清理后均为 0，reserved 在清理后均为 0，退出共享 NPU 张量告警为 0。[逐端与缓存审计](acceptance_audit.json)

新增观察为同一权重的冷缓存请求与缓存命中请求生成输出存在差异，根因仍待定位。该现象在独立冷加载 B 服务中也能复现。

## 全量更新与显存

| 后端 | 阶段 | 接收副本 | 每端张量数 / 字节 | 发送端 allocated | 接收卡 HBM 采样峰值 |
|---|---|---|---|---|---|
| hccl | B | 4 | 6248 / 135927315272 | 全部为 0 | 61555 MiB |
| hccl | A-restored | 4 | 6248 / 135927315272 | 全部为 0 | 57115 MiB |
| npu_ipc | B | 8 | 6248 / 135927315272 | 全部为 0 | 44611 MiB |
| npu_ipc | A-restored | 8 | 6248 / 135927315272 | 全部为 0 | 41954 MiB |

HCCL 使用推理卡 0–3、独立发送卡 7，IPC 使用 8 个同置发送进程。单 packed buffer 为 1280 MiB，显式 KV 为 2 GiB，最大上下文与批处理预算为 2048。A 恢复包括 level-2 sleep、weights wake、真实传输、版本提交、KV wake 和 resume。所有接收副本轮后延迟加载引用均为 0。

B 由完整 checkpoint 的输出头词表行循环置换一次构造，两轮传输全部索引张量。各副本完整 head SHA256 与目标 checkpoint 一致，版本分别提交为 B、A-restored。[变体清单](../../variant_manifest.json)

## 有效缓存命中与输出对照

提示词重复次数由 8 增至 24，使 prefix caching 产生实际命中。两种后端、每个副本、两个接口在更新后首次请求 cached_tokens=0，重复请求为 256。B 的冷/命中输出分别对照独立 B 服务的相应基准，恢复 A 的冷/命中输出分别对照初始 A 的相应基准，全部四项比较严格相等。缓存失效由 pause(clear_cache=true) 执行。

| 后端 | 阶段 | 冷请求与缓存命中请求 token IDs 不同 |
|---|---|---|
| hccl | B | 8/8 |
| hccl | A-restored | 5/8 |
| npu_ipc | B | 16/16 |
| npu_ipc | A-restored | 8/16 |

上述冷热差异仍需针对 prefix cache replay、注意力状态及数值路径定位。后续精度验收应继续保存相同提示词、相同副本的冷请求与命中请求，以及对应 logits/logprob 和缓存状态。

## 源码与运行证据

IPC packed 源码 SHA256 为 1f8e4edc8f691bbb4cdbfbe2eef0c37c8163ad8090471eeb3b661e801c8f0b1e。修复为在复用单缓冲的每个 chunk 导出独立 rebuild 参数，接收重建与引用计数逐次对应。原始旧实现 SHA256 为 2bfcde4bdf2fd2cb3638af8ad49cce0736a3c2d01081d147e22cf9d2081780aa。[源码版本](source_versions.json) [引用计数补丁](../../patches/ipc_chunk_reference_counters.patch)

运行前后的测试脚本与当前资源守护脚本保存在 snapshots。单卡测试在每轮执行 GC、ipc_collect、empty_cache 后记录分配量。全量发送端同时保留清理前和清理后的分配记录。[脚本快照](snapshots/) [单卡命令](single_card_fixed_command.json) [全量命令](transfer_command.json)

每个 backend 目录保存 producer.log、server.log、hbm.csv、逐 rank 发送计数、原始请求响应和 receiver JSONL。退出码分别保留为 producer.exit 和单卡 .exit。

## 资源规则与释放

资源守护按北京时间 12:00 后发现其他 NPU 作业时停止本次带有运行标记的进程。服务器时钟相对客户端基准慢 8 小时，守护使用 28800 秒校正和单调时钟计算实际时间。12:00 后记录的其他 NPU 进程样本为 0，让出触发次数为 0。监测日志保留全部原始采样。[资源规则](settings.json) [监测日志](resource_guard.jsonl)

实验已结束，8 张卡均无运行进程，本实验剩余进程为 0。[释放核验](release_verification.json) [设备快照](released_npu_smi.txt)

## 后续问题

原始全量更新 OOM 与 IPC 缓冲滞留已经闭环。当前遗留为本 Top32 配置下冷缓存和缓存命中请求之间的输出差异，其定位与修复需要后续专项验证。

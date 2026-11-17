# HCCL 与 NPU IPC 权重更新定位计划

目标为 177.42.22.1 的 CANN 9.3.0 环境。实验目录为 /mnt/share/c00954197/Workspace/dsv41/basic/weight_update。

## 模型与对照

使用 40 层、32 路由专家、Top-6、关闭 Engram 的 BF16 checkpoint。A 为前一实验准备的模型视图，B 对 head.weight 做一次输出词表行循环置换。两轮更新均传输全部 6248 个张量，共 135927315272 字节。B 使用独立冷加载服务提供同拓扑的双接口数值对照。

## 执行矩阵

| 后端 | 推理拓扑 | 发送端 | 全量更新验收 |
|---|---|---|---|
| HCCL | NPU0–3，TP1/DP4/EP4 | 独立 NPU7 | A→B→A，版本、完整字节数、全头权重哈希、双接口数值 |
| NPU IPC | NPU0–7，TP1/DP8/EP8 | 每卡一个发送进程 | A→B→A，版本、完整字节数、物理卡对应、全头权重哈希、双接口数值 |

B 更新执行 pause(clear_cache=true) 与原生 start/update/finish。A 恢复加入 level-2 sleep、仅唤醒权重、真实传输、唤醒 KV 与 resume。逐副本、逐接口对照各自基准的文本、token IDs、逐 token logprob。

## HBM 定位与规避

最大单张量为 1323827200 字节，约 1.23 GiB。设置 1280 MiB 单 packed buffer、2 GiB 显式 KV cache、2048 上下文与批处理预算、256 MiB HCCL 通信缓冲。发送端逐张量读取 checkpoint，保持原始名称。接收端采集每批张量数、字节数、延迟加载引用量与分配峰值，宿主机按阶段采样全部设备的物理 HBM。

发现失败后保留日志、完整请求、传输进度和内存现场，清理本轮进程后依照证据修复并使用新运行目录复验。源模型文件通过只读挂载引用，B 的唯一实体变体分片在本实验目录生成。

## 参考

PR #53541 的真实权重传输、B 提交、A 恢复、版本与数值对照流程。MindSpeed Agent model-rl-case-guard 的 checkpoint-native 数据源、名称映射、低显存容量与双接口验收方法。

# CANN930 专项验证

2026-11-17 在 `177.42.22.1`、`c00954197_dsv41_cann930` 容器完成 Sleep level-2 与全量权重更新验证。模型为 40 层、32 路由专家、Top-6、BF16，Engram 关闭。[环境与源码版本](../../environment/cann930_environment_20261117.md)

| 专项 | 实机验收结果 | 执行与证据 |
|---|---|---|
| Sleep level-2 | 五轮输出严格一致，8 副本双接口共 80 次恢复后请求，最大单卡 HBM 增量 40 MiB | [报告](sleep_level2/REPORT.md)、[服务脚本](sleep_level2/scripts/start_server.sh)、[客户端](sleep_level2/scripts/verify_sleep2.py)、[五轮输出](sleep_level2/five_cycle_outputs.txt)、[逐卡显存](sleep_level2/memory_five_cycles.csv) |
| HCCL | DP4/EP4 推理与独立发送端完成全量 A→B→A | [报告](weight_update/REPORT.md)、[运行入口](weight_update/scripts/run_host.py)、[逐端计数](weight_update/runs/transfer_20261117_03/hccl_producer/full_transfer_audit.json) |
| NPU IPC | DP8/EP8 与 8 个同置发送进程完成全量 A→B→A | [逐端计数](weight_update/runs/transfer_20261117_03/npu_ipc_producer/full_transfer_audit.json)、[状态](weight_update/validation_status.json) |
| IPC 发送缓冲回收 | 最新补丁通过 CPU 回归，2026-11-18 单卡 20 轮及 8 卡全量复验通过，allocated 为 0 | [补丁](weight_update/patches/ipc_chunk_reference_counters.patch)、[后续步骤](weight_update/pending_validation.md) |

每轮权重更新传输全部 6248 个索引张量，共 135927315272 字节。B 通过输出头词表行置换构造，使用同拓扑独立冷加载 B 对照。恢复 A 的流程包含 sleep level-2、分阶段唤醒、真实权重传输和版本提交。

两个专项分别保存失败复现、修复补丁、CPU 回归日志、服务端命令和输出、原始请求响应及显存记录。实机回环使用的 packed 源码版本与后续 CPU 修复版本在权重报告中列明。

权重实验的保存证据可在 CPU 上重新核对。

```bash
python3 reduced/cann930_20261117/weight_update/scripts/audit_saved_evidence.py
```

2026-11-18 的实机复验补充了有效缓存命中、完整回收与退出告警检查，新增冷/热缓存输出差异观察。[最新报告](weight_update/runs/retest_20261118_01/REPORT.md)

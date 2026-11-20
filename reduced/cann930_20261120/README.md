# CANN930 2026-11-20专项验收

177.42.22.1，40层32专家BF16，Engram关闭。TP>1/DCP/SP/PP按当前模型支持范围排除。

| 专项 | 当前结果 | 报告 |
|---|---|---|
| 路由返回、并发与缓存一致性 | 最终153请求全部通过，双接口cold/hit/sleep结果及专家严格一致，日志全段0 ERROR/Traceback | [路由报告](routing_replay/REPORT.md) |
| 1M长输入、async、chunked prefill、NPU图 | 100万token输入，43 token自然结束，148.02秒，检索3/3 | [长输入报告](long_context/REPORT.md) |
| FP4/FP8导出抽样 | 34张量、4624标量BF16位模式全部一致 | [核验报告](../../model/dequant_validation_20261120/REPORT.md) |

既有[Sleep level-2五轮](../cann930_20261117/sleep_level2/REPORT.md)与[HCCL/IPC全量A→B→A](../cann930_20261117/weight_update/runs/retest_20261118_01/REPORT.md)通过记录继续保留。当前资源策略为外部占用或进程停止即让出，无自动重启；本轮最终正常释放8张卡。

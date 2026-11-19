# CANN930 完整模型验证

2026-11-17 的 12 轮性能压测和 GPQA Diamond 实测见报告。完整模型使用 384 路由专家、Engram 与视觉模块，TP1/PCP8/DP2/EP16，异步调度，CCU_SCHED。

- [正式报告](REPORT.md)
- [原始执行汇总](combined_run/results_summary.json)
- [三场景四轮性能指标](performance_results/)
- [GPQA 原始评分与最终 content 审计](precision_results/)
- [当前脚本入口](scripts/)
- [实测脚本快照](combined_run/snapshots/)
- [Gloo CPU 超时补丁](patches/cpu_timeout_fix.patch)

本轮 GPQA 使用 32768 输出预算，最终 content 评分为 164/198，82.828283%，24 题达到输出上限。当前入口已适配新的模型和 precision_cann930 路径，运行快照保留当时参数。预测及遥测 JSONL 使用 gzip 无损归档。

2026-11-19 的 65536 预算三轮结果已归档，平均准确率为 90.235690%。[最新三轮精度](../cann930_20261119_precision/REPORT.md)

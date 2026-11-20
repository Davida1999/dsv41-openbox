# DSV41 Flash 模型开箱交付

更新日期：2026-11-20。[正式交付报告](DSV41模型开箱交付件.md)按模板组织，统一CANN930环境。完整384专家＋Engram用于性能与GPQA；32专家、Engram关闭用于本轮RL与1M专项。

| 模块 | 当前结论 | 交付入口 |
|---|---|---|
| 基础推理与支持范围 | TP1/PCP8/DP2/EP16及图、async、chunked prefill组合通过；TP>1/DCP/SP/PP排除本轮 | [正式报告](DSV41模型开箱交付件.md) |
| Sleep level-2 | 五轮、80次恢复后请求严格一致；最大单卡稳态增量40 MiB | [Sleep报告](reduced/cann930_20261117/sleep_level2/REPORT.md) |
| HCCL / NPU IPC | 全量A→B→A、OOM及IPC回收已闭环 | [实机报告](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/REPORT.md) |
| 专家返回 / 缓存replay | 新实机验收通过；最终图模式153请求、全日志0 ERROR/Traceback、冷热和sleep/wake严格一致 | [路由报告](reduced/cann930_20261120/routing_replay/REPORT.md) |
| 1M长序列 | 100万token输入、148.02秒、自然输出43 token、检索3/3；async与chunked prefill实测完成 | [1M报告](reduced/cann930_20261120/long_context/REPORT.md) |
| GPQA Diamond | 四次完整198题；当前三轮平均90.235690%，官方参考方法差异及T检验已记录 | [精度与统计](full/gpqa_four_runs/REPORT.md) |
| 反量化补证 | 独立抽样34张量、4624值逐位一致 | [核验报告](model/dequant_validation_20261120/REPORT.md) |

完整模型CANN930三场景各四轮性能已完成；主场景0.8硬件分析仍需双机资源。[环境](environment/cann930_environment_20261117.md) · [本轮总览](reduced/cann930_20261120/README.md) · [归档清单](ARCHIVE.md)

在仓库根目录运行`shasum -a 256 -c manifest.sha256`核对完整性。

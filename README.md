# DSV41 Flash 模型开箱交付

更新日期：2026-11-20。正式交付文档：[DSV41模型开箱交付件](DSV41模型开箱交付件.md)。报告按模型结构、开箱打通、基本功能、RL功能、性能与精度五部分组织，统一CANN930环境。完整模型为384专家且Engram开启，RL专项为32专家且Engram关闭。

| 模块 | 当前结论 | 交付入口 |
|---|---|---|
| 基础推理及功能 | TP1/PCP8/DP2/EP16及图模式、async、chunked prefill组合运行通过；其他并行专项待验证 | [正式报告](DSV41模型开箱交付件.md)、[当前服务脚本](full/cann930_20261119_precision/scripts/)、[保存的请求示例](full/cann930_basic/) |
| Sleep level-2 | 缩减模型5轮、80次恢复后请求严格一致；最大单卡稳态显存增加40 MiB | [专项报告](reduced/cann930_20261117/sleep_level2/REPORT.md) |
| HCCL / NPU IPC | 缩减模型完整A→B→A通过；OOM与IPC缓冲滞留闭环，冷热缓存输出差异待定位 | [最新实机报告](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/REPORT.md) |
| 专家路由返回 | 双请求HTTP500；候选补丁CPU回归通过，NPU修复复验待完成 | [阶段报告](reduced/cann930_20261118/routing_replay/REPORT.md) |
| 性能 | 完整模型三场景各4轮，逐轮64/64成功；0.8硬件目标待验收 | [CANN930性能](full/cann930_20261117/REPORT.md) |
| GPQA Diamond | 四次完整198题；当前65536预算三轮平均90.235690%；官方DeepInfer参考93.4%，方法尚未完全对齐 | [四次结果、官方对照与T检验](full/gpqa_four_runs/REPORT.md) |

精度验收范围为GPQA Diamond。完整模型RL、独立单curl记录、1M有效回答及剩余并行项目见正式报告中的待办。

- [模型结构与反量化](model/模型结构分析.md)
- [CANN930环境](environment/cann930_environment_20261117.md)
- [最新三轮精度原始记录](full/cann930_20261119_precision/REPORT.md)
- [归档清单](ARCHIVE.md)

仓库根目录运行`shasum -a 256 -c manifest.sha256`核对文件完整性；统计重算命令见四次精度报告。

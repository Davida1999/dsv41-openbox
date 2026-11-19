# DSV41 Flash 模型开箱交付

更新日期为 2026-11-19。主交付件见下方链接。[DSV41模型开箱交付件](DSV41模型开箱交付件.md)

本轮验证环境为 `177.42.22.1` 专项及 `177.42.23.1/24.1` 完整模型使用的 `c00954197_dsv41_cann930` 容器，CANN 9.3.0。使用 40 层、32 路由专家、BF16、关闭 Engram 的模型，Sleep level-2 五轮恢复、HCCL 与 NPU IPC 全量 A→B→A 已通过。IPC 发送缓冲引用计数补丁于 2026-11-18 完成单卡 20 轮及 8 卡全量实机复验，分配量回收与退出告警检查通过。当前待定位冷/热缓存请求的输出差异。完整 384 专家模型在 23/24 双节点完成 CANN930 三场景各四轮性能压测和 GPQA 正式评测。2026-11-19 的 GPQA 65536 预算三轮最终 content 分别为 180/198、178/198、178/198，平均准确率为 90.235690%。三轮未解析最终选项分别为 2、1、2 题，均计错。路由、长输入有效回答、Math-500 和 TextVQA 等专项仍待完成。

| 当前专项 | 结论 | 报告 |
|---|---|---|
| Sleep level-2 | 五轮、8 副本、双接口共 80 次恢复后输出严格一致，最大单卡 HBM 增量 40 MiB | [Sleep 报告](reduced/cann930_20261117/sleep_level2/REPORT.md) |
| HCCL 全量权重更新 | 每轮每端 6248 张量、135927315272 字节，B 与冷加载 B 一致，恢复 A 与初始 A 一致 | [权重更新报告](reduced/cann930_20261117/weight_update/REPORT.md) |
| NPU IPC 全量权重更新 | 完整回环和 2026-11-18 回收复验通过，两轮每个发送端 allocated 为 0，冷热缓存输出差异待定位 | [当前状态](reduced/cann930_20261117/weight_update/validation_status.json) |

- [GPQA 65536 预算最新三轮精度](full/cann930_20261119_precision/REPORT.md)
- [CANN930 历史性能与 32768 预算 GPQA](full/cann930_20261117/REPORT.md)
- [模型结构与转换](model/模型结构分析.md)
- [CANN930 当前环境](environment/cann930_environment_20261117.md)
- [本轮实验入口](reduced/cann930_20261117/README.md)
- [1108 历史环境](environment/1108_environment_20261112.md)
- [E23运行脚本](full/E23/scripts/)
- [完整模型单curl](full/E13/results/)
- [基础功能验收矩阵](reduced/basic_feature/VALIDATION_REPORT.md)
- [Profiling对比](profiling/hardware_comparison.md)
- [归档范围](ARCHIVE.md)

E13、E23、E25 及 2026-11-13 基础功能实验保留各自的环境与日期。运行快照中的模型路径和启动参数用于追溯，新的执行入口见 CANN930 环境文档。

在仓库根目录运行 `shasum -a 256 -c manifest.sha256` 校验交付文件。

[2026-11-18 权重更新与 IPC 回收实机报告](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/REPORT.md)

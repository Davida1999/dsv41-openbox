# DSV41 Flash 开箱归档

主交付件：[DSV41模型开箱交付件](DSV41模型开箱交付件.md)。

完整BF16＋Engram的1108/CANN9.2.1推理为部署基线；E13历史完整模型单curl、E25/CANN9.3.0补充性能和32专家专项失败证据分别归档。现状是部分验收完成，不能宣称RL/1M语义/数据集精度全通过。

- [模型结构与转换](model/模型结构分析.md)
- [环境](environment/1108_environment_20261112.md)
- [E23运行脚本](full/E23/scripts/)
- [完整模型单curl](full/E13/results/)
- [裁剪专项](reduced/basic_feature/VALIDATION_REPORT.md)
- [Profiling对比](profiling/hardware_comparison.md)
- [归档范围](ARCHIVE.md)

历史脚本包含当时的IP、容器路径与模型路径，仅在匹配环境复现。归档不自带模型权重或镜像，也没有后台自动启动任务。

校验：`shasum -a 256 -c manifest.sha256`。本仓库保存既有实验材料与本次整理结果；没有将本次静态核验写成新的NPU运行。

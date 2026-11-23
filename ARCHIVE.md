# 归档清单与证据范围

[开箱结果概览](README.md)汇总当前 CANN930 验证结论，[交付报告](DSV41模型开箱交付件.md)记录受测配置、复现入口和验收边界。各验证目录保存支撑结论的脚本、结果和运行快照。

| 目录 | 内容 |
| --- | --- |
| [model/](model/) | 模型配置、权重索引、当前结构图及源码依据、BF16 反量化与独立抽样核验 |
| [environment/](environment/) | CANN930 软件、容器和运行环境 |
| [full/cann930_basic/](full/cann930_basic/) | 基于保存预测重建的请求、实测输出及 curl 示例 |
| [full/cann930_20261117/](full/cann930_20261117/) | 三场景 12 轮性能、首次完整 GPQA 和 8 题预检 |
| [full/cann930_20261119_precision/](full/cann930_20261119_precision/) | 后续三轮 GPQA 的配置、预测、评分和复核脚本 |
| [full/gpqa_four_runs/](full/gpqa_four_runs/) | 四轮 GPQA 统计、官方参考和逐题配对分析 |
| [reduced/cann930_20261117/](reduced/cann930_20261117/) | Sleep level-2、HCCL 与 NPU IPC 权重更新及回收复验 |
| [reduced/cann930_20261120/](reduced/cann930_20261120/) | 专家返回、缓存重放和 1M 输入的最终验收与关键诊断 |

Git 历史保留旧结构图、早期性能基线及 profiling、旧功能归档和后续已复验的路由排查。当前目录中的诊断记录用于解释正式验收报告涉及的失败或修复。

## 服务器资产

完整权重、tokenizer 和镜像二进制由服务器保存。仓库记录配置、索引、版本和运行证据。

- 完整模型位于 `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16`。
- Slim-E32 实体副本位于 `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32`。

## 完整性核对

在仓库根目录运行 `shasum -a 256 -c manifest.sha256`。 [验证记录](validation.json)收录已保存实测结果的离线核验及结构图校验。预测和遥测采用 gzip 无损归档，原始快照保留实测时的路径。

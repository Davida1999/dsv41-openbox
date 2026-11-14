# 归档清单与证据边界

## 工程包含

| 目录 | 内容 | 来源 |
|---|---|---|
| model | 完整模型config、权重索引、转换清单、转换器/历史日志、源码摘录、结构图 | 25服务器共享工作区；不含权重与tokenizer资产 |
| environment | 1108两节点环境快照、CANN930构建报告 | 2026-11-12环境采集与后续构建归档 |
| full/E13 | 完整BF16＋Engram的服务/测试脚本，三组各10次单curl请求/原始响应/统计 | 2026-11-02，27/28节点；历史环境未强行标成1108 |
| full/E23 | 1108服务/客户端脚本，三组各四轮性能JSON、终端、运行快照、日志及退出码 | 2026-11-12，25/26节点 |
| full/E25 | CANN930完整模型脚本、三轮性能终端和环境模型清单 | 独立补充验证，async关闭 |
| reduced/basic_feature | 32专家模型13项功能的配置/脚本/日志/请求/失败与完成记录 | 2026-11-13，26节点1108 |
| profiling | 两轮rank0 kernel CSV、窗口核验、带公式Excel、压缩逐调用JSON、采集脚本 | E23完整模型文本2K→6K，10步decode |
| diagnostics | Gloo定位报告、E23/E25服务日志对比 | 既有诊断记录 |

`manifest.sha256`以本工程根目录为基准。`validation.json`为本次离线归档检查，不是模型运行测试。归档中的历史绝对路径用于追溯，跨机器运行需要替换。

## 保留在本地或服务器的大件

- no_stack rank0 zip：当前聊天本地 `outputs/E23_profile_archives/dp0_pp0_tp0_dcp0_ep0_rank0_161774_20261112115424053_ascend_pt.zip`（若当前目录无zip，则保留同名已解压目录）。
- with_stack rank0 zip：同目录 `dp0_pp0_tp0_dcp0_ep0_rank0_174833_20261112134248332_ascend_pt.zip`。
- 全16rank原始trace与解析数据：服务器 `/workspace/experiments/E23_profile/{no_stack,with_stack}/profiling/`。
- BF16模型权重：共享存储 `/mnt/share/c00954197/workspace/dsv41-flash-bf16`。
- 镜像仅归档版本、ID、环境；未导出镜像二进制。

未将大件缺省为已上传；分析JSON用gzip无损压缩，可用 `gzip -dk hardware_efficiency.json.gz` 解压。Excel中的经验分数不是理论硬件峰值。

## 公开发布检查与改动

- E23 `manifest_text_2k_6k.json` 的历史blocking_reason包含无关其他用户作业标识，公开副本已替换为中性说明；不改变请求数、参数和结果。原文件仍保留在服务器。
- 不包含SSH私钥、公钥、密码、API token、认证配置或模型权重。历史脚本可能引用本机密钥路径，但不携带密钥内容；它们属于复现脚本的历史配置，不是访问凭据。
- 不复制Step5/HY4参考全文和内部wiki截图；章节风格参考，所有实测结论来自DSV41证据。原稿备份保留在仓库外 `outputs/DSV41模型开箱交付件.before.md`。
- 模型及上游源码的使用/分发权利沿用其原授权；本归档不授予模型权重或第三方代码的新许可。
- 历史报告/manifest可能保留当时的prepared或待执行状态，当前判断以原始响应和最终运行结果为准。主文档已注明旧报告中的状态差异。

## 对现有草稿的主要修改

补充模型结构/反量化来源，修正chat接口、4K→32K方向、图片高宽解释及实际KV口径；用JSON生成12轮数值，删除模板示例精度；明确完整模型可替代的基础功能/性能范围，保留RL与1M失败；区分经验评分和理论0.8峰值；增加待开展实验与归档索引。

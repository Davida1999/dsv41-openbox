# CANN930 DSV41 双节点 GPQA 三轮精度测量

三轮评测均已完成，每轮 198 题，共 594 个请求全部成功。三轮使用同一套双节点服务，完成后已核验两台 NPU 资源释放。三轮准确率的算术平均值为 **90.235690%**。

| 轮次 | 正确题数 | 准确率 | 成功请求 | 客户端总耗时 | 达到输出上限题数 | 未解析最终选项题数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 180/198 | 90.909091% | 198/198 | 54 分 20 秒 | 1 | 2 |
| 2 | 178/198 | 89.898990% | 198/198 | 48 分 25 秒 | 0 | 1 |
| 3 | 178/198 | 89.898990% | 198/198 | 61 分 35 秒 | 2 | 2 |

评分使用 AISBench GPQAEvaluator，并以最终 content 提取答案。无最终选项的回答按错误计入 198 题分母。每轮均核对题号 0–197、标准答案、原始 AISBench 分数复算结果及评分差异。表内耗时包含客户端推理、评分和结果审计。服务启动耗时约 28 分 49 秒，单独记录。

| 配置项 | 数值 |
| --- | --- |
| 日期与时区 | 2026-11-19，Asia/Shanghai |
| 主节点 / 子节点 | 177.42.23.1 / 177.42.24.1，每台 8 张 NPU |
| 容器 | c00954197_dsv41_cann930 |
| Docker socket | unix:///run/c00954197-cann930/docker.sock |
| 模型 | /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 |
| 服务配置 | 完整 BF16、Engram CPU offload 和共享内存、TP1 / PCP8 / DP2 / EP16、异步调度、FULL_DECODE_ONLY、CCU_SCHED |
| temperature / top_p / top_k | 1.0 / 0.95 / 0 |
| reasoning_effort / enable_thinking | low / true |
| 输出上限 / 服务上下文上限 | 65536 / 73728 tokens |
| 并发 / seed | 8 / 42 |
| HTTP 请求超时 | 360000 秒 |
| AISBench | 3.1.20261113 |
| Torch / TorchNPU | 2.10.0+cpu / 2.10.0.post7 |
| vLLM / vLLM-Ascend | 0.32.0+empty / 0.19.1rc2.dev3401+g28e9e97ba |

三轮均使用 seed 42，数据集相同，结果反映这一配置下的重复评测。三份实际生成配置已通过审计，运行后六份源脚本及配置的 SHA256 校验通过。数据集 SHA256 为 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。

| 节点 | 引擎初始化次数 | Worker 权重加载完成记录数 | 停止前错误记录数 | 资源清理阶段错误记录数 |
| --- | --- | --- | --- | --- |
| 23 | 1 | 8 | 0 | 0 |
| 24 | 1 | 8 | 0 | 0 |

停止前日志已保存字节位置和 SHA256，资源清理阶段的错误统计及首条记录已写入汇总 JSON，原始日志路径见下表。11 月 17 日曾出现的 UB LINK ERROR 底层根因仍未确认，本次错误统计见上表。

题号均从 0 编号。

第 1 轮的输出上限题号为 `[79]`，最终选项未解析题号为 `[79, 142]`，评分差异题号为 `[]`。预测文件 SHA256 为 `d7f422daff911f101f0554623fa015c79dfcd88c780f5557fef5828bec9d64ce`。
第 2 轮的输出上限题号为 `[]`，最终选项未解析题号为 `[149]`，评分差异题号为 `[]`。预测文件 SHA256 为 `36b31cb9ab88e7848dc6fa77042b6786499533cc26ddcd96fa1c0ace272fde38`。
第 3 轮的输出上限题号为 `[79, 147]`，最终选项未解析题号为 `[79, 147]`，评分差异题号为 `[]`。预测文件 SHA256 为 `cfbb6fdadf66e286e24af5b2b1e24c57637d6589a93fb3694acdf4775bb70694`。

资源释放核验显示，两台本次任务活动进程均为 0，NPU 运行进程均为 0，8000、12346 端口监听均已关闭。两台 /dev/shm 的可用量均已回到启动前基线。具体 HBM 和健康状态见各节点 release_check.json，共享内存复核见 shm_release_review.json。

| 文件 | 完整路径 |
| --- | --- |
| 本报告 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/REPORT.md |
| 汇总与校验数据 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/results_summary.json |
| 三轮控制器 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/run_three_rounds.py |
| 三轮配置审计 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/config_audit.json |
| 第 1 轮 AISBench 结果 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T095717Z_s3im7f/20261119_095718 |
| 第 2 轮 AISBench 结果 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T105138Z_hWw1BM/20261119_105138 |
| 第 3 轮 AISBench 结果 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T114003Z_GWrxTM/20261119_114004 |
| 23 服务日志 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/node23/launcher.log |
| 24 服务日志 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/node24/launcher.log |
| 23 资源释放核验 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/node23/release_check.json |
| 24 资源释放核验 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/node24/release_check.json |
| 23 共享内存复核 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/node23/shm_release_review.json |
| 24 共享内存复核 | /mnt/share/c00954197/Workspace/dsv41/precision_cann930/runs/low_65536_3rounds_20261119_172734/node24/shm_release_review.json |

# CANN930 完整模型 GPQA Diamond 三轮精度报告

2026-11-19 在 177.42.23.1 主节点与 177.42.24.1 子节点完成三轮 GPQA Diamond 测量。每轮使用相同的 198 题，每题生成一次，三轮共 594 个请求全部成功。最终 content 的正确题数依次为 180、178、178，三轮准确率算术平均为 **90.235690%**。

| 轮次 | 正确题数 | 最终 content 准确率 | 成功请求 | 客户端总耗时 | 达输出上限 | 未解析最终选项 |
|---|---|---|---|---|---|---|
| 1 | 180/198 | 90.909091% | 198/198 | 54 分 20 秒 | 1 | 2 |
| 2 | 178/198 | 89.898990% | 198/198 | 48 分 25 秒 | 0 | 1 |
| 3 | 178/198 | 89.898990% | 198/198 | 61 分 35 秒 | 2 | 2 |

评分使用 AISBench GPQAEvaluator，从服务返回的最终 content 提取 `Answer: A/B/C/D`。未解析最终选项的回答按错误计入每轮 198 题分母。三轮 AISBench 原始分数、原始分数复算值与最终 content 分数均一致，评分差异题号均为空。CPU 审计已从保存的预测独立复算全部结果，并核对题号、答案标签、输出预算、配置及预测 SHA256。[独立核对结果](evidence_validation.json) [核对脚本](scripts/audit_saved_results.py)

| 配置项 | 实测设置 |
|---|---|
| 服务节点 | 177.42.23.1 / 177.42.24.1，各 8×Ascend950DT |
| 容器与镜像 | c00954197_dsv41_cann930，镜像 sha256:3614b7afe7000897bf6f1c60a415f2547feb0e033bae0c9bf76c68a76ecf944f |
| Docker socket | unix:///run/c00954197-cann930/docker.sock |
| 模型 | /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 |
| 模型配置 | 完整 BF16、40 层、384 路由专家、Top-6，视觉模块与 Engram 开启 |
| Engram | CPU offload、DP shared memory |
| 并行与执行 | TP1/PCP8/DP2/EP16，异步调度，FULL_DECODE_ONLY，CCU_SCHED |
| 输出预算 / 服务上下文 | 65536 / 73728 tokens |
| temperature / top_p / top_k | 1.0 / 0.95 / 0 |
| 思考配置 | enable_thinking=true，reasoning_effort=low |
| 并发 / seed / warmup | 8 / 42 / 0 |
| 请求尝试 / HTTP 超时 | 每题 1 次 / 360000 秒 |
| 提示词 | AISBench gpqa_gen_0_shot_cot_chat_prompt，0-shot CoT |
| AISBench | 3.1.20261113 |
| Torch / TorchNPU | 2.10.0+cpu / 2.10.0.post7 |
| vLLM / vLLM-Ascend | 0.32.0+empty / 0.19.1rc2.dev3401+g28e9e97ba |

三轮均使用 seed=42，复用同一次模型加载和服务。每个节点引擎初始化 1 次，8 个 worker 均完成权重加载。客户端耗时包含推理、评分及结果审计，服务启动耗时约 28 分 49 秒。三轮共生成 2905546 个 token，最长输入为 2595 token，输入加输出预算为 68131 token。[请求配置](run/request.json) [三轮执行汇总](run/summary.json) [实际生成配置审计](run/config_audit.json)

| 轮次 | 达输出上限的零基题号 | 未解析最终选项的零基题号 | 完整结果 |
|---|---|---|---|
| 1 | 79 | 79、142 | [评分与审计](results/round_1/) |
| 2 | 空 | 149 | [评分与审计](results/round_2/) |
| 3 | 79、147 | 79、147 | [评分与审计](results/round_3/) |

原始预测以 `.jsonl.gz` 无损存放，解压后的 SHA256 与运行汇总逐轮一致。AISBench 配置快照、原始评分明细、汇总表、客户端日志和六份实测脚本已保存。脚本字节与 request.json 记录的 SHA256 一致，数据集 SHA256 为 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。答案标签文件按 AISBench 的选项轮换规则提取，原始数据集使用服务器现有资产。[脚本](scripts/) [数据集校验与标签](dataset_reference.json) [原始汇总](run/results_summary.json)

两端运行期及资源清理阶段的 ERROR 计数均为 0，停止前日志字节位置和 SHA256 已核对。结束后两端本任务进程、NPU 进程和服务端口监听均为 0，`/dev/shm` 可用量回到启动前基线。遥测 JSONL 已采用 gzip 无损归档。[23 节点释放核验](run/node23/release_check.json) [24 节点释放核验](run/node24/release_check.json) [23 共享内存核对](run/node23/shm_release_review.json) [24 共享内存核对](run/node24/shm_release_review.json)

本次结果对应 65536 输出预算、temperature=1.0、top_p=0.95 和 reasoning_effort=low。2026-11-17 的 32768 预算实验保留其原始设置与分数，跨实验差异涉及多项配置变化。精度达标阈值和同条件对照仍待确定。11 月 17 日出现的 UB LINK ERROR 底层根因仍待确认。[32768 预算历史报告](../cann930_20261117/REPORT.md) [本轮原始报告](run/REPORT.md)

在仓库根目录运行以下命令，可使用 Python 标准库重新核对保存证据。

```bash
python3 full/cann930_20261119_precision/scripts/audit_saved_results.py
```

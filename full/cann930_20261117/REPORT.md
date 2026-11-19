# CANN930 完整模型性能与 GPQA 验证报告

2026-11-19 已完成 65536 输出预算的三轮 GPQA，平均准确率为 90.235690%，最新结果见新报告。以下保留 2026-11-17 的实际参数与结果。[最新三轮精度](../cann930_20261119_precision/REPORT.md)

2026-11-17 在 177.42.23.1 主节点和 177.42.24.1 子节点完成。两端使用 `c00954197_dsv41_cann930`，完整 BF16、40 层、384 路由专家、Top-6，视觉模块与 Engram 开启。同一次服务启动依次完成三场景各四轮性能压测、8 题预检和 198 题 GPQA 正式评测。

## 环境与执行配置

| 项目 | 实测配置 |
|---|---|
| CANN / 驱动 | 9.3.0 / 26.2.0.b020 |
| Python / Torch / TorchNPU | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 |
| vLLM / vLLM-Ascend 提交 | 5df63a7203cdb9328d824a27b682e77c7b9928be / 28e9e97ba254b076a88275e8a275556cb9a11664 |
| 并行 | TP1 / PCP8 / DP2 / EP16，每节点 8 卡 |
| 调度与图 | async-scheduling / FULL_DECODE_ONLY |
| 通信 | CCU_SCHED，HCCL enp35s0f2，Gloo enp39s0f0 |
| 上下文 / batched tokens / 每 DP 最大序列数 | 40960 / 2048 / 32 |
| Engram | CPU offload、DP shared memory |
| AISBench | 3.1.20261113，提交 2c7434077ddacf23a55e4eb595cfd4f59244c00e |

两端 Gloo CPU 超时透传修复随补丁归档。停止前日志审计及两端资源释放记录均保留。[补丁](patches/cpu_timeout_fix.patch) [请求与配置](combined_run/request.json) [环境与服务日志](combined_run/) [释放记录](combined_run/release_check.json)

## 三场景四轮性能

每轮 64 请求、并发 64，seed=42、request-rate=10000、ignore-eos。图文输入每请求一张高 1920×宽 1080 的图片。输入长度为文本目标，实际输入包含聊天模板和图像 token。服务启用 prefix caching，12 轮顺序复用缓存。

| 场景 | 轮次 | 成功/失败 | 总输出 token | 耗时 s | 输出 token/s | 平均 TTFT ms | 平均 TPOT ms |
|---|---|---|---|---|---|---|---|
| mm_4k_32k | [1](performance_results/mm_4k_32k/20261116T205522Z_zRmtCa/benchmark.json) | 64/0 | 2097152 | 941.099 | 2228.407 | 19224.264 | 28.101 |
| mm_4k_32k | [2](performance_results/mm_4k_32k/20261116T211122Z_d05oZn/benchmark.json) | 64/0 | 2097152 | 934.647 | 2243.790 | 9596.589 | 28.207 |
| mm_4k_32k | [3](performance_results/mm_4k_32k/20261116T212716Z_WNFjMZ/benchmark.json) | 64/0 | 2097152 | 944.969 | 2219.281 | 8417.828 | 28.564 |
| mm_4k_32k | [4](performance_results/mm_4k_32k/20261116T214320Z_Cqk0Xa/benchmark.json) | 64/0 | 2097152 | 940.154 | 2230.647 | 7982.875 | 28.431 |
| text_2k_6k | [1](performance_results/text_2k_6k/20261116T215920Z_yLSInZ/benchmark.json) | 64/0 | 393216 | 182.020 | 2160.291 | 4865.748 | 28.775 |
| text_2k_6k | [2](performance_results/text_2k_6k/20261116T220236Z_yz1k2a/benchmark.json) | 64/0 | 393216 | 176.355 | 2229.686 | 2062.710 | 28.337 |
| text_2k_6k | [3](performance_results/text_2k_6k/20261116T220547Z_CUc2vU/benchmark.json) | 64/0 | 393216 | 176.542 | 2227.319 | 1270.352 | 28.517 |
| text_2k_6k | [4](performance_results/text_2k_6k/20261116T220857Z_9uRr5a/benchmark.json) | 64/0 | 393216 | 175.986 | 2234.358 | 1159.127 | 28.450 |
| mm_1k_2k | [1](performance_results/mm_1k_2k/20261116T221208Z_scB8wm/benchmark.json) | 64/0 | 131072 | 70.928 | 1847.958 | 8912.948 | 30.046 |
| mm_1k_2k | [2](performance_results/mm_1k_2k/20261116T221333Z_YHqkFW/benchmark.json) | 64/0 | 131072 | 73.521 | 1782.774 | 6931.700 | 32.280 |
| mm_1k_2k | [3](performance_results/mm_1k_2k/20261116T221502Z_hjwCFl/benchmark.json) | 64/0 | 131072 | 74.936 | 1749.108 | 7609.830 | 32.610 |
| mm_1k_2k | [4](performance_results/mm_1k_2k/20261116T221632Z_MyA2B0/benchmark.json) | 64/0 | 131072 | 65.632 | 1997.070 | 5894.833 | 28.910 |

四轮平均输出吞吐分别为图文 4K→32K 的 2230.531 token/s、文本 2K→6K 的 2212.914 token/s、图文 1K→2K 的 1844.228 token/s。12 轮均成功 64、失败 0，完整输出总量逐轮核对通过。

## GPQA Diamond 精度

实测参数为 temperature=0、top_p=1、seed=42、thinking=true、reasoning_effort=100、max_out_len=32768、并发 8。198 题最长输入 2595 token，输入加输出预算为 35363 token。

| 阶段 | 请求成功 | AISBench 原始评分 | 最终 content 评分 | 达输出上限 / 无最终答案 |
|---|---|---|---|---|
| 8 题预检 | 8/8 | 7/8，87.500000% | 7/8，87.500000% | 1 / 1 |
| 198 题正式 | 198/198 | 165/198，83.333333% | 164/198，82.828283% | 24 / 24 |

零基题号 52 的最终 content 为空，AISBench 从推理文本提取答案 D 并计对。最终 content 口径将其计错，两种口径差 0.505051 个百分点。24 题达到 32768 输出上限，均记录为无法提取最终答案。报告保留两种评分与全部题号。[8 题审计](precision_results/gpqa_8/gpqa_audit.json) [198 题审计](precision_results/gpqa_198/gpqa_audit.json) [评分差异](precision_results/gpqa_198/scoring_difference.json)

原始两个客户端审计退出码均为 1，分别涉及 JSONL 中 U+2028 的读取问题和空最终答案的评分差异。修正后复用已生成预测进行 CPU 复核，两轮最终校验退出码均为 0。整体精度达标阈值等待确定。[执行汇总](combined_run/results_summary.json) [修正后的审计脚本](scripts/check_results.py)

本历史报告对应 32768 预算及当时的实际执行配置。2026-11-19 完成的 65536 预算三轮结果在最新报告中记录。

## 运行快照与当前入口

实测模型路径为 `/mnt/share/c00954197/workspace/dsv41-flash-bf16`，precision 工作目录在运行后更名为 precision_cann930。模型已移动到 `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16`。原始快照保留当时路径，当前入口基于这些快照更新模型和工作目录路径，Shell/Python 静态检查结果保存在仓库交付校验中。[当前脚本](scripts/) [实测启动快照](combined_run/snapshots/)

当前入口保留 40960 服务上下文与 32768 GPQA 输出预算。主、子节点进入 CANN930 容器后分别运行 run_master.sh 和 run_worker.sh，客户端按相同顺序串行执行三场景各四轮，再执行 GPQA 8 题和 198 题。

预测与遥测 JSONL 以 gzip 无损保存，解压后的预测 SHA256 可对照原始汇总。数据集保留来源、提示词和校验信息，使用服务器已有 GPQA Diamond 资产。[原始组合实验报告](combined_run/REPORT.md)

## 日志边界

停止前两端服务日志 ERROR 和 Traceback 计数为 0。STOP 后主节点引擎清理期间记录 EngineDeadError，两端最终 remaining_owned 为空，16 张卡和服务端口均已释放。两节点系统时钟约差 10.5 分钟，持续时间采用客户端单调计时。[停服日志审计](combined_run/shutdown_log_review.json) [资源释放](combined_run/release_check.json)

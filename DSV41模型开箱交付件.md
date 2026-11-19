# DSV41 模型开箱交付件

更新日期为 2026-11-19。本轮使用 **c00954197_dsv41_cann930 容器、CANN 9.3.0**。177.42.22.1 的 8 卡专项使用 40 层、32 路由专家、BF16、关闭 Engram 的模型。177.42.23.1/24.1 双机 16 卡性能与精度验证使用完整 384 专家 BF16 模型，Engram 和视觉模块开启。

Sleep level-2 修复后完成五轮恢复验证，8 个副本的双接口共 80 次恢复后请求与首次加载基准严格一致。HCCL 和 NPU IPC 均完成缩减模型全量 A→B→A 更新。IPC 发送缓冲引用计数补丁于 2026-11-18 完成单卡 20 轮和 8 卡全量 A→B→A 实机复验，两轮发送端 allocated 均为 0，退出共享张量告警为 0。有效缓存失效计数和同缓存状态下的数值对照通过，冷缓存与缓存命中请求的输出差异仍待定位。完整模型在 2026-11-17 完成三场景各四轮性能和 32768 预算 GPQA。2026-11-19 使用 65536 输出预算、reasoning_effort=low、temperature=1.0 完成 GPQA 三轮，每轮 198 题，正确数分别为 180、178、178，平均准确率为 90.235690%。三轮全部 594 请求成功，未解析最终选项分别为 2、1、2 题。其余基础功能保留各自的实测环境与日期，整体开箱验收仍有待完成项。

## 验收总览

“已验证”仅指表内受测配置；“部分”表示通路或有限检查完成；“失败/阻塞”保留原始证据；“待开展”没有足够实测证据。完整模型正常推理不能替代未执行的专项动作。

| 验收项 | 当前结论 | 主证据 / 替代范围 |
|---|---|---|
| 完整 BF16＋Engram 基础推理 | CANN930 三场景 12 轮性能及 GPQA 请求完成，E13 保留双接口可读样例 | 完整模型与 32 专家专项分别记录，GPQA 评分及预算命中见精度章节 |
| EP / DP / PCP | CANN930 完整模型 TP1/PCP8/DP2/EP16 组合实测完成 | 三场景 12 轮性能与 GPQA 请求，其他切分档位待专项验证 |
| TP>1 / DCP / SP | TP2失败；DCP/SP前置阻塞 | 1108裁剪32专家专项保留，不能用完整TP1替代 |
| PP | PP2裁剪模型失败；完整Engram基线PP1 | 不标记PP功能通过 |
| 图模式 / async / chunked prefill | CANN930 完整模型 FULL_DECODE_ONLY、async、chunked prefill 组合通过运行验证 | 同一次双节点服务执行性能与 GPQA，独立开关精度对照待验证 |
| 2K文本→6K、1K文本＋图片→2K、4K文本＋图片→32K | CANN930 各四轮 64 成功、0 失败，输出长度完整 | 2026-11-17 双节点完整模型三场景 12 轮，历史 E23/1108 数据分列 |
| sleep level-1 | 1108 历史裁剪模型五轮恢复一致性通过 | 历史配置与结果保留在基础功能报告 |
| sleep level-2 | CANN930 缩减模型五轮通过，双接口文本/token/logprob 全部严格一致 | 80 次恢复后请求，最大单卡 HBM 增量 40 MiB，首轮后最大增长 1 MiB |
| 专家路由返回 / replay | 裁剪模型双请求断言/HTTP500 | 完整模型待验证；不能由普通MoE推理推出replay支持 |
| Checkpoint / HCCL / NPU IPC更新 | CANN930 缩减模型 checkpoint 恢复与两种全量 A→B→A 均通过 | 每轮每端 6248 张量、135927315272 字节，完整 head 哈希和逐副本双接口数值一致；IPC 回收补丁已于 2026-11-18 实机通过，冷热缓存输出差异待定位 |
| 1M输入 | 裁剪模型容量通过，检索0/3 | 不能替代完整模型1M可读回答与精度 |
| 0.8硬件目标 | 单卡decode经验评分通过；理论峰值口径未证明 | E23文本2K→6K profile，不外推4K图文→32K全阶段或16卡 |
| GPQA Diamond | CANN930 65536 预算三轮完成，594/594 请求成功，平均准确率 90.235690% | 每轮 180/198、178/198、178/198，未解析最终选项 2/1/2，原始评分与最终 content 一致，精度阈值待确定 |
| Math-500 / TextVQA | 待开展 | 正式评分、同条件多轮对照及 T 检验等待完成 |

当前 Sleep、权重更新和完整模型性能精度证据均已归档，原始失败运行继续保留。[最新三轮精度](full/cann930_20261119_precision/REPORT.md) [CANN930 历史性能](full/cann930_20261117/REPORT.md)

E13 使用 27/28 节点，E23 使用 1108/CANN9.2.1，E25 使用 CANN9.3.0。其余基础功能在当前 CANN930 环境的复验仍待资源安排。[本轮专项入口](reduced/cann930_20261117/README.md) [基础功能矩阵](reduced/basic_feature/VALIDATION_REPORT.md)

# 模型结构分析

[模型结构图与资产分析](model/模型结构分析.md)包含40层文本主干、384专家Top-6、32层视觉编码器、Engram层[1,14]、A5 KV布局与权重来源。配置与索引均已归档；模型权重和tokenizer资产不上传GitHub。

本资产由FP8/FP4源权重反量化获得。转换器、执行脚本、525分片/48496张量完成日志均见[转换归档](model/conversion/)。没有把旧“尚未开始转换”的准备文档当成最终状态。

当前完整 BF16 模型路径为 `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16`。32 专家实体副本路径为 `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32`，共 503 个分片、6248 个有效张量，约 135.9 GB（126.6 GiB），Engram 关闭且残留数据已剔除。实测运行快照保留实验当时的 checkpoint 路径。

# 模型开箱打通

## 环境配置

| 组件 | 版本／配置 |
| --- | --- |
| 服务器／容器 | 专项 `177.42.22.1`，完整模型 `177.42.23.1/24.1`，均使用 `c00954197_dsv41_cann930` |
| 宿主机系统／架构 | openEuler，Linux 6.6.0-145.3.35.166.oe2403sp3，aarch64 |
| NPU | 每节点 8×Ascend950DT，单卡 HBM 96 GiB，完整模型使用双节点 16 卡 |
| CANN toolkit | `9.3.0`，`V100R001C25B114` |
| Python | `3.12.16` |
| PyTorch | `2.10.0+cpu`，通过 TorchNPU 支持 NPU |
| TorchNPU | `2.10.0.post7` |
| vLLM | `0.32.0+empty`，commit点： `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | `0.19.1rc2.dev3401+g28e9e97ba`，commit点： `28e9e97ba254b076a88275e8a275556cb9a11664` |
| Transformers | `5.19.0` |
| Docker 网络／IPC | `host`／`host` |
| Docker socket | `/run/c00954197-cann930/docker.sock` |
| Gloo 接口 | `enp39s0f0` |
| HCCL 接口 | `enp35s0f2` |

镜像为 `c00954197/vllm-ascend:1108-cann930-20261113`，镜像 ID 为 `sha256:3614b7afe7000897bf6f1c60a415f2547feb0e033bae0c9bf76c68a76ecf944f`。实际源码在上述提交上包含本轮修复补丁，具体文件哈希与运行时状态见环境记录。[CANN930 环境报告](environment/cann930_environment_20261117.md)

1108 两节点环境报告与原始快照用于解释 E23 历史性能。原报告中的 KV BF16 为 CLI 配置，实际 packed cache 布局见模型结构分析。[历史环境](environment/1108_environment_20261112.md) [历史快照](environment/environment_snapshot.json)

本轮修复涉及 RMSNorm 常量 buffer、attention sink 的 layerwise 加载，以及运行时专家通信映射的加载配额。三项已完成实机验证。IPC 逐 chunk 导出 rebuild 参数的后续补丁已于 2026-11-18 完成单卡和 8 卡全量实机验证。[Sleep 补丁](reduced/cann930_20261117/sleep_level2/patches/) [权重更新补丁](reduced/cann930_20261117/weight_update/patches/)

## 当前 GPQA 服务与客户端

主节点为 177.42.23.1，子节点为 177.42.24.1，使用 TP1/PCP8/DP2/EP16、async、FULL_DECODE_ONLY 和 CCU_SCHED。进入容器的命令如下。

```bash
docker -H unix:///run/c00954197-cann930/docker.sock exec -it c00954197_dsv41_cann930 bash -l
```

当前精度实验使用 `/mnt/share/c00954197/Workspace/dsv41/precision_cann930` 中的服务端与客户端，模型路径为 `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16`。主、子节点完成一次模型加载后，在主节点串行执行三轮 GPQA，每轮 198 题。六份实测脚本及配置的 SHA256 与运行记录一致。[主节点](full/cann930_20261119_precision/scripts/run_master.sh) [子节点](full/cann930_20261119_precision/scripts/run_worker.sh) [客户端与审计](full/cann930_20261119_precision/scripts/) [实际配置审计](full/cann930_20261119_precision/run/config_audit.json)

本轮 GPQA 服务上下文为 73728，batched tokens 为 2048，每 DP 最大序列数 32，显存比例 0.96。客户端使用 65536 输出预算、temperature=1.0、top_p=0.95、top_k=0、enable_thinking=true、reasoning_effort=low、seed=42、并发 8，HTTP 超时 360000 秒。

## 完整模型历史服务与客户端脚本

- [主节点脚本](full/E23/scripts/run_master_mm.sh)、[子节点脚本](full/E23/scripts/run_worker_mm.sh)。保持历史原文件；执行前按实际机器核对IP、网卡和模型路径。
- [纯文本2K→6K](full/E23/scripts/run_client_text_2k_6k.sh)、[图文1K→2K](full/E23/scripts/run_client_mm_1k_2k.sh)、[图文4K→32K](full/E23/scripts/run_client_mm_4k_32k.sh)。
- 各正式轮次的实际服务/客户端快照和退出码保存在 `full/E23/<场景>/run_archive/`，优先以运行快照解释结果；当前公共脚本不是历史每轮唯一事实源。

在两端1108容器各开终端，使用实验目录内原脚本：

```bash
# 25主节点，容器内
bash /workspace/experiments/tp1_dp16_20261029/E23/run_master_mm.sh
# 26子节点，容器内另一个终端
bash /workspace/experiments/tp1_dp16_20261029/E23/run_worker_mm.sh
# 等两端ready、模型名正确后，在25执行客户端
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
bash /workspace/experiments/tp1_dp16_20261029/E23/run_client_mm_4k_32k.sh
```

健康接口200只表示API可响应；还需确认两端worker就绪、实际请求输出增长和最终完整长度。

| 参数 | 基线配置 |
|---|---|
| 模型 / served name | 完整 `dsv41-flash-bf16` / `dsv41-bf16-tp1` |
| 并行 | TP1 / PCP8 / DP2；每节点local DP1，子节点start rank1；EP开启 |
| API / DP RPC | 主节点8000 / 12346；单API server；子节点headless |
| 权重 / Engram | BF16，quantization_config=null；cpu_offload=true、dp_shared_memory=true |
| 上下文 / batch预算 | 40960 / 2048 tokens |
| 每实例最大序列 | 32；两DP实例合计64，客户端并发64 |
| KV CLI / block / 利用率参数 | bfloat16 / 128 / 0.96；packed物理布局见模型分析 |
| 图与调度 | FULL_DECODE_ONLY、async scheduling、prefix caching、chunked prefill开启 |
| 多模态 | image=1，video=0 |
| Ascend附加参数 | weight_nz_mode=0，enable_cpu_binding=true，enable_fused_mc2=1，enable_shared_expert_dp=true；npugraph_ex/static_kernel=false |
| Gloo / HCCL/IP | enp39s0f0 / enp35s0f2；HCCL_IF_IP按本节点填写 |

## 完整模型单curl无乱码与正确性样例

原始E13在27主节点、28子节点运行，最大上下文10240、image=0，其余关键模式为TP1/PCP8/DP2、Engram、FULL_DECODE_ONLY、async、chunked prefill。它提供历史功能证据，不宣称使用1108版本复测过相同30条请求。

基础数学请求（在当前服务本机运行，历史地址为27:8000）：

```bash
curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-bf16-tp1","prompt":"计算 123 + 456。只输出整数答案，不要解释。\n答案：","max_tokens":64,"temperature":0,"top_p":1,"top_k":-1,"seed":42,"logprobs":5,"stream":false,"n":1}'
```

返回 `579`。10/10完成，0失败，文本、token与API返回的logprobs一致。[原始请求与响应](full/E13/results/single_curl_math_results/20261102T060812Z_erstg3gs/)、[逐次报告](full/E13/results/single_curl_math_results/20261102T060812Z_erstg3gs/report.txt)。

复杂业务请求使用 **chat接口**，`messages`、`logprobs=true`、`top_logprobs=5`；不能发到completions接口：

```bash
# 在本归档工程根目录执行；请求正文保持原样
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @full/E13/results/single_curl_business_math_results/20261102T062927Z_3lwq6m4g/request.json
```

10/10完成且各项核算正确，最终JSON为：

```json
{"unit_cost":90,"revenue":27900,"sold_cost":18900,"damaged_loss":1080,"ending_units":18,"ending_inventory":1620,"commission":837,"delivery":1680,"profit":5403,"net_cash_flow":3783}
```

[完整报告](full/E13/results/single_curl_business_math_results/20261102T062927Z_3lwq6m4g/report.txt)、[summary](full/E13/results/single_curl_business_math_results/20261102T062927Z_3lwq6m4g/summary.json)。原始输出包含思考文本，不能把人工摘出的最终JSON当成模型仅输出JSON的证据。

第三组活动票务核算10/10完成且数值正确，但不同轮次生成文本与logprobs不完全一致：[原始结果](full/E13/results/single_curl_event_math_results/20261102T064156Z_lkysf0bb/report.txt)。因此不宣称temperature=0即可保证全局确定性，也不宣称跨DP/批大小一致性已验收。

# 基本功能验证

## 并行切分TP/EP、DP

完整模型E13和E23的受测组合为 `--tensor-parallel-size 1 --prefill-context-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel`。使用完整模型无乱码样例及三组性能作为EP/DP组合的通过证据。TP>1不能从此推导。

1108裁剪模型TP2/DP4预热失败：`mixed_quant_sparse_flash_mla`要求N1=64，而TP2为32。原始[TP2日志](reduced/basic_feature/results/node26_F03_tp2_20261113T095042/server.log)。限制指所测版本与A5实现，不对未来版本或其他平台作永久不支持声明。

## CP功能

完整PCP8组合已验证；无需将裁剪PCP2的退化输出当成完整模型质量证据。裁剪PCP1/PCP2在32768输入ID相同条件下均完成，但6/6输出不同。所规划DCP2依赖TP2，未执行，记为阻塞。完整模型PCP1/2/8之间严格数值等价尚未验收。

## SP功能

裁剪模型受TP>1前置限制，未执行有效SP专项；TP1下打开无效开关不能算通过。[配置](reduced/basic_feature/configs/cases/F05_sp.json)。待验证配置为 `--compilation-config '{"pass_config":{"enable_sp":true}}'`，需在满足 TP 前置条件后实测。

## PP功能

当前完整Engram路径使用PP1；裁剪模型关闭Engram后实测PP2仍失败：`hidden_states`为None时调用`unsqueeze`，另一stage报`PPMissingLayer`缺少`layer_idx`。[原始日志](reduced/basic_feature/results/node26_F06_pp2_20261113T115905/server.log)。原完成记录曾误报让卡，报告已根据异常堆栈更正；保留原文件不覆盖。

## 图模式、异步调度、chunked prefill

完整模型脚本显式开启 `FULL_DECODE_ONLY`、`--async-scheduling`、`--enable-chunked-prefill`，E13可读输出与E23正式性能证明组合运行。长输入超过2048调度预算且请求完成，支持分块路径已工作的判断；不伪造逐步分块计数。

裁剪专项保留独立消融：graph相对eager单请求4/4一致、并发1/2一致；async与同步6/6 token一致；chunk与非chunk的6/6输出不同。完整模型组合成功不能把这些专项结果改写为完整模型逐项精度一致。图模式结论仅限decode图，不扩展至所有图模式或prefill图。

# RL功能验证

本轮 Sleep level-2、checkpoint、HCCL 与 NPU IPC 专项使用 CANN930 容器、40 层 32 专家 BF16 模型，Engram 关闭。2026-11-13 的 1108 基础功能记录继续作为其他专项的历史证据。[本轮实验](reduced/cann930_20261117/README.md) [功能矩阵](reduced/basic_feature/VALIDATION_REPORT.md) [历史执行汇总](reduced/basic_feature/results/EXECUTION_SUMMARY_20261113.json)

## sleep mode功能精度验证

**CANN930 缩减模型 Sleep level-2 五轮验收通过。** 按 sleep2→wake(weights)→checkpoint reload→wake(kv_cache) 执行，逐副本定向发送 temperature=0 的单请求。五轮共 80 次恢复后请求，文本、token IDs 和逐 token logprob 与首次加载基准完全一致，最大 logprob 差值为 0。

故障定位到 attention sink 直接 `param.data.copy_` 绕过 layerwise weight_loader，以及无学习权重 RMSNorm 常量未注册为 buffer。修复后两组共 6 项 CPU 回归通过，并完成 TP1/DP8/EP8 的实机五轮验证。

| 轮次 | 一致请求 | sleep 前 HBM | sleep 后 HBM | wakeup 后 HBM | 请求后稳态 HBM |
|---|---|---|---|---|---|
| 1 | 16/16 | 59030–59395 | 8100–8436 | 58996–59341 | 59070–59375 |
| 2 | 16/16 | 59070–59375 | 8140–8436 | 59036–59341 | 59070–59375 |
| 3 | 16/16 | 59070–59375 | 8140–8436 | 59036–59341 | 59070–59375 |
| 4 | 16/16 | 59070–59375 | 8140–8437 | 59036–59341 | 59070–59375 |
| 5 | 16/16 | 59070–59375 | 8174–8465 | 59036–59341 | 59070–59375 |

单位为 MiB，区间覆盖 8 张卡。相对加载基准的最大单卡稳态增量为 40 MiB，首轮恢复后各轮最大增长为 1 MiB。裁剪模型的基准输出仍存在质量限制，完整 384 专家＋Engram 的五轮恢复验证等待资源安排。[完整报告](reduced/cann930_20261117/sleep_level2/REPORT.md) [五轮输出](reduced/cann930_20261117/sleep_level2/five_cycle_outputs.txt) [逐卡显存](reduced/cann930_20261117/sleep_level2/memory_five_cycles.csv) [服务脚本](reduced/cann930_20261117/sleep_level2/scripts/start_server.sh)

1108 环境中 level-1 的五轮一致性记录，以及 level-2 的首轮输出差异继续保留，作为修复前证据。[历史 level-1](reduced/basic_feature/results/node26_F11_sleep1_20261113T100448/) [历史 level-2](reduced/basic_feature/results/node26_F12_sleep2_repro_20261113T102258/)

## routing replay功能验证

1108 裁剪模型启用返回专家后，AuxOutput要求prefix caching；修正后单请求HTTP200，但响应未及时落盘，双请求触发 `unexpected topk_ids batch dim 2` 与HTTP500，不能判专家内容校验通过。[日志](reduced/basic_feature/results/node26_F09_routing_20261113T095702/server.log)。训练侧强制路由replay也未验证。完整MoE正常推理不能替代此项。

## 权重更新功能验证

**CANN930 缩减模型 HCCL 与 NPU IPC 全量 A→B→A 均通过。** 每轮每个发送端及每个接收副本均记录 6248 张量、135927315272 字节。B 对输出头词表行循环置换一次，由同拓扑独立冷加载 B 提供数值对照。版本提交为 `B`、`A-restored`，所有副本完整 head 哈希和双接口文本/token/logprob 均与对应基准严格一致。

OOM 根因为运行时通信映射 `expert_ids_per_ep_rank` 被计入 checkpoint 加载配额，EP4 每层实际加载 283115520 个元素，预期配额多出 32，逐层暂存输入无法释放。修复将该运行时字段加入跳过加载集合，最大采样延迟张量量从失败时约 44.32 GiB 降至 HCCL 2.043457 GiB、IPC 2.087402 GiB，轮后均归零。

| 路径 | 拓扑与验收 | 采样 HBM 与状态 |
|---|---|---|
| checkpoint reload | Sleep level-2 五轮，逐副本双接口恢复一致 | 最大单卡稳态增量 40 MiB |
| HCCL | DP4/EP4 推理 NPU0–3，独立发送端 NPU7，完整 A→B→A | 更新阶段接收卡峰值 61552 MiB |
| NPU IPC | DP8/EP8 与 8 个同置发送进程，完整 A→B→A | 更新阶段接收卡峰值 44611 MiB，IPC 回收补丁已通过 8 卡实机复验 |

2026-11-18 复验的全部逐端计数、分配量、缓存状态与资源记录见最新报告。[实机闭环报告](reduced/cann930_20261117/weight_update/runs/retest_20261118_01/REPORT.md)

显式 KV 为 2 GiB，单 packed buffer 为 1280 MiB，HCCL_BUFFSIZE 为 256 MiB。最大单张量为 1262.5 MiB，当前打包器要求单缓冲可容纳该张量。[完整报告](reduced/cann930_20261117/weight_update/REPORT.md) [HCCL 逐端审计](reduced/cann930_20261117/weight_update/runs/transfer_20261117_03/hccl_producer/full_transfer_audit.json) [IPC 逐端审计](reduced/cann930_20261117/weight_update/runs/transfer_20261117_03/npu_ipc_producer/full_transfer_audit.json)

后续审计发现旧 IPC 实现各发送进程的分配量在 B 轮后为 1.25 GiB、恢复 A 后为 2.5 GiB，退出共出现 8 条共享张量释放告警。逐 chunk 导出新 rebuild 参数的补丁通过新增 5 项 CPU 回归和仓库 106 项 CPU 测试。2026-11-18 单卡修复前 20 轮累计增长 160 MiB，修复后增长为 0；8 卡全量两轮各发送端 allocated 在显式清理前后均为 0，退出共享张量告警为 0。长提示词补测中，更新后首次 cached_tokens 为 0，重复请求为 256；冷/命中请求各自与同缓存状态基准严格一致，冷热状态之间存在生成差异。[验证状态](reduced/cann930_20261117/weight_update/validation_status.json) [后续复验步骤](reduced/cann930_20261117/weight_update/pending_validation.md)

1108 的更新 OOM 和本轮 CANN930 修复前失败日志继续保留。完整 384 专家模型的更新验证需要独立容量预算和实机结果。[1108 HCCL](reduced/basic_feature/results/node26_F13_hccl_20261113T112758/) [1108 IPC](reduced/basic_feature/results/node26_F13_ipc_20261113T115009/) [本轮失败现场](reduced/cann930_20261117/weight_update/diagnostics/oom_reproduction.json)

## 1M长序列功能验证

裁剪模型实现了1000000输入token＋1024输出token，耗时305.99秒，但5%/50%/95%三个位置检索命中0/3，输出重复。只能标记容量通路通过，不能写成“1M精度/有效回答通过”。[原始结果](reduced/basic_feature/results/node26_F14_long_20261113T120549/long_1000000.json)。

完整模型E23上下文40960、4K→32K长输出不等于1M长输入。完整模型1M测试待开展，需保存输入文本版本、真实 tokenizer 计数、请求与输出、内存和检索答案。

# 性能与精度验证

## CANN930 正式性能结果

2026-11-17 的完整模型服务在 23/24 双节点连续运行，按图文 4K→32K、文本 2K→6K、图文 1K→2K 顺序各执行四轮。每轮 64 个请求、并发 64，输出完整。图文图片为高 1920×宽 1080，prefix caching 开启，轮次间复用缓存。

| 场景 | 轮次 | 成功/失败 | 总输出 token | 耗时 s | 输出 token/s | 平均 TTFT ms | 平均 TPOT ms |
|---|---|---|---|---|---|---|---|
| mm_4k_32k | [1](full/cann930_20261117/performance_results/mm_4k_32k/20261116T205522Z_zRmtCa/benchmark.json) | 64/0 | 2097152 | 941.099 | 2228.407 | 19224.264 | 28.101 |
| mm_4k_32k | [2](full/cann930_20261117/performance_results/mm_4k_32k/20261116T211122Z_d05oZn/benchmark.json) | 64/0 | 2097152 | 934.647 | 2243.790 | 9596.589 | 28.207 |
| mm_4k_32k | [3](full/cann930_20261117/performance_results/mm_4k_32k/20261116T212716Z_WNFjMZ/benchmark.json) | 64/0 | 2097152 | 944.969 | 2219.281 | 8417.828 | 28.564 |
| mm_4k_32k | [4](full/cann930_20261117/performance_results/mm_4k_32k/20261116T214320Z_Cqk0Xa/benchmark.json) | 64/0 | 2097152 | 940.154 | 2230.647 | 7982.875 | 28.431 |
| text_2k_6k | [1](full/cann930_20261117/performance_results/text_2k_6k/20261116T215920Z_yLSInZ/benchmark.json) | 64/0 | 393216 | 182.020 | 2160.291 | 4865.748 | 28.775 |
| text_2k_6k | [2](full/cann930_20261117/performance_results/text_2k_6k/20261116T220236Z_yz1k2a/benchmark.json) | 64/0 | 393216 | 176.355 | 2229.686 | 2062.710 | 28.337 |
| text_2k_6k | [3](full/cann930_20261117/performance_results/text_2k_6k/20261116T220547Z_CUc2vU/benchmark.json) | 64/0 | 393216 | 176.542 | 2227.319 | 1270.352 | 28.517 |
| text_2k_6k | [4](full/cann930_20261117/performance_results/text_2k_6k/20261116T220857Z_9uRr5a/benchmark.json) | 64/0 | 393216 | 175.986 | 2234.358 | 1159.127 | 28.450 |
| mm_1k_2k | [1](full/cann930_20261117/performance_results/mm_1k_2k/20261116T221208Z_scB8wm/benchmark.json) | 64/0 | 131072 | 70.928 | 1847.958 | 8912.948 | 30.046 |
| mm_1k_2k | [2](full/cann930_20261117/performance_results/mm_1k_2k/20261116T221333Z_YHqkFW/benchmark.json) | 64/0 | 131072 | 73.521 | 1782.774 | 6931.700 | 32.280 |
| mm_1k_2k | [3](full/cann930_20261117/performance_results/mm_1k_2k/20261116T221502Z_hjwCFl/benchmark.json) | 64/0 | 131072 | 74.936 | 1749.108 | 7609.830 | 32.610 |
| mm_1k_2k | [4](full/cann930_20261117/performance_results/mm_1k_2k/20261116T221632Z_MyA2B0/benchmark.json) | 64/0 | 131072 | 65.632 | 1997.070 | 5894.833 | 28.910 |

三场景四轮平均吞吐分别为 2230.531、2212.914、1844.228 token/s。数据对应本轮 CANN930、CCU_SCHED、异步调度与服务复用条件。[完整报告](full/cann930_20261117/REPORT.md) [执行汇总](full/cann930_20261117/combined_run/results_summary.json)

## 1108 历史性能结果

主性能场景为 **4K 输入→32K 输出**，图文每请求一张图片。

E23三组均64请求、最大并发64、seed42、ignore-eos、request-rate10000、openai-chat；服务端每DP实例32。图片bucket为 `(1920,1080,1)`，该脚本按**高、宽、帧数**解释，实际为高1920、宽1080的竖图。若验收将“1920×1080”严格定义为宽×高的横图，需要补测，不能默认为已满足方向要求。

| 场景 | 请求文本输入 / 输出 | 数据集 | 每轮实际输入总token | 每轮要求输出总token |
|---|---|---|---:|---:|
| text_2k_6k | 2048 / 6144 | random，无图片 | 132992（每条2078，含模板） | 393216 |
| mm_1k_2k | 1024 / 2048 | random-mm，一图 | 130623 | 131072 |
| mm_4k_32k | 4096 / 32768 | random-mm，一图 | 327233 | 2097152 |

实际输入统计包含模板与视觉开销，不冒充每请求服务端input恰为1024/4096。随机图文用于性能，不证明图片语义理解准确率。三组每组复用本组服务串行四轮；没有声明三组从头到尾共用同一次服务。prefix caching开启且相同seed，后续轮次可能缓存命中，不能作为冷启动吞吐。

| 场景 | 轮次/原始JSON | 成功/失败 | 总输出token | 耗时s | 输出tok/s | Mean TTFT ms | Mean TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| text_2k_6k | [1](full/E23/text_2k_6k/round_1_benchmark.json) | 64/0 | 393216 | 180.39 | 2179.76 | 6738.50 | 28.21 |
| text_2k_6k | [2](full/E23/text_2k_6k/round_2_benchmark.json) | 64/0 | 393216 | 171.81 | 2288.61 | 1869.86 | 27.64 |
| text_2k_6k | [3](full/E23/text_2k_6k/round_3_benchmark.json) | 64/0 | 393216 | 171.70 | 2290.09 | 1037.35 | 27.76 |
| text_2k_6k | [4](full/E23/text_2k_6k/round_4_benchmark.json) | 64/0 | 393216 | 175.62 | 2239.00 | 1396.87 | 28.33 |
| mm_1k_2k | [1](full/E23/mm_1k_2k/round_1_benchmark.json) | 64/0 | 131072 | 69.31 | 1891.16 | 6588.12 | 30.37 |
| mm_1k_2k | [2](full/E23/mm_1k_2k/round_2_benchmark.json) | 64/0 | 131072 | 72.53 | 1807.08 | 9012.81 | 30.79 |
| mm_1k_2k | [3](full/E23/mm_1k_2k/round_3_benchmark.json) | 64/0 | 131072 | 71.13 | 1842.74 | 6878.89 | 31.15 |
| mm_1k_2k | [4](full/E23/mm_1k_2k/round_4_benchmark.json) | 64/0 | 131072 | 69.94 | 1873.97 | 7457.73 | 30.27 |
| mm_4k_32k | [1](full/E23/mm_4k_32k/round_1_benchmark.json) | 64/0 | 2097152 | 950.84 | 2205.58 | 15710.79 | 28.51 |
| mm_4k_32k | [2](full/E23/mm_4k_32k/round_2_benchmark.json) | 64/0 | 2097152 | 941.53 | 2227.39 | 7476.15 | 28.48 |
| mm_4k_32k | [3](full/E23/mm_4k_32k/round_3_benchmark.json) | 64/0 | 2097152 | 954.54 | 2197.04 | 7486.10 | 28.88 |
| mm_4k_32k | [4](full/E23/mm_4k_32k/round_4_benchmark.json) | 64/0 | 2097152 | 957.42 | 2190.41 | 7351.35 | 28.98 |

客户端原始终端汇总： [2K→6K](full/E23/text_2k_6k/)、[1K＋图→2K](full/E23/mm_1k_2k/)、[4K＋图→32K](full/E23/mm_4k_32k/)。各轮均核对成功数、失败数和完整输出总量；不能只看退出0/HTTP200。

## 无标杆硬件性能分析

[两轮对比报告](profiling/hardware_comparison.md)、[no_stack硬件报告](profiling/no_stack/hardware_report/hardware_report.md)、[no_stack Excel](profiling/no_stack/hardware_report/hardware_efficiency.xlsx)、[with_stack Excel](profiling/with_stack/hardware_report/hardware_efficiency.xlsx)。

完整BF16＋Engram，纯文本2K→6K，TP1/PCP8/DP2；delay_iterations=1536、max_iterations=10。两轮16 rank均解析完成，10步×40层Sparse MLA窗口核验通过；硬件评分只计算node25/device0。1536为调度迭代延迟，不能换算成精确输出token位置。

| 指标 | CLI with_stack=false | CLI with_stack=true |
|---|---:|---:|
| 计算调用数 | 21970 | 21970 |
| 计算调用累计耗时ms | 218.097 | 219.723 |
| 经验达成率 | 100.4878% | 100.7164% |
| 筛选的重点调用数 | 745 | 759 |
| 原生with_stack / with_modules | false / false | false / true |

口径：每个计算调用以耗时占比为权重；长于等于10μs的调用按最大有效管线ratio/0.8计分，短于10μs全额计分，不截断超过100%的值。通信、搬运、空闲、CPU调度和Engram等待不进入计算分母。这是**经验评分**，不是MFU或理论FLOPS峰值利用率，也不代表16卡整机达到80%。

优化候选为duration>20μs且max_ratio<0.6，主要为Sparse MLA、TransposeBatchMatMul，其次FP4 indexer/QuantLightningIndexer。候选筛选、单卡分析与经验达成率已在Excel和报告中给出，无需重复统计；没有实施优化收益的证据。

no_stack客户端完整结束64成功/0失败、393216输出token；with_stack轮在采集完成后主动终止，不能作为另一轮完整性能通过。当前wrapper将CLI with_stack映射到with_modules，原生Python栈未采集。no_stack旧报告中“尚未执行with_stack”是写作时点，最终状态以两轮对比为准。

**待补：**4K图文→32K的prefill/视觉/稳定decode性能分析，跨rank覆盖与不均衡；若验收要求理论0.8×硬件峰值，需补芯片/精度/shape匹配的峰值或经验上限和计算/访存口径，不把100.49%直接认定该目标已通过。

## 精度验证

2026-11-19 在 177.42.23.1/24.1 的同一套 CANN930 完整模型服务上完成 GPQA Diamond 三轮，每轮 198 题，共 594 个请求全部成功。评分使用 AISBench GPQAEvaluator，以最终 content 提取的选项为准，无可解析最终选项的回答计错。

| 轮次 | 正确题数 | 准确率 | 请求成功 | 客户端总耗时 | 达输出上限 | 未解析最终选项 |
|---|---|---|---|---|---|---|
| 1 | 180/198 | 90.909091% | 198/198 | 54 分 20 秒 | 1 | 2 |
| 2 | 178/198 | 89.898990% | 198/198 | 48 分 25 秒 | 0 | 1 |
| 3 | 178/198 | 89.898990% | 198/198 | 61 分 35 秒 | 2 | 2 |

三轮准确率算术平均为 **90.235690%**。每轮 AISBench 原始分数、原始分数复算和最终 content 分数均一致，评分差异题号均为空。三轮使用相同数据集与 seed=42，输出预算为 65536，temperature=1.0、top_p=0.95、top_k=0、enable_thinking=true、reasoning_effort=low、并发 8。最长输入 2595 token，加输出预算为 68131 token，服务上下文为 73728。[最新精度报告](full/cann930_20261119_precision/REPORT.md) [三轮审计](full/cann930_20261119_precision/evidence_validation.json) [原始预测与评分](full/cann930_20261119_precision/results/)

第 1 轮未解析题号为 79、142，第 2 轮为 149，第 3 轮为 79、147，均为零基题号。输出上限题号分别为 79、空、79/147，所有未解析回答均计入 198 题分母。服务启动耗时约 28 分 49 秒，客户端耗时包含推理、评分与审计。两端各初始化一次引擎，结束后的本任务进程、NPU 进程和服务端口均释放，共享内存可用量回到启动前基线。[运行汇总](full/cann930_20261119_precision/run/results_summary.json)

2026-11-17 的历史实验使用 32768 输出预算、temperature=0、top_p=1、reasoning_effort=100，最终 content 为 164/198、82.828283%，24 题达到输出上限。该轮原始 AISBench 评分为 165/198，零基题号 52 从推理文本提取到正确选项而最终 content 为空，形成 1 题差异。历史参数、评分与修正过程继续保留。[32768 预算历史报告](full/cann930_20261117/REPORT.md)

本次三轮与历史实验涉及输出预算、采样和思考配置的变化。精度达标阈值及同条件对照仍待确定。Math-500、TextVQA 的正式评测、同条件多轮对照和 T 检验仍待完成。

# 问题、规避与CANN9.3.0补充结果

1. **Gloo消息长度不匹配/连接中断：**原网口enp35s0f2上纯CPU通信可复现数据异常，切enp39s0f0后对照通过，E23使用该配置；HCCL仍enp35s0f2。证据见[定位报告](diagnostics/gloo_communication_diagnosis.md)。这定位到通信路径差异，未证明网卡驱动内部根因。
2. **Engram加载慢：**CPU offload有大表读取与内存注册阶段，短暂SSH/设备查询超时不能直接判HBM不足或服务死锁。用日志进度、有界只读重试和请求输出判断；不将单次查询超时作为杀服务依据。
3. **CANN9.3.0裁剪模型异步长输出异常：**历史E24出现507011、AI Vector timeout/trap，async copy_event同步处检测错误；返回64成功但输出5458而非393216也不能判通过。具体根因尚未闭环，不归因成“BF16 KV不兼容”。
4. **完整模型CANN9.3.0 E25：**同权重和Engram，关闭async后4K图文→32K三轮通过；耗时1400.80/1378.84/1370.71秒，吞吐1497.11/1520.95/1529.98 tok/s。见[三轮结果](full/E25/results/)与[环境](environment/cann930_container_build_report.md)。
5. E23平均约2205 tok/s，E25约1516 tok/s。CANN版本和async开关同时改变，不能把约31%的差距直接归因CANN升级。现有ArgSort转AiCPU、internal format等告警并非E25独有；未找到证明稳态性能下降根因的单条告警。[日志对比](diagnostics/E23_E25/)。需补9.2.1＋async关闭的同条件控制组。
6. **Sleep level-2 恢复差异已修复。** attention sink 加载改为调用 layerwise weight_loader，无学习权重 RMSNorm 常量注册为非持久 buffer，CANN930 五轮逐副本双接口验证通过。[修复与实测](reduced/cann930_20261117/sleep_level2/REPORT.md)
7. **全量权重更新接收端 OOM 已修复。** 排除运行时通信映射的加载配额后，HCCL、IPC 均完成全量 A→B→A。IPC 发送端缓冲回收已完成 2026-11-18 单卡及 8 卡实机验证，冷/热缓存输出差异待定位。[修复与实测](reduced/cann930_20261117/weight_update/REPORT.md)
8. **CANN930 完整模型异步配置已复验。** 双节点使用 CCU_SCHED 和 Gloo CPU 超时透传修复，完成 12 轮完整性能与 GPQA 请求。默认通信展开模式超时的底层机制仍待确认。[当前完整模型报告](full/cann930_20261117/REPORT.md)

# 交付物与复现范围

[README](README.md)为工程入口，[归档清单](ARCHIVE.md)说明交付范围，`manifest.sha256`记录文件校验值。本仓库收录运行脚本、实际参数、原始日志/响应/性能 JSON、环境快照、修复补丁、转换记录、失败证据和 profiling 表格。历史 IP、容器及模型路径保留用于追溯，当前执行入口使用 CANN930 环境配置。

镜像仅提供标签、ID和环境证据；尚未导出可下载镜像tar，不能把记录镜像ID写成已交付镜像二进制。原始两轮profiling rank0 zip及全16rank trace为本地/共享存储大件，Git仓库保存分析CSV/Excel和索引，不复制数百MB检查中间文件。

# 待开展的开箱实验

| 优先级 | 实验 / 交付缺口 | 必须保存与通过条件 |
|---|---|---|
| P0 | 完整模型Math-500、TextVQA精度 | 数据/命令/版本、逐题输出、提取与截断、评分和同口径对照/T检验 |
| P0 | 完整模型Level-2 sleep/wake五轮 | 双接口固定greedy请求，文本/token/logprobs恢复一致，各卡逐轮显存无持续增长 |
| P0 | routing专家返回及需要时训练侧replay | 单/多请求原始专家、dtype/shape/范围，实际回放路径无报错 |
| P0 | 冷缓存与命中缓存输出差异定位 | 固定权重、输入及副本，对照 prefix replay、注意力状态与 logits/logprob，保留相同缓存状态的数值对照 |
| P0 | 完整 384 专家模型 checkpoint、HCCL、NPU IPC 更新 | 独立容量预算、全量计数、版本、权重哈希、生成对照和连续更新回收检查 |
| P1 | 完整模型1M有效回答 | tokenizer输入计数、完整请求/响应、检索命中与无乱码；不只容量 |
| P1 | 主场景4K图文→32K硬件验收 | 视觉/prefill/decode、多卡取证；明确经验评分与理论峰值目标区别 |
| P1 | 完整模型功能开关精度对照 | PCP档位、eager/graph、async、chunk，固定输入和比较口径 |
| P1 | 严格横向1920×1080图片补测（如验收要求） | bucket改为高1080宽1920后独立结果，不覆盖现有竖图数据 |
| P1 | TP2/DCP/SP/PP限制闭环 | 对应实现修复或版本支持后重测；当前保持失败/阻塞，不强行启用 |
| P2 | CANN性能单变量对照 | 9.2.1＋async关闭与9.3.0＋async关闭同负载同采样 |
| P2 | 权重与交付补证 | 反量化FP4/FP8数值校验、公共源码可达提交/必要PR、镜像导出校验 |

资源安排后按上述范围补充实测。32 专家模型已完成的 Sleep level-2 与两种全量权重更新结果，见本轮专项报告。

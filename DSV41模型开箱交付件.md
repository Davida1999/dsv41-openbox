# DSV41 模型开箱交付件

归档日期：2026-11-14。部署基线：**1108容器、CANN9.2.1、完整BF16反量化模型＋Engram CPU offload，双机16卡**。

本交付件按《模型开箱交付件模板》组织，参考Step5 preview和HY4的证据呈现方式。完整模型已完成文本单curl、图文推理负载及三组各四轮性能；RL、1M有效回答和数据集精度尚未全部通过。**文档整理完成不等于模型开箱全项验收通过。** 本次只整理既有实验与只读取证，没有新占用NPU或重跑实验。

## 验收总览与证据替代规则

“已验证”仅指表内受测配置；“部分”表示通路或有限检查完成；“失败/阻塞”保留原始证据；“待开展”没有足够实测证据。完整模型正常推理不能替代未执行的专项动作。

| 验收项 | 当前结论 | 主证据 / 替代范围 |
|---|---|---|
| 完整BF16＋Engram基础推理 | 已验证双接口可读文本样例 | E13完整模型30次单curl；替代裁剪模型乱码作为基础输出证据，不代表数据集精度 |
| EP / DP / PCP | 已验证TP1/PCP8/DP2、EP16的组合 | E13单curl＋E23长输出性能；不宣称TP>1或PCP各档精度等价 |
| TP>1 / DCP / SP | TP2失败；DCP/SP前置阻塞 | 1108裁剪32专家专项保留，不能用完整TP1替代 |
| PP | PP2裁剪模型失败；完整Engram基线PP1 | 不标记PP功能通过 |
| 图模式 / async / chunked prefill | 完整模型所列组合已运行 | E13显式开关＋E23正式运行；独立开关精度消融未在完整模型完成 |
| 2K文本→6K、1K文本＋图片→2K、4K文本＋图片→32K | 各四轮64成功、0失败，输出长度完整 | E23/1108，替代裁剪模型对应性能 |
| sleep level-1 | 裁剪模型5轮恢复一致性通过 | 不代表完整模型，也不能替代模板要求的level-2 |
| sleep level-2 | 裁剪模型首轮恢复token不一致 | 完整模型待验证 |
| 专家路由返回 / replay | 裁剪模型双请求断言/HTTP500 | 完整模型待验证；不能由普通MoE推理推出replay支持 |
| Checkpoint / HCCL / NPU IPC更新 | 未验收通过 | level-2 reload后不一致；HCCL/IPC完整更新OOM |
| 1M输入 | 裁剪模型容量通过，检索0/3 | 不能替代完整模型1M可读回答与精度 |
| 0.8硬件目标 | 单卡decode经验评分通过；理论峰值口径未证明 | E23文本2K→6K profile，不外推4K图文→32K全阶段或16卡 |
| Math-500 / TextVQA | 待开展 | 未找到本模型正式评测与同条件参考，不沿用模板示例分数 |

完整模型结果只替换相同验收项的主要证据；裁剪专项保留在[基础功能报告](reduced/basic_feature/VALIDATION_REPORT.md)和[原始结果](reduced/basic_feature/results/)。E13早于1108，使用27/28节点；不能倒写为1108实测。E25为CANN9.3.0补充验证，与1108性能分列。

# 模型结构分析

[模型结构图与资产分析](model/模型结构分析.md)包含40层文本主干、384专家Top-6、32层视觉编码器、Engram层[1,14]、A5 KV布局与权重来源。配置与索引均已归档；模型权重和tokenizer资产不上传GitHub。

本资产由FP8/FP4源权重反量化获得。转换器、执行脚本、525分片/48496张量完成日志均见[转换归档](model/conversion/)。没有把旧“尚未开始转换”的准备文档当成最终状态。

# 模型开箱打通

## 环境配置：1108基线

| 组件 | 版本／配置 |
| --- | --- |
| 容器系统 | Ubuntu 22.04.5 LTS |
| 宿主机系统 | openEuler 24.03 LTS-SP3 |
| 架构／CPU | aarch64，每台 192 逻辑核 |
| NPU | 每台 8×Ascend950DT，单卡 HBM 96 GiB |
| Ascend 驱动 | `25.1.rc3` |
| 固件 | `9.2.0.101.235` |
| CANN／Runtime／HCCL | `9.2.1` |
| Python | `3.12.16` |
| PyTorch | `2.10.0+cpu`，通过 TorchNPU 支持 NPU |
| TorchNPU | `2.10.0.post7` |
| vLLM | `0.32.0+empty`，commit点： `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | `0.19.1rc2.dev3401+g28e9e97ba`，commit点： `28e9e97ba254b076a88275e8a275556cb9a11664` |
| Transformers | `5.19.0` |
| Triton／Triton-Ascend | `3.6.0`／`3.6.0.post2` |
| NumPy | `1.26.4` |
| safetensors／tokenizers | `0.8.0`／`0.23.3` |
| Ray | `2.48.0` |
| Docker 网络／IPC | `host`／`host` |
| 实际共享内存容量 | 约 1.10 TiB |
| E23 Gloo 接口 | `enp39s0f0` |
| E23 HCCL 接口 | `enp35s0f2` |

镜像：`c00954197/vllm-ascend:main-a5-20261108-e8215a9d`；镜像ID：`sha256:59b0f6431b5f81bc0037d485c41bdcccf6af70805090c68bfbedba14ba6bc5e8`。镜像标签后缀不是当前源码commit。

[环境报告](environment/1108_environment_20261112.md)与[两节点原始快照](environment/environment_snapshot.json)记录挂载、导入路径和版本。原环境报告将KV写为BF16，指CLI值；实际packed cache解释以本交付件模型分析为准。该环境证明历史实测，不是任意torch/CANN版本组合的兼容认证。

vLLM源码：`https://github.com/vllm-project/vllm`；Ascend插件：`https://github.com/vllm-project/vllm-ascend`。历史1108快照两仓库已跟踪文件无改动，未统计未跟踪文件。未找到需要附上的本实验专用源码修复PR；Gloo问题通过接口配置规避，不虚构已合入PR。

## 全层服务与客户端脚本

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

裁剪模型受TP>1前置限制，未执行有效SP专项；TP1下打开无效开关不能算通过。[配置](reduced/basic_feature/configs/cases/F05_sp.json)。保留模板 `--compilation-config '{"pass_config":{"enable_sp":true}}'` 作为待验证方向，不提供已经通过的声明。

## PP功能

当前完整Engram路径使用PP1；裁剪模型关闭Engram后实测PP2仍失败：`hidden_states`为None时调用`unsqueeze`，另一stage报`PPMissingLayer`缺少`layer_idx`。[原始日志](reduced/basic_feature/results/node26_F06_pp2_20261113T115905/server.log)。原完成记录曾误报让卡，报告已根据异常堆栈更正；保留原文件不覆盖。

## 图模式、异步调度、chunked prefill

完整模型脚本显式开启 `FULL_DECODE_ONLY`、`--async-scheduling`、`--enable-chunked-prefill`，E13可读输出与E23正式性能证明组合运行。长输入超过2048调度预算且请求完成，支持分块路径已工作的判断；不伪造逐步分块计数。

裁剪专项保留独立消融：graph相对eager单请求4/4一致、并发1/2一致；async与同步6/6 token一致；chunk与非chunk的6/6输出不同。完整模型组合成功不能把这些专项结果改写为完整模型逐项精度一致。图模式结论仅限decode图，不扩展至所有图模式或prefill图。

# RL功能验证

以下最新专项使用1108、40层32专家且Engram/DSpark关闭，与完整模型身份不同。 [完整13项报告](reduced/basic_feature/VALIDATION_REPORT.md)、[执行汇总](reduced/basic_feature/results/EXECUTION_SUMMARY_20261113.json)、[测试脚本](reduced/basic_feature/scripts/)、[各配置](reduced/basic_feature/configs/cases/)。本次没有移植HY4/Step5补丁或重新运行RL测试。

## sleep mode功能精度验证

模板参考：[vLLM #53200](https://github.com/vllm-project/vllm/pull/53200)。

- Level-1：5/5轮temperature=0输出token与基准一致，8卡HBM最后相对初始最大增长80MiB；仅首引擎allocator统计可见，其余卡依赖HBM采样。此结论不代表全部logprobs严格相等或长期无泄漏。
- Level-2：按wake(weights)→reload_weights→wake(kv_cache)执行，两次均首轮token不一致，复现从第一个输出token即不同；未完成5轮，不能标为通过。
- 原始[Level-1记录](reduced/basic_feature/results/node26_F11_sleep1_20261113T100448/)、[Level-2复现](reduced/basic_feature/results/node26_F12_sleep2_repro_20261113T102258/)。完整BF16＋Engram的Level-2五轮输出、logprob和显存检查待开展。

## routing replay功能验证

模板参考：[vLLM #52902](https://github.com/vllm-project/vllm/pull/52902)。裁剪模型启用返回专家后，AuxOutput要求prefix caching；修正后单请求HTTP200，但响应未及时落盘，双请求触发 `unexpected topk_ids batch dim 2` 与HTTP500，不能判专家内容校验通过。[日志](reduced/basic_feature/results/node26_F09_routing_20261113T095702/server.log)。训练侧强制路由replay也未验证。完整MoE正常推理不能替代此项。

## 权重更新功能验证

模板参考：[vLLM #53541](https://github.com/vllm-project/vllm/pull/53541)。

| 路径 | 实测状态 | 后续验收 |
|---|---|---|
| checkpoint reload | 与Level-2恢复组合执行，返回后生成token不一致 | 独立A→B→A、双接口文本/token/logprobs与权重内容检查 |
| HCCL | DP4/EP4推理＋独立sender；0.75/0.55更新OOM；0.50启动KV不足 | 先做全量权重更新内存预算，再完整传输与一致性验证 |
| NPU IPC | DP8/EP8＋8个同卡sender，packed2GiB、0.45、backend=npu_ipc，真实更新OOM | 校验完整B加载和A恢复，不能以部分传输成功代替 |

[HCCL结果](reduced/basic_feature/results/node26_F13_hccl_20261113T112758/)、[NPU IPC结果](reduced/basic_feature/results/node26_F13_ipc_20261113T115009/)。完整384专家模型内存需求更高，不能直接照搬裁剪模型参数重测。

## 1M长序列功能验证

裁剪模型实现了1000000输入token＋1024输出token，耗时305.99秒，但5%/50%/95%三个位置检索命中0/3，输出重复。只能标记容量通路通过，不能写成“1M精度/有效回答通过”。[原始结果](reduced/basic_feature/results/node26_F14_long_20261113T120549/long_1000000.json)。

完整模型E23上下文40960、4K→32K长输出不等于1M长输入。完整模型1M测试待开展：保存真实tokenizer计数、请求与输出、内存、检索答案；模板附件若不可获取，先确定替代文本并注明，不能将1M字符直接当成1M token。

# 性能与精度验证

## 性能需求与正式结果

以当前模板的**4K输入→32K输出**为主场景，图文每请求一张图片。原草稿“多模态32K→4K”标题已纠正。

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

目前没有可归档的本模型Math-500、TextVQA正式评分、竞品同条件多轮数据或T检验。草稿中的Math-500 97.44%、TextVQA 77.539%来自模板示例，已删除，**不是本模型成绩**。单curl算术正确和随机长输出性能不能代替数据集精度。

待执行：完整BF16＋Engram，ais_bench固定数据集版本、prompt、图像处理、tokenizer、采样参数、最大输出长度和seed；保存逐题输出、提取率、截断率与评分。有竞品时用同条件多轮分数进行T检验；无竞品时选模型报告同口径典型集，明确“1%”是绝对百分点还是相对误差。模板所列内部T检验规范本次未读取，不能声称通过该规范。

# 问题、规避与CANN9.3.0补充结果

1. **Gloo消息长度不匹配/连接中断：**原网口enp35s0f2上纯CPU通信可复现数据异常，切enp39s0f0后对照通过，E23使用该配置；HCCL仍enp35s0f2。证据见[定位报告](diagnostics/gloo_communication_diagnosis.md)。这定位到通信路径差异，未证明网卡驱动内部根因。
2. **Engram加载慢：**CPU offload有大表读取与内存注册阶段，短暂SSH/设备查询超时不能直接判HBM不足或服务死锁。用日志进度、有界只读重试和请求输出判断；不将单次查询超时作为杀服务依据。
3. **CANN9.3.0裁剪模型异步长输出异常：**历史E24出现507011、AI Vector timeout/trap，async copy_event同步处检测错误；返回64成功但输出5458而非393216也不能判通过。具体根因尚未闭环，不归因成“BF16 KV不兼容”。
4. **完整模型CANN9.3.0 E25：**同权重和Engram，关闭async后4K图文→32K三轮通过；耗时1400.80/1378.84/1370.71秒，吞吐1497.11/1520.95/1529.98 tok/s。见[三轮结果](full/E25/results/)与[环境](environment/cann930_container_build_report.md)。
5. E23平均约2205 tok/s，E25约1516 tok/s。CANN版本和async开关同时改变，不能把约31%的差距直接归因CANN升级。现有ArgSort转AiCPU、internal format等告警并非E25独有；未找到证明稳态性能下降根因的单条告警。[日志对比](diagnostics/E23_E25/)。需补9.2.1＋async关闭的同条件控制组。

# 交付物与复现范围

[README](README.md)为工程入口，[归档清单](ARCHIVE.md)说明包含和未包含材料；`manifest.sha256`记录归档文件校验值。保留运行脚本、实际参数、原始日志/响应/性能JSON、环境快照、转换记录、功能失败证据和profiling表格。文件中历史IP/路径用于追溯，迁移时需要替换；GitHub按用户选择作公开归档，不上传模型权重、SSH凭据或环境认证信息。

镜像仅提供标签、ID和环境证据；尚未导出可下载镜像tar，不能把记录镜像ID写成已交付镜像二进制。原始两轮profiling rank0 zip及全16rank trace为本地/共享存储大件，Git仓库保存分析CSV/Excel和索引，不复制数百MB检查中间文件。

# 待开展的开箱实验

| 优先级 | 实验 / 交付缺口 | 必须保存与通过条件 |
|---|---|---|
| P0 | 完整模型Math-500、TextVQA精度 | 数据/命令/版本、逐题输出、提取与截断、评分和同口径对照/T检验 |
| P0 | 完整模型Level-2 sleep/wake五轮 | 双接口固定greedy请求，文本/token/logprobs恢复一致，各卡逐轮显存无持续增长 |
| P0 | routing专家返回及需要时训练侧replay | 单/多请求原始专家、dtype/shape/范围，实际回放路径无报错 |
| P0 | checkpoint、HCCL、NPU IPC完整更新 | 内存预算，独立完整A→B→A与内容哈希；生成恢复；泄漏/正常退出检查 |
| P1 | 完整模型1M有效回答 | tokenizer输入计数、完整请求/响应、检索命中与无乱码；不只容量 |
| P1 | 主场景4K图文→32K硬件验收 | 视觉/prefill/decode、多卡取证；明确经验评分与理论峰值目标区别 |
| P1 | 完整模型功能开关精度对照 | PCP档位、eager/graph、async、chunk，固定输入和比较口径 |
| P1 | 严格横向1920×1080图片补测（如验收要求） | bucket改为高1080宽1920后独立结果，不覆盖现有竖图数据 |
| P1 | TP2/DCP/SP/PP限制闭环 | 对应实现修复或版本支持后重测；当前保持失败/阻塞，不强行启用 |
| P2 | CANN性能单变量对照 | 9.2.1＋async关闭与9.3.0＋async关闭同负载同采样 |
| P2 | 权重与交付补证 | 反量化FP4/FP8数值校验、公共源码可达提交/必要PR、镜像导出校验 |

以上为待办计划，本次不执行新的推理、RL、profiling或权重转换任务。

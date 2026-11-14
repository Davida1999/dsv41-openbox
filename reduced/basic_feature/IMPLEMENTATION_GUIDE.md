# DeepSeek-V4.1 BF16 开箱基础功能任务实施指南

交付日期：2026-11-12（Asia/Shanghai）。初版交付为实施材料；2026-11-13已在26单机执行，实际结果与限制见VALIDATION_REPORT.md。

## 1. 固定范围

- 服务器：177.11.26.1，8×Ascend950DT，单卡96GiB。所有实验及发送端均在该服务器。
- 容器：c00954197_vllm_ascend_1108。
- 工作目录：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature`。全部执行命令在容器内运行。
- 唯一模型：`/mnt/share/g00815994/dsv41/DeepSeek-V4.1-Flash-top32-bf16-runtime-codex-20261030`。
- 实际配置：40层、32 routed experts、top-6、Engram关闭；不修改模型目录，不额外裁剪，不使用dummy权重。
- vLLM：5df63a7203cdb9328d824a27b682e77c7b9928be；vLLM-Ascend：28e9e97ba254b076a88275e8a275556cb9a11664。如版本变化，重新审核参数和证据。
- 文本推理范围；不将文本结果扩大为视觉、工具调用或DSpark验收。
- 当前8卡有其他SGLang任务占用。不得停止、重启这些任务；仅在资源释放后的窗口执行。

## 2. 脚本编写约束

脚本须简洁直白：一个脚本负责一种动作，公共HTTP/配置读写收敛到common.py。模型路径、端口、拓扑、请求长度、循环次数、超时、阈值均放入配置；不重复硬编码、不隐藏失败、不构建复杂调度框架。`configs/common.json`提供公共参数，`configs/cases/*.json`只覆盖差异。修改配置前后都保留快照。脚本静态检查通过不代表运行通过。客户端使用与服务一致的DeepSeek V4.1专用tokenizer，thinking默认false，可通过公共配置调整。

目录：configs为配置；scripts为执行入口；results为每次运行独立证据；evidence为交付检查。原模型只读。结果目录不可复用覆盖。报告使用脚本完整路径，不复制全量代码。

## 3. 进入环境与检查

在服务器执行：

```bash
docker exec -it c00954197_vllm_ascend_1108 bash
cd /mnt/share/c00954197/workspace/dsv41-kx/basic_feature
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
CASE=F01_basic
RUN="$PWD/results/${CASE}_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RUN"
bash scripts/preflight.sh "$CASE" "$RUN"
```

preflight只读safetensors header/索引、配置、Git状态和npu-smi；报告各类权重字节量。文件manifest报告的类型须与header交叉核对。实读header：总计529,784,323,912字节，其中Engram 393,857,008,640、MTP 28,456,625,544、其余107,470,689,728。其余约100.1GiB也不等于实际NPU常驻量（包含视觉等权重），IPC单卡容量须实测估算。必须考虑dense复制、EP分片、KV、图工作区、通信buffer及CPU可用内存，不能只用总大小除以卡数。

执行前阅读expert_remap.json、codex_runtime_manifest.json，确认专家0..31与原专家ID映射。preflight不做昂贵的全量权重校验和。正式验收需要记录资产可信来源及必要的分片校验。

## 4. 单项通用执行流程

```bash
# 只打印命令并保存有效配置，不启动服务
bash scripts/serve.sh "$CASE" "$RUN"
# 确认拓扑、端口和独占空闲窗口后启动；脚本也会检查设备占用
bash scripts/serve.sh "$CASE" "$RUN" --launch
# 另一个终端采集HBM；完成后Ctrl-C结束采集，不影响服务
python scripts/collect_memory.py "$CASE" "$RUN"
# 主终端执行本项客户端
bash scripts/run_case.sh "$CASE" "$RUN"
# 基础性能：先查看命令，确认当前CLI支持，再执行
bash scripts/benchmark.sh "$CASE" "$RUN"
bash scripts/benchmark.sh "$CASE" "$RUN" --execute
# 只停止该RUN记录的进程组；结束后核实NPU释放
bash scripts/stop.sh "$RUN"
npu-smi info
```

serve不覆盖已有服务。run_case是客户端，不负责启动服务。一个时刻只运行一个实验服务；客户端case必须与该RUN的effective_config一致。当前已验证block_size=128为必需配置；关闭chunk时batched_tokens不得小于max_model_len。服务启动失败/超时先留日志，不切换其他参数掩盖失败。stop只向记录的进程组发送SIGTERM，不使用pkill；如有未退出子进程，核实所属关系后人工处理。启动端口只绑定127.0.0.1；开发RPC仅用于本实验。

结果至少保留effective_config.json、command.json、server.log、preflight.json、check.log、请求响应、memory.jsonl。失败同样保留结果。原始文本、token IDs和路由可能较大，按RUN归档，不提交模型数据。

## 5. 逐项顺序与对照

| 顺序 | case | 目的/前置条件 | 主要判定 |
|---|---|---|---|
| 1 | F01_basic | DP8/TP1/EP8，eager，同步 | 单请求与双请求无算子报错；人工检查中文/英文/代码/算术输出；记录TTFT/TPOT |
| 2 | F02_ep | 同基线 | 日志确认专家分片、HCCL/dispatch生效；F02_ep4是有容量前提的4卡对照 |
| 3 | F03_tp2 | DP4/TP2/EP8 | BF16不受量化Issue直接证明；与F01对照 |
| 4 | F04_pcp1_control → F04_pcp2 | 32K同输入；PCP2/DP4/TP1/DCP1 | 实际PCP分片，比较相同输入；8K可另建覆盖input_tokens的case |
| 5 | F04_dcp2 | 配置默认阻塞 | 未证明无Engram DCP不支持；审查专用DSA后端后决定，不以PCP限制代替独立DCP结论 |
| 6 | F05_sp | F03通过；同TP2拓扑开启flashcomm1 | SP路径实际生效，与F03对照 |
| 7 | F06_pp2 | 低优先级；已核对SupportsPP声明，实际跨层状态仍待运行 | 审查CED跨层KV和PP实现后执行；完整模型Engram限制不能直接豁免当前模型 |
| 8 | F07_graph | B0通过 | FULL_DECODE_ONLY捕获/回放证据；与F01对照；不宣称prefill图通过 |
| 9 | F08_async | B0通过 | 日志确认异步启用，与F01对照；再运行F08_mixed_control → F08_mixed_async，以8192/128 tokens混合请求复核 |
| 10 | F09_routing | V2，捕获开关 | 解码base64 NumPy；shape、层数、top-6、0..31和原专家映射；不代表强制路由重放 |
| 11 | F10_control → F10_chunk | 相同16K输入，chunk budget=2K | 日志确认多chunk调度；对照一致性 |
| 12 | F11_sleep1 | RL mode，CPU内存充足 | 预热后5次sleep/wake/reload，温度0 token完全一致，无持续HBM增长 |
| 13 | F12_sleep2 | RL mode | weights wake→reload→kv_cache wake；5轮严格一致性 |
| 14 | F13_hccl | 推理0..3卡，sender4卡；先证明4卡容量 | A→B→A流式权重传输；每rank norm变化/恢复、输出恢复、专家加载日志 |
| 15 | F13_ipc | DP8/EP8，8个同设备sender，packed 2GiB，显存比例0.45，backend=npu_ipc | 运行scripts/run_ipc.py；显式127.0.0.1静态rendezvous，自动选择空闲端口；逐DP哈希与恢复输出核验 |
| 16 | F14_long | 前项稳定，独占资源 | 128K→256K→512K→1,000,000输入+1024输出，无截断/OOM |
| 17 | C01_combined | 图/异步/chunk单项均通过 | 三项同时开启双请求回归；不与单项结果混为一谈 |

配置中的blocked是执行前置条件，不是“不支持”的证据。解除blocked必须先将审查依据、容量测算、版本和新命令写入报告。保留原配置副本。

## 6. 精度及性能判定

`check_inference.py`自动检查非空、替换字符，保存单请求及双请求；不能自动识别所有乱码/循环。所有输出需人工检查，B0乱码时禁止将后续精度写通过。

对照命令：

```bash
python scripts/compare.py results/BASE_RUN/inference.json results/TEST_RUN/inference.json
```

必须替换为真实RUN路径，且两边输入IDs相同。比较失败退出码2，仅表示需要分析。拓扑/图/调度浮点变化可能改变greedy文本；记录首个分歧，在相同前缀上比较logprobs或logits，再决定精度。此交付没有实现完整逐算子数值定位，也没有训练侧golden；不得写“完整模型精度通过”。sleep严格要求token IDs完全一致。

端到端tokens/s不等于decode吞吐。benchmark独立收集TTFT、TPOT、吞吐和并发；无业务阈值时报告实测、不写性能达标。应先warmup，再在相同输入输出长度和并发下比较。不使用裁剪模型结果代表完整模型性能。

显存：预热后统一等待settle_seconds。每rank allocated/reserved增长不超过max(256MiB,1%)；5轮HBM还须检查没有持续增长趋势。collect_memory保存原始npu-smi，人工整理每卡曲线。CPU内存不足时不执行level-1。发生权重更新失败时保持暂停，重启从原checkpoint恢复，禁止直接resume带着半更新权重继续测试。

## 7. 特殊项执行范围

### 权重更新

客户端使用安装版本的WeightTransferTrainerFactory，不实例化完整训练模型；metadata来自header，逐张量搬运到sender。仅将norm.weight乘以weight_scale作为B，其余主模型权重仍发送原值，跳过禁用的Engram与未启用的MTP。worker_probe读取model.norm.weight哈希；每rank必须改变并恢复，恢复后token IDs一致。必须结合所有expert接收/加载证据，不能仅以norm哈希声称全专家更新正确。

HCCL/IPC分别验收；IPC现使用scripts/check_ipc_multi.py：8个同设备sender通过NPU IPC传递权重，Gloo仅汇聚句柄；2GiB打包缓冲区限制发送端暂存。scripts/run_ipc.py按配置卡数启动，禁止使用不可解析的容器hostname作rendezvous。

### M级序列

1M定义为1,000,000个输入token，输出1024；总计1,001,024小于配置上限1,048,576。不得取1,048,576输入再追加输出，也不得静默截断。脚本精确构造token IDs，在5%/50%/95%位置插入三个密码。容量测试强制1024输出，检查usage；人工检查检索和输出质量。强制长度生成的文本与自然EOS场景分开解释。若任一阶梯OOM，保存现场并停止，不擅自换模型或量化。

## 8. 网页证据及适用边界

- [Engram检查（当前提交）](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/platform.py#L1869-L1912)：开启Engram时PP=DCP=1；本模型关闭Engram，此项只说明完整模型限制。
- [PCP检查](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/attention/context_parallel/dsa_v41_cp.py#L137-L145)：PCP路径要求DCP1；不等价于禁止独立DCP。
- [SP说明](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/user_guide/feature_guide/sequence_parallelism.md)：TP>1、EP、flashcomm1。
- [A5 TP量化问题](https://github.com/vllm-project/vllm-ascend/issues/20398)：原生MXFP/特定版本，不证明BF16 TP>1普遍不支持。
- [模型部署指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html)：950DT的TP1示例不等于其他拓扑不支持；latest会变化。
- [RL接口/路由/权重更新](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/user_guide/feature_guide/rl.md)。
- [sleep语义](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/user_guide/feature_guide/sleep_mode.md)。

证据豁免只用于条件、硬件、版本匹配且已明确禁止的项；不算通过。当前无Engram模型的DCP/PP保留待审查，不照搬完整模型结论。用报告逐项记录“通过/失败/证据豁免/前置阻塞/未执行”。

## 9. 实施完成条件

每个原始需求都有报告行、case配置、脚本完整路径、证据或未完成原因。未运行统一未执行；前置阻塞必须写明原因。任务完成后核实原checkpoint未修改、无遗留实验进程，保存配置/代码指纹和检查记录。day1/day2等原表时间列不自动解释为具体日期。

## 2026-11-13 执行补充

公共block_size显式设置128。路由返回case启用prefix caching满足AuxOutput前置要求。IPC原单卡客户端容量阻塞方案已替换为8卡DP8/EP8推理+8个同物理卡sender，脚本scripts/check_ipc_multi.py，由scripts/run_ipc.py按配置卡数启动；Gloo仅汇聚IPC句柄，权重使用NPU IPC，packed buffer=2048MiB。先验证实测后再判通过，旧单卡阻塞说明不再代表当前配置。HCCL/IPC通过每个DP写出独立norm哈希快照，避免RPC仅返回首引擎结果导致误判。

M级容量阶梯执行补充：128K/256K/512K仅生成probe_output_tokens=32用于先行容量检查；最终1,000,000输入严格生成1,024输出，要求不变。该参数仅避免在前置阶梯重复执行1K解码。

当前实测版本IPC必须显式使用backend=npu_ipc；backend=ipc会进入上游CUDA接收器。以当前注册表和真实堆栈为准，不依赖RL文档中的别名描述。

## 主机侧自动执行入口

实际执行推荐在26主机运行 `python3 /mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/supervise_host.py 26 <CASE>`。它负责空闲检查、启动、执行客户端和清理；不要同时再手动启动同一case。发现真实外部NPU计算进程时释放本轮资源。当前失败项先读验证报告，不要无差别重跑整个矩阵。

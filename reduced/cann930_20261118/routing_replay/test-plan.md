# DSV4.1 专家路由返回验证方案

日期为 2026-11-18。本轮定位双请求触发的 topk_ids 维度断言与 HTTP 500，并按路由捕获一致性用例验证完整返回专家数据。

## 方法来源与实际覆盖

MindSpeed Agent 的 model-rl-case-guard 提供 scripts/routing/test_moe_routing_replay.py，包含 HTTP 双接口和进程内 routed_experts 检查。通用脚本需要适配 DSV4.1 的嵌套 text_config、真实 DP8 布局、专家 ID 上界和批处理用例。当前 Ascend 原生 e2e 使用 MRV2 的 AuxOutput 路由捕获路径，本轮主验收采用 TP1/DP8/EP8。

PR #52902 的一致性用例覆盖完整 token 路由、prefix cache 命中、sleep level-1 分阶段唤醒及 TP/DP 前端对齐。快照 head 为 ff24198e140cfd2b5e770f43a9dae7f55050794a，当前状态为 closed、未合并。其注释明确训练侧强制路由重放仍有覆盖缺口。本轮按其中路由捕获一致性合同设计，修复依据当前容器的真实调用链。[PR #52902](https://github.com/vllm-project/vllm/pull/52902)

## 环境与固定配置

| 项目 | 方案 |
|---|---|
| 服务器与容器 | 177.42.22.1，c00954197_dsv41_cann930，CANN 9.3.0 |
| 工作目录 | /mnt/share/c00954197/Workspace/dsv41/basic/routing_replay |
| 模型 | /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32 |
| 模型配置 | 40 层、32 路由专家、1 共享专家、Top-6、BF16、Engram 关闭 |
| 拓扑 | TP1/DP8/EP8，PP1/PCP1/DCP1，真实 8 卡 |
| 初始执行模式 | MRV2、eager、同步调度 |
| 路由开关 | enable-return-routed-experts、enable-prefix-caching、enable-prompt-tokens-details |
| token 与缓存预算 | max-model-len=2048，max-num-batched-tokens=2048，max-num-seqs=2，显式 KV 2 GiB |
| 固定参数 | block-size=128，seed=42，temperature=0，return_token_ids=true |
| 通信 | CCU_SCHED，HCCL enp35s0f2，Gloo enp39s0f0，HCCL_BUFFSIZE=256 MiB |
| 服务地址 | 127.0.0.1:18244，启动前核对端口 |
| 生命周期 | 启用 sleep-mode，使用已验证的 RMSNorm、sink 与加载配额修复 |

历史 F09 使用 TP1/DP8/EP8、eager、同步、prefix caching，chunked prefill 关闭。新 baseline 同样关闭 chunked prefill，使故障定位集中在路由捕获和批布局。

## 复现与定位

历史堆栈经 execute_dummy_batch、_dummy_run 和 MoE router 进入 vllm_ascend/patch/worker/patch_routed_experts_capture.py，报 topk_ids 首维为 2，候选期望为 9、1、16、1。优先核对本地 padding 与 DP dummy forward 的元数据。

1. 保存当前环境、模型配置、导入路径、源码哈希及既有补丁。核对 MRV2 AuxOutput 的创建、路由 callback 绑定和响应序列化。
2. 使用当前有效 runner 复现双请求，在故障点记录真实/填充 token 数、topk_ids.shape、num_tokens_across_dp_cpu、local_sizes、DP/TP rank、MoE 通信类型和 dummy 标志。
3. 对照真实 token 归属确认本地 padding、全局 gather 和 dummy 路径的行布局。修复保留维度、范围和请求归属校验。
4. 使用生产 capture 函数增加针对性 CPU 回归，覆盖本地 1 行填充到 2 行、全局 9 行/填充 16 行、不同 DP rank、空闲 rank dummy forward 与真实双请求。非法布局继续产生明确错误。
5. 修复后重跑相同请求与配置，检查路由字段、worker 原始 topk 与请求/token 位置映射。

历史 V1 失败栈用于组件回归。MRV2 的真实 TP1/DP8/EP8 服务承担本轮主验收。各执行模式建立自身基准。

## 实机验证矩阵

| 用例 | 输入与动作 | 关键判定 |
|---|---|---|
| 路由关闭基线 | 双接口、4 个固定短 prompt、DP0–7 | 保存文本、token IDs、logprob 和实际输入长度 |
| 单请求返回 | 开启路由返回，同样逐 DP、逐接口请求 | HTTP 200，完整专家数组，与同输入/同缓存状态基准对照 |
| 同 DP 双请求 | 两个请求定向同一 DP，DP0–7，各覆盖双接口 | 调度证据确认同一步有 2 个真实请求，其他 rank 的 dummy forward 可持续运行 |
| 跨 DP 双请求 | 0/1、2/3、4/5、6/7 四组 rank 配对 | 请求 ID 与路由数据各自对应，专家行无串用 |
| 长短混合压力 | DP0，32 与 512/1024 token 输入，双请求至少 20 轮，每接口覆盖 | 覆盖 prefill/decode 重叠、不同完成时间和 padding 变化，全请求完成 |
| prefix cache 一致性 | 长 prompt 冷请求与重复请求 | cached_tokens 从 0 变为正数，完整缓存 prompt 路由行保留，按 PR 比较 token IDs 与全路由数组 |
| sleep/wake 一致性 | level-1 sleep，weights wake 与最终 wake，随后冷请求 | 缓存计数回到 0，token IDs 与完整专家数组逐元素匹配基准 |
| 图模式回归 | MRV2 FULL_DECODE_ONLY，capture sizes 1/2 | 重跑单请求、双请求及缓存/生命周期用例，检查捕获/回放与路由写入 |
| token-in/token-out | 按 PR 使用显式 token_ids 调用 /inference/v1/generate | 输入、输出 token 数与返回路由逐行对齐 |

并发由同步屏障同时提交，并保存 worker 调度证据。前端并发和实际同一步双请求均需满足用例目标。

已有 Top32 冷/热缓存生成差异纳入 prefix cache 检查。若复现，记录相同 token 输入下首个分歧位置、层和专家向量，再跟踪生成 token 的后续差异。PR 的 token 和路由数组严格一致性要求保留。

## 专家数据验收

对每个成功请求立即保存原始响应，随后使用 base64 解码和 np.load(allow_pickle=False) 读取路由数组。解码失败和 HTTP 500 的原始响应同样落盘。

- 数组形状为 `(P + G - 1, 40, 6)`，P 使用服务端实际 prompt token 数，G 为实际生成 token 数。
- 最后一个采样 token 尚未执行模型 forward，对应行数采用上述减 1 规则。
- 数组 dtype 为整数，逻辑专家 ID 范围为 0–31，每个 token、每层 6 个专家互异。
- worker 采样数据与 HTTP 路由在请求 ID、token 位置、层和 DP rank 上对齐。
- 保存 expert_remap.json 的映射信息，报告逻辑 ID 与原专家 ID 的关系。
- 单请求、双请求和重复运行的完整专家数据有效，服务健康，任务日志中无本项断言、HTTP 500 或设备执行错误。

## 交付内容

| 文件 | 内容 |
|---|---|
| scripts/start_server.sh | 完整环境变量和真实服务端启动命令 |
| scripts/validate_routing.py | 单请求、并发、缓存与生命周期验证客户端 |
| patches/ | 最小修复与针对性组件回归 |
| runs/<run_id>/requests/ | 每个请求的原始输入、HTTP 状态和原始响应 |
| runs/<run_id>/experts/ | 完整 routed_experts.npy、逐 token/层专家 JSON 和映射 |
| runs/<run_id>/routing_summary.json | shape、dtype、范围、行数、唯一性、对齐及一致性结果 |
| runs/<run_id>/server.log | 服务端日志与异常前后上下文 |
| report.md | 根因、修改、命令、真实专家预览、完整文件链接及逐项验收结果 |

## 资源规则

沿用本日资源约定。北京时间 12:00 后出现他人的 NPU 作业时，停止本实验带运行标记的进程并释放设备，保存当前结果。时间使用客户端基准与校正后的服务器时钟。每次服务结束核对本实验进程与 NPU 占用。

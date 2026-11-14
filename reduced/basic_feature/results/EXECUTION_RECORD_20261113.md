# DeepSeek-V4.1 BF16 开箱验证报告（待执行模板）

目录：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature`。仅使用指定32专家/关闭Engram模型与26.1单机。2026-11-13已在26单机开始执行；最新实测见总表和下方执行记录，静态交付验证见evidence/。

## 环境与版本

| 字段 | 值 |
|---|---|
| 执行人/起止时间 | 待填 |
| 容器 | c00954197_vllm_ascend_1108 |
| vLLM / Ascend commit | 5df63a7203cd / 28e9e97ba254；执行时复核 |
| 模型 | /mnt/share/g00815994/dsv41/DeepSeek-V4.1-Flash-top32-bf16-runtime-codex-20261030 |
| 真实dtype/专家/Engram | 待填preflight证据；配置预期BF16/32/关闭 |
| CANN/驱动/Torch/TorchNPU | 待填 |
| 设备空闲窗口/CPU内存 | 待填 |
| 权重指纹/专家映射 | 待填 |

## 结果总表

| 原始需求ID | 功能 | 结果 | 实验记录 |
|---|---|---|---|
| T01 | BF16推理基础功能及开箱性能 | 服务/请求链路可用，输出质量失败 | 见下方T01 |
| T02 | 并行切分EP/TP | EP链路已运行；TP2算子失败 | 见下方T02 |
| T03 | CP功能及精度 | PCP待执行；DCP2前置阻塞 | 见下方T03 |
| T04 | SP功能及精度 | 前置阻塞：TP>1算子限制 | 见下方T04 |
| T05 | PP功能及精度（低优先级） | 未执行 | 见下方T05 |
| T06 | 图模式及精度 | 图捕获/回放与请求链路通过；质量不合格、精度未验收 | 见下方T06 |
| T07 | 异步调度及精度 | 未执行 | 见下方T07 |
| T08 | routing replay返回专家 | 失败：双请求routing capture断言 | 见下方T08 |
| T09 | chunk_prefill | 未执行 | 见下方T09 |
| T10 | sleep level-1 | 通过：5轮恢复一致性和HBM采样检查 | 见下方T10 |
| T11 | sleep level-2 | 失败：两次首轮恢复均token不一致，证据已保存 | 见下方T11 |
| T12 | HCCL/IPC权重更新 | 未执行 | 见下方T12 |
| T13 | M级序列功能及精度 | 未执行 | 见下方T13 |

## T01 BF16推理基础功能及开箱性能

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F01_basic.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：功能打通，无算子错误；性能记录TTFT/TPOT/吞吐，无预设达标门槛。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F01_basic | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T02 并行切分EP/TP

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F02_ep.json`
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F02_ep4.json`
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F03_tp2.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：EP和TP分别判定，双请求无乱码；TP2前置与容量明确。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F02_ep | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F02_ep4 | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F03_tp2 | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T03 CP功能及精度

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F04_pcp1_control.json`
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F04_pcp2.json`
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F04_dcp2.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：PCP、DCP分别记录；DCP当前前置阻塞，不属于证据豁免。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F04_pcp1_control | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F04_pcp2 | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F04_dcp2 | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T04 SP功能及精度

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F05_sp.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：依赖TP2通过；SP路径实际生效，与F03对照。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F05_sp | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T05 PP功能及精度（低优先级）

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F06_pp2.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：当前无Engram模型PP待审查；不直接套用完整模型限制。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F06_pp2 | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T06 图模式及精度

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F07_graph.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：FULL_DECODE_ONLY实际捕获回放，双请求无乱码，与eager对照。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F07_graph | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T07 异步调度及精度

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F08_async.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：异步实际启用，双请求及长短混合正常，与同步对照。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F08_async | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F08_mixed_control / F08_mixed_async | 待填 | 同名configs/cases配置，check_inference.py | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T08 routing replay返回专家

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_routing.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F09_routing.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：路由非空、shape/top-6/专家ID和token/层对应正确。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F09_routing | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T09 chunk_prefill

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F10_control.json`
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F10_chunk.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：16K输入实际拆chunk，与非chunk同输入对照。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F10_control | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F10_chunk | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T10 sleep level-1

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_sleep.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F11_sleep1.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：5轮sleep/wake/reload后token完全一致，无明显显存增长。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F11_sleep1 | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T11 sleep level-2

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_sleep.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F12_sleep2.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：先恢复weights、reload、恢复KV；5轮一致性与显存。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F12_sleep2 | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T12 HCCL/IPC权重更新

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_weight_update.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F13_hccl.json`
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F13_ipc.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：两种接口单独结论；权重A→B→A逐rank变化/恢复，恢复输出一致。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F13_hccl | 待填 | 待填 | 待填 | 待填 | 未执行 |
| F13_ipc | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## T13 M级序列功能及精度

- 执行脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_long_context.py`
- 统一入口：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/run_case.sh`
- 配置：
  - `/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/F14_long.json`
- 公共配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/common.json`
- 启动脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/serve.sh`
- 判定标准：精确1,000,000输入+1024输出，无OOM/截断，人工检查输出/检索。
- 启动/执行命令：按指南替换CASE和RUN；将最终命令完整复制至此：待填。
- 结果：**未执行**。若case含blocked，先完成前置审查；不得写通过。

| 子项/case | RUN绝对路径 | 生效配置/算子路径证据 | 输出/数值检查 | 性能/内存 | 结论 |
|---|---|---|---|---|---|
| F14_long | 待填 | 待填 | 待填 | 待填 | 未执行 |

失败或不适配原因：待填。支撑链接、对应提交和适用范围：待填（参考指南第8节；不能将待验证写成不支持）。

原始日志/响应/路由/曲线完整路径：待填。复测命令及结果：待填。审核人/日期：待填。

## 组合回归

配置：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/configs/cases/C01_combined.json`；脚本：`/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_inference.py`。单项通过后执行，结果：未执行。

## 总体结论

执行中，按总表逐项判定；EP/TP、PCP/DCP等子项分开报告，不混计通过数量。

当前正在执行，最新结果见总表与2026-11-13执行记录；未完成项不计通过。裁剪模型结果不推广为完整模型精度/性能。

## 2026-11-13 执行记录（26单机）

F01首次启动：失败于模型构造，KeyError:16；脚本未显式设置block-size，已按同版本官方模型示例补齐128。失败RUN：results/node26_F01_basic_20261113T092803。DP/EP rank0..7初始化已完成；尚未推理，不能计功能通过。下一轮仅修改block-size。

第二轮 F01（block-size128）：服务启动成功，4个单请求及2个并发请求均返回；但中文/英文/代码/算术均出现明显重复循环，17+25未输出42。结论：启动/请求链路跑通，输出质量不通过；不得作为精度golden。RUN：results/node26_F01_basic_20261113T093344，原文见inference.json。后续增加chunked_prefill=true单变量对照F01_chunk_supported。

第二轮性能：512输入/128输出，10请求，并发2，成功10失败0；输出10.85token/s，mean TTFT1992.75ms，mean TPOT169.91ms。随机请求、benchmark采用当时服务默认temperature（未显式固定），只作开箱观察值；质量退化，不判验收通过。详细数据见同RUN benchmark.log与benchmark结果JSON。

F01_chunk_supported对照完成：4个单请求与关闭chunk基线token完全相同；2个并发请求中1个相同、1个不同（以comparison.json为准），仍重复退化。短输入下开关不是退化唯一原因。此项不是16K真正分chunk验收。继续测试独立通路，但质量/精度不计通过。

F03 TP2/DP4/EP8：失败，真实BF16权重加载完成（每rank约20.5GiB），预热算子报 N1 must be 64 (the only supported query-head count), got 32。RUN：results/node26_F03_tp2_20261113T095042。本版本BF16 TP2受query-head数量限制；不能宣称所有版本/任意TP均不支持。F05 SP依赖TP>1，记前置阻塞；DCP2配置依赖TP2，同样前置阻塞。

F09首次配置校验失败：AuxOutput Connector requires prefix caching；RUN results/node26_F09_routing_20261113T095539。仅为路由case启用prefix_caching后重试。

F09路由返回第二轮：单请求HTTP200，双请求HTTP500，服务RoutedExpertsCapturer断言 topk_ids batch dim 2 (expected 9,1,16,or1 for dp_rank=7,tp_size=1)。结论失败；完整堆栈见results/node26_F09_routing_20261113T095702/server.log，客户端见validation.log。本轮客户端在全部请求完成后才保存响应，故先前单请求原始响应未落盘，不能据此宣称路由内容校验通过。

F11 sleep level-1：5/5轮完成，温度0输出token与同配置初始基准完全一致。8卡HBM最后恢复平台对比初始平台最大增长80MiB，未见明显持续增长；API首引擎allocated为79,462,642,688字节，未增长。接口RPC仅返回首引擎内存，不能称全部rank的allocator已逐个校验；8卡依据为主机npu_watch.jsonl及hbm_periods.json。结论：本配置sleep/reload一致性与采样显存检查通过，模型语义质量仍未通过。RUN：results/node26_F11_sleep1_20261113T100448。

F12 level-2首轮：第一次sleep/wake/reload后Token mismatch，5轮验收未完成，结论失败。RUN results/node26_F12_sleep2_20261113T101611。为补全原始token证据，修改客户端使基准/每轮响应在断言前落盘，并执行相同首轮单次复现F12_sleep2_repro；不修改数值阈值或引擎。

IPC待执行方案更新：configs/cases/F13_ipc.json，主脚本/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/scripts/check_ipc_multi.py；4卡同设备sender，packed2GiB，不再受原单卡脚本映射限制。尚未运行，不能记通过。

F12补证复现：仍失败，恢复前后首个输出token即不同。原始响应：results/node26_F12_sleep2_repro_20261113T102258/sleep_baseline.json与sleep_cycle_1.json；比较：comparison.json。未放宽容差，不运行剩余4轮以冒充5轮验收。

F07图模式：8rank实际ACL graph replay日志；4单请求+2并发返回，无算子错误。与eager比较serial=[true,true,true,true]、parallel=[false,true]，输出仍重复退化。请求耗时serial约4.01–4.09s，parallel约4.40–4.44s（请求级观测，不替代benchmark）；图功能通路通过，质量失败、精度未验收。RUN results/node26_F07_graph_20261113T102953，比较见comparison.json。

F08异步：4单请求+2并发完成，无算子异常；与同步比较{'serial': [True, True, True, True], 'parallel': [True, True]}，仍重复退化。通路通过，质量失败，数值精度未验收。RUN results/node26_F08_async_20261113T103545。

PCP1对照首次配置校验失败：关闭chunk时max_num_batched_tokens<max_model_len。RUN results/node26_F04_pcp1_control_20261113T104345。修正PCP1/2成对配置为max_model_len=batched_tokens=32896，输入仍32768输出128；同时修正F10对照为16512（输入16384+128）及同类混合请求配置。本次是脚本配置修正，非模型功能结论。

F04 PCP1对照：32768输入+128输出，4单请求及2并发均完成，无OOM/算子异常；输出重复/偏离。RUN results/node26_F04_pcp1_control_20261113T104608，作为PCP2同输入通路对照，不作为质量合格golden。

F04 PCP2/DP4/TP1：32K的4单请求+2并发完成，无OOM/算子报错。输入IDs与PCP1相同，输出比较{'serial': [False, False, False, False], 'parallel': [False, False]}；输出仍重复/偏离，精度未验收。存在整数ArgSort转AiCPU警告，不能宣称全AiCore执行。RUN results/node26_F04_pcp2_20261113T105359。

F10真实分块：16384输入、调度预算2048、prefix cache关闭；4单请求+2并发完成，无OOM/算子异常。与非分块相同输入比较{'serial': [False, False, False, False], 'parallel': [False, False]}。有效chunk开关/预算见server.log，scheduler.py以max_num_scheduled_tokens（默认batched_tokens）约束每次调度；metrics_mid*.txt的iteration_tokens口径按完成prefill聚合，不据此声称记录了精确分块次数。输出仍退化，质量/精度未验收。对照RUN results/node26_F10_control_20261113T110120；分块RUN results/node26_F10_chunk_20261113T110941。

F13 HCCL首次真实传输：推理端layer10附近OOM，24MiB申请失败，已allocated86.72GiB/free18.49MiB，未完成更新。保持暂停并终止本次进程；RUN results/node26_F13_hccl_20261113T111653。仅将memory_utilization从0.75降到0.50为reload腾出空间后重试，原checkpoint不变。

HCCL 0.50显存预留重试：启动失败，可用KV cache=-0.20GiB（results/node26_F13_hccl_20261113T112328）。按实测profile开销调整到0.55作最后一次完整更新容量尝试；IPC同4卡模型配置也采用0.55，避免重复已知无KV容量配置。

IPC最终待执行布局：DP8/TP1/EP8 + 每张卡同设备sender（8个），memory_utilization=0.45，packed buffer2GiB。IPC无需额外空闲发送卡，利用已实测DP8每rank约27.5GiB而DP4约38.05GiB的差异，给重载保留空间；不照搬HCCL必须预留单独sender卡的4卡布局。脚本按配置卡数自动启动。

HCCL最终0.55重试：服务启动成功，传输推进至约layer19后推理端再次OOM（24MiB申请失败，allocated85.30GiB/free约21MiB）。完整B更新未完成，保持暂停并终止；A→B→A未完成，不计通过。RUN results/node26_F13_hccl_20261113T112758。结论仅限本次TP1/DP4/EP4、单独第5卡sender、当前版本和完整checkpoint传输；不能扩大为所有HCCL配置不支持。

IPC第一次：推理服务启动成功，torchrun --standalone将rendezvous地址公布为容器主机名c00954197-a5-26，而容器内getaddrinfo实测解析失败；发送端未进入权重更新。RUN results/node26_F13_ipc_20261113T113338。修正启动器为static rendezvous + 127.0.0.1 +动态空闲端口，具体命令落盘ipc_launcher_command.json；不修改全局hosts，重新执行。

IPC第二轮：rendezvous成功，进入start/update阶段，但backend=ipc选择上游CUDA IPCWeightTransferEngine，packed_tensor.py:471调用torch.cuda.get_device_properties，报Torch not compiled with CUDA enabled。核查当前vllm_ascend/distributed/weight_transfer/__init__.py仅注册npu_ipc，未注册ipc别名；与该提交RL文档描述不一致。RUN results/node26_F13_ipc_20261113T114035。修正配置显式backend=npu_ipc并保留序列化环境开关后重试，不修改框架源码。

IPC最终显式npu_ipc轮：8卡DP8/EP8、每卡sender、packed2GiB、memory_utilization0.45；已进入NPU更新路径，但推理端重载OOM（24MiB申请失败，NPU5 allocated81.43GiB/free5.63MiB，reserved85.71GiB；同卡sender另占显存）。B完整更新未完成，A→B→A未完成，不计通过。RUN results/node26_F13_ipc_20261113T115009。真实NPU路径已排除前两轮地址和CUDA后端误选问题；最终结论为本配置完整IPC更新容量失败，不扩大为所有IPC方案不支持。

F06 PP2/DP4/TP1实际失败：PP1阶段hidden_states为None，model.py:1592调用unsqueeze异常；PP0阶段PPMissingLayer缺少layer_idx。RUN results/node26_F06_pp2_20261113T115905/server.log。监测器completion误记YIELDED_TO_OTHER_JOB：相邻npu_watch采样证明相关PID先全部属于本实验，模型报错后退出，/proc消失被误判；不存在已确认的他人竞争。已补充PID存活/僵尸判断与start-time身份缓存，防止PID复用及退出竞态；检查见evidence/guard_exit_race_check.txt。PP结论失败，不是资源阻塞。

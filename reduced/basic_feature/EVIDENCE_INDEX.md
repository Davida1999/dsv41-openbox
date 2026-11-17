# 裁剪模型专项证据索引

本索引指向原始RUN。结论未被完整模型普通推理覆盖。

## T01 BF16基础推理、开箱性能

启动/请求通路通过；输出质量不合格

TP1/DP8/EP8，无算子异常，4单请求+2并发完成；中文/英文/代码/算术均存在重复退化，17+25未正常回答42。不得以HTTP200认定完整验收通过。

- [node26_F01_basic_20261113T093344](results/node26_F01_basic_20261113T093344/)
- [node26_F01_chunk_supported_20261113T094347](results/node26_F01_chunk_supported_20261113T094347/)

## T02 EP/TP并行

EP8通路已运行；TP2失败

F02_ep与F01配置等价，复用基线真实EP8 rank/请求证据。TP2/DP4权重加载成功，预热时mixed_quant_sparse_flash_mla只接受N1=64，而TP2为32；不是原MXFP scale reshape问题。

- [node26_F01_basic_20261113T093344](results/node26_F01_basic_20261113T093344/)
- [node26_F03_tp2_20261113T095042](results/node26_F03_tp2_20261113T095042/)

## T03 CP功能和精度

PCP2通路通过，6/6输出与PCP1不同；DCP2阻塞

32768输入，PCP1与PCP2输入IDs完全相同；单请求及双请求都完成，无OOM。PCP2存在整数ArgSort转AiCPU警告。输出仍退化，不能给出精度通过结论。所规划DCP2依赖TP2，未执行。

- [node26_F04_pcp1_control_20261113T104608](results/node26_F04_pcp1_control_20261113T104608/)
- [node26_F04_pcp2_20261113T105359](results/node26_F04_pcp2_20261113T105359/)

## T04 SP功能和精度

前置阻塞，未执行

当前SP要求TP>1且MoE启用EP；本模型A5注意力路径固定N1=64，TP2实测失败。未将SP开关在TP1下无效开启当作通过。


## T05 PP功能（低优先级）

PP2失败

关闭Engram后实际尝试PP2/DP4/TP1。预热中PP1的hidden_states为None，调用unsqueeze失败；PP0另报PPMissingLayer缺少layer_idx。不是用完整Engram模型PP=1的限制替代实验。

- [node26_F06_pp2_20261113T115905](results/node26_F06_pp2_20261113T115905/)

## T06 图模式及精度

图捕获/回放通过；一致性部分不符、质量未过

FULL_DECODE_ONLY实际捕获，8个rank均记录ACL graph replay active。6请求完成；对eager单请求4/4一致，并发1/2一致。仍重复退化。仅覆盖decode图，不宣称prefill图通过。

- [node26_F07_graph_20261113T102953](results/node26_F07_graph_20261113T102953/)

## T07 异步调度及精度

运行与6请求greedy对照一致；基础质量未过

日志确认async scheduling实际启用；4单请求+2并发均完成，输出token与同步基线6/6一致。此为有限样例一致性，不等同完整模型精度。

- [node26_F08_async_20261113T103545](results/node26_F08_async_20261113T103545/)

## T08 routing replay返回专家

失败：双请求capture断言/HTTP500

AuxOutput要求prefix caching，修正后进入运行。单请求HTTP200，双请求报unexpected topk_ids batch dim 2（expected 9,1,16,or1）。该轮客户端未提前落盘单请求响应，不能宣称专家内容校验通过；之后已改为逐请求保存。只验返回专家，不包含训练侧强制路由replay。

- [node26_F09_routing_20261113T095702](results/node26_F09_routing_20261113T095702/)

## T09 chunk_prefill

分块通路通过；输出与非分块不同、质量未过

16384输入，2K调度预算，prefix cache关闭；4单请求+2并发完成。与非分块相同输入的6个输出均不同。有效预算和scheduler约束支持实际分块判断；iteration_tokens指标按完成prefill聚合，不能作为精确8次分块计数。

- [node26_F10_control_20261113T110120](results/node26_F10_control_20261113T110120/)
- [node26_F10_chunk_20261113T110941](results/node26_F10_chunk_20261113T110941/)

## T10 sleep level-1

通过：5轮一致性和采样显存检查

5/5轮sleep/wake/reload后temperature=0输出token与同配置初始基准完全一致。8卡HBM最终相对初始最大增长80MiB，未见明显持续增长。API仅返回首引擎allocator统计，其allocated不变；其余卡依据主机HBM采样。此通过不代表基线语义质量合格。

- [node26_F11_sleep1_20261113T100448](results/node26_F11_sleep1_20261113T100448/)

## T11 sleep level-2

失败：首轮恢复输出不一致，已复现

严格按weights wake→reload_weights→kv_cache wake执行。两次首轮均Token mismatch；补证复现从第1个输出token即不同，原始响应与comparison.json已保存。未完成5轮，不计显存稳定性通过。

- [node26_F12_sleep2_20261113T101611](results/node26_F12_sleep2_20261113T101611/)
- [node26_F12_sleep2_repro_20261113T102258](results/node26_F12_sleep2_repro_20261113T102258/)

## T12 HCCL/IPC权重更新

两条完整更新路径均未通过（OOM）

HCCL TP1/DP4推理+第5卡sender，0.75和0.55更新时OOM；0.50无KV容量。IPC DP8/EP8+8个同卡sender，packed2GiB、0.45、显式npu_ipc后仍更新OOM。完整B未完成，A→B→A未完成；不能用部分传输或初始哈希认定更新成功。

- [node26_F13_hccl_20261113T112758](results/node26_F13_hccl_20261113T112758/)
- [node26_F13_ipc_20261113T115009](results/node26_F13_ipc_20261113T115009/)

## T13 M级序列功能和精度

容量通过；检索 0/3，检索/语义未通过

前置131072/262144/524288输入各输出32作为容量探针；最终严格1000000输入+1024输出。强制输出长度用于容量验证，不等同自然生成1K连贯回答或完整精度验收。

- [node26_F14_long_20261113T120549](results/node26_F14_long_20261113T120549/)

## 2026-11-17 CANN930 修复复验

| 项目 | 报告与关键证据 |
|---|---|
| 当前环境 | [CANN930 环境](../../environment/cann930_environment_20261117.md) |
| Sleep level-2 | [报告](../cann930_20261117/sleep_level2/REPORT.md)、[五轮输出](../cann930_20261117/sleep_level2/five_cycle_outputs.txt)、[逐卡 HBM](../cann930_20261117/sleep_level2/memory_five_cycles.csv) |
| HCCL | [完整回环逐端计数](../cann930_20261117/weight_update/runs/transfer_20261117_03/hccl_producer/full_transfer_audit.json) |
| NPU IPC | [完整回环逐端计数](../cann930_20261117/weight_update/runs/transfer_20261117_03/npu_ipc_producer/full_transfer_audit.json) |
| 当前验证边界 | [状态](../cann930_20261117/weight_update/validation_status.json)、[IPC 待复验步骤](../cann930_20261117/weight_update/pending_validation.md) |

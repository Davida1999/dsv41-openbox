# E23 no_stack 单卡硬件性能分析

日期：2026-11-12（北京时间）。配置：torch_profiler_with_stack=false，delay_iterations=1536，max_iterations=10。

## 结论

25.1 / device0 的计算算子耗时加权经验达成率为 **100.4878%**，超过约定目标80%，差值20.49个百分点。该结论按 performance-and-evaluation.md 的默认经验上限0.8计算；不是硬件理论 FLOPS 利用率、MFU 或端到端利用率，也不能代表其他15张卡。

超过100%合法：选用每次调用最大的管线 ratio 除以0.8，结果不截断；短于10μs的调用按其耗时权重全额计分。最大的ratio可能来自 scalar/MTE，而非矩阵乘 MAC，因此高分不等于计算峰值接近100%。未提供芯片/shape/dtype匹配的算子专属经验上限，沿用skill默认值。

## 数据与窗口验收

- 两节点各8个rank，共16份原始数据与解析产物；离线解析退出码均为0。
- 所有16个rank均有10次 MODEL_EXECUTE，每次执行恰有40次 mixed_quant_sparse_flash_mla；模型config的num_hidden_layers=40。覆盖该卡执行的40层 decode 图；没有只取首层或按层数重复放大。
- 选择node25、device0、DP0/PCP0/EP0，CSV Device_id全部为0；metadata记录PCP组0–7、EP组0–15。模型为完整BF16资产，CPU Engram offload；TP1/PCP8/DP2、EP16。每卡仅承担其EP专家分片和PCP工作份额，不能当作单卡部署完整模型。
- 稳定decode证据：服务日志窗口内每DP副本Running=32、Waiting=0；TransposeBatchMatMul输入为32×1×4096，PCP本地投影输入为4×5120；图执行重复10次。
- delay_iterations=1536来自启动配置和worker采集机制，trace不含从第1步起的全程计数，不能独立重建绝对步号；没有把1536解释成精确输出token位置或6144的中点。
- 客户端64成功、0失败，393216输出token（每请求6144），132992输入token（每请求2078，含模板额外30token）。总耗时183.40s，输出吞吐2144.04 token/s，Mean TTFT 6762.26ms，Mean TPOT/ITL 28.69ms。这些为带profiling负载的端到端结果。
- TorchNPU元数据：with_stack=false、with_modules=false；Level1 / ACL_AICORE_PIPE_UTILIZATION。

## 测算口径及结果

输入25320条kernel记录；纳入21970条计算调用，排除3350条。排除包括2480条通信/拷贝、800条MoeDistribute通信、70条非计算任务。

| 指标 | 结果 |
|---|---:|
| 计算调用累计耗时 | 218.097 ms |
| 长调用贡献（≥10μs） | 71.0666% |
| 短调用贡献（<10μs） | 29.4212% |
| 总经验达成率 | 100.4878% |
| 优化候选调用 | 745 |

每次调用w=duration/计算调用总duration；长调用贡献w×max_ratio/0.8，短调用贡献w。各调用分别判断后再聚合。ratio输入单位为fraction（0–1），耗时单位μs。总耗时是调用耗时之和，不等于时间线墙钟时间。

## 优化候选筛选

筛选条件为duration>20μs且max_ratio<0.6。以下是候选而不是已证明的根因，不因全局得分达到80%而忽略。

| 算子 | 全部调用数 | 全部调用耗时ms | 计算耗时占比 | 重点调用数 |
|---|---:|---:|---:|---:|
| `mixed_quant_sparse_flash_mla` | 400 | 14.657 | 6.72% | 305 |
| `aclnnTransposeBatchMatMul_TransposeBatchMatMul_TransposeBatchMatMul` | 400 | 8.860 | 4.06% | 400 |
| `_quantize_mxfp4_indexer_kernel` | 80 | 1.266 | 0.58% | 30 |
| `QuantLightningIndexer` | 40 | 0.576 | 0.26% | 10 |

优先检查Sparse MLA的访存/标量路径和输入形状，其次检查TransposeBatchMatMul的小矩阵与布局转换开销；Indexer两项占比更低，后续结合有模块信息的trace定位调用来源。这里没有修改模型或内核，也没有宣称优化已实现收益。

## 限制与后续

通信、CPU调度、Engram访问等待、设备空闲不纳入本经验评分；需另看时间线才能判断整体瓶颈。默认经验上限未经本型号每个算子逐一标定，因此达到本评分目标不能替代理论峰值百分比。尚未执行with_stack=true对照轮次，本报告不包含其结论。

此前自动解析在daemon worker中失败，原始数据完好；已使用独立进程analyse完成解析，无需重新采集。

## 归档

- 窗口证据：../window_audit.json；服务日志证据：../server_window_evidence.log。
- 原始CSV：../node25_device0/kernel_details.csv。
- 硬件详细证据：hardware_efficiency.json；带公式表格：hardware_efficiency.xlsx（已完成导出，Excel公式与Python计算一致；四表预览已检查）。
- node25原始目录：/workspace/experiments/E23_profile/no_stack/profiling/node25/run_20261112T105718Z_RCkTi9/
- node26原始目录：/workspace/experiments/E23_profile/no_stack/profiling/node26/run_20261112T105714Z_pETeXj/
- CSV SHA256：d2f421f0b4273c943bd0fd64b85f9dcf76ffdaa6879f725278e9289468746c5f

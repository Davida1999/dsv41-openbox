# E23 with_stack 配置轮次：单卡硬件性能分析

日期：2026-11-12（北京时间）。使用同一完整BF16模型、TP1/PCP8/DP2/EP16、Engram CPU offload；选node25/device0/DP0/PCP0/EP0，与第一轮一致。

## 结论

按 performance-and-evaluation.md 的默认经验口径，加权达成率为 **100.7164%**，达到80%目标。计算仅针对所选卡，不能外推其他15张卡；不代表硬件理论FLOPS利用率、MFU、端到端吞吐利用率或实测加速比。默认0.8经验上限未按各个A5算子shape/dtype单独标定。

CLI torch_profiler_with_stack=true；实际TorchNPU元数据with_modules=true、with_stack=false。这是当前Ascend wrapper映射的行为，因此本轮是CLI配置对照和模块信息采集，不能称为完整Python调用栈采集。未修改这一映射。

## 采集与验收

- 服务于21:09:14进入启动流程，21:40:53确认就绪；21:41:05下发客户端，客户端实际命令开始记录为21:41:44。
- delay_iterations=1536、max_iterations=10；纯文本random，2048输入目标/6144输出目标、64请求/总并发64，服务端每DP32。
- 16个rank解析产物齐全，所有rank均有10次MODEL_EXECUTE，每步40次Sparse MLA，与config中40层一致。设备0映射通过CSV Device_id与rank metadata确认。
- 窗口为重复decode图；对应32请求的全局投影和PCP本地切分。延迟1536来自启动配置和worker计数机制，trace不能独立重建之前1536步，也不能把其等同输出token数。
- 21:42:36确认16份原始产物后，按用户允许的早停策略结束服务和客户端；21:42:58清理完成。客户端未跑完6144输出，不提供完整请求成功率或完整压测吞吐，亦不标作完整长输出压测成功。
- 解析在独立进程完成，node25约21:44:20完成，node26约21:44:57完成；避免daemon worker自动解析限制。
- NPU不再由本轮服务占用；原始数据、解析结果保留远端。

## 单卡测算

共25320条kernel记录，计算调用21970条；排除3350条（2480通信/拷贝、800 MoeDistribute通信、70非计算任务）。输入ratio为fraction，duration为μs。

| 指标 | 结果 |
|---|---:|
| 计算调用累计耗时 | 219.723 ms |
| 长调用贡献 | 71.4876% |
| 短调用贡献 | 29.2287% |
| 加权经验达成率 | 100.7164% |
| 重点优化调用 | 759 |

每次调用w=duration/计算总duration；duration≥10μs贡献w×max_ratio/0.8，短调用贡献w。通信、拷贝、空闲和CPU调度不计分；ratio可来自scalar、MTE等管线，不专指MAC。结果不截断，因此允许超过100%。

## 优化候选

条件：duration>20μs且max_ratio<0.6。候选不是已定位根因，也没有执行内核修改。

| 算子 | 全部调用数 | 全部耗时ms | 计算耗时占比 | 重点调用数 |
|---|---:|---:|---:|---:|
| `mixed_quant_sparse_flash_mla` | 400 | 14.513 | 6.61% | 315 |
| `aclnnTransposeBatchMatMul_TransposeBatchMatMul_TransposeBatchMatMul` | 400 | 8.885 | 4.04% | 400 |
| `_quantize_mxfp4_indexer_kernel` | 80 | 1.255 | 0.57% | 30 |
| `QuantSparseLightningIndexer` | 40 | 0.762 | 0.35% | 4 |
| `QuantLightningIndexer` | 40 | 0.594 | 0.27% | 10 |

两轮主要候选一致：Sparse MLA和TransposeBatchMatMul优先；Indexer占比较小。建议先对照时间线和输入形状定位访存/标量路径及布局转换，再以相同负载验证优化收益。当前任务只完成分析，没有宣称优化收益。

## 文件

- hardware_efficiency.xlsx：带逐调用公式的Summary/Kernels/Operators/Excluded四表；已通过Excel/Python重算一致性、公式错误扫描和四表预览校验。
- hardware_efficiency.json：完整计算证据；../window_audit.json：16rank窗口和metadata证据。
- ../node25_device0/kernel_details.csv：单卡原始CSV；SHA256：3d246a975571e54fa486c4fbd563c4bb1f116dc23b636bc8fafe01d5e497b518。
- ../node25_device0/source.json：远端原始rank目录；../run_logs.tar.gz：本轮服务和客户端日志。

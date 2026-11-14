# E23 两轮 profiling 硬件性能对比

两轮已使用相同模型、拓扑、负载配置、delay=1536、10步采集；选同一node25/device0，以同一skill公式独立测算。no_stack为主性能结论，with_stack配置为对照。每轮16个rank均通过10步×40层Sparse MLA调用核验。

| 指标 | CLI with_stack=false | CLI with_stack=true |
|---|---:|---:|
| 原生with_stack | false | false |
| 实际with_modules | false | true |
| 计算调用数 | 21970 | 21970 |
| 排除调用数 | 3350 | 3350 |
| 计算调用累计耗时 | 218.097 ms | 219.723 ms |
| 经验达成率 | 100.4878% | 100.7164% |
| 是否达到约定80%目标 | 是 | 是 |
| 重点优化调用数 | 745 | 759 |
| 客户端是否完整结束 | 64成功/0失败，393216输出token | 采集完成后按授权主动终止 |

经验得分差值为0.2286个百分点；计算调用累计耗时差约0.745%。只有各一次采集，不能据此判定性能提升或量化with_stack的稳定开销，更不能以得分倒数推算加速比。

## 如何理解100%以上

skill以每个计算调用的最大管线ratio除以默认经验上限0.8，短于10μs的调用按权重全额记分，不截断结果。分数属于经验计算指标，不是理论FLOPS利用率；其中还可能由scalar或MTE管线主导。通信、空闲、调度、Engram主机访问等待没有计入。因此结论为“达到约定的经验评分目标”，不是“超过硬件理论极限”。

## 优化方向

两轮均优先关注Sparse MLA、TransposeBatchMatMul；Indexer相关算子次之。已提供逐调用筛选和耗时占比，无需重复再做一份同口径单卡评分。若目标是端到端速度，应另看通信/CPU/Engram等待和跨rank不均衡，不能仅用本得分判断。

## 交付与范围

- [第一轮报告](no_stack/hardware_report/hardware_report.md)
- [第二轮报告](with_stack/hardware_report/hardware_report.md)
- 每轮相邻目录有硬件Excel和JSON，包含计算明细、优化筛选、排除证据。
- 当前Ascend wrapper的CLI with_stack开关仅映射到with_modules；本任务没有采到原生Python调用栈。若需要原生调用栈，应单独修改wrapper并评估开销，本次未擅自改变实现。
- 第二轮采集后已终止本任务的服务和客户端，保留容器、模型和数据。

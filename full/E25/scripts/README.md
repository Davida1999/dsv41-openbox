# E25：CANN 9.3.0、完整BF16模型、Engram CPU offload

本目录只准备实验配置，不自动启动容器、服务或性能测试。

## 服务配置

| 项目 | 配置 |
|---|---|
| 容器 | 25、26上的c00954197_vllm_ascend_cann930 |
| 完整模型 | /mnt/share/c00954197/workspace/dsv41-flash-bf16（沿用E23） |
| 模型结构 | 40个文本层、384个路由专家、每token选6个；保留视觉模块 |
| 精度 | BF16，hf-overrides设置quantization_config=null |
| Engram | 原模型第1、14层启用；cpu_offload=true、dp_shared_memory=true |
| 并行 | TP1 / PCP8 / DP2，EP开启，每节点DP1 |
| API模型名 | dsv41-bf16-cann930 |
| 主节点 / 子节点 | 177.11.25.1 / 177.11.26.1，子节点headless、DP start-rank=1 |
| Gloo / HCCL | enp39s0f0 / enp35s0f2 |
| API / DP RPC端口 | 8000 / 12346 |
| 最大上下文 / 批处理tokens | 40960 / 2048 |
| 每实例max-num-seqs | 32 |
| gpu-memory-utilization | 0.96 |
| 调度 | no-async-scheduling；保留prefix caching、chunked prefill |
| 图模式 | FULL_DECODE_ONLY |

直接读取E23完整模型，不建立裁剪模型视图、不过滤Engram索引、不重写原权重。Engram按E23方式放在注册的主机内存中，每节点DP共享表；主机内存注册和权重读取可能较慢。

选择关闭async，是因为E24在930环境中只关闭此项后，3轮图文与2轮文本复放完整通过；这仅是规避候选，不代表已证明根因，也不代表本E25完整模型已经验证通过。E24原始脚本保持原样。CLI kv-cache-dtype沿用bfloat16，A5模型后端的实际物理缓存布局仍由实现决定。

## 客户端

所有场景：64请求、总并发64、seed=42、ignore-eos、长度浮动0；每次执行一轮。输入是文本目标长度，聊天模板和图片会增加实际输入token。

| 脚本 | 输入 | 输出/请求 | 全部完整完成时总输出 | 结果目录 |
|---|---|---:|---:|---|
| run_client_mm_1k_2k.sh | 1024文本token + 1张1920高×1080宽图片 | 2048 | 131072 | perf_results_mm_1k_2k/ |
| run_client_text_2k_6k.sh | 2048文本token，无图片 | 6144 | 393216 | perf_results_text_2k_6k/ |
| run_client_mm_4k_32k.sh | 4096文本token + 1张1920高×1080宽图片 | 32768 | 2097152 | perf_results_mm_4k_32k/ |

每轮创建独立run_<时间戳>目录，保存benchmark.log、benchmark.json、command.sh、client_snapshot.sh、exit_code.txt。
验收时同时核对64成功、0失败、完整输出token总数和服务端日志；不能仅依赖退出码0或HTTP 200，因为流式请求异常截断也可能被benchmark计作成功。

## 手动运行

先确认设备使用许可和占用情况，服务端口8000/12346空闲。E24和E25复用这些端口及相同设备，不应同时拉起。

在25宿主机：

```bash
docker start c00954197_vllm_ascend_cann930
docker exec -it c00954197_vllm_ascend_cann930 bash -lc 'bash /workspace/experiments/tp1_dp16_20261029/E25/run_master_mm.sh'
```

在26宿主机另一个终端：

```bash
docker start c00954197_vllm_ascend_cann930
docker exec -it c00954197_vllm_ascend_cann930 bash -lc 'bash /workspace/experiments/tp1_dp16_20261029/E25/run_worker_mm.sh'
```

两端就绪后，确认25的/health返回200、/v1/models包含dsv41-bf16-cann930，再在25容器内选择一种客户端运行：

```bash
cd /workspace/experiments/tp1_dp16_20261029/E25
bash run_client_mm_1k_2k.sh
# 或：bash run_client_text_2k_6k.sh
# 或：bash run_client_mm_4k_32k.sh
```

服务端终端日志保存到server_logs/。不含自动重试、自动清理占用或循环压测；本次未执行推理。

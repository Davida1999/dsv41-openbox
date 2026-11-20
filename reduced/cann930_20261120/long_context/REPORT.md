# CANN930 1M长序列、异步调度与分块预填充报告

日期：2026-11-20。177.42.22.1、CANN930、40层32专家BF16，Engram关闭。

## 验收结果

**本轮受测配置通过。** 实际输入1,000,000 token，最大输出预算1024；HTTP200，148.02秒后自然结束，输出43 token。输入约5%、50%、95%处的三个检索键全部命中，输出可读且编码正常。不是以配置上限代替实测输入长度。[输入清单](runs/long_context_20261120_12/validation/input_manifest.json) · [原始响应](runs/long_context_20261120_12/validation/response.json) · [离线核对](evidence_audit.json) · [标准库复核脚本](scripts/audit_saved_long_context.py)

```text
The secret key for FIRST is K8Z7Q4.  The secret key for SECOND is M2V9P6.  The secret key for THIRD is R5T1X3.<｜end▁of▁sentence｜>
```

## 实测配置

| 项目 | 设置 |
|---|---|
| 模型 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32` |
| 并行 | TP1/DP8/EP8，PP1/PCP1/DCP1 |
| 上下文 / batched tokens / 最大序列 | 1048576 / 8192 / 1 |
| KV / 显存利用率参数 | 显式16 GiB/卡 / 0.80 |
| 执行 | MRV2、async scheduling、chunked prefill、FULL_DECODE_ONLY、TorchCompile=0 |
| BI / SWA bounded replay | 关闭 / 关闭 |
| 输入 | 真实tokenizer编码的chat模板、合成填充、三处检索键和末尾问题；固定100万token |
| 生成 | temperature=0、top_p=1、seed=42、ignore_eos=false、max_tokens=1024 |
| 图像 | 本轮纯文本，image/video限额均为0 |

实际输入远大于8192的单批预算，完成了长输入分块执行；同轮启用异步调度与NPU图。[实际参数](runs/long_context_20261120_12/actual_server_command.sh.gz)

## 统一交付文件

| 类型 | 文件 |
|---|---|
| 服务端 | [完整启动脚本](scripts/start_server.sh) |
| 请求/校验 | [生成与检查客户端](scripts/verify_long_context.py) |
| 请求/输出 | [完整请求](runs/long_context_20261120_12/validation/request.json.gz) · [原始响应](runs/long_context_20261120_12/validation/response.json) · [可读输出](runs/long_context_20261120_12/validation/output.txt) |
| 日志/统计 | [服务日志](runs/long_context_20261120_12/server.log.gz) · [客户端日志](runs/long_context_20261120_12/client.log.gz) · [结果](runs/long_context_20261120_12/validation/summary.json) |
| 资源 | [占用即退出守护](scripts/resource_policy_guard.py) · [释放记录](runs/long_context_20261120_12/release_review.json) |

在相同服务就绪后，可在仓库根目录复现本轮完整token输入：

```bash
gzip -dc reduced/cann930_20261120/long_context/runs/long_context_20261120_12/validation/request.json.gz | \
  curl --noproxy '*' --max-time 3600 http://127.0.0.1:18244/inference/v1/generate \
    -H 'Content-Type: application/json' -H 'X-data-parallel-rank: 0' --data-binary @-
```

本轮为单次合成文本容量与三处定点检索验证，不替代GPQA数据集精度。完整原始日志保留了当时的旧停服竞态；公共关闭顺序和优雅停服随后已在[最新153请求回归](../routing_replay/REPORT.md)中修复并验证，当前交付脚本包含该修复。

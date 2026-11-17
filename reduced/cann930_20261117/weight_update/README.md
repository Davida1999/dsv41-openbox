# DSV4.1 全量权重更新实验交付

主报告为 `REPORT.md`，状态为 `validation_status.json`，后续设备复验步骤为 `pending_validation.md`。

服务器为 `177.42.22.1`，实验目录为 `/mnt/share/c00954197/Workspace/dsv41/basic/weight_update`。HCCL 和 IPC 全量 A→B→A 已通过。最新 IPC 引用计数补丁完成 CPU 验证，NPU 回收效果等待用户安排设备。

| 目录或文件 | 内容 |
|---|---|
| `scripts` | 启动、权重源、真实传输、oracle、接收追踪、回归与待执行复验脚本 |
| `patches` | 正式补丁、修改前后源码、CPU 测试输出、继承 Sleep2 补丁 |
| `diagnostics` | OOM 证据和已撤回的早期字段假设 |
| `environment` | 环境元数据、源码版本、当前 diff |
| `runs/transfer_20261117_01` | 第一次失败的 HCCL 更新及冷加载对照 |
| `runs/transfer_20261117_02` | 早期字段修改后仍失败的 HCCL 更新，后续复用其冷加载对照 |
| `runs/transfer_20261117_03` | 最终通过的 HCCL、IPC 回环和逐端审计 |
| `evidence_audit.json` | 对保存的原始数据进行 CPU 交叉核对的结果 |
| `variant_manifest.json` | A/B 来源、变更方式、字节数和 head 哈希 |
| `SHA256SUMS` | 包内文件校验清单 |

每个 producer 运行目录包含服务端日志、实际服务启动命令、发送端命令及输出、逐端收发计数、HBM CSV 和逐副本双接口请求响应。`validation/B/receiver` 与 `validation/A-restored/receiver` 保存每个接收副本的加载进度。

模型 safetensors 文件需使用服务器已有资产。重新核对本包的原始证据只需 Python 标准库。

```bash
python3 scripts/audit_saved_evidence.py
```

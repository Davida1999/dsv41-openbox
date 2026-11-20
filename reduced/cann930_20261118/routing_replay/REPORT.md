# CANN930 专家路由返回验证报告

日期为 2026-11-18。模型为 40 层、32 专家、Top-6、BF16，Engram 关闭；177.42.22.1、CANN930、MRV2、TP1/DP8/EP8。

## 验证结果

**未通过。** 单请求返回专家数组 `[20,40,6]`，dtype 为 `uint8`，ID 范围 0–31，各 token/层的 Top-6 专家互异。同 DP 双请求触发 `topk_ids` 维度断言及 HTTP 500。

候选补丁补充 TP1 本地填充布局的识别，根据 dispatch metadata 只复制本地有效行。7 项针对性 CPU 回归、8 项仓库原生 CPU 检查通过。修复后 NPU 复验因设备资源占用未启动，不能判定实机修复通过。

## 交付文件

| 类型 | 文件 |
|---|---|
| 服务端脚本 | [启动脚本](scripts/start_server.sh)、[实际启动命令](runs/routing_20261118_02/actual_server_command.sh.gz) |
| 请求与校验 | [校验客户端](scripts/validate_routing.py)、[请求参数](runs/routing_20261118_02/client_command.json) |
| 请求与输出 | [单请求专家 JSON](runs/routing_20261118_02/validation/single_completions_experts.json)、[专家 NPY](runs/routing_20261118_02/validation/single_completions_experts.npy)、[请求及失败响应](runs/routing_20261118_02/validation/) |
| 日志与统计 | [校验汇总](runs/routing_20261118_02/validation/summary.json)、[服务日志](runs/routing_20261118_02/server.log.gz)、[客户端日志](runs/routing_20261118_02/client.log) |
| 修复补丁与 CPU 结果 | [补丁](patches/local_dp_padding_capture.patch)、[专项回归](patches/local_padding_after.log)、[仓库检查](patches/repository_capture_cpu_tests.log) |

修复前文件 SHA256 为 `36c4c7c269b58bb064cf15fd16b81e4a476d1646d193e5bcb0a057ade9247dfc`；候选修复后为 `63738a9505e2d4b8c020745461c5652c82fdb9b0e7cf1773ddff14dc759995a5`。

## 剩余验证

需在可用 NPU 窗口完成修复后的单请求、同/跨 DP 双请求、长短混合、连续双请求，以及缓存、sleep/wake、图模式的路由一致性检查。完整模型及训练侧强制 replay 均未验收。本目录保存本次定位相关的脚本、输出、日志和补丁；不包含无关项目的进程清单。

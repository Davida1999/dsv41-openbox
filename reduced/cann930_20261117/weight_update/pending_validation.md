# 待资源实机复验

HCCL 与 NPU IPC 的完整 A→B→A 已通过。接收端加载配额修复已完成实机验证。

后续 CPU 复核发现 IPC 发送缓冲每轮滞留约 1.25 GiB，退出时有共享张量告警。已修复每个数据块的引用计数句柄导出方式，新增 5 项 CPU 回归和仓库 106 项 CPU 测试通过。该生命周期补丁的实际回收效果等待设备安排。

## 单卡多轮验证入口

在安排的单张卡上使用 scripts/ipc_reclamation_retest.py，默认 20 轮、8 MiB 缓冲，检查数据一致性和轮后分配量。可传 --baseline-source patches/packed_tensor.before.py 对照旧实现。该脚本已完成语法检查，尚未实机执行。

资源安排后，在宿主机为 `ALLOCATED_NPU_ID` 设置分配的物理卡编号。先执行旧实现作为对照，再在独立进程中测试当前修复。下面命令等待资源窗口执行。

```bash
cd /mnt/share/c00954197/Workspace/dsv41/basic/weight_update
export ALLOCATED_NPU_ID=0
docker -H unix:///run/c00954197-cann930/docker.sock exec \
  -e ASCEND_RT_VISIBLE_DEVICES="$ALLOCATED_NPU_ID" \
  c00954197_dsv41_cann930 python3 \
  /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/scripts/ipc_reclamation_retest.py \
  --report /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/ipc_reclamation_baseline.json \
  --baseline-source /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/patches/packed_tensor.before.py

docker -H unix:///run/c00954197-cann930/docker.sock exec \
  -e ASCEND_RT_VISIBLE_DEVICES="$ALLOCATED_NPU_ID" \
  c00954197_dsv41_cann930 python3 \
  /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/scripts/ipc_reclamation_retest.py \
  --report /mnt/share/c00954197/Workspace/dsv41/basic/weight_update/ipc_reclamation_fixed.json
```

脚本每轮发送 8 个 3 MiB 张量，通过独立接收进程检查所有数据。轮后执行同步、GC、ipc_collect 和 empty_cache，保存 allocated/reserved。当前验收阈值由实验设为相对初始 allocated 增长不超过 1 MiB，实际设备复验后需结合原始记录核对。保存标准输出、退出码、前后 `npu-smi info` 和共享张量退出告警。

## 完整回环复验入口

已有 scripts/run_host.py 支持 --backend npu_ipc 和新的 --tag，启动独立 B 对照与 A 服务执行回环。该步骤需要 8 张卡。

```bash
cd /mnt/share/c00954197/Workspace/dsv41/basic/weight_update
python3 -u scripts/run_host.py --tag ipc_lifecycle_retest_01 --backend npu_ipc
```

使用尚未存在的新 tag，逐轮核对 6248 张量、135927315272 字节、B/A-restored 版本、完整 head 哈希、逐副本双接口输出以及发送端轮后 allocated。单卡回收通过后，再在分配的 8 卡窗口执行此项，确认完整模型回环的缓冲回收及退出告警情况。当前脚本的卡号和网卡配置针对 177.42.22.1，换设备时需调整配置。

## 覆盖补充

本次 prompt token 数为 113，低于 128-token block，前缀命中一直为 0。资源允许时使用产生有效命中的长 prompt，核对更新前确有命中、pause(clear_cache=true) 后首次请求为 0、更新后重复请求重新命中，并保持双接口数值对照。

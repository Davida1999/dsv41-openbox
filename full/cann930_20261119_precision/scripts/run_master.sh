#!/usr/bin/env bash
# 主节点 177.42.23.1，在 c00954197_dsv41_cann930 容器中运行。
set -euo pipefail

cd /mnt/share/c00954197/Workspace/dsv41/precision_cann930
mkdir -p logs
LOG_FILE=$(mktemp "logs/master_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX.log")
exec > >(tee "$LOG_FILE") 2>&1
echo "日志文件 $LOG_FILE"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=5400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SERVER_DEV_MODE=1
export HCCL_IF_IP=177.42.23.1
export VLLM_HOST_IP=177.42.23.1
export GLOO_SOCKET_IFNAME=enp39s0f0
export IP_SOCKET_IFNAME=enp35s0f2
export HCCL_SOCKET_IFNAME=enp35s0f2

# GPQA 精度，完整 BF16 + Engram，TP1 / PCP8 / DP2，异步调度。
exec vllm serve /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 \
  --host 0.0.0.0 \
  --port 8000 \
  --cpu-distributed-timeout-seconds 5400 \
  --safetensors-load-strategy lazy \
  --dtype bfloat16 \
  --seed 42 \
  --hf-overrides '{"quantization_config":null}' \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --enable-expert-parallel \
  --data-parallel-size-local 1 \
  --api-server-count 1 \
  --data-parallel-address 177.42.23.1 \
  --data-parallel-rpc-port 12346 \
  --served-model-name dsv41-bf16-cann930 \
  --max-model-len 73728 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.96 \
  --block-size 128 \
  --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --trust-remote-code \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --additional-config '{"weight_nz_mode":0,"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_shared_expert_dp":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

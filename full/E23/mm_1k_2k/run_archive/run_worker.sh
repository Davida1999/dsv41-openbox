#!/usr/bin/env bash
# E23: 177.11.26.1 的 c00954197_vllm_ascend_1108 容器；沿用 E14，Gloo 改用业务网口。
set -euo pipefail
cd /workspace

mkdir -p /workspace/experiments/tp1_dp16_20261029/E23/runs_1108/run_1k_2k_01/server_logs
LOG_FILE="$(mktemp /workspace/experiments/tp1_dp16_20261029/E23/runs_1108/run_1k_2k_01/server_logs/worker_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX.log)"
exec > >(tee "$LOG_FILE") 2>&1

export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SERVER_DEV_MODE=1
export HCCL_IF_IP="177.11.26.1"
export GLOO_SOCKET_IFNAME="enp39s0f0"
export IP_SOCKET_IFNAME="enp35s0f2"
export HCCL_SOCKET_IFNAME="enp35s0f2"

echo "EXPERIMENT_START role=worker utc=$(date -u +%FT%TZ) run=E23"
echo "$$" > /workspace/experiments/tp1_dp16_20261029/E23/runs_1108/run_1k_2k_01/worker.pid
exec vllm serve /mnt/share/c00954197/workspace/dsv41-flash-bf16 \
  --host 0.0.0.0 \
  --port 8000 \
  --cpu-distributed-timeout-seconds 3600 \
  --safetensors-load-strategy lazy \
  --dtype bfloat16 \
  --hf-overrides '{"quantization_config":null}' \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --enable-expert-parallel \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address 177.11.25.1 \
  --data-parallel-rpc-port 12346 \
  --headless \
  --served-model-name dsv41-bf16-tp1 \
  --max-model-len 40960 \
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
  --additional-config '{"weight_nz_mode":0,"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_shared_expert_dp":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

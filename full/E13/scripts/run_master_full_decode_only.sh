#!/usr/bin/env bash
# Run inside the vLLM-Ascend container on the primary host (177.11.27.1).
set -euo pipefail

# Do not run with /workspace/dsv41 as cwd: its sibling vllm/ source
# directory would shadow the editable vllm package as a namespace package.
cd /workspace

MODEL_PATH="${MODEL_PATH:-/mnt/share/c00954197/workspace/dsv41-flash-bf16}"
NIC_NAME="${NIC_NAME:-enp35s0f2}"
LOCAL_IP="${LOCAL_IP:-177.11.27.1}"
DP_MASTER_IP="${DP_MASTER_IP:-177.11.27.1}"
DP_RPC_PORT="${DP_RPC_PORT:-12346}"
VLLM_PORT="${VLLM_PORT:-8000}"

# 日志目录：默认保存到共享存储，便于容器退出后查看
LOG_DIR="/workspace/experiments/tp1_dp16_20261029/E13"
mkdir -p "${LOG_DIR}"
LOG_DIR="${LOG_DIR}/server_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "${LOG_DIR}/master_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX.log")"

# 保存脚本及 vLLM 的标准输出、错误输出，同时在终端显示
exec > >(tee "${LOG_FILE}") 2>&1

[[ -d "${MODEL_PATH}" ]] || { echo "Missing model: ${MODEL_PATH}"; exit 1; }
if command -v ip >/dev/null 2>&1; then
  ip -4 addr show dev "${NIC_NAME}" | grep -Fq "${LOCAL_IP}/" || {
    echo "${LOCAL_IP} is not configured on ${NIC_NAME}"; exit 1;
  }
else
  echo "[INFO] 'ip' is unavailable in this image; skipping host-network preflight."
fi

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
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export IP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"

# The CLI retains the name --gpu-memory-utilization on Ascend/NPU backends.
echo "EXPERIMENT_START role=master utc=$(date -u +%FT%TZ) run=E13"
echo "$$" > /workspace/experiments/tp1_dp16_20261029/E13/master.pid
exec vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --cpu-distributed-timeout-seconds 3600 \
  --safetensors-load-strategy lazy \
  --dtype bfloat16 \
  --hf-overrides '{"quantization_config":null}' \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --enable-expert-parallel \
  --data-parallel-size-local 1 \
  --api-server-count 1 \
  --data-parallel-address "${DP_MASTER_IP}" \
  --data-parallel-rpc-port "${DP_RPC_PORT}" \
  --served-model-name dsv41-bf16-tp1 \
  --max-model-len 10240 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.96 \
  --block-size 128 \
  --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --trust-remote-code \
  --additional-config '{"weight_nz_mode":0,"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_shared_expert_dp":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'


#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/share/c00954197/Workspace/dsv41/basic
MODE=${1:-eager}
RUN_DIR=${2:?Pass the run directory as the second argument}
mkdir -p "$RUN_DIR"
cd "$ROOT"
printf '%s\n' "$$" > "$RUN_DIR/server.pid"

export DSV41_SLEEP_RUN_DIR="$RUN_DIR"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_IF_IP=177.42.22.1
export HCCL_SOCKET_IFNAME=enp35s0f2
export IP_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=enp39s0f0
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=7200
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export OMP_NUM_THREADS=10
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

command=(
  vllm serve "$ROOT/model"
  --host 127.0.0.1 --port 18242
  --served-model-name dsv41-top32-sleep2
  --dtype bfloat16 --load-format auto
  --tokenizer-mode deepseek_v41 --trust-remote-code
  --enable-scale-out
  --tensor-parallel-size 1 --data-parallel-size 8
  --pipeline-parallel-size 1
  --prefill-context-parallel-size 1 --decode-context-parallel-size 1
  --enable-expert-parallel --enable-sleep-mode
  --worker-extension-cls sleep_probe.SleepProbe
  --max-model-len 8192 --max-num-batched-tokens 8192
  --max-num-seqs 2 --gpu-memory-utilization 0.60
  --block-size 128 --seed 42
  --no-enable-prefix-caching --no-enable-chunked-prefill
  --additional-config '{"weight_nz_mode":0,"rl_config":{"enabled":true}}'
)
case "$MODE" in
  eager) command+=(--enforce-eager --no-async-scheduling) ;;
  graph) command+=(--no-async-scheduling --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}') ;;
  async) command+=(--async-scheduling --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}') ;;
  *) printf 'Unknown mode %s\n' "$MODE" >&2; exit 2 ;;
esac
printf '%q ' "${command[@]}" | tee "$RUN_DIR/actual_server_command.sh"
printf '\n' | tee -a "$RUN_DIR/actual_server_command.sh"
exec "${command[@]}"

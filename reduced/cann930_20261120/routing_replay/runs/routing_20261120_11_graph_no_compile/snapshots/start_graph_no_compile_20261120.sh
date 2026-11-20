#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay
RUN_DIR=${1:?run directory}
MODE=${2:-eager}
mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/server.pid"
cd "$RUN_DIR"
export DSV41_ROUTING_RUN_DIR="$RUN_DIR"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_IF_IP=177.42.22.1
export HCCL_SOCKET_IFNAME=enp35s0f2
export IP_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=enp39s0f0
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=256
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

command=(
  vllm serve /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16-Slim-E32
  --host 127.0.0.1 --port 18244 --api-server-count 1 --served-model-name dsv41-routing
  --dtype bfloat16 --tokenizer-mode deepseek_v41 --trust-remote-code
  --enable-scale-out --tensor-parallel-size 1 --data-parallel-size 8
  --pipeline-parallel-size 1 --prefill-context-parallel-size 1
  --decode-context-parallel-size 1 --enable-expert-parallel
  --enable-return-routed-experts --enable-prefix-caching --no-swa-bounded-replay
  --enable-prompt-tokens-details --enable-sleep-mode
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 2
  --kv-cache-memory-bytes 2147483648 --gpu-memory-utilization 0.60
  --block-size 128 --seed 42 --no-enable-chunked-prefill
  --no-async-scheduling --worker-extension-cls routing_probe_final_20261120.RoutingProbe
  --additional-config '{"weight_nz_mode":0,"rl_config":{"enabled":true}}'
)
case "$MODE" in
  eager) command+=(--enforce-eager) ;;
  graph) command+=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}') ;;
  *) exit 2 ;;
esac
printf '%q ' "${command[@]}" > "$RUN_DIR/actual_server_command.sh"
printf '\n' >> "$RUN_DIR/actual_server_command.sh"
exec numactl --interleave=all "${command[@]}"

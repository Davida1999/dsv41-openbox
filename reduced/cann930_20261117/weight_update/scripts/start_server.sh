#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/share/c00954197/Workspace/dsv41/basic/weight_update
BACKEND=${1:?hccl or npu_ipc}
MODEL=${2:?checkpoint path}
RUN_DIR=${3:?run directory}
mkdir -p "$RUN_DIR"
cd "$ROOT"
printf '%s\n' "$$" > "$RUN_DIR/server.pid"
export DSV41_WEIGHT_RUN_DIR="$RUN_DIR"
case "$BACKEND" in
  hccl) export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3; DP_SIZE=4 ;;
  npu_ipc) export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; DP_SIZE=8 ;;
  *) exit 2 ;;
esac
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_IF_IP=177.42.22.1
export HCCL_SOCKET_IFNAME=enp35s0f2
export IP_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=enp39s0f0
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=256
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_NUM_THREADS=6
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

command=(
  vllm serve "$MODEL" --host 127.0.0.1 --port 18243
  --served-model-name dsv41-weight-update
  --dtype bfloat16 --load-format auto --tokenizer-mode deepseek_v41
  --trust-remote-code --enable-scale-out --tensor-parallel-size 1
  --data-parallel-size "$DP_SIZE" --pipeline-parallel-size 1
  --prefill-context-parallel-size 1 --decode-context-parallel-size 1
  --enable-expert-parallel --enable-sleep-mode
  --weight-transfer-config "{\"backend\":\"$BACKEND\"}"
  --worker-extension-cls weight_probe.WeightProbe
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 2
  --gpu-memory-utilization 0.60 --kv-cache-memory-bytes 2147483648
  --block-size 128 --seed 42 --enforce-eager --no-async-scheduling
  --enable-prefix-caching --enable-chunked-prefill --enable-prompt-tokens-details
  --additional-config '{"weight_nz_mode":0,"rl_config":{"enabled":true}}'
)
printf '%q ' "${command[@]}" > "$RUN_DIR/actual_server_command.sh"
printf '\n' >> "$RUN_DIR/actual_server_command.sh"
cat "$RUN_DIR/actual_server_command.sh"
exec "${command[@]}"

#!/usr/bin/env bash
# E23 profiling：在 c00954197_vllm_ascend_1108 容器中执行：2K 纯文本 -> 6K 输出，不包含图片。
# 前提：服务已就绪，模型名匹配，上下文能容纳文本和输出。
# 本脚本只压测客户端，不启动或修改服务端。
set -euo pipefail

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
case "${1:-}" in
  no_stack|with_stack) MODE="$1" ;;
  *) echo "Usage: $0 {no_stack|with_stack}" >&2; exit 2 ;;
esac
RESULT_ROOT="/workspace/experiments/E23_profile/$MODE/client_results"
mkdir -p "$RESULT_ROOT"
RUN_DIR="$(mktemp -d "${RESULT_ROOT}/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
exec > >(tee "$RUN_DIR/benchmark.log") 2>&1
trap 'echo 130 > "$RUN_DIR/exit_code.txt"; exit 130' INT
trap 'echo 143 > "$RUN_DIR/exit_code.txt"; exit 143' TERM

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}177.11.25.1,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

# 纯文本随机请求；输入 2048 tokens，输出 6144 tokens。
cmd=(vllm bench serve
  --backend openai-chat
  --endpoint /v1/chat/completions
  --base-url http://177.11.25.1:8000
  --trust-remote-code
  --model dsv41-bf16-tp1
  --tokenizer /mnt/share/c00954197/workspace/dsv41-flash-bf16
  --dataset-name random
  --num-prompts 64
  --max-concurrency 64
  --random-input-len 2048
  --random-output-len 6144
  --random-range-ratio 0.0
  --ignore-eos
  --ready-check-timeout-sec 0
  --request-rate 10000
  --seed 42
  --save-result
  --result-dir "$RUN_DIR"
  --result-filename benchmark.json
)

# 留存本次实际命令与客户端脚本，便于对照不同参数的运行。
cp -- "$SCRIPT_PATH" "$RUN_DIR/client_snapshot.sh"
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
printf 'Started UTC: %s\nRun directory: %s\n' "$(date -u +%FT%TZ)" "$RUN_DIR"
if "${cmd[@]}"; then
  status=0
else
  status=$?
fi
printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
printf 'Finished UTC: %s\nExit code: %s\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"

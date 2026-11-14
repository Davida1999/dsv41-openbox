#!/usr/bin/env bash
# E25：在 c00954197_vllm_ascend_cann930 容器中执行：2K 纯文本 -> 6K 输出。
# 前提：服务已就绪，模型名匹配，上下文能容纳文本和输出。
# 本脚本只压测客户端，不启动或修改服务端。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${SCRIPT_DIR}/perf_results_text_2k_6k"
mkdir -p "$RESULT_ROOT"
RUN_DIR="$(mktemp -d "${RESULT_ROOT}/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
exec > >(tee "$RUN_DIR/benchmark.log") 2>&1

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}177.11.25.1,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

cmd=(vllm bench serve
  --backend openai-chat
  --endpoint /v1/chat/completions
  --base-url http://177.11.25.1:8000
  --trust-remote-code
  --model dsv41-bf16-cann930
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
cp -- "${BASH_SOURCE[0]}" "$RUN_DIR/client_snapshot.sh"
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

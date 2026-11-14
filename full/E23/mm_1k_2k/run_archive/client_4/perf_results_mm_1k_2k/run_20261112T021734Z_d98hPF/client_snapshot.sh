#!/usr/bin/env bash
# E23：在 c00954197_vllm_ascend_1108 容器中执行：1K 文本 + 单张竖图 -> 2K 输出。
# 前提：服务已支持图片，模型名匹配，上下文能容纳文本、视觉 token 和输出。
# 本脚本只压测客户端，不启动或修改服务端。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${SCRIPT_DIR}/perf_results_mm_1k_2k"
mkdir -p "$RESULT_ROOT"
RUN_DIR="$(mktemp -d "${RESULT_ROOT}/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
exec > >(tee "$RUN_DIR/benchmark.log") 2>&1

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}177.11.25.1,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

# 图片 bucket 顺序为 (高度, 宽度, 帧数)。修改测试参数可直接编辑下方数值。
cmd=(vllm bench serve
  --backend openai-chat
  --endpoint /v1/chat/completions
  --base-url http://177.11.25.1:8000
  --trust-remote-code
  --model dsv41-bf16-tp1
  --tokenizer /mnt/share/c00954197/workspace/dsv41-flash-bf16
  --dataset-name random-mm
  --num-prompts 64
  --max-concurrency 64
  --random-input-len 1024
  --random-output-len 2048
  --random-range-ratio 0.0
  --random-mm-base-items-per-request 1
  --random-mm-num-mm-items-range-ratio 0.0
  --random-mm-limit-mm-per-prompt '{"image":1,"video":0}'
  --random-mm-bucket-config '{(1920,1080,1):1.0}'
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

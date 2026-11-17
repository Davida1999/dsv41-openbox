#!/usr/bin/env bash
# 服务就绪后，在 177.42.23.1 的 c00954197_dsv41_cann930 容器中运行。
# 2K 纯文本输入，6K 输出，64 请求，并发 64。
set -euo pipefail

cd /mnt/share/c00954197/Workspace/dsv41/performance_cann930
mkdir -p results/text_2k_6k
RESULT_DIR=$(mktemp -d "results/text_2k_6k/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
exec > >(tee "$RESULT_DIR/benchmark.log") 2>&1
echo "结果目录 $RESULT_DIR"

export NO_PROXY=177.42.23.1,localhost,127.0.0.1
export no_proxy="$NO_PROXY"

status=0
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url http://177.42.23.1:8000 \
  --trust-remote-code \
  --model dsv41-bf16-cann930 \
  --tokenizer /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 \
  --dataset-name random \
  --num-prompts 64 \
  --max-concurrency 64 \
  --random-input-len 2048 \
  --random-output-len 6144 \
  --random-range-ratio 0.0 \
  --ignore-eos \
  --ready-check-timeout-sec 0 \
  --request-rate 10000 \
  --seed 42 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename benchmark.json || status=$?
echo "$status" > "$RESULT_DIR/exit_code.txt"
exit "$status"

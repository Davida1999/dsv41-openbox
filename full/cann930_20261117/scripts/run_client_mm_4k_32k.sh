#!/usr/bin/env bash
# 服务就绪后，在 177.42.23.1 的 c00954197_dsv41_cann930 容器中运行。
# 4K 文本 + 单张 1920×1080 图片输入，32K 输出，64 请求，并发 64。
set -euo pipefail

cd /mnt/share/c00954197/Workspace/dsv41/performance_cann930
mkdir -p results/mm_4k_32k
RESULT_DIR=$(mktemp -d "results/mm_4k_32k/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
exec > >(tee "$RESULT_DIR/benchmark.log") 2>&1
echo "结果目录 $RESULT_DIR"

export NO_PROXY=177.42.23.1,localhost,127.0.0.1
export no_proxy="$NO_PROXY"

# 图片 bucket 顺序为（高度，宽度，帧数）。
status=0
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url http://177.42.23.1:8000 \
  --trust-remote-code \
  --model dsv41-bf16-cann930 \
  --tokenizer /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 \
  --dataset-name random-mm \
  --num-prompts 64 \
  --max-concurrency 64 \
  --random-input-len 4096 \
  --random-output-len 32768 \
  --random-range-ratio 0.0 \
  --random-mm-base-items-per-request 1 \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --random-mm-limit-mm-per-prompt '{"image":1,"video":0}' \
  --random-mm-bucket-config '{(1920,1080,1):1.0}' \
  --ignore-eos \
  --ready-check-timeout-sec 0 \
  --request-rate 10000 \
  --seed 42 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename benchmark.json || status=$?
echo "$status" > "$RESULT_DIR/exit_code.txt"
exit "$status"

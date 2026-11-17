#!/usr/bin/env bash
# 用法 bash run_gpqa.sh 198，预检使用 bash run_gpqa.sh 8。
# 在 23 节点的 c00954197_dsv41_cann930 容器内运行。
set -euo pipefail

QUESTIONS=$1
shift
cd /mnt/share/c00954197/Workspace/dsv41/benchmark
mkdir -p /mnt/share/c00954197/Workspace/dsv41/precision_cann930/results
RUN_DIR=$(mktemp -d "/mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
LOG_FILE="$RUN_DIR/client.log"
echo "日志文件 $LOG_FILE"

ais_bench \
  --models vllm_api_general_chat \
  --datasets gpqa_gen_0_shot_cot_chat_prompt \
  --mode all \
  --model-abbr dsv41-bf16-cann930-gpqa \
  --model-name dsv41-bf16-cann930 \
  --path /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 \
  --host-ip 177.42.23.1 \
  --host-port 8000 \
  --batch-size 8 \
  --num-prompts "$QUESTIONS" \
  --max-out-len 32768 \
  --retry 1 \
  --num-warmups 0 \
  --generation-kwargs '{"temperature":0,"top_p":1,"seed":42,"ignore_eos":false,"chat_template_kwargs":{"thinking":true,"reasoning_effort":100}}' \
  --dump-eval-details \
  --work-dir "$RUN_DIR" \
  "$@" 2>&1 | tee "$LOG_FILE"

EXP_DIR=$(sed -n 's/.*Current exp folder: //p' "$LOG_FILE" | tail -n 1)
python3 /mnt/share/c00954197/Workspace/dsv41/precision_cann930/check_results.py "$EXP_DIR" "$QUESTIONS"

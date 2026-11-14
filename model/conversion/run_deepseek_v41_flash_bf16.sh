#!/usr/bin/env bash
set -euo pipefail
source /home/pkg/cann920/cann-9.2.0/set_env.sh
source /mnt/share/g00815994/dsv41_py311/bin/activate
exec python /mnt/share/c00954197/workspace/convert_deepseek_v41_flash_bf16.py \
  --input-dir /mnt/weight/DeepSeek-V4.1-Flash-1017 \
  --output-dir /mnt/share/c00954197/workspace/dsv41-flash-bf16 \
  --chunk-gib 2 \
  --threads 8

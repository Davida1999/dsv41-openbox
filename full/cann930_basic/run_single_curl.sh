#!/usr/bin/env bash
# Reproduction helper reconstructed from a saved GPQA question, not a new NPU run.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_URL=${1:-http://127.0.0.1:8000}
OUTPUT_FILE=${2:?Pass a new output JSON path as the second argument}
curl --noproxy '*' --fail-with-body --silent --show-error \
  "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$SCRIPT_DIR/examples/gpqa_question_0_reconstructed_request.json" \
  --output "$OUTPUT_FILE"

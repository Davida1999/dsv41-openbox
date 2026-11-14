#!/usr/bin/env bash
# Run once against the master API; it forwards the request to both DP engines.
set -euo pipefail
curl --noproxy '*' --fail-with-body --silent --show-error --max-time 120 \
  -X POST http://177.11.25.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  --data '{"profile_prefix":"text_2k_6k","delay_iterations":1536,"max_iterations":10}'
echo 'Profile request accepted. Workers start recording after 1536 execution steps.'

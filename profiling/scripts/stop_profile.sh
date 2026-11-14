#!/usr/bin/env bash
# Close the profiling session on both DP engines; does not stop the service.
set -euo pipefail
curl --noproxy '*' --fail-with-body --silent --show-error --max-time 600 \
  -X POST http://177.11.25.1:8000/stop_profile
echo 'Profiling stop request completed. Verify trace output on both nodes.'

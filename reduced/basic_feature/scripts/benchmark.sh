#!/usr/bin/env bash
# Inspect installed help once before execution; actual benchmark is deliberately separate.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec python "$ROOT/scripts/benchmark.py" "$@"

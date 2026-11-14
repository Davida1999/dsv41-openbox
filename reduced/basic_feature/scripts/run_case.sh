#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CASE=${1:?case ID required}
OUT=${2:?run directory required}
mkdir -p "$OUT"
python - "$CASE" "$OUT" "$ROOT" <<'CHECK'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[3])/'scripts'))
from common import config
saved=json.loads((Path(sys.argv[2])/'effective_config.json').read_text())
assert saved==config(sys.argv[1]), 'Case/config differs from running server snapshot'
assert not saved.get('blocked'),saved.get('blocked')
CHECK
case "$CASE" in
 F09_*) CHECK=check_routing.py;;
 F11_*|F12_*) CHECK=check_sleep.py;;
 F13_ipc) CHECK=run_ipc.py;;
 F13_*) CHECK=check_weight_update.py;;
 F14_*) CHECK=check_long_context.py;;
 *) CHECK=check_inference.py;;
esac
python "$ROOT/scripts/$CHECK" "$CASE" "$OUT" 2>&1 | tee "$OUT/check.log"

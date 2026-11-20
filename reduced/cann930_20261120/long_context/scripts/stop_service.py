#!/usr/bin/env python3
"""Stop processes carrying the exact environment marker of one experiment run."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from process_ownership import belongs_to_run as process_belongs_to_run


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_directory", type=Path)
parser.add_argument("--grace", type=float, default=45.0)
arguments = parser.parse_args()
run = arguments.run_directory.resolve()
assert run.is_relative_to(Path("/mnt/share/c00954197/Workspace/dsv41/basic/long_context/runs"))


def belongs_to_run(pid):
    return process_belongs_to_run(pid, run)


targets = [int(process.name) for process in Path("/proc").iterdir()
           if process.name.isdigit() and belongs_to_run(int(process.name))]
main_pid = None
if (run / "server.pid").exists():
    candidate = int((run / "server.pid").read_text())
    try:
        process = Path(f"/proc/{candidate}")
        if belongs_to_run(candidate) or (os.readlink(process / "cwd") == str(run)
                                         and b"vllm" in (process / "cmdline").read_bytes()):
            main_pid = candidate
            if candidate not in targets:
                targets.append(candidate)
    except OSError:
        pass
fast = (run / "STOP_REQUESTED").exists()
initial = targets if fast or main_pid is None else [main_pid]
for pid in initial:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + (2.0 if fast else arguments.grace)
while time.monotonic() < deadline and any(belongs_to_run(pid) for pid in targets):
    time.sleep(0.25)
forced = []
for pid in targets:
    if belongs_to_run(pid):
        try:
            os.kill(pid, signal.SIGKILL)
            forced.append(pid)
        except ProcessLookupError:
            pass
result = {"signalled": initial, "main_pid": main_pid, "forced": forced,
          "fast_policy_release": fast, "grace_seconds": 2.0 if fast else arguments.grace}
(run / "service_stop.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))

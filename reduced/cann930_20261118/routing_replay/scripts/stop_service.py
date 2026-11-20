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
arguments = parser.parse_args()
run = arguments.run_directory.resolve()
assert run.is_relative_to(Path("/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/runs"))


def belongs_to_run(pid):
    return process_belongs_to_run(pid, run)


targets = [int(process.name) for process in Path("/proc").iterdir()
           if process.name.isdigit() and belongs_to_run(int(process.name))]
for pid in targets:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(8)
forced = []
for pid in targets:
    if belongs_to_run(pid):
        forced.append(pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
result = {"signalled": targets, "forced": forced}
(run / "service_stop.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))

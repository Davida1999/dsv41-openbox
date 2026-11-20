"""Identify this run's processes even when worker titles obscure proc environ."""

import os
from pathlib import Path


def belongs_to_run(pid, run):
    process = Path(f"/proc/{pid}")
    try:
        state = (process / "stat").read_text().split(") ", 1)[1].split()[0]
        if state == "Z":
            return False
        prefix = b"DSV41_ROUTING_RUN_DIR=" + str(run).encode()
        entries = (process / "environ").read_bytes().split(b"\0")
        if any(value == prefix or value.startswith(prefix + b"/") for value in entries):
            return True
        cwd = os.readlink(process / "cwd")
        if cwd != str(run) and not cwd.startswith(str(run) + "/"):
            return False
        name = (process / "comm").read_text().strip()
        command = (process / "cmdline").read_bytes()
        return name.startswith(("VLLM", "VllmWorker", "EngineCore", "asc_dumper")) or any(
            marker in command for marker in (b"multiprocessing.spawn", b"multiprocessing.resource_tracker")
        )
    except OSError:
        return False

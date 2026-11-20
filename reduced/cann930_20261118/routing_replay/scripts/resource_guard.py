"""Release this experiment when another NPU job appears after the user cutoff."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from process_ownership import belongs_to_run


def owned(pid, session):
    return belongs_to_run(pid, session)


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return state != "Z"
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--clock-offset", type=float, required=True)
    parser.add_argument("--observe-only", action="store_true")
    args = parser.parse_args()
    trusted_start = time.time() + args.clock_offset
    monotonic_start = time.monotonic()
    with (args.session / "resource_guard.jsonl").open("a") as log:
        while True:
            try:
                smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired) as error:
                log.write(json.dumps({"host_epoch": time.time(), "smi_error": str(error)}) + "\n")
                log.flush()
                time.sleep(3)
                continue
            pids = sorted({int(match.group(2)) for match in re.finditer(
                r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", smi.stdout, re.MULTILINE)})
            pids = [pid for pid in pids if alive(pid)]
            own = [pid for pid in pids if owned(pid, args.session)]
            foreign = [pid for pid in pids if pid not in own]
            now = trusted_start + time.monotonic() - monotonic_start
            entry = {"trusted_utc_epoch": now, "host_epoch": time.time(),
                     "own_npu_pids": own, "foreign_npu_pids": foreign, "smi_exit": smi.returncode,
                     "exclusive_window": args.observe_only}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if foreign and now >= args.cutoff and not args.observe_only:
                (args.session / "STOP_REQUESTED").write_text("Foreign NPU job after Beijing noon\n")
                (args.session / "yield_npu_smi.txt").write_text(smi.stdout)
                targets = [int(p.name) for p in Path("/proc").glob("[0-9]*") if owned(int(p.name), args.session)]
                for pid in targets:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                time.sleep(8)
                forced = []
                for pid in targets:
                    if owned(pid, args.session):
                        try:
                            os.kill(pid, signal.SIGKILL)
                            forced.append(pid)
                        except ProcessLookupError:
                            pass
                (args.session / "resource_yield.json").write_text(json.dumps(
                    {"foreign_pids": foreign, "signalled_owned": targets, "forced_owned": forced,
                     "trusted_utc_epoch": now}, indent=2) + "\n")
                subprocess.run(["npu-smi", "info"], stdout=(args.session / "after_yield_npu_smi.txt").open("w"))
                return
            status_file = args.session / "session_status.json"
            if status_file.exists():
                status = json.loads(status_file.read_text())
                if status["status"] != "running" and not own:
                    (args.session / "final_npu_smi.txt").write_text(smi.stdout)
                    return
            time.sleep(3)


if __name__ == "__main__":
    main()

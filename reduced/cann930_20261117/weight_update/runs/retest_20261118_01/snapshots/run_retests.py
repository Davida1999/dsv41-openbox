"""Run the IPC reference-counter comparison and both cache-aware full roundtrips."""

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path


ROOT = Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update")
DOCKER = ["docker", "-H", "unix:///run/c00954197-cann930/docker.sock"]
CONTAINER = "c00954197_dsv41_cann930"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    session = ROOT / "runs" / args.tag
    state = {"status": "running", "phase": "single_card_baseline", "started_host_epoch": time.time()}

    def save():
        temporary = session / "session_status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(session / "session_status.json")

    def stop(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    save()
    try:
        for name, baseline in (("baseline", True), ("fixed", False)):
            assert not (session / "STOP_REQUESTED").exists()
            smi = subprocess.check_output(["npu-smi", "info"], text=True)
            assert "No running processes found in NPU 0" in smi
            (session / f"single_card_{name}_before_npu_smi.txt").write_text(smi)
            command = DOCKER + ["exec", "-e", f"DSV41_WEIGHT_RUN_DIR={session}/single_card_{name}",
                "-e", "ASCEND_RT_VISIBLE_DEVICES=0", CONTAINER, "python3", "-u",
                str(ROOT / "scripts/ipc_reclamation_retest.py"), "--report", str(session / f"ipc_{name}.json")]
            if baseline:
                command += ["--baseline-source", str(ROOT / "patches/packed_tensor.before.py")]
            state["phase"] = f"single_card_{name}"
            save()
            (session / f"single_card_{name}_command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (session / f"single_card_{name}.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            (session / f"single_card_{name}.exit").write_text(str(completed.returncode) + "\n")
            state[name + "_exit"] = completed.returncode
            save()
            assert completed.returncode in (0, 2) if baseline else completed.returncode == 0
            (session / f"single_card_{name}_after_npu_smi.txt").write_text(
                subprocess.check_output(["npu-smi", "info"], text=True))
        assert not (session / "STOP_REQUESTED").exists()
        state["phase"] = "full_hccl_and_ipc_cache_roundtrips"
        save()
        command = ["python3", "-u", str(ROOT / "scripts/run_host.py"), "--tag", args.tag + "/transfer", "--backend", "both"]
        (session / "transfer_command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (session / "transfer.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status="passed", phase="complete", finished_host_epoch=time.time())
    except BaseException:
        state.update(status="yielded" if (session / "STOP_REQUESTED").exists() else "failed",
                     error=traceback.format_exc(), finished_host_epoch=time.time())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

"""Start a scoped routing test and release its container processes on exit."""

import argparse
import json
import signal
import subprocess
import time
import traceback
from pathlib import Path


ROOT = Path("/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay")
DOCKER = ["docker", "-H", "unix:///run/c00954197-cann930/docker.sock"]
CONTAINER = "c00954197_dsv41_cann930"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    args = parser.parse_args()
    run = args.run
    state = {"status": "running", "phase": "preflight", "mode": args.mode}

    def save():
        temporary = run / "session_status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(run / "session_status.json")

    def interrupted(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        smi = subprocess.check_output(["npu-smi", "info"], text=True)
        (run / "admission_npu_smi.txt").write_text(smi)
        assert all(f"No running processes found in NPU {index}" in smi for index in range(8)), "NPU resource busy"
        assert not (run / "STOP_REQUESTED").exists()
        shell = f"exec bash {ROOT}/scripts/start_server.sh {run} {args.mode} > {run}/server.log 2>&1"
        command = DOCKER + ["exec", "-d", CONTAINER, "bash", "-lc", shell]
        (run / "server_launcher.json").write_text(json.dumps(command, indent=2) + "\n")
        subprocess.run(command, check=True)
        state["phase"] = "waiting_for_service"
        save()
        for attempt in range(240):
            assert not (run / "STOP_REQUESTED").exists()
            health = subprocess.run(["curl", "--noproxy", "*", "-fsS", "--max-time", "3",
                                     "http://127.0.0.1:18244/health"], capture_output=True)
            if health.returncode == 0:
                break
            pid_file = run / "server.pid"
            if pid_file.exists():
                alive = subprocess.run(DOCKER + ["exec", CONTAINER, "test", "-d",
                                                  "/proc/" + pid_file.read_text().strip()])
                assert alive.returncode == 0, "Service exited during startup"
            if attempt % 20 == 0:
                print("WAITING_FOR_SERVICE", attempt * 3, flush=True)
            time.sleep(3)
        else:
            raise TimeoutError("Routing service startup timeout")
        state["phase"] = "single_and_dual_request_reproduction"
        save()
        command = DOCKER + ["exec", "-e", f"DSV41_ROUTING_RUN_DIR={run}/client",
            CONTAINER, "python3", "-u", str(ROOT / "scripts/validate_routing.py"), "--run", str(run)]
        (run / "client_command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (run / "client.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        (run / "client.exit").write_text(str(result.returncode) + "\n")
        assert result.returncode == 0, "Routing validation failed; see client.log"
        state.update(status="passed", phase="complete")
    except BaseException:
        state.update(status="yielded" if (run / "STOP_REQUESTED").exists() else "failed",
                     error=traceback.format_exc())
        raise
    finally:
        command = DOCKER + ["exec", CONTAINER, "python3", str(ROOT / "scripts/stop_service.py"), str(run)]
        with (run / "stop.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        (run / "final_npu_smi.txt").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
        save()


if __name__ == "__main__":
    main()

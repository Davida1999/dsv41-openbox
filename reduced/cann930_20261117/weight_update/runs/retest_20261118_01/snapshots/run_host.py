#!/usr/bin/env python3
"""Coordinate isolated oracle and A→B→A runs and sample physical HBM."""

import argparse
import csv
import json
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update")
DOCKER = ["docker", "-H", "unix:///run/c00954197-cann930/docker.sock"]
CONTAINER = "c00954197_dsv41_cann930"


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def admit(devices, run):
    smi = subprocess.check_output(["npu-smi", "info"], text=True)
    (run / "admission_npu_smi.txt").write_text(smi)
    assert all(f"No running processes found in NPU {device}" in smi for device in devices), "Devices occupied"


def start_server(backend, model, run):
    shell = "exec setsid bash " + shlex.quote(str(ROOT / "scripts/start_server.sh"))
    shell += " " + shlex.join([backend, str(model), str(run)])
    shell += " > " + shlex.quote(str(run / "server.log")) + " 2>&1 < /dev/null"
    command = DOCKER + ["exec", "-d", CONTAINER, "bash", "-lc", shell]
    save(run / "server_launcher.json", command)
    subprocess.run(command, check=True)
    for attempt in range(240):
        result = subprocess.run(["curl", "--noproxy", "*", "-fsS", "--max-time", "3",
                                 "http://127.0.0.1:18243/health"], capture_output=True)
        if result.returncode == 0:
            return
        if attempt % 12 == 0:
            print("WAITING_FOR_SERVICE", run.name, attempt * 5, flush=True)
        time.sleep(5)
    raise TimeoutError("Service readiness timeout")


def stop_server(run):
    command = DOCKER + ["exec", CONTAINER, "python3", str(ROOT / "scripts/stop_server.py"), str(run)]
    result = subprocess.run(command, capture_output=True, text=True)
    (run / "stop.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    smi = subprocess.check_output(["npu-smi", "info"], text=True)
    (run / "final_npu_smi.txt").write_text(smi)


def sample_memory(run, done):
    path = run / "hbm.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "phase", "npu_id", "used_mib", "total_mib"])
        while not done.is_set():
            result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
            phase_path = run / "phase.txt"
            phase = phase_path.read_text().strip() if phase_path.exists() else "startup"
            lines = result.stdout.splitlines()
            for index, line in enumerate(lines[:-1]):
                device = re.match(r"\|\s*(\d+)\s*\|\s*Ascend", line)
                if device:
                    matches = re.findall(r"(\d+)\s*/\s*(\d+)", lines[index + 1])
                    if matches:
                        used, total = map(int, matches[-1])
                        writer.writerow([time.time(), phase, device.group(1), used, total])
            stream.flush()
            done.wait(1)


def execute(backend, mode, run, oracle=None):
    if mode == "oracle":
        arguments = ["python3", "-u", str(ROOT / "scripts/oracle.py"), "--run", str(run),
                     "--dp-size", "4" if backend == "hccl" else "8"]
    else:
        arguments = ["python3", "-u", str(ROOT / "scripts/producer.py"), "--backend", backend,
                     "--run", str(run), "--oracle", str(oracle)]
        if backend == "npu_ipc":
            arguments = ["torchrun", "--nnodes=1", "--nproc-per-node=8", "--master-addr=127.0.0.1",
                         "--master-port=29644", *arguments[2:]]
    environment = {
        "DSV41_WEIGHT_RUN_DIR": str(run), "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "HCCL_OP_EXPANSION_MODE": "CCU_SCHED", "HCCL_BUFFSIZE": "256",
        "GLOO_SOCKET_IFNAME": "enp39s0f0", "HCCL_IF_IP": "177.42.22.1",
        "HCCL_SOCKET_IFNAME": "enp35s0f2", "OMP_NUM_THREADS": "6",
        "PYTHONPATH": str(ROOT / "scripts"),
    }
    shell = "exec env " + " ".join(shlex.quote(key + "=" + value) for key, value in environment.items())
    shell += " " + shlex.join(arguments)
    command = DOCKER + ["exec", CONTAINER, "bash", "-lc", shell]
    save(run / f"{mode}_command.json", command)
    with (run / f"{mode}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        save(run / f"{mode}_host_process.json", {"pid": process.pid})
        code = process.wait(timeout=7200)
    (run / f"{mode}.exit").write_text(str(code) + "\n")
    if code:
        raise RuntimeError(f"{mode} failed with exit {code}, see {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--backend", choices=["hccl", "npu_ipc", "both"], default="both")
    parser.add_argument("--hccl-oracle", type=Path, default=None)
    arguments = parser.parse_args()
    selected = ["hccl", "npu_ipc"] if arguments.backend == "both" else [arguments.backend]
    controller = ROOT / "runs" / arguments.tag
    controller.mkdir(exist_ok=False)
    state = {"status": "running", "completed": []}
    save(controller / "controller.json", state)
    try:
        for backend in selected:
            oracle_path = arguments.hccl_oracle if backend == "hccl" else None
            modes = [("oracle", ROOT / "model_b"), ("producer", ROOT.parent / "model")]
            if oracle_path is not None:
                oracle_data = json.loads(oracle_path.read_text())
                variant = json.loads((ROOT / "variant_manifest.json").read_text())
                assert oracle_data["status"] == "passed" and len(oracle_data["heads"]) == 4
                assert all(row["head_sha256"] == variant["b_head_sha256"] for row in oracle_data["heads"])
                save(controller / "reused_hccl_oracle.json", {"path": str(oracle_path)})
                modes = [("producer", ROOT.parent / "model")]
            for mode, model in modes:
                run = controller / f"{backend}_{mode}"
                run.mkdir()
                devices = list(range(4)) + [7] if backend == "hccl" else list(range(8))
                admit(devices, run)
                done = threading.Event()
                sampler = threading.Thread(target=sample_memory, args=(run, done), daemon=True)
                sampler.start()
                state["current"] = str(run)
                save(controller / "controller.json", state)
                try:
                    print("START", backend, mode, run, flush=True)
                    start_server(backend, model, run)
                    execute(backend, mode, run, oracle_path)
                    if mode == "oracle":
                        oracle_path = run / "oracle/oracle.json"
                    else:
                        report = json.loads((run / "report.json").read_text())
                        assert report["status"] == "passed" and len(report["rounds"]) == 2
                    state["completed"].append(str(run))
                    save(controller / "controller.json", state)
                    print("COMPLETE", backend, mode, flush=True)
                finally:
                    stop_server(run)
                    done.set()
                    sampler.join(timeout=10)
        state["status"] = "passed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        raise
    finally:
        save(controller / "controller.json", state)


if __name__ == "__main__":
    main()

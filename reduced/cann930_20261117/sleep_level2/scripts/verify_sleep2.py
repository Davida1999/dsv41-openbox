#!/usr/bin/env python3
"""Exercise checkpoint-backed level-2 sleep and retain every response and memory stage."""

import argparse
import csv
import datetime
import json
import math
import re
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--inspect-tensors", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    ranks = [int(rank) for rank in arguments.ranks.split(",")]
    base = "http://127.0.0.1:18242"
    report = {"status": "running", "cycles": [], "memory": [], "max_tokens": arguments.max_tokens,
              "dp_ranks": ranks, "reference_pr": "https://github.com/vllm-project/vllm/pull/53200",
              "checkpoint": "/mnt/share/c00954197/Workspace/dsv41/basic/model"}

    def save(name, data):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def checkpoint():
        save("report.json", report)

    def request(method, path, payload=None, rank=None):
        headers = {"Content-Type": "application/json"}
        if rank is not None:
            headers["X-data-parallel-rank"] = str(rank)
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=3600) as response:
            raw = response.read().decode()
        if path == "/metrics":
            return raw
        return json.loads(raw) if raw.strip() else {}

    def rpc(method, *args, **kwargs):
        return request("POST", "/collective_rpc", {
            "method": method, "args": list(args), "kwargs": kwargs, "timeout": 3600,
        })

    def snapshot(stage, inspect=False):
        report["phase"] = stage
        checkpoint()
        rpc_workers = rpc("sleep2_probe", inspect)
        if isinstance(rpc_workers, dict):
            rpc_workers = rpc_workers["results"]
        smi = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, check=True)
        (output / (stage + "_npu_smi.txt")).write_text(smi.stdout + smi.stderr)
        lines = smi.stdout.splitlines()
        workers = []
        for index, line in enumerate(lines[:-1]):
            device = re.match(r"\|\s*(\d+)\s*\|\s*Ascend", line)
            if device is None:
                continue
            memory_values = re.findall(r"(\d+)\s*/\s*(\d+)", lines[index + 1])
            assert memory_values, (stage, line, lines[index + 1])
            used_mib, total_mib = map(int, memory_values[-1])
            rank = int(device.group(1))
            worker = {"dp_rank": rank, "device_id": rank, "memory_source": "npu-smi",
                      "used_bytes": used_mib * 2**20, "total_bytes": total_mib * 2**20,
                      "free_bytes": (total_mib - used_mib) * 2**20,
                      "allocated_bytes": None, "reserved_bytes": None}
            probe = next((item for item in rpc_workers if item["dp_rank"] == rank), None)
            if probe is not None:
                worker["allocated_bytes"] = probe["allocated_bytes"]
                worker["reserved_bytes"] = probe["reserved_bytes"]
            workers.append(worker)
        assert sorted(worker["device_id"] for worker in workers) == list(range(8)), workers
        metrics = request("GET", "/metrics")
        states = [line for line in metrics.splitlines() if "engine_sleep_state" in line and not line.startswith("#")]
        record = {"stage": stage, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "workers": workers, "worker_rpc_results": rpc_workers,
                  "is_sleeping": request("GET", "/is_sleeping"), "sleep_metrics": states}
        save(stage + "_state.json", record)
        report["memory"].append({"stage": stage, "workers": [
            {key: value for key, value in worker.items() if key != "tensors"} for worker in workers]})
        with (output / "memory.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stage", "dp_rank", "used_mib", "free_mib", "allocated_mib", "reserved_mib"])
            for memory_record in report["memory"]:
                for worker in memory_record["workers"]:
                    writer.writerow([memory_record["stage"], worker["dp_rank"], *[
                        round(worker[key] / 2**20, 3) if worker[key] is not None else "" for key in
                        ("used_bytes", "free_bytes", "allocated_bytes", "reserved_bytes")]])
        checkpoint()
        return record

    def generate(label):
        results = {}
        for rank in ranks:
            for endpoint in ("completions", "chat/completions"):
                payload = {"model": "dsv41-top32-sleep2", "temperature": 0, "top_p": 1,
                           "seed": 42, "max_tokens": arguments.max_tokens, "return_token_ids": True}
                prompt = "The capital of France is"
                if endpoint == "completions":
                    payload.update(prompt=prompt, logprobs=5)
                else:
                    payload.update(messages=[{"role": "user", "content": prompt}], logprobs=True,
                                   top_logprobs=5, chat_template_kwargs={"thinking": False})
                identity = f"dp{rank}_{endpoint.replace('/', '_')}"
                save(label + "_" + identity + "_request.json", payload)
                started = time.monotonic()
                response = request("POST", "/v1/" + endpoint, payload, rank)
                save(label + "_" + identity + "_response.json", response)
                choice = response["choices"][0]
                logprobs = choice.get("logprobs") or {}
                if endpoint == "completions":
                    text = choice["text"]
                    tokens = logprobs.get("tokens")
                    probabilities = logprobs.get("token_logprobs")
                else:
                    text = choice["message"].get("content")
                    details = logprobs.get("content") or []
                    tokens = [token["token"] for token in details]
                    probabilities = [token["logprob"] for token in details]
                assert tokens and probabilities and len(tokens) == len(probabilities), response
                assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in probabilities), response
                results[identity] = {"text": text, "token_ids": choice.get("token_ids"),
                                     "tokens": tokens, "token_logprobs": probabilities}
                assert results[identity]["token_ids"], response
                print(label, identity, f"{time.monotonic()-started:.2f}s", repr(text), flush=True)
        save(label + "_normalized.json", results)
        return results

    try:
        report["inference_world_size"] = request("GET", "/get_world_size?include_dp=true")
        report["control_rpc_fanout"] = "DPLBAsyncMPClient broadcasts to every engine and returns only the first engine result"
        snapshot("initial", arguments.inspect_tensors)
        warmup = generate("warmup")
        baseline = generate("baseline")
        report["baseline_repeat_identical"] = warmup == baseline
        save("baseline.json", baseline)
        initial = snapshot("baseline_steady", arguments.inspect_tensors)
        for cycle in range(1, arguments.cycles + 1):
            prefix = f"cycle_{cycle}"
            before = snapshot(prefix + "_before_sleep")
            started = time.monotonic()
            report["phase"] = prefix + "_sleep"
            checkpoint()
            save(prefix + "_sleep_result.json", request("POST", "/sleep?level=2&mode=abort"))
            sleeping = snapshot(prefix + "_after_sleep")
            assert sleeping["is_sleeping"]["is_sleeping"] is True, sleeping
            before_by_rank = {worker["dp_rank"]: worker for worker in before["workers"]}
            released_bytes = {
                str(worker["dp_rank"]): before_by_rank[worker["dp_rank"]]["used_bytes"] - worker["used_bytes"]
                for worker in sleeping["workers"]
            }
            assert all(value > 1.5 * 2**30 for value in released_bytes.values()), released_bytes
            request("POST", "/wake_up?tags=weights")
            weights_awake = snapshot(prefix + "_after_wake_weights")
            assert weights_awake["is_sleeping"]["is_sleeping"] is True, weights_awake
            report["phase"] = prefix + "_reload"
            checkpoint()
            reload_result = rpc("reload_weights", weights_path=report["checkpoint"], is_checkpoint_format=True)
            save(prefix + "_reload_result.json", reload_result)
            reload_workers = reload_result["results"] if isinstance(reload_result, dict) else reload_result
            assert reload_workers, reload_result
            snapshot(prefix + "_after_reload")
            request("POST", "/wake_up?tags=kv_cache")
            restored = snapshot(prefix + "_after_wakeup", arguments.inspect_tensors)
            assert restored["is_sleeping"]["is_sleeping"] is False, restored
            request("GET", "/health")
            elapsed = time.monotonic() - started
            generated = generate(prefix)
            steady = snapshot(prefix + "_steady")
            comparisons = {}
            for identity, original in baseline.items():
                current = generated[identity]
                comparisons[identity] = {
                    "text_identical": current["text"] == original["text"],
                    "token_ids_identical": current["token_ids"] == original["token_ids"],
                    "tokens_identical": current["tokens"] == original["tokens"],
                    "logprobs_identical": current["token_logprobs"] == original["token_logprobs"],
                    "max_logprob_delta": max((abs(a-b) for a,b in zip(current["token_logprobs"], original["token_logprobs"])), default=0),
                }
            row = {"cycle": cycle, "lifecycle_seconds": elapsed,
                   "released_bytes_by_dp_rank": released_bytes, "comparisons": comparisons}
            report["cycles"].append(row)
            save(prefix + "_comparison.json", row)
            checkpoint()
            if not all(all(value[key] for key in ("text_identical", "token_ids_identical", "tokens_identical", "logprobs_identical"))
                       for value in comparisons.values()):
                raise AssertionError("Restored output differs from the initial loaded-checkpoint baseline")
        steady_samples = [entry for entry in report["memory"] if entry["stage"].endswith("_steady")]
        memory_checks = {}
        for rank in range(8):
            values = [next(worker["used_bytes"] for worker in entry["workers"] if worker["dp_rank"] == rank)
                      for entry in steady_samples]
            memory_checks[str(rank)] = {"steady_used_bytes": values,
                                       "max_growth_bytes": max(values) - values[0],
                                       "pr_post_first_cycle_growth_bytes": max(values[1:]) - values[1]}
        report["memory_checks"] = memory_checks
        report["pr_memory_tolerance_bytes"] = 0.05 * 2**30
        report["memory_passed"] = all(value["pr_post_first_cycle_growth_bytes"] < report["pr_memory_tolerance_bytes"]
                                      for value in memory_checks.values())
        report["status"] = "passed" if report["memory_passed"] and report["baseline_repeat_identical"] else "failed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    checkpoint()
    print("FINAL", report["status"], output, flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

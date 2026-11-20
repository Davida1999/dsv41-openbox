"""Capture every response before validating expert arrays and concurrency."""

import argparse
import base64
import concurrent.futures
import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


BASE = "http://127.0.0.1:18244"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def request(method, endpoint, payload=None, rank=None):
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + endpoint, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run / "validation"
    output.mkdir(exist_ok=True)
    status, raw = request("POST", "/collective_rpc", {
        "method": "begin_routing_trace", "args": [str(args.run / "trace")], "timeout": 120,
    })
    save(output / "trace_start.json", {"status": status, "body": raw})
    assert status == 200, raw
    result = {"status": "running", "cases": []}

    def generate(name, endpoint, prompt, rank, barrier=None):
        payload = {"model": "dsv41-routing", "max_tokens": 16, "temperature": 0, "seed": 42,
                   "ignore_eos": True, "return_token_ids": True, "top_p": 1}
        if endpoint == "completions":
            payload.update(prompt=prompt, logprobs=5)
        else:
            payload.update(messages=[{"role": "user", "content": prompt}], logprobs=True,
                           top_logprobs=5, chat_template_kwargs={"thinking": False})
        save(output / f"{name}_request.json", {"endpoint": endpoint, "dp_rank": rank, "payload": payload})
        if barrier:
            barrier.wait()
        started = time.monotonic()
        status, raw = request("POST", "/v1/" + endpoint, payload, rank)
        (output / f"{name}_response.txt").write_text(raw)
        save(output / f"{name}_http.json", {"status": status, "elapsed_seconds": time.monotonic() - started})
        assert status == 200, (name, status, raw[:2000])
        body = json.loads(raw)
        save(output / f"{name}_response.json", body)
        choice = body["choices"][0]
        encoded = choice.get("routed_experts")
        assert encoded, (name, "Missing routed_experts")
        routes = np.load(io.BytesIO(base64.b64decode(encoded, validate=True)), allow_pickle=False)
        np.save(output / f"{name}_experts.npy", routes, allow_pickle=False)
        save(output / f"{name}_experts.json", {"shape": list(routes.shape), "dtype": str(routes.dtype),
                                              "experts": routes.tolist()})
        expected = (body["usage"]["prompt_tokens"] + len(choice["token_ids"]) - 1, 40, 6)
        assert routes.shape == expected, (name, routes.shape, expected)
        assert np.issubdtype(routes.dtype, np.integer)
        assert routes.min() >= 0 and routes.max() < 32
        assert np.all(np.diff(np.sort(routes, axis=-1), axis=-1) > 0), "Duplicate experts within top-6"
        return {"name": name, "endpoint": endpoint, "dp_rank": rank, "shape": list(routes.shape),
                "dtype": str(routes.dtype), "min": int(routes.min()), "max": int(routes.max()),
                "token_ids": choice["token_ids"], "cached_tokens":
                    (body.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)}

    try:
        for endpoint in ("completions", "chat/completions"):
            suffix = endpoint.replace("/", "_")
            row = generate("single_" + suffix, endpoint, "The capital of France is", 0)
            result["cases"].append(row)
            save(output / "summary.json", result)
            barrier = threading.Barrier(2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(generate, f"same_dp_pair_{suffix}_{index}", endpoint, prompt, 0, barrier)
                           for index, prompt in enumerate(("The capital of France is", "Explain why the sky is blue."))]
                rows = [future.result() for future in futures]
            result["cases"].extend(rows)
            save(output / "summary.json", result)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / "summary.json", result)


if __name__ == "__main__":
    main()

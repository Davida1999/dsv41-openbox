"""HTTP probes shared by the cold-load oracle and the two transfer experiments."""

import json
import math
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:18243"
PROMPT = "Paris is the capital of France. Berlin is the capital of Germany. " * 8


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def request(method, endpoint, payload=None, rank=None):
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    body = None if payload is None else json.dumps(payload).encode()
    query = urllib.request.Request(BASE_URL + endpoint, data=body, headers=headers, method=method)
    with urllib.request.urlopen(query, timeout=3600) as response:
        raw = response.read().decode()
    return json.loads(raw) if raw.strip() else {}


def rpc(method, *arguments):
    return request("POST", "/collective_rpc", {"method": method, "args": list(arguments), "timeout": 3600})


def probe(directory, dp_size, hash_head=True):
    directory = Path(directory)
    rpc("weight_probe", str(directory), hash_head)
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob("dp*_rank*.json"))]
    assert len(rows) == dp_size, (directory, len(rows), dp_size)
    assert sorted(row["dp_rank"] for row in rows) == list(range(dp_size))
    return rows


def generate(directory, dp_size, label):
    result = {}
    directory = Path(directory)
    for rank in range(dp_size):
        for endpoint in ("completions", "chat/completions"):
            identity = f"dp{rank}_{endpoint.replace('/', '_')}"
            payload = {"model": "dsv41-weight-update", "temperature": 0, "top_p": 1, "seed": 42,
                       "max_tokens": 8, "return_token_ids": True}
            if endpoint == "completions":
                payload.update(prompt=PROMPT, logprobs=5)
            else:
                payload.update(messages=[{"role": "user", "content": PROMPT}], logprobs=True,
                               top_logprobs=5, chat_template_kwargs={"thinking": False})
            save(directory / f"{label}_{identity}_request.json", payload)
            started = time.monotonic()
            response = request("POST", "/v1/" + endpoint, payload, rank)
            save(directory / f"{label}_{identity}_response.json", response)
            choice = response["choices"][0]
            if endpoint == "completions":
                text = choice["text"]
                tokens = choice["logprobs"]["tokens"]
                probabilities = choice["logprobs"]["token_logprobs"]
            else:
                text = choice["message"].get("content")
                details = choice["logprobs"]["content"]
                tokens = [item["token"] for item in details]
                probabilities = [item["logprob"] for item in details]
            assert tokens and choice.get("token_ids") and len(probabilities) == len(tokens)
            assert all(math.isfinite(value) for value in probabilities)
            result[identity] = {"text": text, "token_ids": choice["token_ids"],
                                "tokens": tokens, "token_logprobs": probabilities}
            print(label, identity, f"{time.monotonic()-started:.2f}s", repr(text), flush=True)
    save(directory / f"{label}_normalized.json", result)
    return result


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    checks = {}
    for identity, output in actual.items():
        reference = expected[identity]
        checks[identity] = {key: output[key] == reference[key]
                            for key in ("text", "token_ids", "tokens", "token_logprobs")}
    return checks


def require_equal(actual, expected):
    checks = compare(actual, expected)
    if not all(all(row.values()) for row in checks.values()):
        raise AssertionError("Cold-checkpoint oracle mismatch: " + json.dumps(checks))
    return checks

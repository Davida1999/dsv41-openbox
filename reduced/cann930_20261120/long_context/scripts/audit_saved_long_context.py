"""Check saved real-NPU request/response evidence without an NPU or tokenizer."""

import argparse
import gzip
import json
from pathlib import Path


def audit(run):
    folder = run / "validation"
    request = json.loads(gzip.decompress((folder / "request.json.gz").read_bytes()))
    response = json.loads((folder / "response.json").read_text())
    manifest = json.loads((folder / "input_manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    output = (folder / "output.txt").read_text()
    ids = request["token_ids"]
    choice = response["choices"][0]
    assert len(ids) == manifest["prompt_tokens"] == response["usage"]["prompt_tokens"] == 1_000_000
    assert ids[:len(manifest["template_prefix"])] == manifest["template_prefix"]
    assert ids[-len(manifest["template_suffix"]):] == manifest["template_suffix"]
    for needle in manifest["needles"]:
        start = needle["token_position"]
        assert ids[start:start + len(needle["encoded_tokens"]) ] == needle["encoded_tokens"]
        assert needle["key"] in output
    assert len(choice["token_ids"]) == response["usage"]["completion_tokens"] == 43
    assert choice["finish_reason"] == "stop"
    assert request["sampling_params"]["max_tokens"] == 1024
    assert summary["http"] == 200 and summary["status"] == "passed"
    assert output.strip() and "\ufffd" not in output
    return {
        "status": "passed",
        "scope": "Offline audit of actual 1M request and response",
        "prompt_tokens": len(ids),
        "output_tokens": len(choice["token_ids"]),
        "keys_embedded_at_recorded_positions": True,
        "retrieval_hits": len(manifest["needles"]),
        "natural_stop": True,
        "output_text_source": "Live /detokenize of saved response token_ids; tokenizer not included for independent offline decode",
        "output_contains_all_saved_keys": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.run), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()

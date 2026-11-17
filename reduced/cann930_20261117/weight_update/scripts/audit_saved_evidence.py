#!/usr/bin/env python3
"""Check saved transfer evidence using only the Python standard library."""

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def main():
    variant = read_json(ROOT / "variant_manifest.json")
    run = ROOT / "runs/transfer_20261117_03"
    result = {"scope": "CPU audit of saved evidence", "backends": {}}
    for backend, dp_size in (("hccl", 4), ("npu_ipc", 8)):
        directory = run / f"{backend}_producer"
        report = read_json(directory / "report.json")
        assert report["status"] == "passed"
        assert (directory / "producer.exit").read_text().strip() == "0"
        manifest = read_json(directory / "checkpoint_manifest.json")
        assert len(manifest) == variant["tensor_count"]
        assert len({row["name"] for row in manifest}) == len(manifest)
        assert sum(row["bytes"] for row in manifest) == variant["total_tensor_bytes"]
        oracle_directory = (ROOT / "runs/transfer_20261117_02/hccl_oracle/oracle"
                            if backend == "hccl" else run / "npu_ipc_oracle/oracle")
        oracle = read_json(oracle_directory / "oracle.json")
        assert oracle["status"] == "passed"
        assert all(row["head_sha256"] == variant["b_head_sha256"] for row in oracle["heads"])
        validation = directory / "validation"
        initial = read_json(validation / "A_cold_normalized.json")
        assert len(initial) == dp_size * 2
        assert initial == report["initial_A"]
        warm_matches = {"A": initial == read_json(validation / "A_warm_normalized.json")}
        stages = []
        for label, reference, checksum in (
            ("B", oracle["cold"], variant["b_head_sha256"]),
            ("A-restored", initial, variant["a_head_sha256"]),
        ):
            outputs = read_json(validation / f"{label}_normalized.json")
            assert outputs == reference
            assert (outputs != initial) == (label == "B")
            warm_matches[label] = outputs == read_json(validation / f"{label}_warm_normalized.json")
            row = next(item for item in report["rounds"] if item["label"] == label)
            assert row["version"]["weight_version"] == label
            assert all(all(checks.values()) for checks in row["checks"].values())
            heads = [read_json(path) for path in sorted((validation / label / "after").glob("dp*.json"))]
            assert sorted(head["dp_rank"] for head in heads) == list(range(dp_size))
            assert all(head["head_sha256"] == checksum for head in heads)
            receivers = []
            for path in sorted((validation / label / "receiver").glob("*.jsonl")):
                events = [json.loads(line) for line in path.read_text().splitlines()]
                final = events[-1]
                assert final["event"] == "call_complete"
                assert final["tensors"] == variant["tensor_count"]
                assert final["bytes"] == variant["total_tensor_bytes"]
                assert final["deferred_tensor_bytes"] == 0
                assert final["pending_layers"] == []
                receivers.append({"file": path.name, "tensors": final["tensors"],
                                  "bytes": final["bytes"], "final_deferred_bytes": 0,
                                  "max_deferred_bytes": max(event["deferred_tensor_bytes"] for event in events)})
            assert len(receivers) == dp_size
            producers = [read_json(path) for path in sorted(directory.glob(f"{label}_producer_rank*.json"))]
            assert len(producers) == (1 if backend == "hccl" else 8)
            assert all(producer["tensor_count"] == variant["tensor_count"] and
                       producer["tensor_bytes"] == variant["total_tensor_bytes"] for producer in producers)
            with (directory / "hbm.csv").open() as stream:
                samples = [sample for sample in csv.DictReader(stream) if sample["phase"] == label + "_transfer"]
            peak = max(int(sample["used_mib"]) for sample in samples if int(sample["npu_id"]) < dp_size)
            stages.append({"stage": label, "producers": producers, "receivers": receivers,
                           "sampled_receiver_hbm_peak_mib": peak, "dual_api_exact_checks": len(outputs)})
        cache_values = []
        for path in validation.glob("*_response.json"):
            details = read_json(path).get("usage", {}).get("prompt_tokens_details") or {}
            cache_values.append(details.get("cached_tokens", 0))
        warnings = (directory / "producer.log").read_text().count(
            "Producer process has been terminated before all shared NPU tensors released")
        result["backends"][backend] = {
            "roundtrips": "passed", "stages": stages, "warm_outputs_match_each_stage": warm_matches,
            "cached_tokens_max": max(cache_values), "ipc_exit_warnings": warnings,
        }
    first = read_json(ROOT / "runs/transfer_20261117_01/hccl_oracle/oracle/oracle.json")
    second = read_json(ROOT / "runs/transfer_20261117_02/hccl_oracle/oracle/oracle.json")
    assert first["cold"] == second["cold"]
    result["reused_hccl_oracle_matches_prior_cold_load"] = True
    result["resource_release_at_run_end"] = read_json(run / "resource_release.json")
    assert result["resource_release_at_run_end"]["all_npus_free"] is True
    hashes = {}
    for filename in ("meta.before.py", "meta.comm_map.after.py", "packed_tensor.before.py", "packed_tensor.after.py"):
        path = ROOT / "patches" / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["source_sha256"] = hashes
    for path in (ROOT / "scripts").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    result["status"] = "passed"
    (ROOT / "evidence_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved evidence audit passed for HCCL and NPU IPC")


if __name__ == "__main__":
    main()

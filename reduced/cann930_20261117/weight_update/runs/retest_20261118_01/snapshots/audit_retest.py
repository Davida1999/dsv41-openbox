"""Audit the completed retest using saved requests, receiver traces and allocator records."""

import argparse
import csv
import json
from pathlib import Path


WARNING = "Producer process has been terminated before all shared NPU tensors released"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    session = args.session
    variant = read(session.parents[1] / "variant_manifest.json")
    baseline = read(session / "ipc_baseline.json")
    fixed = read(session / "ipc_fixed.json")
    assert len(baseline["rounds"]) == len(fixed["rounds"]) == 20
    assert all(row["tensors"] == 8 for row in baseline["rounds"] + fixed["rounds"])
    assert baseline["max_growth_bytes"] == 20 * 8 * 1024**2
    assert fixed["status"] == "passed" and fixed["max_growth_bytes"] == 0
    assert all(row["allocated_bytes"] == fixed["baseline_allocated_bytes"] for row in fixed["rounds"])
    assert (session / "single_card_fixed.log").read_text().count(WARNING) == 0
    result = {"single_card": {"rounds": 20, "old_growth_bytes": baseline["max_growth_bytes"],
                              "fixed_growth_bytes": 0, "fixed_exit_warnings": 0}, "backends": {}}
    for backend, dp_size in (("hccl", 4), ("npu_ipc", 8)):
        run = session / "transfer" / f"{backend}_producer"
        report = read(run / "report.json")
        assert report["status"] == "passed" and len(report["rounds"]) == 2
        assert (run / "producer.exit").read_text().strip() == "0"
        validation = run / "validation"
        stages = []
        for label, checksum in (("B", variant["b_head_sha256"]), ("A-restored", variant["a_head_sha256"])):
            row = next(item for item in report["rounds"] if item["label"] == label)
            assert row["version"]["weight_version"] == label
            assert all(all(check.values()) for check in row["checks"].values())
            assert len(row["heads"]) == dp_size
            assert all(head["head_sha256"] == checksum for head in row["heads"])
            assert all(value["cold_cached_tokens"] == 0 and value["warm_cached_tokens"] > 0
                       for value in row["cache_checks"].values())
            cold = read(validation / f"{label}_normalized.json")
            cold_reference_path = (session / "transfer" / f"{backend}_oracle/oracle/B_cold_normalized.json"
                                   if label == "B" else validation / "A_cold_normalized.json")
            cold_reference = read(cold_reference_path)
            warm = read(validation / f"{label}_warm_normalized.json")
            warm_reference_path = (session / "transfer" / f"{backend}_oracle/oracle/B_warm_normalized.json"
                                   if label == "B" else validation / "A_warm_normalized.json")
            warm_reference = read(warm_reference_path)
            fields = ("text", "token_ids", "tokens", "token_logprobs")
            cold_reference_checks = {key: {field: value[field] == cold_reference[key][field] for field in fields}
                                     for key, value in cold.items()}
            assert all(all(check.values()) for check in cold_reference_checks.values())
            warm_reference_checks = {key: {field: value[field] == warm_reference[key][field] for field in fields}
                                     for key, value in warm.items()}
            assert all(all(check.values()) for check in warm_reference_checks.values())
            receivers = []
            for path in sorted((validation / label / "receiver").glob("*.jsonl")):
                events = [json.loads(line) for line in path.read_text().splitlines()]
                final = events[-1]
                assert final["tensors"] == variant["tensor_count"] == 6248
                assert final["bytes"] == variant["total_tensor_bytes"] == 135927315272
                assert final["deferred_tensor_bytes"] == 0 and final["pending_layers"] == []
                receivers.append({"file": path.name, "tensors": final["tensors"], "bytes": final["bytes"],
                                  "max_deferred_bytes": max(event["deferred_tensor_bytes"] for event in events)})
            assert len(receivers) == dp_size
            producers = [read(path) for path in sorted(run.glob(f"{label}_producer_rank*.json"))]
            assert len(producers) == (1 if backend == "hccl" else 8)
            assert all(item["tensor_count"] == 6248 and item["tensor_bytes"] == 135927315272 for item in producers)
            assert all(item["allocated_bytes"] == 0 for item in producers)
            with (run / "hbm.csv").open() as stream:
                samples = [sample for sample in csv.DictReader(stream) if sample["phase"] == label + "_transfer"]
            peak = max(int(sample["used_mib"]) for sample in samples if int(sample["npu_id"]) < dp_size)
            stages.append({"stage": label, "producer_records": producers, "receiver_records": receivers,
                           "receiver_hbm_sample_peak_mib": peak, "cache_checks": row["cache_checks"],
                           "cold_reference_checks": cold_reference_checks,
                           "warm_reference_checks": warm_reference_checks,
                           "warm_output_drift_count": sum(not check["warm_token_ids_equal"]
                                                          for check in row["cache_checks"].values())})
        warnings = (run / "producer.log").read_text().count(WARNING)
        assert warnings == 0
        result["backends"][backend] = {"full_roundtrip": "passed", "producer_exit_warnings": warnings,
                                       "stages": stages}
    guard = [json.loads(line) for line in (session / "resource_guard.jsonl").read_text().splitlines()]
    observations = [row for row in guard if "foreign_npu_pids" in row]
    cutoff = read(session / "settings.json")["cutoff_utc_epoch"]
    after_noon = [row for row in observations if row["trusted_utc_epoch"] >= cutoff]
    result["resource_guard"] = {"samples": len(observations),
                                "foreign_process_samples": sum(bool(row["foreign_npu_pids"]) for row in observations),
                                "after_noon_samples": len(after_noon),
                                "after_noon_foreign_process_samples": sum(bool(row["foreign_npu_pids"]) for row in after_noon),
                                "yield_recorded": (session / "resource_yield.json").exists(),
                                "clock_offset_seconds": 28800}
    result["status"] = "passed"
    result["remaining_issue"] = "Greedy output changes between cold and prefix-cache-hit requests"
    (session / "acceptance_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "single_card": result["single_card"],
                      "backends": {name: {"full_roundtrip": row["full_roundtrip"],
                                          "producer_exit_warnings": row["producer_exit_warnings"]}
                                   for name, row in result["backends"].items()},
                      "remaining_issue": result["remaining_issue"]}, indent=2))


if __name__ == "__main__":
    main()

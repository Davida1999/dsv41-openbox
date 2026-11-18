#!/usr/bin/env python3
"""Run complete checkpoint-native A→B→A updates through HCCL or NPU IPC."""

import argparse
import gc
import json
import math
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path

from common import generate, probe, request, require_equal, rpc, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hccl", "npu_ipc"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    arguments = parser.parse_args()
    import torch
    import torch.distributed as distributed
    import torch_npu
    from checkpoint_weight_source import CheckpointWeightSource
    from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient, WeightTransferTrainerFactory
    from vllm_ascend.distributed.weight_transfer import register_engine
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

    root = Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update")
    dp_size = 4 if arguments.backend == "hccl" else 8
    rank = int(os.environ.get("RANK", "0"))
    device = 7 if arguments.backend == "hccl" else int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    if arguments.backend == "npu_ipc":
        distributed.init_process_group("gloo", timeout=timedelta(seconds=300))
        assert distributed.get_world_size() == 8

    def barrier():
        if distributed.is_initialized():
            distributed.barrier()

    def coordinator(action):
        barrier()
        result = action() if rank == 0 else None
        barrier()
        return result

    class VersionedClient(HTTPVLLMWeightSyncClient):
        version = "initial"

        def finish_weight_update(self, weight_version=None):
            super().finish_weight_update(self.version)

    class AuditedSource(CheckpointWeightSource):
        def __iter__(self):
            self.tensor_count = 0
            self.tensor_bytes = 0
            for name, tensor in super().__iter__():
                self.tensor_count += 1
                self.tensor_bytes += tensor.numel() * tensor.element_size()
                if rank == 0 and self.tensor_count % 256 == 0:
                    print("TRANSFER_PROGRESS", client.version, self.tensor_count, name,
                          torch.npu.memory_allocated(), flush=True)
                yield name, tensor

    client = VersionedClient("http://127.0.0.1:18243", timeout=3600)
    manifest = json.loads((root / "variant_manifest.json").read_text())
    source_a = AuditedSource(manifest["checkpoint_a"], torch.device(f"npu:{device}"))
    source_b = AuditedSource(manifest["checkpoint_b"], torch.device(f"npu:{device}"))
    summary = {"backend": arguments.backend, "status": "running", "rounds": [],
               "source_A": source_a.summary(), "source_B": source_b.summary(), "rank": rank}
    save(arguments.run / f"producer_rank{rank}_preflight.json", summary)
    if rank == 0:
        save(arguments.run / "checkpoint_manifest.json", [
            {"name": item.name, "shape": list(item.shape), "dtype": str(item.dtype),
             "bytes": item.dtype.itemsize * math.prod(item.shape)}
            for item in source_a.metadata()
        ])
    if rank == 0:
        assert request("GET", "/get_world_size?include_dp=true")["world_size"] == dp_size
        baseline = generate(arguments.run / "validation", dp_size, "A_cold")
        warmup = generate(arguments.run / "validation", dp_size, "A_warm")
        assert all(row["cached_tokens"] == 0 for row in baseline.values())
        assert all(row["cached_tokens"] > 0 for row in warmup.values())
        summary["initial_cache_checks"] = {key: {"cold_cached_tokens": row["cached_tokens"], "warm_cached_tokens": warmup[key]["cached_tokens"], "warm_token_ids_equal": warmup[key]["token_ids"] == row["token_ids"]} for key, row in baseline.items()}
        heads = probe(arguments.run / "validation/head_A_initial", dp_size)
        assert all(head["head_sha256"] == manifest["a_head_sha256"] for head in heads)
        oracle = json.loads(arguments.oracle.read_text())
        summary["initial_A"] = baseline
        save(arguments.run / "report.json", summary)
    barrier()
    if arguments.backend not in WeightTransferTrainerFactory._registry:
        register_engine()
    if arguments.backend == "hccl":
        info = HCCLTrainerInitInfo(rank=0, master_address="127.0.0.1", master_port=29643,
                                  world_size=dp_size + 1, packed=True,
                                  packed_buffer_size_bytes=1280 * 1024**2, packed_num_buffers=1)
    else:
        info = NPUIPCTrainerInitInfo(rank=rank, packed=True, packed_buffer_size_bytes=1280 * 1024**2)
    producer = None
    try:
        producer = WeightTransferTrainerFactory.trainer_init(info, client=client, source=source_b)
        for label, source, deep_sleep in [("B", source_b, False), ("A-restored", source_a, True)]:
            client.version = label
            producer.source = source
            stage = arguments.run / "validation" / label

            def pause_service():
                (arguments.run / "phase.txt").write_text(label + "_pause")
                request("POST", "/pause?mode=wait&clear_cache=true")
                rpc("begin_weight_trace", str(stage / "receiver"))
                probe(stage / "before", dp_size, False)
                if deep_sleep:
                    request("POST", "/sleep?level=2&mode=abort")
                    assert request("GET", "/is_sleeping")["is_sleeping"] is True
                    probe(stage / "sleep", dp_size, False)
                    request("POST", "/wake_up?tags=weights")
                    assert request("GET", "/is_sleeping")["is_sleeping"] is True
                (arguments.run / "phase.txt").write_text(label + "_transfer")

            coordinator(pause_service)
            started = time.monotonic()
            producer.send_weights()
            torch.npu.synchronize()
            assert source.tensor_count == summary["source_A"]["tensor_count"]
            assert source.tensor_bytes == summary["source_A"]["total_tensor_bytes"]
            allocated_before_cleanup = torch.npu.memory_allocated()
            gc.collect()
            torch.npu.ipc_collect()
            torch.npu.empty_cache()
            torch.npu.synchronize()
            save(arguments.run / f"{label}_producer_rank{rank}.json", {
                "allocated_before_cleanup_bytes": allocated_before_cleanup,
                "tensor_count": source.tensor_count, "tensor_bytes": source.tensor_bytes,
                "allocated_bytes": torch.npu.memory_allocated(), "reserved_bytes": torch.npu.memory_reserved(),
                "elapsed_seconds": time.monotonic() - started, "device": device,
            })
            gc.collect()
            torch.npu.empty_cache()

            def verify_service():
                (arguments.run / "phase.txt").write_text(label + "_verify")
                version = request("GET", "/weight_info")
                assert version["weight_version"] == label, version
                if deep_sleep:
                    request("POST", "/wake_up?tags=kv_cache")
                    assert request("GET", "/is_sleeping")["is_sleeping"] is False
                request("POST", "/resume")
                heads = probe(stage / "after", dp_size)
                expected_hash = manifest["b_head_sha256"] if label == "B" else manifest["a_head_sha256"]
                assert all(head["head_sha256"] == expected_hash for head in heads), heads
                output = generate(arguments.run / "validation", dp_size, label)
                expected = oracle["cold"] if label == "B" else baseline
                checks = require_equal(output, expected)
                warm = generate(arguments.run / "validation", dp_size, label + "_warm")
                cache_checks = {key: {"cold_cached_tokens": value["cached_tokens"], "warm_cached_tokens": warm[key]["cached_tokens"], "warm_text_equal": warm[key]["text"] == value["text"], "warm_token_ids_equal": warm[key]["token_ids"] == value["token_ids"]} for key, value in output.items()}
                assert all(value["cold_cached_tokens"] == 0 and value["warm_cached_tokens"] > 0 for value in cache_checks.values()), cache_checks
                row = {"cache_checks": cache_checks, "version": version, "label": label, "checks": checks, "heads": heads,
                       "output_differs_from_A": output != baseline, "warm_outputs": warm}
                if label == "B":
                    assert output != baseline, "B must visibly change the committed model"
                summary["rounds"].append(row)
                (arguments.run / "phase.txt").write_text(label + "_verified")
                save(arguments.run / "report.json", summary)

            coordinator(verify_service)
        summary["status"] = "passed"
    except BaseException:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        print(summary["error"], flush=True)
        raise
    finally:
        if rank == 0:
            save(arguments.run / "report.json", summary)
        if producer is not None:
            producer.shutdown()
        if distributed.is_initialized():
            distributed.destroy_process_group()
    print("BACKEND_COMPLETE", arguments.backend, rank, summary["status"], flush=True)


if __name__ == "__main__":
    main()

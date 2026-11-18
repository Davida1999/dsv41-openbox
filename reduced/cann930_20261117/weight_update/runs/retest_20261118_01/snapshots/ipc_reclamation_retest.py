#!/usr/bin/env python3
"""Pending one-device IPC data and multi-round reclamation validation."""

import argparse
import ast
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path


def consumer(connection):
    import torch
    import torch_npu
    from vllm_ascend.distributed.weight_transfer.packed_tensor import packed_npu_ipc_consumer

    torch.npu.set_device(0)
    while True:
        message = connection.recv()
        if message is None:
            break
        chunk = message["chunk"]
        weights = packed_npu_ipc_consumer(
            chunk["ipc_handle"], "reclamation-test-device", chunk["names"], chunk["shapes"],
            chunk["dtype_names"], chunk["tensor_sizes"], device_index=0,
        )
        for name, value in weights:
            index = int(name.removeprefix("w"))
            expected = torch.arange(value.numel(), dtype=torch.float32) + message["round"] * 1000 + index
            assert torch.equal(value.cpu().reshape(-1), expected), (message["round"], name)
        value = None
        del weights
        torch.npu.synchronize()
        gc.collect()
        connection.send({"status": "passed", "tensors": len(chunk["names"])})
    connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--buffer-mib", type=int, default=8)
    parser.add_argument("--baseline-source", type=Path, default=None)
    arguments = parser.parse_args()
    import torch
    import torch_npu
    from vllm_ascend.distributed.weight_transfer import packed_tensor

    torch.npu.set_device(0)
    producer = packed_tensor.packed_npu_ipc_producer
    source_path = Path(packed_tensor.__file__)
    if arguments.baseline_source is not None:
        source_path = arguments.baseline_source
        tree = ast.parse(source_path.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "packed_npu_ipc_producer")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = dict(vars(packed_tensor))
        exec(compile(module, str(source_path), "exec"), namespace)
        producer = namespace["packed_npu_ipc_producer"]

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=consumer, args=(child,))
    process.start()
    baseline = torch.npu.memory_allocated()
    result = {"status": "running", "rounds": [], "baseline_allocated_bytes": baseline,
              "source": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    try:
        for round_index in range(arguments.rounds):
            def tensors():
                for index in range(8):
                    value = torch.arange(3 * 1024**2 // 4, dtype=torch.float32) + round_index * 1000 + index
                    yield f"w{index}", value

            count = 0
            for chunk in producer(tensors(), "reclamation-test-device", lambda item: item[1].to("npu"),
                                  buffer_size_bytes=arguments.buffer_mib * 1024**2):
                parent.send({"round": round_index, "chunk": chunk})
                assert parent.poll(60), "Consumer acknowledgement timeout"
                acknowledgement = parent.recv()
                assert acknowledgement["status"] == "passed"
                count += acknowledgement["tensors"]
            assert count == 8
            gc.collect()
            torch.npu.synchronize()
            torch.npu.ipc_collect()
            torch.npu.empty_cache()
            allocated = torch.npu.memory_allocated()
            result["rounds"].append({"round": round_index + 1, "tensors": count,
                                     "allocated_bytes": allocated, "reserved_bytes": torch.npu.memory_reserved()})
            arguments.report.write_text(json.dumps(result, indent=2) + "\n")
            print("RECLAMATION_ROUND", round_index + 1, allocated, flush=True)
        result["max_growth_bytes"] = max(row["allocated_bytes"] - baseline for row in result["rounds"])
        result["status"] = "passed" if result["max_growth_bytes"] <= 1024**2 else "failed"
    finally:
        parent.send(None)
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()
            result["status"] = "failed"
        parent.close()
        arguments.report.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

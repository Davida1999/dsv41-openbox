"""CPU-only contract tests for the deployed packed NPU producer function."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = Path("/vllm-workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/packed_tensor.py")


class IPCChunkLeaseTests(unittest.TestCase):
    def pack(self, inputs, capacity=32):
        tree = ast.parse(SOURCE.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "packed_npu_ipc_producer")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 function], type_ignores=[])
        ast.fix_missing_locations(module)
        exports = []
        stream = SimpleNamespace(synchronizations=0)

        def synchronize():
            stream.synchronizations += 1

        stream.synchronize = synchronize

        def reduce_tensor(buffer):
            record = {"lease": len(exports), "data_ptr": buffer.data_ptr(),
                      "payload": buffer.numpy().tobytes(), "synchronizations": stream.synchronizations}
            exports.append(record)
            return None, record

        namespace = {"torch": torch, "reduce_tensor": reduce_tensor,
                     "DEFAULT_PACKED_BUFFER_SIZE_BYTES": capacity}
        exec(compile(module, str(SOURCE), "exec"), namespace)
        original_empty = torch.empty

        def cpu_empty(*arguments, **keywords):
            keywords["device"] = "cpu"
            result = original_empty(*arguments, **keywords)
            result.fill_(165)
            return result

        with patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: stream), create=True), \
                patch.object(torch, "empty", side_effect=cpu_empty):
            chunks = list(namespace["packed_npu_ipc_producer"](
                iter(inputs), "cpu-contract-device", lambda item: item[1], capacity))
        return chunks, exports

    def test_each_chunk_receives_a_fresh_export_counter(self):
        inputs = [(f"w{index}", torch.arange(8, dtype=torch.float32) + index + 17) for index in range(3)]
        chunks, exports = self.pack(inputs)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(exports), len(chunks))
        self.assertEqual([chunk["ipc_handle"]["cpu-contract-device"]["lease"] for chunk in chunks], [0, 1, 2])

    def test_export_follows_ready_payload(self):
        inputs = [(f"w{index}", torch.arange(8, dtype=torch.float32) + index + 17) for index in range(3)]
        chunks, exports = self.pack(inputs)
        for chunk, (_, tensor) in zip(chunks, inputs):
            export = chunk["ipc_handle"]["cpu-contract-device"]
            self.assertEqual(export["payload"], tensor.view(torch.uint8).numpy().tobytes())
            self.assertGreater(export["synchronizations"], 0)

    def test_one_buffer_is_reused_across_exports(self):
        inputs = [(f"w{index}", torch.arange(4, dtype=torch.float32)) for index in range(5)]
        chunks, exports = self.pack(inputs)
        self.assertEqual(len(exports), len(chunks))
        self.assertEqual(len({entry["data_ptr"] for entry in exports}), 1)

    def test_empty_iterator_has_no_exported_reference(self):
        chunks, exports = self.pack([])
        self.assertEqual(chunks, [])
        self.assertEqual(exports, [])

    def test_zero_element_metadata_exports_once(self):
        chunks, exports = self.pack([("empty", torch.empty(0))])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(exports), 1)
        self.assertEqual(chunks[0]["tensor_sizes"], [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

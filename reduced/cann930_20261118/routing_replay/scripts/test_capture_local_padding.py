"""Exercise the deployed capture function with CPU tensors and explicit DP layouts."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path("/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_routed_experts_capture.py")
COMM = SimpleNamespace(ALLTOALL="alltoall", MC2="mc2", FUSED_MC2="fused_mc2")


def load_capture(counts, comm, padded_rows=2, source_rows=2):
    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts)))
    extra = SimpleNamespace(moe_comm_type=comm, padded_num_tokens=padded_rows,
                            moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(
                                num_tokens=source_rows, replace_allreduce=False)))
    namespace = {"torch": torch, "get_forward_context": lambda: context, "_EXTRA_CTX": extra,
                 "MoECommType": COMM, "_TP_SHARDED_COMM_TYPES": frozenset(vars(COMM).values())}
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_padded_batch_rows", "capture")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *functions], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["capture"]


class LocalPaddingCaptureTests(unittest.TestCase):
    def capturer(self, rank):
        return SimpleNamespace(dp_rank=rank, tp_size=1,
                               device_buffer=torch.full((32, 40, 6), -7, dtype=torch.int32))

    def test_unpadded_local_rows(self):
        capture = load_capture([2, 1, 1, 1, 1, 1, 1, 1], COMM.MC2)
        owner = self.capturer(0)
        values = torch.arange(12, dtype=torch.int32).reshape(2, 6)
        capture(owner, 0, values)
        torch.testing.assert_close(owner.device_buffer[:2, 0], values)

    def test_local_padding_on_idle_dp_ranks(self):
        for comm in vars(COMM).values():
            for rank in (1, 4, 7):
                with self.subTest(comm=comm, rank=rank):
                    capture = load_capture([2, 1, 1, 1, 1, 1, 1, 1], comm)
                    owner = self.capturer(rank)
                    values = torch.tensor([[1, 3, 5, 7, 9, 11], [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
                    capture(owner, 0, values)
                    torch.testing.assert_close(owner.device_buffer[0, 0], values[0])
                    torch.testing.assert_close(owner.device_buffer[1, 0], torch.full((6,), -7, dtype=torch.int32))

    def test_graph_local_padding(self):
        capture = load_capture([3, 1, 1, 1, 1, 1, 1, 1], COMM.MC2, padded_rows=4, source_rows=3)
        owner = self.capturer(4)
        values = torch.arange(24, dtype=torch.int32).reshape(4, 6)
        capture(owner, 0, values)
        torch.testing.assert_close(owner.device_buffer[0, 0], values[0])
        self.assertTrue((owner.device_buffer[1:, 0] == -7).all())

    def test_global_unpadded_layout(self):
        capture = load_capture([2, 1, 1, 1, 1, 1, 1, 1], COMM.MC2)
        owner = self.capturer(4)
        values = torch.arange(54, dtype=torch.int32).reshape(9, 6)
        capture(owner, 0, values)
        torch.testing.assert_close(owner.device_buffer[0, 0], values[5])

    def test_global_padded_layout(self):
        capture = load_capture([2, 1, 1, 1, 1, 1, 1, 1], COMM.MC2)
        owner = self.capturer(4)
        values = torch.arange(96, dtype=torch.int32).reshape(16, 6)
        capture(owner, 0, values)
        torch.testing.assert_close(owner.device_buffer[0, 0], values[8])

    def test_invalid_local_layout_is_rejected(self):
        capture = load_capture([2, 1, 1, 1, 1, 1, 1, 1], COMM.MC2)
        with self.assertRaises(AssertionError):
            capture(self.capturer(4), 0, torch.zeros((3, 6), dtype=torch.int32))

    def test_padding_metadata_is_required(self):
        capture = load_capture([2, 1, 1, 1, 1, 1, 1, 1], COMM.MC2, padded_rows=None, source_rows=None)
        with self.assertRaises(AssertionError):
            capture(self.capturer(4), 0, torch.zeros((2, 6), dtype=torch.int32))


if __name__ == "__main__":
    unittest.main(verbosity=2)

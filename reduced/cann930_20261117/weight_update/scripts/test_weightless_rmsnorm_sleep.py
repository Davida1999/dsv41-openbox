"""CPU regression for the constant tensor used by Ascend's weightless RMSNorm."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm


def create_layer(has_weight):
    # Exercise the production constructor without selecting an accelerator backend.
    layer = object.__new__(RMSNorm)
    with torch.device("cpu"), patch.object(CustomOp, "__init__", nn.Module.__init__):
        RMSNorm.__init__(layer, 512, has_weight=has_weight, dtype=torch.bfloat16)
    return layer


class WeightlessRMSNormSleepTests(unittest.TestCase):
    def test_constant_is_a_nonpersistent_buffer(self):
        layer = create_layer(False)
        self.assertIs(dict(layer.named_buffers()).get("weight"), layer.weight)
        self.assertEqual(dict(layer.named_parameters()), {})
        self.assertEqual(dict(layer.state_dict()), {})
        layer.to(dtype=torch.float32)
        self.assertEqual(layer.weight.dtype, torch.float32)

    def test_level2_backup_restores_constant_in_place(self):
        layer = create_layer(False)
        pointer = layer.weight.data_ptr()
        backup = {name: value.detach().cpu().clone() for name, value in layer.named_buffers()}
        layer.weight.zero_()
        for name, value in layer.named_buffers():
            value.copy_(backup[name])
        torch.testing.assert_close(layer.weight, torch.ones(512, dtype=torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(layer.weight.data_ptr(), pointer)

    def test_learned_weight_keeps_its_checkpoint_contract(self):
        layer = create_layer(True)
        self.assertIsInstance(layer.weight, nn.Parameter)
        self.assertEqual(list(layer.state_dict()), ["weight"])
        self.assertEqual(dict(layer.named_buffers()), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

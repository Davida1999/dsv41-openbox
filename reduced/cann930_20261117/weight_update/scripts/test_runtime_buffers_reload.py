"""Regression for a communication map inflating checkpoint loading quotas."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.model_loader.reload.layerwise import (
    get_layerwise_info,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.model_loader.reload.utils import get_layer_size
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class RuntimeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3))
        self.register_buffer("expert_ids_per_ep_rank", torch.arange(32, dtype=torch.int32), persistent=False)
        self.dispatcher = SimpleNamespace(expert_ids_per_ep_rank=self.expert_ids_per_ep_rank)


class CheckpointModel(nn.Module):
    def __init__(self, count=1):
        super().__init__()
        self.layers = nn.ModuleList([RuntimeMoE() for _ in range(count)])

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            parameter = self.get_parameter(name)
            getattr(parameter, "weight_loader", default_weight_loader)(parameter, value)
            loaded.add(name)
        return loaded


class RuntimeBufferReloadTests(unittest.TestCase):
    def test_quota_counts_checkpoint_weights(self):
        self.assertEqual(get_layer_size(RuntimeMoE()), 6)

    def test_each_layer_releases_deferred_inputs(self):
        model = CheckpointModel(8)
        record_metadata_for_reloading(model)
        initialize_layerwise_reload(model)
        for index, layer in enumerate(model.layers):
            parameter = layer.weight
            parameter.weight_loader(parameter, torch.full((2, 3), float(index + 1)))
            info = get_layerwise_info(layer)
            self.assertEqual(info.loaded_weights, [])
            self.assertFalse(info.can_load())
            self.assertFalse(layer.weight.is_meta)
            torch.testing.assert_close(layer.weight, torch.full((2, 3), float(index + 1)))

    def test_a_b_a_keeps_runtime_state_and_addresses(self):
        model = CheckpointModel()
        record_metadata_for_reloading(model)
        layer = model.layers[0]
        buffers = {name: (value.clone(), value.data_ptr()) for name, value in layer.named_buffers()}
        configuration = SimpleNamespace(
            lora_config=None, load_config=None,
            model_config=SimpleNamespace(quantization=None, dtype=torch.float32),
        )
        for value in [2.0, 1.0, 2.0, 1.0]:
            reload_weights(configuration, model, iter([("layers.0.weight", torch.full((2, 3), value))]), None, True)
            torch.testing.assert_close(layer.weight, torch.full((2, 3), value))
            for name, (expected, pointer) in buffers.items():
                current = layer.get_buffer(name)
                torch.testing.assert_close(current, expected, rtol=0, atol=0)
                self.assertEqual(current.data_ptr(), pointer)
                self.assertIs(current, layer.dispatcher.expert_ids_per_ep_rank)


if __name__ == "__main__":
    unittest.main(verbosity=2)

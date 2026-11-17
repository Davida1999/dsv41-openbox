"""Exercise the production V4.1 sink loader through checkpoint reload on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM


def create_model(local_heads):
    model = object.__new__(AscendDeepseekV41LLMForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_attention_heads=64, n_routed_experts=32, n_shared_experts=1)
    model.num_redundant_experts = 0
    model.model = nn.Module()
    model.model.has_engram = False
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.attn_sink = nn.Parameter(torch.empty(local_heads, dtype=torch.float32))
    layer.self_attn.dsa_attn = nn.Module()
    layer.self_attn.dsa_attn.attn_sink = layer.self_attn.attn_sink
    model.model.layers = nn.ModuleList([layer])
    return model


class AttentionSinkReloadTests(unittest.TestCase):
    def check_reload(self, tp_size, tp_rank):
        checkpoint = torch.arange(64, dtype=torch.float32) + 1
        expected = checkpoint.chunk(tp_size)[tp_rank]
        with torch.device("cpu"), \
                patch("vllm_ascend.models.deepseek_v41.model.get_ascend_config", return_value=SimpleNamespace(mix_placement=False)), \
                patch("vllm_ascend.models.deepseek_v41.model.get_tensor_model_parallel_rank", return_value=tp_rank), \
                patch("vllm_ascend.models.deepseek_v41.model.get_tensor_model_parallel_world_size", return_value=tp_size), \
                patch("vllm_ascend.models.deepseek_v41.model.fused_moe_make_expert_params_mapping", return_value=[]), \
                patch("vllm_ascend.models.deepseek_v41.model.get_spec_layer_idx_from_weight_name", return_value=None), \
                patch("vllm_ascend.models.deepseek_v41.model.is_pp_missing_parameter", return_value=False):
            model = create_model(64 // tp_size)
            record_metadata_for_reloading(model)
            model.load_weights(iter([("layers.0.attn.attn_sink", checkpoint)]))
            parameter = model.model.layers[0].self_attn.attn_sink
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            pointer = parameter.data_ptr()
            with torch.no_grad():
                parameter.zero_()
            configuration = SimpleNamespace(
                lora_config=None,
                load_config=None,
                model_config=SimpleNamespace(quantization=None, dtype=torch.float32),
            )
            reload_weights(configuration, model, iter([("layers.0.attn.attn_sink", checkpoint)]), None, True)
            restored = model.model.layers[0].self_attn.attn_sink
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
            self.assertEqual(restored.data_ptr(), pointer)
            self.assertIs(restored, model.model.layers[0].self_attn.dsa_attn.attn_sink)

    def test_tp1_reload_restores_sink_and_alias(self):
        self.check_reload(1, 0)

    def test_tp2_rank0_keeps_head_slice(self):
        self.check_reload(2, 0)

    def test_tp2_rank1_keeps_head_slice(self):
        self.check_reload(2, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

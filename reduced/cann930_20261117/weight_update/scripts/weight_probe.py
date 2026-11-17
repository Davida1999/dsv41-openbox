"""Worker-local update tracing and full output-head content checksums."""

import hashlib
import json
import time
import types
from pathlib import Path


class WeightProbe:
    def weight_probe(self, directory, hash_head=True):
        import torch

        target = Path(directory).resolve()
        assert target.is_relative_to(Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update/runs"))
        target.mkdir(parents=True, exist_ok=True)
        torch.npu.synchronize()
        result = {"dp_rank": self.parallel_config.data_parallel_index, "rank": self.rank,
                  "device": torch.npu.current_device(), "allocated_bytes": torch.npu.memory_allocated(),
                  "reserved_bytes": torch.npu.memory_reserved()}
        from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors, get_layer_size
        result["moe_tensor_schema"] = [
            {"module": name, "loadable_numel": get_layer_size(module), "tensors": [
                {"name": key, "numel": value.numel(), "shape": list(value.shape),
                 "dtype": str(value.dtype), "skip_loading_quota": key in SKIP_LOAD_TENSORS}
                for key, value in get_layer_tensors(module).items()]}
            for name, module in self.model_runner.get_model().named_modules()
            if type(module).__name__ == "AscendRoutedExperts"
        ]
        free, total = torch.npu.mem_get_info()
        result.update(free_bytes=free, used_bytes=total - free)
        if hash_head:
            parameters = self.model_runner.get_model().named_parameters()
            name, parameter = next((name, value) for name, value in parameters if name.endswith("lm_head.weight"))
            digest = hashlib.sha256()
            with torch.no_grad():
                for start in range(0, parameter.shape[0], 2048):
                    block = parameter[start:start + 2048].detach().cpu().contiguous()
                    digest.update(block.view(torch.uint8).numpy().tobytes())
            result.update(head_name=name, head_shape=list(parameter.shape), head_dtype=str(parameter.dtype),
                          head_sha256=digest.hexdigest(), head_data_ptr=parameter.data_ptr())
        filename = target / f"dp{self.parallel_config.data_parallel_index}_rank{self.rank}.json"
        filename.write_text(json.dumps(result, indent=2) + "\n")
        return result

    def begin_weight_trace(self, directory):
        import torch
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

        target = Path(directory).resolve()
        assert target.is_relative_to(Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update/runs"))
        target.mkdir(parents=True, exist_ok=True)
        self._weight_trace_path = target / f"receiver_dp{self.parallel_config.data_parallel_index}.jsonl"
        self._weight_trace_counts = {"tensors": 0, "bytes": 0, "calls": 0}
        torch.npu.reset_peak_memory_stats()
        if getattr(self, "_weight_trace_installed", False):
            return self._weight_trace_counts
        original = self.model_runner.get_model().load_weights
        owner = self

        def record_progress(event):
            counts = owner._weight_trace_counts
            buffered_bytes = 0
            pending_layers = []
            for layer, info in LAYERWISE_INFO.items():
                if not info.loaded_weights:
                    continue
                pending_layers.append({"class": type(layer).__name__, "loaded": info.load_numel,
                                       "expected": info.load_numel_total})
                for _, arguments in info.loaded_weights:
                    for value in arguments.arguments.values():
                        if isinstance(value, torch.Tensor) and value.device.type == "npu":
                            buffered_bytes += value.numel() * value.element_size()
            entry = {**counts, "epoch": time.time(), "allocated_bytes": torch.npu.memory_allocated(),
                     "reserved_bytes": torch.npu.memory_reserved(), "deferred_tensor_bytes": buffered_bytes,
                     "max_allocated_bytes": torch.npu.max_memory_allocated(), "pending_layers": pending_layers,
                     "event": event}
            with owner._weight_trace_path.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")

        def traced_load(model, weights):
            owner._weight_trace_counts["calls"] += 1

            def tracked_weights():
                for name, tensor in weights:
                    owner._weight_trace_counts["tensors"] += 1
                    owner._weight_trace_counts["bytes"] += tensor.numel() * tensor.element_size()
                    owner._weight_trace_counts["last_name"] = name
                    if owner._weight_trace_counts["tensors"] % 256 == 0:
                        record_progress("receiving")
                    yield name, tensor

            try:
                result = original(tracked_weights())
            except BaseException:
                record_progress("error")
                raise
            record_progress("call_complete")
            return result

        self.model_runner.get_model().load_weights = types.MethodType(traced_load, self.model_runner.get_model())
        self._weight_trace_installed = True
        return self._weight_trace_counts

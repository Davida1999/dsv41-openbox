"""Collect worker-local memory and model-state evidence through collective RPC."""

import hashlib
import os


def tensor_summary(tensor):
    import torch

    result = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
        "stride": list(tensor.stride()),
    }
    if tensor.numel() == 0 or tensor.device.type == "meta":
        return result
    value = tensor.detach()
    if value.numel() <= 262144:
        sample = value.contiguous().cpu()
        result["digest_kind"] = "full"
    elif value.is_contiguous():
        flattened = value.view(-1)
        sample = torch.cat((flattened[:128], flattened[-128:])).cpu()
        result["digest_kind"] = "head_tail_sample"
    else:
        result["digest_kind"] = "metadata_only"
        return result
    result["sha256"] = hashlib.sha256(sample.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


class SleepProbe:
    def sleep2_probe(self, inspect_tensors=False):
        import torch

        dp_rank = self.parallel_config.data_parallel_index
        result = {
            "pid": os.getpid(),
            "rank": self.rank,
            "dp_rank": dp_rank,
            "device_index": torch.npu.current_device(),
        }
        if inspect_tensors and dp_rank == 0:
            model = self.model_runner.get_model()
            tensors = {}
            for name, value in model.named_buffers():
                tensors["buffer:" + name] = tensor_summary(value)
            for module_name, module in model.named_modules():
                for name, value in vars(module).items():
                    if isinstance(value, torch.Tensor):
                        tensors[f"attribute:{module_name}.{name}"] = tensor_summary(value)
            for name, value in model.named_parameters():
                tensors["parameter:" + name] = tensor_summary(value)
            result["tensors"] = tensors
        torch.npu.synchronize()
        free_bytes, total_bytes = torch.npu.mem_get_info()
        result.update(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            used_bytes=total_bytes - free_bytes,
            allocated_bytes=torch.npu.memory_allocated(),
            reserved_bytes=torch.npu.memory_reserved(),
        )
        return result


#!/usr/bin/env python3
"""Stream DeepSeek V4.1 Flash FP8/FP4 safetensors into BF16 safetensors.

The source checkpoint is opened read-only.  Output chunks are committed by
atomic rename, and a JSONL manifest makes interrupted runs resumable.
"""
import argparse
import json
import os
import shutil
import uuid
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def fp8_to_bf16(weight, scale):
    rows, cols = weight.shape
    sr, sc = scale.shape
    assert rows % sr == 0 and cols % sc == 0, (weight.shape, scale.shape)
    br, bc = rows // sr, cols // sc
    return (weight.float().unflatten(0, (-1, br)).unflatten(-1, (-1, bc))
            * scale.float()[:, None, :, None]).flatten(2, 3).flatten(0, 1).bfloat16()


def fp4_to_bf16(weight, scale):
    packed = weight.view(torch.uint8)
    values = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    signs = 1.0 - 2.0 * ((values >> 3) & 1).float()
    decoded = signs * LUT[(values & 7).long()]
    assert decoded.shape[0] == scale.shape[0]
    assert decoded.shape[1] % scale.shape[1] == 0, (decoded.shape, scale.shape)
    return (decoded * scale.float().repeat_interleave(decoded.shape[1] // scale.shape[1], dim=-1)).bfloat16()


def load_manifest(path):
    done, next_part = {}, 1
    if not os.path.exists(path):
        return done, next_part
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if not os.path.exists(os.path.join(os.path.dirname(path), item["file"])):
                raise RuntimeError(f"manifest references missing output: {item['file']}")
            done.update({name: item["file"] for name in item["names"]})
            next_part = max(next_part, item["part"] + 1)
    return done, next_part


def main(src, dst, chunk_gib, threads):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(dst, exist_ok=True)
    torch.set_num_threads(threads)
    manifest = os.path.join(dst, "conversion-manifest.jsonl")
    done, part = load_manifest(manifest)
    files = sorted(glob(os.path.join(src, "*.safetensors")))
    if not files:
        raise RuntimeError("no source safetensors files")
    chunk_limit = int(chunk_gib * 1024 ** 3)
    pending, pending_bytes, completed = {}, 0, len(done)

    def flush():
        nonlocal pending, pending_bytes, part, completed
        if not pending:
            return
        name = f"model-bf16-{part:05d}.safetensors"
        final = os.path.join(dst, name)
        tmp = final + f".{uuid.uuid4().hex}.tmp"
        save_file(pending, tmp, metadata={"format": "pt"})
        os.replace(tmp, final)
        record = {"part": part, "file": name, "names": list(pending), "raw_bytes": pending_bytes}
        with open(manifest, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        completed += len(pending)
        print(f"COMMIT part={part} tensors={len(pending)} completed={completed}", flush=True)
        done.update({k: name for k in pending})
        pending, pending_bytes = {}, 0
        part += 1

    for file_num, path in enumerate(files, 1):
        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = list(reader.keys())
            key_set = set(keys)
            scale_names = {k for k in keys if k.endswith(".scale") and k[:-6] + ".weight" in key_set}
            for key in keys:
                if key in scale_names or key in done:
                    continue
                weight = reader.get_tensor(key)
                scale_key = key[:-7] + ".scale" if key.endswith(".weight") else None
                if scale_key in key_set and weight.dtype == torch.float8_e4m3fn:
                    out = fp8_to_bf16(weight, reader.get_tensor(scale_key))
                elif scale_key in key_set and weight.dtype == torch.int8:
                    out = fp4_to_bf16(weight, reader.get_tensor(scale_key))
                else:
                    out = weight
                pending[key] = out.contiguous()
                pending_bytes += out.numel() * out.element_size()
                if pending_bytes >= chunk_limit:
                    flush()
            print(f"SOURCE_DONE {file_num}/{len(files)} {os.path.basename(path)}", flush=True)
    flush()

    expected = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = list(reader.keys())
            key_set = set(keys)
            expected.update(k for k in keys if not (k.endswith(".scale") and k[:-6] + ".weight" in key_set))
    if set(done) != expected:
        raise RuntimeError(f"output tensor mismatch: missing={len(expected-set(done))}, unexpected={len(set(done)-expected)}")
    total_size = sum(json.loads(line)["raw_bytes"] for line in open(manifest))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": done}, f, indent=2, sort_keys=True)
    for name in ("config.json", "configuration.json", "tokenizer.json", "tokenizer_config.json"):
        source = os.path.join(src, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(dst, name))
    print(f"COMPLETE tensors={len(done)} raw_bytes={total_size}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-gib", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    main(a.input_dir, a.output_dir, a.chunk_gib, a.threads)

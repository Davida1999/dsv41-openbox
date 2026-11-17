#!/usr/bin/env python3
"""Create a checkpoint B view by rolling the BF16 output-head rows once."""

import hashlib
import json
import shutil
import struct
from pathlib import Path

ROOT = Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update")
SOURCE = ROOT.parent / "model"
TARGET = ROOT / "model_b"
CHUNK_BYTES = 64 * 1024 * 1024


def digest_range(path, offset, length):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(CHUNK_BYTES, remaining))
            if not block:
                raise EOFError(path)
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    TARGET.mkdir(exist_ok=False)
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["head.weight"]
    source_shard = (SOURCE / shard).resolve()
    for path in SOURCE.iterdir():
        if path.name == shard or path.name.endswith("manifest.json"):
            continue
        if path.name == "config.json":
            shutil.copyfile(path, TARGET / path.name)
        else:
            (TARGET / path.name).symlink_to(path.resolve())
    target_shard = TARGET / shard
    shutil.copyfile(source_shard, target_shard)
    with source_shard.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_bytes))
    tensor = header["head.weight"]
    assert tensor["dtype"] == "BF16" and len(tensor["shape"]) == 2
    rows, columns = tensor["shape"]
    begin, end = tensor["data_offsets"]
    offset = 8 + header_bytes + begin
    length = end - begin
    row_bytes = columns * 2
    assert length == rows * row_bytes
    with source_shard.open("rb") as source, target_shard.open("r+b") as target:
        source.seek(offset + length - row_bytes)
        target.seek(offset)
        target.write(source.read(row_bytes))
        source.seek(offset)
        target.seek(offset + row_bytes)
        remaining = length - row_bytes
        while remaining:
            block = source.read(min(CHUNK_BYTES, remaining))
            if not block:
                raise EOFError(source_shard)
            target.write(block)
            remaining -= len(block)
    original_digest = digest_range(source_shard, offset, length)
    changed_digest = digest_range(target_shard, offset, length)
    assert original_digest != changed_digest
    manifest = {
        "checkpoint_a": str(SOURCE), "checkpoint_b": str(TARGET),
        "tensor_count": len(index["weight_map"]),
        "total_tensor_bytes": index["metadata"]["total_size"],
        "changed_tensor": "head.weight", "changed_shard": shard,
        "shape": tensor["shape"], "dtype": tensor["dtype"],
        "change": "torch.roll(head.weight, shifts=1, dims=0)",
        "a_head_sha256": original_digest, "b_head_sha256": changed_digest,
        "head_tensor_bytes": length, "cpu_copy_chunk_bytes": CHUNK_BYTES,
    }
    (ROOT / "variant_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

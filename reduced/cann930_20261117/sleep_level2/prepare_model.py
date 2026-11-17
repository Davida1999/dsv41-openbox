#!/usr/bin/env python3
"""Create the sleep level-2 model view using JSON metadata and symlinks only."""
import json
import struct
import tempfile
from pathlib import Path

SOURCE = Path('/mnt/share/g00815994/dsv41/DeepSeek-V4.1-Flash-top32-bf16-runtime-codex-20261030')
ROOT = Path(__file__).resolve().parent


def main():
    target = ROOT / 'model'
    if target.exists():
        raise SystemExit(f'{target} already exists; refusing to overwrite it.')
    config = json.loads((SOURCE / 'config.json').read_text())
    text = config['text_config']
    assert text['num_hidden_layers'] == 40 and text['n_routed_experts'] == 32
    text['engram_layer_ids'] = []
    index = json.loads((SOURCE / 'model.safetensors.index.json').read_text())
    weights = index['weight_map']
    removed = {k: v for k, v in weights.items() if 'engram' in k.lower()}
    kept = {k: v for k, v in weights.items() if k not in removed}
    shards = sorted(set(kept.values()))
    # Read safetensors headers, never tensor payloads, to recompute total_size.
    total = 0
    for shard in shards:
        with (SOURCE / shard).open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError(f'Unexpected header size in {shard}')
            header = json.loads(stream.read(size))
        for key in (k for k, v in kept.items() if v == shard):
            begin, end = header[key]['data_offsets']
            total += end - begin
    index['weight_map'] = kept
    index.setdefault('metadata', {})['total_size'] = total
    excluded = sorted(set(weights.values()) - set(shards))
    manifest = {'source': str(SOURCE), 'layers': 40, 'routed_experts': 32,
                'engram_layer_ids': [], 'removed_keys': removed,
                'excluded_shards': excluded, 'retained_shards': len(shards),
                'indexed_tensor_bytes': total,
                'mixed_engram_shards': sorted(set(removed.values()) & set(shards))}
    with tempfile.TemporaryDirectory(prefix='.model_', dir=ROOT) as temp:
        stage = Path(temp) / 'model'
        stage.mkdir()
        for p in SOURCE.iterdir():
            if p.name in ('config.json', 'model.safetensors.index.json', 'codex_runtime_manifest.json'):
                continue
            if p.is_file() and p.suffix in ('.json', '.model', '.py', '.jinja', '.txt'):
                (stage / p.name).symlink_to(p.resolve())
        for shard in shards:
            (stage / shard).symlink_to((SOURCE / shard).resolve(strict=True))
        (stage / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (stage / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
        (stage / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(target)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

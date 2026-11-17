#!/usr/bin/env python3
"""Capture checkpoint B directly loaded by the inference service."""

import argparse
import json
from pathlib import Path
from common import generate, probe, save

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--dp-size", type=int, required=True)
arguments = parser.parse_args()
cold = generate(arguments.run / "oracle", arguments.dp_size, "B_cold")
warm = generate(arguments.run / "oracle", arguments.dp_size, "B_warm")
heads = probe(arguments.run / "oracle/head_B", arguments.dp_size)
manifest = json.loads(Path("/mnt/share/c00954197/Workspace/dsv41/basic/weight_update/variant_manifest.json").read_text())
assert all(head["head_sha256"] == manifest["b_head_sha256"] for head in heads)
save(arguments.run / "oracle/oracle.json", {"cold": cold, "warm": warm, "heads": heads, "status": "passed"})
print("COLD_B_ORACLE_COMPLETE", flush=True)

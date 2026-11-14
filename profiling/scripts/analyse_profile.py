#!/usr/bin/env python
"""Offline analysis of an existing TorchNPU profiling directory."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiler_path', type=Path, help='Actual trace directory printed by the service')
    args = parser.parse_args()
    path = args.profiler_path.resolve()
    if not path.is_dir():
        parser.error(f'Profiling directory does not exist: {path}')
    from torch_npu.profiler.profiler import analyse
    analyse(profiler_path=str(path))


if __name__ == '__main__':
    main()

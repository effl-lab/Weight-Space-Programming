"""Hash every compiled tensor without copying the large lookup tables.

No questions, predictions, answer labels, fitting, or parameter selection.
Run in separate processes with different PYTHONHASHSEED values and compare
only the `tensors` records; time, hash probe, and RSS are informational. Use
Python -sP with an empty PYTHONPATH for fixed-seed checks: Python -I ignores
PYTHONHASHSEED and therefore cannot force a requested hash seed.
"""
from pathlib import Path
import hashlib
import json
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import build

torch.set_num_threads(4)


def fingerprint(weights):
    result = {}
    for name, tensor in sorted(weights.items()):
        if tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise AssertionError('Expected contiguous CPU tensor: ' + name)
        # A NumPy view and memoryview do not allocate a full tensor copy.
        raw = memoryview(tensor.detach().numpy()).cast('B')
        result[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                            sha256=hashlib.sha256(raw).hexdigest())
    return result


def main():
    started = time.monotonic()
    weights = build.build()
    build_seconds = time.monotonic() - started
    result = dict(tensors=fingerprint(weights), build_seconds=build_seconds,
                  hash_probe=hash('authored-association-memory'),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

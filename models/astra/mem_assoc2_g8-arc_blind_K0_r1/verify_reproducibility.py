"""Verify deterministic weight construction without questions or accuracy scores.

Run: python model/verify_reproducibility.py
Two sequential fresh processes use different Python hash seeds. Each builds
 twice, releasing the first tensors before rebuilding. The supervisor uses only
stdlib; no two full models are resident at once. Hash tensor buffers directly,
without allocating serialized copies. This file is not read by build().
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LIMIT_BYTES = 2_000_000_000


def fingerprint(weights):
    result = {}
    for name, tensor in sorted(weights.items()):
        assert tensor.device.type == 'cpu' and tensor.is_contiguous(), name
        result[name] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(memoryview(tensor.numpy()).cast('B')).hexdigest(),
        }
    return result


def worker():
    import torch
    import build

    torch.set_num_threads(4)
    grammar = frozenset(build.STOP)
    reference = None
    for _ in range(2):
        weights = build.build()
        current = fingerprint(weights)
        assert frozenset(build.STOP) == grammar, 'build() mutated its grammar configuration'
        if reference is not None:
            assert current == reference, 'Repeated builds produced different weights'
        reference = current
        del weights
        gc.collect()
    print(json.dumps({'weights': reference,
                      'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def main():
    started = time.monotonic()
    reference = None
    max_worker_kib = 0
    for seed in ('0', '37'):
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--worker'],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
        )
        if completed.returncode:
            raise RuntimeError(f'Reproducibility worker failed (seed {seed}):\n'
                               f'{completed.stdout}\n{completed.stderr}')
        result = json.loads(completed.stdout)
        if reference is not None:
            assert result['weights'] == reference, 'Weights depend on Python hash seed'
        reference = result['weights']
        max_worker_kib = max(max_worker_kib, result['peak_kib'])
    supervisor_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert (max_worker_kib + supervisor_kib) * 1024 < LIMIT_BYTES, 'Exceeded memory budget'
    elapsed = time.monotonic() - started
    assert elapsed < 20 * 60, 'Exceeded 20-minute verification budget'
    digest = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    print('REPRODUCIBILITY OK: four builds, two fresh processes, two Python hash seeds.')
    print(f'weight_fingerprint={digest}')
    print(f'seconds={elapsed:.3f} max_worker_rss_kib={max_worker_kib} '
          f'supervisor_rss_kib={supervisor_kib}')
    print('No questions or accuracy scores were used.')


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: python verify_reproducibility.py')

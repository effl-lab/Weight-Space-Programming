"""Build /srv/work/build_weights.py and check names and shapes against /srv/work/arch_spec.json.

Does not run the network. Usage: python /srv/work/tools/check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/srv/lib")
from harness import buildcheck  # noqa: E402
from wsp import archs  # noqa: E402

arch = json.loads(Path("/srv/work/arch_spec.json").read_text())["arch"]
weights, problems = buildcheck.build(Path("/srv/work/build_weights.py"))
problems = problems or archs.validate(arch, weights)
if problems:
    print("CHECK FAILED")
    print("\n".join(problems))
    sys.exit(1)
print("CHECK OK")
for k, v in weights.items():
    v = np.asarray(v, dtype=np.float64)
    print(f"{k:22s} {str(v.shape):16s} mean {v.mean():+.4f} std {v.std():.4f} min {v.min():+.4f} max {v.max():+.4f}")

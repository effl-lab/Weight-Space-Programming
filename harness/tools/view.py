"""Render training images as an upscaled PNG grid for viewing.

Usage: python /srv/work/tools/view.py --class cat --n 48 --out /srv/work/views/cat.png
       python /srv/work/tools/view.py --index 0 1 2 3 --out /srv/work/views/some.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ap = argparse.ArgumentParser()
ap.add_argument("--class", dest="cls", choices=CLASSES)
ap.add_argument("--index", type=int, nargs="+")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--scale", type=int, default=3)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
d = np.load("/srv/work/data/train.npz")
x, y = d["x"], d["y"]
if a.index:
    idx = np.array(a.index)
elif a.cls:
    idx = np.flatnonzero(y == CLASSES.index(a.cls))[a.offset:a.offset + a.n]
else:
    idx = np.arange(a.offset, a.offset + a.n)
cols = 8
rows = -(-len(idx) // cols)
s = 32 * a.scale
canvas = Image.new("RGB", (cols * (s + 2), rows * (s + 2)), "white")
for i, j in enumerate(idx):
    im = Image.fromarray(x[j].transpose(1, 2, 0)).resize((s, s), Image.NEAREST)
    canvas.paste(im, ((i % cols) * (s + 2), (i // cols) * (s + 2)))
a.out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(a.out)
print(f"saved {a.out}: {len(idx)} images, indices {idx[:8].tolist()}..., labels {[CLASSES[k] for k in y[idx[:8]]]}...")

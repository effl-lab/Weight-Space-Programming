"""CIFAR-10 splits for weight-writing runs; `--cifar100` builds the same files for CIFAR-100 (fine labels).

train.npz     50,000 train images (visible to the writer in open conditions)
practice.npz  fixed 512-image subset of train, used by the forward tool
test.npz      10,000 test images, never mounted into a writer sandbox
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wsp import paths

PARQUET_DIR = paths.HF_HUB / "datasets--uoft-cs--cifar10/snapshots/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text"
DATA_ROOT = paths.DATA_ROOT
CIFAR_DIR = DATA_ROOT / "cifar"
CIFAR100_REPO = "uoft-cs/cifar100"
CIFAR100_DIR = DATA_ROOT / "cifar100"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
PRACTICE_SIZE = 512
PRACTICE_SEED = 0


def _decode(parquet_path, label="label"):
    df = pd.read_parquet(parquet_path)
    images = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))
                       for r in df["img"]])
    return images.transpose(0, 3, 1, 2).copy(), df[label].to_numpy(np.int64)


def build_splits(out_dir=CIFAR_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr = _decode(PARQUET_DIR / "train-00000-of-00001.parquet")
    x_te, y_te = _decode(PARQUET_DIR / "test-00000-of-00001.parquet")
    assert x_tr.shape == (50000, 3, 32, 32) and x_te.shape == (10000, 3, 32, 32)
    idx = np.sort(np.random.default_rng(PRACTICE_SEED).choice(len(x_tr), PRACTICE_SIZE, replace=False))
    np.savez(out_dir / "train.npz", x=x_tr, y=y_tr)
    np.savez(out_dir / "practice.npz", x=x_tr[idx], y=y_tr[idx], train_index=idx)
    np.savez(out_dir / "test.npz", x=x_te, y=y_te)
    for name, y in [("train", y_tr), ("practice", y_tr[idx]), ("test", y_te)]:
        print(name, len(y), np.bincount(y, minlength=10).tolist())


def build_cifar100(out_dir=CIFAR100_DIR):
    """train.npz, practice.npz, test.npz as for CIFAR-10, and classes.json (fine label index -> name)."""
    import json
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: hf_hub_download(CIFAR100_REPO, f"cifar100/{s}-00000-of-00001.parquet", repo_type="dataset")
             for s in ("train", "test")}
    (x_tr, y_tr), (x_te, y_te) = (_decode(paths[s], "fine_label") for s in ("train", "test"))
    assert x_tr.shape == (50000, 3, 32, 32) and x_te.shape == (10000, 3, 32, 32)
    names = json.loads(pq.read_schema(paths["train"]).metadata[b"huggingface"])["info"]["features"]["fine_label"]["names"]
    assert len(names) == 100 and int(y_tr.max()) == 99
    (out_dir / "classes.json").write_text(json.dumps(names))
    idx = np.sort(np.random.default_rng(PRACTICE_SEED).choice(len(x_tr), PRACTICE_SIZE, replace=False))
    np.savez(out_dir / "train.npz", x=x_tr, y=y_tr)
    np.savez(out_dir / "practice.npz", x=x_tr[idx], y=y_tr[idx], train_index=idx)
    np.savez(out_dir / "test.npz", x=x_te, y=y_te)
    print("cifar100", len(y_tr), len(y_te), np.bincount(y_tr).min(), np.bincount(y_tr).max(), names[:5])


def load(split, root=CIFAR_DIR):
    """Returns float32 images in [0, 1], shape (N, 3, 32, 32), and int64 labels."""
    d = np.load(root / f"{split}.npz")
    return d["x"].astype(np.float32) / 255.0, d["y"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CIFAR_DIR)
    ap.add_argument("--cifar100", action="store_true")
    a = ap.parse_args()
    build_cifar100() if a.cifar100 else build_splits(a.out)

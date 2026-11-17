"""AG News splits for weight-writing runs.

train.npz            120,000 token-id rows and labels (open conditions)
train.jsonl          the same items as text with class names (open conditions)
train_texts.jsonl    the same texts without labels, in a seeded random order with new indices (view conditions)
practice.npz         fixed 512-item subset of train, used by the forward tool
test.npz             7,600 test items, never mounted into a writer sandbox
vocab.txt            id -> token, one per line (id 0 <pad>, id 1 <unk>); part of the architecture

The cipher variant (CIPHER_DIR) relabels word ids 2..2001 with a secret random permutation and has no vocab.txt, so
the writer sees which ids co-occur but not which words they are. train_ids.npz replaces train_texts.jsonl for view
runs. The permutation lives here only: this module is not mounted into writer sandboxes.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from wsp import paths

REPO_ID = "fancyzhx/ag_news"
AGNEWS_DIR = paths.DATA_ROOT / "agnews"
CIPHER_DIR = paths.DATA_ROOT / "agnews_cipher"
CIPHER_SEED = 20261109
CLASSES = ["world", "sports", "business", "scitech"]
VOCAB_SIZE = 2000
SEQ_LEN = 64
PAD, UNK = 0, 1
TOKEN = re.compile(r"[a-z0-9']+")
PRACTICE_SIZE = 512
PRACTICE_SEED = 0
TEXTS_ORDER_SEED = 1  # the original order groups labels (lag-1 agreement 60%), so view runs get a shuffle


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def encode(texts: list[str], vocab: list[str]) -> np.ndarray:
    index = {t: i for i, t in enumerate(vocab)}
    out = np.full((len(texts), SEQ_LEN), PAD, dtype=np.int64)
    for r, text in enumerate(texts):
        ids = [index.get(t, UNK) for t in tokenize(text)][:SEQ_LEN]
        out[r, :len(ids)] = ids
    return out


def _read(split: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_parquet(hf_hub_download(REPO_ID, f"data/{split}-00000-of-00001.parquet", repo_type="dataset"))
    return df["text"].tolist(), df["label"].to_numpy(np.int64)


def build_splits(out_dir: Path = AGNEWS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_text, y_tr = _read("train")
    te_text, y_te = _read("test")
    assert len(tr_text) == 120_000 and len(te_text) == 7_600
    counts = Counter(t for text in tr_text for t in tokenize(text))
    vocab = ["<pad>", "<unk>"] + [t for t, _ in counts.most_common(VOCAB_SIZE)]
    (out_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    x_tr, x_te = encode(tr_text, vocab), encode(te_text, vocab)
    idx = np.sort(np.random.default_rng(PRACTICE_SEED).choice(len(x_tr), PRACTICE_SIZE, replace=False))
    np.savez(out_dir / "train.npz", x=x_tr, y=y_tr)
    np.savez(out_dir / "practice.npz", x=x_tr[idx], y=y_tr[idx], train_index=idx)
    np.savez(out_dir / "test.npz", x=x_te, y=y_te)
    with (out_dir / "train.jsonl").open("w") as f:
        for i, (t, y) in enumerate(zip(tr_text, y_tr)):
            f.write(json.dumps({"index": i, "label": CLASSES[y], "text": t}) + "\n")
    write_unlabelled_texts(tr_text, out_dir)
    unk = (x_tr == UNK).sum() / (x_tr != PAD).sum()
    for name, y in [("train", y_tr), ("practice", y_tr[idx]), ("test", y_te)]:
        print(name, len(y), np.bincount(y, minlength=len(CLASSES)).tolist())
    print(f"unknown-token share {unk:.3f}, truncated rows {np.mean(x_tr[:, -1] != PAD):.3f}")


def write_unlabelled_texts(texts: list[str], out_dir: Path = AGNEWS_DIR) -> None:
    order = np.random.default_rng(TEXTS_ORDER_SEED).permutation(len(texts))
    with (out_dir / "train_texts.jsonl").open("w") as f:
        for i, j in enumerate(order):
            f.write(json.dumps({"index": i, "text": texts[j]}) + "\n")


def cipher_map() -> np.ndarray:
    """old id -> cipher id; padding and unknown keep ids 0 and 1."""
    return np.concatenate([[PAD, UNK], 2 + np.random.default_rng(CIPHER_SEED).permutation(VOCAB_SIZE)])


def build_cipher(src: Path = AGNEWS_DIR, out_dir: Path = CIPHER_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = cipher_map()
    for split in ("train", "practice", "test"):
        d = dict(np.load(src / f"{split}.npz"))
        d["x"] = m[d["x"]]
        np.savez(out_dir / f"{split}.npz", **d)
    x = np.load(out_dir / "train.npz")["x"]
    order = np.random.default_rng(TEXTS_ORDER_SEED).permutation(len(x))
    np.savez(out_dir / "train_ids.npz", x=x[order])
    print("cipher splits written to", out_dir)


def load_cipher(split: str) -> tuple[np.ndarray, np.ndarray]:
    return load(split, CIPHER_DIR)


def load(split: str, root: Path = AGNEWS_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Returns int64 token ids, shape (N, SEQ_LEN), and int64 labels."""
    d = np.load(root / f"{split}.npz")
    return d["x"], d["y"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=AGNEWS_DIR)
    ap.add_argument("--cipher", action="store_true", help="build CIPHER_DIR from the plain splits")
    a = ap.parse_args()
    build_cipher() if a.cipher else build_splits(a.out)

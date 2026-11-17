"""CIFAR-10 as bags of patch words: the bow network of wsp.text_archs on images.

An image is cut into VW_PATCH x VW_PATCH patches at stride VW_STRIDE. Each patch is contrast-normalised and given
the word whose template scores highest. The network counts each word in each image quadrant and scores classes
like bow: F.linear(log1p(counts), score).

vwords     the word templates are fixed and part of the architecture (a k-means codebook of whitened patches of
           the unlabelled training images, built by `python -m wsp.vision_words`); the parameters are the score
           table only
           train_words.npz holds the word of every patch of every training image, without labels (view runs)
vwords_gd  the same computation with the templates as parameters. The forward assigns each patch to one word;
           the gradient is that of a softmax over the word scores (straight-through)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wsp import paths

VW_PATCH = 6
VW_STRIDE = 2
VW_SIDE = (32 - VW_PATCH) // VW_STRIDE + 1  # patches per image side
VW_WORDS = 128
VW_QUADRANTS = 4
VW_DIM = 3 * VW_PATCH * VW_PATCH
VW_CLASSES = 10
NORM_EPS = 0.04
SOFT_TEMP = 0.25  # softmax temperature of the gradient path; the forward is a hard assignment
CODEBOOK = Path("cifar/patch_words.npz")  # under WSP_DATA_ROOT, resolved on use (writer sandboxes have none)
_CODEBOOK: dict = {}


def patch_rows(x):
    """(N, 3, 32, 32) -> (N, VW_SIDE**2, VW_DIM): every patch minus its mean, over its standard deviation."""
    p = F.unfold(x, VW_PATCH, stride=VW_STRIDE).transpose(1, 2)
    p = p - p.mean(2, keepdim=True)
    return p / (p.std(2, keepdim=True) + NORM_EPS)


def quadrant_of_patch(device):
    """(VW_SIDE**2, 4) one-hot: the image quadrant (0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right)."""
    half = torch.arange(VW_SIDE, device=device) // ((VW_SIDE + 1) // 2)
    return F.one_hot((half[:, None] * 2 + half[None, :]).reshape(-1), VW_QUADRANTS).float()


def word_counts(templates, offsets, x):
    """(N, VW_WORDS * 4): how many patches of each quadrant got each word; column = word * 4 + quadrant."""
    scores = patch_rows(x) @ templates.T + offsets
    soft = F.softmax(scores * SOFT_TEMP, dim=2)
    hard = F.one_hot(scores.argmax(2), scores.shape[2]).float()
    assign = hard + soft - soft.detach()
    return torch.einsum("npk,pq->nkq", assign, quadrant_of_patch(x.device)).flatten(1)


def codebook(device):
    """The fixed word templates (VW_WORDS, VW_DIM) and offsets (VW_WORDS,); the same for every network."""
    if device not in _CODEBOOK:
        with np.load(paths.DATA_ROOT / CODEBOOK) as z:
            _CODEBOOK[device] = (torch.from_numpy(z["words.weight"]).to(device),
                                 torch.from_numpy(z["words.bias"]).to(device))
    return _CODEBOOK[device]


def word_map(x):
    """(N, 3, 32, 32) -> (N, VW_SIDE, VW_SIDE) int64: the word of every patch, rows top to bottom."""
    templates, offsets = codebook(x.device)
    return (patch_rows(x) @ templates.T + offsets).argmax(2).reshape(len(x), VW_SIDE, VW_SIDE)


def _vwords_shapes():
    return {"score.weight": (VW_CLASSES, VW_WORDS * VW_QUADRANTS), "score.bias": (VW_CLASSES,)}


def _vwords_forward(w, x):
    return F.linear(torch.log1p(word_counts(*codebook(x.device), x)), w["score.weight"], w["score.bias"])


def _vwords_gd_shapes():
    return {"words.weight": (VW_WORDS, VW_DIM), "words.bias": (VW_WORDS,), **_vwords_shapes()}


def _vwords_gd_forward(w, x):
    counts = word_counts(w["words.weight"], w["words.bias"], x)
    return F.linear(torch.log1p(counts), w["score.weight"], w["score.bias"])


def with_codebook(weights: dict) -> dict:
    """A written vwords network as a vwords_gd network: the fixed templates become parameters."""
    templates, offsets = codebook(torch.device("cpu"))
    return {"words.weight": templates.clone(), "words.bias": offsets.clone(), **weights}


def fit_codebook(x: torch.Tensor, seed: int = 0, iters: int = 30) -> dict[str, np.ndarray]:
    """k-means on ZCA-whitened patches of unlabelled images, folded into linear templates over patch_rows.

    For a whitened patch z and centroid c the nearest centroid maximises 2 z.c - |c|^2, which is linear in the
    normalised patch p (z = (p - mean) @ zca): template 2 zca c, offset -2 (mean @ zca).c - |c|^2.
    """
    g = torch.Generator().manual_seed(seed)
    rows = patch_rows(x[torch.randperm(len(x), generator=g)[:5000]]).reshape(-1, VW_DIM)
    rows = rows[torch.randperm(len(rows), generator=g)[:300_000]]
    mean = rows.mean(0)
    u, s, _ = torch.linalg.svd(torch.cov((rows - mean).T))
    zca = u @ torch.diag((s + 0.1).rsqrt()) @ u.T
    z = (rows - mean) @ zca
    cent = z[torch.randperm(len(z), generator=g)[:VW_WORDS]].clone()
    for _ in range(iters):
        a = torch.cat([(c @ cent.T * 2 - (cent ** 2).sum(1)).argmax(1) for c in z.split(20_000)])
        n = torch.bincount(a, minlength=VW_WORDS)
        new = torch.zeros_like(cent).index_add_(0, a, z) / n.clamp(min=1)[:, None]
        cent = torch.where(n[:, None] > 0, new, cent)
    templates = 2 * cent @ zca.T
    offsets = -2 * (mean @ zca) @ cent.T - (cent ** 2).sum(1)
    return {"words.weight": templates.numpy().astype(np.float32), "words.bias": offsets.numpy().astype(np.float32)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / CODEBOOK)
    a = ap.parse_args()
    from wsp import data
    torch.set_num_threads(4)
    with np.load(data.CIFAR_DIR / "train.npz") as d:
        images = torch.from_numpy(d["x"]).float() / 255
    np.savez(a.out, **fit_codebook(images))
    print(f"wrote {a.out}")
    with torch.no_grad():
        words = torch.cat([word_map(c) for c in images.split(1000)]).numpy().astype(np.uint8)
    np.savez(data.CIFAR_DIR / "train_words.npz", words=words)
    print(f"wrote train_words.npz {words.shape}")

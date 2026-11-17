"""CIFAR-10 as sequences of image tokens from a frozen tokenizer that was trained without labels.

vq    the VQGAN of amused/amused-256: 8,192 codes, one per 16 x 16 pixels of the resized image
dino  patch features of facebook/dinov2-base (self-supervised, 14 x 14 pixel patches), each given the nearest of
      K centroids; the centroids are a spherical k-means of the patch features of unlabelled training images

`python -m wsp.vision_tokens` writes token_dir()/<name>/{train,test}.npz for every entry of TOKENIZERS
(x: (N, T) int16 token ids, rows top to bottom; y: labels) and, for dino, centroids.npz.
`--dataset cifar100 --kind dino` does the same for the dino tokenizers on CIFAR-100, under cifar100_token_dir(),
with centroids fitted on unlabelled CIFAR-100 training images.
`python -m wsp.vision_tokens --task-files` writes what writer runs on TOKENIZER need besides those.

vtok  the bow network of wsp.text_archs over the tokens of TOKENIZER: a class score per token.

`python -m wsp.vision_tokens --embeddings` writes embed_dir()/{train,test,practice}.npz: per image the mean of its
unit-norm dino patch features at 224 pixels, standardised per dimension with the mean and standard deviation over the
unlabelled training images (stats.npz). Same encoder as dino_r224_*, without the step to discrete tokens.
vemb  a linear layer over those VE_DIM embedding dimensions: a class score per dimension.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wsp import paths

TOKENIZERS = {  # name -> (kind, side of the resized image in pixels, vocabulary)
    "vq_r128": ("vq", 128, 8192),
    "vq_r256": ("vq", 256, 8192),
    "dino_r112_k256": ("dino", 112, 256),
    "dino_r112_k1024": ("dino", 112, 1024),
    "dino_r224_k256": ("dino", 224, 256),
    "dino_r224_k1024": ("dino", 224, 1024),
    "dino_r224_k4096": ("dino", 224, 4096),
}
TOKENIZER = "dino_r224_k256"  # read by vtok; picked from docs/experiments/tokenizer_purity.json
VT_VOCAB = 256
VT_TOKENS = 256
VT_CLASSES = 10
VQ_REPO = "amused/amused-256"
DINO_REPO = "facebook/dinov2-base"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_SIDE = 224
VE_DIM = 768
FIT_IMAGES = 5000
FIT_ROWS = 600_000
BATCH = 50  # the VQGAN encoder at 256 x 256 needs about 30 MB per image
SEED = 0


# Resolved on use: writer sandboxes import this module without a data root.
def token_dir() -> Path:
    return paths.DATA_ROOT / "cifar_tokens"


def cifar100_token_dir() -> Path:
    return paths.DATA_ROOT / "cifar100_tokens"


def embed_dir() -> Path:
    return token_dir() / "dino_r224_mean"


def token_counts(x):
    """(N, 256) token ids -> (N, 256) number of the image's patches that got each token."""
    return torch.zeros(x.shape[0], VT_VOCAB, device=x.device).scatter_add_(1, x, torch.ones(x.shape, device=x.device))


def _vtok_shapes():
    return {"score.weight": (VT_CLASSES, VT_VOCAB), "score.bias": (VT_CLASSES,)}


def _vtok_forward(w, x):
    return F.linear(torch.log1p(token_counts(x)), w["score.weight"], w["score.bias"])


def _vemb_shapes():
    return {"score.weight": (VT_CLASSES, VE_DIM), "score.bias": (VT_CLASSES,)}


def _vemb_forward(w, x):
    return F.linear(x, w["score.weight"], w["score.bias"])


def load_embeddings(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns float32 standardised embeddings, shape (N, VE_DIM), and int64 labels."""
    d = np.load(embed_dir() / f"{split}.npz")
    return d["x"].astype(np.float32), d["y"]


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns int64 token ids of TOKENIZER, shape (N, VT_TOKENS), and int64 labels."""
    d = np.load(token_dir() / TOKENIZER / f"{split}.npz")
    return d["x"].astype(np.int64), d["y"]


def write_task_files() -> None:
    """practice.npz (the CIFAR practice images as tokens) and train_tokens.npz (train tokens without labels)."""
    from wsp import data
    out = token_dir() / TOKENIZER
    train = np.load(out / "train.npz")
    idx = np.load(data.CIFAR_DIR / "practice.npz")["train_index"]
    np.savez(out / "practice.npz", x=train["x"][idx], y=train["y"][idx], train_index=idx)
    np.savez(out / "train_tokens.npz", x=train["x"].astype(np.uint8))


def resize(x: torch.Tensor, side: int) -> torch.Tensor:
    return F.interpolate(x, size=side, mode="bicubic", align_corners=False).clamp(0, 1)


def vq_model(device):
    from diffusers import VQModel
    return VQModel.from_pretrained(VQ_REPO, subfolder="vqvae").to(device).eval()


def vq_tokens(model, x: torch.Tensor, side: int) -> torch.Tensor:
    latents = model.encode(resize(x, side)).latents
    return model.quantize(latents)[2][2].reshape(len(x), -1)


def dino_model(device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(DINO_REPO).to(device).eval().half()


def dino_features(model, x: torch.Tensor, side: int) -> torch.Tensor:
    """(N, 3, 32, 32) in [0, 1] -> (N, (side // 14) ** 2, 768) unit-norm patch features."""
    mean, std = (torch.tensor(v, device=x.device).reshape(1, 3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
    h = model(pixel_values=((resize(x, side) - mean) / std).half()).last_hidden_state[:, 1:]
    return F.normalize(h.float(), dim=2)


def nearest(z: torch.Tensor, cent: torch.Tensor) -> torch.Tensor:
    return torch.cat([(c @ cent.T).argmax(1) for c in z.split(65536)])


def spherical_kmeans(z: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    cent = z[torch.randperm(len(z), generator=g)[:k].to(z.device)].clone()
    for _ in range(iters):
        a = nearest(z, cent)
        n = torch.bincount(a, minlength=k)
        mean = F.normalize(torch.zeros_like(cent).index_add_(0, a, z), dim=1)
        cent = torch.where(n[:, None] > 0, mean, cent)
    return cent


def batches(x: np.ndarray, device):
    for i in range(0, len(x), BATCH):
        yield torch.from_numpy(x[i:i + BATCH]).to(device)


def save(out: Path, split: str, tokens: torch.Tensor, y: np.ndarray) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{split}.npz", x=tokens.cpu().numpy().astype(np.int16), y=y)
    print(f"{out.name} {split} {tuple(tokens.shape)} distinct {int(tokens.unique().numel())}", flush=True)


@torch.no_grad()
def write_embeddings() -> None:
    from wsp import data
    device = torch.device("cuda")
    model = dino_model(device)
    emb = {}
    for split in ("train", "test"):
        x, y = data.load(split)
        emb[split] = (torch.cat([dino_features(model, b, EMBED_SIDE).mean(1) for b in batches(x, device)]).cpu(), y)
    mu, sd = emb["train"][0].mean(0), emb["train"][0].std(0)
    embed_dir().mkdir(parents=True, exist_ok=True)
    np.savez(embed_dir() / "stats.npz", mean=mu.numpy(), std=sd.numpy())
    for split, (z, y) in emb.items():
        np.savez(embed_dir() / f"{split}.npz", x=((z - mu) / sd).numpy().astype(np.float16), y=y)
        print(f"{embed_dir().name} {split} {tuple(z.shape)}", flush=True)
    idx = np.load(data.CIFAR_DIR / "practice.npz")["train_index"]
    train = np.load(embed_dir() / "train.npz")
    np.savez(embed_dir() / "practice.npz", x=train["x"][idx], y=train["y"][idx], train_index=idx)


@torch.no_grad()
def main(dataset: str, kinds: tuple[str, ...]) -> None:
    from wsp import data
    device = torch.device("cuda")
    src, dst = {"cifar": (data.CIFAR_DIR, token_dir()), "cifar100": (data.CIFAR100_DIR, cifar100_token_dir())}[dataset]
    splits = {s: data.load(s, src) for s in ("train", "test")}
    x_tr = splits["train"][0]
    fit_idx = np.sort(np.random.default_rng(SEED).choice(len(x_tr), FIT_IMAGES, replace=False))

    if "vq" in kinds:
        model = vq_model(device)
        sample = torch.from_numpy(x_tr[:64]).to(device)
        rec = model.decode(model.encode(resize(sample, 256)).latents).sample
        print(f"vq reconstruction mse at 256 x 256: {F.mse_loss(rec, resize(sample, 256)).item():.5f}")
        for name, (kind, side, _) in TOKENIZERS.items():
            if kind == "vq":
                for split, (x, y) in splits.items():
                    save(dst / name, split, torch.cat([vq_tokens(model, b, side) for b in batches(x, device)]), y)
        del model

    if "dino" in kinds:
        model = dino_model(device)
        for side in sorted({side for kind, side, _ in TOKENIZERS.values() if kind == "dino"}):
            names = {n: k for n, (kind, s, k) in TOKENIZERS.items() if kind == "dino" and s == side}
            z = torch.cat([dino_features(model, b, side).flatten(0, 1) for b in batches(x_tr[fit_idx], device)])
            z = z[torch.randperm(len(z), generator=torch.Generator().manual_seed(SEED))[:FIT_ROWS].to(device)]
            cents = {n: spherical_kmeans(z, k) for n, k in names.items()}
            del z
            for n, cent in cents.items():
                (dst / n).mkdir(parents=True, exist_ok=True)
                np.savez(dst / n / "centroids.npz", centroids=cent.cpu().numpy(), fit_index=fit_idx)
            for split, (x, y) in splits.items():
                parts = {n: [] for n in names}
                for b in batches(x, device):
                    f = dino_features(model, b, side)
                    for n, cent in cents.items():
                        parts[n].append(nearest(f.flatten(0, 1), cent).reshape(len(b), -1))
                for n in names:
                    save(dst / n, split, torch.cat(parts[n]), y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-files", action="store_true")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--dataset", default="cifar", choices=["cifar", "cifar100"])
    ap.add_argument("--kind", nargs="+", default=["vq", "dino"], choices=["vq", "dino"])
    a = ap.parse_args()
    if a.embeddings:
        write_embeddings()
    else:
        write_task_files() if a.task_files else main(a.dataset, tuple(a.kind))

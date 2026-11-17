"""Adam baselines for the multiple-choice pair scorers of wsp.given_archs on ARC.

Each (arch, N, seed) trains on N questions drawn at random from the ARC train split (the answer is a position, so
there is no class to balance). The vocabulary is the VOCAB most frequent words of the train split's questions
and options. For each learning rate the best-validation checkpoint is kept; the rate with the best validation
accuracy is reported with its test accuracy. Finished cells are skipped on restart.

With --init-from, training starts from a written model folder (its vocab.txt and build.py) that uses the same
arch.py; the untrained written model competes as a step-0 checkpoint.
"""
import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wsp import arc, free_model, given_archs, paths

OUT_DIR = paths.DATA_ROOT / "baselines_arc"
VOCAB = 8000
BATCH = 32
EVAL_EVERY = 50
LRS = (3e-3, 1e-3, 3e-4)
FINETUNE_LRS = (1e-3, 3e-4, 1e-4, 3e-5)


def gd_vocab() -> list[str]:
    q, c, _ = arc.load("train")
    counts = Counter(w for text in q + [o for opts in c for o in opts] for w in free_model.words(text))
    return [w for w, _ in counts.most_common(VOCAB)]


def tensors(vocab: list[str], split: str, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, c, y = arc.load(split)
    qi, oi = free_model.encode(vocab, q, c)
    return torch.from_numpy(qi).to(device), torch.from_numpy(oi).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def accuracy(fwd, w, part) -> float:
    q, o, y = part
    hits = sum((fwd(w, q[i:i + 256], o[i:i + 256]).argmax(1) == y[i:i + 256]).sum().item() for i in range(0, len(y), 256))
    return hits / len(y)


def train_one(fwd, start: dict, parts: dict, n: int, seed: int, lr: float, keep_start: bool) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    q, o, y = parts["train"]
    idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:n].to(y.device)
    w = {k: v.clone().requires_grad_() for k, v in start.items()}
    opt = torch.optim.Adam(w.values(), lr=lr)
    steps = max(1500, 30 * n // BATCH)
    best = (accuracy(fwd, start, parts["validation"]), {k: v.detach().cpu() for k, v in start.items()}, 0) \
        if keep_start else (-1.0, None, 0)
    t0 = time.time()
    for step in range(1, steps + 1):
        b = idx[torch.randint(0, len(idx), (min(BATCH, len(idx)),), device=idx.device)]
        loss = F.cross_entropy(fwd(w, q[b], o[b]), y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            acc = accuracy(fwd, w, parts["validation"])
            if acc > best[0]:
                best = (acc, {k: v.detach().cpu().clone() for k, v in w.items()}, step)
    return {"lr": lr, "steps": steps, "best_step": best[2], "val_accuracy": best[0],
            "seconds": round(time.time() - t0, 1)}, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(given_archs.NAMES))
    ap.add_argument("--n-train", nargs="+", type=int, default=[30, 100, 300, 1000, 3358])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--init-from", nargs="+", default=[], help="written model folders that use one of --archs")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    jobs = [(name, None) for name in a.archs] if not a.init_from else [(None, Path(p)) for p in a.init_from]
    for name, folder in jobs:
        if folder is None:
            vocab, tag = gd_vocab(), ""
            start = given_archs.init(name, len(vocab) + 2, 0)
        else:
            source = (folder / "arch.py").read_text()
            name = next(n for n in given_archs.NAMES if given_archs.source(n) == source)
            vocab, tag = free_model.vocab_of(folder), f"_from-{folder.parent.parent.name}"
            sys.path.insert(0, str(folder))
            build = importlib.import_module("build")
            cwd = Path.cwd()
            os.chdir(folder)
            start = {k: v.float().to_dense() for k, v in build.build().items()}
            os.chdir(cwd)
            sys.path.remove(str(folder))
            del sys.modules["build"]
        fwd = importlib.import_module(f"wsp.given_archs.{name}").forward
        parts = {s: tensors(vocab, s, device) for s in ("train", "validation", "test")}
        n_params = sum(math.prod(v.shape) for v in start.values())
        for n in a.n_train:
            for seed in a.seeds:
                out = OUT_DIR / f"gd_{name}_N{n}_s{seed}{tag}.json"
                if out.exists():
                    continue
                if folder is None:
                    start = given_archs.init(name, len(vocab) + 2, seed)
                init = {k: v.to(device) for k, v in start.items()}
                sweep = [train_one(fwd, init, parts, n, seed, lr, folder is not None)
                         for lr in (LRS if folder is None else FINETUNE_LRS)]
                info, w = max(sweep, key=lambda r: r[0]["val_accuracy"])
                w = {k: v.to(device) for k, v in w.items()}
                q, o, y = parts["test"]
                with torch.no_grad():
                    pred = torch.cat([fwd(w, q[i:i + 256], o[i:i + 256]).argmax(1) for i in range(0, len(y), 256)])
                hit = (pred == y).cpu().numpy()
                sub = arc.subset("test")
                res = {"arch": name, "n_train": n, "seed": seed, "init_from": None if folder is None else str(folder),
                       "parameters": n_params, "vocab": len(vocab), **info, "test_accuracy": float(hit.mean()),
                       "test_by_subset": {s: float(hit[sub == s].mean()) for s in arc.SUBSETS},
                       "lr_sweep": [r[0] for r in sweep], "device": a.device}
                out.write_text(json.dumps(res) + "\n")
                print(json.dumps({k: res[k] for k in ("arch", "n_train", "seed", "lr", "test_accuracy")}), flush=True)


if __name__ == "__main__":
    main()

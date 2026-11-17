"""Adam baselines on the same functional architectures as the writer runs.

Each (arch, N, seed) trains on N class-balanced train images drawn from a pool that excludes
a fixed class-stratified 5,000-image validation split. For each learning rate in LRS it keeps the
best-validation checkpoint; the learning rate with the best validation accuracy is reported with
its test accuracy. Finished cells are skipped on restart.

With --init-from, training starts from a writer run's weights.npz instead of a random init, and the
untrained writer weights compete as a step-0 checkpoint, so fine-tuning is reported only when it helps
on validation. --lift first re-expresses those weights in the larger architecture of archs.LIFTS.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wsp import archs, data, evaluate, tasks

OUT_DIR = data.DATA_ROOT / "baselines"
RUNS_DIR = data.DATA_ROOT / "runs"
VAL_PER_CLASS = 500
BATCH = 128
EVAL_EVERY = 50
LRS = (3e-3, 1e-3, 3e-4)
FINETUNE_LRS = LRS + (1e-4, 3e-5)


def split_indices(n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vrng = np.random.default_rng(1234)
    n_classes = int(TRAIN_Y.max()) + 1
    val = np.concatenate([vrng.choice(np.flatnonzero(TRAIN_Y == c), VAL_PER_CLASS, replace=False)
                          for c in range(n_classes)])
    pool = np.setdiff1d(np.arange(len(TRAIN_Y)), val)
    rng = np.random.default_rng(seed)
    per_class = n_train // n_classes
    chosen = np.concatenate([rng.choice(pool[TRAIN_Y[pool] == c], per_class, replace=False)
                             for c in range(n_classes)])
    return chosen, val, pool


def train_one(arch: str, n_train: int, seed: int, lr: float, device: str,
              init: dict[str, torch.Tensor] | None) -> tuple[dict, dict]:
    tr, val, _ = split_indices(n_train, seed)
    x_tr, y_tr = torch.from_numpy(TRAIN_X[tr]).to(device), torch.from_numpy(TRAIN_Y[tr]).to(device)
    x_val, y_val = TRAIN_X[val], TRAIN_Y[val]
    torch.manual_seed(seed)
    start = archs.init_weights(arch, seed) if init is None else init
    w = {k: v.clone().to(device).requires_grad_() for k, v in start.items()}
    opt = torch.optim.Adam(w.values(), lr=lr)
    steps = max(3000, 30 * n_train // BATCH)
    best = (-1.0, None, 0)
    if init is not None:
        best = (evaluate.evaluate(arch, init, x_val, y_val)["accuracy"], init, 0)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(tr), (min(BATCH, len(tr)),), device=device)
        loss = F.cross_entropy(archs.forward(arch, w, x_tr[idx]), y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            cpu_w = {k: v.detach().cpu().clone() for k, v in w.items()}
            acc = evaluate.evaluate(arch, cpu_w, x_val, y_val)["accuracy"]
            if acc is not None and acc > best[0]:
                best = (acc, cpu_w, step)
    return {"lr": lr, "steps": steps, "best_step": best[2], "val_accuracy": best[0],
            "seconds": round(time.time() - t0, 1)}, best[1]


def run_cell(arch: str, n_train: int, seed: int, device: str,
             init: dict[str, torch.Tensor] | None) -> tuple[dict, dict]:
    sweep = [train_one(arch, n_train, seed, lr, device, init) for lr in (LRS if init is None else FINETUNE_LRS)]
    info, weights = max(sweep, key=lambda r: r[0]["val_accuracy"])
    test = evaluate.evaluate(arch, weights, TEST_X, TEST_Y)
    return {"arch": arch, "n_train": n_train, "seed": seed, **info, "test_accuracy": test["accuracy"],
            "per_class": test["per_class"], "lr_sweep": [r[0] for r in sweep]}, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=[a for a in archs.ARCHS if archs.ARCHS[a].task == "cifar"])
    ap.add_argument("--n-train", nargs="+", type=int, default=[100, 1000, 10000, 45000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu is for the small-N cells (N <= 100), where the GPU queue costs more than the run")
    ap.add_argument("--init-from", nargs="+", default=[],
                    help="writer run ids; each is fine-tuned on its own architecture instead of --archs")
    ap.add_argument("--lift", action="store_true",
                    help="with --init-from: fine-tune in the architecture archs.LIFTS maps the run's to")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device; pass --device cpu for small-N cells")
    device = a.device
    global TRAIN_X, TRAIN_Y, TEST_X, TEST_Y
    loaded = None
    jobs = [(arch, None, "") for arch in a.archs] if not a.init_from else [
        (json.loads((RUNS_DIR / r / "result.json").read_text())["arch"], r, f"_from-{r}") for r in a.init_from]
    for arch, run_id, tag in jobs:
        init = None if run_id is None else evaluate.load_weights(arch, RUNS_DIR / run_id / "weights.npz")
        if a.lift:
            arch, lift = archs.LIFTS[arch]
            init = lift(init)
        if loaded != archs.ARCHS[arch].task:
            loaded = archs.ARCHS[arch].task
            TRAIN_X, TRAIN_Y = tasks.TASKS[loaded].load("train")
            TEST_X, TEST_Y = tasks.TASKS[loaded].load("test")
        for n in a.n_train:
            for seed in a.seeds:
                stem = a.out / f"gd_{arch}_N{n}_s{seed}{tag}"
                if stem.with_suffix(".json").exists():
                    continue
                res, w = run_cell(arch, n, seed, device, init)
                res["init_from"] = run_id
                res["device"] = device
                np.savez(stem.with_suffix(".npz"), **{k: v.numpy() for k, v in w.items()})
                stem.with_suffix(".json").write_text(json.dumps(res) + "\n")
                print(json.dumps({k: res[k] for k in ("arch", "n_train", "seed", "lr", "test_accuracy")}), flush=True)


if __name__ == "__main__":
    main()

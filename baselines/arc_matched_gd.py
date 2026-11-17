"""GD accuracy of the transformer of baselines.arc_size_matched at the exact parameter count of individual
writer-designed models, the partners of Figure 1(b): each model is joined there to the transformer of its size (whose
GPU time analysis.inference_speed measured as `matched_transformer`). Same data, optimiser, learning rates and
checkpoint rule as arc_size_matched; all 3,358 ARC train questions, SEEDS seeds. One file per width and seed in
OUT_DIR (arc_size_matched's), named transformer_w<width>_n3358_s<seed>.json.

    python -m baselines.arc_matched_gd --widths 8 24 68 72 76 256 268
"""
import argparse
import json
from pathlib import Path

import torch

from baselines import arc_size_matched as m, train_pair
from wsp import arc

SEEDS = (0, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    vocab = train_pair.gd_vocab()
    parts = {s: train_pair.tensors(vocab, s, device) for s in ("train", "validation", "test")}
    fwd = m.forward_for("transformer")
    for d in a.widths:
        for seed in SEEDS:
            out = m.OUT_DIR / f"transformer_w{d}_n{m.FULL}_s{seed}.json"
            if out.exists():
                continue
            start = {k: v.to(device) for k, v in m.init("transformer", len(vocab) + 2, d, seed).items()}
            sweep = [train_pair.train_one(fwd, start, parts, m.FULL, seed, lr, False) for lr in train_pair.LRS]
            info, w = max(sweep, key=lambda r: r[0]["val_accuracy"])
            w = {k: v.to(device) for k, v in w.items()}
            res = {"family": "transformer", "width": d, "parameters": m.n_params("transformer", len(vocab) + 2, d),
                   "n_train": m.FULL, "seed": seed, **info,
                   "test_accuracy": train_pair.accuracy(fwd, w, parts["test"]), "lr_sweep": [r[0] for r in sweep]}
            out.write_text(json.dumps(res) + "\n")
            print(json.dumps({k: res[k] for k in ("width", "parameters", "seed", "lr", "val_accuracy",
                                                  "test_accuracy")}), flush=True)


if __name__ == "__main__":
    main()

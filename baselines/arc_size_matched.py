"""Gradient-descent baselines on ARC with the parameter count of the writer-designed models.

Four standard families score each (question, option) pair: mlp (mean word embeddings of question and option, then a
one-hidden-layer MLP on [q, o, q*o]), cnn (two width-3 convolutions over question+option, masked max-pool),
transformer (two pre-norm encoder layers, four heads, mean-pool) and ssm (two diagonal linear-recurrence layers with a
learned per-channel decay, mean-pool). The width of each is chosen so that its parameter count is closest to a target:
the non-zero values stored by the best final-generation model of each writer. Data, vocabulary, optimiser, learning
rates and checkpoint selection are those of baselines.train_pair (Adam, best validation step, best rate by validation).
Every cell trains on N questions drawn from ARC train (all 3,358 once; 10, 100 and 1,000 for subsets 0-2).
Usage: python -m baselines.arc_size_matched --families mlp cnn transformer ssm
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines import train_pair
from wsp import arc, free_model

OUT_DIR = Path("docs/experiments/arc_size_matched")
# Numbers stored by the best-validation final-generation model of each writer (docs/experiments/weight_inventory.json):
# non-zero entries of dense tensors plus, for torch sparse tensors, each stored value and its indices.
TARGETS = {"astra": 546_323, "sol": 766_132, "opus": 3_349_538}
FAMILIES = ("mlp", "cnn", "transformer", "ssm")
LAYERS, HEADS = 2, 4
N_TRAIN = (10, 100, 1000)
SEEDS = (0, 1, 2)
FULL = 3358


def shapes(family: str, vocab: int, d: int) -> dict[str, tuple[int, ...]]:
    out = {"embed": (vocab, d)}
    if family == "mlp":
        return out | {"fc1": (3 * d, 2 * d), "fc1.bias": (2 * d,), "score": (2 * d,)}
    if family == "transformer":
        out["pos"] = (free_model.Q_LEN + free_model.O_LEN, d)
    for i in range(LAYERS):
        p = f"block{i}."
        if family == "cnn":
            out |= {p + "conv": (d, d, 3), p + "conv.bias": (d,)}
        elif family == "transformer":
            out |= {p + n: (d, d) for n in ("q", "k", "v", "o")} | {p + n + ".bias": (d,) for n in ("q", "k", "v", "o")}
            out |= {p + "ln1.weight": (d,), p + "ln1.bias": (d,), p + "ln2.weight": (d,), p + "ln2.bias": (d,),
                    p + "fc1": (d, 2 * d), p + "fc1.bias": (2 * d,), p + "fc2": (2 * d, d), p + "fc2.bias": (d,)}
        else:
            out |= {p + "ln.weight": (d,), p + "ln.bias": (d,), p + "in": (d, 2 * d), p + "in.bias": (2 * d,),
                    p + "decay": (d,), p + "out": (d, d), p + "out.bias": (d,)}
    return out | {"ln_f.weight": (d,), "ln_f.bias": (d,), "score": (d,)}


def n_params(family: str, vocab: int, d: int) -> int:
    return sum(math.prod(s) for s in shapes(family, vocab, d).values())


def width_for(family: str, vocab: int, target: int) -> int:
    return min(range(HEADS, 1025, HEADS), key=lambda d: abs(n_params(family, vocab, d) - target))


def init(family: str, vocab: int, d: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, s in shapes(family, vocab, d).items():
        if k.endswith("ln1.weight") or k.endswith("ln2.weight") or k.endswith("ln.weight") or k == "ln_f.weight":
            out[k] = torch.ones(s)
        elif k.endswith("decay"):
            out[k] = torch.linspace(-2, 4, s[0])
        elif len(s) == 1:
            out[k] = torch.zeros(s)
        elif k in ("embed", "pos"):
            out[k] = torch.randn(s, generator=g) * 0.02
        else:
            fan_in = s[1] * s[2] if len(s) == 3 else s[0]
            out[k] = (torch.rand(s, generator=g) * 2 - 1) * math.sqrt(3 / fan_in)
    return out


def pairs(q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    n = q.shape[0]
    return torch.cat([q[:, None, :].expand(n, 4, q.shape[1]), o], dim=2).reshape(n * 4, q.shape[1] + o.shape[2])


def pooled(h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (h * real[..., None]).sum(-2) / real.sum(-1, keepdim=True).clamp(min=1)


def bag(w: dict, ids: torch.Tensor) -> torch.Tensor:
    return pooled(w["embed"][ids], (ids != 0).float())


def encode_cnn(w: dict, ids: torch.Tensor) -> torch.Tensor:
    real = (ids != 0).float()
    h = w["embed"][ids].transpose(1, 2)
    for i in range(LAYERS):
        p = f"block{i}."
        h = h + F.gelu(F.conv1d(h, w[p + "conv"], w[p + "conv.bias"], padding=1))
    h = F.layer_norm(h.transpose(1, 2), (h.shape[1],), w["ln_f.weight"], w["ln_f.bias"])
    return (h + (real[..., None] - 1) * 1e9).amax(1)


def encode_transformer(w: dict, ids: torch.Tensor) -> torch.Tensor:
    real = (ids != 0).float()
    m, length = ids.shape
    d = w["embed"].shape[1]
    h = w["embed"][ids] + w["pos"][:length]
    for i in range(LAYERS):
        p = f"block{i}."
        x = F.layer_norm(h, (d,), w[p + "ln1.weight"], w[p + "ln1.bias"])
        q, k, v = ((x @ w[p + n] + w[p + n + ".bias"]).reshape(m, length, HEADS, d // HEADS).transpose(1, 2)
                   for n in ("q", "k", "v"))
        att = q @ k.transpose(-1, -2) / math.sqrt(d // HEADS) + (real[:, None, None, :] - 1) * 1e9
        h = h + (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(m, length, d) @ w[p + "o"] + w[p + "o.bias"]
        x = F.layer_norm(h, (d,), w[p + "ln2.weight"], w[p + "ln2.bias"])
        h = h + F.gelu(x @ w[p + "fc1"] + w[p + "fc1.bias"]) @ w[p + "fc2"] + w[p + "fc2.bias"]
    return pooled(F.layer_norm(h, (d,), w["ln_f.weight"], w["ln_f.bias"]), real)


def encode_ssm(w: dict, ids: torch.Tensor) -> torch.Tensor:
    real = (ids != 0).float()
    d = w["embed"].shape[1]
    h = w["embed"][ids]
    for i in range(LAYERS):
        p = f"block{i}."
        u, gate = (F.layer_norm(h, (d,), w[p + "ln.weight"], w[p + "ln.bias"]) @ w[p + "in"] + w[p + "in.bias"]).chunk(2, -1)
        a = torch.sigmoid(w[p + "decay"])
        state, ys = torch.zeros_like(u[:, 0]), []
        for t in range(u.shape[1]):
            state = a * state + (1 - a) * u[:, t] * real[:, t, None]
            ys.append(state)
        h = h + (torch.stack(ys, 1) * F.silu(gate)) @ w[p + "out"] + w[p + "out.bias"]
    return pooled(F.layer_norm(h, (d,), w["ln_f.weight"], w["ln_f.bias"]), real)


ENCODERS = {"cnn": encode_cnn, "transformer": encode_transformer, "ssm": encode_ssm}


def forward_for(family: str):
    if family == "mlp":
        def forward(w, q, o):
            qe, oe = bag(w, q)[:, None, :].expand(-1, 4, -1), bag(w, o)
            return F.gelu(torch.cat([qe, oe, qe * oe], -1) @ w["fc1"] + w["fc1.bias"]) @ w["score"]
        return forward
    enc = ENCODERS[family]
    return lambda w, q, o: (enc(w, pairs(q, o)) @ w["score"]).reshape(q.shape[0], 4)


def run_cell(family: str, writer: str, d: int, vocab_size: int, parts: dict, n: int, seed: int, device) -> dict:
    fwd = forward_for(family)
    start = {k: v.to(device) for k, v in init(family, vocab_size, d, seed).items()}
    sweep = [train_pair.train_one(fwd, start, parts, n, seed, lr, False) for lr in train_pair.LRS]
    info, w = max(sweep, key=lambda r: r[0]["val_accuracy"])
    w = {k: v.to(device) for k, v in w.items()}
    q, o, y = parts["test"]
    with torch.no_grad():
        pred = torch.cat([fwd(w, q[i:i + 256], o[i:i + 256]).argmax(1) for i in range(0, len(y), 256)])
    hit = (pred == y).cpu().numpy()
    sub = arc.subset("test")
    return {"family": family, "matched_to": writer, "target_parameters": TARGETS[writer], "width": d,
            "parameters": n_params(family, vocab_size, d), "vocab": vocab_size, "n_train": n, "seed": seed, **info,
            "test_accuracy": float(hit.mean()), "test_by_subset": {s: float(hit[sub == s].mean()) for s in arc.SUBSETS},
            "lr_sweep": [r[0] for r in sweep]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--writers", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--smoke", action="store_true", help="one n=10 cell per family and size, nothing saved")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    vocab = train_pair.gd_vocab()
    vocab_size = len(vocab) + 2
    parts = {s: train_pair.tensors(vocab, s, device) for s in ("train", "validation", "test")}
    cells = [(FULL, 0)] + [(n, s) for n in N_TRAIN for s in SEEDS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for family in a.families:
        for writer in a.writers:
            d = width_for(family, vocab_size, TARGETS[writer])
            for n, seed in ([(10, 0)] if a.smoke else cells):
                out = OUT_DIR / f"{family}_{writer}_n{n}_s{seed}.json"
                if out.exists() and not a.smoke:
                    continue
                t0 = time.time()
                res = run_cell(family, writer, d, vocab_size, parts, n, seed, device)
                if not a.smoke:
                    out.write_text(json.dumps(res) + "\n")
                print(json.dumps({k: res[k] for k in ("family", "matched_to", "width", "parameters", "n_train",
                                                      "seed", "lr", "test_accuracy")} | {"s": round(time.time() - t0)}),
                      flush=True)


if __name__ == "__main__":
    main()

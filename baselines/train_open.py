"""Adam baselines for the given span and generation transformers (wsp.given_archs SPAN and GEN): SQuAD 1.1 and SciQ
generation, the tasks of transformer-sq1 and transformer-gen, scored as the written models are.

Each (task, N, seed) trains on N training items drawn at random. For each learning rate the best-validation
checkpoint is kept; the rate with the best validation F1 is reported with its test F1. Finished cells are skipped on
restart.
  sq1  SQuAD 1.1 training items that are neither an example nor a validation item and whose first answer is found
       word for word inside the first SC_LEN words of the paragraph. The vocabulary is the VOCAB most frequent words
       of those items; other words get hashed ids as in free_model.span_encode. Loss: cross-entropy of the start and
       of the end position (the "no answer" slot is masked).
  gen  SciQ training questions. The vocabulary is the VOCAB most frequent question words and every answer word of the
       training split. Loss: cross-entropy of the next answer word given the question and the words before it, the
       end of the answer included.

    python -m baselines.train_open --task sq1 --n-train 100 1000 10000 0     (0 = every training item)
"""
import argparse
import json
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from wsp import free_model, given_archs, paths, sciq, squad
from wsp.given_archs import gen_transformer, span_transformer

OUT_DIR = paths.DATA_ROOT / "baselines_open"
VOCAB = {"sq1": 20000, "gen": 8000}
BATCH = 32
LRS = (3e-3, 1e-3, 3e-4)
MAX_STEPS = 20000


def squad_parts(device) -> tuple[list[str], dict]:
    held = {r["id"] for s in ("examples", "validation") for r in squad.items(s, "1.1")}
    pool = []
    for r in squad._items("train1.1"):
        if r["id"] in held:
            continue
        context = [t for t, _, _ in free_model.span_tokens(r["context"])][:free_model.SC_LEN]
        answer = [t for t, _, _ in free_model.span_tokens(r["answers"][0])]
        at = next((i for i in range(len(context) - len(answer) + 1) if context[i:i + len(answer)] == answer), None)
        if answer and at is not None and len(answer) <= free_model.MAX_SPAN:
            pool.append((r, at + 1, at + len(answer)))
    counts = Counter(t for r, _, _ in pool for text in (r["question"], r["context"]) for t, _, _ in free_model.span_tokens(text))
    vocab = [w for w, _ in counts.most_common(VOCAB["sq1"])]
    q, c = free_model.span_encode(vocab, [r["question"] for r, _, _ in pool], [r["context"] for r, _, _ in pool])
    parts = {"train": (torch.from_numpy(q).to(device), torch.from_numpy(c).to(device),
                       torch.tensor([[s, e] for _, s, e in pool], device=device))}
    for split in ("validation", "test"):
        questions, contexts, golds = squad.load(split, "1.1")
        q, c = free_model.span_encode(vocab, questions, contexts)
        parts[split] = (torch.from_numpy(q).to(device), torch.from_numpy(c).to(device), contexts, golds)
    return vocab, parts


def squad_loss(w: dict, batch: tuple) -> torch.Tensor:
    q, c, y = batch
    logits = span_transformer.forward(w, q, c)
    logits = torch.cat([torch.full_like(logits[:, :, :1], -1e9), logits[:, :, 1:]], dim=2)
    return F.cross_entropy(logits[:, 0], y[:, 0]) + F.cross_entropy(logits[:, 1], y[:, 1])


@torch.no_grad()
def squad_f1(w: dict, part: tuple) -> dict:
    q, c, contexts, golds = part
    logits = torch.cat([span_transformer.forward(w, q[i:i + 64], c[i:i + 64]) for i in range(0, len(q), 64)]).cpu().numpy()
    return free_model.span_scores(free_model.span_decode_forced(logits, contexts), golds)


def sciq_parts(device) -> tuple[list[str], dict]:
    questions, answers = sciq.load("train")
    counts = Counter(w for text in questions for w in free_model.words(text))
    said = dict.fromkeys(w for a in answers for w in free_model.words(a[0]))
    vocab = list(dict.fromkeys([w for w, _ in counts.most_common(VOCAB["gen"])] + list(said)))
    index = {w: i + 2 for i, w in enumerate(vocab)}
    q = torch.from_numpy(free_model.gen_encode(vocab, questions)).to(device)
    a = torch.zeros(len(questions), free_model.GA_LEN, dtype=torch.int64)
    for k, ans in enumerate(answers):
        ids = [index[w] for w in free_model.words(ans[0])][:free_model.GA_LEN]
        a[k, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
    parts = {"train": (q, a.to(device))}
    for split in ("validation", "test"):
        qs, golds = sciq.load(split)
        parts[split] = (torch.from_numpy(free_model.gen_encode(vocab, qs)).to(device), golds)
    return vocab, parts


def sciq_loss(w: dict, batch: tuple) -> torch.Tensor:
    """Every prefix of every answer in the batch predicts its next word (0 after the last word)."""
    q, a = batch
    steps = torch.arange(free_model.GA_LEN, device=q.device)
    length = (a != 0).sum(1)
    keep = steps[None, :] <= length[:, None].clamp(max=free_model.GA_LEN - 1)
    prefix = a[:, None, :] * (steps[None, None, :] < steps[None, :, None])
    rows = keep.nonzero()
    return F.cross_entropy(gen_transformer.forward(w, q[rows[:, 0]], prefix[rows[:, 0], rows[:, 1]]), a[rows[:, 0], rows[:, 1]])


@torch.no_grad()
def sciq_f1(w: dict, part: tuple, vocab: list[str]) -> dict:
    q, golds = part
    a = torch.zeros(len(q), free_model.GA_LEN, dtype=torch.int64, device=q.device)
    for t in range(free_model.GA_LEN):
        nxt = torch.cat([gen_transformer.forward(w, q[i:i + 256], a[i:i + 256]).argmax(1) for i in range(0, len(q), 256)])
        a[:, t] = nxt * (a[:, :t] != 0).all(1)
    return free_model.span_scores(free_model.gen_text(vocab, a.cpu().numpy()), golds)


def train_one(loss_of, f1_of, start: dict, train: tuple, validation, n: int, seed: int, lr: float) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    size = len(train[0])
    idx = torch.randperm(size, generator=torch.Generator().manual_seed(seed), device="cpu")[:n or size].to(train[0].device)
    w = {k: v.clone().requires_grad_() for k, v in start.items()}
    opt = torch.optim.Adam(w.values(), lr=lr)
    steps = min(MAX_STEPS, max(1500, 30 * len(idx) // BATCH))
    every = max(50, steps // 40)
    best, t0 = (-1.0, None, 0), time.time()
    for step in range(1, steps + 1):
        b = idx[torch.randint(0, len(idx), (min(BATCH, len(idx)),), device=idx.device)]
        loss = loss_of(w, tuple(t[b] for t in train))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            f1 = f1_of(w, validation)["f1"]
            if f1 > best[0]:
                best = (f1, {k: v.detach().clone() for k, v in w.items()}, step)
    return {"lr": lr, "steps": steps, "best_step": best[2], "val_f1": best[0], "seconds": round(time.time() - t0, 1)}, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("sq1", "gen"))
    ap.add_argument("--n-train", nargs="+", type=int, required=True, help="0 = every training item")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    with torch.device(device):
        if a.task == "sq1":
            vocab, parts = squad_parts(device)
            name, rows, loss_of, f1_of = given_archs.SPAN, len(vocab) + 2, squad_loss, squad_f1
        else:
            vocab, parts = sciq_parts(device)
            name, rows, loss_of = given_archs.GEN, len(vocab) + 2, sciq_loss
            f1_of = lambda w, part: sciq_f1(w, part, vocab)
        for n in a.n_train:
            for seed in a.seeds:
                out = OUT_DIR / f"gd_{a.task}_N{n}_s{seed}.json"
                if out.exists():
                    continue
                with torch.device("cpu"):  # init draws from a CPU generator
                    start = given_archs.init(name, rows, seed)
                start = {k: v.to(device) for k, v in start.items()}
                sweep = [train_one(loss_of, f1_of, start, parts["train"], parts["validation"], n, seed, lr) for lr in LRS]
                info, w = max(sweep, key=lambda r: r[0]["val_f1"])
                test = f1_of(w, parts["test"])
                res = {"task": a.task, "arch": name, "n_train": n or len(parts["train"][0]), "seed": seed,
                       "train_pool": len(parts["train"][0]), "parameters": sum(math.prod(v.shape) for v in start.values()),
                       "vocab": len(vocab), **info, "test_f1": test["f1"], "test_exact": test["exact"],
                       "lr_sweep": [r[0] for r in sweep]}
                out.write_text(json.dumps(res) + "\n")
                print(json.dumps({k: res[k] for k in ("task", "n_train", "seed", "lr", "val_f1", "test_f1")}), flush=True)


if __name__ == "__main__":
    main()

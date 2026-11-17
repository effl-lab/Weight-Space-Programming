"""Builds the order-and-extremes weights from seq.txt (see the header of that file)."""
from pathlib import Path

import torch

from textproc import porter

HERE = Path(__file__).resolve().parent


def parse():
    seqs = []
    for line in (HERE / "seq.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        head, _, body = line.partition(":")
        name, lo, hi = [x.strip() for x in head.split("|")]
        items = [it.split() for it in body.split(";")]
        seqs.append((name.rstrip("*"), name.endswith("*"), lo.split(), hi.split(), items))
    return seqs


def build_seq(vocab):
    seqs = parse()
    S = len(seqs)
    V = len(vocab) + 2
    stems = [porter(wd[:-2] if wd.endswith("'s") else wd) for wd in vocab]
    rank = torch.zeros(V, S, dtype=torch.int8)
    hi = torch.zeros(V, S, dtype=torch.int8)
    lo = torch.zeros(V, S, dtype=torch.int8)
    length = torch.zeros(S)
    order = torch.zeros(S)
    for s, (name, rank_only, los, his, items) in enumerate(seqs):
        length[s] = len(items)
        order[s] = 0.0 if rank_only else 1.0
        count = {}
        for it in items:
            for st in {porter(x) for x in it}:
                count[st] = count.get(st, 0) + 1
        key, exact = {}, {}
        for r, it in enumerate(items):
            for x in it:
                exact[x] = r + 1
                if count[porter(x)] == 1:
                    key[porter(x)] = r + 1
        xcount = {}
        for x in exact:
            xcount[x] = sum(x in it for it in items)
        exact = {x: r for x, r in exact.items() if xcount[x] == 1}
        hs, ls = {porter(x) for x in his}, {porter(x) for x in los}
        for i, st in enumerate(stems):
            base = vocab[i][:-2] if vocab[i].endswith("'s") else vocab[i]
            if base in exact:
                rank[i + 2, s] = exact[base]
            elif st in key:
                rank[i + 2, s] = key[st]
            if st in hs or vocab[i] in his:
                hi[i + 2, s] = 1
            if st in ls or vocab[i] in los:
                lo[i + 2, s] = 1
    return {"seq_rank": rank, "seq_hi": hi, "seq_lo": lo, "seq_len": length, "seq_order": order}

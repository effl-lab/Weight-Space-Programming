"""Writes every weight of the scorer from the compiled knowledge base (see NOTES.md for the design).

All weights are float64: the matching heads need very sharp attention.
"""
import math
from pathlib import Path

import numpy as np
import torch

from kb import compile_kb

D = 8
SEQ = 176
Q_LEN = 128

# design constants
MARGIN = 30.0          # attention logit margin (nats) for: exact match beats self, near slots lose
STOP_NORM = 1e6        # stopwords: huge norm on circle 0 at the stop angle -> their readout is ~0
D_HEAD = (0.2, 0.2, 0.2)   # flag increment for a match on each circle (small: readout ~ linear in matches)
ARC = (0.1 * math.pi, 0.9 * math.pi)   # content slots live on this arc of each circle; stop angle opposite
STOP_ANGLE = 1.5 * math.pi
SCORE = 10.0
QB = 0.005          # question content words read -QB: with equal matches, longer options rank higher
PHI = (0.0, 1.0, 1.5, 0.0)   # readout for an option word matched on k circles (k=3 is almost always the same word)
EPS = 1e-5             # torch layer_norm eps
NEG_WORDS = ("not", "except", "least", "cannot", "can't", "isn't", "aren't", "doesn't", "never", "avoid")
NEG_SHARE = 0.5        # part of a negation word's (huge) norm on circle 1 at STOP_ANGLE: what block 1 detects
NEG_OFF = 10.0         # block 1 adds NEG_OFF to every flag when the sequence holds a negation word
NEG_FLIP = 1.0         # ... and then an option word's match score s becomes -NEG_FLIP * s


def ln_flag(f):
    """Layer-normed flag coordinate of a token whose circles have squared norm 3 and whose flag is f."""
    return f / math.sqrt((3 + f * f) / 8 + EPS)


def step_mlp(classes, w, p, flag):
    """Block MLP adding a constant `add` to the flag of every token whose flag is `level`.

    classes: list of (level, add). Between consecutive levels a saturated ramp pair GELU(z) - GELU(z - 1) with
    a steep z makes an exact step (z is at least 15 away from the ramp at every level).
    """
    classes = sorted(classes)
    w[p + "fc2.bias"] += classes[0][1] * flag
    unit = 0
    for (l0, a0), (l1, a1) in zip(classes, classes[1:]):
        if abs(a1 - a0) < 1e-15:
            continue
        x0, x1 = ln_flag(l0), ln_flag(l1)
        assert x1 > x0
        t = (x0 + x1) / 2
        kappa = 32.0 / (x1 - x0)
        for off, sign in ((0.0, 1.0), (-1.0, -1.0)):
            w[p + "fc1"][:, unit] = kappa * flag
            w[p + "fc1.bias"][unit] = -kappa * t + off
            w[p + "fc2"][unit, :] = sign * (a1 - a0) * flag
            unit += 1
    assert unit <= 16, unit


def basis():
    """Orthonormal 8x8 basis whose column 0 is the all-ones direction (removed by layer norm)."""
    m = np.eye(D)
    m[:, 0] = 1.0
    qm, _ = np.linalg.qr(m)
    if qm[0, 0] < 0:
        qm[:, 0] = -qm[:, 0]
    return qm[:, 1:]          # 7 directions with zero mean: circle h uses 2h, 2h+1; flag uses 6


def build():
    vocab_file = [l.strip() for l in Path("vocab.txt").read_text().splitlines()]
    vocab, info = compile_kb()
    assert vocab == vocab_file, "vocab.txt is stale: run python kb.py"
    B = basis()
    flag = B[:, 6]
    V = len(vocab) + 2
    n_slots = max(info["count"])
    delta = (ARC[1] - ARC[0]) / (n_slots + 1)
    rho = delta ** 2 / 4                     # key radius ratio question/option = 1 + rho

    # question content words end at flag -QB, unmatched option words at 0, a match on circle h adds D_HEAD[h]:
    # requires G + Go = sum(D_HEAD) + QB, with Go slightly above G (question keys a bit longer than option keys)
    S = sum(D_HEAD)
    gap_go = 0.0
    for _ in range(100):
        G = (S + QB - gap_go) / 2
        nq2 = 3 + G * G
        no2 = 8 * ((1 + rho) ** 2 * (nq2 / 8 + EPS) - EPS)
        gap_go = (no2 - nq2) / (math.sqrt(no2 - 3) + G)
    Go = G + gap_go
    nq2, no2 = 3 + G * G, 3 + Go * Go
    lq = 1 / math.sqrt(nq2 / 8 + EPS)
    lo = 1 / math.sqrt(no2 / 8 + EPS)
    A = 4 * MARGIN / delta ** 2 / lo ** 2    # q.k scale so that A*lo^2*(R-1) = MARGIN
    alpha = math.sqrt(A * math.sqrt(2))

    emb = np.zeros((V, D))
    stop_vec = STOP_NORM * (math.cos(STOP_ANGLE) * B[:, 0] + math.sin(STOP_ANGLE) * B[:, 1])
    emb[0] = stop_vec
    emb[1] = stop_vec
    index = {w: i + 2 for i, w in enumerate(vocab)}
    for w in info["stop"]:
        emb[index[w]] = stop_vec
    neg_dir = math.cos(STOP_ANGLE) * B[:, 2] + math.sin(STOP_ANGLE) * B[:, 3]     # circle 1, opposite content
    neg_vec = STOP_NORM * (math.sqrt(1 - NEG_SHARE ** 2) * stop_vec / STOP_NORM + NEG_SHARE * neg_dir)
    for w in NEG_WORDS:
        assert w in index and w in info["stop"], w
        emb[index[w]] = neg_vec
    for w, g in info["group_of"].items():
        e = np.zeros(D)
        for h in range(3):
            th = ARC[0] + (info["slots"][g][h] + 1) * delta
            e += math.cos(th) * B[:, 2 * h] + math.sin(th) * B[:, 2 * h + 1]
        emb[index[w]] = e

    pos = np.zeros((SEQ, D))
    pos[:Q_LEN] = -G * flag
    pos[Q_LEN:] = Go * flag

    w = {}
    z8 = np.zeros((D, D))
    for blk in range(2):
        p = f"block{blk}."
        w[p + "ln1.weight"] = np.ones(D)
        w[p + "ln1.bias"] = np.zeros(D)
        w[p + "ln2.weight"] = np.ones(D)
        w[p + "ln2.bias"] = np.zeros(D)
        for n in ("q", "k", "v", "o"):
            w[p + n] = z8.copy()
            w[p + n + ".bias"] = np.zeros(D)
        w[p + "fc1"] = np.zeros((D, 16))
        w[p + "fc1.bias"] = np.zeros(16)
        w[p + "fc2"] = np.zeros((16, D))
        w[p + "fc2.bias"] = np.zeros(D)

    # block 0: three matching heads
    p = "block0."
    for h in range(3):
        w[p + "q"][:, 2 * h] = alpha * B[:, 2 * h]
        w[p + "q"][:, 2 * h + 1] = alpha * B[:, 2 * h + 1]
        w[p + "k"][:, 2 * h] = alpha * B[:, 2 * h]
        w[p + "k"][:, 2 * h + 1] = alpha * B[:, 2 * h + 1]
        vq = (G - QB) * D_HEAD[h] / S
        vo = -Go * D_HEAD[h] / S
        c1 = -(vq - vo) / (G * lq + Go * lo)
        c0 = vq + c1 * G * lq
        w[p + "v"][:, 2 * h] = c1 * flag
        w[p + "v.bias"][2 * h] = c0
        w[p + "o"][2 * h, :] = flag

    # block 0 MLP: reshape the flag of an option word matched on k circles from k*d to PHI[k]*d (exact steps)
    dd = D_HEAD[0]
    f = [k * dd for k in range(4)]
    x = [fk / math.sqrt((3 + fk * fk) / 8 + EPS) for fk in f]
    y = [PHI[k] * dd - f[k] for k in range(4)]
    for k in range(1, 4):
        t = (x[k - 1] + x[k]) / 2
        kappa = 40 / (x[k] - x[k - 1])
        for j, off in ((2 * k - 2, 0.0), (2 * k - 1, -1.0)):
            w[p + "fc1"][:, j] = kappa * flag
            w[p + "fc1.bias"][j] = -kappa * t + off
            w[p + "fc2"][j, :] = (1 if off == 0 else -1) * (y[k] - y[k - 1]) * flag

    # block 1 attention: a constant query looks for a negation word (only they point along neg_dir);
    # every token then gets NEG_OFF added to its flag (stopwords: 0, content words: never attended)
    p = "block1."
    proj = math.sqrt(8) * NEG_SHARE              # neg_dir coordinate of a layer-normed negation word
    w[p + "q.bias"][0] = 40.0 / proj
    w[p + "k"][:, 0] = math.sqrt(2) * neg_dir
    w[p + "v"][:, 0] = neg_dir / proj
    w[p + "o"][0, :] = NEG_OFF * flag
    # block 1 MLP: with a negation, option word scores s -> -NEG_FLIP*s; question words keep -QB
    levels = sorted(set([0.0] + [PHI[k] * dd for k in range(4)]))
    classes = [(-QB, 0.0)] + [(l, 0.0) for l in levels]
    classes += [(NEG_OFF - QB, -NEG_OFF)] + [(NEG_OFF + l, -NEG_OFF - (1 + NEG_FLIP) * l) for l in levels]
    step_mlp(classes, w, p, flag)

    w["ln_f.weight"] = np.ones(D)
    w["ln_f.bias"] = np.zeros(D)
    w["score"] = SCORE * flag
    w["embed"] = emb
    w["pos"] = pos
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in w.items()}

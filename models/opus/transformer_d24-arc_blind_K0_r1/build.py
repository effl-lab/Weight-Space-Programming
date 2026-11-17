"""Builds every weight of the ARC scorer by hand.  See NOTES.md for the design.

Residual layout (24 dims):
  0 A  anchor: a large constant from pos, so layer norm is almost linear.  Option tokens get a ~300x larger
       anchor than question tokens, which makes their word content ~invisible to keys (option keys = sink).
  1 Z  dump: always 0.  Every matrix that reads a layer norm has row Z = -(sum of the other rows), so all its
       columns sum to zero and the layer norm's mean subtraction has no effect.
  2 O  option flag from pos: +FO at the first option position (the sink), -FO at the other option positions.
  3 V  per-token value: match strength found by layer 0 (and a tiny marker on negation words).
  4..23 word association codes: 4 heads of 5 dims.  Once layer 0 has matched, three of these dims are reused:
       NEGD carries 100 x the negation flag, FLAGD 1000 x the option flag, PD the pooled score x PSCALE.
"""
import math
import os

import numpy as np
import torch

import knowledge as kn

D, HEADS, MLP = 24, 4, 48
DH = D // HEADS
Q_LEN, O_LEN = 128, 48
A, Z, O, V = 0, 1, 2, 3
HEAD_DIMS = kn.HEAD_DIMS
NEGD, PD, FLAGD = 23, 22, 21


def env(name, default):
    return float(os.environ.get(name, default))


# ---- scales
BQ, BO, FO = 30.0, 10000.0, 1000.0   # anchors of question / option tokens, option flag
BETA = 100.0                         # match sharpness (score of a perfect match before thresholds)
GAMMA_Q = 95.0                       # threshold on question keys
GAMMA_S = 5.0                        # score of the sink; one perfect match then gives frac = 1/2
LAMBDA = 1.0                         # weight of one head's match fraction in V
NEG_MARK = 0.05                      # V marker of negation words in the embedding
KAPPA_NEG = GAMMA_Q + 25.0           # key score of a negation word in head 3 (beats any match and the sink)
M_NEG = 100.0                        # negation flag is written at this scale into NEGD
CAP = env("ARC_CAP", 0.6)            # per-word match cap
FLAG = 1000.0                        # option flag written into FLAGD by the layer-0 MLP (layer-1 keys, length bonus)
G_POOL = env("ARC_G", 40.0)          # layer-1 key of option tokens (question tokens: key 0, value 0)
TAU = env("ARC_TAU", 0.0)            # soft-max pooling: option token keys G_POOL + TAU * v
PSCALE = 1000.0                      # pooled score is written at this scale into PD
EPS_LEN = env("ARC_EPS", 0.6)        # length bonus: EPS_LEN * n_opt / (n_q + n_opt)

HERE = os.path.dirname(os.path.abspath(__file__))


def std(vec):
    v = np.asarray(vec, np.float64)
    return float(np.sqrt(((v - v.mean()) ** 2).mean() + 1e-5))


def solve_anchor():
    """Anchor of non-sink option positions so that their std equals the sink's (O = -FO instead of +FO)."""
    target = std([BO, 0, FO] + [0] * (D - 3))
    lo, hi = BO * 0.5, BO * 1.5
    for _ in range(200):
        mid = (lo + hi) / 2
        if std([mid, 0, -FO] + [0] * (D - 3)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def dump_fix(W):
    """Make every column of a matrix that reads a layer-norm output sum to zero (row Z absorbs it)."""
    W[Z, :] = 0
    W[Z, :] = -W.sum(0)
    return W


def build():
    vocab, vec, neg = kn.vocab_list()
    with open(os.path.join(HERE, "vocab.txt")) as fh:
        file_vocab = [l.strip() for l in fh.read().splitlines()]
    assert file_vocab == vocab, "vocab.txt is stale: run python build.py"
    nv = len(vocab) + 2

    embed = np.zeros((nv, D))
    for i, w in enumerate(vocab):
        if w in vec:
            embed[i + 2] += vec[w]
        if w in neg:
            embed[i + 2, V] = NEG_MARK
    BO2 = solve_anchor()
    pos = np.zeros((Q_LEN + O_LEN, D))
    pos[:Q_LEN, A] = BQ
    pos[Q_LEN, A], pos[Q_LEN, O] = BO, FO
    pos[Q_LEN + 1:, A], pos[Q_LEN + 1:, O] = BO2, -FO
    sq = std(pos[0])          # layer-norm scale of question tokens
    so = std(pos[Q_LEN])      # ... of option tokens (sink and non-sink alike)
    r6 = math.sqrt(DH)
    # For a token's layer-norm output x (times the dump fix, x @ W = residual @ W / sigma):
    #   o = x[O]*so/FO is +1 on the sink, -1 on other option tokens, 0 on question tokens.
    w = {}

    # ---------------- block 0: every option word looks for its partners among the question words
    Wq, Wk, Wv, Wo = (np.zeros((D, D)) for _ in range(4))
    bq, bk, bv = (np.zeros(D) for _ in range(3))
    for h, dims in enumerate(HEAD_DIMS):
        b = DH * h
        R = kn.mirror(h)
        for k, d in enumerate(dims):
            Wq[d, b + k] = BETA * so * r6        # option query = BETA * code
            Wk[d, b + k] = sq * R[k]             # question key = mirrored code (option keys ~ code / 300)
        g = b + 5                                # gate: question keys -GAMMA_Q, sink +GAMMA_S, others << 0
        bq[g] = r6
        bk[g] = -GAMMA_Q
        Wk[O, g] = (GAMMA_S + GAMMA_Q) * so / FO
        Wo[b, V] = LAMBDA
    # values: 1 on question keys, 0 on the sink; head 3 needs its value dims 0,1 for match and negation, and its
    # q/k dim 4 is taken by the code, so negation is read through the gate dim of head 3 instead (see below)
    for h in range(HEADS):
        b = DH * h
        bv[b] = 1.0
        Wv[O, b] = -so / FO
    # negation: head 0's value dim 1 carries the flag; its keys get +KAPPA_NEG through the gate dim when the
    # word is a negation word (V marker).  A negation word is not a match (value 0) but raises the flag.
    b = 0
    Wk[V, b + 5] = KAPPA_NEG * sq / NEG_MARK
    Wv[V, b] = -sq / NEG_MARK
    Wv[V, b + 1] = sq / NEG_MARK
    Wo[b + 1, NEGD] = M_NEG
    w["block0.q"], w["block0.k"], w["block0.v"], w["block0.o"] = dump_fix(Wq), dump_fix(Wk), dump_fix(Wv), Wo
    w["block0.q.bias"], w["block0.k.bias"], w["block0.v.bias"], w["block0.o.bias"] = bq, bk, bv, np.zeros(D)

    # ---------------- block 0 MLP: cap V, flip it where the question is negated; FLAG on option tokens
    fc1, fb1, fc2, fb2 = np.zeros((D, MLP)), np.zeros(MLP), np.zeros((MLP, D)), np.zeros(D)
    a, B, KN, KG, KC, KNC = 1.0, 8.0, 40.0, 2e5, 20.0, 300.0
    u = 0
    for sgn, anchor in ((1, BO), (-1, BO2)):    # sink (O=+FO) and other options (O=-FO)
        # gate = (sgn*O/FO - A/anchor)*so/sigma is exactly 0 on the intended positions, <= -0.25 elsewhere
        for with_v in (1, 0):                    # flip: V -= 2 V neg
            fc1[V, u] = a * so * with_v
            fc1[NEGD, u] = KN * so / M_NEG
            fc1[O, u] = KG * sgn * so / FO
            fc1[A, u] = -KG * so / anchor
            fb1[u] = B - KN
            fc2[u, V] = -2.0 / a * (1 if with_v else -1)
            u += 1
        for flip in (0, 1):                      # cap: V -> min(V, CAP), the excess flipped too when negated
            fc1[V, u] = KC * so
            fc1[NEGD, u] = KNC * so / M_NEG * flip
            fc1[O, u] = KG * sgn * so / FO
            fc1[A, u] = -KG * so / anchor
            fb1[u] = -KC * CAP - KNC * flip
            fc2[u, V] = (2.0 if flip else -1.0) / KC
            u += 1
        if TAU:                                  # pooling weight: FLAGD += v * TAU * FLAG / G_POOL (options only)
            for with_v in (1, 0):
                fc1[V, u] = a * so * with_v
                fc1[O, u] = KG * sgn * so / FO
                fc1[A, u] = -KG * so / anchor
                fb1[u] = B
                fc2[u, FLAGD] = (1 if with_v else -1) * TAU * FLAG / G_POOL / a
                u += 1
    K = 10.0
    for sgn in (1, -1):                          # gelu(K o) + gelu(-K o) = K |o|: FLAG on option tokens
        fc1[O, u] = sgn * K * so / FO
        fc2[u, FLAGD] = FLAG / K
        u += 1
    w["block0.fc1"], w["block0.fc1.bias"], w["block0.fc2"], w["block0.fc2.bias"] = dump_fix(fc1), fb1, fc2, fb2

    # ---------------- block 1: every token takes the mean of V over the option tokens, into PD
    Wq, Wk, Wv, Wo = (np.zeros((D, D)) for _ in range(4))
    bq, bk, bv = (np.zeros(D) for _ in range(3))
    bq[0] = r6
    Wk[FLAGD, 0] = G_POOL * so / FLAG            # option keys G_POOL, question keys |.| < G_POOL / 3 (code content)
    Wv[V, 0] = so
    Wo[0, PD] = PSCALE
    w["block1.q"], w["block1.k"], w["block1.v"], w["block1.o"] = dump_fix(Wq), dump_fix(Wk), dump_fix(Wv), Wo
    w["block1.q.bias"], w["block1.k.bias"], w["block1.v.bias"], w["block1.o.bias"] = bq, bk, bv, np.zeros(D)

    # ---------------- block 1 MLP: give question tokens the option tokens' anchor and spread, so all tokens read alike
    fc1, fb1, fc2, fb2 = np.zeros((D, MLP)), np.zeros(MLP), np.zeros((MLP, D)), np.zeros(D)
    K, KG = 10.0, 1000.0
    # h1 = gelu(K - KG o), h2 = gelu(K + KG o), h3 = gelu(KG o), h4 = gelu(-KG o); h1+h2-h3-h4 = K (opt), 2K (q)
    add = {A: BO - BQ, O: FO, V: FLAG}
    for u, (cK, cO, sign) in enumerate(((K, -KG, 1), (K, KG, 1), (0, KG, -1), (0, -KG, -1))):
        fb1[u] = cK
        fc1[O, u] = cO * so / FO
        for d, val in add.items():
            fc2[u, d] = sign * val / K
    for d, val in add.items():
        fb2[d] = -val
    w["block1.fc1"], w["block1.fc1.bias"], w["block1.fc2"], w["block1.fc2.bias"] = dump_fix(fc1), fb1, fc2, fb2

    for i in (0, 1):
        for n in ("ln1", "ln2"):
            w[f"block{i}.{n}.weight"] = np.ones(D)
            w[f"block{i}.{n}.bias"] = np.zeros(D)
    w["ln_f.weight"] = np.ones(D)
    w["ln_f.bias"] = np.zeros(D)
    s = np.zeros(D)
    s[PD] = so / PSCALE                          # pooled score
    s[FLAGD] = EPS_LEN * so / FLAG               # FLAG on option tokens: length bonus
    s[Z] = -s.sum()
    w["score"] = s
    w["embed"] = embed
    w["pos"] = pos
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()}


if __name__ == "__main__":
    vocab, vec, neg = kn.vocab_list()
    with open(os.path.join(HERE, "vocab.txt"), "w") as fh:
        fh.write("\n".join(vocab) + "\n")
    groups = kn.read_groups()
    print(f"{len(vocab)} words, {len(groups)} groups")

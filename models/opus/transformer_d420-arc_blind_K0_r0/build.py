"""Builds every weight of the ARC scorer from the hand-written files in this folder (see NOTES.md).

Residual layout (D = 420):
  0..314   CODE: a word's knowledge code; layer 0 adds the question's mean code (Qvec) on top of it
  then single control dims (names below).
Layer 0: heads 0-2 copy the weighted mean of the question words' codes into every token's code dims;
         head 3 measures whether the token's word also occurs in the question (exact-match);
         MLP squares the code dims: |C(o) + Qvec|^2 - |C(o)|^2 = 2 C(o).Qvec + |Qvec|^2.
Layer 1: head 0 takes the weighted mean of that score over the option's words into F (same at every token);
         head 1 raises NEGF when the question holds a negation word; the MLP flips F when NEGF is on.
ln_f and score read F.
"""
import math
import zlib

import numpy as np
import torch

import kb

D, HEADS, MLP, HD = 420, 4, 840, 105
NCODE = 315
names = ["ANCP", "ANCN", "BAL", "ISO", "LWQ", "LWO", "Z0", "Z1", "Z2", "ZSQ", "EXM", "DD", "F", "NEGW", "NEGF",
         "NORM2", "PRI"]
I = {n: NCODE + i for i, n in enumerate(names)}
A = 3000.0                      # anchor size: keeps every token's layer-norm scale the same
SIG = A * math.sqrt(2.0 / D)    # the layer-norm scale of every token (all other dims are tiny next to the anchors)
BIG = 50.0
BETA = 1e4                      # sharpness of the exact-match test
EXM_PENALTY = 10.0              # log-weight removed from an option word that repeats a question word
NEG_KEY = 25.0
RAMP = 0.05                     # log-weight gained per question position

W_UNK_Q, W_UNK_O = 0.3, 0.05
W_STOP = 1e-4
NORM_POWER = 1.0


def balance(m):
    """Make every written row sum to zero (through BAL) so every residual vector keeps mean 0."""
    m[..., I["BAL"]] = 0
    m[..., I["BAL"]] = -m.sum(-1)
    return m


def line_weight(n):
    return 1.0                  # every fact line counts the same, however many words it lists


def knowledge():
    lines = kb.knowledge_lines()
    rng = np.random.default_rng(12345)
    raw, count = {}, {}
    for groups in lines:
        # one code per group; groups of one line are made mutually opposed (a centred simplex)
        r = rng.standard_normal((len(groups), NCODE))
        if len(groups) > 1:
            r = r - r.mean(0, keepdims=True)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        vec = {}
        for gi, g in enumerate(groups):
            for w in g:
                s = kb.stem(w)
                if s:
                    vec[s] = vec.get(s, 0.0) + r[gi]
        vec = {s: v for s, v in vec.items() if np.linalg.norm(v) > 1e-6}
        if not vec:
            continue
        lam = line_weight(len(vec))
        for s, v in vec.items():
            raw[s] = raw.get(s, 0.0) + lam * v / np.linalg.norm(v)
            count[s] = count.get(s, 0) + 1
    codes = {}
    for s, v in raw.items():
        n = np.linalg.norm(v)
        if n > 1e-9:
            codes[s] = v / n ** NORM_POWER
    return codes, count


def word_weight(w, s, count):
    if w in kb.STOP or w in kb.NEGATION:
        return W_STOP, W_STOP
    n = count.get(s, 0)
    if n == 0:
        return W_UNK_Q, W_UNK_O
    base = 1.0 / math.sqrt(1.0 + n / 10.0)
    if w in kb.GENERIC or s in GENERIC_STEMS:
        base *= 0.25
    return base, base


GENERIC_STEMS = {kb.stem(w) for w in kb.GENERIC}


def zcode(s):
    r = np.random.default_rng(zlib.crc32(("z:" + s).encode()))
    return r.uniform(-1, 1, 3)


def build():
    vocab = [l.strip() for l in open("vocab.txt").read().splitlines()]
    V = len(vocab) + 2
    codes, count = knowledge()

    E = np.zeros((V, D))
    E[:, I["ANCP"]] = A
    E[:, I["ANCN"]] = -A
    # unknown word (id 1)
    E[1, I["LWQ"]] = math.log(W_UNK_Q)
    E[1, I["LWO"]] = math.log(W_UNK_O)
    z = zcode("<unk>")
    E[1, I["Z0"]:I["Z2"] + 1] = z
    E[1, I["ZSQ"]] = (z ** 2).sum()
    for k, w in enumerate(vocab):
        r = k + 2
        s = kb.stem(w)
        c = codes.get(s) if (w not in kb.STOP and w not in kb.NEGATION) else None
        if c is not None:
            E[r, :NCODE] = c
            E[r, I["NORM2"]] = (c ** 2).sum()
        wq, wo = word_weight(w, s, count)
        E[r, I["LWQ"]] = math.log(wq)
        E[r, I["LWO"]] = math.log(wo)
        z = zcode(s)
        E[r, I["Z0"]:I["Z2"] + 1] = z
        E[r, I["ZSQ"]] = (z ** 2).sum()
        if w in kb.NEGATION:
            E[r, I["NEGW"]] = 1.0
    balance(E)

    P = np.zeros((176, D))
    P[128:, I["ISO"]] = 1.0
    P[:128, I["LWQ"]] = RAMP * np.arange(128)   # later question words (the question proper) weigh more
    balance(P)

    w = {"embed": E, "pos": P}
    ones = np.ones(D)

    def zeros_block(p):
        for n in ("q", "k", "v", "o"):
            w[p + n] = np.zeros((D, D))
            w[p + n + ".bias"] = np.zeros(D)
        w[p + "ln1.weight"] = ones * SIG
        w[p + "ln1.bias"] = np.zeros(D)
        w[p + "ln2.weight"] = ones * SIG
        w[p + "ln2.bias"] = np.zeros(D)
        w[p + "fc1"] = np.zeros((D, MLP))
        w[p + "fc1.bias"] = np.zeros(MLP)
        w[p + "fc2"] = np.zeros((MLP, D))
        w[p + "fc2.bias"] = np.zeros(D)

    rt = math.sqrt(HD)
    # ---------------- block 0 ----------------
    p = "block0."
    zeros_block(p)
    for h in range(3):          # question-mean of codes
        c0 = h * HD
        w[p + "q.bias"][c0] = rt
        w[p + "k"][I["LWQ"], c0] = 1.0
        w[p + "k"][I["ISO"], c0] = -BIG
        for d in range(HD):
            w[p + "v"][h * HD + d, c0 + d] = 1.0
            w[p + "o"][c0 + d, h * HD + d] = 1.0
    c0 = 3 * HD                 # exact match: logit = -BETA |z_i - z_j|^2 (+ row constant)
    for d in range(3):
        w[p + "q"][I["Z0"] + d, c0 + d] = rt
        w[p + "k"][I["Z0"] + d, c0 + d] = 2 * BETA
    w[p + "q.bias"][c0 + 3] = rt
    w[p + "k"][I["ZSQ"], c0 + 3] = -BETA
    w[p + "v.bias"][c0] = 1.0                # value: 1 on question tokens, 0 on option tokens
    w[p + "v"][I["ISO"], c0] = -1.0
    w[p + "o"][c0, I["EXM"]] = 1.0
    balance(w[p + "o"])
    balance(w[p + "o.bias"])
    # MLP: DD = sum_d x_d^2 - NORM2, using gelu(t) + gelu(-t) ~ sqrt(2/pi) t^2 for small t
    a = 0.08
    cq = 1.0 / (math.sqrt(2 / math.pi) * a * a)
    for d in range(NCODE):
        w[p + "fc1"][d, 2 * d] = a
        w[p + "fc1"][d, 2 * d + 1] = -a
        w[p + "fc2"][2 * d, I["DD"]] = cq
        w[p + "fc2"][2 * d + 1, I["DD"]] = cq
    u = 2 * NCODE               # gelu(t) - gelu(-t) = t exactly
    w[p + "fc1"][I["NORM2"], u] = 1.0
    w[p + "fc1"][I["NORM2"], u + 1] = -1.0
    w[p + "fc2"][u, I["DD"]] = -1.0
    w[p + "fc2"][u + 1, I["DD"]] = 1.0
    w[p + "fc1"][I["PRI"], u + 2] = 1.0      # an option word's own prior, added to its score
    w[p + "fc1"][I["PRI"], u + 3] = -1.0
    w[p + "fc2"][u + 2, I["DD"]] = 1.0
    w[p + "fc2"][u + 3, I["DD"]] = -1.0
    balance(w[p + "fc2"])
    balance(w[p + "fc2.bias"])

    # ---------------- block 1 ----------------
    p = "block1."
    zeros_block(p)
    c0 = 0                      # weighted mean of DD over the option's words
    w[p + "q.bias"][c0] = rt
    w[p + "k"][I["LWO"], c0] = 1.0
    w[p + "k"][I["EXM"], c0] = -EXM_PENALTY
    w[p + "k"][I["ISO"], c0] = BIG
    w[p + "v"][I["DD"], c0] = 1.0
    w[p + "o"][c0, I["F"]] = 1.0
    c0 = HD                     # negation flag from the question's words
    w[p + "q.bias"][c0] = rt
    w[p + "k"][I["NEGW"], c0] = NEG_KEY
    w[p + "k"][I["ISO"], c0] = -2 * NEG_KEY
    w[p + "v"][I["NEGW"], c0] = 1.0
    w[p + "o"][c0, I["NEGF"]] = 1.0
    balance(w[p + "o"])
    balance(w[p + "o.bias"])
    # MLP: F -= 2 F NEGF, as gelu(K F + KB(NEGF-1)) - gelu(-K F + KB(NEGF-1)) = K F when NEGF = 1, ~0 when 0
    K, B = 10.0, 10.0
    w[p + "fc1"][I["F"], 0] = K
    w[p + "fc1"][I["NEGF"], 0] = K * B
    w[p + "fc1.bias"][0] = -K * B
    w[p + "fc1"][I["F"], 1] = -K
    w[p + "fc1"][I["NEGF"], 1] = K * B
    w[p + "fc1.bias"][1] = -K * B
    w[p + "fc2"][0, I["F"]] = -2.0 / K
    w[p + "fc2"][1, I["F"]] = 2.0 / K
    balance(w[p + "fc2"])
    balance(w[p + "fc2.bias"])

    w["ln_f.weight"] = np.zeros(D)
    w["ln_f.weight"][I["F"]] = SIG
    w["ln_f.bias"] = np.zeros(D)
    w["score"] = np.zeros(D)
    w["score"][I["F"]] = 1.0
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()}

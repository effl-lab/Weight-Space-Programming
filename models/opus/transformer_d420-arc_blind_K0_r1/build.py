"""Builds every weight of the ARC scorer from the hand-written fact files (see NOTES.md).

Residual stream layout (D = 420):
  0..311    association code (sum of the vectors of the facts a word appears in); heads 0-2 read 104 dims each
  312..351  lexical identity code (40 dims): detects option words that repeat a question word
  352..     control features and the scalars the layers write

Question positions carry no anchor in layer 1, so LayerNorm scales each question word by 1 / |embedding|; a per-word
filler sets that norm, which is how question words get their own (idf-like) weight. Option positions carry a large
anchor, so option words are only scaled by a constant. The layer-1 MLP adds the same anchor to question tokens, so
from layer 2 on every token is anchored and every LayerNorm is a fixed scaling.
"""
import glob
import math

import numpy as np
import torch

from lexicon import NEGATION, STOP, concept, stem, words

D, HD, MLP = 420, 105, 840
NB, BD = 3, 104
NC = NB * BD
E0, ED = 312, 40
(ANP, ANN, ZF, RO, CW, OW, NEG, PRI, M, X, FMEAN, FMAX, NN, PMEAN, FIN, FIL1, FIL2, BASE, CWQ) = range(352, 371)

SEED = 20261115
ANCHOR = 1000.0
SQ = math.sqrt(HD)
ALPHA_O, ALPHA_Q = 0.5, 0.75    # association = shared fact weight / (n_o ** ALPHA_O * n_q ** ALPHA_Q)
QSCALE = 200.0                  # question key multiplier for a word in one fact (s_q = QSCALE * n ** (ALPHA_O - ALPHA_Q))
EPS = 8.0                       # association logit = EPS * assoc (function words are sinks with logit 0)
ISA_W = 0.5
LAM_M = 10.0
BETA_E, THETA_E = 4.0, 0.55     # identity logit = BETA_E * s_q * (e.e' - THETA_E)
BIG = 1e4
G, KAPPA, OMEGA, OMEGA_MAX, GAMMA = 60.0, 8.0, 16.0, 60.0, 1.0
NEG_L = 12.0
L_MEAN, L_MAX, L_PRI = 1.0, 0.5, 1.0
S_MLP, C_MLP, K_MLP = 10.0, 50.0, 200.0
UNK_OW = 0.3
OW_IDF = 0.0
FUNC_NORM = 30.0                # embedding norm of function words at question positions


def content_tokens(text):
    out = []
    for t in words(text):
        if t in STOP or (len(t) < 2 and not t.isdigit()):
            continue
        c = concept(t)
        if c not in out:
            out.append(c)
    return out


def read_facts():
    """Facts are lines (or ';'-separated parts) of the files in facts/. Every content word of a fact is associated
    with every other."""
    facts, files = [], []
    for fi, path in enumerate(sorted(glob.glob("facts/*.txt"))):
        for line in open(path, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            for part in line.split(";"):
                f = {c: 1.0 for c in content_tokens(part)}
                if len(f) >= 2:
                    facts.append(f)
                    files.append(fi)
    return facts, files


def read_isa():
    isa = {}
    try:
        for line in open("isa.txt", encoding="utf-8"):
            line = line.split("#", 1)[0]
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            parents = content_tokens(right)
            for c in content_tokens(left):
                isa.setdefault(c, []).extend(parents)
    except FileNotFoundError:
        pass
    return isa


def read_priors():
    pri = {}
    try:
        for line in open("priors.txt", encoding="utf-8"):
            line = line.split("#", 1)[0].split()
            if len(line) == 2:
                pri[line[0]] = float(line[1])
    except FileNotFoundError:
        pass
    return pri


def hash_stem(s):
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) % (2 ** 32)
    return h


def fact_groups(facts):
    """Greedy topical grouping of facts into blocks of at most NC facts; the facts of a block get exactly orthogonal
    vectors, so facts that share (idf-weighted) words, which would interfere most, do not interfere at all."""
    df = {}
    for f in facts:
        for c in f:
            df[c] = df.get(c, 0) + 1
    cap = NC - 1                           # zero-sum vectors span NC - 1 dimensions
    k = -(-len(facts) // cap) + 1
    groups = [[] for _ in range(k)]
    counts = [dict() for _ in range(k)]
    for i, f in enumerate(facts):
        best, best_key = None, None
        for g in range(k):
            if len(groups[g]) >= cap:
                continue
            share = sum(counts[g].get(c, 0) / df[c] for c in f)
            key = (share, cap - len(groups[g]))
            if best_key is None or key > best_key:
                best, best_key = g, key
        groups[best].append(i)
        for c in f:
            counts[best][c] = counts[best].get(c, 0) + 1
    return [g for g in groups if g]


def codes(facts, files, isa, rng):
    """Raw code v(c) = sum over facts containing c of w_f * r_f, and the fact mass n(c) = sum of w_f^2.

    The facts of one file (one topic) get mutually orthogonal unit vectors r_f."""
    vec = np.zeros((len(facts), NC))
    for idx in fact_groups(facts):
        g = rng.standard_normal((NC, len(idx)))
        g -= g.mean(0)                     # zero-sum vectors keep every embedding close to zero mean
        qm, _ = np.linalg.qr(g)
        vec[idx] = qm.T
    raw, mass = {}, {}
    for i, f in enumerate(facts):
        wf = min(1.0, math.sqrt(5.0 / len(f)))
        for c, sign in f.items():
            raw[c] = raw.get(c, 0.0) + sign * wf * vec[i]
            mass[c] = mass.get(c, 0.0) + wf * wf
    out_raw, out_mass = dict(raw), dict(mass)
    for c, parents in isa.items():
        ps = [p for p in parents if p in raw and p != c]
        if not ps:
            continue
        mix = sum(raw[p] / math.sqrt(mass[p]) for p in ps) / math.sqrt(len(ps))
        if c in raw:
            m = mass[c]
            out_raw[c] = raw[c] + ISA_W * math.sqrt(m) * mix
            out_mass[c] = m * (1 + ISA_W)
        else:
            out_raw[c] = mix
            out_mass[c] = 1.0
    return out_raw, out_mass


def build():
    vocab = [ln.strip() for ln in open("vocab.txt", encoding="utf-8").read().splitlines()]
    V = len(vocab) + 2
    rng = np.random.default_rng(SEED)
    facts, files = read_facts()
    raw, mass = codes(facts, files, read_isa(), rng)
    ident = {}
    pri = read_priors()

    emb = np.zeros((V, D))
    short = []
    for k, w in enumerate([None] + vocab):
        row = emb[k + 1]
        target = FUNC_NORM
        if w is None:                      # id 1: any word not in vocab.txt
            row[OW] = UNK_OW
        elif w not in STOP:
            c = concept(w)
            if c in raw:
                n = mass[c]
                row[:NC] = raw[c] * n ** -ALPHA_O
                s = stem(w)
                if s not in ident:
                    e = np.random.default_rng(hash_stem(s)).standard_normal(ED)
                    e -= e.mean()
                    ident[s] = e / np.linalg.norm(e)
                row[E0:E0 + ED] = ident[s]
                row[CW] = 1.0
                row[OW] = 1.0 - OW_IDF * math.log10(max(n, 1.0))   # specific words weigh more in the option mean
                s_q = QSCALE * n ** (ALPHA_O - ALPHA_Q)
                target = math.sqrt(2) * ANCHOR / s_q
                row[CWQ] = 1.0 / s_q             # reads as 1 at a question position after LayerNorm
        if w is not None:
            row[NEG] = NEGATION.get(w, 0.0)
            row[PRI] = pri.get(w, 0.0)
        row[ZF] = -row.sum()
        rest = float(row @ row)
        if rest > target ** 2:
            short.append(w)
        fill = math.sqrt(max(target ** 2 - rest, 0.0) / 2)
        row[FIL1], row[FIL2] = fill, -fill

    assert len(short) < 10, f"{len(short)} embeddings cannot reach their norm: {short[:10]}"
    pos = np.zeros((176, D))
    pos[128:, ANP], pos[128:, ANN] = ANCHOR, -ANCHOR
    pos[128:, RO] = 1.0
    pos[:, ZF] = -pos[:, RO]

    sigma0 = ANCHOR * math.sqrt(2.0 / D)   # LayerNorm weight: an anchored token comes out unscaled
    w = {"embed": emb, "pos": pos}
    for i in range(2):
        p = f"block{i}."
        for n in ("ln1", "ln2"):
            w[p + n + ".weight"] = np.full(D, sigma0)
            w[p + n + ".bias"] = np.zeros(D)
        for n in ("q", "k", "v", "o"):
            w[p + n] = np.zeros((D, D))
            w[p + n + ".bias"] = np.zeros(D)
        w[p + "fc1"], w[p + "fc1.bias"] = np.zeros((D, MLP)), np.zeros(MLP)
        w[p + "fc2"], w[p + "fc2.bias"] = np.zeros((MLP, D)), np.zeros(D)

    # ---- layer 1: association heads 0..2, identity head 3 (queries: option words, keys: question words)
    q, k, v, o, qb = w["block0.q"], w["block0.k"], w["block0.v"], w["block0.o"], w["block0.q.bias"]
    for h in range(NB):
        c0 = h * HD
        for i in range(BD):
            q[h * BD + i, c0 + i] = SQ * EPS / QSCALE
            k[h * BD + i, c0 + i] = 1.0
        qb[c0 + 104] = SQ
        k[RO, c0 + 104] = -BIG
        v[CWQ, c0] = 1.0
        o[c0, M], o[c0, ZF] = LAM_M, -LAM_M
    c0 = 3 * HD
    for i in range(ED):
        q[E0 + i, c0 + i] = SQ * BETA_E
        k[E0 + i, c0 + i] = 1.0
    qb[c0 + 104] = SQ
    k[CW, c0 + 104] = -BETA_E * THETA_E
    k[RO, c0 + 104] = -BIG
    v[CWQ, c0] = 1.0
    o[c0, X], o[c0, ZF] = 1.0, -1.0
    # layer-1 MLP: give question tokens the anchor (option tokens already have it)
    f1, f1b, f2 = w["block0.fc1"], w["block0.fc1.bias"], w["block0.fc2"]
    f1[RO, 0], f1b[0] = -100.0, 50.0
    f2[0, ANP], f2[0, ANN] = ANCHOR / 50.0, -ANCHOR / 50.0

    # ---- layer 2: mean and max pooling over the option's content words, negation detector, sign flip in the MLP
    q, k, v, o = w["block1.q"], w["block1.k"], w["block1.v"], w["block1.o"]
    qb, kb = w["block1.q.bias"], w["block1.k.bias"]
    for h, gamma, omega in ((0, 0.0, OMEGA), (1, GAMMA, OMEGA_MAX)):
        c = h * HD + 104
        qb[c] = SQ
        k[RO, c], kb[c] = G, -G
        k[OW, c], k[X, c], k[M, c] = KAPPA, -omega, gamma
    v[M, 0], v[PRI, 1], v[M, HD] = 1.0, 1.0, 1.0
    o[0, FMEAN], o[0, ZF] = 1.0, -1.0
    o[1, PMEAN], o[1, ZF] = 1.0, -1.0
    o[HD, FMAX], o[HD, ZF] = 1.0, -1.0
    c = 2 * HD + 104
    qb[c] = SQ
    k[NEG, c], k[RO, c] = NEG_L, -G
    v[NEG, 2 * HD] = 1.0
    o[2 * HD, NN], o[2 * HD, ZF] = 1.0, -1.0
    # head 3: share of content words among the question tokens = the association heads' baseline
    c = 3 * HD + 104
    qb[c] = SQ
    k[RO, c] = -G
    v[CW, 3 * HD] = 1.0
    o[3 * HD, BASE], o[3 * HD, ZF] = NB * LAM_M, -NB * LAM_M

    f1, f1b, f2 = w["block1.fc1"], w["block1.fc1.bias"], w["block1.fc2"]
    f1[FMEAN, 0], f1[FMAX, 0], f1[NN, 0] = S_MLP * L_MEAN, S_MLP * L_MAX, S_MLP * K_MLP
    f1[BASE, 0] = -S_MLP * (L_MEAN + L_MAX)
    f1b[0] = S_MLP * (C_MLP - K_MLP)
    f2[0, FIN], f2[0, ZF] = -2.0 / S_MLP, 2.0 / S_MLP

    w["ln_f.weight"], w["ln_f.bias"] = np.full(D, sigma0), np.zeros(D)
    score = np.zeros(D)
    score[FMEAN], score[FMAX], score[FIN], score[PRI] = L_MEAN, L_MAX, 1.0, L_PRI
    score[BASE] = -(L_MEAN + L_MAX)
    w["score"] = score
    return {n: torch.tensor(a, dtype=torch.float32) for n, a in w.items()}

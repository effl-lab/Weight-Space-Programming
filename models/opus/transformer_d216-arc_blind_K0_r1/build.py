"""Builds every weight of the ARC scorer from the hand-written knowledge in knowledge/*.txt.

See NOTES.md for the design. Nothing here is trained or fitted: word codes are sums of fixed pseudo-random
vectors (seeded), one per hand-written fact line and paragraph, and every other weight is set by formula.
"""
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import common

D, HEADS, MLP, NPOS = 216, 4, 432, 176
HD = D // HEADS                    # 54 dims per head
CB = 51                            # code dims per head block
NCODE = CB * HEADS                 # 204 code dims
(CONTENT, STOP, ISQ, ISO, M, S, COMP, CONST, ECHO, CONSTN, NEGW, NEGF) = range(NCODE, D)
Q_LEN = 128

# Hand-chosen constants (see NOTES.md).
L = 30.0          # +L and -L in two dims of every row keep every LayerNorm a near-fixed scaling
BETA = 15.0       # sharpness of an association match
THETA = 0.1       # code dot product a match must exceed to count as much as one question stopword
TAU = 1.5         # logit of a question stopword (the sink of the association heads)
BIG = 1000.0      # keeps option tokens from attending to option tokens
PARA = 10         # lines per paragraph
TOPIC = 160       # lines per topic chunk: 16 paragraphs, 4 per head block (44 vectors of at most 51)
PARA_W = 0.0      # weight of paragraph vectors in a code (off: they added crosstalk without helping)
LAMBDA = 0.1      # cost of an option content word, so unmatched words count slightly against an option
GATE_B, GATE_C = 30.0, 4.0     # GELU gates: a flag times GATE_B switches a unit off; GATE_C keeps it linear
ECHO_GAMMA = 40.0  # sharpness of the echo detector (an option word that is also a question word)
ECHO_ETA = 0.65    # projected self-similarity above which a question word counts as the same word
ECHO_MIN = 1.5     # detectors (of ECHO_HEADS) that must fire to drop an option word
ECHO_ON = 1.0      # 1: echoed option words score nothing; 0: off
ECHO_HEADS = 3     # heads of block 1 used as echo detectors; the last head detects negation
NEG_WORDS = ("except", "least")   # question words that make the question ask for the least associated option
NEG_GAMMA = 10.0   # logit of a negation word for the negation head (any other question token: 0)
NEG_ON = 1.0
DERIV_ON = 1.0     # 1: a word whose stem has no code may use the stem left after removing a derivational suffix


def topic_basis(topic: int, block: int, rng_seed: int = 1234) -> np.ndarray:
    """Fixed pseudo-random orthonormal basis (rows) of one head block for one topic chunk."""
    g = np.random.default_rng([rng_seed, topic, block])
    qm, _ = np.linalg.qr(g.standard_normal((CB, CB)))
    return qm.T


def stem_codes(folder: Path) -> dict[str, np.ndarray]:
    """Code of every stem: a sum of fixed vectors, one per fact line it is in and one per paragraph (PARA
    consecutive lines of a file) it is in, normalised to length 1. Lines and paragraphs are grouped into topic
    chunks of TOPIC consecutive lines of one file; inside a chunk every vector is exactly orthogonal to every
    other (a random orthonormal basis per chunk and head block), so facts of one topic do not blur into each
    other; vectors of different chunks are random relative to each other. All vectors of one paragraph share a
    head block, so the facts two words share tend to meet in one head."""
    lines = []                                             # (weight, stems, paragraph key, topic key)
    for weight, ws, fname, lineno in common.knowledge_lines(folder, with_source=True):
        stems = sorted({common.stem(w) for w in ws if w not in common.STOPWORDS})
        stems = [s for s in stems if s not in common.STOPWORDS and len(s) > 1]
        if len(stems) >= 2:
            lines.append((weight, stems, (fname, lineno // PARA), (fname, lineno // TOPIC)))
    topic_ids, slot_count, slots, bases = {}, defaultdict(int), {}, {}
    raw = defaultdict(lambda: np.zeros(NCODE))

    def add(s, key, topic, para, amount):
        if key not in slots:
            t = topic_ids.setdefault(topic, topic)
            b = (para[1] % (TOPIC // PARA)) % HEADS
            n = slot_count[t, b]
            slot_count[t, b] += 1
            if n >= CB:
                raise ValueError(f"topic chunk {topic} block {b} has more than {CB} vectors")
            slots[key] = (t, b, n)
        t, b, r = slots[key]
        if (t, b) not in bases:
            bases[t, b] = topic_basis(zlib.crc32(f"{topic[0]}:{topic[1]}".encode()), b)
        raw[s][b * CB:(b + 1) * CB] += amount * bases[t, b][r]

    in_para = defaultdict(lambda: defaultdict(float))
    for li, (weight, stems, pkey, tkey) in enumerate(lines):
        for s in stems:
            add(s, ("line", li), tkey, pkey, weight)
            in_para[s][(pkey, tkey)] += weight
    for s, paras in in_para.items() if PARA_W > 0 else ():
        for (pkey, tkey), n in paras.items():
            add(s, ("para",) + pkey, tkey, pkey, PARA_W * math.sqrt(n))
    codes = {}
    for s, u in raw.items():
        n = np.linalg.norm(u)
        if n > 0:
            codes[s] = u / n
    return codes


def echo_projection(head: int, rng_seed: int = 99) -> np.ndarray:
    """(NCODE, HD - 1) fixed random projection; the dot product of two projected unit codes is their code dot
    product plus noise of about 0.14, so only a word and itself (or a word always used with it) exceed 0.65."""
    g = np.random.default_rng([rng_seed, head])
    return g.standard_normal((NCODE, HD - 1)) / math.sqrt(HD - 1)


def build() -> dict[str, torch.Tensor]:
    folder = Path(".")
    vocab = [ln.strip() for ln in (folder / "vocab.txt").read_text().splitlines()]
    V = len(vocab) + 2
    codes = stem_codes(folder)

    embed = np.zeros((V, D))
    embed[1, STOP] = 1.0                                   # unknown word: a sink, like a stopword
    for i, w in enumerate(vocab):
        s = next((c for c in (common.stem_candidates(w) if DERIV_ON else [common.stem(w)]) if c in codes), None)
        if w in common.STOPWORDS or s is None:
            embed[i + 2, STOP] = 1.0
        else:
            embed[i + 2, :NCODE] = codes[s]
            embed[i + 2, CONTENT] = 1.0
    for i, w in enumerate(vocab):
        if w in NEG_WORDS:
            embed[i + 2, NEGW] = 1.0
    embed[0] = 0.0
    embed[:, COMP] = -embed.sum(1)                         # every row sums to zero, so LayerNorm's mean is 0

    pos = np.zeros((NPOS, D))
    pos[:Q_LEN, ISQ] = 1.0
    pos[Q_LEN:, ISO] = 1.0
    pos[:, CONST] = L
    pos[:, CONSTN] = -L
    pos[:, COMP] = -pos.sum(1)

    # Every residual row is (code of norm <= 1, a few flags of 1, CONST = L, CONSTN = -L, COMP = -(sum of the
    # rest), a few small written values), so its mean is 0 and its standard deviation is close to sigma: each LayerNorm (weight 1, bias 0) divides by about
    # sigma. Weights that read the residual are multiplied by sigma to undo that.
    sigma = math.sqrt((2 * L * L + 6.0) / D)
    zero = lambda *s: np.zeros(s)                          # noqa: E731
    rt = math.sqrt(HD)
    w = {"embed": embed, "pos": pos}

    def writes(o_or_fc2, row, dim, amount):
        """Make a write of `amount` into residual dim `dim` that keeps the row sum zero (COMP absorbs it)."""
        o_or_fc2[row, dim] += amount
        o_or_fc2[row, COMP] -= amount

    # ---- block 0: association heads. Option token i attends to question tokens j with logit
    #      BETA * (u_i . u_j within the head's block - THETA) for content words, TAU for stopwords; option
    #      tokens are masked by -BIG. The value is 1 for content words, so M_i = mass on associated words.
    q, k, v, o = zero(D, D), zero(D, D), zero(D, D), zero(D, D)
    qb, kb = zero(D), zero(D)
    for b in range(HEADS):
        cols = slice(b * HD, b * HD + CB)
        rows = slice(b * CB, (b + 1) * CB)
        q[rows, cols] = np.eye(CB) * rt * BETA * sigma
        k[rows, cols] = np.eye(CB) * sigma
        c = b * HD + CB                                    # constant channel of the head
        qb[c] = rt
        k[CONTENT, c] = -BETA * THETA * sigma
        k[STOP, c] = TAU * sigma
        k[ISO, c] = -BIG * sigma
        v[CONTENT, b * HD] = sigma
        writes(o, b * HD, M, 1.0)
    # MLP: S_i = M_i - LAMBDA for option content words, 0 for stopwords and question tokens (gated off).
    fc1, fc1b, fc2 = zero(D, MLP), zero(MLP), zero(MLP, D)
    fc1[M, 0] = sigma
    fc1[CONTENT, 0] = -LAMBDA * sigma
    fc1[ISO, 0] = fc1[ISO, 1] = GATE_B * sigma
    fc1[CONTENT, 0] += GATE_B * sigma
    fc1[CONTENT, 1] = GATE_B * sigma
    fc1b[0] = fc1b[1] = GATE_C - 2 * GATE_B
    writes(fc2, 0, S, 1.0)
    writes(fc2, 1, S, -1.0)
    blocks = [dict(q=q, k=k, v=v, o=o, qb=qb, kb=kb, fc1=fc1, fc1b=fc1b, fc2=fc2)]

    # ---- block 1: echo detectors (heads 0..ECHO_HEADS-1) and a negation detector (last head).
    #      An echo head compares fixed random projections of the whole code, so a question word identical to the
    #      option word gives about 1 and any other word about 0 +- 0.14. The negation head puts nearly all its
    #      attention on a negation word of the question if there is one, and reads the NEGW flag.
    q, k, v, o = zero(D, D), zero(D, D), zero(D, D), zero(D, D)
    qb, kb = zero(D), zero(D)
    for h in range(ECHO_HEADS):
        P = echo_projection(h)
        cols = slice(h * HD, h * HD + HD - 1)
        q[:NCODE, cols] = P * rt * ECHO_GAMMA * sigma
        k[:NCODE, cols] = P * sigma
        c = h * HD + HD - 1
        qb[c] = rt
        k[CONTENT, c] = -ECHO_GAMMA * ECHO_ETA * sigma
        k[ISO, c] = -BIG * sigma
        v[CONTENT, h * HD] = sigma
        writes(o, h * HD, ECHO, 1.0)
    h = HEADS - 1
    qb[h * HD] = rt
    k[NEGW, h * HD] = NEG_GAMMA * sigma
    k[ISO, h * HD] = -BIG * sigma
    v[NEGW, h * HD] = sigma
    writes(o, h * HD, NEGF, 1.0)
    # MLP, in pairs of units g(x) = GELU(x + S) - GELU(x), which is S when x >> 0 and 0 when x << 0:
    #   pair 0 (echo): x = C + B(ISO - 1) + B(ECHO - ECHO_MIN)                  writes -S
    #   pair 1 (negation): x = C + B(ISO - 1) + B(NEGF - 1)                      writes -2S
    #   pair 2 (both): x = C + B2(ISO - 1) + B2(NEGF - 1) + 0.6 B2 (ECHO - ECHO_MIN)   writes +2S
    #   so an option word scores S normally, -S in a negated question, and 0 when it echoes the question.
    fc1, fc1b, fc2 = zero(D, MLP), zero(MLP), zero(MLP, D)
    B2 = 70.0
    gates = [({ISO: GATE_B, ECHO: GATE_B}, GATE_C - GATE_B - GATE_B * ECHO_MIN, -ECHO_ON),
             ({ISO: GATE_B, NEGF: GATE_B}, GATE_C - 2 * GATE_B, -2.0 * NEG_ON),
             ({ISO: B2, NEGF: B2, ECHO: 0.6 * B2}, GATE_C - 2 * B2 - 0.6 * B2 * ECHO_MIN, 2.0 * NEG_ON * ECHO_ON)]
    for g, (wts, bias, out) in enumerate(gates):
        for unit, with_s in ((2 * g, 1.0), (2 * g + 1, 0.0)):
            for dim, wt in wts.items():
                fc1[dim, unit] = wt * sigma
            fc1[S, unit] = with_s * sigma
            fc1b[unit] = bias
        writes(fc2, 2 * g, S, out)
        writes(fc2, 2 * g + 1, S, -out)
    blocks.append(dict(q=q, k=k, v=v, o=o, qb=qb, kb=kb, fc1=fc1, fc1b=fc1b, fc2=fc2))

    for i, bl in enumerate(blocks):
        p = f"block{i}."
        w[p + "ln1.weight"], w[p + "ln1.bias"] = np.ones(D), zero(D)
        w[p + "ln2.weight"], w[p + "ln2.bias"] = np.ones(D), zero(D)
        w[p + "q"], w[p + "k"], w[p + "v"], w[p + "o"] = bl["q"], bl["k"], bl["v"], bl["o"]
        w[p + "q.bias"], w[p + "k.bias"], w[p + "v.bias"], w[p + "o.bias"] = bl["qb"], bl["kb"], zero(D), zero(D)
        w[p + "fc1"], w[p + "fc1.bias"] = bl["fc1"], bl["fc1b"]
        w[p + "fc2"], w[p + "fc2.bias"] = bl["fc2"], zero(D)
    lnf = zero(D)
    lnf[S] = 1.0
    w["ln_f.weight"], w["ln_f.bias"] = lnf, zero(D)
    score = zero(D)
    score[S] = 1.0
    w["score"] = score
    return {n: torch.tensor(a, dtype=torch.float32) for n, a in w.items()}

"""Builds every weight of the d=8 transformer scorer from hand-written lexicon files.

See NOTES.md for the design. All arithmetic is float64 so that very sharp attention is exact.
"""
import math
import re
from pathlib import Path

import numpy as np
import torch

D, HEADS, MLP = 8, 4, 16
Q_LEN, O_LEN = 128, 48
HERE = Path(__file__).resolve().parent

# ---- design constants -------------------------------------------------------------------------------------------
F_FLAG = 1000.0     # |u0| of content tokens: + in the question, - in the options (dominates the norm: LN ~ linear)
RHO = 1.0           # norm of each 2-d cluster code
A_FUNC = 1e9        # anchor of function / unknown words (their final state is ~ pure u0, i.e. contributes 0)
A_Q = 1e10          # anchor added by MLP1 to question tokens (their final contribution becomes ~ 0)
SEP = 25.0          # nats separating neighbouring cluster angles in the sharp heads
G_MATCH = 1.0       # weight of one head's match fraction in u1
K_NEG = 10.0        # offset written to u1 by the negation head
LAMBDA = 1.5        # u1 = a + b - LAMBDA min(a, b) for the two sharp heads
MIRROR = False      # single-cluster words: mirror the cluster into the other head (True) or a private code
NEG_SIN = 0.3       # u6 share of a negation word's embedding


def basis():
    """Seven orthonormal directions of R^8 orthogonal to the all-ones vector (LayerNorm removes that one)."""
    m = np.zeros((8, 8))
    m[:, 0] = 1.0
    for i in range(1, 8):
        m[:, i] = 0.0
        m[i - 1, i] = 1.0
        m[i, i] = -1.0
    m[:, 1:] = np.random.RandomState(0).randn(8, 7)
    q, _ = np.linalg.qr(m)
    return q[:, 1:].T  # (7, 8)


# ---- lexicon ------------------------------------------------------------------------------------------------------
VOWELS = set("aeiou")


def inflect(w: str) -> list[str]:
    """Generous inflection of an English base word (over-generation is harmless)."""
    out = [w]
    if not re.fullmatch(r"[a-z]+", w) or len(w) < 3:
        return out
    if w.endswith("s") and not re.search(r"(ss|us|is)$", w):   # already a plural / 3rd person form
        out += [w + "'", w[:-1], w[:-1] + "'s"]
        if w.endswith("ies"):
            out.append(w[:-3] + "y")
        elif re.search(r"(ses|xes|zes|ches|shes)$", w):
            out.append(w[:-2])
        return out
    # plural / 3rd person
    if re.search(r"(s|x|z|ch|sh)$", w):
        out.append(w + "es")
    elif re.search(r"[^aeiou]y$", w):
        out.append(w[:-1] + "ies")
    else:
        out.append(w + "s")
    if w.endswith("f"):
        out.append(w[:-1] + "ves")
    if w.endswith("fe"):
        out.append(w[:-2] + "ves")
    # past / participle
    if w.endswith("e"):
        out += [w + "d", w[:-1] + "ing"]
    elif re.search(r"[^aeiou]y$", w):
        out += [w[:-1] + "ied", w + "ing"]
    else:
        out += [w + "ed", w + "ing"]
        if len(w) <= 5 and w[-1] not in "aeiouwxy" and w[-2] in VOWELS and w[-3] not in VOWELS:
            out += [w + w[-1] + "ed", w + w[-1] + "ing"]
    out += [w + "'s", w + "s'"]
    return out


LEMMA = {}   # form -> the written word it was generated from (its own private code when it has one cluster)


def read_lexicon():
    """lex/*.txt: one cluster per line (words separated by spaces, '#' starts a comment).
    word! = not inflected; word? = low priority (only takes a free slot, like generated inflections).
    Returns clusters as lists of (form, priority, generated): priority 0 head word (first of its line), 1 written,
    2 low; generated = 1 for an inflected form (it inherits the priority of the word it comes from)."""
    clusters = []
    for path in sorted((HERE / "lex").glob("*.txt")):
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip().lower()
            if not line:
                continue
            members = []
            toks = line.split()
            if toks[0].startswith("@"):            # a label naming the line (for hubs.txt), not a word
                members.append((toks[0], 9, 1))
                toks = toks[1:]
            split = []
            for tok in toks:          # the tokenizer splits on '-': a hyphenated entry becomes its parts
                parts = tok.rstrip("?!").split("-")
                if len(parts) > 1:
                    split += [p + "?" for p in parts if p]
                else:
                    split.append(tok)
            toks = split
            for j, tok in enumerate(toks):
                low = tok.endswith("?")
                tok = tok.rstrip("?")
                base = 2 if low else (0 if j == 0 else 1)   # 0: head word of the line, 1: written, 2: low
                if tok.endswith("!"):
                    members.append((tok[:-1], base, 0))
                    LEMMA.setdefault(tok[:-1], tok[:-1])
                else:
                    members += [(f, base, int(i > 0)) for i, f in enumerate(inflect(tok))]
                    for f in inflect(tok):
                        LEMMA.setdefault(f, tok)
            clusters.append(members)
    # hubs.txt: "word headA headB" pins a word (and its inflections) to the lines headed by headA and headB
    heads = {}
    for low in (False, True):     # a line whose head word is written plainly wins over one headed by "word?"
        for ci, members in enumerate(clusters):
            if (members[0][1] == 2) == low:
                heads.setdefault(members[0][0], ci)
    for line in (HERE / "hubs.txt").read_text().splitlines():
        line = line.split("#")[0].strip().lower()
        if not line:
            continue
        word, *hs = line.split()
        forms = inflect(word.rstrip("!")) if not word.endswith("!") else [word[:-1]]
        # an inflection written in a line without the word itself is a word of its own (weather pins not weathering)
        own = {f for c in clusters if not any(m[0] in forms[:1] for m in c) for f, p, g in c if not g and p <= 1}
        forms = [forms[0]] + [f for f in forms[1:] if f not in own]
        missing = [h for h in hs if h not in heads]
        assert not missing, f"hubs.txt: no line headed {missing}"
        for c in range(len(clusters)):
            clusters[c] = [m for m in clusters[c] if m[0] not in forms]
        for h in hs:
            clusters[heads[h]] += [(f, -1, int(i > 0)) for i, f in enumerate(forms)]
    negs = [w.strip() for w in (HERE / "neg.txt").read_text().split() if w.strip()]
    stop = set((HERE / "stop.txt").read_text().split())
    return clusters, negs, stop


def assign(clusters, stop):
    """Gives each cluster a head (0 or 1) and each word at most one cluster per head.

    Each word wants its two best clusters (head word of a line > written member > generated / low, then file order).
    Heads are chosen by a greedy max-cut so that as many words as possible get both, then every word takes, in its
    order of preference, the free slots."""
    prefs = {}
    for ci, members in enumerate(clusters):
        for f, p, g in members:
            if f in stop or f.startswith("@"):
                continue
            prefs.setdefault(f, {})
            prefs[f][ci] = min((p, g), prefs[f].get(ci, (9, 9)))
    order = {w: sorted(d, key=lambda c: (d[c], c)) for w, d in prefs.items()}
    nbr = [dict() for _ in clusters]
    for w, cs in order.items():
        if len(cs) >= 2:
            a, b = cs[0], cs[1]
            if prefs[w][a][0] < 0 and prefs[w][b][0] < 0:
                wt = 100.0   # a word pinned in hubs.txt must get both of its lines
            else:
                wt = 3.0 - 0.5 * (prefs[w][a][0] + prefs[w][b][0]) - 0.5 * (prefs[w][a][1] + prefs[w][b][1])
            nbr[a][b] = nbr[a].get(b, 0) + wt
            nbr[b][a] = nbr[b].get(a, 0) + wt
    head_of = [0] * len(clusters)
    placed = [False] * len(clusters)
    # words pinned in hubs.txt are hard constraints: 2-colour their lines first (breadth first)
    pin = [[] for _ in clusters]
    for w, cs in order.items():
        if len(cs) >= 2 and prefs[w][cs[0]][0] < 0 and prefs[w][cs[1]][0] < 0:
            pin[cs[0]].append(cs[1])
            pin[cs[1]].append(cs[0])
    fixed = [False] * len(clusters)
    for start in range(len(clusters)):
        if fixed[start] or not pin[start]:
            continue
        fixed[start], queue = True, [start]
        while queue:
            c = queue.pop()
            for o in pin[c]:
                if not fixed[o]:
                    head_of[o], fixed[o] = 1 - head_of[c], True
                    queue.append(o)
    placed = list(fixed)
    for ci in range(len(clusters)):
        if fixed[ci]:
            continue
        gain = sum(wt * (1 if head_of[o] == 0 else -1) for o, wt in nbr[ci].items() if placed[o])
        head_of[ci] = 1 if gain > 0 else 0
        placed[ci] = True
    for _ in range(20):
        changed = False
        for ci in range(len(clusters)):
            if fixed[ci]:
                continue
            same = sum(wt for o, wt in nbr[ci].items() if head_of[o] == head_of[ci])
            diff = sum(wt for o, wt in nbr[ci].items() if head_of[o] != head_of[ci])
            if same > diff:
                head_of[ci] ^= 1
                changed = True
        if not changed:
            break
    slots = {}
    for w, cs in order.items():
        s = [None, None]
        for c in cs:
            if s[head_of[c]] is None:
                s[head_of[c]] = c
        slots[w] = s
    return slots, head_of


def build():
    clusters, negs, stop = read_lexicon()
    slots, head_of = assign(clusters, stop)
    negs = [w for w in negs if w not in slots]
    vocab = sorted(slots) + negs
    file_vocab = [l.strip() for l in (HERE / "vocab.txt").read_text().splitlines()]
    assert file_vocab == vocab, "vocab.txt is stale: run python build.py"
    V = len(vocab) + 2
    U = basis()
    u = [torch.tensor(U[i]) for i in range(7)]

    # every cluster has an angle in both heads (its own head, and a mirror angle in the other one for words that
    # have only that cluster); angles follow file order so neighbouring clusters are neighbours on the circle
    # every word with a single cluster gets, in its other head, a private angle shared only with its own
    # inflections: a word matching in both heads is then (nearly always) a repetition of a question word
    lemmas = sorted({LEMMA.get(w, w) for w, sl in slots.items() if None in sl})
    priv = {l: len(clusters) + i for i, l in enumerate(lemmas)}
    nc = len(clusters) + len(lemmas)
    ang = np.arange(nc) * 2 * math.pi / nc

    E = torch.zeros(V, 8, dtype=torch.float64)
    E[1] = A_FUNC * u[0]
    idx = {w: i + 2 for i, w in enumerate(vocab)}
    for w, (c0, c1) in slots.items():
        if MIRROR:
            c0 = c1 if c0 is None else c0
            c1 = c0 if c1 is None else c1
        else:
            c0 = priv[LEMMA.get(w, w)] if c0 is None else c0
            c1 = priv[LEMMA.get(w, w)] if c1 is None else c1
        e = RHO * (math.cos(ang[c0]) * u[2] + math.sin(ang[c0]) * u[3])
        e = e + RHO * (math.cos(ang[c1]) * u[4] + math.sin(ang[c1]) * u[5])
        E[idx[w]] = e
    for w in negs:
        E[idx[w]] = A_FUNC * (math.sqrt(1 - NEG_SIN ** 2) * u[0] + NEG_SIN * u[6])

    P = torch.zeros(Q_LEN + O_LEN, 8, dtype=torch.float64)
    P[:Q_LEN] = F_FLAG * u[0]
    P[Q_LEN:] = -F_FLAG * u[0]

    z = lambda *s: torch.zeros(*s, dtype=torch.float64)
    w = {"embed": E, "pos": P}
    sig = math.sqrt((F_FLAG ** 2 + 2 * RHO ** 2) / 8)   # LayerNorm std of every content token at block 0
    xr = RHO / sig                                       # normalised code norm
    S = SEP / (1 - math.cos(2 * math.pi / nc))           # attention logit of a perfect match
    s = math.sqrt(S * math.sqrt(2)) / xr

    # ---- block 0: two sharp cluster heads; head A writes its match fraction a to u1, head B writes b to u6.
    # The MLP turns them into u1 = a + b - LAMBDA * min(a, b) and clears u6 again. Matching in both heads almost always
    # means the option repeats a question word (single-cluster words have a private code in their other head), so with
    # LAMBDA = 1.5 a repetition counts half of a real association.
    q, k, v, o = z(8, 8), z(8, 8), z(8, 8), z(8, 8)
    qb, kb, vb, ob = z(8), z(8), z(8), z(8)
    xf = F_FLAG / sig                                    # normalised |u0| of content tokens
    for h, (a, b) in enumerate(((2, 3), (4, 5))):
        q[:, 2 * h], q[:, 2 * h + 1] = s * u[a], s * u[b]
        k[:, 2 * h], k[:, 2 * h + 1] = s * u[a], s * u[b]
        v[:, 2 * h] = u[0] / (2 * xf)                      # question content token -> 1, option content token -> 0
        vb[2 * h] = 0.5
    o[0] = G_MATCH * u[1]
    o[2] = G_MATCH * u[6]
    w.update({"block0.q": q, "block0.k": k, "block0.v": v, "block0.o": o,
              "block0.q.bias": qb, "block0.k.bias": kb, "block0.v.bias": vb, "block0.o.bias": ob})
    fc1, fc1b, fc2, fc2b = z(8, MLP), z(MLP), z(MLP, 8), z(8)
    kk = 100.0 / G_MATCH
    fc1[:, 0] = kk * sig * u[6]                 # ~ b
    fc2[0] = (u[1] - u[6]) / kk
    fc1[:, 1] = kk * sig * (u[1] - u[6])        # ~ relu(a - b)
    fc2[1] = LAMBDA * u[1] / kk
    fc1[:, 2] = kk * sig * u[1]                 # ~ a
    fc2[2] = -LAMBDA * u[1] / kk
    w.update({"block0.fc1": fc1, "block0.fc1.bias": fc1b, "block0.fc2": fc2, "block0.fc2.bias": fc2b})
    for n in ("ln1", "ln2"):
        w[f"block0.{n}.weight"], w[f"block0.{n}.bias"] = torch.ones(8, dtype=torch.float64), z(8)

    # ---- block 1: a negation head, then an MLP that anchors the question tokens (so that only option tokens are
    # read out) and flips the sign of the match on negated questions
    q, k, v, o = z(8, 8), z(8, 8), z(8, 8), z(8, 8)
    qb, kb, vb, ob = z(8), z(8), z(8), z(8)
    # head 2: every token looks for a negation word (key along u6), value = 1 on it
    xn = math.sqrt(8) * NEG_SIN
    qb[4] = 1.0
    k[:, 4] = (60.0 / xn) * math.sqrt(2) * u[6]
    v[:, 4] = u[6] / xn
    o[4] = K_NEG * u[1]
    w.update({"block1.q": q, "block1.k": k, "block1.v": v, "block1.o": o,
              "block1.q.bias": qb, "block1.k.bias": kb, "block1.v.bias": vb, "block1.o.bias": ob})
    fc1, fc1b, fc2, fc2b = z(8, MLP), z(MLP), z(MLP, 8), z(8)
    # unit 0: question tokens (and function words) get a huge anchor
    fc1[:, 0] = 10.0 * u[0] / xf
    fc2[0] = (A_Q / 10.0) * u[0]
    # unit 1: on negation u1 = y = K + match -> y - 2 (y - K/2) = -match   (y = x.u1 * sig for a content token)
    kn = 20.0
    fc1[:, 1] = kn * sig * u[1]
    fc1b[1] = -kn * K_NEG / 2
    fc2[1] = (-2.0 / kn) * u[1]
    w.update({"block1.fc1": fc1, "block1.fc1.bias": fc1b, "block1.fc2": fc2, "block1.fc2.bias": fc2b})
    for n in ("ln1", "ln2"):
        w[f"block1.{n}.weight"], w[f"block1.{n}.bias"] = torch.ones(8, dtype=torch.float64), z(8)

    w["ln_f.weight"], w["ln_f.bias"] = torch.ones(8, dtype=torch.float64), z(8)
    w["score"] = 1000.0 * u[1]
    return w


if __name__ == "__main__":
    clusters, negs, stop = read_lexicon()
    slots, head_of = assign(clusters, stop)
    negs = [x for x in negs if x not in slots]
    vocab = sorted(slots) + negs
    (HERE / "vocab.txt").write_text("\n".join(vocab) + "\n")
    two = sum(1 for s in slots.values() if None not in s)
    print(f"{len(clusters)} clusters, {len(vocab)} words ({two} with two clusters)")

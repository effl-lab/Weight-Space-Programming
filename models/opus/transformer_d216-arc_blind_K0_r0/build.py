"""Builds every weight of the ARC scorer from the hand-written knowledge files in this folder.

See NOTES.md for the design. Nothing here is trained: the knowledge base (kb/*.txt) is compiled into word codes
with fixed formulas and a fixed random seed, and the transformer weights are set by hand so that the network
computes, for every option, how well its words cover the (weighted) content words of the question.

Score of an option = (1/N) * sign * sum over question content words j of  w_j * (c_j - rho_j * [option repeats j])
  c_j   = sum over the 4 code blocks h of  A_jh / (A_jh + e^SINK),  A_jh = sum over option content words o of
          exp(LAMBDA * u_jh . u_oh)   (u = word code, unit length over the 4 blocks)
  rho_j = what j would add if the option only repeated j itself (so repeated words count for nothing)
  w_j   = (1 + number of KB facts containing j) ** -KEY_IDF,  sign = -1 for questions with not/except/least/...
  N     = number of real tokens of question + option (the mean pooling of the given architecture).
"""
import math
import os
import re

import numpy as np
import torch

D, HEADS, MLP, HD = 216, 4, 432, 54
Q_LEN, O_LEN = 128, 48
TOKEN = re.compile(r"[a-z0-9']+")

# ---------------------------------------------------------------- design constants
LAMBDA = 16.0       # sharpness of word-word association inside attention (exponent per unit cosine)
SINK = 4.0          # score of the self "sink" a question word attends to; per head c = A / (A + e^SINK)
ALPHA = 0.45        # share of a fact code that is the topic code (topic-level relatedness)
SEED = 1234
KEY_IDF = 1.0       # weight of a question word = (1 + number of facts it is in) ** -KEY_IDF
STD0 = 1000.0       # every layer-0 input row has exactly this std (and zero mean), so LayerNorm is a fixed scale
G_MASK = 30.0       # score penalty that removes a key from attention
PM_GAP = 30.0       # minimum score gap between a question word's own position and any other question position
GATE = 200.0        # gating offset in the MLPs (GELU(x - GATE) = 0)
ID_LAMBDA = 90.0    # sharpness of the identity check between a question word and the option words
ID_THETA = 0.9      # hash dot product above which two words count as the same word
PSCALE = 1000.0     # position-matching vectors are stored at this scale (keeps float32 LayerNorm accurate)
WRITE_VOCAB = False  # set by `python build.py --write-vocab`
AMP = 1000.0        # amplification used to set each question word's final norm (its weight in the score)

# ---------------------------------------------------------------- residual layout
BAL, FIL, ISQ, ISO, STOP, NEGW = 0, 1, 2, 3, 4, 5
PM = (6, 7)                   # position-matching vectors of question positions (for the self sink)
SC = (8, 9, 10, 11)           # per head: -LAMBDA |u_h|^2 (cancels a word's association with itself)
R, NEG, F, Z, PZ, ID, RHO, WA, WB = 12, 13, 14, 15, 16, 17, 18, 19, 20
CODE0 = 21
DH = 45                       # code dims per head
HASH0 = CODE0 + HEADS * DH    # word identity hash
NH = D - HASH0                # 15 dims
assert NH >= 12


# ---------------------------------------------------------------- words
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def words(text):
    return TOKEN.findall(text.lower())


KB_NEG = {"no", "never", "without", "neither", "nor", "instead"}
IRREG = {}
for ln in read_lines("irregular.txt"):
    parts = words(ln)
    if len(parts) >= 2:
        for p in parts[1:]:
            IRREG[p] = parts[0]

NOSTEM = set()
ING_KEEP = {"spring", "string", "evening", "ceiling", "sing", "king", "ring", "wing", "thing", "bring", "swing",
            "sting", "lightning", "ping", "beijing", "viking", "sling", "cling", "fling"}


def _undouble(w):
    if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in "lsz" and w[-1] not in "aeiou":
        return w[:-1]
    return w


ADJ = set(words(" ".join(read_lines("adjectives.txt"))))


def _adj_base(w):
    """Base adjective of a comparative, superlative or -ly adverb (only for adjectives in adjectives.txt)."""
    for suf in ("est", "er", "ly", "ily", "ier", "iest"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            b = w[:-len(suf)]
            for c in (b, b + "e", b[:-1] if len(b) > 2 and b[-1] == b[-2] else b, b + "y"):
                if c in ADJ:
                    return c
    return None


def stem(w):
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    w = w.replace("'", "")
    if w in IRREG:
        w = IRREG[w]
    if w not in ADJ:
        a = _adj_base(w)
        if a:
            w = a
    if w in NOSTEM or len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith(("ches", "shes", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ed") and len(w) > 4 and not w.endswith("eed"):
        w = _undouble(w[:-2])
    elif w.endswith("ing") and len(w) > 5 and w not in ING_KEEP:
        w = _undouble(w[:-3])
    if w in IRREG:
        w = IRREG[w]
    if w.endswith("ie") and len(w) > 4:
        w = w[:-2] + "y"  # calorie, calories -> calory
    if w.endswith("e") and len(w) > 3 and not w.endswith(("ee", "ie", "ye")):
        w = w[:-1]
    return w


def forms(s, surface):
    """Surface forms that stem to stem s (inflections), generated from a stem and one surface form."""
    base = {s, surface}
    if surface.endswith("e"):
        base.add(surface[:-1])
    out = set()
    for b in base:
        for c in (b, b + "e", b + "s", b + "es", b + "ed", b + "d", b + "ing", b + "er", b + "ers", b + "est", b + "ly",
                  b + b[-1:] + "est", b + "ier", b[:-1] + "ier" if b.endswith("y") else b, b[:-1] + "iest" if b.endswith("y") else b,
                  b[:-1] + "ily" if b.endswith("y") else b,
                  b + b[-1:] + "ing", b + b[-1:] + "ed", b + b[-1:] + "er", b + "'s", b + "s'"):
            out.add(c)
        if b.endswith("y"):
            out.update({b[:-1] + "ies", b[:-1] + "ied"})
        if b.endswith("e"):
            out.update({b[:-1] + "ing", b + "s", b + "d"})
    return {c for c in out if stem(c) == s}


# ---------------------------------------------------------------- knowledge base
def load_kb():
    stop = set()
    for ln in read_lines("stop.txt"):
        stop.update(words(ln))
    neg = set(words(" ".join(read_lines("neg.txt"))))
    for ln in read_lines("nostem.txt"):
        NOSTEM.update(words(ln))
    merges = {}
    topics, facts, surface = [], [], {}
    topic = "misc"
    files = sorted(f for f in os.listdir("kb") if f.endswith(".txt"))
    for fn in files:
        for ln in read_lines(os.path.join("kb", fn)):
            t = ln.strip()
            if not t:
                continue
            if t.startswith("#"):
                topic = fn + ":" + t.lstrip("#").strip()
                continue
            if t.startswith("="):
                ws = [w for w in words(t) if w not in stop]
                st = [stem(w) for w in ws]
                for w, s in zip(ws, st):
                    surface.setdefault(s, w)
                for s in st[1:]:
                    merges[s] = st[0]
                continue
            ws = []
            for clause in re.split(r"[,.;:]", t.lower()):
                for wd in words(clause):
                    if (wd in neg and wd != "least") or wd in KB_NEG:
                        break  # the rest of a negated clause is not associated with the fact
                    if wd not in stop:
                        ws.append(wd)
            st = []
            for w in ws:
                s = stem(w)
                surface.setdefault(s, w)
                if s not in st:
                    st.append(s)
            if len(st) >= 2:
                facts.append((topic, st))
                topics.append(topic)

    def root(s):
        seen = 0
        while s in merges and seen < 10:
            s, seen = merges[s], seen + 1
        return s

    facts = [(t, list(dict.fromkeys(root(s) for s in st))) for t, st in facts]
    return stop, neg, merges, root, facts, surface


def compile_codes(facts):
    """Word codes (4 blocks of DH dims, one per association head).

    Every topic (a `#` section of the KB) is placed in one head; topics that share vocabulary are put in different
    heads, so that a word's facts from different topics do not interfere inside one block. Within its head a topic
    gets an orthonormal basis: row 0 is the topic direction, the other rows are the codes of its facts, so facts of
    one topic never interfere. A fact code is ALPHA * topic direction + sqrt(1 - ALPHA^2) * its own row; a word's
    code is the sum of the codes of the facts it occurs in, normalized to length 1."""
    rng = np.random.default_rng(int(SEED))
    topic_names = sorted({t for t, _ in facts})
    for _ in range(len(topic_names) * HEADS):
        rng.standard_normal(DH)  # (kept so that the random stream, hence every code, stays as designed)
    nfacts = {}
    for _, st in facts:
        for s in st:
            nfacts[s] = nfacts.get(s, 0) + 1
    stems = sorted(nfacts)
    sid = {s: i for i, s in enumerate(stems)}
    U = np.zeros((len(stems), HEADS * DH))
    beta = math.sqrt(1 - ALPHA ** 2)
    load = [0] * HEADS
    rng.permutation(len(facts))  # (random stream, see above)
    cnt = {}
    for t, _ in facts:
        cnt[t] = cnt.get(t, 0) + 1
    # topics -> heads: greedy, largest topics first; cost = shared vocabulary with the head's topics (weighted by
    # 1/sqrt(word frequency)) + a penalty for exceeding an even share of facts
    voc = {t: {} for t in topic_names}
    for t, st in facts:
        for s in st:
            voc[t][s] = voc[t].get(s, 0) + 1
    tl = [0] * HEADS
    th = {}
    members = [[] for _ in range(HEADS)]
    mean_load = len(facts) / HEADS
    for t in sorted(topic_names, key=lambda t: (-cnt[t], t)):
        cost = []
        for h in range(HEADS):
            ov = sum(sum(min(voc[t][s], voc[u].get(s, 0)) / math.sqrt(nfacts[s]) for s in voc[t]) for u in members[h])
            cost.append(ov + 2.0 * max(0.0, tl[h] + cnt[t] - mean_load))
        h = int(np.argmin(cost))
        th[t] = h
        members[h].append(t)
        tl[h] += cnt[t]
    basis = {}
    for t in topic_names:
        m = min(cnt[t] + 1, DH)
        Q, _ = np.linalg.qr(rng.standard_normal((DH, m)))
        basis[t] = Q.T  # orthonormal rows; row 0 = topic direction
    seen = {t: 0 for t in topic_names}
    head_of_fact = np.array([th[t] for t, _ in facts])
    for fi, (t, st) in enumerate(facts):
        h = int(head_of_fact[fi])
        load[h] += 1
        r = rng.standard_normal(DH)
        r /= np.linalg.norm(r)
        k = seen[t]
        seen[t] += 1
        if 1 + k < len(basis[t]):  # a topic with more than DH - 1 facts falls back to random codes
            r = basis[t][1 + k]
        c = np.zeros(HEADS * DH)
        c[h * DH:(h + 1) * DH] = ALPHA * basis[t][0] + beta * r
        for s in st:
            U[sid[s]] += c
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
    return stems, sid, U, head_of_fact, load


# ---------------------------------------------------------------- weights
def pm_points(n):
    """n unit vectors (one per position) on a circle; returns them and the scale giving a PM_GAP margin."""
    th = 2 * math.pi * np.arange(n) / n
    p = np.stack([np.cos(th), np.sin(th)], 1)
    g = p @ p.T
    np.fill_diagonal(g, -2)
    gap = 1 - g.max()
    return p, PM_GAP / gap


def build():
    stop, neg, merges, root, facts, surface = load_kb()
    stems, sid, U, head_of_fact, load = compile_codes(facts)
    NF = np.zeros(len(stems))
    for _, st in facts:
        for s in st:
            NF[sid[s]] += 1
    hrng = np.random.default_rng(int(SEED) + 1)
    HSH = hrng.standard_normal((len(stems), NH))
    HSH /= np.linalg.norm(HSH, axis=1, keepdims=True)

    # vocabulary: every inflected form of every stem, plus stop and negation words
    vocab_map = {}
    for s in stems:
        for f in forms(s, surface.get(s, s)):
            vocab_map.setdefault(f, s)
    for s, src in list(surface.items()):
        r = root(s)
        if r in sid:
            for f in forms(s, src):
                vocab_map.setdefault(f, r)
    for f, base in IRREG.items():
        s = root(stem(base))
        if s in sid:
            vocab_map.setdefault(f, s)
    for wd in list(vocab_map):
        if wd in stop or wd in neg:
            del vocab_map[wd]
    vocab = sorted(vocab_map) + sorted(stop | neg)
    vocab = list(dict.fromkeys(v for v in vocab if v and TOKEN.fullmatch(v)))
    if WRITE_VOCAB:
        with open("vocab.txt", "w") as fh:
            fh.write("\n".join(vocab) + "\n")
    # the ids the model receives come from vocab.txt, so it must be exactly the list compiled here
    assert [ln.strip() for ln in read_lines("vocab.txt")] == vocab, "vocab.txt is stale: run python build.py --write-vocab"
    V = len(vocab) + 2

    # ---- embeddings
    K_AMP = (1 + AMP) ** 2 + AMP ** 2 - 1
    E = np.zeros((V, D))
    E[0, STOP] = 1.0
    E[1, STOP] = 1.0  # unknown word: no code
    for i, wd in enumerate(vocab):
        r = i + 2
        if wd in vocab_map:
            j = sid[vocab_map[wd]]
            E[r, CODE0:CODE0 + HEADS * DH] = U[j]
            E[r, HASH0:HASH0 + NH] = HSH[j]
            rho = 0.0
            for h in range(HEADS):
                uh = U[j, h * DH:(h + 1) * DH]
                E[r, SC[h]] = -LAMBDA * float(uh @ uh)
                c = math.exp(LAMBDA * float(uh @ uh))
                rho += c / (c + math.exp(SINK))
            E[r, RHO] = rho  # what the word adds to its own score when the option repeats it
            wq = (1.0 + NF[j]) ** -KEY_IDF  # weight of the word as a question word
            E[r, WA] = math.sqrt(D * STD0 ** 2 * (1.0 / wq ** 2 - 1.0) / K_AMP)
        else:
            E[r, STOP] = 1.0
            if wd in neg:
                E[r, NEGW] = 1.0
    P = np.zeros((Q_LEN + O_LEN, D))
    P[:Q_LEN, ISQ] = PSCALE  # position flags are stored at a large scale too (float32 accuracy)
    P[Q_LEN:, ISO] = PSCALE
    pts, M_pm = pm_points(Q_LEN)
    P[:Q_LEN, PM[0]:PM[1] + 1] = pts * PSCALE
    # pos rows: zero sum and |p|^2 = POS_N, via two dims that embed rows never use
    POS_N = 6 * PSCALE ** 2 + 6.0
    a = -P.sum(1)
    disc = 2 * (POS_N - (P ** 2).sum(1)) - a ** 2
    assert (disc > 0).all()
    z1 = (a + np.sqrt(disc)) / 2
    P[:, Z], P[:, PZ] = z1, a - z1
    # embed rows: zero sum and |e|^2 = D * STD0^2 - POS_N, via BAL and FIL
    S = E.sum(1)
    N = (E ** 2).sum(1)
    C = D * STD0 ** 2 - POS_N
    disc = 4 * S ** 2 - 8 * (S ** 2 + N - C)
    assert (disc > 0).all()
    f = (-2 * S + np.sqrt(disc)) / 4
    E[:, FIL] = f
    E[:, BAL] = -S - f
    assert np.allclose(E.sum(1), 0) and np.allclose((E ** 2).sum(1), C)
    assert np.allclose(P.sum(1), 0) and np.allclose((P ** 2).sum(1), POS_N)
    assert not (np.abs(E).sum(0) * np.abs(P).sum(0)).any()  # disjoint dims, so e.p = 0
    # hence every layer-0 input row has mean 0 and std exactly STD0: LayerNorm is x -> x / STD0

    dt = torch.float64
    w = {"embed": torch.tensor(E, dtype=dt), "pos": torch.tensor(P, dtype=dt)}

    def zeros(*s):
        return torch.zeros(*s, dtype=dt)

    for b in range(2):
        p = f"block{b}."
        for n in ("ln1", "ln2"):
            w[p + n + ".weight"] = torch.ones(D, dtype=dt)
            w[p + n + ".bias"] = zeros(D)
        for n in ("q", "k", "v", "o"):
            w[p + n] = zeros(D, D)
            w[p + n + ".bias"] = zeros(D)
        w[p + "fc1"] = zeros(D, MLP)
        w[p + "fc1.bias"] = zeros(MLP)
        w[p + "fc2"] = zeros(MLP, D)
        w[p + "fc2.bias"] = zeros(D)

    sq = math.sqrt(HD)  # attention divides scores by sqrt(54); queries are multiplied by it
    inv = STD0          # LayerNorm output is x / STD0 (later layers: the added vectors are tiny against STD0)
    c0 = SINK - M_pm

    # ---- layer 0: four association heads (one per code block); the queries that matter are question words.
    # score(j -> k) = LAMBDA u_j.u_k + M_pm p_j.p_k + (SC_j + c0) isQ_k - G stop_k
    # question word j: its own position scores SINK, other question positions are far below, option content
    # words score LAMBDA u_j.u_o, stop words are removed. Value 1 on option words: R = sum_h A_h / (A_h + e^SINK).
    Wq, Wk, Wv, Wo = (w["block0." + n] for n in "qkvo")
    bq = w["block0.q.bias"]
    for h in range(HEADS):
        o = h * HD
        for d in range(DH):
            Wq[CODE0 + h * DH + d, o + d] = LAMBDA * sq * inv
            Wk[CODE0 + h * DH + d, o + d] = inv
        for i, dd in enumerate(PM):
            Wq[dd, o + DH + i] = M_pm * sq * inv / PSCALE
            Wk[dd, o + DH + i] = inv / PSCALE
        Wq[SC[h], o + DH + 2] = sq * inv
        bq[o + DH + 2] = c0 * sq
        Wk[ISQ, o + DH + 2] = inv / PSCALE
        bq[o + DH + 3] = G_MASK * sq
        Wk[STOP, o + DH + 3] = -inv
        Wv[ISO, o] = inv / PSCALE
        Wo[o, R] = 1.0
        Wo[o, Z] = -1.0

    # ---- layer 1 head 0: negation broadcast (1 if the question has a negation word)
    Wq, Wk, Wv, Wo = (w["block1." + n] for n in "qkvo")
    bq = w["block1.q.bias"]
    o = 0
    bq[o + 0] = G_MASK * sq
    Wk[NEGW, o + 0] = inv
    bq[o + 1] = G_MASK * sq
    Wk[ISO, o + 1] = -inv / PSCALE
    Wv[NEGW, o] = inv
    Wo[o, NEG] = 1.0
    Wo[o, Z] = -1.0
    # ---- layer 1 head 1: identity check. Question word j: own position scores 0, option content words score
    # ID_LAMBDA (h_j.h_o - ID_THETA) (about +6 for the same word, below -20 otherwise); ID = 1 if the option repeats j.
    o = HD
    for d in range(NH):
        Wq[HASH0 + d, o + d] = ID_LAMBDA * sq * inv
        Wk[HASH0 + d, o + d] = inv
    for i, dd in enumerate(PM):
        Wq[dd, o + NH + i] = M_pm * sq * inv / PSCALE
        Wk[dd, o + NH + i] = inv / PSCALE
    bq[o + NH + 2] = (0.0 - M_pm - ID_LAMBDA) * sq
    Wk[ISQ, o + NH + 2] = inv / PSCALE
    bq[o + NH + 3] = sq
    Wk[STOP, o + NH + 3] = -G_MASK * inv
    Wk[ISO, o + NH + 3] = -ID_LAMBDA * ID_THETA * inv / PSCALE
    Wv[ISO, o] = inv / PSCALE
    Wo[o, ID] = 1.0
    Wo[o, Z] = -1.0

    # ---- layer 1 MLP, on question content words: F = s (R - RHO [ID]) with s = -1 for a negated question;
    # and the final norm of the token is set to STD0 / weight, so that ln_f multiplies F by the word's weight.
    W1, b1, W2 = w["block1.fc1"], w["block1.fc1.bias"], w["block1.fc2"]
    u = 0
    for term in ("R", "RHO"):
        for sz in (+1, -1):
            for nb in (0, 1):
                W1[R if term == "R" else RHO, u] = sz * inv
                # gate: question content word, and the right negation branch
                W1[ISQ, u] = GATE * inv / PSCALE
                W1[STOP, u] = -GATE * inv
                bias = -GATE
                if nb == 0:
                    W1[NEG, u] = -GATE * inv
                else:
                    W1[NEG, u] = GATE * inv
                    bias -= GATE
                if term == "RHO":
                    W1[ID, u] = 50.0 * inv
                    bias -= 50.0
                b1[u] = bias
                out = sz * (1 if nb == 0 else -1) * (1 if term == "R" else -1)
                W2[u, F] = out
                W2[u, Z] = -out
                u += 1
    for sz in (+1, -1):
        W1[WA, u] = sz * inv
        W2[u, WA] = sz * AMP
        W2[u, WB] = -sz * AMP
        u += 1

    w["ln_f.weight"] = torch.ones(D, dtype=dt)
    w["ln_f.bias"] = zeros(D)
    sc = zeros(D)
    sc[F] = STD0
    w["score"] = sc
    # computed in float64; float32 halves the activation memory and is accurate to ~1e-4 on the score (see NOTES.md)
    return {k: v.float() for k, v in w.items()}


if __name__ == "__main__":
    import sys
    if "--write-vocab" in sys.argv:
        WRITE_VOCAB = True
        build()
        print("wrote vocab.txt")

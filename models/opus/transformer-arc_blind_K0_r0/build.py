"""Hand-wired ARC scorer.  See NOTES.md for the design.

Every number comes from the text files in this folder (kb/*.txt facts, words.txt word lists) through the
fixed formulas below; nothing is trained or fitted.  Run `python build.py` to regenerate vocab.txt.
"""
import math
import zlib
from pathlib import Path

import numpy as np
import torch

D, HEADS, HD, MLP, LQ, LO = 64, 4, 16, 128, 128, 48
HERE = Path(__file__).resolve().parent
SALT = ""                    # fixed; dev tools vary it only to average out the pseudo-random codes

# ---- residual features: coordinates in an orthonormal basis of the zero-sum subspace of R^64, so the
#      LayerNorm mean is always 0 and LayerNorm only rescales by the norm (kept ~C by the CONST feature).
CONST, ISOPT, CONTENT, G, NEG, F1, NN, REC, DUP = range(9)
CF = 9                       # CF .. CF+3: does the word have facts in view h
T0 = 13                      # topic features: NV views of SL dims, one view per layer-0 head
NV, SL = 4, 12
NT = NV * SL
SQ = T0 + NT                 # |P v|^2 for the repeat detector (P = sum of the views)
C = 40.0                     # constant feature: dominates the norm, so LayerNorm is ~ a fixed scale

# ---- design constants (set by hand)
SPARSE = False                # each fact is coded in one view only (else in all views)
BETA = 5.0                   # logit per unit cosine within a view
CENTER = 0.35                # cosine at which an option word counts as much as the question word itself
BIG = 30.0                   # gating logit size
REC_A = 0.06                 # recency weight per question position in the pooling over question words
IMP_A = 0.0                  # question-word importance -IMP_A * log(number of facts), used when pooling
W_MATCH = 10.0               # readout weight of the pooled match
W_DUP = -150.0                # readout weight of exact repeats of question words in the option
G_CURVE = 0.0                # see MLP 0
G_KNEE = 2.0
W_LEN = 5.0                  # readout weight of the option's share of the tokens (a length prior)
BETA2 = 60.0                 # sharpness of the repeat detector
GLOSS_OWN = 0.5              # weight of a gloss line's own code relative to its core words
HUB_N = 4 # a word takes in the codes of at most HUB_N facts (its first ones)
HUB_MIX = 2.5                # later facts of a capped hub get this much of the hub's vector instead
HUB_PURE = False             # capped hubs keep only their own definition lines
HUB_ITERS = 1                # fixed-point iterations of the hub mixing
HUB_HEAD = False             # a word's own lines (where it comes first) define it before other lines
FILE_ORDER = True            # see file_rank
LEMMA = True                 # inflected KB tokens share their stem's facts
FLEN_P = 0.0                 # facts weighted by length ** -FLEN_P
HEADW = 1.0                  # extra weight of a fact for the word that heads its line


def basis():
    m = np.eye(D)[:, :D - 1] - 1.0 / D
    q, _ = np.linalg.qr(np.concatenate([np.ones((D, 1)) / 8, m], 1))
    return q[:, 1:]                                # (64, 63) orthonormal columns, each sums to 0


def read_lists():
    """words.txt sections: [stop] [neg] [prior] [weight] [irregular]."""
    sec, out = None, {"stop": [], "neg": [], "prior": [], "weight": [], "irregular": [], "nolemma": []}
    for line in (HERE / "words.txt").read_text().splitlines():
        line = line.split("#")[0].strip().lower()
        if not line:
            continue
        if line.startswith("["):
            sec = line.strip("[]")
            continue
        out[sec].append(line.split())
    return out


def variants(w):
    """Inflected forms that share the word's vector (over-generation is harmless)."""
    if len(w) < 3 or not w.isalpha():
        return []
    v = {w + "s", w + "es", w + "ed", w + "ing", w + "er", w + "ers", w + "'s", w + "ly"}
    if w.endswith("e"):
        v |= {w[:-1] + "ing", w + "d", w + "r", w + "rs"}
    if w.endswith("y") and w[-2] not in "aeiou":
        v |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ier", w[:-1] + "iest"}
    if w.endswith("f"):
        v |= {w[:-1] + "ves"}
    if w.endswith("fe"):
        v |= {w[:-2] + "ves"}
    if w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou":
        v |= {w + w[-1] + "ed", w + w[-1] + "ing", w + w[-1] + "er"}
    if w.endswith("ate"):
        v |= {w[:-1] + "ion", w[:-1] + "ions"}
    # derived adjectives and nouns: salty saltier, sunny, darkness, rocky
    v |= {w + "y", w + "ier", w + "iest", w + "ness", w + "ful", w + "less"}
    if w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou":
        v |= {w + w[-1] + "y", w + w[-1] + "ier", w + w[-1] + "iest"}
    if w.endswith("e"):
        v |= {w[:-1] + "y"}
    # stems of inflected KB words
    for suf, rep in (("ies", "y"), ("ied", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""),
                     ("ing", "e"), ("ly", ""), ("ness", ""), ("ment", ""), ("ation", "e"), ("ation", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            st = w[:-len(suf)] + rep
            v |= {st, st + "s"}
            if len(st) >= 4 and st[-1] == st[-2]:
                v |= {st[:-1]}
    return sorted(v - {w})


def file_rank(path):
    """Core files first, then their extensions ("...2"), then the relation files; facts listed earlier define words."""
    n = path.stem
    return (2 if n.startswith("relations") else 1 if n[-1].isdigit() else 0, n) if FILE_ORDER else (0, n)


def read_facts():
    facts = []
    for f in sorted((HERE / "kb").glob("*.txt"), key=file_rank):
        fw, gloss = 1.0, False
        for line in f.read_text().splitlines():
            line = line.split("#")[0].strip().lower()
            if not line:
                continue
            if line.startswith("!weight"):
                fw = float(line.split()[1])
                continue
            if line.startswith("!gloss"):
                gloss = True
                continue
            items = []
            for tok in line.split():
                wt = 1.0
                if ":" in tok:
                    tok, s = tok.split(":")
                    wt = float(s)
                items.append((tok, wt * fw))
            facts.append((line, items, gloss))
    return facts


def fact_vector(text):
    """A pseudo-random unit code fixed by the text of the fact: in one view (SPARSE) or in every view."""
    rng = np.random.default_rng(zlib.crc32((SALT + text).encode()))
    r = rng.standard_normal((NV, SL))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    if SPARSE:
        keep = rng.integers(NV)
        r[np.arange(NV) != keep] = 0
    return r


def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-12) * 0.5, 0.0)    # each used view has norm 1/2


def lemma_map(tokens, keep=()):
    """Merge inflected KB tokens into a stem that is itself a KB token (plants -> plant, melting -> melt)."""
    lem = {}
    for t in sorted(tokens, key=len):
        if t in keep:
            continue
        cands = []
        for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
                         ("ied", "y"), ("'s", "")):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                cands.append(t[:-len(suf)] + rep)
        for c in cands:
            if len(c) >= 4 and c[-1] == c[-2] and c[:-1] in tokens:
                c = c[:-1]
            if c in tokens and c != t and not t.endswith("ss"):
                lem[t] = lem.get(c, c)
                break
    return lem


def lexicon():
    lists = read_lists()
    stop = {w for row in lists["stop"] for w in row}
    facts = read_facts()
    keep = {w for row in lists["nolemma"] for w in row}
    lem = lemma_map({w for _, it, _ in facts for w, _ in it if w not in stop}, keep) if LEMMA else {}
    vec, nf = {}, {}
    count = {}
    for text, items, gloss in facts:
        if not gloss:
            for w in {lem.get(w, w) for w, _ in items if w not in stop}:
                count[w] = count.get(w, 0) + 1
    # which facts define each word: the first HUB_N in priority order (lines it heads first, if HUB_HEAD)
    order = {}
    for fi, (text, items, gloss) in enumerate(facts):
        if gloss:
            continue
        ws = [lem.get(w, w) for w, _ in items]
        for j, w in enumerate(ws):
            if w not in stop and ws.index(w) == j:
                order.setdefault(w, []).append(((0 if (HUB_HEAD and j == 0) else 1), fi))
    defining = {w: {fi for _, fi in sorted(v)[:HUB_N]} for w, v in order.items()}
    skipped = []                      # (words of the fact that take its code, hub words that do not)
    for fi, (text, items, gloss) in enumerate(facts):
        if gloss:
            continue
        items = [(lem.get(w, w), wt) for w, wt in items]
        r = fact_vector(text) * len(items) ** -FLEN_P
        seen, hubs = set(), []
        for j, (w, wt) in enumerate(items):
            if w in stop or w in seen:
                continue
            seen.add(w)
            if fi not in defining[w]:
                hubs.append(w)
                continue
            nf[w] = nf.get(w, 0) + 1
            vec[w] = vec.get(w, 0) + wt * (HEADW if j == 0 else 1.0) * r
        if hubs:
            skipped.append(([w for w, _ in items if w in seen and w not in hubs], hubs))
    # a capped hub still relates to the words of its later facts: they take in some of the hub's vector.
    # The hubs' own vectors include such mixes too, so iterate towards a consistent fixed point.
    own = dict(vec)
    hubset = {h for _, hs in skipped for h in hs}
    cur = {w: normalize(v) for w, v in own.items()}
    for _ in range(HUB_ITERS):
        vec = dict(own)
        for others, hubs in skipped:
            mix = HUB_MIX * sum(cur[h] for h in hubs if h in cur) / math.sqrt(len(hubs))
            for w in others:
                if not (HUB_PURE and w in hubset):
                    vec[w] = vec[w] + mix
        cur = {w: normalize(v) for w, v in vec.items()}
    alias = {}
    for row in lists["irregular"]:
        for form in row[1:]:
            alias[form] = row[0]
    for form, base in alias.items():       # irregular forms share facts with their base word
        if base in vec:
            m = vec[base] + vec.get(form, 0)
            vec[base] = vec[form] = m
            nf[base] = nf[form] = nf[base] + nf.get(form, 0)
    # gloss lines: words that are not core KB words get the mean of the line's core words (plus a little
    # of the line's own code); core words are left unchanged, so new vocabulary does not dilute them.
    core = {w: normalize(v) for w, v in vec.items() if np.abs(v).sum() > 0}
    gl = {}
    for text, items, gloss in facts:
        if not gloss:
            continue
        items = [(lem.get(w, w), wt) for w, wt in items if w not in stop]
        cw = [(w, wt) for w, wt in items if w in core]
        r = fact_vector(text)
        for w, wt in items:
            if w in core:
                continue
            g = sum((core[c] * cwt for c, cwt in cw), np.zeros((NV, SL)))
            g = normalize(g) if cw else 0
            gl[w] = gl.get(w, 0) + g + GLOSS_OWN * normalize(r)
            nf[w] = nf.get(w, 0) + 1
    for w, g in gl.items():
        vec[w] = g
    for t, c in lem.items():
        if c in vec:
            vec[t], nf[t] = vec[c], nf[c]
    words = {w: normalize(v) for w, v in vec.items() if np.abs(v).sum() > 0}
    global NFACT
    NFACT = dict(nf)
    final = dict(words)
    for w, v in words.items():           # inflections, unless the form is itself a KB word or a stopword
        for x in variants(w):
            if x not in words and x not in stop:
                final[x] = normalize(final[x] + v) if x in final else v
                NFACT[x] = max(NFACT.get(x, 0), NFACT[w])
    neg = {row[0]: float(row[1]) for row in lists["neg"]}
    prior = {row[0]: float(row[1]) for row in lists["prior"]}
    return final, stop, neg, prior


def make_vocab():
    final, stop, neg, prior = lexicon()
    vocab = sorted(set(final) | stop | set(neg) | set(prior))
    return vocab, final, stop, neg, prior


def build():
    vocab, topic, stop, neg, prior = make_vocab()
    weight = {}
    for row in read_lists()["weight"]:
        for x in row[:-1]:
            weight[x] = float(row[-1])
    have = [l.strip() for l in (HERE / "vocab.txt").read_text().splitlines()]
    assert have == vocab, "vocab.txt is out of date: run python build.py"
    f64 = torch.float64
    B = torch.tensor(basis(), dtype=f64)              # (64, 63)
    V = len(vocab) + 2
    feat = torch.zeros(V, 63, dtype=f64)
    feat[:, CONST] = C
    for i, w in enumerate(vocab):
        r = i + 2
        if w in topic and w not in stop:
            t = topic[w]
            feat[r, T0:T0 + NT] = torch.tensor(t.reshape(-1))
            feat[r, CONTENT] = 1.0
            feat[r, CF:CF + NV] = torch.tensor((np.linalg.norm(t, axis=1) > 1e-9).astype(float))
            feat[r, SQ] = float((t.sum(0) ** 2).sum())
            feat[r, REC] = -IMP_A * math.log(max(NFACT.get(w, 1), 1)) + math.log(weight.get(w, 1.0))
        feat[r, NEG] = neg.get(w, 0.0)
    posf = torch.zeros(LQ + LO, 63, dtype=f64)
    posf[LQ:, ISOPT] = 1.0
    posf[:LQ, REC] = REC_A * torch.arange(LQ, dtype=f64)
    w = {"embed": feat @ B.T, "pos": posf @ B.T}

    def rd(f):          # read vector of feature f in the residual space
        return B[:, f]

    one = rd(CONST) / C          # reads 1 from every token
    sc = 1.0 / math.sqrt(HD)     # arch divides q.k by sqrt(16)

    def mats():
        return [torch.zeros(D, D, dtype=f64) for _ in range(4)] + [torch.zeros(D, dtype=f64) for _ in range(3)]

    # ---- layer 0: each question word q attends, in each view h, over the option's content words
    #   logit = BETA * cos_h(q, o) + BETA * (1 - CENTER)      for option content words
    #   logit = BETA * cos_h(q, q')  (itself: BETA)           for question content words (the sink, value 0)
    #   function words are masked out; if q has nothing in view h it is sent to the sink.
    #   The value is 1 on option content words, so G sums over views a soft count (0..1) of associated words.
    q0, k0, v0, o0, qb0, kb0, vb0 = mats()
    for h in range(HEADS):
        base = h * HD
        for i in range(SL):
            t = T0 + h * SL + i
            q0[:, base + i] = rd(t) * 4 * BETA / sc       # view vectors have norm 1/2
            k0[:, base + i] = rd(t)
        q0[:, base + 13] = BETA * (1 - CENTER) * rd(CF + h) / sc
        k0[:, base + 13] = rd(ISOPT)
        q0[:, base + 14] = BIG * (one - rd(CF + h)) / sc
        k0[:, base + 14] = one - rd(ISOPT)
        qb0[base + 15] = 1.0 / sc
        k0[:, base + 15] = BIG * (rd(CONTENT) - one)
        v0[:, base] = rd(ISOPT) + rd(CONTENT) - one
        o0[base, :] = rd(G)

    # ---- layer 1
    q1, k1, v1, o1, qb1, kb1, vb1 = mats()
    # head 0: recency-weighted mean of G over the question's content words -> F1 (same for every token)
    qb1[0] = 1.0 / sc
    k1[:, 0] = BIG * (rd(CONTENT) - rd(ISOPT))
    qb1[1] = 1.0 / sc
    k1[:, 1] = rd(REC)
    v1[:, 0] = rd(G)
    o1[0, :] = rd(F1)
    # head 1: is there a negation word in the question -> NN
    qb1[16] = 1.0 / sc
    k1[:, 16] = BIG * (0.5 * rd(NEG) - rd(ISOPT))
    v1[:, 16] = rd(NEG)
    o1[16, :] = rd(NN)
    # head 3: exact repeats of a question word in the option -> DUP on that question word.  With P the sum
    #   of the views, logit = BETA2 (2 Pq.Pk - |Pk|^2) = BETA2 (|Pq|^2 - |Pq - Pk|^2): the question word itself
    #   (the sink) and its exact repeats tie, so one repeat gives 1/2.  Option queries and function-word
    #   queries go to the question content words (value 0).
    for i in range(SL):
        q1[:, 48 + i] = sum(rd(T0 + h * SL + i) for h in range(NV)) * 2 * BETA2 / sc
        k1[:, 48 + i] = sum(rd(T0 + h * SL + i) for h in range(NV))
    qb1[62] = 1.0 / sc
    k1[:, 62] = 1000.0 * (rd(CONTENT) - one) - BETA2 * rd(SQ)
    q1[:, 63] = (rd(ISOPT) + one - rd(CONTENT)) / sc
    k1[:, 63] = 400.0 * (one - rd(ISOPT))
    v1[:, 48] = rd(ISOPT) + rd(CONTENT) - one
    o1[48, :] = rd(DUP)

    # ---- MLP 1: F1 <- F1 * (1 - 2 NN)  (relu gate; F1 >= 0 and NN is ~0 or ~1)
    fc1 = torch.zeros(D, MLP, dtype=f64)
    fc2 = torch.zeros(MLP, D, dtype=f64)
    KAP, KOFF = 4.0, 12.0
    fc1[:, 0] = KAP * (rd(F1) + one - KOFF * (one - rd(NN)))
    fc2[0, :] = -2.0 / KAP * rd(F1)

    lnw = torch.full((D,), C / 8, dtype=f64)      # LN(h) = 8 h / |h| * C / 8 ~= h
    z = torch.zeros(D, dtype=f64)
    for p, (q, k, v, o, qb, kb, vb) in {"block0.": (q0, k0, v0, o0, qb0, kb0, vb0),
                                         "block1.": (q1, k1, v1, o1, qb1, kb1, vb1)}.items():
        w[p + "ln1.weight"], w[p + "ln1.bias"] = lnw.clone(), z.clone()
        w[p + "ln2.weight"], w[p + "ln2.bias"] = lnw.clone(), z.clone()
        w[p + "q"], w[p + "k"], w[p + "v"], w[p + "o"] = q, k, v, o
        w[p + "q.bias"], w[p + "k.bias"], w[p + "v.bias"], w[p + "o.bias"] = qb, kb, vb, z.clone()
        w[p + "fc1"] = torch.zeros(D, MLP, dtype=f64)
        w[p + "fc1.bias"] = torch.zeros(MLP, dtype=f64)
        w[p + "fc2"] = torch.zeros(MLP, D, dtype=f64)
        w[p + "fc2.bias"] = z.clone()
    w["block1.fc1"], w["block1.fc2"] = fc1, fc2
    # ---- MLP 0: G <- G + G_CURVE * relu(G - G_KNEE)   (reshapes each question word's match before pooling)
    if G_CURVE:
        f1 = torch.zeros(D, MLP, dtype=f64)
        f2 = torch.zeros(MLP, D, dtype=f64)
        f1[:, 0] = 8.0 * (rd(G) - G_KNEE * one)
        f2[0, :] = G_CURVE / 8.0 * rd(G)
        w["block0.fc1"], w["block0.fc2"] = f1, f2
    w["ln_f.weight"], w["ln_f.bias"] = lnw.clone(), z.clone()
    w["score"] = W_MATCH * rd(F1) + W_DUP * rd(DUP) + W_LEN * rd(ISOPT)
    return {k: v.float().contiguous() for k, v in w.items()}


if __name__ == "__main__":
    vocab = make_vocab()[0]
    (HERE / "vocab.txt").write_text("\n".join(vocab) + "\n")
    print(len(vocab), "words")

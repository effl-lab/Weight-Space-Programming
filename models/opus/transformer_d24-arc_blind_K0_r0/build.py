"""Hand-written weights for the ARC scorer (see NOTES.md for the design).

Every number here is set by a formula from the knowledge file `knowledge.txt` (concept groups written by hand)
and the constants below. Nothing is trained or fitted.
"""
import math
import re

import numpy as np
import torch

D, HEADS, DH, MLP = 24, 4, 6, 48

# ---- logical residual layout (23 dims, all orthogonal to the all-ones vector so LayerNorm is a pure scale) ----
SEM = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 19))]  # concept code per head
G, N, X, S = 19, 20, 21, 22   # G: side+class, N: norm compensation, X: negation flag / n, S: match score
NL = 23

F_SIDE = 3.0                  # G = -F for question tokens, +F for option tokens (from pos)
P_SIDE = 3.0                  # N = +P for question tokens, -P for option tokens (from pos)
C_STOP = math.sqrt(2.0)       # class offset of words without concept codes (keeps every token's norm equal)
BETA = 20.0                   # attention logit for an exact concept match (dot product 1)
FLAG_MU_LAMBDA = 8.0          # strength of the routing term in layer 0
V_SCALE = 0.05                # residual size of the match score
EPS_CONTENT = 0.02            # small credit per known option content word (tie breaker)
NEG_KEY = 16.0                # layer 1: logit of a negation word (per unit of its strength)
NEG_SIDE = 8.0                # layer 1: preference for question-side negation words
V_NEG = 0.25                  # residual size of the negation value n
STARS = True
SEED_HEADS, SEED_CODES = 7, 20261113   # fixed seeds of the random code vectors and head choices
OPTIMIZE = True
ALL_HEADS = True              # every concept has a code in every head
STAR_MIN = 3
PAIR_PENALTY = 2.0
MEMBER_W = 1.0                # weight of a link between two non-key words of a line
CAP = None                    # cap on one option word's summed match score (None: no cap)
KAPPA_U = 0.3                 # residual size of u = Lq / (Lq + Lo)
A_COEF = 20000.0              # size of the norm-dominating component
HUB_MIN = 5

NEGATION = {"except": 1.0, "not": 0.75, "least": 0.85, "cannot": 0.6, "isn't": 0.6, "aren't": 0.6,
            "doesn't": 0.6, "unlikely": 0.6, "false": 0.7, "incorrect": 0.6}
NORM2 = 4.0 + F_SIDE ** 2 + P_SIDE ** 2   # squared norm of every token's layer-0 input
SIGMA0 = math.sqrt(NORM2 / D)

VOWELS = set("aeiou")
KEYS = []
# generic words never used as concept members (they carry no science content)
GENERIC = set("""a an the and or of to in on at by for with from into onto out off over under up down away through
around between before after during while together each both very much many most more less some all no other others same
different own one two three four five can cannot will would may might could should must is are was were be been being has
have had do does did it its this that these those which what who when where why how than then if per unit kind type types
part parts thing things place area way like look looks make makes made making take takes taking use used uses using need
needs help helps new long short high low big small large size increase increases decrease decreases change changes changed
form forms formed move moves moved moving living organisms organism see sees amount total percent years year's thing also
just only even still well back forth such about against along across get gets got put keep kept give gives go goes going
come comes came called call able any every best most likely following true something someone measure measures
measured measuring observe observes observed observing contain contains contained found activities activity""".split())


def basis() -> np.ndarray:
    """24 x 23 orthonormal columns spanning the complement of the all-ones vector, one per logical dim.

    The huge norm component written into N in the last MLP gets its own two coordinates (2 and 3), and the score
    dim S its own pair (0 and 1), so float32 rounding of the huge values never touches the score."""
    cols = np.zeros((D, NL))
    cols[2, N], cols[3, N] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    rest = [i for i in range(D) if i not in (2, 3)]
    g = np.zeros(D)
    g[2] = g[3] = 1.0
    g[rest] = -2.0 / len(rest)
    cols[:, G] = g / np.linalg.norm(g)
    sub = np.zeros((len(rest), len(rest)))
    sub[:, 0] = 1.0
    sub[0, 1], sub[1, 1] = 1.0, -1.0
    sub[:, 2:] = np.eye(len(rest))[:, 2:len(rest)]
    q, _ = np.linalg.qr(sub)
    q = q[:, 1:] * np.sign(q[1, 1] * -1 if False else 1)
    others = [d for d in range(NL) if d not in (N, G, S)]
    full = np.zeros((D, len(rest) - 1))
    full[rest] = q
    cols[:, S] = full[:, 0]
    for j, d in enumerate(others):
        cols[:, d] = full[:, j + 1]
    assert np.allclose(cols.T @ cols, np.eye(NL), atol=1e-9) and np.allclose(cols.sum(0), 0, atol=1e-9)
    return cols


def variants(w: str) -> set:
    """Regular inflections of a word (plural / 3rd person, past, -ing); over-generation is harmless."""
    out = {w}
    if len(w) < 3 or not w.isalpha():
        return out | {w + "'s"}
    # singulars of written plurals (and 3rd person forms)
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    elif w.endswith(("ches", "shes", "sses", "xes", "zes")):
        out.add(w[:-2])
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        out.add(w[:-1])
    if w.endswith(("s", "x", "z", "ch", "sh")):
        out.add(w + "es")
    elif w.endswith("y") and w[-2] not in VOWELS:
        out.add(w[:-1] + "ies")
    else:
        out.add(w + "s")
    if w.endswith("ss") or not w.endswith("s"):
        if w.endswith("e"):
            out |= {w + "d", w[:-1] + "ing"}
        elif w.endswith("y") and w[-2] not in VOWELS:
            out |= {w[:-1] + "ied", w + "ing"}
        else:
            out |= {w + "ed", w + "ing"}
            if len(w) <= 5 and w[-1] not in VOWELS | set("wxy") and w[-2] in VOWELS and w[-3] not in VOWELS:
                out |= {w + w[-1] + "ed", w + w[-1] + "ing"}
    if w.endswith("f"):
        out.add(w[:-1] + "ves")
    if w.endswith("fe"):
        out.add(w[:-2] + "ves")
    if w.endswith("um"):
        out.add(w[:-2] + "a")
    if w.endswith("us"):
        out.add(w[:-2] + "i")
    return out | {v + "'s" for v in out if not v.endswith("s")} | {w + "'"}


def read_concepts(folder: str = "knowledge") -> list:
    """Concept lines. A line "label: a b c" links all its words. A category line "label: key words | members"
    links each key word with every member, but members only through the keys. Sets KEYS[i] to the number of
    key words of category line i (None for ordinary lines)."""
    import glob
    groups = []
    KEYS.clear()
    lines = [l for p in sorted(glob.glob(folder + "/*.txt")) for l in open(p, encoding="utf-8")]
    tok = lambda t: [w for w in re.findall(r"[a-z0-9']+", t.lower())
                     if not (w in GENERIC or w in NEGATION or (len(w) < 2 and not w.isdigit()))]
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        parts = line.split("|", 1)
        ws = []
        for w in tok(parts[0]) + (tok(parts[1]) if len(parts) > 1 else []):
            if w not in ws:
                ws.append(w)
        if len(ws) >= 2:
            groups.append(ws)
            KEYS.append(len([w for i, w in enumerate(tok(parts[0])) if w not in tok(parts[0])[:i]])
                        if len(parts) > 1 else None)
    return groups


def word_slots(groups: list) -> tuple:
    """word -> sequence of 4 concept ids, one per head (before head assignment).

    A word's own concepts are the lines it is listed in, most central first (a word early in a line is central
    to it). A word listed in at least HUB_MIN lines is a hub and also gets a star concept: the hub plus every word
    it shares a line with. Hubs put their star first; other words fill slots their own lines leave free with the
    stars of hubs they share lines with. Inflections inherit the slots of their base word.
    Returns (slots, number of concepts)."""
    ranked, cooc = {}, {}
    for ci, ws in enumerate(groups):
        for pos, w in enumerate(ws):
            ranked.setdefault(w, []).append((pos, len(ws), ci))
            for u in ws:
                if u != w:
                    cooc.setdefault(w, {}).setdefault(u, 0)
                    cooc[w][u] += 1
    own = {w: [c for _, _, c in sorted(r)] for w, r in ranked.items()}
    hubs = sorted(w for w, cs in own.items() if len(cs) >= HUB_MIN) if STARS else []
    star = {h: len(groups) + i for i, h in enumerate(hubs)}
    slots = {}
    for w, cs in own.items():
        st = [star[u] for u, _ in sorted(cooc.get(w, {}).items(), key=lambda t: (-t[1], t[0])) if u in star]
        if w in star:
            seq = [star[w]] + cs[:3]
        elif len(cs) >= 4:
            seq = cs[:4]
        elif len(cs) == 1:
            seq = [cs[0], cs[0]] + st[:2]
        else:
            seq = cs + st[:4 - len(cs)]
        while len(seq) < 4:
            seq = seq + seq[:4 - len(seq)]
        slots[w] = seq
    for w in list(own):
        for v in variants(w):
            if v not in own and v not in slots:
                slots[v] = slots[w]
    return slots, len(groups) + len(hubs)


def optimize_slots(groups: list, passes: int = 5, report: bool = False) -> tuple:
    """Choose, for every listed word and every head, one concept code to hold, so that as many of the written
    links as possible are realized (two words are linked in a head when they hold the same code there).

    Candidate codes are the lines (cliques of the words written together) and the stars of words listed in at
    least STAR_MIN lines (the word plus all words it shares a line with). Every code lives in one head. The
    objective counts, for each pair of words written together w times, w * (1 + 1/2 + 1/4 ...) over the heads
    they share, minus PAIR_PENALTY for each pair that shares a head without ever being written together.
    Deterministic coordinate ascent on this objective; it only uses the knowledge files.
    Returns (word -> 4 code ids or None, number of codes)."""
    rng = np.random.default_rng(SEED_HEADS)
    L = len(groups)
    nb = {}
    lines_of = {}
    neutral = set()
    for ci, ws in enumerate(groups):
        cat = KEYS[ci] if ci < len(KEYS) else None
        nkey = cat if cat is not None else 1
        if cat is None:
            while nkey < len(ws) and ws[nkey][:4] == ws[0][:4]:
                nkey += 1                      # the key term and its written inflections lead the line
        mw = 0.0 if cat is not None else MEMBER_W
        for i, u in enumerate(ws):
            lines_of.setdefault(u, []).append(ci)
            d = nb.setdefault(u, {})
            for j, v in enumerate(ws):
                if v != u:
                    wt = 1.0 if min(i, j) < nkey else mw
                    if wt > 0:
                        d[v] = d.get(v, 0) + wt
                    else:
                        neutral.add((u, v))
    stars = sorted(u for u, ls in lines_of.items() if len(ls) >= STAR_MIN)
    members = [list(ws) for ws in groups] + [[u] + sorted(nb.get(u, {})) for u in stars]
    for u in lines_of:
        nb.setdefault(u, {})
    n = len(members)
    head_of = rng.integers(0, HEADS, n)
    cand = {u: [[] for _ in range(HEADS)] for u in nb}
    for c, ms in enumerate(members):
        for u in ms:
            for h in (range(HEADS) if ALL_HEADS else [head_of[c]]):
                cand[u][h].append(c)
    potential = {}
    for c, ms in enumerate(members):
        for u in ms:
            potential[(u, c)] = sum(nb[u].get(v, 0) for v in ms if v != u)
    holders = {}                               # (code, head) -> insertion-ordered dict of holders
    hold = {u: [None] * HEADS for u in nb}
    order = sorted(nb, key=lambda u: (-sum(nb[u].values()), u))
    for _ in range(passes):
        for u in order:
            for h in range(HEADS):
                old = hold[u][h]
                if old is not None:
                    holders[(old, h)].pop(u, None)
                # number of heads other than h each partner already shares with u
                k = {}
                for h2 in range(HEADS):
                    c2 = hold[u][h2]
                    if h2 != h and c2 is not None:
                        for v in holders[(c2, h2)]:
                            if v != u:
                                k[v] = k.get(v, 0) + 1
                best, best_score = None, 0.0
                for c in cand[u][h]:
                    g = 0.0
                    for v in holders.get((c, h), ()):
                        wv = nb[u].get(v, 0)
                        g += wv * 0.5 ** k.get(v, 0) if wv else (0.0 if (u, v) in neutral else -PAIR_PENALTY)
                    score = g + 0.01 * potential[(u, c)]
                    if score > best_score:
                        best, best_score = c, score
                hold[u][h] = best
                if best is not None:
                    holders.setdefault((best, h), {})[u] = True
    if report:
        tot = sum(sum(d.values()) for d in nb.values()) / 2
        got = 0.0
        false = 0
        for u in nb:
            shared = {}
            for h in range(HEADS):
                c = hold[u][h]
                if c is not None:
                    for v in holders[(c, h)]:
                        if v != u:
                            shared[v] = shared.get(v, 0) + 1
            for v, kk in shared.items():
                if nb[u].get(v):
                    got += nb[u][v] / 2
                else:
                    false += 1
        print(f"codes {n}, realized link weight {got / tot:.3f}, unwritten linked pairs {false // 2}")
    return hold, n


def assign_heads(seq: list, prefs: np.ndarray) -> list:
    """Each slot's concept takes its own most preferred free head, in order."""
    free, out = [0, 1, 2, 3], [None] * HEADS
    for c in seq:
        h = next(h for h in prefs[c] if h in free)
        out[h] = c
        free.remove(h)
    return out


def build_vocab_and_embed():
    groups = read_concepts()
    if OPTIMIZE:
        hold, n_codes = optimize_slots(groups)
        concepts = dict(hold)
        for w in list(hold):
            for v in variants(w):
                if v not in hold and v not in concepts:
                    concepts[v] = hold[w]
    else:
        concepts, n_codes = word_slots(groups)
    words = sorted(set(concepts) | set(NEGATION))
    rng = np.random.default_rng(SEED_CODES)
    codes = []
    for h in range(HEADS):
        c = rng.standard_normal((n_codes + len(words), len(SEM[h])))
        codes.append(c / np.linalg.norm(c, axis=1, keepdims=True))
    prefs = np.array([rng.permutation(HEADS) for _ in range(n_codes)])
    emb = np.zeros((len(words) + 2, NL))
    emb[0, G] = emb[0, N] = C_STOP            # padding (masked anyway)
    emb[1, G] = emb[1, N] = C_STOP            # unknown word: a stop word
    for i, w in enumerate(words):
        row = emb[i + 2]
        if w in NEGATION:
            xi = math.sqrt(2.0) * NEGATION[w]
            c = math.sqrt((4.0 - xi ** 2) / 2.0)
            row[X], row[G], row[N] = xi, c, c
        elif w in concepts:
            held = concepts[w] if OPTIMIZE else assign_heads(concepts[w], prefs)
            for h, c in enumerate(held):
                row[SEM[h]] = codes[h][n_codes + i if c is None else c]   # None: a private code
        else:
            row[G] = row[N] = C_STOP
    return words, emb


def class_point(c: float, side: str) -> np.ndarray:
    """(G, N) of a token of class offset c on a side (layer-0 input, before LayerNorm)."""
    f, p = (-F_SIDE, P_SIDE) if side == "q" else (F_SIDE, -P_SIDE)
    return np.array([f + c, c * F_SIDE / P_SIDE + p])


def build() -> dict:
    words, emb_log = build_vocab_and_embed()
    listed = [line.strip() for line in open("vocab.txt", encoding="utf-8").read().splitlines()]
    if listed != words:
        raise RuntimeError("vocab.txt does not match the knowledge files; run python build.py to rewrite it")
    B = basis()
    T = lambda a: torch.tensor(a, dtype=torch.float32)

    pos_log = np.zeros((176, NL))
    pos_log[:128, G], pos_log[:128, N] = -F_SIDE, P_SIDE
    pos_log[128:, G], pos_log[128:, N] = F_SIDE, -P_SIDE

    w = {"embed": T(emb_log @ B.T), "pos": T(pos_log @ B.T)}
    for i in range(2):
        p = f"block{i}."
        for n in ("ln1", "ln2"):
            w[p + n + ".weight"], w[p + n + ".bias"] = torch.ones(D), torch.zeros(D)

    # ---------------- layer 0: concept matching ----------------
    # x = h / SIGMA0 for every token (equal norms), so in logical units x*SIGMA0 = h.
    s = SIGMA0
    Q0, K0, V0 = (np.zeros((NL, D)) for _ in range(3))
    O0 = np.zeros((D, NL))
    qb0, kb0, vb0 = np.zeros(D), np.zeros(D), np.zeros(D)
    lam_sem = math.sqrt(BETA * math.sqrt(DH)) * s          # q.k/sqrt(6) = BETA * code.code
    # routing: q_flag = mu*(l - l_oc), l = G - 2N - 3X ; k_flag = lam*(-N + 2X)  (in h units)
    ml = FLAG_MU_LAMBDA * math.sqrt(DH)
    mu, lam = math.sqrt(ml), math.sqrt(ml)
    l_oc = F_SIDE + 2 * P_SIDE
    # value: affine in (G, N): 1 on question content, 0 on question stop words, EPS on option content
    pts = [class_point(0.0, "q"), class_point(C_STOP, "q"), class_point(0.0, "o")]
    A = np.array([[a, b, 1.0] for a, b in pts])
    va, vn, vc = np.linalg.solve(A, np.array([1.0, 0.0, EPS_CONTENT]))
    for h in range(HEADS):
        base = h * DH
        for j, d in enumerate(SEM[h]):
            Q0[d, base + j] = lam_sem
            K0[d, base + j] = lam_sem
        fq = base + 5
        Q0[G, fq], Q0[N, fq], Q0[X, fq] = mu * s, -2 * mu * s, -3 * mu * s
        qb0[fq] = -mu * l_oc
        K0[N, fq], K0[X, fq] = -lam * s, 2 * lam * s
        V0[G, base], V0[N, base] = va * s, vn * s
        vb0[base] = vc
        O0[base, S] = V_SCALE
    w["block0.q"], w["block0.k"], w["block0.v"] = T(B @ Q0), T(B @ K0), T(B @ V0)
    w["block0.q.bias"], w["block0.k.bias"], w["block0.v.bias"] = T(qb0), T(kb0), T(vb0)
    w["block0.o"], w["block0.o.bias"] = T(O0 @ B.T), torch.zeros(D)
    # MLP 0: cap each option word's score, S <- min(S, CAP), so a single word (an exact repeat of a question
    # word matches in all four heads) cannot outweigh the rest of the option.
    F1m, b1m, F2m = np.zeros((NL, MLP)), np.zeros(MLP), np.zeros((MLP, NL))
    if CAP is not None:
        k = 200.0
        F1m[S, 0], b1m[0] = k, -k * V_SCALE * CAP / s
        F2m[0, S] = -s / k
    w["block0.fc1"], w["block0.fc1.bias"] = T(B @ F1m), T(b1m)
    w["block0.fc2"], w["block0.fc2.bias"] = T(F2m @ B.T), torch.zeros(D)

    # ---------------- layer 1 ----------------
    # head 0: negation detection, n -> X.  head 1: uniform attention, u = Lq / (Lq + Lo) -> G.
    # sigma at layer 1 is close to SIGMA0 (the score adds at most 4 * V_SCALE to a norm of sqrt(22)).
    Q1, K1, V1 = (np.zeros((NL, D)) for _ in range(3))
    O1 = np.zeros((D, NL))
    qb1, vb1 = np.zeros(D), np.zeros(D)
    qb1[0] = math.sqrt(DH)
    K1[X, 0] = NEG_KEY / math.sqrt(2.0) * s
    K1[G, 0] = -NEG_SIDE * s / F_SIDE
    V1[X, 0] = s / math.sqrt(2.0)
    O1[0, X] = V_NEG
    # G - N is -6 on every question token and +6 on every option token
    V1[G, 6], V1[N, 6], vb1[6] = -s / 12, s / 12, 0.5
    O1[6, G] = KAPPA_U
    w["block1.q"], w["block1.k"], w["block1.v"] = T(B @ Q1), T(B @ K1), T(B @ V1)
    w["block1.q.bias"], w["block1.k.bias"], w["block1.v.bias"] = T(qb1), torch.zeros(D), T(vb1)
    w["block1.o"], w["block1.o.bias"] = T(O1 @ B.T), torch.zeros(D)
    # MLP units 0-3: S <- S * (1 - 2n) using gelu(z) + gelu(-z) ~ sqrt(2/pi) z^2 for small z
    eps = 1.0
    F1 = np.zeros((NL, MLP))
    b1 = np.zeros(MLP)
    F2 = np.zeros((MLP, NL))
    k2 = math.sqrt(2 / math.pi)
    coef = -2.0 * s * s / (V_NEG * 4 * k2 * eps * eps)    # converts x_S*x_X products to the S increment
    for u, (a, b, sg) in enumerate([(1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1)]):
        F1[S, u], F1[X, u] = a * eps, b * eps
        F2[u, S] = sg * coef
    # units 4-5: on option content tokens add a component proportional to u that dominates the final norm, so
    # the final LayerNorm divides by u: the mean over Lq + Lo tokens becomes a plain sum over option tokens / Lq.
    F1[G, 4], F1[N, 4], b1[4] = 1.0, 1.0, 5.0    # x_G + x_N = KAPPA_U * u / sigma on option content
    b1[5] = 5.0
    F2[4, N], F2[5, N] = A_COEF, -A_COEF
    w["block1.fc1"], w["block1.fc1.bias"] = T(B @ F1), T(b1)
    w["block1.fc2"], w["block1.fc2.bias"] = T(F2 @ B.T), torch.zeros(D)

    w["ln_f.weight"], w["ln_f.bias"] = torch.ones(D), torch.zeros(D)
    sc = np.zeros(NL)
    sc[S] = 1000.0
    w["score"] = T(B @ sc)
    return w


def write_vocab() -> None:
    words, _ = build_vocab_and_embed()
    open("vocab.txt", "w").write("\n".join(words) + "\n")


if __name__ == "__main__":
    write_vocab()

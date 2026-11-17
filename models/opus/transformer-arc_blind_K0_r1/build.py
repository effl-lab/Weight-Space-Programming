"""Builds every weight of the ARC scorer from the hand-written knowledge files in knowledge/.

See NOTES.md for the design. Nothing here is trained: the knowledge is a list of word clusters written by hand
(knowledge/*.txt, one cluster per line); every weight below is a closed-form function of those clusters.
"""
import math
import os
import re

import numpy as np
import torch

D, HEADS, DH, MLP = 64, 4, 16, 128
Q_LEN, O_LEN = 128, 48
CD = 13                      # code dims per head slot
C, NEG, PA, IMP, FIL, BAL, S, QC, S2, N, AP, AM = range(52, 64)

R = 12.0                     # norm of every (embedding + position) vector
P = 1.0                      # position flag magnitude
U = 2.5                      # scale of one code slot in the embedding
A = 40.0                     # anchor added after block 0 (keeps later LayerNorms linear)
BETA, THETA = 16.0, 0.56      # attention logit = BETA * (cosine - THETA) for content keys
M = 40.0                     # mask strength
T = 25.0                     # negation-word attention strength
W_S, W_S2 = 0.25, 0.25       # scale of match counts in the residual
CAP_O, CAP_Q = 1.0, 1.0      # per-token cap on match count
W_O, W_Q = 1.0, 1.0          # weight of option-side and question-side matches
SEED = 12345
PER_HEAD = 2
MERGE = True
HUB_CUT = 0
PRI_ALL = True
PRUNE_ROUNDS = 0
PROTECT_PRI = False
MERGE_N = 25
MERGE_TH = 0.5
MERGE_MAX = 14
KEEP_POLICY = "cover"
IMP_POW = 0.5
Q_CAPPED = False             # question side: capped count (True) or specificity * count (False)
EPS_LEN = 0.02               # tiny bonus per option content word (tie break)
NEG_WORDS = ["not", "except", "least", "isn't", "aren't", "doesn't", "don't", "cannot", "can't", "won't"]

TOKEN = re.compile(r"[a-z0-9']+")

STOP = set("""
a an the of in on at to for from by with and or but if then than so as is are was were be been being am do does did
done doing has have had having can could will would shall should may might must which what when where who whom whose
why how that this these those there here it its it's they them their theirs he him his she her hers we us our you your
i me my some any each every either neither other another such very also just only about into onto most more best
better likely following statement statements describe describes described explain explains explained explanation
example examples show shows showed shown called known true correct answer question choice probably possible possibly
generally usually often many much several kind kinds type types way ways something thing things student students
teacher class group someone determine identify happen happens happened would because since while until all both
same also one ones no per part parts form forms take takes taken taking cause causes caused causing during
around process processes make makes made making become becomes became use used uses using help helps helped need needs
needed get gets got give gives given place places area areas amount amounts different certain able unable best
likely due order rather instead mostly mainly usually include includes including contain contains involve involves
result results resulting effect effects affect affects affected change changes changed changing occur occurs
something anything everything nothing still even well like unlike however therefore thus also another
not except least isn't aren't doesn't don't cannot can't won't
""".split())

IRREG = {
    "leaves": "leaf", "wolves": "wolf", "knives": "knife", "halves": "half", "shelves": "shelf", "calves": "calf",
    "mice": "mouse", "teeth": "tooth", "feet": "foot", "children": "child", "geese": "goose", "men": "man",
    "women": "woman", "people": "person", "lenses": "lens", "lens": "lens", "fungi": "fungus", "nuclei": "nucleus",
    "larvae": "larva", "pupae": "pupa", "bacteria": "bacterium", "bacterium": "bacterium", "data": "datum",
    "criteria": "criterion", "phenomena": "phenomenon", "media": "medium", "algae": "alga", "antennae": "antenna",
    "vertebrae": "vertebra", "cacti": "cactus", "stimuli": "stimulus", "axes": "axis", "oases": "oasis",
    "analyses": "analysis", "hypotheses": "hypothesis", "crises": "crisis", "bases": "bas",
    "grew": "grow", "grown": "grow", "ate": "eat", "eaten": "eat", "fell": "fall", "fallen": "fall",
    "froze": "freez", "frozen": "freez", "rose": "ris", "risen": "ris", "ran": "run", "swam": "swim", "flew": "fly",
    "flown": "fly", "began": "begin", "begun": "begin", "broke": "break", "broken": "break", "drank": "drink",
    "gave": "giv", "given": "giv", "made": "mak", "took": "tak", "taken": "tak", "threw": "throw", "thrown": "throw",
    "wore": "wear", "worn": "wear", "blew": "blow", "blown": "blow", "dug": "dig", "caught": "catch",
    "brought": "bring", "built": "build", "held": "hold", "kept": "keep", "lost": "los", "sent": "send",
    "slept": "sleep", "stood": "stand", "struck": "strik", "taught": "teach", "thought": "think", "felt": "feel",
    "fed": "feed", "bred": "breed", "sank": "sink", "sunk": "sink", "shone": "shin", "shrank": "shrink",
    "stung": "sting", "hung": "hang", "laid": "lay", "lain": "lie", "spun": "spin", "wove": "weav", "woven": "weav",
    "bit": "bit", "bitten": "bit", "hid": "hid", "hidden": "hid", "rode": "rid", "ridden": "rid", "drove": "driv",
    "driven": "driv", "wrote": "writ", "written": "writ", "chose": "choos", "chosen": "choos", "stole": "steal",
    "stolen": "steal", "spoke": "speak", "spoken": "speak", "sang": "sing", "sung": "sing", "rang": "ring",
    "rung": "ring", "swung": "swing", "clung": "cling", "dove": "div", "led": "lead", "bled": "bleed", "fled": "flee",
    "fought": "fight", "sought": "seek", "bought": "buy", "lit": "light", "slid": "slid", "shot": "shoot",
    "fought": "fight", "underwent": "undergo", "undergone": "undergo", "went": "go", "gone": "go",
    "saw": "see", "seen": "see", "knew": "know", "drew": "draw", "drawn": "draw", "became": "becom",
    "dying": "die", "died": "die", "dies": "die", "lying": "lie", "tied": "tie", "ties": "tie",
}


def stem(w: str) -> str:
    s = _stem(w)
    return FAM.get(s, s)


def _stem(w: str) -> str:
    if w in IRREG:
        return IRREG[w]
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("es") and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "aeioulsz":
            w = w[:-1]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "aeioulsz":
            w = w[:-1]
    if len(w) > 3 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]
    return w


FAM = {}
FAM_WORDS = []
if os.path.exists("families.txt"):
    for _line in open("families.txt"):
        _ws = TOKEN.findall(_line.lower())
        if len(_ws) >= 2:
            _c = _stem(_ws[0])
            for _w in _ws:
                FAM.setdefault(_stem(_w), _c)
                FAM_WORDS.append(_w)


def forms(w: str) -> set:
    """Surface forms that may appear in text for a written word (kept only if they stem back to it)."""
    c = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "'s", w + "s'"}
    if w.endswith("y"):
        c |= {w[:-1] + "ies", w[:-1] + "ied"}
    if w.endswith("e"):
        c |= {w[:-1] + "ing"}
    if len(w) >= 3 and w[-1] not in "aeiouwxy":
        c |= {w + w[-1] + "ed", w + w[-1] + "ing"}
    s = stem(w)
    return {f for f in c if stem(f) == s}


def read_clusters():
    clusters, topics = [], []
    topic = "misc"
    for fn in sorted(os.listdir("knowledge")):
        if not fn.endswith(".txt"):
            continue
        for line in open(os.path.join("knowledge", fn)):
            line = line.strip().lower()
            if not line:
                continue
            if line.startswith("#"):
                topic = fn + ":" + line.lstrip("#").strip()
                continue
            pri = line.startswith("!")
            line = line.lstrip("!")
            ws = [w for w in TOKEN.findall(line) if w not in STOP and not w.isdigit()]
            stems, surf = [], []
            for w in ws:
                s = stem(w)
                if s not in stems:
                    stems.append(s)
                surf.append(w)
            if len(stems) >= 2:
                clusters.append((stems, surf))
                topics.append(topic + ("!" + stems[0] if pri else ""))
    return clusters, topics


def spread_codes(n, Wt, rng, iters=1500):
    """n unit vectors in CD dims whose weighted pairwise |cosines| are pushed down (a fixed design computation)."""
    X = rng.standard_normal((n, CD))
    X -= 0.5 * X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.fill_diagonal(Wt, 0)
    for t in range(iters):
        G = (X @ X.T) * Wt
        flat = np.abs(G).ravel()
        tau = np.partition(flat, int(0.97 * len(flat)))[int(0.97 * len(flat))]
        Mx = np.where(np.abs(G) > tau, G - np.sign(G) * tau, 0)
        X = X - 0.1 * Mx @ X / Wt.max()
        s = X.sum(1, keepdims=True)
        X -= 0.02 * np.where(np.abs(s) > 0.5, s, 0) / CD
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    for _ in range(50):
        s = X.sum(1, keepdims=True)
        X -= np.where(np.abs(s) > 0.5, s - np.sign(s) * 0.5, 0) / CD
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def merge_clusters(clusters, topics, thresh=0.5, max_size=14):
    if not MERGE:
        return clusters, topics
    """Merge clusters that say nearly the same thing (overlap weighted by word specificity)."""
    for _ in range(6):
        n = {}
        for st, _s in clusters:
            for s in st:
                n[s] = n.get(s, 0) + 1
        wt = {s: 1.0 / math.log(1.0 + c) ** 1.5 for s, c in n.items()}
        inv = {}
        for ci, (st, _s) in enumerate(clusters):
            if PROTECT_PRI and "!" in topics[ci]:
                continue
            for s in st:
                if n[s] <= 40:
                    inv.setdefault(s, []).append(ci)
        cand = {}
        for s, cis in inv.items():
            for i in range(len(cis)):
                for j in range(i + 1, len(cis)):
                    cand[(cis[i], cis[j])] = 1
        scored = []
        for a, b in cand:
            A, B = set(clusters[a][0]), set(clusters[b][0])
            if len(A | B) > max_size or sum(1 for s in A & B if n[s] <= MERGE_N) < 2:
                continue
            ov = sum(wt[s] for s in sorted(A & B)) / min(sum(wt[s] for s in sorted(A)), sum(wt[s] for s in sorted(B)))
            if ov >= thresh:
                scored.append((ov, a, b))
        if not scored:
            break
        scored.sort(reverse=True)
        used, merged = set(), []
        for ov, a, b in scored:
            if a in used or b in used:
                continue
            used |= {a, b}
            st = list(clusters[a][0]) + [s for s in clusters[b][0] if s not in clusters[a][0]]
            pt = topics[a] if "!" in topics[a] else (topics[b] if "!" in topics[b] else topics[a])
            merged.append(((st, clusters[a][1] + clusters[b][1]), pt))
        rest = [(c, t) for i, (c, t) in enumerate(zip(clusters, topics)) if i not in used]
        clusters = [c for c, t in rest] + [c for c, t in merged]
        topics = [t for c, t in rest] + [t for c, t in merged]
    return clusters, topics


def design():
    """Clusters -> per-head unit codes for every stem.  Returns (vocab list, stem of each vocab word, codes)."""
    clusters, topics = read_clusters()
    if HUB_CUT:
        nr = {}
        for st, _s in clusters:
            for s in st:
                nr[s] = nr.get(s, 0) + 1
        hubs = {s for s, c in nr.items() if c > HUB_CUT}
        cut = []
        for (st, sf), t in zip(clusters, topics):
            if "!" not in t:
                keep_i = [i for i, s in enumerate(st) if s not in hubs]
                st = [st[i] for i in keep_i]
            cut.append(((st, sf), t))
        clusters = [c for c, t in cut if len(c[0]) >= 2]
        topics = [t for c, t in cut if len(c[0]) >= 2]
    clusters, topics = merge_clusters(clusters, topics, MERGE_TH, MERGE_MAX)
    rng = np.random.default_rng(SEED)
    nc = len(clusters)
    size = np.array([len(c[0]) for c in clusters])
    members = {}
    for ci, (stems, _) in enumerate(clusters):
        for s in stems:
            members.setdefault(s, []).append(ci)
    dead = set()
    all_members = members
    for _round in range(PRUNE_ROUNDS + 1):
        members = {s: [c for c in cis if c not in dead] for s, cis in all_members.items()}
        members = {s: cis for s, cis in members.items() if cis}
        # which clusters each stem keeps (at most 8): greedy cover of distinct partners, rare partners first
        nmem = {s: len(c) for s, c in members.items()}
        want = {}
        for s, cis in members.items():
            if len(cis) <= PER_HEAD * HEADS:
                want[s] = list(cis)
                continue
            chosen, covered = [], set()
            left = list(cis)
            for ci in cis:
                if (("!" in topics[ci]) if PRI_ALL else topics[ci].endswith("!" + s)) and len(chosen) < PER_HEAD * HEADS:
                    chosen.append(ci)
                    covered |= set(clusters[ci][0])
                    left.remove(ci)
            while left and len(chosen) < PER_HEAD * HEADS:
                def val(ci):
                    if KEEP_POLICY == "small":
                        return -size[ci]
                    if KEEP_POLICY == "subject":
                        return -size[ci] + (100 if clusters[ci][0][0] == s else 0)
                    return sum(1.0 / math.sqrt(nmem[x]) for x in clusters[ci][0] if x != s and x not in covered) \
                        / size[ci] ** 0.3
                best = max(left, key=lambda ci: (val(ci), -ci))
                chosen.append(best)
                covered |= set(clusters[best][0])
                left.remove(best)
            want[s] = chosen
        def prio(s, ci):
            return want[s].index(ci) if ci in want[s] else 99
        wanted_by = [[] for _ in range(nc)]
        for s, cis in want.items():
            for ci in cis:
                wanted_by[ci].append(s)
        # assign clusters to heads so that a word's kept clusters spread over the heads
        load = {s: np.zeros(HEADS) for s in members}
        hload = np.zeros(HEADS)
        head = np.zeros(nc, dtype=int)
        order = sorted(range(nc), key=lambda ci: size[ci])
        for ci in order:
            cost = np.zeros(HEADS)
            for s in wanted_by[ci]:
                l = load[s]
                cost += np.where(l == 0, 0.0, np.where(l == 1, 1.0, 8.0 * l))
            cost += 1e-3 * hload
            h = int(np.argmin(cost))
            head[ci] = h
            hload[h] += 1
            for s in wanted_by[ci]:
                load[s][h] += 1
        keep = {}
        for s, cis in want.items():
            k = []
            for h in range(HEADS):
                k += sorted([ci for ci in cis if head[ci] == h], key=lambda ci: prio(s, ci))[:PER_HEAD]
            keep[s] = k
        cnt = {}
        for s, k in keep.items():
            for c in k:
                cnt[c] = cnt.get(c, 0) + 1
        new_dead = {c for c in range(nc) if c not in dead and cnt.get(c, 0) < 2}
        if not new_dead or _round == PRUNE_ROUNDS:
            break
        dead |= new_dead
    for s in all_members:
        keep.setdefault(s, [])
    keep = {s: [c for c in k if cnt.get(c, 0) >= 2 or not PRUNE_ROUNDS] for s, k in keep.items()}
    live = set(c for k in keep.values() for c in k)
    # codes: per head, unit vectors with low mutual coherence (pairs that share a word pushed hardest)
    codes = np.zeros((nc, CD))
    for h in range(HEADS):
        idx = [c for c in range(nc) if head[c] == h and c in live]
        pos_of = {c: i for i, c in enumerate(idx)}
        Wt = np.ones((len(idx), len(idx)))
        tid = {t: i for i, t in enumerate(sorted(set(topics)))}
        tp = np.array([tid[topics[c]] for c in idx])
        Wt[tp[:, None] == tp[None, :]] = 2.0
        for s, k in keep.items():
            kh = [pos_of[c] for c in k if head[c] == h]
            if len(kh) == 2:
                Wt[kh[0], kh[1]] = Wt[kh[1], kh[0]] = 4.0
        codes[idx] = spread_codes(len(idx), Wt, rng)
    stem_code = {}
    for s, k in keep.items():
        if not k:
            continue
        v = np.zeros((HEADS, CD))
        for ci in k:
            v[head[ci]] += codes[ci]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(n > 0, v / np.maximum(n, 1e-9), 0.0)
        stem_code[s] = v
    vocab = set()
    for stems, surf in clusters:
        for w in surf:
            vocab |= forms(w)
    for w in list(IRREG) + FAM_WORDS:
        if stem(w) in stem_code:
            vocab |= forms(w)
    vocab = sorted(v for v in vocab if stem(v) in stem_code and v not in STOP)
    vocab += [w for w in NEG_WORDS if w not in vocab]
    raw, _ = read_clusters()
    nraw = {}
    for st, _s in raw:
        for s in st:
            nraw[s] = nraw.get(s, 0) + 1
    return vocab, stem_code, (clusters, head, keep, {s: nraw.get(s, 1) for s in stem_code})


def write_vocab():
    vocab, _, _ = design()
    with open("vocab.txt", "w") as f:
        f.write("\n".join(vocab) + "\n")


def gelu_np(x):
    return x * 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def build():
    vocab_file = [l.strip() for l in open("vocab.txt").read().splitlines()]
    _, stem_code, (_, _, _, nraw) = design()
    imp = {s: n ** -IMP_POW for s, n in nraw.items()}
    V = len(vocab_file) + 2
    E = np.zeros((V, D))
    for i, w in enumerate(vocab_file):
        e = np.zeros(D)
        if w in NEG_WORDS:
            e[NEG] = 1.0
        else:
            s = stem(w)
            if s in stem_code and w not in STOP:
                code = stem_code[s]
                for h in range(HEADS):
                    e[CD * h: CD * (h + 1)] = U * code[h]
                e[C] = 1.0
                e[IMP] = imp[s]
        E[i + 2] = e
    # filler / balance: every word vector has zero sum and norm^2 = R^2 - P^2
    target = R * R - P * P
    for i in range(V):
        e = E[i]
        s_ = e.sum()
        m = target - (e * e).sum()
        disc = 2 * m - s_ * s_
        assert disc > 0, (i, s_, m)
        a = (-s_ + math.sqrt(disc)) / 2
        b = -s_ - a
        e[FIL], e[BAL] = a, b
    pos = np.zeros((Q_LEN + O_LEN, D))
    pos[:Q_LEN, PA] = -P
    pos[Q_LEN:, PA] = P

    w = {}
    l0 = 8.0 / R                             # LayerNorm scale in block 0
    l1 = 8.0 / math.sqrt(R * R + 2 * A * A)  # LayerNorm scale in block 1 and the final one
    for b in range(2):
        lam = l0 if b == 0 else l1
        Wq, Wk, Wv, Wo = (np.zeros((D, D)) for _ in range(4))
        bq, bk, bv, bo = (np.zeros(D) for _ in range(4))
        bscale = 4 * BETA / (lam * lam * U * U)
        for h in range(HEADS):
            o = DH * h
            for j in range(CD):
                Wq[CD * h + j, o + j] = 1.0
                Wk[CD * h + j, o + j] = bscale
            # q coord 13 = 1, 14 = isQ, 15 = isO ; isO = (x_PA/(lam P) + 1) / 2
            bq[o + 13] = 1.0
            Wq[PA, o + 14], bq[o + 14] = -1 / (2 * lam * P), 0.5
            Wq[PA, o + 15], bq[o + 15] = 1 / (2 * lam * P), 0.5
            # key coord 13: content offset (all queries)
            Wk[C, o + 13] = -4 * BETA * THETA / lam
            if b == 0:
                # options and negation words are masked for everyone
                Wk[PA, o + 13] += -4 * M / (2 * lam * P)
                bk[o + 13] += -4 * M / 2
                Wk[NEG, o + 13] += -4 * M / lam
                if h == 0:   # question tokens look for negation words
                    Wk[NEG, o + 14] = 4 * (T + M) / lam
            else:
                # question queries skip question content keys; option queries skip option keys
                Wk[QC, o + 14] = -4 * M / lam
                Wk[PA, o + 15] = -4 * M / (2 * lam * P)
                bk[o + 15] = -4 * M / 2
                if h == 0:   # option tokens look for negation words
                    Wk[NEG, o + 15] += 4 * T / lam
            if b == 0:
                Wv[IMP, o + 0] = 1 / lam
            else:          # option content only (C - QC): question queries count matched option words
                Wv[C, o + 0] = 1 / lam
                Wv[QC, o + 0] = -1 / lam
            if h == 0:
                Wv[NEG, o + 1] = 1 / lam
            tgt, ws = (S, W_S) if b == 0 else (S2, W_S2)
            Wo[o + 0, tgt] += ws
            Wo[o + 0, AP] -= ws / 2
            Wo[o + 0, AM] -= ws / 2
            if h == 0:
                Wo[o + 1, N] += 1.0
                Wo[o + 1, AP] -= 0.5
                Wo[o + 1, AM] -= 0.5
        p = f"block{b}."
        w[p + "q"], w[p + "k"], w[p + "v"], w[p + "o"] = Wq, Wk, Wv, Wo
        w[p + "q.bias"], w[p + "k.bias"], w[p + "v.bias"], w[p + "o.bias"] = bq, bk, bv, bo
        w[p + "ln1.weight"], w[p + "ln1.bias"] = np.ones(D), np.zeros(D)
        w[p + "ln2.weight"], w[p + "ln2.bias"] = np.ones(D), np.zeros(D)

    # ---- block 0 MLP: QC = question AND content ; anchor
    f1, b1, f2, b2 = np.zeros((D, MLP)), np.zeros(MLP), np.zeros((MLP, D)), np.zeros(D)
    lam = l0
    al = 10.0
    f1[C, 0] = al / lam
    f1[PA, 0] = -al / (2 * lam * P)
    b1[0] = al * 0.5 - 1.5 * al
    g5 = float(gelu_np(np.array(5.0)))
    f2[0, QC] = 1 / g5
    f2[0, AP] = f2[0, AM] = -0.5 / g5
    b2[AP], b2[AM] = A, -A
    w["block0.fc1"], w["block0.fc1.bias"], w["block0.fc2"], w["block0.fc2.bias"] = f1, b1, f2, b2

    # ---- block 1 MLP: final per-token score written into S2 (replacing it)
    f1, b1, f2, b2 = np.zeros((D, MLP)), np.zeros(MLP), np.zeros((MLP, D)), np.zeros(D)
    lam = l1
    G, G2, K, C0 = 60.0, 30.0, 10.0, 5.0
    col = [0]

    def neuron(coefs, bias, out):
        """pre-activation = sum coef * raw residual dim + bias; out: dict dim -> weight of GELU output."""
        j = col[0]
        col[0] += 1
        for d, c in coefs.items():
            f1[d, j] += c / lam
        b1[j] += bias
        tot = 0.0
        for d, c in out.items():
            f2[j, d] += c
            tot += c
        f2[j, AP] -= tot / 2
        f2[j, AM] -= tot / 2

    # isO = (PA/P + 1)/2 ; gate for tok: G*(g - 1)
    def gate_terms(tok, sign, g):
        coefs, bias = {}, 0.0
        if tok == "O":    # G*(isO - 1) = G*(PA/(2P) - 1/2)
            coefs[PA] = g / (2 * P)
            bias += -g / 2
        else:             # G*(isQ - 1) = G*(-PA/(2P) - 1/2)
            coefs[PA] = -g / (2 * P)
            bias += -g / 2
        if sign > 0:
            coefs[N] = coefs.get(N, 0) - G2
        else:
            coefs[N] = coefs.get(N, 0) + G2
            bias += -G2
        return coefs, bias

    for tok, src, ws, cap, wt in (("O", S, W_S, CAP_O, W_O),) + ((("Q", S2, W_S2, CAP_Q, W_Q),) if Q_CAPPED else ()):
        for sign in (1, -1):
            gc, gb = gate_terms(tok, sign, G)
            c1 = dict(gc)
            c1[src] = c1.get(src, 0) + 1 / ws
            neuron(c1, gb + C0, {S2: sign * wt})
            neuron(dict(gc), gb + C0, {S2: -sign * wt})
            c2 = {d: v for d, v in gc.items()}
            c2[src] = c2.get(src, 0) + K / ws
            neuron(c2, gb - K * cap, {S2: -sign * wt / K})
    if not Q_CAPPED:
        # question tokens: specificity * match count, product from GELU's quadratic part
        eps = 0.06
        k2 = 1.0 / (4 * 2 * 0.3989422804 * eps * eps)
        for sign in (1, -1):
            gc, gb = gate_terms("Q", sign, G)
            gc.pop(PA)
            gb += G / 2       # no position gate needed (S2 is zero on option tokens)
            for sx, out in ((1, 1), (-1, -1)):
                for so in (1, -1):
                    c = dict(gc)
                    c[S2] = c.get(S2, 0) + so * eps / W_S2
                    c[IMP] = c.get(IMP, 0) + so * sx * eps
                    neuron(c, gb, {S2: sign * W_Q * out * k2})
    # remove the old S2 value (pass-through)
    neuron({S2: 1.0}, C0, {S2: -1.0})
    neuron({}, C0, {S2: 1.0})
    # tiny bonus per option content word
    gc, gb = gate_terms("O", 1, G)
    gc.pop(N)
    gc[C] = G
    neuron(gc, gb - G + C0, {S2: EPS_LEN / C0})
    w["block1.fc1"], w["block1.fc1.bias"], w["block1.fc2"], w["block1.fc2.bias"] = f1, b1, f2, b2

    w["embed"] = E
    w["pos"] = pos
    w["ln_f.weight"], w["ln_f.bias"] = np.ones(D), np.zeros(D)
    sc = np.zeros(D)
    sc[S2] = 1.0 / l1
    w["score"] = sc
    # every write into the residual sums to zero except the position flag PA, so LayerNorm subtracts PA/64 from
    # every dim; fold that constant shift back into the matrices that read LayerNorm outputs
    for name in ["block0.q", "block0.k", "block0.v", "block0.fc1", "block1.q", "block1.k", "block1.v", "block1.fc1"]:
        w[name][PA, :] += w[name].sum(0) / 63.0
    w["score"][PA] += w["score"].sum() / 63.0
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()}


if __name__ == "__main__":
    write_vocab()

"""Reads the hand-written knowledge (kb/*.txt, generic.txt, negation.txt) and turns it into word vectors."""
import os
import re

import numpy as np

D = 24
HEAD_DIMS = [list(range(4, 9)), list(range(9, 14)), list(range(14, 19)), list(range(19, 24))]
HERE = os.path.dirname(os.path.abspath(__file__))


def _read_generic():
    out = set()
    path = os.path.join(HERE, "generic.txt")
    for line in open(path):
        out |= set(re.findall(r"[a-z0-9']+", line.split("#")[0].lower()))
    return out


STOP = _read_generic()


def _read_cuestop():
    out = set()
    path = os.path.join(HERE, "cuestop.txt")
    if os.path.exists(path) and os.environ.get("ARC_CUESTOP", "1") == "1":
        for line in open(path):
            out |= set(re.findall(r"[a-z0-9']+", line.split("#")[0].lower()))
    return out


CUESTOP = _read_cuestop()


def singular(w):
    """Likely singular forms of a plural-looking word."""
    out = set()
    if len(w) > 3 and w.endswith("ies"):
        out.add(w[:-3] + "y")
    if len(w) > 3 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        out.add(w[:-2])
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        out.add(w[:-1])
    return out


def inflect(w):
    """Regular inflections of a base word (over-generation is harmless)."""
    out = {w} | singular(w)
    if len(w) < 3 or not w.isalpha() or w.endswith(("ed", "ing")) or (w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3):
        return out
    if w.endswith(("s", "x", "z", "ch", "sh")):
        out.add(w + "es")
    elif w.endswith("y") and w[-2] not in "aeiou":
        out.update({w[:-1] + "ies", w[:-1] + "ied"})
    else:
        out.add(w + "s")
    if w.endswith("e"):
        out.update({w + "d", w[:-1] + "ing"})
    elif not w.endswith("y") or w[-2] in "aeiou":
        out.update({w + "ed", w + "ing"})
    else:
        out.add(w + "ing")
    if len(w) <= 5 and w[-1] in "bdgmnprt" and w[-2] in "aeiou" and w[-3] not in "aeiou":
        out.update({w + w[-1] + "ed", w + w[-1] + "ing"})
    return out


MAX_GROUPS = int(os.environ.get("ARC_MG", 4))         # a hub word (hubs.txt) keeps at most this many groups (one code per head)
MAX_GROUPS_PLAIN = int(os.environ.get("ARC_MGP", 3))   # any other word
MERGE_HEAD = os.environ.get("ARC_MERGE", "1") == "1"   # lines whose first entity word is the same are one group


def read_groups():
    """kb/*.txt: '@ domain' lines set the domain; every other line is one group of words that point to each
    other.  Words before ':' are the group's entity, words after it its cues.  A word written =word is not
    inflected, !word is kept even if generic.  '#' starts a comment.

    A word that would sit in more than MAX_GROUPS groups keeps the groups where it is the entity first, then the
    smallest (most specific) ones."""
    raw = []
    kbdir = os.path.join(HERE, "kb")
    for name in sorted(os.listdir(kbdir)):
        if not name.endswith(".txt"):
            continue
        domain = name[:-4]
        for line in open(os.path.join(kbdir, name)):
            line = line.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("@"):
                domain = line[1:].strip()
                continue
            ent_part, _, cue_part = line.partition(":") if ":" in line else ("", "", line)
            roles = {}
            for part, role in ((cue_part, 1), (ent_part, 0)):
                for tok in re.findall(r"[=!]*[a-z0-9'\-.]+", part.lower()):
                    keep = "!" in tok
                    exact = "=" in tok
                    tok = tok.strip("=!.")
                    if not tok:
                        continue
                    if "-" in tok:          # the tokenizer splits on '-'; the parts alone are not this word
                        continue
                    elif "." in tok:
                        forms = {t for t in tok.split(".") if t}
                    else:
                        forms = {tok} if exact else inflect(tok)
                    for f in forms:
                        if keep or (f not in STOP and tok not in STOP and not (role == 1 and tok in CUESTOP)):
                            roles[f] = min(role, roles.get(f, 9))
            if roles and min(roles.values()) == 1:      # no ':' -> the first word is the entity
                first = re.findall(r"[a-z0-9']+", cue_part.lower())[0]
                for f in inflect(first):
                    if f in roles:
                        roles[f] = 0
            head_tok = re.findall(r"[a-z0-9']+", (ent_part or cue_part).lower())
            if len(roles) >= 2 and min(roles.values()) == 0 and max(roles.values()) == 1:
                raw.append((domain, roles, head_tok[0] if head_tok else ""))
    # groups with the same entity words are one group (the entity matches every cue of both): saves slots
    merged, index = [], {}
    for dom, roles, head_tok in raw:
        key = head_tok if MERGE_HEAD else frozenset(w for w, r in roles.items() if r == 0)
        if key in index:
            tgt = merged[index[key]][1]
            for w, r in roles.items():
                tgt[w] = min(r, tgt.get(w, 9))
        else:
            index[key] = len(merged)
            merged.append((dom, dict(roles)))
    raw = merged
    member = {}
    for gi, (_, roles) in enumerate(raw):
        for w, role in roles.items():
            member.setdefault(w, []).append((role, len(roles), gi))
    hubs = {}
    for line in open(os.path.join(HERE, "hubs.txt")):
        line = line.split("#")[0]
        if ":" in line:
            w, rest = line.split(":", 1)
            hubs[w.strip()] = [x.strip() for x in rest.split("|") if x.strip()]
    ents = [{w for w, r in roles.items() if r == 0} for _, roles in raw]

    def rank(w, t):
        role, size, gi = t
        pref = hubs.get(w) or next((hubs[x] for x in sorted(singular(w)) if x in hubs), [])
        hit = next((i for i, e in enumerate(pref) if inflect(e) & ents[gi]), len(pref) + 1)
        return (0 if role == 0 and not pref else hit, size, role, gi)

    def limit(w):
        return MAX_GROUPS if (w in hubs or any(x in hubs for x in singular(w))) else MAX_GROUPS_PLAIN
    keep = {w: {t[2] for t in sorted(m, key=lambda t: rank(w, t))[:limit(w)]} for w, m in member.items()}
    groups = []
    for gi, (dom, roles) in enumerate(raw):
        ws = sorted(w for w in roles if gi in keep[w])
        rl = {w: roles[w] for w in ws}
        if len(ws) >= 2 and min(rl.values()) == 0 and max(rl.values()) == 1:
            groups.append((dom, ws, rl))
    return groups


MAX_COS = 0.93          # codes in one head are at least this far apart (cosine); a pair this close does not match
SELF_MAX = 0.5          # a code and its mirror image are at most this similar (so a word never matches itself)


def mirror(h):
    """Each head's keys see codes through R = diag(+1, +1, -1, ...): entity code u, cue code R u."""
    n = len(HEAD_DIMS[h])
    return np.array([1.0, 1.0] + [-1.0] * (n - 2))


def codebook():
    """(head, u, R u) code pairs: a deterministic greedy spherical packing in each head's subspace that keeps
    both u and its mirror R u at least MAX_COS apart from every other code."""
    rng = np.random.default_rng(int(os.environ.get("ARC_SEED", 12345)))
    codes = []
    for h, dims in enumerate(HEAD_DIMS):
        n = len(dims)
        R = mirror(h)
        pts = np.zeros((0, n))
        cand = rng.standard_normal((60000, n))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        for c in cand:
            m = R * c
            if c @ m > SELF_MAX:
                continue
            if len(pts) and (np.max(pts @ c) >= MAX_COS or np.max(pts @ m) >= MAX_COS):
                continue
            pts = np.vstack([pts, c, m])
            codes.append((h, c, m))
    return codes


def super_domain(dom):
    for key, sup in (("astro", "space"), ("rock", "earth"), ("weather", "earth"), ("plate", "earth"),
                     ("water", "earth"), ("time", "earth"), ("climate", "earth"), ("earth", "earth"),
                     ("plant", "life"), ("animal", "life"), ("cell", "life"), ("body", "life"),
                     ("genet", "life"), ("ecolog", "life"), ("resource", "life"), ("bio", "life"),
                     ("matter", "phys"), ("energy", "phys"), ("heat", "phys"), ("electr", "phys"),
                     ("force", "phys"), ("machine", "phys"), ("wave", "phys"), ("method", "method"),
                     ("tool", "method"), ("unit", "method"), ("safety", "method")):
        if key in dom:
            return sup
    return dom


def assign_heads(groups, sharing, cap):
    """4-colour the groups so that groups sharing a word get different heads (DSatur, then min-conflicts)."""
    n = len(groups)
    rng = np.random.default_rng(7)
    head = [-1] * n
    load = np.zeros(len(cap))
    order = sorted(range(n), key=lambda g: -len(sharing[g]))
    for _ in range(n):                      # DSatur
        best, key = None, None
        for g in order:
            if head[g] >= 0:
                continue
            sat = len({head[x] for x in sharing[g] if head[x] >= 0})
            k = (sat, len(sharing[g]))
            if key is None or k > key:
                best, key = g, k
        used = [sum(1 for x in sharing[best] if head[x] == h) for h in range(len(cap))]
        h = min(range(len(cap)), key=lambda h: (used[h], load[h] / cap[h]))
        head[best] = h
        load[h] += 1

    def conflicts(g, h):
        return sum(1 for x in sharing[g] if head[x] == h)
    for it in range(20000):                 # min-conflicts repair
        bad = [g for g in range(n) if conflicts(g, head[g])]
        if not bad:
            break
        g = bad[rng.integers(len(bad))]
        load[head[g]] -= 1
        sc = [conflicts(g, h) + 3.0 * load[h] / cap[h] * len(cap) / n + 0.01 * rng.random() for h in range(len(cap))]
        head[g] = int(np.argmin(sc))
        load[head[g]] += 1
    return head


def assign_codes(groups, passes=4):
    """Heads first (groups sharing a word never share a head), then within each head the code that least
    disturbs the groups it may meet in a question (same domain, then same broad subject)."""
    codes = codebook()
    nc = len(codes)
    chead = np.array([h for h, _, _ in codes])
    full = np.zeros((nc, D))
    fullm = np.zeros((nc, D))
    for c, (h, u, m) in enumerate(codes):
        full[c, HEAD_DIMS[h]] = u
        fullm[c, HEAD_DIMS[h]] = m
    gram = np.maximum(full @ full.T, full @ fullm.T)   # best cross-match between two code pairs
    near = np.clip((gram - 0.85) / 0.12, 0, 1) ** 2 + 20.0 * (gram > 0.999)
    word_groups = {}
    for gi, (_, ws, _r) in enumerate(groups):
        for w in ws:
            word_groups.setdefault(w, []).append(gi)
    sharing = [sorted({g2 for w in ws for g2 in word_groups[w] if g2 != g}) for g, (_, ws, _r) in enumerate(groups)]
    cap = np.array([np.sum(chead == h) for h in range(len(HEAD_DIMS))], float)
    gh = assign_heads(groups, sharing, cap)
    doms = [g[0] for g in groups]
    sups = [super_domain(d) for d in doms]
    by_dom, by_sup = {}, {}
    for g, (d, s_) in enumerate(zip(doms, sups)):
        by_dom.setdefault(d, []).append(g)
        by_sup.setdefault(s_, []).append(g)
    choice = [None] * len(groups)
    allowed = [np.where(chead == h)[0] for h in range(len(HEAD_DIMS))]
    total = np.zeros(nc)
    dsum = {d: np.zeros(nc) for d in by_dom}
    ssum = {x: np.zeros(nc) for x in by_sup}

    def cost_of(g):
        return 0.02 * total + 0.7 * dsum[doms[g]] + 0.3 * ssum[sups[g]]

    def place(g, c):
        choice[g] = c
        for acc, sign in ((total, 1), (dsum[doms[g]], 1), (ssum[sups[g]], 1)):
            acc += near[c]

    def remove(g):
        c = choice[g]
        total[:] -= near[c]
        dsum[doms[g]][:] -= near[c]
        ssum[sups[g]][:] -= near[c]
        choice[g] = None

    order = sorted(range(len(groups)), key=lambda g: -len(groups[g][1]))
    for p in range(passes + 1):
        for g in order:
            if choice[g] is not None:
                remove(g)
            cand = allowed[gh[g]]
            place(g, int(cand[np.argmin(cost_of(g)[cand])]))
    final = [float(cost_of(g)[choice[g]]) for g in range(len(groups))]
    return codes, choice, final


def knowledge():
    groups = read_groups()
    codes, choice, _ = assign_codes(groups)
    vec = {}
    for (dom, ws, roles), c in zip(groups, choice):
        h, u, m = codes[c]
        for w in ws:
            full = np.zeros(D)
            full[HEAD_DIMS[h]] = u if roles[w] == 0 else m
            vec[w] = vec.get(w, 0) + full
    neg = set()
    for line in open(os.path.join(HERE, "negation.txt")):
        neg |= set(re.findall(r"[a-z0-9']+", line.split("#")[0].lower()))
    return vec, neg


def vocab_list():
    vec, neg = knowledge()
    return sorted(set(vec) | neg), vec, neg



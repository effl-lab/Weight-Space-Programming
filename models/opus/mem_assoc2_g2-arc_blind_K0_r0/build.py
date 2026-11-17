"""Turns the hand-written knowledge base in kb/ into the weights of the second-order association memory (arch.py).

Run `python build.py --vocab` once after editing kb/ to regenerate vocab.txt; build() reads vocab.txt and kb/.
See NOTES.md for the design.
"""
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

TOKEN = re.compile(r"[a-z0-9']+")
HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "kb")
PRIME = 1000003
H = 1 << 20          # rows of the pair table
R = 40               # recalls per pair row

# ---------------------------------------------------------------- weights of the parts (chosen by reasoning, not fitted)
W_SYM_A = 1.0        # symmetric fact line: every concept <-> every other (first order), scaled 1/sqrt(m-1)
W_SYM_P = 0.3        # symmetric fact line: a pair recalls the rest of the line
W_SYN = 0.4          # near-synonym lines (facts_synonyms.txt) count less than topic lines
W_DIR_A = 1.5        # directed rule cue -> answer (first order), scaled 1/sqrt(#cues)
W_DIR_P = 2.0        # directed rule: a pair of cues recalls the answers
W_REV = 0.3          # a directed rule also supports its cues from its answers (first order only), at this fraction
W_REV_P = 0.0        # ... and through pairs (off: it made answers' many words into noisy cues)
W_EQ_P = 0.5         # 'term = definition': pair (term, definition word) recalls the other definition words
W_NEG = -0.7         # 'cues >! answers': evidence against
W_TAX = 1.0          # taxonomy parent <-> child, decays per level
TAX_DECAY = 0.5
TAX_PAIR = 0.4       # the words of a category name together recall its members only weakly
FAN_REF = 20.0       # first-order rows of hub concepts are scaled by 1/sqrt(1 + fan/FAN_REF)
FAN_IN_REF = 40.0    # ... and answer concepts supported by very many question concepts by 1/sqrt(1 + fan_in/FAN_IN_REF)
HOP2 = 0.5           # total mass of two-hop associations (q -> b -> c) relative to the direct ones
PAIR_UNIT = 1.0      # pair recalls add up sublinearly: v -> sqrt(v * PAIR_UNIT)
MIX = 1.5            # weight of the first-order read q A against the pair read
IDF_COMMON = 0.8     # cap on the weight of general (non-science) words, kb/common.txt
# negation: 'except' is almost always a negated question; 'not' is often just part of a scenario, so it is mild
NEG_WORDS = {"except": ("#except", 2.5), "not": ("#not", 0.4), "isn't": ("#not", 0.4), "aren't": ("#not", 0.4)}
NEG_PARTNER = 0.0
IDF_NEG = 4.5        # above every other concept, so a negation word is always among the J strongest
# direction words: one concept for "more/increase", one for "less/decrease". They carry no topic, so they get no
# first-order associations at all; they only act inside pairs (e.g. (float, water) recalls "less dense", and
# (altitude, #up) recalls "#down pressure").
DIRECTION = {"#up": "more increase increases increased increasing raise raises raised greater high higher highest".split(),
             "#down": ("less fewer decrease decreases decreased decreasing reduce reduces reduced reducing lessen lessens "
                       "low lower lowers lowered lowest").split()}
IDF_DIR = 1.6
DIR_OPP = 0.7        # a pair's recall of one direction counts this much against the other direction
PHRASE_SHARE = True  # words of a multiword name share one answer weight
DEF_CHAIN = 0.5      # a rule whose answer is a defined term also recalls the definition, at this fraction
TAX_INST = 0.5       # rule cues that name a category also fire for its members, at this fraction
TAX_INST_MAX = 150   # ... for categories with at most this many members


def kb_lines(pattern):
    for path in sorted(glob.glob(os.path.join(KB, pattern))):
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip().lower()
                if line:
                    yield line


def words(text):
    return TOKEN.findall(text.lower())


# ---------------------------------------------------------------- morphology
VOWELS = set("aeiou")
PROTECT = {"habitat", "combat", "format", "thermostat", "lens", "news", "species", "series", "always", "gas",
           "lightning", "early", "physics", "mathematics", "genetics", "economics", "electronics", "diabetes",
           "rabies", "measles", "mumps", "glasses", "scissors", "pants", "clothes", "means", "molasses",
           "christmas", "texas", "kansas", "arkansas", "atlas", "canvas", "chaos", "bias", "pancreas",
           "cactus", "fungus", "nucleus", "virus", "octopus", "platypus", "hippopotamus", "walrus",
           "asparagus", "uterus", "esophagus", "stimulus", "hummus", "status", "census", "focus", "bonus",
           "bus", "us", "this", "his", "is", "was", "has", "its", "plus", "thus", "perhaps", "towards"}


def has_vowel(s):
    return any(c in VOWELS for c in s)


def raw_stem(w):
    w = w.replace("'", "")
    if len(w) <= 2 or w.isdigit() or w in PROTECT:
        return w
    suffixed = False
    if w.endswith("ies") and len(w) > 4:
        w, suffixed = w[:-3] + "y", True
    elif w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        w, suffixed = w[:-1], True
    if w in PROTECT:
        return w
    if w.endswith("ing") and len(w) - 3 >= 2 and has_vowel(w[:-3]):
        w, suffixed = w[:-3], True
    elif w.endswith("ied") and len(w) > 4:
        w, suffixed = w[:-3] + "y", True
    elif w.endswith("ed") and not w.endswith("eed") and len(w) - 2 >= 2 and has_vowel(w[:-2]):
        w, suffixed = w[:-2], True
    elif w.endswith("ly") and len(w) >= 6 and not w.endswith("lly"):
        w = w[:-2]
    elif w.endswith("lly") and len(w) >= 7:
        w = w[:-2]
    if w.endswith("ification") and len(w) > 11:
        w = w[:-9] + "ify"
    elif w.endswith("ication") and len(w) - 7 >= 3:
        w = w[:-7] + "ic"
    elif w.endswith("ation") and len(w) - 5 >= 3:
        w = w[:-5]
    elif w.endswith(("ction", "ption", "ution")) and len(w) - 3 >= 4:
        w = w[:-3]
    elif w.endswith("ate") and len(w) - 3 >= 3 and w[-4] not in VOWELS:
        w = w[:-3]
    elif suffixed and w.endswith("at") and len(w) - 2 >= 3 and w[-3] not in VOWELS and w not in PROTECT:
        w = w[:-2]
    if w.endswith("e") and len(w) >= 3:
        w = w[:-1]
    if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in "aeiousy":
        w = w[:-1]
    return w


class Morph:
    def __init__(self):
        self.stop = set()
        for line in kb_lines("stop.txt"):
            self.stop.update(words(line))
        self.same = {}
        for line in kb_lines("same.txt"):
            ws = line.split()
            if not ws:
                continue
            if ws[0].startswith("!"):          # a concept of its own, kept apart from the stem
                ws[0] = ws[0][1:]
                key = ws[0] + "#"
            else:
                key = self.same.get(ws[0], raw_stem(ws[0]))
            for x in ws:
                self.same[x] = key
        for key, ws in DIRECTION.items():
            for x in ws:
                self.same[x] = key
                self.stop.discard(x)
        # adjectives: comparatives and superlatives map to the base
        for line in kb_lines("adj.txt"):
            for a in words(line):
                key = self.same.get(a, raw_stem(a))
                self.same.setdefault(a, key)
                for f in comparatives(a):
                    self.same.setdefault(f, key)

    def stem(self, w):
        w = w.replace("'", "") if w not in self.same else w
        if w in self.same:
            return self.same[w]
        return raw_stem(w)

    def concept_words(self, text):
        """Stems of the content words of a text (stop words dropped)."""
        out = []
        for w in words(text):
            if w in self.stop or w.replace("'", "") in self.stop:
                continue
            s = self.stem(w)
            if s and s not in out:
                out.append(s)
        return out


def comparatives(a):
    out = []
    if a.endswith("y") and len(a) > 2:
        out += [a[:-1] + "ier", a[:-1] + "iest"]
    elif a.endswith("e"):
        out += [a + "r", a + "st"]
    else:
        out += [a + "er", a + "est"]
        if len(a) >= 3 and a[-1] not in "aeiouwy" and a[-2] in VOWELS and a[-3] not in VOWELS:
            out += [a + a[-1] + "er", a + a[-1] + "est"]
    return out


def inflections(w):
    """Candidate inflected forms of a written word (kept only if they stem back to the same concept)."""
    out = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "ly", w + "'s", w + "s'"}
    if w.endswith("e"):
        out |= {w[:-1] + "ing", w[:-1] + "ation", w[:-1] + "ion"}
    if w.endswith("y"):
        out |= {w[:-1] + "ies", w[:-1] + "ied"}
    if w.endswith("ate"):
        out |= {w[:-1] + "ion", w[:-3] + "ation"}
    if len(w) >= 3 and w[-1] not in "aeiouwxy" and w[-2] in VOWELS and w[-3] not in VOWELS:
        out |= {w + w[-1] + "ed", w + w[-1] + "ing"}
    if w.endswith("ify"):
        out |= {w[:-1] + "ication"}
    if w.endswith(("ct", "pt", "ut")):
        out |= {w + "ion", w + "ions"}
    if w.endswith("s") and not w.endswith("ss"):
        out |= {w[:-1]}
    return out


# ---------------------------------------------------------------- knowledge base parsing
def parse_facts(m):
    """Fact lines from kb/facts_*.txt as (kind, left stems, right stems).
    'a b > x y' directed, 'a b = x y' both directions, 'a b c' symmetric."""
    facts = []
    for path in sorted(glob.glob(os.path.join(KB, "facts_*.txt"))):
        for line in kb_lines(os.path.basename(path)):
            parse_line(m, line, facts, "syn" if "synonyms" in path else "sym")
    return facts


def parse_line(m, line, facts, plain):
    if True:
        if ">!" in line:                    # negative rule: the cues argue against these answers
            l, r = line.split(">!", 1)
            facts.append(("neg", m.concept_words(l), m.concept_words(r)))
        elif ">" in line:
            l, r = line.split(">", 1)
            facts.append(("dir", m.concept_words(l), m.concept_words(r)))
        elif "=" in line:
            l, r = line.split("=", 1)
            L, Rr = m.concept_words(l), m.concept_words(r)
            if {"#up", "#down"} <= set(L + Rr):      # "how high or low": no direction meant
                L, Rr = [x for x in L if x not in DIRECTION], [x for x in Rr if x not in DIRECTION]
            facts.append(("def", L, Rr))
            facts.append(("def", Rr, L))
            facts.append(("eq", L, Rr))
        else:   # topic clusters carry no direction
            facts.append((plain, [x for x in m.concept_words(line) if x not in DIRECTION], []))


def parse_tax(m):
    """kb/tax_*.txt lines 'parent: child child ...' (multiword names joined with _ are split into words)."""
    edges = defaultdict(set)
    for line in kb_lines("tax*.txt"):
        if ":" not in line:
            continue
        p, cs = line.split(":", 1)
        P = tuple(m.concept_words(p.replace("_", " ")))
        for c in cs.split(","):
            C = tuple(m.concept_words(c.replace("_", " ")))
            if P and C and P != C:
                edges[P].add(C)
    return edges


def all_written_words():
    ws = set()
    for pat in ("facts_*.txt", "tax*.txt", "same.txt", "adj.txt"):
        for line in kb_lines(pat):
            ws.update(words(line.replace("_", " ")))
    return ws


def bases(w):
    """Guesses at the base form of a written (possibly inflected) word."""
    out = {w}
    if w.endswith("ies"):
        out.add(w[:-3] + "y")
    if w.endswith("es"):
        out.add(w[:-2])
    if w.endswith("s"):
        out.add(w[:-1])
    if w.endswith("ed"):
        out |= {w[:-2], w[:-1]}
        if len(w) > 4 and w[-3] == w[-4]:
            out.add(w[:-3])
    if w.endswith("ing"):
        out |= {w[:-3], w[:-3] + "e"}
        if len(w) > 5 and w[-4] == w[-5]:
            out.add(w[:-4])
    if w.endswith("ied"):
        out.add(w[:-3] + "y")
    return {b for b in out if len(b) >= 2}


def make_vocab():
    m = Morph()
    base = all_written_words()
    vocab = set()
    for w in base:
        if w in m.stop:
            continue
        s = m.stem(w)
        for b in bases(w):
            for f in inflections(b):
                if f not in m.stop and m.stem(f) == s:
                    vocab.add(f)
    for w, s in m.same.items():
        if w not in m.stop:
            vocab.add(w)
    vocab |= set(NEG_WORDS)
    vocab = sorted(v for v in vocab if TOKEN.fullmatch(v))
    with open(os.path.join(HERE, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    return vocab


# ---------------------------------------------------------------- build
def build_tables():
    m = Morph()
    vocab = [line.strip() for line in open(os.path.join(HERE, "vocab.txt"))]
    facts = parse_facts(m)
    tax = parse_tax(m)

    # concepts: every stem of the vocabulary that is not a stop word
    cid = {}
    word_c = np.zeros(len(vocab) + 2, dtype=np.int64)
    for i, w in enumerate(vocab):
        if w in m.stop:
            continue
        s = NEG_WORDS[w][0] if w in NEG_WORDS else m.stem(w)
        if s not in cid:
            cid[s] = len(cid) + 1
        word_c[i + 2] = cid[s]
    C = max(len(cid) + 1, 16)

    def ids(stems):
        return [cid[s] for s in stems if s in cid]
    inv_cid = {v: k for k, v in cid.items()}

    # document frequency over fact lines and taxonomy lines -> idf
    df = np.zeros(C)
    docs = 0
    for kind, L, Rr in facts:
        for c in set(ids(L) + ids(Rr)):
            df[c] += 1
        docs += 1
    for P, cs in tax.items():
        for c in set(ids(P) + [x for ch in cs for x in ids(ch)]):
            df[c] += 1
        docs += 1
    idf = np.log((docs + 1) / (df + 1)).astype(np.float32)
    idf = np.clip(idf, 1.0, None) / math.log(docs + 1) * 4.0   # roughly 0.5 .. 4
    idf[0] = 0.0
    for line in kb_lines("common.txt"):
        for wd in words(line):
            c = cid.get(m.stem(wd))
            if c and wd not in m.stop:
                idf[c] = min(idf[c], IDF_COMMON)

    A = defaultdict(float)
    P = defaultdict(lambda: defaultdict(float))

    # multiword names (taxonomy entries, defined terms): as answers, their words share one weight, so that an
    # option like "wheel and axle" or "carbon dioxide" does not count twice
    phrases = set()
    for par, chs in tax.items():
        for t in [par] + list(chs):
            if 2 <= len(t) <= 3:
                phrases.add(tuple(t))
    for kind, L, Rr in facts:
        if kind == "eq" and 2 <= len(L) <= 3:
            phrases.add(tuple(L))

    def answer_weights(stems):
        """{concept id: weight} for the answer words of a rule; words of one multiword name share a weight."""
        wt = [1.0] * len(stems)
        i = 0
        while i < len(stems):
            for k in (3, 2):
                if PHRASE_SHARE and i + k <= len(stems) and tuple(stems[i:i + k]) in phrases:
                    for j in range(i, i + k):
                        wt[j] = 1.0 / math.sqrt(k)
                    i += k - 1
                    break
            i += 1
        out = {}
        for st, x in zip(stems, wt):
            if st in cid:
                out[cid[st]] = max(out.get(cid[st], 0.0), x)
        return out

    def add_dir(L, Rr, wa, wp, phrase=False):
        L = ids(L)
        aw = {c: 1.0 for c in ids(Rr)} if phrase else answer_weights(Rr)
        Rw = [(r, x) for r, x in aw.items() if r not in L]
        if not L or not Rw:
            return
        # a rule with many answers supports each less; the words of one named thing share one weight
        n_eff = sum(x for _, x in Rw)
        spread = 1.0 / len(Rw) if phrase else min(1.0, math.sqrt(2.0 / max(n_eff, 1.0)))
        wa, wp = wa * spread, wp * spread
        sa = wa / math.sqrt(len(L))
        for a in L:
            for b, x in Rw:
                A[(a, b)] += sa * x
        if len(L) >= 2:
            sp = wp * 2.0 / len(L)
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    lo, hi = min(L[i], L[j]), max(L[i], L[j])
                    row = P[(lo, hi)]
                    for b, x in Rw:
                        row[b] += sp * x

    # an answer that is a defined term also brings its definition words (options often paraphrase the term)
    if DEF_CHAIN > 0:
        defs = defaultdict(list)
        for kind, L, Rr in facts:
            if kind == "eq":
                key = tuple(sorted(set(ids(L))))
                if 1 <= len(key) <= 3:
                    defs[key].extend(c for c in ids(Rr) if c not in key and c not in defs[key])
        by_word = defaultdict(list)
        for key in defs:
            by_word[key[0]].append(key)
        for kind, L, Rr in list(facts):
            if kind != "dir":
                continue
            Ri = set(ids(Rr))
            Li = set(ids(L))
            for r in list(Ri):
                for key in by_word.get(r, ()):
                    if set(key) <= Ri and not (set(key) & Li) and 2 * len(key) >= len(Ri):
                        extra = [c for c in defs[key] if c not in Ri and c not in Li][:8]
                        if extra:
                            facts.append(("chain", L, [inv_cid[c] for c in extra]))
    for kind, L, Rr in facts:
        if kind == "def":
            add_dir(L, Rr, W_DIR_A, W_DIR_P)
        elif kind == "dir":
            add_dir(L, Rr, W_DIR_A, W_DIR_P)
            add_dir(Rr, L, W_REV * W_DIR_A, W_REV_P * W_DIR_P)
        elif kind == "eq":
            # the term together with one word of its definition recalls the rest of the definition
            Li, Ri = ids(L), [r for r in ids(Rr) if r not in ids(L)]
            if Li and len(Ri) >= 2:
                sp = W_EQ_P * min(1.0, math.sqrt(2.0 / (len(Ri) - 1))) / len(Li)
                for a in Li:
                    for r in Ri:
                        lo, hi = min(a, r), max(a, r)
                        row = P[(lo, hi)]
                        for b in Ri:
                            if b != r:
                                row[b] += sp
        elif kind == "neg":
            add_dir(L, Rr, W_NEG * W_DIR_A, W_NEG * W_DIR_P)
        elif kind == "chain":
            add_dir(L, Rr, DEF_CHAIN * W_DIR_A, DEF_CHAIN * W_DIR_P)
        else:
            S = ids(L)
            if len(S) < 2:
                continue
            sw = answer_weights(L)
            k = W_SYN if kind == "syn" else 1.0
            sa = k * W_SYM_A / math.sqrt(len(S) - 1)
            for a in S:
                for b in S:
                    if a != b:
                        A[(a, b)] += sa * sw.get(b, 1.0)
            if len(S) >= 3:
                sp = k * W_SYM_P * 2.0 / len(S)
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        lo, hi = min(S[i], S[j]), max(S[i], S[j])
                        row = P[(lo, hi)]
                        for b in S:
                            if b != S[i] and b != S[j]:
                                row[b] += sp * sw.get(b, 1.0)

    # taxonomy: closure with decay; parent <-> descendant, and (parent words) pairs recall the child
    def descend(p):
        """Descendants of p with their shallowest depth (breadth first, in a fixed order)."""
        seen, frontier, depth = {p}, [p], 0
        while frontier:
            nxt = []
            for q in frontier:
                for ch in sorted(tax.get(q, ())):
                    if ch not in seen:
                        seen.add(ch)
                        nxt.append(ch)
                        yield ch, depth
            frontier, depth = nxt, depth + 1

    for p in sorted(tax):
        down = min(1.0, 3.0 / math.sqrt(len(tax[p])))     # a parent with many children supports each less
        for ch, depth in descend(p):
            w = W_TAX * (TAX_DECAY ** depth)
            add_dir(list(p), list(ch), w * down, w * down * TAX_PAIR, phrase=True)
            add_dir(list(ch), list(p), w, w, phrase=True)

    # direction concepts act only in pairs
    dirc = {cid[k] for k in DIRECTION if k in cid}
    for key in [k for k in A if k[0] in dirc or k[1] in dirc]:
        del A[key]
    for c in dirc:
        idf[c] = IDF_DIR
    # a rule about a category also holds for its members: the pair (member, other cue) recalls the rule's answers
    if TAX_INST > 0:
        members = {}
        for p in sorted(tax):
            if len(p) == 1 and p[0] in cid:
                ms = []
                for ch, depth in descend(p):
                    cs = [c for c in ids(list(ch)) if c != cid[p[0]]]
                    if cs:
                        ms.append((cs, TAX_DECAY ** depth))
                if len(ms) <= TAX_INST_MAX:
                    members[cid[p[0]]] = ms
        for kind, L, Rr in facts:
            if kind not in ("dir", "def"):
                continue
            Li = ids(L)
            Ri = [r for r in ids(Rr) if r not in Li]
            if len(Li) < 2 or not Ri:
                continue
            sp = TAX_INST * W_DIR_P * min(1.0, math.sqrt(2.0 / len(Ri))) * 2.0 / len(Li)
            for a in Li:
                for cs, dec in members.get(a, ()):
                    for c in cs:
                        for b in Li:
                            if b == a or b == c:
                                continue
                            row = P[(min(b, c), max(b, c))]
                            for r in Ri:
                                if r != c:
                                    row[r] += sp * dec / len(cs)
    # a question concept that supports many answers (a hub) gives each of them less
    fan, fan_in = defaultdict(int), defaultdict(int)
    for (a, b) in A:
        fan[a] += 1
        fan_in[b] += 1
    for key in A:
        A[key] /= math.sqrt(1.0 + fan[key[0]] / FAN_REF) * math.sqrt(1.0 + fan_in[key[1]] / FAN_IN_REF)
    # many rules recalling the same answer through the same pair add up sublinearly
    for row in P.values():
        for b in row:
            v = row[b]
            row[b] = math.copysign(math.sqrt(abs(v) * PAIR_UNIT), v)
    # a pair that recalls "more" argues against "less", and the other way round
    up, down = cid.get("#up"), cid.get("#down")
    if up and down and DIR_OPP > 0:
        for row in P.values():
            u, d = row.get(up, 0.0), row.get(down, 0.0)
            if u or d:
                row[up], row[down] = u - DIR_OPP * d, d - DIR_OPP * u
    # negation: the pair (NOT, q) recalls q's first-order answers with negative weight, so that
    # 'which is NOT ...' / '... EXCEPT' turns the question's evidence around
    first = defaultdict(lambda: defaultdict(float))
    for (a, b), v in A.items():
        first[a][b] += MIX * v
    partner = defaultdict(lambda: defaultdict(float))     # strongest recall of q through any one partner
    for (lo, hi), row in list(P.items()):
        for a, x in ((lo, hi), (hi, lo)):
            for b, v in row.items():
                if v > 0:
                    partner[a][b] = max(partner[a][b], NEG_PARTNER * v * idf[x])
    rows = {}
    for a in set(first) | set(partner):
        tot = defaultdict(float, first[a])
        for b, v in partner[a].items():
            tot[b] += v
        rows[a] = list(tot.items())
    for key, gamma in sorted(set(NEG_WORDS.values())):
        n = cid.get(key)
        if not n:
            continue
        idf[n] = IDF_NEG
        for a, items in rows.items():
            items = sorted(items, key=lambda kv: -abs(kv[1]))[:R]
            lo, hi = min(n, a), max(n, a)
            row = P[(lo, hi)]
            for b, v in items:
                row[b] += -gamma * v / IDF_NEG
    return vocab, word_c, C, idf, A, P, cid


def build():
    vocab, word_c, C, idf, A, P, cid = build_tables()
    if A:
        keys = np.array(list(A.keys()), dtype=np.int64)
        vals = np.array(list(A.values()), dtype=np.float32)
        At = torch.sparse_coo_tensor(torch.from_numpy(keys.T), torch.from_numpy(vals), (C, C)).coalesce()
        At = At.to_dense() if C <= 12000 else At
        if HOP2 > 0 and not At.is_sparse:
            # two-hop association: question concept -> b -> c (the second hop either way round)
            sym = 0.5 * (At + At.T)
            two = At @ sym
            two.fill_diagonal_(0.0)
            scale = HOP2 * At.abs().sum() / max(float(two.abs().sum()), 1e-9)
            At = At + scale * two
    else:
        At = torch.zeros(C, C)

    row_ids = torch.zeros((H, R), dtype=torch.int64)
    row_w = torch.zeros((H, R), dtype=torch.float32)
    merged = defaultdict(lambda: defaultdict(float))
    for (lo, hi), row in P.items():
        h = (lo * PRIME + hi) % H
        tgt = merged[h]
        for b, v in row.items():
            tgt[b] += v
    hs, ks, bs, vs = [], [], [], []
    for h, row in merged.items():
        items = sorted(row.items(), key=lambda kv: -abs(kv[1]))[:R]
        for k, (b, v) in enumerate(items):
            hs.append(h); ks.append(k); bs.append(b); vs.append(v)
    hs, ks = torch.tensor(hs, dtype=torch.int64), torch.tensor(ks, dtype=torch.int64)
    row_ids[hs, ks] = torch.tensor(bs, dtype=torch.int64)
    row_w[hs, ks] = torch.tensor(vs, dtype=torch.float32)
    return {
        "concept": torch.from_numpy(word_c),
        "idf": torch.from_numpy(idf.astype(np.float32)),
        "A": At,
        "row_ids": row_ids,
        "row_w": row_w,
        "mix": torch.tensor(MIX, dtype=torch.float32),
    }


if __name__ == "__main__":
    if "--vocab" in sys.argv:
        v = make_vocab()
        print(len(v), "words")

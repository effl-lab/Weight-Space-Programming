"""Text processing shared by build.py and the vocabulary generator: tokenizer, Porter stemmer, concept mapping."""
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9']+")
HERE = Path(__file__).resolve().parent


def words(text):
    return TOKEN.findall(text.lower())


# ---------------------------------------------------------------- Porter stemmer (written from the 1980 description)
def _cons(w, i):
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _measure(s):
    n, i, L = 0, 0, len(s)
    while i < L and _cons(s, i):
        i += 1
    while i < L:
        while i < L and not _cons(s, i):
            i += 1
        if i >= L:
            break
        while i < L and _cons(s, i):
            i += 1
        n += 1
    return n


def _hasvowel(s):
    return any(not _cons(s, i) for i in range(len(s)))


def _doublec(s):
    return len(s) >= 2 and s[-1] == s[-2] and _cons(s, len(s) - 1)


def _cvc(s):
    if len(s) < 3:
        return False
    L = len(s)
    return _cons(s, L - 3) and not _cons(s, L - 2) and _cons(s, L - 1) and s[-1] not in "wxy"


def _rep(w, suf, new, cond):
    """If w ends with suf and cond(stem) holds, replace; returns (word, matched_suffix)."""
    if w.endswith(suf):
        stem = w[: len(w) - len(suf)]
        if cond(stem):
            return stem + new, True
        return w, True
    return w, False


STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
         ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
         ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
         ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log")]
STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
         ("ness", "")]
STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism",
         "ate", "iti", "ous", "ive", "ize"]


def porter(w):
    if len(w) <= 2:
        return w
    # 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # 1b
    extra = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _hasvowel(w[:-2]):
        w, extra = w[:-2], True
    elif w.endswith("ing") and _hasvowel(w[:-3]):
        w, extra = w[:-3], True
    if extra:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _doublec(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"
    # 1c
    if w.endswith("y") and _hasvowel(w[:-1]):
        w = w[:-1] + "i"
    # 2
    for suf, new in STEP2:
        if w.endswith(suf):
            if _measure(w[: -len(suf)]) > 0:
                w = w[: -len(suf)] + new
            break
    # 3
    for suf, new in STEP3:
        if w.endswith(suf):
            if _measure(w[: -len(suf)]) > 0:
                w = w[: -len(suf)] + new
            break
    # 4
    for suf in STEP4:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 1:
                if suf == "ion":
                    if stem.endswith(("s", "t")):
                        w = stem
                else:
                    w = stem
            break
    # 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    # 5b
    if _measure(w) > 1 and _doublec(w) and w.endswith("l"):
        w = w[:-1]
    return w


# ---------------------------------------------------------------- concept mapping
def read_lines(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def load_synonyms():
    """synonyms.txt: 'canonical: variant variant ...' -> variant maps to canonical's concept."""
    syn = {}
    for line in read_lines(HERE / "synonyms.txt"):
        head, _, rest = line.partition(":")
        head = head.strip()
        for v in rest.split():
            syn[v] = head
    return syn


def load_stopwords():
    sw = set()
    for line in read_lines(HERE / "stopwords.txt"):
        sw.update(line.split())
    return sw


class Concepts:
    """word -> concept key (a stem string), or None for stopwords."""

    def __init__(self):
        self.syn = load_synonyms()
        self.stop = load_stopwords()
        self.cache = {}
        self.stopstems = {porter(s) for s in self.stop}
        claims = {}
        for v, head in self.syn.items():
            if head.startswith("="):
                claims.setdefault(porter(v), set()).add(head)
        self.claim = {st: next(iter(h)) for st, h in claims.items() if len(h) == 1}

    def key(self, w):
        if w in self.cache:
            return self.cache[w]
        k = None
        if w.endswith("'s") and len(w) > 3 and w not in self.syn:
            k = self.key(w[:-2])
        elif w.endswith("'") and len(w) > 2:
            k = self.key(w[:-1])
        elif w not in self.stop:
            base = self.syn.get(w, w)
            for _ in range(3):
                if base != w and base in self.syn and self.syn[base] != base:
                    base = self.syn[base]
            if base in self.stop:
                k = None
            elif base.startswith("="):  # '=stem' forces a literal concept key
                k = base
            elif base[:1].isdigit():
                k = base
            elif base == w and porter(w) in self.claim:
                k = self.claim[porter(w)]
            else:
                k = porter(base)
                if k in self.stopstems and base not in self.syn:
                    k = None
        self.cache[w] = k
        return k

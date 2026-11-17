"""Shared text handling: tokenizing, stemming, reading the hand-written knowledge files."""
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9']+")
HERE = Path(__file__).resolve().parent


def words(text):
    return TOKEN.findall(text.lower())


def read_list(name):
    out = []
    for line in (HERE / name).read_text().splitlines():
        line = line.split("#")[0]
        out += words(line)
    return out


STOP = set(read_list("stopwords.txt"))
GENERIC = set(read_list("generic.txt"))
NEGATION = set(read_list("negation.txt"))

# irregular forms and words the suffix rules would damage: surface -> stem
EXCEPT = {}
for _line in (HERE / "irregular.txt").read_text().splitlines():
    _line = _line.split("#")[0].split()
    if len(_line) >= 2:
        for _w in _line[1:]:
            EXCEPT[_w] = _line[0]
        EXCEPT.setdefault(_line[0], _line[0])

VOWEL = re.compile(r"[aeiouy]")
KEEP_DOUBLE = set("lsz")


def _clean(s):
    """Normalise a stem after removing a suffix: undouble a final consonant, drop a final e."""
    if len(s) >= 3 and s[-1] == s[-2] and s[-1] not in KEEP_DOUBLE and s[-1] not in "aeiou":
        s = s[:-1]
    return s


def stem(w):
    if w.endswith("'s"):
        w = w[:-2]
    w = w.strip("'")
    if w in EXCEPT:
        return EXCEPT[w]
    if len(w) <= 3 or w.isdigit():
        return w
    s = w
    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("ied") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith(("sses", "shes", "ches", "xes", "zes")):
        s = s[:-2]
    elif s.endswith("s") and not s.endswith(("ss", "us", "is", "ous")) and len(s) > 3:
        s = s[:-1]
    elif s.endswith("ing") and len(s) > 5 and VOWEL.search(s[:-3]):
        s = _clean(s[:-3])
    elif s.endswith("ed") and not s.endswith("eed") and len(s) > 4 and VOWEL.search(s[:-2]):
        s = _clean(s[:-2])
    if s in EXCEPT:
        return EXCEPT[s]
    if s.endswith("e") and len(s) > 3:
        s = s[:-1]
    return s


def forms(s, seen):
    """Surface forms that should map to stem s: the ones seen, plus regular inflections."""
    bases = set(seen) | {s, s + "e"}
    out = set(seen)
    for b in list(bases):
        cands = [b, b + "s", b + "es", b + "'s", b + "s'", b + "ed", b + "d", b + "ing", b + "er", b + "ers"]
        if b.endswith("e"):
            cands += [b[:-1] + "ing", b[:-1] + "ed"]
        if b.endswith("y"):
            cands += [b[:-1] + "ies", b[:-1] + "ied"]
        if len(b) >= 3 and b[-1] not in "aeiouwxy" and b[-2] in "aeiou":
            cands += [b + b[-1] + "ing", b + b[-1] + "ed"]
        for c in cands:
            if stem(c) == s:
                out.add(c)
    return out


def knowledge_lines():
    """Every fact line: a list of groups, each group a list of stems (groups after the first are contrasted)."""
    lines = []
    for path in sorted((HERE / "knowledge").glob("*.txt")):
        for raw in path.read_text().splitlines():
            raw = raw.split("#")[0].strip()
            if not raw:
                continue
            groups = []
            for part in raw.split("~"):
                toks = [w for w in words(part) if w not in STOP and w not in NEGATION]
                groups.append(toks)
            lines.append(groups)
    return lines

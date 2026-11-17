"""Regenerate model/vocab.txt: every word of the hand-written files plus inflected / possessive forms whose
concept is a known concept (so 'evaporates' or 'insect's' map to the same concept as a word in the facts).
The previous vocabulary is kept too (words never hurt: unknown words are id 1 anyway)."""
import re
import sys
from pathlib import Path

M = Path("/srv/work/model")
sys.path.insert(0, str(M))
from textproc import Concepts, words  # noqa: E402

cm = Concepts()
src = list((M / "kb").glob("*.txt")) + [M / n for n in ("isa.txt", "synonyms.txt", "stopwords.txt")]
base = set()
for p in src:
    base.update(words(p.read_text()))
base.update(words(Path("/srv/work/model/extra_words.txt").read_text()) if Path("/srv/work/model/extra_words.txt").exists() else [])
known = {cm.key(t) for t in base} - {None}

SUF = ["s", "es", "ed", "d", "ing", "er", "ers", "est", "ly", "ness", "ment", "ments", "al", "ally", "ation",
       "ations", "ion", "ions", "ity", "ive", "able", "ful", "less", "ist", "ists", "ism", "ize", "izes", "ized",
       "ic", "ical", "y", "ies", "ied", "ier", "iest", "ily", "ous", "ance", "ence", "ant", "ent"]


def variants(t):
    out = set()
    stems = {t}
    if t.endswith("e"):
        stems.add(t[:-1])
    if t.endswith("y"):
        stems.add(t[:-1] + "i")
        stems.add(t[:-1])
    if len(t) >= 3 and re.match(r".*[^aeiou][aeiou][bdgklmnprt]$", t):
        stems.add(t + t[-1])
    for s in stems:
        for x in SUF:
            out.add(s + x)
    return out


vocab = set(l.strip() for l in (M / "vocab.txt").read_text().splitlines() if l.strip())
old = set(vocab)
vocab |= base
for t in list(base):
    if not re.fullmatch(r"[a-z]+", t):
        continue
    for v in variants(t):
        if v in vocab:
            continue
        k = cm.key(v)
        if k is not None and k in known:
            vocab.add(v)
add = set()
for v in vocab - old:
    if re.fullmatch(r"[a-z]+", v):
        add.add(v + "'s")
        add.add(v + "s'")
vocab |= {a for a in add if cm.key(a) is not None}
vocab.discard("")
(M / "vocab.txt").write_text("\n".join(sorted(vocab)) + "\n")
print(len(vocab), "words;", len(base), "base words")

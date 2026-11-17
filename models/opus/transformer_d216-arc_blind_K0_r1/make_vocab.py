"""Writes vocab.txt: every word of knowledge/*.txt, the inflected forms that stem to the same stem, and the
stopwords. Run by hand after editing the knowledge files (build() only reads vocab.txt)."""
from pathlib import Path

import common


def variants(w: str) -> set[str]:
    out = {w, w + "s", w + "es", w + "ed", w + "ing", w + "d", w + "er", w + "ers"}
    if w.endswith("e"):
        out |= {w[:-1] + "ing", w[:-1] + "ed"}
    if w.endswith("y"):
        out |= {w[:-1] + "ies", w[:-1] + "ied"}
    if len(w) >= 3 and w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou":
        out |= {w + w[-1] + "ing", w + w[-1] + "ed"}
    if w.endswith("s"):
        out.add(w[:-1])
    out |= {w + "'s", w + "s'"}
    base = w[:-1] if w.endswith("e") else w
    out |= {w + "ly", base + "ly", w + "ally", w + "ness", w + "ment", w + "ful", base + "ity", base + "ation",
            base + "ion", base + "ive", w + "al", base + "al", w + "est", base + "est", w + "ist", w + "ic",
            w + "ical", base + "ic", base + "ical", w + "able", base + "able"}
    if w.endswith("y"):
        out |= {w[:-1] + "ily", w[:-1] + "iness", w[:-1] + "ier", w[:-1] + "iest"}
    return out


def main() -> None:
    folder = Path(__file__).parent
    seen = set()
    for _, ws in common.knowledge_lines(folder):
        seen.update(ws)
    vocab = set(common.STOPWORDS) | set(common.IRREGULAR)
    for w in seen:
        s = common.stem(w)
        vocab |= {v for v in variants(w) if s in common.stem_candidates(v)}
    vocab = sorted(v for v in vocab if common.TOKEN.fullmatch(v))
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n")
    print(len(vocab), "words")


if __name__ == "__main__":
    main()

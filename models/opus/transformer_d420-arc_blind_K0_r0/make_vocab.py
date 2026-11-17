"""Writes vocab.txt: every word of the knowledge files with its regular inflections, plus the function words.
Run by hand after editing the knowledge (build() reads the result)."""
import kb

seen = {}
for groups in kb.knowledge_lines():
    for g in groups:
        for w in g:
            seen.setdefault(kb.stem(w), set()).add(w)
for w, s in kb.EXCEPT.items():
    if s in seen:
        seen[s].add(w)
vocab = set(kb.STOP) | set(kb.NEGATION) | set(kb.GENERIC)
for s, ws in seen.items():
    vocab |= kb.forms(s, ws)
vocab = sorted(v for v in vocab if v and kb.TOKEN.fullmatch(v))
open(kb.HERE / "vocab.txt", "w").write("\n".join(vocab) + "\n")
print(len(vocab), "words,", len(seen), "stems")

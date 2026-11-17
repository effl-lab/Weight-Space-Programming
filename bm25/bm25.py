"""BM25 over the text an LLM wrote for a model, on the ARC test questions.

text/<run>.txt holds that text: every non-comment line of the text files the LLM delivered with the model (vocabulary
file excluded), one line per document. The solver is the retrieval solver of the ARC paper (Clark et al. 2018) with
Lucene's defaults: the query is the question plus one option, a line must share at least one non-stopword with the
question and one with the option, and the option's score is the Okapi BM25 score (k1 = 1.2, b = 0.75) of its best
line. Tokens are lowercased runs of letters and digits minus Lucene's English stopwords; --stem strips plural and
-ed/-ing suffixes (Porter's steps 1a-1c), the variant reported in the paper. results.json holds the stored scores.

    python -m bm25.bm25 --stem bm25/text/free-arc_blind_K0_r0.txt [more files]
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from wsp import arc, free_model

K1, B = 1.2, 0.75
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then there these "
           "they this to was will with".split())  # Lucene's default English stopword set


def stem_light(word: str) -> str:
    """Steps 1a-1c of Porter (1980): plurals, -ed/-ing, final y."""
    w = word
    for suf, rep in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
        if w.endswith(suf):
            w = w[: len(w) - len(suf)] + rep
            break
    for suf in ("ing", "ed"):
        if w.endswith(suf) and re.search(r"[aeiou]", w[: -len(suf)]):
            w = w[: -len(suf)]
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif len(w) > 1 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    if w.endswith("y") and re.search(r"[aeiou]", w[:-1]):
        w = w[:-1] + "i"
    return w


def tokens(text: str, stem: bool) -> list[str]:
    out = [t for t in TOKEN.findall(text.lower()) if t not in STOP]
    return [stem_light(t) for t in out] if stem else out


def solve(facts: list[str], qs: list[str], os_: list, stem: bool) -> np.ndarray:
    docs = [Counter(tokens(f, stem)) for f in facts]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    n, avg = len(docs), sum(sum(d.values()) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in d)
    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        length = sum(d.values())
        for t, tf in d.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            rows.append(i)
            cols.append(vocab[t])
            vals.append(idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg)))
    idx = torch.tensor([rows, cols])
    W = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (n, len(vocab))).coalesce()
    Bin = torch.sparse_coo_tensor(idx, torch.ones(len(vals)), (n, len(vocab))).coalesce()

    def bag(text: str) -> torch.Tensor:
        v = torch.zeros(len(vocab))
        ids = [vocab[t] for t in set(tokens(text, stem)) if t in vocab]
        v[ids] = 1.0
        return v
    logits = np.zeros((len(qs), 4), dtype=np.float32)
    for s in range(0, len(qs), 128):
        qv = torch.stack([bag(q) for q in qs[s:s + 128]])                       # (m, V)
        ov = torch.stack([bag(o) for opts in os_[s:s + 128] for o in opts])     # (4m, V)
        m = qv.shape[0]
        query = (qv.repeat_interleave(4, 0) + ov).clamp(max=1.0)
        score = torch.sparse.mm(W, query.t()).t()                               # (4m, lines)
        ok = (torch.sparse.mm(Bin, qv.t()).t().repeat_interleave(4, 0) > 0) & (torch.sparse.mm(Bin, ov.t()).t() > 0)
        logits[s:s + m] = (score * ok).amax(1).reshape(m, 4).numpy()
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--stem", action="store_true")
    a = ap.parse_args()
    stored = json.loads((Path(__file__).parent / "results.json").read_text())
    qs, os_, y = arc.load("test")
    for f in a.files:
        lines = f.read_text(encoding="utf-8").splitlines()
        acc = free_model.accuracy(solve(lines, qs, os_, a.stem), np.asarray(y))["accuracy"]
        was = stored.get(f.stem, {}).get("bm25_stemmed_accuracy" if a.stem else "bm25_accuracy")
        print(f.stem, f"lines={len(lines)}", f"bm25={acc:.4f}", f"stored={was}")


if __name__ == "__main__":
    main()

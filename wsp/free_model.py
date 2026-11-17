"""Models whose structure the writer designs: `free-<subject>` per MMLU subject, free-arc and its variants for ARC, and
design-cifar for CIFAR-10 images (forward(w, x) on float32 images, see IMAGE_RUNNER; no vocab.txt).

The writer delivers a folder (DELIVERABLE):
  vocab.txt   one word per line; line k (0-based) is id k + 2, id 0 is padding, id 1 any other word
  build.py    build() -> dict of name -> torch tensor: every weight, made from files the writer wrote
  arch.py     forward(w, q, o) -> (n, 4) logits; q (n, Q_LEN) and o (n, 4, O_LEN) are int64 word ids
  anything else the writer's build() reads

Text becomes word ids by TOKEN (words, encode); MathBench architectures (<name>-mb) use MATH_TOKEN (math_words,
math_encode), which keeps signs, decimal points and LaTeX commands.
SQuAD architectures (<name>-sq) answer a question about a paragraph: forward(w, q, c) -> (n, 2, SC_LEN + 1) start and
end logits over the paragraph's words, column 0 being "no answer" (span_encode, span_decode, SPAN_RUNNER); the
SQuAD 1.1 architectures (<name>-sq1) are the same with column 0 ignored, since every item has an answer.
Generation architectures (<name>-gen) answer a SciQ question in words: forward(w, q, a) -> (n, V) logits of the next
word given the answer so far, run word by word until id 0 (gen_encode, GEN_RUNNER, gen_text).
arch.py is restricted to a fixed tensor program (static_problems): no branching on data, no Python-side lookup.
In py-arc arch.py is any Python over the raw texts instead, forward(w, questions, options) (py_static_problems,
PY_RUNNER, run_py), and vocab.txt is optional. In fit-arc the folder holds at most FIT_MAX_BYTES (max_bytes_of).
run() builds and runs a candidate in a bwrap with no network and no data other than the candidate itself.
"""
import ast
import json
import re
import shutil
import string
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from wsp import paths

DELIVERABLE = "model"
TOKEN = re.compile(r"[a-z0-9']+")
# MathBench: a word, a run of digits, a LaTeX command with its backslash, or any other single non-space character
MATH_TOKEN = re.compile(r"[a-z']+|\d|\\[a-z]+|[^\sa-z0-9']")
ALL_TOKEN = re.compile(r"[A-Za-z']+|\d|\\[A-Za-z]+|[^\sA-Za-z0-9']")  # MATH_TOKEN before lower-casing
Q_LEN, O_LEN = 128, 48
GQ_LEN, GA_LEN = 64, 5  # generation (<name>-gen): question words read, answer words generated at most
ARCH_IMPORTS = {"torch", "math"}
ARCH_FORBIDDEN_NODES = (ast.If, ast.IfExp, ast.While, ast.Assert, ast.BoolOp, ast.Try, ast.With, ast.Match,
                        ast.Global, ast.Nonlocal, ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef)
ARCH_FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "getattr",
                        "setattr", "vars", "dict", "set", "str", "repr", "int", "float", "bool", "hash", "print",
                        "sorted", "min", "max", "any", "all", "map", "filter", "iter", "next"}
ARCH_FORBIDDEN_ATTRS = {"tolist", "item", "numpy", "load", "save", "data_ptr", "storage", "apply_", "map_"}
# py-arc: what arch.py may not import (a package and its submodules), call or reach as an attribute; names and
# attributes that start with "__" are refused too, and imports are limited to the standard library, torch and numpy.
PY_FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "io", "pathlib", "shutil", "glob", "importlib", "ctypes",
                        "builtins", "pickle", "urllib", "http", "multiprocessing"}
PY_FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                      "vars", "getattr", "setattr", "delattr"}
PY_FORBIDDEN_ATTRS = {"open", "load", "save", "fromfile", "tofile", "loadtxt", "genfromtxt", "memmap", "from_file"}
FIT_MAX_BYTES = 10 ** 9  # fit-arc: the files of the folder together
RUN_TIMEOUT_S = 1200
MEMORY_BYTES = 6 * 2 ** 30  # RLIMIT_DATA of the runner
TENSOR_GB = 2  # measured: tensor allocations fail past about 2.5 GiB under that limit
VENV = paths.VENV
RUNNER = '''import json, resource, sys
resource.setrlimit(resource.RLIMIT_DATA, (%d, %d))
sys.path.insert(0, "/build")
import numpy as np, torch
torch.set_num_threads(4)
import build, arch
w = build.build()
ids = np.load("/build/_ids.npz")
out = {}
with torch.no_grad():
    for split in ("validation", "test", "probe"):
        if split + "_q" in ids:
            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()
np.savez("/build/_logits.npz", **out)
json.dump({"weights": int(sum(v.numel() for v in w.values())),
           "nonzero": int(sum(int((v.to_dense() if v.is_sparse else v).count_nonzero()) for v in w.values()))},
          open("/build/_size.json", "w"))
''' % (MEMORY_BYTES, MEMORY_BYTES)
IMAGE_RUNNER = RUNNER.replace(
    '''        if split + "_q" in ids:
            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''',
    '''        if split + "_x" in ids:
            x = torch.from_numpy(ids[split + "_x"])
            out[split] = torch.cat([arch.forward(w, x[i:i + 64]).float() for i in range(0, len(x), 64)]).numpy()''')


SPAN_RUNNER = RUNNER.replace(
    '''            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''',
    '''            q, c = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_c"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], c[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''')
GEN_RUNNER = RUNNER.replace(
    '''            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''',
    '''            q = torch.from_numpy(ids[split + "_q"])
            a = torch.zeros(len(q), GA_LEN, dtype=torch.int64)
            for t in range(GA_LEN):  # one word per step; an answer ends at its first id 0
                logits = torch.cat([arch.forward(w, q[i:i + 64], a[i:i + 64]).float() for i in range(0, len(q), 64)])
                assert logits.shape == (len(q), V), f"forward returned {tuple(logits.shape)}, not {(len(q), V)}"
                a[:, t] = logits.argmax(1) * (a[:, :t] != 0).all(1)
            out[split] = a.numpy()''').replace(
    '''ids = np.load("/build/_ids.npz")''',
    '''ids = np.load("/build/_ids.npz")
V = len(open("/build/vocab.txt").read().splitlines()) + 2
GA_LEN = 5''')
assert len({RUNNER, IMAGE_RUNNER, SPAN_RUNNER, GEN_RUNNER}) == 4 and "GA_LEN = %d" % GA_LEN in GEN_RUNNER
# The general task (<name>-all): one input of X_LEN ids, an answer of at most A_LEN output ids. Ids 1..N_MARKS - 1 are
# the marks of MARKS; the answer of a "choose" item is the mark of its option.
X_LEN, A_LEN = 512, 64
MARKS = ("", "[CHOOSE]", "[A]", "[B]", "[C]", "[D]", "[ANSWER]", "[PASSAGE]", "[SUMMARIZE]")
N_MARKS = len(MARKS)
CAPS = {"choose": 1, "answer": 30, "summarize": A_LEN}  # output ids an item may write
OPTION_WORDS, QUESTION_WORDS = 64, 64
# The general task reads 512 positions per row, so its runner gets more memory than the others (2026-11-14: a model
# that compared every pair of positions of 64 rows ran out under MEMORY_BYTES).
ALL_MEMORY_BYTES, ALL_TENSOR_GB = 20 * 2 ** 30, 16
ALL_RUNNER = RUNNER.replace("(%d, %d)" % (MEMORY_BYTES, MEMORY_BYTES), "(%d, %d)" % (ALL_MEMORY_BYTES, ALL_MEMORY_BYTES)).replace(
    '''        if split + "_q" in ids:
            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''',
    '''        if split + "_x" in ids:
            x, cap = torch.from_numpy(ids[split + "_x"]), torch.from_numpy(ids[split + "_cap"])
            a = torch.zeros(len(x), A_LEN, dtype=torch.int64)
            for t in range(A_LEN):  # one output id per step; an answer ends at its first id 0 or at its item's cap
                live = ((cap > t) & (a[:, :t] != 0).all(1)).nonzero()[:, 0]
                if len(live) == 0:
                    break
                logits = torch.cat([arch.forward(w, x[live[i:i + 64]], a[live[i:i + 64]]).float() for i in range(0, len(live), 64)])
                assert logits.shape == (len(live), OUT), f"forward returned {tuple(logits.shape)}, not {(len(live), OUT)}"
                a[live, t] = logits.argmax(1)
            out[split] = a.numpy()''').replace(
    '''ids = np.load("/build/_ids.npz")''',
    '''ids = np.load("/build/_ids.npz")
A_LEN, X_LEN, N_MARKS = %d, %d, %d
OUT = N_MARKS + len(open("/build/vocab.txt").read().splitlines()) + X_LEN''' % (A_LEN, X_LEN, N_MARKS))
assert len({RUNNER, IMAGE_RUNNER, SPAN_RUNNER, GEN_RUNNER, ALL_RUNNER}) == 5 and "A_LEN, X_LEN, N_MARKS = %d" % A_LEN in ALL_RUNNER
# The language-model task (<name>-text): a text is cut into pieces, the model predicts the next piece. Piece ids: 0 is
# padding, then BASE_PIECES (the line break, and every lower-case printable character with and without a leading
# space), then the lines of vocab.txt.
CTX_LEN, NEW_LEN = 768, 64
SPACE, BREAK = "\u2581", "\u21b5"
BASE_CHARS = "".join(ch for ch in string.printable if not ch.isspace() and not ch.isupper())
BASE_PIECES = (BREAK, *BASE_CHARS, *(SPACE + ch for ch in BASE_CHARS))
LETTER_COLUMNS = tuple(BASE_PIECES.index(p + ch) for ch in "abcd" for p in (SPACE, ""))  # in the runner's `first`
LM_RUNNER = RUNNER.replace("(%d, %d)" % (MEMORY_BYTES, MEMORY_BYTES), "(%d, %d)" % (ALL_MEMORY_BYTES, ALL_MEMORY_BYTES)).replace(
    '''        if split + "_q" in ids:
            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''',
    '''        if split + "_x" in ids:
            x, cap = torch.from_numpy(ids[split + "_x"]), torch.from_numpy(ids[split + "_cap"])
            a = torch.zeros(len(x), NEW_LEN, dtype=torch.int64)
            first = torch.zeros(len(x), N_BASE)
            for t in range(NEW_LEN):  # one piece per step; a continuation ends at a line break, at id 0 or at its cap
                live = ((cap > t) & (a[:, :t] > 1).all(1)).nonzero()[:, 0]
                if len(live) == 0:
                    break
                logits = torch.cat([arch.forward(w, x[live[i:i + 64]]).float() for i in range(0, len(live), 64)])
                assert logits.shape == (len(live), V), f"forward returned {tuple(logits.shape)}, not {(len(live), V)}"
                if t == 0:
                    first[live] = logits[:, 1:N_BASE + 1]
                a[live, t] = logits.argmax(1)
                x[live] = torch.cat([x[live, 1:], a[live, t:t + 1]], dim=1)
            out[split], out[split + "_first"] = a.numpy(), first.numpy()''').replace(
    '''ids = np.load("/build/_ids.npz")''',
    '''ids = np.load("/build/_ids.npz")
NEW_LEN, N_BASE = %d, %d
V = 1 + N_BASE + len([ln for ln in open("/build/vocab.txt", encoding="utf-8").read().split("\\n") if ln])''' % (NEW_LEN, len(BASE_PIECES)))
assert len({RUNNER, ALL_RUNNER, LM_RUNNER}) == 3 and "NEW_LEN, N_BASE = %d" % NEW_LEN in LM_RUNNER and "x[live, 1:]" in LM_RUNNER
# py-arc: forward gets the raw texts and may return a tensor, an array or nested lists. Size, of w as build() returns
# it: tensors as in RUNNER, and python_values counts every number, string, bool and None in w at any depth, dict keys
# included (a container met twice is walked once); python_chars counts the characters of those strings.
PY_RUNNER = RUNNER.replace(
    '''w = build.build()
''',
    '''w = build.build()
assert isinstance(w, dict), f"build() returned a {type(w).__name__}, not a dict"
tensors, values, seen, stack = [], [], set(), [w]
while stack:
    x = stack.pop()
    if torch.is_tensor(x):
        tensors.append(x)
    elif isinstance(x, (dict, list, tuple, set, frozenset)):
        if id(x) not in seen:
            seen.add(id(x))
            stack += [*x.keys(), *x.values()] if isinstance(x, dict) else list(x)
    else:
        assert x is None or isinstance(x, (bool, int, float, str)), f"w holds a {type(x).__name__}, not a tensor or plain Python data"
        values.append(x)
size = {"weights": int(sum(v.numel() for v in tensors)),
        "nonzero": int(sum(int((v.to_dense() if v.is_sparse else v).count_nonzero()) for v in tensors)),
        "python_values": len(values), "python_chars": sum(len(v) for v in values if isinstance(v, str))}
''').replace(
    '''            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()''',
    '''            q, o = ids[split + "_q"].tolist(), ids[split + "_o"].tolist()
            parts = []
            for i in range(0, len(q), 64):
                s = arch.forward(w, q[i:i + 64], o[i:i + 64])
                s = np.asarray(s.float().numpy() if torch.is_tensor(s) else s, dtype=np.float32)
                assert s.shape == (len(q[i:i + 64]), 4), f"forward returned {s.shape}, not {(len(q[i:i + 64]), 4)}"
                parts.append(s)
            out[split] = np.concatenate(parts)''').replace(
    '''json.dump({"weights": int(sum(v.numel() for v in w.values())),
           "nonzero": int(sum(int((v.to_dense() if v.is_sparse else v).count_nonzero()) for v in w.values()))},
          open("/build/_size.json", "w"))''',
    '''json.dump(size, open("/build/_size.json", "w"))''')
assert len({RUNNER, PY_RUNNER}) == 2 and "python_values" in PY_RUNNER and "parts.append(s)" in PY_RUNNER \
    and "json.dump(size" in PY_RUNNER


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def encode(vocab: list[str], questions: list[str], options: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    index = {w: i + 2 for i, w in enumerate(vocab)}

    def row(text: str, length: int) -> list[int]:
        ids = [index.get(w, 1) for w in words(text)][:length]
        return ids + [0] * (length - len(ids))

    q = np.array([row(t, Q_LEN) for t in questions], dtype=np.int64).reshape(len(questions), Q_LEN)
    o = np.array([[row(t, O_LEN) for t in opts] for opts in options], dtype=np.int64).reshape(len(questions), 4, O_LEN)
    return q, o


def math_words(text: str) -> list[str]:
    return MATH_TOKEN.findall(text.lower())


def math_encode(vocab: list[str], questions: list[str], options: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    index = {w: i + 2 for i, w in enumerate(vocab)}

    def row(text: str, length: int) -> list[int]:
        ids = [index.get(w, 1) for w in math_words(text)][:length]
        return ids + [0] * (length - len(ids))

    q = np.array([row(t, Q_LEN) for t in questions], dtype=np.int64).reshape(len(questions), Q_LEN)
    o = np.array([[row(t, O_LEN) for t in opts] for opts in options], dtype=np.int64).reshape(len(questions), 4, O_LEN)
    return q, o


def encode_of(arch: str) -> Callable:
    """The encode of an architecture: math_encode for MathBench (<name>-mb), encode for any other."""
    return math_encode if arch.endswith("-mb") else encode


def max_bytes_of(arch: str) -> int | None:
    """The cap on the folder's files: FIT_MAX_BYTES for fit-arc, none for any other."""
    return FIT_MAX_BYTES if arch == "fit-arc" else None


SPAN_TOKEN = re.compile(r"[A-Za-z0-9']+")
SQ_LEN, SC_LEN = 64, 512
HASH_IDS = 65536  # ids of words that are not in vocab.txt
MAX_SPAN = 30  # words in an answer, at most


def span_tokens(text: str) -> list[tuple[str, int, int]]:
    """The words of a text, lower-cased, each with its character range in the text."""
    return [(m.group().lower(), m.start(), m.end()) for m in SPAN_TOKEN.finditer(text)]


def span_encode(vocab: list[str], questions: list[str], contexts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Word ids of questions (n, SQ_LEN) and paragraphs (n, SC_LEN); 0 is padding. A word in vocab has id 2 + its
    line; any other word has id len(vocab) + 2 + crc32(word) % HASH_IDS, the same wherever it occurs."""
    index = {w: i + 2 for i, w in enumerate(vocab)}
    base = len(vocab) + 2

    def row(text: str, length: int) -> list[int]:
        ids = [index[t] if t in index else base + zlib.crc32(t.encode()) % HASH_IDS
               for t, _, _ in span_tokens(text)][:length]
        return ids + [0] * (length - len(ids))

    q = np.array([row(t, SQ_LEN) for t in questions], dtype=np.int64).reshape(len(questions), SQ_LEN)
    c = np.array([row(t, SC_LEN) for t in contexts], dtype=np.int64).reshape(len(contexts), SC_LEN)
    return q, c


def span_decode(logits: np.ndarray, contexts: list[str]) -> list[str]:
    """Answer texts from (n, 2, SC_LEN + 1) logits: row 0 start, row 1 end; column 0 is "no answer", column k word
    k - 1 of the paragraph. The answer is the span of words i..j, j - i < MAX_SPAN, with the largest
    start[i] + end[j], cut from the paragraph text; it is "" (no answer) unless that sum is larger than
    start[0] + end[0]."""
    out = []
    for row, text in zip(logits.astype(np.float64), contexts):
        tokens = span_tokens(text)[:SC_LEN]
        start, end = row[0, 1:len(tokens) + 1], row[1, 1:len(tokens) + 1]
        best, span = row[0, 0] + row[1, 0], None
        for k in range(min(MAX_SPAN, len(tokens))):
            total = start[:len(tokens) - k] + end[k:]
            i = int(total.argmax())
            if total[i] > best:
                best, span = total[i], (i, i + k)
        out.append("" if span is None else text[tokens[span[0]][1]:tokens[span[1]][2]])
    return out


# span_decode without its first slot. A function of its own: the source of span_decode is part of the SQuAD 2.0 task
# text and the source of this one of the SQuAD 1.1 task text.
def span_decode_forced(logits: np.ndarray, contexts: list[str]) -> list[str]:
    """Answer texts from (n, 2, SC_LEN + 1) logits: row 0 start, row 1 end; column k is word k - 1 of the paragraph
    and column 0 is ignored. The answer is the span of words i..j, j - i < MAX_SPAN, with the largest
    start[i] + end[j], cut from the paragraph text."""
    out = []
    for row, text in zip(logits.astype(np.float64), contexts):
        tokens = span_tokens(text)[:SC_LEN]
        start, end = row[0, 1:len(tokens) + 1], row[1, 1:len(tokens) + 1]
        best, span = -np.inf, None
        for k in range(min(MAX_SPAN, len(tokens))):
            total = start[:len(tokens) - k] + end[k:]
            i = int(total.argmax())
            if total[i] > best:
                best, span = total[i], (i, i + k)
        out.append("" if span is None else text[tokens[span[0]][1]:tokens[span[1]][2]])
    return out


def gen_encode(vocab: list[str], questions: list[str]) -> np.ndarray:
    """Word ids of questions (n, GQ_LEN): a word in vocab has id 2 + its line, any other word id 1; 0 is padding."""
    index = {w: i + 2 for i, w in enumerate(vocab)}
    rows = [[index.get(t, 1) for t in words(text)][:GQ_LEN] for text in questions]
    return np.array([r + [0] * (GQ_LEN - len(r)) for r in rows], dtype=np.int64).reshape(len(questions), GQ_LEN)


def gen_text(vocab: list[str], answers: np.ndarray) -> list[str]:
    """Answer texts from generated word ids (n, GA_LEN): the words up to the first id 0, joined by spaces (id 1,
    the unknown word, is written as "?")."""
    out = []
    for row in answers.tolist():
        row = row[:row.index(0)] if 0 in row else row
        out.append(" ".join("?" if i == 1 else vocab[i - 2] for i in row))
    return out


def all_tokens(text: str) -> list[tuple[str, int, int]]:
    """The tokens of a text (a word, one digit, a LaTeX command or one other symbol), lower-cased, each with its
    character range in the text."""
    return [(m.group().lower(), m.start(), m.end()) for m in ALL_TOKEN.finditer(text)]


def all_source(item: dict) -> list[tuple[str, str, int, int]]:
    """The input of an item as (token, text it comes from, start, end) per position; a mark has the empty text.
    choose:    [CHOOSE] question [A] option [B] option [C] option [D] option   (an option keeps its first
               OPTION_WORDS tokens, the question its last tokens that still fit)
    answer:    [ANSWER] question [PASSAGE] passage                             (the question keeps its first
               QUESTION_WORDS tokens, the passage what fits)
    summarize: [SUMMARIZE] article                                             (what fits)"""
    def part(text: str) -> list[tuple[str, str, int, int]]:
        return [(t, text, i, j) for t, i, j in all_tokens(text)]

    def mark(name: str) -> list[tuple[str, str, int, int]]:
        return [(name, "", 0, 0)]

    if item["kind"] == "choose":
        options = [part(o)[:OPTION_WORDS] for o in item["options"]]
        room = X_LEN - 1 - sum(1 + len(o) for o in options)
        question = part(item["question"])
        out = mark("[CHOOSE]") + question[max(0, len(question) - room):]
        for letter, o in zip("ABCD", options):
            out += mark(f"[{letter}]") + o
        return out
    if item["kind"] == "answer":
        return (mark("[ANSWER]") + part(item["question"])[:QUESTION_WORDS] + mark("[PASSAGE]") + part(item["passage"]))[:X_LEN]
    return (mark("[SUMMARIZE]") + part(item["article"]))[:X_LEN]


def all_encode(vocab: list[str], items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Input ids (n, X_LEN) and the number of output ids each item may write (n,). Id 0 is padding, ids 1..N_MARKS - 1
    the marks, a token in vocab has id N_MARKS + its line, and any other token has id
    N_MARKS + len(vocab) + crc32(token) % HASH_IDS, the same wherever it occurs."""
    index = {w: i + N_MARKS for i, w in enumerate(vocab)}
    base = N_MARKS + len(vocab)
    rows = []
    for item in items:
        ids = [MARKS.index(t) if not text else index[t] if t in index else base + zlib.crc32(t.encode()) % HASH_IDS
               for t, text, _, _ in all_source(item)]
        rows.append(ids + [0] * (X_LEN - len(ids)))
    return (np.array(rows, dtype=np.int64).reshape(len(items), X_LEN),
            np.array([CAPS[item["kind"]] for item in items], dtype=np.int64))


def all_text(vocab: list[str], items: list[dict], answers: np.ndarray) -> list[str]:
    """Answer texts from output ids (n, A_LEN), read up to the first id 0. An output id below N_MARKS is a mark
    ("A" to "D" for the option marks, nothing for the others), an id below N_MARKS + len(vocab) is that vocabulary
    word, and a larger id k copies input position k - N_MARKS - len(vocab). Copies of positions that follow one
    another in the same text are cut from that text in one piece, so the characters between them are kept."""
    base, out = N_MARKS + len(vocab), []
    for item, row in zip(items, answers.tolist()):
        source = all_source(item)
        row = row[:row.index(0)] if 0 in row else row
        pieces, run = [], None  # run: (text, start, end, last position) of the copy being extended
        for i in row + [0]:
            p = i - base
            if 0 <= p < len(source) and source[p][1]:
                _, text, start, end = source[p]
                if run and run[0] is text and run[3] == p - 1:
                    run = (text, run[1], end, p)
                    continue
                if run:
                    pieces.append(run[0][run[1]:run[2]])
                run = (text, start, end, p)
                continue
            if run:
                pieces.append(run[0][run[1]:run[2]])
                run = None
            if 0 <= p < len(source):
                pieces.append(source[p][0].strip("[]") if source[p][0] in MARKS[2:6] else "")
            elif N_MARKS <= i < base:
                pieces.append(vocab[i - N_MARKS])
            elif 0 < i < N_MARKS:
                pieces.append(MARKS[i].strip("[]") if 2 <= i <= 5 else "")
        out.append(" ".join(x for x in pieces if x))
    return out


def piece_lines(folder: Path) -> list[str]:
    return [ln for ln in (folder / "vocab.txt").read_text(encoding="utf-8").split("\n") if ln]


def piece_problems(lines: list[str]) -> list[str]:
    """Why vocab.txt lines are not valid pieces (empty when they are): a piece is one or more characters of
    BASE_CHARS, optionally after one SPACE mark; no piece twice, and none that is already in BASE_PIECES."""
    allowed = set(BASE_CHARS)
    bad = [ln for ln in lines if not ln.removeprefix(SPACE) or set(ln.removeprefix(SPACE)) - allowed]
    twice = [ln for ln, c in Counter(lines).items() if c > 1 or ln in BASE_PIECES]
    return ([f"vocab.txt: {len(bad)} lines are not pieces, e.g. {bad[0]!r}"] if bad else []) + \
        ([f"vocab.txt: {len(twice)} pieces are repeated or already base pieces, e.g. {twice[0]!r}"] if twice else [])


def piece_vocab(folder: Path) -> list[str]:
    """Every piece of a language-model folder in id order (id 1 is the first): BASE_PIECES, then vocab.txt's lines."""
    return [*BASE_PIECES, *piece_lines(folder)]


def piece_encode(pieces: list[str], texts: list[str]) -> np.ndarray:
    """Piece ids (n, CTX_LEN) of texts, right-aligned and padded with 0 on the left; a longer text keeps its end.
    The text is lower-cased and any character outside BASE_CHARS becomes "?". Each line is split at white space;
    a word is written with a leading SPACE mark and cut greedily into the longest pieces that match, from its
    left. Lines are joined by the BREAK piece."""
    index = {p: i + 1 for i, p in enumerate(pieces)}
    longest = max(len(p) for p in pieces)
    cache: dict[str, list[int]] = {}

    def word(text: str) -> list[int]:
        if text not in cache:
            ids, i = [], 0
            while i < len(text):
                j = next(j for j in range(min(len(text), i + longest), i, -1) if text[i:j] in index)
                ids.append(index[text[i:j]])
                i = j
            cache[text] = ids
        return cache[text]

    rows = []
    for text in texts:
        ids: list[int] = []
        for k, line in enumerate(text.lower().split("\n")):
            ids += [index[BREAK]] * (k > 0)
            for w in line.split():
                ids += word(SPACE + "".join(ch if ch in BASE_CHARS else "?" for ch in w))
        ids = ids[-CTX_LEN:]
        rows.append([0] * (CTX_LEN - len(ids)) + ids)
    return np.array(rows, dtype=np.int64).reshape(len(texts), CTX_LEN)


def piece_text(pieces: list[str], ids: np.ndarray) -> list[str]:
    """Texts of continuations (n, NEW_LEN): the pieces up to the first line break or id 0, joined; a SPACE mark
    becomes a space."""
    out = []
    for row in ids.tolist():
        stop = next((k for k, i in enumerate(row) if i <= 1), len(row))
        out.append("".join(pieces[i - 1] for i in row[:stop]).replace(SPACE, " ").strip())
    return out


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over lower-cased words of letters and digits: the longest common subsequence of the two whole
    texts (no stemming, no split into sentences)."""
    p, r = re.findall(r"[a-z0-9]+", prediction.lower()), re.findall(r"[a-z0-9]+", reference.lower())
    if not p or not r:
        return 0.0
    row = [0] * (len(r) + 1)
    for x in p:
        prev = 0
        for j, y in enumerate(r, 1):
            prev, row[j] = row[j], prev + 1 if x == y else max(row[j], row[j - 1])
    return 2 * row[-1] / (len(p) + len(r)) if row[-1] else 0.0


def all_scores(items: list[dict], answers: np.ndarray, texts: list[str]) -> dict:
    """Per part: accuracy for "choose" (the first output id has to be the mark of the right option), SQuAD F1 for
    "answer", ROUGE-L F1 for "summarize". `groups` and `accuracy` are the task's score: the mean over arc, mmlu, math
    and writing, where writing is the mean of its two parts."""
    from wsp import general
    parts = {}
    for name in general.PARTS:
        rows = [k for k, it in enumerate(items) if it["part"] == name]
        if not rows:
            continue
        kind = items[rows[0]]["kind"]
        if kind == "choose":
            value = float(np.mean([int(answers[k, 0]) == 2 + items[k]["gold"] for k in rows]))
        elif kind == "answer":
            value = span_scores([texts[k] for k in rows], [items[k]["gold"] for k in rows])["f1"]
        else:
            value = float(np.mean([rouge_l(texts[k], items[k]["gold"]) for k in rows]))
        parts[name] = {"score": value, "n": len(rows), "words": float(np.mean([(answers[k] != 0).sum() for k in rows]))}
    groups = {g: float(np.mean([parts[p]["score"] for p in ps])) for g, ps in general.GROUPS.items() if all(p in parts for p in ps)}
    return {"accuracy": float(np.mean(list(groups.values()))) if groups else None, "groups": groups, "parts": parts}


def _normalized(text: str) -> list[str]:
    """Words of an answer as the official SQuAD evaluation compares them."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def span_scores(predictions: list[str], golds: list[list[str]]) -> dict:
    """Official SQuAD 2.0 exact match and F1 (best over an item's gold answers; an item with no gold answer is right
    only with the prediction ""), for all items, for the answerable ones and for the unanswerable ones."""
    em, f1 = [], []
    for pred, gold in zip(predictions, golds):
        p = _normalized(pred)
        answers = [g for g in (_normalized(a) for a in gold) if g] or [[]]
        scores = []
        for g in answers:
            same = sum((Counter(p) & Counter(g)).values())
            if not p or not g:
                scores.append((float(p == g), float(p == g)))
            else:
                precision, recall = same / len(p), same / len(g)
                scores.append((float(p == g), 2 * precision * recall / (precision + recall) if same else 0.0))
        em.append(max(e for e, _ in scores))
        f1.append(max(f for _, f in scores))
    em, f1 = np.array(em), np.array(f1)
    has = np.array([any(_normalized(a) for a in g) for g in golds])
    answered = np.array([bool(p) for p in predictions])

    def part(mask: np.ndarray) -> dict:
        return {"exact": float(em[mask].mean()) if mask.any() else None,
                "f1": float(f1[mask].mean()) if mask.any() else None, "n": int(mask.sum())}

    return {"accuracy": float(f1.mean()), "exact": float(em.mean()), "f1": float(f1.mean()), "n": len(golds),
            "answered": float(answered.mean()), "answerable": part(has), "unanswerable": part(~has)}


def static_problems(source: str, signature: str = "forward(w, q, o)") -> list[str]:
    """Why an arch.py is not a fixed tensor program (empty when it is)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"arch.py syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [f"arch.py: import {a.name} not allowed" for a in node.names
                         if a.name not in ARCH_IMPORTS and a.name != "torch.nn.functional"]
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ("torch", "torch.nn", "torch.nn.functional", "math"):
                problems.append(f"arch.py: from {node.module} import not allowed (line {node.lineno})")
        elif isinstance(node, ARCH_FORBIDDEN_NODES):
            problems.append(f"arch.py: {type(node).__name__} not allowed (line {node.lineno})")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if any(g.ifs for g in node.generators):
                problems.append(f"arch.py: comprehension condition not allowed (line {node.lineno})")
        elif isinstance(node, ast.Name) and (node.id in ARCH_FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"arch.py: name {node.id} not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and (node.attr in ARCH_FORBIDDEN_ATTRS or node.attr.startswith("__")):
            problems.append(f"arch.py: attribute .{node.attr} not allowed (line {node.lineno})")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in tree.body):
        problems.append(f"arch.py: no top-level {signature}")
    return problems[:20]


def py_static_problems(source: str) -> list[str]:
    """Why a py-arc arch.py breaks its rules (empty when it keeps them)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"arch.py syntax error: {e}"]
    allowed = (set(sys.stdlib_module_names) | {"torch", "numpy"}) - PY_FORBIDDEN_MODULES
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or "."]
            problems += [f"arch.py: import {m} not allowed (line {node.lineno})" for m in modules
                         if m.split(".")[0] not in allowed]
        elif isinstance(node, ast.Name) and (node.id in PY_FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"arch.py: name {node.id} not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and (node.attr in PY_FORBIDDEN_ATTRS or node.attr.startswith("__")):
            problems.append(f"arch.py: attribute .{node.attr} not allowed (line {node.lineno})")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in tree.body):
        problems.append("arch.py: no top-level forward(w, questions, options)")
    return problems[:20]


def folder_problems(folder: Path, images: bool = False, signature: str = "forward(w, q, o)", py: bool = False,
                    max_bytes: int | None = None) -> list[str]:
    if folder.is_symlink() or not folder.is_dir():
        return [f"no folder {DELIVERABLE}/"]
    needed = ("build.py", "arch.py") if images or py else ("vocab.txt", "build.py", "arch.py")
    problems = [f"{DELIVERABLE}/{n} missing" for n in needed if not (folder / n).is_file()]
    problems += [f"{p.relative_to(folder)} is a link" for p in folder.rglob("*") if p.is_symlink()]
    if max_bytes is not None and (size := sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())) > max_bytes:
        problems.append(f"{DELIVERABLE}/ holds {size:,} bytes of files, more than {max_bytes:,}")
    if problems:
        return problems
    source = (folder / "arch.py").read_text()
    return py_static_problems(source) if py else static_problems(source, "forward(w, x)" if images else signature)


def vocab_of(folder: Path) -> list[str]:
    return [line.strip() for line in (folder / "vocab.txt").read_text().splitlines()]


def _argv(build_dir: Path, isolate: bool) -> list[str]:
    python = [f"{VENV}/bin/python", "-I", "/build/_run.py" if isolate else str(build_dir / "_run.py")]
    if not isolate:
        return python
    return ["bwrap", "--die-with-parent", "--unshare-all", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin", "--ro-bind", str(VENV), str(VENV),
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(build_dir), "/build", "--chdir", "/build",
            "--setenv", "PATH", f"{VENV}/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp"] + python


def _execute(folder: Path, arrays: dict[str, np.ndarray], runner: str, shapes: dict[str, tuple[int, int]],
             isolate: bool) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds the folder and runs `runner` on `arrays` (saved as _ids.npz); checks every output against `shapes`."""
    tmp = Path(tempfile.mkdtemp(prefix="wsp_free_"))
    try:
        work = tmp / "build"
        shutil.copytree(folder, work)
        np.savez(work / "_ids.npz", **arrays)
        (work / "_run.py").write_text(runner if isolate else runner.replace("/build", str(work)))
        try:
            proc = subprocess.run(_argv(work, isolate), cwd=work, capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
                                  env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return None, {}, [f"build and forward exceeded {RUN_TIMEOUT_S} s"]
        if proc.returncode != 0:
            return None, {}, [f"model exited {proc.returncode}: {proc.stderr[-2000:]}"]
        with np.load(work / "_logits.npz", allow_pickle=False) as z:
            logits = {k: z[k] for k in z.files}
        bad = [k for k, v in logits.items() if v.shape != shapes[k] or not np.all(np.isfinite(v))]
        if bad:
            return None, {}, [f"forward output for {bad} is not finite logits of shape {[shapes[k] for k in bad]}"]
        return logits, json.loads((work / "_size.json").read_text()), []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(folder: Path, batches: dict[str, tuple[list[str], list[list[str]]]], isolate: bool = True,
        encoder: Callable = encode, max_bytes: int | None = None) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds the candidate and runs it on named batches of (questions, options), which `encoder` turns into word ids.

    Returns (logits per batch, size counts, problems). The candidate sees word ids only, never labels.
    """
    problems = folder_problems(folder, max_bytes=max_bytes)
    if problems:
        return None, {}, problems
    vocab = vocab_of(folder)
    ids = {}
    for name, (questions, options) in batches.items():
        ids[f"{name}_q"], ids[f"{name}_o"] = encoder(vocab, questions, options)
    return _execute(folder, ids, RUNNER, {k: (len(q), 4) for k, (q, _) in batches.items()}, isolate)


def run_py(folder: Path, batches: dict[str, tuple[list[str], list[list[str]]]], isolate: bool = True
           ) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds a py-arc candidate and runs it on named batches of (questions, options) as raw texts, never labels."""
    problems = folder_problems(folder, py=True)
    if problems:
        return None, {}, problems
    ids = {}
    for name, (questions, options) in batches.items():
        ids[f"{name}_q"], ids[f"{name}_o"] = np.array(questions, dtype=str), np.array(options, dtype=str).reshape(-1, 4)
    return _execute(folder, ids, PY_RUNNER, {k: (len(q), 4) for k, (q, _) in batches.items()}, isolate)


def run_spans(folder: Path, batches: dict[str, tuple[list[str], list[str]]], isolate: bool = True
              ) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds a SQuAD model and runs it on named batches of (questions, paragraphs)."""
    problems = folder_problems(folder, signature="forward(w, q, c)")
    if problems:
        return None, {}, problems
    vocab = vocab_of(folder)
    ids = {}
    for name, (questions, contexts) in batches.items():
        ids[f"{name}_q"], ids[f"{name}_c"] = span_encode(vocab, questions, contexts)
    return _execute(folder, ids, SPAN_RUNNER, {k: (len(q), 2, SC_LEN + 1) for k, (q, _) in batches.items()}, isolate)


def run_gen(folder: Path, batches: dict[str, list[str]], isolate: bool = True
            ) -> tuple[dict[str, list[str]] | None, dict, list[str]]:
    """Builds a generation model and lets it answer named batches of questions word by word; returns answer texts."""
    problems = folder_problems(folder, signature="forward(w, q, a)")
    if problems:
        return None, {}, problems
    vocab = vocab_of(folder)
    ids = {f"{name}_q": gen_encode(vocab, questions) for name, questions in batches.items()}
    out, size, problems = _execute(folder, ids, GEN_RUNNER, {k: (len(q), GA_LEN) for k, q in batches.items()}, isolate)
    if problems:
        return None, {}, problems
    return {k: gen_text(vocab, v) for k, v in out.items()}, size, []


def run_all(folder: Path, batches: dict[str, list[dict]], isolate: bool = True
            ) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds a model of the general task and lets it answer named batches of items; returns output ids (n, A_LEN)."""
    problems = folder_problems(folder, signature="forward(w, x, a)")
    if problems:
        return None, {}, problems
    vocab = vocab_of(folder)
    ids = {}
    for name, items in batches.items():
        ids[f"{name}_x"], ids[f"{name}_cap"] = all_encode(vocab, items)
    return _execute(folder, ids, ALL_RUNNER, {k: (len(v), A_LEN) for k, v in batches.items()}, isolate)


def run_lm(folder: Path, batches: dict[str, tuple[list[str], list[int]]], isolate: bool = True
           ) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds a language model and lets it continue named batches of (texts, pieces each may write). Returns per
    batch the continuation ids (n, NEW_LEN) and, under <name>_first, the first step's logits of the base pieces."""
    problems = folder_problems(folder, signature="forward(w, x)") or piece_problems(piece_lines(folder))
    if problems:
        return None, {}, problems
    pieces = piece_vocab(folder)
    ids, shapes = {}, {}
    for name, (texts, caps) in batches.items():
        ids[f"{name}_x"], ids[f"{name}_cap"] = piece_encode(pieces, texts), np.array(caps, dtype=np.int64)
        shapes[name], shapes[f"{name}_first"] = (len(texts), NEW_LEN), (len(texts), len(BASE_PIECES))
    return _execute(folder, ids, LM_RUNNER, shapes, isolate)


def run_images(folder: Path, batches: dict[str, np.ndarray], n_classes: int, isolate: bool = True
               ) -> tuple[dict[str, np.ndarray] | None, dict, list[str]]:
    """Builds an image model and runs it on named batches of float32 images (n, 3, 32, 32) in [0, 1]."""
    problems = folder_problems(folder, images=True)
    if problems:
        return None, {}, problems
    arrays = {f"{k}_x": x.astype(np.float32) for k, x in batches.items()}
    return _execute(folder, arrays, IMAGE_RUNNER, {k: (len(x), n_classes) for k, x in batches.items()}, isolate)


def accuracy(logits: np.ndarray, answers: np.ndarray) -> dict:
    """An unanswered item (all logits equal) counts as a uniform guess (1/4 for ARC's four options)."""
    answered = logits.max(1) > logits.min(1)
    right = (logits.argmax(1) == answers) & answered
    return {"accuracy": float(right.mean() + (1 - answered.mean()) / logits.shape[1]), "answered": float(answered.mean()),
            "accuracy_answered": float(right.sum() / max(answered.sum(), 1)), "n": int(len(answers))}

"""Standard pair scorers for multiple-choice questions, given to writers as a fixed arch.py.

Each module is an arch.py that passes wsp.free_model.static_problems. shapes(name, vocab) lists the weights a model
with `vocab` words (ids 0 and 1 included) needs; gradient-descent baselines start from init(name, vocab, seed).
"""
import math
from pathlib import Path

import torch

from wsp import free_model

DIR = Path(__file__).parent
# transformer_d<w>: the same two-layer transformer at width w (MLP 2w), for comparisons at the parameter count of a
# writer-designed model (widths 8, 24) or of an association memory in Figure 1(b) (16 and 36: GPT-6.1 Sol, generations
# 1 and 5; 216 and 420: Claude Opus 5.5, generations 1 and 5; 12: GPT-6 Astra, generation 1;
# analysis.inference_speed's `matched_transformer` widths); "transformer" is width 64
WIDTHS = {"transformer": 64, **{f"transformer_d{w}": w for w in (8, 12, 16, 24, 36, 216, 420)}}
NAMES = (*WIDTHS, "bow")
# SQuAD (<name>-sq): the same two-layer transformer over the paragraph and the question, with a start and an end
# logit per paragraph position
SPAN = "span_transformer"
# Generation (<name>-gen): the same transformer over the question and the answer so far, with logits of the next
# word over the vocabulary
GEN = "gen_transformer"
BASE_LINE = "D, HEADS, LAYERS, MLP = 64, 4, 2, 128"


# Memory architectures (ours, 2026-11-13): a memory of written units filed under topics; a question is routed to
# ROUTES topics and only their units are read, so the work per question does not grow with the memory. The six differ
# in what a unit is and how it is read. arch.py is memory/front.py followed by memory/<name>.py.
COMMON_SHAPES = """V: lines of vocab.txt plus 2.  C: number of concepts (concept 0 means "none").
T: number of topics, at least 4; a question reads T // 4 of them, at most 16.
M: units per topic (pad short topics with empty units); only a topic's first 256 units are read.
An id table is int64, its weight table float32; an unused slot has id 0 and weight 0.

concept: (V,) int64      word id -> concept id in [0, C); word ids 0 and 1 must map to 0
idf: (C,) float32        weight of each concept in a question's or option's bag
topic_ids, topic_w: (T, Lt)   each topic's cue concepts and their weights (routing)"""
MEMORY = {
    "mem_route": ("a routed fact memory with a product read",
                  "fact_ids, fact_w: (T, M, L)  each fact's concepts and weights\nsecond: ()  float32"),
    "mem_attn": ("a routed fact memory read by one attention step",
                 "fact_ids, fact_w: (T, M, L)  each fact's concepts and weights\nbeta: ()  float32"),
    "mem_core": ("a routed fact memory with categories, a product read and negation",
                 "fact_ids, fact_w: (T, M, L)  each fact's concepts and weights\n"
                 "isa_ids, isa_w: (C, P)  each concept's parent concepts (categories) and weights\n"
                 "neg: (C,) float32  1 for concepts that negate a question, else 0\nqpow, second: ()  float32"),
    "mem_chain": ("a routed fact memory read in two hops",
                  "fact_ids, fact_w: (T, M, L)  each fact's concepts and weights\nsecond, hop2: ()  float32"),
    "mem_triple": ("a routed memory of (subject, relation, object) facts read forward, backward and in two hops",
                   "subj_ids, subj_w: (T, M, Ls)  each fact's subject concepts and weights\n"
                   "obj_ids, obj_w: (T, M, Lo)  each fact's object concepts and weights\n"
                   "rel: (T, M) int64  each fact's relation in [0, R)\n"
                   "cue: (C, R) float32  how strongly a question concept asks for each relation\n"
                   "back, hop2: ()  float32"),
    "mem_rule": ("two attention layers over a routed memory of rules (condition -> conclusion)",
                 "cond_ids, cond_w: (T, M, Lc)  each rule's condition concepts and weights\n"
                 "then_ids, then_w: (T, M, Lv)  each rule's conclusion concepts and weights\nbeta, hop2: ()  float32"),
    "mem_assoc": ("an association memory: one concept x concept matrix read by a matrix product",
                  "A: (C, C) float32  A[a, b]: how strongly question concept a supports answer concept b"),
    "mem_assoc2": ("a second-order association memory: pairs of question concepts recall answer concepts",
                   "A: (C, C) float32  A[a, b]: how strongly question concept a supports answer concept b\n"
                   "row_ids, row_w: (H, R)  row (a * 1000003 + b) % H, a < b: the concepts the pair (a, b) recalls\n"
                   "mix: ()  float32"),
}
# The association memories are not routed: their arch.py has only the bags of memory/front.py, and no topics.
UNROUTED = ("mem_assoc", "mem_assoc2")
UNROUTED_SHAPES = """V: lines of vocab.txt plus 2.  C: number of concepts (concept 0 means "none"), at least 16.
An id table is int64, its weight table float32; an unused slot has id 0 and weight 0.

concept: (V,) int64      word id -> concept id in [0, C); word ids 0 and 1 must map to 0
idf: (C,) float32        weight of each concept in a question's or option's bag"""


def structure(name: str) -> str:
    """The sentence of the task text that names the given structure."""
    if name in MEMORY:
        return f"The structure is fixed: {MEMORY[name][0]}, whose arch.py is given below."
    if name == SPAN:
        return ("The structure is fixed: a standard transformer that reads the paragraph and the question as one "
                "sequence, whose arch.py is given below.")
    if name == GEN:
        return ("The structure is fixed: a standard transformer that reads the question and the answer so far as one "
                "sequence, whose arch.py is given below.")
    return f"The structure is fixed: a standard {name.split('_d')[0]} scorer whose arch.py is given below."


def shape_table(name: str) -> str:
    if name in MEMORY:
        return (UNROUTED_SHAPES if name in UNROUTED else COMMON_SHAPES) + "\n" + MEMORY[name][1]
    if name == GEN:
        return "\n".join(f"{k}: ({', '.join('V' if d == 10 ** 9 else str(d) for d in s)}{',' * (len(s) == 1)})"
                         for k, s in shapes(name, 10 ** 9).items())
    first = f"V + {free_model.HASH_IDS}" if name == SPAN else "V"
    return "\n".join(f"{k}: {tuple(first if i == 0 and k == 'embed' else d for i, d in enumerate(s))}"
                     for k, s in shapes(name, 0).items())


def source(name: str) -> str:
    if name in MEMORY:
        front = (DIR / "memory" / "front.py").read_text()
        if name in UNROUTED:
            front = front[:front.index("def route(")].replace("ROUTES = 16\n\n\n", "\n")
        return front.rstrip() + "\n\n\n" + (DIR / "memory" / f"{name}.py").read_text() if name in UNROUTED else \
            front + "\n\n" + (DIR / "memory" / f"{name}.py").read_text()
    if name in WIDTHS:
        text = (DIR / "transformer.py").read_text()
        assert BASE_LINE in text
        return text.replace(BASE_LINE, f"D, HEADS, LAYERS, MLP = {WIDTHS[name]}, 4, 2, {2 * WIDTHS[name]}")
    return (DIR / f"{name}.py").read_text()


def shapes(name: str, vocab: int) -> dict[str, tuple[int, ...]]:
    if name == "bow":
        return {"embed": (vocab, 64), "W": (64, 64), "u": (64,)}
    d = 64 if name in (SPAN, GEN) else WIDTHS[name]
    mlp = 2 * d
    out = {"embed": (vocab, d), "pos": (free_model.Q_LEN + free_model.O_LEN, d)}
    if name == SPAN:
        out = {"embed": (vocab + free_model.HASH_IDS, d), "none": (d,),
               "pos": (1 + free_model.SC_LEN + free_model.SQ_LEN, d)}
    if name == GEN:
        out = {"embed": (vocab, d), "pos": (free_model.GQ_LEN + free_model.GA_LEN, d)}
    for i in range(2):
        p = f"block{i}."
        out.update({p + "ln1.weight": (d,), p + "ln1.bias": (d,), p + "ln2.weight": (d,), p + "ln2.bias": (d,)})
        out.update({p + n: (d, d) for n in ("q", "k", "v", "o")})
        out.update({p + n + ".bias": (d,) for n in ("q", "k", "v", "o")})
        out.update({p + "fc1": (d, mlp), p + "fc1.bias": (mlp,), p + "fc2": (mlp, d), p + "fc2.bias": (d,)})
    out.update({"ln_f.weight": (d,), "ln_f.bias": (d,)})
    out.update({"span": (d, 2)} if name == SPAN else {"out": (d, vocab), "out.bias": (vocab,)} if name == GEN
               else {"score": (d,)})
    return out


def init(name: str, vocab: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, s in shapes(name, vocab).items():
        if k.endswith(("ln1.weight", "ln2.weight", "ln_f.weight")):
            out[k] = torch.ones(s)
        elif len(s) == 1:
            out[k] = torch.zeros(s)
        elif k in ("embed", "pos"):
            out[k] = torch.randn(s, generator=g) * 0.02
        else:
            out[k] = (torch.rand(s, generator=g) * 2 - 1) * math.sqrt(3 / s[0])
    return out

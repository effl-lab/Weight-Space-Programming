"""One task made of four kinds of items, all text in and words out (wsp.free_model.run_all). Host only.

part    kind       items come from                                      scored by
arc     choose     wsp.arc                                              accuracy
mmlu    choose     the cached cais/mmlu files, all subjects             accuracy
math    choose     wsp.mathbench                                        accuracy
squad   answer     wsp.squad, version 1.1                               SQuAD F1
cnn     summarize  CNN/DailyMail 3.0.0 (validation and test files)      ROUGE-L F1 (wsp.free_model.rouge_l)

examples    40 items with answers, the only items of these datasets a writer sees: 10 of arc, 10 of mmlu (one per
            subject, 10 subjects), 10 of math, and 10 of writing (5 of squad, 5 of cnn)
validation  per part, items that are in no other split
test        the reported split; squad and cnn are seeded samples of their test files (N_TEST), the others whole
An item is a dict with "part", "kind", the texts of its kind ("question" and "options", or "question" and "passage",
or "article") and "gold" (the index of the right option, or the reference answers, or the reference summary).
"""
import glob
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from wsp import arc, mathbench, mmlu, paths, squad

CNN_DIR = paths.DATA_ROOT / "cnn_dailymail"
CNN_FILES = {"validation": ("validation-3.0.0.parquet", "65a5ccce932b08f050114ce6e3c39355d563fa24f194052bb2f27d1c6c499c91"),
             "test": ("test-3.0.0.parquet", "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e")}
PARTS = ("arc", "mmlu", "math", "squad", "cnn")
GROUPS = {"arc": ("arc",), "mmlu": ("mmlu",), "math": ("math",), "writing": ("squad", "cnn")}
SPLIT_SEED = 0
N_EXAMPLES = {"mmlu": 10, "squad": 5, "cnn": 5}
N_VALIDATION = {"squad": 500, "cnn": 300}
N_TEST = {"squad": 2000, "cnn": 1000}
EXAMPLE_WORDS = 220  # a cnn example's article is at most this long, to keep the task text short


def _choices(part: str, questions, options, answers) -> tuple[dict, ...]:
    return tuple({"part": part, "kind": "choose", "question": q.strip(), "options": [str(o).strip() for o in opts],
                  "gold": int(a)} for q, opts, a in zip(questions, options, answers))


@lru_cache(maxsize=None)
def _mmlu(split: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{mmlu.SNAPSHOT}/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no MMLU {split} files in {mmlu.SNAPSHOT}")
    return pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)


@lru_cache(maxsize=None)
def _cnn(split: str) -> pd.DataFrame:
    name, sha = CNN_FILES[split]
    raw = (CNN_DIR / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{CNN_DIR / name}: sha256 is not {sha}")
    df = pd.read_parquet(CNN_DIR / name)
    return df.iloc[np.random.RandomState(SPLIT_SEED).permutation(len(df))].reset_index(drop=True)


def _summaries(df: pd.DataFrame) -> tuple[dict, ...]:
    return tuple({"part": "cnn", "kind": "summarize", "article": a.strip(), "gold": " ".join(h.split())}
                 for a, h in zip(df["article"], df["highlights"]))


@lru_cache(maxsize=None)
def items(split: str, part: str) -> tuple[dict, ...]:
    if part == "arc":
        if split == "examples":
            rows = arc._example_rows(arc.frame("train"), 5)
            return _choices(part, rows["question"], [c["text"] for c in rows["choices"]], rows["answer"])
        return _choices(part, *arc.load(split))
    if part == "math":
        return _choices(part, *mathbench.load(split))
    if part == "mmlu":
        if split != "examples":
            df = _mmlu(split)
            return _choices(part, df["question"], df["choices"], df["answer"])
        df = _mmlu("dev")
        subjects = np.random.RandomState(SPLIT_SEED).permutation(sorted(df["subject"].unique()))[:N_EXAMPLES[part]]
        first = df.groupby("subject", sort=False).head(1).set_index("subject").loc[list(subjects)]
        return _choices(part, first["question"], first["choices"], first["answer"])
    if part == "squad":
        if split == "examples":
            rows = squad.items("examples", "1.1")[:N_EXAMPLES[part]]
        elif split == "validation":
            rows = squad.items("validation", "1.1")[:N_VALIDATION[part]]
        else:
            rows = squad.items("test", "1.1")
            rows = [rows[i] for i in np.random.RandomState(SPLIT_SEED).permutation(len(rows))[:N_TEST[part]]]
        return tuple({"part": part, "kind": "answer", "question": r["question"].strip(), "passage": r["context"],
                      "gold": r["answers"]} for r in rows)
    if part == "cnn":
        if split == "test":
            return _summaries(_cnn("test").iloc[:N_TEST[part]])
        df = _cnn("validation")
        short = df["article"].str.split().str.len() <= EXAMPLE_WORDS
        shown = df[short].iloc[:N_EXAMPLES[part]]
        if split == "examples":
            return _summaries(shown)
        return _summaries(df.drop(shown.index).iloc[:N_VALIDATION[part]])
    raise KeyError(part)


def load(split: str) -> list[dict]:
    """Every item of a split, the parts one after another in the order of PARTS."""
    return [it for part in PARTS for it in items(split, part)]


def examples() -> str:
    """The "examples" split with answers, the only items of these datasets a writer sees."""
    blocks = []
    for it in load("examples"):
        if it["kind"] == "choose":
            blocks.append(f"[{it['part']}] {it['question']}\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", it["options"]))
                          + f"\nAnswer: {'ABCD'[it['gold']]}")
        elif it["kind"] == "answer":
            blocks.append(f"[writing: answer from a passage] Passage: {it['passage'].strip()}\nQuestion: {it['question']}\n"
                          f"Answer: {it['gold'][0]}")
        else:
            blocks.append(f"[writing: summary] Article: {it['article']}\nSummary: {it['gold']}")
    return "\n\n".join(blocks)

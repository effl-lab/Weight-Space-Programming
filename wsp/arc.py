"""ARC (AI2 Reasoning Challenge), Easy and Challenge together, four-option questions only. Host only.

train       3,358 questions, for gradient-descent baselines (shown to writers only in fit-arc, wsp.arc.training)
validation  862 questions, for picking learning rate and step
test        3,530 questions, the reported split; `subset` tells Easy from Challenge
Questions with three or five options (under 1% of each split) are dropped.
"""
import glob

import numpy as np
import pandas as pd

from wsp import paths

SNAPSHOT = f"{paths.HF_HUB}/datasets--allenai--ai2_arc/snapshots/*"
SUBSETS = ("ARC-Easy", "ARC-Challenge")
LETTERS = ("A", "B", "C", "D")
EXAMPLE_SEED = 0
PRACTICE_SEED = 0
N_PRACTICE = 512


def frame(split: str) -> pd.DataFrame:
    parts = []
    for s in SUBSETS:
        df = pd.read_parquet(glob.glob(f"{SNAPSHOT}/{s}/{split}-*.parquet")[0])
        df["subset"] = s
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["choices"].map(lambda c: len(c["text"]) == 4)]
    df["answer"] = [list(c["label"]).index(k) for c, k in zip(df["choices"], df["answerKey"])]
    return df.reset_index(drop=True)


def load(split: str) -> tuple[list[str], list[list[str]], np.ndarray]:
    df = frame(split)
    return df["question"].tolist(), [list(c["text"]) for c in df["choices"]], df["answer"].to_numpy(np.int64)


def subset(split: str) -> np.ndarray:
    return frame(split)["subset"].to_numpy()


def _example_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return pd.concat([df[df["subset"] == s].sample(n, random_state=EXAMPLE_SEED) for s in SUBSETS])


def examples(n: int = 5) -> str:
    """n training questions per subset with answers, the only ARC items a writer sees."""
    part = _example_rows(frame("train"), n)
    blocks = []
    for s, q, c, a in zip(part["subset"], part["question"], part["choices"], part["answer"]):
        blocks.append(f"[{s}] " + q.strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, c["text"]))
                      + f"\nAnswer: {LETTERS[a]}")
    return "\n\n".join(blocks)


def training() -> list[dict]:
    """Every training question with its answer letter, the data the fit-arc writer may fit its model on."""
    questions, options, answers = load("train")
    return [{"question": q, "options": o, "answer": LETTERS[a]} for q, o, a in zip(questions, options, answers)]


def practice() -> list[dict]:
    """N_PRACTICE training questions with answers that the fb-arc writer may measure its model on; none has the text
    of a question in examples()."""
    df = frame("train")
    df = df[~df["question"].str.strip().isin(_example_rows(df, 5)["question"].str.strip())]
    part = df.sample(N_PRACTICE, random_state=PRACTICE_SEED)
    return [{"question": q, "options": list(c["text"]), "answer": int(a)}
            for q, c, a in zip(part["question"], part["choices"], part["answer"])]

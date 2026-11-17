"""MMLU splits per subject, read from the cached cais/mmlu parquet files. Host only: never mounted into a sandbox.

dev         5 public questions per subject, shown to writers as examples
validation  11 to 170 questions per subject, for design choices
test        the reported split
"""
import glob

import numpy as np
import pandas as pd

from wsp import paths

SNAPSHOT = f"{paths.HF_HUB}/datasets--cais--mmlu/snapshots/*/all"
LETTERS = ("A", "B", "C", "D")


def load(split: str, subject: str) -> tuple[list[str], list[list[str]], np.ndarray]:
    """Questions, their four options and the index of the correct option."""
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAPSHOT}/{split}-*.parquet")))
    df = df[df["subject"] == subject]
    return df["question"].tolist(), [list(c) for c in df["choices"]], df["answer"].to_numpy(np.int64)


def examples(subject: str) -> str:
    """The dev questions of a subject as writer-facing text with answers."""
    blocks = []
    for q, opts, y in zip(*load("dev", subject)):
        blocks.append(q.strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, opts)) + f"\nAnswer: {LETTERS[y]}")
    return "\n\n".join(blocks)

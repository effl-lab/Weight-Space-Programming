"""MMLU over all 57 subjects as one multiple-choice task (free-mm), four options each. Host only.

examples    10 dev questions, the first of each of 10 subjects in RandomState(SPLIT_SEED) order: the only MMLU items a
            writer sees (wsp.general shows the same ten in the several-task runs)
validation  the MMLU validation split, all subjects (1,531 questions)
test        the MMLU test split, all subjects (14,042 questions), the reported split and the split the pretrained rows of
            analysis.pretrained_open use; `subset` tells the natural-science subjects from the others
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from wsp import general, memory, mmlu

LETTERS = mmlu.LETTERS
SUBSETS = ("Natural science", "Other")
# the subjects named biology, chemistry, physics or astronomy, as in analysis.mmlu_transfer
NATURAL_SCIENCE = tuple(s for s in memory.SUBJECTS if any(k in s for k in ("biology", "chemistry", "physics", "astronomy")))


@lru_cache(maxsize=None)
def frame(split: str) -> pd.DataFrame:
    if split != "examples":
        return general._mmlu(split)
    dev = general._mmlu("dev")
    subjects = np.random.RandomState(general.SPLIT_SEED).permutation(sorted(dev["subject"].unique()))
    subjects = subjects[:general.N_EXAMPLES["mmlu"]]
    return dev.groupby("subject", sort=False).head(1).set_index("subject").loc[list(subjects)].reset_index()


def load(split: str) -> tuple[list[str], list[list[str]], np.ndarray]:
    df = frame(split)
    return df["question"].tolist(), [list(c) for c in df["choices"]], df["answer"].to_numpy(np.int64)


def subset(split: str) -> np.ndarray:
    return np.where(frame(split)["subject"].isin(NATURAL_SCIENCE), SUBSETS[0], SUBSETS[1])


def examples() -> str:
    """The "examples" split with answers and subjects, the only MMLU items a writer sees."""
    df = frame("examples")
    blocks = []
    for s, q, c, a in zip(df["subject"], df["question"], df["choices"], df["answer"]):
        blocks.append(f"[{s.replace('_', ' ')}] " + q.strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, c))
                      + f"\nAnswer: {LETTERS[a]}")
    return "\n\n".join(blocks)

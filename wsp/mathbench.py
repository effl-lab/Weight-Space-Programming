"""MathBench (release v1), the English single-choice questions, four options each. Host only.

examples    10 questions, 5 per subset: the only MathBench items a writer sees
validation  200 questions, each subset in proportion to its size
test        1,121 questions, the reported split; `subset` tells Theory from Application
Theory is the four <level>_knowledge folders (881 questions), Application the folders middle, high and college (450).
The release has no split. This one deals out, per subset and in a seeded random order, the groups of items whose
stripped question text is the same (the release repeats questions with reordered options), so no question text is in
two splits.
"""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd

from wsp import paths

ZIP = paths.DATA_ROOT / "mathbench" / "mathbench_v1.zip"
SHA256 = "15d5097aee95aae0ac692bf90df7a00558f730319d00e63c288f5365a0c7f8db"
SUBSETS = ("Theory", "Application")
LEVELS = {"Theory": ("primary", "middle", "high", "college"), "Application": ("middle", "high", "college")}
LETTERS = ("A", "B", "C", "D")
SPLIT_SEED = 0
N_EXAMPLES = 5  # per subset
N_VALIDATION = 200


def _splits(df: pd.DataFrame) -> np.ndarray:
    """Split of every row. Per subset, the groups come in the order of RandomState(SPLIT_SEED): the first N_EXAMPLES
    one-item groups are "examples", then every group that still fits in the subset's share of N_VALIDATION is
    "validation", and the rest is "test"."""
    rng = np.random.RandomState(SPLIT_SEED)
    split = np.full(len(df), "test", dtype=object)
    for s in SUBSETS:
        part = df[df["subset"] == s]
        groups = list(part.groupby(part["question"].str.strip(), sort=False).indices.values())
        quota = round(N_VALIDATION * len(part) / len(df))
        shown = taken = 0
        for g in rng.permutation(len(groups)):
            rows = part.index[groups[g]]
            if shown < N_EXAMPLES and len(rows) == 1:
                split[rows], shown = "examples", shown + 1
            elif taken + len(rows) <= quota:
                split[rows], taken = "validation", taken + len(rows)
    return split


def frame(split: str) -> pd.DataFrame:
    raw = ZIP.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError(f"{ZIP}: sha256 is not {SHA256}")
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for s in SUBSETS:
            for level in LEVELS[s]:
                folder = f"{level}_knowledge" if s == "Theory" else level
                for line in z.read(f"mathbench_v1/{folder}/single_choice_en.jsonl").decode("utf-8").splitlines():
                    rows.append({**json.loads(line), "subset": s, "level": level})
    df = pd.DataFrame(rows)
    df["answer"] = df["answer"].map(LETTERS.index)
    return df[_splits(df) == split].reset_index(drop=True)


def load(split: str) -> tuple[list[str], list[list[str]], np.ndarray]:
    df = frame(split)
    return df["question"].tolist(), df["options"].tolist(), df["answer"].to_numpy(np.int64)


def subset(split: str) -> np.ndarray:
    return frame(split)["subset"].to_numpy()


def examples() -> str:
    """The "examples" split with answers, the only MathBench items a writer sees."""
    df = frame("examples")
    blocks = []
    for s, q, c, a in zip(df["subset"], df["question"], df["options"], df["answer"]):
        blocks.append(f"[{s}] " + q.strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, c))
                      + f"\nAnswer: {LETTERS[a]}")
    return "\n\n".join(blocks)

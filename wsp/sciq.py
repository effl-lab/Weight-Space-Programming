"""SciQ as a generation task: a science exam question, answered in words without options. Host only.

examples    10 training questions with their answers: the only SciQ items a writer sees
validation  the 1,000 validation questions
test        the 1,000 test questions, the reported split
The answer of an item is its `correct_answer` (1.5 words on average, at most 5 words for 99.7% of the items); the
distractors and the support paragraph of the dataset are not used.
"""
import glob
from functools import lru_cache

import numpy as np

from wsp import paths

HF = f"{paths.HF_HOME}/datasets/allenai___sciq"
SPLIT_SEED = 0
N_EXAMPLES = 10


@lru_cache(maxsize=None)
def _frame(split: str):
    import pyarrow as pa
    files = sorted(glob.glob(f"{HF}/**/*{split}*.arrow", recursive=True))
    if len(files) != 1:
        raise FileNotFoundError(f"SciQ {split}: {len(files)} files in {HF}")
    with pa.memory_map(files[0]) as source:  # the cache file of `datasets`, an Arrow stream
        return pa.ipc.open_stream(source).read_all().to_pandas()


def load(split: str) -> tuple[list[str], list[list[str]]]:
    """(questions, gold answers); every item has one gold answer."""
    if split == "examples":
        df = _frame("train")
        df = df.iloc[np.random.RandomState(SPLIT_SEED).permutation(len(df))[:N_EXAMPLES]]
    else:
        df = _frame(split)
    return [q.strip() for q in df["question"]], [[a.strip()] for a in df["correct_answer"]]


def examples() -> str:
    """The "examples" split with answers, the only SciQ items a writer sees."""
    return "\n\n".join(f"Question: {q}\nAnswer: {a[0]}" for q, a in zip(*load("examples")))

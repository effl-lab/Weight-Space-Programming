"""SQuAD 2.0: a question about a Wikipedia paragraph, answered by a span of the paragraph or not answerable from it.
Host only. With version "1.1" every function reads SQuAD 1.1 instead, where every item has an answer (10 answerable
examples, 1,000 validation items from other articles, and the 10,570 dev items as test).

examples    10 training items, 5 answerable and 5 unanswerable: the only SQuAD items a writer sees
validation  1,000 training items from other articles than the examples
test        the dev set, 11,873 items (5,945 unanswerable), the reported split
An item's gold answers are the distinct answer texts of its annotators, [] when it is unanswerable.
"""
import hashlib
import json
from functools import lru_cache

import numpy as np

from wsp import free_model, paths

DIR = paths.DATA_ROOT / "squad"
FILES = {"train": ("train-v2.0.json", "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"),
         "dev": ("dev-v2.0.json", "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"),
         "train1.1": ("train-v1.1.json", "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955"),
         "dev1.1": ("dev-v1.1.json", "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9")}
SPLIT_SEED = 0
N_EXAMPLES = 5  # per kind
N_VALIDATION = 1000
EXAMPLE_WORDS = 120  # an example's paragraph is at most this long, to keep the task text short
SUBSETS = ("answerable", "unanswerable")


@lru_cache(maxsize=None)
def _items(part: str) -> tuple[dict, ...]:
    name, sha = FILES[part]
    raw = (DIR / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{DIR / name}: sha256 is not {sha}")
    out = []
    for article in json.loads(raw)["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answers = [] if qa.get("is_impossible") else list(dict.fromkeys(a["text"] for a in qa["answers"]))
                out.append({"id": qa["id"], "title": article["title"], "question": qa["question"],
                            "context": paragraph["context"], "answers": answers})
    return tuple(out)


@lru_cache(maxsize=None)
def _train_splits(version: str = "2.0") -> dict[str, tuple[dict, ...]]:
    """Examples: in the order of RandomState(SPLIT_SEED), the first N_EXAMPLES answerable and N_EXAMPLES unanswerable
    items (version 1.1: the first 2 N_EXAMPLES items) with a paragraph of at most EXAMPLE_WORDS words, each from a
    different article. Validation: the next N_VALIDATION items in that order from articles that have no example."""
    items = _items("train" if version == "2.0" else "train1.1")
    order = np.random.RandomState(SPLIT_SEED).permutation(len(items))
    examples, titles, taken = [], set(), {True: 0, False: 0}
    quota = {True: N_EXAMPLES, False: N_EXAMPLES} if version == "2.0" else {True: 2 * N_EXAMPLES, False: 0}
    for i in order:
        it = items[i]
        kind = bool(it["answers"])
        short = len(free_model.span_tokens(it["context"])) <= EXAMPLE_WORDS
        if taken[kind] < quota[kind] and short and it["title"] not in titles:
            examples.append(it)
            titles.add(it["title"])
            taken[kind] += 1
    validation = [items[i] for i in order if items[i]["title"] not in titles][:N_VALIDATION]
    return {"examples": tuple(examples), "validation": tuple(validation)}


def items(split: str, version: str = "2.0") -> tuple[dict, ...]:
    if split == "test":
        return _items("dev" if version == "2.0" else "dev1.1")
    return _train_splits(version)[split]


def load(split: str, version: str = "2.0") -> tuple[list[str], list[str], list[list[str]]]:
    """(questions, paragraphs, gold answers)."""
    rows = items(split, version)
    return [r["question"] for r in rows], [r["context"] for r in rows], [r["answers"] for r in rows]


def subset(split: str) -> np.ndarray:
    return np.array([SUBSETS[0] if r["answers"] else SUBSETS[1] for r in items(split)])


def examples(version: str = "2.0") -> str:
    """The "examples" split with answers, the only SQuAD items a writer sees."""
    return "\n\n".join(f"Paragraph: {r['context'].strip()}\nQuestion: {r['question'].strip()}\n"
                       f"Answer: {r['answers'][0] if r['answers'] else '(no answer)'}"
                       for r in items("examples", version))

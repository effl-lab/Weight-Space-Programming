"""The language-model task (<name>-text): a written model continues a text; what it is tested on is not told to the
writer. Host only.

The test texts are plain prompts of benchmark items, the ones a released base model is given without examples
(analysis.pretrained_open, zero-shot):
  arc, mmlu, math   baselines.arc_released.PLAIN_TEMPLATE; read: the larger of the first step's logits of the pieces
                    "a" and " a" (and so on) per letter, as the letter logits of a released model are read
  squad             "Passage: ..\\nQuestion: ..\\nAnswer:"; the continuation up to its line break (CAPS pieces at most)
  cnn               "Article: ..\\nSummary:", the article cut to ARTICLE_WORDS words
  sciq              "Question: ..\\nAnswer:"
The items are those of wsp.general's test split (squad and cnn: its first N_TEXT items) and SciQ's test split.
"""
import numpy as np

from baselines.arc_released import PLAIN_TEMPLATE
from wsp import free_model, general, sciq

CAPS = {"choose": 1, "squad": 24, "cnn": free_model.NEW_LEN, "sciq": 12}
N_TEXT = {"squad": 1000, "cnn": 500}
ARTICLE_WORDS = 400
PARTS = ("arc", "mmlu", "math", "squad", "cnn", "sciq")


def items(part: str) -> list[dict]:
    if part == "sciq":
        return [{"part": part, "kind": "sciq", "question": q, "gold": a} for q, a in zip(*sciq.load("test"))]
    rows = list(general.items("test", part))
    return rows[:N_TEXT[part]] if part in N_TEXT else rows


def prompt(item: dict) -> str:
    if item["kind"] == "choose":
        return PLAIN_TEMPLATE.format(question=item["question"], **dict(zip("ABCD", item["options"])))
    if item["kind"] == "answer":
        return f"Passage: {item['passage'].strip()}\nQuestion: {item['question']}\nAnswer:"
    if item["kind"] == "summarize":
        return f"Article: {' '.join(item['article'].split()[:ARTICLE_WORDS])}\nSummary:"
    return f"Question: {item['question']}\nAnswer:"


def cap(item: dict) -> int:
    return CAPS["choose" if item["kind"] == "choose" else item["part"]]


def load(split: str) -> list[dict]:
    if split != "test":
        raise KeyError(f"the language-model task has a test split only, not {split!r}")
    return [it for part in PARTS for it in items(part)]


def scores(rows: list[dict], first: np.ndarray, texts: list[str]) -> dict:
    """Per part: accuracy from the first step's letter logits, SQuAD F1, ROUGE-L. `accuracy` is the mean over arc,
    mmlu, math and writing (squad and cnn), as in the general task; sciq is reported apart."""
    parts = {}
    for name in PARTS:
        k = [i for i, it in enumerate(rows) if it["part"] == name]
        if rows[k[0]]["kind"] == "choose":
            letters = first[k][:, list(free_model.LETTER_COLUMNS)]
            pick = letters.reshape(len(k), 4, 2).max(2).argmax(1)
            tie = (letters.max(1) == letters.min(1)).mean()
            parts[name] = {"score": float(np.mean(pick == np.array([rows[i]["gold"] for i in k]))), "n": len(k), "ties": float(tie)}
        elif name == "cnn":
            parts[name] = {"score": float(np.mean([free_model.rouge_l(texts[i], rows[i]["gold"]) for i in k])), "n": len(k)}
        else:
            parts[name] = {"score": free_model.span_scores([texts[i] for i in k], [rows[i]["gold"] for i in k])["f1"], "n": len(k)}
        if rows[k[0]]["kind"] != "choose":
            parts[name]["empty"] = float(np.mean([not texts[i] for i in k]))
    groups = {g: float(np.mean([parts[p]["score"] for p in ps])) for g, ps in general.GROUPS.items()}
    return {"accuracy": float(np.mean(list(groups.values()))), "groups": groups, "parts": parts}

"""Check /srv/work/model: required files, the arch.py restrictions, and that it builds and runs.

It runs your model on /srv/work/my_questions.json if that file exists (a list of objects with "question",
"options" (four strings) and optionally "answer" (0-3)), otherwise on one placeholder question, and prints the
size of the weights. Questions in that file must be ones you made up. In the noq-arc task the file is not read.
In the py-arc task the model gets the texts themselves and the tool also prints the count of Python values in its
weights. In the fit-arc task the model is built and run with only its folder visible, as when it is scored, and
the folder may hold at most 1 GB of files.
In the design-cifar task it runs the model on /srv/work/my_images.npz instead if that file exists (`x` float32
(n, 3, 32, 32) in [0, 1], optionally `y` int64 labels), otherwise on one grey placeholder image; the images must be
ones you made yourself.
In a SQuAD task (<name>-sq) the items of my_questions.json are objects with "question", "context" (a paragraph) and
optionally "answer" (a string; "" when the paragraph does not answer the question), and the tool prints the model's
answer for each (and exact match and F1 when every item has an answer).
In a SQuAD 1.1 task (<name>-sq1) every item has an answer and the model always marks a span. In a generation task
(<name>-gen) the items are objects with "question" and optionally "answer" (a string), and the tool prints the
answer the model writes for each (and exact match and F1 when every item has an answer).
In the general task (<name>-all) the items are objects with "kind" and the texts of that kind: "choose" with
"question" and "options" (four strings), "answer" with "question" and "passage", "summarize" with "article";
optionally "gold" (what you expect, shown next to the output). The tool prints what the model writes for each.
In the language-model task (<name>-text) it reads /srv/work/my_texts.json instead, a list of strings, and prints the
first piece and the continuation the model writes after each (texts you made up).
With --practice (fb-arc task) it runs the model on /srv/work/data/practice.jsonl instead and prints the accuracy only.
Usage: python /srv/work/tools/check_model.py [--practice]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/srv/lib")
from wsp import free_model  # noqa: E402

MODEL = Path("/srv/work/model")
PRACTICE = Path("/srv/work/data/practice.jsonl")
N_CLASSES = 10


def fail(problems: list[str]) -> None:
    print("CHECK FAILED")
    print("\n".join(problems))
    sys.exit(1)


def check_questions(arch: str) -> None:
    mine = Path("/srv/work/my_questions.json")
    use_mine = mine.is_file() and arch != "noq-arc"
    items = json.loads(mine.read_text()) if use_mine else [{"question": "placeholder", "options": ["a", "b", "c", "d"]}]
    batch = ([i["question"] for i in items], [i["options"] for i in items])
    if arch == "py-arc":
        logits, size, problems = free_model.run_py(MODEL, {"probe": batch}, isolate=False)
    else:
        logits, size, problems = free_model.run(MODEL, {"probe": batch}, isolate=arch == "fit-arc",
                                                encoder=free_model.encode_of(arch),
                                                max_bytes=free_model.max_bytes_of(arch))
    if problems:
        fail(problems)
    if arch == "py-arc":
        print(f"CHECK OK\ntensor weights {size['weights']:,} ({size['nonzero']:,} non-zero), {size['python_values']:,} "
              f"Python values ({size['python_chars']:,} characters of strings)")
    else:
        print(f"CHECK OK\nweights {size['weights']:,} ({size['nonzero']:,} non-zero), vocabulary {len(free_model.vocab_of(MODEL)):,} words")
    answers = [i.get("answer") for i in items]
    if use_mine:
        for item, row, a in zip(items, logits["probe"], answers):
            tie = row.max() == row.min()
            print(("tie " if tie else f"pick {int(row.argmax())} ") + ("" if a is None else f"(answer {a}) ") + item["question"][:70])
        if all(a is not None for a in answers):
            print(json.dumps(free_model.accuracy(logits["probe"], np.array(answers))))


def check_gen() -> None:
    mine = Path("/srv/work/my_questions.json")
    items = json.loads(mine.read_text()) if mine.is_file() else [{"question": "placeholder"}]
    answers, size, problems = free_model.run_gen(MODEL, {"probe": [i["question"] for i in items]}, isolate=False)
    if problems:
        fail(problems)
    print(f"CHECK OK\nweights {size['weights']:,} ({size['nonzero']:,} non-zero), vocabulary {len(free_model.vocab_of(MODEL)):,} words")
    if mine.is_file():
        for item, a in zip(items, answers["probe"]):
            print(f"answer {a!r} " + (f"(expected {item['answer']!r}) " if "answer" in item else "") + item["question"][:70])
        if all("answer" in i for i in items):
            print(json.dumps(free_model.span_scores(answers["probe"], [[i["answer"]] for i in items])))


def check_all() -> None:
    mine = Path("/srv/work/my_questions.json")
    items = json.loads(mine.read_text()) if mine.is_file() else [{"kind": "summarize", "article": "placeholder"}]
    items = [{"part": i.get("part", "mine"), **i} for i in items]
    answers, size, problems = free_model.run_all(MODEL, {"probe": items}, isolate=False)
    if problems:
        fail(problems)
    vocab = free_model.vocab_of(MODEL)
    print(f"CHECK OK\nweights {size['weights']:,} ({size['nonzero']:,} non-zero), vocabulary {len(vocab):,} words")
    if mine.is_file():
        for item, ids, text in zip(items, answers["probe"], free_model.all_text(vocab, items, answers["probe"])):
            shown = {"choose": "question", "answer": "question", "summarize": "article"}[item["kind"]]
            print(f"{item['kind']}: {text!r} ({int((ids != 0).sum())} ids) "
                  + (f"(expected {item['gold']!r}) " if "gold" in item else "") + item[shown][:60])


def check_text() -> None:
    mine = Path("/srv/work/my_texts.json")
    texts = json.loads(mine.read_text()) if mine.is_file() else ["placeholder"]
    out, size, problems = free_model.run_lm(MODEL, {"probe": (texts, [free_model.NEW_LEN] * len(texts))}, isolate=False)
    if problems:
        fail(problems)
    pieces = free_model.piece_vocab(MODEL)
    print(f"CHECK OK\nweights {size['weights']:,} ({size['nonzero']:,} non-zero), {len(pieces):,} pieces")
    if mine.is_file():
        for text, ids, shown in zip(texts, out["probe"], free_model.piece_text(pieces, out["probe"])):
            first = pieces[int(ids[0]) - 1] if ids[0] > 0 else "(id 0)"
            print(f"{text[-60:]!r} -> first piece {first!r}, continuation {shown!r}")


def check_spans(forced: bool) -> None:
    mine = Path("/srv/work/my_questions.json")
    items = json.loads(mine.read_text()) if mine.is_file() else [{"question": "placeholder", "context": "a b c d"}]
    batch = ([i["question"] for i in items], [i["context"] for i in items])
    logits, size, problems = free_model.run_spans(MODEL, {"probe": batch}, isolate=False)
    if problems:
        fail(problems)
    print(f"CHECK OK\nweights {size['weights']:,} ({size['nonzero']:,} non-zero), vocabulary {len(free_model.vocab_of(MODEL)):,} words")
    if mine.is_file():
        answers = (free_model.span_decode_forced if forced else free_model.span_decode)(logits["probe"], batch[1])
        for item, a in zip(items, answers):
            print(f"answer {a!r} " + (f"(expected {item['answer']!r}) " if "answer" in item else "") + item["question"][:70])
        if all("answer" in i for i in items):
            print(json.dumps(free_model.span_scores(answers, [[i["answer"]] if i["answer"] else [] for i in items])))


def check_practice(arch: str) -> None:
    items = [json.loads(line) for line in PRACTICE.read_text().splitlines()]
    batch = ([i["question"] for i in items], [i["options"] for i in items])
    logits, _, problems = free_model.run(MODEL, {"probe": batch}, isolate=False, encoder=free_model.encode_of(arch))
    if problems:
        fail(problems)
    res = free_model.accuracy(logits["probe"], np.array([i["answer"] for i in items]))
    print(f"practice accuracy {res['accuracy']:.4f} on {res['n']} questions, {res['answered']:.4f} answered")


def check_images() -> None:
    mine = Path("/srv/work/my_images.npz")
    data = np.load(mine) if mine.is_file() else {"x": np.full((1, 3, 32, 32), 0.5, np.float32)}
    x = np.asarray(data["x"], np.float32)
    if x.ndim != 4 or x.shape[1:] != (3, 32, 32):
        fail([f"my_images.npz x has shape {x.shape}, not (n, 3, 32, 32)"])
    logits, size, problems = free_model.run_images(MODEL, {"probe": x}, N_CLASSES, isolate=False)
    if problems:
        fail(problems)
    print(f"CHECK OK\nweights {size['weights']:,} ({size['nonzero']:,} non-zero)")
    if mine.is_file():
        print("picks", logits["probe"].argmax(1).tolist())
        if "y" in data:
            print(json.dumps(free_model.accuracy(logits["probe"], np.asarray(data["y"]))))


if __name__ == "__main__":
    arch = json.loads(Path("/srv/work/arch_spec.json").read_text())["arch"]
    if sys.argv[1:] == ["--practice"]:
        check_practice(arch)
    else:
        if arch == "design-cifar":
            check_images()
        elif arch.endswith(("-sq", "-sq1")):
            check_spans(forced=arch.endswith("-sq1"))
        elif arch.endswith("-gen"):
            check_gen()
        elif arch.endswith("-all"):
            check_all()
        elif arch.endswith("-text"):
            check_text()
        else:
            check_questions(arch)

"""Datasets the writer can be given: classes, splits and the files a sandbox may mount."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from wsp import arc, data, general, mathbench, memory, mmlu, mmlu_all, sciq, squad, text_data, textlm
from wsp import vision_tokens as vt


@dataclass(frozen=True)
class Task:
    name: str
    classes: tuple[str, ...]
    data_dir: Path
    load: Callable  # split ("train" | "practice" | "test") -> (x, y)
    item: str  # what one example is called in writer-facing text
    open_files: tuple[str, ...]  # mounted read-only at /srv/work/data/<name> in open runs
    view_files: tuple[str, ...]  # the same for view runs; must not hold labels
    arch_files: tuple[str, ...] = ()  # copied to /srv/work/<name> in every run: part of the architecture
    subsets: tuple[str, ...] = ()  # named parts of the test split, reported separately
    subset: Callable | None = None  # split -> the subset name of every item


TASKS = {
    "cifar": Task("cifar", tuple(data.CLASSES), data.CIFAR_DIR, data.load, "images",
                  ("train.npz",), ("train_images.npz",)),
    "cifar_words": Task("cifar_words", tuple(data.CLASSES), data.CIFAR_DIR, data.load, "images",
                        (), ("train_images.npz", "train_words.npz"), ("patch_words_0.png", "patch_words_1.png")),
    "cifar_tokens": Task("cifar_tokens", tuple(data.CLASSES), vt.token_dir() / vt.TOKENIZER, vt.load, "images",
                         (), ("train_tokens.npz",), tuple(f"token_sheet_{i:02d}.png" for i in range(16))),
    "cifar_emb": Task("cifar_emb", tuple(data.CLASSES), vt.embed_dir(), vt.load_embeddings, "images", (), (),
                      tuple(f"dim_sheet_{i:02d}.png" for i in range(vt.VE_DIM // 16))),
    "agnews": Task("agnews", tuple(text_data.CLASSES), text_data.AGNEWS_DIR, text_data.load, "articles",
                   ("train.npz", "train.jsonl"), ("train_texts.jsonl",), ("vocab.txt",)),
    "agnews_cipher": Task("agnews_cipher", tuple(text_data.CLASSES), text_data.CIPHER_DIR, text_data.load_cipher,
                          "articles", ("train.npz",), ("train_ids.npz",)),
}
TASKS.update({f"mmlu-{s}": Task(f"mmlu-{s}", mmlu.LETTERS, Path(mmlu.SNAPSHOT), partial(mmlu.load, subject=s),
                               "questions", (), ()) for s in memory.SUBJECTS})
TASKS["arc"] = Task("arc", arc.LETTERS, Path(arc.SNAPSHOT), arc.load, "questions", (), (), subsets=arc.SUBSETS,
                    subset=arc.subset)
TASKS["mathbench"] = Task("mathbench", mathbench.LETTERS, mathbench.ZIP.parent, mathbench.load, "questions", (), (),
                          subsets=mathbench.SUBSETS, subset=mathbench.subset)
TASKS["mmlu_all"] = Task("mmlu_all", mmlu_all.LETTERS, Path(mmlu.SNAPSHOT), mmlu_all.load, "questions", (), (),
                         subsets=mmlu_all.SUBSETS, subset=mmlu_all.subset)
TASKS["squad"] = Task("squad", (), squad.DIR, squad.load, "questions", (), (), subsets=squad.SUBSETS, subset=squad.subset)
TASKS["squad1"] = Task("squad1", (), squad.DIR, partial(squad.load, version="1.1"), "questions", (), ())
TASKS["sciq_gen"] = Task("sciq_gen", (), Path(sciq.HF), sciq.load, "questions", (), ())
TASKS["general"] = Task("general", (), general.CNN_DIR, general.load, "items", (), ())
TASKS["text"] = Task("text", (), general.CNN_DIR, textlm.load, "texts", (), ())

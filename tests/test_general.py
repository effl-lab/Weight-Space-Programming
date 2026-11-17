import numpy as np

from harness import run, spec, task
from wsp import archs, free_model, general, tasks

# writes [B] for a choose row, the three tokens after [PASSAGE] for an answer row, the first three tokens of the
# article for a summarize row
MODEL = {
    "vocab.txt": "the\ncell\n",
    "build.py": "import torch\n\n\ndef build():\n    return {'v': torch.zeros(2)}\n",
    "arch.py": """import torch


def forward(w, x, a):
    n = x.shape[0]
    base = 9 + w['v'].shape[0]
    t = (a != 0).sum(1)
    kind = x[:, 0]
    passage = (x == 7).float().argmax(1)
    pick = (kind == 1).long() * 3 * (t == 0).long() \\
        + (kind == 6).long() * (base + passage + 1 + t) * (t < 3).long() \\
        + (kind == 8).long() * (base + 1 + t) * (t < 3).long()
    return torch.zeros(n, base + x.shape[1]).scatter_(1, pick[:, None], 1.0)
""",
}
ITEMS = [{"part": "arc", "kind": "choose", "question": "What is 2+2?", "options": ["3", "4", "5", "6"], "gold": 1},
         {"part": "squad", "kind": "answer", "question": "When was the battle?", "passage": "On 14 October 1066, they fought.",
          "gold": ["14 October 1066", "On 14"]},
         {"part": "cnn", "kind": "summarize", "article": "The U.S. team won. Fans cheered.", "gold": "the u s team won"}]


def test_runs_and_scores(tmp_path):
    for name, text in MODEL.items():
        (tmp_path / name).write_text(text)
    out, size, problems = free_model.run_all(tmp_path, {"probe": ITEMS}, isolate=False)
    assert problems == [] and out["probe"].shape == (3, free_model.A_LEN)
    ids = out["probe"]
    assert ids[0].tolist()[:2] == [3, 0] and (ids[1] != 0).sum() == 3 and (ids[2] != 0).sum() == 3
    texts = free_model.all_text(["the", "cell"], ITEMS, ids)
    assert texts == ["B", "On 14", "The U."]
    scores = free_model.all_scores(ITEMS, ids, texts)
    assert scores["parts"]["arc"]["score"] == 1.0 and scores["parts"]["squad"]["score"] == 1.0
    assert abs(scores["parts"]["cnn"]["score"] - 2 * 2 / (2 + 5)) < 1e-9
    assert "writing" not in scores["groups"] or abs(scores["groups"]["writing"] - (1.0 + 4 / 7) / 2) < 1e-9


def test_encode_and_text():
    vocab = ["when", "october"]
    x, cap = free_model.all_encode(vocab, ITEMS)
    assert x.shape == (3, free_model.X_LEN) and cap.tolist() == [1, 30, 64]
    assert x[0, 0] == 1 and 2 in x[0] and 5 in x[0] and x[1, 0] == 6 and x[1, 1] == 9 and x[2, 0] == 8
    source = free_model.all_source(ITEMS[1])
    base = free_model.N_MARKS + len(vocab)
    p = [k for k, s in enumerate(source) if s[0] == "1"][0]
    # digits are single tokens; adjacent copies are cut from the passage in one piece, a gap starts a new piece
    row = np.zeros((1, free_model.A_LEN), dtype=np.int64)
    row[0, :6] = [base + p, base + p + 1, base + p + 2, base + p + 4, 9, 4]
    assert free_model.all_text(vocab, [ITEMS[1]], row) == ["14 October 0 when C"]
    long = {"part": "mmlu", "kind": "choose", "question": "w " * 900 + "last", "options": ["a b"] * 4, "gold": 0}
    tokens = [s[0] for s in free_model.all_source(long)]
    assert len(tokens) == free_model.X_LEN and tokens[0] == "[CHOOSE]" and "last" in tokens and tokens[-3:] == ["[D]", "a", "b"]


def test_rouge_l():
    assert free_model.rouge_l("the cat sat", "the cat sat") == 1.0 and free_model.rouge_l("", "a") == 0.0
    assert abs(free_model.rouge_l("the cat sat down", "a cat was sat") - 2 * 2 / 8) < 1e-9


def test_registered_and_task_text():
    assert archs.task_of("design-all") is tasks.TASKS["general"] and run.fixed_arc("design-all") is None
    assert run.lineage_parent(spec.RunSpec("design-all", "blind", 0, 0, "opus")) is None
    assert spec.RunSpec("design-all", "blind", 0, 0, "sol").run_dir.parent.name == "runs_all_sol"
    text = task.render("design-all", "blind", 0)
    assert general.examples() in text and "def all_text" in text and "arch.forward(w, x[live[i:i + 64]], a[live[i:i + 64]])" in text
    counts = {p: len(general.items("examples", p)) for p in general.PARTS}
    assert counts == {"arc": 10, "mmlu": 10, "math": 10, "squad": 5, "cnn": 5}
    assert len({i["question"] for i in general.items("examples", "mmlu")}) == 10

import numpy as np

from harness import run, spec, task
from wsp import archs, free_model, tasks, textlm

# after "answer:" the model writes " b"; otherwise it repeats the last piece once and then breaks the line
MODEL = {
    "vocab.txt": "▁answer\n▁the\n",
    "build.py": "import torch\n\n\ndef build():\n    return {'v': torch.zeros(1 + 137 + 2)}\n",
    "arch.py": """import torch


def forward(w, x):
    V = w['v'].shape[0]
    last, before = x[:, -1], x[:, -2]
    asked = ((before == 138) & (last == 53)).long()
    again = (last == before).long()
    pick = asked * 81 + (1 - asked) * (again * 1 + (1 - again) * last)
    return torch.zeros(x.shape[0], V).scatter_(1, pick[:, None], 1.0)
""",
}


def test_pieces():
    pieces = [*free_model.BASE_PIECES, "▁answer", "▁the"]
    x = free_model.piece_encode(pieces, ["The Answer:\nThé"])
    row = [pieces[i - 1] for i in x[0].tolist() if i]
    assert row == ["▁the", "▁answer", ":", "↵", "▁t", "h", "?"] and x.shape == (1, free_model.CTX_LEN)
    ids = np.zeros((1, free_model.NEW_LEN), dtype=np.int64)
    ids[0, :4] = [139, 138, 1, 139]
    assert free_model.piece_text(pieces, ids) == ["the answer"]
    assert free_model.piece_problems(["▁the", "ok"]) == [] and len(free_model.piece_problems(["a", "x y", "Hi", "ok", "ok"])) == 2
    assert [free_model.BASE_PIECES[c] for c in free_model.LETTER_COLUMNS] == ["▁a", "a", "▁b", "b", "▁c", "c", "▁d", "d"]


def test_runs_and_scores(tmp_path):
    for name, text in MODEL.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    rows = [{"part": "arc", "kind": "choose", "question": "q?", "options": ["w", "x", "y", "z"], "gold": 1},
            {"part": "sciq", "kind": "sciq", "question": "what is it", "gold": ["b"]}]
    out, size, problems = free_model.run_lm(tmp_path, {"test": ([textlm.prompt(r) for r in rows], [textlm.cap(r) for r in rows])},
                                            isolate=False)
    assert problems == [] and out["test"].shape == (2, free_model.NEW_LEN) and out["test_first"].shape == (2, 137)
    letters = out["test_first"][:, list(free_model.LETTER_COLUMNS)]
    assert letters.reshape(2, 4, 2).max(2).argmax(1).tolist() == [1, 1]
    assert (out["test"][0] != 0).sum() == 1  # a choose item is continued by one piece
    assert free_model.piece_text(free_model.piece_vocab(tmp_path), out["test"]) == ["b", "b b"]


def test_registered_and_task_text():
    assert archs.task_of("design-text") is tasks.TASKS["text"] and run.fixed_arc("design-text") is None
    s = spec.RunSpec("design-text", "blind", 0, 0, "sol")
    assert s.run_dir.parent.name == "runs_text_sol" and run.lineage_parent(s) is None
    text = task.render("design-text", "blind", 0)
    assert "def piece_encode" in text and "arch.forward(w, x[live[i:i + 64]])" in text
    for told in ("question:", "answer:", "passage", "summar", "squad", "mmlu", "four option", "letter", "benchmark", "choice"):
        assert told not in text.lower(), told
    assert len(textlm.load("test")) == 21193

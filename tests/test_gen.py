import numpy as np

from harness import run, spec, task
from wsp import archs, free_model, given_archs, sciq, tasks

MODEL = {
    "vocab.txt": "energy\nsun\ncell\nwalls\n",
    "build.py": "import torch\n\n\ndef build():\n    return {'first': torch.tensor([0., 0., 0., 0., 5., 4.]),"
                " 'next': torch.tensor([[0., 0, 0, 0, 0, 0]] * 4 + [[0., 0, 0, 0, 0, 3], [1., 0, 0, 0, 0, 0]])}\n",
    # the first word is "cell" (id 4); after "cell" comes "walls" (id 5); after "walls" the answer ends
    "arch.py": "import torch\n\n\ndef forward(w, q, a):\n    n = (a != 0).sum(1)\n    last = a.gather(1, (n - 1).clamp(min=0)[:, None])[:, 0]\n"
               "    started = (n > 0).float()[:, None]\n    return w['first'][None] * (1 - started) + w['next'][last] * started\n",
}


def test_generation_runs_word_by_word(tmp_path):
    for name, text in MODEL.items():
        (tmp_path / name).write_text(text)
    answers, size, problems = free_model.run_gen(tmp_path, {"probe": ["What surrounds a plant cell?", "x"]}, isolate=False)
    assert problems == [] and answers == {"probe": ["cell walls", "cell walls"]} and size["weights"] == 42
    scores = free_model.span_scores(answers["probe"], [["cell walls"], ["the cell membrane"]])
    assert scores["exact"] == 0.5 and abs(scores["f1"] - (1 + 0.5) / 2) < 1e-9


def test_gen_text_and_encode():
    vocab = ["energy", "sun"]
    assert free_model.gen_text(vocab, np.array([[3, 2, 0, 3, 0], [0, 2, 2, 2, 2], [1, 2, 3, 2, 3]])) == \
        ["sun energy", "", "? energy sun energy sun"]
    q = free_model.gen_encode(vocab, ["The Sun's energy!"])
    assert q.shape == (1, free_model.GQ_LEN) and q[0, :4].tolist() == [1, 1, 2, 0]


def test_registered_routed_and_task_text():
    assert {a for a in archs.ARCHS if a.endswith("-gen")} == {"free-gen", "transformer-gen"}
    for a in ("free-gen", "transformer-gen"):
        assert archs.task_of(a) is tasks.TASKS["sciq_gen"]
        assert spec.RunSpec(a, "blind", 0, 0, "sol").run_dir.parent.name == "runs_gen_sol"
        assert run.lineage_parent(spec.RunSpec(a, "blind", 0, 0, "opus")) is None
        text = task.render(a, "blind", 0)
        assert sciq.examples() in text and "1000 held-out questions" in text and "def gen_text" in text
        assert "arch.forward(w, q[i:i + 64], a[i:i + 64])" in text and "{" not in text.split("```")[0]
    assert run.fixed_arc("free-gen") is None and run.fixed_arc("transformer-gen") == given_archs.GEN
    assert len(sciq.load("test")[0]) == 1000 and len(sciq.load("examples")[0]) == 10
    assert not set(sciq.load("examples")[0]) & set(sciq.load("test")[0])


def test_given_transformer_generates():
    import torch
    from wsp.given_archs import gen_transformer as arch
    assert not free_model.static_problems(given_archs.source(given_archs.GEN), "forward(w, q, a)")
    g = torch.Generator().manual_seed(0)
    w = {k: torch.randn(s, generator=g) * 0.1 for k, s in given_archs.shapes(given_archs.GEN, 9).items()}
    q = torch.from_numpy(free_model.gen_encode(["a"] * 7, ["what is a cell", "a"]))
    a = torch.tensor([[0, 0, 0, 0, 0], [3, 4, 0, 0, 0]])
    out = arch.forward(w, q, a)
    assert out.shape == (2, 9) and torch.isfinite(out).all()
    changed = arch.forward(w, q, torch.tensor([[0, 0, 0, 0, 0], [3, 5, 0, 0, 0]]))
    assert torch.allclose(out[0], changed[0]) and not torch.allclose(out[1], changed[1])

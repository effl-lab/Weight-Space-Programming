import numpy as np

from harness import run, spec, task
from wsp import archs, free_model, given_archs, squad, tasks


def test_splits():
    ex, val, test = (squad.items(s) for s in ("examples", "validation", "test"))
    assert len(ex) == 10 and sum(1 for r in ex if not r["answers"]) == 5
    assert len(val) == squad.N_VALIDATION and len(test) == 11873 and sum(1 for r in test if not r["answers"]) == 5945
    assert not {r["title"] for r in ex} & {r["title"] for r in val}
    assert not {r["id"] for r in ex + val} & {r["id"] for r in test}
    assert all(len(free_model.span_tokens(r["context"])) <= squad.EXAMPLE_WORDS for r in ex)
    assert squad.load("examples") == squad.load("examples")
    assert squad.examples().count("\nAnswer: (no answer)") == 5 and squad.examples().count("\nQuestion: ") == 10


def test_encode_gives_unknown_words_their_own_id():
    q, c = free_model.span_encode(["the", "norman"], ["Who was Rollo?"], ["The Norman leader, Rollo, came in 911."])
    assert q.shape == (1, free_model.SQ_LEN) and c.shape == (1, free_model.SC_LEN)
    assert c[0, 0] == 2 and c[0, 1] == 3 and c[0, 7] == 0
    assert q[0, 2] == c[0, 3] >= 4 and len({*c[0, :7].tolist()}) == 7
    assert c[0, 2:7].min() >= 4 and c[0, 2:7].max() < 4 + free_model.HASH_IDS


def test_decode_and_scores():
    text = "The Norman leader, Rollo, came in 911."
    logits = np.zeros((3, 2, free_model.SC_LEN + 1))
    logits[0, 0, 4], logits[0, 1, 4] = 3, 3            # word 3: Rollo
    logits[1, 0, 2], logits[1, 1, 3] = 2, 2            # words 1..2: Norman leader
    assert free_model.span_decode(logits, [text] * 3) == ["Rollo", "Norman leader", ""]
    logits[0, 0, 0] = 7                                 # "no answer" outweighs the span
    assert free_model.span_decode(logits[:1], [text]) == [""]
    r = free_model.span_scores(["the Rollo", "leader", "", "x"], [["Rollo"], ["Norman leader", "the leader of Normans"], [], []])
    assert r["exact"] == 0.5 and abs(r["f1"] - (1 + 2 / 3 + 1 + 0) / 4) < 1e-9
    assert r["answerable"]["n"] == 2 and r["unanswerable"] == {"exact": 0.5, "f1": 0.5, "n": 2}
    assert free_model.span_scores([""] * 2, [["a b"], []])["answerable"]["f1"] == 0.0


def test_registered_and_routed():
    assert {a for a in archs.ARCHS if a.endswith("-sq")} == {"free-sq", "transformer-sq"}
    for a in ("free-sq", "transformer-sq"):
        assert archs.task_of(a) is tasks.TASKS["squad"]
        assert spec.RunSpec(a, "blind", 0, 0, "opus").run_dir.parent.name == "runs_sq"
        assert spec.RunSpec(a, "blind", 0, 0, "sol").run_dir.parent.name == "runs_sq_sol"
        assert run.lineage_parent(spec.RunSpec(a, "blind", 0, 0, "opus")) is None
    assert run.fixed_arc("free-sq") is None and run.fixed_arc("transformer-sq") == given_archs.SPAN


def test_task_text():
    free, fixed = task.render("free-sq", "blind", 0), task.render("transformer-sq", "blind", 0)
    for text in (free, fixed):
        assert squad.examples() in text and "11873 held-out items" in text and "{" not in text.split("```")[0]
        assert "def span_decode" in text and "arch.forward(w, q[i:i + 64], c[i:i + 64])" in text
    assert "You choose the structure" in free and "defines `forward(w, q, c)`" in free
    assert given_archs.source(given_archs.SPAN).rstrip() in fixed and "embed: ('V + 65536', 64)" in fixed
    assert "You choose the structure" not in fixed


def test_given_transformer_runs():
    import torch
    from wsp.given_archs import span_transformer as arch
    assert not free_model.static_problems(given_archs.source(given_archs.SPAN), "forward(w, q, c)")
    g = torch.Generator().manual_seed(0)
    w = {k: torch.randn(s, generator=g) * 0.1 for k, s in given_archs.shapes(given_archs.SPAN, 12).items()}
    q, c = free_model.span_encode(["the"] * 10, ["who?", "what was it?"], ["a b c", "the thing was here"])
    out = arch.forward(w, torch.from_numpy(q), torch.from_numpy(c))
    assert out.shape == (2, 2, free_model.SC_LEN + 1) and torch.isfinite(out).all()


def test_squad11_splits_and_task():
    ex, val, test = (squad.items(s, "1.1") for s in ("examples", "validation", "test"))
    assert len(ex) == 10 and len(val) == 1000 and len(test) == 10570 and all(r["answers"] for r in ex + val + test)
    assert not {r["title"] for r in ex} & {r["title"] for r in val}
    assert {a for a in archs.ARCHS if a.endswith("-sq1")} == {"free-sq1", "transformer-sq1"}
    assert archs.task_of("free-sq1") is tasks.TASKS["squad1"] and tasks.TASKS["squad1"].load("test")[2] == [r["answers"] for r in test]
    assert spec.RunSpec("free-sq1", "blind", 0, 0, "opus").run_dir.parent.name == "runs_sq1"
    assert run.fixed_arc("free-sq1") is None and run.fixed_arc("transformer-sq1") == given_archs.SPAN
    free, fixed = task.render("free-sq1", "blind", 0), task.render("transformer-sq1", "blind", 0)
    for text in (free, fixed):
        assert "SQuAD 1.1" in text and "SQuAD 2.0" not in text and "10570 held-out items" in text
        assert "def span_decode_forced" in text and "def span_decode(" not in text and squad.examples("1.1") in text
        assert "unanswerable" not in text and "(no answer)" not in text
    assert "no answer" not in free


def test_forced_decode_ignores_the_first_slot():
    text = "The Norman leader, Rollo, came in 911."
    logits = np.zeros((2, 2, free_model.SC_LEN + 1))
    logits[0, 0, 4], logits[0, 1, 4], logits[0, 0, 0] = 3, 3, 99
    assert free_model.span_decode(logits, [text] * 2) == ["", ""]
    assert free_model.span_decode_forced(logits, [text] * 2) == ["Rollo", "The"]

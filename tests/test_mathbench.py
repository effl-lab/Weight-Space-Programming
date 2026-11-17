import hashlib
import importlib.util
import inspect
from collections import Counter

import numpy as np
import pytest

from harness import run, spec, task
from harness.run import REPO
from wsp import arc, archs, free_model, given_archs, mathbench, tasks

SPLITS = ("examples", "validation", "test")
GIVEN = ("mem_assoc2", "transformer", "transformer_d8", "transformer_d24")
ARC_TASK_SHA = {"mem_route-arc": "56f7f820d76f9988", "mem_attn-arc": "80933a7f67c9dc60",
                "mem_core-arc": "c224be6376c7442a", "mem_assoc2-arc": "6cb8aa7c5e3d5b7d",
                "mem_assoc-arc": "226efb3a3b9c499c", "free-arc": "7d81475cf13a0b03", "noq-arc": "300ddc20eef03883",
                "transformer-arc": "de7a6bb673ad4a0e"}


def test_split_sizes_and_items():
    counts = {}
    for s in SPLITS:
        questions, options, answers = mathbench.load(s)
        counts[s] = Counter(mathbench.subset(s))
        assert len(questions) == len(options) == len(answers) == sum(counts[s].values())
        assert answers.dtype == np.int64 and answers.min() >= 0 and answers.max() <= 3
        assert all(len(o) == 4 and all(isinstance(t, str) and t.strip() for t in o) for o in options)
        assert all(isinstance(q, str) and q.strip() for q in questions)
    assert counts["examples"] == {"Theory": 5, "Application": 5}
    assert counts["validation"] == {"Theory": 132, "Application": 68}
    assert counts["test"] == {"Theory": 744, "Application": 377}
    assert sum(sum(c.values()) for c in counts.values()) == 1331
    assert sum(c["Theory"] for c in counts.values()) == 881 and sum(c["Application"] for c in counts.values()) == 450


def test_levels():
    df = mathbench.frame("test")
    assert set(df["level"][df["subset"] == "Theory"]) == {"primary", "middle", "high", "college"}
    assert set(df["level"][df["subset"] == "Application"]) == {"middle", "high", "college"}


def test_no_question_text_in_two_splits():
    texts = [{q.strip() for q in mathbench.load(s)[0]} for s in SPLITS]
    assert not (texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2])
    assert sum(len(t) for t in texts) == 1320  # 11 question texts appear twice in the release


def test_split_is_fixed():
    h = hashlib.sha256()
    for s in SPLITS:
        questions, options, answers = mathbench.load(s)
        again = mathbench.load(s)
        assert again[0] == questions and again[1] == options and np.array_equal(again[2], answers)
        h.update(repr((s, questions, options, answers.tolist())).encode())
    assert h.hexdigest()[:16] == "6ded92c388c82bc0"


def test_question_text_is_unchanged():
    questions = mathbench.load("test")[0]
    assert any("<<<Answer>>>" in q for q in questions) and any("\\frac" in q for q in questions)


def test_examples_text():
    questions, options, answers = mathbench.load("examples")
    blocks = mathbench.examples().split("\n\n[")
    assert len(blocks) == 10 and sum(b.lstrip("[").startswith("Theory] ") for b in blocks) == 5
    assert mathbench.examples().startswith(f"[Theory] {questions[0].strip()}\nA. {options[0][0]}\nB. {options[0][1]}\n")
    assert mathbench.examples().endswith(f"\nD. {options[-1][3]}\nAnswer: {mathbench.LETTERS[answers[-1]]}")
    assert blocks[5].startswith("Application] ")


def test_registered_and_routed():
    assert {a for a in archs.ARCHS if a.endswith("-mb")} == {"free-mb", *(f"{n}-mb" for n in GIVEN)}
    assert tasks.TASKS["mathbench"].load is mathbench.load and tasks.TASKS["mathbench"].subsets == mathbench.SUBSETS
    assert tasks.TASKS["arc"].subsets == arc.SUBSETS and tasks.TASKS["agnews"].subsets == ()
    assert run.fixed_arc("free-mb") is None
    for a in ("free-mb", *(f"{n}-mb" for n in GIVEN)):
        assert archs.ARCHS[a].task == "mathbench" and archs.ARCHS[a].deliverable == free_model.DELIVERABLE
        assert archs.task_of(a).name == "mathbench"
        assert spec.RunSpec(a, "blind", 0, 0, "opus").run_dir.parent.name == "runs_mb"
        assert spec.RunSpec(a, "blind", 0, 0, "sol").run_dir.parent.name == "runs_mb_sol"
        assert run.lineage_parent(spec.RunSpec(a, "blind", 0, 0, "opus")) is None
        assert run.allowed_data(spec.RunSpec(a, "blind", 0, 0, "opus")) == set()
    for n in GIVEN:
        assert run.fixed_arc(f"{n}-mb") == n
    assert spec.RunSpec("mem_assoc2-arc", "blind", 0, 0, "opus").run_dir.parent.name == "runs_arc"


def test_arc_task_texts_unchanged():
    for a, sha in ARC_TASK_SHA.items():
        assert hashlib.sha256(task.render_arc(a, None).encode()).hexdigest()[:16] == sha, a


def test_given_task_text():
    text = task.render("mem_assoc2-mb", "blind", 0)
    assert given_archs.source("mem_assoc2").rstrip() in text and given_archs.shape_table("mem_assoc2") in text
    assert "MathBench" in text and "ARC" not in text and "science" not in text
    flat = " ".join(text.split())
    assert text.startswith("# Task: write the weights of a model for MathBench mathematics questions\n")
    assert ("You will write every weight of a model that answers multiple-choice mathematics questions of MathBench, "
            "primary school to college level, theory questions (definitions, formulas, properties) and application "
            "questions (problems to solve) together. The structure is fixed:") in flat
    assert ("Five questions from each set, with answers. These are the only MathBench questions you see; the test "
            "questions are different.") in flat
    assert "\n- No MathBench data is available to you apart from the ten questions above.\n" in text
    assert "accuracy on the 1121 held-out test questions" in text
    assert 'MATH_TOKEN = re.compile(r"[a-z\']+|\\d|\\\\[a-z]+|[^\\sa-z0-9\']")\nQ_LEN, O_LEN = 128, 48\n' in text
    assert inspect.getsource(free_model.math_words) in text and inspect.getsource(free_model.math_encode) in text
    assert "[a-z0-9']+" not in text and "def words(" not in text and "def encode(" not in text
    assert task.render("free-mb", "blind", 0).startswith(
        "# Task: design and write a model for MathBench mathematics questions\n")
    with pytest.raises(ValueError):
        task.render("mem_assoc2-mb", "view", 0)


def test_task_text_differs_from_arc_only_in_the_benchmark():
    """An -mb text turns into the -arc text of the same architecture by swapping the four benchmark phrases, the
    example questions, the number of test questions, the tokenizer (its pattern, words and encode) and the note on
    vocab.txt lines."""
    mb, ar = task.BENCHMARK["mathbench"], task.BENCHMARK["arc"]
    swaps = [(mb["subject"], ar["subject"]), (mb["scope"], ar["scope"]), (mb["examples_note"], ar["examples_note"]),
             (f"- No {mb['name']} data is", f"- No {ar['name']} data is"), (mathbench.examples(), arc.examples()),
             (mb["vocab_note"], ar["vocab_note"]),
             (f"accuracy on the {len(mathbench.load('test')[2])} held-out",
              f"accuracy on the {len(arc.load('test')[2])} held-out"),
             (f'MATH_TOKEN = re.compile(r"{free_model.MATH_TOKEN.pattern}")',
              f'TOKEN = re.compile(r"{free_model.TOKEN.pattern}")'),
             (inspect.getsource(free_model.math_words) + "\n\n" + inspect.getsource(free_model.math_encode).rstrip(),
              inspect.getsource(free_model.words) + "\n\n" + inspect.getsource(free_model.encode).rstrip())]
    for n in ("free", *GIVEN):
        text = task.render(f"{n}-mb", "blind", 0)
        for old, new in swaps:
            assert text.count(old) == 1
            text = text.replace(old, new)
        assert text == task.render_arc(f"{n}-arc", None)


def test_prepare_writes_the_given_arch(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run, "access_token_credentials", lambda key: {"claudeAiOauth": {}})
    s = spec.RunSpec("transformer_d24-mb", "blind", 0, 0, "fable")
    work = run.prepare(s, None) / "work"
    assert work.parent.parent.name == "runs_mb_fable"
    assert (work / "TASK.md").read_bytes().decode() == task.render("transformer_d24-mb", "blind", 0)
    assert (work / "model" / "arch.py").read_text() == given_archs.source("transformer_d24")
    assert [p.name for p in (work / "tools").iterdir()] == ["check_model.py"]
    assert not (work / "data").exists() and not (work / "parent").exists()
    free = run.prepare(spec.RunSpec("free-mb", "blind", 0, 0, "fable"), None) / "work"
    assert not (free / "model").exists() and "You choose the structure" in (free / "TASK.md").read_text()


def test_score_reports_both_subsets(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    encoders = []
    monkeypatch.setattr(free_model, "run", lambda folder, batches, encoder, max_bytes: encoders.append(encoder) or (
        {k: np.zeros((len(q), 4)) for k, (q, _) in batches.items()}, {"weights": 0, "nonzero": 0}, []))
    s = spec.RunSpec("mem_assoc2-mb", "blind", 0, 0, "opus")
    folder = s.run_dir / "work" / free_model.DELIVERABLE
    folder.mkdir(parents=True)
    (folder / "vocab.txt").write_text("sum\n")
    (folder / "arch.py").write_text(given_archs.source("mem_assoc2") + "\n")
    assert run.score_free(s, s.run_dir) == {"valid": False, "problems": ["the given arch.py was changed"]}
    (folder / "arch.py").write_text(given_archs.source("mem_assoc2"))
    res = run.score_free(s, s.run_dir)
    assert res["valid"] and res["test"]["n"] == 1121 and res["validation"]["n"] == 200
    assert encoders == [free_model.math_encode]
    assert list(res["test_by_subset"]) == ["Theory", "Application"]
    assert [res["test_by_subset"][k] for k in mathbench.SUBSETS] == [
        {"accuracy": 0.25, "answered": 0.0, "accuracy_answered": 0.0, "n": 744},
        {"accuracy": 0.25, "answered": 0.0, "accuracy_answered": 0.0, "n": 377}]


def test_math_tokens():
    assert free_model.math_words("-0.895") == ["-", "0", ".", "8", "9", "5"]
    assert free_model.math_words("\\frac{1}{2}") == ["\\frac", "{", "1", "}", "{", "2", "}"]
    assert free_model.math_words("f'(X) = 3x^2 \\Delta \\\\ it's 12,500 \u2264 \u03c0") == [
        "f'", "(", "x", ")", "=", "3", "x", "^", "2", "\\delta", "\\", "\\", "it's", "1", "2", ",", "5", "0", "0",
        "\u2264", "\u03c0"]


def test_math_encode_keeps_the_sign():
    vocab = ["-", ".", "0", "5", "8", "9"]
    options = [["-0.895", "0.895", "0", "895"]]
    q, o = free_model.math_encode(vocab, ["-0.895?"], options)
    assert q.shape == (1, free_model.Q_LEN) and o.shape == (1, 4, free_model.O_LEN) and o.dtype == np.int64
    assert q[0, :7].tolist() == [2, 4, 3, 6, 7, 5, 1] and not q[0, 7:].any()
    assert o[0, :, :6].tolist() == [[2, 4, 3, 6, 7, 5], [4, 3, 6, 7, 5, 0], [4, 0, 0, 0, 0, 0], [6, 7, 5, 0, 0, 0]]
    arc_o = free_model.encode(vocab, ["-0.895?"], options)[1]
    assert np.array_equal(arc_o[0, 0], arc_o[0, 1]) and arc_o[0, 0, :3].tolist() == [4, 1, 0]


def test_options_that_differ_by_sign_get_different_ids():
    pairs = [(x.strip(), "-" + x.strip()) for s in ("validation", "test") for opts in mathbench.load(s)[1]
             for x in opts if x.strip() and "-" + x.strip() in [y.strip() for y in opts]]
    assert len(pairs) == 33
    vocab = sorted({w for pair in pairs for x in pair for w in free_model.math_words(x)})
    for plus, minus in pairs:
        o = free_model.math_encode(vocab, [""], [[plus, minus, "", ""]])[1]
        assert not np.array_equal(o[0, 0], o[0, 1]) and 1 not in o
    assert sum(free_model.words(plus) == free_model.words(minus) for plus, minus in pairs) == 33


def test_vocab_lines_keep_every_token(tmp_path):
    """vocab_of strips each line: no MathBench token is changed, dropped or merged by that."""
    tokens = sorted({w for s in SPLITS for q, opts, _ in zip(*mathbench.load(s)) for x in (q, *opts)
                     for w in free_model.math_words(x)})
    assert {"-", ".", "#", "'", '"', "\\", "\\frac", "{"} <= set(tokens) and any(not w.isascii() for w in tokens)
    (tmp_path / "vocab.txt").write_text("\n".join(tokens) + "\n")
    assert free_model.vocab_of(tmp_path) == tokens


def test_encode_of_architecture():
    for a in archs.ARCHS:
        assert free_model.encode_of(a) is (free_model.math_encode if a.endswith("-mb") else free_model.encode)


def test_arc_encode_unchanged():
    """Ids of wsp.free_model.encode as of commit 696910a."""
    vocab = ["what's", "3", "5", "kg", "of", "h2o", "at", "10", "it's", "2", "500", "j", "don't", "x", "ray", "1e", "n",
             "a"]
    questions = ["What's 3.5 kg of H2O at -10 degrees? (approx.)", "N/A"]
    options = [["It's 2,500 J", "don't know", "x-ray: 1e-3", "N/A"], ["-0.895", "0.895", "'", ""]]
    q, o = free_model.encode(vocab, questions, options)
    assert q.shape == (2, 128) and o.shape == (2, 4, 48) and q.dtype == o.dtype == np.int64
    assert [r[r > 0].tolist() for r in q] == [[2, 3, 4, 5, 6, 7, 8, 9, 1, 1], [18, 19]]
    assert [[r[r > 0].tolist() for r in item] for item in o] == [
        [[10, 11, 12, 13], [14, 1], [15, 16, 17, 3], [18, 19]], [[1, 1], [1, 1], [1], []]]
    assert not q[0, 10:].any() and not o[0, 0, 4:].any()
    assert free_model.words("It's 2,500 J; x-ray: 1e-3 (N/A)") == ["it's", "2", "500", "j", "x", "ray", "1e", "3", "n", "a"]


def test_check_model_encodes_with_the_architecture_tokenizer(monkeypatch, capsys):
    spec_ = importlib.util.spec_from_file_location("check_model", REPO / "harness" / "tools" / "check_model.py")
    tool = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(tool)
    encoders = []
    monkeypatch.setattr(tool.free_model, "run", lambda folder, batches, isolate, encoder, max_bytes: (
        encoders.append(encoder) or ({"probe": np.zeros((1, 4))}, {"weights": 0, "nonzero": 0}, [])))
    monkeypatch.setattr(tool.free_model, "vocab_of", lambda folder: [])
    tool.check_questions("mem_assoc2-mb")
    tool.check_questions("mem_assoc2-arc")
    assert encoders == [free_model.math_encode, free_model.encode] and capsys.readouterr().out.count("CHECK OK") == 2

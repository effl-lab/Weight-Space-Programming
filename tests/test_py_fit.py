import importlib.util
import inspect
import json
import subprocess

import pytest

from harness import run, spec, task
from harness.run import REPO
from wsp import arc, archs, free_model, lineage


def _tool():
    module = importlib.util.spec_from_file_location("check_model", REPO / "harness" / "tools" / "check_model.py")
    tool = importlib.util.module_from_spec(module)
    module.loader.exec_module(tool)
    return tool


def _folder(path, build, arch, vocab=None):
    path.mkdir()
    (path / "build.py").write_text(build)
    (path / "arch.py").write_text(arch)
    if vocab is not None:
        (path / "vocab.txt").write_text(vocab)
    return path


def test_registered_as_generation_one_free_designs():
    for v in ("py", "fit"):
        a = f"{v}-arc"
        assert lineage.arc_variant(a) == v and archs.ARCHS[a].task == "arc" and archs.ARCHS[a].deliverable == "model"
        assert lineage.variant_lineage(a) is None and f"{v}2-arc" not in archs.ARCHS and run.fixed_arc(a) is None
        assert run.lineage_parent(spec.RunSpec(a, "blind", 0, 0, "opus")) is None
        assert spec.RunSpec(a, "blind", 0, 0, "opus").run_dir.parent.name == "runs_arc"


def test_py_task_text_changes_only_the_model_contract():
    text, free = task.render("py-arc", "blind", 0), task.render("free-arc", "blind", 0)
    names = lambda s: ", ".join(f"`{n}`" for n in sorted(s))  # noqa: E731
    free_arch = task.ARC_FREE_ARCH.format(q_len=free_model.Q_LEN, o_len=free_model.O_LEN,
                                          names=names(free_model.ARCH_FORBIDDEN_NAMES))
    py_arch = task.PY_ARCH.format(modules=names(free_model.PY_FORBIDDEN_MODULES),
                                  names=names(free_model.PY_FORBIDDEN_NAMES), attrs=names(free_model.PY_FORBIDDEN_ATTRS))
    inputs = task.ARC_INPUTS.format(token='TOKEN = re.compile(r"[a-z0-9\']+")', q_len=free_model.Q_LEN,
                                    o_len=free_model.O_LEN, encode=inspect.getsource(free_model.words) + "\n\n"
                                    + inspect.getsource(free_model.encode).rstrip())
    expected = (free.replace(task.ARC_VOCAB_ITEM.format(vocab_note=""), "")
                .replace(task.ARC_BUILD_ITEM, task.PY_BUILD_ITEM).replace(free_arch, py_arch)
                .replace(inputs, task.PY_INPUTS).replace(free_model.RUNNER.rstrip(), free_model.PY_RUNNER.rstrip()))
    assert expected != free and text == expected
    assert "vocab.txt" not in text and "word id" not in text and "tensor program" not in text
    assert "The forward pass may be any Python program over these texts." in " ".join(text.split())


def test_fit_task_text_changes_only_the_fitting_rule():
    text, free = task.render("fit-arc", "blind", 0), task.render("free-arc", "blind", 0)
    pairs = ((task.ARC_TRAINED, task.FIT_TRAINED), (task.ARC_SIZE, task.FIT_SIZE.format(gb=1)),
             (task.ARC_ONLY_EXAMPLES, task.FIT_EXAMPLES), (task.ARC_RULE_FIT, task.FIT_RULE_FIT),
             (task.ARC_RULE_DATA, task.ARC_RULE_DATA_FIT.format(n=3358)))
    expected = free
    for old, new in pairs:
        assert free.count(old) == 1 and text.count(new) == 1
        expected = expected.replace(old, new)
    assert text == expected
    flat = " ".join(text.split())
    assert "the folder /srv/work/model may hold at most 1 GB of files." in flat
    assert "including gradient descent" in flat and "must not read arc_train.jsonl" in flat
    assert task.ARC_FREE_ARCH.format(q_len=128, o_len=48, names=", ".join(
        f"`{n}`" for n in sorted(free_model.ARCH_FORBIDDEN_NAMES))) in text


@pytest.mark.parametrize("source", [
    "import re, collections, math, json\nimport numpy as np\nimport torch\nfrom torch import nn\n\n\n"
    "class Scorer:\n    pass\n\n\ndef forward(w, questions, options):\n"
    "    out = []\n    for q, opts in zip(questions, options):\n"
    "        if len(q) > 3 and q:\n            out.append([float(len(o)) for o in opts])\n"
    "        else:\n            out.append(list(map(lambda o: 0.0, opts)))\n    return out\n",
])
def test_py_static_check_allows_any_python(source):
    assert free_model.py_static_problems(source) == []


@pytest.mark.parametrize("line, problem", [
    ("import os", "import os not allowed"),
    ("import os.path", "import os.path not allowed"),
    ("from subprocess import run", "import subprocess not allowed"),
    ("import socket", "import socket not allowed"),
    ("import sys", "import sys not allowed"),
    ("import pathlib", "import pathlib not allowed"),
    ("import requests", "import requests not allowed"),
    ("from . import helper", "import . not allowed"),
    ("x = open('vocab.txt').read()", "name open not allowed"),
    ("x = eval('1')", "name eval not allowed"),
    ("exec('x = 1')", "name exec not allowed"),
    ("c = compile('1', 'f', 'eval')", "name compile not allowed"),
    ("m = __import__('os')", "name __import__ not allowed"),
    ("g = getattr(math, 'pi')", "name getattr not allowed"),
    ("import numpy as np\nx = np.load('w.npy')", "attribute .load not allowed"),
    ("import torch\nx = torch.load('w.pt')", "attribute .load not allowed"),
    ("k = ().__class__", "attribute .__class__ not allowed"),
])
def test_py_static_check_refuses(line, problem):
    problems = free_model.py_static_problems(line + "\n\n\ndef forward(w, questions, options):\n    return [[0] * 4]\n")
    assert any(problem in p for p in problems), problems


def test_py_static_check_needs_forward():
    assert free_model.py_static_problems("def score(w, q, o):\n    return 0\n") == [
        "arch.py: no top-level forward(w, questions, options)"]


def test_run_py_gets_texts_and_counts_python_values(tmp_path):
    folder = _folder(tmp_path / "model",
                     "import torch\n\n\ndef build():\n    return {'a': ['x', 'yz'], 'n': 3, 't': torch.tensor([0.0, 2.0])}\n",
                     "import torch\n\n\ndef forward(w, questions, options):\n"
                     "    return torch.tensor([[float(o in q) for o in opts] for q, opts in zip(questions, options)])\n")
    batch = (["Is water wet?", "Is rock hard?"], [["dry", "wet", "cold", "hot"], ["soft", "hard", "red", "blue"]])
    logits, size, problems = free_model.run_py(folder, {"probe": batch}, isolate=False)
    assert problems == [] and logits["probe"].argmax(1).tolist() == [1, 1]
    assert size == {"weights": 2, "nonzero": 1, "python_values": 6, "python_chars": 6}
    for out in ("np.zeros((len(questions), 4))", "[[0, 1, 2, 3] for _ in questions]"):
        (folder / "arch.py").write_text(f"import numpy as np\n\n\ndef forward(w, questions, options):\n    return {out}\n")
        assert free_model.run_py(folder, {"probe": batch}, isolate=False)[2] == []
    (folder / "arch.py").write_text("def forward(w, questions, options):\n    return [[0, 1, 2]]\n")
    assert "not (2, 4)" in free_model.run_py(folder, {"probe": batch}, isolate=False)[2][0]
    (folder / "build.py").write_text("import re\n\n\ndef build():\n    return {'p': re.compile('a')}\n")
    assert "w holds a Pattern" in free_model.run_py(folder, {"probe": batch}, isolate=False)[2][0]
    (folder / "arch.py").write_text("import os\n\n\ndef forward(w, questions, options):\n    return [[0] * 4]\n")
    assert free_model.run_py(folder, {"probe": batch}, isolate=False)[2] == ["arch.py: import os not allowed (line 1)"]


def test_check_tool_runs_py_on_texts(tmp_path, monkeypatch, capsys):
    tool = _tool()
    model = _folder(tmp_path / "model", "def build():\n    return {'words': ['water', 'ice']}\n",
                    "def forward(w, questions, options):\n    return [[float(o in w['words']) for o in opts] for opts in options]\n")
    monkeypatch.setattr(tool, "MODEL", model)
    tool.check_questions("py-arc")
    assert capsys.readouterr().out == ("CHECK OK\ntensor weights 0 (0 non-zero), 3 Python values "
                                       "(13 characters of strings)\n")


def test_check_tool_isolates_and_caps_fit_only(monkeypatch, capsys):
    tool, calls = _tool(), []
    monkeypatch.setattr(tool.free_model, "run", lambda *a, **k: calls.append(k) or (None, {}, ["stop"]))
    for arch in ("fit-arc", "free-arc"):
        with pytest.raises(SystemExit):
            tool.check_questions(arch)
    assert [(c["isolate"], c["max_bytes"]) for c in calls] == [(True, free_model.FIT_MAX_BYTES), (False, None)]


def test_folder_cap(tmp_path):
    folder = _folder(tmp_path / "model", "def build():\n    return {}\n", "def forward(w, q, o):\n    return o\n", "a\n")
    (folder / "w.npy").write_bytes(b"0" * 1000)
    assert free_model.max_bytes_of("fit-arc") == free_model.FIT_MAX_BYTES == 10 ** 9
    assert free_model.max_bytes_of("free-arc") is None and free_model.max_bytes_of("py-arc") is None
    assert free_model.folder_problems(folder, max_bytes=10 ** 6) == []
    size = sum(p.stat().st_size for p in folder.iterdir())
    assert free_model.folder_problems(folder, max_bytes=1000) == [
        f"model/ holds {size:,} bytes of files, more than 1,000"]


def test_prepare_writes_training_questions_for_fit_only(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run, "access_token_credentials", lambda key: {"claudeAiOauth": {}})
    fit = spec.RunSpec("fit-arc", "blind", 0, 0, "fable")
    work = run.prepare(fit, None) / "work"
    rows = [json.loads(line) for line in (work / "data" / "arc_train.jsonl").read_text().splitlines()]
    questions, options, answers = arc.load("train")
    assert rows == arc.training() and len(rows) == 3358 == len(questions)
    assert [r["answer"] for r in rows] == ["ABCD"[a] for a in answers] and [r["options"] for r in rows] == options
    assert [p.name for p in (work / "data").iterdir()] == ["arc_train.jsonl"]
    assert (work / "TASK.md").read_text() == task.render("fit-arc", "blind", 0)
    assert run.allowed_data(fit) == {"/srv/work/data/arc_train.jsonl"} and run.mounted_data(fit) == set()
    for a in ("free-arc", "py-arc"):
        s = spec.RunSpec(a, "blind", 0, 0, "fable")
        assert not (run.prepare(s, None) / "work" / "data").exists() and run.allowed_data(s) == set()


def _self_test(monkeypatch, tmp_path, arch, files):
    out = "".join(f"file=/srv/work/data/{f}\n" for f in files) + "done\n"
    monkeypatch.setattr(run, "sandbox_argv", lambda s, d: [])
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    return run.self_test(spec.RunSpec(arch, "blind", 0, 0, "sol"), tmp_path)


def test_self_test_allows_exactly_the_training_file_for_fit(tmp_path, monkeypatch):
    assert _self_test(monkeypatch, tmp_path, "fit-arc", ["arc_train.jsonl"]) == []
    assert _self_test(monkeypatch, tmp_path, "fit-arc", []) != []
    assert _self_test(monkeypatch, tmp_path, "fit-arc", ["arc_train.jsonl", "test.jsonl"]) != []
    assert _self_test(monkeypatch, tmp_path, "fit-arc", ["practice.jsonl"]) != []
    for a in ("free-arc", "py-arc", "fb-arc"):
        assert _self_test(monkeypatch, tmp_path, a, ["arc_train.jsonl"]) != [], a
    assert _self_test(monkeypatch, tmp_path, "free-arc", []) == []

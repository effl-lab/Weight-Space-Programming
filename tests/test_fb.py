import hashlib
import importlib.util
import json

from harness import run, spec, task
from harness.run import REPO
from wsp import arc, archs, lineage


def test_registered_as_a_free_design_without_parent():
    assert lineage.arc_variant("fb-arc") == "fb" and archs.ARCHS["fb-arc"].task == "arc"
    assert lineage.variant_lineage("fb-arc") is None and "fb2-arc" not in archs.ARCHS
    assert run.fixed_arc("fb-arc") is None
    assert run.lineage_parent(spec.RunSpec("fb-arc", "blind", 0, 0, "opus")) is None
    assert spec.RunSpec("fb-arc", "blind", 0, 0, "opus").run_dir.parent.name == "runs_arc"


def test_task_text_differs_from_free_in_two_places():
    text, free = task.render("fb-arc", "blind", 0), task.render_arc("free-arc", None)
    tools = task.ARC_TOOLS_FB.format(n=512)
    assert text.count(tools) == 1 and text.count(task.ARC_RULE_DATA_FB) == 1 and free.count(task.ARC_RULE_DATA) == 1
    assert text.replace(tools, "").replace(task.ARC_RULE_DATA_FB, task.ARC_RULE_DATA) == free
    assert task.ARC_TOOLS + tools + "\n\n## Rules\n" in text
    flat = " ".join(text.split())
    assert ("`python /srv/work/tools/check_model.py --practice` runs the model on the 512 ARC training questions in "
            "/srv/work/data/practice.jsonl (question, options, answer) and prints its accuracy on them. You may read "
            "that file and use this as often as you like.") in flat
    assert ("- No ARC data is available to you apart from the ten questions above and /srv/work/data/practice.jsonl. "
            "Do not copy practice questions or their answers into the model, and do not set anything in the model by "
            "an automatic search or fit against them; change the model only by your own reasoning about what you "
            "measured. - Do not access the internet") in flat


def test_practice_questions():
    rows = arc.practice()
    assert len(rows) == arc.N_PRACTICE == 512 and rows == arc.practice()
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16] == "405be521c724f3d8"
    assert all(set(r) == {"question", "options", "answer"} and len(r["options"]) == 4 and r["answer"] in range(4)
               for r in rows)
    assert len({r["question"] for r in rows}) == 512 and {r["question"] for r in rows} <= set(arc.load("train")[0])
    shown = arc.examples()
    assert shown.count("\nAnswer: ") == 10 and not any(f"] {r['question'].strip()}\n" in shown for r in rows)


def test_prepare_writes_practice_for_fb_only(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("WSP_CLAUDE_BIN", "/claude")
    monkeypatch.setattr(run, "access_token_credentials", lambda key: {"claudeAiOauth": {}})
    fb, free = (spec.RunSpec(a, "blind", 0, 0, "fable") for a in ("fb-arc", "free-arc"))
    work = run.prepare(fb, None) / "work"
    lines = (work / "data" / "practice.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == arc.practice()
    assert [p.name for p in (work / "data").iterdir()] == ["practice.jsonl"]
    assert (work / "TASK.md").read_text() == task.render("fb-arc", "blind", 0)
    assert not (run.prepare(free, None) / "work" / "data").exists()
    assert run.allowed_data(fb) == {"/srv/work/data/practice.jsonl"} and run.mounted_data(fb) == set()
    assert "practice.jsonl" not in " ".join(run.sandbox_argv(fb, fb.run_dir))


def test_no_other_blind_run_may_see_data():
    for a in archs.ARCHS:
        assert run.allowed_data(spec.RunSpec(a, "blind", 0, 0, "opus")) == (
            {"/srv/work/data/practice.jsonl"} if a == "fb-arc" else {"/srv/work/data/arc_train.jsonl"} if a == "fit-arc"
            else set()), a
    open_run = spec.RunSpec("bow", "open", 0, 0, "opus")
    assert run.allowed_data(open_run) == run.mounted_data(open_run) == {
        "/srv/work/data/train.npz", "/srv/work/data/train.jsonl"}


def test_check_model_practice_prints_the_accuracy_only(tmp_path, monkeypatch, capsys):
    module = importlib.util.spec_from_file_location("check_model", REPO / "harness" / "tools" / "check_model.py")
    tool = importlib.util.module_from_spec(module)
    module.loader.exec_module(tool)
    model = tmp_path / "model"
    model.mkdir()
    (model / "vocab.txt").write_text("water\n")
    (model / "build.py").write_text("import torch\n\n\ndef build():\n    return {'water': torch.tensor(2)}\n")
    (model / "arch.py").write_text(
        "import torch\n\n\ndef forward(w, q, o):\n    return (o == w['water']).sum(-1).float()\n")
    rows = [{"question": "Which one is a liquid?", "options": ["rock", "water", "sand", "ice"], "answer": 1},
            {"question": "Which one is a solid?", "options": ["water", "rock", "steam", "air"], "answer": 1},
            {"question": "Which one is a gas?", "options": ["rock", "sand", "ice", "steam"], "answer": 3}]
    (tmp_path / "practice.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(tool, "MODEL", model)
    monkeypatch.setattr(tool, "PRACTICE", tmp_path / "practice.jsonl")
    tool.check_practice("fb-arc")
    assert capsys.readouterr().out == "practice accuracy 0.4167 on 3 questions, 0.6667 answered\n"

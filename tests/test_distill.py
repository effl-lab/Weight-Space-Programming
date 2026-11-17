import json
import time

from harness import run, task
from wsp import archs, distill


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_read_keeps_valid_rows_and_lists_bad_lines(tmp_path):
    good = {"question": "What gas do plants take in?", "options": ["oxygen", "carbon dioxide", "helium", "neon"],
            "answer": 1}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(good) + "\nnot json\n" + json.dumps({**good, "answer": 4}) + "\n\n")
    rows, problems = distill.read(path)
    assert rows == [good] and [p.split(":")[0] for p in problems] == ["line 2", "line 3"]


def test_stats_counts_duplicates_and_answer_positions():
    r = {"question": "q one", "options": ["a", "b", "c", "d"], "answer": 2}
    s = distill.stats([r, r, {**r, "question": "q two", "answer": 0}])
    assert s["questions"] == 3 and s["exact_duplicates"] == 1 and s["answer_position"] == [1, 0, 2, 0]


def test_task_and_routing():
    text = task.render("distill-arc", "blind", 0)
    assert "check_data.py" in text and "check_model.py" not in text and "arch.py" not in text
    assert "Everything in the data must come from your own knowledge" in text
    assert archs.ARCHS["distill-arc"].deliverable == distill.DELIVERABLE and run.fixed_arc("distill-arc") is None


def test_snapshot_only_when_due_and_changed(tmp_path):
    work = tmp_path / "work" / distill.DELIVERABLE
    work.mkdir(parents=True)
    meta = {"started": time.time() - distill.SNAPSHOT_S - 1}
    _write(work / distill.TRAIN_FILE, [{"question": "q", "options": ["a", "b", "c", "d"], "answer": 0}])
    run.snapshot_data(tmp_path, meta)
    run.snapshot_data(tmp_path, meta)  # not due yet
    run.snapshot_data(tmp_path, meta, force=True)  # due but unchanged
    assert len(list((tmp_path / "snapshots").iterdir())) == 1
    _write(work / distill.TRAIN_FILE, [{"question": "q2", "options": ["a", "b", "c", "d"], "answer": 1}])
    run.snapshot_data(tmp_path, meta, force=True)
    assert len(list((tmp_path / "snapshots").iterdir())) == 2


def test_distill100_same_task_ended_by_cost(tmp_path, monkeypatch):
    assert task.render("distill100-arc", "blind", 0) == task.render("distill-arc", "blind", 0)
    assert run.fixed_arc("distill100-arc") is None
    spec = run.RunSpec("distill100-arc", "blind", 0, 0)
    seg = {"script_sha": "a", "forward_calls_after": 0}
    used = {"output_tokens": 10 * run.K0_OUTPUT_TOKEN_TARGET, "cost_usd_equiv": 99.0}
    monkeypatch.setattr(run.tokens, "totals", lambda _: used)
    meta = {"segments": [seg]}
    assert run._continuation(spec, tmp_path, meta) is not None  # past the token target, under the cost cap
    used["cost_usd_equiv"] = 100.0
    assert run._continuation(spec, tmp_path, meta) is None and meta["ended"] == "cost cap reached"
    meta = {"segments": [seg]}
    assert run._continuation(run.RunSpec("distill-arc", "blind", 0, 0), tmp_path, meta) is None
    assert meta["ended"] == "output token target reached"

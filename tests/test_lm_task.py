"""The released-LM ARC architectures: names, run roots, the edit contract and the scoring-side folder checks."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from baselines import arc_released
from harness import lm_score, spec, task
from wsp import lm_start, lm_task

START = {"embed": torch.arange(12, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16),
         "norm": torch.ones(3, dtype=torch.bfloat16)}


@pytest.fixture
def start(tmp_path):
    save_file(START, tmp_path / lm_start.WEIGHTS)
    return tmp_path


def write_edit(folder, body: str):
    folder.mkdir(exist_ok=True)
    (folder / lm_task.EDIT_FILE).write_text("import torch\n\n\ndef edit(w, tok):\n" + body)
    return folder


@pytest.fixture
def isolated(monkeypatch):
    """apply_edit changes the working directory and sys.path, as it does in its own process."""
    monkeypatch.chdir(".")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "edit", raising=False)


def test_names():
    assert lm_start.NAMES == tuple(sorted(lm_start.NAMES)) and len(lm_start.NAMES) == 15
    assert len(lm_start.ARCHS) == 30
    for a in lm_start.ARCHS:
        name, start_ = lm_start.parse(a)
        assert lm_start.arch(name, start_) == a and start_ in lm_start.STARTS
    assert lm_start.parse("lm-qwen3_5_0_8b-init") == ("qwen3_5_0_8b", "init")
    assert not lm_start.is_lm("free-arc") and not lm_start.is_lm("lm-nomodel-pre")
    with pytest.raises(ValueError):
        lm_start.parse("transformer-arc")


def test_run_dir():
    s = spec.RunSpec("lm-gemma2_2b-init", "blind", 0, 1)
    assert s.run_dir == spec.RUNS_DIR.with_name("runs_lm") / "lm-gemma2_2b-init_blind_K0_r1"
    assert spec.RunSpec("lm-gemma2_2b-init", "blind", 0, 1, "sol").run_dir.parent.name == "runs_lm_sol"
    assert spec.RunSpec("transformer-arc", "blind", 0, 0).run_dir.parent.name == "runs_arc"


def test_slot_weight(monkeypatch):
    sizes = {"lm-gemma2_2b-pre": 2_614_341_888, "lm-gemma3_1b-pre": 999_885_952}
    monkeypatch.setattr(lm_start, "manifest", lambda a: {"parameters": sizes[a]})
    assert [spec.slot_weight(a) for a in (*sizes, "mlp")] == [2, 1, 1]


def test_lm_runs_blind_k0_only():
    with pytest.raises(ValueError):
        task.render("lm-smollm2_135m-pre", "open", 0)


def test_start_weights_lazy_and_private(start):
    w = lm_task.StartWeights(start)
    assert sorted(w) == ["embed", "norm"] and w.shape("embed") == (4, 3)
    w["norm"][0] = 5
    assert w["norm"][0] == 5
    assert lm_task.StartWeights(start)["norm"][0] == 1


def test_edit_problems(start):
    w = lm_task.StartWeights(start)
    assert lm_task.edit_problems([], w) == ["edit returned list, not a dict"]
    out = {"nope": torch.zeros(1), "norm": torch.zeros(2), "embed": torch.zeros(4, 3, dtype=torch.int64)}
    assert lm_task.edit_problems(out, w) == ["'nope': no starting tensor of that name", "norm: shape (2,) != (3,)",
                                             "embed: dtype torch.int64 is not a float dtype"]
    assert lm_task.edit_problems({"norm": torch.full((3,), 1e39, dtype=torch.float64)}, w) == ["norm: not finite in bfloat16"]
    assert lm_task.edit_problems({"norm": torch.ones(3).to_sparse()}, w) == ["norm: not a dense torch tensor"]
    assert lm_task.edit_problems({"norm": torch.zeros(3, dtype=torch.float64)}, w) == []


def test_apply_edit_returns_only_returned_tensors(start, tmp_path, isolated):
    folder = write_edit(tmp_path / "model", "    w['embed'][0] = 7\n    n = w['norm'].float().clone()\n"
                                            "    n[1] = 2\n    return {'norm': n}\n")
    changed, problems = lm_task.apply_edit(folder, start, None)
    assert problems == [] and list(changed) == ["norm"] and changed["norm"].dtype == lm_task.DTYPE
    assert changed["norm"].tolist() == [1, 2, 1]
    assert lm_task.changed_parameters(changed, start) == 1
    w = lm_task.model_weights(start, changed, torch.device("cpu"))
    assert torch.equal(w["embed"], START["embed"]) and torch.equal(w["norm"], changed["norm"])


def test_apply_edit_reads_its_folder(start, tmp_path, isolated):
    folder = write_edit(tmp_path / "model", "    import scale\n    return {'norm': w['norm'] * scale.FACTOR}\n")
    (folder / "scale.py").write_text("FACTOR = float(open('factor.txt').read())\n")
    (folder / "factor.txt").write_text("3")
    changed, problems = lm_task.apply_edit(folder, start, None)
    assert problems == [] and changed["norm"].tolist() == [3, 3, 3]


def test_readout():
    ids = torch.tensor([[1, 7, 9, 3, 0], [1, 9, 3, 0, 0]])
    lengths = torch.tensor([5, 4])
    assert arc_released.readout(ids, lengths, None).tolist() == [4, 3]
    assert arc_released.readout(ids, lengths, 9).tolist() == [2, 1]


def test_mask_of():
    assert lm_task.mask_of("smollm2_135m") is None
    assert lm_task.mask_of("modernbert_base") == 50284 and lm_task.mask_of("ultrabert_150m") == 16


def test_group_max():
    z = torch.tensor([[1.0, 5.0, 2.0, 0.0, 4.0]])
    assert arc_released.group_max(z, [[10, 11], [12], [13, 14]]).tolist() == [[5.0, 2.0, 4.0]]


def test_folder_problems(tmp_path):
    name = "smollm2_135m"
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "arch.py").write_text(lm_start.arch_source(name))
    assert lm_score.folder_problems(folder, name) == ["model/edit.py missing"]
    (folder / "edit.py").write_text("def edit(w, tok):\n    return {}\n")
    assert lm_score.folder_problems(folder, name) == []
    (folder / "link").symlink_to("/etc/passwd")
    (folder / "arch.py").write_text(lm_start.arch_source(name) + "#")
    assert lm_score.folder_problems(folder, name) == ["link is a link", "the given arch.py was changed"]


def test_score_step_runs_locally_and_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(lm_score, "REPO", tmp_path)
    monkeypatch.setattr(lm_score, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "fake_score.py").write_text(
        "import json, sys, pathlib\n"
        "pathlib.Path(sys.argv[1], 'score.json').write_text(json.dumps({'arch': sys.argv[2]}))\n")
    (tmp_path / "fake_fail.py").write_text("import sys\nsys.exit('broken start')\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    venv = Path(sys.prefix)
    assert lm_score.run_score_step("fake_score", run_dir, "lm-x-pre", venv) == []
    assert json.loads((run_dir / lm_score.SCORE).read_text()) == {"arch": "lm-x-pre"}
    (run_dir / lm_score.SCORE).unlink()
    problems = lm_score.run_score_step("fake_fail", run_dir, "lm-x-pre", venv)
    assert len(problems) == 1 and "exited 1" in problems[0] and "broken start" in problems[0]


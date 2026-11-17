import json

import pytest

from harness import run, spec
from wsp import archs, lineage


def test_parse_and_parent():
    assert lineage.parse("free-arc") == (1, "arc")
    assert lineage.parse("free3-nutrition") == (3, "nutrition")
    assert lineage.parse("transformer-arc") is None and lineage.parse("free1-arc") is None
    assert lineage.parse("mem-nutrition") is None and lineage.parse("lm-gemma2_2b-pre") is None
    assert lineage.parent_arch("free-arc") is None
    assert lineage.parent_arch("free-nutrition") == "mem-nutrition"
    assert lineage.parent_arch("free2-arc") == "free-arc"
    assert lineage.parent_arch("free4-nutrition") == "free3-nutrition"


def test_registered_and_routed():
    assert archs.ARCHS["free2-arc"].task == "arc" and archs.ARCHS["free2-nutrition"].task == "mmlu-nutrition"
    assert spec.RunSpec("free2-nutrition", "blind", 0, 0, "opus").run_dir.parent.name == "runs_memory"
    assert spec.RunSpec("free2-arc", "blind", 0, 0, "sol").run_dir.parent.name == "runs_arc_sol"
    assert run.fixed_arc("free2-arc") is None and run.fixed_arc("bow-arc") == "bow"


def _result(root, arch, rep, val, valid=True):
    d = root / f"{arch}_blind_K0_r{rep}"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"arch": arch, "rep": rep, "valid": valid,
                                               "validation": {"accuracy": val}}))
    return d


def test_parent_is_best_valid_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    root = tmp_path / "runs_arc"
    root.mkdir()
    s = spec.RunSpec("free2-arc", "blind", 0, 0, "opus")
    with pytest.raises(FileNotFoundError):
        run.lineage_parent(s)
    _result(root, "free-arc", 0, 0.40)
    best = _result(root, "free-arc", 1, 0.45)
    _result(root, "free-arc", 2, 0.90, valid=False)
    _result(root, "free-arc", 3, 0.45)
    assert run.lineage_parent(s) == best
    assert run.lineage_parent(spec.RunSpec("free-arc", "blind", 0, 0, "opus")) is None


def test_arc_variants_registered_and_routed():
    for v in lineage.ARC_VARIANTS:
        assert lineage.arc_variant(f"{v}-arc") == v and lineage.parse(f"{v}-arc") is None
        assert archs.ARCHS[f"{v}-arc"].task == "arc" and run.fixed_arc(f"{v}-arc") is None
        assert spec.RunSpec(f"{v}-arc", "blind", 0, 0, "opus").run_dir.parent.name == "runs_arc"
    assert lineage.arc_variant("free-arc") is None and lineage.arc_variant("bow-arc") is None


def test_nofact_variant_adds_only_its_sentence():
    from harness import task
    text = task.render_arc("nofact-arc", None)
    assert task.ARC_NOFACT in text and text.replace(task.ARC_NOFACT, "") == task.render_arc("free-arc", None)
    assert run.fixed_arc("nofact-arc") is None and run.lineage_parent(spec.RunSpec("nofact-arc", "blind", 0, 0, "opus")) is None


def test_conv_parent_is_best_of_last_finished_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    root = tmp_path / "runs_arc"
    root.mkdir()
    s = spec.RunSpec("conv-arc", "blind", 0, 0, "opus")
    with pytest.raises(FileNotFoundError):
        run.lineage_parent(s)
    _result(root, "free-arc", 0, 0.40)
    _result(root, "free2-arc", 0, 0.50)
    best = _result(root, "free2-arc", 1, 0.55)
    (root / "free3-arc_blind_K0_r0").mkdir()
    assert run.lineage_parent(s) == best
    assert run.lineage_parent(spec.RunSpec("par-arc", "blind", 0, 0, "opus")) is None


def test_variant_lineage_parent_and_task(tmp_path, monkeypatch):
    from harness import task
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    root = tmp_path / "runs_arc"
    root.mkdir()
    _result(root, "par-arc", 0, 0.40)
    best = _result(root, "par-arc", 1, 0.45)
    _result(root, "free-arc", 0, 0.60)
    assert lineage.variant_lineage("par3-arc") == ("par", 3) and lineage.variant_lineage("par-arc") == ("par", 1)
    assert lineage.variant_lineage("conv-arc") is None
    assert lineage.arc_variant("par2-arc") == "par" and run.fixed_arc("noq2-arc") is None
    assert run.lineage_parent(spec.RunSpec("par2-arc", "blind", 0, 0, "opus")) == best
    assert run.lineage_parent(spec.RunSpec("par-arc", "blind", 0, 0, "opus")) is None
    parent = {"arch": "par-arc", "validation": {"accuracy": 0.45, "n": 862}}
    assert "parallel hardware" in task.render_arc("par2-arc", parent)
    assert "The parent model (generation 1)" in task.render_arc("par2-arc", parent)


def test_given_lineage_parent_and_task(tmp_path, monkeypatch):
    from harness import task
    from wsp import given_archs
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    root = tmp_path / "runs_arc"
    root.mkdir()
    best = _result(root, "mem_assoc2-arc", 0, 0.48)
    _result(root, "mem_assoc2-arc", 1, 0.47)
    _result(root, "mem_assoc-arc", 0, 0.60)
    assert lineage.given_lineage("mem_assoc2-arc") == ("mem_assoc2", 1)
    assert lineage.given_lineage("mem_assoc2_g3-arc") == ("mem_assoc2", 3)
    assert lineage.given_lineage("mem_assoc2_g1-arc") is None and lineage.given_lineage("mem_assoc-arc") is None
    assert lineage.given_lineage("mem_assoc22-arc") is None and lineage.given_lineage("free2-arc") is None
    assert lineage.given_arch("mem_assoc2", 1) == "mem_assoc2-arc"
    assert archs.ARCHS["mem_assoc2_g2-arc"].task == "arc" and "mem_assoc_g2-arc" not in archs.ARCHS
    assert run.fixed_arc("mem_assoc2_g2-arc") == "mem_assoc2" and lineage.arc_variant("mem_assoc2_g2-arc") is None
    assert spec.RunSpec("mem_assoc2_g2-arc", "blind", 0, 0, "sol").run_dir.parent.name == "runs_arc_sol"
    assert run.lineage_parent(spec.RunSpec("mem_assoc2_g2-arc", "blind", 0, 0, "opus")) == best
    assert run.lineage_parent(spec.RunSpec("mem_assoc2-arc", "blind", 0, 0, "opus")) is None
    with pytest.raises(FileNotFoundError):
        run.lineage_parent(spec.RunSpec("mem_assoc2_g3-arc", "blind", 0, 0, "opus"))
    first = task.render_arc("mem_assoc2-arc", None)
    parent = {"arch": "mem_assoc2-arc", "validation": {"accuracy": 0.48, "n": 862}}
    text = task.render_arc("mem_assoc2_g2-arc", parent)
    section = task.PARENT_GIVEN.format(generation=1, n_val=862, accuracy=48.0)
    assert "arch.py stays as given" in section and text.replace("\n" + section, "") == first
    assert given_archs.source("mem_assoc2").rstrip() in text
    assert lineage.given_lineage("mem_route_g2-arc") == ("mem_route", 2) and run.fixed_arc("mem_route_g4-arc") == "mem_route"
    routed = task.render_arc("mem_route_g2-arc", {"arch": "mem_route-arc", "validation": {"accuracy": 0.5, "n": 862}})
    assert routed.replace("\n" + task.PARENT_GIVEN.format(generation=1, n_val=862, accuracy=50.0), "") == \
        task.render_arc("mem_route-arc", None)


def test_conv1_parent_is_best_of_generation_one(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    root = tmp_path / "runs_arc"
    root.mkdir()
    s = spec.RunSpec("conv1-arc", "blind", 0, 0, "opus")
    with pytest.raises(FileNotFoundError):
        run.lineage_parent(s)
    _result(root, "free-arc", 0, 0.40)
    best = _result(root, "free-arc", 1, 0.45)
    _result(root, "free2-arc", 0, 0.60)
    _result(root, "free2-arc", 1, 0.60)
    assert run.lineage_parent(s) == best
    assert lineage.arc_variant("conv1-arc") == "conv1" and run.fixed_arc("conv1-arc") is None


def test_design_cifar_routed():
    assert archs.ARCHS["design-cifar"].task == "cifar" and lineage.parse("design-cifar") is None
    assert run.fixed_arc("design-cifar") is None and lineage.arc_variant("design-cifar") is None
    assert spec.RunSpec("design-cifar", "blind", 0, 0, "opus").run_dir.parent.name == "runs"
    assert run.lineage_parent(spec.RunSpec("design-cifar", "blind", 0, 0, "opus")) is None


def test_image_model_folder_problems(tmp_path):
    from wsp import free_model
    (tmp_path / "build.py").write_text("def build():\n    return {}\n")
    (tmp_path / "arch.py").write_text("import torch\n\ndef forward(w, x):\n    return x.mean((2, 3))\n")
    assert free_model.folder_problems(tmp_path, images=True) == []
    assert free_model.folder_problems(tmp_path) == ["model/vocab.txt missing"]
    (tmp_path / "arch.py").write_text("import torch\n\ndef forward(w, q, o):\n    return q\n")
    assert free_model.folder_problems(tmp_path, images=True) == []
    (tmp_path / "arch.py").write_text("import torch\n\ndef score(w, x):\n    return x\n")
    assert free_model.folder_problems(tmp_path, images=True) == ["arch.py: no top-level forward(w, x)"]


def test_accuracy_counts_ties_as_uniform_guess():
    import numpy as np
    from wsp import free_model
    assert free_model.accuracy(np.zeros((2, 4)), np.array([0, 1]))["accuracy"] == 0.25
    assert free_model.accuracy(np.zeros((2, 10)), np.array([0, 1]))["accuracy"] == 0.1


def test_transformer_widths():
    from wsp import given_archs
    for name, d in given_archs.WIDTHS.items():
        src = given_archs.source(name)
        assert f"D, HEADS, LAYERS, MLP = {d}, 4, 2, {2 * d}" in src and not free_model_problems(src)
        assert given_archs.shapes(name, 10)["embed"] == (10, d) and run.fixed_arc(f"{name}-arc") == name
    assert given_archs.source("transformer") == (given_archs.DIR / "transformer.py").read_text()


def free_model_problems(src):
    import ast
    from wsp import free_model
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, free_model.ARCH_FORBIDDEN_NODES)]

"""Final score of a released-LM run (lm-<name>-<start>), after the writer has ended. The writer never sees it.

On the host, the delivered model/edit.py runs once more as in the check tool, in a bwrap with no network and no data
other than the start directory and a copy of the model folder; the tensors it returns are saved as
run_dir/edit.safetensors (bfloat16). A score step, run as a subprocess in the run's Python environment, overlays them
on the starting weights on the GPU, runs the arch forward with the letter readout of wsp.lm_task on ARC validation and
test, and writes run_dir/score.json. A run whose edit already ran or whose score.json exists resumes from there.

    python -m harness.lm_score <run_dir> <arch>      the score step (needs a GPU)
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import load_file

from baselines import arc_released
from wsp import arc, lm_start, lm_task, paths
from wsp.free_model import DELIVERABLE

REPO = Path(__file__).resolve().parents[1]
VENV = paths.VENV
SLM_ARC_ENV = "WSP_SLM_VENV"  # an environment with newer transformers, as these families' released models need
SLM_ARC_FAMILIES = {"qwen3_5", "rwkv7", "ultrabert"}
LOG_DIR = REPO / "logs" / "lm_score"
EDIT_OUT, EDIT_INFO, SCORE = "edit.safetensors", "edit.json", "score.json"
LOGITS = "letter_logits.npz"  # per item, the logit of every letter token
# Mounted at /srv/lib for a released-LM run, with wsp/lm_archs/<name>.py, besides the files every sandbox has
# (wsp/__init__.py and wsp/lm_start.py, which wsp/archs.py imports); the scoring-side edit gets all of them.
LIB_FILES = ("wsp/lm_task.py", "wsp/arc.py", "wsp/lm_weights.py", "baselines/__init__.py", "baselines/arc_released.py")
BASE_LIB_FILES = ("wsp/__init__.py", "wsp/paths.py", "wsp/lm_start.py")
RUN_TIMEOUT_S = lm_task.EDIT_TIMEOUT_S + 600  # the edit's own limit plus imports, loading and saving
SCORE_LIMIT_S = 2 * 24 * 3600
EDIT_RUNNER = '''import json, sys
sys.path.insert(0, "/srv/lib")
from pathlib import Path
import torch
from safetensors.torch import save_file
from wsp import lm_task
lm_task.limit_memory()
torch.set_num_threads(4)
start = Path("/srv/start")
changed, problems = lm_task.apply_edit(Path("/build/model"), start, lm_task.tokenizer(start))
if problems:
    sys.exit("EDIT PROBLEMS\\n" + "\\n".join(problems))
save_file(changed, "/build/out/edit.safetensors")
json.dump({"changed_tensors": sorted(changed), "changed_parameters": lm_task.changed_parameters(changed, start)},
          open("/build/out/edit.json", "w"))
'''


def lib_binds(name: str, files: tuple[str, ...] = LIB_FILES) -> list[str]:
    argv = []
    for f in (*files, f"wsp/lm_archs/{name}.py"):
        argv += ["--ro-bind", str(REPO / f), f"/srv/lib/{f}"]
    return argv


def folder_problems(folder: Path, name: str) -> list[str]:
    if folder.is_symlink() or not folder.is_dir():
        return [f"no folder {DELIVERABLE}/"]
    problems = [f"{p.relative_to(folder)} is a link" for p in folder.rglob("*") if p.is_symlink()]
    if not (folder / lm_task.EDIT_FILE).is_file():
        problems.append(f"{DELIVERABLE}/{lm_task.EDIT_FILE} missing")
    arch_py = folder / "arch.py"
    if not arch_py.is_file() or arch_py.read_text() != lm_start.arch_source(name):
        problems.append("the given arch.py was changed")
    return problems


def run_edit(arch: str, folder: Path, run_dir: Path) -> list[str]:
    """Runs the delivered edit isolated; on success run_dir holds EDIT_OUT and EDIT_INFO."""
    name, _ = lm_start.parse(arch)
    return run_isolated_edit(folder, run_dir, lm_start.start_dir(arch), lib_binds(name, BASE_LIB_FILES + LIB_FILES),
                             EDIT_RUNNER)


def run_isolated_edit(folder: Path, run_dir: Path, start: Path, binds: list[str], runner: str) -> list[str]:
    """Runs `runner` in a bwrap with no network, the start directory at /srv/start, the library files of `binds` and
    a copy of the model folder at /build/model; the runner writes EDIT_OUT and EDIT_INFO to /build/out."""
    build = run_dir / "edit_build"  # on disk, not in /tmp: the folder may hold large files
    shutil.rmtree(build, ignore_errors=True)
    shutil.copytree(folder, build / "model")
    (build / "out").mkdir()
    (build / "_run.py").write_text(runner)
    argv = ["bwrap", "--die-with-parent", "--unshare-all", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin", "--ro-bind", str(VENV), str(VENV),
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(start), "/srv/start", *binds,
            "--bind", str(build), "/build", "--chdir", "/build",
            "--setenv", "PATH", f"{VENV}/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp",
            "--setenv", "HF_HOME", "/tmp/hf", "--setenv", "HF_HUB_OFFLINE", "1",
            f"{VENV}/bin/python", "-I", "/build/_run.py"]
    try:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return [f"edit exceeded {RUN_TIMEOUT_S} s"]
        if proc.returncode != 0:
            return [f"edit exited {proc.returncode}: {proc.stderr[-2000:]}"]
        shutil.move(build / "out" / EDIT_OUT, run_dir / EDIT_OUT)
        shutil.move(build / "out" / EDIT_INFO, run_dir / EDIT_INFO)
        return []
    finally:
        shutil.rmtree(build, ignore_errors=True)


def run_score_step(module: str, run_dir: Path, arch: str, venv: Path) -> list[str]:
    """Runs `python -m <module> <run_dir> <arch>` with the Python of `venv`, offline, logging to LOG_DIR; problems
    if it fails, runs past SCORE_LIMIT_S or writes no score.json."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{run_dir.name}.out"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(REPO)}
    env.pop("TRANSFORMERS_CACHE", None)
    with log.open("w") as f:
        try:
            proc = subprocess.run([f"{venv}/bin/python", "-m", module, str(run_dir), arch], cwd=REPO, env=env,
                                  stdout=f, stderr=subprocess.STDOUT, timeout=SCORE_LIMIT_S)
        except subprocess.TimeoutExpired:
            return [f"score step wrote no {SCORE} within {SCORE_LIMIT_S} s (log {log})"]
    if proc.returncode != 0 or not (run_dir / SCORE).exists():
        return [f"score step exited {proc.returncode}: {log.read_text(errors='replace')[-1500:]}"]
    return []


def score(arch: str, run_dir: Path) -> dict:
    folder = run_dir / "work" / DELIVERABLE
    name, _ = lm_start.parse(arch)
    problems = folder_problems(folder, name)
    if not problems and not (run_dir / EDIT_INFO).exists():
        problems = run_edit(arch, folder, run_dir)
    if problems:
        return {"valid": False, "problems": problems}
    if not (run_dir / SCORE).exists():
        family = lm_start.manifest(arch)["family"]
        venv = paths.env_path(SLM_ARC_ENV) if family in SLM_ARC_FAMILIES else VENV
        problems = run_score_step("harness.lm_score", run_dir, arch, venv)
        if problems:
            return {"valid": False, "problems": problems}
    if not (run_dir / DELIVERABLE).exists():
        shutil.copytree(folder, run_dir / DELIVERABLE)
    s, info = json.loads((run_dir / SCORE).read_text()), json.loads((run_dir / EDIT_INFO).read_text())
    return {"valid": True, "problems": [], "test_accuracy": s["test"], "practice_accuracy": None,
            "test_by_subset": s["test_by_subset"], "validation": s["validation"],
            "changed_tensors": len(info["changed_tensors"]), "changed_parameters": info["changed_parameters"],
            "parameters": s["parameters"], "score_host": s["host"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("arch", choices=lm_start.ARCHS)
    a = ap.parse_args()
    name, _ = lm_start.parse(a.arch)
    start, manifest = lm_start.start_dir(a.arch), lm_start.manifest(a.arch)
    if lm_start.sha256(start / lm_start.WEIGHTS) != manifest["weights_sha256"]:
        sys.exit(f"LM_SCORE_FAILED {start / lm_start.WEIGHTS} does not match its manifest")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = lm_task.tokenizer(start)
    groups = arc_released.letter_tokens(tok)
    mask = lm_task.mask_of(name)
    w = lm_task.model_weights(start, load_file(a.run_dir / EDIT_OUT), device)
    out, logits = {}, {}
    for split in ("validation", "test"):
        q, o, y = arc.load(split)
        letters, logits[split] = lm_task.letter_logits(lm_task.forward_of(name), w, lm_task.encode(tok, q, o, mask),
                                                       groups, lm_task.pad_id(tok), device, mask)
        out[split] = letters.argmax(1) == y
    sub = arc.subset("test")
    np.savez(a.run_dir / LOGITS, **logits)
    result = {"validation": float(out["validation"].mean()), "test": float(out["test"].mean()),
              "test_by_subset": {s: float(out["test"][sub == s].mean()) for s in arc.SUBSETS},
              "n": {s: int(len(v)) for s, v in out.items()}, "parameters": sum(v.numel() for v in w.values()),
              "letter_tokens": [tok.convert_ids_to_tokens(g) for g in groups], "host": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(0),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
              "weights_sha256": manifest["weights_sha256"]}
    tmp = a.run_dir / (SCORE + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(a.run_dir / SCORE)
    print(json.dumps({k: result[k] for k in ("validation", "test", "test_by_subset")}), flush=True)


if __name__ == "__main__":
    main()

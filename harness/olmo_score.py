"""Hidden score of an OLMo run (wsp/olmo_start.py), after the writer has ended; the writer never sees it.

As for the released-LM ARC runs (harness/lm_score.py), the delivered model/edit.py runs once more in an isolated bwrap
and its tensors are saved as run_dir/edit.safetensors (float32); a score step on the GPU then measures the edited and
the unedited start on the same items and writes run_dir/score.json:
  heldout_loss   mean next-token loss on the held-out web documents of HELDOUT (both tasks)
  arc            ARC validation and test accuracy with the base-model prompt and letter readout (olmo-sft only)
These pick which runs go to the pretraining and SFT comparisons; those comparisons themselves run elsewhere.

    python -m harness.olmo_score <run_dir> <arch>      the score step (needs a GPU)
"""
import argparse
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from baselines import arc_released
from harness import lm_score
from wsp import arc, lm_start, lm_task, olmo2_1b, olmo_start, olmo_task, paths
from wsp.free_model import DELIVERABLE

REPO = Path(__file__).resolve().parents[1]
HELDOUT = paths.DATA_ROOT / "olmo_heldout.jsonl"  # scripts/prepare_olmo_heldout.py
# Mounted at /srv/lib for an OLMo run besides the files every sandbox has (harness/run.py LIB_FILES); the scoring-side
# edit gets EDIT_LIB_FILES.
LIB_FILES = ("wsp/lm_task.py", "wsp/olmo_task.py", "wsp/arc.py", "baselines/__init__.py", "baselines/arc_released.py")
EDIT_LIB_FILES = lm_score.BASE_LIB_FILES + ("wsp/olmo2_1b.py", "wsp/olmo_start.py") + LIB_FILES
EDIT_RUNNER = '''import json, resource, sys
sys.path.insert(0, "/srv/lib")
from pathlib import Path
import torch
from safetensors.torch import save_file
from wsp import lm_task, olmo_task
resource.setrlimit(resource.RLIMIT_DATA, (olmo_task.DATA_BYTES, olmo_task.DATA_BYTES))
torch.set_num_threads(4)
start = Path("/srv/start")
changed, problems = olmo_task.apply_edit(Path("/build/model"), start, lm_task.tokenizer(start))
if problems:
    sys.exit("EDIT PROBLEMS\\n" + "\\n".join(problems))
save_file(changed, "/build/out/edit.safetensors")
json.dump({"changed_tensors": sorted(changed), "changed_parameters": lm_task.changed_parameters(changed, start)},
          open("/build/out/edit.json", "w"))
'''


def lib_binds(files: tuple[str, ...] = LIB_FILES) -> list[str]:
    argv = []
    for f in files:
        argv += ["--ro-bind", str(REPO / f), f"/srv/lib/{f}"]
    return argv


def score(arch: str, run_dir: Path) -> dict:
    folder = run_dir / "work" / DELIVERABLE
    problems = [] if (folder / lm_task.EDIT_FILE).is_file() else [f"{DELIVERABLE}/{lm_task.EDIT_FILE} missing"]
    if not problems and not (run_dir / lm_score.EDIT_INFO).exists():
        problems = lm_score.run_isolated_edit(folder, run_dir, olmo_start.start_dir(arch),
                                                lib_binds(EDIT_LIB_FILES), EDIT_RUNNER)
    if problems:
        return {"valid": False, "problems": problems}
    if not (run_dir / lm_score.SCORE).exists():
        problems = lm_score.run_score_step("harness.olmo_score", run_dir, arch, paths.VENV)
        if problems:
            return {"valid": False, "problems": problems}
    if not (run_dir / DELIVERABLE).exists():
        shutil.copytree(folder, run_dir / DELIVERABLE)
    s = json.loads((run_dir / lm_score.SCORE).read_text())
    info = json.loads((run_dir / lm_score.EDIT_INFO).read_text())
    return {"valid": True, "problems": [], "test_accuracy": s.get("arc", {}).get("edited", {}).get("test"),
            "practice_accuracy": None, "olmo": s, "changed_tensors": len(info["changed_tensors"]),
            "changed_parameters": info["changed_parameters"], "score_host": s["host"]}


@torch.no_grad()
def arc_accuracy(w: dict, tok, device) -> dict:
    groups = arc_released.letter_tokens(tok)
    flat = torch.tensor([t for g in groups for t in g], device=device)
    out = {}
    with torch.device(device):
        for split in ("validation", "test"):
            q, o, y = arc.load(split)
            prompts = lm_task.encode(tok, q, o)
            tokens = torch.empty(len(prompts), len(flat), device="cpu")
            for s in range(0, len(prompts), arc_released.EVAL_BATCH):
                ids, lengths = arc_released.batch_tensors(prompts[s:s + arc_released.EVAL_BATCH],
                                                          lm_task.pad_id(tok), device)
                logits = olmo2_1b.logits(w, ids)[torch.arange(len(ids)), lengths - 1]
                tokens[s:s + len(ids)] = logits[:, flat].cpu()
            out[split] = float((arc_released.group_max(tokens, groups).argmax(1).cpu().numpy() == y).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("arch", choices=olmo_start.ARCHS)
    a = ap.parse_args()
    start, manifest = olmo_start.start_dir(a.arch), olmo_start.manifest(a.arch)
    if lm_start.sha256(start / lm_start.WEIGHTS) != manifest["weights_sha256"]:
        sys.exit(f"LM_SCORE_FAILED {start / lm_start.WEIGHTS} does not match its manifest")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = lm_task.tokenizer(start)
    docs = [json.loads(line)["text"] for line in HELDOUT.read_text().splitlines()]
    ids = olmo_task.encode_texts(tok, docs)
    result = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "heldout_docs": len(ids),
              "weights_sha256": manifest["weights_sha256"]}
    for which, changed in (("unedited", {}), ("edited", load_file(a.run_dir / lm_score.EDIT_OUT))):
        w = olmo_task.model_weights(start, changed, device)
        with torch.device(device):
            losses = olmo_task.text_losses(w, ids)
        result.setdefault("heldout_loss", {})[which] = float(np.mean(losses))
        if a.arch == "olmo-sft":
            result.setdefault("arc", {})[which] = arc_accuracy(w, tok, device)
        del w
        torch.cuda.empty_cache()
    tmp = a.run_dir / (lm_score.SCORE + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(a.run_dir / lm_score.SCORE)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

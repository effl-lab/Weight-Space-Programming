"""Build a written model folder and score it on ARC validation and test exactly as harness.run scores the run that
wrote it (harness.run.score on a copy of the folder). The architecture, rep and writer come from the folder's
result.json; the printed result is compared with the stored one.

    python -m harness.score_model models/opus/free5-arc_blind_K0_r1
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from harness import run
from harness.spec import RunSpec
from wsp import free_model

COMPARED = ("valid", "test_accuracy", "test", "validation", "test_by_subset", "words", "weights", "nonzero")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    a = ap.parse_args()
    stored = json.loads((a.folder / "result.json").read_text())
    spec = RunSpec(stored["arch"], stored["data_access"], stored["forward_budget"], stored["rep"], stored["writer"])
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(a.folder, Path(tmp) / "work" / free_model.DELIVERABLE,
                        ignore=shutil.ignore_patterns("result.json", "__pycache__"))
        scored = run.score(spec, Path(tmp))
    out = {k: scored.get(k) for k in COMPARED}
    print(json.dumps({"scored": out, "matches_stored": {k: out[k] == stored.get(k) for k in COMPARED},
                      "problems": scored.get("problems")}, indent=1))


if __name__ == "__main__":
    main()

"""Run the generations of a lineage (wsp.lineage: a free design, an ARC variant or a given architecture) until the
best validation accuracy of a generation does not exceed the best of all earlier generations. Each generation is one
harness.dispatch call; the next one starts only when every rep of the previous generation has result.json. Test
accuracy is never read here.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from harness.spec import RunSpec
from wsp import lineage

REPO = Path(__file__).resolve().parents[1]
WAIT_S = 300


def best_validation(arch: str, reps: int, writer: str) -> float | None:
    """Best validation accuracy over the valid reps, None while a rep has no result.json yet."""
    results = []
    for rep in range(reps):
        f = RunSpec(arch, "blind", 0, rep, writer).run_dir / "result.json"
        if not f.exists():
            return None
        results.append(json.loads(f.read_text()))
    valid = [r["validation"]["accuracy"] for r in results if r["valid"]]
    if not valid:
        raise RuntimeError(f"{arch}: no valid rep, the lineage cannot continue")
    return max(valid)


def wait_for(arch: str, reps: int, writer: str) -> float:
    while (v := best_validation(arch, reps, writer)) is None:
        time.sleep(WAIT_S)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help='an MMLU subject or "arc"')
    ap.add_argument("--writer", default="opus")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dispatch-args", default="", help="passed on to harness.dispatch, e.g. '--profiles c'")
    ap.add_argument("--variant", choices=lineage.LINEAGE_VARIANTS, help="an ARC variant lineage (par or noq)")
    ap.add_argument("--given", choices=lineage.GIVEN_LINEAGES, help="a given-architecture ARC lineage")
    a = ap.parse_args()
    if a.variant and a.given:
        ap.error("--variant and --given name different lineages")
    name = ((lambda g: lineage.variant_arch(a.variant, g)) if a.variant
            else (lambda g: lineage.given_arch(a.given, g)) if a.given else (lambda g: lineage.arch(g, a.task)))
    history = [] if a.task == "arc" else [wait_for(f"mem-{a.task}", 1, a.writer)]
    for g in range(1, lineage.MAX_GENERATION + 1):
        arch = name(g)
        if g > 1:
            subprocess.run([sys.executable, "-m", "harness.dispatch", "--archs", arch, "--conditions", "blind-K0",
                            "--reps", str(a.reps), "--writer", a.writer, *a.dispatch_args.split()], cwd=REPO,
                           check=True)
        best = wait_for(arch, a.reps if g > 1 or a.task == "arc" else 1, a.writer)
        print(json.dumps({"generation": g, "arch": arch, "best_validation": best, "earlier": history}), flush=True)
        if history and best <= max(history):
            print(json.dumps({"stopped": arch, "reason": "no gain over the earlier generations"}), flush=True)
            return
        history.append(best)
    print(json.dumps({"stopped": name(lineage.MAX_GENERATION), "reason": "generation limit"}))


if __name__ == "__main__":
    main()

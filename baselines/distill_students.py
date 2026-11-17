"""Students of the distill-arc baseline: the standard small networks of baselines.arc_size_matched, trained by gradient
descent on training data a writer wrote (wsp.distill), with the optimiser, learning rates and checkpoint rule of
baselines.train_pair (Adam, best validation step, best rate by ARC validation accuracy). The vocabulary is the most
frequent words of the writer's data; nothing from ARC train is used.

One output per (data file, family, size) in OUT_DIR. A data file is the run's final data/train.jsonl or one of its
snapshots (run_dir/snapshots/<seconds>_<digest>.jsonl), so that accuracy can be read against the writer's cost at that time.

    python -m baselines.distill_students <run_dir> [--snapshots] [--families ...] [--device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from baselines import arc_size_matched, train_pair
from wsp import arc, distill, free_model

OUT_DIR = Path("docs/experiments/distill_students")
SEED = 0


def data_files(run_dir: Path, snapshots: bool) -> list[Path]:
    final = run_dir / distill.DELIVERABLE / distill.TRAIN_FILE
    snaps = sorted((run_dir / "snapshots").glob("*.jsonl")) if snapshots else []
    return snaps + [final]


def tensors(vocab: list[str], questions: list[str], options: list[list[str]], answers, device) -> tuple:
    q, o = free_model.encode(vocab, questions, options)
    return torch.from_numpy(q).to(device), torch.from_numpy(o).to(device), torch.as_tensor(answers).to(device)


def train_students(run_dir: Path, path: Path, families: list[str], device) -> None:
    rows, _ = distill.read(path)
    vocab = distill.vocab(rows)
    parts = {"train": tensors(vocab, [r["question"] for r in rows], [r["options"] for r in rows],
                              [r["answer"] for r in rows], device)}
    for split in ("validation", "test"):
        q, c, y = arc.load(split)
        parts[split] = tensors(vocab, q, c, y, device)
    tag = "final" if path.parent.name == distill.DELIVERABLE else path.stem
    for family in families:
        fwd = arc_size_matched.forward_for(family)
        for writer, target in arc_size_matched.TARGETS.items():
            out = OUT_DIR / f"{run_dir.parent.name}__{run_dir.name}__{tag}__{family}_{writer}.json"
            if out.exists():
                continue
            d = arc_size_matched.width_for(family, len(vocab) + 2, target)
            start = {k: v.to(device) for k, v in arc_size_matched.init(family, len(vocab) + 2, d, SEED).items()}
            t0 = time.time()
            sweep = [train_pair.train_one(fwd, start, parts, len(rows), SEED, lr, False) for lr in train_pair.LRS]
            info, w = max(sweep, key=lambda r: r[0]["val_accuracy"])
            w = {k: v.to(device) for k, v in w.items()}
            res = {"run": f"{run_dir.parent.name}/{run_dir.name}", "data": tag, "questions": len(rows),
                   "vocab": len(vocab), "family": family, "size_of": writer, "width": d,
                   "parameters": arc_size_matched.n_params(family, len(vocab) + 2, d), **info,
                   "test_accuracy": train_pair.accuracy(fwd, w, parts["test"]), "lr_sweep": [r[0] for r in sweep],
                   "seconds_total": round(time.time() - t0, 1)}
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res) + "\n")
            print(json.dumps({k: res[k] for k in ("data", "questions", "family", "size_of", "lr", "val_accuracy",
                                                  "test_accuracy")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--snapshots", action="store_true", help="also every snapshot, not only the final data")
    ap.add_argument("--families", nargs="+", default=list(arc_size_matched.FAMILIES),
                    choices=arc_size_matched.FAMILIES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for path in data_files(a.run_dir, a.snapshots):
        train_students(a.run_dir, path, a.families, torch.device(a.device))


if __name__ == "__main__":
    main()

"""Evaluate a weights.npz on a split of its architecture's task."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wsp import archs


def load_weights(arch: str, path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    problems = archs.validate(arch, arrays)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in arrays.items()}


@torch.no_grad()
def predict(arch: str, weights: dict[str, torch.Tensor], x: np.ndarray, batch: int = 1000) -> np.ndarray:
    out = [archs.forward(arch, weights, torch.from_numpy(x[i:i + batch])) for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def evaluate(arch: str, weights: dict[str, torch.Tensor], x: np.ndarray, y: np.ndarray) -> dict:
    logits = predict(arch, weights, x)
    if not np.all(np.isfinite(logits)):
        return {"accuracy": None, "error": "non-finite logits", "n": len(y)}
    pred = logits.argmax(1)
    k = logits.shape[1]
    conf = np.zeros((k, k), dtype=int)
    np.add.at(conf, (y, pred), 1)
    return {"accuracy": float((pred == y).mean()), "n": int(len(y)),
            "per_class": (conf.diagonal() / conf.sum(1)).round(4).tolist(),
            "confusion": conf.tolist(),
            "pred_hist": np.bincount(pred, minlength=k).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(archs.ARCHS))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["test", "practice", "train"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(4)
    x, y = archs.task_of(a.arch).load(a.split)
    res = {"arch": a.arch, "split": a.split, "weights": str(a.weights),
           **evaluate(a.arch, load_weights(a.arch, a.weights), x, y)}
    text = json.dumps(res)
    if a.out:
        a.out.write_text(text + "\n")
    print(json.dumps({k: res[k] for k in ("arch", "split", "accuracy", "n")}))


if __name__ == "__main__":
    main()

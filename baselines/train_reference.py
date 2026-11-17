"""Context line only: a standard large CNN trained with SGD on the same 45,000-image pool.

ResNet-18 (CIFAR stem), random crop + flip, SGD momentum 0.9, lr 0.1 one-cycle, 40 epochs. Not one
of the compared architectures; it shows how far the small compared models sit from a typical
GPU-trained CIFAR-10 result.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from baselines import train_gd
from wsp import data

EPOCHS = 40
BATCH = 256
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def resnet18() -> nn.Module:
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def augment(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,))
    x = x[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)


@torch.no_grad()
def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, dev: str) -> float:
    model.eval()
    pred = torch.cat([model((torch.from_numpy(x[i:i + 1000]).to(dev) - MEAN.to(dev)) / STD.to(dev)).argmax(1).cpu()
                      for i in range(0, len(x), 1000)])
    return float((pred.numpy() == y).mean())


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = "cuda"
    train_gd.TRAIN_X, train_gd.TRAIN_Y = data.load("train")
    tr, val, _ = train_gd.split_indices(45000, 0)
    x = torch.from_numpy(train_gd.TRAIN_X[tr]).to(dev)
    y = torch.from_numpy(train_gd.TRAIN_Y[tr]).to(dev)
    torch.manual_seed(0)
    model = resnet18().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = EPOCHS * (len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        for k in range(len(tr) // BATCH):
            idx = perm[k * BATCH:(k + 1) * BATCH]
            xb = (augment(x[idx]) - MEAN.to(dev)) / STD.to(dev)
            loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    x_te, y_te = data.load("test")
    res = {"model": "resnet18", "n_train": 45000, "epochs": EPOCHS,
           "n_params": sum(p.numel() for p in model.parameters()),
           "val_accuracy": accuracy(model, train_gd.TRAIN_X[val], train_gd.TRAIN_Y[val], dev),
           "test_accuracy": accuracy(model, x_te, y_te, dev), "seconds": round(time.time() - t0, 1)}
    out = train_gd.OUT_DIR / "reference_resnet18.json"
    out.write_text(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()

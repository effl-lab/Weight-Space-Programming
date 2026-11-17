import math

import torch
import torch.nn.functional as F

D, HEADS, LAYERS, MLP = 64, 4, 2, 128


def forward(w, q, a):
    """The sequence is the question's words, then the answer's words so far. The logits of the next word, over the
    vocabulary (id 0 ends the answer), are read from the last layer's state at the first empty answer position."""
    n, L = q.shape
    ids = torch.cat([q, a], dim=1)
    slot = L + (a != 0).sum(1).clamp(max=a.shape[1] - 1)
    rows = torch.arange(n)
    real = (ids != 0).float()
    real = real.scatter(1, slot[:, None], 1.0)
    h = w["embed"][ids] + w["pos"]
    for i in range(LAYERS):
        p = f"block{i}."
        x = F.layer_norm(h, (D,), w[p + "ln1.weight"], w[p + "ln1.bias"])
        qq, k, v = (((x @ w[p + m]) + w[p + m + ".bias"]).reshape(n, h.shape[1], HEADS, D // HEADS)
                    .transpose(1, 2) for m in ("q", "k", "v"))
        att = qq @ k.transpose(-1, -2) / math.sqrt(D // HEADS) + (real[:, None, None, :] - 1) * 1e9
        x = (torch.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(n, h.shape[1], D)
        h = h + x @ w[p + "o"] + w[p + "o.bias"]
        x = F.layer_norm(h, (D,), w[p + "ln2.weight"], w[p + "ln2.bias"])
        h = h + F.gelu(x @ w[p + "fc1"] + w[p + "fc1.bias"]) @ w[p + "fc2"] + w[p + "fc2.bias"]
    h = F.layer_norm(h, (D,), w["ln_f.weight"], w["ln_f.bias"])
    return h[rows, slot] @ w["out"] + w["out.bias"]

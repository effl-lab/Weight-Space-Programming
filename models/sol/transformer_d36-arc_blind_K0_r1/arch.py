import math

import torch
import torch.nn.functional as F

D, HEADS, LAYERS, MLP = 36, 4, 2, 72


def encode(w, ids):
    """(m, L) word ids -> (m, D): mean over real tokens of the last layer's states."""
    real = (ids != 0).float()
    h = w["embed"][ids] + w["pos"][: ids.shape[1]]
    for i in range(LAYERS):
        p = f"block{i}."
        x = F.layer_norm(h, (D,), w[p + "ln1.weight"], w[p + "ln1.bias"])
        q, k, v = (((x @ w[p + n]) + w[p + n + ".bias"]).reshape(ids.shape[0], ids.shape[1], HEADS, D // HEADS)
                   .transpose(1, 2) for n in ("q", "k", "v"))
        att = q @ k.transpose(-1, -2) / math.sqrt(D // HEADS) + (real[:, None, None, :] - 1) * 1e9
        a = (torch.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(ids.shape[0], ids.shape[1], D)
        h = h + a @ w[p + "o"] + w[p + "o.bias"]
        x = F.layer_norm(h, (D,), w[p + "ln2.weight"], w[p + "ln2.bias"])
        h = h + F.gelu(x @ w[p + "fc1"] + w[p + "fc1.bias"]) @ w[p + "fc2"] + w[p + "fc2.bias"]
    h = F.layer_norm(h, (D,), w["ln_f.weight"], w["ln_f.bias"])
    return (h * real[..., None]).sum(1) / real.sum(1, keepdim=True).clamp(min=1)


def forward(w, q, o):
    """Each option is read together with its question: the sequence is question ids then option ids."""
    n = q.shape[0]
    seq = torch.cat([q[:, None, :].expand(n, 4, q.shape[1]), o], dim=2).reshape(n * 4, q.shape[1] + o.shape[2])
    return (encode(w, seq) @ w["score"]).reshape(n, 4)

import math

import torch
import torch.nn.functional as F

D, HEADS, LAYERS, MLP = 64, 4, 2, 128


def forward(w, q, c):
    """The sequence is one "no answer" slot, the paragraph's words, then the question's words. The slot and every
    paragraph position get a start logit and an end logit from the last layer's states: (n, 2, 1 + paragraph length)."""
    n, L = c.shape
    ids = torch.cat([c, q], dim=1)
    real = torch.cat([torch.ones(n, 1), (ids != 0).float()], dim=1)
    h = torch.cat([w["none"].expand(n, 1, D), w["embed"][ids]], dim=1) + w["pos"]
    for i in range(LAYERS):
        p = f"block{i}."
        x = F.layer_norm(h, (D,), w[p + "ln1.weight"], w[p + "ln1.bias"])
        qq, k, v = (((x @ w[p + m]) + w[p + m + ".bias"]).reshape(n, h.shape[1], HEADS, D // HEADS)
                    .transpose(1, 2) for m in ("q", "k", "v"))
        att = qq @ k.transpose(-1, -2) / math.sqrt(D // HEADS) + (real[:, None, None, :] - 1) * 1e9
        a = (torch.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(n, h.shape[1], D)
        h = h + a @ w[p + "o"] + w[p + "o.bias"]
        x = F.layer_norm(h, (D,), w[p + "ln2.weight"], w[p + "ln2.bias"])
        h = h + F.gelu(x @ w[p + "fc1"] + w[p + "fc1.bias"]) @ w[p + "fc2"] + w[p + "fc2.bias"]
    h = F.layer_norm(h, (D,), w["ln_f.weight"], w["ln_f.bias"])
    return (h[:, : L + 1] @ w["span"]).transpose(1, 2)

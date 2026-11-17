"""Small AG News classifiers over token ids, registered in wsp.archs with task "agnews".

Input: int64 token ids, shape (N, 64); id 0 is padding, id 1 an unknown word, ids 2.. index vocab.txt.
Output: (N, 4) logits in the order world, sports, business, scitech.
"""
import math

import torch
import torch.nn.functional as F

TEXT_VOCAB = 2002
TEXT_LEN = 64
TEXT_CLASSES = 4
EMB = 16
CNN_CH = 32
TF_DIM = 24
TF_HEADS = 2
TF_MLP = 48


def token_counts(x):
    """(N, 64) ids -> (N, 2002) number of times each id occurs; padding (id 0) is not counted."""
    c = torch.zeros(x.shape[0], TEXT_VOCAB, device=x.device).scatter_add_(1, x, torch.ones(x.shape, device=x.device))
    c[:, 0] = 0
    return c


def _bow_shapes():
    return {"score.weight": (TEXT_CLASSES, TEXT_VOCAB), "score.bias": (TEXT_CLASSES,)}


def _bow_forward(w, x):
    return F.linear(torch.log1p(token_counts(x)), w["score.weight"], w["score.bias"])


def _fasttext_shapes():
    return {"embed": (TEXT_VOCAB, EMB), "score.weight": (TEXT_CLASSES, EMB), "score.bias": (TEXT_CLASSES,)}


def _fasttext_forward(w, x):
    real = (x != 0).unsqueeze(-1).float()
    mean = (F.embedding(x, w["embed"]) * real).sum(1) / real.sum(1).clamp(min=1)
    return F.linear(mean, w["score.weight"], w["score.bias"])


def _cnn1d_shapes():
    return {"embed": (TEXT_VOCAB, EMB), "conv.weight": (CNN_CH, EMB, 3), "conv.bias": (CNN_CH,),
            "score.weight": (TEXT_CLASSES, CNN_CH), "score.bias": (TEXT_CLASSES,)}


def _cnn1d_forward(w, x):
    real = (x != 0).unsqueeze(1).float()
    e = F.embedding(x, w["embed"]).transpose(1, 2) * real                # (N, 16, 64), padding embeds as 0
    h = F.relu(F.conv1d(e, w["conv.weight"], w["conv.bias"], padding=1))  # (N, 32, 64)
    h = h * real                                                          # padding positions read 0
    return F.linear(h.amax(2), w["score.weight"], w["score.bias"])


def _transformer_shapes():
    d = TF_DIM
    return {"embed": (TEXT_VOCAB, d), "pos": (TEXT_LEN, d),
            "block0.ln1.weight": (d,), "block0.ln1.bias": (d,),
            "block0.q.weight": (d, d), "block0.q.bias": (d,), "block0.k.weight": (d, d), "block0.k.bias": (d,),
            "block0.v.weight": (d, d), "block0.v.bias": (d,), "block0.o.weight": (d, d), "block0.o.bias": (d,),
            "block0.ln2.weight": (d,), "block0.ln2.bias": (d,),
            "block0.fc1.weight": (TF_MLP, d), "block0.fc1.bias": (TF_MLP,),
            "block0.fc2.weight": (d, TF_MLP), "block0.fc2.bias": (d,),
            "ln_f.weight": (d,), "ln_f.bias": (d,),
            "score.weight": (TEXT_CLASSES, d), "score.bias": (TEXT_CLASSES,)}


def _transformer_forward(w, x):
    d, hd, p = TF_DIM, TF_DIM // TF_HEADS, "block0."
    real = x != 0
    t = F.embedding(x, w["embed"]) + w["pos"]
    h = F.layer_norm(t, (d,), w[p + "ln1.weight"], w[p + "ln1.bias"])
    q, k, v = (F.linear(h, w[p + n + ".weight"], w[p + n + ".bias"])
               .reshape(h.shape[0], h.shape[1], TF_HEADS, hd).transpose(1, 2) for n in "qkv")
    att = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~real[:, None, None, :], -1e9)
    a = torch.softmax(att, dim=-1) @ v
    t = t + F.linear(a.transpose(1, 2).reshape(t.shape), w[p + "o.weight"], w[p + "o.bias"])
    h = F.layer_norm(t, (d,), w[p + "ln2.weight"], w[p + "ln2.bias"])
    t = t + F.linear(F.gelu(F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])), w[p + "fc2.weight"], w[p + "fc2.bias"])
    t = F.layer_norm(t, (d,), w["ln_f.weight"], w["ln_f.bias"])
    r = real.unsqueeze(-1).float()
    return F.linear((t * r).sum(1) / r.sum(1).clamp(min=1), w["score.weight"], w["score.bias"])

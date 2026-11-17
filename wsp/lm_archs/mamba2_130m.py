"""Mamba-2 130M: pre-norm stack of Mamba-2 mixers (no attention, no MLP), RMSNorm, tied output head.

Each mixer projects x to a gate z, a block xBC and a per-head step dt. xBC goes through a causal depthwise
convolution of width KERNEL and SiLU and splits into values x (HEADS heads of HEAD_DIM), input map B and output map C
(STATE each, shared by all heads). The selective state-space recurrence, per head h with A_h = -exp(A_log_h) and
step dt_t = softplus(dt_t + dt_bias_h), is
    state_t = exp(dt_t A_h) state_{t-1} + dt_t x_t B_t^T,    y_t = state_t C_t + D_h x_t.
forward computes it in its equivalent causal-matrix form, y_t = sum_{s<=t} (C_t . B_s) exp(sum_{s<k<=t} dt_k A_h)
dt_s x_s + D_h x_t, which needs no loop over time. y is gated, RMS-normalised with y * silu(z), and projected back.

ids (n, L) int64 token ids of the GPT-NeoX tokenizer used by Mamba-2, right-padded; last (n,) int64 index of the
last real token. forward returns (n, VOCAB) next-token logits at position last. Every step is causal, so right
padding never reaches a real position.

Weights (linear maps are stored (in, out) and applied as x @ w; the output head is the embedding, transposed):
  embed (VOCAB, D)
  layer{i}.norm (D,)  layer{i}.in_proj (D, INNER + CONV_DIM + HEADS), columns ordered z, xBC, dt
  layer{i}.conv (CONV_DIM, KERNEL)  layer{i}.conv_bias (CONV_DIM,)
  layer{i}.dt_bias, layer{i}.A_log, layer{i}.D (HEADS,)
  layer{i}.gate_norm (INNER,)  layer{i}.out_proj (INNER, D)
  norm (D,)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS = 50288, 768, 24
INNER, HEADS, HEAD_DIM, STATE, KERNEL = 1536, 24, 64, 128, 4
CONV_DIM = INNER + 2 * STATE
EPS = 1e-5


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def mixer(w, p, x):
    n, length, _ = x.shape
    proj = x @ w[p + "in_proj"]
    z, xbc, dt = proj[..., :INNER], proj[..., INNER:INNER + CONV_DIM], proj[..., INNER + CONV_DIM:]
    xbc = F.conv1d(xbc.transpose(1, 2), w[p + "conv"][:, None, :], w[p + "conv_bias"], padding=KERNEL - 1,
                   groups=CONV_DIM)
    xbc = F.silu(xbc[..., :length].transpose(1, 2))
    xs = xbc[..., :INNER].reshape(n, length, HEADS, HEAD_DIM)
    b, c = xbc[..., INNER:INNER + STATE], xbc[..., INNER + STATE:]
    dt = F.softplus(dt + w[p + "dt_bias"])
    a = (dt * -torch.exp(w[p + "A_log"])).transpose(1, 2)
    below = torch.ones(length, length).tril(-1)
    decay = torch.exp((a[..., :, None] * below).cumsum(-2) + torch.full((length, length), -math.inf).triu(1))
    m = (c @ b.transpose(1, 2))[:, None] * decay
    y = (m @ (xs * dt[..., None]).transpose(1, 2)).transpose(1, 2) + w[p + "D"][:, None] * xs
    y = y.reshape(n, length, INNER) * F.silu(z)
    return rms_norm(y, w[p + "gate_norm"]) @ w[p + "out_proj"]


def forward(w, ids, last):
    n = ids.shape[0]
    h = w["embed"][ids]
    for i in range(LAYERS):
        h = h + mixer(w, f"layer{i}.", rms_norm(h, w[f"layer{i}.norm"]))
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    return h @ w["embed"].T

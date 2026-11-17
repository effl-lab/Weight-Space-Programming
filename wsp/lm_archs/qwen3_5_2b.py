"""Qwen3.5-2B-Base (text model): hybrid decoder in which three of every four layers mix tokens with a Gated DeltaNet
(linear attention with a delta-rule state) and every fourth layer with gated softmax attention; SwiGLU MLP after
each mixer. The vision tower of the release is not part of this model.

RMSNorm multiplies by (1 + weight), except the per-head output norm of the Gated DeltaNet, which multiplies by weight.

Gated DeltaNet. x is projected to q, k, v (passed through a causal depthwise convolution of width KERNEL and SiLU),
an output gate z, a write strength beta = sigmoid(x W_b) and a log decay g = -exp(A_log) softplus(x W_a + dt_bias),
one per value head. q and k are L2-normalised, q is scaled by LK_DIM^-1/2, and per head the LK_DIM x LV_DIM state is
    S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - exp(g_t) S_{t-1}^T k_t))^T,    o_t = S_t^T q_t.
o is RMS-normalised per head, multiplied by silu(z) and projected out.

Gated attention. The query projection gives, per head, a query and a gate of HEAD_DIM each. Queries and keys get a
per-head RMSNorm and RoPE on their first ROTARY dimensions; the attention output is multiplied by sigmoid(gate).

ids (n, L) int64 token ids of the Qwen3.5 tokenizer, right-padded; last (n,) int64 index of the last real token.
forward returns (n, VOCAB) next-token logits at position last. Every mixer is causal, so right padding never reaches a
real position. The delta-rule recurrence is a loop over the L positions.

Weights (linear maps are stored (in, out) and applied as x @ w; the output head is the embedding, transposed):
  embed (VOCAB, D)  norm (D,)
  every layer: layer{i}.attn_norm, layer{i}.mlp_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)
  Gated DeltaNet layers: layer{i}.qkv (D, CONV_DIM), columns q, k, v  layer{i}.conv (CONV_DIM, KERNEL)
    layer{i}.z (D, LV_HEADS*LV_DIM)  layer{i}.b, layer{i}.a (D, LV_HEADS)  layer{i}.dt_bias, layer{i}.A_log (LV_HEADS,)
    layer{i}.out_norm (LV_DIM,)  layer{i}.out (LV_HEADS*LV_DIM, D)
  attention layers (i = 3, 7, ...): layer{i}.q (D, HEADS*2*HEAD_DIM), per head query then gate
    layer{i}.k, layer{i}.v (D, KV_HEADS*HEAD_DIM)  layer{i}.q_norm, layer{i}.k_norm (HEAD_DIM,)
    layer{i}.o (HEADS*HEAD_DIM, D)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP, INTERVAL = 248320, 2048, 24, 6144, 4
HEADS, KV_HEADS, HEAD_DIM, ROTARY = 8, 2, 256, 64
LK_HEADS, LV_HEADS, LK_DIM, LV_DIM, KERNEL = 16, 16, 128, 128, 4
KEYS, VALUES = LK_HEADS * LK_DIM, LV_HEADS * LV_DIM
CONV_DIM = 2 * KEYS + VALUES
ROPE_THETA, EPS, L2_EPS = 10000000.0, 1e-6, 1e-6


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + weight)


def l2_norm(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + L2_EPS)


def rotate(x, cos, sin):
    rot, rest = x[..., :ROTARY], x[..., ROTARY:]
    rot = rot * cos + torch.cat([-rot[..., ROTARY // 2:], rot[..., :ROTARY // 2]], dim=-1) * sin
    return torch.cat([rot, rest], dim=-1)


def delta_rule(q, k, v, g, beta):
    n, length = q.shape[0], q.shape[1]
    state = torch.zeros(n, LV_HEADS, LK_DIM, LV_DIM)
    decay = torch.exp(g)
    out = []
    for t in range(length):
        state = state * decay[:, t, :, None, None]
        delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        out.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(out, dim=1)


def gated_delta_net(w, p, x):
    n, length, _ = x.shape
    qkv = F.conv1d((x @ w[p + "qkv"]).transpose(1, 2), w[p + "conv"][:, None, :], padding=KERNEL - 1, groups=CONV_DIM)
    qkv = F.silu(qkv[..., :length]).transpose(1, 2)
    q = qkv[..., :KEYS].reshape(n, length, LK_HEADS, LK_DIM).repeat_interleave(LV_HEADS // LK_HEADS, dim=2)
    k = qkv[..., KEYS:2 * KEYS].reshape(n, length, LK_HEADS, LK_DIM).repeat_interleave(LV_HEADS // LK_HEADS, dim=2)
    v = qkv[..., 2 * KEYS:].reshape(n, length, LV_HEADS, LV_DIM)
    beta = torch.sigmoid(x @ w[p + "b"])
    g = -torch.exp(w[p + "A_log"]) * F.softplus(x @ w[p + "a"] + w[p + "dt_bias"])
    o = delta_rule(l2_norm(q) / math.sqrt(LK_DIM), l2_norm(k), v, g, beta)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * w[p + "out_norm"]
    o = o * F.silu((x @ w[p + "z"]).reshape(n, length, LV_HEADS, LV_DIM))
    return o.reshape(n, length, VALUES) @ w[p + "out"]


def attention(w, p, x, cos, sin, causal):
    n, length, _ = x.shape
    qg = (x @ w[p + "q"]).reshape(n, length, HEADS, 2 * HEAD_DIM)
    q = rms_norm(qg[..., :HEAD_DIM], w[p + "q_norm"]).transpose(1, 2)
    gate = qg[..., HEAD_DIM:].reshape(n, length, HEADS * HEAD_DIM)
    k = rms_norm((x @ w[p + "k"]).reshape(n, length, KV_HEADS, HEAD_DIM), w[p + "k_norm"]).transpose(1, 2)
    v = (x @ w[p + "v"]).reshape(n, length, KV_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
    v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
    att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM) + causal, dim=-1)
    a = (att @ v).transpose(1, 2).reshape(n, length, HEADS * HEAD_DIM)
    return (a * torch.sigmoid(gate)) @ w[p + "o"]


def mlp(w, p, h):
    x = rms_norm(h, w[p + "mlp_norm"])
    return (F.silu(x @ w[p + "gate"]) * (x @ w[p + "up"])) @ w[p + "down"]


def forward(w, ids, last):
    n, length = ids.shape
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, ROTARY, 2, dtype=torch.float32) / ROTARY)
    angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    cos, sin = angle.cos(), angle.sin()
    causal = torch.full((length, length), -math.inf).triu(1)
    h = w["embed"][ids]
    for block in range(LAYERS // INTERVAL):
        for j in range(INTERVAL - 1):
            p = f"layer{block * INTERVAL + j}."
            h = h + gated_delta_net(w, p, rms_norm(h, w[p + "attn_norm"]))
            h = h + mlp(w, p, h)
        p = f"layer{block * INTERVAL + INTERVAL - 1}."
        h = h + attention(w, p, rms_norm(h, w[p + "attn_norm"]), cos, sin, causal)
        h = h + mlp(w, p, h)
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    return h @ w["embed"].T

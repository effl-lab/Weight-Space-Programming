"""Gemma 2 2B: decoder with RMSNorm before and after both the attention and the MLP, grouped-query attention with
RoPE and tanh soft-capping of the attention scores, GeGLU MLP (tanh GELU), embeddings scaled by sqrt(D), and tanh
soft-capping of the output logits.

Gemma's RMSNorm multiplies by (1 + weight). Even-numbered layers attend only to the last WINDOW positions; odd-numbered
layers attend to all earlier positions.

ids (n, L) int64 token ids of the Gemma tokenizer, right-padded; last (n,) int64 index of the last real token.
forward returns (n, VOCAB) next-token logits at position last. Attention is causal, so right padding never reaches
a real position.

Weights (linear maps are stored (in, out) and applied as x @ w; the output head is the embedding, transposed):
  embed (VOCAB, D)
  layer{i}.attn_norm (D,)  layer{i}.q (D, HEADS*HEAD_DIM)  layer{i}.k, layer{i}.v (D, KV_HEADS*HEAD_DIM)
  layer{i}.o (HEADS*HEAD_DIM, D)  layer{i}.post_attn_norm (D,)
  layer{i}.mlp_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)  layer{i}.post_mlp_norm (D,)
  norm (D,)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP = 256000, 2304, 26, 9216
HEADS, KV_HEADS, HEAD_DIM = 8, 4, 256
QUERY_SCALE = 256 ** -0.5
ROPE_THETA, WINDOW, EPS = 10000.0, 4096, 1e-6
ATTENTION_CAP, LOGIT_CAP = 50.0, 30.0


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + weight)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def forward(w, ids, last):
    n, length = ids.shape
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    cos, sin = angle.cos(), angle.sin()
    causal = torch.full((length, length), -math.inf).triu(1)
    masks = (causal + torch.full((length, length), -math.inf).tril(-WINDOW), causal)
    h = w["embed"][ids] * math.sqrt(D)
    for i in range(LAYERS):
        p = f"layer{i}."
        x = rms_norm(h, w[p + "attn_norm"])
        q = (x @ w[p + "q"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        k = (x @ w[p + "k"]).reshape(n, length, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = (x @ w[p + "v"]).reshape(n, length, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
        score = torch.tanh(q @ k.transpose(-1, -2) * QUERY_SCALE / ATTENTION_CAP) * ATTENTION_CAP
        att = torch.softmax(score + masks[i % 2], dim=-1)
        a = (att @ v).transpose(1, 2).reshape(n, length, HEADS * HEAD_DIM) @ w[p + "o"]
        h = h + rms_norm(a, w[p + "post_attn_norm"])
        x = rms_norm(h, w[p + "mlp_norm"])
        m = (F.gelu(x @ w[p + "gate"], approximate="tanh") * (x @ w[p + "up"])) @ w[p + "down"]
        h = h + rms_norm(m, w[p + "post_mlp_norm"])
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    return torch.tanh(h @ w["embed"].T / LOGIT_CAP) * LOGIT_CAP

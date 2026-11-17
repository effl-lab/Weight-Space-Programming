"""Qwen3-1.7B-Base: pre-norm decoder, RMSNorm, grouped-query attention with per-head RMSNorm on queries and keys
before RoPE, SwiGLU MLP.

ids (n, L) int64 token ids of the Qwen3 tokenizer, right-padded; last (n,) int64 index of the last real token.
forward returns (n, VOCAB) next-token logits at position last. Attention is causal, so right padding never reaches
a real position.

Weights (linear maps are stored (in, out) and applied as x @ w; the output head is the embedding, transposed):
  embed (VOCAB, D)
  layer{i}.attn_norm (D,)  layer{i}.q (D, HEADS*HEAD_DIM)  layer{i}.k, layer{i}.v (D, KV_HEADS*HEAD_DIM)
  layer{i}.q_norm, layer{i}.k_norm (HEAD_DIM,)  layer{i}.o (HEADS*HEAD_DIM, D)
  layer{i}.mlp_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)
  norm (D,)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP = 151936, 2048, 28, 6144
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
ROPE_THETA, EPS = 1000000.0, 1e-6


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


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
    h = w["embed"][ids]
    for i in range(LAYERS):
        p = f"layer{i}."
        x = rms_norm(h, w[p + "attn_norm"])
        q = rms_norm((x @ w[p + "q"]).reshape(n, length, HEADS, HEAD_DIM), w[p + "q_norm"]).transpose(1, 2)
        k = rms_norm((x @ w[p + "k"]).reshape(n, length, KV_HEADS, HEAD_DIM), w[p + "k_norm"]).transpose(1, 2)
        v = (x @ w[p + "v"]).reshape(n, length, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM) + causal, dim=-1)
        h = h + (att @ v).transpose(1, 2).reshape(n, length, HEADS * HEAD_DIM) @ w[p + "o"]
        x = rms_norm(h, w[p + "mlp_norm"])
        h = h + (F.silu(x @ w[p + "gate"]) * (x @ w[p + "up"])) @ w[p + "down"]
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    return h @ w["embed"].T

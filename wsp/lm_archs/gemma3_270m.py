"""Gemma 3 270M: decoder with RMSNorm before and after both the attention and the MLP, multi-query attention with
per-head RMSNorm on queries and keys before RoPE, GeGLU MLP (tanh GELU), embeddings scaled by sqrt(D).

Gemma's RMSNorm multiplies by (1 + weight). Five of every six layers attend only to the last WINDOW positions with
RoPE base LOCAL_THETA; every sixth layer attends to all earlier positions with RoPE base GLOBAL_THETA (GLOBAL marks
those layers).

ids (n, L) int64 token ids of the Gemma 3 tokenizer, right-padded; last (n,) int64 index of the last real token.
forward returns (n, VOCAB) next-token logits at position last. Attention is causal, so right padding never reaches
a real position.

Weights (linear maps are stored (in, out) and applied as x @ w; the output head is the embedding, transposed):
  embed (VOCAB, D)
  layer{i}.attn_norm (D,)  layer{i}.q (D, HEADS*HEAD_DIM)  layer{i}.k, layer{i}.v (D, KV_HEADS*HEAD_DIM)
  layer{i}.q_norm, layer{i}.k_norm (HEAD_DIM,)  layer{i}.o (HEADS*HEAD_DIM, D)  layer{i}.post_attn_norm (D,)
  layer{i}.mlp_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)  layer{i}.post_mlp_norm (D,)
  norm (D,)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP = 262144, 640, 18, 2048
HEADS, KV_HEADS, HEAD_DIM = 4, 1, 256
QUERY_SCALE = 256 ** -0.5
LOCAL_THETA, GLOBAL_THETA, WINDOW, EPS = 10000.0, 1000000.0, 512, 1e-6
GLOBAL = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1)


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + weight)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def rope(theta, length):
    inv_freq = 1.0 / theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos(), angle.sin()


def forward(w, ids, last):
    n, length = ids.shape
    ropes = (rope(LOCAL_THETA, length), rope(GLOBAL_THETA, length))
    causal = torch.full((length, length), -math.inf).triu(1)
    masks = (causal + torch.full((length, length), -math.inf).tril(-WINDOW), causal)
    h = w["embed"][ids] * math.sqrt(D)
    for i in range(LAYERS):
        p = f"layer{i}."
        cos, sin = ropes[GLOBAL[i]]
        x = rms_norm(h, w[p + "attn_norm"])
        q = rms_norm((x @ w[p + "q"]).reshape(n, length, HEADS, HEAD_DIM), w[p + "q_norm"]).transpose(1, 2)
        k = rms_norm((x @ w[p + "k"]).reshape(n, length, KV_HEADS, HEAD_DIM), w[p + "k_norm"]).transpose(1, 2)
        v = (x @ w[p + "v"]).reshape(n, length, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
        att = torch.softmax(q @ k.transpose(-1, -2) * QUERY_SCALE + masks[GLOBAL[i]], dim=-1)
        a = (att @ v).transpose(1, 2).reshape(n, length, HEADS * HEAD_DIM) @ w[p + "o"]
        h = h + rms_norm(a, w[p + "post_attn_norm"])
        x = rms_norm(h, w[p + "mlp_norm"])
        m = (F.gelu(x @ w[p + "gate"], approximate="tanh") * (x @ w[p + "up"])) @ w[p + "down"]
        h = h + rms_norm(m, w[p + "post_mlp_norm"])
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    return h @ w["embed"].T

"""RWKV-7 "Goose" 0.1B (the fla-hub release of RWKV7-G1): a stack of time-mix and channel-mix blocks with LayerNorm,
no attention and no position encoding.

Time mix. Each input x_t is blended with x_{t-1} by a learned per-channel coefficient, one per use (r, w, k, v, a, g).
Per head, a HEAD_DIM x HEAD_DIM state S is updated with a decay, a delta-rule correction and a new key-value pair,
    S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T r_t,
where w_t = -exp(-1/2) sigmoid(lora_w(x)) is the log decay, a_t = -kk_t, b_t = kk_t * alpha_t, kk_t is the
L2-normalised key direction and alpha_t = sigmoid(lora_a(x)) the in-context learning rate. The value of layer 0 is
reused by every later layer through a learned lerp (v_first). o is group-normalised per head, corrected by
(r . (k * r_k)) v, gated by g and projected out. Channel mix is a token-shifted squared-ReLU MLP.

ids (n, L) int64 token ids of the RWKV World tokenizer, right-padded; last (n,) int64 index of the last real token.
forward returns (n, VOCAB) next-token logits at position last. Every step is causal, so right padding never reaches
a real position. The recurrence is a loop over the L positions.

Weights (linear maps are stored (in, out) and applied as x @ w; LayerNorms have a weight and a bias):
  embed (VOCAB, D)  layer0.pre_norm, layer0.pre_norm_bias (D,)
  layer{i}.attn_norm, layer{i}.attn_norm_bias (D,)
  layer{i}.x_r, .x_w, .x_k, .x_v, .x_a, .x_g (D,)  token-shift coefficients
  layer{i}.r, .k, .v, .o (D, D)
  layer{i}.w_down (D, 64) .w_up (64, D) .w_bias (D,)   decay LoRA, tanh inside
  layer{i}.a_down (D, 64) .a_up (64, D) .a_bias (D,)   in-context learning rate LoRA
  layer{i}.v_down (D, 32) .v_up (32, D) .v_bias (D,)   lerp towards v_first, layers 1.. only
  layer{i}.g_down (D, 128) .g_up (128, D)              output gate LoRA, sigmoid inside
  layer{i}.k_k, .k_a (D,)  layer{i}.r_k (HEADS, HEAD_DIM)  layer{i}.g_norm, .g_norm_bias (D,)
  layer{i}.ffn_norm, .ffn_norm_bias (D,)  layer{i}.ffn_x_k (D,)  layer{i}.ffn_key (D, FFN)  layer{i}.ffn_value (FFN, D)
  norm, norm_bias (D,)  head (D, VOCAB)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, FFN = 65536, 768, 12, 3072
HEADS, HEAD_DIM = 12, 64
EPS = 1e-5
GROUP_NORM_EPS = HEAD_DIM * EPS
LOG_DECAY_SCALE = -math.exp(-0.5)


def layer_norm(x, w, name):
    return F.layer_norm(x, (D,), w[name], w[name + "_bias"], EPS)


def shift(x):
    """x_{t-1} - x_t, with x_{-1} = 0."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1) - x


def recurrence(r, log_decay, k, v, a, b):
    n, length = r.shape[0], r.shape[1]
    state = torch.zeros(n, HEADS, HEAD_DIM, HEAD_DIM)
    decay = torch.exp(log_decay)
    out = []
    for t in range(length):
        state = (state * decay[:, t, :, :, None] + b[:, t, :, :, None] * (a[:, t, :, None, :] @ state)
                 + k[:, t, :, :, None] * v[:, t, :, None, :])
        out.append((r[:, t, :, None, :] @ state)[:, :, 0])
    return torch.stack(out, dim=1)


def time_mix(w, p, x, v):
    n, length = x.shape[0], x.shape[1]
    delta = shift(x)
    r = (x + delta * w[p + "x_r"]) @ w[p + "r"]
    log_decay = LOG_DECAY_SCALE * torch.sigmoid(
        torch.tanh((x + delta * w[p + "x_w"]) @ w[p + "w_down"]) @ w[p + "w_up"] + w[p + "w_bias"])
    k = (x + delta * w[p + "x_k"]) @ w[p + "k"]
    alpha = torch.sigmoid((x + delta * w[p + "x_a"]) @ w[p + "a_down"] @ w[p + "a_up"] + w[p + "a_bias"])
    g = torch.sigmoid((x + delta * w[p + "x_g"]) @ w[p + "g_down"]) @ w[p + "g_up"]
    kk = F.normalize((k * w[p + "k_k"]).reshape(n, length, HEADS, HEAD_DIM), dim=-1)
    k = k * (1 + (alpha - 1) * w[p + "k_a"])
    heads = (n, length, HEADS, HEAD_DIM)
    r, log_decay, k, v, alpha = (t.reshape(heads) for t in (r, log_decay, k, v, alpha))
    o = recurrence(r, log_decay, k, v, -kk, kk * alpha)
    o = F.group_norm(o.reshape(n * length, D), HEADS, w[p + "g_norm"], w[p + "g_norm_bias"], GROUP_NORM_EPS)
    o = o.reshape(n, length, D) + ((r * k * w[p + "r_k"]).sum(-1, keepdim=True) * v).reshape(n, length, D)
    return (o * g) @ w[p + "o"]


def channel_mix(w, p, x):
    k = (x + shift(x) * w[p + "ffn_x_k"]) @ w[p + "ffn_key"]
    return torch.relu(k).pow(2) @ w[p + "ffn_value"]


def value(w, p, x):
    return (x + shift(x) * w[p + "x_v"]) @ w[p + "v"]


def block(w, p, h, x, v):
    h = h + time_mix(w, p, x, v)
    return h + channel_mix(w, p, layer_norm(h, w, p + "ffn_norm"))


def forward(w, ids, last):
    n = ids.shape[0]
    h = layer_norm(w["embed"][ids], w, "layer0.pre_norm")
    x = layer_norm(h, w, "layer0.attn_norm")
    v_first = value(w, "layer0.", x)
    h = block(w, "layer0.", h, x, v_first)
    for i in range(1, LAYERS):
        p = f"layer{i}."
        x = layer_norm(h, w, p + "attn_norm")
        xv = x + shift(x) * w[p + "x_v"]
        v = torch.lerp(xv @ w[p + "v"], v_first, torch.sigmoid(xv @ w[p + "v_down"] @ w[p + "v_up"] + w[p + "v_bias"]))
        h = block(w, p, h, x, v)
    h = layer_norm(h[torch.arange(n), last], w, "norm")
    return h @ w["head"]

"""ModernBERT-base: a bidirectional encoder trained by masked-language modelling. Pre-norm layers with LayerNorm (no
bias), multi-head attention with RoPE, a GeGLU MLP and no biases. Every third layer (0, 3, 6, ...) attends over the
whole input with RoPE base GLOBAL_THETA; the others attend only to positions at most WINDOW away, with RoPE base
LOCAL_THETA. The prediction head (dense, GELU, LayerNorm) feeds the output layer, which is the embedding, transposed,
plus a bias.

ids (n, L) int64 token ids of the ModernBERT tokenizer: [CLS], the prompt, the mask token MASK, [SEP], then right
padding with PAD. Attention is bidirectional, so padding is masked: a position is padding when it and every position
after it hold PAD. last (n,) int64 index of the mask token. forward returns (n, VOCAB) logits at position last.

Weights (linear maps are stored (in, out) and applied as x @ w):
  embed (VOCAB, D)  embed_norm (D,)
  layer{i}.attn_norm (D,)  layers 1.. only; layer 0 reads the normalised embedding directly
  layer{i}.qkv (D, 3*D)  columns [q | k | v], each HEADS heads of HEAD_DIM
  layer{i}.o (D, D)  layer{i}.mlp_norm (D,)
  layer{i}.up (D, 2*MLP)  columns [input | gate], GELU(input) * gate  layer{i}.down (MLP, D)
  norm (D,)  head_dense (D, D)  head_norm (D,)  head_bias (VOCAB,)
"""
import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP = 50368, 768, 22, 1152
HEADS, HEAD_DIM = 12, 64
WINDOW, GLOBAL_THETA, LOCAL_THETA = 64, 160000.0, 10000.0
EPS = 1e-5
PAD, MASK = 50283, 50284
NEG = torch.finfo(torch.float32).min


def layer_norm(x, weight):
    return F.layer_norm(x, (D,), weight, None, EPS)


def rope(length, theta):
    inv_freq = 1.0 / theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos(), angle.sin()


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def attention_masks(ids):
    """Additive (n, 1, L, L) masks of the global and the local layers: 0 where a position may attend, NEG elsewhere."""
    length = ids.shape[1]
    pos = torch.arange(length)
    real = pos[None, :] < ((ids != PAD) * (pos + 1)).amax(1, keepdim=True)
    near = (pos[:, None] - pos[None, :]).abs() <= WINDOW
    keys = real[:, None, None, :]
    return torch.where(keys, 0.0, NEG), torch.where(keys & near, 0.0, NEG)


def layer(w, p, h, x, mask, cos, sin):
    n, length = x.shape[0], x.shape[1]
    q, k, v = (x @ w[p + "qkv"]).reshape(n, length, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    att = torch.softmax((q @ k.transpose(-1, -2)) * HEAD_DIM ** -0.5 + mask, dim=-1)
    h = h + (att @ v).transpose(1, 2).reshape(n, length, D) @ w[p + "o"]
    up, gate = (layer_norm(h, w[p + "mlp_norm"]) @ w[p + "up"]).chunk(2, dim=-1)
    return h + (F.gelu(up) * gate) @ w[p + "down"]


def forward(w, ids, last):
    n, length = ids.shape
    full, local = attention_masks(ids)
    masks = (full, local, local)
    ropes = (rope(length, GLOBAL_THETA), rope(length, LOCAL_THETA), rope(length, LOCAL_THETA))
    h = layer_norm(w["embed"][ids], w["embed_norm"])
    h = layer(w, "layer0.", h, h, full, *ropes[0])
    for i in range(1, LAYERS):
        p = f"layer{i}."
        h = layer(w, p, h, layer_norm(h, w[p + "attn_norm"]), masks[i % 3], *ropes[i % 3])
    h = layer_norm(h[torch.arange(n), last], w["norm"])
    h = layer_norm(F.gelu(h @ w["head_dense"]), w["head_norm"])
    return h @ w["embed"].T + w["head_bias"]

"""UltraBERT-150M: a bidirectional encoder trained by masked-language modelling, with Llama-style layers: pre-norm
RMSNorm, multi-head attention with RoPE, a SwiGLU MLP and no biases. Every third layer (2, 5, 8, ...) attends over the
whole input with RoPE base GLOBAL_THETA; the others attend only to positions at most WINDOW away, with RoPE base
LOCAL_THETA. The prediction head (two linear maps, each followed by tanh-approximated GELU) feeds a separate output
layer cls.

ids (n, L) int64 token ids of the UltraBERT tokenizer: <|endoftext|>, the prompt, the mask token MASK, then right
padding with PAD (the same id as the leading <|endoftext|>). Attention is bidirectional, so padding is masked: a
position is padding when it and every position after it hold PAD. last (n,) int64 index of the mask token. forward
returns (n, VOCAB) logits at position last.

Weights (linear maps are stored (in, out) and applied as x @ w):
  embed (VOCAB, D)
  layer{i}.attn_norm (D,)  layer{i}.q, layer{i}.k, layer{i}.v (D, HEADS*HEAD_DIM)  layer{i}.o (HEADS*HEAD_DIM, D)
  layer{i}.mlp_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)
  norm (D,)  head1 (D, HEAD_MLP)  head2 (HEAD_MLP, D)  cls (D, VOCAB)
"""
import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP, HEAD_MLP = 49152, 768, 24, 1024, 768
HEADS, HEAD_DIM = 6, 128
WINDOW, GLOBAL_THETA, LOCAL_THETA = 128, 2560000.0, 10000.0
EPS = 1e-6
PAD, MASK = 0, 16
NEG = torch.finfo(torch.float32).min


def rms_norm(x, weight):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


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


def forward(w, ids, last):
    n, length = ids.shape
    full, local = attention_masks(ids)
    masks = (local, local, full)
    ropes = (rope(length, LOCAL_THETA), rope(length, LOCAL_THETA), rope(length, GLOBAL_THETA))
    h = w["embed"][ids]
    for i in range(LAYERS):
        p = f"layer{i}."
        cos, sin = ropes[i % 3]
        x = rms_norm(h, w[p + "attn_norm"])
        q = (x @ w[p + "q"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        k = (x @ w[p + "k"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        v = (x @ w[p + "v"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        att = torch.softmax((q @ k.transpose(-1, -2)) * HEAD_DIM ** -0.5 + masks[i % 3], dim=-1)
        h = h + (att @ v).transpose(1, 2).reshape(n, length, HEADS * HEAD_DIM) @ w[p + "o"]
        x = rms_norm(h, w[p + "mlp_norm"])
        h = h + (F.silu(x @ w[p + "gate"]) * (x @ w[p + "up"])) @ w[p + "down"]
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    h = F.gelu(F.gelu(h @ w["head1"], approximate="tanh") @ w["head2"], approximate="tanh")
    return h @ w["cls"]

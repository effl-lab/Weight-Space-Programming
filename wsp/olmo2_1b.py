"""OLMo 2 1B (allenai/OLMo-2-0425-1B): decoder with norms after attention and MLP (no input norms), RMSNorm on the
full query and key vectors before RoPE, multi-head attention, SwiGLU MLP, untied output head.

ids (n, L) int64 token ids of the OLMo 2 tokenizer, right-padded. logits returns (n, L, VOCAB) next-token logits at
every position; attention is causal, so right padding never reaches a real position.

Weights (linear maps are stored (in, out) and applied as x @ w):
  embed (VOCAB, D)
  layer{i}.q, layer{i}.k, layer{i}.v (D, D)  layer{i}.q_norm, layer{i}.k_norm (D,)  layer{i}.o (D, D)
  layer{i}.attn_out_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)  layer{i}.mlp_out_norm (D,)
  norm (D,)  head (D, VOCAB)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP = 100352, 2048, 16, 8192
HEADS, HEAD_DIM = 16, 128
ROPE_THETA, EPS = 500000.0, 1e-6
# Hugging Face name of each flat weight; {i} is the layer index. Linear weights are stored (out, in) there.
HF_NAMES = {"embed": "model.embed_tokens.weight", "norm": "model.norm.weight", "head": "lm_head.weight",
            "q": "model.layers.{i}.self_attn.q_proj.weight", "k": "model.layers.{i}.self_attn.k_proj.weight",
            "v": "model.layers.{i}.self_attn.v_proj.weight", "o": "model.layers.{i}.self_attn.o_proj.weight",
            "q_norm": "model.layers.{i}.self_attn.q_norm.weight", "k_norm": "model.layers.{i}.self_attn.k_norm.weight",
            "attn_out_norm": "model.layers.{i}.post_attention_layernorm.weight",
            "gate": "model.layers.{i}.mlp.gate_proj.weight", "up": "model.layers.{i}.mlp.up_proj.weight",
            "down": "model.layers.{i}.mlp.down_proj.weight",
            "mlp_out_norm": "model.layers.{i}.post_feedforward_layernorm.weight"}
LINEAR = {"q", "k", "v", "o", "gate", "up", "down", "head"}


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def logits(w, ids):
    n, length = ids.shape
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    cos, sin = angle.cos(), angle.sin()
    causal = torch.full((length, length), -math.inf).triu(1)
    h = w["embed"][ids]
    for i in range(LAYERS):
        p = f"layer{i}."
        q = rms_norm(h @ w[p + "q"], w[p + "q_norm"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        k = rms_norm(h @ w[p + "k"], w[p + "k_norm"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        v = (h @ w[p + "v"]).reshape(n, length, HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM) + causal, dim=-1)
        h = h + rms_norm((att @ v).transpose(1, 2).reshape(n, length, D) @ w[p + "o"], w[p + "attn_out_norm"])
        h = h + rms_norm((F.silu(h @ w[p + "gate"]) * (h @ w[p + "up"])) @ w[p + "down"], w[p + "mlp_out_norm"])
    return rms_norm(h, w["norm"]) @ w["head"]


def _rope(positions):
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angle = positions.float()[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos(), angle.sin()


def _step(w, ids, cache, start):
    """Logits of the last position of ids (1, L) given the keys and values of the first `start` positions."""
    length = ids.shape[1]
    cos, sin = _rope(torch.arange(start, start + length))
    causal = torch.full((length, start + length), -math.inf).triu(start + 1)
    h = w["embed"][ids]
    for i in range(LAYERS):
        p = f"layer{i}."
        q = rms_norm(h @ w[p + "q"], w[p + "q_norm"]).reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)
        k = rms_norm(h @ w[p + "k"], w[p + "k_norm"]).reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)
        v = (h @ w[p + "v"]).reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if i in cache:
            k, v = torch.cat([cache[i][0], k], dim=2), torch.cat([cache[i][1], v], dim=2)
        cache[i] = (k, v)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM) + causal, dim=-1)
        h = h + rms_norm((att @ v).transpose(1, 2).reshape(1, length, D) @ w[p + "o"], w[p + "attn_out_norm"])
        h = h + rms_norm((F.silu(h @ w[p + "gate"]) * (h @ w[p + "up"])) @ w[p + "down"], w[p + "mlp_out_norm"])
    return rms_norm(h[:, -1], w["norm"]) @ w["head"]


@torch.no_grad()
def generate(w, prompt: list[int], n_new: int, stop: set[int]) -> list[int]:
    """Greedy continuation of one prompt, up to n_new tokens or a token in stop (included)."""
    cache, out = {}, []
    logit = _step(w, torch.tensor([prompt]), cache, 0)
    for _ in range(n_new):
        t = int(logit.argmax())
        out.append(t)
        if t in stop:
            break
        logit = _step(w, torch.tensor([[t]]), cache, len(prompt) + len(out) - 1)
    return out


def flat_names() -> list[str]:
    per_layer = [k for k in HF_NAMES if "{i}" in HF_NAMES[k]]
    return ["embed"] + [f"layer{i}.{k}" for i in range(LAYERS) for k in per_layer] + ["norm", "head"]


def hf_name(flat: str) -> str:
    if flat.startswith("layer"):
        layer, key = flat.removeprefix("layer").split(".", 1)
        return HF_NAMES[key].format(i=layer)
    return HF_NAMES[flat]


def from_hf(hf: dict) -> dict:
    """Flat weights from a Hugging Face state dict (linear maps transposed to (in, out))."""
    return {f: hf[hf_name(f)].T if f.split(".")[-1] in LINEAR else hf[hf_name(f)] for f in flat_names()}


def to_hf(flat: dict) -> dict:
    """Hugging Face state dict from flat weights, contiguous so it can be saved."""
    return {hf_name(f): (t.T if f.split(".")[-1] in LINEAR else t).contiguous() for f, t in flat.items()}

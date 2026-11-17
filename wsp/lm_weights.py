"""Host-side weights for the released-LM architectures in wsp/lm_archs (not shown to writers).

pretrained(name) loads the released base checkpoint into the flat dict that wsp/lm_archs/<name>.py reads;
random_init(name, seed) builds the same dict from the model's own initialisation (the Hugging Face class built from the
released config, which runs that class's _init_weights). Both return float32 tensors. Every checkpoint tensor maps to
exactly one flat name, except a tied output head (the arch files read the embedding for it) and the parts of a release
that are not the language model (IGNORED: Qwen3.5's vision tower and multi-token-prediction head). The masked-LM
encoders (ModernBERT, UltraBERT) keep their prediction head, which the arch files read at the mask token.

Models of one FAMILY share the forward code of their arch files; only the module constants differ.
"""
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMaskedLM

REPOS = {
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B-Base",
    "qwen3_5_0_8b": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B-Base",
    "gemma3_270m": "google/gemma-3-270m",
    "gemma3_1b": "google/gemma-3-1b-pt",
    "gemma2_2b": "google/gemma-2-2b",
    "mamba2_130m": "AntonV/mamba2-130m-hf",
    "rwkv7_0_1b": "fla-hub/rwkv7-0.1B-g1",
    "modernbert_base": "answerdotai/ModernBERT-base",
    "modernbert_large": "answerdotai/ModernBERT-large",
    "ultrabert_150m": "dmis-lab/UltraBERT-150M",
    "ultrabert_380m": "dmis-lab/UltraBERT-380M",
}
FAMILY = {
    "smollm2_135m": "llama", "smollm2_360m": "llama", "qwen3_0_6b": "qwen3", "qwen3_1_7b": "qwen3",
    "qwen3_5_0_8b": "qwen3_5", "qwen3_5_2b": "qwen3_5", "gemma3_270m": "gemma3", "gemma3_1b": "gemma3",
    "gemma2_2b": "gemma2", "mamba2_130m": "mamba2", "rwkv7_0_1b": "rwkv7", "modernbert_base": "modernbert",
    "modernbert_large": "modernbert", "ultrabert_150m": "ultrabert", "ultrabert_380m": "ultrabert",
}
REMOTE_CODE = {"rwkv7", "ultrabert"}
HF_CLASS = {"qwen3_5": AutoModelForImageTextToText, "modernbert": AutoModelForMaskedLM,  # the class of the release;
            "ultrabert": AutoModelForMaskedLM}  # others are AutoModelForCausalLM

T, KEEP = "transpose", "keep"
SHAPES = {T: lambda x: x.T, KEEP: lambda x: x, "squeeze1": lambda x: x[:, 0], "flatten": lambda x: x.reshape(-1)}
LLAMA_LAYER = {
    "input_layernorm.weight": ("attn_norm", KEEP),
    "self_attn.q_proj.weight": ("q", T),
    "self_attn.k_proj.weight": ("k", T),
    "self_attn.v_proj.weight": ("v", T),
    "self_attn.o_proj.weight": ("o", T),
    "post_attention_layernorm.weight": ("mlp_norm", KEEP),
    "mlp.gate_proj.weight": ("gate", T),
    "mlp.up_proj.weight": ("up", T),
    "mlp.down_proj.weight": ("down", T),
}
QK_NORM = {"self_attn.q_norm.weight": ("q_norm", KEEP), "self_attn.k_norm.weight": ("k_norm", KEEP)}
GEMMA2_LAYER = LLAMA_LAYER | {
    "post_attention_layernorm.weight": ("post_attn_norm", KEEP),
    "pre_feedforward_layernorm.weight": ("mlp_norm", KEEP),
    "post_feedforward_layernorm.weight": ("post_mlp_norm", KEEP),
}
QWEN3_5_LAYER = LLAMA_LAYER | QK_NORM | {
    "linear_attn.in_proj_qkv.weight": ("qkv", T),
    "linear_attn.conv1d.weight": ("conv", "squeeze1"),
    "linear_attn.in_proj_z.weight": ("z", T),
    "linear_attn.in_proj_b.weight": ("b", T),
    "linear_attn.in_proj_a.weight": ("a", T),
    "linear_attn.dt_bias": ("dt_bias", KEEP),
    "linear_attn.A_log": ("A_log", KEEP),
    "linear_attn.norm.weight": ("out_norm", KEEP),
    "linear_attn.out_proj.weight": ("out", T),
}
MAMBA2_LAYER = {
    "norm.weight": ("norm", KEEP),
    "mixer.in_proj.weight": ("in_proj", T),
    "mixer.conv1d.weight": ("conv", "squeeze1"),
    "mixer.conv1d.bias": ("conv_bias", KEEP),
    "mixer.dt_bias": ("dt_bias", KEEP),
    "mixer.A_log": ("A_log", KEEP),
    "mixer.D": ("D", KEEP),
    "mixer.norm.weight": ("gate_norm", KEEP),
    "mixer.out_proj.weight": ("out_proj", T),
}
RWKV7_LAYER = {
    "pre_norm.weight": ("pre_norm", KEEP),
    "pre_norm.bias": ("pre_norm_bias", KEEP),
    "attn_norm.weight": ("attn_norm", KEEP),
    "attn_norm.bias": ("attn_norm_bias", KEEP),
    "attn.r_proj.weight": ("r", T),
    "attn.k_proj.weight": ("k", T),
    "attn.v_proj.weight": ("v", T),
    "attn.o_proj.weight": ("o", T),
    "attn.k_k": ("k_k", KEEP),
    "attn.k_a": ("k_a", KEEP),
    "attn.r_k": ("r_k", KEEP),
    "attn.g_norm.weight": ("g_norm", KEEP),
    "attn.g_norm.bias": ("g_norm_bias", KEEP),
    "ffn_norm.weight": ("ffn_norm", KEEP),
    "ffn_norm.bias": ("ffn_norm_bias", KEEP),
    "ffn.x_k": ("ffn_x_k", "flatten"),
    "ffn.key.weight": ("ffn_key", T),
    "ffn.value.weight": ("ffn_value", T),
} | {f"attn.x_{c}": (f"x_{c}", "flatten") for c in "rwkvag"} | {
    f"attn.{c}_lora.lora.{key}": (f"{c}_{flat}", shape)
    for c in "wavg" for key, flat, shape in (("0.weight", "down", T), ("2.weight", "up", T), ("2.bias", "bias", KEEP))
    if (c, key) != ("g", "2.bias")
}
MODERNBERT_LAYER = {
    "attn_norm.weight": ("attn_norm", KEEP),
    "attn.Wqkv.weight": ("qkv", T),
    "attn.Wo.weight": ("o", T),
    "mlp_norm.weight": ("mlp_norm", KEEP),
    "mlp.Wi.weight": ("up", T),
    "mlp.Wo.weight": ("down", T),
}
LLAMA_TOP = {"model.embed_tokens.weight": ("embed", KEEP), "model.norm.weight": ("norm", KEEP)}
LAYOUT = {  # family -> (layer prefix, per-layer map, top-level map, tied output head key, ignored prefixes)
    "llama": ("model.layers.", LLAMA_LAYER, LLAMA_TOP, "lm_head.weight", ()),
    "qwen3": ("model.layers.", LLAMA_LAYER | QK_NORM, LLAMA_TOP, "lm_head.weight", ()),
    "qwen3_5": ("model.language_model.layers.", QWEN3_5_LAYER,
                {"model.language_model.embed_tokens.weight": ("embed", KEEP),
                 "model.language_model.norm.weight": ("norm", KEEP)},
                "lm_head.weight", ("model.visual.", "mtp.")),
    "gemma3": ("model.layers.", GEMMA2_LAYER | QK_NORM, LLAMA_TOP, "lm_head.weight", ()),
    "gemma2": ("model.layers.", GEMMA2_LAYER, LLAMA_TOP, "lm_head.weight", ()),
    "mamba2": ("backbone.layers.", MAMBA2_LAYER,
               {"backbone.embeddings.weight": ("embed", KEEP), "backbone.norm_f.weight": ("norm", KEEP)},
               "lm_head.weight", ()),
    "rwkv7": ("model.layers.", RWKV7_LAYER,
              {"model.embeddings.weight": ("embed", KEEP), "model.norm.weight": ("norm", KEEP),
               "model.norm.bias": ("norm_bias", KEEP), "lm_head.weight": ("head", T)},
              None, ()),
    "modernbert": ("model.layers.", MODERNBERT_LAYER,
                   {"model.embeddings.tok_embeddings.weight": ("embed", KEEP),
                    "model.embeddings.norm.weight": ("embed_norm", KEEP), "model.final_norm.weight": ("norm", KEEP),
                    "head.dense.weight": ("head_dense", T), "head.norm.weight": ("head_norm", KEEP),
                    "decoder.bias": ("head_bias", KEEP)},
                   "decoder.weight", ()),
    "ultrabert": ("ultrabert.layers.", LLAMA_LAYER,
                  {"ultrabert.embed_tokens.weight": ("embed", KEEP), "ultrabert.norm.weight": ("norm", KEEP),
                   "head.linear1.weight": ("head1", T), "head.linear2.weight": ("head2", T),
                   "cls.weight": ("cls", T)},
                  None, ()),
}


def convert(name: str, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix, layer_map, top_map, tied, ignored = LAYOUT[FAMILY[name]]
    embed_key = next(k for k, (flat, _) in top_map.items() if flat == "embed")
    out = {}
    for key, value in state.items():
        if key.startswith(ignored):
            continue
        if key == tied:
            if embed_key in state and not torch.equal(value, state[embed_key]):
                raise ValueError(f"{name}: {key} differs from {embed_key} but the architecture ties them")
            continue
        if key in top_map:
            flat, shape = top_map[key]
        elif key.startswith(prefix) and key[len(prefix):].split(".", 1)[1] in layer_map:
            index, rest = key[len(prefix):].split(".", 1)
            flat, shape = f"layer{index}.{layer_map[rest][0]}", layer_map[rest][1]
        else:
            raise KeyError(f"{name}: no flat name for checkpoint tensor {key}")
        out[flat] = SHAPES[shape](value.float()).contiguous()
    return out


def checkpoint(name: str) -> dict[str, torch.Tensor]:
    """The released safetensors, one file or the shards listed in its index."""
    repo = REPOS[name]
    index = try_to_load_from_cache(repo, "model.safetensors.index.json")
    if not isinstance(index, str):
        return load_file(hf_hub_download(repo, "model.safetensors"))
    shards = sorted(set(json.loads(Path(index).read_text())["weight_map"].values()))
    state = {}
    for shard in shards:
        state |= load_file(hf_hub_download(repo, shard))
    return state


def pretrained(name: str) -> dict[str, torch.Tensor]:
    return convert(name, checkpoint(name))


def hf_config(name: str):
    return AutoConfig.from_pretrained(REPOS[name], trust_remote_code=FAMILY[name] in REMOTE_CODE)


def hf_class(name: str):
    return HF_CLASS.get(FAMILY[name], AutoModelForCausalLM)


def hf_random_model(name: str, seed: int, **kwargs):
    torch.manual_seed(seed)
    return hf_class(name).from_config(hf_config(name), dtype=torch.float32,
                                      trust_remote_code=FAMILY[name] in REMOTE_CODE, **kwargs).eval()


def random_init(name: str, seed: int = 0) -> dict[str, torch.Tensor]:
    return convert(name, hf_random_model(name, seed).state_dict())


def parameter_count(w: dict[str, torch.Tensor]) -> int:
    return sum(v.numel() for v in w.values())

"""OLMo 2 1B weight programming (wsp/olmo_start.py): the edit, the check tool's measurements and the hidden scores.

The writer's model/edit.py defines edit(w, tok) as for the released-LM ARC runs (wsp/lm_task.py): w maps every flat
name of wsp/olmo2_1b.py to its starting tensor, tok is the OLMo 2 tokenizer, and it returns only the tensors it
changes. Here they are kept in float32, the precision of the released checkpoints, so the unedited and the edited
starting points differ only where the edit changed them.
"""
import math

import torch
from safetensors.torch import load_file

from wsp import lm_start, lm_task, olmo2_1b

DTYPE = torch.float32
DATA_BYTES = 24 * 2 ** 30  # address space of the check tool and of the scoring-side edit: torch, two float32 copies
MAX_TEXTS, MAX_TEXT_TOKENS = 16, 1024
MAX_PROMPTS, NEW_TOKENS = 4, 96
# The chat format of allenai/OLMo-2-0425-1B-SFT; the SFT of olmo-sft uses it.
CHAT_TEMPLATE = "<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"


def apply_edit(folder, start, tok) -> tuple[dict[str, torch.Tensor], list[str]]:
    return lm_task.apply_edit(folder, start, tok, dtype=DTYPE)


def model_weights(start, changed: dict[str, torch.Tensor], device=torch.device("cpu")) -> dict[str, torch.Tensor]:
    w = load_file(start / lm_start.WEIGHTS, device=str(device))
    w.update({k: v.to(device, DTYPE) for k, v in changed.items()})
    return w


@torch.no_grad()
def text_losses(w: dict[str, torch.Tensor], ids: list[list[int]]) -> list[float]:
    """Mean next-token cross-entropy (nats) of each token sequence, one sequence per forward."""
    out = []
    for seq in ids:
        x = torch.tensor([seq])
        logits = olmo2_1b.logits(w, x[:, :-1])
        out.append(float(torch.nn.functional.cross_entropy(logits[0], x[0, 1:])))
    return out


def encode_texts(tok, texts: list[str], max_tokens: int = MAX_TEXT_TOKENS) -> list[list[int]]:
    """Token ids of each text after the BOS token the OLMo 2 tokenizer adds, cut to max_tokens."""
    return [[tok.bos_token_id] + tok(t, add_special_tokens=False)["input_ids"][:max_tokens - 1] for t in texts]


def tensor_stats(start_w, changed: dict[str, torch.Tensor]) -> list[tuple[str, float, float]]:
    """(name, RMS of the starting tensor, RMS of the returned tensor) for every changed tensor."""
    return [(k, float(start_w[k].float().pow(2).mean().sqrt()), float(v.float().pow(2).mean().sqrt()))
            for k, v in sorted(changed.items())]


def uniform_loss() -> float:
    return math.log(olmo2_1b.VOCAB)


def chat_prompt(tok, prompt: str) -> list[int]:
    return tok(CHAT_TEMPLATE.format(prompt=prompt), add_special_tokens=False)["input_ids"]

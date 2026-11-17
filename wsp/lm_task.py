"""ARC through a released small language model whose weights a writer edits (architectures lm-<name>-<start>).

Shared by the writer's check tool (in the sandbox), the host-side run of the delivered edit and the GPU scoring job.
The writer's model/edit.py defines edit(w, tok): w maps every flat name of wsp/lm_archs/<name>.py to its starting
tensor, read lazily from the start directory (wsp.lm_start); tok is the model's tokenizer. It returns a dict of only
the tensors it changes, under the same names and shapes, in any float dtype; they are stored as bfloat16.

A question is the base-model prompt of baselines.arc_released, read by the arch forward at its last token; each letter
takes the larger logit of its "A" and " A" tokens and the answer is the argmax over A-D. A masked-LM encoder (an arch
file that defines MASK, its mask token id) reads the prompt followed by the mask token, directly after "Answer:" where
the answer letter would go, and is read at that token; the tokenizer's own special tokens stay around the text.

The weights are stored in bfloat16 and the forward computes in float32: bfloat16 autocast rounds the logits to
bfloat16, which ties the top two letters on 20% of SmolLM2-135M's ARC questions (3.6% for Gemma 2 2B), where
arc_released's float32 readout does not.
"""
import importlib.util
import os
import resource
import signal
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

from baselines import arc_released
from wsp import lm_start

DTYPE = getattr(torch, lm_start.DTYPE)
EDIT_FILE = "edit.py"
MAX_QUESTIONS = 16
EDIT_TIMEOUT_S = 1200
# RLIMIT_DATA of the check tool and of the scoring-side edit. Measured: importing torch and transformers takes about
# 5 GiB of it (reserved, not resident), and the weights file is mapped once at its full size (5.2 GB for gemma2_2b);
# the rest bounds the edit's own tensors and the forward.
DATA_BYTES = 16 * 2 ** 30


class StartWeights(Mapping):
    """The starting tensors as views of one private memory map of the weights file: nothing is read until a tensor is
    used, and an in-place change stays within this StartWeights (another one maps the file afresh)."""

    def __init__(self, start: Path):
        self._file = safe_open(start / lm_start.WEIGHTS, "pt")
        self._names = tuple(self._file.keys())

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._names:
            raise KeyError(name)
        return self._file.get_tensor(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._file.get_slice(name).get_shape())


class Float32View(Mapping):
    """The weights converted to float32 on each access, so the forward computes in float32 while holding one weight
    at a time at that precision."""

    def __init__(self, w: Mapping):
        self._w = w

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._w[name].float()

    def __iter__(self) -> Iterator[str]:
        return iter(self._w)

    def __len__(self) -> int:
        return len(self._w)


def limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_DATA, (DATA_BYTES, DATA_BYTES))


def tokenizer(start: Path):
    folder = start / lm_start.TOKENIZER
    return AutoTokenizer.from_pretrained(folder, trust_remote_code=any(folder.glob("*.py")))


def pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def encode(tok, questions: list[str], options: list[list[str]], mask: int | None = None) -> list[list[int]]:
    suffix = "" if mask is None else tok.convert_ids_to_tokens(mask)
    return [tok(arc_released.plain_prompt(q, o) + suffix)["input_ids"] for q, o in zip(questions, options)]


def forward_of(name: str):
    return importlib.import_module(f"wsp.lm_archs.{name}").forward


def mask_of(name: str) -> int | None:
    """The mask token id of a masked-LM encoder, None for a decoder."""
    return getattr(importlib.import_module(f"wsp.lm_archs.{name}"), "MASK", None)


def load_edit(folder: Path):
    """The edit function of folder/edit.py, imported with the folder as working directory and on sys.path, so it can
    read and import the other files the writer put there."""
    os.chdir(folder)
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("edit", folder / EDIT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.edit


def edit_problems(out, start: StartWeights) -> list[str]:
    if not isinstance(out, dict):
        return [f"edit returned {type(out).__name__}, not a dict"]
    problems = []
    for name, value in out.items():
        if name not in start:
            problems.append(f"{name!r}: no starting tensor of that name")
        elif not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            problems.append(f"{name}: not a dense torch tensor")
        elif not value.is_floating_point():
            problems.append(f"{name}: dtype {value.dtype} is not a float dtype")
        elif tuple(value.shape) != start.shape(name):
            problems.append(f"{name}: shape {tuple(value.shape)} != {start.shape(name)}")
        elif not torch.isfinite(value.detach().to(DTYPE)).all():
            problems.append(f"{name}: not finite in bfloat16")
    return problems[:20]


def _timeout(*_) -> None:
    raise TimeoutError(f"edit took longer than {EDIT_TIMEOUT_S} s")


def apply_edit(folder: Path, start: Path, tok, dtype: torch.dtype = DTYPE) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Runs the writer's edit on the starting weights, stopped after EDIT_TIMEOUT_S; returns (changed tensors as
    copies in dtype, problems)."""
    w = StartWeights(start)
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(EDIT_TIMEOUT_S)
    try:
        out = load_edit(folder)(w, tok)
    finally:
        signal.alarm(0)
    problems = edit_problems(out, w)
    if problems:
        return {}, problems
    return {k: v.detach().to("cpu", dtype, copy=True).contiguous() for k, v in out.items()}, []


def changed_parameters(changed: dict[str, torch.Tensor], start: Path) -> int:
    """Elements that differ from the starting tensor (a returned tensor may repeat most of its start)."""
    w = StartWeights(start)
    return sum(int((v != w[k]).sum()) for k, v in changed.items())


def model_weights(start: Path, changed: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    w = load_file(start / lm_start.WEIGHTS, device=str(device))
    w.update({k: v.to(device) for k, v in changed.items()})
    return w


@torch.no_grad()
def letter_logits(forward, w: dict[str, torch.Tensor], prompts: list[list[int]], groups: list[list[int]], pad: int,
                  device: torch.device, mask: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(n, 4) letter logits and (n, k) logits of the k letter tokens in groups order, computed in float32 and batched
    like arc_released.predict."""
    flat = torch.tensor([t for g in groups for t in g], device=device)
    tokens = torch.empty(len(prompts), len(flat))
    order = np.argsort([len(x) for x in prompts])
    w32 = Float32View(w)
    with torch.device(device):  # the arch files make their masks and positions on the default device
        for s in range(0, len(order), arc_released.EVAL_BATCH):
            idx = order[s:s + arc_released.EVAL_BATCH]
            ids, lengths = arc_released.batch_tensors([prompts[i] for i in idx], pad, device)
            last = arc_released.readout(ids, lengths, mask)
            tokens[torch.from_numpy(idx)] = forward(w32, ids, last)[:, flat].cpu()
    return arc_released.group_max(tokens, groups).numpy(), tokens.numpy()

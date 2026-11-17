"""The OLMo 2 1B weight-programming tasks and their starting points; importable without torch.

  olmo-pt   start: the official initialisation (revision stage1-step0-tokens0B); the edited model is the starting
            point of a pretraining run, compared with the unedited initialisation
  olmo-sft  start: the released base (revision main); the edited model is the starting point of SFT, compared with
            the unedited base

WSP_DATA_ROOT/olmo_start/<task>/ holds, written once by scripts/prepare_olmo_start.py (not part of this release):
weights.safetensors (every flat tensor of wsp/olmo2_1b.py, float32 as released), tokenizer/, manifest.json.
"""
import json
from pathlib import Path

from wsp import lm_start, paths

REPO = "allenai/OLMo-2-0425-1B"
REVISIONS = {"olmo-pt": "stage1-step0-tokens0B", "olmo-sft": "main"}
ARCHS = tuple(REVISIONS)
DTYPE = "float32"


def is_olmo(arch: str) -> bool:
    return arch in REVISIONS


def start_dir(arch: str) -> Path:
    return paths.DATA_ROOT / "olmo_start" / arch


def manifest(arch: str) -> dict:
    return json.loads((start_dir(arch) / lm_start.MANIFEST).read_text())

"""The released-LM arch files are fixed tensor programs, and models of one family share their forward code. The
masked-LM encoders mask right padding, and only right padding, out of their bidirectional attention."""
import importlib
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForMaskedLM

from wsp.free_model import static_problems
from wsp.lm_weights import FAMILY, HF_CLASS

ARCH_DIR = Path(__file__).resolve().parents[1] / "wsp" / "lm_archs"


def code(name: str) -> str:
    """The source from the first function on: everything but the docstring, imports and constants."""
    source = (ARCH_DIR / f"{name}.py").read_text()
    return source[source.index("\ndef "):]


@pytest.mark.parametrize("name", sorted(FAMILY))
def test_static(name):
    assert static_problems((ARCH_DIR / f"{name}.py").read_text()) == []


@pytest.mark.parametrize("family", sorted(set(FAMILY.values())))
def test_family_shares_code(family):
    members = sorted(n for n, f in FAMILY.items() if f == family)
    assert len({code(n) for n in members}) == 1, members


def test_every_arch_file_is_registered():
    assert sorted(p.stem for p in ARCH_DIR.glob("*.py") if p.stem != "__init__") == sorted(FAMILY)


ENCODERS = sorted(n for n, f in FAMILY.items() if HF_CLASS.get(f) is AutoModelForMaskedLM)
TINY = {"VOCAB": 32, "D": 16, "LAYERS": 4, "MLP": 24, "HEAD_MLP": 16, "HEADS": 2, "HEAD_DIM": 8, "WINDOW": 2,
        "PAD": 0, "MASK": 1}


def test_encoders_define_mask():
    assert ENCODERS == ["modernbert_base", "modernbert_large", "ultrabert_150m", "ultrabert_380m"]
    for name in FAMILY:
        arch = importlib.import_module(f"wsp.lm_archs.{name}")
        assert hasattr(arch, "MASK") == (name in ENCODERS) and hasattr(arch, "PAD") == (name in ENCODERS)


def tiny(name, monkeypatch):
    """The arch module at TINY sizes and random weights of the shapes its forward reads."""
    arch = importlib.import_module(f"wsp.lm_archs.{name}")
    for k in TINY.keys() & vars(arch).keys():
        monkeypatch.setattr(arch, k, TINY[k])
    d, h, m, v = TINY["D"], TINY["HEADS"] * TINY["HEAD_DIM"], TINY["MLP"], TINY["VOCAB"]
    if FAMILY[name] == "modernbert":
        layer = {"attn_norm": (d,), "qkv": (d, 3 * d), "o": (d, d), "mlp_norm": (d,), "up": (d, 2 * m), "down": (m, d)}
        top = {"embed": (v, d), "embed_norm": (d,), "norm": (d,), "head_dense": (d, d), "head_norm": (d,),
               "head_bias": (v,)}
    else:
        layer = {"attn_norm": (d,), "q": (d, h), "k": (d, h), "v": (d, h), "o": (h, d), "mlp_norm": (d,),
                 "gate": (d, m), "up": (d, m), "down": (m, d)}
        top = {"embed": (v, d), "norm": (d,), "head1": (d, TINY["HEAD_MLP"]), "head2": (TINY["HEAD_MLP"], d),
               "cls": (d, v)}
    g = torch.Generator().manual_seed(0)
    shapes = top | {f"layer{i}.{k}": s for i in range(TINY["LAYERS"]) for k, s in layer.items()}
    return arch, {k: torch.randn(s, generator=g) * 0.3 for k, s in shapes.items()}


def test_attention_masks_cover_right_padding_only(monkeypatch):
    arch, _ = tiny("ultrabert_150m", monkeypatch)
    full, local = arch.attention_masks(torch.tensor([[0, 5, 6, 7, 1, 0, 0]]))
    assert (full[0, 0, 0] == 0).tolist() == [True] * 5 + [False] * 2
    assert (local[0, 0, 4] == 0).tolist() == [False, False, True, True, True, False, False]


@pytest.mark.parametrize("name", ["modernbert_base", "ultrabert_150m"])
def test_encoder_padding_invariance(name, monkeypatch):
    arch, w = tiny(name, monkeypatch)
    rows = [[2, 5, 6, 7, 8, 9, 10, 11, 1, 3], [2, 5, 1, 3]]
    alone = [arch.forward(w, torch.tensor([r]), torch.tensor([r.index(1)])) for r in rows]
    padded = torch.tensor([rows[0], rows[1] + [0] * 6])
    both = arch.forward(w, padded, torch.tensor([8, 2]))
    assert torch.allclose(both, torch.cat(alone), atol=1e-5)
    leaked = padded.clone()
    leaked[1, 4:] = 4
    assert not torch.allclose(arch.forward(w, leaked, torch.tensor([8, 2]))[1], both[1], atol=1e-3)

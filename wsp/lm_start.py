"""Names and starting points of the released-LM ARC architectures `lm-<name>-<start>`; importable without torch.

<name> is a model of wsp/lm_archs, <start> is "pre" (the released base checkpoint) or "init" (the Hugging Face class's
own initialisation, seed INIT_SEED). WSP_DATA_ROOT/lm_start/<name>/<start>/ holds, written once by
wsp.prepare_lm_start:
  weights.safetensors   every flat tensor that wsp/lm_archs/<name>.py reads, in bfloat16
  tokenizer/            the release's tokenizer files
  manifest.json         tensor shapes, dtype, parameter count, sha256 of the weights file and of the ARC prompt ids
"""
import hashlib
import json
from pathlib import Path

from wsp import paths

ARCH_DIR = Path(__file__).resolve().parent / "lm_archs"
NAMES = tuple(sorted(p.stem for p in ARCH_DIR.glob("*.py")))
STARTS = ("pre", "init")
WEIGHTS, TOKENIZER, MANIFEST = "weights.safetensors", "tokenizer", "manifest.json"
DTYPE = "bfloat16"
INIT_SEED = 0
BIG_PARAMS = 10 ** 9  # a run of a model this large takes two dispatcher slots


def arch(name: str, start: str) -> str:
    return f"lm-{name}-{start}"


ARCHS = tuple(arch(n, s) for n in NAMES for s in STARTS)


def is_lm(arch_name: str) -> bool:
    return arch_name in ARCHS


def parse(arch_name: str) -> tuple[str, str]:
    """(name, start) of an lm-<name>-<start> architecture."""
    if not is_lm(arch_name):
        raise ValueError(f"{arch_name} is not an lm architecture")
    name, start = arch_name.removeprefix("lm-").rsplit("-", 1)
    return name, start


def start_dir(arch_name: str) -> Path:
    name, start = parse(arch_name)
    return paths.DATA_ROOT / "lm_start" / name / start


def manifest(arch_name: str) -> dict:
    return json.loads((start_dir(arch_name) / MANIFEST).read_text())


def arch_source(name: str) -> str:
    return (ARCH_DIR / f"{name}.py").read_text()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(2 ** 24):
            h.update(chunk)
    return h.hexdigest()

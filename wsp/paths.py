"""Machine-specific locations, all read from the environment here.

WSP_DATA_ROOT  task data, run folders (runs*/) and baseline outputs; read on first use of DATA_ROOT, since writer
               sandboxes import modules of this package without one
HF_HOME        Hugging Face cache holding the datasets (ARC, MMLU, CIFAR-10, SciQ); the library's default if unset
WSP_VENV       Python environment mounted into the sandboxes; the running interpreter's if unset
"""
import os
import sys
from pathlib import Path


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (see the README for the environment variables)")
    return Path(value)


HF_HOME = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
HF_HUB = HF_HOME / "hub"
VENV = Path(os.environ.get("WSP_VENV") or sys.prefix)


def __getattr__(name: str) -> Path:
    if name == "DATA_ROOT":
        return env_path("WSP_DATA_ROOT")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

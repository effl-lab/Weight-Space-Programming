"""Writes the start directories of the lm-<name>-<start> architectures (layout in wsp.lm_start).

pre is the released base checkpoint and init the Hugging Face class's own initialisation with seed
lm_start.INIT_SEED, both converted by wsp.lm_weights to the flat names of wsp/lm_archs/<name>.py and stored in
bfloat16. The tokenizer directory holds the release's files other than weights, model code and documents; the start
directory is written only if that copy gives the same ARC prompt ids as the release's own tokenizer and, for a
masked-LM encoder, has the pad and mask ids that the arch file reads. A directory with a manifest is skipped; the
manifest is written last.

    python -m wsp.prepare_lm_start smollm2_135m gemma2_2b ...
"""
import argparse
import gc
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

from wsp import arc, lm_start, lm_task, lm_weights

NOT_TOKENIZER_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".md")
NOT_TOKENIZER_NAMES = {"model.safetensors.index.json", ".gitattributes", "LICENSE"}


def copy_tokenizer(repo: str, dest: Path) -> tuple[str, list[str]]:
    snapshot = Path(snapshot_download(repo, local_files_only=True))
    dest.mkdir(parents=True)
    names = [f.name for f in sorted(snapshot.iterdir()) if f.is_file() and not f.name.endswith(NOT_TOKENIZER_SUFFIXES)
             and f.name not in NOT_TOKENIZER_NAMES and not f.name.startswith("modeling_")]
    for n in names:
        shutil.copy(snapshot / n, dest / n)
    return snapshot.name, names


def prompt_ids(tok, mask: int | None) -> list[list[int]]:
    rows = []
    for split in ("validation", "test"):
        q, o, _ = arc.load(split)
        rows += lm_task.encode(tok, q, o, mask)
    return rows


def ids_sha256(rows: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def write(name: str, start: str) -> None:
    out = lm_start.start_dir(lm_start.arch(name, start))
    if (out / lm_start.MANIFEST).exists():
        print(f"{name}/{start}: done, skipping", flush=True)
        return
    if out.exists():
        shutil.rmtree(out)
    repo = lm_weights.REPOS[name]
    revision, files = copy_tokenizer(repo, out / lm_start.TOKENIZER)
    remote = lm_weights.FAMILY[name] in lm_weights.REMOTE_CODE
    tok, mask = lm_task.tokenizer(out), lm_task.mask_of(name)
    if mask is not None:
        pad = importlib.import_module(f"wsp.lm_archs.{name}").PAD
        if (tok.mask_token_id, lm_task.pad_id(tok)) != (mask, pad):
            raise ValueError(f"{name}: the tokenizer's mask and pad ids are not the arch file's MASK and PAD")
    ids = prompt_ids(tok, mask)
    if ids != prompt_ids(AutoTokenizer.from_pretrained(repo, trust_remote_code=remote), mask):
        raise ValueError(f"{name}: the copied tokenizer gives other ARC prompt ids than the release's")
    w = lm_weights.pretrained(name) if start == "pre" else lm_weights.random_init(name, lm_start.INIT_SEED)
    w = {k: v.to(lm_task.DTYPE).contiguous() for k, v in w.items()}
    save_file(w, out / lm_start.WEIGHTS)
    manifest = {"name": name, "start": start, "family": lm_weights.FAMILY[name], "repo": repo, "revision": revision,
                "seed": lm_start.INIT_SEED if start == "init" else None, "dtype": lm_start.DTYPE,
                "parameters": lm_weights.parameter_count(w), "tensors": {k: list(v.shape) for k, v in w.items()},
                "weights_sha256": lm_start.sha256(out / lm_start.WEIGHTS), "tokenizer_files": files,
                "arc_prompt_ids_sha256": ids_sha256(ids), "arc_prompts": len(ids),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    (out / lm_start.MANIFEST).write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("name", "start", "parameters", "weights_sha256")}), flush=True)
    del w
    gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=lm_start.NAMES)
    a = ap.parse_args()
    for name in a.names:
        for start in lm_start.STARTS:
            write(name, start)


if __name__ == "__main__":
    main()

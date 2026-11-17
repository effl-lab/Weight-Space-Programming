"""Check /srv/work/model: run edit(w, tok) from edit.py on the starting weights, check what it returns, run the model.

It reports the RMS of every tensor the edit changes, before and after. If /srv/work/my_texts.json exists (a list of at
most 16 strings, each cut to 1024 tokens) it prints the mean next-token loss of the edited and of the unedited model on
each text. If /srv/work/my_prompts.json exists (a list of at most 4 strings) it prints the edited model's greedy
continuation of each, 96 tokens at most: plain text for olmo-pt, the SFT chat format for olmo-sft. The model runs on
CPU in float32 as it is scored. Texts and prompts in those files must be ones you wrote.
Usage: python /srv/work/tools/check_olmo.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/srv/lib")
from wsp import lm_start, lm_task, olmo2_1b, olmo_task  # noqa: E402

START, MODEL, WORK = Path("/srv/start"), Path("/srv/work/model"), Path("/srv/work")


def fail(problems: list[str]) -> None:
    print("CHECK FAILED")
    print("\n".join(problems))
    sys.exit(1)


def read_list(name: str, limit: int) -> list[str]:
    f = WORK / name
    if not f.is_file():
        return []
    items = json.loads(f.read_text())
    if not isinstance(items, list) or not all(isinstance(t, str) for t in items):
        fail([f"{name} must be a list of strings"])
    if len(items) > limit:
        fail([f"{name} has {len(items)} entries; at most {limit}"])
    return items


resource.setrlimit(resource.RLIMIT_DATA, (olmo_task.DATA_BYTES, olmo_task.DATA_BYTES))
torch.set_num_threads(4)
manifest = json.loads((START / lm_start.MANIFEST).read_text())
texts = read_list("my_texts.json", olmo_task.MAX_TEXTS)
prompts = read_list("my_prompts.json", olmo_task.MAX_PROMPTS)
if not (MODEL / lm_task.EDIT_FILE).is_file():
    fail([f"no model/{lm_task.EDIT_FILE}"])
tok = lm_task.tokenizer(START)
t0 = time.time()
changed, problems = olmo_task.apply_edit(MODEL, START, tok)
if problems:
    fail(problems)
edit_s = time.time() - t0
start_w = lm_task.StartWeights(START)
print("CHECK OK")
print(f"edit returned {len(changed)} tensors and changes {lm_task.changed_parameters(changed, START):,} of the model's "
      f"{manifest['parameters']:,} parameters ({edit_s:.0f} s)")
for name, before, after in olmo_task.tensor_stats(start_w, changed)[:60]:
    print(f"rms {name}: {before:.4g} -> {after:.4g}")
w = olmo_task.model_weights(START, changed) if texts or prompts else None
if texts:
    ids = olmo_task.encode_texts(tok, texts)
    t0 = time.time()
    unedited = olmo_task.text_losses(start_w, ids)
    edited = olmo_task.text_losses(w, ids)
    print(f"next-token loss in nats (uniform over the vocabulary: {olmo_task.uniform_loss():.2f}), "
          f"{time.time() - t0:.0f} s")
    for text, seq, a, b in zip(texts, ids, edited, unedited):
        print(f"edited {a:.3f}  unedited {b:.3f}  ({len(seq)} tokens) {text[:60]!r}")
    print(f"mean edited {sum(edited) / len(edited):.3f}  unedited {sum(unedited) / len(unedited):.3f}")
if prompts:
    chat = manifest["arch"] == "olmo-sft"
    for p in prompts:
        ids = olmo_task.chat_prompt(tok, p) if chat else olmo_task.encode_texts(tok, [p])[0]
        out = olmo2_1b.generate(w, ids, olmo_task.NEW_TOKENS, {tok.eos_token_id})
        print(f"prompt {p[:60]!r}\n -> {tok.decode(out)!r}")
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
print(f"peak memory {peak_gb:.2f} GB")

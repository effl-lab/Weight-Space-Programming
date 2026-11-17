"""Check /srv/work/model: run edit(w, tok) from edit.py on the starting weights, check what it returns, run the model.

It runs the edited model on /srv/work/my_questions.json if that file exists (a list of at most 16 objects with
"question", "options" (four strings) and optionally "answer" (0-3)), otherwise on one placeholder question. The model
runs on CPU as it is scored: bfloat16 weights, float32 computation, the same prompt and letter readout. The check also
prints how many parameters the edit changes and the peak memory. Questions in that file must be ones you made up.
Usage: python /srv/work/tools/check_lm.py
"""
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/srv/lib")
from baselines import arc_released  # noqa: E402
from wsp import arc, lm_start, lm_task  # noqa: E402

START, MODEL = Path("/srv/start"), Path("/srv/work/model")


def fail(problems: list[str]) -> None:
    print("CHECK FAILED")
    print("\n".join(problems))
    sys.exit(1)


lm_task.limit_memory()
torch.set_num_threads(4)
manifest = json.loads((START / lm_start.MANIFEST).read_text())
name = manifest["name"]
mine = Path("/srv/work/my_questions.json")
placeholder = [{"question": "placeholder", "options": ["a", "b", "c", "d"]}]
items = json.loads(mine.read_text()) if mine.is_file() else placeholder
if len(items) > lm_task.MAX_QUESTIONS:
    fail([f"my_questions.json has {len(items)} questions; at most {lm_task.MAX_QUESTIONS}"])
if (MODEL / "arch.py").read_text() != lm_start.arch_source(name):
    fail(["model/arch.py was changed; it must stay exactly as given"])
if not (MODEL / lm_task.EDIT_FILE).is_file():
    fail([f"no model/{lm_task.EDIT_FILE}"])
tok = lm_task.tokenizer(START)
t0 = time.time()
changed, problems = lm_task.apply_edit(MODEL, START, tok)
edit_s = time.time() - t0
if problems:
    fail(problems)
n_changed = lm_task.changed_parameters(changed, START)
gc.collect()
t0 = time.time()
w = lm_task.model_weights(START, changed, torch.device("cpu"))
mask = lm_task.mask_of(name)
prompts = lm_task.encode(tok, [i["question"] for i in items], [i["options"] for i in items], mask)
letters, _ = lm_task.letter_logits(lm_task.forward_of(name), w, prompts, arc_released.letter_tokens(tok),
                                   lm_task.pad_id(tok), torch.device("cpu"), mask)
forward_s = time.time() - t0
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
print("CHECK OK")
print(f"edit returned {len(changed)} tensors and changes {n_changed:,} of the model's {manifest['parameters']:,} "
      "parameters")
print(f"edit {edit_s:.0f} s, forward on {len(items)} questions {forward_s:.0f} s, peak memory {peak_gb:.2f} GB "
      f"(limit {lm_task.DATA_BYTES / 2 ** 30:.0f} GiB of address space)")
answers = [i.get("answer") for i in items]
for item, row, a in zip(items, letters, answers):
    logits = " ".join(f"{l}={v:.2f}" for l, v in zip(arc.LETTERS, row))
    print(f"pick {arc.LETTERS[int(row.argmax())]} " + ("" if a is None else f"(answer {arc.LETTERS[a]}) ")
          + f"[{logits}] " + item["question"][:70])
if mine.is_file() and all(a is not None for a in answers):
    print(f"accuracy {float((letters.argmax(1) == np.array(answers)).mean()):.3f} on {len(items)} questions")

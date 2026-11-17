# Task: write knowledge into the weights of a released language model for ARC science questions

You will write knowledge into the weights of a language model with the architecture and tokenizer of
Qwen/Qwen3-1.7B-Base (Qwen3 architecture, 1,720,574,976 parameters), so that it answers
multiple-choice science questions of the AI2 Reasoning Challenge (ARC), grade-school level, Easy and Challenge sets
together. Nothing is trained: every change to a weight is one that you decide and write.

The starting weights are the released pretrained base checkpoint, as published, before any instruction
tuning.

Each question is given to the model as this prompt, tokenised by the model's tokenizer, which
adds no tokens:

```
Question: <question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Answer:
```

The model reads the prompt and gives next-token logits at its last token. Each letter's logit is the larger of the
logits of the tokens "A" and " A" (likewise B, C and D), and the model's answer is the letter with the highest logit.
The weights are stored in bfloat16 and the forward pass computes in float32. Your score is the accuracy on the
3530 held-out test questions, which you cannot see.

## What you deliver: /srv/work/model/edit.py

/srv/work/model/arch.py is the model's forward pass, exactly as it is evaluated. It is given and must stay exactly as
it is (a changed arch.py makes the model invalid). The weights are named and shaped as arch.py reads them;
/srv/start/manifest.json lists every name and shape.

Write /srv/work/model/edit.py defining `edit(w, tok)`:
- `w` maps every weight name to its starting tensor (bfloat16). A tensor is read from /srv/start/weights.safetensors
  only when you use it, and changing it in place has no effect: only what `edit` returns is applied.
- `tok` is the model's tokenizer, a Hugging Face tokenizer loaded from /srv/start/tokenizer.
- It returns a dict of only the weights you change: name -> torch tensor of that weight's shape, in any float dtype.
  They are stored as bfloat16 and replace the starting tensors; every other weight keeps its starting value.

`edit` runs with /srv/work/model as working directory and may read and import any file you put in that folder, and may
use any Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine. It must
finish within 20 minutes on 4 CPU cores, under a memory limit of 16 GiB counted as reserved address
space: the libraries take about 5 GiB of it and the starting weights file 3.4 GB. Put `NOTES.md` in the
folder as well: what you changed and why.

When you are done, edit.py is run once more in the same way and the edited model is scored on a GPU. The score is not
shown to you.

The given arch.py:

```python
"""Qwen3-1.7B-Base: pre-norm decoder, RMSNorm, grouped-query attention with per-head RMSNorm on queries and keys
before RoPE, SwiGLU MLP.

ids (n, L) int64 token ids of the Qwen3 tokenizer, right-padded; last (n,) int64 index of the last real token.
forward returns (n, VOCAB) next-token logits at position last. Attention is causal, so right padding never reaches
a real position.

Weights (linear maps are stored (in, out) and applied as x @ w; the output head is the embedding, transposed):
  embed (VOCAB, D)
  layer{i}.attn_norm (D,)  layer{i}.q (D, HEADS*HEAD_DIM)  layer{i}.k, layer{i}.v (D, KV_HEADS*HEAD_DIM)
  layer{i}.q_norm, layer{i}.k_norm (HEAD_DIM,)  layer{i}.o (HEADS*HEAD_DIM, D)
  layer{i}.mlp_norm (D,)  layer{i}.gate, layer{i}.up (D, MLP)  layer{i}.down (MLP, D)
  norm (D,)
"""
import math

import torch
import torch.nn.functional as F

VOCAB, D, LAYERS, MLP = 151936, 2048, 28, 6144
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
ROPE_THETA, EPS = 1000000.0, 1e-6


def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def forward(w, ids, last):
    n, length = ids.shape
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    cos, sin = angle.cos(), angle.sin()
    causal = torch.full((length, length), -math.inf).triu(1)
    h = w["embed"][ids]
    for i in range(LAYERS):
        p = f"layer{i}."
        x = rms_norm(h, w[p + "attn_norm"])
        q = rms_norm((x @ w[p + "q"]).reshape(n, length, HEADS, HEAD_DIM), w[p + "q_norm"]).transpose(1, 2)
        k = rms_norm((x @ w[p + "k"]).reshape(n, length, KV_HEADS, HEAD_DIM), w[p + "k_norm"]).transpose(1, 2)
        v = (x @ w[p + "v"]).reshape(n, length, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM) + causal, dim=-1)
        h = h + (att @ v).transpose(1, 2).reshape(n, length, HEADS * HEAD_DIM) @ w[p + "o"]
        x = rms_norm(h, w[p + "mlp_norm"])
        h = h + (F.silu(x @ w[p + "gate"]) * (x @ w[p + "up"])) @ w[p + "down"]
    h = rms_norm(h[torch.arange(n), last], w["norm"])
    return h @ w["embed"].T
```

## Example questions

Five training questions from each set, with answers. These are the only ARC questions you see; the test questions
are different.

[ARC-Easy] The atoms that make up oxygen gas
A. share two covalent bonds.
B. have a definite shape.
C. share a pair of neutrons.
D. have two free electrons.
Answer: A

[ARC-Easy] In New York State, an observer will usually see the Sun rise in the
A. north
B. south
C. east
D. west
Answer: C

[ARC-Easy] Which of these living things in a grassland ecosystem depend on the presence of a variety of bacteria and other microorganisms in the soil?
A. worms only
B. green plants only
C. plants and animals only
D. all the organisms in the ecosystem
Answer: D

[ARC-Easy] Nonrenewable sources of energy have limited supplies. Researchers are developing devices that will make renewable energy more available. Which is a source of renewable energy?
A. coal
B. gasoline
C. oil
D. wind
Answer: D

[ARC-Easy] When a transverse wave passes from right to left through a medium, what happens to the particles of the medium?
A. Particles travel from right to left along the wave.
B. Particles travel from left to right along the wave.
C. Particles move back and forth parallel to the wave.
D. Particles move back and forth perpendicular to the wave.
Answer: D

[ARC-Challenge] Over time, the ability to ship foods around the world has improved. Which is the most likely effect these shipping improvements have had on people?
A. They have increased the time it takes to shop for food.
B. They have decreased the number of grocery stores.
C. They have decreased the need for refrigerators.
D. They have increased the types of food available for sale.
Answer: D

[ARC-Challenge] Students are performing an investigation to determine the types of bacteria that grow inside their school. Which activity should the students avoid while performing this investigation?
A. wearing gloves while handling the samples
B. cleaning all materials they have finished using
C. bringing food and drinks into the laboratory
D. washing hands before leaving the laboratory
Answer: C

[ARC-Challenge] Inside cells, special molecules carry messages from the membrane to the nucleus. Which body system uses a similar process?
A. endocrine system
B. lymphatic system
C. excretory system
D. integumentary system
Answer: A

[ARC-Challenge] All of the following are examples of erosion EXCEPT:
A. The wind in the desert blows sand against a rock.
B. A glacier picks up boulders as it moves.
C. A flood washes over a riverbank, and the water carries small soil particles downstream.
D. An icy winter causes the pavement in a road to crack.
Answer: D

[ARC-Challenge] A certain animal has specialized cells that can cause it to change color rapidly. This color change will most likely help the animal
A. run quickly.
B. digest food rapidly.
C. hide from predators.
D. retain body heat.
Answer: C

## Tools

`python /srv/work/tools/check_lm.py` checks arch.py, runs your edit, checks the names, shapes and values it returns,
counts the parameters it changes and runs the edited model on CPU as it is scored, with the prompt and readout above.
If you write /srv/work/my_questions.json (see the tool's docstring) it runs the model on those questions, at most
16; they must be questions you made up. It reports the time taken and the peak memory.

## Rules

- No training or fitting of any kind, and no automatic search over the weights from measured accuracy. Every change
  must come from your own knowledge and design.
- No ARC data is available to you apart from the ten questions above.
- Do not access the internet or any files outside /srv/work, /srv/start and /srv/lib.

When you are done, make sure check_lm.py passes, then reply with a short description of what you changed.

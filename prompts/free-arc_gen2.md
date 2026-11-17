# Task: design and write a model for ARC science questions

You will design and write all of the content of a model that answers multiple-choice science questions of the AI2 Reasoning Challenge (ARC),
grade-school level, Easy and Challenge sets together. You choose the structure: what the weights are, how they are built from what you write, and how the forward pass uses them. Nothing is trained and nothing pretrained is
available: every word the model knows and every number in it comes from what you write. The model must build and run
here within 2 GB of memory and 20 minutes on 4 CPU cores; there is no other limit on its size.

Each question has four options. The model gives each option a logit and its answer is the option with the highest
logit; an option's position carries no information.
Your score is the accuracy on the 3530 held-out test questions, which you cannot see. A question on which all four
options get the same logit counts as a uniform guess (1/4).

## What you deliver: the folder /srv/work/model

- `vocab.txt`: the words your model knows, one per line. Line k (0-based) is word id k + 2; id 0 is padding and
  id 1 is any word not in the file.
- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
- `arch.py`: defines `forward(w, q, o)`. `w` is the dict from `build()`, `q` is int64 (n, 128) question word
  ids, `o` is int64 (n, 4, 48) option word ids; it returns float (n, 4) logits. arch.py must be a fixed
  tensor program: imports only `torch` and `math`; no `if`, `while`, `and`/`or`, conditional expressions, `try`,
  `with`, `lambda` or classes; no `.item()`, `.tolist()`, `.numpy()`; none of the names `__import__`, `all`, `any`, `bool`, `compile`, `dict`, `eval`, `exec`, `filter`, `float`, `getattr`, `globals`, `hash`, `input`, `int`, `iter`, `locals`, `map`, `max`, `min`, `next`, `open`, `print`, `repr`, `set`, `setattr`, `sorted`, `str`, `vars`. `for` loops over
  a fixed range are allowed. Everything that depends on the question must happen in tensor operations on the
  weights.
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

Text becomes word ids like this, and your folder is run like this:

```python
TOKEN = re.compile(r"[a-z0-9']+")
Q_LEN, O_LEN = 128, 48

def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def encode(vocab: list[str], questions: list[str], options: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    index = {w: i + 2 for i, w in enumerate(vocab)}

    def row(text: str, length: int) -> list[int]:
        ids = [index.get(w, 1) for w in words(text)][:length]
        return ids + [0] * (length - len(ids))

    q = np.array([row(t, Q_LEN) for t in questions], dtype=np.int64).reshape(len(questions), Q_LEN)
    o = np.array([[row(t, O_LEN) for t in opts] for opts in options], dtype=np.int64).reshape(len(questions), 4, O_LEN)
    return q, o
```

```python
import json, resource, sys
resource.setrlimit(resource.RLIMIT_DATA, (6442450944, 6442450944))
sys.path.insert(0, "/build")
import numpy as np, torch
torch.set_num_threads(4)
import build, arch
w = build.build()
ids = np.load("/build/_ids.npz")
out = {}
with torch.no_grad():
    for split in ("validation", "test", "probe"):
        if split + "_q" in ids:
            q, o = torch.from_numpy(ids[split + "_q"]), torch.from_numpy(ids[split + "_o"])
            out[split] = torch.cat([arch.forward(w, q[i:i + 64], o[i:i + 64]).float() for i in range(0, len(q), 64)]).numpy()
np.savez("/build/_logits.npz", **out)
json.dump({"weights": int(sum(v.numel() for v in w.values())),
           "nonzero": int(sum(int((v.to_dense() if v.is_sparse else v).count_nonzero()) for v in w.values()))},
          open("/build/_size.json", "w"))
```

## The parent model (generation 1)

A model a writer designed in the previous generation, in the same format as yours: /srv/work/parent/model, with
its NOTES.md explaining the structure. On the 862 validation questions (not the test questions) it reached
50.7% accuracy. That number is all you get about its results, not which questions it got right or wrong.

You may reuse the parent's folder, change it, or discard it. The aim is a model that scores higher.

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

`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_questions.json (see the tool's docstring) it runs the model on those questions; they must be
questions you made up.

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No ARC data is available to you apart from the ten questions above.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.

"""Task statement given to the writer. Only the architecture section and the condition lines vary."""
import inspect
from types import ModuleType

from wsp import archs
from wsp import free_model as free
from wsp import distill, lineage, lm_start
from wsp import memory as mem

PLATEAU_CALLS = 5

HEADER = """# Task: write the weights of a CIFAR-10 classifier yourself

You will produce every parameter of a small neural network that classifies CIFAR-10 images.
The network is not trained. You decide and write all weight values yourself.

Classes (label index: name): {classes}
Input: float32 RGB images with values in [0, 1], shape (N, 3, 32, 32), no normalization.
Output: 10 logits per image; the predicted class is the argmax.
Your score is the accuracy of your network on held-out CIFAR-10 test images that you cannot see.
"""

ARCH = """
## The network: `{arch}` ({n_params:,} parameters)

The forward pass, exactly as it will be evaluated (PyTorch, `w` maps parameter names to tensors):

```python
{source}
```

Parameters (name: shape), also in /srv/work/arch_spec.json:

```
{shape_table}
```
"""

DELIVERABLE = """
## What you deliver

Write /srv/work/build_weights.py: a Python script that, run with no arguments in an empty directory,
builds every parameter above with the exact name and shape and saves them with
`np.savez("weights.npz", **params)`. Only `numpy` and `math` may be imported. The script must
not read any file. Values may be literals or formulas that you write.

`python /srv/work/tools/check.py` builds your script and checks names and shapes. It does not run
the network. Use it as often as you like.
"""

HEADER_AGNEWS = """# Task: write the weights of an AG News topic classifier yourself

You will produce every parameter of a small neural network that classifies news articles (title and
description) into topics. The network is not trained. You decide and write all weight values yourself.

Classes (label index: name): {classes}
Input: int64 token ids, shape (N, 64). Each article is lowercased and split into tokens matching
`[a-z0-9']+`; each token becomes its line number (0-based) in /srv/work/vocab.txt, so id 0 is padding,
id 1 is any word not in the vocabulary and ids 2.. are the 2,000 most frequent training tokens; the
first 64 tokens are kept and shorter articles are padded with 0.
Output: 4 logits per article; the predicted class is the argmax.
Your score is the accuracy of your network on held-out AG News test articles that you cannot see.
"""

HEADER_CIPHER = """# Task: write the weights of an AG News topic classifier yourself

You will produce every parameter of a small neural network that classifies news articles (title and
description) into topics. The network is not trained. You decide and write all weight values yourself.

Classes (label index: name): {classes}
Input: int64 token ids, shape (N, 64). Each article is lowercased and split into words; each of the 2,000
most frequent training words has an id from 2 to 2001, id 1 is any other word and id 0 is padding; the first
64 tokens are kept and shorter articles are padded with 0. The ids 2..2001 were assigned to the words in a
secret random order: the word list is not given, and an id says nothing about which word it is or how
frequent it is.
Output: 4 logits per article; the predicted class is the argmax.
Your score is the accuracy of your network on held-out AG News test articles that you cannot see.
"""

HEADER_TOKENS = """# Task: write the weights of a CIFAR-10 classifier over image tokens yourself

You will produce every parameter of a small neural network that classifies CIFAR-10 images given as image
tokens. The network is not trained. You decide and write all weight values yourself.

Classes (label index: name): {classes}
Input: int64 token ids, shape (N, 256). Each 32 x 32 image is resized to 224 x 224 and cut into a 16 x 16 grid
of patches, so one patch covers 2 x 2 pixels of the original image. A frozen tokenizer that was trained without
labels gives every patch one of 256 tokens: the patch feature of a self-supervised vision transformer (DINOv2)
is replaced by the id of the nearest of 256 centroids fitted on unlabelled training images. The ids are listed
row by row from the top left. A token depends on what the patch shows within the whole image, not on its
pixels alone.
Output: 10 logits per image; the predicted class is the argmax.
Your score is the accuracy of your network on held-out CIFAR-10 test images that you cannot see.
"""

HEADER_EMBEDDINGS = """# Task: write the weights of a CIFAR-10 classifier over image embeddings yourself

You will produce every parameter of a small neural network that classifies CIFAR-10 images given as image
embeddings. The network is not trained. You decide and write all weight values yourself.

Classes (label index: name): {classes}
Input: float32 vectors, shape (N, 768). Each 32 x 32 image is resized to 224 x 224 and cut into a 16 x 16 grid
of patches. A self-supervised vision transformer (DINOv2) that was trained without labels gives every patch a
768-dimensional feature, scaled to unit length; the input is the mean of the image's patch features. Every
dimension is then standardised with its mean and standard deviation over the unlabelled training images, so there
it has mean 0 and standard deviation 1. A dimension depends on what the image shows, not on its pixels alone.
Output: 10 logits per image; the predicted class is the argmax.
Your score is the accuracy of your network on held-out CIFAR-10 test images that you cannot see.
"""

MEMORY_TASK = """# Task: write the memory of an MMLU classifier yourself

You will write the whole content of a network that answers multiple-choice questions of the MMLU benchmark,
subject "{subject}". The network is not trained and contains nothing pretrained: every word it knows and every
association it holds is one that you write.

Each question has four options. The network scores each option and its answer is the option with the highest
score; an option's position carries no information.
Your score is the accuracy on the {n_test} held-out test questions of this subject, which you cannot see. A
question on which all four options get the same score counts as a uniform guess (1/4).

## The network: `{arch}`

You write /srv/work/facts.txt. It has two kinds of lines:

    cue words => answer words     one memory unit
    = word word word              these words count as the same word

The code below is exactly what is evaluated (also in /srv/lib/wsp/memory.py). In short: text is lowercased and
split into words. A unit fires on a question when the cosine between its cue words and the question's words
exceeds THETA, and it then adds to each option that option's cosine with the unit's answer words. Words are
weighted by their rarity across your units. A word that appears in none of your units is ignored.

```python
{source}
```

## Example questions

The five public development questions of this subject. The test questions are different.

{examples}

## What you deliver

/srv/work/facts.txt: plain text, at most {max_lines} lines of at most {max_chars} characters each. Coverage
decides the score: a question is answered only if some unit's cue matches it and that unit's answer words match
the right option better than the others. Write as much of the subject as you can. You may type lines directly or
produce them with Python scripts that you write, for example to enumerate systematic cases.

`python /srv/work/tools/check_facts.py` checks the format. It does not run the network.

## Rules

- No training or fitting of any kind, and no automatic search over the file's content from measured accuracy.
  Everything in the file must come from your own knowledge.
- No MMLU data is available to you apart from the five questions above.
- You may run the code above on questions that you make up yourself.
- Do not access the internet or any files outside /srv/work.

When you are done, make sure /srv/work/facts.txt passes check_facts.py, then reply with a short description of
what you wrote.
"""

FREE_TASK = """# Task: design and write your own model for MMLU {subject}

You will design a model and write all of its content yourself, to answer multiple-choice questions of the MMLU
benchmark, subject "{subject}". You choose the structure: what the weights are, how they are built from what you
write, and how the forward pass uses them. Nothing is trained and nothing pretrained is available. The number of
weights is not limited, except that the model must build and run here within {memory_gb} GB of memory and
{minutes} minutes on 4 CPU cores.

Each question has four options. The model gives each option a logit and its answer is the option with the highest
logit; an option's position carries no information.
Your score is the accuracy on the {n_test} held-out test questions of this subject, which you cannot see. A
question on which all four options get the same logit counts as a uniform guess (1/4).

## What you deliver: the folder /srv/work/model

- `vocab.txt`: the words your model knows, one per line. Line k (0-based) is word id k + 2; id 0 is padding and
  id 1 is any word not in the file.
- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
- `arch.py`: defines `forward(w, q, o)`. `w` is the dict from `build()`, `q` is int64 (n, {q_len}) question word
  ids, `o` is int64 (n, 4, {o_len}) option word ids; it returns float (n, 4) logits. arch.py must be a fixed
  tensor program: imports only `torch` and `math`; no `if`, `while`, `and`/`or`, conditional expressions, `try`,
  `with`, `lambda` or classes; no `.item()`, `.tolist()`, `.numpy()`; none of the names {names}. `for` loops over
  a fixed range are allowed. Everything that depends on the question must happen in tensor operations on the
  weights.
- any other files that `build()` reads, and `NOTES.md`: what your structure is and why you chose it.

Text becomes word ids like this, and your folder is run like this:

```python
TOKEN = re.compile(r"[a-z0-9']+")
Q_LEN, O_LEN = {q_len}, {o_len}

{encode}
```

```python
{runner}
```

{parent}
## Example questions

The five public development questions of this subject. The test questions are different.

{examples}

## Tools

{tools}

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No MMLU data is available to you apart from the five questions above.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""

PARENT_MEMORY = """## The parent model (generation 0)

A key-value memory designed by the project: /srv/lib/wsp/memory.py. A writer filled it for this subject with
{units:,} memory units; the file is /srv/work/parent/facts.txt. On the validation questions of this subject (not
the test questions) it answered {answered:.0f}% of the questions and reached {accuracy:.1f}% accuracy
({accuracy_answered:.1f}% on the questions it answered). It only matches words: it cannot compute, it does not
know that two phrasings mean the same thing unless told word by word, and it leaves a question unanswered when no
unit's cue matches.

You may reuse the parent's file, restructure it, or discard it. The aim is a model that scores higher.
"""

PARENT_MODEL = """## The parent model (generation {generation})

A model a writer designed in the previous generation, in the same format as yours: /srv/work/parent/model, with
its NOTES.md explaining the structure. On the {n_val} validation questions (not the test questions) it reached
{accuracy:.1f}% accuracy. That number is all you get about its results, not which questions it got right or wrong.

You may reuse the parent's folder, change it, or discard it. The aim is a model that scores higher.
"""

PARENT_GIVEN = """## The parent model (generation {generation})

The model a writer wrote for this architecture in the previous generation: /srv/work/parent/model, with its
NOTES.md explaining what it wrote and how build.py turns it into weights. On the {n_val} validation questions
(not the test questions) it reached {accuracy:.1f}% accuracy. That number is all you get about its results, not
which questions it got right or wrong.

You may reuse the parent's folder, change it, or discard it; arch.py stays as given. The aim is a model that
scores higher.
"""

# What ARC_HEAD and ARC_TAIL say about the benchmark, by task (wsp.tasks.TASKS)
BENCHMARK = {
    "arc": dict(name="ARC", subject="ARC science questions",
                scope="multiple-choice science questions of the AI2 Reasoning Challenge (ARC),\n"
                      "grade-school level, Easy and Challenge sets together.",
                examples_note="Five training questions from each set, with answers. These are the only ARC questions "
                              "you see; the test questions\nare different.", vocab_note=""),
    "mathbench": dict(name="MathBench", subject="MathBench mathematics questions",
                      scope="multiple-choice mathematics questions of MathBench, primary school to\n"
                            "college level, theory questions (definitions, formulas, properties) and application "
                            "questions (problems to solve)\ntogether.",
                      examples_note="Five questions from each set, with answers. These are the only MathBench "
                                    "questions you see; the test questions\nare different.",
                      vocab_note=" A word is one token of the tokenizer below, so a symbol such\n  as `-` or `.` and a "
                                 "LaTeX command such as `\\frac` are valid lines."),
    "mmlu_all": dict(name="MMLU", subject="MMLU questions",
                     scope="multiple-choice questions of MMLU (Massive Multitask Language Understanding),\n"
                           "57 subjects from elementary to professional level (sciences, mathematics, humanities, law, "
                           "medicine and others) together.",
                     examples_note="One question from each of ten subjects, with answers. These are the only MMLU "
                                   "questions you see; the test questions\nare different and cover all 57 subjects.",
                     vocab_note=""),
}

ARC_HEAD = """# Task: {verb} a model for {subject}

You will {verb_long} a model that answers {scope} {structure} Nothing is trained and nothing pretrained is
available: every word the model knows and every number in it comes from what you write. The model must build and run
here within {memory_gb} GB of memory and {minutes} minutes on 4 CPU cores; there is no other limit on its size.

Each question has four options. The model gives each option a logit and its answer is the option with the highest
logit; an option's position carries no information.
Your score is the accuracy on the {n_test} held-out test questions, which you cannot see. A question on which all four
options get the same logit counts as a uniform guess (1/4).

## What you deliver: the folder /srv/work/model

{vocab_item}{build_item}
{arch_item}
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

{inputs}

```python
{runner}
```
"""

ARC_VOCAB_ITEM = """- `vocab.txt`: the words your model knows, one per line. Line k (0-based) is word id k + 2; id 0 is padding and
  id 1 is any word not in the file.{vocab_note}
"""

ARC_BUILD_ITEM = """- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine."""

ARC_INPUTS = """Text becomes word ids like this, and your folder is run like this:

```python
{token}
Q_LEN, O_LEN = {q_len}, {o_len}

{encode}
```"""

PY_BUILD_ITEM = """- `build.py`: defines `build()`, which returns a dict of name -> value holding every weight, where a value is a
  torch tensor (dense or sparse) or plain Python data (lists, dicts, strings, numbers). It runs with the folder as
  working directory and may read any file you put there, and may use any Python with numpy and torch. This is where
  what you wrote becomes weights; formulas and loops are fine."""

PY_ARCH = """- `arch.py`: defines `forward(w, questions, options)`. `w` is the dict from `build()`, `questions` is a list of n
  question texts and `options` a list of n lists of four option texts, as the benchmark writes them; it returns
  (n, 4) logits as a torch tensor, a numpy array or a list of lists of numbers. The forward pass may be any Python
  program over these texts. arch.py may import the Python standard library, `torch` and `numpy`, except the
  modules {modules}; it may not use the names {names}, the attributes {attrs}, or any name or attribute that starts
  with `__`. It may not read files or use the network."""

PY_INPUTS = "Your folder is run like this:"

ARC_FREE_ARCH = """- `arch.py`: defines `forward(w, q, o)`. `w` is the dict from `build()`, `q` is int64 (n, {q_len}) question word
  ids, `o` is int64 (n, 4, {o_len}) option word ids; it returns float (n, 4) logits. arch.py must be a fixed
  tensor program: imports only `torch` and `math`; no `if`, `while`, `and`/`or`, conditional expressions, `try`,
  `with`, `lambda` or classes; no `.item()`, `.tolist()`, `.numpy()`; none of the names {names}. `for` loops over
  a fixed range are allowed. Everything that depends on the question must happen in tensor operations on the
  weights."""

ARC_FIXED_ARCH = """- `arch.py` is given: it is already in the folder and must stay exactly as it is (a changed arch.py makes the
  model invalid). Your `build()` must return exactly these weights, where V is the number of lines of vocab.txt
  plus 2:

```
{shape_table}
```

The given arch.py:

```python
{arch_source}
```"""

DESIGN_CIFAR = """# Task: design and write a CIFAR-10 classifier yourself

You will design and write all of the content of a model that classifies CIFAR-10 images. You choose the structure:
what the weights are, how they are built from what you write, and how the forward pass uses them. Nothing is trained
and nothing pretrained is available: every number in the model comes from what you write. The model must build and run
here within {memory_gb} GB of memory and {minutes} minutes on 4 CPU cores; there is no other limit on its size.

Classes (label index: name): {classes}
Input: float32 RGB images with values in [0, 1], shape (N, 3, 32, 32), no normalization.
Output: 10 logits per image; the predicted class is the argmax. An image on which all ten logits are equal counts as a
uniform guess (1/10).
Your score is the accuracy on the {n_test} held-out CIFAR-10 test images, which you cannot see.

## What you deliver: the folder /srv/work/model

- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
- `arch.py`: defines `forward(w, x)`. `w` is the dict from `build()`, `x` is float32 (n, 3, 32, 32); it returns float
  (n, 10) logits. arch.py must be a fixed tensor program: imports only `torch` and `math`; no `if`, `while`,
  `and`/`or`, conditional expressions, `try`, `with`, `lambda` or classes; no `.item()`, `.tolist()`, `.numpy()`;
  none of the names {names}. `for` loops over a fixed range are allowed. Everything that depends on the image must
  happen in tensor operations on the weights.
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

Your folder is run like this:

```python
{runner}
```

## Tools

`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_images.npz with an array `x` (float32, shape (n, 3, 32, 32), values in [0, 1]) and optionally `y`
(int64 labels), it runs the model on those images; they must be images you made yourself, not CIFAR-10 images.

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No CIFAR-10 data is available to you.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""

ARC_TOOLS = """`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_questions.json (see the tool's docstring) it runs the model on those questions; they must be
questions you made up."""

ARC_TOOLS_NOQ = """`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it on one placeholder
question. Do not make up questions to run the model on, with this tool or in any other way: build the model from your
knowledge alone."""

ARC_TOOLS_FB = """

`python /srv/work/tools/check_model.py --practice` runs the model on the {n} ARC training questions in
/srv/work/data/practice.jsonl (question, options, answer) and prints its accuracy on them. You may read that file and
use this as often as you like."""

ARC_RULE_DATA = "- No ARC data is available to you apart from the ten questions above."
ARC_RULE_DATA_FB = """- No ARC data is available to you apart from the ten questions above and
  /srv/work/data/practice.jsonl. Do not copy practice questions or their answers into the model, and do not set
  anything in the model by an automatic search or fit against them; change the model only by your own reasoning about
  what you measured."""

ARC_TRAINED = """Nothing is trained and nothing pretrained is
available: every word the model knows and every number in it comes from what you write."""
ARC_SIZE = "; there is no other limit on its size."
FIT_TRAINED = """Nothing pretrained is available:
every word the model knows and every number in it comes from what you write or fit on the ARC training questions."""
FIT_SIZE = ", and the folder /srv/work/model may hold at most\n{gb} GB of files."
ARC_ONLY_EXAMPLES = "These are the only ARC questions you see; the test questions\nare different."
FIT_EXAMPLES = "The test questions are different."
ARC_RULE_FIT = """- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design."""
FIT_RULE_FIT = """- You may fit or train the model on the questions in /srv/work/data/arc_train.jsonl in any way, including gradient
  descent, on this machine's CPU during your run, and you may measure its accuracy on them. Save what you fit as files
  in /srv/work/model (for example .npy or .pt files): build.py must build the weights from the files in that folder
  and must not read arc_train.jsonl, which is not there when check_model.py builds the model or when it is scored."""
ARC_RULE_DATA_FIT = """- No ARC data is available to you apart from the ten questions above and /srv/work/data/arc_train.jsonl, the
  {n:,} ARC training questions with their answers, one JSON object per line (question, options, answer letter); the
  ten questions above are among them, and no validation or test question is."""

ARC_PARALLEL = " Design that structure for parallel hardware, so that its forward pass runs well on a GPU."

ARC_NOFACT = (" Do not build it as a store of facts, rules or word associations that is searched or matched against the\n"
              "question and the options: no fact list, knowledge base, key-value or association table, and no scoring by\n"
              "which stored words the question and an option share. What the model knows must be held in another form, for\n"
              "example distributed vectors that every weight contributes to, or features that are computed.")

PARENT_CONVERT = """## The parent model (generation {generation})

A model a writer designed in an earlier generation, in the same format as yours: /srv/work/parent/model, with its
NOTES.md explaining the structure. On the {n_val} validation questions (not the test questions) it reached
{accuracy:.1f}% accuracy. That number is all you get about its results, not which questions it got right or wrong.

Rewrite the parent's model for parallel hardware: keep what it knows, and restructure how the weights are laid out
and how the forward pass uses them so that it runs well on a GPU. You may change anything else in the folder too.
"""

DISTILL_ARC = """# Task: write the training data for a small model that answers ARC science questions

You will write the training data for a small model that answers multiple-choice science questions of the AI2
Reasoning Challenge (ARC), grade-school level, Easy and Challenge sets together. You do not build the model. After you
finish, standard small neural networks are trained by gradient descent on your data and nothing else, and then scored.
Nothing pretrained is available to them: every word they know and everything they learn comes from what you write.

## What happens to your data

- The networks score each (question, option) pair, and the answer is the option with the highest score; an option's
  position carries no information. Four standard families are trained: an MLP over mean word embeddings, a
  convolutional network, a two-layer transformer encoder and a two-layer state-space model, each at about 0.5, 0.8 and
  3.3 million parameters.
- Their vocabulary is the {vocab:,} most frequent words of your questions and options; any other word is unknown to
  them. Text becomes word ids as shown below, and a question keeps its first {q_len} words and an option its first
  {o_len}.
- Each network is trained with Adam at three learning rates. The checkpoint and learning rate with the best accuracy
  on held-out ARC validation questions are kept; you never see those questions or that accuracy.
- Your score is the test accuracy of the network chosen by validation accuracy, on the {n_test} held-out test
  questions, which you cannot see.

## What you deliver: the folder /srv/work/{deliverable}

- `{train_file}`: your training questions, one JSON object per line: `{{"question": "...", "options": ["...", "...",
  "...", "..."], "answer": k}}` with k from 0 to 3.
- Optionally the files and programs you used to produce `{train_file}` (for example a `make_data.py` that turns
  facts you wrote into questions), and `NOTES.md`: what your data covers and why.

Text becomes word ids like this:

```python
TOKEN = re.compile(r"[a-z0-9']+")
Q_LEN, O_LEN = {q_len}, {o_len}

{encode}
```
"""

DISTILL_TOOLS = """`python /srv/work/tools/check_data.py` checks `{train_file}`: the format, the number of questions, exact
duplicates, the spread of the answer position, and how many of your words fall outside the {vocab:,}-word vocabulary.
It trains nothing and reports no accuracy."""

ARC_TAIL = """
## Example questions

{examples_note}

{examples}

## Tools

{tools}

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No {name} data is available to you apart from the ten questions above.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""

LM_TASK = """# Task: write knowledge into the weights of a released language model for ARC science questions

You will write knowledge into the weights of a language model with the architecture and tokenizer of
{repo} ({family} architecture, {n_params:,} parameters), so that it answers
multiple-choice science questions of the AI2 Reasoning Challenge (ARC), grade-school level, Easy and Challenge sets
together. Nothing is trained: every change to a weight is one that you decide and write.

{start}

Each question is given to the model as this prompt, tokenised by the model's tokenizer, which
{added}:

```
{prompt}
```

{readout} Each letter's logit is the larger of the
logits of the tokens "A" and " A" (likewise B, C and D), and the model's answer is the letter with the highest logit.
The weights are stored in bfloat16 and the forward pass computes in float32. Your score is the accuracy on the
{n_test} held-out test questions, which you cannot see.

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
finish within {minutes} minutes on 4 CPU cores, under a memory limit of {memory_gib} GiB counted as reserved address
space: the libraries take about 5 GiB of it and the starting weights file {weights_gb:.1f} GB. Put `NOTES.md` in the
folder as well: what you changed and why.

When you are done, edit.py is run once more in the same way and the edited model is scored on a GPU. The score is not
shown to you.

The given arch.py:

```python
{arch_source}
```

## Example questions

Five training questions from each set, with answers. These are the only ARC questions you see; the test questions
are different.

{examples}

## Tools

`python /srv/work/tools/check_lm.py` checks arch.py, runs your edit, checks the names, shapes and values it returns,
counts the parameters it changes and runs the edited model on CPU as it is scored, with the prompt and readout above.
If you write /srv/work/my_questions.json (see the tool's docstring) it runs the model on those questions, at most
{max_questions}; they must be questions you made up. It reports the time taken and the peak memory.

## Rules

- No training or fitting of any kind, and no automatic search over the weights from measured accuracy. Every change
  must come from your own knowledge and design.
- No ARC data is available to you apart from the ten questions above.
- Do not access the internet or any files outside /srv/work, /srv/start and /srv/lib.

When you are done, make sure check_lm.py passes, then reply with a short description of what you changed.
"""

OLMO_HEAD = """# Task: {title}

You will edit the weights of OLMo 2 1B ({repo}, {n_params:,} parameters) by hand. Nothing is trained by you: every
change to a weight is one that you decide and write.

{start}

## What happens to your model afterwards

{afterwards}

## What you deliver: /srv/work/model/edit.py

/srv/work/model/arch.py is the model's forward pass. It is given for reference and is not run from your folder; the
weights are named and shaped as it reads them, and /srv/start/manifest.json lists every name and shape. The weights
are stored and computed in float32.

Write /srv/work/model/edit.py defining `edit(w, tok)`:
- `w` maps every weight name to its starting tensor (float32). A tensor is read from /srv/start/weights.safetensors
  only when you use it, and changing it in place has no effect: only what `edit` returns is applied.
- `tok` is the OLMo 2 tokenizer, a Hugging Face tokenizer loaded from /srv/start/tokenizer.
- It returns a dict of only the weights you change: name -> torch tensor of that weight's shape, in any float dtype.
  They are stored as float32 and replace the starting tensors; every other weight keeps its starting value.

`edit` runs with /srv/work/model as working directory and may read and import any file you put in that folder, and may
use any Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine. It must
finish within {minutes} minutes on 4 CPU cores, under a memory limit of {memory_gib} GiB counted as reserved address
space: the libraries take about 5 GiB of it and the starting weights file {weights_gb:.1f} GB. Put `NOTES.md` in the
folder as well: what you changed and why.

When you are done, edit.py is run once more in the same way and the edited model is measured; the measurement is not
shown to you.

The given arch.py:

```python
{arch_source}
```

## Tools

`python /srv/work/tools/check_olmo.py` runs your edit, checks the names, shapes and values it returns, counts the
parameters it changes and prints the RMS of each changed tensor before and after. If you write /srv/work/my_texts.json
(at most {max_texts} strings) it prints the next-token loss of the edited and the unedited model on each; if you write
/srv/work/my_prompts.json (at most {max_prompts} strings) it prints the edited model's greedy continuation of each
({prompt_format}). Texts and prompts in those files must be ones you wrote. See the tool's docstring.

## Rules

- No training or fitting of any kind: no gradient steps, and no automatic search over the weights driven by a
  measured loss or score. Every change must come from your own knowledge and design. Weights written as formulas, or
  generated with code from text and tables you wrote yourself, are allowed.
- No data is available to you: no {data_name}, no evaluation set.
- Do not access the internet or any files outside /srv/work, /srv/start and /srv/lib.

When you are done, make sure check_olmo.py passes, then reply with a short description of what you changed.
"""

OLMO_TEXT = {
    "olmo-pt": dict(
        title="write a better starting point for pretraining OLMo 2 1B",
        start="Your starting point is OLMo 2 1B's official initialisation (revision {revision}): the random weights its "
              "pretraining began from, never trained. Its next-token loss is about {uniform:.1f} nats, uniform over "
              "the {vocab:,}-token vocabulary.",
        afterwards="Your edited model and the unedited initialisation are each pretrained from scratch, with the same "
                   "recipe, data order and number of steps: the OLMo 2 1B recipe (AdamW with warmup and decay of the "
                   "learning rate, weight decay) on about 5 billion tokens of Dolma 3, a mix of web pages, code, "
                   "academic papers and other text. The two are compared on next-token loss on held-out text at the "
                   "same numbers of training tokens during pretraining, and on standard benchmarks at the end. Your "
                   "aim is an edited model that learns faster and ends better. Both are trained with the optimizer "
                   "settings of the unedited initialisation, so weights whose scale is far from it may train badly.",
        prompt_format="plain text continuation", data_name="pretraining text"),
    "olmo-sft": dict(
        title="edit OLMo 2 1B before its supervised fine-tuning",
        start="Your starting point is the released OLMo 2 1B base model (revision {revision}): pretrained on about 4 "
              "trillion tokens, not instruction-tuned.",
        afterwards="Your edited model and the unedited base are each fine-tuned with the same supervised fine-tuning "
                   "(SFT) recipe on the same instruction data (Dolci-Instruct-SFT, conversations of instructions and "
                   "answers), in this chat format, where the model learns to write the answer and then "
                   "<|endoftext|>:\n\n```\n{chat}```\n\nAfter SFT both are evaluated in the same chat format on "
                   "standard benchmarks: instruction following (IFEval), grade-school math (GSM8K), knowledge (MMLU) "
                   "and others. Your aim is an edited model that scores higher after SFT than the unedited base does.",
        prompt_format="in the chat format above", data_name="instruction data"),
}

LM_START = {
    "pre": "The starting weights are the released pretrained base checkpoint, as published, before any instruction\n"
           "tuning.",
    "init": "The starting weights are a random initialisation: the model's own initialisation scheme (that of its\n"
            "Hugging Face implementation) with seed {seed}. They hold nothing from pretraining; only the tokenizer is\n"
            "the released one.",
}
LM_FAMILY = {"llama": "Llama", "qwen3": "Qwen3", "qwen3_5": "Qwen3.5", "gemma3": "Gemma 3", "gemma2": "Gemma 2",
             "mamba2": "Mamba-2", "rwkv7": "RWKV-7", "modernbert": "ModernBERT", "ultrabert": "UltraBERT"}
LM_READOUT = "The model reads the prompt and gives next-token logits at its last token."
LM_MASKED_READOUT = ("The model is a bidirectional encoder trained by masked-language modelling: it reads the whole\n"
                     "prompt at once, the mask token {mask} included, and gives logits for the token at the position of\n"
                     "that mask token, which stands where the answer letter goes.")

RULES_COMMON = """
## Rules

- No training or fitting of any kind. Do not use gradient descent, least squares, pseudo-inverses,
  regression, clustering, PCA/SVD, or any other procedure that computes weight values from data.
  Do not search over weight values automatically (random search, evolutionary search,
  hill-climbing loops, or any loop that adjusts weights from measured accuracy). The weight values
  must come from you.
- Do not access the internet or any files outside /srv/work.
"""

RULE_BLIND = """- No CIFAR-10 data is available to you. Use only your own knowledge of the dataset.
- You cannot run the network on images.
"""

RULE_OPEN = """- You may inspect the training set freely: /srv/work/data/train.npz holds `x` (uint8, shape
  (50000, 3, 32, 32)) and `y` (int64 labels). `python /srv/work/tools/view.py --help` renders images as
  a PNG grid that you can open with the Read tool. You may compute any descriptive statistics you
  like with Python to understand the data. If a statistic informs a weight, write the number
  yourself; do not have code copy computed statistics into the weights.
"""

RULE_VIEW = """- You may look at the training set, without its labels as numbers:
  /srv/work/data/train_images.npz holds `x` (uint8, shape (50000, 3, 32, 32)) with no labels, and
  /srv/work/views/<class>_<k>.png (k = 0, 1, 2) each show 64 training images of one class. Open the
  PNGs with the Read tool to look at them. Do not load these PNGs into code.
- You may compute class-agnostic statistics of the unlabeled images with Python (for example the
  overall range of a colour or edge measure). Do not compute statistics for individual classes by
  any means. Class-specific weight values must come from your own judgment.
"""

RULE_WORD_SHEETS = """- /srv/work/patch_words_0.png and /srv/work/patch_words_1.png are part of the architecture. For every patch
  word (its number is at the left) they show the mean of the training patches that get the word, then 6 such
  patches, as raw pixels enlarged. Open them with the Read tool.
"""

RULE_BLIND_WORDS = """- No CIFAR-10 data is available to you apart from the two patch word sheets. Use only your own knowledge of
  the dataset.
- You cannot run the network on images.
""" + RULE_WORD_SHEETS

RULE_VIEW_WORDS = """- You may look at the training set, without its labels as numbers:
  /srv/work/data/train_images.npz holds `x` (uint8, shape (50000, 3, 32, 32)) with no labels, and
  /srv/work/data/train_words.npz holds `words` (uint8, shape (50000, 14, 14)), the patch word of every patch of
  the same images in the same order.
- /srv/work/views/<class>_<k>.png (k = 0, 1, 2) each show 64 training images of one class, 8 per row.
  /srv/work/views/<class>_<k>.txt lists patch words of the same 64 images, one image per line in the PNG's
  row-by-row order: for each quadrant (TL, TR, BL, BR) the words of 9 of its 49 patches (patch rows and columns
  1, 3, 5 and 8, 10, 12), row by row. Open these class files with the Read tool only: do not load them into
  code or search them with Grep or Glob.
- You may compute class-agnostic statistics of the unlabelled images and words with Python (for example word
  frequencies, or which words occur together). Do not compute statistics for individual classes by any means.
  Code may show you statistics; it must not turn them into class assignments or weight values (for example by
  counting, over the words that occur with a word, the classes you gave them). Class-specific weight values
  must come from your own judgment.
- Patch words of images are statistics of the inputs. Anything computed with your score values on training
  images counts as running the network.
""" + RULE_WORD_SHEETS

RULE_TOKEN_SHEETS = """- /srv/work/token_sheet_00.png to token_sheet_15.png are part of the architecture. Each shows 16 tokens, one
  per row, with the token id at the left. A row has 5 pairs of tiles for 5 patches with that token, drawn at
  random from 2,000 unlabelled training images: the image the patch lies in, then the same image with every
  patch of this token at full brightness and all other patches dimmed. Open them with the Read tool.
"""

RULE_DIMENSION_SHEETS = """- /srv/work/dim_sheet_00.png to dim_sheet_47.png are part of the architecture. Each shows 16 dimensions, one
  per row, with the dimension index at the left. A row shows the 5 images with the highest value of that dimension,
  then, after a bar, the 5 images with the lowest value, among 2,000 unlabelled training images. Open them with the
  Read tool.
"""

RULE_BLIND_EMBEDDINGS = """- No labelled CIFAR-10 data is available to you. The dimension sheets show unlabelled training images; apart
  from them, use only your own knowledge of the dataset.
- You cannot run the network on images.
""" + RULE_DIMENSION_SHEETS

RULE_BLIND_TOKENS = """- No labelled CIFAR-10 data is available to you. The token sheets show unlabelled training images; apart
  from them, use only your own knowledge of the dataset.
- You cannot run the network on images.
""" + RULE_TOKEN_SHEETS

RULE_VIEW_TOKENS = """- You may look at the training set, without its labels as numbers:
  /srv/work/data/train_tokens.npz holds `x` (uint8, shape (50000, 256)), the tokens of the 50,000 training
  images with no labels.
- /srv/work/views/<class>_<k>.png (k = 0, 1, 2) each show 64 training images of one class, 8 per row.
  /srv/work/views/<class>_<k>.txt lists, for the same 64 images, one image per line in the PNG's row-by-row
  order, the 10 most frequent tokens of the image as id:count (count = how many of its 256 patches got the
  token). Open these class files with the Read tool only: do not load them into code or search them with Grep
  or Glob.
- You may compute class-agnostic statistics of the unlabelled tokens with Python (for example token
  frequencies, or which tokens occur together). Do not compute statistics for individual classes by any means.
  Code may show you statistics; it must not turn them into class assignments or weight values (for example by
  counting, over the tokens that occur with a token, the classes you gave them). Class-specific weight values
  must come from your own judgment.
- Tokens of images are statistics of the inputs. Anything computed with your score values on training images
  counts as running the network.
""" + RULE_TOKEN_SHEETS

RULE_OPEN_AGNEWS = """- You may inspect the training set freely: /srv/work/data/train.jsonl holds the 120,000 training
  articles, one JSON object per line with `index`, `label` (class name) and `text`, and
  /srv/work/data/train.npz holds the same articles as token ids `x` (int64, shape (120000, 64)) and labels
  `y`. You may compute any descriptive statistics you like with Python to understand the data. If a
  statistic informs a weight, write the number yourself; do not have code copy computed statistics into
  the weights.
"""

RULE_VIEW_AGNEWS = """- You may read the training set, without its labels:
  /srv/work/data/train_texts.jsonl holds the 120,000 training articles (`index`, `text`) with no labels,
  and /srv/work/views/<class>_<k>.txt (k = 0, 1, 2) each hold 64 training articles of one class. Read the
  class files with the Read tool only: do not load them into code or search them with Grep or Glob.
- You may compute class-agnostic statistics of the unlabelled texts with Python (for example overall
  word frequencies). Do not compute statistics for individual classes by any means. Class-specific weight
  values must come from your own judgment.
"""

RULE_VIEW_CIPHER = """- You may read the training set, without its labels:
  /srv/work/data/train_ids.npz holds `x` (int64, shape (120000, 64)), the 120,000 training articles as
  ids with no labels, and /srv/work/views/<class>_<k>.txt (k = 0, 1, 2) each hold 64 training articles of one
  class, one article per line as space-separated ids. Read the class files with the Read tool only: do not
  load them into code or search them with Grep or Glob.
- You may compute class-agnostic statistics of the unlabelled ids with Python (for example overall id
  frequencies or co-occurrence). Do not compute statistics for individual classes by any means.
  Class-specific weight values must come from your own judgment.
- Do not try to work out which word an id stands for, and do not use any guess of that kind.
"""

RULE_BLIND_AGNEWS = """- No AG News data is available to you apart from /srv/work/vocab.txt. Use only your own knowledge of the
  dataset and of news writing.
- You cannot run the network on articles.
"""

RULE_INPUT_STATS = """- Statistics of the inputs themselves (words, token ids, token counts) are descriptive statistics.
  Anything computed with your weight values on training articles counts as running the network.
"""

ARCH_RULES = {"white": """- Running part of the network, or a copy of part of it (for example `_ratio_features`), on images
  counts as running the network.
"""}

RULE_NO_FORWARD = """- Do not run the network, or any re-implementation of it, on {items}.
"""

RULE_FORWARD = """- Do not run the network, or any re-implementation of it, on {items} yourself. The only way to
  run it is `python /srv/work/tools/forward.py`, which builds your current script and reports the
  overall accuracy and the count of predictions per class on 512 fixed practice {items} drawn from the
  training set. This report is allowed feedback. Every network you submit must be a classifier for
  this task; do not submit networks built to measure properties of the practice {items}.
  You have {budget} calls in total; calls beyond that are refused. A call whose build fails does not
  count. The run ends early if the best practice accuracy has not improved over {plateau}
  consecutive calls.
"""

FOOTER = """
When you are done, make sure /srv/work/build_weights.py passes check.py, then reply with a short
description of how your network computes its prediction.
"""


def arch_section(arch: str) -> str:
    fns = [archs.ARCHS[arch].forward, *archs.ARCHS[arch].shown]
    source = "\n\n".join(inspect.getsource(f) for f in fns).rstrip()
    if archs.ARCHS[arch].consts:
        module = inspect.getmodule(archs.ARCHS[arch].forward)
        consts = {k: getattr(module, k) for k in archs.ARCHS[arch].consts}
        source = "\n".join(f"{k} = {v!r}" for k, v in consts.items()) + "\n\n" + source
    table = "\n".join(f"{k}: {s}" for k, s in archs.shapes(arch).items())
    return ARCH.format(arch=arch, n_params=archs.n_params(arch), source=source, shape_table=table)


TEXTS = {"cifar": (HEADER, RULE_OPEN, RULE_VIEW, RULE_BLIND),
         "cifar_words": (HEADER, None, RULE_VIEW_WORDS, RULE_BLIND_WORDS),
         "cifar_tokens": (HEADER_TOKENS, None, RULE_VIEW_TOKENS, RULE_BLIND_TOKENS),
         "cifar_emb": (HEADER_EMBEDDINGS, None, None, RULE_BLIND_EMBEDDINGS),
         "agnews": (HEADER_AGNEWS, RULE_OPEN_AGNEWS + RULE_INPUT_STATS, RULE_VIEW_AGNEWS + RULE_INPUT_STATS,
                    RULE_BLIND_AGNEWS),
         "agnews_cipher": (HEADER_CIPHER, None, RULE_VIEW_CIPHER + RULE_INPUT_STATS, None)}


def render_memory(arch: str) -> str:
    from wsp import mmlu  # reads the benchmark files; host only
    fns = [archs.ARCHS[arch].forward, *archs.ARCHS[arch].shown]
    consts = "\n".join(f"{k} = {getattr(mem, k)!r}" for k in archs.ARCHS[arch].consts)
    subject = arch.removeprefix("mem-")
    return MEMORY_TASK.format(subject=subject.replace("_", " "), arch=arch, n_test=len(mmlu.load("test", subject)[2]),
                              source=consts + "\n\n" + "\n\n".join(inspect.getsource(f) for f in fns).rstrip(),
                              examples=mmlu.examples(subject), max_lines=mem.MAX_LINES, max_chars=mem.MAX_LINE_CHARS)


def parent_section(parent: dict) -> str:
    v = parent["validation"]
    if parent["arch"].startswith("mem-"):
        return PARENT_MEMORY.format(units=parent["units"], answered=100 * v["answered"], accuracy=100 * v["accuracy"],
                                    accuracy_answered=100 * v["accuracy_answered"])
    if (given := lineage.given_lineage(parent["arch"])) is not None:
        return PARENT_GIVEN.format(generation=given[1], n_val=v["n"], accuracy=100 * v["accuracy"])
    vl = lineage.variant_lineage(parent["arch"])
    generation = vl[1] if vl else lineage.parse(parent["arch"])[0]
    return PARENT_MODEL.format(generation=generation, n_val=v["n"], accuracy=100 * v["accuracy"])


def render_free(arch: str, parent: dict) -> str:
    from wsp import mmlu
    subject = lineage.parse(arch)[1]
    return FREE_TASK.format(
        subject=subject.replace("_", " "), n_test=len(mmlu.load("test", subject)[2]),
        memory_gb=free.TENSOR_GB, minutes=free.RUN_TIMEOUT_S // 60, q_len=free.Q_LEN, o_len=free.O_LEN,
        names=", ".join(f"`{n}`" for n in sorted(free.ARCH_FORBIDDEN_NAMES)),
        encode=inspect.getsource(free.words) + "\n\n" + inspect.getsource(free.encode).rstrip(),
        runner=free.RUNNER.rstrip(), parent=parent_section(parent), examples=mmlu.examples(subject))


def render_distill_arc() -> str:
    from wsp import arc, distill
    common = dict(vocab=distill.VOCAB, q_len=free.Q_LEN, o_len=free.O_LEN, n_test=len(arc.load("test")[2]),
                  deliverable=distill.DELIVERABLE, train_file=distill.TRAIN_FILE,
                  encode=inspect.getsource(free.words) + "\n\n" + inspect.getsource(free.encode).rstrip())
    tail = ARC_TAIL.format(examples=arc.examples(), tools=DISTILL_TOOLS.format(**common), **BENCHMARK["arc"])
    tail = tail.replace("in the model must come from your own knowledge and design",
                        "in the data must come from your own knowledge and design")
    tail = tail.replace("search over the model's content or structure", "search over the data")
    tail = tail.replace("make sure check_model.py passes, then reply with a short description of your model",
                        "make sure check_data.py passes, then reply with a short description of your data")
    return DISTILL_ARC.format(**common) + tail


def render_arc(arch: str, parent: dict | None) -> str:
    from wsp import arc
    given = lineage.given_lineage(arch)
    return render_choice(arch, given[0] if given else arch.removesuffix("-arc"), arc, parent)


def render_mathbench(arch: str) -> str:
    from wsp import mathbench
    return render_choice(arch, arch.removesuffix("-mb"), mathbench, None)


def render_mmlu_all(arch: str) -> str:
    from wsp import mmlu_all
    return render_choice(arch, arch.removesuffix("-mm"), mmlu_all, None)


def render_choice(arch: str, name: str, data: ModuleType, parent: dict | None) -> str:
    """Task text of an ARC or MathBench architecture. `name` is the given architecture of a fixed one and `data` the
    benchmark's module (wsp.arc, wsp.mathbench, wsp.mmlu_all)."""
    from wsp import given_archs
    phrases = BENCHMARK[archs.ARCHS[arch].task]
    variant = lineage.arc_variant(arch)
    encode = free.encode_of(arch)
    token, words = ("MATH_TOKEN", free.math_words) if encode is free.math_encode else ("TOKEN", free.words)
    common = dict(memory_gb=free.TENSOR_GB, minutes=free.RUN_TIMEOUT_S // 60, q_len=free.Q_LEN, o_len=free.O_LEN,
                  n_test=len(data.load("test")[2]), token=f'{token} = re.compile(r"{getattr(free, token).pattern}")',
                  encode=inspect.getsource(words) + "\n\n" + inspect.getsource(encode).rstrip(),
                  runner=(free.PY_RUNNER if variant == "py" else free.RUNNER).rstrip())
    common |= dict(vocab_item="" if variant == "py" else ARC_VOCAB_ITEM.format(**phrases),
                   build_item=PY_BUILD_ITEM if variant == "py" else ARC_BUILD_ITEM,
                   inputs=PY_INPUTS if variant == "py" else ARC_INPUTS.format(**common))
    if lineage.parse(arch) or variant:
        head = dict(verb="design and write", verb_long="design and write all of the content of",
                    structure="You choose the structure: what the weights are, how they are built from what you write, "
                              "and how the forward pass uses them." + (ARC_PARALLEL if variant == "par" else "")
                              + (ARC_NOFACT if variant == "nofact" else ""),
                    arch_item=PY_ARCH.format(**{k: ", ".join(f"`{n}`" for n in sorted(v)) for k, v in (
                        ("modules", free.PY_FORBIDDEN_MODULES), ("names", free.PY_FORBIDDEN_NAMES),
                        ("attrs", free.PY_FORBIDDEN_ATTRS))}) if variant == "py"
                    else ARC_FREE_ARCH.format(q_len=free.Q_LEN, o_len=free.O_LEN,
                                              names=", ".join(f"`{n}`" for n in sorted(free.ARCH_FORBIDDEN_NAMES))))
    else:
        head = dict(verb="write the weights of", verb_long="write every weight of",
                    structure=given_archs.structure(name),
                    arch_item=ARC_FIXED_ARCH.format(shape_table=given_archs.shape_table(name),
                                                    arch_source=given_archs.source(name).rstrip()))
    if variant in lineage.CONVERT_VARIANTS:
        v = parent["validation"]
        lineage_part = "\n" + PARENT_CONVERT.format(generation=lineage.parse(parent["arch"])[0], n_val=v["n"],
                                                     accuracy=100 * v["accuracy"])
    else:
        lineage_part = "\n" + parent_section(parent) if parent else ""
    tools = ARC_TOOLS_NOQ if variant == "noq" else ARC_TOOLS
    if variant == "fb":
        tools += ARC_TOOLS_FB.format(n=data.N_PRACTICE)
    tail = ARC_TAIL.format(examples=data.examples(), tools=tools, **phrases)
    if variant == "fb":
        tail = tail.replace(ARC_RULE_DATA, ARC_RULE_DATA_FB)
    head_text = ARC_HEAD.format(**common, **head, **phrases)
    if variant == "fit":
        head_text = head_text.replace(ARC_TRAINED, FIT_TRAINED).replace(
            ARC_SIZE, FIT_SIZE.format(gb=free.FIT_MAX_BYTES // 10 ** 9))
        tail = tail.replace(ARC_ONLY_EXAMPLES, FIT_EXAMPLES).replace(ARC_RULE_FIT, FIT_RULE_FIT).replace(
            ARC_RULE_DATA, ARC_RULE_DATA_FIT.format(n=len(data.load("train")[0])))
    return head_text + lineage_part + tail


SQUAD_TASK = """# Task: {verb} a model that answers questions about a paragraph

You will {verb_long} a model for SQuAD 2.0. Each item is a question and a Wikipedia paragraph. The model marks the
words of the paragraph that answer the question, or says that the paragraph does not answer it. {structure} Nothing is
trained and nothing pretrained is available: every word the model knows and every number in it comes from what you
write. The model must build and run here within {memory_gb} GB of memory and {minutes} minutes on 4 CPU cores; there
is no other limit on its size.

Your score is the F1 of the official SQuAD 2.0 evaluation on {n_test} held-out items, which you cannot see: the
word overlap between the model's answer and the closest of the reference answers, after lower-casing and removing
punctuation and the articles a, an, the. About half of the items cannot be answered from their paragraph; on those
the only right output is "no answer", and "no answer" on an answerable item scores 0.

## What you deliver: the folder /srv/work/model

- `vocab.txt`: the words your model knows, one per line. Line k (0-based) is word id k + 2 and id 0 is padding. A
  word that is not in the file still has an id of its own (see below), so a name or a number has the same id in the
  question and in the paragraph.
- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
{arch_item}
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

The output of `forward` is float (n, 2, {c_len} + 1): row 0 holds start logits and row 1 end logits. Column 0 stands
for "no answer" and column k for word k - 1 of the paragraph. The answer is the words i..j, at most {max_span} of
them, with the largest start[i] + end[j], cut from the paragraph text; it is "no answer" unless that sum is larger
than start[0] + end[0].

Text becomes word ids, your folder is run, and the output becomes an answer like this:

```python
SPAN_TOKEN = re.compile(r"{token}")
SQ_LEN, SC_LEN = {q_len}, {c_len}
HASH_IDS = {hash_ids}
MAX_SPAN = {max_span}

{encode}
```

```python
{runner}
```

## Example items

Five answerable and five unanswerable training items, with answers. These are the only SQuAD items you see; the
test items are different and come from other articles.

{examples}

## Tools

`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_questions.json (see the tool's docstring) it runs the model on those items and prints its answers; they
must be items you made up.

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No SQuAD data is available to you apart from the ten items above.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""
SQUAD_FREE_ARCH = """- `arch.py`: defines `forward(w, q, c)`. `w` is the dict from `build()`, `q` is int64 (n, {q_len}) question word
  ids, `c` is int64 (n, {c_len}) paragraph word ids. arch.py must be a fixed
  tensor program: imports only `torch` and `math`; no `if`, `while`, `and`/`or`, conditional expressions, `try`,
  `with`, `lambda` or classes; no `.item()`, `.tolist()`, `.numpy()`; none of the names {names}. `for` loops over
  a fixed range are allowed. Everything that depends on the question or the paragraph must happen in tensor
  operations on the weights."""


# SQuAD 1.1 (<name>-sq1): SQUAD_TASK with these passages replaced; every item has an answer
SQUAD1_CHANGES = (
    ("a model for SQuAD 2.0. Each item is a question and a Wikipedia paragraph. The model marks the\n"
     "words of the paragraph that answer the question, or says that the paragraph does not answer it.",
     "a model for SQuAD 1.1. Each item is a question and a Wikipedia paragraph that contains the answer. The model\n"
     "marks the words of the paragraph that answer the question."),
    ("the official SQuAD 2.0 evaluation", "the official SQuAD 1.1 evaluation"),
    (" About half of the items cannot be answered from their paragraph; on those\n"
     "the only right output is \"no answer\", and \"no answer\" on an answerable item scores 0.",
     " Every item has an answer in its\nparagraph."),
    ("Column 0 stands\nfor \"no answer\" and column k for word k - 1 of the paragraph.",
     "Column k stands\nfor word k - 1 of the paragraph and column 0 is ignored."),
    ("; it is \"no answer\" unless that sum is larger\nthan start[0] + end[0].", "."),
    ("Five answerable and five unanswerable training items, with answers.", "Ten training items with their answers."),
)
GEN_TASK = """# Task: {verb} a model that answers science questions in words

You will {verb_long} a model that answers science exam questions of the SciQ dataset. There are no options: the
model writes the answer itself, one word at a time. {structure} Nothing is trained and nothing pretrained is
available: every word the model knows and every number in it comes from what you write. The model must build and run
here within {memory_gb} GB of memory and {minutes} minutes on 4 CPU cores; there is no other limit on its size.

Your score is the F1 on {n_test} held-out questions, which you cannot see: the word overlap between the model's
answer and the reference answer, after lower-casing and removing punctuation and the articles a, an, the. Reference
answers are short, one or two words for most questions and never more than {a_len} words here.

## What you deliver: the folder /srv/work/model

- `vocab.txt`: the words your model knows, one per line. Line k (0-based) is word id k + 2; id 0 is padding and
  id 1 is any word not in the file. The model can only write words that are in this file.
- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
{arch_item}
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

The model writes an answer word by word. `forward` is called {a_len} times; each time it gets the question and the
answer written so far and returns logits over the vocabulary, float (n, V) with V = lines of vocab.txt plus 2. The
word with the largest logit is appended to the answer. Id 0 ends the answer: once it is chosen nothing more is added.

Text becomes word ids, your folder is run, and the output becomes an answer like this:

```python
TOKEN = re.compile(r"{token}")
GQ_LEN, GA_LEN = {q_len}, {a_len}

{encode}
```

```python
{runner}
```

## Example questions

Ten training questions with their answers. These are the only SciQ questions you see; the test questions are
different.

{examples}

## Tools

`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_questions.json (see the tool's docstring) it runs the model on those questions and prints the answers
it writes; they must be questions you made up.

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No SciQ data is available to you apart from the ten questions above.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""
GEN_FREE_ARCH = """- `arch.py`: defines `forward(w, q, a)`. `w` is the dict from `build()`, `q` is int64 (n, {q_len}) question word
  ids, `a` is int64 (n, {a_len}) ids of the answer words written so far (0 where no word is written yet); it returns
  float (n, V) logits of the next word. arch.py must be a fixed
  tensor program: imports only `torch` and `math`; no `if`, `while`, `and`/`or`, conditional expressions, `try`,
  `with`, `lambda` or classes; no `.item()`, `.tolist()`, `.numpy()`; none of the names {names}. `for` loops over
  a fixed range are allowed. Everything that depends on the question or the answer so far must happen in tensor
  operations on the weights."""


ALL_TASK = """# Task: design and write one model for four kinds of text tasks

You will design and write all of the content of one model that reads a text and writes words. The same model has to
do four things: answer science exam questions (ARC), answer exam questions of 57 school and university subjects
(MMLU), answer mathematics questions (MathBench), and write (answer a question from a passage, SQuAD 1.1; summarise a
news article, CNN/DailyMail). You choose the structure: what the weights are, how they are built from what you write,
and how the forward pass uses them. Nothing is trained and nothing pretrained is available: every word the model
knows and every number in it comes from what you write. The model must build and run here within {memory_gb} GB of
memory and {minutes} minutes on 4 CPU cores (the build and all {n_val} + {n_test} validation and test items together);
there is no other limit on its size.

Your score is the mean of four numbers on held-out items, which you cannot see:
- ARC: accuracy on {n_arc} four-option questions;
- MMLU: accuracy on {n_mmlu} four-option questions;
- MathBench: accuracy on {n_math} four-option questions;
- writing: the mean of the F1 on {n_squad} passage questions (word overlap between the model's answer and the
  reference answer, after lower-casing and removing punctuation and the articles a, an, the) and of ROUGE-L on
  {n_cnn} article summaries (the longest common subsequence of words between the model's summary and the reference
  summary, as an F1; reference summaries have about 55 words).

## One interface for every item

Every item reaches the model as one row of {x_len} input ids, and the model writes its answer as output ids, one per
step. Input ids: 0 is padding; 1 to 8 are the marks [CHOOSE], [A], [B], [C], [D], [ANSWER], [PASSAGE], [SUMMARIZE];
a token of `vocab.txt` has id 9 + its line (0-based); any other token has id 9 + V + crc32(token) % {hash_ids} (V =
lines of vocab.txt), the same wherever it occurs. The three kinds of rows are
- `[CHOOSE] question [A] option [B] option [C] option [D] option`: the answer is one output id, the mark of the right
  option (2 for A, 3 for B, 4 for C, 5 for D). Only the first output id is read.
- `[ANSWER] question [PASSAGE] passage`: the answer is a short phrase, at most {cap_answer} output ids.
- `[SUMMARIZE] article`: the answer is a summary, at most {cap_summary} output ids.

An output id is 0 (end of the answer), a mark, a vocabulary word (9 + its line), or a copy of an input position:
output id 9 + V + p writes the token at position p of the row. Copies of positions that follow one another are cut
from the original text in one piece, so "14 October 1066" or "U.S." come out as written there.

## What you deliver: the folder /srv/work/model

- `vocab.txt`: the tokens your model knows, one per line, lower-case.
- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
- `arch.py`: defines `forward(w, x, a)`. `w` is the dict from `build()`, `x` is int64 (n, {x_len}) input ids, `a`
  is int64 (n, {a_len}) output ids written so far (0 where nothing is written yet); it returns float
  (n, 9 + V + {x_len}) logits of the next output id. arch.py must be a fixed tensor program: imports only `torch`
  and `math`; no `if`, `while`, `and`/`or`, conditional expressions, `try`, `with`, `lambda` or classes; no
  `.item()`, `.tolist()`, `.numpy()`; none of the names {names}. `for` loops over a fixed range are allowed.
  Everything that depends on the input or the answer so far must happen in tensor operations on the weights.
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

`forward` is called once per step for the items that are still writing. The output id with the largest logit is
appended to the answer; id 0 ends it. Mind the time limit: a summary may take {cap_summary} steps.

Text becomes ids, your folder is run, and the output becomes an answer like this:

```python
ALL_TOKEN = re.compile(r"{token}")
X_LEN, A_LEN, HASH_IDS = {x_len}, {a_len}, {hash_ids}
MARKS = {marks}
N_MARKS = len(MARKS)
CAPS = {caps}
OPTION_WORDS, QUESTION_WORDS = {option_words}, {question_words}

{encode}
```

```python
{runner}
```

## Example items

Forty items with their answers: ten of ARC, ten of MMLU, ten of MathBench and ten of writing (five passage questions,
five summaries). These are the only items of these datasets you see; the test items are different.

{examples}

## Tools

`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_questions.json (see the tool's docstring) it runs the model on those items and prints what it writes;
they must be items you made up.

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No data of these datasets is available to you apart from the forty items above.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""


def render_all() -> str:
    from wsp import general
    n = {p: len(general.items("test", p)) for p in general.PARTS}
    encode = "\n\n\n".join(inspect.getsource(f).rstrip() for f in (free.all_tokens, free.all_source, free.all_encode,
                                                                    free.all_text))
    return ALL_TASK.format(memory_gb=free.ALL_TENSOR_GB, minutes=free.RUN_TIMEOUT_S // 60, n_val=len(general.load("validation")),
                           n_test=sum(n.values()), n_arc=n["arc"], n_mmlu=n["mmlu"], n_math=n["math"], n_squad=n["squad"],
                           n_cnn=n["cnn"], x_len=free.X_LEN, a_len=free.A_LEN, hash_ids=free.HASH_IDS,
                           cap_answer=free.CAPS["answer"], cap_summary=free.CAPS["summarize"],
                           names=", ".join(f"`{x}`" for x in sorted(free.ARCH_FORBIDDEN_NAMES)),
                           token=free.ALL_TOKEN.pattern, marks=free.MARKS, caps=free.CAPS,
                           option_words=free.OPTION_WORDS, question_words=free.QUESTION_WORDS, encode=encode,
                           runner=free.ALL_RUNNER.rstrip(), examples=general.examples())


TEXT_TASK = """# Task: design and write a language model

You will design and write a language model by hand. It reads a text and predicts what comes next, one piece at a time.
You choose the structure: what the weights are, how they are built from what you write, and how the forward pass uses
them. Nothing is trained and nothing pretrained is available: everything the model knows comes from what you write.

The model will be tested on tasks that are not disclosed to you. Each test gives the model a plain English text and
reads its continuation: sometimes only the first piece, sometimes up to {new_len} pieces or until the model writes a
line break. There are no example items. The model must build and run here within {memory_gb} GB of memory and
{minutes} minutes on 4 CPU cores (the build and about {n_test} texts with about {steps} continuation steps in all);
there is no other limit on its size.

## What you deliver: the folder /srv/work/model

- `vocab.txt`: the pieces your model adds to the base pieces, one per line. A piece is one or more of the characters
  of BASE_CHARS, optionally after one `{space}` mark, which stands for the space before a word. No piece twice and
  none that is a base piece.
- `build.py`: defines `build()`, which returns a dict of name -> torch tensor (dense or sparse) holding every
  weight. It runs with the folder as working directory and may read any file you put there, and may use any
  Python with numpy and torch. This is where what you wrote becomes weights; formulas and loops are fine.
- `arch.py`: defines `forward(w, x)`. `w` is the dict from `build()`, `x` is int64 (n, {ctx_len}) piece ids, the
  text so far, right-aligned and padded with 0 on the left; it returns float (n, V) logits of the next piece, with
  V = 1 + {n_base} + lines of vocab.txt. arch.py must be a fixed tensor program: imports only `torch` and `math`; no
  `if`, `while`, `and`/`or`, conditional expressions, `try`, `with`, `lambda` or classes; no `.item()`, `.tolist()`,
  `.numpy()`; none of the names {names}. `for` loops over a fixed range are allowed. Everything that depends on the
  text must happen in tensor operations on the weights.
- any other files that `build()` reads, and `NOTES.md`: what your model is and why.

Piece ids: 0 is padding; ids 1 to {n_base} are the base pieces (the line break, then every character of BASE_CHARS
without and with the `{space}` mark); line k (0-based) of vocab.txt has id {n_base} + 1 + k. Text is lower-cased
before it is cut into pieces, so the model reads and writes lower case only.

A text becomes piece ids, your folder is run, and a continuation becomes text like this:

```python
CTX_LEN, NEW_LEN = {ctx_len}, {new_len}
SPACE, BREAK = "\\u2581", "\\u21b5"
BASE_CHARS = {base_chars!r}
BASE_PIECES = (BREAK, *BASE_CHARS, *(SPACE + ch for ch in BASE_CHARS))

{encode}
```

```python
{runner}
```

In the runner, `cap` is the number of pieces a text may be continued by, and `first` (the first step's logits of
the base pieces) is one of the things a test may read.

## Tools

`python /srv/work/tools/check_model.py` checks the folder, builds the model and runs it. If you write
/srv/work/my_texts.json (a list of strings) it prints the model's continuation of each; they must be texts you
made up.

## Rules

- No training or fitting of any kind, and no automatic search over the model's content or structure from measured
  accuracy. Everything in the model must come from your own knowledge and design.
- No data is available to you.
- Do not access the internet or any files outside /srv/work and /srv/lib.

When you are done, make sure check_model.py passes, then reply with a short description of your model.
"""


def render_text() -> str:
    from wsp import textlm
    rows = textlm.load("test")
    encode = "\n\n\n".join(inspect.getsource(f).rstrip() for f in (free.piece_encode, free.piece_text))
    return TEXT_TASK.format(new_len=free.NEW_LEN, ctx_len=free.CTX_LEN, memory_gb=free.ALL_TENSOR_GB,
                            minutes=free.RUN_TIMEOUT_S // 60, n_test=f"{round(len(rows), -3):,}",
                            steps=f"{round(sum(textlm.cap(r) for r in rows), -4):,}", space=free.SPACE,
                            n_base=len(free.BASE_PIECES), base_chars=free.BASE_CHARS,
                            names=", ".join(f"`{x}`" for x in sorted(free.ARCH_FORBIDDEN_NAMES)), encode=encode,
                            runner=free.LM_RUNNER.rstrip())


def render_gen(arch: str) -> str:
    from wsp import given_archs, sciq
    if arch == "free-gen":
        head = dict(verb="design and write", verb_long="design and write all of the content of",
                    structure="You choose the structure: what the weights are, how they are built from what you write, "
                              "and how the forward pass uses them.",
                    arch_item=GEN_FREE_ARCH.format(q_len=free.GQ_LEN, a_len=free.GA_LEN,
                                                   names=", ".join(f"`{n}`" for n in sorted(free.ARCH_FORBIDDEN_NAMES))))
    else:
        head = dict(verb="write the weights of", verb_long="write every weight of",
                    structure=given_archs.structure(given_archs.GEN),
                    arch_item=ARC_FIXED_ARCH.format(shape_table=given_archs.shape_table(given_archs.GEN),
                                                    arch_source=given_archs.source(given_archs.GEN).rstrip()))
    encode = "\n\n\n".join(inspect.getsource(f).rstrip() for f in (free.words, free.gen_encode, free.gen_text))
    return GEN_TASK.format(memory_gb=free.TENSOR_GB, minutes=free.RUN_TIMEOUT_S // 60, n_test=len(sciq.load("test")[0]),
                           q_len=free.GQ_LEN, a_len=free.GA_LEN, token=free.TOKEN.pattern, encode=encode,
                           runner=free.GEN_RUNNER.rstrip(), examples=sciq.examples(), **head)


def render_squad(arch: str) -> str:
    from wsp import given_archs, squad
    first = arch.endswith("-sq1")
    template = SQUAD_TASK
    for old, new in SQUAD1_CHANGES if first else ():
        if template.count(old) != 1:
            raise ValueError(f"SQUAD_TASK does not have exactly one {old!r}")
        template = template.replace(old, new)
    if arch.startswith("free-"):
        head = dict(verb="design and write", verb_long="design and write all of the content of",
                    structure="You choose the structure: what the weights are, how they are built from what you write, "
                              "and how the forward pass uses them.",
                    arch_item=SQUAD_FREE_ARCH.format(q_len=free.SQ_LEN, c_len=free.SC_LEN,
                                                     names=", ".join(f"`{n}`" for n in sorted(free.ARCH_FORBIDDEN_NAMES))))
    else:
        head = dict(verb="write the weights of", verb_long="write every weight of",
                    structure=given_archs.structure(given_archs.SPAN),
                    arch_item=ARC_FIXED_ARCH.format(shape_table=given_archs.shape_table(given_archs.SPAN),
                                                    arch_source=given_archs.source(given_archs.SPAN).rstrip()))
    decode = free.span_decode_forced if first else free.span_decode
    version = "1.1" if first else "2.0"
    encode = "\n\n\n".join(inspect.getsource(f).rstrip() for f in (free.span_tokens, free.span_encode, decode))
    return template.format(memory_gb=free.TENSOR_GB, minutes=free.RUN_TIMEOUT_S // 60,
                           n_test=len(squad.load("test", version)[0]),
                             q_len=free.SQ_LEN, c_len=free.SC_LEN, hash_ids=free.HASH_IDS, max_span=free.MAX_SPAN,
                             token=free.SPAN_TOKEN.pattern, encode=encode, runner=free.SPAN_RUNNER.rstrip(),
                             examples=squad.examples(version), **head)


def render_design_cifar() -> str:
    from wsp import data
    classes = ", ".join(f"{i}: {c}" for i, c in enumerate(data.CLASSES))
    return DESIGN_CIFAR.format(memory_gb=free.TENSOR_GB, minutes=free.RUN_TIMEOUT_S // 60, classes=classes,
                               n_test=len(data.load("test")[1]), runner=free.IMAGE_RUNNER.rstrip(),
                               names=", ".join(f"`{n}`" for n in sorted(free.ARCH_FORBIDDEN_NAMES)))


def render_lm(arch: str) -> str:
    from baselines import arc_released
    from wsp import arc, lm_task
    m, start = lm_start.manifest(arch), lm_start.start_dir(arch)
    tok, mask = lm_task.tokenizer(start), lm_task.mask_of(m["name"])
    prompt = arc_released.PLAIN_TEMPLATE.format(question="<question>",
                                                **{l: f"<option {l}>" for l in arc.LETTERS})
    readout = LM_READOUT
    if mask is not None:
        prompt += tok.convert_ids_to_tokens(mask)
        readout = LM_MASKED_READOUT.format(mask=tok.convert_ids_to_tokens(mask))
    plain, special = tok("x", add_special_tokens=False)["input_ids"], tok("x")["input_ids"]
    at = next(i for i in range(len(special)) if special[i:i + len(plain)] == plain)
    before, after = special[:at], special[at + len(plain):]
    if after and mask is None:
        raise ValueError(f"{arch}: the tokenizer adds tokens after the text")
    ends = [f"{' '.join(tok.convert_ids_to_tokens(t))} at the {where}"
            for t, where in ((before, "start"), (after, "end")) if t]
    return LM_TASK.format(
        added="adds " + " and ".join(ends) if ends else "adds no tokens", readout=readout,
        repo=m["repo"], family=LM_FAMILY[m["family"]], n_params=m["parameters"],
        start=LM_START[m["start"]].format(seed=m["seed"]), prompt=prompt, n_test=len(arc.load("test")[2]),
        minutes=lm_task.EDIT_TIMEOUT_S // 60, memory_gib=lm_task.DATA_BYTES // 2 ** 30,
        weights_gb=(start / lm_start.WEIGHTS).stat().st_size / 1e9, arch_source=lm_start.arch_source(m["name"]).rstrip(),
        examples=arc.examples(), max_questions=lm_task.MAX_QUESTIONS)


def render_olmo(arch: str) -> str:
    from wsp import lm_task, olmo2_1b, olmo_start, olmo_task
    m, start = olmo_start.manifest(arch), olmo_start.start_dir(arch)
    t = OLMO_TEXT[arch]
    chat = olmo_task.CHAT_TEMPLATE.format(prompt="<instruction>") + "<answer><|endoftext|>\n"
    return OLMO_HEAD.format(
        title=t["title"], repo=m["repo"], n_params=m["parameters"],
        start=t["start"].format(revision=m["revision"], uniform=olmo_task.uniform_loss(), vocab=olmo2_1b.VOCAB),
        afterwards=t["afterwards"].format(chat=chat), minutes=lm_task.EDIT_TIMEOUT_S // 60,
        memory_gib=olmo_task.DATA_BYTES // 2 ** 30, weights_gb=(start / lm_start.WEIGHTS).stat().st_size / 1e9,
        arch_source=inspect.getsource(olmo2_1b).rstrip(), max_texts=olmo_task.MAX_TEXTS,
        max_prompts=olmo_task.MAX_PROMPTS, prompt_format=t["prompt_format"], data_name=t["data_name"])


def render(arch: str, data_access: str, forward_budget: int, parent: dict | None = None) -> str:
    from wsp import olmo_start
    if olmo_start.is_olmo(arch):
        if data_access != "blind" or forward_budget:
            raise ValueError("OLMo architectures run blind-K0 only")
        return render_olmo(arch)
    if lm_start.is_lm(arch):
        if data_access != "blind" or forward_budget:
            raise ValueError("released-LM architectures run blind-K0 only")
        return render_lm(arch)
    if arch in distill.ARCHS:
        if data_access != "blind" or forward_budget:
            raise ValueError("distill-arc runs blind-K0 only")
        return render_distill_arc()
    if arch.endswith("-arc"):
        if data_access != "blind" or forward_budget:
            raise ValueError("ARC architectures run blind-K0 only")
        return render_arc(arch, parent)
    if arch.endswith("-mb"):
        if data_access != "blind" or forward_budget:
            raise ValueError("MathBench architectures run blind-K0 only")
        return render_mathbench(arch)
    if arch.endswith("-mm"):
        if data_access != "blind" or forward_budget:
            raise ValueError("MMLU architectures run blind-K0 only")
        return render_mmlu_all(arch)
    if arch.endswith(("-sq", "-sq1")):
        if data_access != "blind" or forward_budget:
            raise ValueError("SQuAD architectures run blind-K0 only")
        return render_squad(arch)
    if arch.endswith("-gen"):
        if data_access != "blind" or forward_budget:
            raise ValueError("generation architectures run blind-K0 only")
        return render_gen(arch)
    if arch.endswith("-all"):
        if data_access != "blind" or forward_budget:
            raise ValueError("the general task runs blind-K0 only")
        return render_all()
    if arch.endswith("-text"):
        if data_access != "blind" or forward_budget:
            raise ValueError("the language-model task runs blind-K0 only")
        return render_text()
    if arch == "design-cifar":
        if data_access != "blind" or forward_budget:
            raise ValueError("design-cifar runs blind-K0 only")
        return render_design_cifar()
    if archs.ARCHS[arch].deliverable == free.DELIVERABLE:
        if data_access != "blind" or forward_budget:
            raise ValueError("free-design architectures run blind-K0 only")
        return render_free(arch, parent)
    if archs.ARCHS[arch].deliverable == mem.DELIVERABLE:
        if data_access != "blind" or forward_budget:
            raise ValueError("memory architectures run blind-K0 only")
        return render_memory(arch)
    t = archs.task_of(arch)
    header, rule_open, rule_view, rule_blind = TEXTS[t.name]
    classes = ", ".join(f"{i}: {c}" for i, c in enumerate(t.classes))
    rules = RULES_COMMON + {"open": rule_open, "view": rule_view, "blind": rule_blind}[data_access]
    if data_access in ("open", "view"):
        rules += (RULE_FORWARD.format(budget=forward_budget, plateau=PLATEAU_CALLS, items=t.item)
                  if forward_budget > 0 else RULE_NO_FORWARD.format(items=t.item))
    if data_access in ("open", "view"):
        rules += ARCH_RULES.get(arch, "")
    return (header.format(classes=classes) + arch_section(arch) + DELIVERABLE + rules + FOOTER)

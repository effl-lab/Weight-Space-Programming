# A hand-written association scorer for ARC

Nothing in this model is trained or fitted. Every number comes from `build.py`, which compiles the
hand-written science facts in `knowledge/*.txt` into weights by fixed formulas. `make_vocab.py` writes
`vocab.txt` from the same files; `common.py` holds the tokenizer, a small hand-written stemmer, and the stopword list.

## Knowledge

`knowledge/*.txt` holds about 5,100 lines written by hand (32 files). Each line is one fact cluster: a set of words that
belong together, e.g. `cumulonimbus clouds tall thunderstorms lightning heavy rain hail` or
`ovary flower develops into fruit`. The topics cover the grade-school curriculum: plants, animals, ecology,
cells and genetics, the human body, rocks and plate tectonics, weather and the water cycle, astronomy, matter
and chemistry, energy, forces, electricity and magnetism, waves, inquiry and safety, resources, technology,
New York State earth science, a glossary of science terms, and typical everyday scenarios.

## Word codes (the embedding)

Each word is stemmed, and every stem gets a 204-dim code. The code is the sum of one fixed vector for each fact
line containing the stem, normalised to length 1. So the dot product of two codes is roughly the cosine of
their line co-membership: high for words that share facts, near 0 otherwise. I also tried adding weaker
paragraph-level vectors. They did not help the exact scorer and they added crosstalk, so they are off
(`PARA_W = 0`).

The vectors are pseudo-random with fixed seeds. They are also structured, so that the facts within one topic
do not blur into each other:
* Lines are grouped into topic chunks of 160 consecutive lines of a file. Inside a chunk, every line vector is a
  distinct row of a random orthonormal basis (QR of a Gaussian matrix seeded by the file name and chunk index,
  so adding a file does not change the others). Facts that
  compete as distractors are usually in the same topic, and there they are exactly orthogonal. Vectors from
  different chunks overlap only randomly.
* The code is split into 4 blocks of 51 dims, one per attention head. All the lines of a paragraph (10
  consecutive lines) go to the same block, so the facts two related words share tend to meet in the same head.

The vocabulary holds every knowledge word with its inflections, possessives ("earth's") and common
derivations ("rapidly"). A form whose own stem has no code falls back to the stem left after removing a
derivational suffix. Prefixes are never removed, because they can reverse the meaning ("nonrenewable").

Stopwords (function words and question boilerplate such as "which", "best", "student", "describes") and unknown
words get no code. They get only a STOP flag.

## The network (arch.py, unchanged)

Residual layout: dims 0–203 hold the code. Then come flags (CONTENT, STOP, ISQ / ISO for question or option
position, from the position embedding), a constant pair +L / −L (L = 30), and a compensation dim. The
compensation dim makes every row sum to zero, so LayerNorm (weight 1, bias 0) acts as a near-constant scaling
that the reading weights undo. The rest are value dims M, S and ECHO, which later layers write.

**Block 0, attention: associations.** In each head, an option token attends to the question tokens. The logit is
β·(code·code within the head's block − θ) for a question content word and τ for a question stopword (the
stopwords are the attention sink). Option tokens are masked with −1000 through the ISO flag. The value is 1 for
content words, so M (summed over the 4 heads) is the attention mass that lands on question words associated
with this option word. β = 15, θ = 0.1, τ = 1.5.

**Block 0, MLP: gating.** Pairs of GELU units (GELU(x + c) − GELU(c), which is linear for large c) copy M − λ
(λ = 0.1) into the score dim S. A unit switches off unless the token is an option content word, so question
tokens and option stopwords score 0.

**Block 1, attention: echo and negation detectors.** Distractors in ARC often repeat words of the question,
and an exact repeat would otherwise give the strongest possible match. Heads 0–2 each compare a fixed random
53-dim projection of the full code. A word and itself score about 1; any other pair scores about 0 ± 0.14. With a
threshold of 0.65 and a sharp slope, ECHO counts how many of the 3 detectors call the option word a repeat of a
question word. Head 3 puts its attention on any question word flagged as a negation ("except", "least") and
reads the flag, so NEGF ≈ 1 in a negated question.

**Block 1, MLP: echo removal and negation.** Three pairs of gated GELU units combine these. A repeated word
(ECHO ≥ 1.5) contributes nothing. In a negated question every other option word's S changes sign, so the
least associated option wins. "not" is deliberately not a negation word: in ARC it is used descriptively ("a
plant that does not get light") about as often as it negates the question.

**Output.** ln_f and `score` read only S. The logit is the mean over all real tokens of S, so an option's score is
the sum of its content words' association mass divided by the sequence length (question tokens contribute 0).

## Why this design

* A first-order co-occurrence association between question and option words is the classic baseline for
  science multiple choice. It needs a word × word table, and that table only fits through the 216-dim attention
  bilinear form as a compressed code. Random codes give crosstalk of about 1/√51 per head. Making the codes
  orthogonal within a topic, and keeping a paragraph's facts in one head, removes most of the crosstalk that
  matters.
* I checked every design decision against an exact (uncompressed) version of the same scorer, run on questions I
  wrote myself (`/srv/work/dev`, not part of the model). Exact-match removal helped by about 3 points there. Single
  runs vary by about ±1.5 points with the random seed of the bases, so I compared knowledge changes as the
  average over 3 seeds.
  Position weighting, IDF weighting and a length-independent sink did not help, so they are not in the model.
* No measured accuracy was used to search over the knowledge or structure automatically. The few constants
  (β, θ, τ, λ) were set by hand after looking at my own made-up questions.

## Results on my own questions

On 317 questions I wrote myself (dev set, not shipped), the model scores about 0.63–0.65 (chance is 0.25). The
exact uncompressed scorer scores about 0.70, so compression into 204 dims costs about 5 points. The real ARC test
is harder and much broader than my questions, so I expect a clearly lower score there.

## Known limits

It is a bag-of-words association model. It cannot do arithmetic. Negation is handled only for "except" and
"least", so "NOT" questions still get the most associated option. It cannot tell "increase" from "decrease" when both co-occur with the same words, and
it knows only what the knowledge files say.

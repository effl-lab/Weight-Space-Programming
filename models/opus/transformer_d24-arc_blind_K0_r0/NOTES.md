# A hand-written concept-matching scorer for ARC

Nothing here is trained. Every weight is computed in `build.py` from formulas and from the knowledge I wrote in
`knowledge/*.txt` (2,149 concept lines, 69 of them category lists, over about 4,000 written words;
27,000 vocabulary entries with inflections). The architecture (`arch.py`) is fixed;
the design below is how that fixed two-layer, 24-dim, 4-head transformer is made to compute a useful score.

## What the model computes

For each option, the logit is (up to a positive factor that is the same for the four options of a question)

    sum over content words t of the option of   (1 - 2n) * ( sum over heads h of  S_h(t) + eps * (1 - S_h(t)) )

* `S_h(t)` is the share of attention that option word `t` gives to question content words in head `h`. A word
  holds one 5-dim (head D: 4-dim) unit "concept code" per head and attention logits are 20 x (dot product of
  codes), so attention lands on question words holding exactly the same code as `t`; the word itself (and the
  other option words) act as the sink. `S_h` is therefore about m/(m+1) when m question words share a concept
  with `t` in head `h`, and about 0 otherwise.
* `n` is about 1 when the question contains a negation word (EXCEPT, NOT, LEAST, ...), which flips the ranking.
* `eps` is a very small credit per known content word (a tie breaker that favours options with more science words).

## Knowledge and how it becomes codes

`knowledge/*.txt`: one concept per line, e.g. `roots function: roots root absorb absorbs water minerals ...`.
Words written on a line are linked. Category lines `key words | members` link the keys with every member but not
the members with each other (e.g. `mammal mammals | whale dog bat ...`). Generic words (`which`, `make`,
`measure`, ...) are dropped, and regular inflections, singulars and possessives inherit their base word's codes.

Each word has only four slots (one code per head) and two words are linked in a head only if they hold the same
code there. `optimize_slots` decides which code each word holds in each head: candidates are the lines a word is
on and "stars" (a frequent word plus all words it shares a line with); every candidate has a random code in every
head. A deterministic coordinate ascent maximizes the number of written links realized (with diminishing credit
for the same pair matching in several heads) minus a penalty for pairs that would be linked without ever having
been written together. This is a compiler step over my own knowledge files only; it never sees any question.

## How the fixed transformer is made to do this

Residual stream (23 logical dims, all orthogonal to the all-ones vector so LayerNorm is a pure scaling): 19 code
dims (heads A, B, C: 5, head D: 4), G (side + word class), N (norm compensation), X (negation), S (score).

* Every token has exactly the same norm at the input (stop words and negation words carry offsets in G, N, X that
  compensate for their missing codes), so LayerNorm divides everything by one known constant.
* Layer 0, all four heads: 5 query/key dims match codes, the 6th dim routes. Its query term is zero for option
  content words and strongly negative otherwise; its key term is lowest on question stop words. So question
  tokens, option stop words and negation words all attend only to question stop words, whose value is 0: only
  option content words get a non-zero score, and question tokens are identical across the four options. Values
  are an affine function of (G, N): 1 on question content words, 0 on question stop words, eps on option words.
* Layer 1, head 0: attends to negation words (key from X, with a strong preference for the question side);
  value n is written into X. MLP 1 multiplies, S <- S (1 - 2n), with gelu(z) + gelu(-z) ~ 0.8 z^2.
* Length normalization: arch.py averages over all Lq + Lo tokens, which would divide each option's evidence by its
  length and let any match shared by all options become a length bias. Layer 1, head 1 attends uniformly and
  writes u = Lq / (Lq + Lo) (value = 1 on question tokens). MLP 1 turns u into a huge component (about 2e4 u) in
  its own two float32 coordinates, which dominates the norm, so the final LayerNorm divides by u and the mean
  over tokens becomes (sum over option tokens) / Lq: a plain sum over option words.
* The orthonormal basis keeps the score and the huge component on separate coordinates so float32 rounding of
  the latter never touches the score.

## Why

Without training, the only knowledge source is what I can write down. Grade-school science questions are often
answerable by "which option is most strongly associated with the question's key terms" (the classic IR / PMI
baselines on ARC-Easy), so the model is an associative matcher over hand-written concept groups, with sharp
(identity-like) matching to keep random collisions of codes rare, saturation per head, a sign flip for negated
questions, and an exact sum over option words.

Known limitations: bag of words (no word order, so "A between B and C" equals "C between B and A"); each word
can hold only four concepts, so very frequent words (water, energy, heat) keep only a few of their links; on
questions whose distractors are as associated with the question as the answer (much of ARC-Challenge) the
association signal is near chance or worse.

## Design choices and how they were checked

I wrote ~300 ARC-style questions of my own (not ARC data) and used them only to debug the circuit and to compare
a few structural options, averaging over six seeds of the random codes because a single seed varies by +-5%:
summing over option words instead of the architecture's length-dependent mean (+5 to 6 points), the code
assignment optimizer (vs. a heuristic), splitting lines that mix both halves of a contrast ("like poles repel /
opposite poles attract"), category lines, and dropping generic words such as "measure". Constants such as the
match sharpness (BETA), the per-word credit (EPS_CONTENT) and the pair penalty were flat within noise. The
model scores about 62-64% on those questions; they are easier than ARC and were written by the same person as
the knowledge, so the ARC test accuracy will be clearly lower.

## Files

* `build.py`: the circuit (`build()`), the knowledge compiler (`read_concepts`, `variants`, `optimize_slots`) and
  `python build.py`, which rewrites `vocab.txt` from the knowledge files. `build()` refuses a stale vocab.txt.
* `knowledge/*.txt`: the hand-written concept lines.
* `vocab.txt`: every word that has codes (written words, their inflections, negation words).

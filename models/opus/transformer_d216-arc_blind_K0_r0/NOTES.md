# A hand-written ARC scorer

Nothing in this model is trained or fitted. Every number comes from `build.py`, which compiles a knowledge base I
wrote by hand (`kb/*.txt`, about 3,700 short science facts in 270 topics) plus word lists (`stop.txt`, `neg.txt`,
`irregular.txt`, `adjectives.txt`, `nostem.txt`) into the weights of the fixed 2-layer, d=216 transformer in
`arch.py`. The only randomness is a fixed seed used to draw code vectors.

## What the network computes

The score of an option is an IDF-weighted "how well do the option's words cover the question's words":

    score = (1/N) * s * sum over question content words j of  w_j * (c_j - rho_j * [the option repeats j])
    c_j   = sum over 4 code blocks h of  A_jh / (A_jh + e^SINK),   A_jh = sum over option content words o of exp(LAMBDA u_jh . u_oh)

* `u` is a word's code (180 dims = 4 blocks of 45, unit length), so `u_j . u_o` is a compressed version of "how
  many facts the two words share" (cosine of their fact-incidence vectors).
* Each block saturates (`A / (A + e^SINK)`), so one strong match counts about as much as it can, and many weak
  matches cannot add up to a strong one.
* `w_j = 1/(1 + number of facts with j)`: rare, specific question words matter, hub words ("water", "energy") hardly.
* An option word that literally repeats a question word gets no credit for that (distractors often reuse question
  words): `rho_j` is exactly what the repeat itself contributes, and it is subtracted when a hash check sees the repeat.
* `s = -1` when the question contains not / except / least / never / cannot (and contractions): then the least
  associated option wins.
* `N` (question + option length) is the architecture's mean pooling; it is the same up to the option length.

On my own made-up practice questions (236 questions in four sets, written by me in ARC style) this gets about 60-63%;
real ARC questions, especially the Challenge set (selected against word-association solvers), will be harder.

## How the weights implement it

Residual stream layout (216 dims): flags (question position, option position, stop word, negation word), two
position-matching dims, per-block self-similarity terms, output slots, 180 code dims, a 15-dim word-identity hash,
and two dims (BAL, FIL) that give every layer-0 input row exactly zero mean and the same norm. With that, every
LayerNorm is just a fixed scaling, so the attention scores are exactly the bilinear forms written below. Weights
are computed in float64 and returned as float32 (peak memory about 1.05 GB, 3,530 questions in about a minute);
position flags and position vectors are stored at scale 1000 so that float32 LayerNorm rounding is not amplified
by the large self-sink coefficients. The float32 scores match a numpy reference of the formula to about 2e-4.

* **Layer 0, 4 heads (one per code block).** Each question content word queries every token: option content words
  score `LAMBDA u_j.u_o`; its own position scores `SINK` (two position dims on a circle with a large scale make
  "same position" stand out by at least 60, and a per-word term `-LAMBDA |u_jh|^2` cancels the word's similarity with
  itself); other question words are pushed far below by the position dims; stop words are masked. The value is 1 on
  option tokens, so the head outputs `A/(A+e^SINK)`; the four heads add into one slot `R` = `c_j`.
* **Layer 1, head 0** broadcasts "the question has a negation word" (negation keys score +30).
  **Head 1** is the identity check: a question word attends to option words with score `90 (h_j.h_o - 0.9)` on
  random unit hashes `h` (shared by all forms of a stem) against a self sink at 0, giving `ID ~ 1` iff the option
  repeats the word.
* **Layer 1 MLP** computes, on question content words only, `F = s (R - rho [ID])` with GELU gates
  (`gelu(x) - gelu(-x) = x`, and an offset of -200 switches a pair off), and inflates the token's norm by
  `1/w_j` (two spare dims). The final LayerNorm divides by that norm, so `score . ln_f(h)` = `w_j F`; the mean
  pooling then sums over the question words. Layer 0's MLP is unused (zero).

## Knowledge compilation

* `kb/*.txt`: one fact per line; `#` starts a topic; `= a b c` merges words into one (e.g. evaporate/evaporation).
  A fact's words are its content words (stop words removed). In a fact, words after a negation (`not`, `no`,
  `without`, ...) in the same clause are dropped, so "spiders are arachnids not insects" does not link spider-insect.
  Generic category words of questions (tool, organism, property, process, ...) are stop words: all four options
  usually belong to the category, so they only add noise.
* Words are reduced to stems by a small hand-written stemmer (plural, -ed, -ing, comparatives/-ly for listed
  adjectives, irregular forms); `vocab.txt` lists every inflected form of every stem, plus stop and negation words.
  Unknown words count as stop words.
* Codes: every topic is placed in one of the 4 blocks (greedy, so that topics sharing vocabulary land in different
  blocks); within its block a topic gets an orthonormal basis, row 0 being the topic direction and the other rows
  its facts' codes (fact code = 0.45 topic + 0.89 own row). A word's code is the normalized sum of its facts'
  codes. So facts of one topic never interfere; interference only comes from different topics in the same block.

## Design choices and what I checked

All choices were made by reasoning, then checked with a handful of comparisons on my own practice questions and
with a numpy reference implementation (`score = network` to 1e-4). No automatic search was run. Things that
mattered: question-centric scoring (+5 points over the option-centric version I built first), discounting repeated
words (+4), spreading each word's facts over the blocks with orthonormal codes inside topics (+3 over topic-random
codes), IDF weighting of question words (+2), category words as stop words (+1). Compressing the KB into 45-dim
blocks costs about 5 points compared with the same formula on 400-dim codes.

Limitations: it is a bag-of-words associator. It cannot compare quantities, follow multi-step reasoning, or tell
cause from effect; on ARC-Challenge, which was built from questions that word co-occurrence solvers get wrong,
it will be near chance.

## Files

`build.py` (all weight construction; `python build.py --write-vocab` regenerates `vocab.txt`, and `build()` checks
that `vocab.txt` matches), `kb/*.txt` (the knowledge), `stop.txt`, `neg.txt`, `irregular.txt`, `adjectives.txt`,
`nostem.txt` (word lists), `vocab.txt`, `arch.py` (given, unchanged).

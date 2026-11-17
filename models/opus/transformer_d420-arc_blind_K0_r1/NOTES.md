# A hand-written association scorer for ARC

Nothing is trained. Every number comes from text I wrote (`facts/*.txt`, `isa.txt`, `priors.txt`, the word lists in
`lexicon.py`) turned into weights by formulas in `build.py`.

## Idea

An option is good when its words are strongly associated, in a body of science facts, with the question's words,
the way an IR solver would look the answer up in a textbook. The "textbook" is ~3,900 short facts I wrote,
one per line, covering the grade-school curriculum: space, weather, geology, cells, plants, animals, ecology, the
human body, heredity, matter, forces, energy, electricity and magnetism, waves, inquiry and safety, environment,
technology, and everyday science. Some files are written as terse cue lines ("glacier erosion carves valley"),
which strengthen the pairs that questions actually ask about. All content words of a fact count as associated.

* `lexicon.py`: stop words; a small consistent stemmer (inflections share a stem); derivational families that share
  one concept (pollen/pollinate/pollination, sun/solar, magnet/magnetic, …); the negation words
  (not, except, least, n't forms).
* `isa.txt`: "frog : amphibian", "granite : igneous rock", "toaster : electrical heat". A word inherits half
  of its categories' associations, so specific option words with few facts still know what they are.
* `priors.txt`: one mild prior (options saying "only" are usually distractors).

## How the transformer computes it

Word code: every fact gets a unit vector in a 312-dim space, and a concept's code is the sum of its facts' vectors. The
facts are greedily grouped by shared (idf-weighted) words into blocks of ≤311, and the vectors within a block are exactly
orthogonal (QR of a random zero-sum matrix), so related facts do not interfere. The dot product of two codes is then
(number of shared facts) plus projection noise. The association between option word o and question word q is

    A(o, q) = shared-fact weight / (n_o^0.5 · n_q^0.75)          (n = a word's fact mass)

The option side uses 0.5, so every option word has the same noise variance. The question side uses 0.75, an
idf-like weight, so generic question words count less.

* **Layer 1**, heads 0-2 (104 code dims each): every option token attends to the question tokens with logit
  4·A(o, q). Question function words are sinks with logit 0 and value 0. The output, a soft sum of the option
  word's associations with the question, is written to a scalar M. The question-side weight n_q^-0.75 is
  per word, which a bilinear form cannot do on its own. So question positions carry no anchor in layer 1, LayerNorm
  scales each question token by 1/|embedding|, and a per-word "filler" component sets that norm. Option positions
  carry a large constant anchor, so their LayerNorm is a fixed scaling. The layer-1 MLP then adds the same anchor to
  question tokens, so from layer 2 on every LayerNorm is a fixed scaling.
* **Layer 1**, head 3: a 40-dim random identity code per stem detects option words that repeat a question word
  (flag X). Repeated words are excluded from the option's score. Distractors often reuse question words, and
  excluding them was clearly better on my test questions (163 → 175 of 240).
* **Layer 2**: head 0 is the mean of M over the option's content words (repeats excluded; the option's function
  words carry the exact neutral baseline as a fallback). Head 1 is a soft max of M. Head 2 detects a negation word
  in the question. Head 3 computes the question's baseline (share of content words), which is subtracted. The MLP
  computes −2·N·score with a ReLU-style GELU gate, so for NOT/EXCEPT/LEAST questions the ranking is reversed.
* Every token ends up holding the same option score, so the final mean pooling returns it. `score` reads it, plus
  the prior feature.

## What I checked

Only on questions I made up (240 across five sets, written before or alongside the facts), averaged over 4 random
seeds of the projection: about 73% correct. These are easier than ARC. I expect real ARC to be clearly lower:
the Challenge set was built to defeat exactly this kind of word-association method. Things that did **not** help
and were removed: soft-OR with a threshold, cosine normalization on both sides, explicit contrast facts
("renewable != coal"), and an idf weight on option words. Known blind spots: word order and roles ("light is faster
than sound" and "sound is faster than light" score the same), arithmetic, multi-step reasoning, and fine
distinctions between two on-topic terms when the discriminating cue is a very common word.

`vocab.txt` is generated from the fact words (every inflected form whose stem has a code), plus the stop, negation
and prior words.

# A hand-written associative scorer for ARC in a d=8 transformer

Nothing here is trained. The model is a word-association matcher: an answer option scores well when its words
belong to the same hand-written *concept* as words of the question (e.g. "anemometer" and "wind", "chloroplast" and
"sunlight", "igneous" and "magma"). Every number comes from the lexicon files and the constants in `build.py`.

## Files

- `lex/*.txt`: the knowledge, about 690 concept lines written by hand, grouped by topic (measurement, science
  practice, matter, chemistry, energy, forces, electricity, waves, earth materials and processes, weather, astronomy,
  cells, plants, ecology, animals, adaptation, heredity, human body, health, resources and environment).
  One line is one concept: its first word heads the line (or `@label` names it). `word?` marks a weak member that only
  takes a free slot, and `word!` turns off inflection. Hyphenated entries are split the way the tokenizer splits them.
- `hubs.txt`: pins frequent words ("sun", "food", "heat", "oxygen", ...) to exactly the two concepts they should cue.
- `stop.txt`: words that are never concept members (function words and words too generic to discriminate).
- `neg.txt`: negation words (`not`, `except`, `least`).
- `build.py`: expands every word into its inflections (plural, -ed, -ing, possessive, singular), resolves
  memberships, writes `vocab.txt` (run `python build.py`), and builds all weights in float64.
- `arch.py`: the given architecture, unchanged.

## How the weights compute the score

Residual basis: seven orthonormal directions u0..u6 orthogonal to the all-ones vector, which LayerNorm removes.

- **Words.** Each content word has two *cluster codes*, unit vectors at an angle in the planes (u2,u3) (head A) and
  (u4,u5) (head B). Every concept gets an angle on a circle and sits in one of the two heads. A greedy max-cut gives
  each word at most one concept per head, with pinned words treated as hard constraints. A word with only one concept
  gets a *private* angle, shared only with its own inflections, in its other head.
  Function and unknown words are a huge vector on u0 (1e9). LayerNorm then makes them a pure u0 vector, which reads
  out as 0 and gives no attention keys.
- **Position.** It adds +1000·u0 for question positions and −1000·u0 for option positions. All content tokens
  therefore have exactly the same norm, which is what makes the very sharp matching exact.
- **Block 0, heads A and B.** Q = K = (scaled) cluster code. Logits are about 1e8, so attention is a hard match
  within an angle: each option token attends to its own position and to every token sharing its angle. The value is
  1 for question tokens and 0 for option tokens, read from the sign of u0. So head A outputs
  a = (#matching question tokens)/(#matching tokens, self included), and head B outputs b likewise.
  a is written to u1 and b to u6.
- **Block 0, MLP.** Three GELU units, used in their linear regime, compute u1 = a + b − 1.5·min(a, b) and clear u6.
  Because of the private codes, matching in both heads almost always means the option just repeats a question word,
  so a repetition counts half of a real association.
- **Block 1, head 2 (negation).** A constant query looks for a key on u6, which only negation words carry, and adds
  10·u1 when the question contains "not", "except" or "least".
- **Block 1, MLP.** Unit 1 maps y = 10 + m to −m, so on negated questions the option with the fewest associations
  wins. Unit 0 adds a 1e10 anchor to question tokens, so only option tokens are read out.
- **Readout.** The final LayerNorm and `score` (along u1) give each option content token its score m, and the mean
  over tokens gives the logit: the sum of the option words' match scores divided by the sequence length.

All arithmetic is float64: a perfect match is about 1e8 logits and neighbouring angles are separated by 25 nats.

## Why this design

With d = 8 there are only 7 usable directions, and all of them are taken: anchor/flag, readout, two 2-d codes, and
one spare that carries head B and the negation key. So each word gets two concept slots. The lexicon is organised
so that cue words point at the answers they cue. Experiments that did not help on my own questions were dropped:
soft "neighbouring concept" heads, an exact sum over option words (implemented with a 1/N head; no gain once ties were
counted fairly), IDF-like weighting (not separable from the option side), and mirroring single-concept words into
both heads.

## Expected behaviour

On about 400 grade-school questions I wrote myself, about 70% are correct. When the model is decisive it is right
85–90% of the time, but about 40% of my questions end near a tie. Real ARC, especially the Challenge set (filtered
against word co-occurrence methods), is harder and has a broader vocabulary, so expect much less there.

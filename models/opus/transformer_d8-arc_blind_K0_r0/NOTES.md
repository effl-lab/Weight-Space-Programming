# A hand-written ARC scorer in a d=8 transformer

Nothing here is trained or fitted. All knowledge is in `knowledge/` (written by hand); `kb.py` compiles it into
a vocabulary and per-word "slots"; `build.py` turns slots into float64 weights by formula.

## Idea

An option is good when its words are *associated* with the question's words. Associations are hand-written
clusters (`knowledge/clusters_*.txt`, ~1200 lines like
`barometer_def: barometer, measures air pressure, atmospheric pressure, falling pressure, storms coming, ...`).
Two words are associated when they share a cluster. The model counts, for every option word, on how many of
three "circles" it shares a cluster with some question word, maps that count k to a score PHI[k], and sums.

## How the weights do it

The residual stream has 8 dims; LayerNorm removes the all-ones direction, leaving 7 orthonormal directions:
six form three 2-d **circles**, one is the **flag**.

* **Embedding.** Every content word (a lemma group: inflections, derivations and listed synonyms share a row)
  has one slot angle on each circle; a cluster occupies one slot on one circle and its members share it. A
  word is in at most 3 clusters (one per circle); words in no cluster on a circle get a private slot there.
  Slots lie on an arc of each circle; stopwords / unknown words are a huge (1e6) vector at the opposite angle,
  so after LayerNorm they never match and their final readout is ~0.
* **Positions.** Question positions add `-G` to the flag, option positions `+G_o`, with `G_o` a hair larger, so
  after LayerNorm a question word's key is very slightly longer (ratio 1+rho) than an option word's own key.
* **Block 0 attention (3 heads, one per circle)** with q = k = the circle coordinates times a huge scale
  (weights are float64). An option word attends to itself unless a question word sits on *exactly* the same
  slot (then the longer question key wins by 30 nats); neighbouring slots lose by 30 nats. The value is a
  linear function of the flag (question words carry one value, option words another), so each matched circle
  adds `D_HEAD` to the option word's flag; question words end at a fixed level `-QB`, unmatched option words at 0.
* **Block 0 MLP** maps the discrete flag levels k*D (k matched circles) to PHI[k]*D with exact steps (pairs of
  steep GELU ramps). PHI = (0, 1, 1.5, 0): k = 3 almost always means the *same word* occurs in the question,
  which is not evidence (ARC distractors often echo the question), so it scores 0.
* **Block 1 attention (negation).** Negation words (not, except, least, cannot, avoid, ...) are stopword-like
  vectors with an extra component on circle 1. A head with a constant query finds them and adds a large offset
  to every token's flag; **block 1 MLP** then maps option scores s -> -s (exact steps), so a NOT/EXCEPT
  question prefers the *least* associated option.
* **Readout.** `score` reads the flag after the final LayerNorm; the logit is the mean over the sequence, i.e.
  sum of option-word scores divided by length. The small negative level `-QB` of question words makes longer
  options win when the match evidence is equal (mild length prior / tie-breaker).

## Knowledge compilation (kb.py)

* `stop.txt`: function words and generic question words (never matched).
* `lemmas.txt`: words that are one concept; regular inflections, possessives, singulars and safe derivations
  (-ly, -ness, -ity, -ation, ...) are generated automatically.
* `clusters_*.txt`: one cluster per line; phrases contribute all their words; terms containing a negation are
  skipped (so "no definite shape" does not tie "shape" to gas); clusters with the same head term are merged.
* `hubs.txt`: for ~250 over-used words (water, energy, heat, plant, ...) the 3 clusters they keep, chosen by
  hand. Other words pick up to 3 clusters greedily to cover the most partner words.
* Clusters are coloured onto the 3 circles so that no word needs two slots on one circle (weighted conflicts,
  local search + simulated annealing; 6 fixed seeds, the lowest-cost colouring is kept). The result is cached in
  `colour_cache.json`, keyed by a hash of kb.py and the knowledge files (recomputed if they change, ~45 s).
* Numbers that appear in clusters (0, 32, 100, 212, 24, 365, ...) are content words; other numbers are unknown.
* Generic category words of questions ("instrument", "tool", "property", "process") are stopwords: every option
  is usually such a thing, so they are no evidence.

`python kb.py` rewrites `vocab.txt`; `build.py` recompiles the knowledge and asserts it matches.

## Known limits

Bag-of-words association cannot tell "vapor into liquid" from "liquid into vapor", does not reason, and only
has 3 cluster memberships per word, so hub words lose some links. ARC-Challenge questions were selected to
defeat this kind of association, so most of the accuracy comes from the Easy set.

# A hand-written association scorer for ARC

Nothing here is trained or fitted. The knowledge is about 2,600 lines of science facts that I wrote in
`knowledge/*.txt`, plus the word lists `stopwords.txt`, `generic.txt`, `negation.txt` and `irregular.txt`.
`build.py` turns those files into the weights of the fixed transformer in `arch.py` with explicit formulas.
`make_vocab.py` (run by hand) writes `vocab.txt`: every knowledge word, its regular inflections, and the
function words.

## What the model computes

For a question Q and an option O:

    score(O) = sign(Q) * sum_o b_o * 2 C(o) . Qvec,     Qvec = sum_q a_q C(q)

- `C(w)` is the knowledge code of a word's stem (`kb.stem`: light suffix stripping plus the irregular table).
  Each fact line gets a random unit vector in 315 dims (seeded, so it is reproducible). A word's code is the
  sum of the vectors of the lines it appears in, scaled to unit length (cosine). So `C(a).C(b)` approximates
  "how much of a's and b's knowledge they share". A line can contrast groups with `~`: the groups get mutually
  opposed vectors (a centred simplex), so for example "rises east ~ sets west" makes *rise* favour *east* and
  count against *west*.
- `a_q`: weights of the question words. Stopwords get ~0. A word that appears in n lines gets
  1/sqrt(1 + n/10), and generic words get a quarter of that. Weights also grow by e^0.05 per position, because
  the question proper usually comes last.
- `b_o`: the same weights for option words, normalised over the option (a weighted mean). An option word that
  also occurs in the question is weighted down by e^-10: repeating the question tells nothing about which
  option is right. Unknown words get a small weight.
- `sign(Q)` is -1 when the question contains *except*, *least* or *not*.

## How the transformer does it (residual dims: 315 code + 17 control)

- **Embedding**: code, log question weight, log option weight, a 3-d hash `z` of the stem and |z|^2, a negation
  flag, |C|^2, and a pair of +-3000 anchors. The anchors make every token's LayerNorm scale identical, so with
  LayerNorm gains set to that scale each LayerNorm acts as the identity. Every written vector is balanced to
  zero sum through one spare dim, so the LayerNorm mean is zero. The position embedding marks option positions
  and adds the 0.05/position ramp.
- **Layer 0, heads 0-2**: the query is constant and the key is log weight minus 50 on option positions. Each
  head copies 105 code dims of the question tokens, so every token receives Qvec, added onto its own code.
- **Layer 0, head 3**: exact match. The logit is -1e4 |z_i - z_j|^2 (written as a bilinear form plus a key
  term) and the value is "is a question token". So the output is the share of same-stem tokens that sit in the
  question.
- **MLP 0**: gelu(t) + gelu(-t) ~ sqrt(2/pi) t^2 squares each code dim. The sum is |C(o) + Qvec|^2. Minus |C(o)|^2
  (gelu(t) - gelu(-t) = t passes that through exactly), this gives 2 C(o).Qvec + |Qvec|^2. The last term is the
  same for all four options.
- **Layer 1, head 0**: constant query. Option tokens are keyed by log option weight minus 10 x exact match
  (question tokens get -50). The value is that score, so every token gets the option's weighted mean.
- **Layer 1, head 1**: is any question word a negation word (key 25)?
- **MLP 1**: F -= 2 F NEGF, using gelu(KF + KB(NEGF-1)) - gelu(-KF + KB(NEGF-1)).
- **ln_f / score** read that one dim. It has the same value at every token, so the final mean over tokens returns it.

`/srv/work/scripts/verify.py` checks the transformer against a direct numpy computation of the formula above.
They agree to about 5e-4. An earlier 1% disagreement came from LayerNorm rounding times a 1e4 masking constant;
the masks are now 50.

## Why this shape (checked on my own made-up questions, not ARC)

- Random line codes in 315 dims leak crosstalk. Simulations with exact (orthogonal) codes scored about 10
  points higher, so crosstalk is the main loss after missing knowledge. Things that didn't help: soft-max
  (attention-sharpened) matching, codes sharing a topic component, and orthogonalising neighbouring lines.
  Things that did: equal weight for every line whatever its length, and cosine normalisation.
- The exact-match head costs a fourth code head (~400 dims instead of 315). It was worth about as much as
  those extra dims, and it keeps options that just repeat question words from winning.
- Generic words and stopwords are down-weighted by hand lists, not by any statistics from ARC.

## Limits

This is bag-of-words association. It cannot order words ("vapor into liquid" vs "liquid into vapor"), compute,
or chain reasoning steps. Questions that need those, much of ARC-Challenge, fall back to topical association,
which the Challenge set was built to defeat. The negation flip also fires on scenario sentences that merely
contain "not".

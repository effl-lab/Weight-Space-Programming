# A hand-wired association scorer for ARC

Nothing here is trained. The knowledge is about 1,700 hand-written lines in `kb/*.txt`. `knowledge.py` turns
them into word vectors, and `build.py` wires a fixed 2-layer, 24-dim transformer (the given `arch.py`) into a
matcher that scores each option by how strongly its words are tied to the question's words.

## The knowledge (`kb/`, `generic.txt`, `cuestop.txt`, `hubs.txt`, `negation.txt`)
- Each line is one concept, written as `entity words : cue words`, for example
  `igneous : magma lava cools volcanic granite basalt`. Regular inflections and singulars are added
  automatically. `=w` means "do not inflect"; `!w` means "keep w even if it is generic".
- `generic.txt` lists function words and question boilerplate that are never linked.
- `cuestop.txt` lists common science words (earth, sun, water, energy...). They may be answers (entities), but
  they are too common in questions to serve as cues.
- Lines with the same first entity word are merged into one group.
- `negation.txt` lists the words that turn a question around (NOT, EXCEPT, least, false...).

## Words as codes
- The 20 association dims form 4 heads of 5 dims. In each head every group gets a pair of codes (u, Ru).
  R = diag(1, 1, -1, -1, -1) is a fixed reflection.
- Entity words of the group store u and cue words store Ru. Keys see a word through R.
- So an option word and a question word match only if one is the entity and the other a cue of the same group.
  A word never matches itself or its own synonyms, which removes "the option repeats a question word" noise.
- The codes come from a deterministic greedy spherical packing: codes are at most cos 0.93 apart, and
  u·Ru ≤ 0.5. That gives about 400 pairs per head.
- A word gets one code per head. It therefore keeps at most 3 groups, or 4 for the hubs listed in `hubs.txt`.
  When it would sit in more, it keeps its own entity groups and its hub preferences first, then the most
  specific (smallest) groups.
- Groups are given heads by DSatur plus min-conflicts colouring, so groups sharing a word never share a head.
  Codes within a head are then chosen to keep groups from the same subject far apart.
- On word pairs from my test questions this gave no false matches and no missed matches.

## The network
Residual dims: anchor A, dump Z, option flag O, value V, and 20 code dims. After matching, three of the code dims
are reused for the negation flag, the option flag and the pooled score.
- **Layer-norm control.**
  - Every matrix that reads a layer norm has its columns summing to zero, through the always-zero dump row Z,
    so the mean subtraction does nothing.
  - A large anchor keeps the layer-norm scale nearly constant, so the network is close to linear.
  - Option tokens get a ~300x larger anchor than question tokens. Their word content is then almost
    invisible to keys, so the first option token works as a content-free sink.
- **Layer 0 attention (4 heads).** Each option word queries the question words with β·code.
  - Question keys sit at a threshold, the sink at +5, other option tokens far below.
  - One perfect match gives an attention fraction of 1/2, two give 2/3, and near-miss codes give ~0.
  - The fraction (value 1 on question keys, 0 on the sink) is added to V.
  - Head 0 also gives negation words a very high key. Their value raises a negation flag (written ×100 into
    a spare dim) instead of a match.
- **Layer 0 MLP.** Built from gated GELU pairs, using gelu(x+B) - gelu(B) ≈ x when open and ≈ 0 when the gate
  closes. Each gate is exactly 0 on its own position class, whatever the token's scale. The MLP:
  - caps a word's match at 0.6;
  - flips its sign when the question is negated;
  - writes an option flag (×1000).
- **Layer 1 attention.** Every token takes the mean of V over the option tokens (keys from the option flag)
  and writes it into the score dim.
- **Layer 1 MLP.** Gives question tokens the option tokens' anchor and spread, so every token reads the same
  value through the final layer norm.
- **Score.** logit = mean match over the option's words + 0.6 × n_option / (n_question + n_option). The second
  term is a small bonus for longer options that mostly breaks ties.

## Measured on my own questions (made up, ARC style; 341 questions in 4 sets)
- Accuracy is about 63–66% on sets I looked at while writing the knowledge base, and about 58% on the last set
  (68 questions) that was never used to tune the knowledge base.
- Real ARC questions are longer and harder, especially the Challenge set, so expect clearly less.
- Many questions get no signal at all (no option word tied to a question word). These count about 1/4.
- The main limits are:
  - coverage of the knowledge base;
  - the 3-slot limit per word (about 30% of written memberships are dropped);
  - the purely associative scoring, which cannot tell "takes in CO2" from "gives off O2".

## Known weakness
- An option with no tokens at all (only symbols) gets a meaningless, typically high score, because pooling
  then falls back onto the question tokens.

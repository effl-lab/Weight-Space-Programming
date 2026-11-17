# Hand-written science association transformer

This is an **untrained, analytically constructed** instance of the supplied
12-dimensional, four-head, two-block transformer. `arch.py` is unchanged.
No pretrained model, corpus, ARC files, internet source, optimizer, fitting,
accuracy-driven search, or random parameter initialization was used.

## Authored knowledge

- `knowledge.tsv` supplies science entities, synonyms, and topic memberships.
  It covers biology, ecology, anatomy, scientific investigation, chemistry,
  matter, energy, mechanics, electricity, waves, geology, weather, and astronomy.
- `associations.tsv` supplies directional cue-to-concept associations, such as
  fulcrum → lever, boiling → vaporization, and hormone → endocrine.
- `refinements.tsv` explicitly separates meanings that the coarse glossary
  conflated, including transparent/opaque, north/south, above/below,
  higher/lower, balanced/unbalanced forces, and saturated/unsaturated solutions.
  It also supplies a small number of additional science distinctions.
- `aliases.tsv` supplies explicit spelling variants and negative contractions.
  The builder adds possessives only for manually enumerated scientific nouns.
  Each alias copies its canonical word's entire embedding, including negation
  metadata and lexical priors. Aliases add no new semantic codes.
- `build.py` adds explicit number-word equivalences, function words, a few
  disambiguations, negation markers, and small language priors. It compiles
  this information directly to ordinary dense float64 tensors. Double precision
  prevents cancellation error in the large-anchor feature projections from
  being amplified by sharp attention.
- `vocab.txt` is the sorted union of these authored words. Vocabulary generation
  is deterministic; the delivered file is already generated. Unknown words
  have a neutral embedding, rather than a guessed science meaning.

Semantic codebooks consist of points on a circle, not learned word vectors.
The neutral code is the zero vector, not a point on the circle: unknown and
neutral function words therefore cannot act as nearby science-concept keys.
Each word has a question-cue code, an answer-entity code, and a science-topic
code. Adjacent entity codes follow the manually written ontology's order.

## Computation implemented by the weights

The twelve coordinates contain the three two-dimensional codes, a
question/option segment bit, a salience bit, a negation channel, a score
channel, and two large fixed normalization anchors.

1. **Block 0 attention** broadcasts explicit negation from question tokens,
   not option tokens, using the score channel as temporary storage. Its MLP
   erases each token's original negation marker, moves a thresholded question
   flag into the negation channel, and restores the score channel's small
   lexical prior. Thus an option-local marker cannot accidentally change the
   question flag through the residual connection. The same MLP keeps question
   cue codes but replaces option cue codes with the options' answer-entity
   codes, supplying a self-match reference for the next block's softmax.
2. **Block 1 attention** measures four forms of question support: a sharp
   directed association, a sharp entity/synonym match, shared science topic,
   and a small, softer contribution from neighboring ontology entries.
   Its values indicate question-side versus option-side keys. An exact
   option-side self-match competes with question keys, so unrelated question
   words do not automatically count as support. The three sharp heads derive
   their inverse temperatures from codebook geometry; the deliberately soft
   fourth head retains its separate neighborhood-matching role.
3. **Block 1's MLP** retains evidence on salient option tokens only and reverses
   its sign for explicit exception/negation questions. It clears the salience
   channel after using it. The final score reads the difference of the score
   and cleared channels, canceling LayerNorm centering without subtracting the
   large anchors in the final dot product.
4. The supplied architecture averages over non-padding tokens and returns a
   scalar for each candidate. No parameter refers to an option's A/B/C/D
   position. Position weights identify only the fixed question/option boundary
   at token 128; they contain no question IDs or answer-position information.

The anchors make LayerNorm approximately affine on the small feature values.
Linear projections explicitly compensate for its centering. Bounded gates are
implemented with analytically chosen GELU pairs. All unused weights are zero.
No nonstandard tensor subclasses, hooks, evaluator modifications, or extra
inference code are involved.

### Analytic separation for exact matches

For `N` nonneutral unit-circle codes, the minimum self-versus-other dot-product
separation is `d = min(1, 2 sin²(pi/N))`; the minimum includes the neutral zero
key. With inverse temperature `T`, up to 128 wrong question keys contribute
at most `128 exp(-T d)` relative to an exact option self-key in ideal code
geometry. The option-key bonus only improves that bound.

The sharp heads use `T = 2 log(128 / 1e-6) / d`. This reserves half the geometric
separation for normalization/gating perturbations while targeting at most
`1e-6` leakage under that margin. This is a closed-form construction, not a
parameter search or fitting step. It replaces fixed temperatures whose
separation weakened as the authored codebook grew. The facts, head mixture
weights, priors, and deliberately soft fourth head are unchanged.

## Reproduction and checks

`build.build()` reads the five delivered text files and returns exactly the
37 required tensors. It does not write files. Running `python build.py`
explicitly regenerates the vocabulary from the authored lexicon.

The checker is run on original, independently written smoke-test questions.
The separate implementation tests check determinism, finite outputs,
option-permutation equivariance, question conditioning, and the negation gate.
These are implementation checks, not a fitting loop. No parameter is selected
from measured question accuracy.

Final review checks passed: `check_model.py` reports **CHECK OK**, with 3,760
vocabulary entries and 49,860 scalar weights (32,944 nonzero). Implementation
checks cover deterministic rebuilding, the unchanged architecture, valid
single-token vocabulary entries, distinct embeddings for opposing meanings,
question-scoped negation with option-local markers and negative contractions,
preservation of lexical priors, exact embedding equality for all declared
aliases, zero semantic vectors for neutral tokens, absence of meaningful
support from all-unknown questions, padding-position invariance, empty-input
behavior, option-permutation equivariance, finite outputs, and conditioning.
All-pairs code-vector checks independently verify the three sharp-head bounds
from the delivered matrices. A numerical attention test with 128 repetitions
of an adjacent wrong entity code produced question-key mass of approximately
`3.5e-15`; repeating the exact entity instead retained more than 90% question
mass. These are mechanism tests, not science-answer accuracy measurements.

A separate delivery audit copies only the eight deliverable files into a fresh
package directory, parses the exact tensor contract from `TASK.md`, and builds
in independent processes. All 37 names and shapes match; every weight is an
ordinary finite float64 CPU tensor. Changing the working directory, Python
hash seed, and ambient default floating dtype produces byte-identical weights.
Building does not modify the package. This audit found no additional weight
or scoring correction to make, so the model was left unchanged.

The sorted name/shape/dtype/tensor-byte SHA-256 produced by
`/srv/work/check_delivery.py` is:
`652cd8abe27f2f37f57ad6b723f9b325c30b4da044dd7e556503c96e52b420f7`.

A 3,530-question **synthetic repeated-token resource test**, using four CPU
threads and the evaluator's full sequence lengths and batch size, took about
10.1 seconds. Peak resident memory for the test process was 1,259 MiB, below
2 GiB. These synthetic inputs are not ARC questions and measure no accuracy.

The initial implementation's float32 feature-recovery test failed; float64
reduced maximum recovery roundoff from approximately 8.9e-4 to 1.9e-12. During
review, the vocabulary-validity regression check caught an unreachable
hyphenated entry (`half-life`); it was removed. The builder now validates
single-token entries and unique codebook names, and orders marker insertion
deterministically. A subsequent design review found that placing the neutral
code on the semantic circle gave it unintended neighbors; it now has zero
coordinates. Explicit aliases improve known-word coverage without inferring
new science relations. Alias canonical forms, conflicts, and chains are
validated at build time. No checker answer results were used to select these
changes.

## Limitations

This is a lexical science heuristic, not a pretrained language model. The
small fixed state cannot hold general reasoning machinery. Each word has one
main entity sense and one directed cue target, so polysemy and multiple valid
relations are compressed aggressively. Question negation reverses relatedness
rather than performing full logical reasoning; answer-clause negation is not
parsed. There is no arithmetic, syntactic
parsing, multihop inference, or learned calibration. Accuracy on the unseen
ARC questions has not been measured and is not claimed.

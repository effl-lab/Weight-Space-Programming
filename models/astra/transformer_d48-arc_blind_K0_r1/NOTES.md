# Hand-authored science-association transformer

## Deliverable and constraints

`build.py` defines `build()` and returns exactly the ordinary, dense float32 tensors
required by the supplied architecture. `arch.py` is unchanged, with SHA-256:

```
28f37cc068049ebb97182718688a83edf4e8d80ec68d42db35b894121d44bd7e
```

The vocabulary has 5,453 entries. The model has 308,352 scalar parameters,
of which 204,082 are nonzero.

The four hand-authored source tables are:

- `concepts.tsv`: 411 science concepts and their word aliases, covering physics,
  chemistry, Earth and space science, biology, ecology, measurement, and methods.
- `rules.tsv`: 618 directed association rules from question words to answer
  concepts. Negative coefficients express incompatible associations. Entries with
  coefficient zero are documentary references, not model features.
- `categories.tsv`: 75 concept-family rows defining broader semantic categories,
  such as tree -> plant -> producer and mammal -> vertebrate -> animal.
- `forms.tsv`: explicitly authored alternate spellings and inflections. Existing
  concept aliases and rule cues take precedence over these fallback forms.

The source tables and `vocab.txt` are the only data files that the builder reads.
The builder neither reads questions nor performs inference itself. No training,
fitting, pretrained weights, downloaded resources, additional ARC questions, or
accuracy-based search were used. Source validation and circuit tests do not
optimize parameters.

## Review and revisions

The design reviews identified and addressed seven weaknesses without using
multiple-choice accuracy as feedback:

1. **Answer-side information was being erased.** The original broad question
   stop list also removed meaningful answer categories such as “people,”
   “investigation,” “diagram,” and “number.” Answer embeddings now use a separate,
   narrower grammatical stop list. These words retain their answer concepts while
   remaining suppressed as generic question cues. The ambiguous grammatical word
   “he” still does not acquire the helium concept.
2. **Negated options were treated as affirmative.** Block 0 now detects negation
   independently within each segment. Its MLP reverses question cue vectors for
   question negation and option answer vectors for option negation. Negation in
   one segment cannot trigger the other segment's detector. This is still a
   segment-wide approximation, not syntactic negation-scope analysis.
3. **Every option received the same averaged question representation.** Block 1
   now uses all four heads. Two preserve the global summary; the other two form
   option-conditioned summaries. The latter can emphasize compatible question
   cues instead of always diluting them among unrelated background words. Global
   and focused summaries receive equal weight. The attention gain and mixture
   weights were chosen analytically, not searched or tuned on answers.
4. **Answer concepts lacked their broader categories.** A previous review added
   explicit category inheritance. A tree answer now contains attenuated plant,
   producer, and living-thing features rather than only its tree feature. A
   category at distance `d` contributes `0.5**d`; multiple paths use the shortest
   distance, not repeated contributions. Explicit meanings retain unit weight.
   This is deterministic compilation of authored memberships, not graph learning.
5. **Word forms and common examples were missing.** A previous review added
   explicit morphological fallbacks, animal and tree examples, and the concepts
   buoyancy, floating, and sinking. These changes added 544 vocabulary entries.
   Alternate forms copy their complete base embeddings, including control
   features, and cannot overwrite existing explicitly authored word meanings.
   That lexical review did not change the transformer circuits.
6. **Approximate erasure leaked into question-token scoring.** A previous review
   found small nonzero compatibility increments at question positions: up to
   approximately `2.76e-6` on a synthetic token probe. Block 1's multiplication
   neurons now include the fixed offset `6*(segment_role - 1)`. It is approximately
   zero for options and -12 for questions, suppressing question-side GELU outputs
   without consuming any additional neurons. Tests verify zero question-side
   compatibility increments even when residual answer features are deliberately
   amplified. The vocabulary and knowledge tables were unchanged in that review.
7. **Plausibility priors ignored question polarity.** The latest review corrected
   the answer-word priors for negated and exception questions. Previously, a word
   such as “always” incurred the same small negative prior even when the question
   asked for an incorrect statement. A second block-0 head now broadcasts question
   negation, independently of option-local negation. Two additional MLP neurons
   reverse option priors when question negation is active. Prior magnitudes,
   vocabulary, and knowledge tables were not changed, and no accuracy feedback
   was used to choose this correction.

These are implementation and design improvements. No claim of improved held-out
accuracy is made, since held-out questions and results were not accessed.

## Lexical knowledge

Each concept receives a fixed, deterministic 20-dimensional direction computed
from its name with SHAKE-256. A small, similarly constructed domain component
provides broad topical similarity. Important contrasts, such as hot/cold,
renewable/nonrenewable, east/west, and increasing/decreasing, have opposite
vectors. These hash-derived vectors are random indexing codes, not learned word
embeddings.

A word has separate question-side and answer-side vectors. Its answer vector
represents its aliased concepts and attenuated broader categories. Shared
categories are counted once even when several explicit meanings reach them.
Its question vector combines a small self-association with the authored factual
links. This asymmetry supports links
such as thermometer -> temperature, ribosome -> protein, and sunrise -> east,
rather than just rewarding repeated words. Grammatical words carry no semantic
vectors. Generic question words use broader filtering than answer words.
Alternate forms and possessives inherit their base word's complete features.
Possessives of alternate forms resolve directly to the explicit root, so copying
is independent of vocabulary order. Category references, lexical bases, conflicts,
and cycles are validated during compilation.

Twenty question coordinates and twenty answer coordinates occupy the first 40
embedding dimensions. The remaining eight coordinates hold lexical salience,
negation, segment identity, two numerical anchors, a segment-local negation
summary, an output accumulator, and a zero reference. After block-0 attention,
the salience coordinate also carries a common offset encoding question negation.

## Transformer computation

1. **Positions identify segments.** Positions 0–127 are the question, and
   positions 128–175 are the option. No weight represents the option's index
   among the four choices. A mild question-recency bias emphasizes later cues.
2. **Block 0 detects local and question-level negation.** One head uses a product
   of query and key segment signs to restrict attention to the query's own
   segment. Within that segment, negation markers receive a strong key preference.
   Its summary coordinate holds question negation at question positions and
   option negation at option positions. A second head attends only to question
   keys and broadcasts approximately eight times the question-negation indicator
   as a common offset in salience. Adding a constant to all question keys cancels
   in softmax; only the already-small layer-normalization scale variation can
   perturb this invariance.
3. **Block 0 gates lexical roles, polarity, and priors.** Analytic GELU gates
   erase question-side answer vectors and option-side cue vectors. Separate gates
   reverse retained vectors in negated segments. Small priors for absolute or
   implausible words, such as “always” and “magically,” apply only to option
   tokens. They are negative for ordinary questions and positive for negated or
   exception questions. The fixed gate offset `4*salience + 32*role - 40` detects
   the broadcast at option positions while remaining inactive at question
   positions. These gates use 88 of the 96 hidden neurons.
4. **Block 1 combines global and focused question evidence.** The first two
   heads compute a salience-weighted question summary, ten coordinates per head.
   The other two use the corresponding ten option-answer coordinates to query
   question cues. Their attention logits add four times the partial semantic dot
   product to the salience and segment biases. Each output half is an equal
   mixture of its global and focused summaries. All four heads exclude option
   keys, so an option cannot supply its own factual evidence.
5. **Block 1 scores compatibility.** Its MLP approximates twenty products
   between the resulting question-summary coordinates and each option word's
   answer coordinates. It uses the quadratic, even component of GELU and four
   neurons per product, for 80 of the 96 hidden neurons. A segment-dependent
   preactivation offset suppresses these neurons at question positions rather
   than relying solely on approximate answer-feature erasure in block 0. The
   scalar readout extracts their sum. The supplied mean pooling aggregates this
   evidence over real tokens.

The numerical anchors make the initial embedding norm constant and keep later
layer-normalization scales nearly constant. Subtracting the reference coordinate
cancels layer normalization's mean shift. Thus the hand-constructed gates and
multiplication circuit remain approximately in their intended units. Unused
neurons and connections have explicit zero weights.

## Reproducibility and validation

`vocab.txt` is the sorted vocabulary derived from the concept aliases, rule cues,
authored alternate forms, and hand-authored control-word lists in `build.py`.
For a fresh copy without a vocabulary, `python build.py --write-vocab` creates it;
this command deliberately refuses to overwrite an existing file. Ordinary
`build()` validates the existing vocabulary and does not write files. The lexical
review regenerated the vocabulary after checking that no previous entries would
be removed. The subsequent numerical and polarity reviews retained that vocabulary
unchanged.

The required command passed after the revisions:

```
python /srv/work/tools/check_model.py
CHECK OK
weights 308,352 (204,082 non-zero), vocabulary 5,453 words
```

Additional checks passed:

- Exactly the required tensor names, shapes, and float32 types.
- Finite weights and outputs, including empty-input behavior.
- Deterministic rebuilding, option-order equivariance, and identical-option ties.
- Preservation of answer-only concepts without reintroducing generic question
  cues.
- Exact full-embedding inheritance for all alternate and possessive forms.
- Precedence of existing explicit words over morphological fallbacks.
- Category attenuation, shortest-path deduplication, and explicit-meaning
  precedence.
- Independent question and option negation, testing all four polarity cases.
- Correct sign of every lexical prior for ordinary, “not,” and “except” inputs,
  using both one-token and maximum-length 128-token question segments.
- A uniform question-polarity broadcast that is not activated by option-only
  negation, including the dual-purpose word “never.”
- No reactivation of lexical priors at question positions.
- Role erasure and sign reversal in intermediate transformer states.
- Agreement between the focused attention circuit and its direct analytic
  formula.
- Zero question-side compatibility increments, including deliberately amplified
  residual answer features.
- A segment-gated multiplication truth table covering signs, zeros, and maximum
  single-coordinate inputs, within its analytic approximation-error bound.
- Exact invariance to changing the padding token's embedding to a content word.
- The unchanged architecture's SHA-256 digest.

Tests used synthetic token arrays and isolated lexical probes, not labeled
multiple-choice questions. On four CPU threads, construction took 0.076 seconds
and a maximum-length batch of 64 questions took 0.147 seconds. Peak RSS for the
test process was 982.7 MiB. These are local measurements, not a full held-out run.
The required checker used its default placeholder.

### Full-workload delivery review

A subsequent review tested the complete task-sized workload rather than
extrapolating from one batch. It used 3,530 synthetic, maximum-length token-ID
sequences with four options each, the prescribed 128/48 segment lengths, and
batches of 64 on four CPU threads. No ARC questions, answer labels, or accuracy
measurements were involved.

- Build time: **0.080 seconds**.
- Inference time for all **14,120 option scores**: **14.415 seconds**.
- Build, input allocation, and inference together: **14.502 seconds**.
- Peak process RSS: **1,343.3 MiB**, below even a decimal 2 GB limit.
- All output logits were finite and had the required `(3530, 4)` shape.
- Two fresh builds, running with the model folder as their working directory and
  Python hash seeds 1 and 997, produced identical canonical weight digests:

  ```
  1988c7329be5156c06d1f1c6109bc070cac34e9d8a04d9faaeb9983a186fae6b
  ```

- Source, vocabulary, knowledge-table, and architecture hashes were unchanged by
  construction and inference.

The checks are implemented in `/srv/work/review_delivery.py`; this is a validation
script, not a file read by the model. They found no additional implementation or
resource-budget failure. The weights were therefore retained rather than adding
further speculative heuristics. This review strengthens delivery validation; it
does not establish held-out accuracy or remove the limitations below.

## Limitations

This is a compact association scorer, not a trained language model. Twenty
semantic dimensions cause collisions between unrelated facts. Its word-based
summaries lose syntax, phrase boundaries, quantities, causal direction, and most
multi-step reasoning. Focused attention can emphasize accidental matches as well
as genuine ones, and each focused head sees only half the semantic coordinates.
Negation remains segment-wide rather than scope-aware, and polysemous words mix
their meanings. Mean pooling also introduces an option-length effect. These are
disclosed limitations of a fixed, fully hand-authored construction.

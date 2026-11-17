# Hand-authored science association transformer

This is an **untrained lexical knowledge and association model**, implemented
entirely in the supplied, unchanged two-block transformer. It is not a pretrained
language model. It uses no downloaded files, ARC corpus, optimization, fitting,
parameter search, or accuracy-based selection.

## Knowledge

`knowledge.py` contains my explicitly authored elementary-science vocabulary and
associations. There are 52 semantic axes covering biology, physical science,
Earth/space science, laboratory safety, and experimental reasoning. Some axes
have opposed ends, such as renewable/nonrenewable, conductor/insulator,
heating/cooling, erosion/weathering, and rotation/revolution. Additional explicit
associations combine these same coordinates. Deterministic spelling rules add
plural and possessive aliases. Explicitly written inflection families cover
common science verbs and a few irregular nouns, preserving separately authored
forms rather than replacing their meanings. Standard negated contractions and
interrogative contractions have explicit syntax metadata. None of these rules
learns from text or changes the evaluator's tokenizer.

Each base word's semantic embedding is the bounded sum of its authored
associations. A newly added grammatical alias receives the same vector as its
source. Possessives also retain their base word's lexical metadata: for example,
`earth's` and `earth` have exactly identical embeddings. This matters because the
specified tokenizer preserves straight apostrophes rather than stripping them.
Unlisted words have zero semantic features. Small language priors discourage
absolute expressions such as “always” and “only.” No option index is represented.

## How the fixed transformer computes a score

- **Stable layer normalization.** Two equal and opposite constant embedding
  coordinates of magnitude 4,096 dominate the norm. Linear projections subtract
  a zero-reference coordinate to cancel the layer-normalization mean. The large
  anchors reduce semantic/control interference and baseline drift in float32.
  The resulting operations approximate ordinary algebra on the semantic
  coordinates while still using the exact required architecture.
- **Question and option roles.** The fixed 128/48 token layout is recorded in
  positional coordinates. Padding is handled by the original architecture.
- **First block.** One attention head selects the most recent negation or
  interrogative marker in the question. A later “which,” “what,” etc. resets a
  background sentence's negation; a later “not,” “except,” etc. sets it. Option
  markers cannot affect this decision. The prior coordinate temporarily carries
  the interrogative flag; the MLP clears that temporary flag before scoring.
  Another head computes a length-sensitive, zero-association attention baseline.
  A third head separately detects assertion-negation words in the current
  option, with question keys suppressed. Its word list includes “not,” “no,”
  “never,” and “without,” but deliberately excludes request modifiers such as
  “least” and “avoid.” The question-negation lexical coordinate is reused for
  this broadcast option bit after its input role is complete. The output
  coordinate temporarily carries the option-negation lexical flag and is then
  cleared. Positional GELU gates also clear the first question token's semantic
  coordinates, making that position a null attention key, and round both
  negation bits. The first word is consequently not a semantic cue; this remains
  an intentional limitation, not a learned tokenizer behavior.
- **Second block.** Four heads compare an option token's scientific features
  with the real question tokens. Option keys are suppressed. Each head returns
  the probability of attending to question content rather than the null key.
  Subtracting the first block's baseline gives positive evidence for aligned
  associations and negative evidence for opposed associations. Both the content
  heads and their baseline use the same modest positional recency bias:
  `-3 + 1.5 * position / 128` relative to the null key. This favors later query
  words over earlier background narration without needing punctuation tokens.
- **Readout.** A hand-wired MLP routes evidence only at option positions, using
  the XOR of question-request negation and option-assertion negation. Exactly
  one of the two negations reverses semantic evidence; both negate it twice.
  Language priors depend on question polarity alone: adding “not” to an option
  must not turn its “only” penalty into a reward. The supplied mean pooling and
  final linear score then produce one logit per option. The gates rely on the
  identity `GELU(x) - GELU(-x) = x` and on GELU's near-linear/near-zero tails;
  no learned approximator is used.

All tensors are explicitly allocated as dense float32, independently of the
caller's PyTorch default dtype. `build.py` constructs every required weight,
including zero weights. A per-build cache normalizes each distinct semantic
vector once, rather than repeating the identical work for grammatical aliases.
This cache preserves every weight bit-for-bit and is not a learned or persistent
parameter cache. `build()` reads only the supplied local knowledge and
vocabulary. Running `python build.py` serializes the deterministic vocabulary;
the vocabulary is already included in this folder.

## Limitations

This is a bounded-capacity association heuristic, not a general science solver.
It cannot reliably parse multiword relations, distinguish all causal directions,
perform arithmetic, or reason over long descriptions. Polysemy, negation scope,
and several facts sharing a coordinate can produce incorrect answers. The
language priors and the first-token null-key construction are also imperfect.
The option-negation flag is statement-wide: it does not understand nested scope,
“not only ... but also,” or multiple negations within one option. There is no
claim of measured held-out accuracy.

## Verification

The required `python /srv/work/tools/check_model.py` was run successfully.
Separate implementation checks exercise finite logits, padding invariance,
question-to-question isolation, option-order equivariance, and a full 64-question
batch. They do not measure science-question accuracy or optimize any weights.

The review added marker-only unit tests for negation resets, option-marker
isolation, clearing temporary features while preserving ordinary priors, the
null-key construction, and the analytic recency baseline. Maximum-length inputs
also produce finite logits. The architecture is checked against the literal
source in TASK.md, not merely against expected parameter shapes.

The further review added unit tests for option-only negation detection,
question/option flag isolation, and clearing the temporary lexical score flag.
A synthetic continuous-feature truth table exercises every question/option
polarity combination, both token roles, and positive/zero/negative evidence.
It verifies the actual final MLP, including independent language-prior polarity.
No science questions or answer labels are used in this truth table.

Lexical-coverage tests additionally verify exact embedding identity for
possessives and newly added inflections, the polarity of expanded contractions,
compatibility with the task's apostrophe-preserving tokenizer, and exact
end-to-end logit invariance under alias substitution. These are grammar and
implementation checks, not science-answer accuracy measurements.

Builder regression checks compare every tensor with this model's own
pre-refactor weights, not pretrained weights. All weights are bit-identical
under both float32 and float64 caller defaults. The revised builder took
approximately 0.5 seconds in those checks; no inference behavior changed.

Final revised check: **CHECK OK**, 640,576 weights (30,279 nonzero), and 8,782
vocabulary entries including rule-generated aliases. A 64-question batch took
0.192 seconds; the implementation-invariant test process peaked at 983.6 MiB.

A separate full-sized resource check built the model and scored **3,530
synthetic question/option pairs** at maximum token lengths, using the evaluator's
batch size of 64 and four CPU threads. Build plus inference took **10.476
seconds**, with **1,246.4 MiB** peak resident memory, below the 20-minute and
2-GB limits. These inputs were random token IDs, not ARC questions, and only
output shape and finiteness were checked. None of these checks measures accuracy.

## Review changes and rationale

1. Increased balanced anchors from 128 to 4,096 to reduce layer-normalization
   drift; this is numerical conditioning, not a fitted hyperparameter.
2. Replaced “any question negation reverses the score” with the explicit
   latest-marker routing described above. This handles a background “not”
   followed by a fresh interrogative clause. It is still a scope heuristic and
   does not fully parse nested clauses or double negatives.
3. Added the same modest recency preference to attention and its baseline,
   reflecting that a question's request often follows its background text.
4. Added explicit instrument/quantity associations (length/ruler, time/stopwatch,
   temperature/thermometer, volume/graduated cylinder, and mass/balance), along
   with qualitative versus quantitative measurement vocabulary.
5. Added option-assertion negation, which the earlier scorer ignored. The final
   MLP now explicitly routes all four combinations of question/option negation,
   without reversing language priors merely because an option contains “not.”
6. Added exact possessive aliases and explicit inflection families, addressing
   formerly unrecognized forms such as `earth's`, `plants'`, `attracting`, and
   `eroded` without changing their source meanings. Expanded standard negated
   contractions and interrogative forms such as `what's`. The attention and
   scoring structure was not changed in this lexical-coverage review.
7. Cached duplicate semantic-vector normalization locally during construction
   and made all tensor dtypes explicit. Exact regression tests show that this
   implementation change preserves every weight. A full-sized synthetic stress
   test now verifies the task's time and memory requirements, rather than
   relying only on a single-batch measurement.

No science-answer accuracy was measured to select these changes, and no claim
is made that the revised model's held-out accuracy has been empirically improved.

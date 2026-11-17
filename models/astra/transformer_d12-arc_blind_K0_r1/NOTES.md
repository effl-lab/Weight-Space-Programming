# Handwritten science-association transformer

## Deliverable

This is a deterministic, untrained model for the supplied, **unchanged**
`arch.py`. It has **7,481 vocabulary entries**, **748 internal concept symbols**,
and **94,512 scalar weights** (74,683 nonzero). All weights are ordinary dense
`torch.float64` tensors. No pretrained embeddings, downloaded resources,
training examples, fitted parameters, or accuracy-driven search are used.

Files used by `build()`:

- `vocab.txt`: sorted authored vocabulary; padding and unknown retain IDs 0 and 1.
- `knowledge.tsv`: handwritten concepts, memberships, and related-answer links.
- `overrides.tsv`: handwritten word-level refinements.
- `corrections.tsv`: membership corrections and additional authored vocabulary.
- `families.tsv`: 42 authored broader-to-narrower concept-family definitions.
- `build.py`: compiles the descriptions into every required weight using fixed
  formulas, with explicit concept aliases, possessive spelling rules, lexical
  priors, and query rules. It reads no questions, answers, checkpoints, or
  evaluator output. Running it as a script materializes `vocab.txt`; `build()`
  itself only reads deliverable files and returns tensors.

The vocabulary covers elementary mechanics, matter, chemistry, waves, Earth
science, astronomy, cells, organisms, ecology, resources, and investigation.
Authored associations include positive charge → proton, chlorophyll →
sunlight/photosynthesis, ribosome → protein, and transverse waves → perpendicular
motion. Memberships also group synonyms and biological instances.

## Review improvements

1. **Repair concept references.** A static audit found four unreachable related
   concepts: `dissolve`, `permeability`, `scattering`, and `decrease`. Authored
   aliases connect them to existing keys. Other inconsistent names were unified,
   and `build()` now rejects unreachable related/query targets.
2. **Preserve possessives.** The tokenizer keeps apostrophes. Fixed English
   spelling rules now preserve the meanings of forms such as “earth's” and
   “plants'”, inheriting memberships and specialized query rules. Explicit
   entries take precedence. All **4,192 possessive spelling equivalences** have
   been checked for exactly identical embeddings; no corpus mining was used.
3. **Correct conflations and expand coverage.** “Balanced” describes force
   balance rather than a measuring instrument; nighttime is distinct from
   daytime. Added entries cover instruments, ions, geographic and astronomical
   terms, cell structures, plant anatomy, and ecology.
4. **Question polarity.** A first-block head detects authored question cues such
   as “not,” “except,” “least,” and “incorrect.” The standalone token “non” is
   included because the prescribed tokenizer splits “non-renewable.”
5. **Option polarity and cross-segment cancellation.** A second
   first-block head detects logical negators, such as “not,” “cannot,” and
   negative contractions, **only within the first three option tokens**. It
   excludes question-selection words such as “except” and “least.” This fixes
   the previous circuit's near-indifference to an option's leading negation,
   while deliberately ignoring later contrasts such as “…but does not…”. The
   flags combine by XOR: one active segment reverses association, while two
   active segments cancel. This is a heuristic capability improvement, not a
   measured accuracy improvement.
6. **Broader concept families.** Exact-symbol queries could miss
   known subtypes whenever their two lexical key slots used other concepts.
   Handwritten families now let queries such as instrument, animal, organ,
   machine, rock, and element recognize their declared narrower concepts.
   There are **286 inherited broader-to-narrower pairs**, compiled into the same
   two-dimensional query slots by a closed-form geometric construction. Leaf
   concepts retain exact-symbol matching. The review also separates “machine”
   from mechanical energy and adds nonmetal/metalloid terminology. No family,
   ordering, or parameter was selected from answer accuracy.
7. **Tokenizer-compatible contractions (latest review).** The prescribed regex
   retains ASCII apostrophes but splits curly-apostrophe forms such as “isn’t”
   into `isn` and `t`. Thirteen explicitly listed, unambiguous fragments now have
   the same marker embedding as their complete negative contractions. Five
   previously missing complete negative contractions were also added. The
   ordinary words `can`, `don`, `won`, and `haven` are deliberately excluded from
   fragment matching. No tokenizer or architecture code was changed.

## How twelve dimensions implement the scorer

Each ordinary token contains two two-dimensional **queries**, two
 two-dimensional **key symbols**, a position/polarity coordinate, an evidence
coordinate, and two normalization-anchor coordinates. Keys are equally spaced
points on a unit circle. Queries address either a single symbol or a contiguous
angular interval for an authored concept family. These are symbolic addresses,
not learned semantic embeddings.

The default first query is the word's primary concept; the second is an
explicitly related concept. Keys encode primary and alternate memberships.
Handwritten query rules handle some definitions and weakly weight ambiguous
verbs. Negators instead carry out-of-band markers, which are removed before
concept matching.

### Family geometry

`families.tsv` defines a forest, with at most one parent per concept. The compiler
checks all names, rejects cycles and repeated parents, and lays out the forest
in deterministic depth-first order. Every family's members, including itself,
then occupy a contiguous interval of equally spaced unit-circle keys.

For an interval with midpoint angle `m` and angular half-width `h`, its query is
`(cos(m), sin(m)) / cos(h)`. Its dot product with a unit key is at least one
exactly inside that interval. A leaf has `h = 0`, so it remains an exact-symbol
query. A fixed confidence adjustment preserves the existing attention margin
at interval endpoints. Neighboring nonmembers retain a large rejection margin.

This uses no extra embedding dimensions or transformer layers. Query radius is
explicitly bounded by 1.25 so the polarity circuit remains valid; the actual
maximum for the authored forest is **1.010902592**. Families are lexical science
relationships, not a complete or sense-disambiguated formal ontology.

### Normalization anchors

Large, equal-and-opposite anchor coordinates and centered feature writes make
layer normalization nearly linear on the small semantic features. Float64
keeps the sharp attention thresholds and gates numerically reliable.

### First block: scoped flags, XOR, and key masking

Initial position codes are +1 for question tokens, -2 for the first three
option tokens, and -1 for the remaining option tokens. The first attention
head reads question-side markers. The second reads logical-negation markers
in the option prefix. A 32-unit position advantage exceeds a marker's maximum
17.25-unit lexical advantage, so markers outside the intended region cannot
win against ordinary real tokens inside it. This isolation assumes a nonempty
question and option, as in normal task inputs.

Ordinary query coordinates are bounded by [-1.25,1.25], while marker coordinates
are sixteen. Absent/present head reads are therefore bounded by
[-1.25/16,1.25/16] and approximately one, respectively. They are broadcast as position shifts of 1/4 and 1/8.
Their four discrete combinations are 0, 1/8, 1/4, and 3/8. Only the two middle
levels require reversing the score.

The first MLP uses 23 of its 24 hidden units to:

- clear question keys and question-side lexical priors;
- erase both out-of-band markers from the query coordinates;
- recognize the two middle levels with clipped ramps in each position band;
- restore ordinary segment codes and write
  `p = (+1 for question, -1 for option) + 0.25 * (question_flag XOR option_flag)`;
- add a compensating evidence offset for the second block.

The conservative absent-read noise bound is 15/512. Ramp positions, widths, and GELU scaling
are chosen analytically with separation from that bound. No classifier is
trained and no runtime control flow is added to the supplied architecture.

### Second block: four concept comparisons

The four heads test all query/key-slot pairings. Question tokens supply null
keys. Option keys receive a threshold offset, overcome only by symbols inside
the query's declared family. Circle spacing determines the attention scale
analytically; even neighboring nonmembers have a large rejection margin.

The XOR-polarity shift adds the same logit offset to every key, so it cancels
in softmax. Values encode the option indicator minus a polarity-dependent
constant; the residual offset from block zero cancels that constant. The
resulting evidence remains the attention mass on matching option tokens.

### Evidence readout

The second MLP retains question evidence at full strength and only 2% of
option-internal evidence, preventing trivial self-matches from dominating.
It reverses both contributions when XOR is one. Its multiplier is +1/-1 on
question positions and +0.02/-0.02 on option positions.

The weak option residual supplies a specificity/coherence preference. Small,
explicit priors discourage absolute wording such as “always” and “never” when
the net polarity is positive; they reverse with the evidence. Final layer
normalization and the score vector read the mean evidence coordinate.

The only within-option position distinction is the three-token negation
prefix. **There is no answer-index preference:** all four options use the same
independent scorer, including when their polarity flags differ.

## Limitations

This is a small associative circuit, not a trained language model. A word has
at most two query targets and two key memberships. The model does not generally
parse syntax, resolve word senses, perform arithmetic, or implement multi-step
reasoning. Lexical overlap can favor distractors, and related choices may be
indistinguishable. Broader families can increase irrelevant association, and the
forest cannot encode arbitrary multiple inheritance. Ambiguous terms such as
“nucleus” are not disambiguated by context. The answer-language priors are
heuristics, not facts.

Negation is still approximate. The question flag is existential across the
question; the option flag is existential within its first three tokens.
The model does not understand “not only,” internal clause scope, or multiple
negators inside one segment. A question flag and an option flag cancel, but
this is **not general double-negation parsing**. A main-predicate negator later
than the option prefix is ignored. “Least” may be numerical rather than
associative. Choosing the least-associated option need not identify a false
statement. These limitations can cause incorrect reversals or missed reversals.

Contraction-fragment handling is deliberately partial: curly-apostrophe
“can’t,” “don’t,” “won’t,” and “haven’t” are not recognized via their ambiguous
fragments, though their complete ASCII-apostrophe spellings are recognized.
Recognized punctuation variants preserve polarity, not necessarily identical
logits: the fixed tokenizer produces different token counts.

Unknown words have no concept features. Mean pooling introduces a
question/option-length dependence. **Held-out accuracy has not been measured,
and no claim of improved test accuracy is made.** Changes were chosen from
static consistency and circuit reasoning, not answer-accuracy feedback.

## Verification

`python /srv/work/tools/check_model.py` passed:

```
CHECK OK
weights 94,512 (74,683 non-zero), vocabulary 7,481 words
```

`/srv/work/verify_model.py`, outside the deliverable, uses only authored,
unlabeled inputs. It verifies:

- exact required weight names, finite inference, and batch consistency;
- identical-option and unknown-token ties, plus empty-input handling;
- first-block question-key masking and centered residuals;
- both polarity heads' segment isolation and the option-prefix boundary;
- every normal vocabulary word against false question activation, including
  options containing question-only selection cues;
- every question negator and every option negator, including marker erasure;
- the first MLP's XOR truth table at all absent-read noise corners for all
  three input position bands;
- the second MLP's sign/attenuation truth table;
- end-to-end single-segment reversal and cross-segment cancellation, without
  assigning gold answers to the token probes;
- mixed-polarity option independence and option-permutation equivariance;
- exact embedding equality for all possessive spelling equivalences and
  end-to-end spelling invariance in questions and options;
- vocabulary uniqueness and prescribed-tokenizer compatibility;
- every one of the **559,504 concept pairs**, comparing geometric matches with
  independently computed reachability in the authored family graph;
- unit key norms, bounded query radii, and family/nonfamily rejection margins;
- 104 contraction-spelling polarity cases using the exact prescribed tokenizer:
  13 contractions, four apostrophe spellings, and both question and option roles;
- four explicit ambiguous-word guards against spurious negation activation.

For the compiled geometry, the minimum declared-match logit is **14.000000**
and the maximum nonmember logit is **-143.912742**. These are algebraic compiler
checks against authored declarations, not answer labels or measured ARC accuracy.

The latest `--full` run processed **3,530 repetitions of an authored, unlabeled
input**, in batches of 64 on four CPU threads. This was a resource load test,
**not an ARC evaluation**. Observed results:

- Full synthetic 3,530-question inference: **31.251 seconds**.
- Peak process resident memory: **1,309.6 MiB**, below 2 GiB.
- Initial 64-question batch: **1.262 seconds**.
- Maximum question-key masking residual: **3.07e-11**.
- Build, mechanical checks, and full load test combined: **33.624 seconds**.

Timing varies with host load. No answer accuracy was calculated, no parameter
sweep was run, and no parameters were selected from these test outputs.
The supplied `arch.py` was never edited.

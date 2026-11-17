# Hand-written science-association transformer

## Delivered files

- `arch.py`: the supplied transformer, byte-for-byte unchanged.
- `knowledge.txt`: 490 hand-authored science association groups in four tables.
- `aliases.txt`: hand-written spelling, inflection, and terminology equivalents.
- `vocab.txt`: 3,460 word forms from those tables, aliases, explicit modifiers,
  and a fixed plural-spelling rule.
- `build.py`: deterministic formulas returning all 62,704 float32 parameters.
  It reads only `knowledge.txt`, `aliases.txt`, and `vocab.txt` in its folder.

There is no training, fitting, pretrained representation, downloaded resource,
question lookup, or inference-time Python shortcut. The scorer is precisely the
provided `arch.forward`; all lexical knowledge is compiled into ordinary dense
embedding, attention, normalization, and feed-forward weights.

## Design

The embedding contains four two-dimensional circular dictionary addresses. Each
first-layer attention head uses one address to associate scientific terms in an
option with related terms in its question. Topics include biology, body systems,
ecology, astronomy, geology, weather, chemistry, energy, waves, electricity,
scientific method, instruments, engineering, and measurement.

A spelling has one address per head. Earlier table definitions ordinarily win;
a small explicit sense table gives measurement and physical-quantity words their
intended meanings instead of their first incidental association. For example,
“temperature” must connect to thermometers rather than being captured by a heat
transfer row. Particle-charge and ozone/UV definitions also receive explicit
priority over incidental associations. Sense overrides cover the same fixed
spelling variants as the tables; conflicting explicit directives are rejected.
Aliases, including common chemical formulas, fill missing addresses without
overwriting explicit senses.

Two large, opposite embedding coordinates, +8192 and -8192, stabilize
layer-normalization scale. These exact power-of-two anchors give an exactly
representable base variance of 2^23; the other features perturb it only at roughly
float32 precision. This makes high-resolution dictionary matching less sensitive
to which unrelated semantic features are present on a word.
Projections subtract a near-zero scratch coordinate to cancel centering without
subtracting large anchor products. Scratch coordinate 15 is zero until the second
attention output. The first MLP clears the no-longer-needed importance channel
11, which then supplies the reference for the second MLP and final score.
These are finite float32 arithmetic operations in the prescribed architecture.

The first attention layer uses the fixed question/option position boundary.
Option queries attend to matching scientific terms in the question; ordinary
question words supply a zero-valued softmax baseline. Option keys are strongly
suppressed. An exact association has a logit approximately 8 above the baseline.
The angular scale is calculated directly from table size so neighboring addresses
are separated by at least 24 logit units. No scale is selected using accuracy.

The first MLP removes question-token scores, clears answer-local negation flags,
recycles the importance coordinate, and decodes a cautious-direction control.
The second attention layer retrieves question negation and cautious-context
markers, cross-averages signed increase/decrease terms between the question and
option, and pools the option's mean association score. That option mean is broadcast to all
real tokens in a separate score channel. Reading the broadcast score avoids
question-token dilution and makes duplicating all answer content approximately
score-invariant. Answer words do not become question negation controls.

The second MLP applies question polarity and a small qualitative-direction bonus.
The bonus also reverses for negative questions. Its gated GELU hinges implement
an explicit piecewise-linear function; a fixed power-of-two input scale makes
the GELUs approximate ReLUs in the required activation regions. For nonempty,
uncautioned inputs, cross-averaging makes the final direction contribution
approximately `mean(question_direction) * mean(option_direction)`, with the
appropriate question-polarity sign. It is independent of the relative question
and option lengths. Small explicit lexical priors discourage absolute claims
and favor basic laboratory safety. Final normalization, mean pooling, and all
other computations remain exactly those in the supplied architecture.

## First review changes

The initial version had 2,803 vocabulary entries and 52,192 parameters. The review
made general construction changes, not accuracy-selected changes:

1. Isolated question negation by clearing answer-local polarity features before
   question negation is broadcast.
2. Replaced diluted token-sum scoring with broadcast option-mean scoring.
3. Made the qualitative-direction bonus follow question polarity consistently.
4. Recycled scratch space to implement pooling without changing the architecture
   or introducing numerically fragile large-anchor cancellation.
5. Added explicit ambiguous-word senses, authored aliases, missing inflections,
   and 85 supplementary association groups covering instruments, engineering,
   cell division, sensory organs, and other elementary-science concepts.

## Second review changes

The first review had 3,405 words and 61,824 parameters. This review made these
additional, analytically chosen changes:

1. Increased normalization anchors to the exact power of two 8192. This reduces
   scale drift in the sharp circular-address comparisons without changing the
   architecture or choosing a value from question performance.
2. Removed three misleading aliases: the pronoun “he” is not helium, an isolated
   diagram label “c” is not reliably carbon, and “potentially” is not synonymous
   with potential energy. Lowercasing makes element-symbol ambiguity unavoidable.
3. Added common multi-character metric abbreviations such as cm, kg, mg, ml,
   kW, and kPa, plus the corresponding unit vocabulary. Single-letter labels
   remain conservative rather than being indiscriminately treated as units.
4. Added eight explicit planet-attribute groups, separate from the broad planet
   class. This preserves general planet associations while distinguishing facts
   such as Jupiter's size, Saturn's rings, and Uranus's sideways orientation.
5. Added strict alias-token and vocabulary-consistency checks to detect stale
   delivery files before inference.

## Third review changes

The previous version had 3,449 words and 62,528 parameters. This review changes
the direction heuristic, not the science associations:

1. Added an explicit cautious-context control for density, volume, mass,
   distance, duration/time, pressure, resistance, wavelength, and inverse wording.
   A direct sign-agreement rule cannot establish the causal relationship for
   these quantities without knowing the relevant equation and what is fixed.
2. Packed this control as value 2 in the initial direction feature. Two first-
   layer MLP units remove that categorical value before direction averaging and
   mark only question occurrences with a .01 role offset. Answer occurrences
   cannot trigger the question-level control.
3. Used the previously unused fourth second-layer attention head to retrieve
   that marker. It shifts the direction gate into a region where the bonus is
   identically zero. The science association score and negation logic remain
   active; this is abstention from one heuristic, not algebraic reasoning.
4. Added white-box checks of categorical decoding, question-only scope, bonus
   suppression in cautious contexts, and retention of the uncautioned gate.
   These tests inspect activations and piecewise arithmetic, not answer labels.

All constants are analytically chosen with separated activation regions. The
changes fit inside the existing 16-dimensional, four-head, two-layer model.

## Fourth review changes

The vocabulary, parameter count, science associations, and priors are unchanged.
This review fixed the direction circuit and strengthened its tests:

1. An end-to-end arithmetic test revealed that the previous direction gate's
   terms canceled each other. Earlier tests checked latent gate values and
   length consistency but did not isolate the actual nonzero score contribution.
2. Replaced the incorrect conditional-linear construction with genuinely gated
   GELU hinges. Their exact intended function is
   `f(t) = R(t) - R(-t) - 2.5 R(t-.15) + 1.5 R(t-.25)`, where `R` is ReLU.
   Thus `f(t)=t` in the ordinary direction region and `f(t)=0` in the cautious
   region. A fixed scale of 1024 approximates these hinges without fitting.
3. Made direction attention cross the segment boundary in both directions.
   Question tokens receive the option's mean signed direction; option tokens
   receive the question's mean. An extra value component cancels the residual
   role offset. The final mean of these local products is the product of the
   two segment means, without an unintended option-length multiplier.
4. Added checks for a nonzero direction contribution, opposite signs, question
   negation, question and option duplication, and end-to-end cautious-context
   suppression. A diagnostic ablation isolates the direction term strictly for
   arithmetic verification; it is not used for answer-accuracy selection.

## Fifth review changes

The previous version had 3,451 words and 62,560 parameters. This review adds and
corrects general science definitions without changing the scoring circuit:

1. Gave the particle-charge definitions explicit priority in the fourth science
   dictionary: protons are positive, electrons negative, and neutrons uncharged.
   Previously incidental circuit and repulsion rows captured the relevant words
   before the particle-definition rows. Other dictionaries retain the electrical,
   magnetic, and acid/base associations, including neutral pH 7.
2. Added missing chemical-formula aliases: H2O for water, C6H12O6 for glucose,
   CH4 for methane, O3 for ozone, and HCl for hydrochloric acid. Existing explicit
   formula senses are retained; aliases fill missing semantic addresses.
3. Linked methane to fossil fuels and molecular chemistry, hydrochloric acid to
   acid vocabulary, and the stratosphere/ozone group to ultraviolet radiation.
   These are elementary scientific associations written from general knowledge,
   not additional question/answer examples.
4. Added direct embedding checks for the three distinct charge definitions,
   all new formula aliases, ozone/UV, and preservation of alternate senses.
   No multiple-choice answer labels were used to select the changes.

## Sixth review changes

A label-free audit found 15 inconsistencies between explicit sense overrides and
forms generated by the existing fixed spelling rule. For example, “temperature”
had its curated measurement sense while “temperatures” retained an incidental
heat-transfer sense. Similar gaps affected masses, volumes, distances, currents,
resistances, velocities, heats, and several planet/atmosphere spellings.

1. Applied each authored sense override to all of its existing fixed spelling
   variants, using exactly the same rule as the underlying dictionary tables.
   No new words, scientific facts, association groups, or scoring circuits were
   introduced.
2. Added a build-time assertion rejecting conflicting explicit sense directives
   for the same spelling in the same head.
3. Added checks of all 175 distinct head/form pairs covered by the sense table,
   verifying their exact intended addresses and science-importance markers after
   alias compilation. All checks pass; the previously detected gaps are closed.

The vocabulary, parameter count, nonzero count, and architecture are unchanged.
This is a compilation-consistency correction; no held-out accuracy gain has been
measured or claimed.

No changes in any review were selected by searching measured accuracy. The
existing authored questions were not used to choose weights, thresholds, or
vocabulary additions. No ARC data beyond the supplied examples was accessed.

## Limitations

This is a lexical reasoning heuristic, not a general language model. It cannot
reliably distinguish grammatical roles, solve quantitative science problems,
resolve complicated negation, interpret diagrams, or reason through multi-step
experiments. Its
associations are symmetric: a question about what a plant releases can still be
confused with what it consumes. A global negation flag can misread incidental
uses of “not.” Unknown words provide no semantic knowledge. The normalized
option mean can favor a concise, highly associated answer over a detailed one.

The direction feature remains a weak heuristic, not a universal causal law.
The cautious-context gate now suppresses it for named quantities with potentially
inverse or conditional relationships. This is conservative: it also abstains in
some contexts where a direct relationship would be valid, and it cannot catch
every causal exception. It does not solve equations or recover the correct sign
of an inverse relationship. Dominant science associations remain unchanged.

## Verification

`python /srv/work/tools/check_model.py` passes and reports:

- 62,704 parameters;
- 24,378 nonzero entries;
- 3,460 vocabulary words.

`/srv/work/verify_model.py` verifies exact parameter names and shapes, finite
float32 values, deterministic repeated builds, and byte-for-byte equality of
`arch.py` with the task specification. Its four-thread stress test of 3,530
synthetic maximum-length questions in 64-question batches completed in 10.116
seconds, including weight construction, with peak process RSS of 855.6 MiB.
Synthetic IDs, not ARC questions or labels, were used for this resource test.

`/srv/work/review_checks.py` separately verifies actual latent-state question/
answer negation isolation, scratch clearing, question-score clearing, option-
permutation equivariance, score consistency when answer content is duplicated,
and semantic-address equivalence for spelling and metric-unit aliases. It also
checks exclusion of ambiguous aliases, compilation of all eight planet facts,
and the cautious-context decoding, scope, and direction-suppression invariants.
End-to-end direction-active tests cover question/option duplication, opposite
signs, question negation, nonzero output, and cautious-context suppression. The
isolated direction contribution matches its mean-product formula with maximum
absolute error of approximately 8.26e-6 on the arithmetic probes.
Additional definition checks verify distinct particle charges, chemical formula
aliases, the ozone/UV association, and preservation of other dictionary senses.
All 175 explicit sense/head spelling pairs are checked against their intended
circular addresses after compilation, including fixed inflection variants.
Normalized feature reconstruction has a maximum absolute error of approximately
1.79e-7 across the complete embedding table with the question role marker set.
All checks pass. These are arithmetic and lexical invariants, not accuracy-based
parameter selection.

The ten independently authored diagnostic questions in
`/srv/work/my_questions.json` still yield nine correct answers. The same
plant-gas-release question remains incorrect. This small construction check is
not a held-out accuracy estimate, and it does not establish an accuracy gain.

# Hand-authored science relation model

## What this is

This is a deterministic, non-neural knowledge model for four-option grade-school science questions. It has no pretrained component, learned parameters, fitting procedure, random initialization, network dependency, or external corpus. Its science content and inference coefficients were written by hand for this task.

The seven knowledge files cover biology and ecology; mechanics, energy, electricity and waves; chemistry and matter; geology, weather and astronomy; scientific inquiry and engineering; and everyday scientific relationships. `lexicon.txt` supplies inflections, synonyms, and multiword concepts. `relations.txt` supplies directed transformations. `taxonomy.txt` supplies directed category memberships. `properties.txt` and `comparisons.txt` supply hand-written scalar facts and comparison rules.

## Files and compilation

- `vocab.txt`: 16,927 tokenizable words, with the required offset of two for padding and unknown IDs.
- `build.py`: `build()` reads only the authored model files and returns named Torch tensors. The tensors encode 3,890 concepts, 4,359 retrieval contexts, and 9,430 signed assertion entries, including mechanically generated reverse retrieval links.
- `arch.py`: the fixed tensor forward program. It imports only `torch` and `math`; there is no input-dependent Python control flow, text processing, file access, or conversion of tensors to Python values.
- `biology.txt`, `physics.txt`, `chemistry.txt`, `earth_space.txt`, `inquiry.txt`, `extra.txt`, `concepts.txt`: authored propositions.
- `lexicon.txt`: authored lexical equivalences.
- `relations.txt`: ordered source-to-destination relations.
- `taxonomy.txt`: directed class memberships. Compilation yields 1,363 inheritance links, including a bounded transitive closure, and 81 parent classes.
- `properties.txt`: 238 hand-authored named reference entries. The comparison module supports 30 properties in total, including numeric-only dimensions.
- `comparisons.txt`: 65 hand-authored rules specifying context, property, comparative cues, and optional implicit direction.

A knowledge line has the form:

```
question concepts | supported alternatives separated by semicolons | contradicted alternatives
```

Underscores denote lexical phrases, such as `carbon_dioxide`. A leading `=` requests a strong converse lookup for a definition. Other affirmative assertions receive weaker, associative reverse retrieval links: these help identify a process from its description without treating every association as a logically certain biconditional. A leading `!` marks an explicitly negative-question rule, which is not inverted again by question negation.

Duplicate assertions retain their strongest authored confidence. An automatically generated weak reverse link therefore cannot shadow a later explicit fact. Retrieval priority also accounts for whether a context has strong authored evidence or only weak reverse links.

Vocabulary generation is deterministic: `python build.py --vocab`. It collects the authored words, explicit aliases, conservative generated inflections, and a fixed numeric vocabulary. Scalar reference values become weights rather than extra numeric vocabulary entries. No question or test files are read by `build()`.

## Forward pass

1. **Concept recognition.** Token IDs are mapped to concepts; two- and three-token phrases are matched with tensor comparisons. Both concept presence and first-occurrence position are retained.
2. **Directed category inheritance.** Named members inherit their classes: for example, a robin is a bird and a vertebrate, but a bird does not imply a robin. A bounded, precompiled transitive closure supplies these features with fixed distance attenuation. Inheritance does not invent textual positions. A modest class constraint can match the first explicitly named category near `which` or `what` to the options.
3. **Limited lexical similarity.** A small hand-selected soft neighborhood handles nearby expressions. Inheritance is separate from synonyms; invalid equivalences such as invertebrate/exoskeleton and large/strong have been removed.
4. **Context retrieval.** Weighted concept coverage retrieves relevant question contexts. A steep coverage function rewards several consistent clues rather than an isolated topic word. Term specificity uses frequency in the authored knowledge itself, not any question dataset.
5. **Option evidence.** Each option is matched to assertion completions. Repeated words common to the options, and words already supplied in the question, receive less discriminating weight. A fixed top-evidence aggregation combines support and contradiction.
6. **Polarity.** Affirmation/absence in assertions and options is distinguished. Question cues such as `not`, `except`, `least`, and `avoid` reverse ordinary evidence, while explicitly written exception rules retain their meaning. Negation in a preceding narrative is excluded when it falls before the final interrogative cue.
7. **Directed relationships.** Token/phrase positions distinguish, for example, electrical-to-mechanical energy from mechanical-to-electrical energy. A `from` between the endpoints reverses the surface order. These relationships also identify devices or processes from ordered descriptions in questions.
8. **Small symbolic modules.** Equality/opposition rules handle magnetic poles and electric charges, including repeated or plural identities. A dimension-aware arithmetic module uses 18 explicitly authored formulas for density, mass, volume, speed, distance, time, work, kinetic energy, force, acceleration, power, and Ohm's law.
9. **Scalar comparisons.** A tensor-only ranking module compares named reference entities, explicit measurements, and unambiguous one-step derived values. It requires all four options to have known, unambiguous values for the selected property and contributes no comparison evidence otherwise.

There is no option-position prior. Unknown and padded content supplies no evidence. Identical options receive identical logits.

### Arithmetic details

The numeric module recognizes base and compound units, cubic dimensions, short decimal fragments, and many common unit aliases. Both questions and options are converted to consistent dimensions. Consequently, physically equivalent numeric options receive equal arithmetic evidence; an option with an incompatible explicit unit does not receive a numeric match.

The module requires exactly one identifiable measurement for each required input dimension and refuses division by zero. It does not average multiple measurements into an invented input. A formula's output must not already be supplied as a measured quantity. Formula exponents, coefficients, dimensions, and target concepts are explicit tensors constructed from hand-written formula specifications.

Kinetic energy uses the standard formula `0.5 * mass_kg * speed_squared`. Valid zero-valued numerators produce exactly zero rather than a small logarithmic-clamp approximation. Nonfinite calculated values are gated off and replaced with zero.

The unit registry uses the same canonicalization as the lexicon, so aliases such as `volt`/`volts`/`voltage` do not disagree between recognition and arithmetic.

### Comparison details

Named reference properties cover material density, thermal and electrical conductivity, specific heat, phase-change temperatures, mineral hardness, planetary properties, electromagnetic frequencies, and atomic number and mass. Each reference line is `property | entity | value`. Entries with the same value within a property share an evidence group, preventing equal-valued aliases from creating artificial ambiguity. Different-valued entities in one option make its named value ambiguous.

Explicit numeric options are converted to common units. The same conservative formula evaluator can derive an option's density or kinetic energy from its measurements. Four options consisting solely of one numeric value each can instead be ranked under an implied common unit. This bare-number assumption is not mixed with explicitly unit-bearing choices or numbers embedded in descriptions.

Comparison direction uses literal comparative cues rather than synonym-collapsed magnitude words. The module distinguishes `least dense` from `least likely`: probability phrases do not act as ordinary magnitude cues, while `least likely` separately reverses an implicit preference such as floating in water. Final-interrogative context has priority over earlier narrative context. The strongest sufficiently supported property, or tied properties, supplies pairwise ranking evidence with numerical tie tolerance.

Unknown values, different-valued multi-entity options, and negated options suppress comparison evidence. This abstention applies to the comparison module, not necessarily to the model's independent retrieval modules.

## Why this design

A hand-authored model cannot acquire a general-purpose language representation by training under these rules. Explicit scientific relationships provide useful coverage without pretending to have one. Directed retrieval preserves the distinction between a topic and its answer; explicit contradictions help separate nearby distractors. Phrase recognition avoids collapsing concepts such as carbon dioxide and oxygen. Directed category inheritance, ordered relations, scalar comparisons, and narrowly scoped symbolic operations cover reasoning failures that pure bag-of-words matching cannot resolve.

All normalization, confidence multipliers, and module coefficients are fixed design choices. There was no parameter search, fitting, or automatic selection using measured accuracy. Questions and numerical specifications used in checks were independently authored for this task. No held-out ARC data was accessed.

## Review changes

The first review added broader knowledge about classification, genetics, scientific inference, environmental systems, materials, climate, and astronomy. It also corrected first-seen confidence shadowing in the compiler; separated directed type membership from lexical similarity; narrowed question-negation scope; and replaced the original integer-only, input-unit-only arithmetic with dimension-aware input/output calculations and inverse formulas.

The subsequent review added scalar comparison rules and named reference tables, conservative numeric and derived-value comparisons, equivalent-value grouping, and comparative/probability cue separation. It also added kinetic energy and exact zero-numerator handling to the shared formula evaluator.

These are implementation and coverage improvements, not a measured claim of higher accuracy on the unseen ARC test set.

## Verification and resources

The required checker passes on the updated model (`CHECK OK`). Its sixteen independently authored smoke questions yield 15/16 correct, unchanged from the previous reviewed version; the roots-function exception question remains incorrect. This small smoke result is not a held-out accuracy estimate and was not used for parameter search.

Additional deterministic checks pass for:

- static tensor-program restrictions;
- output shape and finite values;
- zero-padding and entirely unknown inputs;
- option-permutation equivariance;
- batch-composition invariance;
- exact ties for identical options;
- twelve independently authored arithmetic specifications;
- equivalent-unit equality and incompatible-unit rejection;
- abstention on ambiguous repeated measurements;
- directed inheritance, including no subtype inference from a parent class;
- absence of the former erroneous invertebrate/exoskeleton implication;
- sixteen independently authored comparison specifications;
- comparison ties, bare numeric choices, and mixed bare/explicit-unit abstention;
- comparison abstention for unknown, ambiguous, negated-option, and non-comparative cases;
- kinetic energy with both nonzero and zero speed.

A four-thread CPU run over 3,530 synthetic questions, made only by repeating independently authored smoke-test questions, took **15.271 seconds** for forward evaluation; construction took **0.349 seconds**. Peak process resident memory was **845.2 MiB**. Tensor weights occupy **227,784,059 bytes**. This workload check is not an accuracy estimate on ARC, but verifies substantial headroom under the 2 GB / 20 minute limits.

## Limitations

The model has no general language understanding. It can fail on unusual paraphrases, long multi-step narratives, ambiguous scope of negation, comparisons requiring uncoded world knowledge, diagrams, and detailed experimental scenarios. Its ontology describes typical grade-school categories and does not perform general word-sense disambiguation.

Scalar references are approximate and conditional: material properties depend on composition and temperature, phase-change temperatures on pressure, and planetary temperatures on the definition used. Variable materials are coarsely grouped where close rankings would be unreliable. Outer-planet temperature entries are omitted rather than mixing effective, minimum, and cloud-top temperatures. Rotation periods are sidereal and are not equated with generic solar-day length. Comparison rules do not supply a general model of those conditions or of compound linguistic scope.

Arithmetic remains deliberately narrow. The supplied tokenizer removes decimal points, commas, signs, and operators; some numeric distinctions are therefore unrecoverable. Short decimals are inferred only in constrained local patterns, and three fractional digits are inferred only after a leading zero to avoid confusing common thousands separators. Arbitrary numerical expressions, algebraic parsing, and all possible unit notations are not supported. Weak converse links are retrieval associations, not a full theorem prover. Smoke-test correctness is not a claim about unseen ARC accuracy.

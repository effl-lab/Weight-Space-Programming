# Authored science model — generation 2

## Summary

This is a deterministic, hand-authored knowledge-and-reasoning model for four-choice elementary and middle-school science. It extends the supplied parent model. It contains no pretrained model, downloaded corpus, learned embedding, random initialization, gradient computation, fitted parameter, or external service dependency.

The model combines directed scientific assertions, conservative whole-statement completion, lexical predicate oppositions, one-way taxonomic inheritance, ordered transformations, qualitative causal reasoning, explicit experimental-variable role binding, and the parent's arithmetic and property-comparison modules. All inference is a fixed Torch tensor program. The Python compiler reads only the authored files in this folder and returns all inference weights as tensors.

The aim is broader and more reliable science coverage than the parent, **not a claim of measured improvement on unseen ARC questions**. No held-out questions, labels, or per-question parent results were available or accessed.

## Deliverables and compilation

- `vocab.txt`: **20,665** words, one per line; IDs start at 2. Padding and unknown words contribute no lexical evidence.
- `build.py`: `build()` compiles authored content into a dictionary of tensors. `python build.py --vocab` deterministically regenerates the vocabulary.
- `arch.py`: `forward(w, q, o)` returns `(n, 4)` float logits. It imports only `torch` and `math`. Its Python loops are over eight fixed phrase-matching chunks and three fixed positions within a phrase. All input-dependent decisions are tensor operations.
- `biology.txt`, `physics.txt`, `chemistry.txt`, `earth_space.txt`, `inquiry.txt`, `extra.txt`, `concepts.txt`: the parent's authored science assertions, with reviewed corrections. In particular, chemistry now distinguishes isotope mass number, atomic mass, and abundance-weighted average atomic mass.
- `extensions.txt`, `scenarios.txt`: additional authored biological, chemical, physical, geological, astronomical, environmental, and inquiry knowledge. Mechanism explanations and concise argument-bound facts supplement topic definitions.
- `lexicon.txt`, `lexicon_extra.txt`: explicit aliases, inflections, and multiword concepts. The compiler also generates conservative inflections by fixed spelling rules.
- `taxonomy.txt`, `taxonomy_extra.txt`: directed member-to-class links.
- `entities.txt`: multiword entity names whose component words must not be independently typed or used as named scalar references.
- `review.txt`: scientific clarifications added during review, including alloys versus elements, natural gas versus methane, and compounds versus their constituent elements.
- `opposites.txt`: explicit alternative predicates such as increase/decrease, absorb/release, and parallel/perpendicular.
- `causal.txt`: qualitative monotonic laws with an optional contextual requirement.
- `experiment_grammar.txt`: literal cue sets for requested variable roles, active/passive verbs, clause boundaries, fixed-value qualifiers, and quantity heads. The compiler turns these authored grammar rules into token lookup tensors.
- `relations.txt`, `properties.txt`, `comparisons.txt`: the parent's ordered relations, scalar reference values, and comparison selectors.

An assertion has the form:

```
question concepts | supported alternatives separated by semicolons | contradicted alternatives
```

Underscores denote lexical phrases. `=` requests a strong converse lookup for a definition. Other positive assertions receive weaker associative reverse links; these are retrieval aids, not logical biconditionals. `!` marks an explicit exception-question rule that should not be inverted again by question negation.

The compiler deduplicates assertions, retaining the strongest authored confidence. Term specificity is calculated only from the authored assertions, not from question data. No question files or verification results are read by `build()`.

### Compiled representation

| Component | Size |
|---|---:|
| Concepts, including the null slot | 4,579 |
| Retrieval contexts | 6,210 |
| Signed assertion entries, including reverse links | 13,437 |
| Phrase patterns | 350 |
| Directed inheritance links, including bounded closure | 2,042 |
| Predicate-opposition pairs | 39 |
| Qualitative causal laws | 121 |
| Total tensor elements | 95,188,341 |
| Nonzero tensor elements | 222,019 |
| Tensor storage | 381,070,277 bytes |

Dense matrices intentionally trade a modest amount of memory for straightforward CPU matrix multiplication. Most entries are zero, but no external sparse retrieval engine is required.

## Inference

### 1. Concept recognition and inheritance

Token IDs map to canonical concepts. Two- and three-word phrases are recognized with tensor comparisons. Features retain both presence and first textual position. Named organisms and materials inherit categories through a precompiled, attenuated directed closure: a spider can imply arachnid and animal, but animal never implies spider. A small inherited soft lexical neighborhood supplies limited paraphrase support.

Phrase comparisons are processed in eight fixed chunks to bound temporary memory. Authored entity names additionally shield their occupied token spans from standalone inheritance and scalar-reference lookup. `Carbon dioxide` is a compound, not an independently named carbon sample; `salt water` is a mixture, not pure salt plus pure water samples; a `sea star` is not an astronomical star. A separately mentioned component elsewhere in the text remains available. Non-class component words remain available for associative retrieval, so this is not a general semantic parser.

The reviewed ontology distinguishes metallic materials from pure elements: alloys inherit metal and mixture, but metal alone no longer implies element. Named elemental metals retain explicit element memberships. Methane and natural gas are separate concepts: methane is a compound, while natural gas is a mixture mainly containing methane.

### 2. Directed evidence retrieval

Question coverage retrieves authored contexts. A steep coverage function and specificity weighting reward several consistent clues more than an isolated topic word. Options are evaluated against the retrieved completions, with reduced weight for words repeated in every option or already present in the question. Fixed top-evidence aggregation combines support and contradiction.

Explicit opposite predicates suppress partial matches. For example, an option containing a decrease should not receive unqualified support merely because it shares the noun in an assertion about an increase. Oppositions are used only when the expected predicate is absent and its opposite occurs explicitly. An option containing both predicates is not globally rejected by this mechanism.

Question scope prioritizes the last `which`/`what` request over subordinate `why`/`how`/`when` cues. Thus a subordinate clause does not erase a main-clause `not`; earlier narrative negation is still excluded when it precedes the selected request. Additive constructions such as `not only` are not treated as logical negation. Avoidance is interpreted as a negative-answer request only with a nearby directive such as `should avoid`, rather than reversing ordinary questions about adaptations that help an organism avoid predators. Explicit exception rules keep their authored polarity.

An additional answer-request mask distinguishes a negative premise from a negative answer request. Standalone `why` questions and adjacent `explain why` / `reason why` constructions introduce premises: `why does ice not sink` does not request a false answer. A preceding `which statement does not explain why` still requests a negative answer. The complete premise remains available to scientific retrieval; only request-polarity and symbolic selector cues are masked. Comparison direction, pair-choice selection, and experimental-role requests use this same mask, avoiding numerical rankings or role classifications merely mentioned in an explanation premise. Recognition is intentionally limited to these explicit constructions; it does not parse arbitrary relative clauses or all explanatory paraphrases. This is a limited grammar heuristic, not unrestricted scope resolution.

### 3. Whole-statement completion backoff

Directed retrieval can miss evidence when a description and its name occur on different sides of an authored relation. A secondary module therefore measures how much of a complete positive statement is covered jointly by the question and an option. Only option features absent from the question can add coverage. Question coverage, complete-statement coverage, and distinctive option precision all contribute to the score.

This reuses the existing query and assertion matrices; it does not manufacture a training set or allocate a second full knowledge matrix. It is a lower-weight backoff, not a replacement for directed reasoning.

Crucially, **only affirmative facts participate**. An authored false completion cannot be contraposed into a negative definition of its subject. This prevents an invalid reverse inference from a partially matching misconception.

### 4. Qualitative causal laws

A causal line is:

```
context | changed quantity | responding quantity | monotonic sign
```

Examples include resistance versus current, force versus acceleration, frequency versus wavelength, humidity versus evaporation rate, and vegetation versus erosion. These statements assume other relevant factors are held fixed.

The module binds literal increase/decrease cues to the nearest named quantity. Distance is measured to the quantity's **whole phrase span**, not only its first word. A nearby change to resistance therefore does not automatically become a change to a more distant voltage. The last explicit occurrence of a responding quantity is preferred, so a component such as `effort` inside `effort arm` does not shadow the quantity actually asked about.

Option direction is likewise locally bound. Words inside multiword quantity names, such as `fall` in `fall speed`, are not mistaken for direction cues. A short answer may omit a target explicitly supplied by the question, but an answer naming a different quantity cannot borrow that target. Missing, conflicting, or weakly identified quantities cause this module to abstain.

The authored slope sign is multiplied by the observed source-change sign and compared with each option's change sign. Negated local cues such as `does not increase` cause abstention rather than being treated as evidence of a decrease. Explicitly fixed, constant, or unchanged quantities suppress conflicting generic causal predictions; the module does not manufacture a precise alternative result. It is not a numerical simulator or an unconstrained causal graph search.

### 5. Requested-class selection

The parent used the first named class near `which` or `what` as a category constraint. That remains the fallback. A limited copular parser additionally recognizes patterns such as `which animal is an arachnid`. In such a case, arachnid is the requested class, rather than merely animal.

Only a short copular bridge containing permitted function words can override the initial noun. An intervening content verb blocks the override: `which animal eats a plant` still requests an animal, not a plant. Literal category positions, rather than inferred categories, control selection.

### 6. Retained symbolic modules

- Ordered relations distinguish transformations such as electrical-to-mechanical and mechanical-to-electrical energy, using endpoint positions and a `from` cue.
- Magnetic poles and electric charges have explicit equality/opposition reasoning, including repeated identities. A separate pair-choice module handles questions asking which pair attracts or repels. It counts literal identities in each option, distinguishing two repeated north poles from a north/south pair, and recognizes an explicit `two` or `both` with one named identity. The requested interaction and pair cue must occur in the question scope, and the magnetic/electric domain must be unambiguous. Options with mixed domains, unsupported multiplicities, contradictory same/opposite qualifiers, or negation receive no pair-choice evidence. This does not infer induced charges or polarity for unspecified materials, solve multi-object forces, or resolve spatial arrangements. The retained inverse lookup for a single counterpart is now disabled for pair-valued options, so two south poles are not treated as one south pole relative to a narrative north pole. The new contribution uses the existing authored interaction scale, without fitting.
- Dimension-aware arithmetic uses the parent's 18 authored formulas for density, mass, volume, speed, distance, time, work, kinetic energy, force, acceleration, power, and Ohm's law. Ambiguous measurements and division by zero are gated off. Equivalent explicit units receive equivalent arithmetic evidence. Recognized but unsupported powers and quotients are rejected rather than silently treated as their numerator: square meters are not length, grams per second are not mass, and joules per second squared are not power. Cubic lengths and squared time denominators in acceleration remain supported. Only entirely numeric options may borrow an implied unit; a value followed by an unknown or unsupported unit cannot use that fallback. The arithmetic module abstains on negated options and negative questions instead of supporting a negated equality.
- Scalar comparisons use named reference values, explicit measurements, and conservatively derived quantities. All four choices must be known and unambiguous for the selected property. Every concept in the property selector must be recognized somewhere in the question; a generic topic word alone cannot select an unrelated numerical attribute. Scoped coverage still controls the strength of identified selectors. Comparison direction distinguishes comparative magnitude from probability phrases such as `least likely`.
- Mass number is the integer proton-plus-neutron count of an isotope, not an alias of atomic mass. Atomic mass is the mass of an atom; element-average values depend on isotope abundances. The retained element reference values are rounded averages and are not used as isotope-specific mass numbers. Separate lexical concepts and assertions preserve these distinctions.

There is no option-position prior. Permuting choices permutes logits. Identical choices tie exactly, and entirely padded or unknown inputs produce equal zero logits.

### 7. Explicit experimental-variable roles

A bounded tensor grammar binds short narrative spans to manipulated inputs, measured responses, and explicitly fixed factors. It activates only when the final question names exactly one role near `variable` or `factor`. Narrative evidence must precede that question; a generic definition question supplies no binding.

Active verbs bind a following noun span (`varies the temperature`, `records the height`). Authored passive forms with a nearby copula bind a preceding span (`the pressure is varied`). Spans extend at most eight tokens and stop at recognized clause boundaries or other role verbs. Keeping or holding something is not enough to establish a constant: a nearby explicit `same`, `constant`, or equivalent qualifier is required. Locally negated verbs and qualifiers do not establish bindings.

Choices must match all their recognized non-ignored content words against one role span. Quantity-head checks prevent treating `water` as an answer naming `temperature of water`. There is no taxonomic expansion in this matching step. Unknown words or negation in a choice disable its role contribution. Multiple bindings for one role, or a choice matching multiple roles, cause abstention rather than merging distinct descriptions. An unambiguous match contributes bounded support for the requested role or contradiction for a different role.

This is a narrow grammar, not a general experimental-design solver. It does not infer manipulated factors from treatment tables, resolve pronouns, identify control groups, or conclude that every recorded quantity is necessarily a response in an arbitrarily complex experiment. Unrecognized syntax falls back to the other modules. All cue lists and the fixed contribution scale were authored directly, without parameter search.

## Development and verification

Content and coefficients were authored directly. There was no training, parameter fitting, automatic architecture search, or accuracy-driven parameter sweep. Synthetic examples and algebra specifications were used to inspect and correct implementation errors. They are **development checks, not an independent accuracy estimate**.

The required command passes:

```
python /srv/work/tools/check_model.py
CHECK OK
```

Final deterministic checks passed for:

- the restricted `arch.py` syntax and required interface;
- finite `(n, 4)` outputs;
- option-permutation equivariance;
- batch-composition and batch-order invariance;
- exact ties for identical options;
- zero logits on padded and entirely unknown inputs;
- **242/242** qualitative algebra specifications: each of the 121 authored laws in both change directions, checking the isolated module rather than fitting end-to-end question accuracy;
- explicit opposition and suppression when both predicates occur;
- density arithmetic, equivalent unit expressions, and incompatible-unit rejection;
- 29 entity/type specifications, including alloys, compounds, mixtures, multiword entities, and separately mentioned component words;
- 12 question-scope specifications plus option-negation contrasts;
- six causal negation/fixed-quantity specifications;
- scalar-lookup abstention when a water-vapor option lacks its own reference value, instead of incorrectly using liquid-water density;
- 40 isolated experimental-role specifications covering active/passive voice, role requests, clause ordering, quantity heads, local negation, missing constant qualifiers, ambiguous roles, and unknown options;
- full-inference finiteness, option-permutation, batch invariance, and exact identical-option ties on additional inputs that exercise experimental-role binding;
- 30 supported measurement specifications, including cubic volumes, compound density units, unit conversions, and acceleration;
- 16 unsupported-dimension abstention specifications;
- seven arithmetic validity/negation specifications, including preserved work and force calculations;
- seven atomic-identity and scalar-selection specifications, including abstention when isotope mass number cannot be established from element-average references;
- 153 interaction specifications: 56 ordered-pair/multiplicity/direction/negation checks, 32 abstention checks, one request-scope check, and 64 retained direct-interaction/single-counterpart checks;
- full-inference finite outputs, option-permutation equivariance, batch invariance, and identical-option ties on inputs exercising pair-choice reasoning;
- 30 explanation-scope specifications: 24 premise/request polarity contrasts, one semantic-context retention check, and five symbolic-module abstention checks;
- full-inference finiteness, option permutation, batch invariance, and identical-option ties on explanation-scope inputs.

The final twenty independently authored science smoke questions are answered correctly. Some were used during debugging; their result should not be interpreted as expected ARC accuracy. Verification scripts and inputs are outside the model folder, and the compiler does not read them.

### Resources

A four-thread CPU run constructed the reviewed model in **1.83 seconds**. Forward evaluation of **3,530 synthetic inputs**, obtained by repeating only the authored smoke questions in batches of 64, took **68.96 seconds**. Peak process resident memory across the verification run was **1,253.7 MiB**. The number of synthetic inputs matches the test workload size, but their content is not ARC test data.

This is comfortably inside the requested 2 GB memory and 20 minute runtime limits. Fixed tensor dimensions make the main allocation sizes independent of which words occupy the supplied 128-token questions and 48-token options.

## Limitations

This is not a general language model. It has no learned syntax, world model, or unrestricted multi-step reasoning. It can fail on unfamiliar paraphrases, diagrams, long narratives, negation scope, subtle word senses, and experimental designs requiring more than the authored relationships.

Whole-statement completion is associative evidence, not proof. The causal module assumes fixed other factors and can abstain when multiple changes, embedded phrases, or unusual ordering make roles unclear. Category recognition does not constitute full semantic parsing.

Scalar properties are approximate and conditional on material, temperature, pressure, and other circumstances. Numeric interpretation inherits the task tokenizer's loss of punctuation, signs, and decimal points. The arithmetic module consequently supports only constrained numerical forms rather than arbitrary expressions.

Neither the synthetic results nor the architectural changes establish that the model exceeds the parent's reported 42.1% validation accuracy. Only the unavailable held-out evaluation can determine that.

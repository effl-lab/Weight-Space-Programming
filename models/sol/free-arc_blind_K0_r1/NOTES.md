# Authored Science Memory — reviewed version

This is a deliberately designed, untrained retrieval-and-rules model for four-option school science questions. All scientific statements, synonym groups, rules, and numerical constants were written for this task. No external text, additional ARC data, pretrained model, optimizer, fitting procedure, or accuracy-driven search was used.

## Contents

- **2,932 authored knowledge records** in the topic `.tsv` files. Each record contains question-side clues and a supported answer-side statement. Coverage includes cells, genetics, evolution, ecology, human physiology, matter and chemistry, motion, forces, energy, heat, optics, waves, electricity, magnetism, geology, weather, climate, astronomy, experimental design, laboratory safety, and practical applications.
- `aliases.txt`: inflections and word-level equivalences. Scientifically distinct terms remain distinct: breathing versus cellular respiration, starch versus the carbohydrate category, mushroom versus fungus, sediment versus sedimentary, and bone versus skeleton.
- `phrases.txt`: multiword scientific concepts.
- `concepts.txt`: authored links between alternative multiword names, abbreviations, and formulas, such as *carbon dioxide / CO2*, *red blood cell / erythrocyte*, and *dependent variable / responding variable*.
- `quantities.txt`: 70 scientific quantities, each with separate higher/increased, lower/decreased, and unchanged/constant state concepts. Alternative names for a quantity share these states; different quantities remain distinct.
- `causal_science.tsv`: 275 additional authored conditional relationships, including gas laws, density, forces, circuits, waves, heat, solutions, weather, ecology, physiology, and measurement uncertainty.
- `opposites.txt`: contrasting attributes used to identify incompatible support. The compiler also adds opposed states for each quantity.
- `overrides.txt`: specific descriptions superseding embedded positive phrases, such as *no definite shape* superseding *definite shape*.
- `ranks.tsv`: authored scientific orderings.
- `extra_words.txt`: additional vocabulary and ordinary word forms.
- `vocab.txt`: **38,580 entries**, generated deterministically from the authored material, productive spelling/inflection rules, and integer literals 0–10,000. Unused integer literals support arithmetic without allocating empty retrieval dimensions.
- `prepare.py`: reproducible vocabulary generation. Run `python prepare.py` after editing the source text. It reads no questions or labels.

## Weight construction

`build.build()` compiles the authored material into word and phrase lookups, binary clue and answer matrices, contradiction tables, and numerical rules. A closed-form inverse-document-frequency formula on the authored records reduces the influence of ubiquitous terms. This is deterministic compilation, not parameter optimization or training.

The compiled model has **3,661 concept features** and **25,860,914 tensor elements**, of which **112,820 are nonzero**. The principal memory matrices are dense for efficient CPU multiplication. Every returned weight is a Torch tensor.

## Forward pass

`arch.forward(w, q, o)` is a fixed tensor program with only Torch/math imports and no question-dependent Python branching or lookup. Its loops have fixed bounds: phrase widths 2, 3, and 4, and the three permitted unit-prefix distances.

1. **Concept extraction:** normalize words and detect two-, three-, and four-word concepts after grammatical-word removal. Abbreviations and formulas share features with their expanded names; overlapping equivalence groups are merged transitively. Unknown content words and clause connectors such as *and*, *but*, and *as* act as phrase barriers rather than being removed. Quantity-state concepts distinguish *pressure increases* from *volume increases*. If conflicting states are extracted for the same quantity, those derived state features are suppressed conservatively; ordinary word features remain.
2. **Question focus:** combine the whole question with its direct request. Prefer the final *which/what* clause; subordinate *why/when* clauses cannot replace that request. Without a selection clause, retain the first other interrogative.
3. **Memory retrieval:** combine weighted clue coverage with option-to-answer matching and a smaller reverse-relation contribution. Discount wording shared across options or repeated from the question.
4. **Incompatible support:** penalize opposing attributes only when both the memory and the option unambiguously select opposite members. Statements explicitly discussing both alternatives do not incur this penalty.
5. **Evidence aggregation:** combine the three strongest supporting memories for each option.
6. **Scoped negative requests:** reverse evidence for explicit exclusion requests. Negative conditions inside subordinate clauses, such as a question about evidence that a plant is *not receiving light*, do not automatically reverse the whole question. Direct double negation is handled by parity.
7. **Ordered comparisons:** use independent entity entries and prefer a specific multiword entity over its component words. An *organ system* entry cannot overwrite the *organ* entry, and *inner core* cannot overwrite *outer core* through their shared word. Directly negated comparisons reverse the ordering contribution, while *least* is not reversed twice.
8. **Guarded arithmetic:** calculate density, speed, work, power, basic circuit quantities, and frequency/period for recognizable number-and-unit expressions. Ordinary centimeters/meters are lengths, not volumes; cubic markers or explicit volume units are required. A previous unit cannot donate its number to a later unit. Compound rate units are not misread as standalone distances. Zero divisors and ambiguous adjacent numbers disable inappropriate calculations. Numeric answers must have an explicitly supported output-unit pattern or be unitless. Ratio units require numerator-before-denominator order. Authored equivalent derived forms, such as joules, newton-meters, and watt-seconds for work, include scale conversion. An ordinary article *a* is not treated as an ampere outside a numerical/unit context.

No option index is a feature. Option operations are permutation-equivariant. Empty and entirely unknown inputs produce finite, equal logits.

## Review changes

The review corrected over-broad lexical equivalences, added explicit formula/name equivalences and longer phrases, scoped question focus and negation, added conservative contradiction handling, isolated ranking entities, and repaired dimensional/unit parsing. It also added **211 authored records** in `relationships.tsv`, including molecular building blocks, cell transport, biological distinctions, physiological mechanisms, population relationships, and experimental reasoning.

The second review added property-specific change states and 275 conditional science records. It also preserved transitive phrase aliases, introduced clause barriers and conflicting-state abstention, required dimensionally compatible numeric answers, supported authored derived-unit alternatives, filtered article/ampere ambiguity, and corrected directly negated ordered comparisons.

These changes were selected from code inspection and scientific knowledge, not from measured question accuracy. No model variants were searched or fitted.

## Verification

`python /srv/work/tools/check_model.py` passes.

`/srv/work/verify_model.py` performs implementation-level tests, not dataset accuracy evaluation:

- exact compiler/inference feature agreement on **5,881 authored strings**, including both sides of every memory record;
- preservation of important scientific lexical distinctions;
- compound alias equivalence, unknown-word barriers, and negative phrase overrides;
- property-scoped change states, clause barriers, conflicting-state abstention, and transitive aliases;
- direct checks of question-focus, request-negation, and comparison-polarity helpers;
- unit dimensionality, ratio order, derived-unit scale conversion, article/ampere guards, number-association guards, and authored arithmetic formulas;
- ranking-entry isolation;
- finite `(64, 4)` logits, option-permutation equivariance, and equal logits for empty/unknown inputs;
- 56 forward batches of 64 examples, approximately the requested workload.

On this host with four Torch CPU threads, building took approximately **0.47 seconds**, 56 forward batches took approximately **5.6 seconds**, and peak process resident memory during the extended checks was approximately **592 MiB**. Tests used authored text and implementation invariants, not held-out questions or accuracy labels.

## Limitations

This is not a pretrained language model, and retrieval does not provide general language understanding. The memory is not exhaustive. Complex negation, causal direction, hypothetical scenarios, unfamiliar paraphrases, and long multi-step reasoning remain limitations. Property-state matching is local rather than a full grammatical or causal parser: telegraphic clauses, changes over time, and multiple objects with different states can cause conservative abstention or incorrect attachment. Arithmetic intentionally supports simple recognizable integer-and-unit expressions; ambiguous decimals, multiword numbers, unsupported composite units, and general multi-step calculations are not generally parsed. No held-out accuracy or accuracy improvement is claimed.

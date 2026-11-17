# Authored Science Memory, generation 2

## Design and provenance

This is an untrained, hand-authored science-memory and symbolic-reasoning model. It retains the parent's lexical retrieval architecture and adds explicit conditional constraints, taxonomic inheritance with named exceptions, applied-science explanations, and guarded quantitative reasoning.

No external corpus, pretrained parameters, ARC data beyond the task description, optimizer, fitting, or accuracy-driven model search was used. All declarations and numerical constants are authored; compilation consists of normalization, logical inheritance, matrix construction, and closed-form weighting of the authored memory. No improvement in held-out accuracy is claimed without evaluation.

## Knowledge sources

- **3,895 directly authored retrieval records** in the topic `.tsv` files, including the parent's records, **905 additional scenario/explanation records** in `applied_reasoning.tsv`, `physical_scenarios.tsv`, and `evidence_scenarios.tsv`, and **58 review records** in `review_science.tsv`. Coverage includes life science, ecology, physiology, chemistry, mechanics, heat, waves, electricity, Earth science, astronomy, experimental design, safety, and engineering.
- **3,210 deterministically derived records** from `taxonomy.txt` and `properties.txt`. The compiler computes category ancestry and inherits explicitly declared properties. For example, a named mammal inherits mammal milk/lung properties and vertebrate backbone membership. Named entities remain distinct lexical concepts; a hamster is not made synonymous with every mammal. `exceptions.txt` omits particular inherited properties for mercury, carbon, seahorses, and moss. Assertions reject cyclic/incomplete category closure and invalid exception declarations.
- **239 conjunctive rules** in `logic.txt`. Each rule declares question requirements, supported option requirements, contradicted option requirements, and an authored confidence. Semicolons denote conjunctions and commas denote alternatives. A question atom prefixed with `!` must be absent; one prefixed with `?` must appear in the focused request. Prefixes may be combined to forbid a concept specifically within that request. These are scientific declarations, not dataset question/answer records.
- `aliases.txt`, `phrases.txt`, and `concepts.txt`: inflections, scientific multiword concepts, equivalent names, and formulas. Added negative concepts distinguish *no new substance*, *energy not required*, and *without a backbone* from their embedded positive assertions.
- `quantities.txt`: property-specific increased, decreased, and unchanged states. The compiler additionally recognizes explicit unchanged wording such as *mass does not change* and *no change in mass*.
- `opposites.txt` and `overrides.txt`: incompatible assertions and suppression of embedded positive phrases. Suppression also removes raw lexical features that alias the overridden concept.
- `ranks.tsv`: authored comparisons, extended with sound transmission, mineral hardness, geologic chronology, sediment size, planetary properties, and subatomic particle masses.
- `extra_words.txt` and `parser_words.txt`: additional vocabulary and explicit literal vocabulary for raw-token parser/scope controls. Compilation asserts that every declared parser control and physical-unit word exists in the vocabulary; missing guards cannot silently disappear.

There are **7,105 compiled retrieval records**, **4,399 concept features**, and **42,092 vocabulary entries**. Vocabulary includes deterministic inflection variants and integer strings 0–10,000. `python prepare.py` regenerates `vocab.txt` from these declarations without reading questions.

## Build

`build.build()` and `extension.py` compile the declarations into Torch tensors:

1. Word normalization, phrase-code lookup, equivalent-concept mappings, and quantity-state tables.
2. Clue and answer matrices for the direct and inherited records.
3. Closed-form inverse-document-frequency weights computed only from authored knowledge.
4. Contradiction and negative-phrase override tables.
5. A shared exact-concept matrix plus padded conjunction/alternative indices for whole-question and focused-request constraints.
6. Ordered comparisons, number/unit tables, physical formulas, atom-count markers, and cycle/displacement parsing controls.

Every returned weight is a tensor. The model has **76,348,649 tensor elements**, including **172,498 nonzero elements**. Retrieval matrices are dense for efficient CPU multiplication; the knowledge itself is sparse.

## Fixed tensor forward pass

`arch.py` imports only Torch and math. All question-dependent work is performed with tensor operations. Its Python loops have fixed bounds, and it contains no question-dependent branching, file access, token-to-text decoding, or Python lookup.

1. **Extract concepts.** Normalize known words and match two-, three-, and four-content-word concepts. Unknown content and clause connectors are phrase barriers. Apply negative-phrase overrides and suppress ambiguous conflicting quantity states.
2. **Focus the request.** Combine whole-question evidence with the direct interrogative clause, retaining the parent's scope-aware focus policy.
3. **Retrieve scientific support.** Match question clues to fact keys and options to fact answers, with a smaller reverse contribution. Discount shared option wording and question repetition. Penalize unambiguous opposing properties and aggregate the three strongest memories.
4. **Apply explicit logic.** A rule activates only when all its whole-question and focused-request requirements are met and its forbidden concepts are absent. Option requirements are also conjunctive. Support and contradiction are pooled separately by maximum rather than summed, so overlapping declarations do not accumulate artificial votes. The resulting signed evidence is added to retrieval evidence.
5. **Handle exclusion requests.** Reverse retrieval and logical evidence for scoped negative requests. The separate ordering contribution accounts for comparison polarity without reversing *least* twice.
6. **Apply authored orders.** Prefer specific multiword entities over embedded component terms.
7. **Apply dimensional arithmetic.** Retain density, speed, work, power, and circuit calculations with dimensional checks and unit conversion. Select the first explicit requested mathematical quantity (including *how long*) instead of activating every formula keyword mentioned later. Compute frequency from cycle count/time and period from elapsed time/cycle count; compute total duration from count/frequency or count times an explicitly stated period; compute wave speed from wavelength times frequency. A period/single-cycle cue must link to the time reading through a short grammatical span; an arbitrary observation duration is not assumed to be one period. A multi-cycle count is a barrier to that period assignment. A request for one cycle does not overwrite a larger supplied cycle count. Known frequency must agree with supplied period or cycle/time constraints. Equivalent formulas are pooled by maximum, not added as duplicate votes; conflicting predictions for the requested quantity cause abstention. The simple force-times-distance work formula abstains on explicit angle, perpendicular, opposite, or circular-motion cues rather than silently assuming parallel force.
8. **Propagate atom counts.** Relate atomic number, proton count, neutron count, and mass number. Explicit neutral electron counts can also supply proton/atomic number, and explicit neutrality allows electron-count predictions. Nearby numbers must attach unambiguously through grammatical bridge words rather than crossing unrelated content. Contradictory supplied counts (including different repeated values for the same count), particle-change wording, charged contexts for neutrality-dependent inference, adjacent numerical tokens, and incompatible option units cause conservative abstention.
9. **Compute water displacement.** Require exactly two compatible volume readings and a submersion/displacement context. Compute their difference, optionally dividing an unambiguous object mass by that volume for density. Support volume/density unit conversion. Floating, pouring, spilling, evaporation, excess readings, and ambiguous numerical input disable this helper. When both volume and density occur in the request, the first requested quantity selects the calculation rather than an incidental later mention.

No option-position parameter exists. The entire forward pass is option-permutation equivariant. Empty and wholly unknown inputs produce finite equal logits.

## Review corrections

The review tightened scientific preconditions and scope rather than selecting variants by accuracy:

- Requested renewable/nonrenewable categories, experimental variables, selected cell functions, circuit types, weather-pressure categories, core states, and apparent brightness can be distinguished from background mentions using focused requirements.
- Gas-pressure changes require a closed rigid/fixed-volume container, not merely a sealed flexible balloon. Particle-speed changes require temperature changes, not heating alone during a possible phase change.
- Earth/Moon weight changes encode direction. Potential-energy changes during falling are separated from kinetic-energy claims requiring stronger assumptions; terminal velocity does not imply increasing kinetic energy.
- Loudness and pitch rules no longer contradict independent changes in frequency and amplitude. Both properties may change together.
- Day/night requirements do not rely on matching a phrase across the clause connector *and*.
- Named inheritance exceptions prevent typical category properties from becoming false universals.
- Explicit parser vocabulary and compiler assertions fix a discovered implementation bug in which an out-of-vocabulary word such as *poured* could silently disable an abstention guard.
- Minimum-reading reductions use infinity for non-readings rather than finite sentinels that could be smaller than valid converted quantities.
- A further numerical review separated supplied periods from elapsed durations, added total duration from period times count, prioritized the first requested quantity, rejected conflicting wave/atom constraints and formula predictions, and prevented equivalent formulas from casting duplicate votes.
- Explicit nonparallel/circular force cues disable the parallel-force work shortcut.
- Prokaryotic/bacterial contexts disable generic nucleus/mitochondrion rules. Focused photosynthesis disables the mitochondrion/ATP rule, since ATP can be produced in chloroplasts. Laboratory avoidance/safety rules now require the relevant request rather than an incidental background mention of danger.

## Verification and resource use

`python /srv/work/tools/check_model.py` passes, including architecture restrictions, build, and execution.

The implementation-only checks in `/srv/work/verify_model.py` passed:

- exact compiler/forward feature equivalence on **11,392 distinct authored strings**;
- extraction of every declared rule atom and representability of every rule's required conjunction;
- omission of each declared inheritance exception;
- unchanged-state wording and negative-phrase overrides;
- direct tests of dimensional arithmetic, period/duration roles, cycle identities, first-request selection, wavelength/frequency speed, displacement volume/density, atom-count propagation, conflicting repeated counts, formula agreement, duplicate-evidence invariance, and nonparallel-work abstention;
- exact logical-evidence tests for request scope, travel direction, physical preconditions, terminal velocity, day/night, independent pitch/loudness changes, photosynthetic ATP, prokaryotic exceptions, and lab-safety request scope;
- finite `(64, 4)` output;
- equal logits for empty/unknown inputs and identical options;
- option-permutation equivariance;
- **56 complete forward batches of 64 examples**.

On this host with four Torch CPU threads, the latest extended verification built in approximately **0.97 seconds**, ran the 56-batch workload in **15.7 seconds**, and peaked at **928 MiB** resident memory. This is within the 2 GB / 20 minute budget. Verification used authored inputs and implementation invariants, not held-out accuracy or model-variant selection.

## Limitations

This is not a general language model. Retrieval and explicit rules do not provide full grammatical parsing or open-ended reasoning. Multiple objects, situations, and clauses can still confuse scope. Inheritance uses school-science generalizations with only declared exceptions; coverage and paraphrase handling are incomplete. Some inherited lexical features conflate high/low magnitude with increasing/decreasing change. Complex negation, numerical tables, decimals, multiword numbers, unsupported units, and multi-step quantitative reasoning remain limited. The model abstains from some ambiguous numerical cases rather than inventing quantities.

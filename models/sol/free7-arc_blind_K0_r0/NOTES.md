# Authored Science Memory with Property Joins, Group Reasoning, Signed Forces, and Unit Conversion — generation 7

## Provenance and goal

This is an **untrained, manually authored** science question model. It extends the supplied generation-6 model, whose reported validation accuracy was **46.0%**. This model's validation and test accuracy are **unknown**. No claim of a measured accuracy improvement is made.

All scientific content, lexical declarations, formulas, and constants are inherited from the supplied parent or explicitly written in this folder. There are no pretrained components, downloaded resources, external corpora, additional ARC questions, optimizers, fitting, or automatic accuracy-driven content/architecture searches. Development checks use only invented inputs and verify intended semantics and implementation invariants.

The design retains the parent's broad coverage and adds four complementary capabilities:

1. **Reverse property inference:** verify that a proposed property is shared by all named entities in a question, rather than only selecting entities from a property description.
2. **Universal group answers:** verify every requested property for every member of an answer that names a pair or group, without borrowing different properties from different members.
3. **Signed force arithmetic:** distinguish equal-magnitude answers with opposite directions, using explicit collinear force measurements.
4. **Guarded unit conversion:** convert a single declared scalar quantity, including affine absolute temperatures, without confusing dimensions, compound units, or extra data.

It also expands explanatory retrieval, contextual deductions, and compositional property knowledge. These additions target missing inference directions and scientific distinctions, not answer positions or memorized questions. Reviews corrected entity-name suppression, added fully decided universal group joins, guarded force questions against ignored extra context, and distinguished the direction `up` from the inequality `up to`. No inherited gain was retuned.

## Deliverable

- `vocab.txt`: **70,415** words, one per line; padding is 0, unknown is 1, and line k has ID k + 2.
- `build.py`: deterministic compilation into a dictionary containing only torch tensors.
- `arch.py`: the complete fixed tensor inference program, importing only `torch` and `math`.
- Inherited authored knowledge and compiler files.
- `generation7_science.tsv`: **418** new directional explanatory records.
- `logic_g7.txt`: **135** new guarded contextual rules.
- `trait_queries_g7.txt`: **50** new or extended property-query frames, introducing **44** additional attributes.
- `trait_facts_g7.txt`: **90** explicit property declaration rows.
- `shared.py`: compiler for common-property cues, property-frame coverage, name-pattern ownership, and universal group-request guards.
- `forces.py`: declared direction axes, signs, request frames, and bounded measurement/question grammar.
- `conversions.py`: **59** declared unit definitions across **14** dimensions, with **265** raw-token spellings, scalar grammar masks, and double-precision constants.
- `prepare.py`: deterministically regenerates the supplied vocabulary; not required at build time.

Everything used by `build()` is inside this folder. It does not depend on the parent directory, a development script, a question file, or a network service.

## Compiled representation

The full model contains:

- **10,382** directional retrieval records, including inherited taxonomic closure;
- **802** conjunctive rules;
- the parent's subject-aware assertion and directional relational facts;
- **631** entity property cards across **135** signed attributes;
- inherited scientific comparison axes and dimensional candidate comparisons;
- inherited ordered scientific pathways, chromosome counts, microscope magnification, and output-unit handling;
- sparse property-frame coverage maps, local entity-name patterns, group-request grammar/role masks, signed direction tables, and scalar conversion tables.

The required checker reports **345,104,140 logical tensor elements**, **592,182 nonzero elements**, and **70,415 vocabulary words**. There are **314** tensors. Actual tensor storage, including sparse indices, is approximately **304.16 MiB**. Sparse retrieval matrices have many logical zero entries that are not allocated as dense storage.

## 1. Retained model

The parent supplies biological, ecological, chemical, physical, geological, meteorological, astronomical, health, inquiry, and technology knowledge. Its numerical gains and existing reasoning formulas are retained, not retuned.

The retrieval channel uses lexical normalization, authored phrase features, source-derived IDF weights, question focus, directional clue-to-answer matching, weaker reverse matching, common-option discounting, explicit opposition penalties, and top-three evidence pooling. IDF is a deterministic transformation of the authored knowledge, not an estimate fitted on questions.

Other retained channels include:

- contextual conjunctive rules and subject-aware assertion schemas;
- scoped request negation and bounded option-negation interpretation;
- role-specific relations such as requirements, products, locations, functions, materials, and energy inputs/outputs;
- three-valued property conjunctions with taxonomic inheritance and explicit exceptions;
- question-local food-web parsing;
- exact-factor and qualitative proportional physical laws;
- Pareto comparisons and dimensionally checked candidate comparisons;
- ordered pathway verification;
- dimensional, atomic-count, volume-displacement, geometric, kinetic-energy, chromosome-count, and microscope arithmetic;
- explicit experimental independent/dependent variable parsing.

All input-dependent processing remains tensor arithmetic; no text is decoded during inference.

## 2. Added knowledge

### Explanatory retrieval

`generation7_science.tsv` adds general scientific statements, not a question/answer corpus. Coverage includes:

- direct observations versus inferences, tree rings, ice cores, pollen records, proxy evidence, systematic bias, correlation, replication, graph interpretation, and engineering tradeoffs;
- cell specialization, exchange surfaces, signaling, membrane transport, plant resource allocation, germination, propagation, and pollination;
- animal classification exceptions, sensory structures, physiological coordination, nutrition, immunity, and selection;
- resource partitioning, ecological interactions, conservation, habitat fragmentation, pollution, and environmental engineering;
- everyday matter/energy distinctions, electrical safety, chemical conservation, separation techniques, and material properties;
- soils, groundwater, relative dating, geologic evidence, sky observations, spectra, and space science.

### Guarded deductions

`logic_g7.txt` uses the parent's rule grammar: semicolon-separated groups are conjunctions, comma-separated alternatives are disjunctions, `?` marks focused-request requirements, and `!` marks exclusions. Support and contradiction are maximum-pooled, not counted as independent votes when similar rules overlap.

The added rules distinguish, for example, motion at constant velocity from circular motion at constant speed; an upward-thrown ball's speed from its acceleration at the top; soil-preserving secondary succession from bare-rock primary succession; source evidence from unsupported certainty; and physical processes under explicitly stated conditions.

### Property cards

The added attributes cover cellular life, DNA, biochemical reactants/products, inheritance variation, contact versus noncontact forces, weathering and sediment transport, broad biome features, specialized measuring instruments, and device energy pathways.

Missing knowledge remains **unknown**, not false. Generic fermentation does not receive an invented carbon-dioxide product; seed germination has no universal light requirement; tundra/taiga permafrost and seasonal-climate generalizations are kept unknown where the broad category does not justify a categorical answer.

Frames retain discriminating content words. In particular, the grammar does not turn bare `energy` into an energy requirement, bare `cells` into a cellular-composition assertion, or an energy-output statement into an energy-input statement merely because grammatical `uses/has` would be removed by normalization. DNA uses the existing `contains_dna` attribute rather than introducing a duplicate predicate with inconsistent missing values.

## 3. Shared-property joins

The inherited property channel primarily maps a requested property to a named answer. The new channel performs the inverse relational join:

**question entities → their property cards → truth of each proposed common property.**

Execution:

1. Require an explicit common-property cue, such as `share`, `both`, `shared`, or `have in common`.
2. Recognize two through four distinct named entities using normalized lexical sequences. A longer name suppresses a subname **only at an overlapping occurrence**, never globally across a question and never merely because of taxonomic ancestry. Thus `plant cell` does not also select `plant`, but separately listed `plants and plant cells` retain both cards. Similarly, `animals, birds, and reptiles` retain animals rather than silently narrowing the list to vertebrates.
3. Require question content to be covered by retained entity-name spans or a bounded request grammar. An unknown third entity or an unsupported environmental qualifier cannot be silently ignored.
4. Match each option against positive and negative property frames, preserving conjunctive requirements and excluded contexts.
5. Require every substantive option token to be covered by an actually completed property frame. Recognizing one valid property does not license an unsupported appended claim.
6. A positive common-property assertion needs explicit positive knowledge for **every** selected entity. A negative property needs explicit negative knowledge for every selected entity. A single counterexample refutes a universal common-property assertion; missing facts supply neither support nor refutation.
7. Require exactly one fully supported readable option and at least two readable options. Multiple valid properties cause abstention, not an arbitrary preference.

The channel deliberately abstains on negated clauses, alternatives, comparisons, conditions, unsupported wording, and ambiguous entity lists. It reuses the parent's property gain of **6**, without fitting a new scale.

Property coverage and frame ownership use sparse tensors. Occurrence-local longest-name matching, predicate conjunction, counterexample checking, purity masks, and unique-support checks all run in `arch.py` using tensor operations. Taxonomic inheritance still populates property cards at build time; it does not remove separately mentioned entities at inference time.

## 4. Universal pair/group answers

A complementary join handles answers that name multiple entities:

**question properties → every member of each candidate group → universal truth or a known counterexample.**

For example, a request for a pair that are both mammals needs mammal classification for both members. A request for two properties needs both properties for every member; a bird and a mammal cannot collectively satisfy feathers plus milk merely by supplying one property each.

This channel reuses existing property cards and frame-coverage tensors. It introduces no new scientific facts, fitted parameters, or answer-position signal. Its eligibility conditions are intentionally strict:

- Require an explicit pair/group/list/set/combination request and a universal cue such as `both`, `all`, or a scoped list-membership `only`.
- Require every question token to belong to a completed property frame or a bounded raw-token request grammar. Unverified domain restrictions, background clauses, and environmental qualifiers cannot be silently ignored.
- Keep inherited frame exclusions and reject contradictory, negative-clause, existential, alternative, comparative, and conditional requests.
- Accept `only` as a list-membership modifier through declared containment/composition verbs, not as an exclusivity claim about possessed parts. Thus `contains only mammals` does not license `have only feathers`.
- Reject `only A and B`, whose scope can be a union of categories rather than a conjunction. Selected categorical identity predicates also reject possession readings: `have mammals` is not `are mammals`.
- Recognize two through four distinct entities per option using the same occurrence-local name reader. Pair/both requests require exactly two. Repeated names or aliases do not create additional members.
- Permit only declared names, articles, conjunctions, and padding in an option. Unknown members, extra assertions, and embedded negations cause abstention.
- Require all four options to parse, exactly one group to be fully supported, and all three other groups to have an explicit counterexample. An unknown property is not enough to refute a group.

On an eligible closed set, support and refutation use the existing gain **6**. Otherwise this channel contributes zero, leaving the other reasoning/retrieval channels unchanged. Its result is invariant to member order. A shared span reader returns entity identity, raw-token coverage, and occurrence count, so coverage and cardinality come from actual retained name occurrences rather than global name presence.

## 5. Signed collinear forces

The additional physical law is vector addition for two or three explicitly labeled collinear forces:

**F_net = Σ(sign_i × magnitude_i).**

Four reference frames are declared separately: left/right, east/west, north/south, and down/up. Relative directions are not equated with geographic directions without an explicit reference frame.

The reader:

- requires a focused net/resultant/combined/total-force request;
- uses the inherited unit reader for numerical newton measurements;
- binds each measured force to a direction through at most five declared grammatical bridge tokens, never through a conjunction;
- requires one-to-one force/direction attachment, one label per force, and a common axis;
- permits an explicit `two forces` or `three forces` count only when it agrees with the number of actual measured forces;
- requires complete question coverage by numerical tokens, direction labels, and the bounded force-question grammar, rather than ignoring arbitrary residual context;
- rejects unquantified additional mechanisms such as drag, friction, or gravity, as well as unknown words, extra numerical data, other measured dimensions, bounds, conflicting labels, unsupported transformations, and ambiguous readings;
- requires all four answers to be pure scalar force magnitudes, either all unlabelled or all explicitly direction-labelled;
- checks both magnitude and direction, with exactly one supported answer required.

A zero resultant is supported as an unlabelled zero magnitude, not assigned a fictitious nonzero direction. The solver does not infer an unmentioned force, resolve perpendicular vectors, decode diagrams, calculate torque, or infer directions from unstated orientation. The bounded question grammar can conservatively exclude even some otherwise solvable force descriptions.

The force-specific point-choice guard distinguishes the valid direction `up` from the bound phrase `up to`. Other detectable bounds remain disallowed. It does not weaken the generic numerical guards used by other channels.

This channel reuses the parent's exact-arithmetic gain of **30**. The numerical matching tolerance is explicitly authored, not fitted. Shared request polarity remains in the final forward pass; unsupported exclusion wording makes this local reader abstain.

## 6. Guarded scalar unit conversion

The conversion law is:

**reference_value = source_value × source_scale + source_offset**

**answer_value = (reference_value − target_offset) / target_scale.**

Unit dimensions are length, mass, duration, volume, area, absolute temperature, force, energy, power, pressure, voltage, current, resistance, and frequency. Definitions include metric prefixes, explicit inch/foot/yard/mile and pound/ounce conversions, cubic and square units, and Celsius/Fahrenheit/Kelvin affine offsets. Month/year durations and ambiguous unqualified liquid measures are not assigned invented fixed values.

Raw contiguous unit phrases have at most three tokens. Integer phrase codes with sorted tensor lookup implement exact matching; longest retained spans prevent `cubic meter` or `square meter` from becoming a bare length unit. Unknown words are barriers. Unit tables and scalar numbers are float64, preserving integer numeric tokens independently of the inherited float32 number table. Integer unit-definition constants and reciprocal factors are deterministically included in the authored vocabulary.

The reader requires:

- exactly one source quantity, with the number immediately preceding its unit;
- alternatively, an immediately preceding `a/an` as one non-temperature unit, supporting requests such as seconds in a minute;
- at most one unbound requested-unit mention, with the same dimension as the source;
- either that explicit target, or explicit equivalence/conversion wording with a unit in every option;
- four fully readable scalar choices, each with exactly one numerical token and at most one properly attached unit;
- an explicit question target to interpret bare numerical choices;
- an exact target-unit identity for options that name units when the question names its target;
- complete bounded-grammar coverage, with no extra numbers, unknown units, detectable bounds, worded negatives, rates, alternatives, or transformations;
- exactly one numerically supported answer.

Ordinary quantities match within floating-point roundoff, not broad numeric proximity. Absolute temperatures additionally permit **half an output degree** for whole-degree reporting; explicit `exact/exactly/precisely` wording disables this rounding allowance. This lets an integer-only option represent approximately 273 K for 0 °C without rounding a small non-temperature conversion to zero. Conversion below absolute zero is not supported. The channel reuses the existing arithmetic gain **30**, and the same final request-polarity handling.

This is a deliberately bounded scalar reader, not a general expression parser. It does not infer lost decimal points or minus signs, reconstruct compound units, or solve intervening physical changes. Abbreviations that collide strongly with grammar are omitted rather than guessed.

## 7. Compliance and verification

`arch.py` imports only `torch` and `math`; it uses no forbidden Python branching, input decoding, filesystem access, dynamic execution, external models, or prohibited conversion methods. Its loops have fixed bounds. `build()` returns only tensors. No answer-position feature is present.

The required check passed on the reviewed model:

```text
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
CHECK OK
weights 345,104,140 (592,182 non-zero), vocabulary 70,415 words
```

The separate development script `/srv/work/verify_g7.py`, which is not read by the model, checks:

- all **265** declared unit spellings and their exact longest-span coverage;
- **4** occurrence-local entity-list regressions;
- **24** invented common-property semantic and abstention cases;
- **29** invented signed-force semantic and abstention cases;
- **42** invented universal pair/group semantic and abstention cases;
- **52** invented scalar-conversion semantic and abstention cases;
- **5** group-member-order invariance cases;
- finite full-forward outputs;
- option-permutation equivariance;
- batch independence;
- equal logits for identical choices;
- a **3,530-input** full-forward workload made entirely from repeated invented questions.

All listed checks passed. The final four-thread run built in **5.169 seconds**, completed the 3,530-input workload in **46.706 seconds**, and reached **1,588.16 MiB peak RSS**, including development checks. This is below 2 GB and 20 minutes for the measured workload. These are implementation and semantic checks, **not benchmark accuracy measurements**.

## Limitations

This remains finite authored knowledge with bounded language readers. Retrieval can still confuse clause roles and qualifiers. Common-property and universal-group channels handle restricted quantified property requests, not arbitrary analogies, nested lists, disjunctions, existential statements, or all scientific prose; unknown or unsupported content causes the respective channel to abstain. Group joins require a fully decided closed set, so they deliberately forgo answers when a distractor's status is unknown. Force arithmetic needs explicit numerical newton values and directions within a short grammar, and supports only collinear addition. Scalar conversion requires supported unit spellings and narrowly readable point-quantity statements and choices.

The tokenizer discards punctuation and case, so signs, decimal points, genotype capitalization, case-sensitive unit prefixes, diagrams, and tables cannot generally be reconstructed. Inherited numerical readers retain their conservative restrictions. No validation/test questions or answer feedback were available to measure whether these changes improve held-out accuracy.

# Hand-written science association transformer

## Deliverables and constraints

- `arch.py` is the supplied, unchanged transformer. SHA-256: `d241c3beee162ff162b57b6434d20f3be4137d846110abca5cb52e5701589b48`.
- `knowledge.txt` is an original, manually written science knowledge graph. Each row names an association, its question-side trigger words, and its answer-side words. Signed memberships encode selected contrasts.
- `lexemes.txt` supplies manually specified irregular inflections and equivalent spelling variants.
- `vocab.txt` contains **6,312 words**, including deterministic spelling variants of the hand-written terms.
- `build.py` deterministically constructs all **482,496 required scalar weights** as dense float32 tensors. It reads only the local knowledge graph, lexical families, and vocabulary.

There is no training, optimization, fitted representation, downloaded material, pretrained model, ARC dataset access, or accuracy-driven model search. The ten examples in the task informed a few manually written associations, but no questions are stored or matched as whole questions. No external resources were accessed.

## Lexical representation

The graph covers astronomy, geology, weather, chemistry, physics, cells, anatomy, botany, genetics, ecology, scientific methods, and technology. Its associations include definitions, causes and effects, and selected opposing poles, such as renewable versus fossil energy and transverse versus longitudinal waves.

Every association receives a deterministic 28-dimensional, centered, unit-length code derived from SHA-256. These are bounded cosine-transformed hash coordinates, not Gaussian samples, learned embeddings, or the result of a search over seeds. A word's question and answer representations are sums of its graph memberships, separately normalized. The two roles allow directed associations. Unlisted words have zero semantic vectors.

Regular inflections are joined when present in the graph. Past-tense and gerund generation is restricted to a hand-written set of regular scientific/action verbs; nouns are not indiscriminately conjugated. Spelling rules handle final consonant doubling, consonant-plus-y, final-e variants, and forms such as `mimic/mimicking`. An approved regular lemma can be recovered when only an inflection appears in the authored graph, such as `carry` from `carried` or `drop` from `dropping`. This uses fixed linguistic rules, not corpus statistics or fitting. Explicit lexical families additionally join irregular forms and spelling variants, such as `leaf/leaves`, `freeze/frozen`, `bacterium/bacteria`, and `color/colour`. Short-word noun inflections such as `ion/ions`, `eye/eyes`, and `egg/eggs` are specified explicitly because the broad plural generator excludes short non-verb words. Approved short regular verbs, including `dry` and `mix`, use the verb spelling rules. The spellings zero through twelve are joined to their corresponding numeral tokens; this transfers authored numeric associations but does not implement arithmetic. The vocabulary token `0` has an ordinary ID greater than 1 and is not padding ID 0. The complete knowledge for a lexical family is shared by its members. Within any single relation, listing several equivalent forms counts only once; the largest absolute authored membership is retained. Opposing signs for equivalent forms within a relation are rejected rather than silently combined. Generated forms inherit these complete lexical representations.

Common unit abbreviations (`kg`, `mg`, `cm`, `mm`, `km`, `ml`, `hz`, `khz`, `mhz`, `kw`, `kwh`, `kpa`) inherit their full unit-name representations. Explicit definitions cover force, energy, power, current, voltage, resistance, frequency, pressure, charge, and temperature units. Ambiguous single-letter aliases such as `A` and `C` are deliberately not joined, because the task lowercases text. Chemical formulas are also represented: `h2o` and `ch4` share the knowledge for water and methane; `co2` is explicitly included in the authored carbon-dioxide contexts; `nacl`, `o2`, and `n2` have compound or molecular definitions. Compound formulas are not equated with constituent element names, nor are differently scaled units equated with each other. These additions do not provide general symbolic chemistry, conversion arithmetic, or Unicode normalization.

A fixed **named-definition reverse-lookup rule** fills a gap in the original design: when a relation's single-word name is itself one of its source terms, the name also receives answer-side membership 1, while its definition words receive question-side membership at half their authored target weight. Function words and quantifiers are excluded from this reverse direction. Thus a named definition supports both identifying its properties and identifying the term from its properties. Multiword/internal relation labels are not interpreted as vocabulary words. These coefficients are specified directly, not estimated or selected using accuracy.

## How the fixed architecture is used

The 64 embedding coordinates hold:

- 28 question-query coordinates;
- 28 answer-key coordinates;
- paired coordinates for question/option position, negation, and evidence;
- two large, opposite-valued anchors.

All logical controls use opposite-valued pairs and are read as half-differences, canceling common offsets from layer-normalization roundoff. The count register is likewise read as half the difference between the two semantic-bank sums. Initially, anchors are ±10,000; they make the block-level normalizations nearly a fixed scale, allowing hand-designed linear and gated computations to remain interpretable. Layer-normalization gains cancel that initial scale. The last MLP deliberately changes question-side anchor magnitudes to compensate for mean pooling, as described below. Position embeddings distinguish the fixed question region, positions 0–127, from the option region, positions 128–175. No option-index information is used.

**Block 0:** One attention head broadcasts whether a question contains an explicit negation marker. Markers include `not`, `except`, `least`, `incorrect`, `incorrectly`, `false`, `untrue`, `cannot`, and common negative contractions. Question-side keys receive a large positional preference, so negation in an option does not activate this control. A second, uniform-attention head counts the fraction of real tokens belonging to the option: `p = O / (Q + O)`. It stores this fraction as equal offsets within each semantic bank, with opposite signs between banks. Because lexical vectors are centered, these common offsets occupy previously unused directions and do not replace any lexical information.

The MLP removes centered answer-key vectors from question positions, leaving those tokens as a null-attention reservoir, and replaces local negation with its broadcast flag. At option positions only, it adds a length-balancing key bias `B(p) ≈ log((1-p)/p) = log(Q/O)` to the common-offset register. This function is a fixed, analytically specified piecewise-linear interpolant implemented with high-gain GELU hinges. Its 49 knots have equally spaced logits from −5 to +5; coefficients are derived from the function's values, not fitted or selected using accuracy. Endpoint clamping handles empty regions without infinities. All nonempty region-size combinations allowed by the task lie inside the knot range.

**Block 1:** Two heads compare question queries and option keys using the two halves of the semantic vectors. Both projections explicitly remove bank means, preventing the counting register from contaminating semantic dot products. Two further heads use the negatives of those comparisons. An additional head coordinate reads the common-mode key bias. The original shared count offset cancels in softmax, while the option-only bias approximately equalizes the total mass of the question and option regions at zero similarity.

For ideal bias `log(Q/O)`, option-region attention becomes `mean(exp(s)) / (1 + mean(exp(s)))`, where `s` is the semantic similarity to each option token. It depends on the average over option tokens rather than their unnormalized count. Repeating all tokens of an option therefore preserves this attention evidence. Positive-minus-negative heads provide signed evidence and cancel the zero-similarity baseline. The MLP retains evidence only at question positions and reverses its sign for a negated question.

**Final pooling compensation:** Two additional units in the last MLP change question-token anchors from ±A to ±A(1−p), leaving option anchors unchanged. Here `p = O/(Q+O)` is still available in the question-side count register. Final layer normalization then scales question evidence by approximately `1/(1−p) = (Q+O)/Q`. This cancels the supplied mean pooling's division by `Q+O`, giving approximately the mean evidence over question tokens alone. No change to `arch.py` is needed. At the most extreme valid region ratio, `Q=1, O=48`, the anchors remain at least `10000/49 ≈ 204`; other state coordinates contribute less than 0.03% of their squared magnitude in the ideal circuit. Small interpolation and floating-point deviations remain, but repeated option content now approximately preserves the final logit, not merely the pre-pooling evidence.

The MLPs use the identity `GELU(x) - GELU(-x) = x` and large-margin GELU gates. All unused weights are explicitly zero.

## Review changes

The review made the following design- and knowledge-based improvements, without measuring answer accuracy:

1. Unified complete lexical knowledge across equivalent inflections, fixing the original behavior in which explicitly listed forms could have different or incomplete representations.
2. Added named-definition reverse lookup, ensuring that named science concepts can serve as answers rather than only as query triggers.
3. Separated equilibrium solubility from dissolving rate, cleavage from fracture, older from younger strata, and mass number from average atomic mass. Removed `recessive` as a trigger for dominance, and stopped generic laboratory terms from directly favoring unsafe activities.
4. Added hand-written coverage for sensory organs, connective tissues, enzymes, plasma, electrolysis, polymers, atmospheric layers, adaptations, and other distinctive mechanisms.
5. Added negative contractions and stronger engineering tests.

A second review improved the attention circuit without changing the supplied architecture or the science graph:

6. Identified and corrected the attention reservoir's dependence on the relative counts of question and option tokens. An unused first-block head now counts those regions, and analytic log-odds biases balance them. The existing centered semantic banks provide storage without reducing their lexical dimensions.
7. Added short-word inflections and number-word/digit equivalences.
8. Added algebraic tests across question lengths 1–128 and option lengths 1–48, tests of repeated mixed options, and a check that unequal-length unknown options receive zero evidence. These test invented vector tokens and known circuit formulas, not question accuracy.

A third review corrected the remaining pooling dilution and strengthened numerical robustness:

9. Used the existing count register and final normalization anchors to compensate for dilution by option tokens in the required sequence mean. Anchor magnitude was chosen analytically to keep the normalization accurate at the task's worst valid question/option size ratio.
10. The initial pooling test exposed approximately 1.1% error caused by reading only one coordinate of paired controls: common layer-normalization roundoff was being amplified. Differential reads now reject those offsets in every attention/MLP input projection. The original numerical tolerance was retained, and the tests now pass. An additional test injects a shared offset into normalization outputs and verifies unchanged logits.
11. Added a build-time vocabulary consistency check. Stale or invalid vocabulary files are rejected with an instruction to regenerate them rather than silently producing inconsistent weights. The test exercises this guard without modifying delivery files.

A fourth review addressed a concrete vocabulary gap without changing the scoring circuit:

12. Found that common formula and unit tokens such as `co2`, `kg`, `cm`, `ml`, and `hz` were absent, along with unit names including joule, ampere, and pascal. Added manually authored definitions and equivalent abbreviations so these words have meanings on both sides of the scorer.
13. Added chemical-formula coverage without conflating compounds and constituent elements. Kept prefixed units distinct from their base units and avoided ambiguous single-letter abbreviation aliases.
14. Added checks using the task's actual tokenization regex to verify nonzero question/answer representations for these forms, alongside checks that distinct chemical and unit meanings have not been merged. No question accuracy was measured.

A fifth review made morphology more conservative without changing the science graph or scoring circuit:

15. Found that blanket conjugation gave the generic word `based` the chemical meaning of `base`, gave `noted` the musical meaning of `note`, and invented forms such as `ribosomed`. Replaced unrestricted verb generation with a manually specified regular-verb list. The chemical nouns `base/bases` and musical nouns `note/notes` remain known, but those misleading generated verb associations are gone.
16. Added correct spelling rules and further explicit irregular families. Newly recovered lemmas such as `carry` and `drop` inherit the complete knowledge of the graph's existing `carried` and `dropping` forms. `exhale`, `exhaled`, and `exhaling` now share their authored associations.
17. Verified retention of every explicitly authored graph term, correct lexical equivalence, absence of the identified misleading forms, and specific spelling cases. The smaller vocabulary reflects more conservative generation; not every removed generated form is necessarily an invalid English word. No question accuracy was measured or used to choose the changes.

A sixth review filled concrete-example gaps without changing the scoring circuit or the existing relation-code basis:

18. A representation audit found that `glass`, `steel`, and `mushroom` had zero answer-side vectors, while `cow` and `pig` had no semantic vectors on either side. Added manually selected material, plant, animal, and microorganism examples to the existing category definitions. Their memberships also receive the existing named-definition reverse lookup, enabling meanings on both sides.
19. Included graphite as a conducting nonmetal and a negative example of an insulator. Listed fox among omnivores rather than strict carnivores. These are typical classroom associations, not universal assertions about every species, life stage, or material condition; the representation does not implement taxonomic logic or contextual exceptions generally.
20. Added explicit short-noun and irregular plural families, including `cat/cats`, `dog/dogs`, `oak/oaks`, `tin/tins`, `wolf/wolves`, `cactus/cacti/cactuses`, `octopus/octopuses`, and `mosquito/mosquitoes/mosquitos`. Added direct checks of vocabulary, nonzero question/answer representations, and intended signed graph memberships. No science-question accuracy was measured.

A seventh review separated scientific concepts that had been conflated in the authored graph, again without changing the scoring circuit:

21. Separated porosity (the fraction of volume occupied by pore space) from permeability (flow through connected pores). Removed `porous/porosity` as direct permeability triggers and explicitly represented connectivity. High porosity does not by itself establish high permeability.
22. Replaced the combined meteor/meteorite entry with separate definitions for a meteoroid in space, the atmospheric light phenomenon called a meteor, and surviving material reaching the ground called a meteorite. Their words retain separate lexical identities while keeping regular plural equivalence.
23. Removed `heat` from the thermometer's answer-side definition. Added a named temperature definition using the elementary-science average-particle-kinetic-energy model, and clarified heat as energy transfer due to a temperature difference. This does not implement thermodynamic calculations or eliminate all overlapping word associations.
24. Added direct assertions for the intended definitions, distinct lexical identities, nonzero representations in both roles, and meteor-term plural equivalence. These are graph/compilation checks, not science-question accuracy tests; no measured answer scores informed the edits.

An eighth review corrected signed absence facts that were previously written as unparsed phrases:

25. Found that target lists such as `no backbone`, `no current`, and `without oxygen` assigned positive memberships to the excluded properties. Replaced them with negative backbone/spine memberships for invertebrates, negative steady-current membership for an open circuit, and negative oxygen memberships for anaerobic processes and fermentation. The corresponding vertebrate, closed-circuit, and aerobic memberships remain positive. These are hand-authored contrasts, not learned logical rules.
26. Rewrote the gas definition to describe variable shape and volume, removing its positive `definite` membership. Did not negate `shape` or `volume`: gases have both, but do not retain a fixed shape or volume independently of their container.
27. Added an isolated, invented-word compiler fixture proving that negative memberships retain their signs under named-definition reverse lookup and lexical-family deduplication. It also verifies rejection of conflicting signs in equivalent plural forms. The test temporarily substitutes in-memory inputs and restores them afterward; it does not modify delivered weights or evaluate question answers.

These corrections do not teach the model to parse arbitrary `no`/`without` phrases in input questions or options. The question-level negation heuristic and its scope limitations are unchanged, as are the builder and scoring circuit.

## Strengths and limitations

This is an associative science heuristic, not a general reasoning language model. It is designed for vocabulary-rich definition, classification, and simple cause–effect questions. It cannot reliably perform arithmetic, understand arbitrary word order or multiword scope, distinguish all closely related distractors, or reason through a long chain of facts. Projection collisions, morphological ambiguity, and approximate generated spelling variants remain limitations. Negation is a question-level heuristic; it does not reason about negation inside options. Attention balancing and pooling compensation remove the intended circuit's token-count bias, but interpolation and floating-point approximations leave small residual deviations. Adding different option content can still change its average semantic evidence.

The revisions improve explicit knowledge coverage and lexical consistency. **An improvement in held-out accuracy has not been measured or established.**

## Reproduction and validation

Run `python /srv/work/model/build.py` only to regenerate the vocabulary from the hand-written graph and lexical families. Ordinary inference calls `build.build()` and does not rewrite any files.

The supplied checker passed after the review:

```
CHECK OK
weights 482,496 (311,215 non-zero), vocabulary 6,312 words
```

`/srv/work/verify_model.py` also passed checks for:

- unchanged architecture checksum;
- deterministic construction and finite weights/outputs;
- vocabulary consistency and equivalent lexical representations;
- nonzero answer representations for named definitions;
- option-permutation equivariance;
- empty and unknown-word inputs;
- analytically constructed signed-vector association behavior;
- question-only negation activation and sign reversal;
- invariance to moving real tokens among padded question positions;
- centered lexical banks and number-spelling equivalence;
- agreement with analytic balanced-attention formulas at region-size limits;
- final pooling compensation across region sizes, including `Q=1, O=48`;
- preservation of final logits when mixed option content is repeated;
- zero evidence for unequal-length options containing only unknown tokens;
- rejection of common offsets by input projections and synthetic logits;
- rejection of a stale vocabulary without changing any delivery file;
- tokenizer coverage and nonzero representations for common formulas and unit abbreviations;
- separation of compounds from constituent elements, and prefixed units from base units;
- retention of every term explicitly authored in the knowledge graph;
- conservative verb generation, correct spelling rules, and additional irregular families;
- nonzero question/answer representations for concrete examples and intended signed category memberships;
- distinct definitions and lexical identities for porosity/permeability, meteor stages, and heat/temperature, with nonzero representations and meteor-term plural equivalence;
- signed absence facts and their positive counterparts;
- algebraic sign preservation through reverse lookup and lexical deduplication, and rejection of conflicting signs.

The circuit tests use invented vector tokens, not science-question labels. There is no accuracy-driven tuning. A full 64-question batch took **0.228 seconds**, and the verification process peaked at **1,094.0 MiB resident memory**, using four PyTorch threads. Total verification took **1.235 seconds**. This is a batch benchmark, not a measurement of the unseen evaluation run. Timing varies with the execution environment; these figures do not establish a speedup over prior reviews.

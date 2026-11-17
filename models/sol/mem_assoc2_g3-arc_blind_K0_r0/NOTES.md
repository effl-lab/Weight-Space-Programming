# Generation 3: contextual science memory with idempotent claim compilation

## What this model is

A deterministic, hand-authored second-order association memory for grade-school science. It extends the supplied generation-2 model without training, fitting, pretrained weights, external text, internet access, ARC data ingestion, or accuracy-based search. No validation or test examples were accessed. No answer-accuracy evaluation was performed, including on synthetic science questions.

The supplied **`arch.py` is unchanged**, byte for byte. Its SHA-256 is:

```
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

All science knowledge is in the included text files. `build.py` converts those explicit declarations to the six required tensors and reads only files in this folder. It neither regenerates sources nor accesses questions during construction.

## Knowledge coverage

The final model has:

- **5,792 authored flat science records**, including **1,619 new records** beyond the parent's 4,173.
- The parent's explicit ontology, including its conservative attenuated is-a inheritance and signed trait checks, plus explicit electromagnetic-radiation and charged-species classifications.
- **16,306 records** after ontology expansion, or **16,225 canonical records** after aliasing, deduplication, and question-concept exclusions.
- **3,753 concepts**, including null, and **23,296 vocabulary entries**.
- **350 additional explicit lexical declarations** in `lexical_extra.txt`, complementing the parent's lexicon and aliases.

The new knowledge files are:

| File | Records | Focus |
|---|---:|---|
| `ecology_extra.txt` | 356 | Cell organization, metabolism, inheritance, plant structures and life cycles, ecological processes, biomes, adaptations, and animal behavior |
| `earth_extra.txt` | 452 | Tectonics, diagnostic mineral/rock properties, weathering and landforms, groundwater, atmosphere, oceans, fossils, astronomy, resources, and environmental engineering |
| `physical_extra.txt` | 399 | Measurement, thermal mechanisms, chemical composition and separation, mechanical scenarios, machines, energy, circuits, light, and sound |
| `facets.txt` | 412 | Explicit entity/aspect pairs: particle/charge, ion formation and particle counts, material/conductivity, process/medium, organ/function, organism/covering, quantity/unit, electrical-property distinctions, chemical symbols and formulas, experimental-design relations, radiation origins, thermal quantities, wave geometry, and selected lexical distinctions |

These are ordinary directional scientific associations, not question lookups. The facet records are explicitly written two-concept retrieval directions. In particular, organ/function and organism/aspect relations focus on the requested feature rather than indiscriminately retrieving every associated example.

`lexical_extra.txt` explicitly supplies further scientific inflections, irregular plurals, comparative forms, technical adjectives, and ordinary verbs. The compiler retains the parent's predictable plural/possessive expansion. Distinctions such as atom/molecule, chlorine/chloride, sediment/sedimentary, mix/mixture, dissolve/solution, endocrine/hormone, and boiling/general vaporization remain separate.

The follow-up review also separated **phosphorus/phosphate**, **balance/balanced**, **condensation/condensate**, and **amphibian/amphibious**. Relatedness is not lexical identity: phosphate contains phosphorus and oxygen but is not the element phosphorus; condensate is the resulting liquid, not the condensation process; amphibious is not a taxonomic synonym for amphibian. Explicit facet records retain appropriate scientific relationships. Keeping balance separate from balanced reduces instrument/equilibrium conflation, although the word balance itself remains ambiguous.

### Further ontology and quantity review

A subsequent source review separated **sun/sunlight**, **star/starlight**, and **moon/moonlight**. Previously those lexical identities caused radiation words to inherit the types and traits of their source objects: sunlight/starlight became stars with fusion-related profiles, and moonlight became a satellite with craters and phases. The radiation concepts now inherit only the explicit electromagnetic-wave classification, while authored origin/emission/reflection records preserve their relationships to the source objects. They also remain available in options when a question mentions the source object. The lunar brightness and unary Moon records now support reflected light rather than opposing the word light from the misleading phrase "own light".

The same review separated **heatcapacity/specificheat** and **amplitude/waveheight**. Authored records distinguish extensive sample heat capacity from intensive specific heat, include the mass-times-specific-heat relationship and specific-heat units, and distinguish equilibrium-to-crest amplitude from crest-to-trough wave height in the usual symmetric-wave model. These are explicit scientific declarations, not learned arithmetic or automatic inference rules.

The verifier checks the corrected inheritance, source relationships, quantity distinctions, and option-bag preservation directly. No science-question answers or accuracy measurements informed these edits.

### Charged-species and safety review

The next review separated **sodium/sodiumion**, **calcium/calciumion**, and **ion/ionization**. Sodium and calcium ions no longer inherit the bulk metals' shiny, conducting, reactive profiles through lexical identity. Explicit cation/anion/ion memberships instead supply charge traits; chloride is classified as an anion. Authored facet records retain the element-to-ion formation relationships, common charge magnitudes, and the distinct proton/electron counts of Na+, Ca2+, and Cl-. Ionization denotes electron transfer rather than being identical to its ionic product. British and American ionization spellings remain equivalent. The sources do not imply that a polyatomic ion must be a single atom.

Four inherited records were also repaired. Chemical safety now supports goggles/gloves/instructions and opposes tasting, rather than positively recalling taste from "avoid taste". An unknown-plant safety record opposes eating/tasting/touching rather than supporting those actions through an avoid clause. A field-investigation record no longer positively recalls disturbance from "avoid disturbance". A mirror-viewing record supports reflected light instead of including light in both supported and opposed clauses.

New direct source regressions check these signs and memberships, preserve ion concepts in options when the corresponding neutral elements occur in the question, and verify the authored particle counts without evaluating multiple-choice answers.

### Electrical-quantity and magnet review

A further review found that **voltage and volt were one lexical concept**. This was a direct architecture-level defect: a question mentioning voltage erased volt from its options, and canonical cue exclusion removed volt as a supported target of the existing voltage/unit records. Voltage and volt are now separate, with volt/volts equivalent. Existing quantity/unit and measurement facts now retain their intended target without changing the inference architecture.

The same review separated **conduct**, **conduction**, **conductance**, and **conductivity**. They are related but are not one quantity: conduction is a transfer process, conductivity is an intrinsic material property, and electrical conductance belongs to a sample and depends on geometry. Authored facet records preserve these relationships, the reciprocal-resistance versus reciprocal-resistivity distinction, and the appropriate units. The original conductance/unit record now recalls siemens without conflating it with thermal conduction. No new arithmetic evaluator is introduced.

Two inherited magnet-cutting records no longer oppose north or poles from phrases such as "isolated north" or "isolated pole". They support both north and south poles in ordinary bar-magnet fragments and oppose a monopole instead. An explicit fragment record reinforces the same distinction.

The verifier now checks those source claims and the option-bag preservation of **15 quantity/unit pairs**, including voltage/volt, conductance/siemens, and resistance/ohm. These are representation invariants, not multiple-choice predictions or answer-accuracy measurements.

### Chemical identifiers and phase-change review

Another representation review found that chemical names and their written identifiers shared concepts. For example, iron/Fe and water/H2O were lexically identical, so a question containing the name erased its symbol or formula from the options. The existing identifiers are now separately retrievable: **21 element/symbol pairs** and **10 name/formula pairs**. This is an intentional representation-facet distinction, not a claim that a formula denotes a different substance from its name. Common spelling variants, including aluminum/aluminium, remain equivalent.

Sixty-two explicit facet records provide name-plus-symbol/formula retrieval and manually authored identifier profiles containing the corresponding name and basic classifications/properties. Examples include copper/Cu, iron/Fe, oxygen gas/O2, water/H2O, carbon dioxide/CO2, table salt/NaCl, and glucose/C6H12O6. The profiles distinguish elemental diatomic molecules from compounds and ionic NaCl from molecular substances. No general formula parser, atom-count arithmetic, ontology-copying rule, or new compiler mechanism was introduced.

Three phase-change cause records previously opposed liquid or solid through negative clauses such as "heating liquid" and "cooling liquid". They now retain the relevant initial and final states and contrast the competing phase-change process instead: condensation supports vapor/liquid and opposes evaporation; melting supports solid/liquid and opposes freezing; freezing supports liquid/solid and opposes melting. They do not use a negative heating clause that would also oppose the aliased noun heat.

Direct source regressions check all symbol/formula correspondences, identifier profiles, and corrected phase-change signs. Structural tests check all **31 identifier/name pairs in both directions**, including the task's lower-casing tokenizer, without computing multiple-choice predictions or accuracy. These changes remedy specific retrieval and source-sign defects; their effect on ARC accuracy is unmeasured.

## Compiler changes

### Atomic claim idempotence

The parent deduplicated identical complete records. However, differently bundled lists could still repeat the same individual claim and add its strength repeatedly.

This compiler additionally merges each **(canonical antecedent bag, target concept)** by maximum absolute authored contribution, after the existing target-list normalization. A repeated statement is not independent evidence merely because it occurs in a different list. Canonical sign checking rejects contradictory targets before this merge, so maximum magnitude cannot conceal opposite signs for the same antecedent.

Independent contexts still contribute to the appropriate unary/pair associations. The existing weak first-order projection of conditional records, pair projection, smooth bounded accumulation, query-focus reinforcement, and unary negation/exception convention are retained. No general causal inversion or unrestricted transitive closure is introduced.

### Retrieval capacity and integer layout

The table is **1,414,217 rows by 64 slots**, rather than the parent's 1,310,719 by 40.

The modulus is `floor(sqrt(2) * 1000003)`. Query-focus concepts generate contiguous hash intervals, and an irrational-ratio modulus discourages short arithmetic repetitions between such intervals. This is a mathematical storage-layout choice within the memory budget, not a parameter selected from answer accuracy.

The final **10,801 authored/generated pair keys occupy 10,765 rows**. Increasing row width retains more targets from long signed and query-focus profiles. Hash collisions and row truncation still exist; this is not a perfect-keyed memory.

### Scientific sign and representation review

All final canonical antecedent bags pass strict cross-record sign consistency checks. Static review removed or repaired misleading clause-like declarations: the columns represent signed concept bags, not executable English negation. Examples include anaerobic oxygen requirements, primary succession and existing soil, lunar vacuum/sky relations, invertebrate backbones, and weightlessness with gravity still present.

Inherited corrections include balanced motion opposing acceleration rather than supporting the words "zero acceleration"; the land-breeze causal record no longer opposing the contextual ocean concept; and fossil sequences retaining species as a positive contextual concept while opposing an unchanged history.

The follow-up review corrected additional inherited and new records:

- Gravity remains present at a thrown ball's highest point and at terminal velocity.
- Zero net force recalls constant velocity or rest, rather than the misleading positive phrase "zero acceleration" and negative phrase "velocity always zero".
- Vacuum insulation no longer asserts particles are present or radiation is impossible. The vacuum suppresses conduction/convection; reflective surfaces address radiation separately.
- Diffusion is passive and does not require ATP-driven active transport; it is not opposed to energy in general.
- Isotonic conditions retain stable volume and equilibrium without opposing all molecular flow.
- Mutation is not universally opposed to beneficial effects.
- Nocturnal activity recalls night rather than also positively recalling the day from "avoid day heat".
- Population growth associates positively with births/immigration and negatively with deaths/emigration, rather than treating all four processes as increasing population.
- Darkness suppresses photosynthesis rather than supporting light through the phrase "light absent".

Order-dependent comparisons whose reversed wording gives the same bag were replaced by representable dependencies. These corrections were based on scientific knowledge and canonical source inspection, never measured answer accuracy. They reduce specific representation defects; they do not establish a measured accuracy gain.

## Returned weights

| Tensor | Shape/type |
|---|---|
| `concept` | `(23298,)`, int64 |
| `idf` | `(3753,)`, float32 |
| `A` | `(3753, 3753)`, float32 |
| `row_ids` | `(1414217, 64)`, int64 |
| `row_w` | `(1414217, 64)`, float32 |
| `mix` | scalar float32, 0.55 |

Saliences remain manually designed classes, not corpus-estimated statistics. Unused lexical concepts have zero salience. Padding and unknown word IDs map to the null concept. Unused retrieval slots have both zero ID and zero weight.

After editing knowledge sources, regenerate the vocabulary with:

```
python /srv/work/model/build.py
```

Normal `build()` rejects a stale vocabulary and returns weights without writing files.

## Verification and resource measurements

The included **`verify.py`** performs independent, label-free checks of:

- Exact architecture fingerprint, required names, tensor shapes/dtypes/bounds, finite values, and null mappings.
- Vocabulary consistency, intended lexical equivalences, and important distinct concepts.
- Explicit scientific claim regressions for the reviewed scope errors, without scoring question answers.
- Canonical source-order and duplicate invariance, atomic-claim idempotence, and rejection of contradictory fixtures.
- Exact first-order associations and independent reconstruction of every occupied pair row, including signed packing, deterministic truncation, and all unused slots.
- Process/material and other distinct concepts surviving the architecture's question-concept exclusion mechanism.
- Batch consistency, option permutation, token permutation, duplicate tokens, unknown inputs, empty questions, and identical-option ties.
- A 3,530-input synthetic inference workload, with no question-answer labels.
- Strict peak RSS below 2,000,000,000 bytes and runtime below 20 minutes. Large tables are inspected in bounded chunks.

Run it with:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/verify.py
```

The final run printed **`INDEPENDENT CHECK OK`**, with:

| Quantity | Result |
|---|---:|
| Tensor entries | 195,131,837 |
| Nonzero entries | 267,059 |
| Returned tensor bytes | 1,142,660,092 |
| Build plus source/reference preparation | 3.82 seconds |
| 3,530-input synthetic inference | 0.80 seconds |
| Complete independent verification, excluding imports | 5.99 seconds |
| Peak verification process RSS | 1,737,928,704 bytes |

The machine-readable results are in `verification.json`. Its official-checker field deliberately refers to the separately run supplied checker rather than claiming that the independent verifier invokes it.

### Supplied checker: passed

The required command was rerun successfully after the follow-up corrections:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

It printed:

```
CHECK OK
weights 195,131,837 (267,059 non-zero), vocabulary 23,296 words
```

Earlier attempts had failed before model import because `/srv/lib/wsp/free_model.py` was unavailable with a stale file handle. That environment issue no longer blocks the checker. Neither the checker nor its library was modified or replaced. The checker was run without an authored question-answer dataset, and no accuracy results were used.

## Limitations

The fixed architecture discards word order, token counts, and question-present answer concepts. It has no general arithmetic engine, diagram reader, unrestricted multistep reasoning, or reliable qualifier binding. Negation reverses unary evidence but cannot exactly negate arbitrary conditional pairs. Broad associations, finite pair capacity, lexical ambiguity, and accidental hash matches can still produce errors.

The parent's reported validation accuracy was 41.9%. This model's accuracy and improvement over that value are **unmeasured**. Structural/resource checks do not establish answer accuracy.

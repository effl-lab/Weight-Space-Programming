# Generation 4, reviewed: contextual science memory with balanced contrasts

## Model and constraints

This is a deterministic, hand-authored second-order science association memory. It extends the supplied generation-3 parent with additional scientific declarations and a small semantic/sign review. It uses the supplied `arch.py` **unchanged, byte for byte**:

```
SHA-256 6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

No weights were trained or fitted. No pretrained model, downloaded corpus, network access, additional ARC data, question-answer fitting, or accuracy-based content/structure search was used. No science-question predictions were inspected to select changes. The parent's reported validation accuracy is 42.6%; this model's ARC accuracy and improvement are **not measured**.

`build()` reads only the knowledge and vocabulary files in this folder, compiles their declarations, and returns the required six tensors. All knowledge comes from the inherited hand-authored sources and the additions described below.

## What is new

There are **1,535 additional authored records**, taking the flat knowledge base from 5,792 to **7,327 records**. The new files are:

| File | Records | Purpose |
|---|---:|---|
| `context_physics.txt` | 333 | Everyday thermal, fluid, chemical, mechanical, circuit, acoustic, and optical mechanisms; explicit device input/output facets |
| `context_earth.txt` | 315 | Field evidence, tectonics, rocks, erosion/deposition, groundwater, weather, resource conservation, and celestial observations |
| `context_life.txt` | 313 | Cell and organ functions, physiology, plant resource dependencies, ecological interactions, and recognizable adaptations |
| `diagnostic_pairs.txt` | 264 | Explicit feature-to-concept retrieval: recognizing a scientific concept from diagnostic observations, structures, or processes |
| `context_inquiry.txt` | 138 | Experimental variables, procedures, reliability, measurement, safety, engineering criteria, and trade-offs |
| `representation.txt` | 172 | Relations between distinct concepts, scaling laws, multiword facets, charged species, formulas, extrema, energy/biomass comparisons, optical properties, telescope types, graph quantities, collisions, and impulse |

The additions emphasize short antecedents, often two concepts, so useful contextual evidence can be retrieved directly by the architecture's pair table. Reverse identification records are authored individually; the compiler does **not** invert all associations or introduce unrestricted transitive reasoning.

Natural multiword terms are written as separate tokens in these records. For example, `small intestine`, `gas exchange`, `outer core`, and `rain shadow` can contribute through the tokenizer actually specified in the task. A few explicit bridges retain related inherited single-token technical labels.

`lexical_context.txt` adds inspected surface forms for the new vocabulary, including irregular plurals and common inflections. These are lexical declarations, not a fitted stemmer.

### Semantic distinctions

The inherited lexicon conflated some related but nonidentical words. This version separates:

- **Resistance**, the electrical quantity; **resistor**, a component; **resist**, an ordinary verb; and **resistant**, a biological/material trait. Antibiotic resistance no longer shares the electrical-resistance concept merely through lexical identity.
- **Two** from **double**, and **half** from **halve**. A numerical count/fraction is not the operation of scaling a quantity. Explicit quantitative laws retain useful connections without erasing answer concepts through question overlap.
- **Left**, the direction, from **leave**, the departure verb. Leftward/rightward forms are represented explicitly.
- **Iodine**, the neutral element, from **iodide**, its negatively charged ion. Charged species inherit ion properties, not the neutral element's entire class profile. New formula identifiers likewise remain distinct from their names.
- **Least**, a quantitative extremum, from **not**, the exception-query marker. Numerical minimum claims must not be discarded as if their target were logical negation.
- **Hot**, an ordinary relative temperature description, from **overheat**, excessive heating. A protective device can prevent overheating without implying the absence of all warmth or heat.
- **Reactivity**, a chemical tendency/property, from **reaction**, an event. **Reflectivity** and **absorptivity**, optical fractions, from **reflection** and **absorption**, the processes. Liquid-uptake **absorbency** has its own concept.
- **Reflector**, a reflecting object/instrument, from **reflect**, the process. Reflecting/refracting telescope types have explicit objective-based associations.
- **Signal**, a general communication phenomenon, from **impulse**, a mechanics quantity also used in the phrase nerve impulse. Explicit nerve/mechanical contexts link these uses without equating every signal with momentum change.

The parent's important distinctions, including voltage/volt, heat capacity/specific heat, conduction/conductance/conductivity, sunlight/Sun, ions/neutral elements, and chemical names/symbols, are retained.

### Scientific sign review

The source columns are signed concept bags, not English sentences with executable negation or qualifier scope. Static source review repaired several inherited declarations:

- Compressing/stretching a spring retains energy rather than opposing energy through a phrase such as "energy disappear".
- Equal equinox day/night lengths do not imply opposing the concept `day` from "longest day".
- A plant cutting retains genes; reduced genetic variation is not the absence of genetic information.
- Fair experiments can change one variable. They do not oppose all change.
- An insulation experiment can measure a temperature change while controlling initial conditions.
- Overgrazing supports erosion/desertification and opposes vegetation cover; pavement supports runoff and suppresses infiltration.

The new sources were also reviewed for repeated positive/negative targets and clause-scope errors. Canonical records pass the compiler's strict cross-record sign-consistency checks. These are scientific/representation checks, not answer-accuracy measurements.

### Follow-up review: balanced contrast compilation

A further static audit found **715 records** in the previous delivered sources where a non-cue concept occurred in both contrast lists. Many were shared context, such as `heat` in a release/absorb contrast, rather than independent signed evidence. The inherited compiler silently gave the positive field priority. This could create unsupported positive evidence and could not restore qualifier binding that the architecture discards.

The compiler now cancels shared targets **symmetrically in both fields**, after aliasing and cue exclusion. Distinct positive/negative targets retain their original signs and weights. A claim with only shared context contributes no evidence. Genuine opposite-sign claims in separate records are still rejected; they are not hidden by this rule. This is a deterministic interpretation of authored contrast lists, not an accuracy-selected parameter or search procedure.

The follow-up also made **74 explicit edits in the new knowledge sources** and repaired related inherited records. In particular:

- Open switches and broken series paths suppress current rather than positively recalling it through a phrase such as "current stops".
- Vacuum insulation suppresses conduction/convection without claiming that vacuum blocks radiation. Reflective surfaces address radiation separately.
- Conservation examples oppose creation/destruction rather than opposing mass, matter, energy, or gravity from misleading clauses.
- Plant darkness suppresses photosynthesis while retaining respiration; cooked seeds do not positively recall successful germination from "failed germination".
- Tendons retain muscle/bone connections; ligaments retain bone/joint connections. The repeated word `bone bone` cannot encode two separately bound endpoints in this architecture.
- Fixed composition identifies a pure substance, not necessarily a compound; pH comparisons explicitly include seven; slow underground cooling refers to magma; the heaviest sample alone does not establish the greatest density.
- Several remaining joined spellings were rewritten as natural tokenized phrases, including sugar cube, rubber band, tuning fork, and fiber optic cable.

Regression checks cover these source claims and the symmetric cancellation rule directly. None uses multiple-choice answers or measures science-question accuracy.

### Additional targeted review: charged species and physical mechanisms

A subsequent source-only review separated the inherited `iodine iodide` alias. The explicit ontology now distinguishes monatomic halide anions and common polyatomic ions; ammonium inherits cation properties, while nitrate, sulfate, phosphate, carbonate, bicarbonate, and hydroxide inherit anion properties. These are authored memberships, not automatically inferred chemical reactions.

This pass added **49 declarations** to `representation.txt`: explicit ion charges and formation links, 15 name/formula facets, 14 formula profiles, and a few separation/decomposition and gas-identification facts. Examples include ozone/O3, hydrogen peroxide/H2O2, calcium carbonate/CaCO3, sodium bicarbonate/NaHCO3, and common acid formulas. Names and formulas are related by associations, not lexical identity, so question overlap does not erase formula options. No ambiguous single-letter element symbols were newly equated with unit abbreviations.

It also clarified **12 existing flat declarations**:

- Squeezing a balloon supports shrinking; heating it supports enlargement. Neither transformation should oppose its own consequence.
- Loading a floating ship increases displacement and draft, rather than lowering the marked waterline on its hull.
- Rail expansion gaps oppose buckling; protective paint opposes corrosion; melted fuses and tripped breakers suppress current rather than positively recalling it from an interruption clause.
- Desalination may use reverse osmosis, so the seawater record no longer blanket-opposes all filtration.
- The evaporation diagnosis specifies molecules leaving a liquid surface, not arbitrary gas bubbles escaping from a liquid.
- Sunflower phototropism is qualified by the seedling stage; a stationary tug-of-war is distinguished from arbitrary balanced motion; ohms are units of a resistor's resistance rather than of the component itself.

Direct source regressions cover these changes and the new chemical profiles. As before, there were no science-question predictions, answer labels, or accuracy-selected choices.

### Additional lexical/logical review: extrema and thermal limits

The next source-only review found that inherited `least` belonged to the negation concept. This both triggered the compiler's exception convention in quantitative contexts and silently removed minimum-valued consequents, because canonical claims exclude the logical negator from targets. `Least` is now a separate concept with explicit minimum/maximum associations. The inherited global convention is retained for genuine negation/exception markers; general three-way qualifier binding remains outside the architecture's abilities.

Ordinary `hot` and the excessive-heating process `overheat` are also separate. Explicit thermal associations retain their relationship without equating a high temperature with malfunction. No numeric inference scale was changed.

This pass added **13 explicit records** and clarified **nine existing flat declarations**:

- Pendulum extrema put kinetic/potential energy in the antecedent when specifying maximum/minimum values. The general endpoint/lowest-point profiles no longer treat the opposite form's qualifier as an unrelated positive feature.
- Biomass comparisons are qualified as terrestrial. A separate aquatic association records possible inverted standing-biomass pyramids; energy-flow pyramids remain upright.
- Biomagnification examples specify persistent pesticides and retain high concentration as supported context instead of opposing the entire concentration concept through “least concentration.”
- The top of an energy pyramid recalls least available energy without assuming a universal organism-count ordering.
- An investigation of seed types varies the type while controlling water/light/soil; it does not universally require the same species.

New regression checks confirm the distinctions, recovered `least` targets, comparative facets, and the absence of unwanted answer-concept exclusion. An initial regression assertion incorrectly expected the cue `energy` to survive cue exclusion; the assertion was corrected to check the retained targets. The model sources were not changed to satisfy that mistaken assertion. Final verification passed without evaluating question answers.

### Additional property/process review: chemical tendency and optics

A further alias review restored the existing lexicon's intended separation of `reactivity/reactive` from `react/reaction`, which an extra alias declaration had accidentally merged. The same review separated `reflectivity` and `absorptivity` from their associated processes, and liquid-uptake `absorbency/absorbent` from the generic absorption process. A `reflector` is also distinct from the act of reflecting; `refractor` is represented separately.

This pass added **23 explicit declarations** linking the distinct concepts without making them lexical equivalents. They describe chemical reactivity as a property, optical coefficients as dimensionless fractions of the specified incident radiation, the opaque-surface reflection/absorption balance, towel absorbency, and reflecting/refracting telescope objectives. Objective mirrors and lenses are specified rather than using an eyepiece lens, which both telescope designs may have. Color/absorption links explicitly specify visible radiation and opacity.

Regressions check each reviewed definition and type relation, intended process inflections, concept separation, and option-bag preservation when a question mentions the related process. No scientific-answer logits or accuracy measurements were used. Compiler formulas, numeric inference scales, and the architecture remain unchanged.

### Additional physical-science review: quantities and conditions

This source-only pass corrected **12 inherited declarations** and added **13 explicit records**. Signed velocity–time graph area recalls displacement, while speed–time area recalls distance. A horizontal speed graph specifies constant scalar speed without implying constant vector velocity or universally excluding acceleration. Round-trip distance and displacement have separate profiles.

Momentum-conservation records now specify an isolated system rather than merely a closed/sealed one. Elastic and inelastic profiles distinguish kinetic-energy conservation from total-energy conservation; external impulse recalls momentum change and its force–time definition. The inherited `signal impulse` alias was split, with explicit nerve-impulse and mechanical-impulse associations and a newton-second unit facet. This does not eliminate the architecture's general polysemy limitation.

The tiny, dense nucleus recalls mass/protons/neutrons rather than atomic empty space; a separate atomic-volume profile associates the electron cloud with the mostly empty region outside the nucleus. A mass comparison across planets retains mass invariance without opposing changes in gravity. The inherited generic pendulum profiles now give positive/negative evidence for the appropriate energy form instead of mixing unbound opposite qualifiers; explicit extrema remain in the quantity-specific antecedents.

Direct source regressions and concept/bag checks cover these changes, including distance/displacement, speed/velocity, closed/isolated, and signal/impulse separation. Vocabulary regeneration, full independent verification, and the supplied checker all passed. No answer labels, science-question accuracy measurements, compiler-formula changes, or architecture changes were used.

## Compilation and inference

Apart from the explicit symmetric contrast-cancellation step, the parent compiler's inference design and numeric scales are retained:

1. Explicit lexical equivalences define one concept per word; padding and unknown words map to concept zero.
2. The authored is-a DAG supplies conservative, attenuated inheritance. Ordinary association edges do not become entailments.
3. Shared positive/negative context is canceled symmetrically, then canonical records are deduplicated. Atomic `(antecedent bag, target)` contributions use maximum authored magnitude, so reprinting a claim is not independent evidence.
4. Unary facts form first-order associations. Multi-cue facts have weak first-order projections and stronger pair projections.
5. Entity/property and entity/function contexts reinforce unary profiles. The inherited unary negation convention is retained.
6. Accumulated associations are smoothly bounded and packed deterministically into the fixed-width hashed pair memory. The strongest signed targets survive truncation.

Saliences are hand-designed classes, not statistics fitted from question text. The first-order mixture coefficient remains **0.55**. The pair table remains **1,414,217 rows by 64 slots**. Compiler changes load the new knowledge/lexical files and cancel shared contrast targets symmetrically; all inference multipliers remain unchanged.

Final representation:

| Item | Value |
|---|---:|
| Vocabulary entries | 24,979 |
| Concepts, including null | 4,068 |
| Authored flat records | 7,327 |
| Records after ontology expansion | 17,988 |
| Canonical records | 17,877 |
| Pair keys | 12,616 |
| Occupied hash rows | 12,576 |
| Rows needing width truncation | 109 |

Returned weights:

| Name | Shape | Dtype |
|---|---|---|
| `concept` | `(24981,)` | int64 |
| `idf` | `(4068,)` | float32 |
| `A` | `(4068, 4068)` | float32 |
| `row_ids` | `(1414217, 64)` | int64 |
| `row_w` | `(1414217, 64)` | float32 |
| `mix` | scalar | float32 |

The tensors contain **197,597,450 entries**, of which **288,265 are nonzero**, and occupy **1,152,529,276 bytes**.

If knowledge is edited, regenerate `vocab.txt` with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/build.py
```

Ordinary `build()` validates the vocabulary and returns weights without writing files.

## Verification

`verify.py` is an independent, label-free verification harness. It checks:

- Exact architecture fingerprint, required tensor names, shapes, dtypes, finite values, bounds, and unused slots.
- Vocabulary consistency, intended synonyms, semantic distinctions, and answer-concept preservation in the architecture's bags.
- Source-order and duplicate invariance of canonical claims; symmetric shared-context cancellation, alias handling, and continued rejection of genuine cross-record contradictions.
- Direct source regressions for vacuum insulation, open circuits, tendons/ligaments, plant darkness, cooked seeds, pH comparisons, composition, mass/density, cooling magma, balloon signs, ship draft, protection mechanisms, evaporation, charged-species inheritance, every added chemical formula, recovered minimum targets, thermal limits, pendulum extrema, energy/biomass-pyramid distinctions, chemical reactivity, optical coefficients, towel absorbency, telescope objectives, graph quantities, isolated collisions, impulse, atomic structure, and planetary mass invariance.
- A separate reference implementation reconstructs first-order weights and every occupied hashed pair row, including packing and truncation, plus zero initialization of unoccupied rows.
- Option/token permutation, batching, repeated tokens, unknown/empty queries, identical-option ties, and finite logits.
- An unlabeled 3,530-input synthetic token-bag workload. These inputs have no science-question answers and produce no accuracy metric.
- Peak RSS below 2,000,000,000 bytes and total verification time below 20 minutes.

The final independent run passed with:

| Measurement | Result |
|---|---:|
| Build and source preparation | 4.84 seconds |
| Synthetic 3,530-input inference | 0.90 seconds |
| Full verification, excluding imports | 10.25 seconds |
| Peak process RSS | 1,800,384,512 bytes |

Run it using:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/verify.py
```

Machine-readable results are in `verification.json`. The supplied `check_model.py` was also run separately and passed; it was not modified. No authored multiple-choice evaluation file was used.

## Limitations

The architecture is still a bag-of-concepts memory, not a language model or general reasoning engine. It discards order and multiplicity, removes question-present concepts from options, uses only the 16 most salient question concepts for pair retrieval, and cannot reliably bind arbitrary qualifiers or distinguish reversed comparisons with identical bags. Numerical laws are explicit associations, not general arithmetic. Hash collisions, truncation, broad associations, and ambiguous words remain. The checks establish a valid, reproducible, resource-compliant model, not a measured accuracy improvement.

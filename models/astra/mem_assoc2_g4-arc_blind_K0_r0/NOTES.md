# Generation 4: expanded, lexically connected science association memory

## Delivery and provenance

This folder implements the supplied second-order association architecture with
**9,132 handwritten science rules**, a net **1,532 rules beyond the supplied
generation-3 parent**. It uses no training, fitting, pretrained model, external
corpus, internet access, held-out questions, or accuracy-based
search. The parent's supplied 45.2% validation accuracy is background information;
this revision's ARC accuracy has not been measured.

`arch.py` is unchanged. Its SHA-256 is:

```
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

`build()` reads only the vocabulary and authored knowledge files in this folder.
It returns the required six tensors. No question file or prediction is read during
weight construction.

## Changes from the parent

### 1. Explicit lexical connections

`aliases_extra.txt` adds handwritten inflections and spelling equivalences. For
example, **lowers/lowered**, **emissions**, **crystallizes**, **secretion**, and
**filled** now share concepts with the corresponding existing definitions.

The compiler resolves an entire old alias group when a supplemental line connects
its root, rather than assigning just one spelling inconsistently. Existing
possessive, quotation, and conservative plural expansion is retained. There is no
learned stemming or external dictionary.

Scientific distinctions are preserved, including weather/weathering,
reproduce/reproducible, speed/velocity, mass/weight, heat/temperature,
rotation/revolution, conduction/convection, conductivity/conduction,
porosity/permeability, and genotype/phenotype. Supplemental aliases are audited
against all existing source rules after canonicalization.

### 2. Short diagnostic and conditional knowledge

Nine new source files supply definitions, descriptive recall cues, comparisons,
and practical mechanisms:

- `facts_g4_physical.txt`: mechanics, force and motion comparisons, pressure,
  buoyancy, heat transfer, phase changes, sound, and optics.
- `facts_g4_living.txt`: cell transport and signaling, plant growth and dispersal,
  physiology, immunity, inheritance, ecological mechanisms, and adaptations.
- `facts_g4_earth.txt`: geological identification and evidence, weathering and
  sediment transport, groundwater, weather and climate, tectonics, and astronomy.
- `facts_g4_matter.txt`: particle models, reactions, materials, solutions,
  separation, atomic structure, nuclear science, circuits, and electrical devices.
- `facts_g4_semantic_review.txt`: force interactions versus equilibrium,
  quantity-specific comparisons, interruption and protection mechanisms,
  acoustics, normal-incidence refraction, and solubility versus dissolving speed.
- `facts_g4_notation.txt`: contextual scientific symbols, named units, chemical
  formulas, compound names, vitamin names, and standard temperature reference points.
- `facts_g4_biology_review.txt`: relative tonicity, guard-cell turgor, root oxygen
  deprivation, girdling, prevention versus treatment, inheritance, and qualified
  ecological mechanisms.
- `facts_g4_earth_review.txt`: signed soil-conservation effects, quantity-specific
  hydrology, pore connectivity, phase distinctions, seismic propagation, stellar
  properties, and land versus floating sea ice.
- `facts_g4_chemistry_review.txt`: electrolyte carriers and phase conditions,
  molecular solutes, fuel-specific products, catalyst selectivity versus equilibrium,
  and reaction enthalpy signs.

Many cues contain just two diagnostic concepts. Comparisons assume other relevant
quantities are held fixed; they are associations, not a general algebra solver.
Conditional facts can use a negative source scale to request **pair-only recall**,
without unconditional first-order backoff. Explicit negative answer targets are
used when a positive bag would otherwise support the excluded process.

A source audit caught and prompted correction of signed-bag conflicts such as
positive support for decay in a preservation rule and positive resistance in a
superconductor definition. These repairs came from the meaning of source rules,
not from answer predictions. Duplicate rules still do not accumulate additional
weight.

### 3. Reduced accidental pair recall

The hash table has **1,048,573 rows**, a fixed doubling from the parent's 524,287.
A larger address space reduces expected collisions between unrelated concept
pairs, including unrelated query pairs that otherwise read an occupied row.
This was one design decision, not a search over hash sizes or measured accuracy.
Every resolved nonzero row target is retained; the current resulting width is 70.
Width is derived from the largest resolved hashed row, not selected by accuracy
or a parameter search.

The parent's remaining weight formulas are unchanged: independent positive and
negative maxima, short-target normalization, broad first-order damping,
negation-preserving cue pairs, bounded symbolic contrasts and definition
inversions, semantic salience, and the first-order mixture coefficient **0.65**.

## Follow-up semantic review

A second source-only review repaired concrete encoding problems in both the
initial G4 additions and inherited statements. The net rule count increased by
65, without changing the compiler formulas, salience policy, mixture coefficient,
or hash-table dimensions policy.

- **Lost condition:** `switch on` collapsed to just `switch` because `on` is a
  grammatical stopword. It is now `switch activated`, pair-only, with an explicit
  closed/current outcome and inhibition of open. Deactivated-switch and coil
  rules retain their conditions as well.
- **Absent outcomes:** a disconnected electromagnet no longer positively recalls
  its magnetic field. The evacuated-bell rule recalls silence and inhibits sound.
  The older bell rule was recast as a separate medium/transmission mechanism.
- **Mixed directions:** pendulum-length and contact-width comparisons no longer
  put increase and decrease in the same target bag. Wire-length/thickness changes
  retain the electrical quantity in separate pair-only rules. This does not solve
  arbitrary role binding, but avoids asserting both directions unconditionally.
- **Different physical conditions:** balanced forces on one object are not
  automatically a Newton-third-law interaction between different bodies.
  Constant velocity inhibits acceleration; coasting with friction recalls
  deceleration rather than constant motion. Instantaneous rest is distinguished
  from sustained equilibrium.
- **Positive bags with implicit negation:** stainless corrosion resistance,
  acoustic damping, insulating handles, material brittleness/strength, and buffer
  stability now use explicit signs where appropriate. Related inherited duplicate
  rules were corrected, rather than leaving opposing source assertions.
- **Optical and solution distinctions:** virtual/upright mirror images are
  qualified by a plane mirror; normal incidence need not bend a refracted ray.
  Stirring and crushing leave the equilibrium solubility limit unchanged, rather
  than inhibiting the concept of dissolved amount itself.

Aliases now cover acoustic/acoustics, damped/damping, energization, deactivation,
and related mechanism terms. Damping remains distinct from dampness, activated
from deactivated, frictionless from friction, and resultant from result.

Negative weights for reduced conduction, corrosion, or reflection mean inhibitory
evidence; they do not claim that every such process is eliminated absolutely.
The model still cannot recover all negation scopes or comparative roles.

Source regressions reject the replaced lossy cues, check the repaired signs and
alias distinctions, compile the reviewed conditional rules in isolation, and
check three reviewed definitions directly in first-order memory. No question
accuracy or answer ranking was used to select any repair.

## Notation and biology review

A further source-only review added a net 181 rules and repaired lexical ambiguity
and missing biological conditions. No measured answer accuracy guided this work.

- **Ambiguous letters:** C/F/K/N are no longer unconditional aliases for Celsius,
  Fahrenheit, Kelvin, and Newton. Named units can recall their symbols, but the
  independent symbols require contextual cue pairs for units, elements, vitamins,
  or compass directions. Full-model regressions require their outgoing first-order
  rows to be zero. This caught an inherited uppercase `vitamin C` source rule that
  still allowed unconditional backoff; it and the inherited vitamin-D rule were
  made pair-only without weakening the regression.
- **Compound identity:** CO2 now denotes carbon dioxide, not generic dioxide.
  Explicit formula aliases distinguish CO2/SO2, H2O/H2O2, and NaCl/generic salt.
  Other ambiguous spellings, such as CO versus cobalt after lowercasing, are not
  globally merged.
- **Finite lexical compilation:** eight explicitly authored compound-name pairs
  have exact one-token alternatives. The compiler preserves original facts and
  expands cue, positive-target, and negative-target spellings deterministically.
  It preserves signs and scales, deduplicates variants, skips masked tautologies,
  and never compresses a pair-only condition to a single cue. Both authored and
  compiled rules are audited. This is not inference-time phrase parsing, inferred
  chemistry, generic fact reversal, or multi-hop reasoning. The fixed architecture
  still treats spelled-out names as separate words.
- **Biological conditions:** cell-water movement now uses relative tonicity rather
  than assuming every saltwater/freshwater condition has a fixed effect. Drought,
  closed stomata, waterlogged roots, skin barriers, nocturnal desert behavior, and
  wetland loss retain their relevant conditions or explicit inhibited outcomes.
  Vaccination is separated from antiviral treatment, and inbreeding is not treated
  as evidence that every recessive allele is harmful. Duplicate inherited rules
  were corrected where necessary.

Regression fixtures test formula alternatives in both cue and answer roles,
including negative CO2 recall for a photosynthetic-products condition. These are
isolated tensor contracts, not multiple-choice questions or accuracy labels.
The handwritten weight formulas, salience policy, mixture coefficient, and fixed
hash-table size remain unchanged in this review.

## Earth-science condition review

The Earth-science source-only review added 36 handwritten rules and corrected existing
rules without changing the architecture, compiler, or weight formulas.

- **Suppressed processes:** windbreaks, contour plowing, terraces, and cover crops
  now support soil retention/conservation and inhibit erosion where appropriate,
  rather than positively recalling erosion from phrases such as "reduced erosion."
  Matching inherited records were repaired as well.
- **Hydrological roles:** ordinary impermeable paving supports runoff and inhibits
  infiltration. Permeable paving retains the opposite contextual relationship.
  Separate paving/runoff, paving/infiltration, and paving/recharge cue pairs express
  quantity-specific directions rather than mixing increase and decrease in one
  answer bag. Porosity remains distinct from permeability.
- **Geological and atmospheric conditions:** glassy obsidian inhibits crystalline
  texture; this describes the bulk texture rather than denying sparse inclusions.
  Inland/landfall hurricanes retain the weakening condition. Dew and frost keep
  liquid-condensation and solid-deposition roles separate. Relative-humidity
  comparisons assume fixed vapor content before condensation.
- **Contextual comparisons:** stellar red/blue temperature relationships and
  seismic P/S propagation are pair-only rather than unconditional associations
  from a color, a letter, or the generic concept of a star. Land-ice melt and
  floating-sea-ice melt retain distinct school-level displacement mechanisms.
- **Lexical connections:** pave/paved/paving, intensification forms, and dissipation
  forms are explicitly connected; pavement remains a distinct surface noun.

Expanded source contracts caught inherited positive-scale duplicates for bare
hillsides, landfall hurricanes, and red/blue stars. Those source records were
corrected rather than weakening the tests. That review added 38 isolated
Earth-science cue-pair contracts and expanded signed first-order definition
checks to five. There was no answer-label evaluation or accuracy-guided selection.

These are bounded associations under the stated conditions, not a general
quantitative simulator. Negative support for a reduced process does not assert
that it disappears entirely, and the architecture still cannot parse arbitrary
phrases such as "prevents erosion" compositionally.

## Chemistry condition review

The chemistry review adds a net 29 handwritten rules and repairs existing chemistry
records. No architecture, compiler, salience, mixture, or hash-size change was made.

- **Electrical phase conditions:** ionic solids and crystalline sodium chloride
  support fixed ions and insulation, while molten/dissolved conditions support
  mobile ions and conduction. These rules, including inherited duplicates, are
  pair-only; melting or dissolving alone must not assign ionic conductivity to
  every material. Electrolyte current carriers are distinguished from the electrons
  carrying current in external metal wires.
- **Molecular solutes:** sugar/sucrose dissolving in otherwise pure water does not
  by itself supply mobile ions. Nonelectrolyte and ionization/dissolution concepts
  remain distinct. Weak versus strong electrolytes concern degree of ionization,
  not solution concentration. A source check found that inherited aliases collapse
  "strength" and "strong"; the generic definition therefore uses the distinct cue
  `electrolyte classification` rather than `electrolyte strength`.
- **Fuel-specific products:** generic combustion recalls oxidation, oxygen, fuel,
  and heat, not universal carbon-dioxide-and-water products. Explicit complete
  hydrocarbon/methane rules retain those products; incomplete conditions recall
  possible carbon monoxide and soot. Hydrogen, magnesium, and sulfur have separate
  product associations. Compound concepts distinguish sulfur dioxide from carbon
  dioxide without assigning opposing signs to their shared word "dioxide."
- **Catalysts:** pathway and quantity conditions retain pair-only recall. Existing
  correct activation/enthalpy/equilibrium rules were preserved. Added rules separate
  faster reaction rates from unchanged equilibrium endpoints at fixed temperature.
  An unqualified "same products" assertion was removed because catalysts can affect
  selectivity among competing reactions.

Forty additional chemistry cue pairs have source/isolated-tensor contracts;
seven signed first-order definitions are now checked. Formula alternatives,
duplicate invariance, all prior regressions, and the required checker pass.
The deterministic retain-all-targets policy now derives row width 70 instead of
55; no row-width or hash-size search was performed. Measured memory remains below
the task budget. No multiple-choice labels or accuracy measurements guided this
review, and the bag-and-pair representation still cannot enforce arbitrary
multi-condition chemistry or resolve every contextual exception.

## Circuit conditions and negation contracts

The latest review preserves the 9,132-rule count and all compiler weight formulas,
while repairing inherited unconditional backoff from electrical conditions.

- Disconnected batteries, broken filaments, melted fuses, open/incomplete paths,
  and reconnection/closure now retain their conditions through pair-only recall.
  A disconnected-battery statement must not teach the isolated word "battery"
  to inhibit current. Corresponding inherited open/closed duplicates were repaired.
- A loose connection now recalls intermittent/unreliable contact rather than
  universally asserting a fully open circuit. These circuit associations describe
  ordinary powered circuits, not arbitrary transient or multipath networks.
- Source contracts cover fourteen additional two-concept cues and seven additional
  multi-concept conditions, including canonical cue order and alias variants.
  Isolated tensor tests verify that these reviewed pairs introduce no first-order
  backoff and preserve explicit target signs.
- New compiler fixtures verify that a three-concept negated condition generates
  only negator-containing pairs and no first-order memory. Separate fixtures test
  direct-definition inversion for `not`/`except`, and the priority of an authored
  exception for the exact semantic pair. Fixture facts are not added to the full
  knowledge base and are not answer labels.
- The source audit now rejects a lone-negator cue: backoff is disabled and no pair
  can form, so such a rule would silently produce no recall. No current authored
  rule had that problem; this is a preventive validation check. Existing aliases
  for no/without/absent/never were verified to retain negation.

All prior tests and the required checker pass. This review used source semantics
and compiler contracts only, with no question accuracy measurement or fitting.

## Representation

Source records use the inherited format:

```
question cues | supported answer concepts | inhibited concepts | optional scale
```

Weights are compiled deterministically. The vocabulary maps surface forms to
concepts, `idf` gives fixed semantic salience, `A` stores first-order associations,
and `row_ids`/`row_w` store pair-conditioned associations at the hash addresses
required by `arch.py`. Negative scale means pair-only, not reversal of the
positive/negative target lists.

| Quantity | Value |
|---|---:|
| Authored source rules | 9,132 |
| Unique compiled lexical variants | 9,175 |
| Vocabulary entries | 65,268 |
| Concepts, including none | 4,112 |
| First-order matrix | 4,112 × 4,112 |
| Pair table | 1,048,573 × 70 |
| Occupied pair rows | 15,404 |
| Tensor elements | 163,778,147 |
| Nonzero elements | 303,446 |
| Tensor storage | 948,974,108 bytes, about 905.0 MiB |

Most vocabulary entries are morphological, possessive, or quotation forms, not
independent concepts.

## Verification

Run:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/build.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/audit.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify_g4.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

All checks passed. The required checker reported **CHECK OK**, with the dimensions
and nonzero count above. It used its default placeholder, not ARC examples.

`verify_g4.py` checks architecture identity, vocabulary consistency, alias
equivalences and distinctions, source signs, isolated pair-only compilation, zero
first-order weights for those isolated rules and ambiguous-symbol rows, exact
compound-name alternatives and signed formula recall, duplicate-rule invariance,
negation-preserving pair generation and explicit-exception priority, tensor
shapes/dtypes/bounds, finite values, unused-slot conventions, question masking,
option permutation, token-order and duplicate-token invariance, alias-substitution
invariance, and unknown-input ties.

The inference checks use original synthetic concept bags with **no labels and no
accuracy computation**. No output ranking selected any content or parameter.

Measured with four Torch threads:

| Measurement | Result |
|---|---:|
| Full build | 3.506 seconds |
| 4,096 synthetic inputs, batches of 64 | 1.167 seconds |
| Peak RSS of the full regression process | 1,062,228 KiB, about 1.01 GiB |

The regression includes isolated and duplicate-rule builds as well as the full
model. It remains below the 2 GB memory budget and the 20-minute runtime limit.
Synthetic timing demonstrates execution capacity, not ARC accuracy.

## Limitations

The fixed architecture is a bag-and-pair memory. It cannot reliably bind arbitrary
roles, resolve polysemy, represent all negation scopes, interpret diagrams, or
perform general numerical reasoning. It only uses sixteen question concepts for
pair recall and scores answer concepts additively. Hash collisions remain
possible. The additions and lexical repairs are intended to improve coverage and
recall specificity, but improved held-out accuracy is not claimed without an
external evaluation.

# Generation 6: expanded descriptive science association memory

## Deliverable and provenance

This is a deterministic, manually authored extension of the supplied generation-5
model. It retains the fixed second-order association architecture and the parent's
weight formulas. It adds **1,701 handwritten science rules** in seven files, for
**12,302 authored rules** in total, plus explicit lexical variants and eight
additional compound-name substitutions.

Nothing is trained, fitted, pretrained, downloaded, or selected using measured
answer accuracy. No external files, internet resources, or ARC question sets were
consulted. The ten questions in TASK.md were not used as a test suite. The supplied
parent's reported 46.4% validation accuracy is background information, not a result
for this revision. Improved held-out accuracy is the goal, not a verified claim.

`arch.py` is unchanged. Its SHA-256 is:

```
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

## What was added

| File | Authored rules | Main coverage |
|---|---:|---|
| `facts_g6_living.txt` | 306 | Cell structures, transport, plant growth, anatomy, reproductive structures, inheritance, osmotic conditions |
| `facts_g6_earth.txt` | 271 | Rock descriptions, weathering, sediment transport, groundwater, soils, weather, geologic evidence, glacier mass budgets |
| `facts_g6_physical.txt` | 320 | Everyday forces, collisions, pressure, machines, thermal transfer, energy conversions, charge carriers, wave media, sound, optics, quantity-specific comparisons |
| `facts_g6_ecology.txt` | 243 | Feeding roles, symbioses, population limitations, succession, adaptations, pollution, conservation |
| `facts_g6_chemistry.txt` | 236 | Atomic structure, crystalline/amorphous solids, solutions, reactions, chemical names, separation, preservation, pH thresholds, electrodes |
| `facts_g6_inquiry.txt` | 174 | Evidence, experimental controls, measurement errors, instrument use, engineering, safety |
| `facts_g6_space.txt` | 151 | Sky patterns, hemispheric visibility, seasons, space instruments, stars, planets, lunar phases, eclipse shadow regions, tides |

The additions emphasize short, concrete descriptions that can recall a scientific
mechanism even when the question does not name that mechanism. Examples include
paper crumpling and drag, capillary uptake in a wick, milk pasteurization versus
homogenization, sediment sorting, gas exchange across alveoli, and observed
properties of minerals. These are authored associations, not automatically reversed
facts or inferred multi-hop chains.

`aliases_g6.txt` adds explicit inflections, spelling variants, and formula tokens.
Examples include crumpled/crumple, wrung/wring, haemoglobin/hemoglobin,
keratinised/keratin, and pasteurised/pasteurization. Important distinctions remain:
endotherms are not endothermic reactions; acclimation is not evolutionary
adaptation; porosity is not permeability; cohesion is not adhesion; and glucose is
not every sugar.

The compiler additionally recognizes authored name alternatives for carbon
monoxide, sodium hydroxide, sodium bicarbonate, sodium hydrogen carbonate, calcium
hydroxide, sulfuric acid, nitric acid, and acetic acid. Formula aliases such as
NaOH, NaHCO3, H2SO4, HNO3, CH3COOH, and C6H12O6 retain compound identity. Matching
uses complete token boundaries and preserves both original and substituted
records. It does not map a compound indiscriminately onto one of its elements.

## Construction of every weight

Source records have the format:

```
question cues | supported answer concepts | inhibited concepts | optional scale
```

A negative scale means pair-only recall, not reversed target signs. The inherited
compiler:

1. Resolves authored lexical aliases and conservative surface variants into a
   single concept per vocabulary word. Padding and unknown words map to concept 0.
2. Assigns fixed semantic salience values, without corpus or question statistics.
3. Compiles signed first-order associations into `A` and concept-pair associations
   into the hashed recall tables. Targets are normalized by the number of
   recallable concepts; multi-cue backoff is weaker than direct definitions.
4. Combines duplicate evidence using independent positive and negative maxima,
   so repeated facts do not behave like training frequency.
5. Retains the parent's limited opposition and negation transformations, pair-only
   conditions, generic-cue damping, and first-order mixture of **0.65**.
6. Hashes pair identities into **1,048,573** rows and retains every nonzero target.
   Row width is the widest resulting row, not an accuracy-tuned hyperparameter.

The only compiler changes from the parent are loading `aliases_g6.txt` and adding
the eight explicit compound-name alternatives. All weight formulas, salience
rules, hash dimensions, and inference code are otherwise inherited.

`build()` reads only local vocabulary, alias, and fact files and returns exactly
`concept`, `idf`, `A`, `row_ids`, `row_w`, and `mix`, with the specified dtypes.
Verification scripts and this document are not consulted during weight building.

## Source review and checks

Source-level audits identified places where ordinary prose would create an
incorrect bag association. For example, “prevents backflow” must inhibit backflow,
not positively recall it merely because the word is present. Similar repairs
cover smothered flames, opaque obstructions, absent chlorophyll, stationary work,
and atmosphere-free telescope observations. Conditional comparisons retain
pair-only encoding where appropriate. Broad reductions and absolute absences are
not treated as interchangeable claims.

### Follow-up review: preserve selectors and bind quantities explicitly

A source-only review added 33 rules and corrected overlapping inherited records,
without changing the compiler or evaluating answer accuracy:

- **Impact protection:** mixed target bags such as “increase stopping time, reduce
  force” were replaced with neutral cushioning/impulse mechanisms. Separate
  airbag, padding, cushion, and helmet pairs now condition explicitly on force
  versus time/duration. Comparisons hold mass and velocity change fixed and refer
  to average force, not every instantaneous peak in a real impact.
- **Ramps and gears:** opposite changes in effort and travel distance no longer
  share a target bag in the reviewed ramp records. Quantity-specific slope pairs
  assume an ideal ramp with fixed load and rise. Gear-size descriptions retain
  ratio/teeth relationships without assigning speed and torque directions before
  driver/driven roles are established.
- **Pressure and orbital comparisons:** the unqualified quantity `submarine
  depth` no longer asserts an increase. Deeper and shallower conditions recall
  opposite pressure directions. Orbital comparisons retain closer/farther in
  two-cue rules rather than leaking short/long periods into every planet/Sun pair;
  these comparisons assume the same central mass.
- **Other dropped selectors:** faster/slower vibration and red/blue star rules
  retain the selector in their two-cue condition. Generic sound/vibration and
  star/temperature pairs no longer inherit both incompatible directions from
  these new descriptive rules.
- **Mechanisms and thresholds:** an inherited space-telescope record had
  positively recalled atmospheric distortion; it now inhibits distortion.
  Refrigerator coils are neutral heat exchangers until condenser/evaporator
  roles are specified. Protostar heating recalls gravitational contraction rather
  than assuming sustained hydrogen fusion. Rebounding does not prove an elastic
  collision. Permanent deformation requires exceeding the elastic limit.
- **Materials and timescales:** brittle nonmetals do not universally imply
  insulating behavior. Rigid solids are distinguished from the slowly deforming
  solid mantle when describing convection. Genuine inflections such as
  cushion/cushioning and steep/steeper are connected without merging depth with
  deeper or force with duration.

Matching records in inherited files were corrected as well; new all-source
regression guards prevent those reviewed overlaps from silently restoring the
same mistakes. These remain limited associations under stated assumptions, not
an arbitrary role-binding or negation parser.

### Chemistry and biology review: thresholds, lineages, and quantifiers

A further source-only review added 26 records and corrected matching inherited
records. It did not change any compiler formula or use answer accuracy.

- **pH thresholds:** expanding “pH below seven” and “pH above seven” into every
  pair had put incompatible acid/base claims into the shared pH/seven row.
  Comparison-plus-threshold cues now retain `below seven`, `above seven`, or
  `equal seven` together. The bare pH/seven default remains neutral. The explicit
  threshold rules use a manually chosen pair-only magnitude of 2 so their signed
  evidence overrides that weaker co-occurrence default. This priority was
  designed symbolically, not chosen by a score search. Four composition tests
  verify signs, not option rankings. These rules assume the usual aqueous pH
  convention near 25 C; numeric thresholds outside that domain are not a general
  inequality parser. Above/below without a threshold no longer assert a class.
- **Atomic neutrality and phases:** neutral atoms recall equal proton/electron
  counts rather than being confused with neutral subatomic particles. A foam is
  not automatically liquid: liquid and solid foam associations have separate
  conditions.
- **Electrodes and preservation:** anode/cathode polarity is conditioned on
  galvanic discharge versus electrolysis. Pasteurization, refrigeration, and
  freezing do not imply sterilization. The new records distinguish microbial
  reduction or possible survival from spore destruction; they do not claim that
  every microbe survives every preservation treatment.
- **Osmosis and blood cells:** saltwater alone does not specify tonicity relative
  to tissue. Potato shrinkage now requires the hypertonic selector. Mature
  mammalian erythrocytes lack nuclei, while avian and other reviewed nonmammalian
  erythrocytes retain them; erythroblasts are nucleated precursors. Isotonic,
  hypertonic, and hypotonic erythrocyte descriptions retain their selectors.
- **Mutation transmission:** the blanket inhibition of inheritance for a somatic
  mutation was removed. Mitotic daughter-cell lineages and some vegetative
  offspring provide transmission routes, distinct from germline transmission in
  sexual reproduction. These are possible routes, not guarantees for every
  mutation.
- **Quantifiers:** `both` is now an explicitly authored concept instead of being
  discarded by the grammar filter. Mutualism descriptions preserve `both benefit`
  as their cue rather than teaching that any mentioned benefit means mutualism.
  The word remains distinct from `one` and `all`. This limited fix does not solve
  arbitrary role assignments or all three-way ecological descriptions.

### Inquiry and environmental review: testability, qualifiers, and signed harm

A third source-only review added four records and corrected existing descriptions
and lexical grouping. Compiler formulas and inference remain unchanged.

- **Testability versus outcome:** `falsifiable`/`falsifiability` no longer share a
  concept with `falsified`/`falsify`. Being open to an empirical test does not mean
  a claim has already been refuted. Definitions now recall testability and
  evidence rather than accomplished disproof. Hypothesis refutation and fabricated
  evidence have separate contextual records. The different senses of the verb
  `falsify` still require context; this is not full word-sense disambiguation.
- **Controls:** positive/negative control selectors stay in two-concept cues.
  Adding a third word, `expected`, had projected response evidence onto a generic
  control/expected pair. These rules describe expected results of functioning
  assays, not guaranteed observations.
- **Measurement:** consistency alone no longer projects inaccuracy or bias from
  the longer instrument/consistent/wrong cue. Scattered readings and finer scale
  divisions likewise retain their qualifiers. Reported digits refer to reporting
  resolution, not demonstrated repeatability or accuracy. Averaging independent
  readings of a stable measurand reduces random uncertainty of the mean, but
  does not remove a persistent systematic offset.
- **Environmental effects:** rain-garden descriptions recall infiltration and
  storage without assigning the same direction to infiltration and runoff.
  Oil-contaminated feathers inhibit insulation/waterproofing rather than
  positively recalling those impaired functions; overlapping inherited records
  were corrected too. Reforestation recalls soil stabilization and inhibits
  erosion, comparing established vegetation with cleared ground under comparable
  rainfall. Signed inhibition expresses impairment/reduction here, not necessarily
  complete absence of a function or process.

The source audit caught a proposed pair whose `data` token was discarded as
ordinary grammar. It was rewritten as `evidence fabricated`, preserving two
meaningful concepts without changing the grammar filter. All-source guards also
caught and corrected another inherited oil/feathers overlap.

### Physical and Earth-science review: structure, texture, and mass balance

A fourth source-only review added seven records and corrected overbroad
associations, without changing compiler formulas or measuring answer accuracy.

- **Solid structure:** a solid's particles need not occupy a repeating lattice.
  Generic solid/particle descriptions now recall localized positions and vibration;
  separate crystalline/solid and amorphous/solid pairs distinguish lattice order
  from irregular arrangement. Inhibiting a lattice for an amorphous material does
  not deny local short-range structural order.
- **Crystal size:** coarse/fine selectors remain in two-concept cues, so opposite
  cooling histories no longer project onto the same unqualified rock/crystal pair.
  These are conventional igneous-texture comparisons with other crystal-growth
  conditions comparable, not universal size-to-cooling laws for every rock.
- **Silica and viscosity:** unspecified magma silica recalls composition and
  viscosity, not an automatic increase or explosive eruption. Higher/lower silica
  pairs describe silicate melts at comparable temperature, water content and
  crystal content. Explosivity also depends on gas and eruption dynamics.
- **Glaciers:** retreat/advance describe terminus movement rather than guaranteeing
  a particular whole-glacier mass budget. Ice flow, surges and calving can decouple
  front position from total mass changes. Separate balance, accumulation and
  ablation records cover mass inputs and losses, including sublimation and calving
  rather than equating all ice loss with melting.

### Electricity and waves review: carriers and material media

A fifth source-only review added seven pair-only records and corrected two
unqualified definitions. It did not change construction formulas or evaluate
answer accuracy.

- **Current carriers:** generic electric current recalls charge flow rather than
  assuming electrons in every medium. Metal/current recalls electron drift;
  electrolyte/current recalls ion transport and inhibits electrons as the bulk
  carriers. The latter describes ordinary aqueous/molten ionic electrolytes,
  not plasmas or mixed ionic/electronic conductors. Electron transfer at electrodes
  remains distinct from transport through the electrolyte.
- **Direction convention:** conventional current uses the direction of positive
  charge flow, opposite to electron drift. These explicit two-concept conditions
  do not merge ions with electrons or change the existing electrode chemistry.
- **Wave media:** a generic disturbance no longer presumes a material medium.
  Mechanical waves recall material oscillations; electromagnetic waves recall
  electric/magnetic fields and can propagate through vacuum. The latter is not
  a claim that electromagnetic waves cannot also travel through matter.

Concept numbering and hash occupancy change with the vocabulary. The electricity
review produced a widest row of 66 targets; the later astronomy revision produces
57. Row width is determined by retaining every compiled nonzero target, not by
accuracy tuning. Both measured revisions remained within the task memory budget.

### Astronomy review: hemisphere selectors and eclipse shadow regions

A sixth source-only review added four records and corrected existing descriptions.
It did not change compiler formulas or measure answer accuracy.

- **Polaris visibility:** north/south remains in each two-concept cue. The previous
  three-concept southern/hemisphere/Polaris description also projected invisibility
  onto an unqualified hemisphere/Polaris pair. The revised comparisons use ideal
  pole-star geometry at ordinary latitudes, not precision predictions near the
  equator where refraction and Polaris's angular offset from the pole matter.
- **Partial eclipses:** a generic partial eclipse no longer specifies penumbra.
  Partial solar observations involve the Moon's penumbra; a partial lunar eclipse
  involves only part of the Moon entering Earth's umbra. Penumbral lunar eclipses
  recall faint dimming. Annular eclipses recall a sunlight ring and antumbra, with
  separate concepts retained for umbra, antumbra, penumbra and penumbral.
- **Comet tails:** the new generic tail-direction record now includes radiation
  pressure as well as solar wind. The inherited explicit ion/dust-tail distinction
  was already present and was retained. This is the ordinary outward-tail picture,
  not a model of every projection effect or apparent comet antitail.

### Cell and leaf review: retain physical-state qualifiers

A seventh source-only review corrected three existing rules without increasing
the rule count or changing compiler formulas.

- **Cellular boundaries:** flexible/rigid remains in each two-concept cue, recalling
  membrane/transport or wall/support respectively. The old three-concept cues
  projected both structures onto the generic cell/boundary pair. The new pairs
  have no unconditional first-order backoff. These are cellular descriptions,
  not claims about every flexible or rigid boundary in engineering.
- **Leaf pores:** closed remains in the two-concept cue for water conservation
  and reduced transpiration. The old leaf/pores/closed cue also inhibited
  transpiration for a leaf/pores pair that did not specify closure. Inhibition
  concerns reduced stomatal water loss; it does not mean that all cuticular
  water loss stops. The cue is interpreted in the leaf-pore context, not as a
  claim about arbitrary pores in rocks or filters.

All-source guards protect the generic cell/boundary and leaf/pores pairs from
these reviewed projections. Three new isolated tensor contracts verify that the
qualified pairs still recall their intended consequences without first-order
backoff. These limited fixes do not supply general three-way condition binding.

These checks were semantic and structural, not comparisons against question
answers. The suites verify:

- vocabulary consistency, lexical distinctions, and compound matching boundaries;
- no empty cues, entirely masked targets, or explicit identical-cue sign conflicts;
- duplicate-rule idempotence and finite, correctly typed tensors;
- unchanged architecture, masking, option permutation, token permutation,
  repeated-token invariance, and unknown-input ties;
- **131 inherited conditional contracts** and **129 new isolated conditional
  source/tensor contracts**, including zero first-order backoff for the selected
  pair-only records;
- 55 additional all-source guards against reintroducing reviewed ambiguous
  directions, dropped selectors, and incorrect mechanism claims;
- four symbolic pH-composition checks requiring the correct signs when a
  threshold selector and the bare pH/seven pair are both present.

The inherited generation-4 suite required one compatibility update: its old test
expected `carbon monoxide` not to have a phrase alternative. It now explicitly
checks the new `carbonmonoxide` alternative and retains the negative boundary test
with `carbon monoxideish`. The generation-5 suite is unchanged. Generation-6
isolated contracts check newly authored records; they do not claim that every
older overlapping definition lacks the parent's weaker first-order backoff.

All of the following passed on the final model:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/build.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/audit.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify_g4.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify_g5.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify_g6.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

The required checker used its default placeholder, not a custom labeled question
set. It reported:

```
CHECK OK
weights 140,812,621 (359,453 non-zero), vocabulary 73,876 words
```

## Final size and measured execution

| Quantity | Value |
|---|---:|
| Authored rules | 12,302 |
| Unique compiled lexical variants | 12,191 |
| Vocabulary entries | 73,876 |
| Concepts including none | 4,604 |
| First-order matrix | 4,604 × 4,604 |
| Pair tables | 1,048,573 × 57 each |
| Occupied pair rows | 18,236 |
| Tensor elements | 140,812,621 |
| Nonzero tensor entries | 359,453 |
| Tensor storage | 802,620,640 bytes |
| Measured full build, four CPU threads | 2.896 seconds |
| 4,096 synthetic inputs, batches of 64 | 1.201 seconds |
| Peak RSS of full inherited structural regression | 1,402,964 KiB, about 1.34 GiB |

The synthetic inputs exercise execution and invariants; they measure no science
accuracy. Measured construction and inference are within the 2 GB and 20-minute
budget on four CPU threads.

## Limitations

This remains a bag-and-pair memory, not a general scientific reasoner. It cannot
reliably bind arbitrary comparisons to their quantities, resolve all negation
scopes or homographs, inspect figures, or solve general numerical equations.
Only the sixteen strongest question concepts enter pair recall. Hash collisions
and additive option-length effects remain possible. The larger authored knowledge
base expands recall coverage but does not remove these architectural limitations.

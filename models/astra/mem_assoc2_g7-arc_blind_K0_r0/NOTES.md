# Generation 7: condition-preserving science association memory

## Deliverable and provenance

This model extends the supplied, manually authored generation-6 model. It has
**13,969 authored science rules**, a net increase of **1,667 records** over the
parent, and **117 new lexical alias groups**. The six new fact files contain
1,663 rules; reviewed inherited files account for four additional records.
All additions are written from general science knowledge.
Nothing is trained or fitted. No external corpus, pretrained weights, network
access, ARC dataset, question labels, or accuracy-driven parameter search was
used. The ten example questions in TASK.md were not evaluated.

The supplied parent's reported validation accuracy is 46.4%. This revision aims
to improve coverage and reduce incorrect recall, but its held-out accuracy has
not been measured and no improvement is claimed as an observed result.

`arch.py` is unchanged. Its SHA-256 remains:

```
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

## Changes from the parent

### 1. Preserve unambiguous comparative conditions

The parent already preserved negators when projecting a multi-concept rule into
pairs. It could, however, project a rule such as `gas temperature increase` onto
an unconditional `gas / temperature` pair, discarding the comparative condition.
Its first-order backoff could also distribute that conditional claim to each
individual cue.

The new `condition_guard` recognizes a deliberately small, authored selector set:

```
increase decrease more less high low fast slow above below deep shallow
```

If a rule has multiple cue concepts and exactly one distinct selector, each
recalled pair must contain that selector, and single-cue backoff is suppressed.
A rule with two different selectors may express a comparison between conditions;
these cases retain the parent's behavior instead of guessing an argument role.
Negation retains priority over this guard. Single-concept definitions retain their
ordinary first-order path.

Selector salience is explicitly set to 1.03 so these retained conditions are less
likely to disappear in the architecture's top-16 selection. The parent's stronger
1.06 polarity/reactant/product selectors and 1.12 negators still take precedence.
These are semantic design choices, not fitted values.

This does **not** implement full third-order logic, word-order interpretation,
negation scope, or general comparison reasoning. It removes a specific source of
unqualified associations while staying inside the fixed architecture.

#### Follow-up review: explicitly mandatory source qualifiers

**125 rules** now use an authored `@word` marker in their cue, for example:

```
@heated sealed gas | pressure increase collisions
pitch @longer string | slower vibration lower frequency
rock @fine @dark | basalt igneous
```

Every emitted pair must include every marked concept. The first example emits
heating/sealed and heating/gas pairs, but not an unconditional sealed/gas pair.
The third emits only the fine/dark pair. Marked rules have no singleton backoff.
Negation and automatic comparative guards remain additional constraints, and
unrepresentable conditions fail the audit/build instead of silently losing a
qualifier. The marker is source syntax only: no new tokenizer behavior, runtime
parser, weight name, or architecture change is introduced.

Lexical variants cannot change a mandatory concept into a different concept or
collapse a mandatory pair into a singleton. The audit rejects unknown markers,
markers in answer fields, malformed markers, and more conditions than a pair can
represent. Explicit markers do not change numeric weight formulas or salience.

The review corrected qualifier loss in gas heating/cooling, ramps, vibrating
strings, surface reflection, atomic particle changes, periodic-table groups,
bond breaking/forming, wax phase changes versus combustion, diaphragm movement,
skin circulation, rock descriptions, seasons, and stellar lifetime comparisons.
It also corrected signed descriptions of vacuum-gap insulation, broken circuits,
absorbed sound, insulating materials, solid ionic crystals, catalyst consumption,
heart-valve backflow, fossil preservation, and morning/evening shadow directions.
Inherited records that would reintroduce the reviewed candle, sealed-gas, and
fossil-preservation errors were corrected too. Inhibited processes may be reduced
rather than completely absent; these remain limited school-science associations.

These changes came from source-level science/encoding review, not measured answer
accuracy. Twenty-seven all-source regression cases now forbid the specifically reviewed
claims from leaking onto unqualified concept pairs.

#### Ecology and inquiry review: bind effects to the changed quantity or method

A further source-only review applied the same explicit-qualifier mechanism to:

- **Ecological roles:** predator/prey/producer reductions retain both the changed
  population and the reduction selector. Taxonomic and feeding descriptions
  preserve decisive traits and adult life stages. New carnivory descriptions no
  longer assert a fixed secondary-consumer level when the prey's diet is unknown.
- **Water treatment:** settling, filtration, chlorination, boiling, and
  desalination retain their method selectors rather than assigning all removed
  substances to every water-treatment pair.
- **Environmental effects:** photosynthesis under algal shading and shell
  calcification under acidification are inhibited rather than asserted as
  beneficial outputs. Operational solar-panel/wind-turbine generation inhibits
  fuel combustion; this is not a claim of zero manufacturing or lifecycle impact.
- **Inquiry:** supported/rejected hypotheses, controlled equality of conditions,
  seed-specific dependent measurements, microscope focus, and graph-slope
  qualifiers no longer disappear during pair projection.
- **Quantity binding:** an unspecified microscope field has an inverse
  relationship with magnification, not simultaneous unqualified increases and
  decreases. Four explicit comparison records bind magnification to its change
  or level. High/low blood records require glucose or sugar rather than recalling
  insulin/glucagon from every high/low blood-related quantity.
- **Safety and materials:** a wire covering is distinguished from the conducting
  wire. Heating and sealing are both retained in a pressure-hazard condition.
  Closed microbial cultures inhibit exposure. The added telescope hazard is
  conditioned on unfiltered sunlight, and protective-filter descriptions retain
  the filter selector.

Overlapping inherited microscope, hypothesis-contradiction, blood-glucose, and
acidification descriptions were corrected as well. Compiler weight formulas,
hash capacity, salience, and the fixed architecture were not changed in this
review. No question accuracy was measured.

#### Optics and distillation review: avoid unbound relation targets

Specific red/blue object, green-leaf, and red-filter descriptions now recall
reflection or transmission of the observed color without also recalling the
absorption of other wavelengths in the same unbound target bag. Absorption of
those other wavelengths remains scientifically valid; it is not marked as
inhibited. General selective-absorption and chlorophyll knowledge is retained.
Overlapping inherited descriptions were corrected as well.

Colored-illumination records retain both relevant colors, rather than claiming
that an arbitrary red object is dark or arbitrary white paper is red. These use
the ordinary idealized school model of broad-reflecting white paper and a
selectively red-reflecting surface. As elsewhere, pair recall cannot fully bind
an arbitrary ordered illumination/surface relation.

Distillation descriptions require the flask or condenser cue for the respective
boiling or cooling mechanism. Generic distillation knowledge still includes both
evaporation and condensation; only the component-specific projections changed.
Four further unqualified-projection regressions and four observed-color source
regressions cover these changes. Architecture, compiler formulas, vocabulary size,
and hash capacity remain unchanged. No accuracy evaluation was used.

#### Absence and inhibition review: encode signs, not negating prose

Four generation-7 descriptions were corrected:

- Flexibility supports bending/pliability and inhibits breaking during ordinary
  bending within the useful range. It does not assert immunity to failure.
- Brittleness supports fracture and inhibits plastic deformation, without
  claiming that brittle materials have no elastic strain.
- Antibiotic/virus cues support ineffectiveness and inhibit effectiveness rather
  than recalling generic treatment as a positive outcome.
- Antibiotic/bacterial cues support treatment, inhibition, and killing while
  inhibiting growth, under the ordinary susceptible-bacteria assumption. Existing
  resistance/selection records retain the resistant-organism exception.

These four records are pair-only, so their inhibited processes do not leak onto
an isolated material or pathogen concept. Three inherited antibiotic descriptions
were also constrained: viral ineffectiveness requires both virus and antibiotic,
not a generic antibiotic/infection pair or an antibiotic singleton. Four signed
source regressions, one additional projection regression, and a cross-source
pair-only check protect these changes. No training or accuracy testing was used.

#### Population-isolation review: distinguish barriers from positive exchange

Isolation and fragmentation descriptions no longer place "reduced gene flow" in
a positive target bag. Simply negating the two words would also incorrectly
inhibit genes themselves. Instead, these records support population separation,
barriers, and possible divergence/speciation. Geographic or habitat isolation
inhibits between-population migration; reproductive isolation inhibits
between-population interbreeding. These are limited exchange associations, not
claims that all movement stops or that speciation is inevitable.

Two new pair-only records bind decreased exchange to migration or interbreeding,
rather than predicting an unqualified decrease for every isolated-population
quantity. The inbreeding/drift description now retains both small population size
and isolation. Road-fragmentation effects retain fragmentation rather than
inhibiting migration for every habitat/road pair. Positive gene-flow descriptions
for forest, wildlife, and habitat corridors remain unchanged and are regression
checked.

An initial source-audit conflict exposed an overlapping inherited "migration
barriers" record with positive migration evidence; it and the related road
record were repaired. A new regression initially referred to "species isolation"
instead of the actual "species isolated" cue; the test was corrected to inspect
the existing source concept, without changing aliases or weights to satisfy it.
Final audits, source/tensor checks, reproducibility checks, and the required model
checker all pass. No answer accuracy was measured.

### 2. Recognize ordinary spaced scientific terms

The compiler adds a closed list of **40 spaced/fused spelling alternatives**, in
both directions. Examples:

- `food chain` / `foodchain`
- `food web` / `foodweb`
- `water cycle` / `watercycle`
- `fossil fuel` / `fossilfuel`
- `sea floor` / `seafloor`
- `ground water` / `groundwater`
- `light year` / `lightyear`
- `cell wall` / `cellwall`
- `tuning fork` / `tuningfork`
- `pressure cooker` / `pressurecooker`

This matters because the supplied tokenizer cannot combine adjacent words into
one token. The compiler therefore supplies both the original association and
its ordinary spaced lexical realization in the weight tables. It does not alter
the tokenizer, match phrases at inference time, or merge a compound with one of
its component concepts.

The parent's 16 formula/name substitutions are retained. Expansion uses complete
token boundaries, preserves signs and scales, and never turns an explicitly
pair-only condition into a singleton rule. Duplicate evidence still uses maxima,
not frequency accumulation. There are **14,258 unique compiled lexical records**.

`aliases_g7.txt` supplies explicit inflections and spelling variants such as
stomatal/stomata, dragged/drag, dried/dry, retinal/retina, leukocyte/leucocyte,
caliper/calliper, and hypothalamus/hypothalamic. It does not conflate speed with
velocity, mass with weight, or increase with decrease.

### 3. Additional hand-authored science coverage

| File | New rules | Coverage |
|---|---:|---|
| `facts_g7_mechanisms.txt` | 295 | Contact forces, inertia, fluid pressure, machines, thermal mechanisms, optics, sound, electricity, device energy conversions |
| `facts_g7_life.txt` | 271 | Cell descriptions, membrane transport, organ interactions, homeostasis, plant structures, reproduction, inheritance, evolution evidence |
| `facts_g7_earthspace.txt` | 304 | Rock/mineral descriptions, geologic evidence, surface processes, groundwater, weather mechanisms, sky observations, astronomy |
| `facts_g7_matter.txt` | 281 | Material properties, separation, physical/chemical changes, particles, atomic identity, bonding, periodic patterns, reactions and laboratory evidence |
| `facts_g7_ecosystems.txt` | 260 | Animal descriptions, feeding roles, cycles, populations, conservation, environmental mechanisms, adaptations and succession |
| `facts_g7_reasoning.txt` | 252 | Evidence, controls, variable-role bridges, instruments, graphs, models, engineering, laboratory safety |

Rules use the inherited format:

```
question cues | supported answer concepts | inhibited answer concepts | optional strength
```

Negative strength means pair-only recall, not reversed target signs. Mechanisms
and quantities are separated where mixing them would create ambiguous bags, for
example squeezing/pressure versus squeezing/volume, cushioning/force versus
cushioning/duration, and graph slope versus graph area. Signed records distinguish
litmus colors, interrupted electrical current, impaired transpiration, erosion
prevention, and other explicit contrasts. Source audits were used to find and
repair encoding inconsistencies, not to optimize question predictions.

Comparative physical claims assume the usual school-science idealizations and
other relevant conditions held fixed. The knowledge base is not a numerical
solver, a role-binding parser, or a complete treatment of every scientific
exception.

### 4. More hash addresses, with no target truncation

The hash table has **1,572,869 rows**, a fixed approximately 50% enlargement over
the parent's 1,048,573 rows. This reduces the expected frequency of unrelated
concept pairs recalling each other's facts. The size was chosen from the memory
budget, not question performance. The widest resulting row determines the column
count, **59**; every nonzero recalled target is retained.

## Weight construction

`build()` reads only local vocabulary, aliases, and fact files. It returns exactly:

| Tensor | Shape | Type |
|---|---|---|
| `concept` | `(77,406,)` | int64 |
| `idf` | `(4,779,)` | float32 |
| `A` | `(4,779, 4,779)` | float32 |
| `row_ids` | `(1,572,869, 59)` | int64 |
| `row_w` | `(1,572,869, 59)` | float32 |
| `mix` | scalar | float32 |

The vocabulary has **77,404 entries**. Many are conservative inflections,
possessives, and quote-token spellings inherited from the parent, not additional
independent concepts. Padding and unknown words both map to concept zero.

Apart from the described guard, selector salience, lexical expansion, and hash
capacity changes, the parent compiler's numeric formulas are retained:

- Signed first-order and pair evidence is normalized by recallable target count.
- Positive and negative maxima are maintained separately and then combined;
  copying a fact does not increase its strength by repetition.
- Broad first-order rows are damped.
- Limited authored oppositions and definition-negation transforms are retained.
- The first-order mixture remains **0.65**.
- No retrieval, parsing, corpus access, or rule traversal occurs in `forward`.

## Verification and resource use

`audit.py` verifies token encodability, distinct selectors, usable cue/target
concepts, finite nonzero rule strengths, and explicit sign consistency for equal
cue bags. Both the authored records and their compiled lexical variants pass.

The compiler review centralized the comparative-selector list shared by the audit
and builder. The audit now rejects two mandatory anchors plus a separate retained
comparison, which cannot fit into a pair; previously only the build caught that
case. Negation still takes precedence over comparisons. It also requires all
comparative selectors to remain present and their opposite meanings distinct.
Regression tests cover impossible conditions, valid anchor/guard combinations,
negation priority, and accidental selector aliasing. These validation changes do
not change authored science content or numeric weight formulas.

`verify.py` additionally checks:

- The exact unchanged architecture hash.
- Comparative and negation guard behavior, explicit mandatory qualifiers,
  marker validation, and anchor identity through lexical expansion.
- Twenty-seven all-source regressions against unqualified pair projections,
  twenty-one signed regression cases for reviewed inhibited processes and shadow
  directions, four source regressions for observed-color reflection/transmission,
  and cross-source pair-only preservation of viral antibiotic ineffectiveness.
- Five source regressions against unbound exchange claims from isolation, plus
  three checks preserving positive corridor/gene-flow associations.
- Spaced/fused lexical variants and complete-token matching.
- Encodability of every token in all 56 declared chemical-name and spaced/fused
  alternatives; lexical-compilation idempotence; and identical compiled evidence
  after reversing source-rule order and duplicating every source rule. These
  source-only checks neither score answers nor modify weights.
- Pair-only condition preservation and important lexical distinctions.
- Required weight names, shapes, dtypes, finite values, ID ranges, and empty slots.
- Synthetic-input option-permutation equivariance, repeated-token invariance,
  question-concept masking, and uniform all-unknown logits.
- A resource stress run of 4,480 full-length deterministic token patterns, with
  finite logits, build/inference time below 20 minutes, and peak process RSS
  strictly below a conservative **2,000,000,000 bytes**. These synthetic patterns
  have no natural-language question labels and are not scored for accuracy.

These are source/tensor correctness checks, not answer-accuracy tests. No question
accuracy was measured or used to select content or hyperparameters.

A four-thread verification run reported:

- Build time: **12.46 seconds**.
- One batch of 64 synthetic forward inputs: **0.027 seconds**.
- Full-length stress inference on 4,480 synthetic inputs: **1.726 seconds**.
- Tensor storage: **1,205,584,984 bytes**.
- Peak process RSS, including invariant checks and stress inference:
  **1,475,520 KiB** (approximately **1.41 GiB**, below decimal 2 GB).
  RSS measurements vary between runs; earlier verification runs were about
  1.73 GiB, also within the enforced budget.

The separate `python /srv/work/model/verify_reproducibility.py` check passed:

- Four builds, two per fresh process, with Python hash seeds 0 and 37.
- Byte-identical hashes for every tensor, including shapes and dtypes.
- No mutation of the builder's grammar configuration.
- **75.411 seconds** total, **1,645,232 KiB** maximum worker RSS, and
  **16,640 KiB** supervisor RSS. Their sum is below decimal 2 GB.
- Models are built sequentially and released before rebuilding. Tensor buffers
  are hashed directly without allocating serialized copies; the supervisor uses
  only the standard library and never loads a second model.

The aggregate tensor fingerprint was:

```
56fc1011d72f9f0d0f15a90a15687f685d108f0ae87c01d79102eabbf05cf6f7
```

This verifies reproducibility for the tested environment, not scientific answer
accuracy or cross-version numerical identity.

The required final validation, `python /srv/work/tools/check_model.py`, passed:

```
CHECK OK
weights 208,519,569 (394,700 non-zero), vocabulary 77,404 words
```

The checker used its placeholder input; no ARC questions or answer accuracy were
examined. `NOTES.md` and verification scripts are not read by `build()`.

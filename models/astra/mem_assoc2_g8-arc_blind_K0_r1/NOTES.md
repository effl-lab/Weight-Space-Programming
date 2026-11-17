# Generation 8: reviewed, condition-preserving science memory

## Deliverable and provenance

This is the task's fixed second-order concept-association architecture, extending
the supplied generation-7 model with **1,423 hand-authored science records** and
**120 supplementary lexical alias records**. The complete source has **15,389
records**, compiling to **15,669 unique lexical records**, **80,252 vocabulary
entries**, and **4,962 concepts including zero**.

Every weight comes from the supplied authored parent or from science, spelling,
and fixed formulas written in this folder. There is no training, fitting,
pretrained model, external corpus, internet access, question retrieval, or
accuracy-driven search. No ARC questions were evaluated, including the examples.
The parent's reported validation accuracy is 47.5%; this model's held-out accuracy
is unknown. Improved coverage and encoding are intended benefits, not a claim of
measured accuracy improvement.

`arch.py` is byte-identical to the supplied architecture and the parent's copy.
Its SHA-256 is:

```
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

## Knowledge additions

| Source | Records | Coverage |
|---|---:|---|
| `facts_g8_living.txt` | 259 | Flower anatomy, pollination, seed dispersal, germination, life cycles, animal structures, cells, digestion, circulation and health |
| `facts_g8_ecology.txt` | 230 | Symbiosis, adaptations, behavior, food webs, cycles, pollution, conservation, inheritance and evolution |
| `facts_g8_physical.txt` | 332 | Separation, material properties, particles, reactions, thermal processes, forces, waves and electricity |
| `facts_g8_earthspace.txt` | 329 | Rocks, minerals, surface processes, groundwater, soils, weather, climate, sky geometry, stars and planets |
| `facts_g8_inquiry.txt` | 273 | Experimental reasoning, measurement, units, graphs, instruments, sampling, engineering, resources and safety |

`aliases_g8.txt` supplies inflections and spelling equivalents, including
moult/molting, oesophagus/esophagus, pseudopodia/pseudopod, and
catalysing/catalyze. Some records reaffirm existing aliases without multiplying
weights. Porosity/permeability, accuracy/precision, and Celsius/Fahrenheit remain
distinct.

The parent's normalization formulas, mixture, semantic salience, hash capacity,
and condition-preserving mechanisms are retained. The compiler reads the extra
alias file and releases obsolete temporary dictionaries before allocating the
dense recall tables. A subsequent compiler review adds a symbolic precedence
rule: an inferred opposition cannot refute explicit positive evidence for the
same semantic cue and target. That evidence-combination change is detailed below.

Earlier follow-up reviews corrected ten inherited records: two platelet wound
descriptions, one scientific-objectivity description, four chemical indicator
records, and three paving/soil-infiltration descriptions. A subsequent conditional
review revised 47 records across 16 source files, including removal of one ambiguous
inherited record. A protective-outcome review revised another 15 records across
11 files. An experiment-role and thermos review revised 34 records across four
files. A graph-binding review revised 20 records across four files. The latest
temperature-reference review revises 19 records across five files, including two
removals; details appear below. Separate positive/negative maxima prevent duplicate
facts from acting like training frequencies.

## Encoding and follow-up review

The rule format is:

```
question cues | supported answer concepts | inhibited answer concepts | optional strength
```

A negative strength means **pair-only recall**, not reversal of answer signs.
An `@word` requires that concept in every emitted pair. It is source syntax,
not an inference-time parser or tokenizer change.

### Preserve decisive conditions

The source explicitly retains wet/dry seed storage, nitrogen/magnesium
deficiency, electron gain/loss, sugar/salt conductivity, intrusive/extrusive
cooling, lunar day/night temperatures, stellar colors, and boiling/freezing
reference temperatures with their Celsius/Fahrenheit scales.

The first requested follow-up review made **66 individually authored replacements**
across the new files, plus three inherited corrections. The subsequent
cross-source review is documented below. In particular:

- Red/violet wavelength and frequency comparisons are four separate signed
  assertions, each requiring its color and quantity. An unordered target bag
  containing both colors and both comparative directions is no longer used.
- Graph slopes and areas retain the relevant axis quantity; formula descriptions
  require the two physical quantities instead of projecting onto generic
  words such as "product" or "quotient".
- Annual/biennial life cycles, finch beak types, experimental variable roles,
  summer/winter solstices, and material deformation descriptions retain their
  discriminating qualifiers.
- Whale/fish and bat/bird comparisons cannot give singleton fish/bird cues
  mammalian traits.
- Heating hazards require both heating and sealing. Lunar atmospheric records
  retain the Moon rather than declaring every thin atmosphere weatherless.

### Encode inhibition rather than negating prose

The review corrected assertions where words such as "without", "limited",
"insufficient", or "prevents" would otherwise place an inhibited process in
positive target evidence:

- Vacuum sound records support silence and inhibit propagation/transmission.
- Wave energy-transfer records support oscillation and inhibit bulk matter
  transport, using the ordinary school-wave idealization.
- Pale dark-grown seedlings support etiolation and inhibit chlorophyll.
- Reproductive isolation supports separation/speciation and inhibits
  interbreeding; adding a species cue cannot restore positive interbreeding.
- Platelet wound records support clotting/sealing and inhibit bleeding.
- Scientific objectivity supports evidence/impartiality and inhibits bias.
- Patternless scatterplots inhibit correlation rather than positively recalling
  "no apparent correlation" as an unordered positive bag.
- Phloem girdling supports damage/starvation and inhibits transport; overheating
  enzymes supports denaturation/inactivity rather than positively recalling
  "loss of function".

Source descriptions for algal shading, coral acidification, fur/feather oiling,
refrigeration, and smoking were also narrowed to their mechanisms. A reduction
or inhibition is not generally a claim of complete absence. These remain
limited grade-school associations rather than quantified physiological models.

Earlier reviews also signed indicator colors, static friction, soil
conservation, sterilization, and atmospheric distortion in space telescopes.
All corrections came from scientific meaning and source consistency, not
question predictions or measured accuracy.

### Check emitted pairs, not just complete source cues

Equal-cue sign checks can miss an error in a longer rule: a three-concept cue can
emit a pair already assigned the opposite sign by a two-concept rule. The new
regressions inspect actual compiler-emitted pairs across **all source files**
for **15 reviewed mechanisms**. They require the expected inhibited evidence and
forbid explicit positive evidence for the reviewed inhibited process.

### Subsequent cross-source consistency review

The read-only `python model/diagnose_pairs.py` diagnostic enumerates explicit
positive/negative overlaps after pair projection, restricted to pairs involving
new source cues. It initially
identified eight overlaps. Manual scientific review corrected their causes:

- Black-hole descriptions no longer put "escape" in a positive bag as part of
  the phrase "escape speed exceeds light"; they support trapped light instead.
- Wet cobalt-chloride indicator records support pink and inhibit blue. Both the
  hydration condition and the specific metal are mandatory; water/chloride,
  dry/chloride, and water/sulfate pairs alone cannot recall reagent colors.
- Oiled feathers inhibit waterproofing as well as insulation; "impaired
  waterproofing" is no longer a positive target bag.
- Frozen soil, compacted soil, and ordinary impermeable pavement support runoff
  and inhibit infiltration. Corresponding inherited descriptions were corrected.
  This does not mean that every partly frozen or compacted soil has zero flow.
- Paving's groundwater effects require the paving qualifier, including through
  spaced/fused lexical expansion. Generic ground/water pairs retain positive
  infiltration knowledge rather than inheriting a pavement-specific inhibition.
- Drought no longer recalls recharge as a positive outcome; a separate drought/
  recharge pair supports a decrease. Compaction/permeability and compaction/
  porosity similarly have explicit quantity-specific directional records.
- Stainless steel and chromium plating support protective materials and inhibit
  corrosion, rather than placing "corrosion resistance" in an unordered positive
  bag that conflicts with existing protective records.

Related inspection also signed the lack of crystals in glassy rock and preserved
both dam and migration in the fish-passage barrier assertion. The soil-compaction
review exposed one additional inherited positive-infiltration description, which
was corrected rather than weakening the new consistency check.

The same read-only diagnostic now reports **zero explicit projected-sign overlaps
involving the new cue set**. This is a limited source-consistency result: it does
not inspect inferred opposition signs, prove every scientific claim, resolve all
word senses, or measure answer accuracy. Overlaps were inspected manually rather
than treated as universally invalid or automatically optimized away.

Permanent regressions additionally forbid the three unqualified reagent-color
projections and unqualified groundwater inhibition, while requiring retained
positive groundwater infiltration. These are source-level assertions before
hashing, not evaluations of question answers. They do not claim that unrelated
hash collisions are impossible.

### Explicit evidence outranks heuristic opposition

The latest compiler review addressed a distinct issue from erroneous source
facts. The inherited compiler generates negative evidence from a closed set of
25 semantic opposition groups. Such an inference is useful by default but is
not universally true: a process or object can legitimately involve both members,
or the two words can describe different quantities.

For example, the source explicitly supports:

- Condensation as well as evaporation in the water cycle.
- Emission as well as absorption at a black surface.
- Meiosis as well as mitosis as forms of cell division.
- High current alongside low resistance in a short circuit.

Previously, a narrower source statement could infer an opposite that subtracted
from these explicit associations. The compiler now keeps inferred opposition
evidence separate until all authored evidence is complete, then merges it using
this fixed symbolic rule:

> Apply an inferred negative only if there is no explicit positive for the same
> semantic singleton-or-pair and target.

This is provenance precedence, not a fitted numerical threshold. An explicit
positive blocks the default regardless of magnitude. Explicit negative evidence
is **not** erased: if both positive and negative facts were authored, their
independent maxima still combine as before. Where no explicit positive exists,
the inherited strongest-negative behavior remains unchanged.

The rule applies before hashing and separately to first-order and pair evidence.
Support from a different pair, a different target, or a singleton cannot exempt
an unrelated pair. Results do not depend on source order or duplication. The
25 groups and their candidate-inference formula are unchanged and centralized
for shared use by compiler and regression tests. No fact content, vocabulary,
normalization exponent, mixture coefficient, salience, or table capacity was
changed in this compiler review.

`verify_defaults.py` checks the symbolic precedence, retention of explicit
negatives, unaffected unopposed defaults, semantic-key isolation, rejection of
malformed positive defaults, idempotence, and insertion-order invariance. It also
checks the four real-source examples above. Across the current source,
**229 pair/target slots** and **1,044 singleton/target slots** have explicit
positive evidence protected from an inferred negative. These are source counts,
not answer-accuracy measurements, and were not used to tune numeric parameters.

This does not solve general argument binding or remove every scientifically
ambiguous source association; it prevents a heuristic default from overruling
more directly authored knowledge.

### Further conditional-source review

This review revised **47 source records across 16 files**: 46 replacements and
one removal. Decisive qualifiers are mandatory in pair projection for string
length, rolling surfaces, upward/downward projectile motion, solution additions,
heated versus cold air, river-bank position, circuit-bridge materials, iron
surface area, touch/conduction materials, food-chain length, water-table weather
conditions, and stellar colors. Spaced/fused lexical variants retain these guards.

Spatial hot-air "rise" was replaced by "upward": the existing rise alias means
quantitative increase, which incorrectly implied increasing density. A drought
record no longer recalls recharge positively merely because its prose meant
reduced recharge. The ambiguous "boiling point elevation" record was removed:
elevation maps to geographic altitude, whose boiling-point effect has the opposite
direction. Vacuum and pressure-cooker records now preserve their actual conditions.
The vocabulary, architecture, compiler formulas, and numeric parameters are unchanged.

`verify_conditions.py` checks **36 cue cases**, **56 compiled variants**, and
**10 all-source projection exclusions**, plus spatial/density and altitude
safeguards. Initial checks exposed five further unguarded descriptions (two river
banks, two stellar colors, and tile conduction), which were corrected. An initially
overstrict river-bend exclusion was narrowed: generic bends legitimately involve
both erosion and deposition, but do not alone determine faster or slower flow.
No questions, answer labels, or accuracy measurements guided these changes.

### Protective outcomes versus hazardous conditions

The protective-outcome review corrected **15 source records across 11 files**, without changing
vocabulary, architecture, compiler formulas, or numeric parameters. Ten electrical
safety descriptions had placed shock in positive evidence as part of phrases like
"shock protection" or "prevent shock". They now support insulation, resistance,
grounding, or protection as appropriate and inhibit shock. These assertions are
pair-only; longer cues retain the protective material or electrical condition.

Two sealed-culture descriptions now support containment/protection and inhibit
contamination, with one also inhibiting escape. An outside-host virus description
supports inactivity/host dependence and inhibits reproduction. Shoe treads support
friction/traction and inhibit slipping. These are ordinary school-level mechanisms,
not promises that protective measures eliminate every possible accident or that
sealing makes an already contaminated culture sterile.

The all-source regressions also exposed a distinct conditional hazard: "frayed
electrical insulation" was projecting shock/fire evidence onto generic electrical
insulation. It now requires both fraying and the electrical context. The hazard
itself remains supported; it was not deleted to satisfy a sign check.

`verify_protection.py` checks **six mandatory-condition cases**, **11 all-source
inhibited pairs**, and **ten retained positive-context cases**. The latter preserve
shock/fire hazards for frayed insulation and wet electrical contexts, slipping for
smooth tires, reproduction with a viral host, and contamination as something a
sterile culture control checks. Beneficial friction, insulation, and protection
associations also remain. These are authored-source regressions, not question
predictions or answer-accuracy measurements.

### Experimental variable roles and thermos mechanisms

Reviewing insulation experiments exposed a role-binding problem in inherited
source descriptions. A three-concept cue such as "insulation experiment dependent"
could emit insulation/experiment without its dependent-role qualifier, or
experiment/dependent without its insulation topic. The first loses the distinction
between manipulated factors, measured outcomes, and controls; the second exports
an experiment-specific measurement to unrelated experiment types.

This review makes **34 replacements across four files**. In 21 conventional
experiment descriptions (fertilizer, friction, insulation, dissolving solutes,
pendulums, magnets, evaporation, and absorbency), both the topic and variable role
are now mandatory. Another 12 definition records retain their independent,
dependent, or controlled/constant role together with the experiment, variable,
or measurement concept. Their positive target lists are unchanged. These rules
illustrate common experimental designs; actual variable assignments still depend
on the described protocol. No general language-level role parser is claimed.

The remaining replacement corrects the thermos-lid description: it supports
sealing, insulation, and retention, while inhibiting convection/evaporation heat
loss instead of placing those inhibited processes in a positive bag. This refers
to loss through the flask opening, not elimination of all internal fluid motion.
Radiation through a vacuum gap and radiation/reflection associations for shiny
or silvered thermos surfaces remain supported.

`verify_roles.py` checks **21 topic/role cases**, **11 definition bindings**
(including alias-equivalent source records), and **four all-source thermos
mechanism cases**. It inspects actual compiler-emitted pairs and retained target
concepts. All checks are source assertions, not science-question evaluations.
Vocabulary, compiler formulas, numeric parameters, and architecture are unchanged.

### Axis, plotted-line, and motion-graph bindings

The graph-binding review makes **20 replacements across four files**, retaining the
existing target assertions while preserving their graph-specific conditions.
Horizontal and vertical axis descriptions now explicitly require the axis and
its direction. A horizontal plotted line instead requires line/horizontal, and
a horizontal trend requires trend/horizontal. A generic graph/horizontal pair
can no longer receive these reviewed axis-specific and line-specific conclusions.

Motion-graph rules now retain both the plotted quantity and the relevant line
shape, slope, area, or direction of change. Thus distance/flat recalls rest,
speed/flat recalls constant speed, velocity/slope recalls acceleration, and
velocity/area recalls displacement. Time/flat, time/slope, and time/area alone
cannot inherit these quantity-specific conclusions from the reviewed source.
Elapsed time on the horizontal axis remains an explicit school-graph assumption;
the fixed second-order model cannot bind three arbitrary quantities or parse a
plotted image. Existing generic first-order associations are not claimed to solve
axis/line ambiguity globally.

`verify_graphs.py` checks **18 binding cases**, **21 compiled variants** (including
matching existing guarded source), **nine all-source projection exclusions**, and
**17 retained mechanism cases**. Vocabulary, numeric parameters, compiler formulas,
and architecture are unchanged. No question predictions or accuracy measurements
were used to choose these corrections.

### Temperature reference points retain both phase and scale

Inspection of water's phase-change descriptions exposed inherited numeric rules
that mixed Celsius/Fahrenheit values in a single unordered target bag or lost
the phase or scale when projecting three cues into pairs. The latest review makes
**17 replacements and two removals across five files**. Water freezing/boiling
reference points now require both the phase change and temperature scale, matching
the already guarded generation-8 records. A purity/boiling assertion explicitly
retains Celsius. Two redundant mixed-scale records were removed; their numerical
knowledge remains in the qualified records. No density records were changed in
this review, and density/phase-change mechanisms remain covered by regressions.

`verify_temperatures.py` checks **12 phase/scale cases**, **21 compiled reference
variants**, **nine all-source unqualified-pair exclusions**, and **five retained
mechanisms**. Unqualified water/freezing, water/boiling, and water/scale pairs no
longer recall these reference numbers. Tests retain ordinary water expansion,
ice density, vapor/bubble formation, and pressure-dependent boiling associations.
Numeric word/digit aliases are checked, and Celsius/Fahrenheit/Kelvin remain distinct.

These reference points assume pure water at ordinary atmospheric pressure; Kelvin
values 273 and 373 are explicitly school-level approximations. With only two
mandatory concepts per pair, the water/pressure context cannot simultaneously be
bound along with phase and scale. This limitation remains documented rather than
being hidden behind an invalid three-condition rule. Ambiguous unit letters c/f/k
are not globally merged with named temperature units, since they also have chemical
uses. No arbitrary unit conversion, numeric solver, or universal boiling-point
claim is introduced. Vocabulary, architecture, compiler formulas, and numeric
parameters are unchanged; no answer accuracy guided the review.

## Weight construction

1. Read local aliases and facts; generate conservative inflections, possessives,
   and quote-token forms; verify the saved vocabulary.
2. Audit authored and compiled facts. Apply the inherited closed lists of
   formula/name and spaced/fused spelling alternatives.
3. Map words to concepts. Padding and unknown word IDs map to concept zero.
4. Apply fixed semantic salience and preserve important polarity/comparison cues.
5. Compile signed first-order associations and qualified second-order pairs,
   normalized by usable target count and cue count. Keep inferred opposition
   defaults separate from explicitly authored evidence.
6. Merge defaults with explicit-positive precedence, retain independent signed
   maxima, damp broad first-order rows, and apply definition-negation backoff.
   Hash only after semantic operations finish.
7. Release obsolete intermediate dictionaries, then allocate tables without
   truncating any nonzero recalled target.

Only the six required tensors are returned:

| Tensor | Shape | dtype |
|---|---|---|
| `concept` | `(80254,)` | int64 |
| `idf` | `(4962,)` | float32 |
| `A` | `(4962, 4962)` | float32 |
| `row_ids` | `(1572869, 57)` | int64 |
| `row_w` | `(1572869, 57)` | float32 |
| `mix` | scalar, value 0.65 | float32 |

Tensor storage is **1,174,990,056 bytes**. Row width is derived from the widest
populated row; every target fits. There is no runtime fact traversal.
`build()` reads only its local vocabulary, alias, and fact files. Notes and
verification scripts do not contribute weights.

## Final verification

All of these checks passed after the follow-up corrections:

- `python model/audit.py`: authored and compiled records are encodable; no
  masked-only assertions or explicit sign conflicts for equal cue bags.
- `python model/verify_g8.py`: **60 condition cases**, **30 signed source cases**,
  **15 all-source projected-pair cases**, reagent/groundwater qualifier
  regressions, positive-groundwater preservation, and lexical distinctions.
- `python model/verify_conditions.py`: 36 conditional cue cases, 56 compiled
  variants, ten all-source exclusions, and spatial/density/altitude safeguards.
- `python model/verify_protection.py`: six mandatory-condition cases, 11
  all-source inhibited pairs, and ten retained positive-context cases.
- `python model/verify_roles.py`: 21 topic/role cases, 11 definition bindings,
  and four all-source thermos mechanism cases.
- `python model/verify_graphs.py`: 18 binding cases, 21 compiled variants,
  nine all-source exclusions, and 17 retained mechanism cases.
- `python model/verify_temperatures.py`: 12 phase/scale cases, 21 compiled
  variants, nine all-source exclusions, five retained mechanisms, and unit aliases.
- `python model/diagnose_pairs.py`: zero explicit sign overlaps within its scope.
- `python model/verify_defaults.py`: symbolic default precedence, explicit-sign
  retention, semantic-key isolation, idempotence, insertion-order invariance,
  and four real-source examples without question scoring.
- `python model/verify.py`: inherited semantic regressions; unchanged
  architecture; lexical-expansion idempotence and source-order invariance;
  tensor keys, shapes, dtypes, finite values, ID ranges and unused slots;
  option permutation, repeated-token invariance, question masking and OOV ties.
- `python model/verify_reproducibility.py`: four sequential builds in two fresh
  processes with two Python hash seeds, producing byte-identical tensor hashes.

The final four-thread verification run measured:

- Build: **10.244 seconds**.
- One synthetic batch of 64: **0.022 seconds**.
- Stress inference on **4,480** full-length deterministic synthetic token
  patterns: **1.179 seconds**.
- Peak process RSS including audits and invariant checks: **1,804,556 KiB**
  (**1,847,865,344 bytes**), below decimal **2 GB**.

Reproducibility verification took **47.733 seconds** in total. Maximum worker RSS
was **1,632,832 KiB**, with **16,128 KiB** supervisor RSS; their sum is
**1,688,535,040 bytes**, also below 2 GB. RSS and timing vary between runs.
The final aggregate tensor fingerprint:

```
26931997bb9a5f465d6b0f1768e9acd673ae16e946f161cf8fb5f4f53ffa8220
```

The required `python /srv/work/tools/check_model.py` passed on its placeholder:

```
CHECK OK
weights 204,013,727 (418,191 non-zero), vocabulary 80,252 words
```

No synthetic inputs carried answer labels or were scored for science accuracy.
These tests establish source/tensor correctness, determinism and resource use,
not held-out answer accuracy.

## Limitations

This is a bag-of-concepts memory, not a language model or symbolic solver. It
cannot generally bind arbitrary argument roles, distinguish word order,
implement third-order conditions, solve arithmetic, or parse diagrams. Top-16
cue selection, question/answer concept masking, ambiguous words and hash
collisions remain. Scientific comparisons assume ordinary school idealizations
with other relevant conditions held fixed.

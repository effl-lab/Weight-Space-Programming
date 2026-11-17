# Generation 3: expanded everyday-science association memory

## Delivery and provenance

This model compiles **7,600 hand-authored science rules** into the fixed six-tensor
second-order association memory. The supplied **`arch.py` is byte-for-byte
unchanged**. No pretrained weights, training, fitting, external corpus, internet
access, held-out questions, or accuracy-based search were used.

The supplied generation-2 parent is the starting point. Its reported validation
accuracy was 44.4%; that is background information, **not a measured accuracy for
this model**. This revision adds **1,986 rules** over the supplied parent,
explicit everyday word forms, and source-level semantic corrections. The initial
expansion added 1,626 rules; the encoding review added 41 more and revised 24
existing rules; the definition review added 213 rules and corrected five further
source statements. The conditional-source review revised 36 existing rules and
added one separate respiration rule. The protective-property review revised 27
existing rules and added nine definitions/mechanism rules. The astronomy review
revised 20 source statements, split one alternative-phase rule into two, and
added 32 definitions/qualified relationships. The reaction-energy review revised
12 existing rules and added 29 definitions/qualified relationships. The genetics
review revised nine existing rules and added 34 definitions/qualified
relationships. The compiler, salience formulas, hash dimensions policy, and
mixture coefficient are inherited unchanged. No question predictions or measured
accuracy selected the additions or review changes.

## What changed

Four new knowledge files broaden the range of descriptions that recall science
concepts. They include both familiar applications and alternate descriptive cues
for established facts; overlap is intentional, but duplicate evidence is not
frequency reinforcement in this compiler.

- **`facts_situations_physical.txt`**: phase changes in household materials,
  conduction and insulation, conservation of matter, separation methods,
  pressure, buoyancy, collisions, machines, energy transformations, musical
  instruments, optics, circuits, and magnetic devices.
- **`facts_situations_living.txt`**: descriptions of cell functions, physiology,
  nutrition and hygiene, plant observations, reproduction and dispersal,
  adaptation mechanisms, inheritance, food webs, pollution, and conservation.
- **`facts_situations_earth.txt`**: erosion and deposition, soils and groundwater,
  water treatment, weather observations, climate, resources, geological evidence,
  apparent celestial motion, lunar phases, orbits, and astronomical instruments.
- **`facts_situations_inquiry.txt`**: evidence and uncertainty, experimental
  controls, replication and sampling, measurement procedures, instrument choice,
  laboratory safety, interpreting graphs, and engineering decisions.

Most new rules use short diagnostic cue combinations. Rules needing explicitly
conditional rather than unconditional evidence can use a negative scale, which
requests pair-only compilation without reversing target signs. A few cues still
contain more than two concepts and inherit the parent's pairwise approximation.

`aliases.txt` now recognizes additional authored inflections and spellings, such
as wilt/wilted/wilting, uproot/uprooted, plow/plough, and
accumulate/accumulation. The sequential selector `next` is retained rather than
silently removed as grammar. The original distinctions between scientific
quantities and mechanisms are retained, and the inherited possessive,
quotation, contraction, and plural expansion remains in place.

A source review also corrected the inherited space-telescope advantage rule.
Its former target text, "avoids atmospheric distortion absorption", supplied
positive evidence for distortion and absorption because targets are bags, not
sentences. It now supports clear images and resolution, locates the telescope
above the atmosphere, and explicitly inhibits distortion and absorption.
New rules about open circuits, nonmagnetic materials, absence of lunar weather,
and other negated outcomes likewise use explicit signed targets rather than
assuming a target word such as "not" reverses the following word.

## Follow-up encoding review

This review examined source meanings and deterministic encoding contracts, not
smoke-question correctness or accuracy. `facts_encoding_review.txt` adds 41
handwritten definitions and qualified relationships. The corrections include:

1. **Density comparisons:** "mass same volume greater" and "volume same mass
   greater" had identical cue bags but opposite consequences. The revised cues
   use "same volume heavier" and "same mass larger", which remain distinguishable.
   These are still pairwise approximations of a fixed-quantity comparison, not
   a general mechanism for binding variables.
2. **Relative polarity:** repeated tokens such as "north north" cannot identify
   two poles in a set-valued bag. Those rules now require an explicit relative
   selector such as "identical". Like/unlike charge and pole relations are
   pair-only, avoiding unconditional claims that north or positive alone repel.
3. **Interrupted states:** disconnected wires, circuit gaps, and switched-off
   electromagnets no longer positively assert current. Conditional rules inhibit
   current/field evidence and have no first-order backoff. An electromagnet's
   temporary nature is separately described as control by a switch and current.
4. **Oxygen removal:** the condition is now the exact pair "oxygen removed".
   The old three-word cue generated a misleading flame/oxygen pair that omitted
   removal. Smothering has an explicit extinguished outcome and negative
   combustion evidence, also pair-only.
5. **Absence and blocked outcomes:** black-hole descriptions inhibit escape;
   small mosses recall nonvascular diffusion rather than positively asserting
   vascular transport; sterile conditions inhibit microorganism or contamination
   evidence. These changes are explicit source edits, not a general and
   potentially incorrect automatic inversion of sentences containing "prevent".
6. **Vocabulary contracts:** British `fossilisation` now matches `fossilization`,
   not the remains concept `fossil`. Interruption and trapping inflections have
   explicit aliases. The source audit now protects `next` from grammar removal.

The expanded regression suite checks the distinct density cue bags, removal of
lossy repeated-polarity cues, the corrected aliases and signed definitions, and
isolated compilation of **27 pair-only rules**. Their positive and negative
memory entries are checked directly, together with a completely zero first-order
matrix. These are compiler/source contracts, not multiple-choice answer tests.

That encoding review changed the maximum resolved hash-row width incidentally
from 57 to 55. The definition review brought it back to 57; the
protective-property review returned it to 55, unchanged by the astronomy review.
The reaction-energy review gave a width of 56; the genetics review returns it
to 55. No hash size, row width,
salience, or mixture coefficient was selected using accuracy or a parameter
search.

## Definition and physical-meaning review

A source-only inventory identified scientific terms used as answer targets but
lacking definitions to recall when they appear in questions. The inventory
counted authored source occurrences, not question frequencies or correctness.
The new `facts_definition_bridges.txt` supplies **213 handwritten definitions
and qualified links**, including:

- ammeters, voltmeters, ohmmeters, adsorption, and measurement methods;
- amylase, pepsin, endosperm, cuticles, cytoskeletons, structural biopolymers,
  microbial nitrogen compounds, immune signals, and nucleic-acid components;
- drag, thrust, torque, deceleration, resonance, photons, optical phenomena,
  photovoltaic devices, and material properties;
- moraines, erratics, glacial striations, recrystallization, desalination,
  leaching, volcanic material, and atmospheric auroras.

The additions are manually selected definitions, not automatic reversal of every
source association. Explicit aliases connect each term's own morphological
forms, while keeping adsorption/absorption, nucleotide/nucleoside,
intermolecular/intramolecular, malleable/ductile, and transform/transformer
separate.

Five existing statements were corrected by scientific meaning:

1. The former claim that water vapor generally has greater particle kinetic
   energy than liquid was replaced by a qualified latent-vaporization rule.
   Latent heat at the standard pure-substance, constant-pressure transition
   changes intermolecular potential energy without raising temperature.
2. Melting is described by disrupted attractions, disorder, and mobility, not
   a universal increase in particle separation; water is an important exception
   to that geometric generalization.
3. Freezing is described by ordering and heat release, not a universal decrease
   in particle speed or spacing during the phase transition.
4. The phase-change rule now specifies intermolecular potential energy rather
   than unqualified "bonds", which could suggest chemical decomposition.
5. An element's atomic mass is qualified as an isotope-weighted average. Mass
   number remains the proton-plus-neutron count, and individual atomic mass is
   still linked to the nucleus; the words `atom` and `atomic` remain aliases.

New source and tensor regressions verify selected definition coverage, nonempty
first-order recall, aliases, preserved distinctions, instrument/quantity links,
signed exclusions, and the five corrected source contracts. No answer ranking,
accuracy measurement, or smoke-output comparison selected these changes.

## Conditional-source review

This source-only review examined conditions that were lost during pairwise
compilation. For example, `food web remove predator` formerly generated a
food/web recall pair even without removal. The revised `predator removed` cue
contains just the two required concepts and uses a negative scale for pair-only
evidence. Neither predator nor removal alone receives first-order support from
that rule. This uses the existing compiler; no new rule format or weighting
formula was introduced.

The review revised **36 source rules** and added **one separate rule**, covering:

- Predator/prey changes, scarcity, organism removal, and decomposer removal.
  Selected cues retain the organism and its condition instead of generating
  generic food/web pairs. These describe simple food-web tendencies with other
  influences held unchanged, not universal ecological predictions.
- Darkness and light-blocked leaves. Positive bags such as "less photosynthesis"
  still supported photosynthesis; the revised rules explicitly inhibit it.
  Covered leaves assume an opaque, light-blocking covering. Already stored food
  is distinguished from ongoing photosynthesis rather than claiming that all
  starch instantly disappears.
- Photosynthesis versus respiration in darkness. The old combined target
  "stops respiration continues" could not bind its two verbs. Photosynthesis
  stopping and respiration continuing now have separate conditional rules.
- Tree/vegetation removal and solar generation at night. Their adverse outcomes
  no longer receive unconditional first-order backoff from those rules.
- Mixed target directions. "Foxes decrease grass increase" was an unbound bag
  containing both directions; the revised removal rule uses the qualified
  consequence concepts predator starvation and vegetation recovery instead.
  This is still associative evidence, not general role binding.

An isolated regression compiles all **37 reviewed conditional rules**, checks
that every cue has exactly two concepts and every scale is pair-only, verifies
all declared target signs, and requires a zero first-order matrix. Its occupied
memory rows must be exactly those addressed by the authored cue pairs. Tests
also reject the former lossy cues and the repaired positive-target mistakes.
This establishes source/compilation contracts, not an accuracy improvement.

Vocabulary was regenerated after these edits. Removing a now-unused concept
also shifts some concept IDs and hash addresses; those incidental changes were
not selected using any prediction output.

## Protective-property review

This review corrected **27 existing rules** whose positive target bags often
included a prevented outcome. For example, "resists burning combustion" still
supported combustion, "without fracture" supported fracture, and "prevents
backflow" supported backflow. These terms now receive explicit negative
evidence, while useful mechanisms and anatomical context remain positive.

Corrections cover nonflammability, malleability, impermeability, hygiene,
disinfection, valves, cartilage cushioning, food preservation, sterile
technique, handwashing, thermal insulation, expansion gaps, and engine cooling.
Impermeability inhibits penetration/infiltration rather than all water movement:
surface runoff above an impermeable layer is still possible. Epiglottis rules
retain the cartilage flap and swallowing mechanism while inhibiting aspiration
and choking. The ideal inertial-trajectory rule now specifies equilibrium and
inhibits acceleration; it does not claim that individual balanced forces cannot
exist.

`facts_protective_review.txt` adds **nine handwritten rules** for aseptic
conditions, anticoagulants, waterproofing, sanitation, disinfectants, aspiration,
and three qualified protective mechanisms. These additions were authored from
scientific meaning, not generated from a rule-word frequency or answer score.
Where a property describes reduction rather than absolute elimination, negative
weights are inhibitory evidence, not a guarantee of perfect prevention. The
model still cannot bind arbitrary prevention scopes in natural-language options.

The regression suite checks explicit source signs, actual first-order inhibition
for the reviewed definitions, and isolated pair-memory signs and exact occupied
rows for the reviewed pair-only mechanisms. It also compiles the negative-only
nonflammability definition in isolation and checks its symbolic not/except
inversion. An initially overstrict regression rejected a valid, unchanged
railroad-gap mechanism rule; the test was corrected to permit general mechanism
evidence while requiring a protective rule and rejecting positive support for
the prevented outcome. All final checks pass.

The compiler and aliases were unchanged. Vocabulary regeneration added a net
four concepts, incidentally changing some hash addresses; the resolved row width
at that stage was 55. Neither the new addresses nor the smaller memory footprint was chosen
by accuracy-based search.

## Astronomy review

This source-level review revised **20 existing statements**, including one
alternative-phase statement split into two rules, and added **32 handwritten
rules** in `facts_astronomy_review.txt`. The net addition is 33 rules.

- Generic summer/winter solstice and spring/autumn equinox definitions no longer
  assume Northern Hemisphere calendar months. Four explicit hemisphere-season
  pairs supply the appropriate March/September equinox associations.
- Spring/neap tidal definitions are pair-only. Their evidence no longer backs
  off unconditionally to the season word `spring` or to tides without their
  qualifier. New and full moons are separate alternative cues for spring tides;
  quarter phase, perpendicular geometry, and alignment likewise retain a tidal
  cue rather than asserting a tidal type from a generic sun/moon pair.
- Speed and gravitational strength change along an ideal elliptical orbit, but
  the orbital period does not change each time the body moves closer or farther.
  Two source rules no longer attach a shorter/longer year to those instantaneous
  positions. New perihelion/aphelion period rules explicitly preserve the year.
  These describe ideal two-body behavior, not perturbation calculations.
- Ion-tail solar-wind effects are distinguished from dust-tail radiation
  pressure. General comet-tail descriptions now include both influences.
- Definitions cover sidereal versus synodic reference periods, synchronous
  rotation, libration, far-side illumination, obliquity, precession, occultation,
  annularity, totality, retrograde/prograde direction, and semimajor geometry.
  They do not claim that the lunar far side is permanently dark or that apparent
  retrograde motion requires a planet physically reversing its orbit.

Explicit word forms include perihelia/aphelia and precession, libration, and
synchronization variants. Rotation and revolution, sidereal and synodic, and
solar versus Earth-centered orbital extrema remain distinct concept IDs.

The expanded regression checks month-neutral generic definitions, hemisphere
contracts, removal of the lossy tidal cues, and isolated compilation of the
reviewed pair-only rules with no first-order evidence. It verifies signs and
exact occupied rows, source distinctions for orbital periods and comet tails,
new definition recall, and alias equivalence/distinction. All checks passed
without using answer ranks or accuracy to choose any change.

## Reaction-energy and equilibrium review

This review revised **12 existing rules** and added **29 handwritten rules** in
`facts_reaction_review.txt`. The changes separate kinetic effects from
thermodynamic quantities and retain the conditions needed for each statement.

- `catalyst reaction heat` formerly generated a reaction/heat pair asserting
  unchanged enthalpy even without a catalyst. It is now the exact pair
  `catalyst enthalpy`. Related activation, equilibration, reverse-reaction,
  pathway, and regeneration rules also retain an explicit catalyst cue and
  suppress unconditional first-order evidence.
- The ambiguous cue "reaction speeds unchanged" has been replaced by a
  qualified acceleration association. Heat release and heat absorption now
  have separate pair-only exothermic/endothermic associations.
- Individual bond breaking/dissociation requires energy; bond formation releases
  it. Their signed rules distinguish these contributions rather than claiming
  that breaking bonds alone supplies the net released energy of a reaction.
- Enthalpy is defined as a state function involving internal energy, pressure,
  and volume, not as heat itself. A separate enthalpy-change rule relates it to
  constant-pressure heat under the usual pressure-volume-work assumption.
- Chemical equilibrium has equal forward/reverse rates and constant
  concentrations. The rules do not assert that the concentrations must be
  equal or must be unequal. Reactions continue at dynamic equilibrium.
- Open, closed, and isolated systems are distinguished by qualified exchange
  rules. Inhibiting matter or energy exchange does not assert that a system
  contains no matter or no energy.

Explicit aliases normalize equilibration, dissociation, regeneration, and
reversibility forms while keeping equilibrium/equilibration,
dissolution/dissociation, and catalyst/catalysis distinct.

The expanded regression rejects the replaced ambiguous cues, checks scientific
source contracts, and compiles **26 selected pair-only rules** in isolation.
It requires zero first-order support, the exact authored occupied hash rows,
and the declared positive/negative signs. Additional checks protect constant
versus equal concentration, the enthalpy definition, boundary-exchange scope,
new definitions, and alias distinctions. All final checks passed; none used
question correctness or answer rankings to select model content.

## Genetics and individual-response review

This review revised **nine existing rules** and added **34 handwritten rules**
in `facts_genetics_review.txt`.

- Heterozygosity means different alleles, not necessarily complete dominance.
  Recessive expression no longer universally requires two copies: separate
  homozygous and hemizygous rules distinguish the usual diploid case from
  expression when only one copy is present. Codominance and incomplete dominance
  have their own definitions.
- Plant cloning recalls a shared genotype rather than guaranteed identical
  traits. Environment, phenotypic plasticity, and subsequent mutation can
  produce differences among clonal descendants.
- Mutation effects include neutral, beneficial, and harmful possibilities;
  mutation is not defined as necessarily changing a protein or visible trait.
  Synonymous, missense, nonsense, and frameshift definitions distinguish coding
  effects without asserting that synonymous changes have no possible cellular
  consequences.
- Somatic mutations are no longer categorically excluded from all transmission:
  they can pass to mitotic descendants and some asexual offspring. Germline and
  somatic lineages remain distinct. This removes a universal claim rather than
  claiming that ordinary acquired scars pass to sexually produced offspring.
- Acclimatization is an individual physiological adjustment, distinct from
  population-level evolutionary adaptation. Acquired-change and exercise rules
  retain their conditions rather than inhibiting genetics unconditionally.
- New definitions cover penetrance, expressivity, pleiotropy, polygenic traits,
  genetic drift, founder effects, bottlenecks, gene flow, and reproductive
  fitness. Pleiotropy and polygenic targets use distinct wording: simply swapping
  "one gene / multiple traits" into "multiple genes / one trait" would produce
  the same bag and lose the role distinction.

Explicit aliases preserve acclimatization/adaptation, genotype/phenotype,
dominance/codominance, and homozygous/heterozygous/hemizygous distinctions while
recognizing each term's own forms. The regression checks repaired source
contracts, direct definition recall, alias equivalence/distinction, distinct
pleiotropy/polygenic target bags, and eight selected pair-only rules compiled
in isolation with zero first-order support and the declared memory signs.
All final checks pass. These are source and encoding improvements, not a measured
improvement in question-answer accuracy.

## Representation and weights

Each source fact has the form:

```text
question cues | supported answer concepts | optional inhibited concepts | optional strength
```

`build.py` reads only the vocabulary and authored sources in this folder. It
returns exactly:

- `concept`: int64 word-to-concept lookup; padding and unknown IDs map to zero;
- `idf`: float32, manually specified semantic salience;
- `A`: dense float32 first-order support/inhibition matrix;
- `row_ids` and `row_w`: int64/float32 hashed pair-recall tables;
- `mix`: scalar float32, **0.65**.

The inherited deterministic compiler uses independent positive and negative
maxima, so repeated rules do not accumulate like training observations. It
normalizes target lists, damps broad first-order rows, suppresses unconditional
backoff for negated or pair-only conditions, preserves negators when generating
pairs, and adds bounded symbolic contrasts and not/except inversions. Exact
semantic pairs are resolved before hashing. Every resolved nonzero hash-row
target is retained.

Inference uses the question's strongest sixteen concepts and their 120 pairs.
It reads only the compiled tensors; it does not retrieve source text or inspect
question files while building.

Current dimensions:

| Quantity | Value |
|---|---:|
| Authored rules | 7,600 |
| Vocabulary token forms | 60,592 |
| Concepts, including none | 3,885 |
| Dense first-order matrix | 3,885 × 3,885 |
| Pair memory | 524,287 × 55 |
| Tensor elements | 72,829,275 |
| Nonzero tensor elements | 275,898 |
| Tensor storage | 406,902,616 bytes, about 388.1 MiB |

Vocabulary forms include finite spelling/possessive/quotation variants; they are
not 60,592 independent scientific concepts.

## Reproduction and verification

Regenerate the vocabulary after changing authored source files:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/build.py
```

Verification commands:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/audit.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

**All three verification commands passed.** The required checker reported
`CHECK OK`, with 72,829,275 weights and 275,898 nonzero elements. It also executed
twelve original, unlabeled smoke questions covering multiple subject areas.
Those outputs were an execution check, not an accuracy benchmark; no knowledge,
aliases, compiler settings, or weights were changed in response to them.

The source audit and expanded label-free regression suite check
source consistency, tensor shapes/dtypes/bounds, finite values, zero unused
slots, architecture identity, alias distinctions, spelling equivalence,
question masking, option/token order invariance, duplication invariance,
unknown-input ties, build determinism, negation-preserving pairs, and isolated
hash-collision/exception semantics. They do not fit rules to answers.

Measured locally with four Torch threads:

| Check | Result |
|---|---:|
| Build time | 0.739 seconds |
| Ten forward batches of 64 synthetic inputs | 0.266 seconds |
| Full expanded regression suite | 5.945 seconds |
| Peak regression RSS | 1,314,420 KiB, about 1.25 GiB |

The regression includes multiple builds and simultaneous comparison tensors.
It remains below the 2 GiB limit, with substantial time margin under 20 minutes.

Architecture SHA-256:

```text
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

Tensor fingerprint from the regression suite:

```text
734aa56a78a919b916e3238ffe896b216693a53125e384e30cb4c73a35f7f76b
```

## Limitations

This remains a bag-and-pair association memory, not a language model or a general
reasoning system. It cannot reliably recover word order, bind arbitrary roles,
perform general arithmetic, inspect diagrams, or model complete negation scope.
It has only sixteen pair-active concepts per question, can suffer hash collisions
and polysemy, and scores options additively without length normalization.

The new descriptive coverage and source repairs are motivated by science and
encoding semantics. **No improved ARC accuracy is claimed without evaluation.**

# Generation 5: descriptive and conditional science association memory

## Model and provenance

This model extends the supplied generation-4 memory with **1,469 handwritten
rules**, for **10,601 authored rules** in total. It uses the supplied fixed
second-order association architecture. Nothing is trained, fitted, downloaded,
or pretrained. No ARC questions beyond those in TASK.md were accessed; no
question labels, model accuracy, or answer rankings guided the changes.

The parent's reported 45.5% validation accuracy is background information, not a
measurement of this revision. Improved held-out accuracy is an aim, not a
verified result.

`arch.py` is unchanged, with SHA-256:

```
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

## What changed and why

### Descriptions recall scientific concepts

Many elementary science prompts describe an observation instead of naming its
mechanism. Ten new files supply explicitly authored links and conditional reviews:

| Source file | Rules | Coverage |
|---|---:|---|
| `facts_g5_life_descriptions.txt` | 221 | Cell structures, physiology, plant processes, inheritance, animal classification |
| `facts_g5_physical_descriptions.txt` | 252 | Observable material behavior, heat, separation, motion, simple machines, waves, optics |
| `facts_g5_earth_descriptions.txt` | 295 | Rock identification, landforms, soil, hydrology, weather, geological evidence, astronomy |
| `facts_g5_ecology_applications.txt` | 254 | Food webs, population changes, adaptations, succession, pollution, conservation |
| `facts_g5_inquiry_materials.txt` | 232 | Operational experimental descriptions, instruments, engineering, safety, atoms, circuits |
| `facts_g5_conditional_review.txt` | 67 | Quantity-specific directions, trophic transfer, introduced species, dormancy, error types, dilution |
| `facts_g5_mechanics_review.txt` | 53 | Motion qualifiers, projectile components, gravity, pendulum quantities, waves, photon proportionalities |
| `facts_g5_matter_review.txt` | 25 | Concentration distinctions, bond energetics, phase-change plateaus, molecular identity, heat measurement |
| `facts_g5_inquiry_review.txt` | 35 | Sampling uncertainty and bias, random assignment, control types, correlation, graph conditions, measurement patterns |
| `facts_g5_electric_review.txt` | 35 | Electrical quantities and units, conditional Ohmic comparisons, transformer selectors, induction paths, unmagnetized materials |

For example, `colored celery` recalls xylem and water transport; `interlocking
rock` recalls igneous crystallization; and `particles collisions` recalls gas
pressure. These are manually supplied associations, not automatic reversals of
all inherited facts or inference-time retrieval.

Short two-concept cues improve the chance of matching a paraphrased description.
Conditional population, electrical, and physical relationships use pair-only
recall when isolated cue words would give misleading unconditional evidence.

### Lexical coverage without merging distinct science concepts

`aliases_g5.txt` supplies explicit inflections and spelling variants such as
swollen/swell, shriller/shrill, percolating/percolation, recharging/recharge,
acidified/acidification, and subducting/subduction. The compiler reads this file
using the parent's existing whole-group alias resolution.

An inherited merger of **spongy** with **sponge** was separated: the texture of
leaf tissue is not the filter-feeding animal or an absorbent object. Distinctions
such as biomagnification/bioaccumulation, heat/temperature, mass/weight,
weather/weathering, and reproduction/reproducibility remain intact.

### Explicit signs and preserved conditions

Source review and label-free contracts caught phrases whose literal word bags
would encode the wrong claim. Examples repaired in new or overlapping inherited
rules include:

- `zero acceleration`: support zero/balance and inhibit acceleration;
- `reduced conduction`: support insulation and inhibit conduction/convection;
- boiled seeds: inhibit germination only under the boiling condition, not from
  the isolated word seed;
- frozen soil: support runoff and inhibit infiltration;
- lunar crater preservation: inhibit erosion rather than positively recalling it
  from the phrase "little erosion";
- wire comparisons: separate resistance and current cues instead of placing
  increase and decrease together in one undifferentiated answer bag.

A tripped breaker's excessive-current cause remains distinct from the
current-stopping effect of an opened breaker. Source tests check conditions and
signs, not which option wins on a question.

### Follow-up review: quantities, conditions, and scientific distinctions

The requested follow-up used source meaning and compiler contracts only. It added
67 rules and corrected overlapping inherited/new statements without changing any
weight formula or evaluating answer accuracy.

- **Trophic levels:** the phrase "higher trophic level" had accidentally placed
  the canonical concept *increase* in an energy-pyramid target bag. The general
  definition now recalls trophic transfer and heat loss. Separate higher/lower
  level conditions encode the appropriate energy directions. They do not assert
  that every standing-biomass pyramid is upright; aquatic biomass can be inverted.
- **Introduced versus invasive:** introduction describes origin and transport;
  it does not imply harmful spread. General introduced-species rules now recall
  nonnative origin, while explicitly invasive conditions retain ecological harms.
  Decomposers are associated with detritus and all trophic levels, not asserted to
  be the universal endpoint of every food chain.
- **Reduced quantity versus absent quantity:** for reviewed reductions, the
  quantity remains a positive recall target alongside *decrease*, with *increase*
  inhibited. This matters when an option says "reduced drag" or a question asks
  which quantity was reduced. Streamlining, shell water conservation, spines,
  resource partitioning, experimental blinding, thermal pollution, and calorimeter
  insulation use this encoding. Quantity-specific pairs add narrower directional
  recall. This is not a blanket rule that inhibitory targets are inappropriate;
  true absences and excluded processes still use explicit inhibition.
- **Dormancy and error mechanisms:** lower metabolism does not imply death or
  absent respiration. Repetition/averaging can reduce random error but does not
  automatically remove systematic bias. Calibration and blinding have distinct
  roles. Dilution rules distinguish fixed solute amount from concentration, and
  comparisons assume the other relevant component amount is held fixed.
- **Conditional consistency:** matching inherited records were corrected as well;
  reviewed conditional facts cannot silently reintroduce unconditional backoff
  through another spelling or duplicate source file. New aliases connect introduced
  inflections, partitioning, metabolic/metabolism, masking, allocation, and blinding.

These are limited associations under their stated conditions. Additive bags still
cannot bind arbitrary directions to different quantities in the same sentence.

### Mechanics and wave review

A further source-only review added 53 rules and repaired inherited records whose
pair expansion lost essential physical qualifiers. All formulas and architecture
remain unchanged; no answer rankings or accuracy measurements were used.

- **Constant speed versus constant velocity:** straight and turning conditions
  must not both write categorical outcomes into the unqualified constant-speed
  pair. That pair now recalls magnitude only. Turning/circular motion recalls
  changing direction and acceleration; uniform rectilinear motion recalls
  equilibrium. Sustained rest is distinguished from instantaneous rest.
- **Projectile components:** horizontal-component rules no longer create a
  misleading projectile/vacuum pair that inhibits total acceleration. Generic
  ball/apex recall does not assign an unqualified zero. Vertical velocity and
  gravitational acceleration have separate apex cue pairs.
- **Relationships versus directions:** gravity/mass, gravity/distance,
  pendulum/length, and photon relations recall dependencies or proportionalities,
  not an unconditional increase or decrease. Wave comparison rules retain their
  frequency/wavelength selector rather than projecting a change onto every
  wave/frequency question.
- **Pendulum energy:** extreme-position definitions identify the energy at its
  maximum without putting maximum and minimum into the same target bag for two
  different energy quantities. Period, mass, damping, and amplitude have separate
  associations.
- **Explicit assumptions:** gravity comparisons fix other masses/distances;
  frequency/wavelength comparisons fix propagation speed; pendulums use the
  small-angle model; projectile components neglect air resistance; linear-wave
  amplitude comparisons hold frequency and medium fixed.

The regression suite checks every source fact containing each reviewed
unqualified pair, preventing a differently ordered or longer duplicate from
reintroducing the lost-condition error. These repairs do not make the fixed
architecture a general parser for arbitrary multi-condition statements.

### Matter and energy review

A source-only matter review added 25 rules and corrected overlapping records.
No answer scores or accuracy measurements were used.

- **Concentration versus concentrated:** the neutral quantity and the adjective
  describing a relatively high solute ratio now have distinct concepts. Plural
  *concentrations* still maps to *concentration*. This also prevents question
  masking from treating the neutral quantity as the high-concentration state.
- **Concentration versus saturation:** a concentrated solution can be unsaturated,
  and a dilute solution can be saturated when solubility is low. These combined
  conditions are pair-only across all matching records, rather than making either
  adjective imply the other condition without context.
- **Bond energetics:** all matching breaking/formation records now preserve their
  two-cue condition. Breaking absorbs energy; formation releases energy. Existing
  signed rules cannot be bypassed by unconditional backoff from duplicate spellings.
  Whole-reaction energetics depends on the balance of the separate steps.
- **Heat and mass:** an ambiguous target bag combining more energy, larger
  temperature change, and resistance was replaced with the variables in the
  sensible-heating relationship, without asserting an unconditional direction.
- **Phase-change descriptions:** new plateau associations explicitly assume pure
  substances, fixed pressure, and equilibrium phase coexistence. Physical changes
  of molecular substances distinguish intermolecular attractions from intact
  intramolecular bonds, without asserting chemical decomposition. These assumptions
  do not imply that every evaporation process has constant temperature.

That review brought the suite to 94 signed-condition contracts, including 14 new
contracts, and added checks for lexical distinctions and the neutral heat/mass
relationship. The architecture and compiler formulas remained unchanged.

### Experimental design, measurement, and graph review

The source-only inquiry review added 35 rules, repaired matching records, and
expanded verification to 117 conditional contracts. No question outputs, labels,
answer rankings, or accuracy measurements guided the changes.

- **Sample size versus representativeness:** larger samples of independent
  observations reduce sampling uncertainty under a fixed design and population;
  they do not automatically
  remove selection bias or guarantee representativeness. Smaller samples have
  greater sampling uncertainty, not necessarily selection bias. General sample-size
  definitions now name dependencies without asserting an unconditional direction.
- **Sampling versus assignment:** random sampling concerns selection from a
  population; random assignment concerns treatment groups and confounding.
  Matching records retain their two-cue condition. Random procedures have expected
  statistical benefits, not guaranteed perfect balance in every realization.
  Representative samples are defined by population characteristics, not by a claim
  that random selection is the only possible way to obtain one.
- **Bias and measurement:** increasing or repeating a biased recruitment process
  need not eliminate its bias. Systematic sampling at regular intervals is distinct
  from systematic measurement error. Clustered readings indicate precision, not
  accuracy by themselves; persistent offsets from a reference indicate bias.
  These measurement descriptions assume a stable measurand.
- **Control groups:** general control definitions specify a baseline/reference,
  not mandatory absence of treatment. Specific active, placebo, positive, and
  negative controls have separate pair-only associations. Assay-control outcomes
  describe expected behavior of a properly functioning procedure.
- **Correlation:** association alone does not prove causation, but it also does not
  rule causation out. A source that inhibited causation itself was corrected to
  inhibit proof instead. All matching reviewed correlation rules are pair-only.
- **Graphs:** a straight line has constant slope but need not pass through the
  origin or rise. A rising line does not necessarily express direct proportionality.
  Explicit origin/linearity and direct/inverse proportionality associations retain
  their selectors. Distance-versus-time slopes recall speed; displacement-versus-time
  slopes recall velocity. These motion-graph associations explicitly assume elapsed
  time on the horizontal axis; the model does not inspect diagrams.
- **Lexical coverage:** handwritten inflections connect biased/bias,
  assigned/assignment, correlated/correlation, and other inquiry descriptions.
  Sampling/assignment, correlation/causation, uncertainty/bias, and
  accuracy/precision/resolution remain distinct concepts.

All weight formulas, hash dimensions, salience rules, and the supplied architecture
remain unchanged. The inherited regression suite is unchanged. New source guards
cover overlapping longer cues as well as exact pairs, while isolated compilation
checks require zero unconditional backoff for the reviewed conditional records.

### Electricity and magnetism review

The latest source-only review added 35 rules and 14 conditional contracts, bringing
verification to 131 contracts. No question accuracy or answer rankings were used.

- **Device versus electrical quantity:** resistor, resistance, and resistivity now
  have separate concepts. Resistance depends on geometry and material; resistivity
  is a material property at specified conditions. Conductance is distinguished
  from conductivity. Authored unit and reciprocal relationships distinguish ohms,
  ohm-metres, siemens, and siemens per metre without changing the compiler.
- **Unmagnetized versus nonmagnetic:** these are no longer aliases. Unmagnetized
  iron is still ferromagnetic and can be attracted. New associations distinguish
  orientations across domains from alignment within a domain, and define remanence.
  The inherited elementary nonmagnetic examples describe ordinary strong
  ferromagnetic attraction, not the absence of every possible magnetic response.
- **Ohmic comparisons:** bare current/voltage and current/resistance pairs now
  express proportionality rather than an unconditional increase or decrease.
  Explicit resistance and voltage comparisons retain their selectors, assuming
  uniform ohmic conductors and the other relevant electrical quantity held fixed.
- **Coils and transformers:** general winding/current definitions no longer assert
  a field-strength increase without a comparison. Transformer turns refer to the
  primary/secondary ratio, not an unspecified turn-count increase. Step-up/down
  conditions have separate pair-only recall, including up/down cues for hyphenated
  wording. Frequency and power statements use the ordinary ideal, linear AC model.
- **Induction:** the general definition recalls induced voltage/emf from changing
  linked magnetic flux. Specific open/closed path conditions distinguish induced
  voltage from induced current, neglecting capacitive transients. Existing broader
  descriptive associations remain subject to the architecture's context limitations.

All matching records for the reviewed comparisons were corrected. Source guards
check electrical unit distinctions and overlapping cues; isolated tensor tests
check conditional signs and absence of unconditional backoff. The architecture,
weight formulas, and inherited regression suite remain unchanged.

## Weight construction

The inherited weight formulas are retained. The only compiler change is loading
the additional authored alias file.

Source format:

```
question cues | supported answer concepts | inhibited concepts | optional scale
```

A negative scale requests pair-only recall; it does not reverse target signs.
Surface words map to concepts. Fixed semantic salience supplies `idf`. The
first-order matrix supplies broad backoff, while hashed concept pairs recall
signed answer concepts. Independent positive and negative maxima make duplicate
source records idempotent. Short-target normalization, generic-cue damping,
negation-preserving pairs, bounded contrasts, definition inversion, and the
first-order mixture **0.65** are unchanged.

The address space remains **1,048,573** rows. Every nonzero resolved row target is
retained. Row width is derived from the widest resulting row, not selected by
search or evaluation. Changes in concept numbering and hashed row unions account
for the different width from the parent.

`build()` reads only the vocabulary and authored science/alias sources in this
folder, and returns exactly the six required tensors. Verification scripts are
not consulted during construction.

## Final dimensions and measured resources

| Quantity | Value |
|---|---:|
| Authored rules | 10,601 |
| Unique compiled lexical variants | 10,551 |
| Vocabulary entries | 68,808 |
| Concepts including none | 4,312 |
| First-order matrix | 4,312 × 4,312 |
| Pair table | 1,048,573 × 57 |
| Occupied pair rows | 16,591 |
| Tensor elements | 138,203,789 |
| Nonzero tensor elements | 326,436 |
| Tensor storage | 792,165,040 bytes (about 755.5 MiB) |
| Measured full build, four threads | 1.871 seconds |
| 4,096 synthetic inputs, batches of 64 | 1.077 seconds |
| Peak RSS, inherited full regression process | 1,371,996 KiB (about 1.31 GiB) |

Synthetic inputs test execution and invariants, not science-answer accuracy.
The measured build and runtime are comfortably within 2 GB and 20 minutes.

## Verification

Run with four CPU threads:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/build.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/audit.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify_g4.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify_g5.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

The source audit and both regression suites pass. The inherited suite is kept
unchanged. It checks lexical distinctions, formula variants, conditional signs,
negation, duplicate-rule invariance, tensor validity, question masking, option
permutation, token-order and repeated-token invariance, and unknown-input ties.
The new suite now checks **131 signed conditional contracts**, additional alias
checks, introduced/invasive and trophic-direction distinctions, mechanical
qualifier preservation across relevant source records, concentration and
molecular-bond distinctions, sampling/control/graph conditions, electrical
quantities and transformer selectors, and isolated compilation requiring zero
unconditional backoff.

The source audit, both regression suites, and required checker passed on the
final model. The checker reported **CHECK OK**, **138,203,789 weights**,
**326,436 nonzero entries**, and **68,808 vocabulary words**. It used its default
placeholder, not ARC questions or a labeled custom question set.

## Limitations

This remains a bag-and-pair memory, not a general reasoner. It cannot reliably
bind arbitrary comparative roles, resolve every negation scope or homograph,
interpret diagrams, or perform general arithmetic. Only the sixteen strongest
question concepts participate in pair recall; hash collisions remain possible.
Additive option scoring can confuse a mentioned quantity with a claim about that
quantity. Signed and pair-only rules reduce specific encoding errors but do not
remove these architectural limits.

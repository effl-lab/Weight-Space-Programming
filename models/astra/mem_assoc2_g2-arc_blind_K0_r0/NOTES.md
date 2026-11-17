# Reviewed hand-authored science association memory

## Model and provenance

This model compiles **5,614 handwritten science rules** into the six tensors
required by the supplied second-order association-memory architecture.
**`arch.py` is unchanged.** There is no training, fitting, pretrained model,
external corpus, internet access, or accuracy-based search.

The parent supplied 3,077 rules and a symbolic compiler. The present model has
2,537 more rules in net count, plus corrections to inherited and newly authored
rules. Coverage emphasizes familiar situations, causal mechanisms, adaptations,
environmental effects, measurement, and reverse descriptions. The aim is broader
and more faithful recall, not merely a larger technical glossary.

Held-out accuracy is unknown. The parent's reported 42.1% is not a measured
result for this version. Neither smoke predictions nor accuracy measurements
selected the review changes; the review used source semantics, structural audits,
and synthetic compiler regression fixtures.

## Knowledge files

Nine inherited files cover biology, chemistry, physics, Earth science,
astronomy, methods, reverse recall, conceptual distinctions, and foundations.
They remain the foundation, with targeted scientific and encoding corrections.
Nine additional files supply application knowledge and review cross-links:

- **`facts_applications.txt`**: heat transfer, insulation, phase changes,
  expansion, friction, inertia, pressure, buoyancy, machines, circuits,
  magnetism, sound, optics, and energy conversions.
- **`facts_life_applications.txt`**: organism adaptations, plant transport and
  dispersal, pollinators, ecological changes, conservation, nutrient cycles,
  cell mechanisms, heredity, evolution, physiology, and health.
- **`facts_matter_applications.txt`**: particles, physical and chemical changes,
  separation methods, solutions, materials, reactions, atoms, and nuclei.
- **`facts_earth_applications.txt`**: landforms, weathering, erosion, soils,
  rocks, tectonics, fossils, groundwater, weather, climate, resources, and space.
- **`facts_inquiry_applications.txt`**: evidence, hypotheses, controls,
  replication, measurement, instruments, graphs, models, engineering, and safety.
- **`facts_crosslinks.txt`**: exception categories, comparisons, definitions,
  tokenization variants, and descriptive recall links.
- **`facts_review.txt`**: qualified comparisons, circuit states, metamorphosis,
  measurement meanings, conductivity, indicators, eclipses, device roles,
  isotopic averages, and fronts.
- **`facts_solution_review.txt`**: concentration and dilution, solubility versus
  dissolution rate, solution phases, colloids and suspensions, catalyst effects,
  and distinctions between rust, tarnish, and other corrosion.
- **`facts_mechanisms_review.txt`**: adhesion/cohesion, capillarity, plant tropisms,
  molting, nymph development, and insect respiratory structures.

`aliases.txt` supplies explicit morphological and semantic equivalences. Related
but distinct concepts remain separate where needed: pollinator/pollination,
engine/motor, voltage/volt, endotherm/endothermic, and complete/closed, for example.
The vocabulary contains **52,280 supported token forms**, mapped into **3,375
concepts including none**. Most additional forms are deterministic possessive
and quotation variants, not new scientific concepts. Padding and unknown word
IDs map to concept zero and supply no evidence.

## Follow-up review improvements

1. **Preserve discriminating words.** The inherited grammar list discarded
   `above`, `below`, and `last`. Thus “pH above seven,” “pH below seven,” and
   “pH seven” previously collapsed into the same cue. These selectors now have
   explicit concepts; above/below also receive comparison salience. The audit
   rejects discarded or accidentally merged selectors.
2. **Do not drop a condition's negation.** A rule with three or more cue concepts
   previously generated all cue pairs, including pairs without its negator.
   For example, a condition meaning “satellite without gravity” could assert
   its consequence for the ordinary satellite/gravity pair. Negated rules now
   generate only pairs containing a negator. Their unconditional backoff remains
   disabled. This is still an approximation, not full three-way logic.
3. **Apply semantic precedence before hashing.** Pair identities are retained
   until logical compilation is complete. An authored exception protects only
   its exact semantic pair, not an unrelated pair that happens to share a hash
   row. Signed collision unions are formed only at final storage. Synthetic
   regressions exercise both a real hash collision and exact-pair precedence.
4. **Represent poor conduction correctly.** Rules such as “nonmetal | poor
   conductor” previously supported the conductor concept and could infer
   negative insulation evidence. They now explicitly support insulation/high
   resistance and inhibit conduction, with a qualified graphite exception.
   Open-switch rules likewise use explicit absent/stopped states rather than
   treating “stop current” as positive current evidence.
5. **Separate overloaded aliases.** Close, closed, complete, and incomplete no
   longer conflate proximity, circuit state, and completeness. Explicit circuit
   links preserve their appropriate relationship. Complete/incomplete
   metamorphosis and closeness to a true measurement retain their own meanings.
6. **Respect the loss of word order.** Reversed litmus-color sequences, reversed
   electrical/mechanical conversion descriptions, eclipse-object permutations,
   and reversed warm/cold-front wording cannot encode different meanings in a
   bag. Unqualified cases now recall a shared family; qualified final-color,
   input/output, visibility, and advancing-front descriptions provide more
   specific evidence when their cues are present.
7. **Expand regression coverage without fitting.** New tests cover selector
   retention, separated aliases, conductivity signs, negation-preserving pairs,
   hash-collision isolation, and exact-pair exception precedence. These tests
   check declared compiler semantics, not science-question accuracy.

## Orthographic review

The fixed tokenizer retains ASCII apostrophes. Consequently, `earth's`,
`plants'`, and `'density'` are whole tokens rather than automatically normalized
words. Previously these forms lost their scientific concepts as unknown tokens.

The compiler now expands possessives and single-quote variants, including tokens
at the beginning or end of quoted phrases. Expansion preserves explicit aliases,
adds no concepts, and is finite and idempotent. Additional authored aliases cover
negative contractions such as `wasn't`, `didn't`, and `wouldn't`, and spelling
variants such as `behaviour`, `aeroplane`, and `labelled`.

That orthographic review increased the vocabulary from 8,194 to 51,756 token
forms while retaining the then-current 5,496 science rules and 3,345 concepts.
A before/after comparison verified that `idf`, `A`, `row_ids`, `row_w`, and `mix`
were bit-identical across that change, and every original word retained its
concept ID. Only the lookup expanded, increasing tensor storage by 348,496 bytes.
This is a historical preservation check, not a claim that later science
revisions leave those tensors unchanged. Always encode word IDs using the
current vocabulary.

New label-free tests verify root preservation, idempotence, explicit-alias
precedence, tokenizer behavior, and exact bag/logit equivalence for representative
possessive and quoted spellings in both questions and options.

## Chemistry review

A source-level review corrected seven existing statements and added 68 authored
rules. No question predictions, labels, or accuracy measurements selected these
changes. The compiler, salience formulas, hash size, mixture coefficient, and
architecture were not changed in this review.

- The volume-based concentration definition now refers to **solution volume**,
  not solvent volume. Separate molarity/molality definitions retain their
  different denominators and units.
- Dilution lowers concentration without reducing the total solute particle
  count. Color statements now describe concentration/absorption rather than
  implying that adding solvent destroys particles.
- A solvent is not necessarily liquid. Explicit examples cover gaseous solutions
  and solid alloys, alongside aqueous solutions.
- Dissolution rate is distinct from equilibrium solubility. Stirring/crushing
  can speed dissolving without changing the equilibrium limit at fixed
  temperature and solvent quantity. The constancy rules are **pair-only**,
  avoiding an unconditional assertion that stirring changes nothing.
- Solutions, suspensions, colloids, and emulsions have separate concepts and
  rules for particle size, settling, filtration, and light scattering. Soap's
  oil-dispersing role now recalls an emulsion rather than a coarse suspension.
- Catalyst rules distinguish faster kinetics/lower activation energy from
  unchanged equilibrium position and reaction enthalpy. Thermodynamic exceptions
  are pair-only and explicitly signed.
- Silver tarnish now recalls sulfur and black sulfide. General oxidation no
  longer implies rust universally; rust is linked specifically to iron.

Regression additions check the corrected source definitions and distinct
concepts. Four actual chemistry rules are covered by the isolated conditional-rule
fixture, which verifies signed pair evidence and zero first-order backoff. These
are source and compiler contract tests, not multiple-choice accuracy tests.
Broader context and scientific exceptions remain limited by the fixed bag/pair
architecture.

## Mechanism coverage review

An inventory of authored cue/target concepts found several meaningful terms that
were used only as targets, with no definitions to recall when those terms appear
as cues. This is a source-coverage check, not an accuracy measure. Fifty additional
rules provide selected definitions and qualified mechanisms; arbitrary relations
are not automatically reversed.

- Adhesion and cohesion now have distinct definitions and links to surface
  tension, wetting, droplets, and menisci.
- **Capillarity is no longer aliased to capillary.** The liquid phenomenon has
  its own definition. Qualified rules connect capillary action to liquid
  movement and blood capillaries to exchange. Water rise and mercury depression
  are distinguished; the generic phenomenon is not restricted to upward motion.
- The existing general phototropism definition now describes directional growth
  in response to light, rather than assuming it is always toward light.
  Gravitropism, hydrotropism, thigmotropism, and auxin have explicit definitions.
  Positive/negative direction rules are pair-only, so the words positive and
  negative alone do not unconditionally recall plant-growth directions.
- Molting, exoskeletons, nymphs, spiracles, and insect tracheae receive definitions
  and qualified growth/respiration links. Geotropism is an explicit alias of
  gravitropism; distinct tropism types remain separate.

Regression checks protect the capillary/capillarity distinction, definition
coverage and nonempty first-order recall for nine selected terms, adhesion versus
cohesion signs, and the direction-neutral general phototropism definition. The
conditional-rule fixture now includes six signed tropism rules alongside four
chemistry exceptions. The compiler, salience formulas, hash size, and architecture
remain unchanged. The smaller maximum memory-row width in this revision is an
incidental result of the new concept IDs and resulting hash layout, not a searched
or accuracy-selected parameter.

## Punctuation and negation review

The fixed ASCII tokenizer recognizes `isn't` as one token but splits `isn’t`
into `isn` and `t`. The latter previously lost its negation. Thirteen explicit
aliases now map unambiguous fragments to the existing `not` concept:
`isn`, `aren`, `wasn`, `weren`, `doesn`, `didn`, `hasn`, `hadn`, `wouldn`,
`shouldn`, `couldn`, `mustn`, and `needn`. Existing finite quotation/possessive
expansion brings this change to 104 additional token forms.

A before/after structural comparison verified that `idf`, `A`, `row_ids`,
`row_w`, and `mix` remain bit-identical, and all 52,176 previous word-to-concept
mappings are preserved. No science rules, concepts, compiler formulas, or
architecture changed in this pass. Tensor storage grew by only 832 bytes.

Tests cover exact concept-bag and logit equivalence for all thirteen contractions
under right/left curly, modifier-letter, and fullwidth apostrophes, both in
questions and in options. Existing hyphenated `non-` handling was already correct;
new regressions preserve its equivalence to explicit `not` for five example terms.
These are unlabeled equivalence tests, not answer-selection benchmarks.

The repair deliberately does not map the valid words `can`, `don`, `won`, or
`haven` to negation, and does not give the trailing `t` fragment evidence. Thus
curly-apostrophe versions of `can't`, `don't`, `won't`, and `haven't` remain a
limitation rather than corrupting those words globally. Split contractions also
consume an extra token, so equivalence near the fixed 128/48-token truncation
boundaries is not guaranteed. Negation scope within options remains subject to
the same bag-of-concepts limitations as before.

## Rule representation and compilation

Each fact has the form:

```text
question cues | positive answer concepts | optional negative concepts | optional strength
```

An omitted strength is 1. A negative strength requests pair-only evidence using
its magnitude; it does not reverse the answer-field signs.

The deterministic compiler:

1. Normalizes words, resolves alias roots, and audits source consistency.
2. Assigns fixed semantic salience. Broad inquiry wording is below specific
   science concepts; production, requirement, comparisons, and negation have
   explicit priorities. Disconnected concepts receive zero salience.
3. Compiles weak single-cue backoff into dense `A`. Multi-cue conditions have
   weaker backoff, broad rows are damped, and negated/pair-only rules have none.
4. Compiles eligible exact cue pairs, retaining independent positive and negative
   maxima. Repetition is not frequency-based reinforcement; opposing evidence
   cancels independently of source order.
5. Normalizes by unmasked target count and adds explicit contrast inhibition and
   bounded symbolic `not`/`except` inversions, respecting exact-pair exceptions.
6. Folds the results into **H = 524287** hash rows, approximately twice the
   parent's address space. Width is the maximum resolved row occupancy, **55**.
   All resolved authored recall is retained, including unavoidable hash unions.
7. Uses the inherited **`mix = 0.65`**. Every fact is folded into tensors; inference
   does not retrieve facts or read source text.

Vocabulary generation uses conservative plural expansion with explicit aliases
first. Generated aliases are resolved to final canonical roots, and redundant
pluralizations of already-inflected source words are discarded rather than
creating disconnected concepts such as `organses`.

`build()` reads only files in this folder. It does not read questions, labels,
probe outputs, evaluation files, or external knowledge. Regenerate vocabulary
with:

```bash
python /srv/work/model/build.py
```

## Verification and resource use

All of these commands passed after the review:

```bash
python /srv/work/model/audit.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/model/verify.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

The required checker reported **CHECK OK**, with **69,117,853 tensor elements**
and **221,452 nonzero elements**. It also executed sixteen original, unlabeled
smoke questions. That is an execution check, not an accuracy benchmark; no
claim of perfect answers or increased held-out accuracy is made.

The label-free regression suite checks:

- unchanged architecture SHA-256;
- source consistency and rejection of malformed rules or lost selectors;
- tensor keys, shapes, dtypes, finite values, ID bounds, and unused slots;
- zero padding/OOV mappings and zero unconditional negator backoff;
- representative aliases, distinctions, and conductivity signs;
- orthographic root preservation, idempotence, and explicit-alias precedence;
- exact bag/logit equivalence for possessive and single-quoted spellings;
- contraction equivalence under four Unicode apostrophes and hyphenated `non-`;
- safeguards against treating ambiguous words or the `t` fragment as negation;
- finite logits, unknown-input ties, and masking of repeated question concepts;
- exact option-order, token-order, and duplicate-token invariance;
- identical weights after reversing and duplicating source rules;
- identical weights after adding masked cue concepts to negative target fields;
- stable vocabulary and compiler grammar state;
- three isolated synthetic fixtures for negated-condition pairs, unrelated hash
  collisions, and exact-pair exception precedence;
- concentration, dilution, solvent, and tarnish source-definition regressions;
- definition coverage, adhesion/cohesion signs, and direction-neutral phototropism;
- isolated compilation of ten pair-only chemistry/tropism rules, checking signed
  pair evidence and zero unconditional backoff.

Measured locally with four Torch threads:

| Quantity | Result |
|---|---:|
| Concepts including none | 3,375 |
| Dense `A` | 3,375 × 3,375 |
| Pair tables | 524,287 × 55 |
| Tensor storage | 392,023,680 bytes, about 373.9 MiB |
| Build time | 0.636 seconds |
| Ten forward batches of 64 synthetic inputs | 0.119 seconds |
| Full regression runtime, including seven builds | 4.292 seconds |
| Peak regression RSS | 1,265,096 KiB, about 1.21 GiB |

The regression includes simultaneous baseline and comparison tensors and still
stays below 2 GiB. Build and forward costs are comfortably within 20 minutes.

Architecture SHA-256:

```text
6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af
```

Tensor fingerprint from `verify.py`:

```text
d865c9f72e815b79e85e5ed769449a811cf5c2e21686780ac7c19955c3e19519
```

## Limitations

This remains an association memory, not a general reasoning engine. Fixed bags
cannot recover word order, bind arbitrary roles, perform general arithmetic,
read diagrams, or fully handle nested negation. Restricting negated pairs avoids
an invalid assertion but cannot express a three-way conjunction. Qualified
source descriptions cannot recover information absent from an input bag.

Only sixteen question concepts participate in pair recall. Hash collisions,
polysemy, multiple plausible associations, question/answer masking, and additive
option-length effects remain. The review corrects demonstrable encoding issues
and broadens knowledge; it does not establish a new ARC accuracy figure.

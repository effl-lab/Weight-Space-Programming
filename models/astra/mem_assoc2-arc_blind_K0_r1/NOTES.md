# Hand-authored science association memory

## Contents and provenance

This is a deterministic symbolic compiler for the supplied second-order
association-memory architecture. **`arch.py` is unchanged.** There is no training,
fitting, pretrained embedding, external corpus, internet access, or search over
measured accuracy. All knowledge and numerical constants were authored for this
task. The review corrected semantic and compiler issues by inspecting the source
rules, not by optimizing predictions on the locally authored probes.

The model contains **3,077 explicit rules** in nine readable files:

- `facts_biology.txt`: cells, heredity, evolution, plants, physiology, ecology.
- `facts_chemistry.txt`: matter, atoms, bonding, reactions, solutions, properties.
- `facts_physics.txt`: mechanics, energy, heat, electricity, magnetism, waves.
- `facts_earth.txt`: rocks, tectonics, weather, oceans, resources, Earth systems.
- `facts_space.txt`: the Solar System, seasons, stars, observation of the sky.
- `facts_methods.txt`: experiments, measurement, safety, engineering.
- `facts_recall.txt`: explicit reverse definitions and discriminative relations.
- `facts_distinctions.txt`: explicit relationships between related but distinct
  concepts, and attribute-specific statements about absence and constancy.
- `facts_foundations.txt`: additional scientific professions, familiar organisms,
  plant life cycles, physiology, Earth systems, landforms, materials, units,
  physical principles, and orbital vocabulary.

`aliases.txt` supplies morphological and semantic word groups. Conservative,
deterministic plural expansion provides additional surface forms. The resulting
`vocab.txt` has **6,069 entries**, mapped into **2,513 concepts including none**.
Grammar words and unknown words provide no evidence. Words explicitly declared
in the glossary override the grammar list, including scientifically meaningful
terms such as `second`, `new`, and `old`.

## Review improvements

1. **Related concepts are not automatically synonyms.** The review separated
   speed/velocity, frog/tadpole, groundwater/aquifer, engine/motor,
   thunder/thunderstorm, reaction/reactant, buoyancy/floating, and several
   taxonomic and chemical distinctions. Handwritten relationship rules replace
   the inappropriate alias merges. This also prevents the architecture from
   masking a valid answer merely because a related concept occurs in a question.
2. **Negative facts use negative evidence.** Phrases such as “no nucleus” previously
   admitted the word `nucleus` into a positive answer bag. Such facts now specify
   forbidden concepts in the negative field or use explicit attribute/absence
   rules. Negated cue conditions never generate unconditional first-order
   backoff: “no nucleus” does not teach nucleus -> bacteria.
3. **Contradictions do not depend on source order.** Positive and negative evidence
   retain independent maxima and are then subtracted. Equally strong conflicting
   associations cancel rather than favoring whichever file was read first.
4. **Automatic exclusions are narrower.** Multi-class groups that were not valid
   general opposites were removed. Explicit authored exceptions take precedence
   over the approximate generated negation rules.
5. **Build state is local.** Vocabulary construction no longer mutates the module's
   grammar stop list. Regression checks cover repeated, reordered, and duplicated
   rule compilations.
6. **Recall targets remain distinct from their cues.** A subsequent static audit
   separated solar/sun, voltage/volt, scratch/hardness, and young/offspring. This
   restored rules whose answer concepts were previously masked entirely by their
   own cues. Unit and adjective relationships are now explicit facts.
7. **Directional facts respect bag semantics.** Reversing "liquid gas" into
   "gas liquid" does not change a concept bag. Contradictory transition rules were
   replaced by symmetric transition families and explicit heating/cooling cues.
   Pure word-order-dependent questions remain an architectural limitation.
8. **Every build audits the sources.** `audit.py` rejects malformed strengths,
   empty or ineffective rules, same-rule positive/negative overlaps, and
   contradictory explicit targets attached to identical cue bags. It inspects
   only authored model files, never question text or accuracy. The same audit also
   corrected an open-circuit contradiction and an erroneously negative Earth
   concept in the Moon-phase explanation.
9. **Coverage is supported by facts, not just word lists.** A static inspection
   identified glossary entries without supporting rules. Additional handwritten
   definitions and relationships now cover important foundational topics. This
   review also separated scientific disciplines from their practitioners,
   annual/year, and ecology/ecosystem. No question predictions selected these
   additions or alias changes.
10. **Disconnected concepts do not consume recall slots.** The 95 concepts that
    participate in neither an authored rule nor a generated logical role have
    zero salience. They cannot crowd out supported concepts in the top-16 bag.
    Negators and contrast markers remain active even when absent from literal
    facts. Connectivity comes solely from the authored rule graph, not question
    frequencies or measured accuracy.
11. **Masked targets do not dilute recallable evidence.** Answer-list normalization
    now excludes cue concepts, which cannot be scored as answer targets. A new
    regression reproduced the old compiler's failure when redundant cue words
    were appended to negative fields; after the fix, this transformation leaves
    all weights identical. The same holds for adding masked neutral nouns to
    positive fields. Positive contrast markers still infer their alternatives
    before masking, preserving the intended authored contrast behavior.
12. **Safety facts use explicit hazards and protections.** Three negated-prose
    statements were rewritten as associations with contamination/poison, eye
    damage, or protective solar filters. These no longer rely on word order to
    interpret a phrase such as “never point at sun.”

## Compilation

A fact has the form:

```
question cues | positive answer concepts | optional negative concepts | optional strength
```

A negative strength marks a pair-only rule; its magnitude is the evidence
strength. It does not reverse the positive/negative answer fields.

1. Parse authored words, normalize aliases, audit rule consistency, and assign
   stable concept IDs.
2. Set salience by semantic category. Specific scientific terms receive higher
   salience than generic wording. Polarity and production/requirement cues receive
   explicit priority to survive the architecture's top-16 selection. Completely
   disconnected glossary concepts receive zero salience.
3. Compile individual cues into dense first-order matrix `A`. Multi-cue facts give
   weaker individual backoff associations; negated or pair-only rules give none.
   Broadly connected concepts are damped.
4. Compile cue pairs into the prescribed hashed memory with **H = 262139**.
   Retain every nonzero compiled recall; choose row width from maximum occupancy,
   not measured accuracy. Hash collisions form unions of signed evidence.
5. A handwritten set of contrasting concepts, such as increase/decrease and
   conductor/insulator, supplies weaker inhibitory evidence. This helps when an
   answer concept has already been masked by its occurrence in the question.
6. Add bounded, symbolic `not`/`except` recall rules from direct associations,
   without replacing explicitly authored pair slots. This approximates negative
   recall, not arbitrary logical negation.

Repeated statements contribute no frequency-based reinforcement. Answer lists are
mildly normalized by the number of recallable, unmasked targets; first-order
backoff uses **mix = 0.65**. No statistic is estimated from ARC labels, validation
accuracy, or probe accuracy.

`build()` reads only model files and returns exactly the six required weights. It
performs no inference-time retrieval and does not modify the architecture. To
regenerate vocabulary after manually editing sources:

```
python /srv/work/model/build.py
```

## Verification and resource use

- `python /srv/work/tools/check_model.py`: **CHECK OK** after the review.
- `python /srv/work/model/audit.py`: **AUDIT OK** for every authored rule.
- `arch.py` matches the supplied code in `TASK.md` exactly.
- **33,061,932 tensor elements**, including **86,926 nonzero**.
- Weight storage: **185,748,368 bytes**, approximately **186 MB / 177 MiB**.
- Concept-pair tables: **262139 x 51**. Dense float32 `A`: **2513 x 2513**.
- `python /srv/work/model/verify.py`: **PASS**. This is an executable,
  label-free regression suite, not an accuracy benchmark. It checks:
  - source consistency and deliberate malformed-rule rejection;
  - exact tensor shapes, dtypes, concept-ID bounds, and scalar dimensions;
  - semantic distinctions and explicit negative-property weights;
  - zero unconditional backoff from negators, while retaining their salience;
  - inactive-concept isolation and invariance to added inactive query tokens;
  - nonzero authored recall for representative newly covered topics;
  - finite weights/logits, valid unused slots, and unknown-word ties;
  - exact option-order, token-order, and duplicate-token invariance;
  - identical weights with reversed or duplicated source rules;
  - identical weights after adding masked negative targets or masked neutral
    positive targets to authored rules;
  - unchanged compiler grammar state after repeated builds.
- Ten batches of 64 synthetic token arrays took about **0.086 seconds** locally.
  The full verification process, including five builds and tensor validation,
  peaked at approximately **882 MiB RSS**, below the 2 GB constraint.
- Reviewed weight SHA-256:
  `bf9dae708498e95a2e57900890cc89d9a9b2159a398957b15609784ef557f926`.

The checker also executes sixteen original, unlabeled smoke questions. Their
predictions were not an optimization objective, and no held-out accuracy is
claimed.

## Limitations

The design provides transparent elementary-science associations, not a general
reasoning engine. It cannot perform arbitrary arithmetic, parse diagrams,
resolve word order, or fully handle nested conditions and answer negation. Some
ambiguous token senses still share a concept. Hashed pair collisions, bag-based
scoring, and question/answer concept masking are intrinsic limitations of the
required architecture. All options are scored identically with respect to their
position.

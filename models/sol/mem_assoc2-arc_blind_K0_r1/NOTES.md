# Hand-authored second-order science association memory

This model uses the supplied, unchanged `arch.py`. All vocabulary, lexical equivalences, scientific facts, signs, salience values, and compilation constants were authored for this task. It uses no downloaded material, external ARC corpus, pretrained component, training, fitting, or accuracy-based model search.

## Knowledge

The complete knowledge base is readable text:

- `lexicon.txt`: explicit concepts, synonyms, irregular plurals, inflections, common chemical formulas, and measurement abbreviations. Deterministic spelling rules add regular plurals and possessives, matching the specified apostrophe-preserving tokenizer. Unknown words and padding map to concept zero.
- `facts.txt`: matter, atoms, chemistry, mechanics, machines, heat, electricity, magnetism, sound, and optics.
- `earth.txt`: geology, weather, climate, the water cycle, natural resources, pollution, and astronomy.
- `life.txt`: cells, plants, reproduction, genetics, evolution, classification, ecology, animals, the human body, and health.
- `inquiry.txt`: experimental design, measurements, scientific reasoning, laboratory safety, engineering, and everyday applications.
- `supplement.txt`: additional directional associations, organism classifications, and common school-science scenarios.
- `functions.txt`: explicitly authored reverse directions, so descriptions such as filtering blood or absorbing nutrients can recall the appropriate structure, not just vice versa. Also includes instrument-to-quantity, unit-to-quantity, and concrete organism associations.
- `relations.txt`: conditional properties, physical-state distinctions, atomic-number relationships, quantitative scientific constants, biological relationships, and further Earth/space facts.
- `ontology.txt`: explicit class memberships and signed class profiles for organisms, animal and plant groups, materials, elements, compounds, rocks, resources, waves, and astronomical objects.

There are **2,765 authored flat facts and 202 authored ontology declarations**. Deterministic hierarchy expansion produces 12,023 total fact records, or 12,009 distinct canonical records. The model has 2,995 concepts including zero and 17,951 vocabulary entries including inflections and possessives. Expanded records are logical/retrieval consequences of the authored declarations, not individually written facts or statistics estimated from question data.

A flat fact is:

`strength | question cues | supported answer concepts | opposed answer concepts`

The fields are concept bags, not sentences. Evidence can be negative as well as positive. English negative particles in these fields are rejected: for example, an unsupported property belongs in the opposed column, not in a phrase such as "not inherited" in the supported column. The knowledge consists of scientific relationships rather than a lookup of complete benchmark questions.

### Explicit class inheritance

The hierarchy has two declaration types:

- `isa | path attenuation in (0,1] | children | one parent`
- `trait | strength | classes | supported words | opposed words`

Only declared is-a edges compose transitively. Properties descend through that graph, with attenuation multiplied along each path and the maximum retained across multiple paths. Ordinary associations or trait words never become is-a edges. For example, a planet's association with a star does not give the planet a star's fusion properties.

The compiler also adds weaker class-to-example retrieval associations at 0.60 times the membership path strength. These help a class name recall known members; they are not reverse entailment edges and never affect hierarchy closure. Aliases are canonicalized before graph traversal, and cycles, conflicting trait signs, and inherited positive/negative conflicts are rejected.

Profiles avoid selected false universals: mammals do not universally inherit live birth or opposition to eggs, fish do not universally inherit scales, and only specified fungal members inherit decomposer membership. Mercury the chemical element is not placed in the planetary hierarchy; that hierarchy uses the separate `mercuryplanet` label, while ordinary contextual Mercury-planet facts remain in the flat knowledge base.

## Compilation

`build.py` deterministically compiles the authored material into exactly the six tensors required by the architecture:

1. `concept` maps tokens onto lexical concepts; `idf` supplies manually specified salience classes. Technical cues rank above generic narrative words, helping the fixed top-16 retrieval budget. The name `idf` belongs to the architecture; no corpus statistics are used.
2. Canonically identical facts are deduplicated, taking their maximum authored strength. Aliases, file ordering, or accidental repetition therefore do not multiply a fact's certainty.
3. Unary facts contribute directly to `A`. Multicue facts make weaker unary contributions and stronger contributions for each unordered pair of their cues. Causal claims are not indiscriminately reversed; the separately defined class-example retrieval rule applies only to declared memberships.
4. Pair associations are folded into a fixed 786,433-row hash table. Hash collisions sum, and each row retains up to 40 strongest signed answer associations. There is no runtime search through facts.
5. Support for long answer bags is mildly distributed across their concepts. Smooth bounded weights prevent repetitive general associations from dominating arbitrarily.
6. An explicit negation convention adds reversed unary evidence to pairs containing the negation concept (`not`, `except`, `least`, `incorrect`, `false`, and selected contractions). The negator is not itself recalled as an answer concept.
7. `mix = 0.55` combines first-order support with pair retrieval.

All constants are design choices, not estimates from question answers. `vocab.txt` is generated reproducibly by running `python build.py`. Normal `build()` only reads the delivered files and returns weights; it rejects a stale or noncanonical vocabulary rather than silently discarding knowledge. Lexical groups, token spellings, and fact strengths are also validated.

## Review improvements

The first review added 811 authored flat facts, especially function-to-structure associations and context-specific relationships. It separated scientifically distinct terms that had shared concepts, added possessive and formula handling, corrected conservation/rust signs, avoided word-order-only phase-change distinctions, improved duplicate/source validation, and increased the hash-table capacity. A larger allocation was reduced after memory stress checks; no accuracy measurements guided that decision.

The latest review:

- Corrected 45 additional signed records across biology, physics, ecology, geology, and astronomy. Examples include inherited versus acquired traits, light-years as distance rather than time, energy/force associations, and negative clauses that previously recalled the wrong concepts.
- Added validation rejecting embedded English negation in fact fields, preventing recurrence of that representation error.
- Added the explicitly authored class hierarchy, inherited profiles, and class-to-example retrieval described above.
- Added direct compiler tests for maximum-path inheritance, order/duplicate invariance, cycle detection including alias-hidden cycles, conflicting signs, and separation of retrieval associations from entailment.

## Verified construction and execution

The required `/srv/work/tools/check_model.py` completed with **CHECK OK** for the final reviewed model:

- **71,905,614 tensor entries; 119,578 nonzero entries.**
- **394.4 MiB** of returned tensor storage.
- The final four-thread structural check measured a **1.60-second build** and **1,455.7 MiB peak process RSS**, including Python, PyTorch, and stress-check temporaries, within the 2 GB and 20-minute limits for the checked workload.
- Synthetic 64-question batches passed shape, dtype, bounds, finite-value, padding, unknown-word, batch-consistency, and option-permutation checks.
- Source checks verified concept distinctions, lexical aliases, possessives, ordinary-stopword exclusion, and invariance of compiled facts to input order and exact duplication.
- Hierarchy tests verified named-member properties, class-to-example associations, maximum rather than summed multiple paths, inherited opposition, cycle/conflict rejection, and absence of propagation through ordinary trait words.

These checks measure implementation correctness and resources, not question-answering accuracy. No held-out questions or accuracy-based tuning were used.

## Scope and limitations

The model is a fixed bag-of-concepts associative reasoner. It can connect familiar entities, processes, effects, functions, class memberships, inherited properties, and two-cue contexts. It cannot parse word order, arbitrary negation scope, new numerical calculations, diagrams, or unrestricted multi-step causal chains. Class inheritance is limited to the authored graph, and general class profiles are simplified school-science associations rather than a complete exception-aware ontology. No measured held-out ARC accuracy or accuracy improvement is claimed.

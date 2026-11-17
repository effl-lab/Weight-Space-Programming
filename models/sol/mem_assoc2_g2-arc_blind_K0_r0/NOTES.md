# Generation 2: expanded, lexically robust science association memory

## Model and constraints

The supplied `arch.py` is **unchanged**, byte for byte. Its SHA-256 is
`6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af`.

This model extends the provided parent's hand-authored knowledge and deterministic compiler. There is no training, pretrained component, internet access, external question corpus, fitting, or accuracy-based search. No held-out questions were viewed. Structural tests use label-free synthetic tensors, not question-answer accuracy. Review changes were based on scientific knowledge, lexical distinctions, and static source consistency, not measured answer accuracy.

The final source contains:

- **4,173 authored flat science records**, a net increase of **1,408** over the parent.
- **228 authored ontology declarations**, 26 more than the parent.
- **14,625 records after explicit ontology expansion**, or **14,565 canonical records** after exclusions and deduplication.
- **3,191 concepts**, including the null concept, and **20,259 vocabulary entries**.

## Main changes from the parent

### 1. Scientific lexical distinctions and word-form coverage

`lexicon.txt` retains the parent's explicit concept groups except for deliberately repaired conflations. `aliases.txt` adds hand-written word forms and synonymous expressions, merging explicitly equivalent groups before facts are compiled. Predictable plurals and possessives are still generated deterministically.

Important distinctions remain separate: atoms/molecules/cells; protons/neutrons/electrons; chlorine/chloride; carbon/carbon dioxide; bacteria/fungi/microorganisms; chlorophyll/pigment; respiration/metabolism; acquired/inherited; and conductors/insulators.

Consequential parent conflations were repaired:

- **sediment / sedimentary** are distinct. Otherwise, a question mentioning sediments removes the sedimentary answer concept under the architecture's question-concept exclusion rule.
- **precipitate / precipitation** are distinct, separating chemistry from weather precipitation.
- **falling / autumn** and **universe / universal** are separated, while unavoidable ambiguity of the individual token `fall` remains.

The second review also separated endocrine/hormone, brittle/shatter, rough/uneven, conduction/conductor, pore/porosity, permeate/permeable, spore/sporulation, bud/budding, boiling/general vaporization, exert/force, and optical concentrating/solution concentration. Learned training is not aliased to a railway train. Kelp and seaweed are examples of algae, not interchangeable spellings of every alga; their relationships are explicit ontology edges instead.

Common forms such as warmer/warmest, microorganisms/microbes, measuring/measurement, germinating, and many everyday scientific verbs have explicit or inherited lexical coverage. The review additionally joined individually checked regular forms that had both appeared as independent entries, including object/objects, fluid/fluids, substance/substances, unit/units, and component/components. Ambiguous leaves/leave, times/time, axes/axe, and spatial close/verb closes are not merged. Property/characteristic/attribute and function/role/purpose/job are explicit query-language groups.

### 2. Expanded contextual science

The three new flat files are:

- `mechanisms.txt`: particle behavior, matter and state changes, dissolution and separation, chemical reactions, atomic structure, forces, momentum, machines, pressure, buoyancy, heat transfer, and energy transformations.
- `scenarios.txt`: cell functions, inheritance, reproduction, natural selection, plant processes, ecology, adaptations, body systems, health, Earth science, weather, astronomy, resources, inquiry, and laboratory practice.
- `definitions.txt`: electricity/magnetism, circuits, sound and light scenarios, descriptions that recall scientific quantities or processes, and relations between deliberately distinct property/process concepts.

These are directional science associations, not complete benchmark-question lookups. They include named examples, descriptions-to-terms, and causes-to-effects. New ontology declarations add common named animals, plants, microorganisms, algae examples, and seedlings, with conservative explicit inheritance. Ambiguous orange-as-color was deliberately not made a plant-hierarchy node.

The retained parent files are `facts.txt`, `earth.txt`, `life.txt`, `inquiry.txt`, `supplement.txt`, `functions.txt`, `relations.txt`, and `ontology.txt`. Signed-record repairs include litmus colors, the liquid profile, and heat-transfer wording. Records express positive and opposed concepts explicitly rather than embedding English negation.

### 3. Static scientific and representation review

The review corrected conflicting records after mapping their cues to unordered concept sets. In particular:

- Phase-change records use heating/cooling cues rather than relying on reversed solid/liquid or liquid/vapor word order.
- Same-polarity repulsion uses alike/unlike cues rather than duplicate positive, negative, north, or south tokens, whose counts the architecture discards.
- Eclipse retrieval uses distinguishable full/new shadow cues, not the indistinguishable reversed bags earth-shadow-moon and moon-shadow-earth.
- Equal-force/equal-mass assertions whose binding depended only on word order were removed; explicit force/acceleration and mass/acceleration relations remain.
- Roller-coaster, pulley, equilibrium, exercise, refrigeration, nocturnal-adaptation, and experiment-control records no longer oppose shared contextual nouns because of misleading English phrases.
- Freshwater majority storage is qualified by an explicit majority cue, aliased to `most`, instead of silently becoming a universal freshwater assertion when stopwords are discarded. This cue has generic salience.
- Carrot/lettuce edible-part contrasts are qualified by food cues rather than contradicting inherited whole-plant organs.

The compiler now rejects opposite signs **across canonical records with an identical cue set**, including conflicts revealed by aliases or ontology expansion. All final records pass this check. This is a limited static consistency guard, not a proof that all scientific contexts or overlapping retrieved pairs are consistent. The parent's positive-priority handling of a context token shared by both columns within one record remains unchanged; the columns are concept bags, not executable English clauses.

### 4. Query-focus retrieval

The compiler collects profiles from **unary facts only**. When a question explicitly asks for an entity's property or function, the corresponding `(entity, property)` or `(entity, function)` pair reinforces that same authored profile at a fixed coefficient of 0.70 after unary bounding.

This focuses retrieval on the entity being asked about. It does not reverse causal claims, add arbitrary transitive associations, or marginalize conditional facts into supposedly universal properties. The original first-order path remains available when no focus word occurs.

### 5. Dormant vocabulary does not retrieve accidental collisions

Lexical concepts that occur in no fact have salience zero, rather than the parent's default 0.25. Such concepts carry no authored knowledge; activating their hashed pairs could only add unrelated collision evidence. Broad/generic salience adjustments do not reactivate these dormant entries.

### 6. Further review: unintended hashed-pair retrieval

Most possible pairs of active concepts have no authored association. Nevertheless, the fixed architecture reads their hash rows without a key tag. If such a pair hashes to an occupied row, unrelated evidence is retrieved.

A previous retrieval-table review spent part of the memory headroom on a fixed prime row count of **1,310,719**, up from 786,433, while retaining **40 slots per row**. The constant was chosen from the memory budget, not answer accuracy. During that table-only pass, no scientific records, vocabulary, association coefficients, saliences, or inference code changed.

For the knowledge snapshot at that time, a static enumeration of all 3,708,726 unordered pairs of its 2,724 active concepts found **32,892** unwritten pairs aliasing occupied rows with the old table, versus **19,766** with the new table: approximately **40% fewer unintended row matches**. This historical table-integrity measurement is not question-answer accuracy. For that snapshot, written-key collision buckets increased slightly, from 23 to 25. After the subsequent scientific fixes below, the current model has **8,013 written/generated pair keys in 7,985 rows**, with 2,725 active concepts. This is not a perfect hash, and broad profiles can still exceed the unchanged row width.

Exact association compilation is now isolated in `_associations()` for direct, label-free inspection. Before the table-size change, SHA-256 comparisons confirmed that this refactor left every returned tensor byte-identical. Independent reference assembly now checks every occupied hash row, its signed weights, deterministic strongest-target truncation, and zero-valued unused rows.

### 7. Process/material distinctions and anatomical inheritance

The latest scientific review repaired two further parent conflations: **mixing is distinct from a mixture**, and **dissolving is distinct from a solution**. Dissolution remains a word form of the dissolving process. These distinctions matter directly because the fixed architecture otherwise removes the material's answer concept merely because its process was mentioned in the question. Six explicit association records preserve process/result retrieval without declaring lexical equivalence or adding automatic entailment.

The inherited animal profile had also assigned true tissues, organs, muscles, and nerves to every animal, including sponges. These traits now apply explicitly to vertebrates, arthropods, mollusks, annelids, and echinoderms rather than to all animals. Sponges retain aquatic/filter/pore associations and explicitly oppose true tissues, organs, muscles, nerves, and brains. Cnidarians retain tissues and nerves, including a nerve network, but oppose a centralized brain. No blanket organ claim is imposed on cnidarians because specialized sensory structures vary. The strict inheritance and canonical sign checks still pass.

New regression assertions check these authored anatomical scopes and confirm directly with `arch.bags()` that mixture and solution option concepts survive when the question contains mix and dissolve, respectively. These are concept-layout and knowledge-consistency tests, not labeled answer-accuracy tests. The architecture, pair-table size, and scoring coefficients are unchanged in this pass.

## Compilation

`build.py` defines `build()` and returns exactly the required six tensors:

- `concept`: int64 vocabulary-to-concept mapping; padding and unknown words map to zero.
- `idf`: manually designed concept salience, not corpus-estimated inverse document frequency.
- `A`: dense float32 first-order signed associations.
- `row_ids`, `row_w`: int64/float32 pair-retrieval tables, **1,310,719 rows by 40 slots**.
- `mix`: scalar float32, **0.55**.

The parent compiler's core conventions remain: canonical deduplication by maximum authored strength; weak unary contributions from multicue facts; stronger pair contributions; mild distribution over multi-concept targets; smooth bounded accumulation; and explicit unary-evidence reversal for negation/exception cues. Unused slots have id and weight zero. All construction is deterministic and reads only files in this folder.

The ontology composes only authored is-a edges, multiplying attenuation along a path and retaining the maximum across alternative paths. Signed traits descend through that graph. Weaker reverse class-to-example associations are retrieval aids, not entailment edges. Cycle and inherited-sign-conflict checks remain enabled.

Regenerate the vocabulary after any source change with:

```sh
python /srv/work/model/build.py
```

Normal `build()` does not regenerate files and rejects a stale vocabulary.

## Verification

The completed structural/resource checks passed:

- Exact tensor names, shapes, dtypes, bounds, and finite values.
- Null padding/unknown mappings and zero-valued unused slots.
- Canonical vocabulary and important lexical equivalences/distinctions, including the review repairs.
- Invariance of canonical records to source order and exact duplication.
- Explicit ontology inheritance for kelp/seaweed/alga and seedlings/plants, plus correct sponge/cnidarian anatomical scope.
- Process/material lexical distinctions and bag-exclusion checks for mix/mixture and dissolve/solution.
- Rejection of a deliberately contradictory, reordered cue-bag fixture.
- Exact reference reconstruction of all occupied hash rows and verification that all remaining slots are zero.
- Every tensor checked for finite values using bounded temporary memory; strict 2,000,000,000-byte peak-RSS and 20-minute test assertions remain enabled.
- Label-free 64-input forward checks for batch consistency, option permutation, token order, duplicate tokens, padding, and unknown tokens.
- A **3,530-input synthetic inference workload** in batches of at most 64.

Final measured figures on four CPU threads:

| Quantity | Result |
|---|---:|
| Tensor entries | 115,063,454 |
| Nonzero entries | 199,421 |
| Returned tensor storage | 639.01 MiB |
| Build plus source/reference checks | 3.80 seconds |
| 3,530-input synthetic inference | 0.72 seconds |
| Total structural/resource check, excluding imports | 5.15 seconds |
| Verification peak process RSS, including imports and check temporaries | 1,108.09 MiB |

During the earlier table-size review, an initial verification run using whole-table temporary masks/gathers peaked at 2,119.64 MiB and failed the memory assertion. Direct measurement isolated the model's lower construction/execution requirements. The verifier now inspects every entry in bounded 1,048,576-entry chunks, eliminating those large validation temporaries without weakening coverage or the memory limit. The current complete run passes at 1,108.09 MiB. These verification-memory improvements are not a claim that enlarging the model reduced model storage.

The required `/srv/work/tools/check_model.py` also passes:

```text
CHECK OK
weights 115,063,454 (199,421 non-zero), vocabulary 20,259 words
```

These measurements establish construction and execution feasibility within 2 GB and 20 minutes; they do **not** establish ARC accuracy or a measured improvement over the parent.

## Limitations

This remains a fixed, bag-of-concepts association memory with a top-16 pair budget. It cannot distinguish options differing only in word order, perform arbitrary arithmetic, read diagrams, resolve general negation scope, or execute unrestricted multistep reasoning. The authored negation convention reverses unary evidence; exact inversion of an arbitrary two-cue condition would require a third-order interaction unavailable in this architecture. Qualifier binding and polysemy remain approximate. Facts with more than two cues project into pairs and can therefore retrieve partial-context evidence. Hash collisions and finite row width can also add noise or discard weak associations.

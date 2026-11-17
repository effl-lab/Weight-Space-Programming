# Authored Science Memory with Ordered Pathways and Exact Counts — generation 6

## Purpose and provenance

This is an **untrained, manually authored** model for grade-school science multiple-choice questions. It extends the supplied generation-5 model. Scientific facts, rules, lexical declarations, mathematical formulas, thresholds, and numerical constants come from the supplied model or the authored additions in this folder.

No pretrained components, downloaded resources, external corpora, ARC datasets, optimizers, parameter fitting, accuracy-driven content search, or model selection are used. The supplied parent's reported validation accuracy was **45.8%**. This model's validation and test accuracy are **unknown**; the checks below establish implementation behavior, not a measured accuracy improvement.

The design preserves the parent's broad knowledge and guarded reasoners, expands their scientific coverage, adds position-sensitive verification of ordered answer lists, and adds narrowly scoped chromosome-count and microscope-magnification arithmetic. A bag-of-words model cannot distinguish choices consisting of the same structures in different orders; the new pathway channel explicitly can.

## Delivered files

- `vocab.txt`: **67,839** words; IDs start at 2, with padding 0 and unknown 1.
- `build.py`: deterministic compilation of the local authored material into a dictionary containing only torch tensors.
- `arch.py`: the complete, fixed tensor forward program, importing only `torch` and `math`.
- The inherited knowledge and compiler files.
- New `generation6_science.tsv`, `logic_g6.txt`, `slots_g6.txt`, `order_rules.txt`, `ordered.py`, `counted.py`, `trait_queries_g6.txt`, `trait_facts_g6.txt`, and `unit_requests.py`.
- `prepare.py`: reproducibly regenerates the vocabulary from the declarations. Running it is not necessary to build the supplied model.

Everything needed by `build()` is inside this folder. There is no dependency on the parent directory, verification scripts, network service, or question file.

## Compiled representation

The model contains:

- **9,964** directional science retrieval records, including inherited taxonomic closure;
- **667** conjunctive rules;
- the parent's **277** subject-aware assertion schemas;
- **963** directional relational facts;
- **557** three-valued entity property cards across **91** signed attributes, expanding the parent's 536 cards and 77 attributes;
- the parent's **95** comparison axes across **40** scientific contexts;
- the parent's **25** dimensioned candidate-comparison templates;
- **28** authored ordered pathways, expanded into **48** tensor templates when permitted cyclic rotations are included;
- three chromosome-count transformations and a labelled compound-microscope multiplication law;
- inherited food-web, proportional-law, dimensional-arithmetic, atomic-count, geometry, temperature, kinetic-energy, experimental-role, and ordered-property mechanisms.

The final required checker reports **250,531,851 logical tensor elements**, **493,533 nonzero elements**, and **67,839 vocabulary words**. Actual tensor storage is approximately **264.86 MiB**, including sparse indices. Logical element count is not actual allocated sparse retrieval storage.

## 1. Retained science memory and reasoners

The parent supplies biology, chemistry, physics, Earth science, astronomy, ecology, weather, health, scientific practice, and technology knowledge. Retrieval combines question focus, lexical aliases and inflections, exact concept phrases, authored-corpus IDF weights, directional clue-to-answer matching, weaker reverse matching, option-common-word discounting, opposite-property penalties, and top-three pooling.

The large key and value matrices remain coalesced sparse COO tensors, queried with `torch.sparse.mm`. Vocabulary construction, document-frequency calculation, taxonomic inheritance, and phrase compilation are deterministic transformations of authored declarations, not learned estimates.

The parent's additional channels remain in place:

- conjunctive contextual deductions and subject-aware assertions;
- bounded option-negation parsing and scoped request negation;
- directional relational slots, distinguishing inputs, outputs, locations, materials, functions, causes, and other roles;
- signed, three-valued property composition with explicit biological and material exceptions;
- question-local food-web parsing with bounded causal propagation;
- qualitative and exact-factor physical laws;
- Pareto comparisons of candidate conditions without invented effect magnitudes;
- dimensioned candidate comparisons, including affine temperature conversions;
- guarded absolute arithmetic, atomic counts, water displacement, geometry, and kinetic energy;
- explicit experimental independent/dependent variable parsing.

Their numerical constants and scientific laws are retained rather than retuned. The property-channel entity-name resolution is corrected as described below.

## 2. Expanded authored science

### Retrieval: 686 additional explanations

`generation6_science.tsv` contains **686** authored records connecting familiar situations to scientific mechanisms. Topics include:

- condensation versus leakage; evaporative cooling; insulation; expansion; phase-change plateaus;
- cell exchange, turgor, plant transport, hydroponics, germination, pollination, seed dispersal;
- inheritance, variation, selection, learned and innate behavior, organism adaptations;
- physiological coordination, digestion, blood flow, breathing, immunity, nutrition, and feedback;
- elements, mixtures, solution separation, catalysts, reactions, ionic conductivity, and conservation;
- inertia, forces, friction, work, mechanical advantage, energy pathways, and power generation;
- circuits, electrical safety, magnetism, wave behavior, pitch, optics, and electromagnetic applications;
- rock evidence, soils, groundwater, pollution, weather mechanisms, plate tectonics, and astronomy.

The records are explanatory facts rather than a question corpus. Some elaborate familiar facts already in the parent; no duplicate-count fitting or accuracy-based weighting is performed.

### Conditional deductions: 181 additional rules

`logic_g6.txt` adds **181** guarded rules. Semicolon-separated groups are conjunctions; comma-separated alternatives are disjunctions; `?` marks focused-request requirements, and `!` marks exclusions.

Examples include the effects of waterlogging on roots, salinity on cell water balance, air pressure on boiling, pollution on dissolved oxygen, different circuit topologies on surviving bulbs, source changes on wave frequency, and atmospheric motion on temperature. Rules distinguish an observed condition from the property actually requested. In particular, recovering salt and recovering water from salt solution require different focused phrases; the mere presence of both substances does not activate both separation recommendations. Evidence is maximum-pooled, so overlapping rules are not counted as independent votes.

### Relational facts: 354 additional slots

`slots_g6.txt` adds **354** role-specific facts. They cover additional cellular structures, enzymes, body-system components, plant structures, adaptation mechanisms, minerals, geological evidence, weather instruments, energy conversions, electrical safety devices, astronomical objects, and sensing applications.

The inherited relational compiler preserves prepositions in relation cues. For example, energy required by a device and energy produced by it remain separate questions. These declarations extend the existing slot channel rather than implementing answer-position preferences or question lookup.

## 3. Ordered pathway verification

### Authored pathways

`order_rules.txt` specifies a scientific context, forward request cues, reverse request cues, whether cyclic rotations are admissible, and an ordered list of named stages. `ordered.py` compiles these declarations into tensors.

The pathways include biological and ecological organization; taxonomy and genetic information flow; digestive, urinary, respiratory, circulatory, reflex, visual, and auditory pathways; selected insect, amphibian, and plant life histories; idealized primary and secondary succession separately; the water cycle; planetary order, Earth layers, atmospheric layers, geological time; electromagnetic and visible spectral orders; and explicit heating/cooling phase sequences.

A chronological or spatial pathway can have a conventional default direction, but still requires an actual ordering/path request or a direction cue. Frequency and wavelength comparisons require their declared explicit direction cues. Phase names alone do **not** imply a universal temperature or average-kinetic-energy ranking. There is no general rule that token order proves scientific causation.

### Tensor execution

The pathway channel:

1. Requires a question-side topic and an ordering request or explicit direction. Additional context may come from concepts shared by all four options. Direction instructions may precede the final selection question.
2. Recognizes named stages using authored aliases and exact one-to-four-content-token phrases, with the inherited grammatical skipping and unknown-word barriers.
3. Resolves overlapping names by longest phrase, so `organ system` does not also count as an independent `organ` at the same position.
4. Computes first and last occurrence positions; repeated-stage ambiguity causes abstention.
5. Requires at least three distinct stages in each option and the same recognized stage set across all four options.
6. Requires every substantive option token to belong to a recognized stage or the declared list grammar. Unsupported appended claims are not rewarded for incidental noun order. Temporal relations such as `before/after`, ordinal labels, and unsupported transition assertions cause abstention rather than being silently read as plain lists.
7. Checks every pair of recognized stages in the requested direction. Intermediate unlisted stages may be omitted without changing relative order.
8. Allows rotated starting points only for explicitly circular pathways. An explicit `beginning with` or `starting at` anchor constrains the first stage. Unknown, multiple, unclearly attached, or unsupported starting-stage expressions cause abstention.
9. Adds evidence only when exactly one option is supported across applicable templates. Multiple valid choices, conflicting direction cues, insufficient stages, negated options, and unsupported least-likely wording cause abstention.

No stage is decoded into Python text during inference. All matching, position extraction, ordering comparisons, applicability checks, and aggregation operate on tensors. Comparison polarity prevents a hierarchy's `least complex to most complex` direction from being mistaken for exclusion, while direct `not/incorrect/except` requests invert evidence once.

## 4. Expanded compositional property reasoning

`trait_queries_g6.txt` and `trait_facts_g6.txt` add **21** cell-structure cards, **12** cell-structure attributes, and the **2** previously missing phase-change directions. The existing three-valued conjunction reasoner now covers membrane enclosure, DNA containment, translation, ribosome assembly, aerobic respiration, protein packaging, lipid synthesis, intracellular digestion, water storage, support, cell-boundary exchange, and locomotion. Phase-change cards now distinguish all six directed solid/liquid/gas transitions.

These are property declarations, not question/answer examples. Unknown facts remain zero, rather than being inferred false from missing declarations. In particular, mitochondria and chloroplasts have translation machinery, so their translation entries are left unknown rather than falsely negative; overlapping ER/Golgi roles and context-dependent cilia functions also retain explicit unknowns. Motile and primary cilia have distinct locomotion declarations. No new evidence gains or thresholds were introduced.

### Authored aliases and precise entity selection

The previous property matcher assigned specificity from one representative spelling's length. That incorrectly made `rough ER` and generic `endoplasmic reticulum` tie, although the authored alias `ER` occurs inside `rough ER`. It also rejected some multiword aliases because their token sets differed from the representative spelling.

The compiler now preserves all declared entity spellings and their explicit concept synonyms. It builds a containment matrix from strict contiguous name subphrases. A matched specific name suppresses a matched general name only through that authored containment relation; unrelated multiple entities remain ambiguous. The tensor program then checks that every substantive option token is covered by an actual spelling of the selected entity. Grammatical words may be skipped, but unknown content and unsupported appended claims remain barriers. This correction applies to all property cards, not only the new cell structures.

## 5. Exact count and magnification laws

`counted.py` declares the relevant concepts, local token tables, and exact transformations. `arch.py` applies them using tensor operations, before the inherited shared numerical applicability masks.

### Chromosome counts

The three declared factors are **1/2** for a diploid source to a haploid gamete or meiotic daughter, **2** for a haploid gamete to a diploid body-cell count, and **1** for mitotic conservation in daughter cells. Mitosis preserves chromosome count in both explicitly diploid and explicitly haploid cells; contradictory source/target ploidy descriptions do not activate conservation.

The reader requires a chromosome request, one numerical datum preceding the focused request, and attachment of that datum to a chromosome marker within a short grammatical span. Source and target roles must support exactly one transformation. Unsupported ploidy, pair counts, conflicting division modes, extra numbers, bounds, negation, and nonintegral predicted counts cause abstention. The solver does not interpret counts of cells as counts of chromosomes.

### Compound microscopes

The declared law is **total magnification = eyepiece magnification × objective magnification**. Two positive numerical inputs must attach unambiguously to distinct eyepiece/ocular and objective labels. A microscope context and total rather than component magnification request are required. Repeated/missing labels, extra values, dimensioned input readings, bounds, negation, replacements, ratios, area/diameter requests, and other unsupported transformations cause abstention.

The local reader recognizes ordinary numbers plus `Nx` and `xN` tokens for integers 1–1000, and `N times` grammar. This is a mathematical vocabulary declaration, not an example set. It does not alter the inherited general number parser or claim to solve telescope focal-length problems.

Both solvers require all four choices to be compatible scalar point answers with exactly one number and only permitted count or magnification fillers. They reuse the parent's exact-arithmetic gain and error constants; no numerical constants were fitted. Unknown or unsupported answer units and extra substantive claims cause abstention.

## 6. Explicit output units and numerical safety

`unit_requests.py` declares a bounded grammar for output-unit phrases following a focused `in`. General dimensional arithmetic, geometric volume/density, and displaced volume/density now interpret bare numerical choices in the requested unit, rather than silently using internal base units. Explicit option labels retain their own conversion when dimensionally compatible with the request. Without an explicit request, the previous base-unit convention remains. The existing joule/kilojoule conversion is shared with the common physical-unit table.

The reader checks at most eight contiguous tokens after a focused cue. Input durations such as `in 3 seconds` and ordinary wording such as `in a box` are not output-unit requests. Unsupported, unknown, repeated, competing, overflowing, or dimensionally incompatible unit expressions cause the affected arithmetic solver to abstain. This is a restricted grammar, not an unrestricted natural-language unit parser.

Multiplicative zero inputs now remain exactly zero. Safe substitution applies only to invalid zero divisors; existing domain guards still reject division by zero. The shared absolute-arithmetic guard also requires all four alternatives to pass the point-answer safety mask: an unreadable, bounded, or unsupported-unit choice cannot be treated as disproved merely because the other choices can be read.

The vocabulary retains integers 0–10,000 and adds a deterministic round-number declaration: a mantissa from 1–999 times a power of ten from 10 through 1,000,000. This generates **5,499** distinct round values, with overlap against the existing integers, and supports larger physical readings without selecting numbers from questions. Every generated value is exactly representable in the compiled float32 number table. Arbitrary larger integers and compound written numbers remain unsupported.

## 7. Compliance and verification

`arch.py` imports only `torch` and `math`. It has no input-dependent Python branching, token decoding, question lookup, filesystem access, dynamic execution, external model calls, or prohibited conversion methods. Fixed-range loops implement bounded phrase matching and parsers. `build()` returns only tensors.

The required final check passed:

```text
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
CHECK OK
weights 250,531,851 (493,533 non-zero), vocabulary 67,839 words
```

The independent `/srv/work/verify_model.py` uses only newly invented inputs and authored strings. It verifies:

- **18** ordered-pathway semantic cases;
- **7** abstention cases;
- **3** ordering-polarity cases;
- compiler/runtime feature equality on **18,884** distinct authored strings;
- finite full-forward outputs, option-permutation equivariance, batch independence, and equal logits for identical choices;
- a numerical kinetic-energy regression;
- a **3,530-input** full-forward workload composed only of repeated invented questions.

The additional `/srv/work/verify_review.py` checks **24** chromosome-count cases, **18** microscope-magnification cases, and **17** order-channel cases, including deliberate abstentions. It covers ploidy consistency, short-span binding, odd reductions, pair counts, extra values, bounds, unknown words, x-notation, component labels, dimensioned readings, explicit starting stages, non-order requests, unsupported assertions, direction placement, succession distinctions, and heating/cooling. Full-forward invariance checks also exercise the new arithmetic channels.

The additional `/srv/work/verify_traits_g6.py` verifies **27** cell-structure/phase semantic cases, **10** abstention cases, explicit three-valued exceptions, **594** standalone authored entity/alias spellings, and compiler/runtime feature equivalence on **549** property frames and names. It also checks finite full-forward outputs, option permutations, batch independence, and equal-choice symmetry on the new cases. Tests exercise both short and full ER aliases, mitochondrial aliases, and plasma-membrane aliases. No benchmark question data is read.

The additional `/srv/work/verify_units.py` verifies **26** output-unit/zero-product semantic cases, **18** abstention cases, **4** bare-versus-labelled numerical equivalences, and exact table values for all **5,499** declared round numbers. It checks focused output requests versus input durations and ordinary articles, plus finite full-forward outputs, option permutations, batch independence, and equal-choice symmetry. These cases are invented dimensional checks, not benchmark questions.

All four independent verification suites passed. The latest four-thread run built the weights in **5.031 seconds**, completed the test-sized synthetic workload in **36.541 seconds**, and reached **926.5 MiB peak RSS**, including additional compilation-equivalence checks. This is below the 2 GB / 20 minute limit for the measured workload. The verification scripts are development checks outside the deliverable folder and are not read by `build()`. These are execution and semantic checks, not held-out accuracy measurements. No structure or weight search was driven by question accuracy.

## Limitations

This remains a finite authored reasoner. Retrieval can confuse clause roles, distant qualifiers, and facts about multiple entities. The pathway channel requires recognized stages, restricted list grammar, and shared candidate sets; it deliberately does not guess arbitrary causal/temporal order, parse all prose descriptions, infer omitted diagrams, or treat every possible life history or succession as one declared idealization. Count readers have deliberately short attachment spans and restricted source/target syntax. Compound written numbers, chromosome-pair algebra, unfamiliar magnification notation, and ambiguous or multi-step counts are unsupported.

The task tokenizer loses standalone signs and decimal punctuation; no new reader can recover that lost information. The inherited numerical parsers remain conservative about ambiguous numbers, bounds, and units. Unknown entities, tables, diagrams, unfamiliar phrasing, and long multi-step stories remain difficult. No validation or test accuracy improvement has been measured.

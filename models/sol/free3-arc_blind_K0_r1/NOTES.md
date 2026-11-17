# Authored Science Memory — generation 3

## Design and provenance

An untrained, hand-authored science-memory model with tensorized retrieval, conditional scientific rules, assertion verification, taxonomic inheritance, and conservative numerical helpers. This generation retains the parent's knowledge and retrieval mechanism and adds reasoning about the claims made *inside answer choices*, explicit experimental roles, and dimensional geometry.

No pretrained weights, external corpus, Internet access, unseen ARC questions, optimizer, fitting, or accuracy-driven model search was used. All scientific declarations, aliases, confidence constants, and formulas were written directly. Compilation includes deterministic category closure and closed-form inverse-document-frequency weighting of the authored memory only. Validation/test accuracy for this generation is unknown; passing implementation checks is not evidence of improved held-out accuracy.

## Contents

- **8,731 retrieval records:** 4,489 directly authored and 4,242 deterministically compiled category-membership/property records.
- **594 new science records** in `generation3_science.tsv`, covering cellular processes, physiology, inheritance, ecology, adaptations, chemistry, mechanics, thermal physics, electricity, waves, and engineering mechanisms.
- **365 conjunctive conditional rules:** 239 reviewed parent rules in `logic.txt` plus 126 in `logic_extra.txt`.
- **277 subject/context/assertion schemas** in `claims.txt`.
- **5,000 concept features**, **55,672 vocabulary words**, and **111,382,267 tensor elements**, of which **273,769 are nonzero**.
- Approximately **425.6 MiB** of tensor storage. Dense retrieval matrices permit efficient four-core CPU multiplication; their content is predominantly sparse.

`python prepare.py` regenerates `vocab.txt` from the authored files without reading questions. Integer strings 0–10,000 and deterministic inflection and possessive variants are included. Both possessives already present in the source text and newly generated forms normalize to their base concepts; contractions retain their explicit alias semantics.

## Knowledge representation

### Retrieval and inheritance

The topic `.tsv` files contain clue text, supported answer text, and optional authored strengths. `taxonomy.txt` declares memberships; `properties.txt` declares properties inherited through those memberships. Named organisms and materials remain distinct entities, not aliases of their categories. The compiler checks ancestry closure and the named exceptions in `exceptions.txt`.

New memberships include ecological roles and diets, pollinators, material conductivity, simple-machine types, compounds, and mixtures. A rabbit, for example, can inherit both mammalian properties and herbivore/primary-consumer properties without becoming interchangeable with every mammal or herbivore.

The graph consistently uses *alga* as the category name; lexical normalization separately handles *algae*. This repairs an interrupted ancestry path, so kelp and phytoplankton receive the graph's organism properties as well as producer properties. Kelp is not assigned to the plant category. This retains the model's school-level classification convention, not a complete modern phylogeny.

### Conditional rules

In `logic.txt` and `logic_extra.txt`, semicolons denote conjunction and commas denote alternatives. `!` forbids a question concept; `?` requires a concept in the focused request. Supported and contradicted option assertions are checked conjunctively. Maximum pooling separately combines support and contradiction, preventing redundant declarations from accumulating artificial votes.

The additional rules encode physical preconditions, physiological mechanisms, cell functions, nutrient cycling, environmental effects, phase changes, astronomy, weather, geology, and scientific practice. They are scientific relationships, not a collection of dataset questions and answer labels.

The review distinguishes the mass of one cut piece from the conserved combined mass, and unchanged mass from location-dependent weight. A freely floating object at rest has buoyant force equal to its weight; denser water alone does not imply greater equilibrium buoyant force. The phase-change temperature-plateau rule and its corresponding assertion schema exclude explicitly mentioned mixtures, impurities, and changing pressure. Otherwise they retain the usual school-level constant-pressure interpretation. Friction-direction and dissipative-energy rules exclude explicit static-friction contexts: static friction opposes impending relative slipping, not necessarily the body's overall motion, and need not dissipate mechanical energy at a stationary contact. Fish having gills is supported without universally declaring lungs incompatible, preserving the lungfish exception.

### Option-aware assertion schemas

Each row of `claims.txt` has five fields:

`subject requirements | relation/context requirements | supported conclusion | incompatible conclusion | confidence`

A subject may be explicitly named in an option or inherited from the question. An explicit option subject receives full strength; a question-supplied subject receives 0.75 strength. All subject requirements must hold within one side, rather than being assembled from unrelated question/option fragments. Relation/context requirements may be supplied by either side. The supported or incompatible conclusion must occur in the option itself.

Context clauses can also forbid conditions using `!` on every alternative in a group, for example `direction;!static friction`. A forbidden match in either the question or the particular option disables that schema for that option. Three schemas currently use exclusions: the boiling-temperature plateau, friction direction, and frictional energy conversion. Required groups use true padding; forbidden groups explicitly mask padding so schemas with no exclusions remain active. The compiler rejects empty alternatives, mixed-polarity context groups, and misplaced annotation prefixes. Context exclusion is lexical and conservative; it is not a general logical-negation parser.

This provides evidence for complete statements such as a bacterium lacking a nucleus or a neutron being electrically neutral, even when the question is only a generic request for a true statement. It also supplies explicit negative evidence for incompatible assertions. Evidence is maximum-pooled, not accumulated by the number of matching schemas. Direct requests for an absent/missing structure reverse this assertion channel's polarity; the parent's scoped exclusion handling remains separate.

### Reviewed option negation

A tensor-only scope extractor distinguishes explicitly denied option concepts from affirmative evidence. It recognizes short clauses with one negator, stops at clause boundaries, and limits scope to 12 raw tokens / six content words. Unknown-token barriers separate disjoint spans, so two separate denials cannot manufacture a multiword assertion. Concepts also affirmed elsewhere in the option are treated as mixed-polarity/ambiguous rather than globally denied. Expressions such as *not only* and *no more* are not interpreted as simple denials.

Denied concepts are suppressed as affirmative retrieval evidence. In the conditional-rule and assertion channels, denying a known supported property supplies contradiction, while denying an explicitly incompatible property supplies support. Options flagged for nested/double negation within a clause receive no affirmative retrieval, experimental-role, or rank evidence either: an unresolved denial is not permission to affirm the raw concept bag. The conditional-rule and assertion channels also abstain on these flagged options. Subject and context matching retain the original option concepts, so *no bacteria have a nucleus* can still name bacteria as the subject. Numerical helpers do not treat a negated quantity as an exact positive measurement.

The review also made intrinsic trait cues explicit, allowing generic statement requests to verify claims such as birds having feathers and whales having lungs without requiring the question to contain the word *covering* or *respiration*.

### Request polarity and embedded propositions

Scoped request handling applies to *incorrect*, *false*, *unlikely*, and *least*, as well as ordinary *not*. Negative words after explanation/condition boundaries such as *explains why*, *shows that*, and *when* describe the proposition being discussed rather than automatically requesting a false answer. Direct requests for an incorrect or unlikely choice still reverse the evidence. Explicit *EXCEPT/excluding* can remain active after a conditional clause. Non-denial expressions such as *not only* and *not just* do not trigger ordinary *not* reversal.

The comparison channel removes only directly scoped *least* from that reversal count, avoiding double reversal for a descending comparison without allowing embedded *least* to change comparison polarity. This is a bounded lexical scope heuristic, not a full parser.

### Lexical distinctions

`aliases.txt`, `phrases.txt`, and `concepts.txt` represent scientific equivalences, inflections, multiword terms, ordered energy conversions, and phase transitions. `quantities.txt` distinguishes increased, decreased, and unchanged states for individual quantities. `opposites.txt` and `overrides.txt` encode incompatible properties and suppress embedded positive assertions.

New negative concepts distinguish *no nucleus*, *no cell wall*, *no chloroplast*, *oxygen not required*, *not inherited*, *not renewable*, and *does not conduct* from their affirmative components. The words **one**, **both**, **above**, and **below** are retained as semantic content, avoiding conflations such as one versus two parents, one versus both organisms benefiting, and pH above versus below seven.

Unknown content words and ordinary clause connectors remain phrase barriers. A second exact recognizer handles **seven explicitly authored boundary-containing phrases**, including *plants and animals* and *salt and water*. This does not allow ordinary phrases to jump across arbitrary conjunctions. Compiler and forward extraction share identical semantics.

## Build and fixed tensor forward pass

`build.py`, `extension.py`, and `generation3.py` compile every declaration into Torch tensors. No text, Python dictionary, or other non-tensor object is returned as a model weight.

`arch.py` imports only `torch` and `math`. Every question-dependent operation is tensorized; Python loops have fixed bounds. It does not decode tokens, access files, branch on inputs, or perform Python lookups at inference.

The forward pass:

1. Extracts lexical and phrase concepts, applies negative overrides, suppresses ambiguous quantity states, and identifies bounded option-denial scopes.
2. Focuses the direct interrogative request while retaining weaker whole-question context.
3. Retrieves clue-to-answer evidence from affirmed option concepts, with a smaller reverse path. Shared option wording and repetition of question words are discounted; unambiguous opposing properties are penalized. The three strongest records are pooled.
4. Adds signed conjunctive scientific-rule evidence and subject-aware option-assertion evidence.
5. Extracts experimental roles from explicit causal language and adds role-matching evidence.
6. Handles scoped exclusion requests and authored ordered comparisons.
7. Adds guarded dimensional, atomic-count, water-displacement, and geometric calculations.

There is no option-position parameter. The complete computation is option-permutation equivariant.

### Experimental-variable roles

The parser recognizes *effect of X on Y*, *how X affects Y*, and, at lower strength, *changes X and measures Y*. It extracts token spans for the manipulated quantity and measured response, converts those spans to concepts, removes shared concepts, and compares each role with the choices.

The role channel requires an explicit independent/dependent-variable request and matching evidence for both roles among the choices. Multiple candidate causal constructions or action verbs cause abstention. Action-based parsing requires active verbs before the direct request and does not guess from passive clauses. It does not infer causality merely from two quantities co-occurring.

### Ordered comparisons

Authored ranks compare recognized entities only. Each rank channel centers values over the recognized choices, leaving unknown choices at neutral evidence. This prevents descending comparisons from placing every recognized choice below an unknown one; neutrality also survives request-polarity reversal. The computation is invariant to an arbitrary additive shift of an ordinal scale and preserves pairwise evidence differences between recognized choices. Vacuum is excluded from the sound-speed ordering because sound does not propagate there, rather than having a comparable speed of zero.

### Numerical reasoning

The parent's guarded calculations are retained: density, speed, work, power, circuits, wave cycles/frequency/period, atomic and mass numbers, and water displacement. Their unit checks, first-request selection, ambiguity guards, and formula-consistency checks remain in place.

Explicit *per* now requires one quotient separator with positive-power units before it and negative-power units after it. Thus newton-meters can express work, but newtons per meter cannot; joules per hour can express power with conversion. Repeated or misplaced quotient separators cause abstention, and a quotient is not accepted as a simple volume. When no separator survives tokenization, existing ordered-unit interpretations remain available because a slash is discarded by the task tokenizer. Exact-value evidence follows the same scoped request-negation polarity as verbal evidence, so a direct *not* or *incorrect* request does not receive a positive numerical vote for the matching value.

A shared final guard applies to every numerical helper: it requires an unnegated point-value option, rejects unknown token IDs anywhere in the option, and abstains on authored bound/range cues and common explicitly unsupported units. Thus *24 J* can receive exact-work evidence, but *less than 24 J*, *24 cm2*, *24 pascals*, or a number followed by an unknown unit cannot. The declared unsupported units include area, temperature, pressure, percentages, and customary-unit forms not handled by the numerical formulas. This intentionally trades coverage for avoiding high-confidence dimension and bound errors; it does not solve inequalities or supply new conversions. Bare numerical options remain supported.

New geometry computes a rectangular box/prism's volume from exactly three explicit length readings, or a cube's volume from one explicit side/edge length. It converts mixed metric units and can derive density when exactly one mass is provided. The earliest requested volume/density quantity selects the output. Unsupported or ambiguous shapes—including hollow, spherical, cylindrical, conical, triangular, and pyramidal contexts—disable this helper. Numerical evidence requires compatible, singly numerical choices; it does not invent missing dimensions.

## Verification and resource use

`python /srv/work/tools/check_model.py` **passes**, including the architecture restrictions, build, and execution.

The separately run `/srv/work/verify_model.py` checked:

- exact compiler/forward feature equality on **14,018 distinct authored strings**;
- extractability of every declared atom;
- separation of positive and incompatible assertion feature sets;
- negative-phrase suppression, literal conjunctions, and cardinality distinctions;
- possessive/base-concept equivalence and contraction preservation;
- affirmative/negative statement verification with explicit option subjects;
- signed option assertions, clause barriers, mixed-polarity repeated concepts, and nested-negation abstention;
- intrinsic trait contexts, conserved combined mass, and floating-force preconditions;
- absent-structure requests;
- causal-role and active-action parsing plus conservative abstention;
- rectangular/cubic volume, mixed-unit conversion, density, and unsupported-shape abstention;
- explicit quotient dimensions, power-unit conversion, misplaced/repeated separators, and malformed-volume abstention;
- neutral unknown-option rank evidence, arbitrary-origin invariance, rank symmetry, and exclusion of vacuum from sound-speed ordering;
- numerical request-polarity reversal and negated-choice abstention in an isolated complete forward pass;
- exact-point guards for bounds, unknown words, and unsupported units across formula, geometry, displacement, and nuclear-count helpers;
- ambiguous-negation abstention across retrieval, experimental roles, ordered ranks, and the complete forward pass;
- forbidden assertion contexts from questions and options, true/false padding behavior, pressure-changing phase conditions, and static/sliding-friction separation;
- direct versus embedded request polarity, non-denial idioms, and trailing *EXCEPT*;
- consistent alga ancestry and removal of universal fish/lung incompatibility from the logical-rule channel;
- independence of logits from unrelated questions sharing a mixed batch;
- finite outputs for empty and unknown inputs, and equal scores for identical choices;
- option-permutation equivariance;
- **56 complete batches of 64 invented examples**, totaling 3,584 examples.

With four Torch threads, the latest verification built in approximately **1.2 seconds**, ran its 56-batch workload in **20.4 seconds**, and peaked at approximately **1,128 MiB** resident memory. These are implementation/resource checks, not accuracy measurements or model-selection runs, and are within the 2 GB / 20 minute budget.

## Limitations

This is not a pretrained language model or a general grammatical reasoner. Lexical overlap can still confuse objects, clauses, negation, and causal direction. Option-negation scope is bounded and cannot recover punctuation removed by tokenization; nested or complex negation is not generally resolved. Assertion verification only covers authored schemas and does not establish the truth of arbitrary sentences. Rule/schema context guards do not automatically scope or qualify separate generic retrieval records, which can still supply broader evidence. Taxonomic inheritance uses school-level generalizations with declared exceptions. Some lexical quantity patterns still conflate magnitude and change. Experimental-role parsing is intentionally narrow. Tokenization removes slash, multiplication, and percent punctuation, leaving some compound-unit expressions and unitless-looking values inherently ambiguous; explicit *per* and declared unsupported unit words are checked, but missing punctuation cannot be reconstructed reliably. Exact-value guards deliberately abstain on unknown option words, which can exclude otherwise interpretable values, and cannot recognize every possible unsupported unit or inequality phrasing. Decimals, numerical tables, unsupported units, ambiguous measurements, and complex multistep calculations remain limited. Conservative channels abstain when their required structures are absent.

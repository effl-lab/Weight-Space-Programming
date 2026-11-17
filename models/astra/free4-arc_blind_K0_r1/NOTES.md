# Authored science model — generation 4, reviewed

## Summary

This is a deterministic, hand-authored science model built on the supplied generation-3 parent. It combines sparse scientific-assertion retrieval with finite tensor reasoning modules. This generation adds **1,330 authored assertion lines**, a conservative lexical supplement, and an explicit **collinear net-force calculation** module. Reviews add clause-local retrieval polarity, conditional-premise scope handling, repairs to five split lexical families, request-local class selection that distinguishes category requests from incidental class mentions, bounded English-cardinal parsing with complete number-to-unit binding, explicit directional grammar that distinguishes transformations from unordered endpoint lists, and identity-aware pair binding for magnetic/electric interactions.

The supplied parent's reported validation accuracy was **44.8%**. No held-out questions, labels, or per-question results were available or accessed. The additions aim to improve coverage and reasoning, but no held-out improvement is claimed.

There is no training, parameter fitting, pretrained component, downloaded corpus, random initialization, external service, or automated accuracy-based search. All additions are authored scientific knowledge, literal grammar, or mathematical operations. Specificity normalization uses only frequencies within the authored knowledge, not question data. The inherited retrieval coefficients and existing symbolic scoring coefficients are unchanged.

## Delivered files

- `vocab.txt`: **24,750** words. Line k maps to token ID k+2; padding and unknown remain IDs 0 and 1.
- `build.py`: defines `build()` and compiles all weights into a dictionary of Torch tensors. It reads only files in this folder. `python build.py --vocab` deterministically regenerates the vocabulary.
- `arch.py`: defines `forward(w, q, o)`. Its imports are only Torch and math, and all question-dependent operations are tensor operations. The required checker accepts its restricted syntax.
- `mechanisms_physics.txt`: motion, interactions, machinery, heat, circuits, magnetism, sound, and optics, connecting mechanisms to everyday observations.
- `mechanisms_life.txt`: cell and organ functions, plant experiments, adaptations, reproduction, inheritance, evolution, and health.
- `mechanisms_earth.txt`: geological evidence, surface processes, tectonics, water, weather, climate, and astronomical observations.
- `mechanisms_matter.txt`: particle explanations, reactions, conservation, mixtures, separation, material selection, atoms, and periodicity.
- `mechanisms_ecology_inquiry.txt`: ecological dependencies, resources, environmental effects, fair testing, evidence, measurement, models, engineering, and safety.
- `lexicon_mechanisms.txt`: conservative additional lexical forms, spelling variants, and two/three-word scientific phrases. It complements the parent's three lexicon files.
- `force_grammar.txt`: literal direction, request, bridge, exclusion, and option-wrapper vocabulary for the new force calculator.
- `scope_grammar.txt`: literal connective boundaries, negators, fillers, and conditional/causal request-premise cues for the reviewed scope handling.
- `class_grammar.txt`: literal prefixes, copular bridges, and generic subject wrappers for requested-category binding.
- `relation_grammar.txt`: literal directional prepositions, result predicates, passive-agent syntax, and supported intervening words for the ordered scientific relations.
- `interaction_grammar.txt`: literal quantity bridges, typed polarity qualifiers, and exclusions for simple two-object magnetic/electric interactions.
- The parent's scientific assertion, taxonomy, property, comparison, opposition, causal, entity, and grammar files remain included and in use.

The five new knowledge files contain 1,330 non-comment assertion lines in total. Some intentionally overlap existing knowledge while expressing distinct contexts or mechanisms. The compiler deduplicates identical canonical assertions, so a written line is not necessarily a new independent weight entry.

Assertion syntax remains:

```
context concepts | supported completions separated by semicolons | contradicted completions
```

The final field is optional. Underscores represent phrases. `=` permits stronger reverse definition lookup; ordinary affirmative assertions receive weaker reverse associations. `!` marks an explicitly directed exception context. These are knowledge statements, not stored ARC question/answer records.

## Representation and retained inference

1. **Lexical recognition:** authored aliases, inflections, phrase matches, and bounded soft neighborhoods produce concept features and positions. Entity shielding distinguishes names such as carbon dioxide from independently mentioned component substances.
2. **Type inheritance:** directed taxonomy propagates member-to-class evidence, not class-to-member evidence. Inferred question classes have weaker retrieval strength than explicitly named concepts.
3. **Sparse retrieval:** normalized concept-to-context and concept-to-completion maps are coalesced sparse COO tensors. Fixed sparse matrix multiplication computes context and answer coverage. Context completeness, specificity, predicate oppositions, distinctive answer content, and completion precision control evidence.
4. **Signed aggregation:** fixed top-evidence aggregation combines supported and contradicted completions. Limited request/option negation and exception handling supply polarity. No option-position prior is used.
5. **Statement completion:** a lower-weight backoff recognizes affirmative facts whose content is distributed between question and option.
6. **Retained reasoning:** direction-sensitive transformations; magnetic/charge interactions; 18 dimension-aware arithmetic formulas; physical-applicability guards; scalar comparisons; qualitative causal laws; requested-class selection; experimental-variable binding; explicit one-edge food-web population effects; and goal-conditioned mixture separation.

All these remain finite mechanisms, not unrestricted language understanding or a general theorem prover.

## New collinear force calculator

The new `collinear_forces()` module handles narrow numerical questions such as two or three explicitly stated force magnitudes followed by their directions, with a final request for a net, resultant, combined, or total force.

- Every input magnitude must have a complete attached force unit recognized by the existing dimensional parser.
- Directions must follow the unit through a short supported literal bridge, for example `11 newtons to the right`.
- Signed direction vectors use four separate coordinate families: left/right, east/west, north/south, and down/up. Mixing families causes abstention rather than inventing a geometric relationship.
- All measurements must belong to the premise before the final request, and each must bind exactly one direction. Repeated/conflicting directions cause abstention.
- Two or three forces must be present and collinear. Their signed magnitudes are added; absolute resultant magnitude and direction are compared with each choice.
- Explicit angular, inclined-plane, frictional, collision, torque, acceleration, and selected multi-object/separate-force wording disables this module. Negated questions also disable it.
- Each supported option contains one parsed number, at most one complete force-unit occurrence, at most one direction, and only allowed wrapper words. Unsupported units, unknown assertions, extra values, multiple directions, or negation cause option-level abstention.
- A directionless numeric option can answer magnitude alone. A zero resultant cannot support an asserted nonzero direction.

Numerical agreement uses the inherited arithmetic convention, an exponential of relative log error. Its manually specified contribution scale is 3.0, matching the existing arithmetic module; no coefficient was searched or fitted.

This is not a general mechanics parser. It does not recover punctuation-erased negative signs, solve arbitrary vector geometry, infer omitted forces, bind arbitrary object identities, or handle directions appearing before the measured force. The existing decimal-token ambiguity also remains. Zero contribution means this module abstains; the rest of the model still answers.

## Review: scope and lexical consistency

### Conditional premises are not exception requests

`answer_scope()` now recognizes literal conditional and causal suffixes after the initial request word, in addition to its existing why-clause handling. For example, the negation in `What happens if a seed does not receive water?` describes a premise; it does not ask for a false answer. Conversely, the main-clause negation in `Which statement is not true if ...?` is retained. The same scoped request is available to the existing modules that already use `answer_scope()`.

### Retrieval polarity follows matched content

Previously any option negation reversed every retrieved fact matched by that option. Now `scoped_option_features()` separates affirmative and negated concept evidence using literal connective boundaries. A negator applies forward within its supported clause. A short terminal ellipsis such as `rubber does not` can negate its local subject, but unknown following words count as content rather than disappearing. The existing `not only`, `not just`, and `not merely` exemptions remain in force, and neither/nor lists do not double-negate.

For each completion, `signed_agreement()` computes the difference between affirmative and negated coverage divided by their sum. This supplies the polarity and confidence of that particular match. Conflicting occurrences of the same concept cancel; a negated explanation no longer automatically reverses an unrelated main-clause fact. Whole-statement backoff scopes both the context and completion halves, preserving evidence when an option supplies only the fact's subject.

No retrieval scoring constant was fitted or changed. Entirely affirmative matching supplies a factor of exactly +1. Twelve forward equivalence specifications verify unchanged scope/retrieval scores against the preserved pre-review inference code when the new request scoping is inactive. To isolate this earlier regression from the later class-selector repair, both runs use identical copied test weights with the class-request contribution disabled; delivered weights are not modified. Separate class specifications and full-forward invariance checks use the actual model weights. The existing symbolic modules retain their own negation gates; this review is not a wholesale rewrite of their semantics.

This is still bounded lexical scope, not a semantic entailment engine. Punctuation is absent; an `and` boundary can be ambiguous; inserted subordinate clauses, nested verbs, quantifiers, double negation, and object binding are not generally solved. Mixed matches receive graded evidence rather than a proof of conjunction. The compiler's existing treatment of negation inside authored completion strings is unchanged, so mixed affirmative/negative authored fragments can still be imperfectly represented.

### Five restored lexical families

An internal audit of the authored alias tables found that later self-root declarations had split previously declared families. The final lexical supplement now reunifies:

- acid / acids / acidic;
- base / bases / basic / alkaline / alkalinity;
- conduct / conductor / conduction / conductive / conductivity;
- permeable / permeability;
- burn / burned / burning / burnt / combust / combustion.

Deliberate distinctions remain, including low versus the lowering action, offspring versus young, skeleton versus skeletal, muscle versus muscular, and the opposing chemical/material categories. These are repairs to authored concept identity, not learned synonym embeddings.

## Second review: requested-category binding

The requested-class bonus previously used the first concept positions from the entire question. An earlier background mention could therefore hide the same class in the final request. `requested_classes()` now uses concept features and positions extracted from the scoped final request.

- Direct requests such as `which animal` or `which of the following animals` require supported literal prefixes.
- Copular category requests such as `which sample is a metal` require a supported generic subject or an already recognized direct class subject, plus a short supported bridge. A predicate category can still specialize the direct subject class.
- Questions about statements, explanations, characteristics, and similar unsupported subjects no longer receive a category bonus merely because a class occurs nearby.
- Only class features with strength at least 0.99 can supply a literal target. Soft semantic neighbors may aid retrieval but cannot masquerade as explicitly requested class names.
- Every supported bridge word is explicitly seeded in the vocabulary, including `considered`, which was previously unknown despite appearing in the intended bridge table.
- Request-local features are shared with the existing comparison module. Eight dedicated comparisons remain exactly equal to the pre-class-review calculations. The class-membership coefficient remains 0.75; no scoring coefficient was fitted or changed.

This is a conservative finite grammar, not a general parser. Unsupported modifiers or subject forms may cause the class bonus to abstain, and compound category constraints are not generally solved. Other retrieval and reasoning modules still contribute.

## Third review: composed number words and unit binding

A direct measurement-parser audit found that `twenty one grams` was interpreted as 1 gram, `two hundred grams` as 100 grams, and `one hundred and twenty five grams` as 5 grams. Only the last numeric word had an adjacent unit. Several teen and tens words were also missing from the numeric table.

The compiler now includes the complete zero-to-nineteen and tens vocabulary and a literal five-column cardinal grammar table. `number_spans()` performs fixed tensor operations in three stages: tens plus units, a one-to-nine multiplier plus `hundred`, and a one-to-999 multiplier plus `thousand`. The scale stages can incorporate a smaller word-number, optionally introduced by `and`. This supports conventional spelled-out nonnegative integers through 999,999, including hyphenated tens after tokenization. Existing standalone `hundred` and `thousand` values remain available.

Each recognized number has one value at its first token and a complete token span. Consumed words no longer become separate measurements. Units attach after the entire span, not after its last numeric fragment. Arithmetic option validation, force-direction binding, and bare-number comparisons use these spans and only accept internal conjunctions covered by a valid numeral. Separated measurements such as `five grams and six grams` remain distinct.

Adjacent unresolved numeric fragments, including fragments separated by `and`, are rejected rather than using the final fragment's unit. The grammar deliberately abstains on unsupported constructions such as `twelve hundred`, `2 hundred`, or `twenty 1`. Existing digit/decimal parsing and symbolic score coefficients are unchanged. This is literal grammar and base-ten arithmetic, not fitting or learned number recognition.

Limitations remain: mixed digit/word expressions, number words for decimals or fractions, ordinals, and larger word scales are outside this bounded grammar. It does not prove that every unsupported expression embedded in prose is a single quantity. Punctuation-erased digit pairs retain the inherited ambiguity between decimal points, separators, and fraction notation; signs are not recovered. The class, scope, and scientific-assertion mechanisms were not retuned in this review.

## Fourth review: directional relations require directional wording

The ordered-relation module previously treated any two recognized endpoints as a directed transformation. For a motor context, `electrical energy and mechanical energy` received a +1.8 directional contribution, while the same unordered list in reverse order received -1.8. A passive-agent construction such as `mechanical energy is produced by electrical energy` was also assigned the wrong direction because only the word `from` reversed lexical order.

`relation_directions()` now applies the same bounded tensor grammar to option-side transformations and question-side inverse lookup across the 35 existing authored relations:

- Both endpoints must be recognized, at distinct positions no more than 12 tokens apart.
- `to`, `into`, and related directional prepositions supply forward orientation; `from` supplies reverse orientation.
- Active result predicates such as `becomes`, `produces`, and `generates` can supply direction without a preposition.
- A supported past participle with `by` reverses the endpoints. A passive auxiliary plus a participle without a directional preposition or agent does not invent an omitted source.
- Explicit prepositions take precedence over result verbs. Conflicting forward and reverse cues cancel.
- Every intervening token must belong to the authored grammar. Unordered lists, coordination, causal-clause boundaries, unknown intervening content, and negation within the bridge cause this directional component to abstain.

The existing 1.8 contribution scale, scientific relation table, context weights, and other reasoning modules are unchanged. No coefficients or knowledge assertions were selected from measured accuracy.

This is not a general event parser. It retains the existing concept-position representation, so repeated mentions and soft-neighbor positions can still obscure the intended endpoints. It does not recover arrows erased by tokenization, infer an omitted endpoint, or parse arbitrary passive, coordinated, or multi-clause descriptions. Material outside the tested bridge is not fully validated. The rest of the model still supplies retrieval evidence when this component abstains.

## Fifth review: interaction identity and multiplicity

The old interaction rule treated a generic `two` or `both` anywhere in the question as evidence that a single named polarity applied to both objects. It therefore assigned repulsion to a description of one positive object and one neutral object, and to a north pole facing an unknown pole. A generic `same` also meant same polarity, so `same size` could cancel attraction evidence from explicitly positive and negative charges.

`interaction_pairs()` now shares identity and multiplicity logic between question-described interactions and pair-valued choices:

- Literal north/south and positive/negative identities are counted separately. Magnetic and electric identities are not combined into one pair.
- A number or `both` must bind locally through a supported literal construction. The existing complete-number parser prevents `twenty two` from being mistaken for `two`.
- Predicative groups such as `both objects are positive` can repeat one named identity, but do not duplicate one label in `two poles are north and south`. Explicit larger groups abstain.
- A supported pair contains exactly two counted members. Contradictory like/unlike qualifiers suppress the match.
- Like/same/opposite qualifiers must bind a nearby plural `poles` or `charges` through the supported grammar. Size, shape, location, and charge-magnitude wording are not polarity evidence.
- Neutral, uncharged, unspecified, unknown, and unmagnetized descriptions abstain from this simple rule. This does not assert that charged and neutral bodies cannot attract; polarization is different physics.
- Inverse counterpart lookup requires one affirmed query identity and one option identity, not a pre-described pair, larger group, mixed domain, or excluded counterpart. Bare magnetic-pole contexts can now supply the magnetic domain without requiring the separate word `magnet`.

The 1.8 interaction coefficient and scientific like/unlike laws are unchanged. This is bounded grammar, not a general object or discourse binder. Singular `same charge` alone does not establish a pair; distant references, arbitrary pronouns, nested negation, and complex group descriptions remain unsupported. Exclusion words anywhere in the parsed description conservatively suppress this component, even when a fuller parser could isolate an unrelated clause. Other model components remain active.

## Verification actually performed

The required command passes:

```
python /srv/work/tools/check_model.py
CHECK OK
weights 177,377,682 (330,788 non-zero), vocabulary 24,750 words
```

The logical weight count includes implicit zeros in sparse matrices and is not actual storage size.

`/srv/work/verify_model.py`, outside the delivered folder, checks:

- **28** exact force-calculation specifications across the four coordinate families, same/opposite directions, cancellation, three-force addition, abbreviations, number words, and decimals.
- **15** context-abstention specifications for unsupported geometry, missing/ambiguous directions, non-force requests, unsuitable units, negation, and selected separate-object contexts.
- **8** invalid-option specifications covering extra assertions, repeated/incomplete units, multiple directions, wrong coordinate families, negation, extra measurements, and unknown units.
- **3** retained arithmetic specifications: density, speed, and Ohm's law.
- Finite `(n, 4)` output, option-permutation equivariance, batch reversal, single-item/batch consistency, exact identical-option ties, and zero output for wholly padded or unknown inputs.
- Sparse-column normalization and sparse/dense projection equivalence on fixed submatrices.
- A 3,530-input, four-thread synthetic execution and resource checks.

The first specification run exposed one semantic bug: the force binder accepted the first of two consecutive conflicting direction words. The bridge/count logic was corrected so this is ambiguous and receives no force contribution. All specifications then passed. No scoring coefficient was altered.

The review additionally ran `/srv/work/scope_specs.py`: **28** text cases with **90** literal polarity checks, **5** predicate-agreement checks, **20** question-polarity cases, **12** forward equivalence cases against the pre-review architecture with the class contribution disabled on both sides as explained above, and **28** mixed-choice full-forward invariance inputs using actual weights. The old question-scoping logic disagreed with **7** of the 20 scope specifications; the reviewed logic passes all of them.

`/srv/work/lexicon_specs.py` verifies the **5** repaired families across **22** forms, **10** preserved distinctions, and **5** phrase-level feature equivalences.

`/srv/work/class_specs.py` checks **26** supported category requests, **18** non-category-request abstentions, **36** repeated-class cases, **4** member-to-category cases, **8** exact comparison-equivalence cases, and **56** full-forward invariance inputs. The prior selector disagreed with **21** of the 44 supported/abstention specifications. During development, this suite exposed the missing bridge vocabulary word and the mistaken use of soft-neighbor positions; both were repaired. An older scope-equivalence fixture also initially differed because the class repair intentionally changed its membership bonus, so that regression now isolates scope behavior rather than requiring the repaired class selector to reproduce the old bug.

`/srv/work/number_specs.py` checks **2,053** cardinal values and spans: every integer from 0 through 999 with and without the optional conjunction, representative thousand-scale boundaries through 999,999, hyphenated tens, and legacy standalone scale words. It also checks **16** unsupported-number abstentions, **4** separated-measurement cases, **16** legacy measurement equivalences, **5** arithmetic cases, **8** invalid arithmetic options, **3** force calculations, **3** comparisons, and **11** full-forward invariance inputs. Legacy equivalences compare measurement values and counts against the preserved pre-number-review architecture; no unused tensor slot is treated as a semantic output.

`/srv/work/relation_specs.py` checks **560** orientation cases and **560** unsupported-bridge abstentions instantiated across all **35** authored endpoint pairs, plus **10** endpoint/domain cases, **8** missing-endpoint cases, and **12** full-forward integration and invariance inputs. The prior direction rule disagreed with **105** orientation cases and all **560** abstention cases. Full-model differences against the preserved pre-relation-review architecture equal the independently reconstructed directional-contribution differences within floating-point tolerance, under identical weights. These templated counts measure grammar coverage, not independent science-question accuracy.

One initial endpoint fixture incorrectly assumed bare `mechanical` was already recognized as mechanical energy. The lexicon does not provide that mapping. The unsupported abbreviated wording was retained as an abstention test; supported energy-phrase variants were tested separately, without adding a broad synonym to the model.

`/srv/work/interaction_specs.py` checks **98** supported pair descriptions, **44** unsupported-pair abstentions, **8** inconsistent-pair cases, **10** forward interactions, **6** unknown/unsupported-counterpart abstentions, **4** inverse counterpart cases, **4** pair-choice cases, and **24** full-forward integration and invariance inputs. The old module disagreed with **4** of the six counterpart-abstention specifications. The new suite initially exposed an additional error: `three poles are north and south` was counted as a pair because it contains two identity labels. The explicit larger-group guard repairs this. Full-model changes equal the independently reconstructed changes in the two repaired interaction modules within floating-point tolerance under identical weights.

All seven suites and the required checker pass on the current model. These are deterministic semantic, arithmetic, and invariance checks, not held-out accuracy measurements.

The checker also runs **20 independently written science smoke questions**. Before the first review the result was **16 correct**; after the first review and again after the class-selection, number-parser, directional-relation, and interaction reviews the result is **17 correct**. The remaining errors concern paved-surface runoff, the rationale for repeated trials, and closed-system reaction mass. They expose associative-retrieval and sentence-interpretation limitations. No parameter sweep, accuracy-based search, or question-specific fact addition was performed. This small, reused authored set is not an independent ARC benchmark or a test-accuracy estimate.

The parent's extensive specialized regression suites were not supplied here and are not claimed as rerun. The checks above are those performed for this delivery. Verification scripts, inputs, and results are outside the model folder and are not build dependencies.

## Measured resources

Four-thread CPU execution, with batch size 64 for the 3,530-input workload:

| Quantity | Measurement |
|---|---:|
| Build time | 0.93 seconds |
| Forward time, 3,530 synthetic inputs | 55.76 seconds |
| Peak process resident memory | 1,424.58 MiB |
| Actual tensor storage including sparse indices | 35,110,950 bytes |
| Vocabulary words | 24,750 |
| Concepts including null | 5,392 |
| Retrieval contexts | 10,471 |
| Signed completion entries including reverse links | 20,886 |
| Shared phrase patterns | 529 |

The synthetic workload repeats independently authored specification inputs, not held-out ARC data. Peak memory includes imports, build, and the verification sequence. The measured run is below the 2 GB memory and 20 minute runtime limits.

## Remaining limitations

Coverage is finite and paraphrase-sensitive. Strong lexical overlap can still outweigh the intended causal explanation; negation scope, universal claims, irrelevant clauses, narrative referents, and competing facts are only partially handled. Multi-hop reasoning, diagrams, tables, unfamiliar names, and elaborate word problems remain difficult. Tokenization erases punctuation and signs. More authored knowledge does not eliminate these structural limitations, and only the unavailable held-out evaluation can establish whether this generation improves on 44.8%.

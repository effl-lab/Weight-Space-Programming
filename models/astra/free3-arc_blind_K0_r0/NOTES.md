# Authored science model — generation 3, reviewed

## Summary

A deterministic, hand-authored model for elementary and middle-school science multiple-choice questions. It extends the supplied generation-2 model with **997 additional scientific assertion lines**, additional lexical forms, sparse retrieval, conservative retrieval-precision checks, an explicit food-web population-effect module, and goal-conditioned separation-method reasoning.

Nothing is trained. There is no pretrained model, external corpus, network dependency, random initialization, gradient computation, parameter fitting, or automatic architecture/content search. Scientific facts, grammar cues, coefficients, formulas, and vocabulary-generation rules are authored directly. The supplied parent is the only reused model.

The parent's reported validation accuracy is **43.5%**. No validation/test questions or per-question results were available or accessed. This model aims to improve on that baseline, but **no held-out improvement is claimed**.

## Files and compilation

- `vocab.txt`: 23,365 words; zero-based line k has token ID k+2. IDs 0 and 1 are padding and unknown.
- `build.py`: `build()` reads only the authored model files and returns a dictionary of Torch tensors. `python build.py --vocab` regenerates the vocabulary deterministically.
- `arch.py`: `forward(w, q, o)` accepts the specified integer tensors and returns finite floating-point `(n, 4)` logits. Imports are only Torch and math. All input-dependent work is tensor computation.
- `biology.txt`, `physics.txt`, `chemistry.txt`, `earth_space.txt`, `inquiry.txt`, `extra.txt`, `concepts.txt`, `extensions.txt`, `scenarios.txt`, `review.txt`: retained parent assertions.
- `practical_science.txt`: **203** new assertion lines about everyday mechanisms, machines, materials, heat, optics, sound, and electrical devices.
- `life_applications.txt`: **248** new lines connecting biological structures, experiments, observations, health, ecology, and inheritance.
- `earth_applications.txt`: **222** new lines about astronomical observations, geological evidence, surface processes, weather, water, and resources.
- `matter_applications.txt`: **168** new lines about physical/chemical changes, separation, particle models, atomic structure, materials, and reaction evidence.
- `inquiry_applications.txt`: **156** new lines about variables, fair comparisons, measurement, evidence, experimental limitations, design, and laboratory safety.
- `lexicon.txt`, `lexicon_extra.txt`, `lexicon_new.txt`: canonical concepts, synonyms, irregular morphology, and two/three-word phrases. Fixed spelling rules generate additional inflections.
- `ecology_grammar.txt`: literal feeding verbs, active/passive markers, population-change cues, clause boundaries, and request words for the new trophic module.
- `separation_grammar.txt`, `separation_materials.txt`, `separation_methods.txt`, `separation_rules.txt`: literal goal grammar, recognized material/mixture signatures, eight named method classes, and 20 manually authored suitability rules. They compile into token tables, short phrase patterns, and small dense reasoning maps.
- `formula_conditions.txt`: nine authored physical-applicability guards, scoped to the whole context or final request and compiled into token-to-condition and condition-to-formula maps.
- The parent's taxonomy, entity, opposition, causal, experimental-grammar, ordered-relation, property, and comparison files remain in use.

Assertion syntax is:

```
question concepts | supported alternatives separated by semicolons | contradicted alternatives
```

The third field is optional. Underscores denote phrases. `=` enables a stronger reverse definition lookup; ordinary affirmative assertions also get weak associative reverse links. `!` marks an already-directed exception rule. These are authored knowledge statements, not question/answer records.

The compiler deduplicates assertions, assigns specificity weights from frequencies in the authored knowledge itself, and compiles the fixed reasoning tables. This normalization uses no question data. It does not read the smoke-test questions, diagnostic scripts, or verification outputs.

## Representation and inference

### 1. Lexical recognition and scientific inheritance

Token lookups, phrase matching, explicit aliases, and fixed soft semantic neighborhoods produce concept features and textual positions. Directed taxonomy propagates member-to-class evidence, never class-to-member evidence. Multiword entity shields prevent examples such as `carbon dioxide` from being treated as an independently named carbon sample for typing or scalar lookup.

Review repaired nine lexical families whose later entries had split the intended canonical features: salt/sodium chloride, conduct/conductor, insulate/insulator, color/coloration, pollute/polluted, decrease/reduce, respond/response, explain/explanation, and oxidize/oxidation. This restores shared retrieval evidence across these authored forms. It does not blindly merge every conflicting synonym group: distinctions such as offspring versus young, or a body structure versus a body-system adjective, remain as previously authored.

### 2. Sparse knowledge retrieval

The two largest maps, concept-to-context and concept-to-assertion, are compiled directly as **coalesced sparse COO tensors**. No large dense zero-filled version is allocated by `build()`.

`project()` applies each sparse linear map with `torch.sparse.mm`, flattening and restoring leading batch dimensions. This is the same normalized linear projection as its dense counterpart; the architecture remains a fixed tensor program, not an external search engine.

Retrieval favors highly covered, specific question contexts and scores their positive and negative completions. Explicit opposite predicates suppress contradictory partial matches. Fixed top-evidence aggregation combines the strongest matching statements. There is no option-position prior.

### 3. Conservative retrieval precision

Two general safeguards supplement the parent scoring:

- **Inferred question types are weaker than explicit context.** A class feature having no textual position receives a factor of one half in the retrieval question representation. Full taxonomic features remain available to option typing and symbolic modules. Merely mentioning water therefore supplies weaker evidence for a general question about compounds than explicitly mentioning `compound` would.
- **Answer precision complements answer coverage.** The weighted fraction of distinctive option content supported by a completion contributes a bounded factor between one half and one. Unknown option tokens carry uncertainty mass instead of disappearing from the precision denominator. The whole-statement backoff also counts this uncertainty mass.

These are fixed, manually selected safeguards, not fitted coefficients. They reduce some partial-match errors but do not establish logical entailment or guarantee that an irrelevant addition can never improve the overall score through a different fact.

### 4. Whole-statement completion

A lower-weight backoff scores affirmative facts jointly covered by question and option. Only novel option content adds coverage. Authored false completions do not become reverse facts. This helps when the description and the name occur on opposite sides of a knowledge relation, but remains associative rather than deductive.

### 5. Explicit trophic population effects

The new module recognizes short, explicit clauses such as:

```
A eats B
A feeds on B
B is eaten by A
```

It binds known organism names near a literal verb, distinguishes active/passive direction, respects selected clause boundaries, and rejects locally negated edges. Organism identities come from the authored taxonomy rather than from assumed real-world diets. A stated relationship in the question supplies the edge; a species name alone does not.

An explicit increase/decrease is bound to its nearest organism. It must be a premise before the final request or in an explicit conditional clause. Exactly one unambiguous changed population is required. Ordinary `grass grows` is not interpreted as an exogenous population increase.

The reviewed binder additionally requires a supported short bridge between the organism and the change marker. Nearby competing attributes (mass, speed, temperature, food supply, and others) or unparsed coordination cause abstention. Thus `fox mass increases` is not a population increase. Explicit removal or extinction remains a valid **input premise**, but is not inferred as an output merely from a qualitative decline.

For a direct feeding relationship and otherwise fixed conditions:

- An increase in food supplies positive support for an increase in its consumer.
- An increase in consumers supplies positive support for a decrease in their food population.
- Decreases reverse these signs.

Effects are combined conservatively; conflicting direct effects cancel. An option must name one affected organism, or omit the name when the request names exactly one target. An explicitly named option must agree with any explicitly requested organism: a true statement about mice is not supported as an answer about rabbits.

Only simple population-change options with supported vocabulary are scored. Unknown names, competing attributes, exact numerical amounts, extinction predictions, extra clauses, negation, and multiple organism names cause abstention. Bare target borrowing therefore cannot turn `the temperature decreases` into a population answer. The request itself is checked for unsupported attributes and unknown words. A trailing `if`/`when` clause is excluded from target/request matching while remaining available as an input premise.

Missing relationships, ambiguous changes, and unsupported targets also cause abstention. This deliberately conservative grammar may miss valid but elaborate explanations; they remain available to the other retrieval modules.

This is **one-edge, qualitative, all-else-equal reasoning**. It does not simulate ecological equilibrium, infer arbitrary diets, interpret lost arrow punctuation, perform unrestricted trophic cascades, or infer no effect just because an edge was not recognized. Contributions are bounded and independent of choice position.

### 6. Goal-conditioned separation methods

The second review adds a deliberately narrow method-selection module. It distinguishes the desired product from the material being removed: recovering water from saltwater favors distillation and rejects evaporation alone; recovering salt favors evaporation, with distillation as a less direct alternative. Removing one component of a recognized binary mixture treats the other as the desired product. Generic separation requests use fallback suitability rows instead of inventing a product goal.

Literal one-, two-, and three-word patterns recognize every material occurrence, including repeated mentions after an earlier description. Mixture names supply authored signatures, rather than unrestricted taxonomic ingredients. Multiword material names shield their component words; selected excluded compounds and water vapor also shield their words and disable this module. The recognized signature must contain exactly two identifiers and match an authored rule.

A product binds within a short supported bridge after a separation/recovery/removal cue and before `from`, a mixture head, or another action. Multiple distinct products, unbound explicit recovery goals, and ambiguous removal requests cause abstention. Specific product rules supersede generic rules for the same mixture. Dissolution cues suppress rules that assume solid particles.

The final request must ask for a method or separation action. Competing measurement/identification tasks, explanatory/property requests, selected chemical/biological-purification contexts, unknown question words, and negation disable the module. Each option must contain exactly one recognized literal method and only allowed wrapper/phrase words. Unsupported assertions, multiple methods, unknown words, and negated options receive no contribution. Elaborate procedural options remain with ordinary retrieval.

Suitability scores are authored as 1 for a customary suitable method, .5 for a viable less direct alternative, -1 for an ineffective method, and 0 for unlisted evidence. The contribution scale is fixed at 1.35; neither these values nor the grammar were searched or fitted against measured accuracy. This is finite classroom-level reasoning, not a general chemistry parser: unfamiliar materials, context scope, reaction products, multi-stage procedures, equipment limitations, and elaborate purity requirements remain limitations. Abstention here means zero additional separation evidence, not that the full model stops answering.

### 7. Reviewed numerical scope and complete units

The third review preserves the 18 authored formulas, their coefficients, and the numerical agreement scale. Formula selection now uses literal canonical quantity names in the final answer-request scope, rather than any topic anywhere in the question. A background statement about speed no longer licenses a numerical speed answer to a final question about color.

The unit parser also returns the span consumed by each recognized unit. A following quotient, exponent, or additional unit invalidates an incomplete prefix: `cubic meters per second` is not treated as plain volume, and `meters per second per kilogram` is not treated as plain speed. Supported primitive, cubic-volume, density, speed, power, and acceleration units retain their dimensional conversions. This stricter parser also feeds the numerical comparison module.

Arithmetic options must contain exactly one parsed value and account for every token using that number and its complete attached unit span. Bare numbers may borrow the requested dimension; explicit units must match it. Optional `about` or `approximately` prefixes are permitted before the number. Unknown words, extra assertions, unsupported qualifiers, additional units, and multiple values cannot silently disappear. Decimal joining requires digit tokens; a spelled number followed by a separate digit is not manufactured into a decimal.

The task tokenizer still erases punctuation. Adjacent digit tokens can therefore be a decimal or an ambiguous separated pair, and negative signs cannot be recovered. The retained decimal heuristic cannot resolve that lost information. Formula applicability, object identity, direction, long-distance scope, and elaborate word problems remain limited; this is not general symbolic algebra. Unsupported numerical phrasing receives no extra arithmetic evidence, while the rest of the model remains active.

### 8. Physical applicability and requested comparison attributes

The fourth review separates dimensional validity from physical applicability. Nine authored exclusion rules in `formula_conditions.txt` withhold a formula when explicit wording conflicts with its assumptions. They distinguish unsigned collinear work from angled, opposing, frictional, or variable-force situations; transferred work from a particular stored energy; translational kinetic energy from rotational energy, changes, or initial/final-state problems; net force from individual force components; and average speed or constant-speed travel from unsupported instantaneous/extreme-speed or accelerating-motion requests.

These checks apply to direct arithmetic and to derived values used in comparisons. Comparisons also bind a nearby explicit scalar attribute after a comparative cue. If requested work is unavailable, the module cannot instead rank a merely mentioned force. The binding filters authored rule contexts rather than forcing identical input/output properties: the retained rule that compares mass at fixed volume using density remains available. Multiple distinct bound attributes cause abstention. Comparative operators such as `highest` rank described options and are not themselves treated as a demand for an unobserved peak speed.

The 18 formulas and their numerical coefficients are unchanged. These are conservative literal guards, not a vector-mechanics solver or proof that every unblocked formula is applicable. Angled work is withheld, not evaluated with a cosine; negative work is not fabricated after tokenization has erased signs. Some valid problems with unrelated blocking words, explicitly matched initial/final states, or additional assumptions will also be withheld. Zero symbolic evidence leaves ordinary retrieval active.

### 9. Retained symbolic modules

The parent modules remain integrated:

- Request-scope, exception, and limited explanatory-premise negation handling.
- Direction-sensitive energy transformations.
- Magnetic-pole and charge interactions, including pair multiplicity.
- Dimension-aware arithmetic with 18 formulas and unit checking.
- Scalar-property comparisons with known reference values and conservative abstention.
- 121 authored qualitative causal laws with local quantity/direction binding.
- Requested-class selection from explicit question wording.
- Explicit manipulated/measured/fixed experimental-variable role binding.

The representation still distinguishes mass number from average atomic mass, alloys from pure elements, and compounds from separately named components.

## Verification performed for this generation

The required command passes:

```
python /srv/work/tools/check_model.py
CHECK OK
weights 134,088,895 (277,576 non-zero), vocabulary 23,365 words
```

The weight count includes logical zero entries in sparse tensors; it is not actual storage size.

Additional deterministic checks in `/srv/work/verify_model.py` passed:

- **72** direct trophic monotonicity specifications spanning six organism pairs, both changed endpoints, both change directions, and active/feed-on/passive syntax.
- **10** trophic abstention specifications for missing/negated edges, missing or conflicting changes, unsupported targets, ordinary growth, and outcome cues that are not premises.
- **8** option-binding specifications, including named targets, reversed direction, multiple names, and negated choices.
- Finite output and required `(n, 4)` shape.
- Option-permutation equivariance.
- Batch-order and batch-composition invariance.
- Exact ties for identical options.
- Equal zero logits for entirely padded or entirely unknown inputs.
- Sparse/dense projection equivalence on fixed submatrices, for two- and three-dimensional feature arrays.
- Unit-sum normalization of all compiled context/assertion columns.

Additional review checks in `/srv/work/review_specs.py` passed:

- **31** abstention specifications covering other quantities, unknown targets, exact magnitudes, unwarranted extinction, unsupported bridges, conflicting/negated changes, and options about the wrong organism.
- **9** supported specifications covering ordinary qualitative effects, explicit removal/extinction as premises, active/passive/feed-on syntax, trailing conditions, and requested-target agreement.
- **9** lexical-family specifications checking shared recognized canonical features.
- Full-forward finiteness, option permutation, batch order/composition, and exact identical-option ties on these additional cases.

The initial review specifications reproduced implementation errors, including nonzero trophic support for temperature changes and extinction claims. Repairs addressed those semantics directly; no scoring coefficient was changed and no accuracy-based parameter search was run. All review checks now pass.

The second review adds `/srv/work/separation_specs.py`, with **24 supported cases** and **38 abstention cases**, all passing. These check recovery versus removal, repeated material mentions, generic versus product-specific requests, mixtures and compound shields, dissolution conflicts, ambiguous goals, unsupported question types, and invalid option content. Module-level and full-forward finiteness, permutation, batch-order, and single-item consistency checks also pass. An initial positive specification abstained because `wants` was unknown; the ordinary `want` inflection family was added to the lexicon. No coefficient changes or accuracy-based search were used.

The third review adds `/srv/work/numerical_specs.py`: **24 calculation specifications** spanning all 18 formulas plus decimal, approximation, number-word, and zero cases; **20 unit-conversion specifications**; and **35 abstention specifications** for unsupported units/options, zero denominators, ambiguous operands, and unrelated final requests. Three additional varied-option checks verify arithmetic ordering and dimension rejection. Full-forward finite output, option permutation, batch reversal, and single-item consistency also pass. The initial run reproduced 12 failures before the semantic repairs; all these checks now pass. All previously listed regression suites were rerun successfully. No formula coefficient or scoring scale was changed.

The fourth review adds `/srv/work/physics_specs.py`, with **24 physical-applicability abstention cases**, **10 supported cases**, **one energy-kind disambiguation case**, and **eight comparison cases**, all passing. The initial run reproduced 24 failures, including simultaneous full support for kinetic energy and unrelated work values. Comparison testing additionally reproduced a wrong-attribute fallback and prompted the explicit attribute binder. Regression checks preserve conditional mass/volume/density comparisons and distinguish derived values from explicitly given component-force values. Full-forward finiteness, option permutation, batch reversal, and single-item consistency pass. All earlier suites and the required checker were rerun successfully, without fitting or changing any formula coefficient or scoring scale.

The 24 independently authored smoke questions in `/srv/work/my_questions.json` remain at **21 correct answers**, compared with 20 before the separation review. The water-recovery case is now correct. Three still fail: paraphrased parallel-circuit behavior, paved-surface runoff, and investigation of disagreeing results. These expose remaining retrieval/paraphrase limitations. The smoke set and diagnostic inspection were used during engineering, so **21/24 is not an independent evaluation or an ARC accuracy estimate**. There was no automated parameter sweep or accuracy-driven search.

Parent verification results are not claimed as rerun in this generation. The checks listed above are the checks actually run here. Verification inputs and scripts live outside the delivered model folder and are not model dependencies.

## Resources

A four-thread CPU verification run measured:

| Quantity | Result |
|---|---:|
| Build time | 0.993 seconds |
| Forward time for 3,530 synthetic inputs, batches of 64 | 30.616 seconds |
| Peak process resident memory during verification | 1,039.47 MiB |
| Actual tensor storage, including sparse indices | 29,010,839 bytes |
| Concepts including null | 5,051 |
| Retrieval contexts | 8,248 |
| Signed assertion entries including reverse links | 16,939 |
| Shared lexical phrase patterns | 388 |

The synthetic workload repeats only the authored smoke questions; its content is not held-out ARC data. Fixed input dimensions determine the main intermediate allocations. The measured run is comfortably below 2 GB and 20 minutes.

## Limitations

This is not a general language model. It has a finite authored knowledge base and limited grammar. It can miss paraphrases, overvalue shared keywords, mistake scope, and fail on unfamiliar narratives, diagrams, tables, long-distance references, or multi-step reasoning. Tokenization loses punctuation and numeric signs, which limits arithmetic and sentence-boundary parsing. Explicit negation and predicate oppositions are conservative heuristics, not unrestricted semantic logic.

The expanded content, sparse execution, and reasoning additions are deliberate design improvements, but only the unavailable held-out evaluation can establish whether this model exceeds the parent's reported accuracy.

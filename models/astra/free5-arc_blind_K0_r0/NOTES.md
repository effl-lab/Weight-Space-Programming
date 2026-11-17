# Authored science model — generation 5, reviewed

## Summary

This is a deterministic, hand-authored science model extending the supplied generation-4 parent. It combines the parent's sparse scientific-assertion retrieval and finite reasoning modules with **589 additional explanatory assertions**, **255 structured statement rules**, **58 lexical supplement lines**, and a new **option-local statement verifier**. A fixed two-microbatch forward pass bounds transient memory.

The supplied parent's reported validation accuracy is **45.2%**. No held-out questions, labels, or per-question results were available or accessed. These changes are intended to improve coverage and distinguish true statements from superficially similar false statements; no held-out accuracy improvement is claimed.

There is no training, fitting, pretrained component, downloaded corpus, random initialization, external service, or accuracy-based search. Knowledge, grammar, numerical formulas, and scoring constants are authored. The parent's retrieval and reasoning coefficients are unchanged. Specificity normalization uses frequencies only within the authored knowledge, not question data.

## Files and representation

- `vocab.txt`: **25,351** vocabulary entries. Line k maps to word ID k+2; padding and unknown remain IDs 0 and 1.
- `build.py`: compiles all model weights from the delivered files into a dictionary of Torch tensors. It reads no question files. `python build.py --vocab` regenerates the vocabulary deterministically.
- `arch.py`: fixed tensor inference, importing only Torch and math, with no question-dependent Python control flow or string lookup.
- `explanatory_bridges.txt`: **589** additional assertion lines covering particle explanations, conservation, phase changes, mixtures, thermal effects, waves, optics, circuits, mechanics, cells, physiology, plant experiments, inheritance, ecology, resources, surface processes, geology, weather, astronomy, inquiry, measurement, and engineering.
- `claims.txt`: **255** structured factual rules, compiled into **695** signed subject/predicate entries. These are elementary science propositions, not stored multiple-choice question records.
- `claim_grammar.txt`: literal request, exclusion, and generic-stem vocabulary for the statement verifier.
- `lexicon_bridges.txt`: **58** technical-name and lexical-variant lines, including short scientific phrases.
- The parent's assertion, lexicon, taxonomy, property, comparison, causal, opposition, entity, and grammar files remain included and used.

The main assertion format is inherited:

```
context concepts | supported completions separated by semicolons | contradicted completions
```

The last field is optional. Underscores denote phrases; `=` requests stronger reverse-definition lookup; `!` identifies an explicitly directed exception context. Identical canonical assertions are deduplicated during compilation.

The compiled model contains **5,518 concepts**, **11,410 retrieval contexts**, and **22,398 signed retrieval completion entries**, plus the separate claim tables and retained reasoning weights. Large concept-to-fact maps are sparse COO tensors, not dense allocated matrices. Signed assertions that become indistinguishable after unordered encoding are conservatively omitted, as described in the third review below.

## Retained inference

The parent's architecture remains the foundation:

1. Literal aliases, inflections, short phrase recognition, entity shielding, and bounded soft semantic neighbors yield concept features and positions.
2. Directed taxonomy propagates member-to-class evidence without treating a class as evidence for every member.
3. Sparse projections score context coverage and candidate completion coverage. Specificity, predicate opposition, distinctive answer content, and precision govern evidence.
4. Signed top-evidence aggregation handles supported and contradicted completions, scoped option negation, and negative/exception requests.
5. A weaker whole-statement backoff allows a fact's content to be distributed between stem and choice.
6. Fixed reasoning modules handle directed transformations, magnetic/electric interactions, 18 dimension-aware arithmetic formulas, applicability guards, scalar comparisons, qualitative causal laws, requested classes, experimental-variable roles, one-edge food-web effects, binary-mixture separation, and explicitly collinear force addition.

These are finite mechanisms, not unrestricted natural-language understanding or a general theorem prover.

## New option-local statement verification

### Motivation

Ordinary retrieval expects a fact's subject or context in the question. A stem such as a request for a correct statement about particles may leave both the subject and predicate inside each option. Conversely, lexical overlap alone does not distinguish a proton carrying positive charge from a proton carrying negative charge, or a density ratio with its numerator and denominator reversed.

The new module explicitly verifies curated complete statements **inside each option**, rather than treating every generally true fact as an answer to every question.

### Knowledge format

```
topic alternatives | subject | true ordered predicates ; alternatives | false ordered predicates
```

For example, the rules encode proton/positive-charge and proton/negative-charge relationships, state-of-matter properties, acid/base properties, organ functions, ecological roles, and measurement instruments. Topic terms are a disjunction; subject and predicate concepts are conjunctions. Predicate concepts retain authored order rather than being sorted into an unordered bag.

All rules are compiled into normalized sparse maps, padded ordered concept-ID tables, signs, and literal grammar tables. Negated predicates in the authored text are normalized into their underlying predicate with the corresponding truth sign. No runtime text processing occurs.

### Activation and safeguards

- The request must explicitly concern a statement, claim, description, or fact, and have one supported judgment head requesting correctness, truth, falsity, or description. A negative head and one local negation compose: `not false` requests true statements, unlike `not true`. Unsupported or ambiguous request grammar abstains.
- Explanatory, hypothetical, conditional, and selected experimental request wording disables this module. It is not a causal-explanation selector.
- Each rule needs a literal topical anchor in the question, the complete named subject in the question, or a stem made entirely of supported generic statement-request words.
- **The complete subject and complete predicate must both be present in the option.** A subject mentioned only in the question cannot fill a missing subject in the choice.
- Only literal-strength features with known positions count. Soft semantic neighbors and inferred taxonomy cannot establish a required word.
- The subject must precede the predicate through a supported literal bridge. Arbitrary intervening verbs cannot turn co-occurrence into identity or possession. The bridge permits components of its own multiword subject, not unrelated scientific words. Predicate concepts must appear in authored order, and the full matched span is bounded to 20 tokens. This distinguishes ordered color changes and ratios instead of equating reversed endpoints.
- Predicate gaps may contain their own literal concepts, phrase components, and a small grammatical filler list. Arbitrary verbs and unlisted directional prepositions cannot be silently ignored inside a matched predicate.
- The option must contain no unknown tokens, be at most 24 tokens long, and avoid authored conditional, explanatory, modal, and quantifier exclusions.
- The matched proposition must account for at least 60% of the option's weighted literal concept content. Extra assertions cannot freely borrow the truth of a small matching fragment.
- Recognized phrase components are credited in the precision numerator when the phrase belongs to the proposition. A technical name such as `graduated cylinder` is not incorrectly penalized for appearing as both a phrase concept and component words.
- `above`, `below`, `in`, `from`, and `by` retain literal identity and position in this module, despite being stop words in inherited retrieval. This prevents pH inequalities from collapsing and preserves explicitly authored prepositional roles.
- A single option negator may reverse a claim only when it precedes the complete predicate in its clause. Internal, trailing, other-clause, or multiple negators cause that claim to abstain. A negated predicate must not cross a recognized clause boundary. A supported negative judgment request reverses the resulting evidence according to its composed polarity. Conflicting supported and contradicted matches offset each other.

The strongest positive and negative claim evidence contribute at a manually specified scale of **1.35**, the same scale used by some retained symbolic modules. No coefficient was searched or fitted.

### Limitations

This module deliberately abstains on many valid paraphrases. For example, reversing the grammatical order from “current stops” to “stops current” can fail an ordered predicate match. It does not implement arbitrary passive syntax, nested negation, quantifier logic, pronoun resolution, repeated-entity binding, or general conjunctive entailment. A topical anchor is only a coarse relevance test. Subordinate or multi-assertion sentences can remain ambiguous even with the exclusions and precision gate. Other model components still answer when this module contributes zero.

## Review: assertion binding and negation

The review found two structural errors in the new statement verifier:

1. Its option-wide negation flag could reverse a matched proposition because of a negator inside an unsupported construction or after the predicate, rather than a negator governing the assertion.
2. Subject and predicate order alone allowed an intervening verb to change the relationship unnoticed: “plastic touches an electrical conductor” could be scored as if it asserted “plastic is an electrical conductor.”

`claim_bindings()` now uses a literal bridge table and per-subject lexical-component tables to validate the link. Prefix sums test the bridge without Python branching. Clause-local prefix negator counts permit a single supported negation and withhold evidence for other placements. This is intentionally conservative: even an understandable “ribosomes make no proteins” abstains because its negator is internal to the authored predicate, whereas “ribosomes do not make proteins” is supported. A second unsupported clause is not allowed to flip the first clause's claim.

The review also corrected the solute proposition to **dissolves in solvent**, preserving `in` explicitly rather than equating it with the solvent's **dissolves solute** relation. The other authored `from` and `by` occurrences are likewise retained. The main retrieval stop list, assertion sources, inherited reasoning coefficients, claim scale, and precision threshold were not retuned.

This remains a finite grammar. It does not provide general active/passive voice interpretation, prove all clauses of a compound statement, or assign full semantic roles to every unlisted preposition. Supported bridge words are deliberately limited; many valid paraphrases abstain.

## Second review: judgment heads and predicate gaps

### Judgment polarity is composed and shared

Previously, `inaccurate` was accepted as a judgment word but inherited question polarity did not mark it negative. The inherited any-negative-word rule also equated `not false` with `not true`, and could mistake an incidental negative adjective in a topic for the requested truth value.

`claim_request_polarity()` now binds one literal judgment head through a bounded copular, descriptive, or directly attributive construction. Positive and negative heads have explicit signs. One attached negator, negative descriptive adverb, `least`, or exception operator composes with that sign. For example, `is inaccurate` and `is not accurate` request false statements; `is not inaccurate` requests true statements. A supported topic adjective before the head does not automatically reverse the request. Multiple or trailing judgments, unbound/double operators, and explicitly unsupported operators abstain.

The same valid result is shared through `question_scope()` with retrieval and the retained reasoning modules. This prevents the new claim module from correctly interpreting a request while the rest of the model uses the opposite polarity. The override requires an explicit factual-request noun, a valid bound head, and absence of the existing explanatory/hypothetical/experimental exclusions. Unsupported requests retain the previous general question polarity; the claim verifier itself abstains on unsupported judgment grammar. This is not a general logical parser.

### Predicate content must form the authored relation

Subject–predicate bridging alone was insufficient: a predicate matcher could ignore a preposition inside its own span and equate `solvent dissolves solute` with `solvent is dissolved in solute`. It could also splice predicate words around an unrelated lexical verb.

The compiler now builds a grouped lexical table for predicate concepts and their phrase components. Tensor prefix sums verify every gap against that table and a small literal grammatical-filler list. Authored relational words such as `in`, `from`, and `by` must be present where the predicate specifies them, and cannot be freely inserted into another relation. Thus the passive solvent example abstains rather than being treated as the active dissolving fact. This is conservative literal validation, not general passive-voice interpretation.

Six manually authored, explicit-preposition variants preserve conventional descriptions of density division, particles fixed in position, and blood moving away from the heart. These expand the 255 rule lines from 689 to **695 signed entries**. No retrieval coefficient, claim scale, precision threshold, or scientific numerical parameter was fitted or searched.

## Third review: collisions in unordered retrieval

A structural audit of the authored sources found **43 groups** with opposite effective signs but identical encoded context and completion concept sets. These usually represent sound science whose essential relation is lost by unordered encoding: numerator versus denominator, phase-transition endpoints, particle counts attached to different elements, pH inequalities whose prepositions are stop words, or which variable is held constant. They are not necessarily contradictions in the original scientific sentences. The inherited positive/negative aggregation is asymmetric, so retaining both signs does not reliably make such evidence neutral.

`build.py` now uses `retrieval_records()` to inspect all explicit assertions before generating associations. It groups by exactly the compiler's canonical context set, completion set, and directed-exception flag, and composes the authored assertion sign with completion negation. If a group has both effective signs, neither assertion nor a reverse association derived from it is emitted. Confidence cannot select a winner. Noncolliding completions in the same source line remain usable; slash-separated contexts are checked individually. This is a deterministic representation-integrity rule, not question scoring, training, or an accuracy-driven content search.

A second check prevents a generated reverse association from contradicting explicit evidence or entering an explicitly ambiguous group. It also omits mutually opposed reverse-only associations. This removed **four** inherited predator–prey inverse associations: an effect-to-cause association must not override an explicitly authored forward ecological response. Compatible weak and strong reverse associations retain their original confidence factors. Separate numerical, qualitative, directional, and ordered-claim modules are unchanged.

The final sparse retrieval tensors have **zero opposite-sign groups with identical encoded keys**. This does not establish general scientific consistency. The filter does not recover erased order, detect every partial-overlap contradiction, or distinguish reversed wording when only one direction was authored. It deliberately sacrifices ambiguous retrieval coverage rather than inventing a relation parser. Authored source text remains intact, and ordinary authored-corpus specificity normalization is recomputed over the retained contexts without fitting any coefficients.

## Fourth review: explicit decimal phrases

The inherited numeric parser treated the fractional tail of an explicit word decimal as a standalone measurement. For example, `one point five grams` supplied 5 grams, and `twenty five point two meters` supplied 2 meters. Unsupported constructions such as `one point twenty five grams` could likewise leak a trailing cardinal into arithmetic.

`explicit_decimals()` now composes a supported whole cardinal, the literal token `point`, and a fractional part of up to three decimal places. Fractional digits may be individual digit words or literal digit tokens, including leading-zero groups such as `05`. It preserves zero values, scales units normally, and exposes the whole phrase as one number span for both measurements and bare numerical options. The implementation uses tensor operations and a fixed three-step loop. The only new weight is the existing vocabulary ID of `point`; no vocabulary, scientific constants, coefficients, or thresholds were fitted.

The parser identifies contiguous numeric/point runs, including internal `and` connectors, and requires the supported decimal to cover its entire run. Unsupported precision, repeated points, fractional cardinals such as `twenty five`, missing integer parts, and malformed adjacent-number constructions invalidate the run rather than leaving a trailing fragment. Ordinary separated uses such as `the point is at five meters` retain their measurement. Existing punctuation-derived decimals, cardinal composition, dimensional guards, and zero-denominator guards remain in place.

This is a bounded literal grammar, not general numerical language understanding. It deliberately abstains on `point five` without a whole part and on more than three fractional places. Tokenization still destroys punctuation and signs, so some distinct original numeric expressions are unavoidably indistinguishable. Arithmetic applicability and object-binding limitations are not removed.

## Fifth review: implied units and fixed-variable comparisons

The comparison module previously offered each bare numerical option as a value of every supported physical quantity. A rule comparing volume at fixed mass could therefore consume the options as densities and reverse their order, even though the requested values were volumes. `comparison_attributes()` now shares the existing nearby attribute binding between rule selection and implied-unit selection. Four bare numeric choices inherit only one requested physical property. If no attribute is bound after a comparative, one unambiguous literal property in the request may establish the implied quantity; multiple unbound properties do not. This preserves requests such as `Which mass is greatest?` without treating a mass, volume, speed, and density as interchangeable numbers. Explicit units, independently derived values, and named reference properties retain their separate handling.

The fixed-variable comparison contexts also lost their roles: `same volume mass` and `same mass volume` encoded to the same unordered concepts, allowing opposite rules to cancel. Three authored phrase concepts now preserve the conditions in `comparisons.txt`: `same_volume`, `same_mass`, and `same_heat`. Thus mass at fixed volume ranks with density, volume at fixed mass ranks inversely with density, and temperature change at fixed mass and heat ranks inversely with specific heat. Existing lexical aliases also recognize adjacent `equal volume` and `identical mass`. The ordinary phrase compiler handles these concepts; no vocabulary entries, scientific reference values, or scoring coefficients were fitted or added.

These are deliberately literal conditions, not a general algebraic constraint parser. A distant, reversed, negated, or elaborately paraphrased constant-quantity condition is not fully understood. The nearby-attribute binder remains bounded to four following tokens, and ambiguous implied units abstain. The changes do not establish arbitrary object identity or validate every descriptive clause.

## Sixth review: empty batches and bounded-memory inference

The public input contract permits an `(n, 128)` question tensor and `(n, 4, 48)` option tensor, but the previous wrapper failed for `n=0` with an ambiguous zero-element reshape inside inference. The actual evaluation loop supplies nonempty batches; this repair improves the public function's robustness rather than claiming an assessment-score gain.

`forward()` still calls the full inference program on two shape-determined microbatches. It now appends one all-padding sentinel input, sets the microbatch width to `(n + 2) // 2`, and repeats only the sentinel when necessary. Both calls to the internal core therefore have nonempty inputs, including when `n=0`. Slicing the concatenated output back to `n` removes every sentinel output and returns a floating `(0, 4)` tensor for an empty input. No Python conditional, data-dependent loop, learned prior, or weight change is introduced.

The split depends only on batch shape, never question content, option position, or model confidence. All four options remain together, and no reduction combines different examples. The wrapper evaluates one or two extra padding rows per call; for the runner's outer batch size of 64, each microbatch contains 33 rows. This bounds dense transient fact-evidence memory for the supplied runner. The internal `forward_core()` helper still expects a nonempty batch; the public wrapper enforces that precondition without changing nonempty predictions within floating-point tolerance.

## Verification actually performed

The required command passes on the final model:

```
python /srv/work/tools/check_model.py
CHECK OK
weights 215,136,629 (366,107 non-zero), vocabulary 25,351 words
```

The logical weight count includes implicit zeros in sparse matrices; it is not allocated storage.

`/srv/work/verify_model.py`, outside the deliverable and not a build dependency, verifies:

- **72 signed semantic specifications** for the new module, covering particle charges, matter states, pH inequalities, ordered litmus color changes, sound, conservation, question/option negation, generic requests, organs, cells, ecology, astronomy, measurement, circuits, direct-subject relevance, and descriptive requests.
- **24 abstention specifications** for unsupported requests, unrelated topics, reversed subject/predicate roles, unknown words, extra explanations, and selected quantifiers.
- Finite `(n, 4)` output, option-permutation equivariance, batch reversal, singleton/batch consistency, direct-core/microbatch equivalence, exact identical-option ties, and zero outputs for wholly padded or unknown inputs.
- **Three retained numerical specifications**: density, speed, and collinear net force.
- A **3,530-input synthetic workload** on four CPU threads, using repetitions of independently authored specification inputs.

One initial run exposed a real precision-accounting bug: a multiword technical subject was charged for both its phrase and its component concepts, but only the phrase was credited. The compiler now credits the matched phrase components consistently. A separate retained-force fixture used the word `another`, which the parent's conservative force grammar explicitly excludes. That fixture was rewritten to the supported single-object grammar; the inherited exclusion was not removed. All final specifications pass.

The reviewed model reran all of those checks successfully. In addition, `/srv/work/claim_review_specs.py` verifies **56** statement-binding specifications. The fixtures have no unknown option words, so abstention checks exercise the new grammar rather than the unknown-word gate. The suite covers supported affirmative/negative assertions, unsupported intervening verbs, internal/trailing/double/other-clause negation, explicitly authored prepositions, multiword subjects, and negative question requests. Under identical current weights, the preserved pre-review inference disagrees with **17** of these specifications; the reviewed inference passes all 56. Full-model differences equal the separately computed claim-contribution differences within floating-point tolerance. Option permutation, batch reversal, singleton consistency, and identical-option ties also pass on these new inputs.

The second review reran both previous suites successfully and added `/srv/work/claim_review2_specs.py`:

- **48 request-polarity specifications**, including negative judgment words, `not true` versus `not false`, contractions, descriptive and attributive heads, topic adjectives, exceptions, ambiguity, and unsupported operators.
- **224 signed claim specifications** across 56 four-option inputs. This total includes **32 predicate-gap specifications** for prepositions, unrelated intervening verbs, grammatical fillers, and phrase components.
- Direct verification that supported request polarity is shared with general `question_scope()`, while unsupported request forms preserve its previous polarity.
- Full-forward option permutation, batch reversal, singleton consistency, and exact identical-option ties.
- Contribution isolation under identical current weights. Where general request polarity changes intentionally, the preserved old architecture is also evaluated with the new scope function to isolate the separate claim-module change. Where request polarity is unchanged, the direct old/new difference equals the claim difference.

The preserved pre-second-review claim inference disagrees with **118** of the signed specifications under the same current weights; the reviewed implementation passes all of them. These are templated grammar and semantic checks, not independent science-question accuracy observations. An initial new fixture used the topic `energy` for heating/cooling rules whose declared topic list is `matter particles temperature`; those two checks therefore abstained. The fixture stem was corrected to a declared topic rather than changing the model or thresholds to fit the fixture.

The third review reran all three previous suites and the required checker successfully. `/srv/work/retrieval_review_specs.py` additionally verifies **19 compiler specifications** and source-order invariance of their emitted record multisets. They cover ordered-ratio and context-binding collisions, erased inequalities, effective negation signs, confidence independence, directed exceptions, slash variants, compatible reverse factors, unsupported causal inverses, and reverse-only conflicts. Direct inspection of the delivered sparse tensors verifies zero opposite-sign groups, **five** discarded-collision/reverse-absence checks, and **four** preserved explicit ecological directions. `/srv/work/audit_science.py` reports the original 43 source-level collisions. Both scripts and their results are outside the deliverable and are not build dependencies. All new specifications passed on their first execution.

The fourth review reran all previous suites and the required checker successfully. `/srv/work/numeric_review_specs.py` adds **40 measurement specifications**, **four bare-number/span specifications**, and **six arithmetic specifications**, including malformed-decimal abstention, zero denominators, zero numerators, unit conversion, and compound units. It also checks finite full-model logits, option permutation, batch reversal, every case as a singleton, core/microbatch equivalence, and identical-option ties. The preserved prior parser disagrees with **29** measurement specifications under identical current weights; this is a grammar-correctness comparison, not question accuracy. The first run passed all 40 measurement checks but exposed a fixture-count mistake: `twenty five point two` has four tokens, not five. Correcting that expected count required no model change. The final full sequence passes. This script, its result JSON, and `/srv/work/numeric_review_before.py` are outside the deliverable and are not build dependencies.

The fifth review reran all previous suites and the required checker successfully. `/srv/work/comparison_review_specs.py` adds **31 comparison specifications**, covering bare-number unit binding, attributes preceding comparative cues, ambiguous requests, mixed bare/explicit options, incompatible units, unit conversion, equal-valued choices, derived density, fixed-variable material comparisons, and specific-heat conditions. Every specified pairwise order or abstention passes. It checks finite full-model logits, option permutation, batch reversal, representative singletons, core/microbatch equivalence, and exact identical-option ties. With identical current weights, the preserved pre-review inference disagrees with **two** bare-number specifications; full-model differences equal the isolated comparison-contribution differences within tolerance. That controlled comparison tests the inference change, not the old versus new authored phrase contexts. All new specifications passed on their first execution. The script, result JSON, and `/srv/work/comparison_review_before.py` are outside the deliverable and are not build dependencies.

The sixth review reran every previous suite and the required checker successfully. `/srv/work/batch_review_specs.py` verifies **11 batch sizes** (`0, 1, 2, 3, 7, 8, 31, 32, 63, 64, 65`) against the preserved prior wrapper on eight authored or padding/unknown inputs. Empty output shape, floating dtype, and device are checked explicitly. Nonempty outputs match the prior inference within tolerance. Additional checks cover genuinely noncontiguous inputs, partitions containing interleaved empty batches, option permutation, batch reversal, core equivalence on nonempty inputs, identical-option ties, and zero output for padded or wholly unknown inputs. Input tensors and their underlying noncontiguous storage remain unchanged. A fresh build matches every dense weight tensor and every sparse tensor's indices and values after repeated inference, checking both deterministic compilation and absence of weight mutation. All new specifications passed on their first execution after the empty-input failure was reproduced. The script, its result JSON, and `/srv/work/batch_review_before.py` are outside the deliverable and are not build dependencies.

These are semantic, arithmetic, invariance, and execution checks, not ARC accuracy measurements. The parent's historical regression suites were not supplied and are not claimed as rerun. The required checker used its default placeholder input; the substantive runtime checks are in the independent verification script.

## Measured resources

Four-thread CPU execution, outer batch size 64:

| Quantity | Measurement |
|---|---:|
| Build time | 0.91 seconds |
| Forward time, 3,530 synthetic inputs | 50.48 seconds |
| Peak process resident memory | 932.76 MiB |
| Actual tensor storage, including sparse indices | 42,352,690 bytes |
| Vocabulary entries | 25,351 |
| Concepts | 5,518 |
| Retrieval contexts | 11,410 |
| Retrieval completion entries | 22,398 |
| Separate signed claim entries | 695 |

The measured workload is below the 2 GB memory and 20 minute runtime limits. Its questions are authored specifications, not held-out ARC questions.

## Remaining model limitations

Coverage is finite and paraphrase-sensitive. Strong associative overlap can still outrank the intended explanation. Sentence scope, arbitrary object identity, conditionals, negation, quantification, diagrams, tables, multi-hop reasoning, unfamiliar vocabulary, and elaborate numerical word problems are only partially handled. Tokenization erases punctuation and signs. Added knowledge and bounded statement verification do not eliminate those limitations; only the unavailable held-out evaluation can establish a score improvement over the supplied parent.

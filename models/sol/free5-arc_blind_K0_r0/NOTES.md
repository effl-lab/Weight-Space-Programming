# Authored Science Memory with Guarded Candidate Reasoning — generation 5

## Provenance and purpose

This is an **untrained, manually authored** model for elementary multiple-choice science. It extends the supplied generation-4 model rather than replacing its broad science memory. All scientific statements, lexical declarations, numerical laws, thresholds, and parser rules originate in the supplied parent or the files written for this model. There are no pretrained weights, downloaded resources, external corpora, ARC datasets, optimizers, parameter fitting, or accuracy-driven model searches.

The parent was reported to achieve **45.2% validation accuracy**. This model's validation and test accuracy are **unknown**. The verification results below establish implementation properties, not an accuracy improvement.

The main additions target three weaknesses of bag-of-words retrieval:

1. A named answer must satisfy the requested **properties**, including implicit conjunctions and ordinary single-property classification.
2. Comparing **candidate conditions** requires composing their effects without assigning unsupported numerical magnitudes.
3. Numerical measurements **inside the choices** need to be evaluated as candidate situations, not mistaken for proposed scalar answers.

## Final representation

The compiled model contains:

- **9,278 retrieval records**, including taxonomic closure and **547 newly authored explanatory records**;
- **486 conjunctive rules**, including **121 new conditional deductions**;
- the parent's **277 subject-aware assertion schemas**;
- **609 relational factual slots**, including **197 new slots**;
- **536 entity property cards** over **77 signed attributes**;
- **95 comparison axes** across **40 scientific contexts**;
- **25 numerical comparison templates**: 12 two-measurement laws and 13 direct dimensioned-measurement comparisons, including affine temperature conversion;
- the parent's food-web parser, proportional/qualitative laws, ordered comparisons, experimental-role parser, and guarded numerical solvers;
- a separate guarded absolute kinetic-energy solver using mass and speed from the question;
- **58,296 vocabulary entries**, with the specified padding and unknown IDs;
- **158,272,427 logical tensor elements**, with **373,196 nonzero elements**;
- approximately **168.7 MiB of actual tensor storage**, including sparse indices.

`vocab.txt` is generated deterministically by `prepare.py` from the authored material and lexical declarations. The inherited number vocabulary, inflection generation, exact concept phrases, and unknown-word phrase barriers are retained. Vocabulary construction and IDF weighting are compilation, not fitting.

## 1. Retained parent model

The parent provides broad biology, chemistry, physics, geology, astronomy, weather, ecology, health, technology, and scientific-practice coverage. Its fixed tensor execution combines:

- directional clue/answer retrieval, IDF features, question focus, opposite-property penalties, option-common-word discounting, and top-three pooling;
- taxonomic membership and inherited factual properties with exceptions;
- conjunctive rules and subject-aware assertion schemas;
- bounded option-negation scopes and scoped request polarity;
- relation-specific slots that distinguish, for example, producing oxygen from consuming oxygen;
- question-local food-web parsing and bounded signed causal propagation;
- exact-factor and qualitative physical laws, with conservative ambiguity guards;
- ordered factual comparisons;
- dimensional arithmetic, atomic counts, displacement, and rectangular/cubic geometry.

These mechanisms are retained. Experimental-role frames additionally accept explicit `whether X affects Y` and `if X affects Y` constructions, and observing a response is recognized alongside measuring, recording, and counting it. This does not infer causal direction from an unoriented `relationship between` phrase.

## 2. Sparse science retrieval

The parent's two large binary retrieval matrices are now **coalesced sparse COO tensors**. They are compiled directly from sets of active feature IDs, without constructing their full dense forms during the build. Document frequencies, feature masses, and opposite-feature tables are derived from exactly the same authored sets.

The forward pass uses `torch.sparse.mm` for question and option retrieval. This changes storage and computation, not the retrieval formula. Integer-valued feature-product checks verify equality with the corresponding dense matrices. The smaller storage leaves substantial room below the memory budget despite the larger knowledge base.

## 3. Expanded signed entity properties

`trait_queries_g5.txt` and `trait_facts_g5.txt` extend the inherited property declarations. The combined table covers organism classes, anatomy, reproduction, plant groups, materials, energy resources, chemistry, geological rock classes, phase changes, wave types, and selected plant structures.

Values remain explicitly **true (+1), false (-1), or unknown (0)**. Missing facts do not become false. The nearest explicit taxonomic ancestor wins; direct declarations override ancestors, and equal-priority contradictions fail compilation. Scientific exceptions inherited from the parent, including egg-laying mammals, conductive graphite, nonconductive diamond, liquid mercury, and wall-less mycoplasma, remain available. New cards explicitly distinguish ordinary salts, sugar, distilled water, acids, and bases.

The parent activated its property channel only for conjunctions of at least two properties with an explicit conjunction word. This model retains that branch and adds a tightly guarded **closed-candidate branch**:

- all four options must be recognized, pure entity names;
- at least three choices must have a known value for every requested property;
- there must be at least one complete supported choice and at least one known contradiction;
- an unknown fact cannot satisfy a requirement;
- conflicting requests and relevant unsupported contexts cause abstention.

The branch supports single-property classification, implicit conjunctions such as a mammal that lays eggs, and coupled labels such as compound / not element / not mixture. A name embedded inside an appended assertion is not treated as a pure entity choice. Alias-normalized names are accepted only through the inherited lexical representation.

A single direct request negator, including a supported contraction such as `isn't`, uses the shared output polarity. Background, embedded, repeated, and exceptional negation remain guarded. Thus a supported property is inverted once for a direct negative request, not once in the property table and again at the output.

The review also narrows uncertain scientific declarations: broad reptile, amphibian, insect, arachnid, and crustacean groups do not imply universal egg laying; named common cases carry explicit facts instead. Clean water does not imply chemical purity. Photosynthesis absorbing light does not establish a heat-flow sign. These cases are explicitly unknown in the property table, rather than asserted as universal truths.

## 4. Partial-order comparison of candidate conditions

`comparison_topics.txt` declares scientific contexts and excluded contexts. `comparison_axes.txt` declares independent qualitative factors and the settings that increase or decrease an outcome. For example:

- evaporation: temperature, humidity, moving air, exposed surface;
- reaction rate: temperature, concentration, surface area, catalyst;
- wire resistance: length, thickness, temperature;
- kinetic energy: mass and speed;
- current: voltage and resistance;
- erosion, runoff, infiltration, transpiration, heat transfer, friction, and other elementary processes.

`comparisons.py` compiles these declarations into feature-group, incidence, and parser-control tensors. The tensor forward pass:

1. Identifies the scientific context and an explicit maximizing or minimizing request.
2. Checks each candidate's signed settings on each relevant axis.
3. Uses an axis only when all four candidates have an unambiguous setting on it.
4. Abstains if a relevant axis is partially specified or contradictory across the candidates.
5. Computes pairwise **Pareto dominance** across the eligible axes.
6. Adds evidence only for a candidate that strictly dominates every other candidate.

It does **not** add arbitrary temperature, humidity, or stirring magnitudes. Warm/humid and cold/dry conditions are incomparable without further information. By contrast, warm/dry conditions dominate both when those are the fully stated factors. Multiple applicable contexts must agree on the winning option. Unsupported negation and specified ambiguous conditions cause abstention. Context exclusions apply to both the question and the choices; for example, perpendicular force or circular motion prevents use of the simple work comparison. Qualitative gravitational potential energy, like its numerical counterpart, abstains in explicit differing-planet or gravitational-field contexts rather than silently assuming common gravity. Relativistic/photon exclusions also guard the appropriate massive-body comparisons.

`most likely` and `least likely` are not themselves mistaken for a physical maximizing/minimizing request. Comparison polarity is combined with the parent's scoped exclusion handling. No option-position parameter is used.

## 5. Exact numerical candidate comparisons

A separate channel evaluates supported numerical descriptions in the options:

- density = mass / volume;
- speed = distance / time;
- acceleration = force / mass;
- work = force × distance;
- power = work / time;
- current = voltage / resistance;
- voltage = current × resistance;
- resistance = voltage / current;
- energy = power × time;
- momentum proportional to mass × speed;
- kinetic energy proportional to mass × speed squared;
- gravitational potential energy proportional to mass × height under common gravity.

The common factors one-half and gravitational acceleration cancel in these ordinal candidate comparisons. This channel does not compute absolute energy; a separate kinetic-energy solver is described in section 7. Law-specific exclusions are checked in both questions and choices: work excludes angle-dependent or oppositely directed force cases, energy comparisons exclude relevant relativistic contexts, and gravitational potential energy excludes differing-planet/field and spring contexts.

Thirteen additional templates directly compare single measurements of mass, volume, length, time, force, energy, power, voltage, current, resistance, frequency, speed, or temperature. The same supported dimension can answer related requests such as shortest wavelength or greatest work.

Absolute temperatures are converted to Kelvin using authored affine formulas: **K = C + 273.15**, **K = (F - 32) × 5/9 + 273.15**, and the identity for Kelvin. The reader accepts `Celsius`/`C`, `Fahrenheit`/`F`, and `kelvin`/`kelvins`/`K`, immediately after a number or through bounded `degree`/`degrees`/`of` bridges. Temperature, hottest/coldest, and warmer/cooler extreme requests can activate this template. Equivalent Celsius/Fahrenheit readings tie under the same roundoff safeguard as other numerical comparisons. A numeric reading of zero is valid, including absolute zero in Kelvin.

The affine branch applies only to static, absolute readings. Temperature differences, changes, increases, reductions, gains, losses, warming/cooling, and explicit `negative`/`minus` wording are guarded in both the question and choices. Missing or repeated units, extra numbers, unsupported scales, bounded readings, and orphan physical units cause abstention. The extension's choice mask permits temperature units only in this local dimensioned reader; the inherited arithmetic and geometric solvers continue rejecting them.

The channel reuses the inherited dimensioned-unit reader and conversion tables, with a local extension for explicit speed units. It recognizes bounded length/time syntax such as `meters per second`, `kilometers per hour`, and tokenized `m/s`, plus `mps`, `kph`, `kmh`, and `mph`. A compound speed consumes its length and denominator units rather than also becoming a distance measurement. Unassigned physical units cause abstention. The inherited arithmetic formulas are unchanged; their combined output now has the shared bounded-input guard described below.

Every candidate must supply exactly the template's one or two bound numerical measurements and satisfy its physical domain. Divisors must be strictly positive; momentum, kinetic energy, and gravitational potential energy additionally require positive mass. Other supported nonnegative inputs may be zero, including a cart's speed, displacement in a work calculation, elapsed time in an energy calculation, voltage across a positive resistance, and scalar measurements. Zero product factors are preserved rather than clamped upward. Invalid zero divisors cause abstention, not infinities.

Repeated, missing, incompatible, bounded, extra unbound numbers, unsupported squared units, and ambiguous bindings cause abstention. The question must request a single compatible template and a clear extreme. The channel compares converted magnitudes, not the original numbers in isolation.

Numerical extrema use a scale-relative roundoff bound of **eight float32 machine epsilons**. This is an arithmetic-precision safeguard, not a fitted parameter or an assumption about measurement uncertainty. It prevents equivalent units such as 90 km/h and 25 m/s from acquiring unequal evidence through rounding. All-equal candidates abstain, and candidates tied at a nontrivial extreme receive equal evidence. The tolerance scales with magnitude rather than imposing an arbitrary absolute cutoff.

This branch deliberately does not guess whether a bare number is speed, height, mass, or time, and does not parse arbitrary algebra or general compound units. Mass-unit scale factors cancel for the supported ordinal force/mass comparison. Temperature comparisons support nonnegative integer readings in the named scale; tokenized decimals and explicit negative-word readings abstain. The required tokenizer discards standalone minus signs and decimal punctuation, so a lost sign cannot be recovered from the provided IDs; this is a limitation of the input representation, not a supported signed-temperature parser.

## 6. Scoped bounds on absolute-arithmetic inputs

The exact-value solvers share an `absolute_arithmetic` wrapper that retains the inherited answer-choice safety mask and additionally checks numerical inputs in the question. A bound such as `mass under 24 grams` cannot justify the exact density computed from 24 grams. Supported bounded inputs therefore suppress absolute numerical evidence rather than guessing the bound's endpoint. Formula, nuclear, displacement, rectangular/cubic geometry, and the new kinetic-energy evidence all pass through this wrapper. Request polarity remains applied exactly once afterward.

The guard links inequality, range, and minimum/maximum cues to numbers through a short authored grammar, then checks that the number is attached to a physical unit, cycle count, or atomic-count marker. It covers prefix forms such as `less than`, `at least`, `at most`, `between ... and`, `range of ... to`, and `up to`, as well as postfix forms such as `24 grams or less` and `24 grams minimum`.

The guard is scoped rather than a global keyword ban. Unrelated counts such as `at least 3 students` do not invalidate exact mass and volume readings. `Most likely` does not become an upper bound; standalone spatial `up` and postfix `above` do not become inequalities. A detected bounded scientific datum conservatively suppresses all absolute-arithmetic branches, even when a particular branch might not need it. This is a bounded grammar, not interval arithmetic or a general inequality parser; unsupported long-distance wording remains a limitation.

## 7. Absolute kinetic energy

A separate solver calculates **KE = (1/2) m v²** for a single nonrelativistic massive body. It uses the local dimensioned reader to bind exactly one positive mass and one nonnegative speed in the question. Because the reader stores mass in grams and speed in meters per second, the authored coefficient is **0.5 / 1000**, producing joules. Supported compound and named speed units are the same as in section 5. Zero speed correctly produces zero energy.

All four choices must be compatible scalar energy answers, each containing exactly one number and at most one explicit energy unit. The solver supports `joule`/`joules`/`J` and `kilojoule`/`kilojoules`/`kJ`. Bare-number choices default to joules, or to the unit explicitly requested by a focused `in joules` or `in kilojoules` phrase. Explicit units on choices override this default. An output-unit request is excluded from input-measurement parsing; multiple requested output units cause abstention. Equivalent J/kJ answers receive equal evidence.

Activation requires a focused kinetic-energy request, with the kinetic keyword preceding competing quantity words in the focused request. Exactly two input numbers and two bound measurements are required. Missing or extra measurements, orphan or unsupported units, squared-time syntax, zero mass, and incompatible answer units cause abstention. Explicit negation, relativistic/photon contexts, specific or combined energy, collisions, changes/differences, ratios/fractions/percentages, efficiency, and factor-change wording are excluded. The shared wrapper also rejects supported bounded scientific inputs. These conservative guards are finite lexical rules, not a general semantic parser.

This solver reuses the inherited scalar-answer gain and error formula. Its physical constant, conversions, and applicability declarations are authored; no parameters were fitted and no held-out accuracy was measured.

## 8. Additional authored science content

`generation5_science.tsv` adds concrete explanations of adaptation, plant structures, germination, interdependence, microorganisms, particle models, conservation, household physical changes, solutions, simple forces, electrical safety, sound, optics, heat transfer, weather, geological evidence, environmental decisions, and astronomy.

`slots_g5.txt` adds directional functional and structural facts about sensory organs, hormones, nervous communication, plant reproduction, measuring instruments, electrical devices, energy conversions, astronomical structures, soil horizons, topographic evidence, and conservation methods.

`logic_g5.txt` adds guarded deductions about the effects of removing roots or leaves, limiting resources, conservation during phase changes and reactions, familiar optical situations, electrical and thermal examples, erosion, runoff, groundwater, and selected Earth/space observations. Rule evidence remains maximum-pooled rather than accumulating duplicated statements as independent votes.

## Fixed tensor execution and compliance

`build()` returns only tensors and reads only local authored declarations. Its operations are deterministic lexical compilation, feature construction, taxonomic closure, and mathematical transformations of the declarations.

`arch.py` imports only `torch` and `math`. It contains no input-dependent Python branching, token decoding, file reads, dynamic execution, question lookup dictionaries, or pretrained components. Input-dependent work is tensor computation. Fixed-range loops are inherited for bounded phrase and parser operations. The required architecture checker passes.

The model is option-permutation equivariant, independent of unrelated batch members, and gives equal logits to identical choices. All model files are self-contained in this folder; verification scripts outside it are not needed by `build()` or `forward()`.

## Verification and resources

The required check passes:

```
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
CHECK OK
```

`/srv/work/verify_generation5.py` uses only newly invented inputs and authored declaration strings. It checks:

- property-table semantics, including single attributes, conjunctions, implicit conjunctions, coupled classifications, and explicit exceptions;
- compositional comparisons and conservative abstention on incomparable or partially specified conditions;
- **39 numerical semantic cases**, including mixed-unit momentum and kinetic energy, gravitational potential energy, compound speeds, direct scalar comparisons, valid zero inputs, and **12 absolute-temperature comparison cases**;
- **18 temperature abstention cases**, Celsius/Fahrenheit full and partial ties, temperature unit invariance, and unchanged inherited temperature-unit rejection;
- option-local applicability exclusions, missing/repeated/extra measurements, orphan or unsupported units, zero-divisor rejection, and equal converted-value ties (including cubic units, all-zero inputs, and partially tied extrema);
- **22 absolute-arithmetic semantic cases** covering exact formulas, atomic counts, cycle frequency, displacement, geometry, bound-scope regressions, and **10 absolute kinetic-energy cases**;
- **20 kinetic-energy applicability/abstention cases**, equivalent mass/speed inputs, and equivalent joule/kilojoule output ties;
- **13 bounded-input abstention cases** and **6 bound-scope checks**, including prefix/postfix inequalities, ranges, unrelated counts, and spatial wording;
- **14 property**, **9 qualitative comparison**, and **4 relational slot** semantic cases, plus experimental-role frames;
- direct-request versus background property-negation scope, including contractions and repeated negators;
- compiler/runtime feature equality on **27,423 authored strings**;
- dense/sparse retrieval-product equivalence;
- finite full outputs, option permutation, batch independence, and identical-choice ties;
- a **3,584-input** complete-forward workload using repeated invented cases.

The latest four-thread verification run built the weights in **4.394 seconds**, completed the synthetic workload in **22.169 seconds**, and reached **926.7 MiB peak RSS**, including the dense-equivalence checks. The required checker subsequently reported `CHECK OK`, with 158,272,427 logical tensor elements, 373,196 nonzero elements, and 58,296 vocabulary words. These figures are well below the 2 GB / 20 minute limit. The synthetic workload is a resource and execution check, not a held-out accuracy evaluation.

## Limitations

This remains a finite authored reasoner. Retrieval and conjunctive bags can confuse clause roles or causal direction. Unknown entities, unfamiliar phrasing, diagrams, tables, long-distance syntax, and multi-step stories remain difficult. The new comparisons cover only declared axes and require a universal dominance winner rather than guessing magnitudes for competing effects. Exact candidate calculations require a small set of cleanly bound dimensions. Property cards describe ordinary school-level contexts with explicit unknowns, not every biological exception or altered physical condition. The model intentionally abstains in many unsupported cases; the inherited channels supply the remaining evidence.

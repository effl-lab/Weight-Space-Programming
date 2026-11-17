# Generation 5: duplicate-resistant contextual science memory

This is a hand-authored second-order association model for grade-school science. It reuses the supplied generation-4 knowledge base, adds six contextual knowledge files and explicit lexical forms, and changes how repeated projections are combined. `arch.py` is unchanged.

## Knowledge

New files:

- `causal_physics.txt`: particle/bulk distinctions; thermal processes; chemical diagnostics and conservation; mechanics; circuits; waves; optical and electrical technologies.
- `causal_life.txt`: cellular functions; metabolic inputs and outputs; heredity; plant structures and reproduction; organ-system interactions; ecological dependencies and adaptations.
- `causal_earth.txt`: geological observations and mechanisms; erosion and groundwater; atmospheric processes; astronomical geometry; resources and historical evidence.
- `reasoning_inquiry.txt`: experimental variables, evidence, reproducibility, measurement, models, safety, and engineering decisions.
- `decision_contexts.txt`: practical precautions, conservation choices, failure consequences, and condition/response contexts.
- `review_relations.txt`: 343 source-review additions covering measurement/control distinctions, quantity-specific mechanical trends, natural multiword terminology, chemical classes, biological signaling, fossil evidence, microscopy, vaccination, sensory functions, buoyancy, skeleton classification, explicit input/output or numerator/denominator roles, and dimensional unit notation.
- `lexical_g5.txt`: explicitly written inflections and close lexical equivalents. Scientific processes, their products, and their agents are not automatically treated as synonyms.

These are declarations of the form `strength | unordered question cues | supported answer concepts | opposed answer concepts`. Most added antecedents have two concepts so they can be retrieved directly by the fixed architecture. Some longer antecedents describe qualified processes. Positive/negative fields are concept bags, not executable English clauses.

The first five files add **1,160 authored records**. `review_relations.txt` adds **343**. The flat knowledge base now has **8,832** records, including two additional records created by splitting unbound atom-count statements into element-specific claims.

The inherited ontology mechanism is retained, with reviewed class profiles: only explicitly authored is-a edges compose transitively. Ordinary associations do not become entailments. A few inherited qualifier-scope errors were corrected during source checks: cutting a magnet does not remove its poles; erosion prevention does not oppose plants; a nighttime land breeze does not oppose all instances of faster cooling. The first verification run also exposed an inherited lexical conflation of **pollination** (a process) with **pollinator** (an agent). These now have distinct concepts; the corrected verification passed.

## Follow-up review

A source-only review repaired additional claims whose English qualifier scope did not survive conversion to signed concept bags. Examples:

- Precision need not imply accuracy, but it does not oppose true values or accuracy.
- Proteins can supply energy; phloem transports water along with sugars; germinating seeds exchange gases. A primary role does not imply the absence of every secondary role.
- A magnet broken in half still has poles. A cast fossil can contain an impression; it is not an empty mold.
- Controlled experiments can change their independent variable. “Avoid changing all variables” must not become a blanket opposition to change or mass.
- Thermal equilibrium implies equal temperatures and zero net transfer, not absent thermal energy. Boiling plateaus retain constant temperature as positive context.
- Friction supports walking, and losses of useful energy do not destroy energy.

The review also separates the **alcohol class** from its example **ethanol**, with explicit associations for ethanol, methanol, and isopropanol. `No` joins the inherited query-negation concept; quantitative `least` remains distinct. Explicit multiword contexts distinguish endothermic reactions from endothermic animals, and recognize spellings such as `x rays` and `warm blooded` under the task's tokenizer.

New source regressions check these distinctions and the absence of unsupported blanket negatives. The initial source audit found two further inherited scope conflicts (experimental surface changes and fossil impressions); both were corrected before the successful verification run. No question-answer predictions or accuracy measurements were used.

The compiler now releases source objects and exact-pair dictionaries before allocating the large tables, and evaluates the same float32 bounded-unary formula in place. This reduces temporary memory use without changing the architecture, coefficients, table dimensions, or mathematical inference rule.

## Representation and class-profile review

A further source-only review separated six inherited lexical conflations:

- `microscopic` (a size/observability property) from `microscope` (an instrument).
- `vaccination` (an intervention) from `vaccine` (the administered material).
- `float` (a state/motion) from `buoyancy` (a force).
- `hearing`, `smell`, and `taste` from `ear`, `nose`, and `tongue` respectively.

These distinctions are especially important because the fixed architecture removes option concepts already present in the question. Genuine inflections remain aliases; the related but distinct concepts are connected through authored science associations instead. The review adds 53 contextual records without changing inference or numeric coefficients.

Vertebrates inherit backbone and skeleton, but no longer universally inherit bone. Explicit cartilaginous-fish and bony-fish classes preserve the distinction for sharks versus salmon and other named examples. Corresponding broad flat records were corrected. Microorganisms inherit `microscopic`, not the instrument `microscope`.

Two buoyancy records were repaired so that a negative qualified phrase does not cancel positive buoyancy evidence. An unusable sinking-force comparison was replaced with terminal-speed context: a sinking object can have balanced weight, buoyancy, and drag and therefore zero acceleration.

Additional regressions check class profiles, signed evidence, and the six lexical distinctions. Feature-level tests use two distinct one-token concepts in each direction, with four identical options, to verify that question-overlap removal no longer erases the related target concept. They do not score answers or select model changes by predictions. The expanded label-free verification passed before the supplied checker was rerun.

## Canceled-claim and biological-compartment review

A source-only audit found **23 declarations whose signed targets canceled completely** after canonicalization. Examples include generator conversion written as `mechanical electrical` versus `electrical mechanical`, speed formulas written as `distance divide time` versus `time divide distance`, and swapped atom-count descriptions. These pairs are indistinguishable as concept bags; formerly the records silently contributed nothing.

All 23 declarations were rewritten into representable associations. An additional **58 review records** bind the distinguishing role in the antecedent: converter input versus output, formula numerator versus denominator, element-specific atom counts, eclipse shadow caster versus receiver, and wind-pressure origin versus destination. Generic conversion records retain the associated energy forms without asserting that the reverse-order bag is their opposite. These declarations improve representation, but do not overcome the architecture's inability to distinguish answer options with identical concept bags.

The compiler now rejects a declaration when shared positive/negative targets cancel all remaining signed evidence, with an error asking for the distinguishing role in the cues. Partial shared-target cancellation remains symmetric. Verification tests both cases and checks the role-specific authored claims. The first expanded verification run caught a second, earlier `earth rotation direction` declaration supporting both west and east; it was corrected to eastward before verification passed.

Biological corrections remove blanket opposition between cytoplasm and genetic material, between prokaryotes and membranes, and between viral structure and membranes. Context-specific records cover bacterial nucleoids, eukaryotic chromosome location, and lipid envelopes of enveloped viruses. Photosynthesis retains oxygen as a product-related association, treats sunlight as an energy requirement rather than a material reactant, and no longer loses both light and chemical energy to reversed-bag cancellation. Drying-food preservation is described through dehydration and growth inhibition rather than an unbound `remove ... microbial growth` clause.

These revisions use no question-answer scoring, parameter search, or training. Inference, numeric coefficients, and `arch.py` are unchanged.

## Dimensional-notation review

The next review added **83 measurement records** and explicit lexical forms for attached-exponent unit tokens such as `m2`, `cm2`, `m3`, `cm3`, and `mm3`. These were previously absent even though the fixed tokenizer preserves each as a single word. Linear, square, and cubic units now remain separate concepts; `cc` and single-word cubic-centimeter spellings are aliases of `cm3`. Multiword `square centimeter` and `cubic centimeter` cues retrieve their corresponding notation and dimensions through authored associations.

Additional records cover SI unit dimensions, customary speed notation, common metric prefixes, equivalent volumes, and fixed conversion scale factors. Kilowatts represent power while kilowatt-hours represent energy. These are hand-authored scientific constants and relations, not learned conversions or a general arithmetic evaluator. Conversion-pair scale factors do not resolve arbitrary conversion direction or calculate results for arbitrary input values.

An inherited acceleration-unit record lost its positive meter component because `meter alone` appeared in the negative field; that negative clause was removed. Pressure-unit records no longer oppose the kilogram component of compound SI base-unit notation. The first expanded verification exposed a second such pressure record; it was corrected before verification passed.

New source and feature-level regressions check the dimensions, aliases, and surviving answer features. The tokenizer checks explicitly document that `cm3` is one token, `cm^3` becomes `cm` plus `3`, and `kg/m3` becomes `kg` plus `m3`. This does not recover slash order, bind detached exponents, or distinguish case-sensitive symbols after lowercasing. The word `micrometer` remains polysemous between a unit and an instrument; a separate screw-gauge context supplies the instrument association.

## Compilation change: idempotent projected evidence

The parent already deduplicated identical antecedents, but then **summed** their unary and pair projections. The same scientific association could therefore become strong merely because it appeared in several differently qualified descriptions. This can favor frequently mentioned answer concepts and allow a weak conditional backoff to compete with a direct fact.

The new compiler retains the strongest positive and strongest negative proof separately for each projected `(cue, target)` or `(cue pair, target)`, then subtracts their magnitudes. Thus:

- Reprinting or rephrasing a same-sign projected claim does not repeatedly amplify it.
- Opposing conditional projections can cancel; positive evidence does not win merely because its column is processed first.
- Direct one- and two-concept declarations keep the largest projection scales. More qualified facts retain the inherited attenuation.
- Distinct active question cues and cue pairs still add at inference, exactly as required by `arch.py`.

The inherited target-list normalization, pair attenuation, first-order backoff coefficient, bounded nonlinearities, property/function focus convention, negation convention, and `mix = 0.55` are retained. No numeric parameter was selected by accuracy. The memory layout remains 1,414,217 hashed rows with at most 64 slots per row. Hash collisions sum only at the final storage step. Strongest-magnitude packing is deterministic.

## Build-efficiency and integrity review

A further implementation review replaced per-occurrence scalar salience writes with a few indexed assignments over authored concept-role sets. The broad/generic class declarations and all numeric coefficients are unchanged. The order is still output-only defaults, cue defaults, broad overrides, generic overrides, then the negator override. Dormant concepts remain zero; source reordering and duplication do not change salience.

Every returned tensor was SHA-256 hashed before and after the change. **All six tensors are byte-for-byte identical**, including shapes and dtypes. Independent builds with controlled Python hash seeds 17 and 93 also produced the same six fingerprints; their hash probes differed, confirming that the requested seeds took effect. `reproducibility.json` records the fingerprints and measured resources. This review changes build efficiency and verification, not the model's knowledge, weights, inference, or expected accuracy.

The pre-change fingerprint run built in 4.69 seconds. The updated controlled-seed runs built in 4.08 and 4.16 seconds. These are individual local measurements, not a guaranteed speedup benchmark. Fingerprinting uses views rather than copying the large tensors. The stronger complete verification measured 1,770,905,600 bytes of peak RSS, still below two GB.

`verify.py` now checks every vocabulary-to-concept entry and every authored salience value against a separately enumerated reference. It tests cue/output precedence, aliases, dormant broad/generic entries, and source-order/duplicate invariance. Small cloned lookup/salience tables are deliberately corrupted to ensure these checks reject in-bounds but incorrect values; the actual model tensors are not mutated.

To obtain a fresh tensor fingerprint:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/reproducibility.py
```

For controlled hash-seed comparisons, use the following for each requested seed and compare only the `tensors` records. Python `-I` ignores `PYTHONHASHSEED`, so it is deliberately replaced here by `-sP` with an empty `PYTHONPATH`. The utility explicitly imports the user-authored model directory.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=17 PYTHONPATH= python -sP /srv/work/model/reproducibility.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=93 PYTHONPATH= python -sP /srv/work/model/reproducibility.py
```

## Reproducibility and compliance

`build()` reads only files in this directory and returns exactly the six required weight tensors. It performs no training, fitting, parameter optimization, data ingestion, network access, or evaluation-dependent branching. Vocabulary is generated solely from the authored lexicon, aliases, science records, and conservative spelling rules. Padding and unknown words map to concept zero.

No ARC questions beyond the ten examples in the task were accessed. No additional question-answer examples, measured accuracy, parent predictions, or held-out labels were used to make changes. The parent's reported 44.1% validation accuracy is context only. This model's ARC accuracy is **not measured**, and improvement is not claimed as a verified result.

Regenerate the vocabulary after source edits:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/build.py
```

Run the label-free verification and supplied checker:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python /srv/work/tools/check_model.py
```

`verify.py` checks the exact architecture fingerprint, source signs, the new projection rule, shapes/dtypes/bounds, hashing and packing, reference inference, permutation/batching invariants, and a 3,530-input synthetic workload without answer labels. Results and resource measurements are recorded in `verification.json`.

## Verified representation and resources

| Item | Value |
|---|---:|
| Vocabulary entries | 26,366 |
| Concepts, including null | 4,293 |
| Records after ontology expansion | 19,741 |
| Canonical records | 19,638 |
| Returned tensor entries | 199,480,287 |
| Returned tensor storage | 1,160,066,172 bytes |
| Build time | 3.92 seconds |
| Synthetic 3,530-input inference | 0.97 seconds |
| Complete label-free verification | 8.03 seconds |
| Peak verification RSS | 1,770,905,600 bytes |

These measurements are from the successful `verify.py` run. They are below the two-GB and twenty-minute limits. The supplied checker is also run separately; it does not measure ARC accuracy here.

## Limitations

This remains a fixed bag-of-concepts model. It cannot bind arbitrary qualifiers, interpret general arithmetic, distinguish every polysemous word, or recover word order. It removes option concepts present in the question and uses only the 16 strongest question concepts for pair retrieval. Unary exception reversal is approximate. Source checks and execution tests establish model validity and resource use, not held-out answer accuracy.

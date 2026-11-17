# Hand-authored science transformer — scientific word forms and cross-projection scoring

## Deliverables and compliance

- `arch.py` remains the supplied, unmodified architecture.
- `knowledge.tsv` contains 660 manually authored science concepts, single-word aliases, and directed associations covering elementary physics, chemistry, astronomy, Earth science, biology, ecology, anatomy, laboratory methods, and resources. The latest review left this file unchanged.
- `word_forms.tsv` lists 212 additional scientific inflections, derived forms, spelling variants, and closely related lexical forms, each mapped explicitly to an existing base word. It is hand-authored, not extracted from a corpus.
- `vocab.txt` contains 21,997 unique tokenizer-compatible entries: 2,563 base words/forms, 212 curated word forms, and 19,222 generated possessive/quotation variants. These aliases do not add new concepts or factual associations.
- `build.py` constructs all 37 required dense float32 tensors: 1,102,464 scalar weights, of which 939,818 are nonzero. It reads only `vocab.txt`, `knowledge.tsv`, and `word_forms.tsv` in its folder. Calling `build()` does not write anything. Every pre-existing word embedding, both special-token embeddings, and all 36 non-embedding tensors were preserved bit-for-bit in this review.

There is no training, fitting, pretrained model, downloaded corpus, benchmark lookup, or accuracy-driven search. The knowledge graph and design constants are authored. Hash-derived vectors form a fixed, data-independent codebook, not learned embeddings. Revisions are based on circuit properties and manually identified lexical distinctions, not selection using answer accuracy.

## Improvements made during review

### Recognize curated scientific word forms

The latest review added a finite, explicitly authored table of 212 previously unknown forms, including `photosynthesizing`, `refracting`, `enzymatic`, `mitotic`, `geologic`, and British spellings such as `fertilisation`. The table also includes selected closely related forms, such as thawing/melting and oscillation/vibration.

There is no general stemmer or automatic suffix-stripping rule. Each form points directly to an existing base embedding; existing dictionary entries always take precedence. In particular, the model does not strip prefixes in ways that would turn an opposite such as “insoluble” into “soluble.” The new `insolubility` form explicitly maps to `insoluble`.

All 20,301 previous entries, the special tokens, the concept graph, the supplied architecture, and all non-embedding tensors were verified unchanged against hashes captured before editing. All 212 new forms and their quoted/possessive variants are exact embedding copies. For every new form, replacing it with its designated source gives exactly identical logits in both question and option positions: 424 end-to-end equivalence checks. These checks verify the intended implementation, not science-answer accuracy or universal word-sense equivalence.

### Preserve known meanings across apostrophe-bearing token forms

The specified tokenizer retains ASCII apostrophes inside tokens and at their boundaries. Consequently, `earth's`, `'chlorophyll'`, and `plants'` were previously unknown even though `earth`, `chlorophyll`, and `plants` were known.

An earlier revision added deterministic, one-hop aliases for possessive forms and leading/trailing single-quotation marks, including quoted possessives. These rules also apply to the newly curated word forms. Each generated form copies all 48 embedding coordinates from its authored source, preserving semantics, salience, negation markers, and scalar priors exactly. Explicitly authored forms always take precedence; an existing entry is never replaced by a generated interpretation. Pure numerals are not expanded, avoiding confusion between word quotation and feet/minutes marks.

This is a build-time vocabulary expansion, not a modified tokenizer or additional inference-time program. Mechanical expansion may include uncommon or ungrammatical surface forms; they add no new facts. It does not parse possessive relationships or quotation scope. Previously unknown forms can now change predictions, but every previously known form retains its original representation.

The review was checked against hashes captured from our own authored model before editing. No archived weights are used by `build()`, and the preservation checks do not fit or select model parameters. In addition to checking every alias row, 22 formatting-equivalence cases produce exactly identical logits.

### Preserve contrasting word meanings

An earlier review found that some broad entries treated opposite words as interchangeable aliases. For example, “fast” and “slow” shared the speed concept, while “increase” and “decrease” both mapped to population growth. This could make choices differing only in those words indistinguishable at the input.

The dictionary now preserves distinct meanings for those terms and other contrasts, including push/pull, balanced/unbalanced, heavier/lighter, long/short, concentrated/dilute, saturated/unsaturated, convex/concave, and converging/diverging. Generic parent concepts remain available without conflating their contrasting members. That revision introduced 36 additional concepts and 94 additional base vocabulary words.

Associations were also corrected where necessary: attraction relates to pulling, repulsion to pushing, centripetal motion to inward direction, and centrifugal terminology to outward direction and inertia. Echo is represented as sound reflection, separately from resonance's frequency and amplitude associations. Convex/concave geometry is kept separate from converging/diverging behavior rather than incorrectly applying one rule to both lenses and mirrors.

A regression check verifies 23 contrasting word pairs have disjoint alias-concept memberships and distinct semantic embeddings. These are dictionary-integrity checks, not labeled questions or a measured accuracy target. The review does not add numerical comparison or causal reasoning; it removes forced lexical equivalences.

### Explicit similarity instead of attention mass

The first design measured how much attention landed on scientific words, compared with background attention. A question whose words all had identical salience therefore gave zero semantic evidence. The next revision retrieved association vectors and computed an explicit twenty-dimensional similarity using an 80-neuron GELU circuit. This fixed that degeneracy, including for single-token questions.

### Separate the projection that selects from the projection that scores

Same-projection attention can select a question word because of an accidental random-code match, then score that same match as positive evidence. An earlier circuit revision cross-connected the two retrieval heads:

- Head 0 selects using the **second** ten-dimensional projection, but retrieves and scores the **first** projection.
- Head 1 selects using the **first** projection, but retrieves and scores the **second** projection.

In an ideal independent, zero-mean noise model, the projection used to score is not biased positive merely by selection in the other projection. True associations encoded in both projections can still contribute. Both halves of the concept code, domain mixture, and association code are now normalized separately, avoiding a shared normalization factor that couples them.

This is not a promise to eliminate all collisions or improve measured ARC accuracy. Domain structure, ambiguous words, and the authored associations are not independent noise. The change addresses the specific same-projection selection mechanism and passes direct synthetic circuit checks.

### Negation token coverage

The vocabulary now includes 19 authored negative contractions, such as `isn't`, `doesn't`, `can't`, and `shouldn't`. There are 28 explicit negation/exception markers in total, including `never`. A single constant list controls both vocabulary construction and negation weights, preventing mismatches. All markers still act only when present in the question, not in an option.

## Representation

Each concept receives two separately normalized ten-dimensional code vectors, each of norm `sqrt(10)`. A weaker domain code supplies broad topical information. A second twenty-dimensional representation combines the concept with its explicitly listed associates, again normalizing each half separately. Synonyms share representations; ambiguous words average their meanings. A fixed list of broad words receives reduced salience. Unknown words have zero semantic features.

The code halves are deterministic functions of separate portions of a SHAKE-256 digest. They have no shared centering or normalization operation. No random seed or codebook is selected using accuracy.

| Coordinates | Function |
| --- | --- |
| 0–19 | Token's concept code |
| 20–39 | Associated-concept code; replaced by retrieved question context |
| 40 | Question/answer segment |
| 41–42 | Position coordinate and its square |
| 43 | Scalar evidence, including a weak authored absolute-word prior |
| 44 | Explicit negation marker |
| 45 | Salience |
| 46–47 | Opposite-sign normalization anchors |

After first-block scoring, coordinates 0 and 1 are reclaimed for pooled evidence and detected question negation.

Large opposite-sign anchors keep the layer-normalization scale nearly constant. Their average serves as a zero reference: subtracting it from a feature cancels the layer-normalization mean without consuming another coordinate. Positions 0–127 are question positions and 128–175 are answer positions. No feature identifies which of the four choices is being scored.

## First block: cross-projection retrieval and explicit similarity

Heads 0 and 1 retrieve the two association-vector halves as described above. Attention uses `dot / sqrt(10)`, a question-only positional bias, and a modest salience preference.

Heads 2 and 3 copy the current token's original association coordinates and subtract them from the residual. Their ideal positional logits are `-20 * (i - j)^2`, selecting the token itself. Position coordinates are scaled to keep this calculation well conditioned in float32. The residual association slots then contain retrieved question context instead of the answer's original associations.

The feed-forward block computes a twenty-dimensional dot product divided by twenty. For `G = GELU` and small `e`,

```
G(z) + G(-z) = sqrt(2/pi) * z^2 + O(z^4)

[G(e(a+b)) + G(-e(a+b)) - G(e(a-b)) - G(-e(a-b))]
    / [4 * sqrt(2/pi) * e^2] = a*b + O(e^2)
```

Four neurons per coordinate, with `e = 0.05`, implement the products. Four more reclaim coordinates 0 and 1 using `GELU(x) - GELU(-x) = x`.

## Second block: pooling and negation

Two heads pool evidence over answer tokens only: a content-weighted mean and a smaller soft-maximum contribution. Another head detects question-side negation/exception markers. A two-neuron GELU circuit reverses the pooled evidence when that detector is active. The final score reads pooled evidence relative to the mean of the anchors.

There is also a weak authored penalty on absolute words such as “always” and “only.” It is not an estimated answer-frequency prior.

## Limitations

This remains a lexical science-association model, not a language model or symbolic reasoner. Twenty-dimensional codes have collisions; attention retrieves the halves separately; associations are not logical implications; and averaged meanings lose word-sense distinctions. It does not reliably represent multiword entities, grammatical roles, arithmetic, causal chains, or scoped negation. Related distractors can outrank correct answers. Global negation reversal can misread a subordinate “not,” including a contracted one. Numerical circuits are approximate because of GELU's higher-order terms, float32 arithmetic, and layer normalization.

No held-out ARC accuracy or accuracy improvement is claimed. The original science smoke-test questions are not used to fit weights or select revisions. No model weights were changed after inspecting this revision's smoke-test predictions.

## Verification

The required checker returned:

```
CHECK OK
weights 1,102,464 (939,818 non-zero), vocabulary 21,997 words
```

Independent tests in `/srv/work/test_circuit.py`, `/srv/work/test_orthography.py`, and `/srv/work/test_word_forms.py` passed:

- Finite float32 tensors and logits; all 37 required tensors present.
- Exact preservation of all 20,301 previous vocabulary entries, both special-token embeddings, all 36 non-embedding tensors, `arch.py`, and `knowledge.tsv` against pre-edit hashes. The earlier 2,563-base-word preservation check also still passes.
- All 19,434 lexical aliases exactly equal their one-hop source embeddings; explicit entries take precedence.
- All curated and generated forms match the task tokenizer as single tokens.
- All 212 curated forms have valid base sources and correct quoted/possessive extensions.
- All 424 question-side and option-side word-form substitution checks produce exactly identical logits.
- Twenty-two possessive/quotation equivalence cases also produce exactly identical logits, including quoted options and negation markers.
- Separate projection normalization and exact tokenizer-compatible vocabulary generation.
- All concept references resolve; 23 contrasting word pairs retain distinct meanings and embeddings.
- The deterministic attention sweep covers every authored base entry; the remaining vocabulary entries are covered by exhaustive alias equality checks.
- Cross-projection attention compared with its direct numerical formula: maximum probability error **0.00005793**.
- Positional self-copying across that sweep: minimum self-attention probability **0.99988556**.
- GELU dot products compared with direct multiplication: maximum absolute error **0.000701** in that sweep.
- Artificial, nonlinguistic vectors exhibiting a one-projection collision produce approximately zero evidence; two-projection agreement matches the analytical score within GELU approximation tolerance.
- Single-token scores compared with direct embedding dot products.
- Padding-position invariance within numerical tolerance.
- Choice-permutation equivariance and exact ties for identical choices.
- Negation sign reversal for all **28** markers, and confirmation that answer-side markers do not trigger question negation.
- A fully occupied 64-question batch (256 sequences of 176 tokens) ran in **0.162 seconds** on four CPU threads. Peak circuit-verification-process resident memory was **962.6 MiB**, below 2 GB. The weight tensors themselves occupy approximately **4.21 MiB**.

These are numerical-correctness and resource tests, not an accuracy-search loop.

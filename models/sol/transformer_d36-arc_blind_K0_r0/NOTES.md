# Handwritten science-association transformer

This is an **untrained**, entirely hand-specified model. `arch.py` is unchanged.
`build.py` returns all 37 required float32 weight tensors, including explicit
zeros for unused connections. It reads `knowledge.tsv`, `extra_facts.tsv`,
`lexical_families.txt`, and `vocab.txt`, all inside the delivered folder.
There are no downloaded models, optimizers, fitted parameters, ARC datasets,
or accuracy-based searches.

## Knowledge

`knowledge.tsv` and `extra_facts.tsv` contain manually authored associations
for elementary biology, ecology, chemistry, matter, energy, forces, waves,
electricity, geology, weather, astronomy, measurement, and scientific
investigation. Added coverage includes cell transport, enzymes, inheritance,
endothermic/exothermic reactions, optical phenomena, atomic ions, scientific
units, and practical technologies. Each row names a concept, its signed
strength, and associated words. Negative strengths encode some contrasts,
such as renewable versus fossil energy sources, transverse versus longitudinal
wave motion, and active versus passive transport's energy requirements.

A SHA-256-based, fixed signed projection gives each named concept a direction
in 24 dimensions. A word's embedding is the normalized sum of its concept
directions. This is a direct formula, not an embedding learned from text or
labels. `lexical_families.txt` explicitly identifies 284 lexical families,
including irregular plurals, British/American spellings, unit names, and
scientific process forms such as “photosynthesizing,” “refracted,” and “froze.” Each
family receives the union of its associations and identical salience/prior
metadata, so equivalent forms have exactly equal full embeddings. Deterministic
spelling rules add further inflected forms; explicit entries take precedence.

A fixed list of grammatical function words is excluded from science association
construction. This prevents words such as “the,” “without,” and “most” from
acquiring scientific meanings simply because they appear in a fact description.
Negation flags and lexical priors remain separate from semantic vectors. Words
without associations, including unknown words, also have zero semantic vectors.
A small set of handwritten priors favors scientific evidence and penalizes
overconfident absolutes.

## How the fixed transformer computes a score

The 36 state channels contain 24 concept coordinates and 12 circuit channels:
salience, original squared norm, a reference channel, question/option identity,
a working score, a lexical prior, pooled prior, pooled question negation,
a local negation marker, output, and two constant anchors.

1. Positions 0–127 mark question tokens; positions 128–175 mark answer tokens.
   The first attention layer pools question concepts, with higher salience
   for distinctive science terms and an effectively hard exclusion of
   zero-vector grammatical/unknown tokens. A fixed positional recency factor
   has a 24-token half-life, emphasizing the actual query over a long preamble.
   The four heads transport six concept coordinates each. The pooled vector
   `y` is added to each token's own vector `x`.
2. The first MLP uses an analytic three-neuron GELU square formula for each
   of the 24 coordinates, using all 72 hidden units. Richardson extrapolation
   cancels the fourth-order error term. The embedding supplies `-||x||²`
   directly in the working residual channel, so the resulting score is
   approximately `s(x) = ||x+y||² - ||x||² = 2 x·y + ||y||²`.
3. The second attention layer uses all four heads:
   - **Question negation:** detect question-scoped cues such as “not,”
     “except,” “least,” and negative contractions; ignore option negations.
   - **Shared offset:** repeat the first layer's question weighting. Since
     the weighted mean of question vectors is `y`, the weighted mean of
     `s(x)` is `3 ||y||²`. Subtract one third of this mean. No zero-concept
     word, anchor token, or special sentinel is required.
   - **Answer relevance:** average `s(x)` over answer tokens using key
     `8 ||x||² + 1.5 salience`. A concept-bearing word has `||x||² = 2.25`,
     giving a content advantage of `exp(18)` before salience weighting.
     Thus even 47 zero-concept tokens cannot materially dilute one recognized
     term. Existing salience metadata also emphasizes distinctive answer
     terms over broad words. Together with the offset subtraction, this gives
     approximately `2 (weighted mean answer vector) · y`.
   - **Answer prior:** independently average the lexical priors over answer
     tokens, so negation does not reverse the answer-quality prior.
4. The last MLP conditionally reverses the semantic score for negated
   questions and adds the pooled prior. The completed score is broadcast to
   every real token. The given final mean pooling therefore preserves it,
   rather than diluting it according to question length or option length.
   No answer position or choice label is used.

Two opposite constant channels, +512 and -512, stabilize layer normalization.
Every circuit read subtracts a reference channel to cancel layer normalization's
mean shift. GELU arithmetic and gates are analytic approximations; they are
not trained approximators.

### Analytic square formula

For `a = 1/16`, each coordinate uses

```text
[ (3/2) GELU(a z) + (7/6) GELU(-a z) - (1/6) GELU(2 a z) ]
----------------------------------------------------------------
                      sqrt(2/pi) a²
```

This equals `z² - a⁴ z⁶/10 + O(a⁶ z⁸)`. Its coefficients follow from
`(4 S_a - S_(2a))/3`, where `S_a` is the symmetric two-GELU square
approximation, and the exact identity `GELU(t) - GELU(-t) = t`. No coefficients
were fitted. The three-read construction frees enough capacity to cancel the
quartic term in every coordinate. Moving the known negative word norm into
the embedding removes the need for separate subtraction neurons.

## Review improvements

A design review identified two avoidable weaknesses in the original circuit:
shared-offset subtraction depended on zero-concept question words, and scoring
only answer tokens before the fixed final mean caused length dilution. The
revised circuit removes that dependency by the weighted-mean identity above
and broadcasts a separately normalized answer score. It also adds positional
recency and recognizes more question-negation spellings.

A subsequent knowledge-layer review added explicitly authored facts, exact
lexical-family equivalence, and function-word filtering. A grounding check
caught one initially ungrounded alias family (`boulder/boulders`), which was
resolved with its rock/glacial-transport association. These changes were made
from science and linguistic knowledge, not question predictions. Neither the
lexicon nor the circuit was selected or tuned against measured accuracy.

The numerical review replaced the original two-neuron square approximation
with the higher-order three-neuron formula above, using all 72 first-layer
MLP units instead of 50. Larger constant anchors reduce layer-normalization
perturbations. The intended semantic scoring rule and all handwritten science
associations remain unchanged by this numerical refinement.

The process-vocabulary review added 48 explicit families for common scientific
verbs, irregular past forms, and related process nouns. These reuse existing
handwritten associations rather than adding new random feature directions.
The families also propagate the existing salience and prior metadata, avoiding
inconsistent scores for equivalent forms. Distinct senses remain separate:
meteorological “weather” is not merged with “weathering,” and breathing is not
merged with cellular respiration. The existing prior distinction between
“reproduce” and “reproduction” is preserved.

The pooling review removed residual dilution by zero-vector filler. The norm
coefficient 8 follows from the known 2.25 content-vector norm: `47 exp(-18)`
is below one millionth, even before the salience advantage over function words.
Question pooling uses the same content gate; its question/option segment gap
was increased to prevent answer leakage when a question has no known concepts.
Answer pooling uses half the question's salience gain. All these rules are
fixed analytic/design choices, not results of measured-accuracy selection.

## Scope and limitations

This is a compact semantic heuristic, not a general language model. It can
connect familiar science terms and some contrasting concepts, but cannot do
reliable multi-step deduction, arbitrary numerical calculations, syntactic
role assignment, or exact factual retrieval. Projecting many associations
into 24 dimensions inevitably introduces collisions. Negation detection is
simple and does not resolve grammatical scope or double negation. The scoring
circuit assumes nonempty question and answer text, as normal ARC items have.
Small numerical deviations arise from layer normalization and GELU arithmetic.

## Verification

The required command `python /srv/work/tools/check_model.py` passes. The model
has **6,754 vocabulary entries, 271,188 weights, and 170,827 nonzero weights**.
The checker was run on its placeholder input, not on held-out ARC questions.

Separate arithmetic, lexicon, and synthetic-token checks verified:

- the square stencil on 601 fixed points in `[-3, 3]`: maximum absolute error
  0.00108942 in float64 and 0.00108719 in float32, below the analytic 0.0012 bound;
- use of all 72 square neurons and exact negative-norm initialization;
- exact full-embedding equality across all 284 lexical families;
- coverage of explicit process forms and preservation of distinct word senses;
- zero semantic vectors for all 140 listed function words present in the vocabulary;
- grounding and vocabulary coverage for the additional concepts;
- agreement with the analytic scoring formula within 0.003 on the synthetic
  token cases, including single-word questions;
- invariance within 0.001 when identical question or option tokens are repeated;
- invariance within 0.001 under insertion of up to 127 zero-vector, zero-prior
  question tokens and 47 such answer tokens, using function words and OOV tokens;
- answer-position permutation equivariance;
- exact ties for identical options;
- finite float32 weights and finite full-batch logits;
- exactly equal weight tensors from two independent `build()` calls.

The final resource check directly ran **3,530 synthetic items in 56 batches**
with batch size 64 and four CPU threads. All question and option arrays were
full-length and non-padding (128 question tokens and 48 tokens per option),
exercising the maximum 176-token sequence size. This used random valid token
IDs from the handwritten vocabulary, not ARC text or answer labels.

- Build time: **1.293 seconds**.
- Complete 3,530-item synthetic forward pass: **9.343 seconds**.
- Peak resident memory over the full check: **1,096.8 MiB**.
- Check body including two builds: **11.861 seconds**.

These are measured full-workload resource results, replacing the previous
single-batch extrapolation. Both the 2 GiB memory budget and 20-minute time
budget passed. The final review retained the existing weights rather than
making additional speculative semantic changes. No held-out accuracy claim
is made.

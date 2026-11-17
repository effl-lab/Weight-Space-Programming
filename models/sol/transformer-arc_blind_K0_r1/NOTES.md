# Hand-written science association transformer — reviewed version

## Deliverable and constraints

This is an untrained, deterministic model for the supplied, unchanged `arch.py`.
`build()` returns exactly the required dense float32 tensors. There are no
checkpoints, downloaded resources, labeled training examples in the model,
optimizers, fitted weights, or accuracy-driven parameter searches. No internet
or held-out ARC data was accessed.

Files used by the build:

- `knowledge.txt`: 784 manually written science association/contrast rows,
  covering chemistry, mechanics, electricity, waves, geology, weather,
  astronomy, biology, ecology, resources, measurement, inquiry, and safety.
- `aliases.txt`: explicitly written word forms and scientific spelling variants.
- `vocab.txt`: 3,539 unique, tokenizer-compatible words, in sorted order.
- `build.py`: the complete numerical construction of the weights.

The reviewed model has **305,024 weights**, of which **167,919 are nonzero**.
The vocabulary generator is `python build.py`; normal evaluation only calls
`build()` and does not rewrite the vocabulary. The build rejects stale
vocabulary, conflicting or untokenizable aliases, duplicate fact identifiers,
and nonpositive/nonfinite fact strengths.

## Review changes and their rationale

The revision was chosen by examining the circuit and the written science facts,
not by searching or fitting against question accuracy.

1. **Option-level pooling and broadcast.** Previously, evidence was placed only
   at option-token positions, then divided by the architecture's total real-token
   count. That introduced dependence on question length and rewarded repeating
   answer text. Two previously unused attention heads now pool evidence and a
   small qualifier prior over the option and broadcast them to all positions.
   The final mean therefore reads an approximately option-level score rather
   than diluting it by the question/option length ratio.
2. **Meaningful-word pooling.** Scientific option words have `exp(8)` times the
   pooling weight of null/function words. This keeps a few extra grammatical
   words from substantially diluting the evidence. If no option word is known,
   attention still has a well-defined fallback over its real tokens.
3. **Smaller normalization error.** The fixed opposite ballast coordinates were
   increased from +/-10 to +/-100, with analytically matched normalization
   weights. Their squared norm is now 20,000, compared with a non-ballast state
   of order 10. This reduces unintended normalization-induced gain variation
   and baseline mismatch by roughly two orders of magnitude. This is not a fit.
4. **Correct representation of absence.** Ten knowledge rows previously included
   text such as "no antenna" on a positive side. The compiler encodes signed
   sets, not the grammar of these descriptions, so that incorrectly associated
   the absent property with the subject. The rows for volcanic glass, vacuum,
   prokaryotes, anaerobic respiration, nonvascular plants, arachnids, fungal
   nutrition, the lunar surface, primary succession, and frictionless motion
   now express those contrasts directly in the appropriate signed sets.
5. **Word-form coverage and integrity.** Added ordinary inflections for existing
   knowledge, such as `rise/rises/rising` and `carry/carries/carried`, and more
   explicit contractions for negation. Repeated canonical alias groups are
   merged instead of overwriting earlier forms. Neutral aliases are expanded
   consistently too. Every present alias group is mechanically checked to share
   the same semantic embedding.

6. **Semantic-vector readout for dense questions.** The original
   heads returned only science-word attention mass. When every question word is
   scientific, those values are nearly constant regardless of affinity. The
   heads now additionally return the retrieved semantic vectors. Ninety-two
   previously unused block-zero MLP neurons add a small, explicitly derived
   dot-product-like score using the residual vector. This removes that
   information-loss bottleneck without altering any facts, word codes,
   vocabulary, or `arch.py`. The constants are fixed by design, not selected
   using accuracy.

7. **Scientific distinctions and possessives (earlier review).** That pass left
   `build.py`, all circuit constants, and `arch.py` unchanged. It separates
   over-broad aliases for chlorine/chloride, ammonia/ammonium, spinal/spine,
   microscopic/microorganism, and Arctic/Antarctic versus the ambiguous word
   polar. The word polar itself still has both chemical and geographical senses.
   `Hg` is now tied to quicksilver's chemical facts instead of inheriting all
   facts about the planet Mercury. `Sun's` now shares the Sun's full word vector,
   rather than only the solar-energy adjective's vector. Six manually written
   facts retain useful relationships between the separated concepts while
   expressing charge, anatomy, microbial life, and polar geography correctly.
   The cell-theory description uses "fundamental unit" rather than teaching an
   association with the chemical sense of "basic." Ordinary singular and plural
   possessive forms of common science nouns are included explicitly, because
   the tokenizer retains their apostrophes. These are manual knowledge and
   dictionary corrections, not changes selected from measured accuracy.

8. **Guaranteed null grammatical anchors (latest review).** The compiler formerly
   used `setdefault` when inserting neutral words. This left `in` with a science
   vector because the weathering fact says "in place." Such incidental grammar
   should not become scientific evidence or cease to provide a null baseline.
   An explicit handwritten set of closed-class words now has zero semantic
   vectors, including equivalent aliases, regardless of incidental fact text.
   Directional/comparative terms and negation controls are not blanket-cleared.
   The regression test checks that each anchor's complete embedding equals the
   unknown-word/null embedding, including its science flag and curvature offset.
   `Who`, `whom`, and `whose` are now explicitly included as neutral words; the
   other anchors were already in the vocabulary. The 784 fact rows, scientific
   aliases, scoring constants, and original `arch.py` were not changed. A separate
   unlabeled, maximum-length resource test now exercises all 3,530 synthetic
   inputs using the task's batching protocol, without any accuracy measurement.

These are concrete coverage, representation, and circuit improvements. They do
**not** establish that held-out ARC accuracy increased; that accuracy is unknown.

## Where the knowledge lives

Each knowledge row supplies a strength, a set of related terms, and optionally a
contrasting set. A deterministic SHAKE-256 code assigns the row three independent
15-dimensional, zero-mean vectors. Related terms receive the same vector;
contrasting terms receive its negative. A word's vector is the sum of its fact
contributions. Aliases share the canonical word's vector. Each of its three
15-dimensional parts is normalized to the same fixed norm.

The hash is only a reproducible code assignment, not a source of science facts
or a fitted embedding. All factual content and relative fact strengths are in
the handwritten text files. Signed facts provide useful contrasts such as
renewable/nonrenewable, conductor/insulator, and transverse/longitudinal.

The remaining embedding coordinates hold science-word membership, negation,
a small explicitly specified prior against absolute qualifiers, a deterministic
original-word smooth-square offset, and circuit workspace. Two opposite constant coordinates stabilize layer normalization.
Positions distinguish the fixed question region from the option region; they
do not distinguish answer labels or option ordering.

## How the fixed transformer computes a score

1. **Block zero, heads 0–2:** The three semantic views let each option word query
   scientific associations in the question. A large positional key offset
   excludes option words from the retrieved context. Science words pay a fixed
   attention log-prior. Non-science question words provide zero-valued null
   alternatives, so an association can increase attention mass on meaningful
   question words rather than merely redistribute a fixed mass among them.
   Each head also retrieves its 15 semantic coordinates and adds one quarter
   of that vector to the corresponding semantic residual coordinates.
2. **Block zero, head 3:** Retrieve whether the question contains an exception or
   negation cue, independently of option text.
3. **Block zero MLP:** Combine the three semantic outputs using their mean minus
   a disagreement penalty. GELU differences supply exact linear terms;
   symmetric GELUs approximate absolute differences. Agreement across views
   suppresses isolated random-projection collisions. An additional smooth-square
   circuit reads the retrieved semantic vectors, as described below.
4. **Block one, head 0:** Retrieve the null-query score from non-scientific
   question words, allowing approximately question-specific baseline removal.
5. **Block one, head 1:** Separately detect explicit negation inside the option.
6. **Block one, heads 2–3:** Broadcast the meaningful-option-word mean evidence
   and the all-option-word mean qualifier prior, respectively.
7. **Block one MLP:** Large-offset GELU differences select the appropriate sign,
   subtract the null baseline, and reverse evidence for question negation XOR
   option negation. The resulting score coordinate is present at every real
   token, not only at answer-token positions.
8. **Final score:** Read the dedicated score coordinate minus a reserved reference
   coordinate. This cancels the layer-normalization mean. The supplied
   architecture performs its required mean over real tokens.

All attention projections, offsets, normalization parameters, MLP circuits,
embeddings, and the score vector are supplied by `build.py`. Unused connections
are explicitly zero. Inference uses only the original transformer operations;
there is no Python question-answering callback or custom tensor behavior.

## Auxiliary semantic similarity: explicit algebra, not training

Let `u` be an original word vector and `a` the concatenated retrieved vectors.
The attention residual contains approximately `u + eps*a`, where `eps = 1/4`.
The identity

```
(||u + eps*a||^2 - ||u||^2) / (2*eps)
    = dot(u, a) + eps/2 * ||a||^2
```

turns the squared residual norm into a compatibility score plus a small
retrieved-context coherence term. It does not require any null question word.
The contribution is weighted by `1/16`, keeping it auxiliary to the existing
three-view science-mass score.

Symmetric GELUs supply the smooth square

```
phi(z) = sqrt(pi/2) / t^2 * (GELU(t*z) + GELU(-t*z)),  t = 1/32.
```

It approximates `z^2` for the bounded semantic coordinates. Ninety MLP neurons
implement the 45 symmetric pairs, and two more subtract the original word's
`sum(phi(u_j))`. This original-word offset is evaluated analytically with `erf`
and stored in embedding coordinate 60. It is a direct formula on the already
handwritten embeddings, not a fitted statistic or an evaluation-data estimate.
There are still only 128 hidden units; 100 of them are now used in block zero.
Layer normalization introduces small residual deviations from the ideal identity.

## Validation

The required command is:

```
python /srv/work/tools/check_model.py
```

`arch.py` is unchanged. Its SHA-256, checked explicitly by the mechanical test
script as well as by the required checker, is:

```
d241c3beee162ff162b57b6434d20f3be4137d846110abca5cb52e5701589b48
```

The following checks in `/srv/work/verify_model.py` passed after the revision:

- Repeated builds are bit-for-bit deterministic.
- Weights and outputs are finite float32 values.
- The vocabulary matches the compiled knowledge exactly.
- Present aliases share identical semantic embeddings.
- Explicit grammatical anchors have exactly the same complete embeddings as
  unknown/null words, even when they occur incidentally in science fact text.
- Six scientifically distinct term pairs have distinct embeddings rather than
  being collapsed by aliases; Sun's and other tested possessives match their
  intended noun embeddings.
- Permuting options permutes logits correspondingly.
- Identical options receive identical logits.
- Single-batch and 64-question-batch results agree.
- Duplicating the entire question or repeating option text preserves logits
  within the stated numerical tolerance; the largest measured absolute
  difference in that metamorphic check was 0.006371.
- Adding six null/function words to each one-word option changes its pooled
  evidence only slightly, consistent with their `exp(-8)` relative weight.
- A fully populated 64-question batch at the maximum sequence lengths remains
  finite, as do empty and wholly unknown inputs.
- On six fixed, synthetic vector fixtures, the auxiliary MLP agrees with its
  exact smooth-square formula within 0.00000194, and with the ideal quadratic
  identity within 0.00007299. These are arithmetic checks, not QA accuracy tests.
- An all-scientific-token question fixture produces a content-dependent logit
  spread of about 19.29, confirming that the constant-value saturation path is
  no longer the only available evidence. This is not a benchmark score.

With four PyTorch threads, the measured build took approximately **0.35 seconds**
and a 64-question forward pass approximately **0.18 seconds**. Peak process memory
across these checks was **1,090 MiB**, below 2 GiB.

The additional `/srv/work/check_model_resources.py` test processed **3,530**
synthetic, unlabeled inputs with every question position (128) and option position
(48) populated. It used four PyTorch threads and the task's batches of 64. The
full inference took **9.57 seconds**, or **11.20 seconds** including imports,
build, and fixture construction; peak process memory was **1,303.4 MiB**. All
logits were finite, and the script asserts the complete run is below 20 minutes
and 2,000,000,000 bytes of peak resident memory. This is a resource/computation
check, not an ARC test or an accuracy estimate. These are local measurements,
not guarantees for every environment.

The latest required checker reported **CHECK OK** and answered **11/12** of the
original self-written questions in `/srv/work/my_questions.json` correctly. This
is a smoke-test regression from the preceding revision's 12/12: the xylem item
is now incorrect. The null-anchor correction was designed before running that
check, from the compiler/circuit contract, and has not been tuned or reversed
using the answer results. Those questions are not ARC data and are not a
representative accuracy benchmark. No parameter or content search was performed
using their answers. Held-out ARC accuracy and whether it improved are unknown.

## Limitations

This is a compact association engine, not a pretrained language model. It cannot
reliably perform arithmetic, parse complicated scope, distinguish every causal
role, or carry out multi-step scientific reasoning. Negation handling is shallow,
and word-level pooling can overvalue a plausible distractor. Hash collisions,
polysemy, and competing fact associations remain possible. Null calibration is
weak when the question contains no non-scientific words. Layer normalization is
only approximately a fixed scale, so pooling invariance is not mathematically
exact. No held-out ARC data was accessed, and held-out accuracy remains unknown.

# Hand-compiled science-association transformer

This is a manually designed, untrained semantic compatibility scorer. It uses
exactly the provided `arch.py`; the inference code, tokenization, architecture,
and option-order convention have not been changed.

## Knowledge and word forms

`knowledge.py` contains 622 hand-written conceptual neighborhoods covering life
science, human biology, ecology, chemistry, everyday materials, energy,
mechanics, electricity, waves, astronomy, geology, weather, measurement, and
experimental methods. It also specifies lexical equivalences, function words,
weak plausibility priors, and explicit negation terms. These are general science
associations, not stored question/answer pairs.

`morphology.py` specifies English inflection families, irregular plurals, common
British/American spellings, and regular plural/possessive rules. Word forms such
as `plant`, `plants`, `plant's`, and `plants'` share one embedding. The compiler
unifies forms **before** accumulating conceptual memberships, so merely listing
both singular and plural in a neighborhood does not count the association
twice. Neighborhood-size weights also count unique canonical content words,
not repeated spellings or grammatical filler. Explicitly different scientific
senses such as meteorological `weather` and rock `weathering` are not merged by
an indiscriminate suffix stripper.

Canonicalization can collapse several alias declarations onto one destination.
The compiler combines distinct target concepts once, ignores identity aliases,
and reads from a fixed original-vector snapshot. Consequently dictionary order
cannot select the winning alias, duplicate inflections cannot overweight a
source, and a self-alias cannot erase an informative semantic link.

A coverage audit checks declarations against compiled semantic vectors, not
question-answer accuracy. It identified previously empty science word families.
Explicit cognate links now cover `digest`/`digestion`, `consume`/`consumer`,
`react`/`reaction`, and `mix`/`mixture`, including their declared inflections.
Additional hand-written definitions cover calories, vaporization, paleontology,
design criteria, data analysis, formulas, radius, dietary fiber, scientific
theories, and laws. Mass measurement now includes weighing. British/American
forms such as `analyze`/`analyse`, `center`/`centre`, and the spellings of
vaporization share their appropriate word-family embeddings. None of these
additions was selected using measured accuracy.

Each concept receives a reproducible 15-dimensional random-feature code with a
smaller contribution from its broad science topic. A word's embedding
superposes its conceptual neighborhoods. Broadly shared words are attenuated,
and small neighborhoods receive somewhat more weight. For 115 concepts,
hand-specified signed contrast axes distinguish alternatives such as
renewable/nonrenewable energy, conductors/insulators, melting/freezing, and
independent/dependent variables.

Three neighborhoods additionally contain explicit **negative memberships**:
sound requires matter rather than a vacuum; classical energy conservation and
closed chemical-system mass conservation exclude creation/destruction of the
conserved quantity. Excluded words retain the broad topic component but reverse
the fact-specific code. This avoids treating an explicitly incompatible member
as a positive association merely because it appears in a fact's description.

Codes come from SHA-256 labels and a fixed normal random-feature construction,
not pretrained embeddings. `vocab.txt` has 8,937 entries, including inflected,
possessive, and spelling variants. Padding and unknown words have zero lexical
features.

## Transformer circuit

The 36 channels initially hold 15 lexical features, a negation cue, 15 question
context features, a lexical prior, a position marker, a temporary content flag,
and two constant opposite anchors. Channels are reused between stages.

* **Question retrieval:** Block 0 has two content-sensitive heads and two
  averaging heads. Position markers suppress option keys and mildly favor later
  question tokens. A temporary lexical-content flag favors known science words
  over grammatical filler by approximately `exp(3)`.
* **Compatibility:** Its MLP uses four-GELU finite differences to approximate
  each product between an option feature and the retrieved question feature.
  A modest self-energy subtraction discounts incidental similarity. Product
  units are shut off at question positions.
* **Priors and scratch space:** Paired neurons use the exact identity
  `GELU(z)-GELU(-z)=z` to position-gate the weak lexical prior. After their context
  has been consumed by the product units, two channels are cleared and reused
  for the option-only prior and a question-negation flag. The temporary content
  flag is also removed, to layer-normalization precision.
* **Two-sided negation:** Block 1 independently detects negation in the question
  and in the current option. Its MLP reverses semantic compatibility if either
  side is negative, but restores the sign if both are negative. Shared-offset
  GELU difference gates implement this parity rule without a cubic-product
  approximation. Contractions and tokenized `non-` prefixes are covered.
* **Prior polarity:** Plausibility priors follow the question's polarity only.
  A negative option therefore does not turn an implausibility penalty into a
  bonus. `Avoid` remains a hazardous-action concept, not a global negation cue.
* **Readout:** The final score vector reads compatibility plus the signed prior
  while canceling the final layer normalization's mean. The supplied architecture
  performs the prescribed real-token mean pooling and independent option scoring.

Opposite anchors of magnitude 32 stabilize layer normalization. Feature-reading
matrices subtract their average to cancel the common mean shift exactly. The
remaining normalization rescaling is small but not mathematically zero. The
block-0 product circuit uses the small, binary-exact scale `1/8` to reduce
higher-order GELU error. Unused parameters and neurons are explicitly zero.

All tensors are finite, deterministic, dense float32. Neither the build nor the
design uses gradient descent, fitting, accuracy-driven search, external files,
internet access, ARC data beyond the task's permitted examples, or pretrained
weights.

## Scope and limitations

This is an associative science baseline, not a general reasoning model. It
cannot reliably calculate arithmetic, resolve long causal chains, distinguish
every role within a relation, or recover clause-level negation scope. The
negation circuit detects the presence of a cue within each side; it does not
parse nested negations or exceptions such as "not only." Narrow random-feature
codes can produce accidental similarities. No option index is preferred.

## Verification

The supplied `check_model.py` passes: **349,776 weights**, **137,558 nonzero**, and
**8,937 vocabulary entries**.

Separate structural/numerical checks establish that:

* `arch.py` matches the literal task specification;
* weights are finite and bitwise deterministic;
* explicit inflection and possessive families have identical embeddings;
* newly covered science terms have nonzero semantic vectors and content flags,
  with their declared inflections/spelling variants sharing identical embeddings;
* reversing alias declarations leaves every semantic vector bitwise unchanged;
* adding redundant word forms and grammatical filler to conceptual descriptions
  leaves every semantic vector bitwise unchanged;
* each isolated signed neighborhood gives excluded members opposite fact codes;
* option permutations permute logits, and batch composition does not change
  scores beyond rounding;
* artificial feature inputs satisfy all four question/option negation parity
  cases;
* option negation preserves the sign of a plausibility penalty;
* a lexical prior at the final question position is completely shut off.

An entire **3,530-row synthetic pass** took **25.21 seconds** on four threads,
with **937.9 MiB** peak resident memory for the checking process. The synthetic
inputs were repeated artificial rows, not real evaluation questions. These
checks measure circuit behavior and resources, not question-answer accuracy,
and were not used to search or fit parameters or content. Held-out accuracy is
unknown.

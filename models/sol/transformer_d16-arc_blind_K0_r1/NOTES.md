# Hand-written science association transformer

This model is an analytic circuit in the supplied, unchanged two-layer,
16-dimensional transformer. There is no training, fitting, pretrained material,
external corpus, or accuracy-driven architecture/content search.

## Knowledge

`knowledge.tsv` is an explicitly authored science dictionary, covering mechanics,
energy, matter, chemistry, astronomy, Earth science, weather, cells, genetics,
plants, animals, ecology, human biology, laboratory work, and measurement. Each
word has at most two question-side association labels and two answer-side concept
labels. These are asymmetric: a question cue can point to a scientific property
or function, while an answer word can be labelled with that property or function.
Explicit spelling and ontology normalizations are in `aliases.tsv`. The revised
lexicon includes additional taxonomy, plant development, cell functions,
biochemistry, anatomy, laboratory instruments, and investigation vocabulary.
There are 7,680 spellings and 1,288 active concept labels. The scientific
concept inventory is unchanged by the latest review. `word_aliases.tsv` adds
explicit lexical equivalents: noun possessives, alternate scientific spellings,
selected inflections, and negative contractions. Balanced ASCII-single-quoted
forms of known words are also generated mechanically, because those quote marks
survive the task's tokenizer. Every equivalent copies its source word's complete
features, including exclusion markers.

Compilation checks that every spelling is reachable by the supplied tokenizer,
rejects alias cycles and duplicate spellings, and omits concept labels occurring
on only one side of the query/key relation. Such labels cannot represent a match;
this is static dictionary validation, not fitting or a search over accuracy.

`vocab.txt` is the sorted list of dictionary spellings. Unrecognized words have no
semantic labels. A deterministic Fibonacci-sphere construction converts concept
names to distinct unit directions in three coordinates. This is a geometric
encoding of the written dictionary, not an embedding learned from text.

## Circuit

The 16 residual coordinates hold two three-dimensional query codes, two
three-dimensional answer codes, a normalization anchor, a question/option
role, an evidence accumulator, and a packed exclusion/length register. Positional embeddings mark
positions 128--175 as answer positions, exactly as the prescribed tokenizer does.

1. **Block 0 attention** broadcasts a question-side exclusion marker. Answer-side
   negation does not set the question's marker. A second, uniform attention head
   measures the fraction of real tokens belonging to the option. This fraction
   is stored below the 100-unit spacing of the exclusion marker.
2. **Block 0 MLP** removes answer-key codes from question positions. The remaining
   question positions become null competitors for the next attention operation.
   It folds local-plus-broadcast exclusion markers back to one marker using a
   step subtraction that preserves the fractional length register.
3. **Block 1 attention** uses the four heads for all four pairings of the two
   query labels with the two answer labels. A fixed negative answer-position
   offset means that only a close code match wins against the null competitors.
   Values identify answer positions, so the accumulated attention mass measures
   matched scientific associations rather than merely returning another word.
   A role-dependent query component prevents option queries from generating
   self-match evidence when a real question token is present. Dyadic value
   projections keep null question values exactly zero in floating-point tests.
4. **Block 1 MLP** keeps evidence at question positions, suppresses answer-side
   self-matching, and reverses its sign for exclusion-style questions. Evidence
   is read relative to the role coordinate, which is exactly zero at question
   positions. This removes the previous word-dependent semantic-coordinate
   baseline. The semantic mean is used only in the widely separated role gates.
   Ten further hidden units decode the fractional length register and change
   only the anchor, compensating for option-length dilution in mean pooling.
5. The prescribed final layer normalization, mean pooling, and linear scorer
   convert the accumulated evidence into option logits. No architecture code is
   changed to implement the compensation.

All matrices, biases, normalization parameters, positions, and vocabulary
embeddings are generated explicitly by `build.py`. Unused connections are zero.
The large constant anchor and float64 arithmetic keep the normalization/gating
identities numerically stable; the active features and gates remain small
relative to the anchor. The anchor is a numerical device, not stored knowledge.
GELU pairs implement signed linear functions and well-separated conditional
arithmetic without changing the architecture.

## Limitations

This is a compact associative science heuristic, not a pretrained language model
or a general symbolic reasoner. Word-level concepts cannot fully disambiguate
polysemy, phrases, negation scope, arithmetic, multi-step deductions, or which
relation a question asks about. The new circuit approximately removes dilution
from unrelated option tokens, but matching remains sensitive to wording,
repetition, and the number of actual associations. No held-out questions have
been accessed and no claim of held-out accuracy is made.

## Verification

Use `python /srv/work/tools/check_model.py` to check the required files, immutable
architecture, tensor shapes, build, and forward pass. The optional questions in
`/srv/work/my_questions.json` are newly authored smoke tests, not ARC data. They
are not used by `build()` or to fit or search the model.

Completed checks in this environment:

- Supplied checker: **CHECK OK**.
- Vocabulary: **7,680** words; **1,288** active concepts.
- Weights: **130,224** scalars, **82,081** nonzero.
- Build time: approximately **0.130 seconds**.
- One full-length batch of 64 questions and four options: approximately
  **0.766 seconds**, with **1,224.0 MiB** peak resident memory.
- All **3,299** pre-extension word embeddings and the padding/unknown rows were
  verified bitwise unchanged. Vocabulary ids change with sorting, but each
  original word's vector remains identical. Non-embedding weights are unchanged
  except for **26** explicitly audited role-gate/value-scaling entries described
  below; the earlier fingerprint baseline has not been replaced.
- Direct second-attention traces verified exactly zero option-side evidence
  updates for maximal artificial self-matches, with question lengths 1 and 128,
  option lengths 1 and 48, and both exclusion polarities. Question-side retrieval
  was preserved relative to a copy with only the new query gate removed
  (`rtol=0`, `atol=1e-8`).
- All **542** explicit lexical aliases and **3,840** quoted equivalents were
  verified to have exactly the same embedding as their source word, including
  exclusion flags. One explicit alias was already defined consistently in the
  base lexicon; **541** unquoted spellings are new.
- Exact required tensor names/shapes and finite weights were verified.
- Direct artificial-code tests verified all four attention match paths,
  negligible null-match evidence, exclusion-sign reversal, isolation of
  question markers from option markers, and option-permutation equivariance.
- Length compensation was verified on artificial codes with question lengths
  1 and 128, option lengths 1, 16, 32, and 48, and both evidence polarities
  (`rtol=1e-4`, `atol=1e-5`). These are circuit invariants, not question accuracy.
- Full-batch finite outputs, exact equal logits for identical options, zero
  output for entirely padded inputs, and the empty-question guard were verified.
- `arch.py` is unchanged; SHA-256:
  `305fca3c0d510de38233efd45d511fc568aa24afc5ab6b7576682ad1c78f2de8`.

These are validity and resource checks, not an estimate of ARC accuracy.

## Earlier review changes

The review corrected an analytically identifiable scoring offset: the old MLP
scored evidence minus a semantic-code mean, even when there was no match. The
revised evidence reference is the zero question-role coordinate. The role-gate
coefficients were re-derived for that reference, and the fixed anchor was
increased from 1e6 to 1e8 to reduce residual normalization error while retaining
float64 precision and the same memory footprint.

Ontology corrections separated problem-solving from chemical solutions, ocean
currents from electrical current, and the spring season from an elastic spring.
Ambiguous surface words retain explicitly limited senses instead of merging
those entire concept classes. Explicit synonym/spelling corrections and newly
written scientific facts fill additional coverage gaps. The unreachable
hyphenated vocabulary entry was replaced with a tokenizer-valid spelling.

All revisions were based on circuit algebra and authored science knowledge.
There was no accuracy sweep, parameter selection against results, or use of
additional ARC data. `/srv/work/verify_model.py` contains the structural tests;
it is not used by `build()` and does not contain science questions or labels.
The supplied checker was run after the model revisions were complete. Neither
those smoke outputs nor any held-out outputs were used to fit the revision.

## Review: option-length compensation

Previously, question evidence was divided by `nq + no` by the mandated mean
pool, where `nq` and `no` are real question and option token counts. Appending
unrelated option words therefore weakened positive evidence and made negative
evidence less negative. This was an algebraic effect of pooling, independent of
any question results.

The revised first block's uniform head measures `f = no / (nq + no)`. Its value
is packed into the exclusion register: after folding a duplicate local marker,
the register is approximately `100 * flag + f`. Strengthening exclusion
attention to a logit separation of 32 makes its discrete marker precise enough
for this use. The fold now subtracts a fixed 100 above the local-marker threshold
instead of reflecting the register about a threshold, preserving `f`.

At the final MLP, the register relative to role lies in one of the intervals
`-100 + f`, `f`, or `100 + f`. Three clipped ramps and two reset steps decode the
fraction with ten GELU units. They change the normalization anchor from `A` to
`A * (1-f)`. The unchanged final layer norm consequently scales evidence by
approximately `1/(1-f) = (nq+no)/nq`, cancelling option-length dilution. Question
evidence is thus approximately averaged over question tokens, not over the
combined question/option length. Only unrelated extra tokens are intended to be
invariant; additional scientific content can still change the score.

The decoded fraction is capped at `255/256` to keep the anchor positive for a
degenerate empty question. This cap cannot affect a valid nonempty question:
with at most 48 option tokens, `f <= 48/49`. The initial anchor is now `1e10`,
which keeps normalization error small even after this bounded rescaling while
float64 retains enough precision for concept codes. Tensor shapes, vocabulary,
scientific dictionary, option-position neutrality, and architecture are unchanged.
No accuracy measurement was used to select these constants or changes.

## Review: conservative lexical coverage

The fixed tokenizer preserves ASCII apostrophes, so a known noun's possessive
form, a contraction, or an ASCII-single-quoted word previously often became an
unknown token. The new explicit lexical table and quoted-spelling rule address
this without changing any scientific association or scoring parameter.

Examples of the representation equivalences are `cell's` / `cells'` -> `cell`,
`neurone` -> `neuron`, `fertilisation` -> `fertilization`, and `isn't` / `cannot`
-> the existing `not` exclusion marker. Quoted known words copy the same full
feature vectors. Possessives are listed for selected nouns rather than generated
with unrestricted stemming. Contractions retain the existing global negation
heuristic and do not solve its scope limitations. Curly-apostrophe contractions
are not claimed to be normalized when the fixed tokenizer splits them.

The lexical compiler requires all sources to exist in the authored base lexicon,
rejects duplicate aliases and unreachable spellings, and refuses to overwrite a
conflicting existing meaning. Validation caught `solidified`, whose existing
geological association differs from the proposed freezing alias; that alias was
removed, leaving the original entry untouched. No question output was involved.

Before the revision, fingerprints were recorded for all original word vectors,
special rows, and non-embedding weights. `/srv/work/verify_lexical_extension.py`
checks those against `/srv/work/lexical_review_baseline.json` and proves exact
feature equality for every new equivalent. These are structural regression
artifacts generated from this hand-written model, not training data; `build()`
does not read them. At the lexical-review stage, all original vectors and
scoring tensors were unchanged while vocabulary coverage increased from 3,299
to 7,680 spellings. The concept directions, numerical circuit, and `arch.py`
were unchanged by that extension. The subsequent numerical revision below is
explicitly accounted for by the same regression helper.

The full structural suite and supplied checker were run after the extension.
No accuracy-driven selection, fitting, additional ARC questions, external data,
or internet access was used. Held-out accuracy remains unmeasured.

## Latest review: early self-match suppression and exact null values

The final MLP already removes option-side evidence, but previously the second
attention block could first generate large option self-matches. Suppressing
those matches earlier avoids relying only on later floating-point cancellation.
The new fourth query component adds `(role - evidence) / 32` in each second-block
head. Before that attention operation, question role and evidence are both zero,
so question queries retain their original fourth component, 2. Option queries
instead have a component approximately `2 + 100/32 = 5.125`.

In the ideal unit-code calculation, even an option's maximal self-match has
logit at most `TEMP - (1 + ROLE/64) * (TEMP - MATCH) = -25579.5`.
When a real question/null competitor is present, this margin makes the option
self-attention mass underflow to zero in float64. The final MLP gate remains as
a guard, including degenerate empty-question cases; exact-zero option attention
is not claimed for empty questions.

The first exact-zero regression assertion exposed approximately `3.6e-7` residue
from non-binary projections of equal coordinates. Rather than relaxing that
assertion, the value projections were changed to exact power-of-two contrasts:
marker values use unit contrasts, and role values use contrasts divided by 128.
Compensating output scalings preserve their intended algebraic contributions.
The query-role coefficient is exactly `1/32`. Thus equal-coordinate null values
remain exactly zero in the direct-attention tests.

Exactly 26 tensor entries differ from the pre-lexical fingerprint baseline:
four in `block0.v`, two in `block0.o`, eight in `block1.q`, eight in `block1.v`,
and four in `block1.o`. Only the eight query-gate connections are newly nonzero.
The regression helper checks each new value explicitly, restores the old values
in test-only copies, and checks the original fingerprints for all remaining
entries. Vocabulary, all embeddings, concept directions, scientific knowledge,
tensor shapes, and `arch.py` are unchanged by this review.

Artificial-code tracing now verifies exactly zero option evidence updates under
maximal self-alignment in all four heads, while preserving question retrieval
at the tested length and polarity boundaries. All earlier structural checks and
the required checker also pass. These changes follow circuit algebra and
numerical invariants, not question accuracy; no training, fitting, additional
ARC data, external data, or accuracy-driven selection was used.

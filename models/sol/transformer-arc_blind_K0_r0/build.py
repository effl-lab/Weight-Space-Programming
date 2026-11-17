"""A completely authored, untrained transformer science-association scorer.

There is no optimization, dataset access, model download, or generated training
corpus here. All matrices are assembled from elementary algebra and the explicit
lexical associations in knowledge.py. See NOTES.md for the circuit design.
"""
from pathlib import Path
import math
import re

import torch

from knowledge import AXES, PRIORS, NEGATIONS, QUESTION_MARKERS, OPTION_NEGATIONS, INFLECTIONS, MORPHOLOGY

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")
D = 64
SEM = list(range(45)) + list(range(54, 61))
PRIOR, NEG, OPTION, FIRST, BODY = 45, 46, 47, 48, 49
NEG_ALL, EVIDENCE, OUTPUT, BASE = 50, 51, 52, 53
# NEG carries lexical question-negation strength before block 0 and the
# broadcast option-negation bit afterward. OUTPUT temporarily carries an
# option-negation lexical flag, cleared before the actual score is computed.
OPT_NEG = NEG
ANCHOR_P, ANCHOR_N, REF = 61, 62, 63
# Large, balanced anchors make feature/norm interference negligible in float32.
# Their magnitude is an algebraic conditioning choice, not an accuracy fit.
ANCHOR = 4096.0
LN_SCALE = math.sqrt(2 * ANCHOR * ANCHOR / D)
ATT_OFFSET = 3.0
RECENCY = 1.5
GAIN = 8.0


def lexicon():
    """Return the authored lexical vectors, with conservative spelling aliases."""
    assert len(AXES) == len(SEM) == 52
    table = {}
    for j, (_, groups) in enumerate(AXES):
        for strength, text in groups:
            for word in set(TOKEN.findall(text)):
                table.setdefault(word, [0.0] * len(SEM))[j] += strength
    if MORPHOLOGY:
        # Orthography only: do not infer any labels or inspect any questions.
        # Existing authored words always take precedence over derived aliases.
        additions = {}
        for word, vector in sorted(table.items()):
            if len(word) < 4 or not word.isalpha() or word.endswith('s'):
                continue
            if word.endswith('y') and word[-2] not in 'aeiou':
                plural = word[:-1] + 'ies'
            elif word.endswith(('ch', 'sh', 'x', 'z')):
                plural = word + 'es'
            else:
                plural = word + 's'
            if plural not in table:
                additions.setdefault(plural, vector[:])
        table.update(additions)
    # These families are explicitly written grammar knowledge. Keep separately
    # authored forms intact even when a family contains one of those forms.
    for source, forms in INFLECTIONS:
        assert source in table, f'Missing authored inflection source: {source}'
        for word in forms.split():
            table.setdefault(word, table[source][:])
    if MORPHOLOGY:
        # The task's tokenizer retains straight apostrophes. Preserve a noun's
        # scientific features in both singular and plural possessive spellings.
        # Generating lexical aliases is not a runtime tokenizer modification.
        possessives = {}
        for word, vector in sorted(table.items()):
            if len(word) < 3 or not word.isalpha():
                continue
            forms = [word + "'s"]
            if word.endswith('s'):
                forms.append(word + "'")
            for form in forms:
                if form not in table:
                    possessives[form] = vector[:]
        table.update(possessives)
    for word in set(PRIORS) | NEGATIONS | QUESTION_MARKERS | OPTION_NEGATIONS:
        table.setdefault(word, [0.0] * len(SEM))
    # Explicit zero-feature function words make the vocabulary easy to inspect.
    for word in ("a an the and or of to from for in on at by with as is are was were "
                 "be been being do does did has have had what which when where why "
                 "how who this that these those it its they their them there than "
                 "then will would can could should may might most more less likely "
                 "best example examples following statement statements describes "
                 "about during through into over under between before after each "
                 "some any other such so because if while two three four five").split():
        table.setdefault(word, [0.0] * len(SEM))
    return table


def build():
    torch.set_num_threads(4)
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    assert len(vocab) == len(set(vocab)) and all(TOKEN.fullmatch(t) for t in vocab)
    table = lexicon()
    assert set(table) <= set(vocab), 'Regenerate vocab.txt from lexicon()'
    w = {}
    w['embed'] = torch.zeros((len(vocab) + 2, D), dtype=torch.float32)
    # Equal and opposite anchors stabilize all layer norms, including unknowns.
    w['embed'][:, ANCHOR_P] = ANCHOR
    w['embed'][:, ANCHOR_N] = -ANCHOR
    # Most vocabulary rows are grammatical aliases of an existing vector.
    # Compute each distinct normalization once, preserving the exact scalar
    # operations of the uncached builder. The cache is local to this build.
    semantic_cache = {}
    zero_semantics = (0.0,) * len(SEM)
    for row, word in enumerate(vocab, 2):
        raw = tuple(table.get(word, zero_semantics))
        if raw not in semantic_cache:
            vector = torch.tensor(raw, dtype=torch.float32)
            vector.clamp_(-2.0, 2.0)
            vector *= min(1.0, 4.5 / max(float(vector.norm()), 1e-12))
            semantic_cache[raw] = vector
        w['embed'][row, SEM] = semantic_cache[raw]
        # The prior channel temporarily doubles as a question-marker flag.
        # Block 0 consumes that flag and clears it before the scoring readout.
        # Possessive rows inherit their base's lexical metadata as well as its
        # semantic vector. Interrogative contractions such as "what's" likewise
        # retain the syntax-marker role of "what".
        canonical = word
        if word.endswith("'s") and word[:-2] in table:
            canonical = word[:-2]
        elif word.endswith("'") and word[:-1] in table:
            canonical = word[:-1]
        prior = PRIORS.get(canonical, 0.0)
        w['embed'][row, PRIOR] = 1.0 if canonical in QUESTION_MARKERS else prior
        # NEG+PRIOR is exactly one on a negation marker, even for a word
        # such as 'never' which also has a negative option-language prior.
        w['embed'][row, NEG] = (1.0 - prior) if canonical in NEGATIONS else 0.0
        w['embed'][row, OUTPUT] = float(canonical in OPTION_NEGATIONS)
    w['pos'] = torch.zeros((176, D), dtype=torch.float32)
    w['pos'][128:, OPTION] = 1.0
    # FIRST is 1 at the null position, negative normalized position on the
    # remaining question tokens, and 0 on options. It supports both the null
    # gate and recency without consuming another representation coordinate.
    w['pos'][0, FIRST] = 1.0
    w['pos'][1:128, FIRST] = -torch.arange(1, 128, dtype=torch.float32) / 128.0
    w['pos'][1:128, BODY] = 1.0
    for i in range(2):
        p = f'block{i}.'
        for norm in ('ln1', 'ln2'):
            w[p + norm + '.weight'] = torch.full((D,), LN_SCALE, dtype=torch.float32)
            w[p + norm + '.bias'] = torch.zeros(D, dtype=torch.float32)
        for name in ('q', 'k', 'v', 'o'):
            w[p + name] = torch.zeros((D, D), dtype=torch.float32)
            w[p + name + '.bias'] = torch.zeros(D, dtype=torch.float32)
        w[p + 'fc1'] = torch.zeros((D, 128), dtype=torch.float32)
        w[p + 'fc1.bias'] = torch.zeros(128, dtype=torch.float32)
        w[p + 'fc2'] = torch.zeros((128, D), dtype=torch.float32)
        w[p + 'fc2.bias'] = torch.zeros(D, dtype=torch.float32)
    w['ln_f.weight'] = torch.full((D,), LN_SCALE, dtype=torch.float32)
    w['ln_f.bias'] = torch.zeros(D, dtype=torch.float32)
    w['score'] = torch.zeros(D, dtype=torch.float32)
    w['score'][OUTPUT] = 1.0
    w['score'][REF] = -1.0

    def linear_feature(matrix, source, target, value):
        # Layer norm subtracts a mean. Subtracting a feature with raw value zero
        # cancels that mean, leaving a nearly identity-scaled raw feature.
        matrix[source, target] += value
        matrix[REF, target] -= value

    def mlp_unit(prefix, unit, features, bias, outputs):
        for source, value in features.items():
            linear_feature(w[prefix + 'fc1'], source, unit, value)
        w[prefix + 'fc1.bias'][unit] = bias
        for target, value in outputs.items():
            w[prefix + 'fc2'][unit, target] = value

    # BLOCK 0 / HEAD 0: select the LAST negation or interrogative marker.
    # Markers dominate ordinary words; among markers, four logit units/token
    # make the most recent win. A later 'which/what/...' therefore resets a
    # background sentence's 'not'. The first-position sentinel is earlier than
    # any real body position, which is appropriate for this competition.
    p = 'block0.'
    w[p + 'q.bias'][0] = 4.0
    linear_feature(w[p + 'k'], NEG, 0, 1024.0)
    linear_feature(w[p + 'k'], PRIOR, 0, 1024.0)
    linear_feature(w[p + 'k'], FIRST, 0, -512.0)
    linear_feature(w[p + 'k'], OPTION, 0, -100000.0)
    linear_feature(w[p + 'v'], NEG, 0, 1.0)
    w[p + 'o'][0, NEG_ALL] = 1.0

    # BLOCK 0 / HEAD 1: query-length-dependent, zero-similarity baseline.
    # The first question token is a null competitor with value zero. All other
    # real question tokens have key bias -3 and value one. Padding is masked
    # by arch.py, not by assumptions about the length of the question.
    w[p + 'q.bias'][16] = 4.0
    linear_feature(w[p + 'k'], BODY, 16, -ATT_OFFSET - RECENCY)
    linear_feature(w[p + 'k'], FIRST, 16, -RECENCY)
    w[p + 'k.bias'][16] = RECENCY
    linear_feature(w[p + 'k'], OPTION, 16, -1000.0)
    linear_feature(w[p + 'v'], BODY, 16, 1.0)
    w[p + 'o'][16, BASE] = 1.0

    # BLOCK 0 / HEAD 2: independently detect assertion negation in THIS OPTION.
    # Each sample is one question/option pair; option keys dominate question
    # keys by 1000 logit units, so a question's 'not' cannot leak into this bit.
    w[p + 'q.bias'][32] = 4.0
    linear_feature(w[p + 'k'], OUTPUT, 32, 60.0)
    linear_feature(w[p + 'k'], OPTION, 32, 1000.0)
    linear_feature(w[p + 'v'], OUTPUT, 32, 1.0)
    # Reuse NEG after its lexical role is finished. The broadcast result lives
    # in [4, 5.22] if set, versus [0, 1.22] otherwise, allowing a clean decode.
    w[p + 'o'][32, OPT_NEG] = 4.0

    # BLOCK 0 / MLP: turn the first real token into a content-free null key.
    # GELU(z)-GELU(-z) == z exactly. A large positional gate turns this
    # subtraction off everywhere other than position zero.
    for j, dim in enumerate(SEM):
        mlp_unit(p, 2*j, {dim: 8.0, FIRST: 128.0}, -128.0, {dim: -1/8})
        mlp_unit(p, 2*j+1, {dim: -8.0, FIRST: 128.0}, -128.0, {dim: 1/8})
    # The max-attention negation result is rounded with a smooth clipped ramp.
    # Remove its old residual first, then add a robust approximately 0/1 bit.
    u = 2 * len(SEM)
    mlp_unit(p, u, {NEG_ALL: 1.0}, 0.0, {NEG_ALL: -1.0})
    mlp_unit(p, u+1, {NEG_ALL: -1.0}, 0.0, {NEG_ALL: 1.0})
    mlp_unit(p, u+2, {NEG_ALL: 64.0}, -16.0, {NEG_ALL: 1/32})
    mlp_unit(p, u+3, {NEG_ALL: 64.0}, -48.0, {NEG_ALL: -1/32})
    # Clear the temporary question-marker flag without changing ordinary
    # language priors, whose magnitudes are all below the ramp's threshold.
    mlp_unit(p, u+4, {PRIOR: 64.0}, -16.0, {PRIOR: -1/32})
    mlp_unit(p, u+5, {PRIOR: 64.0}, -48.0, {PRIOR: 1/32})
    # Replace the mixed lexical/broadcast NEG coordinate with the decoded
    # option-only bit. A clipped ramp between 1.5 and 2.5 separates its ranges.
    mlp_unit(p, u+6, {OPT_NEG: 1.0}, 0.0, {OPT_NEG: -1.0})
    mlp_unit(p, u+7, {OPT_NEG: -1.0}, 0.0, {OPT_NEG: 1.0})
    mlp_unit(p, u+8, {OPT_NEG: 64.0}, -96.0, {OPT_NEG: 1/64})
    mlp_unit(p, u+9, {OPT_NEG: 64.0}, -160.0, {OPT_NEG: -1/64})
    # OUTPUT was a lexical flag at input, but must be zero before readout.
    mlp_unit(p, u+10, {OUTPUT: 1.0}, 0.0, {OUTPUT: -1.0})
    mlp_unit(p, u+11, {OUTPUT: -1.0}, 0.0, {OUTPUT: 1.0})
    assert u + 12 <= 128

    # BLOCK 1: four topic-association heads. An option token can attend only
    # to question tokens or the null competitor. The matched probability minus
    # the preceding block's baseline is evidence of association (or opposition).
    p = 'block1.'
    for head in range(4):
        start = head * 16
        dims = SEM[head * 15:(head + 1) * 15]
        for local, dim in enumerate(dims):
            # Dot products / sqrt(16): (2*2)/4 gives coefficient one.
            linear_feature(w[p + 'q'], dim, start + local, 2.0)
            linear_feature(w[p + 'k'], dim, start + local, 2.0)
        w[p + 'q.bias'][start + 15] = 4.0
        linear_feature(w[p + 'k'], BODY, start + 15, -ATT_OFFSET - RECENCY)
        linear_feature(w[p + 'k'], FIRST, start + 15, -RECENCY)
        w[p + 'k.bias'][start + 15] = RECENCY
        linear_feature(w[p + 'k'], OPTION, start + 15, -1000.0)
        linear_feature(w[p + 'v'], BODY, start, 1.0)
        w[p + 'o'][start, EVIDENCE] = GAIN

    # BLOCK 1 / MLP: score only option tokens, using question/option XOR
    # polarity. A negated assertion flips semantic evidence; a question asking
    # for a false assertion flips it again. Absolute-language priors depend on
    # question polarity alone: an option's 'not' must not reward its 'only'.
    # Each matching binary-control pattern opens exactly one pair of gates.
    # In active branches GELU(C+e)-GELU(C-e) ~= 2e; others are in the zero tail.
    gate = 128.0
    for qneg in (0, 1):
        for oneg in (0, 1):
            branch = 2 * qneg + oneg
            out_sign = 1.0 if qneg == oneg else -1.0
            evidence = {EVIDENCE: 1.0, BASE: -4 * GAIN,
                        PRIOR: 1.0 if oneg == 0 else -1.0}
            for flip in (0, 1):
                sign = 1.0 if flip == 0 else -1.0
                features = {k: sign * value for k, value in evidence.items()}
                features[OPTION] = gate
                features[NEG_ALL] = (2 * qneg - 1) * gate
                features[OPT_NEG] = (2 * oneg - 1) * gate
                mlp_unit(p, 2*branch + flip, features,
                         -(qneg + oneg + .5) * gate,
                         {OUTPUT: out_sign * sign * .5})

    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in w.values())
    return w


if __name__ == '__main__':
    # Explicit one-time vocabulary serialization. build() itself only reads it.
    (ROOT / 'vocab.txt').write_text('\n'.join(sorted(lexicon())) + '\n')
    print(f'Wrote {len(lexicon())} authored vocabulary entries.')

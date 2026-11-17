"""Compile a manually specified science lexicon into the fixed transformer.

There is no learning, optimization, dataset loading, or accuracy-based search.
All constants below specify circuits or fixed semantic random features.
"""
from pathlib import Path
import hashlib
import math

import numpy as np
import torch

from knowledge import (ALIASES, CONTRASTS, NEGATIVE_MEMBERS, NEGATION, PRIORS,
                       STOP, entries, morphology)
from morphology import canonical

D = 36
SEM = 15
ANCHOR = 32.0
LN_SCALE = math.sqrt(2 * ANCHOR**2 / D)


def code(label):
    # A reproducible random-feature basis; not pretrained word vectors.
    seed = int.from_bytes(hashlib.sha256(label.encode('utf8')).digest()[:8], 'little')
    v = np.random.default_rng(seed).standard_normal(SEM)
    return v / np.linalg.norm(v)


def resolve_aliases(original, aliases, forms):
    """Merge distinct alias targets once, independently of declaration order.

    Morphological normalization can map several declarations to one destination.
    Identity aliases must not erase informative links, and repeated spellings
    must not give a source extra weight. Read exclusively from the original
    neighborhood vectors so aliases do not cascade in dictionary order.
    """
    targets = {}
    for word, base in aliases.items():
        word, base = canonical(word, forms), canonical(base, forms)
        if word != base and base in original:
            targets.setdefault(word, set()).add(base)
    result = dict(original)
    for word in sorted(targets):
        bases = sorted(targets[word])
        target = np.mean([original[base] for base in bases], axis=0)
        magnitude = float(np.mean([np.linalg.norm(original[base]) for base in bases]))
        norm = np.linalg.norm(target)
        if norm < 1e-12:
            # Conflicting evidence that exactly cancels supplies no direction.
            continue
        target = target * magnitude / norm
        vec = 0.35 * original[word] + 0.65 * target if word in original else target
        norm = np.linalg.norm(vec)
        if norm >= 1e-12:
            result[word] = vec * magnitude / norm
    return result


def semantic_vectors():
    vectors, counts = {}, {}
    forms = morphology()
    for topic, name, row in entries():
        # Scientific opposites share a contrast axis with opposite signs rather
        # than receiving unrelated codes that can accidentally look alike.
        contrast, polarity = CONTRASTS.get(name, ('concept:' + name, 1.0))
        topic_code = 0.38 * code('topic:' + topic)
        concept_code = 0.925 * polarity * code(contrast)
        positive = topic_code + concept_code
        positive /= np.linalg.norm(positive)
        negative = topic_code - concept_code
        negative /= np.linalg.norm(negative)
        # Unite inflections before both membership and neighborhood-size counts.
        # Redundant spelling variants or grammar must not change a fact's weight.
        members = {canonical(word, forms) for word in row if word not in STOP}
        members.difference_update(STOP)
        strength = (8.0 / max(4, len(members))) ** 0.5
        excluded = {canonical(word, forms)
                    for word in NEGATIVE_MEMBERS.get((topic, name), ())}
        for word in members:
            vectors.setdefault(word, np.zeros(SEM))
            vectors[word] += strength * (negative if word in excluded else positive)
            counts[word] = counts.get(word, 0) + 1
    for word, vec in vectors.items():
        n = np.linalg.norm(vec)
        magnitude = 1.2 * min(1.0, (6.0 / counts[word]) ** 0.25)
        vectors[word] = vec * magnitude / max(n, 1e-12)
    vectors = resolve_aliases(vectors, ALIASES, forms)
    for word, base in forms.items():
        if base in vectors:
            vectors[word] = vectors[base].copy()
    return vectors


def put_feature(matrix, feature, column, scale=1.0):
    """Read a coordinate independently of layer norm's mean subtraction.

The two large opposite anchors average to zero before normalization. Subtract
that average from any read coordinate, undoing the common LN mean exactly.
"""
    matrix[feature, column] += scale
    matrix[34, column] -= 0.5 * scale
    matrix[35, column] -= 0.5 * scale


def build():
    vocab = Path(__file__).with_name('vocab.txt').read_text().splitlines()
    w = {'embed': torch.zeros((len(vocab) + 2, D)),
         'pos': torch.zeros((176, D)), 'score': torch.zeros(D)}
    vectors = semantic_vectors()
    forms = morphology()
    for i, word in enumerate(vocab, 2):
        base = canonical(word, forms)
        if word not in STOP and base not in STOP and word in vectors:
            w['embed'][i, :SEM] = torch.from_numpy(vectors[word].astype(np.float32))
            # Temporarily reuse the score channel as a lexical-content flag.
            # Block 0 consumes it for key selection, then removes it in the MLP.
            w['embed'][i, 33] = 0.125
        negbase = ALIASES.get(word, ALIASES.get(base, base))
        w['embed'][i, 15] = float(negbase in NEGATION)
        w['embed'][i, 31] = PRIORS.get(word, PRIORS.get(base, 0.0))

    # Question slots and option slots are known from the supplied architecture.
    # Padding is still masked exclusively by arch.py, not by vocabulary tricks.
    w['pos'][:, 34] = ANCHOR
    w['pos'][:, 35] = -ANCHOR
    w['pos'][:128, 32] = -1.0 - 0.4 * torch.arange(128) / 127
    w['pos'][128:, 32] = 1.0

    for b in range(2):
        p = f'block{b}.'
        for ln in ('ln1', 'ln2'):
            w[p + ln + '.weight'] = torch.full((D,), LN_SCALE)
            w[p + ln + '.bias'] = torch.zeros(D)
        for n in ('q', 'k', 'v', 'o'):
            w[p + n] = torch.zeros((D, D))
            w[p + n + '.bias'] = torch.zeros(D)
        w[p + 'fc1'] = torch.zeros((D, 72))
        w[p + 'fc1.bias'] = torch.zeros(72)
        w[p + 'fc2'] = torch.zeros((72, D))
        w[p + 'fc2.bias'] = torch.zeros(D)

    # Block 0: two content-sensitive heads and two question-average heads.
    # Head dimension 8 is the question/option mask; the other eight positions
    # carry semantic queries and/or values.
    p = 'block0.'
    for head in range(4):
        offset = 9 * head
        features = list(range(8)) if head % 2 == 0 else list(range(8, SEM))
        w[p + 'q.bias'][offset + 8] = 1.0
        put_feature(w[p + 'k'], 32, offset + 8, -24.0)
        # exp(72 * 0.125 / sqrt(9)) = exp(3): prefer known science
        # content to grammatical filler, including in the average heads.
        put_feature(w[p + 'k'], 33, offset + 8, 72.0)
        for j, d in enumerate(features):
            if head < 2:
                put_feature(w[p + 'q'], d, offset + j, math.sqrt(18.0))
                put_feature(w[p + 'k'], d, offset + j, math.sqrt(18.0))
            put_feature(w[p + 'v'], d, offset + j)
            w[p + 'o'][offset + j, 16 + d] = 0.6 if head < 2 else 0.4

    # A four-GELU finite-difference identity computes products. Around zero,
    # GELU(z)+GELU(-z) = sqrt(2/pi)*z^2 + O(z^4).
    # The 15 products compare a word with its retrieved question neighborhood.
    # A modest self-energy subtraction centers incidental semantic similarity.
    t = 0.125
    coefficient = 1.0 / (4.0 * math.sqrt(2.0 / math.pi) * t*t)
    unit = 0
    for d in range(SEM):
        for sign, pair, outsign in ((1, 1, 1), (-1, 1, 1),
                                    (1, -1, -1), (-1, -1, -1)):
            # a +/- (b - 0.27*a), with both polarities.
            put_feature(w[p + 'fc1'], d, unit, sign * t * (1 - pair * 0.27))
            put_feature(w[p + 'fc1'], 16 + d, unit, sign * t * pair)
            # All four units are disabled at question positions. Consequently
            # only option/question compatibility enters the final mean pool.
            put_feature(w[p + 'fc1'], 32, unit, 8.0)
            w[p + 'fc1.bias'][unit] = -8.0
            w[p + 'fc2'][unit, 33] = coefficient * outsign
            unit += 1

    # GELU(z)-GELU(-z)=z exactly. Apply the same positional shutoff
    # to both neurons: unlike multiplying by (marker+1)/2, this remains
    # disabled even where the question's recency marker is less than -1.
    for sign in (1.0, -1.0):
        put_feature(w[p + 'fc1'], 31, unit, sign)
        put_feature(w[p + 'fc1'], 32, unit, 8.0)
        w[p + 'fc1.bias'][unit] = -8.0
        # Keep plausibility separate: option-local negation should reverse
        # semantic content, not turn an implausibility penalty into a bonus.
        w[p + 'fc2'][unit, 17] = sign
        unit += 1

    # Release the reused content-flag channel. The two large anchors make
    # normalized feature reads nearly identity-scaled. The exact GELU
    # odd-part identity removes the flag to that same normalization precision.
    for sign in (1.0, -1.0):
        put_feature(w[p + 'fc1'], 33, unit, sign)
        w[p + 'fc2'][unit, 33] = -sign
        unit += 1

    # The contextual values have already been consumed by the product units.
    # Reuse two of their channels for a global-negation flag and an option-only
    # prior. Parallel MLP reads still see the original context for all products.
    for scratch in (16, 17):
        for sign in (1.0, -1.0):
            put_feature(w[p + 'fc1'], scratch, unit, sign)
            w[p + 'fc2'][unit, scratch] = -sign
            unit += 1

    # Block 1: independent OR detectors for question and option negation.
    # Negation in an option must not affect the question detector, or vice versa.
    # Their sum changes the option marker to +1, -1, or -3 (zero, one, or both
    # sides negated). Preserve the question-only flag for plausibility priors.
    p = 'block1.'
    for head, mask_sign in ((0, -1.0), (1, 1.0)):
        offset = 9 * head
        w[p + 'q.bias'][offset + 8] = 1.0
        put_feature(w[p + 'k'], 32, offset + 8, mask_sign * 60.0)
        put_feature(w[p + 'k'], 15, offset + 8, 36.0)
        put_feature(w[p + 'v'], 15, offset)
        w[p + 'o'][offset, 32] = -2.0
    w[p + 'o'][0, 16] = 1.0

    # Odd GELU differences act as linear gates when their shared offset is
    # strongly positive and vanish when it is negative. This implements the
    # parity rule: reverse similarity for either negation, but not for both.
    # Unlike the earlier small-signal product, no cubic approximation is needed.
    unit = 0
    for bias, coefficient in ((0.0, -1.0), (-16.0, 1.0)):
        for sign in (1.0, -1.0):
            put_feature(w[p + 'fc1'], 33, unit, sign)
            put_feature(w[p + 'fc1'], 32, unit, -8.0)
            w[p + 'fc1.bias'][unit] = bias
            w[p + 'fc2'][unit, 33] = coefficient * sign
            unit += 1

    # Add the option prior, reversing it for a negative QUESTION only. A word
    # such as 'never' must not acquire a plausibility bonus merely because its
    # own OPTION was negative. The prior was position-gated in block 0.
    for sign in (1.0, -1.0):
        put_feature(w[p + 'fc1'], 17, unit, sign)
        w[p + 'fc2'][unit, 33] = sign
        unit += 1
    for sign in (1.0, -1.0):
        put_feature(w[p + 'fc1'], 17, unit, sign)
        put_feature(w[p + 'fc1'], 16, unit, 16.0)
        w[p + 'fc1.bias'][unit] = -8.0
        w[p + 'fc2'][unit, 33] = -sign
        unit += 1

    w['ln_f.weight'] = torch.full((D,), LN_SCALE)
    w['ln_f.bias'] = torch.zeros(D)
    w['score'][33] = 20.0
    w['score'][34] = -10.0
    w['score'][35] = -10.0
    return w

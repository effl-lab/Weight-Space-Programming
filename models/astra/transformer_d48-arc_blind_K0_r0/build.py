"""Hand-authored associative transformer, with an explicit GELU dot-product circuit.

No learning, fitting, external data, or accuracy-based selection is performed.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
D = 48
ANCHOR = 1000.0
LN_SCALE = ANCHOR * math.sqrt(2.0 / D)
# Embedding coordinates: own[0:20], associations[20:40], six controls, anchors.
SEG, T, T2, EVIDENCE, NEGATION, SALIENCE = 40, 41, 42, 43, 44, 45
ANCHOR_PLUS, ANCHOR_MINUS = 46, 47
# Coordinates 0 and 1 are reclaimed after the first block's similarity circuit.
POOLED, NEG_DETECTED = 0, 1
PRODUCT_EPS = 0.05
NEGATORS = tuple("not except least avoid incorrect false cannot never neither "
                 "isn't aren't wasn't weren't doesn't don't didn't can't couldn't "
                 "won't wouldn't shouldn't mustn't hasn't haven't hadn't needn't "
                 "oughtn't shan't".split())


def knowledge():
    records = []
    for line in (ROOT / 'knowledge.tsv').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, domain, aliases, links = (part.strip() for part in line.split('|'))
        records.append((name, domain, aliases.split(), links.split()))
    return records


def base_vocabulary():
    words = set()
    for name, domain, aliases, links in knowledge():
        words.update(aliases)
    words.update(NEGATORS)
    words.update('only always all every none'.split())
    return sorted(words)


def curated_forms():
    """Read the finite, authored list of scientific word-form aliases."""
    result = {}
    for line in (ROOT / 'word_forms.tsv').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        source, spellings = (part.strip() for part in line.split('|'))
        for spelling in spellings.split():
            assert spelling.isascii() and spelling.isalpha() and spelling.islower()
            assert spelling not in result or result[spelling] == source, spelling
            result[spelling] = source
    return result


def lexical_forms():
    # The required tokenizer retains ASCII apostrophes, including quote marks
    # at token boundaries. Normalize these forms through copied embeddings,
    # not by modifying the tokenizer or introducing text-time inference code.
    # Explicitly authored entries always take precedence over generated forms.
    base = base_vocabulary()
    forms = {word: word for word in base}
    for spelling, source in curated_forms().items():
        assert source in base, (spelling, source)
        forms.setdefault(spelling, source)
    for word, source in list(forms.items()):
        if word.isalpha():
            forms.setdefault(word + "'s", source)
    # One pass over base words, curated forms, and possessives handles both quoted words and
    # quoted possessives. Pure numerals are excluded to avoid treating a feet
    # or minutes mark as a word-quotation/possessive marker.
    for word, source in list(forms.items()):
        if any(c.isalpha() for c in word):
            for variant in ("'" + word, word + "'", "'" + word + "'"):
                forms.setdefault(variant, source)
    return forms


def vocabulary():
    return sorted(lexical_forms())


def normalize_halves(x):
    # Separate normalizers prevent either random projection from depending on
    # the other one's norm. Every concept has norm sqrt(10) in each half.
    halves = np.asarray(x, dtype=np.float64).reshape(2, 10)
    norms = np.linalg.norm(halves, axis=1, keepdims=True)
    return (halves / np.maximum(norms, 1e-12) * math.sqrt(10.0)).reshape(20)


def code(label):
    # Two independently normalized, data-independent spherical code halves.
    # There is deliberately no shared centering or norm across the halves.
    b = hashlib.shake_256(label.encode()).digest(80)
    x = np.frombuffer(b, dtype='<u4').astype(np.float64)
    x = (x + 0.5) / 4294967296.0 - 0.5
    return normalize_halves(x)


def build():
    records = knowledge()
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    assert len(vocab) == len(set(vocab))
    forms = lexical_forms()
    assert set(vocab) == set(forms), 'Regenerate vocab.txt from the authored forms'
    index = {s: i + 2 for i, s in enumerate(vocab)}
    names = {r[0] for r in records}
    assert len(names) == len(records)
    own = {}
    for name, domain, aliases, links in records:
        x = 0.90 * code('concept:' + name) + 0.44 * code('domain:' + domain)
        own[name] = normalize_halves(x)
    target = {}
    for name, domain, aliases, links in records:
        x = 0.70 * own[name]
        for link in links:
            parts = link.split(':')
            other = parts[0]
            assert other in names, (name, other)
            weight = float(parts[1]) if len(parts) == 2 else 1.0
            x = x + weight * own[other]
        target[name] = normalize_halves(x)

    w = {'embed': torch.zeros((len(vocab) + 2, D)), 'pos': torch.zeros((176, D))}
    w['embed'][:, ANCHOR_PLUS] = ANCHOR
    w['embed'][:, ANCHOR_MINUS] = -ANCHOR
    accum = {}
    for name, domain, aliases, links in records:
        for word in aliases:
            accum.setdefault(word, []).append(name)
    generic = set('energy matter object objects material materials substance substances system systems change changes changing process processes environment environmental organism organisms animal animals plant plants earth water air living life natural nature scientist scientists scientific investigation experiment experiments investigate study studies'.split())
    for word, concepts in accum.items():
        i = index[word]
        a = np.mean([own[c] for c in concepts], axis=0)
        b = np.mean([target[c] for c in concepts], axis=0)
        salience = 0.45 if word in generic else 1.0
        w['embed'][i, :20] = torch.tensor(a * salience, dtype=torch.float32)
        w['embed'][i, 20:40] = torch.tensor(b * salience, dtype=torch.float32)
        w['embed'][i, SALIENCE] = salience
    for word in NEGATORS:
        w['embed'][index[word], NEGATION] = 1.0
    for word, prior in {'only': -0.025, 'always': -0.035, 'never': -0.035, 'none': -0.015}.items():
        w['embed'][index[word], EVIDENCE] = prior
    # Direct, one-hop aliases preserve every feature, including salience,
    # negative-question markers, and authored scalar priors, bit for bit.
    for variant, source in forms.items():
        if variant != source:
            w['embed'][index[variant]].copy_(w['embed'][index[source]])

    w['pos'][:128, SEG] = -1.0
    w['pos'][128:, SEG] = 1.0
    # Keep positional coordinates appreciably above float32 cancellation noise,
    # while their squared coordinate remains small relative to the anchors.
    t = (torch.arange(176, dtype=torch.float32) - 87.5) / 16.0
    w['pos'][:, T] = t
    w['pos'][:, T2] = t.square()

    for layer in range(2):
        p = f'block{layer}.'
        for ln in ('ln1', 'ln2'):
            w[p + ln + '.weight'] = torch.full((D,), LN_SCALE)
            w[p + ln + '.bias'] = torch.zeros(D)
        for n in ('q', 'k', 'v', 'o'):
            w[p + n] = torch.zeros((D, D))
            w[p + n + '.bias'] = torch.zeros(D)
        w[p + 'fc1'] = torch.zeros((D, 96))
        w[p + 'fc1.bias'] = torch.zeros(96)
        w[p + 'fc2'] = torch.zeros((96, D))
        w[p + 'fc2.bias'] = torch.zeros(D)

    def diff(matrix, source, output, coefficient):
        # The mean of the opposite anchors is a zero reference before LN.
        # Subtracting their average exactly cancels LN's feature-mean term.
        matrix[source, output] += coefficient
        matrix[ANCHOR_PLUS, output] -= 0.5 * coefficient
        matrix[ANCHOR_MINUS, output] -= 0.5 * coefficient

    root12 = math.sqrt(12.0)
    p = 'block0.'
    # Cross-projection retrieval: attention computed in one ten-dimensional
    # view retrieves values from the other. Scoring uses the retrieved view,
    # not the view which selected the token. This avoids mechanically rewarding
    # same-projection noise that happened to receive high softmax attention.
    # Dot/sqrt(10) is the standard variance normalization for ten coordinates.
    for head in (0, 1):
        base = 12 * head
        w[p + 'q.bias'][base + 10] = 1.0
        w[p + 'q.bias'][base + 11] = 1.0
        diff(w[p + 'k'], SEG, base + 10, -30.0 * root12)
        diff(w[p + 'k'], SALIENCE, base + 11, math.log(2.0) * root12)
        for j in range(10):
            source = 10 * head + j
            selector = 10 * (1 - head) + j
            diff(w[p + 'q'], selector, base + j, math.sqrt(12.0 / 10.0))
            diff(w[p + 'k'], 20 + selector, base + j, 1.0)
            diff(w[p + 'v'], 20 + source, base + j, 1.0)
            w[p + 'o'][base + j, 20 + source] = 1.0

    # Heads 2/3 subtract the token's old association vector from its residual.
    # The positional logit is -20*(i-j)^2. The self token is therefore selected
    # independent of semantics, even when the sequence has no function words.
    positional_strength = 20.0 * 16.0 ** 2
    for half in (0, 1):
        base = 12 * (2 + half)
        diff(w[p + 'q'], T, base, 2.0 * positional_strength * root12)
        diff(w[p + 'k'], T, base, 1.0)
        w[p + 'q.bias'][base + 1] = -positional_strength * root12
        diff(w[p + 'k'], T2, base + 1, 1.0)
        diff(w[p + 'q'], T2, base + 2, -positional_strength * root12)
        w[p + 'k.bias'][base + 2] = 1.0
        for j in range(10):
            source = 20 + 10 * half + j
            diff(w[p + 'v'], source, base + j, 1.0)
            w[p + 'o'][base + j, source] = -1.0

    # GELU(z)+GELU(-z) = sqrt(2/pi)*z^2 + O(z^4).
    # Four units thus implement a*b with a controlled small-input expansion.
    # Summing twenty products gives the actual retrieved-vector similarity.
    coefficient = 1.0 / (20.0 * 4.0 * math.sqrt(2.0 / math.pi) * PRODUCT_EPS ** 2)
    signs = ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0),
             (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0))
    for j in range(20):
        for offset, (sa, sb, so) in enumerate(signs):
            unit = 4 * j + offset
            diff(w[p + 'fc1'], j, unit, sa * PRODUCT_EPS)
            diff(w[p + 'fc1'], 20 + j, unit, sb * PRODUCT_EPS)
            w[p + 'fc2'][unit, EVIDENCE] = so * coefficient

    # Reclaim only two semantic coordinates, after their products were read.
    # GELU(x)-GELU(-x)=x provides deterministic identity cancellation.
    for j in (POOLED, NEG_DETECTED):
        unit = 80 + 2 * j
        diff(w[p + 'fc1'], j, unit, 1.0)
        diff(w[p + 'fc1'], j, unit + 1, -1.0)
        w[p + 'fc2'][unit, j] = -1.0
        w[p + 'fc2'][unit + 1, j] = 1.0

    p = 'block1.'
    for head, amount in ((0, 0.75), (2, 0.25)):
        base = head * 12
        w[p + 'q.bias'][base] = root12
        diff(w[p + 'k'], SEG, base, 30.0)
        diff(w[p + 'k'], SALIENCE, base, 2.0)
        if head == 2:
            diff(w[p + 'k'], EVIDENCE, base, 1.5)
        diff(w[p + 'v'], EVIDENCE, base, 1.0)
        w[p + 'o'][base, POOLED] = amount
    w[p + 'q.bias'][12] = root12
    diff(w[p + 'k'], SEG, 12, -30.0)
    diff(w[p + 'k'], NEGATION, 12, 20.0)
    diff(w[p + 'v'], NEGATION, 12, 1.0)
    w[p + 'o'][12, NEG_DETECTED] = 1.0

    gain, margin = 10.0, 16.0
    for unit, sign in ((0, 1.0), (1, -1.0)):
        diff(w[p + 'fc1'], POOLED, unit, sign * gain)
        diff(w[p + 'fc1'], NEG_DETECTED, unit, margin * gain)
        w[p + 'fc1.bias'][unit] = -0.5 * margin * gain
        w[p + 'fc2'][unit, POOLED] = -sign / gain

    w['ln_f.weight'] = torch.full((D,), LN_SCALE)
    w['ln_f.bias'] = torch.zeros(D)
    w['score'] = torch.zeros(D)
    w['score'][POOLED] = 1.0
    w['score'][ANCHOR_PLUS] = -0.5
    w['score'][ANCHOR_MINUS] = -0.5
    return w


if __name__ == '__main__':
    (ROOT / 'vocab.txt').write_text('\n'.join(vocabulary()) + '\n')

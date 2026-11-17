"""Compile a hand-written science knowledge graph into the fixed transformer.

No training, fitted parameters, datasets, or pretrained representations are used.
The SHA-256 projection is a fixed coordinate system, not a parameter search.
"""
from pathlib import Path
import hashlib
import math
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
D = 64
N = 28
# After pooling compensation the smallest valid anchor is ANCHOR / 49.
# 10000 keeps non-anchor energy below 0.03% of anchor energy even there.
ANCHOR = 10000.0
POS, NEG, EVID = 56, 58, 60


def entries():
    for line in (ROOT / 'knowledge.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        label, sources, targets = [s.strip() for s in line.split('|')]
        yield label, parse(sources), parse(targets)


def parse(text):
    out = []
    for term in text.split():
        bits = term.split(':')
        word = bits[0]
        weight = float(bits[1]) if len(bits) > 1 else 1.0
        assert re.fullmatch("[a-z0-9']+", word), word
        out.append((word, weight))
    return out


# Verb morphology is allowed only for these manually specified scientific/action
# lemmas. In particular, chemical 'base' and musical 'note' must not give their
# meanings to the generic discourse words 'based' and 'noted'. Irregular verbs
# use explicit families in lexemes.txt rather than guessed past-tense forms.
REGULAR_VERBS = set('''
    absorb accelerate accumulate adapt add adjust affect agree allow analyze apply
    attract avoid balance boil bounce burn calculate camouflage carry change
    classify clean close collide combine compare compete compress condense conduct
    conserve contain contract control convert cool copy crack cross crystallize
    decompose decrease depend deposit describe design detect develop diffuse digest
    disappear displace dissolve divide double drain drop dry emit enlarge enter
    erode evaporate expand experiment explain exhale fertilize filter fix float flow
    flower force form generate germinate glide heat hibernate hunt increase infect
    inherit insulate investigate ionize irrigate lift live look lower lubricate
    magnify measure melt migrate mimic mix model mold move multiply mutate observe
    occur open orbit organize oscillate pass perform photosynthesize pick place
    plan plant pollinate predict prevent produce propel protect pull pump push
    radiate rain react receive recycle reduce reflect refract release remove repair
    repel repeat reproduce research resist respond retain revolve roll rotate rub
    rust sample save scatter scratch select separate settle share shift slow solve
    sort specialize squeeze stop store stretch sublimate support surround taste test
    tilt trace transfer transform transport transpire trap travel turn use vary vibrate
    visit warm wash water weather work
'''.split())
DOUBLE_FINAL = {'control', 'drop', 'occur', 'plan', 'propel', 'repel', 'rub',
                'stop', 'transfer', 'trap'}


def forms(word):
    """Authored spelling rules, not a learned stemmer or corpus-derived analyzer."""
    yield word
    is_verb = word in REGULAR_VERBS
    if not word.isalpha() or (len(word) < 4 and not is_verb):
        return
    if len(word) >= 4:
        yield word + "'s"
    if word.endswith('y') and word[-2] not in 'aeiou':
        yield word[:-1] + 'ies'
    elif word.endswith(('ch', 'sh', 'x', 'ss')):
        yield word + 'es'
    elif not word.endswith('s'):
        yield word + 's'
    if not is_verb:
        return
    if word in DOUBLE_FINAL:
        yield word + word[-1] + 'ed'
        yield word + word[-1] + 'ing'
    elif word.endswith('c'):
        yield word + 'ked'
        yield word + 'king'
    elif word.endswith('ie'):
        yield word + 'd'
        yield word[:-2] + 'ying'
    elif word.endswith('e'):
        yield word + 'd'
        yield (word if word.endswith(('ee', 'ye', 'oe')) else word[:-1]) + 'ing'
    elif word.endswith('y') and word[-2] not in 'aeiou':
        yield word[:-1] + 'ied'
        yield word + 'ing'
    else:
        yield word + 'ed'
        yield word + 'ing'


NEGATIONS = {
    'not', 'except', 'least', 'incorrect', 'incorrectly', 'false', 'untrue',
    "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't",
    'cannot', "can't", "won't", "wouldn't", "shouldn't", "couldn't",
    "hasn't", "haven't", "hadn't",
}


def lexical_families():
    for line in (ROOT / 'lexemes.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            yield line.split()


def lexical_index():
    """Join known inflections before compiling relations, without estimating weights."""
    rows = list(entries())
    assert len({label for label, _, _ in rows}) == len(rows), 'Duplicate relation label'
    families = list(lexical_families())
    base = {word for _, s, t in rows for word, _ in s + t} | NEGATIONS
    base.update(word for family in families for word in family)
    # Recover an authored regular lemma even if the graph only mentions an
    # inflection, e.g. 'carried' -> 'carry', 'dropping' -> 'drop'. This is exact
    # application of the spelling rules, not inference from an external corpus.
    base.update({verb for verb in REGULAR_VERBS if any(form in base for form in forms(verb))})
    parent = {word: word for word in base}

    def root(word):
        while parent[word] != word:
            parent[word] = parent[parent[word]]
            word = parent[word]
        return word

    def join(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    for word in sorted(base - NEGATIONS):
        for variant in forms(word):
            if variant in base and variant not in NEGATIONS:
                join(word, variant)
    for family in families:
        for word in family[1:]:
            join(family[0], word)
    canonical = {word: root(word) for word in sorted(base)}
    return rows, canonical


def vocabulary():
    _, canonical = lexical_index()
    return sorted({v for word in canonical for v in forms(word)})


def concept_code(label):
    # Fixed bounded coordinates obtained directly from a hash (not Gaussian).
    raw = hashlib.sha256(('science-graph-v1:' + label).encode()).digest()
    u = (np.frombuffer(raw[:28], dtype=np.uint8).astype(np.float64) + 0.5) / 256.0
    # A bounded, zero-mean projection avoids dimension-dependent biases.
    vec = np.cos(2.0 * math.pi * u)
    vec -= vec.mean()
    return vec / np.linalg.norm(vec)


# Do not turn grammar and quantifiers in a definition into content-bearing queries.
DEFINITION_STOP = set('''
    a an the of and or as at for by with from in into on out up down to
    is are was were be been being do does did have has had this that these those
    it its they their them one two three four five six seven eight nine ten
    same different more less most least much many some all any each both no not
    only very than then when where which what who how can could will would
    should may might
'''.split())


def make_embeddings(vocab):
    rows, canonical = lexical_index()
    roots = {root: np.zeros((2, N), dtype=np.float64) for root in canonical.values()}
    for label, sources, targets in rows:
        code = concept_code(label)
        # A named definition supports both term -> properties and properties -> term.
        # The half-strength reverse direction is specified, not estimated from data.
        # Multiword/internal relation labels are deliberately not treated as words.
        if label in canonical and any(canonical[word] == canonical[label]
                                      for word, _ in sources):
            sources = sources + [(word, 0.5 * weight) for word, weight in targets
                                 if word not in DEFINITION_STOP]
            targets = targets + [(label, 1.0)]
        for side, terms in enumerate((sources, targets)):
            memberships = {}
            for word, weight in terms:
                lemma = canonical[word]
                previous = memberships.get(lemma, 0.0)
                assert previous * weight >= 0, (label, lemma, 'conflicting signs')
                # Listing both 'electron' and 'electrons' must not double the fact.
                if abs(weight) > abs(previous):
                    memberships[lemma] = weight
            for lemma, weight in memberships.items():
                roots[lemma][side] += weight * code
    nodes = {word: roots[lemma] for word, lemma in canonical.items()}
    inherited = {}
    for word, lemma in canonical.items():
        for variant in forms(word):
            if variant not in nodes:
                inherited.setdefault(variant, set()).add(lemma)
    for word, lemmas in inherited.items():
        nodes[word] = np.mean([roots[lemma] for lemma in sorted(lemmas)], axis=0)
    emb = np.zeros((len(vocab) + 2, D), dtype=np.float32)
    emb[:, 62], emb[:, 63] = ANCHOR, -ANCHOR
    for i, word in enumerate(vocab, 2):
        if word in nodes:
            for side in range(2):
                vec = nodes[word][side]
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    # Bounded-norm lexical vectors keep common terms from dominating.
                    emb[i, side * N:(side + 1) * N] = vec / max(1.0, norm)
        if word in NEGATIONS:
            emb[i, NEG], emb[i, NEG + 1] = 1.0, -1.0
    return torch.from_numpy(emb)


def build():
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    if vocab != vocabulary():
        raise ValueError('vocab.txt is stale or invalid; regenerate it with build.py')
    w = {'embed': make_embeddings(vocab), 'pos': torch.zeros(176, D)}
    w['pos'][:128, POS] = 3.0
    w['pos'][:128, POS + 1] = -3.0
    w['pos'][128:, POS] = -3.0
    w['pos'][128:, POS + 1] = 3.0
    ln_gain = ANCHOR / math.sqrt(32.0)
    for b in range(2):
        p = f'block{b}.'
        for ln in ('ln1.', 'ln2.'):
            w[p + ln + 'weight'] = torch.full((D,), ln_gain)
            w[p + ln + 'bias'] = torch.zeros(D)
        for name in ('q', 'k', 'v', 'o'):
            w[p + name] = torch.zeros(D, D)
            w[p + name + '.bias'] = torch.zeros(D)
        w[p + 'fc1'] = torch.zeros(D, 128)
        w[p + 'fc1.bias'] = torch.zeros(128)
        w[p + 'fc2'] = torch.zeros(128, D)
        w[p + 'fc2.bias'] = torch.zeros(D)
    w['ln_f.weight'] = torch.full((D,), ln_gain)
    w['ln_f.bias'] = torch.zeros(D)
    w['score'] = torch.zeros(D)
    w['score'][EVID], w['score'][EVID + 1] = 0.5, -0.5

    # Block 0 attention: a soft OR of question-side negation indicators.
    p = 'block0.'
    w[p + 'q.bias'][0] = 4.0  # cancels the attention sqrt(16) divisor
    # Read paired controls differentially: this cancels roundoff in the
    # layer-normalization mean instead of amplifying it in later arithmetic.
    w[p + 'k'][NEG, 0], w[p + 'k'][NEG + 1, 0] = 12.0, -12.0
    w[p + 'k'][POS, 0], w[p + 'k'][POS + 1, 0] = 10.0, -10.0
    w[p + 'v'][NEG, 0], w[p + 'v'][NEG + 1, 0] = 0.5, -0.5
    w[p + 'o'][0, EVID] = 1.0
    w[p + 'o'][0, EVID + 1] = -1.0

    # Block 0 head 1 counts the option fraction p = O / (Q + O).
    # Lexical banks are centered, so their two common modes can carry a scalar
    # without losing any of the 28-dimensional semantic representation.
    # Store +p/N in each query coordinate and -p/N in each answer coordinate.
    # The overall hidden-state mean stays zero; later projections remove these
    # common modes explicitly before computing semantic dot products.
    w[p + 'v'][POS, 16], w[p + 'v'][POS + 1, 16] = -1.0 / 12.0, 1.0 / 12.0
    w[p + 'v.bias'][16] = 0.5
    w[p + 'o'][16, :N] = 1.0 / N
    w[p + 'o'][16, N:2 * N] = -1.0 / N

    # FFN helpers. GELU(t)-GELU(-t)=t, exactly in real arithmetic.
    def linear_unit(block, j, inputs, output, bias=0.0):
        p = f'block{block}.'
        for dim, value in inputs.items():
            if dim in (POS, NEG, EVID):
                w[p + 'fc1'][dim, j] += value / 2.0
                w[p + 'fc1'][dim + 1, j] -= value / 2.0
            else:
                w[p + 'fc1'][dim, j] += value
        w[p + 'fc1.bias'][j] = bias
        for dim, value in output.items():
            w[p + 'fc2'][j, dim] = value

    # Remove centered answer-key vectors at question positions, preserving the
    # counting register in the bank's common mode. The large-margin GELU gates
    # act as linear/zero switches. Every lexical vector has zero bank mean.
    for d in range(N):
        for sign, j in ((1.0, 2 * d), (-1.0, 2 * d + 1)):
            inputs = {N + k: -sign * 8.0 / N for k in range(N)}
            inputs[N + d] += sign * 8.0
            inputs[POS] = 8.0
            linear_unit(0, j, inputs, {N + d: -sign / 16.0})
    # Replace the local negation bit by its broadcast version and clear scratch space.
    for sign, j in ((1.0, 56), (-1.0, 57)):
        linear_unit(0, j, {EVID: sign, NEG: -sign},
                    {NEG: sign, NEG + 1: -sign})
    for sign, j in ((1.0, 58), (-1.0, 59)):
        linear_unit(0, j, {EVID: sign},
                    {EVID: -sign, EVID + 1: sign})

    # Analytically specified log-odds correction, not regression or fitting.
    # At option positions add B(p) = log((1-p)/p) = log(Q/O) to the counting
    # register; at question positions add zero. A fixed linear interpolant of
    # log-odds uses 49 knots whose logits are equally spaced from -5 to 5.
    # Endpoint clamping keeps empty-region inputs finite. Every valid nonempty
    # pair of regions (Q <= 128, O <= 48) is inside the knot range.
    knot_logits = np.linspace(-5.0, 5.0, 49)
    knots = 1.0 / (1.0 + np.exp(-knot_logits))
    values = -knot_logits
    slopes = np.diff(values) / np.diff(knots)
    changes = np.diff(np.concatenate(([0.0], slopes, [0.0])))

    def count_output(scale):
        return {**{d: scale / N for d in range(N)},
                **{N + d: -scale / N for d in range(N)}}

    # Constant term, gated to options using their negative position marker.
    linear_unit(0, 60, {POS: -8.0}, count_output(float(values[0]) / 24.0))
    hinge_gain = 2048.0
    for j, (knot, change) in enumerate(zip(knots, changes), 61):
        # (sum(query bank) - sum(answer bank))/2 = p, invariant to a
        # shared layer-normalization mean offset in either semantic bank.
        inputs = {**{d: hinge_gain / 2.0 for d in range(N)},
                  **{N + d: -hinge_gain / 2.0 for d in range(N)}}
        inputs[POS] = -hinge_gain / 3.0
        # Option preactivation is gain*(p-knot); question preactivation has
        # another -2*gain, making every hinge inactive on question tokens.
        linear_unit(0, j, inputs, count_output(float(change) / hinge_gain),
                    bias=-hinge_gain * (1.0 + float(knot)))

    # Block 1: two centered semantic comparisons and their negative counterparts.
    # The common-mode key bias is p on questions and p+log(Q/O) on options.
    # The shared p cancels in softmax. At zero semantic similarity both regions
    # therefore receive equal total attention regardless of their token counts.
    # Positive-minus-negative attention then supplies signed, balanced evidence.
    p = 'block1.'
    for head in range(4):
        half, sign = head % 2, (1.0 if head < 2 else -1.0)
        start = head * 16
        for d in range(14):
            coord = half * 14 + d
            # Project out the counting register, retaining exactly the original
            # centered lexical coordinates in each semantic attention head.
            w[p + 'q'][:N, start + d] = -4.0 * sign / N
            w[p + 'q'][coord, start + d] += 4.0 * sign
            w[p + 'k'][N:2 * N, start + d] = -6.0 / N
            w[p + 'k'][N + coord, start + d] += 6.0
        w[p + 'q.bias'][start + 14] = 4.0
        w[p + 'k'][:N, start + 14] = 0.5
        w[p + 'k'][N:2 * N, start + 14] = -0.5
        w[p + 'v'][POS, start], w[p + 'v'][POS + 1, start] = -1.0 / 12.0, 1.0 / 12.0
        w[p + 'v.bias'][start] = 0.5
        w[p + 'o'][start, EVID] = sign * 0.5
        w[p + 'o'][start, EVID + 1] = -sign * 0.5

    # Replace raw evidence by question-only evidence, reversing it for negated queries.
    for sign, j in ((1.0, 0), (-1.0, 1)):
        linear_unit(1, j, {EVID: sign}, {EVID: -sign, EVID + 1: sign})
    for sign, j in ((1.0, 2), (-1.0, 3)):
        linear_unit(1, j, {EVID: sign * 8.0, POS: 32.0 / 3.0, NEG: -80.0},
                    {EVID: sign / 16.0, EVID + 1: -sign / 16.0})
    for sign, j in ((1.0, 4), (-1.0, 5)):
        linear_unit(1, j, {EVID: sign * 8.0, POS: 32.0 / 3.0, NEG: 80.0},
                    {EVID: -sign / 16.0, EVID + 1: sign / 16.0}, bias=-80.0)

    # Compensate the architecture's final mean pooling without changing arch.py.
    # On question tokens (sum(query bank)-sum(answer bank))/2 = p = O/(Q+O),
    # since the option-only correction is absent. Set anchors to +/-A*(1-p).
    # Final layer normalization consequently scales question evidence by
    # approximately 1/(1-p) = (Q+O)/Q. Its contribution after mean pooling is
    # therefore the mean over question tokens, rather than over Q+O tokens.
    # The +/-8 positional gate suppresses this update at option positions:
    # their common-mode register can contain p+log(Q/O), but its magnitude is
    # below 8 even at the clamped endpoints. Options retain their large anchors.
    for sign, j in ((1.0, 6), (-1.0, 7)):
        inputs = {**{d: sign * 2.0 for d in range(N)},
                  **{N + d: -sign * 2.0 for d in range(N)}}
        inputs[POS] = 32.0 / 3.0
        linear_unit(1, j, inputs,
                    {62: -sign * ANCHOR / 8.0, 63: sign * ANCHOR / 8.0})
    return w


if __name__ == '__main__':
    (ROOT / 'vocab.txt').write_text('\n'.join(vocabulary()) + '\n')
    print('Wrote deterministic vocabulary.')

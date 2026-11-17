"""A completely hand-authored, untrained lexical science transformer.

Four attention heads are four independent associative dictionaries.  The
embedding stores a circular address in each dictionary, not a learned vector.
The first block measures question/answer associations.  Its feed-forward
network removes the question's own score.  The second block broadcasts a
negation flag and conditionally reverses the association score.
"""
from pathlib import Path
import math
import re
import torch

ROOT = Path(__file__).resolve().parent
D = 16
# An exact power-of-two anchor keeps the base variance exactly representable.
# Its magnitude bounds non-anchor variance effects to roughly float32 precision,
# important when a circular-address match subtracts large attention logits.
ANCHOR = 8192.0
LN_SCALE = ANCHOR / math.sqrt(8.0)
MATCH_SCALE = 1600.0
MATCH_OFFSET = 1592.0
ROLE_MARGIN = 5000.0


def variants(word):
    """A small, fixed spelling rule, not corpus-derived morphology."""
    yield word
    if len(word) > 3 and word.isalpha():
        if word.endswith('y') and word[-2] not in 'aeiou':
            yield word[:-1] + 'ies'
        elif word.endswith(('ch', 'sh', 'ss', 'x')):
            yield word + 'es'
        elif not word.endswith(('s', 'ing', 'ed', 'ic', 'al', 'ous', 'ive')):
            yield word + 's'


def knowledge():
    """Read only the authored association tables, never questions or labels."""
    groups = [[] for _ in range(4)]
    head = None
    for raw in (ROOT / 'knowledge.txt').read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            head = int(line.strip('[]'))
            continue
        label, text = line.split(':', 1)
        assert head is not None
        words = text.split()
        assert all(re.fullmatch(r"[a-z0-9']+", x) for x in words), label
        groups[head].append((label, words))
    return groups


# Small truthfulness/safety priors. These are deliberately much weaker than
# even a single successful dictionary association.
PRIORS = {
    'always': -.12, 'never': -.12, 'only': -.06, 'instantly': -.12,
    'immediately': -.025, 'completely': -.055, 'nothing': -.07,
    'unlimited': -.08, 'identical': -.025, 'everywhere': -.035,
    'unprotected': -.10, 'tasting': -.08, 'taste': -.035,
    'carefully': .035, 'goggles': .055, 'gloves': .045,
    'washing': .04, 'wash': .035, 'accurate': .025,
    'repeated': .02, 'repeat': .02, 'evidence': .025,
    'all': .015, 'both': .015, 'variety': .015,
}
UP = set('increase increases increased increasing greater larger higher more faster fast hotter hot heating heated warmer warm expand expands expanded expanding expansion rise rises rising gain gains gaining longer strongest stronger'.split())
DOWN = set('decrease decreases decreased decreasing smaller lower less fewer slower slow colder cold cooling cooled cooler cool contract contracts contracted contracting contraction sink sinks sinking lose loses losing shorter weakest weaker'.split())

# A direct sign-agreement heuristic is unjustified for ratios, elapsed time,
# resistance, and distance-dependent laws. Abstain from that *small bonus* in
# these question contexts rather than pretending to know the causal equation.
CAUTIOUS = {'inverse', 'inversely'}
for _quantity in ('density', 'volume', 'mass', 'distance', 'duration',
                  'time', 'pressure', 'resistance', 'wavelength'):
    CAUTIOUS.update(variants(_quantity))

NEGATION = {
    'not', 'except', 'least', 'incorrect', 'incorrectly', 'false',
    'avoid', 'avoided', 'unsafe', 'disprove', 'disproves',
    "isn't", "aren't", "doesn't", "cannot", "can't",
}


def aliases():
    """Explicit spelling/inflection equivalences, not inferred from examples."""
    result = []
    for raw in (ROOT / 'aliases.txt').read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        canonical, text = line.split(':', 1)
        forms = text.split()
        assert re.fullmatch(r"[a-z0-9']+", canonical) and forms
        assert all(re.fullmatch(r"[a-z0-9']+", form) for form in forms)
        result.append((canonical, forms))
    return result


# Deliberate senses for ambiguous general words. In particular, measurement
# words must not be captured by the first incidental physical association.
SENSES = {
    1: {
        'mercury_fact': 'mercury innermost closest smallest',
        'venus_fact': 'venus hottest retrograde',
        'earth_fact': 'earth habitable life',
        'mars_fact': 'mars red reddish rusty ironoxide',
        'jupiter_fact': 'jupiter largest biggest giant',
        'saturn_fact': 'saturn rings ring',
        'uranus_fact': 'uranus sideways',
        'neptune_fact': 'neptune farthest outermost',
    },
    2: {
        'stratosphere': 'stratosphere ozone ultraviolet uv',
    },
    3: {
        # Preserve particle definitions instead of incidental circuit/repulsion
        # senses. Other heads retain electricity, magnets, and acid/base senses.
        'protons': 'proton protons positive positively',
        'electrons': 'electron electrons negative negatively',
        'neutrons': 'neutron neutrons neutral uncharged',
        'mass': 'mass gram grams kilogram kilograms balance scale',
        'volume': 'volume cubic milliliter milliliters liter liters cylinder',
        'temperature': 'temperature thermometer thermometers degree degrees celsius fahrenheit kelvin',
        'length': 'length distance meter meters centimeter centimeters kilometer kilometers ruler meterstick',
        'particlesolid': 'solid solids rigid tightly',
        'particleliquid': 'liquid liquids flow flowing sliding',
        'particlegas': 'gas gases gaseous compressible freely',
        'current': 'current ampere amperes amps',
        'voltage': 'voltage volt volts battery batteries',
        'resistance': 'resistance resistor resistors ohm ohms',
        'speed': 'speed velocity fast faster',
        'force': 'force forces push pushes pushing pull pulls pulling newton newtons',
        'heatflow': 'heat thermal hot warmer cold cooler',
    }
}


def vocabulary():
    result = set(PRIORS) | NEGATION | UP | DOWN | CAUTIOUS
    for head in knowledge():
        for _, words in head:
            for word in words:
                result.update(variants(word))
    for _, forms in aliases():
        for form in forms:
            result.update(variants(form))
    return sorted(result)


def build():
    torch.set_num_threads(4)
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    assert len(set(vocab)) == len(vocab)
    assert set(vocab) == set(vocabulary()), 'Regenerate vocab.txt after editing the lexicon'
    assert all(re.fullmatch(r"[a-z0-9']+", word) for word in vocab)
    index = {word: i + 2 for i, word in enumerate(vocab)}
    w = {'embed': torch.zeros(len(vocab) + 2, D),
         'pos': torch.zeros(176, D)}
    e = w['embed']
    # Dominant, symmetric anchors make layer normalization essentially a known
    # linear map. Channel 15 is the zero reference through the second attention
    # projection. The first MLP then frees channel 11 as the final zero reference.
    e[:, 12] = ANCHOR
    e[:, 13] = -ANCHOR
    w['pos'][:128, 14] = 1.0
    groups = knowledge()
    for head, table in enumerate(groups):
        used = set()
        for number, (_, words) in enumerate(table):
            angle = 2.0 * math.pi * number / len(table)
            a, b = math.cos(angle), math.sin(angle)
            for word in words:
                for form in variants(word):
                    # An address has one meaning per dictionary. Explicit
                    # earlier definitions have precedence over spelling aliases.
                    if form in used or form not in index:
                        continue
                    used.add(form)
                    row = index[form]
                    e[row, head * 2] = a
                    e[row, head * 2 + 1] = b
                    e[row, 11] = 1.0
    for head, senses in SENSES.items():
        labels = {label: number for number, (label, _) in enumerate(groups[head])}
        assigned = {}
        for label, words in senses.items():
            angle = 2 * math.pi * labels[label] / len(groups[head])
            for word in words.split():
                # Apply the same fixed spelling rule as the underlying tables;
                # otherwise plurals can retain an incidental earlier sense.
                for form in variants(word):
                    assert form not in assigned or assigned[form] == label, (
                        f'Conflicting sense overrides for {form!r} in head {head}'
                    )
                    assigned[form] = label
                    row = index[form]
                    e[row, 2 * head] = math.cos(angle)
                    e[row, 2 * head + 1] = math.sin(angle)
                    e[row, 11] = 1.0
    for canonical, forms in aliases():
        assert canonical in index, f'Unknown alias source: {canonical}'
        source = index[canonical]
        for form in forms:
            for spelling in variants(form):
                row = index[spelling]
                for head in range(4):
                    start = 2 * head
                    # An explicit dictionary sense is never overwritten by an
                    # alias. Aliases also fill missing heads of known spellings.
                    if not e[row, start:start + 2].count_nonzero():
                        e[row, start:start + 2] = e[source, start:start + 2]
                e[row, 11] = max(e[row, 11], e[source, 11])
    for word, prior in PRIORS.items():
        if word in index:
            e[index[word], 8] = prior
    for word in NEGATION:
        e[index[word], 10] = 1.0
    for word in UP:
        e[index[word], 9] = 1.0
    for word in DOWN:
        e[index[word], 9] = -1.0
    assert not CAUTIOUS.intersection(UP | DOWN)
    assert not NEGATION.intersection(UP | DOWN), 'Negation tokens must have zero signed direction'
    for word in CAUTIOUS:
        # Pack a distinct categorical control into the direction coordinate;
        # the first MLP decodes it before any direction pooling takes place.
        e[index[word], 9] = 2.0

    for layer in range(2):
        p = f'block{layer}.'
        for norm in ('ln1', 'ln2'):
            w[p + norm + '.weight'] = torch.full((D,), LN_SCALE)
            w[p + norm + '.bias'] = torch.zeros(D)
        for name in ('q', 'k', 'v', 'o'):
            w[p + name] = torch.zeros(D, D)
            w[p + name + '.bias'] = torch.zeros(D)
        w[p + 'fc1'] = torch.zeros(D, 32)
        w[p + 'fc1.bias'] = torch.zeros(32)
        w[p + 'fc2'] = torch.zeros(32, D)
        w[p + 'fc2.bias'] = torch.zeros(D)

    def project(matrix, source, target, factor, reference=15):
        # Subtract a zero scratch state, rather than the large anchors: this
        # removes centering without cancellation between large matrix products.
        matrix[source, target] += factor
        matrix[reference, target] -= factor

    p = 'block0.'
    for head in range(4):
        base = 4 * head
        # Analytically separate neighboring dictionary addresses: their logits
        # differ by at least 24, independent of the authored table's length.
        spacing = 2.0 * math.pi / len(groups[head])
        scale = max(MATCH_SCALE, 24.0 / (1.0 - math.cos(spacing)))
        offset = scale - (MATCH_SCALE - MATCH_OFFSET)
        for j in range(2):
            project(w[p + 'q'], 2 * head + j, base + j, math.sqrt(2 * scale))
            project(w[p + 'k'], 2 * head + j, base + j, math.sqrt(2 * scale))
        w[p + 'q.bias'][base + 2] = 2.0
        # All option keys receive a large negative offset. Question function
        # words have key=0 and value=0: the softmax's "no association" baseline.
        project(w[p + 'k'], 11, base + 2, -offset)
        project(w[p + 'k'], 14, base + 2, ROLE_MARGIN)
        w[p + 'k.bias'][base + 2] = -ROLE_MARGIN
        # Question queries attend only to the question, keeping their raw score
        # independent of the option; the MLP will then erase that score.
        project(w[p + 'q'], 14, base + 3, 2.0)
        project(w[p + 'k'], 14, base + 3, ROLE_MARGIN)
        project(w[p + 'v'], 11, base, 1.0)
        w[p + 'o'][base, 8] = 1.0

    # Subtract s on question tokens, and zero on option tokens, using the
    # difference of two GELUs. The inputs are far into their linear regions.
    # G(s,c) = [gelu(10(s+10(c-.5)))-gelu(100(c-.5))]/10.
    def gated_score(layer, control, multiplier, score_channel=8):
        prefix = f'block{layer}.'
        reference = 11 if layer == 1 else 15
        project(w[prefix + 'fc1'], score_channel, 0, 10.0, reference)
        for unit in (0, 1):
            project(w[prefix + 'fc1'], control, unit, 100.0, reference)
            w[prefix + 'fc1.bias'][unit] = -50.0
        w[prefix + 'fc2'][0, score_channel] = multiplier / 10.0
        w[prefix + 'fc2'][1, score_channel] = -multiplier / 10.0

    gated_score(0, 14, -1.0)
    # Clear the *answer's* local negation feature before question negation is
    # broadcast. Otherwise an answer containing 'not' would invert just itself.
    project(w[p + 'fc1'], 10, 2, 10.0)
    for unit in (2, 3):
        project(w[p + 'fc1'], 14, unit, -100.0)
        w[p + 'fc1.bias'][unit] = 50.0
    w[p + 'fc2'][2, 10] = -.1
    w[p + 'fc2'][3, 10] = .1
    # Importance has served its purpose. Recycle it into a near-zero scratch
    # coordinate using GELU(z)-GELU(-z)=z; residual error is only LN scale error.
    project(w[p + 'fc1'], 11, 4, 1.0)
    project(w[p + 'fc1'], 11, 5, -1.0)
    w[p + 'fc2'][4, 11] = -1.0
    w[p + 'fc2'][5, 11] = 1.0
    # Decode packed value 2: remove it from the signed direction feature, and
    # tag only *question* occurrences by a small offset in the role channel.
    # Values -1, 0, +1 remain untouched; two far-from-boundary GELUs suffice.
    project(w[p + 'fc1'], 9, 6, 100.0)
    w[p + 'fc1.bias'][6] = -150.0
    w[p + 'fc2'][6, 9] = -.04
    project(w[p + 'fc1'], 9, 7, 100.0)
    project(w[p + 'fc1'], 14, 7, 100.0)
    w[p + 'fc1.bias'][7] = -250.0
    w[p + 'fc2'][7, 14] = .0002

    # A high-key negation token is retrieved if it occurs in the question.
    # Negations inside an answer do not change the question's polarity.
    p = 'block1.'
    w[p + 'q.bias'][0] = 2.0
    project(w[p + 'k'], 10, 0, 30.0)
    project(w[p + 'k'], 14, 0, 80.0)
    project(w[p + 'v'], 10, 0, 1.0)
    w[p + 'o'][0, 10] = 1.0
    # Cross-boundary direction pooling: question queries average the option,
    # and option queries average the question. A second value component returns
    # the opposing role. Adding it and subtracting 1 cancels the residual role,
    # so coordinate 14 holds .1*opposing_mean on both sides (before caution).
    # For nonempty uncautioned q/o, the final mean of u*opposing_mean equals
    # mean(q_direction)*mean(o_direction), independent of either token count.
    w[p + 'q.bias'][4] = 2.0
    project(w[p + 'q'], 14, 4, -4.0)
    project(w[p + 'k'], 14, 4, 80.0)
    project(w[p + 'v'], 9, 4, 1.0)
    project(w[p + 'v'], 14, 5, 1.0)
    w[p + 'o'][4, 14] = .1
    w[p + 'o'][5, 14] = 1.0
    w[p + 'o.bias'][14] = -1.0
    # Pool the option's mean score and broadcast it to every real token.
    # Reading only this channel removes question-token dilution and an implicit
    # preference for answer verbosity. All options use the identical operation.
    w[p + 'q.bias'][8] = 2.0
    project(w[p + 'k'], 14, 8, -80.0)
    project(w[p + 'v'], 8, 8, 1.0)
    w[p + 'o'][8, 15] = 1.0
    # The previously unused head retrieves a cautious-context role tag. Real
    # question words have role 1 or 1.01; a 100-logit difference selects a tag
    # if present. Its .01 value produces a .4 offset in the direction gate.
    w[p + 'q.bias'][12] = 2.0
    project(w[p + 'k'], 14, 12, 10000.0)
    project(w[p + 'v'], 14, 12, 1.0)
    w[p + 'v.bias'][12] = -1.0
    w[p + 'o'][12, 14] = 40.0
    gated_score(1, 10, -2.0, score_channel=15)

    # A small qualitative-direction bonus: increased temperature / faster
    # particles, cooling / contraction, etc. Not a general causal simulator.
    # f(t)=R(t)-R(-t)-2.5*R(t-.15)+1.5*R(t-.25), where R is ReLU.
    # Thus f(t)=t for |t|<=.1 and f(t)=0 for t>=.25. Cautious contexts place
    # all real-token t in [.3,.51]. Each hinge is gated by u=sign using a
    # NEGATIVE shift for inactive selectors, not a positive linear offset:
    # the latter would erase the hinge and cancel all terms identically.
    # 1024 is a fixed power-of-two ReLU approximation scale, not a fitted gain.
    relu_scale = 1024.0
    unit = 2
    hinges = ((1., 0., 1.), (-1., 0., -1.),
              (1., .15, -2.5), (1., .25, 1.5))
    for negative_only, correction in ((False, 1.0), (True, -2.0)):
        for sign in (1.0, -1.0):
            for slope, threshold, coefficient in hinges:
                project(w[p + 'fc1'], 14, unit, slope * relu_scale, reference=11)
                project(w[p + 'fc1'], 9, unit, 2 * sign * relu_scale, reference=11)
                w[p + 'fc1.bias'][unit] = -(2 + threshold) * relu_scale
                if negative_only:
                    # AND with retrieved negation. A question negation token
                    # can have control 2 but has u=0, so its two signs cancel.
                    project(w[p + 'fc1'], 10, unit, 2 * relu_scale, reference=11)
                    w[p + 'fc1.bias'][unit] -= 2 * relu_scale
                w[p + 'fc2'][unit, 15] = 10 * correction * sign * coefficient / relu_scale
                unit += 1
    assert unit <= 32

    # Read only the broadcast answer score, using recycled scratch coordinate
    # 11 to cancel final layer normalization's centering.
    w['ln_f.weight'] = torch.full((D,), LN_SCALE)
    w['ln_f.bias'] = torch.zeros(D)
    w['score'] = torch.zeros(D)
    w['score'][15] = 1.0
    w['score'][11] = -1.0
    return w


if __name__ == '__main__':
    # This only enumerates the authored lexicon; it does not train a model.
    (ROOT / 'vocab.txt').write_text('\n'.join(vocabulary()) + '\n')
    print(f'Wrote {len(vocabulary())} vocabulary entries.')

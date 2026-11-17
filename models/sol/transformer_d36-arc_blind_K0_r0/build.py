"""An entirely hand-designed, untrained semantic-association transformer.

The circuit is specified here; the hand-written facts and lexical families
are in knowledge.tsv, extra_facts.tsv, and lexical_families.txt.
There is no optimizer, data loading, model download, or accuracy-based selection.
"""
from pathlib import Path
import hashlib
import math
import re

import torch

ROOT = Path(__file__).resolve().parent
D = 36
SEM = 24
SAL, NORM, REF, SEG, WORK, PRIOR, POOLPRIOR, NEGPOOL, NEG, OUT = range(24, 34)
BASE = 512.0
LN_SCALE = math.sqrt(2 * BASE * BASE / D)
QUADRATIC_STEP = 1 / 16
# For a content vector of squared norm 2.25, gain 8 gives exp(18).
# Even 47 zero-vector words then have total relative weight < 1e-6.
QUESTION_CONTENT_GAIN = 8.0
QUESTION_SEGMENT_GAIN = 28.0
ANSWER_CONTENT_GAIN = 8.0
ANSWER_SALIENCE_GAIN = 1.5
NEGATION_WORDS = frozenset("not except least incorrect incorrectly inaccurate false never cannot can't doesn't don't isn't aren't wasn't weren't won't wouldn't shouldn't couldn't".split())
# Function words may occur in fact descriptions, but must not acquire a
# scientific meaning merely from being part of those descriptions.
SEMANTIC_STOPWORDS = frozenset("""a an the this that these those which what when
where why how who whose is are was were be being been am do does did doing
has have had having of to from in into on onto at by with without for and
or but if then than as it its it's they their them we our us you your i me
my he she his her all any each every either neither some no none not only
never always own can could may might must should would will shall cannot
can't doesn't don't isn't aren't wasn't weren't won't wouldn't shouldn't
couldn't because therefore so during before after while between among
through over under within outside inside about around across also very
too really enough even once again such following most least best likely possible
probably usually generally often sometimes""".split())


def _lexical_families():
    families = []
    seen = set()
    for line in (ROOT / 'lexical_families.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        family = tuple(line.split())
        assert not (seen & set(family)), f'Overlapping lexical family: {family}'
        assert all(re.fullmatch(r"[a-z0-9']+", word) for word in family)
        seen.update(family)
        families.append(family)
    return families


def _knowledge():
    entries = {}
    for filename in ('knowledge.tsv', 'extra_facts.tsv'):
        for line in (ROOT / filename).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tag, strength, text = [p.strip() for p in line.split('|')]
            strength = float(strength)
            for word in text.split():
                assert re.fullmatch(r"[a-z0-9']+", word), word
                if word not in SEMANTIC_STOPWORDS:
                    entries.setdefault(word, {})[tag] = strength
    # Orthographic forms and irregular plurals share the complete union of
    # their hand-written associations, not just one spelling's partial facts.
    for family in _lexical_families():
        combined = {}
        for word in family:
            for tag, strength in entries.get(word, {}).items():
                if tag in combined:
                    assert combined[tag] == strength, (family, tag)
                combined[tag] = strength
        assert combined, f'Lexical family has no hand-written facts: {family}'
        for word in family:
            entries[word] = dict(combined)
    # Conservative, deterministic spelling coverage. Existing explicit entries
    # always take precedence; no corpus or observed questions are involved.
    explicit = set(entries)
    for word, groups in list(entries.items()):
        if len(word) < 4 or not word.isalpha():
            continue
        variants = set()
        if word.endswith('y') and word[-2] not in 'aeiou':
            variants.update((word[:-1] + 'ies', word[:-1] + 'ied'))
        elif word.endswith(('ch', 'sh', 'x', 'z')):
            variants.add(word + 'es')
        elif not word.endswith('s'):
            variants.add(word + 's')
        if word.endswith('e'):
            variants.update((word + 'd', word[:-1] + 'ing'))
        elif not word.endswith(('s', 'y', 'ing', 'ed')):
            variants.update((word + 'ed', word + 'ing'))
        for variant in sorted(variants - explicit):
            target = entries.setdefault(variant, {})
            for tag, strength in groups.items():
                target.setdefault(tag, strength)
    return entries


def _direction(tag):
    # A fixed, data-independent random projection, not learned embeddings.
    digest = hashlib.sha256(('science-hand-circuit:' + tag).encode()).digest()
    return torch.tensor([1.0 if (digest[i // 8] >> (i % 8)) & 1 else -1.0
                         for i in range(SEM)]) / math.sqrt(SEM)


def build():
    knowledge = _knowledge()
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    assert len(set(vocab)) == len(vocab)
    w = {'embed': torch.zeros(len(vocab) + 2, D), 'pos': torch.zeros(176, D)}
    for block in range(2):
        p = f'block{block}.'
        for ln in ('ln1', 'ln2'):
            w[p + ln + '.weight'] = torch.full((D,), LN_SCALE)
            w[p + ln + '.bias'] = torch.zeros(D)
        for name in ('q', 'k', 'v', 'o'):
            w[p + name] = torch.zeros(D, D)
            w[p + name + '.bias'] = torch.zeros(D)
        w[p + 'fc1'] = torch.zeros(D, 72)
        w[p + 'fc1.bias'] = torch.zeros(72)
        w[p + 'fc2'] = torch.zeros(72, D)
        w[p + 'fc2.bias'] = torch.zeros(D)
    w['ln_f.weight'] = torch.full((D,), LN_SCALE)
    w['ln_f.bias'] = torch.zeros(D)
    w['score'] = torch.zeros(D)

    e = w['embed']
    e[:, 34], e[:, 35] = BASE, -BASE
    e[:, SAL] = -1.0
    low_salience = set('energy matter object objects material materials substance substances thing things organism organisms animal animals plant plants earth water air living nonliving system systems process processes environment environments change changes example examples form forms type types kind kinds cause causes result results effect effects state states property properties physical chemical science scientific student students experiment experiments investigation investigations scientist scientists'.split())
    low_salience.update('one two first second third fourth both same different greater less more higher lower most make makes made making use uses used using measure measures measured measuring measurement measurements tool tools instrument instruments example examples shows show shown based following best likely possible usually normally general certain typical'.split())
    high_salience = set('renewable nonrenewable photosynthesis evaporation condensation precipitation freezing melting sublimation respiration transpiration pollination fertilization producer producers consumer consumers decomposer decomposers herbivore herbivores carnivore carnivores omnivore omnivores erosion weathering deposition convection conduction radiation gravity friction inertia magnetism magnetic eclipse eclipses rotation revolution rotate revolves orbit orbits sedimentary igneous metamorphic mitochondria chloroplast chloroplasts nucleus nuclei membrane membranes predator predators prey camouflage inherited acquired dominant recessive soluble insoluble soluble conductivity insulator insulators conductor conductors hypothesis variable variables controlled independent dependent control extinct extinction endangered adaptation adaptations renewable fossil fossils constellation constellations electricity electrical circuit circuits covalent ionic proton protons neutron neutrons electron electrons earthquake earthquakes volcano volcanoes tectonic'.split())
    high_salience.update('osmosis diffusion enzyme enzymes catalyst catalysts endothermic exothermic homogeneous heterogeneous refraction reflection wavelength frequency amplitude density buoyancy neutralization fusion fission eukaryote eukaryotic prokaryote prokaryotic symbiosis mutualism parasitism commensalism biodiversity'.split())
    positive_prior = {
        'evidence': .11, 'data': .06, 'observations': .07, 'observe': .05,
        'measure': .05, 'measuring': .05, 'repeat': .09, 'repeated': .07,
        'reliable': .07, 'accurate': .07, 'accuracy': .06, 'controlled': .05,
        'compare': .04, 'comparison': .04, 'testable': .10,
        'energy': .015, 'organisms': .025, 'survive': .04, 'survival': .04,
        'reproduce': .04, 'reproduction': .035, 'protect': .035,
        'available': .025, 'diversity': .04, 'different': .015,
        'perpendicular': .025, 'covalent': .025,
    }
    negative_prior = {
        'always': -.20, 'never': -.22, 'only': -.13, 'completely': -.10,
        'nothing': -.15, 'instantly': -.14, 'impossible': -.12,
        'identical': -.035, 'exactly': -.04, 'disappear': -.07,
        'disappears': -.07, 'unlimited': -.10, 'randomly': -.06,
        'guess': -.08, 'guesses': -.08, 'opinion': -.09, 'opinions': -.09,
        'believe': -.07, 'belief': -.07, 'magic': -.25,
    }
    # Lexical equivalence applies to metadata as well as concept coordinates.
    for family in _lexical_families():
        if high_salience.intersection(family):
            high_salience.update(family)
        elif low_salience.intersection(family):
            low_salience.update(family)
        for priors in (positive_prior, negative_prior):
            values = [priors[word] for word in family if word in priors]
            if values:
                assert all(value == values[0] for value in values), family
                priors.update(dict.fromkeys(family, values[0]))
    directions = {t: _direction(t) for groups in knowledge.values() for t in groups}
    for row, word in enumerate(vocab, 2):
        groups = knowledge.get(word, {})
        if groups:
            vec = sum((directions[t] * s for t, s in groups.items()), torch.zeros(SEM))
            # Equal bounded lengths prevent highly polysemous words dominating.
            length = float(vec.norm())
            if length > 1e-8:
                vec = vec * (1.5 / length)
                e[row, :SEM] = vec
                e[row, NORM] = vec.square().sum()
                # Supply the subtraction directly in the residual stream.
                # This frees all 72 first-layer MLP units for 24 squares.
                e[row, WORK] = -e[row, NORM]
                e[row, SAL] = 1.05
        if word in low_salience and groups:
            e[row, SAL] = .30
        if word in high_salience and groups:
            e[row, SAL] = 1.55
        e[row, NEG] = float(word in NEGATION_WORDS)
        e[row, PRIOR] = positive_prior.get(word, 0) + negative_prior.get(word, 0)
    w['pos'][:128, SEG] = 1.0
    w['pos'][128:, SEG] = -1.0
    # Later question tokens have a fixed 24-token attention half-life.
    # This concentrates on the actual query rather than a long preamble.
    w['pos'][:128, SAL] = torch.arange(128) * (math.log(2) / (3 * 24))

    def difference(matrix, column, source, coefficient):
        # Cancel layer norm's subtraction of the channel mean.
        matrix[source, column] += coefficient
        matrix[REF, column] -= coefficient

    # First attention: mean of QUESTION concepts, excluding zero-vector filler.
    # The wider segment gap prevents option leakage even for all-OOV questions.
    # All heads have identical attention, and transport six concept channels.
    p = 'block0.'
    for head in range(4):
        col = head * 9
        w[p + 'q.bias'][col] = 3.0  # cancels attention's sqrt(9)
        difference(w[p + 'k'], col, SAL, 3.0)
        difference(w[p + 'k'], col, NORM, QUESTION_CONTENT_GAIN)
        difference(w[p + 'k'], col, SEG, QUESTION_SEGMENT_GAIN)
        for j in range(6):
            source = head * 6 + j
            difference(w[p + 'v'], col + j, source, 1.0)
            w[p + 'o'][col + j, source] = 1.0

    # Let S_a(z) = (GELU(a*z)+GELU(-a*z))/(sqrt(2/pi)*a^2).
    # Richardson extrapolation (4*S_a-S_(2a))/3 cancels its quartic error.
    # GELU(t)-GELU(-t)=t reduces the four reads to THREE:
    # [3/2*GELU(a*z) + 7/6*GELU(-a*z) - 1/6*GELU(2*a*z)]
    #     / (sqrt(2/pi)*a^2) = z^2 - a^4*z^6/10 + O(a^6*z^8).
    # Use all 72 units. WORK already contains -||x||^2 from the embedding.
    a = QUADRATIC_STEP
    quadratic_scale = 1.0 / (math.sqrt(2 / math.pi) * a * a)
    for j in range(SEM):
        for side, (multiplier, coefficient) in enumerate(
                ((1.0, 3 / 2), (-1.0, 7 / 6), (2.0, -1 / 6))):
            unit = 3 * j + side
            difference(w[p + 'fc1'], unit, j, multiplier * a)
            w[p + 'fc2'][unit, WORK] = coefficient * quadratic_scale

    # The second layer broadcasts the complete option score to EVERY token.
    # Final mean pooling therefore does not dilute it by question length.
    p = 'block1.'
    for head in range(4):
        w[p + 'q.bias'][head * 9] = 3.0

    # Head 0: question-scoped negation, excluding negations in the option.
    difference(w[p + 'k'], 0, NEG, 24.0)
    difference(w[p + 'k'], 0, SEG, 28.0)
    difference(w[p + 'v'], 0, NEG, 1.0)
    w[p + 'o'][0, NEGPOOL] = 1.0

    # Head 1: reuse the FIRST layer's question weights. Since E[x] = y,
    # E[2*x dot y + ||y||^2] / 3 = ||y||^2. This offset estimate requires
    # neither a zero-concept token nor a special sentinel word.
    difference(w[p + 'k'], 9, SAL, 3.0)
    difference(w[p + 'k'], 9, NORM, QUESTION_CONTENT_GAIN)
    difference(w[p + 'k'], 9, SEG, QUESTION_SEGMENT_GAIN)
    difference(w[p + 'v'], 9, WORK, 1.0)
    w[p + 'o'][9, OUT] = -1.0 / 3.0

    # Head 2: salience-weighted OPTION content, not grammatical filler.
    # Known vectors have squared norm 2.25. Gain 8 suppresses zero-vector
    # dilution below 1e-6 even at the maximum 48-token option length.
    # Half the question salience gain emphasizes distinctive answer terms
    # without completely discarding broad but meaningful science words.
    difference(w[p + 'k'], 18, SEG, -28.0)
    difference(w[p + 'k'], 18, NORM, ANSWER_CONTENT_GAIN)
    difference(w[p + 'k'], 18, SAL, ANSWER_SALIENCE_GAIN)
    difference(w[p + 'v'], 18, WORK, 1.0)
    w[p + 'o'][18, OUT] = 1.0

    # Head 3: mean lexical prior, separately pooled so question negation
    # reverses the semantic comparison but not the answer-quality prior.
    difference(w[p + 'k'], 27, SEG, -28.0)
    difference(w[p + 'v'], 27, PRIOR, 1.0)
    w[p + 'o'][27, POOLPRIOR] = 1.0

    # OUT now holds 2 * (mean answer concept) dot y. Negated questions
    # receive -2*OUT as a gated correction. GELU pairs implement linear
    # reads exactly when their shared gate is large and positive.
    gate = 24.0
    for side, sign in enumerate((1.0, -1.0)):
        difference(w[p + 'fc1'], side, OUT, sign)
        difference(w[p + 'fc1'], side, NEGPOOL, 2 * gate)
        w[p + 'fc1.bias'][side] = -gate
        w[p + 'fc2'][side, OUT] = -sign
        difference(w[p + 'fc1'], 2 + side, POOLPRIOR, sign)
        w[p + 'fc2'][2 + side, OUT] = sign

    w['score'][OUT] = 1.0
    w['score'][REF] = -1.0
    return w

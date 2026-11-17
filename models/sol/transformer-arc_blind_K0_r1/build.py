"""An entirely hand-designed, untrained transformer science scorer.

Knowledge is in knowledge.txt and aliases.txt. No examples, labels, optimization,
pretrained arrays, or inference-time Python scoring are used.
"""
from pathlib import Path
import hashlib
import math
import re

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")

# Fixed algebraic circuit constants, selected by design, never fitted.
CONTEXT_MIX = 1 / 4
DOT_WEIGHT = 1 / 16
CURVE_STEP = 1 / 32

# These closed-class tokens supply null attention anchors, not science facts.
# Keep directional/comparative terms and negation controls outside this set.
GRAMMATICAL_ANCHORS = frozenset(
    "a an the and or of to in at by for from with which what who whom whose "
    "when where why how is are was were be been being do does did will would "
    "should could it its their they them this that these those then than as if "
    "because during".split()
)


def square_features(vector):
    """The exact smooth-square function realized by symmetric GELUs.

    phi(z) = sqrt(pi/2)/t**2 * (GELU(t*z) + GELU(-t*z)).
    It is approximately z**2 near zero. Storing its original-word value
    removes word-dependent curvature offsets without examples or fitting.
    """
    return math.sqrt(math.pi / 2) / CURVE_STEP * sum(
        float(z) * math.erf(CURVE_STEP * float(z) / math.sqrt(2))
        for z in vector
    )


def knowledge():
    aliases = {}
    spellings = {}
    for line in (HERE / "aliases.txt").read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        forms = line.split()
        if any(TOKEN.fullmatch(form) is None for form in forms):
            raise ValueError("Alias cannot be emitted by the tokenizer: " + line)
        root = forms[0]
        spellings.setdefault(root, set()).update(forms)
        for form in forms:
            if form in aliases and aliases[form] != root:
                raise ValueError("Conflicting alias: " + form)
            aliases[form] = root

    vectors = {}
    fact_names = set()
    for line in (HERE / "knowledge.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, strength, positive, negative = [s.strip() for s in line.split('|')]
        if name in fact_names or not math.isfinite(float(strength)) or float(strength) <= 0:
            raise ValueError("Invalid or duplicate fact: " + name)
        fact_names.add(name)
        # Independent, reproducible balanced codes, NOT fitted embeddings. Every
        # fact contributes equal energy to each of the three attention heads.
        raw = hashlib.shake_256(('science-fact:' + name).encode()).digest(45)
        code = np.array([1.0 if b & 1 else -1.0 for b in raw], dtype=np.float64)
        for start in (0, 15, 30):
            part = code[start:start + 15]
            part -= part.mean()
            norm = np.linalg.norm(part)
            if norm == 0:
                part[:] = np.arange(15) - 7
                norm = np.linalg.norm(part)
            part /= norm
        code *= float(strength)
        for text, sign in ((positive, 1), (negative, -1)):
            # A canonical concept contributes once per fact even when several
            # of its inflections are included in the human-readable row.
            roots = {aliases.get(word, word) for word in TOKEN.findall(text)}
            for root in roots:
                vectors.setdefault(root, np.zeros(45, dtype=np.float64))
                vectors[root] += sign * code

    words = {}
    for root, vector in vectors.items():
        # Equal per-head norms make null attention rates independent of the
        # particular answer word, and prevent high-degree generic words from
        # overwhelming specific words solely through vector magnitude.
        for start in (0, 15, 30):
            part = vector[start:start + 15]
            norm = np.linalg.norm(part)
            if norm > 1e-10:
                part *= math.sqrt(0.75) / norm
        for form in spellings.get(root, {root}):
            words[form] = vector.copy()

    negations = set("not never except least incorrect incorrectly false avoid avoids avoiding unlikely cannot can't doesn't don't isn't aren't wasn't weren't shouldn't wouldn't couldn't won't hasn't haven't hadn't mustn't inaccurate unsupported disproves disprove unrelated".split())
    priors = {'only': -0.015, 'always': -0.02, 'never': -0.02,
              'entirely': -0.012, 'completely': -0.012, 'may': 0.003,
              'can': 0.003, 'sometimes': 0.003}
    neutral = set("a an the and or of to in on at by for from with without which what when where why how is are was were be been being do does did will would should could it its their they them this that these those then than as if so because during into through over under between among most more less some any all each every one two three four first second third best likely example examples following shown describes describe statement statements explains explain answer answers question students student scientist scientists investigation experiment number amount process processes occurs occur happens happen results result causes cause change changes changing increases increase increased decreases decrease decreased large small high low higher lower greatest smallest largest least greatest enough much many same different".split())
    for word in negations | neutral | set(priors):
        root = aliases.get(word, word)
        for form in spellings.get(root, {word}):
            words.setdefault(form, np.zeros(45, dtype=np.float64))
    # Fact descriptions may contain incidental grammar, e.g. "in place".
    # setdefault above would preserve the resulting spurious science vector.
    # Explicitly clear these anchors and any equivalent alias forms instead.
    for word in GRAMMATICAL_ANCHORS:
        root = aliases.get(word, word)
        for form in spellings.get(root, {word}):
            words[form] = np.zeros(45, dtype=np.float64)
    return words, negations, priors


def write_vocab():
    words, _, _ = knowledge()
    (HERE / 'vocab.txt').write_text(''.join(w + '\n' for w in sorted(words)))


def build():
    words, negations, priors = knowledge()
    vocab = (HERE / 'vocab.txt').read_text().splitlines()
    if vocab != sorted(words):
        raise ValueError('Vocabulary is stale or malformed; regenerate with python build.py')
    w = {}
    def zeros(name, shape):
        w[name] = torch.zeros(shape, dtype=torch.float32)
        return w[name]

    embed = zeros('embed', (len(vocab) + 2, 64))
    # Fixed, zero-sum ballast makes layer normalization nearly a fixed scale.
    # Even unknown words have a well-defined state, but no scientific meaning.
    ballast = 100.0
    embed[:, 62] = ballast
    embed[:, 63] = -ballast
    option_negations = set("not never cannot can't doesn't don't isn't aren't wasn't weren't shouldn't wouldn't couldn't won't hasn't haven't hadn't mustn't".split())
    for i, word in enumerate(vocab, 2):
        vector = words.get(word)
        if vector is not None:
            embed[i, :45] = torch.tensor(vector, dtype=torch.float32)
            embed[i, 45] = float(np.linalg.norm(vector) > 1e-8)
            embed[i, 60] = square_features(vector)
        embed[i, 46] = float(word in negations)
        embed[i, 47] = priors.get(word, 0.0)
        embed[i, 54] = float(word in option_negations)
    pos = zeros('pos', (176, 64))
    pos[:128, 48] = -1
    pos[128:, 48] = 1

    # The non-ballast state has O(10) squared norm versus 20,000 here, so
    # all five layer normalizations stay close to an identity scale. This
    # controls baseline and broadcast errors analytically, without fitting.
    ln_scale = math.sqrt(2 * ballast * ballast / 64)
    for i in range(2):
        p = f'block{i}.'
        for ln in ('ln1', 'ln2'):
            w[p + ln + '.weight'] = torch.full((64,), ln_scale)
            zeros(p + ln + '.bias', (64,))
        for proj in ('q', 'k', 'v', 'o'):
            zeros(p + proj, (64, 64))
            zeros(p + proj + '.bias', (64,))
        zeros(p + 'fc1', (64, 128))
        zeros(p + 'fc1.bias', (128,))
        zeros(p + 'fc2', (128, 64))
        zeros(p + 'fc2.bias', (64,))

    def add_feature(matrix, input_dim, output_dim, value):
        # Difference from a reserved zero coordinate removes LN's mean.
        matrix[input_dim, output_dim] += value
        matrix[61, output_dim] -= value

    p = 'block0.'
    # Three independent views of the fact vectors. Option tokens query only
    # question tokens. Scientific keys pay a fixed log-prior; non-science
    # question words serve as a zero-valued null alternative.
    for head in range(3):
        offset = 16 * head
        for j in range(15):
            dim = 15 * head + j
            add_feature(w[p+'q'], dim, offset+j, math.sqrt(24))
            add_feature(w[p+'k'], dim, offset+j, math.sqrt(24))
            # Preserve the actual retrieved semantic vector as well as the
            # science-word attention mass. The residual becomes u + eps*a.
            add_feature(w[p+'v'], dim, offset+j, 1)
            w[p+'o'][offset+j, dim] = CONTEXT_MIX
        w[p+'q.bias'][offset+15] = 4
        add_feature(w[p+'k'], 48, offset+15, -24)
        add_feature(w[p+'k'], 45, offset+15, -2.5)
        add_feature(w[p+'v'], 45, offset+15, 1)
        w[p+'o'][offset+15, 56+head] = 1
    # Fourth head: existential question negation, independent of option text.
    w[p+'q.bias'][63] = 4
    add_feature(w[p+'k'], 48, 63, -24)
    add_feature(w[p+'k'], 46, 63, 20)
    add_feature(w[p+'v'], 46, 48, 1)
    w[p+'o'][48, 59] = 1

    # Consensus score = mean(heads) - (1/6) sum pairwise soft-absolute
    # differences. This discourages random one-head hash collisions.
    # GELU(z)-GELU(-z)=z exactly, so linear terms require only two neurons.
    f1, b1, f2 = w[p+'fc1'], w[p+'fc1.bias'], w[p+'fc2']
    for dim in (56, 57, 58):
        add_feature(f1, dim, 0, 1/3)
        add_feature(f1, dim, 1, -1/3)
    f2[0, 50], f2[1, 50] = 1, -1
    unit = 2
    temperature = 32.0
    for a, b in ((56, 57), (56, 58), (57, 58)):
        for sign in (1, -1):
            add_feature(f1, a, unit, sign * temperature)
            add_feature(f1, b, unit, -sign * temperature)
            f2[unit, 50] = -1 / (6 * temperature)
            unit += 1

    # Auxiliary similarity from the retrieved vectors, using previously unused
    # MLP neurons. The identity
    #   (||u + eps*a||^2 - ||u||^2)/(2*eps)
    #     = dot(u, a) + eps/2 * ||a||^2
    # supplies content sensitivity even when all question values have science
    # membership 1. Symmetric small-input GELUs realize a smooth square; the
    # original-word curvature offset is cached analytically in coordinate 60.
    gain = DOT_WEIGHT / (2 * CONTEXT_MIX)
    square_gain = gain * math.sqrt(math.pi / 2) / (CURVE_STEP ** 2)
    for dim in range(45):
        for sign in (1, -1):
            add_feature(f1, dim, unit, sign * CURVE_STEP)
            f2[unit, 50] = square_gain
            unit += 1
    add_feature(f1, 60, unit, 1)
    add_feature(f1, 60, unit+1, -1)
    f2[unit, 50], f2[unit+1, 50] = -gain, gain
    assert unit + 2 <= 128

    # Retrieve the null-query baseline computed in block zero. Restrict
    # attention to non-scientific question words, whose semantic query is zero.
    p = 'block1.'
    w[p+'q.bias'][15] = 4
    add_feature(w[p+'k'], 48, 15, -24)
    add_feature(w[p+'k'], 45, 15, -24)
    add_feature(w[p+'v'], 50, 0, 1)
    w[p+'o'][0, 51] = 1

    # Separately detect explicit negation INSIDE the option. Position gating
    # keeps option words from changing the question's exception flag.
    w[p+'q.bias'][31] = 4
    add_feature(w[p+'k'], 48, 31, 24)
    add_feature(w[p+'k'], 54, 31, 20)
    add_feature(w[p+'v'], 54, 16, 1)
    w[p+'o'][16, 55] = 1

    # Head 2 pools semantic evidence over meaningful OPTION words. Null words
    # have exp(-8) relative weight; if no word is known they still supply a
    # well-defined fallback. Uniform queries broadcast the option-level score
    # to every real token, avoiding the old question-length dilution.
    w[p+'q.bias'][47] = 4
    add_feature(w[p+'k'], 48, 47, 24)
    add_feature(w[p+'k'], 45, 47, 8)
    add_feature(w[p+'v'], 50, 32, 1)
    w[p+'o'][32, 49] = 1

    # Head 3 averages the small qualifier prior over ALL option words,
    # independently of the science-word weighting above.
    w[p+'q.bias'][63] = 4
    add_feature(w[p+'k'], 48, 63, 24)
    add_feature(w[p+'v'], 47, 48, 1)
    w[p+'o'][48, 53] = 1

    # The last MLP reads the broadcast score rather than a token-local score.
    # Reverse evidence for question negation XOR option negation. Each pair
    # of GELUs is a large-offset switch: GELU(gate+s)-GELU(gate).
    f1, b1, f2 = w[p+'fc1'], w[p+'fc1.bias'], w[p+'fc2']
    for case, (qneg, oneg) in enumerate(((0, 0), (1, 1), (1, 0), (0, 1))):
        a, b = 2*case, 2*case+1
        sign = 1 if qneg == oneg else -1
        for n in (a, b):
            add_feature(f1, 59, n, 24 if qneg else -24)
            add_feature(f1, 55, n, 24 if oneg else -24)
            b1[n] = 12 - 24 * (qneg + oneg)
        add_feature(f1, 49, a, 1)
        add_feature(f1, 51, a, -1)
        add_feature(f1, 53, a, 1)
        f2[a, 52], f2[b, 52] = sign, -sign

    w['ln_f.weight'] = torch.full((64,), ln_scale)
    zeros('ln_f.bias', (64,))
    score = zeros('score', (64,))
    score[52], score[61] = 100, -100
    return w


if __name__ == '__main__':
    write_vocab()

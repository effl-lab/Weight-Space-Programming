"""Compile hand-written lexical science knowledge into ordinary transformer tensors.

No optimization, pretrained parameters, corpus, examples, or model search is used.
Run `python build.py --write-vocab` once to materialize the deterministic vocabulary.
"""
from pathlib import Path
import hashlib
import math
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
D, F = 48, 20
SAL, NEG, ROLE, ANCHOR1, ANCHOR2, NEGCTX, OUT, REF = range(40, 48)
NORM2 = 2_000_000.0
SIGMA = math.sqrt((NORM2 + 1.0) / D - (1.0 / D) ** 2)
TOKEN = re.compile(r"[a-z0-9']+")

STOP = set("""
a an the and or but of to in on at by for from with as than that this these those
which what who whom whose when where why how is are was were be been being
it its it's they them their theirs he him his she her hers we us our you your
has have had having do does did doing done can could would should will shall
may might must also such some any each every other another most much many
very often usually generally sometimes typically likely probably following
example examples shown show shows showing illustrated diagram picture
student students teacher teachers scientist scientists researchers researcher
scientific science study studied studying studies investigation investigations
investigate investigates investigating question questions answer answers
best better describes described description describe statement statements
happens happen happened occurs occur occurred occurring result results
process processes object objects thing things certain particular several
use uses used using useful need needs needed necessary purpose provide
provides provided providing help helps helped helping make makes made making
cause causes caused causing affect affects affected affecting effect effects
because due during while after before then first next last still instead
about into onto through throughout among between within outside inside
there here either whether way ways one two three four five six seven eight
nine ten number numbers part parts type types kind kinds form forms
person people child children man woman boy girl john mary bob jane
""".split())
# Question filtering is intentionally broader than answer filtering. For example,
# "people" is usually an unhelpful question cue but a meaningful answer category.
# Only grammatical words are stripped from answer-side concept embeddings.
ANSWER_STOP = set("""
a an the and or but of to in on at by for from with as than that this these those
which what who whom whose when where why how is are was were be been being
it its it's they them their theirs he him his she her hers we us our you your
has have had having do does did doing done can could would should will shall
may might must also because during while after before then there here
""".split())
NEGATORS = set("not except least false incorrect incorrectly avoid unlikely never cannot can't doesn't don't isn't aren't won't".split())
PRIORS = {
    "only": -0.025, "always": -0.040, "never": -0.040,
    "instantly": -0.030, "completely": -0.020, "unlimited": -0.025,
    "magically": -0.080, "magic": -0.080, "impossible": -0.015,
}
# Exact antipodal codes give these natural contrasts an explicit signed meaning.
OPPOSITES = {
    "cold": "heat", "decrease": "increase", "slow": "fast",
    "small": "large", "contract": "expand", "down": "up",
    "negative": "positive", "lose": "gain", "smooth": "rough",
    "repulsion": "attraction", "insulator": "conductor",
    "insoluble": "solution", "nonrenewable": "renewable",
    "nonliving": "life", "west": "east", "south": "north",
    "condensation": "evaporation", "freezing": "melting",
    "deposition": "erosion", "physicalchange": "chemicalchange",
    "sinking": "floating",
}


def _lines(name):
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if line:
            yield line


def knowledge():
    concepts = {}
    for line in _lines("concepts.tsv"):
        name, domain, aliases = (part.strip() for part in line.split("|"))
        if name in concepts:
            raise ValueError(f"Duplicate concept: {name}")
        words = aliases.split()
        if any(TOKEN.fullmatch(word) is None for word in words):
            raise ValueError(f"Invalid lexical alias in {name}")
        concepts[name] = (domain, words)
    rules = []
    for line in _lines("rules.tsv"):
        source, destination = (part.strip() for part in line.split("|"))
        cues = set()
        for word in source.split():
            if word.startswith("@"):
                cues.update(concepts[word[1:]][1])
            else:
                if not TOKEN.fullmatch(word):
                    raise ValueError(f"Invalid cue {word}")
                cues.add(word)
        targets = []
        for item in destination.split():
            parts = item.split(":")
            name = parts[0]
            strength = float(parts[1]) if len(parts) == 2 else 1.0
            if strength == 0:
                continue
            if name not in concepts:
                raise ValueError(f"Unknown target concept {name!r}: {line}")
            targets.append((name, strength))
        if targets:
            rules.append((cues, targets))
    return concepts, rules


def lexical_forms(concepts, rules):
    """Explicit spellings override inherited forms; no stemming or text corpus."""
    explicit = STOP | NEGATORS | set(PRIORS)
    for _, aliases in concepts.values():
        explicit.update(aliases)
    for cues, _ in rules:
        explicit.update(cues)
    inherited = {}
    for line in _lines("forms.tsv"):
        base, variants = (part.strip() for part in line.split("|"))
        if base not in explicit:
            raise ValueError(f"Unknown lexical base: {base}")
        for variant in variants.split():
            if TOKEN.fullmatch(variant) is None:
                raise ValueError(f"Invalid lexical variant: {variant}")
            if variant in explicit:
                continue
            if variant in inherited and inherited[variant] != base:
                raise ValueError(f"Conflicting lexical bases for {variant}")
            inherited[variant] = base
    words = explicit | set(inherited)
    # Flatten possessives to their explicit root. This prevents copy order from
    # depending on lexical sorting and preserves the entire root feature vector.
    for word in sorted(words):
        if len(word) > 2 and "'" not in word and word not in STOP:
            possessive = word + "'s"
            if possessive not in words:
                inherited[possessive] = inherited.get(word, word)
    words.update(inherited)
    return sorted(words), inherited


def vocabulary():
    return lexical_forms(*knowledge())[0]


def category_ancestors(concepts):
    """Compile the hand-authored acyclic category graph, using shortest paths."""
    parents = {}
    for line in _lines("categories.tsv"):
        child, targets = (part.strip() for part in line.split("|"))
        if child not in concepts or any(p not in concepts for p in targets.split()):
            raise ValueError(f"Unknown category in {line}")
        if child in parents:
            raise ValueError(f"Duplicate category row: {child}")
        parents[child] = sorted(set(targets.split()))
    ancestors, active = {}, set()

    def visit(name):
        if name in active:
            raise ValueError(f"Cyclic category membership at {name}")
        if name in ancestors:
            return ancestors[name]
        active.add(name)
        result = {}
        for parent in parents.get(name, []):
            result[parent] = min(result.get(parent, 1), 1)
            for ancestor, depth in visit(parent).items():
                result[ancestor] = min(result.get(ancestor, depth + 1), depth + 1)
        active.remove(name)
        ancestors[name] = result
        return result

    for name in concepts:
        visit(name)
    return ancestors


def membership_weights(names, ancestors):
    """An authored category gets 1/2 per edge, counted once at its closest path."""
    weights = {name: 1.0 for name in sorted(set(names))}
    for name in sorted(set(names)):
        for ancestor, depth in ancestors[name].items():
            weights[ancestor] = max(weights.get(ancestor, 0.0), 0.5**depth)
    return weights


def _direction(label):
    """Fixed random indexing, NOT an embedding learned from any observations."""
    raw = hashlib.shake_256(label.encode("utf-8")).digest(4 * F)
    values = np.frombuffer(raw, dtype="<u4").astype(np.float64)
    values = (values + 0.5) / 2**32 - 0.5
    return values / np.linalg.norm(values)


def _lexical_vectors(vocab):
    concepts, rules = knowledge()
    codes = {}
    for name, (domain, _) in concepts.items():
        # A small common-domain component supplements the individual concept code.
        v = _direction("science-concept:" + name) + 0.20 * _direction("domain:" + domain)
        codes[name] = v / np.linalg.norm(v)
    for opposite, original in OPPOSITES.items():
        codes[opposite] = -codes[original]

    index = {word: i + 2 for i, word in enumerate(vocab)}
    n = len(vocab) + 2
    answer = np.zeros((n, F), dtype=np.float64)
    cues = np.zeros_like(answer)
    counts = np.zeros(n, dtype=np.float64)
    is_concept = np.zeros(n, dtype=bool)
    has_rule = np.zeros(n, dtype=bool)
    ancestors = category_ancestors(concepts)
    meanings = {}
    for name, (_, aliases) in concepts.items():
        for word in set(aliases):
            if word not in ANSWER_STOP:
                meanings.setdefault(word, set()).add(name)
    for word, names in meanings.items():
        i = index[word]
        # Explicit meanings have unit weight. Inherited category meanings decay
        # geometrically. Shared ancestors are not counted repeatedly, even when
        # a word belongs to multiple families (e.g. a hawk is a bird and predator).
        for name, strength in sorted(membership_weights(names, ancestors).items()):
            answer[i] += strength * codes[name]
        is_concept[i] = True
    norms = np.linalg.norm(answer, axis=1, keepdims=True)
    answer /= np.maximum(1.0, norms)
    # Small self-association; stronger, asymmetric factual links are added below.
    # Do not let retained answer concepts reintroduce generic question cues.
    cues += 0.25 * answer
    question_stop_ids = np.array([index[word] for word in sorted(STOP)], dtype=np.int64)
    cues[question_stop_ids] = 0.0
    for words, targets in rules:
        v = sum((strength * codes[name] for name, strength in targets),
                np.zeros(F, dtype=np.float64))
        v /= max(1.0, math.sqrt(sum(strength**2 for _, strength in targets)))
        for word in words:
            if word in STOP or word in NEGATORS:
                continue
            i = index[word]
            cues[i] += v
            counts[i] += 1
            has_rule[i] = True
    cues /= np.sqrt(1.0 + 0.5 * counts[:, None])
    norms = np.linalg.norm(cues, axis=1, keepdims=True)
    cues /= np.maximum(1.0, norms / 1.4)

    embed = np.zeros((n, D), dtype=np.float64)
    embed[:, :F] = cues
    embed[:, F:2*F] = answer
    embed[:, SAL] = math.log(0.025)
    embed[is_concept, SAL] = math.log(0.65)
    embed[has_rule, SAL] = math.log(1.0)
    embed[question_stop_ids, SAL] = math.log(0.025)
    for word, i in index.items():
        if word in NEGATORS:
            embed[i, NEG] = 1.0
        embed[i, OUT] = PRIORS.get(word, 0.0)
    _, inherited = lexical_forms(concepts, rules)
    for variant, base in sorted(inherited.items()):
        embed[index[variant]] = embed[index[base]]
    # Balance both mean and norm so layer normalization is initially an affine
    # operation with known scale. The two anchors never carry knowledge.
    sums = embed.sum(axis=1)
    squares = (embed * embed).sum(axis=1)
    delta = np.sqrt(2.0 * (NORM2 - squares) - sums * sums)
    embed[:, ANCHOR1] = (-sums + delta) / 2.0
    embed[:, ANCHOR2] = (-sums - delta) / 2.0
    return torch.tensor(embed, dtype=torch.float32)


def _put_difference(matrix, column, coordinate, value):
    """Read h[coordinate]-h[REF], eliminating layer-norm mean subtraction."""
    matrix[coordinate, column] += value
    matrix[REF, column] -= value


def build():
    vocab = (ROOT / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if vocab != vocabulary():
        raise ValueError("vocab.txt does not match the deterministic knowledge vocabulary")
    w = {
        "embed": _lexical_vectors(vocab),
        "pos": torch.zeros(176, D),
        "ln_f.weight": torch.full((D,), SIGMA),
        "ln_f.bias": torch.zeros(D),
        "score": torch.zeros(D),
    }
    for block in range(2):
        p = f"block{block}."
        for ln in ("ln1", "ln2"):
            w[p + ln + ".weight"] = torch.full((D,), SIGMA)
            w[p + ln + ".bias"] = torch.zeros(D)
        for name in ("q", "k", "v", "o"):
            w[p + name] = torch.zeros(D, D)
            w[p + name + ".bias"] = torch.zeros(D)
        w[p + "fc1"] = torch.zeros(D, 96)
        w[p + "fc1.bias"] = torch.zeros(96)
        w[p + "fc2"] = torch.zeros(96, D)
        w[p + "fc2.bias"] = torch.zeros(D)

    # Input positions, rather than option numbers, identify the two segments.
    w["pos"][:128, ROLE] = -1.0
    w["pos"][128:, ROLE] = 1.0
    # Mild recency preference: the actual interrogative often follows background.
    w["pos"][:128, SAL] = torch.arange(128) * 0.012

    # Block 0: segment-local negation detection. The role product gives keys
    # from the query's own segment a +24 bias and other-segment keys a -24 bias.
    # Thus question and option negation are summarized independently, using one
    # head and the same NEGCTX coordinate at their respective token positions.
    p = "block0."
    w[p + "q.bias"][0] = math.sqrt(12)
    _put_difference(w[p + "k"], 0, NEG, 10.0)
    _put_difference(w[p + "q"], 1, ROLE, math.sqrt(12))
    _put_difference(w[p + "k"], 1, ROLE, 24.0)
    _put_difference(w[p + "v"], 0, NEG, 1.0)
    w[p + "o"][0, NEGCTX] = 1.0

    # A second head broadcasts QUESTION negation to both segments. Store it as
    # a common +8 offset in salience: adding the same value to all question keys
    # cancels in the next block's softmax (up to the tiny anchor-scale error).
    # Thus question polarity is available to option-prior gates without taking
    # a semantic coordinate or confusing it with option-local negation.
    w[p + "q.bias"][12] = math.sqrt(12)
    _put_difference(w[p + "k"], 12, NEG, 10.0)
    _put_difference(w[p + "k"], 12, ROLE, -24.0)
    _put_difference(w[p + "v"], 12, NEG, 1.0)
    w[p + "o"][12, SAL] = 8.0

    # Gating identity: GELU(x+b)-GELU(b) is approximately x for b>>0,
    # approximately zero for b<<0. Semantic coordinates are bounded by 1.4;
    # gate margins are 12. Sharing the GELU(b) neuron saves MLP width.
    next_unit = 0

    def subtract_gate(coordinates, role_weight, neg_weight=0.0, bias=0.0,
                      multiplier=1.0, sal_weight=0.0):
        nonlocal next_unit
        base = next_unit
        next_unit += 1
        _put_difference(w[p + "fc1"], base, ROLE, role_weight)
        _put_difference(w[p + "fc1"], base, NEGCTX, neg_weight)
        _put_difference(w[p + "fc1"], base, SAL, sal_weight)
        w[p + "fc1.bias"][base] = bias
        for coordinate in coordinates:
            unit = next_unit
            next_unit += 1
            _put_difference(w[p + "fc1"], unit, coordinate, 1.0)
            _put_difference(w[p + "fc1"], unit, ROLE, role_weight)
            _put_difference(w[p + "fc1"], unit, NEGCTX, neg_weight)
            _put_difference(w[p + "fc1"], unit, SAL, sal_weight)
            w[p + "fc1.bias"][unit] = bias
            w[p + "fc2"][unit, coordinate] -= multiplier
            w[p + "fc2"][base, coordinate] += multiplier

    subtract_gate(range(F), 12.0)                # Erase option-side cues.
    subtract_gate(range(F, 2*F), -12.0)           # Erase question-side answers.
    subtract_gate(range(F), -12.0, 24.0, -24.0,
                  multiplier=2.0)               # Invert negated question cues.
    subtract_gate(range(F, 2*F), 12.0, 24.0, -24.0,
                  multiplier=2.0)               # Invert negated option meanings.
    subtract_gate([OUT], -12.0)                  # Priors apply only to options.
    # Normal option salience is in [log(0.025), 0]. With question negation it
    # exceeds 4.26, even for a 128-token question with only one negator.
    # b=4*salience+32*role-40 is therefore <=-8 normally, >=9 for a negated
    # question's options, and always negative at question positions. Negating
    # the prior when this gate is active aligns it with exception questions.
    subtract_gate([OUT], 32.0, bias=-40.0, multiplier=2.0, sal_weight=4.0)
    assert next_unit <= 96

    # Block 1 uses all four heads. The first pair preserves the full, global
    # salience-weighted question summary. The second pair adds option-conditioned
    # summaries: each head reads ten answer coordinates as its semantic query.
    # In each output half, global and focused evidence receive equal weight.
    # All heads exclude option keys, so answers cannot supply their own evidence.
    # The similarity gain 4 gives a bounded cue substantial but not exclusive
    # influence; it is a design constant, not an empirically selected parameter.
    p = "block1."
    for head in range(4):
        start = head * 12
        half = head % 2
        focused = head >= 2
        w[p + "q.bias"][start + 10] = math.sqrt(12)
        _put_difference(w[p + "k"], start + 10, SAL, 1.0)
        _put_difference(w[p + "k"], start + 10, ROLE, -24.0)
        for j in range(10):
            coordinate = half * 10 + j
            _put_difference(w[p + "v"], start + j, coordinate, 1.0)
            w[p + "o"][start + j, coordinate] = 0.5
            if focused:
                _put_difference(w[p + "q"], start + j, F + coordinate,
                                4.0 * math.sqrt(12))
                _put_difference(w[p + "k"], start + j, coordinate, 1.0)

    # Fixed analytic multiplier using GELU's even quadratic component:
    # [g(t(c+a))+g(-t(c+a))-g(t(c-a))-g(-t(c-a))]
    #       / [4*t*t*sqrt(2/pi)]  ~= c*a.
    # Twenty products require 80 of the 96 hidden neurons; no fitting occurs.
    # Add 6*(role-1) to each preactivation: approximately 0 for option tokens
    # and -12 for question tokens. The latter saturates GELU at zero even if
    # block 0's approximate erasure leaves small answer-coordinate residuals.
    # This prevents question-token compatibility leakage without extra neurons.
    t = 0.125
    scale = 1.0 / (4.0 * t * t * math.sqrt(2.0 / math.pi))
    for j in range(F):
        for k, (sc, sa, so) in enumerate(((1, 1, 1), (-1, -1, 1),
                                         (1, -1, -1), (-1, 1, -1))):
            unit = 4*j + k
            _put_difference(w[p + "fc1"], unit, j, sc*t)
            _put_difference(w[p + "fc1"], unit, F+j, sa*t)
            _put_difference(w[p + "fc1"], unit, ROLE, 6.0)
            w[p + "fc1.bias"][unit] = -6.0
            w[p + "fc2"][unit, OUT] = so * scale
    w["score"][OUT] = 1.0
    w["score"][REF] = -1.0
    return w


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-vocab"]:
        raise SystemExit("Usage: python build.py --write-vocab")
    vocab = vocabulary()
    with (ROOT / "vocab.txt").open("x", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    print(f"Wrote {len(vocab)} vocabulary entries.")

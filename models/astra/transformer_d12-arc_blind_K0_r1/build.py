"""Compile an authored science lexicon into the fixed 12-dimensional transformer.

There is no optimization, data-dependent fitting, checkpoint, or external input.
All numerical parameters below are chosen from the algebra of the circuit.
Run this file once to materialize the deterministically ordered vocab.txt.
"""
from pathlib import Path
import math

import torch

ROOT = Path(__file__).resolve().parent
DTYPE = torch.float64
D = 12
ANCHOR = 1_000_000.0
LN_SCALE = ANCHOR / math.sqrt(6.0)
GATE = 32.0
OPTION_RETAIN = 0.02
NEG_MARKER = 16.0
NEG_SHIFT = 0.25
OPTION_NEG_SHIFT = 0.125
OPTION_NEG_PREFIX = 3
STEP_SCALE = 512.0
MAX_QUERY_RADIUS = 1.25

# The prescribed ASCII-token regex splits curly-apostrophe contractions into
# e.g. 'isn', 't'. Preserve negation for unambiguous fragments only. Do NOT mark
# 'can', 'don', 'won', or 'haven': those are ordinary affirmative words too.
CONTRACTION_FRAGMENTS = {
    'isn': "isn't", 'aren': "aren't", 'wasn': "wasn't", 'weren': "weren't",
    'doesn': "doesn't", 'didn': "didn't", 'couldn': "couldn't",
    'wouldn': "wouldn't", 'shouldn': "shouldn't", 'hasn': "hasn't",
    'hadn': "hadn't", 'mustn': "mustn't", 'needn': "needn't",
}

# Segment-aware polarity heuristics. Marker detection, prefix scoping, XOR,
# and reversal are implemented entirely by the existing weight tensors.
NEGATORS = frozenset((
    "not except least incorrect incorrectly false cannot non "
    "isn't aren't wasn't weren't doesn't don't didn't can't "
    "couldn't wouldn't shouldn't won't hasn't haven't hadn't mustn't needn't"
).split()) | frozenset(CONTRACTION_FRAGMENTS)
# Question-selection words such as EXCEPT are not logical NOT in an answer.
# Option negators are read only in its first three tokens, avoiding a global
# reversal for later contrasts such as '... but does not ...'.
OPTION_NEGATORS = NEGATORS - {"except", "least", "incorrect", "incorrectly", "false"}

# Explicitly authored equivalences repair inconsistent names in the tables.
# They are not inferred from examples or selected using measured accuracy.
CONCEPT_ALIASES = {
    "allele": "gene",
    "cooling": "cold",
    "decrease": "reduce",
    "dissolve": "solution",
    "enzyme": "catalyst",
    "information": "data",
    "nonliving": "abiotic",
    "nutrition": "nutrient",
    "orbit": "revolution",
    "pattern": "trend",
    "permeability": "porous",
    "reaction": "chemical_change",
    "recycle": "recycling",
    "role": "function",
    "scattering": "scatter",
    "shell": "exoskeleton",
    "spreading": "spread",
    "stability": "stable",
}

# Small, explicitly authored answer-language priors. They are deliberately much
# weaker than a single question-to-answer concept match after the option gate.
PRIORS = {
    "only": -0.75,
    "always": -0.60,
    "never": -0.60,
    "impossible": -0.50,
    "instantly": -0.50,
    "nothing": -0.30,
    "none": -0.30,
    "unrelated": -0.30,
    "everything": -0.20,
    "completely": -0.15,
}

# Exceptionally ambiguous verbs get low-confidence directed associations rather
# than the nearly binary matches used for concrete scientific terminology.
# Values: first target, second target, exact-match attention logit.
QUERY_RULES = [
    ("release releases released releasing", "oxygen", "heat", 1.5),
    ("absorb absorbs absorbed absorbing uptake", "water", "sunlight", 1.5),
    ("inhale inhaling inhalation", "oxygen", "lung", 12.0),
    ("exhale exhaling exhalation", "co2", "lung", 12.0),
    ("sunrise", "east", "rotation", 14.0),
    ("sunset", "west", "rotation", 14.0),
    ("positive positively", "proton", "positive", 14.0),
    ("negative negatively", "electron", "negative", 14.0),
    ("neutral uncharged", "neutron", "neutral", 14.0),
    ("covalent", "electron", "bond", 14.0),
    ("ionic", "charge", "bond", 14.0),
    ("chlorophyll", "sunlight", "photosynthesis", 14.0),
    ("ribosome ribosomes ribosomal", "protein", "ribosome", 14.0),
    ("mitochondria mitochondrion mitochondrial", "energy", "respiration", 14.0),
    ("endocrine", "hormone", "message", 14.0),
    ("transverse", "perpendicular", "wave", 14.0),
    ("longitudinal compressional", "parallel", "sound", 14.0),
    ("control controls controlled controlling", "control_group", "nervous", 12.0),
]


def lexicon():
    """Parse authored tables; later entries refine earlier ones, then unify names."""
    records = {}
    concepts = set()
    for filename in ("knowledge.tsv", "overrides.tsv", "corrections.tsv"):
        for line_number, raw in enumerate((ROOT / filename).read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = [s.strip() for s in line.split("|")]
            if len(fields) != 4:
                raise ValueError(f"{filename}:{line_number}: expected four fields")
            primary, alternate, related, tokens = fields
            primary, alternate, related = (
                CONCEPT_ALIASES.get(c.lower(), c.lower())
                for c in (primary, alternate, related)
            )
            concepts.update(c for c in (primary, alternate, related) if c != "-")
            for word in tokens.split():
                if word != word.lower() or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789'" for c in word):
                    raise ValueError(f"Invalid vocabulary token: {word!r}")
                records[word] = (primary, alternate, related)
    query_rules = {}
    for words, first, second, strength in QUERY_RULES:
        first = CONCEPT_ALIASES.get(first, first)
        second = CONCEPT_ALIASES.get(second, second)
        concepts.update((first, second))
        for word in words.split():
            if word not in records:
                raise ValueError(f"Query rule needs a lexical entry: {word}")
            query_rules[word] = (first, second, strength)
    # The supplied tokenizer retains apostrophes. Give possessive spellings the
    # SAME authored meaning and query rule as their base word, rather than UNK.
    # This is a fixed English spelling rule, not vocabulary mined from data.
    # Explicit entries take precedence; only a snapshot of base words is used.
    for word in tuple(records):
        if word.isalpha():
            spellings = [word + "'s"]
            if word.endswith('s'):
                spellings.append(word + "'")
            for spelling in spellings:
                if spelling not in records:
                    records[spelling] = records[word]
                    if word in query_rules:
                        query_rules[spelling] = query_rules[word]
    keys = {c for primary, alternate, _ in records.values() for c in (primary, alternate)}
    targets = {related for _, _, related in records.values()}
    targets.update(c for first, second, _ in query_rules.values() for c in (first, second))
    if targets - keys:
        raise ValueError(f"Unreachable authored query targets: {sorted(targets - keys)}")
    if not set(CONTRACTION_FRAGMENTS.values()) <= NEGATORS:
        raise ValueError('A contraction fragment lacks its negative source spelling')
    if set(CONTRACTION_FRAGMENTS) & set(records):
        raise ValueError('A contraction fragment conflicts with a scientific lexical entry')
    vocabulary = sorted(set(records) | set(PRIORS) | NEGATORS)
    return records, sorted(concepts), query_rules, vocabulary


def concept_geometry(concepts):
    """Compile authored families into unit keys and bounded angular-cap queries.

    Depth-first order makes each family's descendants contiguous on the circle.
    A query at the interval midpoint with radius sec(half_width) has dot product
    >= 1 with exactly its own family, including itself. A leaf remains an exact
    symbol query. This is a closed-form construction, not a search or fitting.
    """
    known = set(concepts)
    children = {}
    parent_of = {}
    for line_number, raw in enumerate((ROOT / 'families.tsv').read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        fields = [field.strip() for field in line.split('|')]
        if len(fields) != 2 or not fields[1]:
            raise ValueError(f'families.tsv:{line_number}: expected parent | children')
        parent = CONCEPT_ALIASES.get(fields[0], fields[0])
        if parent not in known or parent in children:
            raise ValueError(f'Unknown or repeated family parent: {parent}')
        members = [CONCEPT_ALIASES.get(c, c) for c in fields[1].split()]
        for child in members:
            if child not in known or child in parent_of or child == parent:
                raise ValueError(f'Unknown, repeated-parent, or self-linked child: {child}')
            parent_of[child] = parent
        children[parent] = sorted(members)

    order, spans, active = [], {}, set()

    def visit(concept):
        if concept in active:
            raise ValueError(f'Cycle in authored concept families: {concept}')
        active.add(concept)
        start = len(order)
        order.append(concept)
        for child in children.get(concept, ()):
            visit(child)
        spans[concept] = (start, len(order))
        active.remove(concept)

    for root in sorted(known - set(parent_of)):
        visit(root)
    if len(order) != len(known) or set(order) != known:
        raise ValueError('Concept families contain a cycle or omit a symbol')

    count = len(order)
    keys, queries = {}, {}
    for i, concept in enumerate(order):
        angle = 2.0 * math.pi * (i + 0.5) / count
        keys[concept] = (math.cos(angle), math.sin(angle))
        start, end = spans[concept]
        midpoint = math.pi * (start + end) / count
        half_width = math.pi * (end - start - 1) / count
        if half_width >= math.pi / 2:
            raise ValueError(f'Family angular interval is too wide: {concept}')
        radius = 1.0 / math.cos(half_width)
        if radius > MAX_QUERY_RADIUS:
            raise ValueError(f'Family exceeds the polarity circuit radius bound: {concept}')
        queries[concept] = (radius * math.cos(midpoint), radius * math.sin(midpoint))
    return keys, queries, children


def _zero(*shape):
    return torch.zeros(shape, dtype=DTYPE)


def _balanced_add(row, feature, amount):
    """Write a zero-sum feature direction, preserving the normalization anchor."""
    row[feature] += amount
    row[10] -= amount / 2.0
    row[11] -= amount / 2.0


def build():
    records, concepts, query_rules, vocabulary = lexicon()
    on_disk = (ROOT / "vocab.txt").read_text().splitlines()
    if on_disk != vocabulary:
        raise ValueError("vocab.txt does not match the authored, sorted lexicon")

    # Equally spaced unit keys, with exact leaf queries and angular-cap family
    # queries. At a family boundary the matching dot product is exactly one.
    # The next nonmember has a rejection margin at least as large as the leaf
    # case (approximately 16*pi**2 in logit space). All scales are algebraic.
    count = len(concepts)
    attention_scale = max(1_000_000.0, 8.0 * count * count)
    match_logit = 14.0
    threshold = 1.0 - match_logit / attention_scale
    codes, query_codes, _ = concept_geometry(concepts)

    w = {"embed": _zero(len(vocabulary) + 2, D), "pos": _zero(176, D)}
    for i in range(2):
        p = f"block{i}."
        for layer in ("ln1", "ln2"):
            w[p + layer + ".weight"] = torch.full((D,), LN_SCALE, dtype=DTYPE)
            w[p + layer + ".bias"] = _zero(D)
        for name in ("q", "k", "v", "o"):
            w[p + name] = _zero(D, D)
            w[p + name + ".bias"] = _zero(D)
        w[p + "fc1"] = _zero(D, 24)
        w[p + "fc1.bias"] = _zero(24)
        w[p + "fc2"] = _zero(24, D)
        w[p + "fc2.bias"] = _zero(D)
    w["ln_f.weight"] = torch.full((D,), LN_SCALE, dtype=DTYPE)
    w["ln_f.bias"] = _zero(D)
    w["score"] = _zero(D)
    w["score"][9] = 1.0

    # Layout: query A [0:2], key A [2:4], query B [4:6], key B [6:8],
    # segment [8], evidence [9], fixed +/- normalization anchor [10:12].
    def put_code(row, start, concept, strength=None):
        if concept == "-":
            return
        radius = 1.0 if strength is None else 1.0 + (strength - match_logit) / attention_scale
        x, y = (codes if strength is None else query_codes)[concept]
        row[start] = radius * x
        row[start + 1] = radius * y

    for idx, word in enumerate(vocabulary, 2):
        row = w["embed"][idx]
        if word in records:
            primary, alternate, related = records[word]
            put_code(row, 2, primary)
            if alternate != primary:
                put_code(row, 6, alternate)
            if word in query_rules:
                first, second, strength = query_rules[word]
                put_code(row, 0, first, strength)
                if second != first:
                    put_code(row, 4, second, strength)
            else:
                put_code(row, 0, primary, 14.0)
                second = related if related != "-" else alternate
                if second != primary:
                    put_code(row, 4, second, 12.0)
        row[9] = PRIORS.get(word, 0.0)
        if word in NEGATORS:
            row[:10] = 0.0
            row[0] = NEG_MARKER
            if word in OPTION_NEGATORS:
                row[1] = NEG_MARKER

    # Include anchors even for padding and UNK. Every initial vector is centered.
    sums = w["embed"][:, :10].sum(dim=1)
    w["embed"][:, 10] = ANCHOR - sums / 2.0
    w["embed"][:, 11] = -ANCHOR - sums / 2.0
    w["pos"][:128, 8] = 1.0
    w["pos"][128:, 8] = -1.0
    w["pos"][128:128 + OPTION_NEG_PREFIX, 8] = -2.0
    w["pos"][:, 10] = -w["pos"][:, 8] / 2.0
    w["pos"][:, 11] = -w["pos"][:, 8] / 2.0

    # Block 0: two constant-query heads read separate polarity markers. Ordinary
    # coordinates are bounded by MAX_QUERY_RADIUS=1.25; markers are +16. Head 0
    # prefers questions; head 1 prefers the first three option tokens (code -2).
    # The 32-unit position advantage exceeds the maximum 17.25-unit lexical
    # advantage. Thus neither head borrows a marker from the wrong segment, and
    # late option markers cannot defeat a non-marker in the option prefix.
    # Absent reads are bounded by +/-1.25/16; present reads are approximately 1.
    p = "block0."
    for head, feature, position_sign, amount in (
        (0, 0, 1.0, NEG_SHIFT), (1, 1, -1.0, OPTION_NEG_SHIFT)
    ):
        j = 3 * head
        w[p + "q.bias"][j] = 1.0
        w[p + "k"][feature, j] = 4.0 * math.sqrt(3.0)
        w[p + "k"][8, j] = 4.0 * math.sqrt(3.0) * 32.0 * position_sign
        w[p + "v"][feature, j] = 1.0 / NEG_MARKER
        _balanced_add(w[p + "o"][j], 8, amount)

    # Clear keys and lexical priors on questions. The raw segment shift is too
    # small to change these branches: relu(z+4*p)-relu(4*p) equals z on questions
    # and zero on options. GELU at scale 32 realizes this to machine precision.
    masked_features = (2, 3, 6, 7, 9)
    common = len(masked_features)
    w[p + "fc1"][8, common] = 4.0 * GATE
    for neuron, feature in enumerate(masked_features):
        w[p + "fc1"][feature, neuron] = GATE
        w[p + "fc1"][8, neuron] = 4.0 * GATE
        _balanced_add(w[p + "fc2"][neuron], feature, -1.0 / GATE)
        _balanced_add(w[p + "fc2"][common], feature, 1.0 / GATE)

    # Erase both out-of-band markers before semantic matching. Normal concept
    # coordinates never reach this gate's threshold of +2.
    for feature, neuron in ((0, 6), (1, 7)):
        w[p + "fc1"][feature, neuron] = GATE
        w[p + "fc1.bias"][neuron] = -2.0 * GATE
        _balanced_add(w[p + "fc2"][neuron], feature,
                      -NEG_MARKER / ((NEG_MARKER - 2.0) * GATE))

    # The two broadcasts give shifts 0, 1/8, 1/4, 3/8 for neither, option only,
    # question only, and both. The middle two require a sign reversal: XOR.
    # Each initial position band (+1 question, -2 prefix, -1 remaining option)
    # gets a pair of clipped steps enclosing those two middle levels.
    # Normal-read noise is bounded by (1/4+1/8)*1.25/16 = 15/512, safely outside
    # the width-.02 ramps at offsets 1/16 and 5/16. Scale 512 gives a large
    # GELU margin. A seventh ramp restores the ordinary question/option sign.
    # Result: p = ordinary_segment + NEG_SHIFT * (question_NOT XOR option_NOT).
    # The evidence write pre-cancels the later attention value offset.
    w[p + "fc1"][8, 8] = GATE
    w[p + "fc1.bias"][8] = 4.0 * GATE
    _balanced_add(w[p + "fc2"][8], 8, -1.0 / GATE)
    _balanced_add(w[p + "fc2.bias"], 8, 3.0)
    ramps = []
    for base in (1.0, -2.0, -1.0):
        for offset_value, coefficient in ((0.0625, 1.0), (0.3125, -1.0)):
            center = base + offset_value
            ramps.append((center - 0.01, center + 0.01,
                          coefficient * NEG_SHIFT, coefficient * NEG_SHIFT / 2.0))
    ramps.append((-0.25, 0.25, 2.0, 0.0))
    for ramp, (left, right, pos_amount, score_amount) in enumerate(ramps):
        for edge, (threshold_value, sign) in enumerate(((left, 1.0), (right, -1.0))):
            neuron = 9 + 2 * ramp + edge
            w[p + "fc1"][8, neuron] = STEP_SCALE
            w[p + "fc1.bias"][neuron] = -threshold_value * STEP_SCALE
            factor = sign / ((right - left) * STEP_SCALE)
            _balanced_add(w[p + "fc2"][neuron], 8, pos_amount * factor)
            _balanced_add(w[p + "fc2"][neuron], 9, score_amount * factor)

    # Block 1: four heads perform all query-slot/key-slot combinations. Real
    # question tokens supply null keys; option keys compete with those nulls.
    # The negation flag adds the SAME logit offset to every key and therefore
    # cancels in softmax. Only symbols in the query's family overcome threshold.
    # Values are option_indicator - NEG_SHIFT*negated/2; the residual evidence
    # offset from block 0 cancels the latter term exactly.
    p = "block1."
    pairings = ((0, 2), (0, 6), (4, 2), (4, 6))
    dot_scale = math.sqrt(attention_scale * math.sqrt(3.0))
    offset = attention_scale * math.sqrt(3.0) * threshold / 2.0
    for head, (query_start, key_start) in enumerate(pairings):
        j = 3 * head
        for axis in range(2):
            w[p + "q"][query_start + axis, j + axis] = dot_scale
            w[p + "k"][key_start + axis, j + axis] = dot_scale
        w[p + "q.bias"][j + 2] = 1.0
        w[p + "k"][8, j + 2] = offset
        w[p + "k.bias"][j + 2] = -offset
        w[p + "v"][8, j] = -0.5
        w[p + "v.bias"][j] = 0.5
        _balanced_add(w[p + "o"][j], 9, 0.25)

    # Read question evidence at full strength. Retain only 2% of option-internal
    # evidence: otherwise every option word would earn a large self-match bonus.
    # This weak residual supplies a scientific-specificity/coherence tie-breaker
    # and the explicitly listed language priors; it carries no answer-position
    # information. This gate is valid for the entire designed evidence range.
    w[p + "fc1"][9, 0] = GATE
    w[p + "fc1"][8, 0] = -4.0 * GATE
    w[p + "fc1"][8, 1] = -4.0 * GATE
    _balanced_add(w[p + "fc2"][0], 9, -(1.0 - OPTION_RETAIN) / GATE)
    _balanced_add(w[p + "fc2"][1], 9, (1.0 - OPTION_RETAIN) / GATE)

    # An odd number of active segment flags reverses association polarity.
    # Each selector g(s,p,t)=relu(s+32*(p-t))-relu(32*(p-t)) is either s or zero.
    # After the option gate above, these corrections implement multipliers
    # +1/-1 on question evidence and +retain/-retain on option evidence.
    selectors = ((1.125, -2.0), (-0.875, -2.0 * OPTION_RETAIN),
                 (0.0, 2.0 * OPTION_RETAIN))
    for selector, (threshold_value, correction) in enumerate(selectors):
        a, b = 2 + 2 * selector, 3 + 2 * selector
        w[p + "fc1"][9, a] = GATE
        for neuron in (a, b):
            w[p + "fc1"][8, neuron] = 32.0 * GATE
            w[p + "fc1.bias"][neuron] = -32.0 * GATE * threshold_value
        _balanced_add(w[p + "fc2"][a], 9, correction / GATE)
        _balanced_add(w[p + "fc2"][b], 9, -correction / GATE)

    assert all(torch.isfinite(t).all().item() for t in w.values())
    return w


if __name__ == "__main__":
    _, concepts, _, vocabulary = lexicon()
    (ROOT / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    print(f"Wrote {len(vocabulary)} vocabulary words and {len(concepts)} authored concept codes.")

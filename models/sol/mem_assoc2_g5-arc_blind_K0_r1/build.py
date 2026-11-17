"""Compile hand-written science knowledge into the supplied association memory.

There is no fitting, corpus ingestion, network access, or test-dependent behavior.
The text files are the complete source of the model's knowledge.
"""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import re
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")
STOP = set('''a an the of and or to is are be by on in for with from than then
also only alone always all any each every both other often normally usually
necessarily no cannot without never must most main many much some very'''.split())


NEGATIVE_PARTICLES = {'no', 'not', 'never', 'cannot', 'without', 'neither'}


def fact_words(text):
    """Signs belong to the supported/opposed columns, not English clauses."""
    words = TOKEN.findall(text.lower())
    embedded = NEGATIVE_PARTICLES.intersection(words)
    if embedded:
        raise ValueError('Embedded negation in fact field; use explicit signed '
                         'concepts instead: ' + text)
    return [word for word in words if word not in STOP]


def ontology_facts(canonicalize):
    """Conservative, deterministic class inheritance from an authored DAG.

    Only explicit is-a links are transitive. Ordinary associations in A are
    never treated as entailments. Multiple proof paths use max, not sum.
    """
    parents = defaultdict(dict)
    traits = defaultdict(list)
    nodes = set()
    path = ROOT / 'ontology.txt'
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].strip().lower()
        if not line:
            continue
        parts = [part.strip() for part in line.split('|')]
        kind = parts[0]
        if kind not in ('isa', 'trait') or len(parts) != (4 if kind == 'isa' else 5):
            raise ValueError('Invalid ontology declaration: ' + line)
        strength = float(parts[1])
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError('Invalid ontology strength: ' + line)
        subjects = sorted({canonicalize(w) for w in fact_words(parts[2])})
        if not subjects:
            raise ValueError('Missing ontology subject: ' + line)
        nodes.update(subjects)
        if kind == 'isa':
            supers = [canonicalize(w) for w in fact_words(parts[3])]
            if len(supers) != 1 or strength > 1:
                raise ValueError('Is-a link needs one parent and attenuation in (0, 1]: ' + line)
            parent = supers[0]
            nodes.add(parent)
            for subject in subjects:
                parents[subject][parent] = max(strength, parents[subject].get(parent, 0.0))
        else:
            yes = sorted({canonicalize(w) for w in fact_words(parts[3])})
            no = sorted({canonicalize(w) for w in fact_words(parts[4])})
            if set(yes).intersection(no):
                raise ValueError('Conflicting trait signs: ' + line)
            for subject in subjects:
                traits[subject].append((strength, yes, no))

    cache = {}
    visiting = set()

    def ancestors(node):
        if node in cache:
            return cache[node]
        if node in visiting:
            raise ValueError('Cycle in is-a hierarchy at: ' + node)
        visiting.add(node)
        result = {node: 1.0}
        for parent, attenuation in sorted(parents[node].items()):
            for ancestor, confidence in ancestors(parent).items():
                result[ancestor] = max(result.get(ancestor, 0.0), attenuation * confidence)
        visiting.remove(node)
        cache[node] = result
        return result

    out = []
    for node in sorted(nodes):
        support, oppose = {}, {}
        for ancestor, confidence in sorted(ancestors(node).items()):
            if ancestor != node:
                support[ancestor] = max(confidence, support.get(ancestor, 0.0))
                # Retrieving a known example of a class is useful too. This
                # reverse association is NOT added to the entailment graph.
                out.append((0.60 * confidence, [ancestor], [node], []))
            for strength, yes, no in traits[ancestor]:
                for word in yes:
                    support[word] = max(strength * confidence, support.get(word, 0.0))
                for word in no:
                    oppose[word] = max(strength * confidence, oppose.get(word, 0.0))
        support.pop(node, None)
        oppose.pop(node, None)
        conflict = set(support).intersection(oppose)
        if conflict:
            raise ValueError('Inherited trait conflict for ' + node + ': ' + str(sorted(conflict)))
        out.extend((value, [node], [word], []) for word, value in sorted(support.items()))
        out.extend((value, [node], [], [word]) for word, value in sorted(oppose.items()))
    return out


def source():
    # Canonical concepts and explicit lexical aliases.
    groups = []
    occupied = {}
    for raw in (ROOT / 'lexicon.txt').read_text().splitlines():
        line = raw.split('#', 1)[0].strip().lower()
        if not line:
            continue
        ws = list(dict.fromkeys(line.split()))
        if any(TOKEN.fullmatch(w) is None for w in ws):
            raise ValueError('Invalid lexical token: ' + line)
        existing = {occupied[w] for w in ws if w in occupied}
        if existing:
            raise ValueError('Overlapping lexical groups: ' + line)
        idx = len(groups) + 1
        groups.append(ws)
        for w in ws:
            occupied[w] = idx
    # Additional authored aliases may join existing synonymous groups. This is
    # lexical equivalence only; science associations never induce synonymy.
    for raw in '\n'.join((ROOT / name).read_text() for name in
                         ('aliases.txt', 'lexical_extra.txt', 'lexical_context.txt',
                          'lexical_g5.txt')).splitlines():
        line = raw.split('#', 1)[0].strip().lower()
        if not line:
            continue
        ws = list(dict.fromkeys(line.split()))
        if any(TOKEN.fullmatch(w) is None for w in ws):
            raise ValueError('Invalid alias token: ' + line)
        matched = sorted({occupied[w] for w in ws if w in occupied})
        if matched:
            idx = matched[0]
            members = list(dict.fromkeys([w for i in matched for w in groups[i - 1]] + ws))
            for i in matched[1:]:
                groups[i - 1] = []
            groups[idx - 1] = members
        else:
            idx = len(groups) + 1
            groups.append(ws)
            members = ws
        for word in members:
            occupied[word] = idx
    groups = [g for g in groups if g]
    occupied = {word: i for i, g in enumerate(groups, 1) for word in g}
    facts = []
    authored = '\n'.join((ROOT / name).read_text() for name in
                         ('facts.txt', 'earth.txt', 'life.txt', 'inquiry.txt',
                          'supplement.txt', 'functions.txt', 'relations.txt',
                          'mechanisms.txt', 'scenarios.txt', 'definitions.txt',
                          'ecology_extra.txt', 'earth_extra.txt',
                          'physical_extra.txt', 'facets.txt',
                          'context_physics.txt', 'context_earth.txt',
                          'context_life.txt', 'diagnostic_pairs.txt',
                          'context_inquiry.txt', 'representation.txt',
                          'causal_physics.txt', 'causal_life.txt',
                          'causal_earth.txt', 'reasoning_inquiry.txt',
                          'decision_contexts.txt', 'review_relations.txt'))
    for raw in authored.splitlines():
        line = raw.split('#', 1)[0].strip().lower()
        if not line:
            continue
        parts = [s.strip() for s in line.split('|')]
        if len(parts) != 4:
            raise ValueError('Expected weight | cues | supported | opposed: ' + line)
        weight = float(parts[0])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Fact strengths must be finite and positive: ' + line)
        fields = [fact_words(part) for part in parts[1:]]
        if not fields[0]:
            raise ValueError('Fact has no content cues: ' + line)
        for field in fields:
            for word in field:
                if word not in occupied:
                    occupied[word] = len(groups) + 1
                    groups.append([word])
        facts.append((weight, *fields))
    # Inherit only along the separately authored is-a graph. Concept names
    # are canonicalized before closure, so aliases cannot form hidden cycles.
    def canonicalize(word):
        return groups[occupied[word] - 1][0] if word in occupied else word

    inherited = ontology_facts(canonicalize)
    for weight, cues, yes, no in inherited:
        for word in cues + yes + no:
            if word not in occupied:
                occupied[word] = len(groups) + 1
                groups.append([word])
        facts.append((weight, cues, yes, no))
    # Predictable surface forms only, not a learned stemmer. Explicit groups win.
    no_plural = {'gas', 'mass', 'glass', 'grass', 'less', 'cross', 'loss', 'process',
                 'species', 'series', 'axis', 'this', 'his', 'is', 'as', 'has',
                 'does', 'always', 'sometimes', 'news', 'physics', 'diabetes',
                 'osmosis', 'analysis', 'synthesis', 'photosynthesis', 'hypothesis'}
    for idx, group in enumerate(groups, 1):
        word = group[0]
        if len(word) < 3 or not word.isalpha() or word in no_plural:
            continue
        if word.endswith('y') and word[-2] not in 'aeiou':
            plural = word[:-1] + 'ies'
        elif word.endswith(('s', 'sh', 'ch', 'x', 'z')):
            plural = word + 'es'
        else:
            plural = word + 's'
        if plural not in occupied:
            occupied[plural] = idx
            group.append(plural)
    # The specified tokenizer preserves apostrophes: Earth's and plants' must
    # recall the same concepts as earth and plants, not become unknown words.
    for idx, group in enumerate(groups, 1):
        additions = []
        for word in group:
            if not word.isalpha() or len(word) < 3:
                continue
            forms = [word + "'s"]
            if word.endswith('s'):
                forms.append(word + "'")
            for form in forms:
                if form not in occupied:
                    occupied[form] = idx
                    additions.append(form)
        group.extend(additions)
    return groups, occupied, facts


def write_vocab():
    groups, _, _ = source()
    words = sorted({word for group in groups for word in group})
    (ROOT / 'vocab.txt').write_text('\n'.join(words) + '\n')


def canonical_facts(mapping, facts):
    """Canonical contrastive claims, without arbitrary positive-sign priority.

    A concept printed in BOTH contrast lists is shared context, not signed
    evidence. The bag architecture cannot bind it to two different qualifiers.
    Cancel that overlap symmetrically before atomic claims are accumulated.
    """
    merged = {}
    for weight, cues, yes, no in facts:
        src = tuple(sorted({mapping[w] for w in cues}))
        excluded = set(src) | {mapping['not']}
        positive = {mapping[w] for w in yes} - excluded
        negative = {mapping[w] for w in no} - excluded
        shared = positive & negative
        pos = tuple(sorted(positive - shared))
        neg = tuple(sorted(negative - shared))
        if shared and not (pos or neg):
            raise ValueError('All signed targets cancel for cues: '
                             + ' '.join(cues)
                             + '; move the distinguishing role into the cues')
        if not src or not (pos or neg):
            continue
        key = (src, pos, neg)
        merged[key] = max(weight, merged.get(key, 0.0))
    # Equal bags are indistinguishable to arch.py, including reversed cue order.
    # Reject contradictory signed targets rather than letting weights hide them.
    signs = defaultdict(lambda: (set(), set()))
    for src, pos, neg in sorted(merged):
        supported, opposed = signs[src]
        conflict = supported.intersection(neg) | opposed.intersection(pos)
        if conflict:
            raise ValueError('Opposite signs for identical cue bag: '
                             + str(src) + ' -> ' + str(sorted(conflict)))
        supported.update(pos)
        opposed.update(neg)
    return [(merged[key], *key) for key in sorted(merged)]


def _retain(row, target, value):
    """A repeated projected claim is one proof, not independent evidence.

    Keep the strongest proof on EACH side. Opposing contextual projections
    can cancel; a positive sign never silently wins a contradiction.
    """
    positive, negative = row.get(target, (0.0, 0.0))
    if value >= 0:
        positive = max(positive, float(value))
    else:
        negative = max(negative, float(-value))
    row[target] = (positive, negative)


def _associations(mapping, facts, C):
    """Compile duplicate-resistant signed associations before fixed hashing.

    Same-sign unary/pair projections use max, as do alternate proof paths in
    the ontology. Distinct active question clues still add during inference.
    No decisions here depend on question data, labels, or measured accuracy.
    """
    A = np.zeros((C, C), dtype=np.float32)
    first_order = defaultdict(dict)
    evidence = defaultdict(dict)
    unary_profiles = defaultdict(dict)
    atomic = defaultdict(dict)
    for weight, src, pos, neg in canonical_facts(mapping, facts):
        targets = [(b, weight / max(1., len(pos) ** 0.40)) for b in pos]
        targets += [(b, -weight * 0.78 / max(1., len(neg) ** 0.40)) for b in neg]
        for target, value in targets:
            previous = atomic[src].get(target, 0.0)
            if abs(value) > abs(previous):
                atomic[src][target] = value
    for src, ts in sorted(atomic.items()):
        targets = sorted(ts.items())
        first = 1.0 if len(src) == 1 else 0.16 / len(src)
        for a in src:
            for target, value in targets:
                _retain(first_order[a], target, first * value)
                if len(src) == 1:
                    unary_profiles[a][target] = value
        if len(src) >= 2:
            pair_scale = 1.0 / max(1., (len(src) * (len(src) - 1) / 2) ** 0.45)
            for pair in combinations(src, 2):
                for target, value in targets:
                    _retain(evidence[pair], target, value * pair_scale)
    # Query-focus reinforcement uses the authored unary profile only. Repeating
    # an explicit (anchor, function/property) fact is not an extra proof.
    for anchor, profile in sorted(unary_profiles.items()):
        for focus in sorted({mapping['property'], mapping['function']}):
            if anchor == focus or anchor == mapping['not']:
                continue
            row = evidence[tuple(sorted((anchor, focus)))]
            for target, value in sorted(profile.items()):
                if target not in (anchor, focus):
                    _retain(row, target, 0.70 * 3.0 * np.tanh(value / 3.0))
    for a, ts in sorted(first_order.items()):
        for target, (positive, negative) in sorted(ts.items()):
            A[a, target] = positive - negative
    # The in-place form avoids several full C-by-C temporaries; the formula
    # and float32 arithmetic are unchanged.
    A /= 3.0
    np.tanh(A, out=A)
    A *= 3.0
    pairs = {pair: {target: positive - negative
                    for target, (positive, negative) in sorted(ts.items())
                    if abs(positive - negative) > 1e-10}
             for pair, ts in sorted(evidence.items())}
    # Inherited exception/negation convention: reverse the bounded first-order
    # profile. This is logic authored into the weights, not a runtime branch.
    negator = mapping['not']
    for a in range(1, C):
        if a == negator:
            continue
        key = tuple(sorted((a, negator)))
        row = pairs.setdefault(key, {})
        for b in np.flatnonzero(A[a]):
            if b != negator:
                row[int(b)] = row.get(int(b), 0.) - 1.25 * float(A[a, b])
    return A, pairs


# Hand-authored salience classes, not corpus-derived IDFs. Alias membership is
# resolved to concept IDs before applying the classes; generic overrides broad.
SALIENCE_BROAD = '''change increase decrease energy water animal plant organism material
object substance system environment earth human use make produce cause need
result part move process example different similar best most least more less
greater smaller large small high low hot cold living nonliving grow survive
investigation experiment student scientist science measure time temperature
force motion speed light heat food cell rock air gas liquid solid'''.split()

SALIENCE_GENERIC = '''change form create make produce use contain fill flow spread
fixed contact direct indirect angle horizontal vertical up down forward back
toward away ratio equal unequal high low large small more less amount size
shape space distance length height width thick thin depth wet dry empty
path position location rate reach hide capture escape receive send carry
lift hold remove leave enter exit pass allow block prevent maintain regulate
support anchor attach connect join combine break bend split cut tear crush
rub slide roll swing bounce impact climb swim fly run walk jump breathe
chew swallow sleep wake grow develop mature age old young birth death parent
mother father male female test record collect compare evaluate explain
interpret analyze revise confirm reject prove apply replicate calibrate
estimate identify represent simplify improve replenish benefit harm risk
function structure characteristic role behavior learn experience practice
skill rapid slow sudden gradual local world introduce isolate barrier
repair replace heal measure observe infer predict experiment investigate
student scientist question problem example result effect best similar
different same part number type specific final starting resulting majority exert'''.split()


def _salience(mapping, facts, C):
    """Apply the unchanged authored precedence using a few indexed writes.

    A cue is active regardless of source order; repeating a fact adds no
    salience. A lexical entry with no fact stays dormant, so it cannot retrieve
    an accidental hash collision. No statistics or question data are used.
    """
    cue_ids, target_ids = set(), set()
    for _, cues, yes, no in facts:
        cue_ids.update(mapping[word] for word in cues)
        target_ids.update(mapping[word] for word in yes + no)
    active = cue_ids | target_ids
    broad = active & {mapping[word] for word in SALIENCE_BROAD if word in mapping}
    generic = active & {mapping[word] for word in SALIENCE_GENERIC if word in mapping}
    idf = torch.zeros((C,), dtype=torch.float32)
    for ids, value in ((target_ids, 0.72), (cue_ids, 1.0),
                       (broad, 0.82), (generic, 0.48)):
        if ids:
            idf[torch.tensor(sorted(ids), dtype=torch.int64)] = value
    idf[mapping['not']] = 1.15
    idf[0] = 0.0
    return idf


def build():
    groups, mapping, facts = source()
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    expected = sorted(mapping)
    if vocab != expected:
        raise ValueError('vocab.txt is stale or noncanonical; regenerate with python build.py')
    C = max(16, len(groups) + 1)
    concept = torch.tensor([0, 0] + [mapping[w] for w in vocab], dtype=torch.int64)
    idf = _salience(mapping, facts, C)
    A, pairs = _associations(mapping, facts, C)
    # No source text is needed once the tensors and exact pair claims exist.
    # Release it before allocating the large fixed-width lookup tables.
    del groups, mapping, facts, vocab, expected
    # The low-ID query-focus concepts create contiguous hash intervals. A
    # modulus near sqrt(2) * PRIME discourages short arithmetic repetitions,
    # while 64 slots per row keep total storage within the two-GB budget.
    # This is an integer-layout design, not an accuracy-selected parameter.
    H = 1414217
    rows = defaultdict(dict)
    for (a, b), ts in pairs.items():
        row = rows[(a * 1000003 + b) % H]
        for t, value in ts.items():
            row[t] = row.get(t, 0.) + value
    del pairs
    # Fixed sparse width bounds memory and scatter work. Retain the strongest
    # signed associations; the rule depends only on authored weights.
    # The inherited 64-slot layout keeps returned tensors near one GiB.
    R = min(64, max([1] + [len(row) for row in rows.values()]))
    row_ids = torch.zeros((H, R), dtype=torch.int64)
    row_w = torch.zeros((H, R), dtype=torch.float32)
    for h, ts in rows.items():
        nonzero = [(t, value) for t, value in ts.items() if abs(value) > 1e-10]
        entries = sorted(nonzero, key=lambda x: (-abs(x[1]), x[0]))[:R]
        for j, (t, value) in enumerate(entries):
            row_ids[h, j] = t
            row_w[h, j] = 5.0 * np.tanh(value / 5.0)
    return dict(concept=concept, idf=idf, A=torch.from_numpy(A),
                row_ids=row_ids, row_w=row_w, mix=torch.tensor(0.55, dtype=torch.float32))


if __name__ == '__main__':
    write_vocab()

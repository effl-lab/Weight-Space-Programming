"""Label-free specification, source, and resource checks; no science QA scoring."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build
import arch

torch.set_num_threads(4)
START = time.monotonic()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_checks(groups, mapping, facts):
    require(hashlib.sha256((ROOT / 'arch.py').read_bytes()).hexdigest() ==
            '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af',
            'architecture fingerprint')
    require((ROOT / 'vocab.txt').read_text().splitlines() == sorted(mapping),
            'canonical vocabulary')
    for a, b in [('atom', 'atoms'), ('avoid', 'avoiding'), ('not', 'no'),
                 ('photosynthesis', 'photosynthesize'), ('alveolus', 'alveoli'),
                 ('petrification', 'petrified'), ('stoma', 'stomatal'),
                 ('vaccinate', 'vaccination'), ('cartilage', 'cartilaginous'),
                 ('hearing', 'heard'), ('smell', 'smelling'), ('taste', 'tasted'),
                 ('m2', 'squaremetre'), ('cm3', 'cc'),
                 ('cm3', 'cubiccentimeter'), ('mps2', 'metrespersecondsquared'),
                 ('kilowatthour', 'kwh'), ('micrometer', 'micron')]:
        require(mapping[a] == mapping[b], 'lexical equivalence: ' + a)
    for a, b in [('pollinate', 'pollinator'), ('ethanol', 'alcohol'),
                 ('endothermic', 'endotherm'), ('electron', 'proton'),
                 ('resistance', 'resistor'), ('least', 'not'),
                 ('voltage', 'volt'), ('signal', 'impulse'),
                 ('sun', 'sunlight'), ('iodine', 'iodide'),
                 ('microscopic', 'microscope'), ('vaccination', 'vaccine'),
                 ('float', 'buoyancy'), ('ear', 'hearing'),
                 ('nose', 'smell'), ('tongue', 'taste'),
                 ('cartilaginousfish', 'bonyfish'), ('meter', 'm2'),
                 ('m2', 'm3'), ('centimeter', 'cm2'), ('cm2', 'cm3'),
                 ('millimeter', 'mm3'), ('mps', 'mps2'),
                 ('kilowatt', 'kilowatthour')]:
        require(mapping[a] != mapping[b], 'concept distinction: ' + a)
    # Canonicalization includes a full source contradiction check.
    canonical = build.canonical_facts(mapping, facts)
    records = defaultdict(lambda: (set(), set()))
    for _, src, yes, no in canonical:
        records[src][0].update(yes)
        records[src][1].update(no)
    def positive(cues, targets):
        key = tuple(sorted({mapping[x] for x in cues.split()}))
        require({mapping[x] for x in targets.split()} <= records[key][0],
                'authored positive claim: ' + cues + ' -> ' + targets)
    positive('dehydration kidney response', 'concentrated urine')
    positive('magnet cut halves', 'pole north south')
    positive('erosion prevent', 'plant vegetation')
    positive('astronaut weightless', 'gravity orbit')
    positive('photosynthesis energy input', 'sunlight')
    positive('photosynthesis matter input', 'water carbon dioxide')
    positive('respiration matter output', 'carbon dioxide water')
    positive('vacuum heat transfer', 'radiation')
    positive('outer core evidence', 'liquid')
    positive('magnet halves', 'pole north south')
    positive('walking ground', 'friction')
    positive('efficiency loss', 'energy')
    positive('boiling plateau', 'constant temperature')
    positive('alcohol examples', 'ethanol methanol')
    positive('phloem transported solution', 'water sugar')
    positive('seed gas exchange', 'oxygen respiration')
    positive('nervous synapse', 'chemical neurotransmitter')
    positive('stem cell', 'differentiation')
    positive('falling kinetic', 'increase')
    positive('falling potential', 'decrease')
    positive('rising kinetic', 'decrease')
    positive('rising potential', 'increase')
    positive('microscopic observation', 'microscope magnification')
    positive('microorganism', 'microscopic')
    positive('vaccination purpose', 'prevent disease')
    positive('vaccine administration', 'vaccination')
    positive('hearing', 'ear sound')
    positive('smell', 'nose')
    positive('taste', 'tongue')
    positive('submerged sinking', 'buoyancy')
    positive('floating equilibrium', 'buoyancy weight equal')
    positive('sinking terminal speed', 'balanced buoyancy drag')
    positive('shark', 'cartilage backbone')
    positive('dogfish', 'cartilage vertebrate')
    positive('salmon', 'bone vertebrate')
    positive('generator energy input', 'mechanical')
    positive('generator energy output', 'electrical')
    positive('motor energy input', 'electrical')
    positive('motor energy output', 'mechanical')
    positive('microphone energy input', 'sound')
    positive('microphone energy output', 'electrical')
    positive('speaker energy input', 'electrical')
    positive('speaker energy output', 'sound')
    positive('speed formula numerator', 'distance')
    positive('speed formula denominator', 'time')
    positive('water molecule hydrogen count', 'two')
    positive('water molecule oxygen count', 'one')
    positive('dioxide molecule carbon count', 'one')
    positive('dioxide molecule oxygen count', 'two')
    positive('solar eclipse shadow', 'moon')
    positive('lunar eclipse shadow', 'earth')
    positive('earth rotation direction', 'east')
    positive('wind pressure origin', 'high')
    positive('wind pressure destination', 'low')
    positive('prokaryote boundary', 'membrane')
    positive('bacteria dna location', 'cytoplasm')
    positive('photosynthesis', 'oxygen')
    positive('photosynthesis energy', 'light chemical')
    positive('m2', 'area')
    positive('cm2', 'area')
    positive('m3', 'volume')
    positive('cm3', 'volume milliliter')
    positive('dm3', 'volume liter')
    positive('mps', 'speed')
    positive('mps2', 'acceleration')
    positive('area si unit', 'm2')
    positive('volume si unit', 'm3')
    positive('acceleration si unit', 'mps2')
    positive('square centimeter', 'cm2')
    positive('cubic centimeter', 'cm3')
    positive('unit acceleration', 'meter second squared')
    positive('kilowatt', 'power')
    positive('kilowatthour', 'energy')
    positive('kilowatthour joule conversion', '3600000')
    require(build.TOKEN.findall('cm3') == ['cm3'], 'attached exponent token')
    require(build.TOKEN.findall('cm^3') == ['cm', '3'], 'separated exponent limitation')
    require(build.TOKEN.findall('kg/m3') == ['kg', 'm3'], 'compound unit tokens')
    for cues, targets in [('microorganism', 'microscope'), ('shark', 'bone'),
                          ('vertebrate', 'bone')]:
        key = tuple(sorted({mapping[x] for x in cues.split()}))
        require(not ({mapping[x] for x in targets.split()} & records[key][0]),
                'unsupported class trait: ' + cues + ' -> ' + targets)
    def not_opposed(cues, targets):
        key = tuple(sorted({mapping[x] for x in cues.split()}))
        require(not ({mapping[x] for x in targets.split()} & records[key][1]),
                'unsupported blanket opposition: ' + cues + ' -> ' + targets)
    not_opposed('precision', 'true accuracy value')
    not_opposed('protein function', 'energy')
    not_opposed('phloem function', 'water')
    not_opposed('magnet halves', 'pole')
    not_opposed('thermal equilibrium definition', 'energy heat')
    not_opposed('experiment pendulum length', 'change')
    not_opposed('experiment friction surface', 'change mass')
    not_opposed('fossil cast', 'impression')
    not_opposed('floating equilibrium', 'buoyancy')
    not_opposed('saltwater float', 'buoyancy')
    not_opposed('cytoplasm', 'genetic code')
    not_opposed('prokaryote organelle', 'membrane')
    not_opposed('virus structure', 'membrane')
    not_opposed('photosynthesis', 'oxygen')
    not_opposed('energy transformation respiration', 'chemical')
    not_opposed('generator energy transformation', 'electrical')
    not_opposed('unit acceleration', 'meter')
    not_opposed('unit pressure', 'kilogram')
    reactant_key = tuple(sorted({mapping['photosynthesis'], mapping['reactant']}))
    require(mapping['sunlight'] not in records[reactant_key][0],
            'energy source is distinct from material reactants')
    shark_key = (mapping['shark'],)
    require(mapping['bone'] in records[shark_key][1], 'cartilaginous skeleton distinction')
    return len(canonical)


def projection_checks():
    row = {}
    for x in (2., 1., 2., -0.5, -0.25):
        build._retain(row, 4, x)
    require(row[4] == (2., 0.5), 'signed max projection')
    m = {'not': 1, 'property': 2, 'function': 3,
         'a': 4, 'b': 5, 'c': 6, 'x': 7, 'y': 8}
    f = [(4., ['a', 'b'], ['x'], []),
         (2., ['a'], ['y'], [])]
    a, p = build._associations(m, f, 16)
    a2, p2 = build._associations(m, list(reversed(f)) + f * 3, 16)
    require(np.array_equal(a, a2) and p == p2, 'source order/duplicate invariance')
    extra = (4., ['a', 'b', 'c'], ['x'], [])
    a3, p3 = build._associations(m, f + [extra] * 5, 16)
    require(p3[(4, 5)][7] == p[(4, 5)][7], 'repeated qualified pair backoff')
    require(a3[4, 7] == a[4, 7], 'repeated qualified unary backoff')
    conflicted = f + [(1., ['a', 'b'], [], ['x'])]
    try:
        build.canonical_facts(m, conflicted)
    except ValueError:
        pass
    else:
        raise AssertionError('contradictory identical cues were accepted')
    shared = build.canonical_facts(m, [(2., ['a'], ['x', 'y'], ['x'])])
    require(shared == [(2., (4,), (8,), ())], 'symmetric shared-target cancellation')
    for yes, no in [(['x'], ['x']), (['x', 'y'], ['y', 'x'])]:
        try:
            build.canonical_facts(m, [(2., ['a'], yes, no)])
        except ValueError as exc:
            require('All signed targets cancel' in str(exc), 'actionable cancellation error')
        else:
            raise AssertionError('fully canceled signed claim was silently accepted')


def lexical_weight_checks(w, mapping, facts):
    # Independently enumerate each concept's authored role and precedence.
    # This catches errors that reference inference would share with the model,
    # such as an in-bounds but incorrect vocabulary-to-concept table.
    expected_lookup = np.array([0, 0] + [mapping[word] for word in sorted(mapping)],
                               dtype=np.int64)
    require(np.array_equal(w['concept'].numpy(), expected_lookup), 'exact concept lookup')
    cues = {mapping[word] for _, src, _, _ in facts for word in src}
    targets = {mapping[word] for _, _, yes, no in facts for word in yes + no}
    broad = {mapping[word] for word in build.SALIENCE_BROAD if word in mapping}
    generic = {mapping[word] for word in build.SALIENCE_GENERIC if word in mapping}
    expected_idf = np.zeros(len(w['idf']), dtype=np.float32)
    for concept in range(1, len(expected_idf)):
        value = 1.0 if concept in cues else 0.72 if concept in targets else 0.0
        if value and concept in broad:
            value = 0.82
        if value and concept in generic:
            value = 0.48
        if concept == mapping['not']:
            value = 1.15
        expected_idf[concept] = value
    require(np.array_equal(w['idf'].numpy(), expected_idf), 'exact authored salience')


def salience_checks():
    mapping = {'not': 1, 'water': 2, 'hydric': 2, 'make': 3, 'sensor': 4,
               'target': 5, 'unused': 6, 'energy': 7, 'solid': 8, 'function': 9}
    facts = [(2., ['sensor', 'hydric'], ['target', 'energy'], []),
             (1., ['make'], ['sensor'], [])]
    expected = torch.tensor([0., 1.15, 0.82, 0.48, 1., 0.72, 0., 0.82] + [0.] * 8,
                            dtype=torch.float32)
    actual = build._salience(mapping, facts, 16)
    require(torch.equal(actual, expected), 'salience roles, aliases, and dormant overrides')
    repeated = build._salience(mapping, list(reversed(facts)) + facts * 3, 16)
    require(torch.equal(actual, repeated), 'salience source order/duplicate invariance')


def corruption_checks(w, mapping, facts):
    # Deliberately malformed small tables must fail even though bounds and
    # dtypes still satisfy the architecture. The real model is never mutated.
    changed = dict(w, concept=w['concept'].clone())
    changed['concept'][2] = int(w['concept'][2]) % (len(w['idf']) - 1) + 1
    try:
        lexical_weight_checks(changed, mapping, facts)
    except AssertionError as exc:
        require(str(exc) == 'exact concept lookup', 'lookup corruption diagnostic')
    else:
        raise AssertionError('in-bounds lookup corruption passed verification')
    changed = dict(w, idf=w['idf'].clone())
    changed['idf'][mapping['water']] = 0.48
    try:
        lexical_weight_checks(changed, mapping, facts)
    except AssertionError as exc:
        require(str(exc) == 'exact authored salience', 'salience corruption diagnostic')
    else:
        raise AssertionError('incorrect salience passed verification')


def weight_checks(w, groups, mapping):
    require(set(w) == {'concept', 'idf', 'A', 'row_ids', 'row_w', 'mix'}, 'weight names')
    C = max(16, len(groups) + 1)
    V = len(mapping) + 2
    H, R = w['row_ids'].shape
    require(w['concept'].shape == (V,), 'concept shape')
    require(w['idf'].shape == (C,), 'salience shape')
    require(w['A'].shape == (C, C), 'unary shape')
    require(w['row_w'].shape == (H, R), 'pair weight shape')
    require(H == 1414217 and 1 <= R <= 64, 'memory layout')
    require(w['mix'].shape == (), 'mixture scalar')
    for name, value in w.items():
        require(value.device.type == 'cpu' and not value.is_sparse, 'dense CPU tensor: ' + name)
        require(value.dtype == (torch.int64 if name in ('concept', 'row_ids') else torch.float32),
                'dtype: ' + name)
    require(w['concept'][:2].tolist() == [0, 0], 'padding and unknown concepts')
    require(int(w['concept'].min()) >= 0 and int(w['concept'].max()) < C, 'word concept bounds')
    require(float(w['idf'][0]) == 0., 'null salience')
    for name in ('idf', 'A', 'mix'):
        require(bool(torch.isfinite(w[name]).all()), 'finite tensor: ' + name)
    require(bool((w['idf'] >= 0).all()), 'nonnegative salience')
    require(bool((w['A'][0] == 0).all() and (w['A'][:, 0] == 0).all()), 'null unary row/column')
    for offset in range(0, H, 8192):
        ids, values = w['row_ids'][offset:offset+8192], w['row_w'][offset:offset+8192]
        require(int(ids.min()) >= 0 and int(ids.max()) < C, 'pair concept bounds')
        require(bool(torch.isfinite(values).all()), 'finite pair weights')
        require(bool((values[ids == 0] == 0).all()), 'unused slot weights')


def packing_checks(w, mapping, facts):
    C = len(w['idf'])
    expected_A, pairs = build._associations(mapping, facts, C)
    require(np.array_equal(w['A'].numpy(), expected_A), 'unary tensor matches compiler')
    rows = defaultdict(dict)
    H, R = w['row_ids'].shape
    for (a, b), targets in pairs.items():
        h = (a * 1000003 + b) % H
        for target, value in targets.items():
            rows[h][target] = rows[h].get(target, 0.) + value
    occupied = sorted(rows)
    occupied_mask = np.zeros(H, dtype=np.bool_)
    truncation = 0
    for h in occupied:
        occupied_mask[h] = True
        entries = sorted(((t, v) for t, v in rows[h].items() if abs(v) > 1e-10),
                         key=lambda x: (-abs(x[1]), x[0]))
        truncation += int(len(entries) > R)
        entries = entries[:R]
        ids = np.zeros(R, dtype=np.int64)
        values = np.zeros(R, dtype=np.float32)
        for j, (target, value) in enumerate(entries):
            ids[j] = target
            values[j] = 5. * np.tanh(value / 5.)
        require(np.array_equal(w['row_ids'][h].numpy(), ids), 'packed IDs')
        require(np.array_equal(w['row_w'][h].numpy(), values), 'packed values')
    for offset in range(0, H, 8192):
        absent = torch.from_numpy(~occupied_mask[offset:offset+8192])
        require(not bool(w['row_ids'][offset:offset+8192][absent].count_nonzero()),
                'empty-row IDs')
        require(not bool(w['row_w'][offset:offset+8192][absent].count_nonzero()),
                'empty-row weights')
    return {'pair_keys': len(pairs), 'occupied_hash_rows': len(occupied),
            'truncated_rows': truncation}


def reference_forward(w, q, o):
    # Independent bag construction and sparse-row reduction for small inputs.
    # Use torch topk only to match its specified tie-breaking among equal IDFs.
    C = len(w['idf'])
    concept, idf = w['concept'].numpy(), w['idf'].numpy()
    answer = []
    for qq, oo in zip(q.numpy(), o.numpy()):
        qset = set(concept[qq]) - {0}
        qb = np.zeros(C, dtype=np.float32)
        for a in qset:
            qb[a] = idf[a]
        top = torch.from_numpy(qb).topk(16)
        read = np.zeros(C, dtype=np.float64)
        for i in range(16):
            for j in range(i+1, 16):
                a, b = sorted((int(top.indices[i]), int(top.indices[j])))
                strength = float(top.values[i]) * float(top.values[j])
                h = (a * 1000003 + b) % w['row_ids'].shape[0]
                np.add.at(read, w['row_ids'][h].numpy(), strength * w['row_w'][h].numpy())
        read += float(w['mix']) * (qb @ w['A'].numpy())
        scores = []
        for option in oo:
            option_set = set(concept[option]) - qset - {0}
            scores.append(sum(read[a] * float(idf[a]) for a in option_set))
        answer.append(scores)
    return torch.tensor(answer, dtype=torch.float32)


def lexical_bag_checks(w, mapping):
    # These are feature-preservation checks, not question-answer predictions.
    # Four identical one-token options ensure that no answer label is involved.
    vocabulary = (ROOT / 'vocab.txt').read_text().splitlines()
    index = {word: i + 2 for i, word in enumerate(vocabulary)}
    pairs = [('microscopic', 'microscope'), ('vaccination', 'vaccine'),
             ('float', 'buoyancy'), ('hearing', 'ear'),
             ('smell', 'nose'), ('taste', 'tongue'),
             ('meter', 'm2'), ('m2', 'm3'), ('cm2', 'cm3'),
             ('mps', 'mps2'), ('kilowatt', 'kilowatthour')]
    for left, right in pairs:
        for cue, target in [(left, right), (right, left)]:
            q = torch.zeros((1, 128), dtype=torch.int64)
            o = torch.zeros((1, 4, 48), dtype=torch.int64)
            q[0, 0] = index[cue]
            o[:, :, 0] = index[target]
            qb, ob = arch.bags(w, q, o)
            require(float(qb[0, mapping[cue]]) > 0., 'active cue: ' + cue)
            require(bool((ob[0, :, mapping[target]] > 0.).all()),
                    'distinct answer feature retained: ' + cue + '/' + target)
            o[:, :, 0] = index[cue]
            _, shared = arch.bags(w, q, o)
            require(not bool(shared.count_nonzero()), 'shared lexical feature excluded')


def inference_checks(w):
    generator = torch.Generator().manual_seed(70517)
    V = len(w['concept'])
    q = torch.zeros(12, 128, dtype=torch.int64)
    o = torch.zeros(12, 4, 48, dtype=torch.int64)
    q[:, :19] = torch.randint(2, V, (12, 19), generator=generator)
    o[:, :, :7] = torch.randint(2, V, (12, 4, 7), generator=generator)
    # Some deliberately shared concepts exercise question/answer exclusion.
    o[:, :, 0] = q[:, 0, None]
    with torch.no_grad():
        logits = arch.forward(w, q, o)
        require(bool(torch.isfinite(logits).all()), 'finite logits')
        reference = reference_forward(w, q[:4], o[:4])
        require(torch.allclose(logits[:4], reference, atol=1e-5, rtol=1e-4), 'reference inference')
        perm = torch.tensor([3, 0, 2, 1])
        require(torch.allclose(arch.forward(w, q, o[:, perm]), logits[:, perm], atol=1e-6),
                'option permutation')
        require(torch.equal(arch.forward(w, q[:, torch.randperm(128, generator=generator)], o), logits),
                'question token permutation')
        repeated = q.clone()
        repeated[:, 32:51] = q[:, :19]
        require(torch.equal(arch.forward(w, repeated, o), logits), 'question token multiplicity')
        option_repeated = o.clone()
        option_repeated[:, :, 16:23] = o[:, :, :7]
        require(torch.equal(arch.forward(w, q, option_repeated), logits), 'option token multiplicity')
        one_at_time = torch.cat([arch.forward(w, q[i:i+1], o[i:i+1]) for i in range(12)])
        require(torch.allclose(one_at_time, logits, atol=1e-5, rtol=1e-4), 'batch invariance')
        zq, zo = torch.zeros_like(q), torch.zeros_like(o)
        require(not bool(arch.forward(w, zq, zo).count_nonzero()), 'empty input tie')
        require(not bool(arch.forward(w, zq + 1, zo + 1).count_nonzero()), 'unknown input tie')
        identical = o[:, :1].expand(-1, 4, -1)
        equal = arch.forward(w, q, identical)
        require(torch.equal(equal[:, :1].expand(-1, 4), equal), 'identical options tie')
        shared = q[:, :48, None].transpose(1, 2).expand(-1, 4, -1)
        require(not bool(arch.forward(w, q, shared).count_nonzero()), 'question concept exclusion')
        started = time.monotonic()
        # Synthetic word bags, not ARC or authored QA examples; no answer labels.
        for start in range(0, 3530, 64):
            n = min(64, 3530 - start)
            qq = torch.zeros(n, 128, dtype=torch.int64)
            oo = torch.zeros(n, 4, 48, dtype=torch.int64)
            qq[:, :32] = torch.randint(2, V, (n, 32), generator=generator)
            oo[:, :, :12] = torch.randint(2, V, (n, 4, 12), generator=generator)
            require(bool(torch.isfinite(arch.forward(w, qq, oo)).all()), 'synthetic workload')
    return time.monotonic() - started


def main():
    groups, mapping, facts = build.source()
    canonical_count = source_checks(groups, mapping, facts)
    projection_checks()
    salience_checks()
    t = time.monotonic()
    w = build.build()
    build_seconds = time.monotonic() - t
    weight_checks(w, groups, mapping)
    lexical_weight_checks(w, mapping, facts)
    corruption_checks(w, mapping, facts)
    statistics = packing_checks(w, mapping, facts)
    lexical_bag_checks(w, mapping)
    workload_seconds = inference_checks(w)
    elapsed = time.monotonic() - START
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(rss < 2_000_000_000, 'peak RSS exceeds two GB: ' + str(rss))
    require(elapsed < 1200, 'verification exceeds twenty minutes')
    result = dict(passed=True, vocabulary=len(mapping), concepts=len(groups)+1,
                  records_including_ontology=len(facts), canonical_records=canonical_count,
                  weights=sum(v.numel() for v in w.values()),
                  weight_bytes=sum(v.numel()*v.element_size() for v in w.values()),
                  build_seconds=build_seconds, synthetic_3530_seconds=workload_seconds,
                  verification_seconds=elapsed, peak_rss_bytes=rss, **statistics)
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

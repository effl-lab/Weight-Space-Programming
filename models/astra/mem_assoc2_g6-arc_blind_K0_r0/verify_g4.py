"""Label-free source/compiler/runtime contracts; no benchmark questions or fitting."""
from pathlib import Path
import gc
import hashlib
import resource
import time

import numpy as np
import torch
import arch
import build
from audit import audit_rules


REVIEWED_PAIRS = {
    'coasting friction', 'interaction forces', 'pendulum shorter', 'pendulum longer',
    'lubricated axle', 'static friction', 'pressure wider', 'pressure narrower',
    'wood handle', 'thermos stopper', 'double glazing', 'pitch higher', 'pitch lower',
    'carpet echo', 'curtain echo', 'sound soft', 'vacuum bell', 'refraction oblique',
    'stainless steel', 'plastic handle', 'strength material', 'switch activated',
    'electromagnet disconnected', 'constant velocity', 'balanced force',
    'unbalanced force', 'balanced forces', 'newton third', 'longer resistance',
    'longer current', 'thicker resistance', 'thicker current', 'shorter resistance',
    'shorter current', 'thinner resistance', 'thinner current', 'solubility stirring',
    'solubility crushing', 'stainless chromium', 'switch deactivated',
    'coil deactivated', 'coil energized', 'continuous circuit', 'disconnected motor',
    'disconnected battery', 'mechanical equilibrium', 'static equilibrium',
    'buffer ph', 'acoustic absorption', 'sound evacuated', 'vacuum transmission',
    'refraction normal', 'refraction frequency', 'period shorter', 'period longer',
    'viral treatment', 'viral prevention', 'antibiotic resistance', 'resistant bacteria',
    'cell hypertonic', 'cell hypotonic', 'wall turgor', 'waterlogged roots',
    'plant drought', 'stomata closed', 'bark removed', 'red cell', 'red biconcave',
    'white cell', 'skin barrier', 'vitamin d', 'vitamin c', 'inbreeding recessive',
    'inbreeding risk', 'producer decline', 'predator increase', 'predator scarcity',
    'prey scarcity', 'carrying drought', 'population overcrowding',
    'immigration population', 'emigration population', 'invasive competition',
    'invasive predator', 'habitat fragmentation', 'wetland removal',
    'forest clearing', 'nocturnal desert', 'c temperature', 'c vitamin',
    'c element', 'f temperature', 'f element', 'k temperature', 'k vitamin',
    'k element', 'n force', 'n element', 'n compass', 'm length', 's time',
}


EARTH_REVIEWED_PAIRS = {
    'windbreak trees', 'contour plowing', 'terrace slope', 'bare hillside',
    'root network', 'burned hillside', 'paved surface', 'obsidian cooling',
    'hurricane inland', 'seismic p', 'star red', 'star blue', 'contour farming',
    'windbreak purpose', 'terrace farming', 'windbreak farming', 'cover crop',
    'permeable pavement', 'windbreak speed', 'paving runoff', 'paving infiltration',
    'paving recharge', 'permeable paving', 'compaction infiltration',
    'compaction runoff', 'permeability isolated', 'saturated zone',
    'obsidian texture', 'hurricane landfall', 'humidity warming', 'humidity cooling',
    'dew phase', 'frost phase', 'seismic shear', 'seismic compression',
    'landice melting', 'seaice melting', 'thermal seawater',
}
REVIEWED_PAIRS |= EARTH_REVIEWED_PAIRS

CHEMISTRY_REVIEWED_PAIRS = {
    'ionic solid', 'ionic molten', 'ionic dissolved', 'sodiumchloride crystal',
    'sodiumchloride dissolved', 'sodiumchloride molten', 'sodiumchloride solid',
    'catalyst pathway', 'catalyst quantity', 'catalyst equilibrium',
    'catalyst activation', 'catalyst enthalpy', 'catalyst forward',
    'catalyst reverse', 'catalyst consumed', 'catalyst selectivity',
    'incomplete combustion', 'carbonaceous incomplete', 'methane combustion',
    'hydrogen burning', 'electrolyte carriers', 'metal carriers',
    'ionic melting', 'ionic dissolution', 'saltwater carriers',
    'sugar aqueous', 'sugar conductivity', 'sucrose dissolved',
    'weak electrolyte', 'strong electrolyte', 'hydrocarbon complete',
    'hydrocarbon incomplete', 'methane complete', 'methane incomplete',
    'carbon complete', 'carbon incomplete', 'magnesium combustion',
    'sulfur combustion', 'endothermic enthalpy', 'exothermic enthalpy',
}
REVIEWED_PAIRS |= CHEMISTRY_REVIEWED_PAIRS

CIRCUIT_REVIEWED_PAIRS = {
    'complete circuit', 'incomplete circuit', 'close circuit', 'close switch',
    'closed circuit', 'open circuit', 'switch open', 'switch closed',
    'battery disconnected', 'broken filament', 'loose wire', 'fuse melted',
    'circuit reconnection', 'wire disconnected',
}
CIRCUIT_REVIEWED_CUES = CIRCUIT_REVIEWED_PAIRS | {
    'closed switch path', 'open switch path', 'circuit complete path',
    'circuit broken path', 'circuit bridge metal', 'circuit bridge plastic',
    'bulb broken wire',
}
REVIEWED_PAIRS |= CIRCUIT_REVIEWED_PAIRS


def source_review_contracts(vocab, aliases, rules):
    known = set(vocab)
    def concepts(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}
    def terms(text):
        return concepts(text)
    forbidden = {'switch on', 'force coasting', 'equal opposite objects',
                 'force equal opposite', 'wire length increase',
                 'wire thickness increase', 'mirror image'}
    assert not forbidden.intersection(r[0] for r in rules)
    requirements = {
        'vacuum bell': ('silence', 'sound'),
        'electromagnet disconnected': ('absent stopped', 'field current'),
        'constant velocity': ('balanced equilibrium', 'acceleration'),
        'switch activated': ('closed current', 'open'),
        'stainless steel': ('iron chromium', 'corrosion'),
        'buffer': ('ph stability', 'change'),
        'brittleness': ('fracture shatter', 'ductility plasticity'),
        'soundproof': ('absorption insulation', 'transmission'),
        'pressure wider': ('decrease', 'increase'),
        'pressure narrower': ('increase', 'decrease'),
        'pendulum shorter': ('faster', 'slower'),
        'pendulum longer': ('slower', 'faster'),
        'wall turgor': ('support firm', 'bursting'),
        'skin barrier': ('protection', 'infection dehydration'),
        'stomata closed': ('water conservation', 'transpiration'),
        'nocturnal desert': ('night cooler', 'daytime'),
        'windbreak': ('trees shelter conservation', 'erosion'),
        'windbreak trees': ('soil conservation', 'erosion'),
        'contour plowing': ('soil retention', 'runoff erosion'),
        'pavement': ('impermeable runoff', 'infiltration'),
        'paved surface': ('runoff', 'infiltration'),
        'permeable pavement': ('infiltration recharge', 'runoff'),
        'obsidian cooling': ('glassy amorphous', 'crystals'),
        'hurricane inland': ('weakens', 'intensification'),
        'star red': ('cooler', 'hotter'),
        'star blue': ('hotter', 'cooler'),
        'paving recharge': ('decrease', 'increase'),
        'dew phase': ('liquid condensation', 'solid'),
        'frost phase': ('solid deposition', 'liquid'),
        'ionic solid': ('fixed ions insulator', 'conductivity'),
        'ionic molten': ('mobile ions conductor', 'insulator'),
        'ionic dissolved': ('mobile ions conductor', 'insulator'),
        'sodiumchloride solid': ('fixed ions insulator', 'conductivity'),
        'sodiumchloride molten': ('mobile ions conductivity', 'insulator'),
        'electrolyte carriers': ('mobile ions', 'electrons'),
        'metal carriers': ('mobile electrons', 'ions'),
        'sugar aqueous': ('molecules nonelectrolyte', 'ions'),
        'nonelectrolyte': ('dissolved molecules', 'ionization'),
        'hydrogen burning': ('water energy', 'carbon dioxide'),
        'hydrocarbon complete': ('carbon dioxide water', 'monoxide soot'),
        'methane complete': ('carbon dioxide water', 'monoxide soot'),
        'catalyst enthalpy': ('unchanged', 'increase decrease'),
        'endothermic enthalpy': ('positive', 'negative'),
        'exothermic enthalpy': ('negative', 'positive'),
        'incomplete circuit': ('open stopped', 'current'),
        'battery disconnected': ('stopped', 'current'),
        'broken filament': ('open stopped', 'current'),
        'fuse melted': ('open stopped', 'current'),
        'loose wire': ('intermittent unreliable', ''),
    }
    for cue, (positive, negative) in requirements.items():
        matching = [r for r in rules if concepts(r[0]) == concepts(cue)]
        assert matching, cue
        assert any(terms(positive) <= concepts(p) and terms(negative) <= concepts(n)
                   for _, p, n, _ in matching), cue
        assert all(not (terms(negative) & concepts(p)) for _, p, _, _ in matching), cue
    ambiguous_symbols = concepts('c f k n')
    reviewed_conditions = {frozenset(concepts(cue))
                           for cue in EARTH_REVIEWED_PAIRS | CHEMISTRY_REVIEWED_PAIRS
                           | CIRCUIT_REVIEWED_CUES}
    for cue, positive, negative, scale in rules:
        if frozenset(concepts(cue)) in reviewed_conditions:
            assert scale < 0, ('Reviewed condition needs pair-only recall', cue)
        if concepts(cue) == concepts('combustion'):
            assert not (concepts('carbon dioxide water') & concepts(positive)), cue
        assert cue != 'catalyst products', 'Unqualified catalyst product assertion'
        if concepts(cue) & ambiguous_symbols:
            assert scale < 0, ('Ambiguous symbol needs contextual recall', cue)
        if cue in {'vitamin c', 'vitamin d'}:
            assert scale < 0, ('Vitamin identity needs contextual recall', cue)
        if concepts(cue) in (concepts('solubility stirring'), concepts('solubility crushing')):
            assert 'amount' not in concepts(negative), cue
        if cue == 'gas solubility':
            assert not (concepts('increase decrease') & concepts(positive)), cue
        if cue == 'viral treatment':
            assert not (concepts('vaccination') & concepts(positive)), cue


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    expected_arch = '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af'
    assert hashlib.sha256((root / 'arch.py').read_bytes()).hexdigest() == expected_arch
    vocab, aliases, rules = build.lexicon()
    assert vocab == (root / 'vocab.txt').read_text().splitlines()
    assert not audit_rules(vocab, aliases, rules)
    source_review_contracts(vocab, aliases, rules)
    assert len(vocab) == len(set(vocab))
    assert build.lexicon() == (vocab, aliases, rules)
    compiled_rules = build.compile_rules(vocab, aliases, rules)
    assert not audit_rules(vocab, aliases, compiled_rules)
    assert build.compile_rules(vocab, aliases, rules + rules) == compiled_rules
    assert build.phrase_variants('carbon dioxide water') == (
        'carbon dioxide water', 'carbondioxide water')
    assert len(build.phrase_variants('calcium carbonate carbon dioxide')) == 4
    # Generation 6 explicitly adds carbon monoxide as a compound-name variant.
    assert build.phrase_variants('carbon monoxide') == ('carbon monoxide', 'carbonmonoxide')
    for text in ('dioxide', 'carbon monoxideish', 'carbon dioxideish', "carbon dioxide's"):
        assert build.phrase_variants(text) == (text,)
    bridge = [('carbon dioxide', 'carbondioxide', '', 1.0)]
    assert build.compile_rules(vocab, aliases, bridge) == bridge
    pair_only = [('carbon dioxide', 'oxygen', '', -1.0)]
    assert build.compile_rules(vocab, aliases, pair_only) == pair_only

    def canonical(word):
        assert word in vocab, word
        return aliases.get(word, word)

    for left, right in [
        ('lowers', 'decrease'), ('lowered', 'decrease'),
        ('emissions', 'release'), ('crystallizes', 'crystallization'),
        ('secretion', 'secrete'), ('filled', 'fill'),
        ('replication', 'repeat'), ('reproducibility', 'reproducible'),
        ('refractive', 'refraction'), ('weathered', 'weathering'),
        ('calibrated', 'calibrate'), ('germinated', 'germination'),
        ('acoustic', 'acoustics'), ('damped', 'damping'),
        ('deactivated', 'deactivate'), ('energized', 'energize'),
        ('durability', 'durable'), ('incident', 'incidence'),
        ('paved', 'paving'), ('intensifying', 'intensification'),
        ('dissipated', 'dissipation'),
        ('ionised', 'ionization'), ('solvated', 'solvation'),
        ('no', 'not'), ('without', 'not'), ('absent', 'not'), ('never', 'not'),
        ('co2', 'carbondioxide'), ('so2', 'sulfurdioxide'),
        ('nacl', 'sodiumchloride'), ('caco3', 'calciumcarbonate'),
        ('h2o2', 'hydrogenperoxide'), ('sio2', 'silica'),
        ('nh3', 'ammonia'), ('ch4', 'methane'), ('hcl', 'hydrogenchloride'),
        ('h2o', 'water'), ('o2', 'oxygen'), ('n2', 'nitrogen'),
        ('cm', 'centimeter'), ('kg', 'kilogram'), ('ml', 'milliliter'),
        ('hypertonicity', 'hypertonic'), ('waterlogged', 'waterlogging'),
        ('b1', 'thiamine'), ('b12', 'cobalamin'), ('d3', 'cholecalciferol'),
        ("earth's", 'earth'), ("'copper'", 'copper'),
    ]:
        assert canonical(left) == canonical(right), (left, right)
    for left, right in [
        ('weather', 'weathering'), ('reproduce', 'reproducible'),
        ('speed', 'velocity'), ('rotation', 'revolution'),
        ('conduct', 'conduction'), ('conduction', 'convection'),
        ('resist', 'resistance'), ('absorb', 'adsorption'),
        ('genotype', 'phenotype'), ('mitosis', 'meiosis'),
        ('mass', 'weight'), ('heat', 'temperature'),
        ('permeability', 'porosity'), ('acid', 'base'),
        ('above', 'below'), ('before', 'after'), ('open', 'closed'),
        ('damping', 'damp'), ('activated', 'deactivated'),
        ('frictionless', 'friction'), ('resultant', 'result'),
        ('c', 'celsius'), ('c', 'carbon'), ('c', 'ascorbic'),
        ('f', 'fahrenheit'), ('f', 'fluorine'),
        ('k', 'kelvin'), ('k', 'potassium'), ('k', 'phylloquinone'),
        ('n', 'newton'), ('n', 'nitrogen'), ('n', 'north'),
        ('co2', 'dioxide'), ('co2', 'so2'), ('h2o2', 'h2o'),
        ('nacl', 'salt'), ('hypertonic', 'hypotonic'),
        ('hypotonic', 'isotonic'), ('hypoxia', 'anoxia'),
        ('paving', 'pavement'), ('dew', 'frost'), ('landice', 'seaice'),
        ('permeable', 'impermeable'), ('amorphous', 'crystalline'),
        ('electrolyte', 'nonelectrolyte'), ('ionization', 'dissolution'),
        ('selectivity', 'selection'), ('complete', 'incomplete'),
    ]:
        assert canonical(left) != canonical(right), (left, right)

    # Source-specific pair-only contracts. These are direct tensor sign tests,
    # not multiple-choice predictions. No scores select a design change.
    cues = {'kinetic faster', 'kinetic slower', 'vacuum insulation',
            'fossil preservation', 'circuit gap', 'atom expansion'} | REVIEWED_PAIRS
    selected = [r for r in rules if r[0] in cues and r[3] < 0]
    assert {r[0] for r in selected} == cues
    assert all(r[3] < 0 for r in selected)
    original_lexicon = build.lexicon
    try:
        build.lexicon = lambda: (vocab, aliases, selected)
        isolated = build.build()
        assert isolated['A'].count_nonzero().item() == 0
        index = {word: i + 2 for i, word in enumerate(vocab)}

        def concepts(text, w):
            return sorted({int(w['concept'][index[t]])
                           for t in build.TOKEN.findall(text.lower()) if t in index})

        wanted_rows = set()
        for cue, pos, neg, scale in selected:
            qs = concepts(cue, isolated)
            assert len(qs) == 2, (cue, qs)
            a, b = qs
            row = (a * 1000003 + b) % isolated['row_ids'].shape[0]
            wanted_rows.add(row)
            actual = {int(k): float(v) for k, v in
                      zip(isolated['row_ids'][row], isolated['row_w'][row]) if int(k)}
            for target in set(concepts(pos, isolated)) - set(qs):
                assert actual[target] > 0, (cue, target, actual)
            for target in set(concepts(neg, isolated)) - set(qs):
                assert actual[target] < 0, (cue, target, actual)
        occupied = set(torch.where(isolated['row_w'].abs().sum(1) != 0)[0].tolist())
        assert occupied == wanted_rows
        build.lexicon = lambda: (vocab, aliases, selected + selected)
        duplicate = build.build()
        assert all(torch.equal(isolated[k], duplicate[k]) for k in isolated)
        del isolated, duplicate

        # The reviewed single-concept definitions must also carry their signs in
        # first-order memory, independently of any multiple-choice question.
        definition_cues = {'buffer', 'soundproof', 'brittleness', 'windbreak', 'pavement',
                           'nonelectrolyte', 'combustion'}
        definitions = [r for r in rules if r[0] in definition_cues]
        assert len(definitions) == len(definition_cues)
        build.lexicon = lambda: (vocab, aliases, definitions)
        direct = build.build()
        for cue, pos, neg, scale in definitions:
            qs = concepts(cue, direct)
            assert len(qs) == 1 and scale > 0
            qid = qs[0]
            for target in set(concepts(pos, direct)) - set(qs):
                assert direct['A'][qid, target] > 0, (cue, target)
            for target in set(concepts(neg, direct)) - set(qs):
                assert direct['A'][qid, target] < 0, (cue, target)
        del direct

        # Formula/name variants preserve signed scientific roles and conditions.
        # These authored tensor fixtures have no answer options or score labels.
        lexical_facts = [
            ('carbon dioxide respiration', 'water', '', -1.0),
            ('photosynthesis products', 'oxygen glucose', 'carbon dioxide', -1.0),
            ('calcium carbonate acid', 'carbon dioxide water', '', -1.0),
        ]
        expanded = build.compile_rules(vocab, aliases, lexical_facts)
        assert ('calciumcarbonate acid', 'carbondioxide water', '', -1.0) in expanded
        assert all(r[3] == -1.0 for r in expanded)
        assert not audit_rules(vocab, aliases, expanded)
        build.lexicon = lambda: (vocab, aliases, lexical_facts)
        lexical = build.build()
        assert lexical['A'].count_nonzero().item() == 0
        for cue, target, sign in [
            ('co2 respiration', 'water', 1),
            ('photosynthesis products', 'co2', -1),
            ('calciumcarbonate acid', 'co2', 1),
            ('calciumcarbonate acid', 'water', 1),
        ]:
            qs = concepts(cue, lexical)
            assert len(qs) == 2
            row = (qs[0] * 1000003 + qs[1]) % lexical['row_ids'].shape[0]
            target_id = concepts(target, lexical)[0]
            found = lexical['row_w'][row][lexical['row_ids'][row] == target_id]
            assert found.numel() == 1 and float(found[0]) * sign > 0, (cue, target)
        del lexical

        # Negation is retained in every recalled pair, even for positive-scale
        # authored conditions. These are compiler fixtures, not answer labels.
        assert build.recall_pairs([2, 3, 5], {2}) == ((2, 3), (2, 5))
        assert build.recall_pairs([2, 3, 5], {7}) == ((2, 3), (2, 5), (3, 5))
        assert build.recall_pairs([2], {2}) == ()
        for lone_negator in ('not', 'no', 'except'):
            errors = audit_rules(vocab, aliases, [(lone_negator, 'current', '', 1.0)])
            assert any('no usable concept pair' in error for error in errors)
        negated_facts = [('circuit not closed', 'open', 'current', 1.0)]
        assert not audit_rules(vocab, aliases, negated_facts)
        build.lexicon = lambda: (vocab, aliases, negated_facts)
        negated = build.build()
        assert negated['A'].count_nonzero().item() == 0
        negator = concepts('not', negated)[0]
        qs = concepts('circuit not closed', negated)
        expected_pairs = build.recall_pairs(qs, {negator})
        assert len(expected_pairs) == 2
        assert all(negator in pair for pair in expected_pairs)
        expected_rows = {(a * 1000003 + b) % negated['row_ids'].shape[0]
                         for a, b in expected_pairs}
        actual_rows = set(torch.where(negated['row_w'].abs().sum(1) != 0)[0].tolist())
        assert actual_rows == expected_rows
        for row in expected_rows:
            for term, sign in [('open', 1), ('current', -1)]:
                target = concepts(term, negated)[0]
                found = negated['row_w'][row][negated['row_ids'][row] == target]
                assert found.numel() == 1 and float(found[0]) * sign > 0
        del negated

        # Direct-definition inversion respects an explicitly authored exception
        # for the exact semantic pair. This does not add facts to the full model.
        fixture = [('copper', 'conductor', '', 1.0)]
        for explicit_exception in (False, True):
            facts = fixture + ([('copper not', 'conductor', '', -1.0)]
                               if explicit_exception else [])
            build.lexicon = lambda: (vocab, aliases, facts)
            inverted = build.build()
            target = concepts('conductor', inverted)[0]
            for negator_word in ('not', 'except'):
                a, b = concepts('copper ' + negator_word, inverted)
                row = (a * 1000003 + b) % inverted['row_ids'].shape[0]
                found = inverted['row_w'][row][inverted['row_ids'][row] == target]
                assert found.numel() == 1
                if explicit_exception and negator_word == 'not':
                    assert float(found[0]) == 1.0
                else:
                    assert float(found[0]) < 0
            del inverted
    finally:
        build.lexicon = original_lexicon
    gc.collect()

    start = time.perf_counter()
    weights = build.build()
    elapsed_build = time.perf_counter() - start
    assert set(weights) == {'concept', 'idf', 'A', 'row_ids', 'row_w', 'mix'}
    C = len(weights['idf'])
    H, R = weights['row_ids'].shape
    assert C >= 16 and H == 1048573
    assert weights['concept'].shape == (len(vocab) + 2,)
    assert weights['concept'].dtype == weights['row_ids'].dtype == torch.int64
    assert weights['A'].shape == (C, C)
    assert weights['row_w'].shape == (H, R)
    assert weights['mix'].shape == ()
    assert float(weights['mix']) == float(np.float32(.65))
    assert weights['concept'][:2].tolist() == [0, 0]
    assert int(weights['concept'].min()) == 0
    assert int(weights['concept'].max()) < C
    assert weights['idf'][0] == 0 and bool((weights['idf'] >= 0).all())
    for name in ('idf', 'A', 'row_w', 'mix'):
        assert weights[name].dtype == torch.float32
    assert bool(torch.isfinite(weights['A']).all())
    assert bool(torch.isfinite(weights['idf']).all())
    occupied_count = 0
    for start in range(0, H, 8192):
        ids = weights['row_ids'][start:start + 8192]
        values = weights['row_w'][start:start + 8192]
        assert int(ids.min()) >= 0 and int(ids.max()) < C
        assert bool(torch.isfinite(values).all())
        assert torch.equal(ids == 0, values == 0)
        occupied_count += int((ids != 0).any(1).count_nonzero())
    assert weights['A'][0].count_nonzero() == 0
    assert weights['A'][:, 0].count_nonzero() == 0
    # Ambiguous symbols have no unconditional outgoing sense assignment.
    for symbol in ('c', 'f', 'k', 'n'):
        symbol_id = int(weights['concept'][index[symbol]])
        assert weights['A'][symbol_id].count_nonzero() == 0, symbol
    for long_name, symbol in [('celsius', 'c'), ('fahrenheit', 'f'),
                               ('kelvin', 'k'), ('newton', 'n')]:
        a = int(weights['concept'][index[long_name]])
        b = int(weights['concept'][index[symbol]])
        assert weights['A'][a, b] > 0, (long_name, symbol)

    # Original synthetic concept bags, with no answer labels or accuracy scores.
    index = {word: i + 2 for i, word in enumerate(vocab)}
    def encode(text, length):
        tokens = [index.get(t, 1) for t in build.TOKEN.findall(text.lower())][:length]
        return tokens + [0] * (length - len(tokens))
    texts = ['circuit gap', 'kinetic faster', 'fossil preservation',
             'vacuum insulation', 'magnet material', 'plant turgid']
    options = ['current copper metal', 'thermal energy radiation',
               'water pressure root', 'mineral sediment burial']
    q = torch.tensor([encode(s, 128) for s in texts], dtype=torch.int64)
    o = torch.tensor([[encode(s, 48) for s in options] for _ in texts], dtype=torch.int64)
    with torch.no_grad():
        logits = arch.forward(weights, q, o)
        assert logits.shape == (6, 4) and bool(torch.isfinite(logits).all())
        permutation = [2, 0, 3, 1]
        assert torch.equal(arch.forward(weights, q, o[:, permutation]), logits[:, permutation])
        assert torch.equal(arch.forward(weights, q.flip(1), o.flip(2)), logits)
        repeated_q = q.clone()
        repeated_q[:, 30:32] = q[:, :2]
        repeated_o = o.clone()
        repeated_o[:, :, 30:33] = o[:, :, :3]
        assert torch.equal(arch.forward(weights, repeated_q, repeated_o), logits)
        unknown = arch.forward(weights, torch.ones_like(q), o)
        assert unknown.count_nonzero() == 0
        same = torch.tensor([[encode(s, 48)] * 4 for s in texts], dtype=torch.int64)
        assert arch.forward(weights, q, same).count_nonzero() == 0
        # Alias substitutions have exactly the same bags and logits.
        alternate_q = q.clone()
        alternate_q[1] = torch.tensor(encode('kinetic rapidly', 128))
        assert torch.equal(arch.forward(weights, alternate_q, o), logits)
        qb = q.repeat(11, 1)[:64]
        ob = o.repeat(11, 1, 1)[:64]
        start = time.perf_counter()
        for _ in range(64):
            result = arch.forward(weights, qb, ob)
            assert bool(torch.isfinite(result).all())
        elapsed_infer = time.perf_counter() - start

    elements = sum(t.numel() for t in weights.values())
    tensor_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    nonzero = sum(int(t.count_nonzero()) for t in weights.values())
    print('VERIFY OK: aliases, source signs, duplicate-rule invariance, tensors, masking,')
    print('option/token permutations, duplicate tokens, OOV ties, finite inference.')
    print(f'Rules={len(rules)} Compiled_variants={len(compiled_rules)} '
          f'Vocab={len(vocab)} C={C} H={H} R={R}')
    print(f'Elements={elements} Nonzero={nonzero} Occupied_rows={occupied_count}')
    print(f'Tensor_bytes={tensor_bytes} Build_seconds={elapsed_build:.3f}')
    print(f'4096_synthetic_forward_seconds={elapsed_infer:.3f}')
    print(f'Peak_RSS_KiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')


if __name__ == '__main__':
    main()

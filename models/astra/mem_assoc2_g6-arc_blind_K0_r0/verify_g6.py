"""Generation-6 source and compiler contracts; no questions, labels, or fitting."""
from pathlib import Path
import hashlib
import itertools
import torch
import build
from audit import audit_rules


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    assert hashlib.sha256((root / 'arch.py').read_bytes()).hexdigest() == (
        '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af')
    vocab, aliases, rules = build.lexicon()
    assert vocab == (root / 'vocab.txt').read_text().splitlines()
    assert not audit_rules(vocab, aliases, rules)
    known = set(vocab)

    def terms(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text.lower()) if t in known}

    for a, b in [
        ('crumpled', 'crumple'), ('wrung', 'wring'), ('scurried', 'scurry'),
        ('clung', 'cling'), ('malleability', 'malleable'), ('ductility', 'ductile'),
        ('haemoglobin', 'hemoglobin'), ('oesophagus', 'esophagus'),
        ('faeces', 'feces'), ('keratinised', 'keratin'),
        ('pasteurised', 'pasteurization'), ('sodiumhydroxide', 'naoh'),
        ('sodiumbicarbonate', 'nahco3'), ('sulfuricacid', 'h2so4'),
        ('nitricacid', 'hno3'), ('aceticacid', 'ch3cooh'), ('glucose', 'c6h12o6'),
        ('steeper', 'steep'), ('shallower', 'shallow'), ('deeper', 'deep'),
        ('cushioning', 'cushion'), ('exceeded', 'exceed'),
        ('avian', 'bird'), ('voltaic', 'galvanic'),
        ('electrolytic', 'electrolysis'), ('7', 'seven'),
        ('falsifiable', 'falsifiability'), ('falsified', 'falsify'),
    ]:
        assert a in known and b in known and terms(a) == terms(b), (a, b)
    for a, b in [
        ('endotherm', 'endothermic'), ('ectotherm', 'exothermic'),
        ('acclimation', 'adaptation'), ('homogenization', 'pasteurization'),
        ('cohesion', 'adhesion'), ('porosity', 'permeability'),
        ('phosphorescent', 'fluorescent'), ('photovoltaic', 'photoelectric'),
        ('sodiumchloride', 'sodiumhydroxide'), ('glucose', 'sugar'),
        ('sulfuricacid', 'sulfur'), ('nitricacid', 'nitrogen'),
        ('depth', 'deep'), ('force', 'duration'), ('steep', 'shallow'),
        ('condenser', 'evaporator'), ('reversible', 'permanent'),
        ('both', 'one'), ('both', 'all'), ('below', 'above'),
        ('galvanic', 'electrolysis'), ('electrolyte', 'electrolysis'),
        ('erythrocyte', 'erythroblast'), ('hypertonic', 'hypotonic'),
        ('sterilization', 'pasteurization'), ('sterilization', 'refrigeration'),
        ('falsifiable', 'falsified'), ('precision', 'resolution'),
        ('precision', 'accuracy'), ('random', 'systematic'),
        ('mechanical', 'electromagnetic'), ('electron', 'ion'),
        ('penumbra', 'penumbral'), ('umbra', 'antumbra'),
        ('northern', 'southern'),
    ]:
        assert a in known and b in known and terms(a) != terms(b), (a, b)

    # Authored compound names expand only at complete lexical boundaries.
    for phrase, compound in build.PHRASE_ALTERNATIVES:
        assert compound in known, compound
        assert compound in build.phrase_variants(phrase), phrase
        for suffix in ("'s", 'ish'):
            text = phrase + suffix
            assert compound + suffix not in build.phrase_variants(text), text
    for formula in ('naoh', 'nahco3', 'h2so4', 'hno3', 'ch3cooh', 'c6h12o6'):
        assert terms(formula) not in (terms('hydrogen'), terms('oxygen'), terms('carbon'))

    # All are explicit two-concept conditions. They must not silently turn into
    # unconditional assertions or invert a signed consequence during compilation.
    conditions = {
        'holding stationary': ('zero', 'work displacement'),
        'pushing immovable': ('zero', 'work displacement'),
        'opaque obstruction': ('shadow absorption', 'transmission'),
        'variegated white': ('unpigmented', 'chlorophyll photosynthesis'),
        'potato saltwater': ('water osmosis movement', ''),
        'potato freshwater': ('water osmosis swell', ''),
        'potato hypertonic': ('water osmosis shrink', ''),
        'potato hypotonic': ('water osmosis swell', ''),
        'chlorophyll removed': ('photosynthesis decrease', 'increase'),
        'vessels dilate': ('heat cooling', ''),
        'vessels constrict': ('heat conservation', ''),
        'resource partition': ('competition decrease coexistence', 'increase'),
        'predator removed': ('prey population increase', 'decrease'),
        'habitat isolated': ('gene flow decrease', 'increase'),
        'urban infiltration': ('decrease', 'increase'),
        'solution evaporating': ('concentration increase', 'decrease'),
        'stainless chromium': ('protective oxide', 'corrosion'),
        'smother flame': ('extinguished', 'oxygen combustion'),
        'soil compacted': ('pore space decrease', 'increase'),
        'snow dirty': ('absorption melting increase', 'decrease'),
        'submarine deeper': ('water pressure increase', 'decrease'),
        'submarine shallower': ('water pressure decrease', 'increase'),
        'moon airless': ('vacuum', 'weather'),
        'mutation somatic': ('body cell mitotic lineage', ''),
        'scar inherited': ('acquired injury', 'genetic'),
        'blinding observer': ('expectation bias decrease', 'increase'),
        'control active': ('treatment comparison', ''),
        'control placebo': ('expectation comparison', ''),
        'airbag time': ('stopping increase', 'decrease'),
        'airbag duration': ('stopping increase', 'decrease'),
        'airbag force': ('decrease', 'increase'),
        'padding time': ('stopping increase', 'decrease'),
        'padding duration': ('stopping increase', 'decrease'),
        'padding force': ('decrease', 'increase'),
        'cushion time': ('stopping increase', 'decrease'),
        'cushion duration': ('stopping increase', 'decrease'),
        'cushion force': ('decrease', 'increase'),
        'helmet time': ('stopping increase', 'decrease'),
        'helmet duration': ('stopping increase', 'decrease'),
        'helmet force': ('decrease', 'increase'),
        'steep force': ('increase', 'decrease'),
        'steep effort': ('increase', 'decrease'),
        'steep distance': ('shorter', 'longer'),
        'shallow force': ('decrease', 'increase'),
        'shallow effort': ('decrease', 'increase'),
        'shallow distance': ('longer', 'shorter'),
        'refrigerator condenser': ('heat released', 'absorbed'),
        'refrigerator evaporator': ('heat absorbed', 'released'),
        'condenser heat': ('released', 'absorbed'),
        'evaporator heat': ('absorbed', 'released'),
        'elastic exceeded': ('permanent deformation', 'reversible'),
        'rigid heat': ('conduction radiation', 'convection'),
        'vibration faster': ('higher pitch frequency', 'lower'),
        'vibration slower': ('lower pitch frequency', 'higher'),
        'star blue': ('hotter temperature', 'cooler'),
        'star red': ('cooler temperature', 'hotter'),
        'orbit closer': ('shorter period', 'longer'),
        'orbit farther': ('longer period', 'shorter'),
        'external gears': ('opposite rotation', ''),
        'internal gears': ('same rotation direction', ''),
        'below seven': ('acid', 'base neutral'),
        'above seven': ('base', 'acid neutral'),
        'equal seven': ('neutral', 'acid base'),
        'ph seven': ('neutral', ''),
        'mammal erythrocyte': ('hemoglobin oxygen', 'nucleus'),
        'avian erythrocyte': ('nucleus hemoglobin oxygen', ''),
        'reptile erythrocyte': ('nucleus hemoglobin oxygen', ''),
        'amphibian erythrocyte': ('nucleus hemoglobin oxygen', ''),
        'fish erythrocyte': ('nucleus hemoglobin oxygen', ''),
        'erythrocyte isotonic': ('balanced water movement stable', ''),
        'erythrocyte hypertonic': ('water loss shrink', ''),
        'erythrocyte hypotonic': ('water uptake swell', ''),
        'somatic transmission': ('mitotic daughter cells lineage', ''),
        'germline transmission': ('gamete offspring inheritance', ''),
        'vegetative mutation': ('somatic lineage offspring transmission', ''),
        'both benefit': ('mutualism', 'parasitism commensalism competition'),
        'both harmed': ('competition', 'mutualism'),
        'foam liquid': ('gas dispersed bubbles', ''),
        'foam solid': ('gas trapped pores matrix', ''),
        'galvanic anode': ('negative charge', 'positive'),
        'galvanic cathode': ('positive charge', 'negative'),
        'electrolysis anode': ('positive charge', 'negative'),
        'electrolysis cathode': ('negative charge', 'positive'),
        'pasteurization purpose': ('pathogen decrease food preservation', ''),
        'pasteurization spores': ('resistant survival', 'sterilization'),
        'sterilization spores': ('destruction microbial elimination', ''),
        'refrigeration microbial': ('slow metabolism growth', 'sterilization'),
        'freezing microbes': ('survival dormant', 'sterilization'),
        'canning pressure': ('elevated temperature spores destruction', ''),
        'control positive': ('known expected response', ''),
        'control negative': ('background baseline', 'response'),
        'readings scattered': ('random variation uncertainty', ''),
        'consistent wrong': ('precise inaccurate bias', ''),
        'reported digits': ('resolution significant figures', ''),
        'finer divisions': ('resolution smaller increments', ''),
        'averaging random': ('uncertainty decrease', 'increase'),
        'averaging systematic': ('persistent bias', 'eliminated'),
        'hypothesis falsified': ('contradictory evidence revision', ''),
        'evidence fabricated': ('dishonest unreliable', ''),
        'oil feathers': ('contamination plumage damage', 'insulation waterproofing'),
        'reforest watershed': ('infiltration roots soil stabilization', 'erosion'),
        'solid particles': ('fixed positions vibration', ''),
        'crystalline solid': ('ordered lattice vibration', ''),
        'amorphous solid': ('irregular arrangement vibration', 'lattice'),
        'crystals coarse': ('slow cooling growth', 'rapid'),
        'crystals fine': ('rapid cooling', 'slow'),
        'silica higher': ('viscosity increase', 'decrease'),
        'silica lower': ('viscosity decrease', 'increase'),
        'glacier retreat': ('terminus receding', ''),
        'glacier advance': ('terminus extension ice flow', ''),
        'glacier balance': ('mass budget accumulation ablation', ''),
        'glacier ablation': ('melting sublimation calving loss', ''),
        'glacier accumulation': ('snowfall deposition mass gain', ''),
        'metal current': ('electron drift', ''),
        'electrolyte current': ('ion transport', 'electron'),
        'conventional current': ('positive charge flow direction', ''),
        'electron drift': ('opposite conventional current direction', ''),
        'mechanical wave': ('material medium oscillation', ''),
        'electromagnetic wave': ('electric magnetic fields radiation', ''),
        'electromagnetic vacuum': ('propagation transmission', ''),
        'southern polaris': ('below horizon', 'visible'),
        'northern polaris': ('above horizon visible', ''),
        'eclipse partial': ('incomplete shadow coverage', ''),
        'penumbral eclipse': ('lunar faint dimming earth shadow', ''),
        'annular eclipse': ('antumbra sunlight ring', ''),
        'antumbra eclipse': ('annular sunlight ring', ''),
        'boundary flexible': ('membrane selective transport', ''),
        'boundary rigid': ('wall support protection', ''),
        'pores closed': ('water conservation', 'transpiration'),
    }
    # Check ALL sources, including overlapping inherited facts. A longer cue
    # must not project a selector-dependent direction onto an unqualified pair.
    for pair, forbidden in (
        ('star temperature', 'hot cold'),
        ('sound vibration', 'increase decrease'),
        ('planet sun', 'shorter longer faster slower'),
        ('gear driven', 'faster slower increase decrease smaller larger'),
        ('space telescope', 'distortion'),
        ('ph seven', 'acid base'),
        ('neutral atom', 'neutron'),
        ('foam bubbles', 'liquid'),
        ('oil feathers', 'insulation waterproofing'),
        ('reforest watershed', 'erosion'),
        ('solid particles', 'ordered lattice crystalline'),
        ('rock crystals', 'slow rapid intrusive extrusive'),
        ('cell boundary', 'wall'),
    ):
        matches = [r for r in rules if terms(pair) <= terms(r[0])]
        assert matches, pair
        assert all(not (terms(forbidden) & terms(p)) for _, p, _, _ in matches), pair
    for cue, forbidden in (
        ('submarine depth', 'increase decrease'),
        ('airbag', 'increase decrease longer shorter'),
        ('airbag collision', 'increase decrease longer shorter'),
        ('helmet cushion', 'increase decrease longer shorter smaller larger'),
        ('padding collision', 'increase decrease longer shorter'),
        ('landing bend', 'increase decrease longer shorter'),
        ('catching cushion', 'increase decrease longer shorter'),
        ('ramp length', 'longer shorter increase decrease more less'),
        ('ramp steep', 'longer shorter increase decrease more less smaller larger'),
        ('ramp shallow', 'longer shorter increase decrease more less smaller larger'),
        ('ramp longer', 'increase decrease more less smaller larger'),
        ('ramp shorter', 'increase decrease more less smaller larger'),
        ('refrigerator coils', 'absorb release'),
        ('planet distance period', 'farther longer shorter faster slower'),
        ('protostar heating', 'fusion'),
        ('elastic limit', 'permanent'),
        ('ph above', 'acid base neutral'),
        ('ph below', 'acid base neutral'),
        ('potato saltwater', 'shrink loss hypertonic'),
        ('rain garden', 'increase decrease reduction'),
        ('reported digits', 'precision accuracy'),
        ('magma silica', 'explosive increase decrease'),
        ('glacier retreat', 'melting exceeds accumulation'),
        ('glacier advance', 'accumulation exceeds melting'),
        ('electric current', 'electron'),
        ('wave disturbance', 'medium'),
        ('eclipse partial', 'penumbra umbra'),
    ):
        matches = [r for r in rules if terms(cue) == terms(r[0])]
        assert matches, cue
        assert all(not (terms(forbidden) & (terms(p) | terms(n)))
                   for _, p, n, _ in matches), cue
    # Guards cover overlapping inherited records, not just the new source file.
    for cue, _, negative, _ in rules:
        qs = terms(cue)
        if terms('leaf pores') <= qs:
            assert not (terms('transpiration') & terms(negative)), cue
        if qs == terms('somatic mutation'):
            assert not (terms('inherited offspring transmission') & terms(negative)), cue
        if terms('erythrocyte') <= qs and terms('nucleus') & terms(negative):
            assert terms('mammal') <= qs, cue
    for pair in ('species benefit', 'relationship benefit'):
        assert not any(terms(pair) <= terms(cue) and terms('mutualism') & terms(positive)
                       for cue, positive, _, _ in rules), pair
    for threshold in ('below seven', 'above seven', 'equal seven'):
        matches = [r for r in rules if terms(r[0]) == terms(threshold)]
        assert matches and all(r[3] == -2.0 for r in matches), threshold
    assert 'both' in known and terms('both'), 'Quantifier discarded as grammar'
    # Allow empty matches: these selector-free pairs need not have a rule at all.
    for pair, forbidden in (
        ('control expected', 'response'),
        ('instrument consistent', 'inaccurate bias wrong'),
        ('instrument readings', 'random uncertainty precision'),
        ('instrument divisions', 'smaller finer'),
    ):
        assert not any(terms(pair) <= terms(cue) and terms(forbidden) & terms(positive)
                       for cue, positive, _, _ in rules), pair
    for cue, positive, negative, _ in rules:
        if terms('falsifiable') <= terms(cue):
            assert not (terms('falsified disproof disprove dishonest unreliable') &
                        (terms(positive) | terms(negative))), cue
    # A hemisphere mention alone cannot select pole-star visibility. Check both
    # positive below-horizon evidence and explicit visibility inhibition.
    for cue, positive, negative, _ in rules:
        if terms('polaris hemisphere') <= terms(cue):
            assert not (terms('below') & terms(positive)), cue
            assert not (terms('visible') & terms(negative)), cue
    tails = [r for r in rules if terms(r[0]) == terms('comet tail direction')]
    assert tails and all(terms('radiation') <= terms(r[1]) for r in tails)
    # Test the newly authored records in isolation. Older definitions may retain
    # deliberately weaker first-order backoff and are covered by inherited suites.
    new_rules = []
    for path in sorted(root.glob('facts_g6*.txt')):
        for raw in path.read_text().splitlines():
            raw = raw.split('#', 1)[0].strip()
            if not raw:
                continue
            fields = [s.strip().lower() for s in raw.split('|')]
            new_rules.append((fields[0], fields[1], fields[2] if len(fields) >= 3 else '',
                              float(fields[3]) if len(fields) == 4 else 1.0))
    selected = []
    for cue, (positive, negative) in conditions.items():
        qs = terms(cue)
        assert len(qs) == 2, (cue, qs)
        matching = [r for r in new_rules if terms(r[0]) == qs]
        assert matching, cue
        assert all(r[3] < 0 for r in matching), cue
        assert any(terms(positive) <= terms(p) and terms(negative) <= terms(n)
                   for _, p, n, _ in matching), cue
        selected.extend(matching)
    selected = sorted(set(selected))
    assert not audit_rules(vocab, aliases, selected)
    old = build.lexicon
    try:
        build.lexicon = lambda: (vocab, aliases, selected)
        w = build.build()
    finally:
        build.lexicon = old
    assert w['A'].count_nonzero().item() == 0
    index = {term: i + 2 for i, term in enumerate(vocab)}

    def ids(text):
        return {int(w['concept'][index[t]]) for t in build.TOKEN.findall(text.lower()) if t in index}

    for cue, (positive, negative) in conditions.items():
        a, b = sorted(ids(cue))
        row = (a * 1000003 + b) % w['row_ids'].shape[0]
        recall = {int(i): float(v) for i, v in zip(w['row_ids'][row], w['row_w'][row]) if int(i)}
        for text, sign in ((positive, 1), (negative, -1)):
            for target in ids(text) - {a, b}:
                assert recall.get(target, 0) * sign > 0, (cue, text, target)
    assert all(float(w['idf'][i]) > 0 for i in ids('both'))
    # Test symbolic composition, not multiple-choice outputs or answer rankings.
    # A below/above threshold must inhibit the bare co-occurrence default even
    # when the question bag also contains pH and seven. No parameter search occurs.
    def recall(text):
        qs = ids(text)
        result = {}
        for a, b in itertools.combinations(sorted(qs), 2):
            row = (a * 1000003 + b) % w['row_ids'].shape[0]
            strength = float(w['idf'][a] * w['idf'][b])
            for target, value in zip(w['row_ids'][row], w['row_w'][row]):
                target = int(target)
                if target and target not in qs:
                    result[target] = result.get(target, 0.0) + strength * float(value)
        return result
    for cue, positive, negative in (
        ('ph below seven', 'acid', 'base neutral'),
        ('ph above seven', 'base', 'acid neutral'),
        ('ph equal seven', 'neutral', 'acid base'),
        ('ph seven', 'neutral', ''),
    ):
        actual = recall(cue)
        for text, sign in ((positive, 1), (negative, -1)):
            for target in ids(text):
                assert actual.get(target, 0) * sign > 0, (cue, text, target)
    counts = {}
    for p in sorted(root.glob('facts_g6*.txt')):
        counts[p.name] = sum(bool(s.split('#', 1)[0].strip()) for s in p.read_text().splitlines())
    assert len(counts) == 7 and min(counts.values()) >= 100
    print(f'G6 VERIFY OK: {len(conditions)} signed source/tensor contracts; '
          'lexical distinctions, compound boundaries, no unconditional backoff.')
    print(f'G6 authored additions={sum(counts.values())}; files={counts}')


if __name__ == '__main__':
    main()

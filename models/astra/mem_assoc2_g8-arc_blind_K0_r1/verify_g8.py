"""Generation-8 source regressions. No question predictions or accuracy scores."""
from pathlib import Path
import build
from audit import audit_rules


def main():
    vocab, aliases, rules = build.lexicon()
    known = set(vocab)

    def cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}

    selectors = cs(' '.join(sorted(build.COMPARATIVE_SELECTORS)))
    negators = cs('not except')
    compiled = build.compile_rules(vocab, aliases, rules)
    assert not audit_rules(vocab, aliases, compiled)

    # Each pair compiled from these specific assertions retains the condition.
    mandatory_cases = [
        ('seed dry storage', 'dry'),
        ('seed wet storage', 'wet'),
        ('plant magnesium deficiency', 'magnesium deficiency'),
        ('plant nitrogen deficiency', 'nitrogen deficiency'),
        ('ion electron loss', 'loss'),
        ('ion electron gain', 'gain'),
        ('sugar water conductivity', 'sugar'),
        ('salt water conductivity', 'salt'),
        ('rock intrusive cooling', 'intrusive'),
        ('rock extrusive cooling', 'extrusive'),
        ('moon day temperature', 'day'),
        ('moon night temperature', 'night'),
        ('star blue color', 'blue'),
        ('star red color', 'red'),
        ('freezing water celsius', 'freezing celsius'),
        ('boiling water celsius', 'boiling celsius'),
        ('freezing water fahrenheit', 'freezing fahrenheit'),
        ('boiling water fahrenheit', 'boiling fahrenheit'),
        ('motion graph flat distance', 'flat distance'),
        ('motion graph flat speed', 'flat speed'),
        ('light red wavelength', 'red wavelength'),
        ('light violet wavelength', 'violet wavelength'),
        ('light red frequency', 'red frequency'),
        ('light violet frequency', 'violet frequency'),
        ('whale fish distinction', 'whale'),
        ('bat bird distinction', 'bat'),
        ('reproductive isolation species', 'reproductive isolation'),
        ('annual plant cycle', 'annual'),
        ('biennial plant cycle', 'biennial'),
        ('finch thick beak', 'thick'),
        ('finch thin beak', 'thin'),
        ('mass volume quotient', 'mass volume'),
        ('distance time quotient', 'distance time'),
        ('force area quotient', 'force area'),
        ('work time quotient', 'work time'),
        ('mass acceleration product', 'mass acceleration'),
        ('force distance product', 'force distance'),
        ('voltage current product', 'voltage current'),
        ('current resistance product', 'current resistance'),
        ('frequency wavelength product', 'frequency wavelength'),
        ('distance time slope', 'distance slope'),
        ('speed time slope', 'speed slope'),
        ('speed time area', 'speed area'),
        ('heat sealed container', 'heat sealed'),
        ('moon atmosphere thin', 'moon thin'),
        ('summer solstice daylight', 'summer solstice'),
        ('winter solstice daylight', 'winter solstice'),
        ('experiment independent variable', 'independent'),
        ('experiment dependent variable', 'dependent'),
        ('experiment controlled variable', 'controlled'),
        ('water cobalt chloride', 'water cobalt'),
        ('water copper sulfate', 'water copper'),
        ('stainless steel property', 'stainless steel'),
        ('chromium plating property', 'chromium plating'),
        ('rock glassy texture', 'glassy'),
        ('soil frozen rain', 'frozen'),
        ('soil compacted', 'compacted'),
        ('urban pavement runoff', 'pavement'),
        ('dam fish migration', 'dam migration'),
        ('black hole gravity', 'black hole'),
    ]
    # Inspect only new records: old broader records have their own inherited tests.
    authored = []
    for path in sorted(Path(build.ROOT).glob('facts_g8_*.txt')):
        for line in path.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                fields = [part.strip() for part in line.split('|')]
                authored.append((fields[0], fields[1], fields[2] if len(fields) > 2 else '',
                                 float(fields[3]) if len(fields) > 3 else 1.0))
    for cue, required in mandatory_cases:
        matched = [r for r in authored if cs(r[0]) == cs(cue)]
        assert matched, cue
        for q, p, n, scale in matched:
            for variant, _, _, _ in build.compile_rules(vocab, aliases, [(q, p, n, scale)]):
                qs = sorted(cs(variant))
                anchors = cs(' '.join(build.ANCHOR.findall(variant)))
                pairs = build.recall_pairs(qs, negators, build.condition_guard(qs, selectors), anchors)
                assert pairs and all(cs(required).issubset(pair) for pair in pairs), (cue, pairs)

    signed_cases = [
        ('acid litmus', 'red', 'blue'),
        ('base litmus', 'blue', 'red'),
        ('static friction', 'traction', 'sliding'),
        ('water cobalt chloride', 'pink', 'blue'),
        ('water copper sulfate', 'blue', 'white'),
        ('space telescope', 'resolution', 'distortion absorption'),
        ('contour plowing', 'conservation', 'erosion runoff'),
        ('terrace farming', 'conservation', 'erosion runoff'),
        ('sterilization equipment', 'clean', 'microorganisms contamination'),
        ('sound space vacuum', 'silence', 'propagation transmission'),
        ('wave energy transfer', 'oscillation', 'bulk matter'),
        ('seedling pale dark', 'etiolation', 'chlorophyll'),
        ('platelets wound', 'clotting', 'bleeding'),
        ('phloem girdling', 'starvation', 'translocation transport'),
        ('scientific objectivity', 'evidence', 'preference bias'),
        ('scatterplot no pattern', 'unrelated', 'correlation'),
        ('reproductive isolation species', 'speciation', 'interbreeding'),
        ('light red wavelength', 'long', 'short'),
        ('light violet wavelength', 'short', 'long'),
        ('light red frequency', 'low', 'high'),
        ('light violet frequency', 'high', 'low'),
        ('stainless steel property', 'chromium oxide', 'corrosion'),
        ('chromium plating property', 'coating', 'corrosion'),
        ('oil spill feathers', 'hypothermia', 'insulation waterproofing'),
        ('rock glassy texture', 'amorphous', 'crystals'),
        ('soil compacted', 'runoff', 'infiltration'),
        ('soil frozen rain', 'runoff', 'infiltration'),
        ('pavement rain', 'runoff', 'infiltration'),
        ('urban pavement runoff', 'flooding', 'infiltration'),
        ('black hole gravity', 'trapped light', 'escape'),
    ]
    for cue, positive, negative in signed_cases:
        matched = [r for r in authored if cs(r[0]) == cs(cue)]
        assert matched, cue
        assert cs(positive).issubset(set().union(*(cs(r[1]) for r in matched))), cue
        assert cs(negative).issubset(set().union(*(cs(r[2]) for r in matched))), cue
        assert all(not (cs(r[1]) & cs(negative)) for r in matched), cue

    # Full cue-bag checks alone miss contradictions introduced by longer cues.
    # Inspect the actual projected pairs across ALL source files for these reviewed
    # mechanisms. This checks authored signs, not logits or measured accuracy.
    projected_cases = [
        ('sound vacuum', 'propagation transmission'),
        ('wave transfer', 'matter'),
        ('seedling dark', 'chlorophyll'),
        ('reproductive isolation', 'interbreeding'),
        ('scientific objectivity', 'bias'),
        ('platelet wound', 'bleeding'),
        ('phloem girdling', 'translocation transport'),
        ('black hole', 'escape'),
        ('water cobalt', 'blue'),
        ('oil feathers', 'waterproofing'),
        ('soil frozen', 'infiltration'),
        ('soil compacted', 'infiltration'),
        ('pavement runoff', 'infiltration'),
        ('stainless steel', 'corrosion'),
        ('rock glassy', 'crystals'),
    ]
    projected = {tuple(sorted(cs(cue))): (cue, cs(inhibited), set())
                 for cue, inhibited in projected_cases}
    for q, p, n, scale in compiled:
        qs = cs(q)
        candidate_pairs = [pair for pair in projected if set(pair).issubset(qs)]
        if not candidate_pairs:
            continue
        anchors = cs(' '.join(build.ANCHOR.findall(q)))
        emitted = build.recall_pairs(sorted(qs), negators,
                                     build.condition_guard(qs, selectors), anchors)
        for pair in candidate_pairs:
            if pair in emitted:
                cue, inhibited, observed = projected[pair]
                assert not (cs(p) - qs) & inhibited, (cue, q, p)
                observed.update((cs(n) - qs) & inhibited)
    for cue, inhibited, observed in projected.values():
        assert inhibited.issubset(observed), (cue, inhibited - observed)

    # Reagent colors depend on the particular metal as well as its hydration
    # state. Paving must not inhibit generic groundwater infiltration/recharge.
    forbidden_pairs = {
        tuple(sorted(cs('water chloride'))): cs('pink blue'),
        tuple(sorted(cs('dry chloride'))): cs('pink blue'),
        tuple(sorted(cs('water sulfate'))): cs('white blue'),
    }
    groundwater_pair = tuple(sorted(cs('ground water')))
    groundwater_positive = set()
    for q, p, n, scale in compiled:
        qs = cs(q)
        anchors = cs(' '.join(build.ANCHOR.findall(q)))
        emitted = build.recall_pairs(sorted(qs), negators,
                                     build.condition_guard(qs, selectors), anchors)
        for pair, targets in forbidden_pairs.items():
            assert pair not in emitted or not ((cs(p) | cs(n)) & targets), (pair, q)
        if groundwater_pair in emitted:
            assert not cs(n) & cs('infiltration recharge'), q
            groundwater_positive.update(cs(p) - qs)
        if cs('paved groundwater').issubset(qs) and cs(n) & cs('infiltration recharge'):
            assert emitted and all(cs('paved').issubset(pair) for pair in emitted), q
            assert scale < 0 or anchors, q
    assert cs('infiltration').issubset(groundwater_positive)

    # Explicit inflections merge; scientifically different concepts remain distinct.
    for left, right in [('moulted', 'molting'), ('hatchlings', 'hatch'),
                        ('oesophagus', 'esophagus'), ('pseudopodia', 'pseudopod'),
                        ('catalysing', 'catalyze'), ('centigrade', 'celsius')]:
        assert cs(left) and cs(left) == cs(right), (left, right)
    for left, right in [('permeability', 'porosity'), ('celsius', 'fahrenheit'),
                        ('anther', 'stigma'), ('ovule', 'ovary'),
                        ('adhesion', 'cohesion'), ('precision', 'accuracy')]:
        assert cs(left) and cs(right) and cs(left).isdisjoint(cs(right)), (left, right)
    print(f'G8 SOURCE CHECK OK: {len(authored)} additions; '
          f'{len(mandatory_cases)} condition cases; {len(signed_cases)} sign cases; '
          f'{len(projected_cases)} all-source pair cases.')
    print('No questions, labels, accuracy measurements, or model selection used.')


if __name__ == '__main__':
    main()

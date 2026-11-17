"""Role/topic binding and thermos source regressions; no question predictions."""
from collections import defaultdict
import build


def main():
    vocab, aliases, source = build.lexicon()
    known = set(vocab)

    def cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}

    def required(text):
        assert set(build.TOKEN.findall(text)).issubset(known), text
        return cs(text)

    negators = required('not except')
    selectors = required(' '.join(sorted(build.COMPARATIVE_SELECTORS)))

    def pairs(q):
        cues = cs(q)
        return build.recall_pairs(sorted(cues), negators,
                                 build.condition_guard(cues, selectors),
                                 cs(' '.join(build.ANCHOR.findall(q))))

    # Each asserted topic/role has representative supported concepts. These
    # examples describe conventional experiments, not all possible protocols.
    domain_cases = [
        ('fertilizer', 'dependent', 'growth height biomass'),
        ('fertilizer', 'independent', 'concentration'),
        ('friction', 'dependent', 'distance speed force'),
        ('friction', 'independent', 'surface texture material'),
        ('friction', 'constant', 'object mass slope'),
        ('insulation', 'dependent', 'temperature cooling rate'),
        ('insulation', 'independent', 'material thickness'),
        ('insulation', 'constant', 'water volume temperature container'),
        ('solute', 'dependent', 'time dissolving rate'),
        ('solute', 'independent', 'temperature stirring particle size'),
        ('solute', 'constant', 'amounts solvent'),
        ('pendulum', 'dependent', 'period swing time'),
        ('pendulum', 'independent', 'string length'),
        ('pendulum', 'constant', 'mass release angle'),
        ('magnet', 'dependent', 'paperclips attraction distance'),
        ('magnet', 'independent', 'strength coil current'),
        ('magnet', 'constant', 'paperclips distance'),
        ('evaporation', 'dependent', 'water volume time'),
        ('evaporation', 'independent', 'temperature surface area airflow'),
        ('absorbency', 'dependent', 'water volume'),
        ('absorbency', 'constant', 'sample size water time'),
    ]
    checked = 0
    for topic, role, supported in domain_cases:
        cue = required(f'{topic} experiment {role}')
        mandatory = required(f'{topic} {role}')
        assert len(mandatory) == 2
        matched = [rule for rule in source if cs(rule[0]) == cue]
        assert matched, (topic, role)
        for rule in matched:
            assert required(supported).issubset(cs(rule[1]) - cs(rule[0])), rule
            for q, p, n, scale in build.compile_rules(vocab, aliases, [rule]):
                emitted = pairs(q)
                assert emitted and all(set(pair) == mandatory for pair in emitted), (q, emitted)
                assert cs(' '.join(build.ANCHOR.findall(q))) == mandatory, q
                checked += 1

    definitions = [
        ('experiment independent purpose', 'experiment independent'),
        ('experiment dependent purpose', 'experiment dependent'),
        ('experiment constants purpose', 'experiment constants'),
        ('controlled variable meaning', 'controlled variable'),
        ('manipulated variable meaning', 'manipulated variable'),
        ('responding variable meaning', 'responding variable'),
        ('independent variable meaning', 'independent variable'),
        ('dependent variable meaning', 'dependent variable'),
        ('constant variable meaning', 'constant variable'),
        ('initial independent variable', 'independent variable'),
        ('final dependent measurement', 'dependent measurement'),
    ]
    for cue, binding in definitions:
        matched = [rule for rule in source if cs(rule[0]) == required(cue)]
        assert matched, cue
        for rule in matched:
            for q, p, n, scale in build.compile_rules(vocab, aliases, [rule]):
                emitted = pairs(q)
                assert emitted and all(required(binding) == set(pair) for pair in emitted), (q, emitted)

    # Positive associations for genuine remaining heat-transfer mechanisms are
    # retained. A vacuum suppresses conduction/convection, not radiation.
    positive, negative = defaultdict(set), defaultdict(set)
    for q, p, n, scale in build.compile_rules(vocab, aliases, source):
        for pair in pairs(q):
            positive[pair].update(cs(p) - cs(q))
            negative[pair].update(cs(n) - cs(q))
    key = tuple(sorted(required('thermos lid')))
    assert required('convection evaporation').issubset(negative[key])
    assert not required('convection evaporation').intersection(positive[key])
    assert required('insulation retention').issubset(positive[key])
    key = tuple(sorted(required('thermos vacuum')))
    assert required('radiation').issubset(positive[key])
    assert required('conduction convection').issubset(negative[key])
    for cue in ('thermos shiny', 'thermos silver'):
        key = tuple(sorted(required(cue)))
        assert required('radiation reflection').issubset(positive[key]), cue

    print(f'ROLES OK: {len(domain_cases)} topic/role cases, {checked} compiled domain variants,')
    print(f'  {len(definitions)} definition bindings, four all-source thermos mechanism cases.')
    print('Source assertions only; no questions, labels, or accuracy measurements.')


if __name__ == '__main__':
    main()

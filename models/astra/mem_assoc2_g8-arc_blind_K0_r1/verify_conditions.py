"""Source-only regressions for inherited conditional science descriptions."""
import build


def main():
    vocab, aliases, source = build.lexicon()
    known = set(vocab)

    def cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}

    negators = cs('not except')
    selectors = cs(' '.join(sorted(build.COMPARATIVE_SELECTORS)))

    def pairs(q):
        cues = cs(q)
        anchors = cs(' '.join(build.ANCHOR.findall(q)))
        return build.recall_pairs(sorted(cues), negators,
                                 build.condition_guard(cues, selectors), anchors)

    cases = [
        ('rolling ball carpet', 'carpet'),
        ('rolling ball smooth', 'smooth'),
        ('ball upward speed', 'upward'),
        ('ball downward speed', 'downward'),
        ('bulb metal bridge', 'metal'),
        ('bulb plastic bridge', 'plastic'),
        ('circuit bridge metal', 'metal'),
        ('circuit bridge plastic', 'plastic'),
        ('short string sound', 'short'),
        ('long string sound', 'long'),
        ('solution add water', 'add water'),
        ('solution add solute', 'add solute'),
        ('air heated density', 'heated'),
        ('air hot density', 'hot'),
        ('air cold density', 'cold'),
        ('warm air density', 'warm'),
        ('food chain long', 'long'),
        ('food chain shorten', 'shorten'),
        ('foodchain shorter', 'shorter'),
        ('cold metal touch', 'metal'),
        ('cold wood touch', 'wood'),
        ('cold tile touch', 'tile'),
        ('cold metal wood', 'cold metal'),
        ('iron wool burning', 'wool'),
        ('iron block burning', 'block'),
        ('water table drought', 'drought'),
        ('water table rainfall', 'rainfall'),
        ('water table wet season', 'wet'),
        ('river bend outside', 'outside'),
        ('river bend inside', 'inside'),
        ('river bend outer', 'outer'),
        ('river bend inner', 'inner'),
        ('star color blue', 'star blue'),
        ('star color red', 'star red'),
        ('boiling vacuum', 'vacuum'),
        ('boiling pressure cooker', 'pressure cooker'),
    ]
    checked = 0
    for cue, condition in cases:
        matched = [r for r in source if cs(r[0]) == cs(cue)]
        assert matched, cue
        for rule in matched:
            for q, p, n, scale in build.compile_rules(vocab, aliases, [rule]):
                emitted = pairs(q)
                assert emitted and all(cs(condition).issubset(pair) for pair in emitted), (cue, q, emitted)
                checked += 1

    # Spatial upward motion must not be encoded as increasing density.
    hot = [r for r in source if cs(r[0]) == cs('air hot density')]
    assert hot
    assert any(cs('upward').issubset(cs(r[1])) for r in hot)
    assert all(not cs('increase').intersection(cs(r[1])) for r in hot)
    drought = [r for r in source if cs(r[0]) == cs('water table drought')]
    assert drought and all(not cs('recharge').intersection(cs(r[1])) for r in drought)
    assert not any(cs(q) == cs('boiling point elevation') and
                   cs('increase').intersection(cs(p)) for q, p, n, scale in source)

    # Inspect every compiled source, not only the corrected full cue bags.
    forbidden = [
        ('string sound', 'high low'),
        ('solution add', 'increase decrease'),
        ('air density', 'increase decrease'),
        ('ball speed', 'increase decrease'),
        ('bulb bridge', 'open closed'),
        ('circuit bridge', 'open closed'),
        ('river bend', 'fast slow'),
        ('cold touch', 'fast slow'),
        ('iron burning', 'fast slow'),
        ('star color', 'high low hot cold'),
    ]
    forbidden = [(tuple(sorted(cs(cue))), cs(targets)) for cue, targets in forbidden]
    for q, p, n, scale in build.compile_rules(vocab, aliases, source):
        emitted = set(pairs(q))
        for pair, targets in forbidden:
            assert not (pair in emitted and targets.intersection(cs(p) | cs(n))), (q, p, n, pair)
    print(f'CONDITIONS OK: {len(cases)} cue cases, {checked} compiled variants,')
    print(f'  {len(forbidden)} all-source projection exclusions, spatial/density and altitude safeguards.')
    print('Source semantics only; no question predictions or accuracy scores.')


if __name__ == '__main__':
    main()

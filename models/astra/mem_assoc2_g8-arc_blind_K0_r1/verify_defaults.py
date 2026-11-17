"""Verify symbolic opposition precedence, without questions or accuracy tests."""
from collections import defaultdict
from copy import deepcopy
import build


def main():
    # A default is weaker in provenance, regardless of its numerical magnitude.
    authored = {
        (1, 2): {3: (0.01, 0.0), 4: (0.0, -0.2), 5: (0.4, -0.8)},
        (1, 9): {3: (0.5, 0.0)},
        1: {7: (0.4, 0.0)},
    }
    defaults = {
        (1, 2): {3: (0.0, -10.0), 4: (0.0, -0.7), 5: (0.0, -10.0), 6: (0.0, -0.3)},
        (1, 8): {3: (0.0, -0.6)},
        (1, 7): {7: (0.0, -0.6)},
        9: {3: (0.0, -0.4)},
    }
    original_authored, original_defaults = deepcopy(authored), deepcopy(defaults)
    build.merge_opposition_defaults(authored, defaults)
    assert authored[(1, 2)][3] == (0.01, 0.0)       # explicit positive survives
    assert authored[(1, 2)][4] == (0.0, -0.7)       # negative-only max rule retained
    assert authored[(1, 2)][5] == (0.4, -0.8)       # explicit contradiction NOT erased
    assert authored[(1, 2)][6] == (0.0, -0.3)       # default still fills an empty slot
    assert authored[(1, 8)][3] == (0.0, -0.6)       # no immunity from another pair
    assert authored[(1, 7)][7] == (0.0, -0.6)       # singleton cannot block a pair
    assert authored[9][3] == (0.0, -0.4)            # pair cannot block a singleton
    assert defaults == original_defaults          # input defaults not mutated
    again = deepcopy(authored)
    build.merge_opposition_defaults(again, defaults)
    assert again == authored                      # idempotent
    reversed_authored = dict(reversed(list(original_authored.items())))
    reversed_defaults = {key: dict(reversed(list(targets.items())))
                         for key, targets in reversed(list(original_defaults.items()))}
    build.merge_opposition_defaults(reversed_authored, reversed_defaults)
    assert reversed_authored == authored          # map order cannot affect values
    try:
        build.merge_opposition_defaults({}, {1: {2: (0.1, 0.0)}})
    except ValueError:
        pass
    else:
        raise AssertionError('Positive default evidence was not rejected')

    groups = [{'increase', 'decrease'}, {'absorb', 'release'}]
    assert build.inferred_opposites({'increase'}, set(), groups) == {'decrease'}
    assert not build.inferred_opposites({'increase', 'decrease'}, set(), groups)
    assert not build.inferred_opposites({'increase'}, {'decrease'}, groups)
    assert build.inferred_opposites({'absorb'}, set(), groups) == {'release'}

    # Inspect real compiled source evidence, not outputs on science questions.
    vocab, aliases, source = build.lexicon()
    known = set(vocab)
    def cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}
    groups = [cs(line) for line in build.OPPOSITION_TEXT.splitlines()]
    negators = cs('not except')
    selectors = cs(' '.join(sorted(build.COMPARATIVE_SELECTORS)))
    explicit = defaultdict(dict)
    inferred = defaultdict(dict)
    def add(table, key, target, value):
        pos, neg = table[key].get(target, (0.0, 0.0))
        table[key][target] = (max(pos, value), min(neg, value))

    for q, p, n, scale in build.compile_rules(vocab, aliases, source):
        qs, ps, ns = cs(q), cs(p), cs(n)
        guards = build.condition_guard(qs, selectors)
        anchors = cs(' '.join(build.ANCHOR.findall(q)))
        pairs = build.recall_pairs(sorted(qs), negators, guards, anchors)
        allow_singleton = scale > 0 and not qs & negators and not guards and not anchors
        default = build.inferred_opposites(ps, ns, groups)
        for targets, sign, table in [(ps, 1.0, explicit), (ns, -1.0, explicit),
                                     (default, -0.55, inferred)]:
            targets = targets - qs
            if not targets:
                continue
            base = sign * abs(scale) / len(targets) ** 0.35
            for target in targets:
                for pair in pairs:
                    add(table, pair, target, base / max(1.0, (len(qs) - 1) ** 0.7))
                if allow_singleton:
                    for singleton in qs:
                        add(table, singleton, target, base / max(1, len(qs)) ** 1.5)

    protected = [(key, target, explicit[key][target])
                 for key, targets in inferred.items() for target in targets
                 if explicit.get(key, {}).get(target, (0.0, 0.0))[0] > 0.0]
    cases = [('water cycle', 'condensation'),
             ('black surface', 'release'),
             ('cell division', 'meiosis'),
             ('short circuit', 'high')]
    for cue, target_text in cases:
        key = tuple(sorted(cs(cue)))
        targets = cs(target_text)
        assert len(key) == 2 and len(targets) == 1, (cue, targets)
        target = next(iter(targets))
        assert explicit[key][target][0] > 0, (cue, target_text)
        assert inferred[key][target][1] < 0, (cue, target_text)
    build.merge_opposition_defaults(explicit, inferred)
    assert all(explicit[key][target] == old for key, target, old in protected)
    pairs = sum(isinstance(key, tuple) for key, target, old in protected)
    print('DEFAULT PRECEDENCE OK: provenance priority, retained explicit negatives,')
    print('  unchanged unopposed defaults, semantic-key isolation, idempotence, order invariance.')
    print(f'Protected {pairs} pair/target slots and {len(protected) - pairs} singleton/target slots.')
    print('Four real-source examples checked; no question predictions or accuracy scores.')


if __name__ == '__main__':
    main()

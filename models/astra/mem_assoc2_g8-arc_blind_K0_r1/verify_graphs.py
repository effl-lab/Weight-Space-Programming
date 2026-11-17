"""Verify authored graph bindings before hashing, without question scoring."""
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

    selectors = required(' '.join(sorted(build.COMPARATIVE_SELECTORS)))
    negators = required('not except')

    def pairs(q):
        cues = cs(q)
        return build.recall_pairs(sorted(cues), negators,
                                 build.condition_guard(cues, selectors),
                                 cs(' '.join(build.ANCHOR.findall(q))))

    cases = [
        ('experimental graph horizontal axis', 'horizontal axis'),
        ('experimental graph vertical axis', 'vertical axis'),
        ('graph horizontal axis', 'horizontal axis'),
        ('graph vertical axis', 'vertical axis'),
        ('graph horizontal line', 'horizontal line'),
        ('graph horizontal trend', 'horizontal trend'),
        ('distance time flat', 'distance flat'),
        ('distance time straight', 'distance straight'),
        ('distance time steep', 'distance steep'),
        ('distance time curved', 'distance curved'),
        ('distance time horizontal', 'distance horizontal'),
        ('speed time increasing', 'speed increasing'),
        ('speed time decreasing', 'speed decreasing'),
        ('speed time flat', 'speed flat'),
        ('speed time area', 'speed area'),
        ('velocity time slope', 'velocity slope'),
        ('velocity time horizontal', 'velocity horizontal'),
        ('velocity time area', 'velocity area'),
    ]
    variants = 0
    for cue, mandatory in cases:
        matched = [rule for rule in source if cs(rule[0]) == required(cue)]
        assert matched, cue
        for rule in matched:
            for q, p, n, scale in build.compile_rules(vocab, aliases, [rule]):
                emitted = pairs(q)
                assert emitted and all(set(pair) == required(mandatory) for pair in emitted), (q, emitted)
                variants += 1

    positive, negative = defaultdict(set), defaultdict(set)
    for q, p, n, scale in build.compile_rules(vocab, aliases, source):
        for pair in pairs(q):
            positive[pair].update(cs(p) - cs(q))
            negative[pair].update(cs(n) - cs(q))

    # Missing the axis/line distinction or the plotted quantity must not supply
    # these specialized conclusions through an unqualified pair.
    excluded = [
        ('graph horizontal', 'independent constant unchanged'),
        ('graph vertical', 'dependent'),
        ('graph line', 'constant unchanged'),
        ('time flat', 'stopped stationary constant'),
        ('time horizontal', 'stopped rest constant'),
        ('time steep', 'speed'),
        ('time curved', 'acceleration'),
        ('time slope', 'speed acceleration'),
        ('time area', 'distance displacement'),
    ]
    for cue, targets in excluded:
        key = tuple(sorted(required(cue)))
        assert len(key) == 2
        assert not required(targets).intersection(positive[key] | negative[key]), (cue, targets, positive[key])

    # Ordinary school plotting conventions are retained. Motion-graph slope
    # and area interpretations here assume elapsed time on the horizontal axis.
    retained = [
        ('horizontal axis', 'independent'),
        ('vertical axis', 'dependent'),
        ('horizontal line', 'unchanged'),
        ('horizontal trend', 'constant'),
        ('distance flat', 'stationary'),
        ('speed flat', 'constant'),
        ('distance horizontal', 'rest'),
        ('velocity horizontal', 'constant'),
        ('distance steep', 'speed'),
        ('distance curved', 'acceleration'),
        ('speed increasing', 'acceleration'),
        ('speed decreasing', 'deceleration'),
        ('distance slope', 'speed'),
        ('velocity slope', 'acceleration'),
        ('speed slope', 'acceleration'),
        ('speed area', 'distance'),
        ('velocity area', 'displacement'),
    ]
    for cue, targets in retained:
        key = tuple(sorted(required(cue)))
        assert len(key) == 2 and required(targets).issubset(positive[key]), (cue, targets, positive[key])

    print(f'GRAPHS OK: {len(cases)} binding cases, {variants} compiled variants,')
    print(f'  {len(excluded)} all-source projection exclusions, {len(retained)} retained mechanism cases.')
    print('No question predictions, labels, or accuracy measurements used.')


if __name__ == '__main__':
    main()

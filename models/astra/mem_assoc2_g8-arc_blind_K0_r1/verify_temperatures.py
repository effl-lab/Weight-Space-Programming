"""Check phase/scale bindings in authored water reference-point knowledge."""
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

    # Pure water at ordinary atmospheric pressure; Kelvin values are rounded.
    scales = [('celsius', 'zero', 'hundred'), ('c', 'zero', 'hundred'),
              ('fahrenheit', 'thirtytwo', 'twohundredtwelve'),
              ('f', 'thirtytwo', 'twohundredtwelve'),
              ('kelvin', '273', '373'), ('k', '273', '373')]
    all_values = required('zero hundred thirtytwo twohundredtwelve 273 373')
    assert len(all_values) == 6
    for word, numeral in [('zero', '0'), ('hundred', '100'),
                          ('thirtytwo', '32'), ('twohundredtwelve', '212')]:
        assert required(word) == required(numeral), (word, numeral)
    named = [required(scale) for scale in ('celsius', 'fahrenheit', 'kelvin')]
    assert all(not a.intersection(b) for i, a in enumerate(named) for b in named[i + 1:])
    # Preserve multi-use chemical symbols instead of globally turning them into
    # temperature-unit aliases. Their authored phase/scale pair supplies context.
    for symbol, name in [('c', 'celsius'), ('f', 'fahrenheit'), ('k', 'kelvin')]:
        assert required(symbol) != required(name)

    references = []
    variants = 0
    for scale, freezing, boiling in scales:
        for phase, expected in [('freezing', freezing), ('boiling', boiling)]:
            mandatory = required(f'{phase} {scale}')
            assert len(mandatory) == 2
            matched = [r for r in source if mandatory.issubset(cs(r[0]))
                       and cs(r[1]).intersection(all_values)]
            assert matched, (phase, scale)
            for rule in matched:
                assert cs(rule[1]).intersection(all_values) == required(expected), rule
                for q, p, n, strength in build.compile_rules(vocab, aliases, [rule]):
                    anchors = cs(' '.join(build.ANCHOR.findall(q)))
                    emitted = pairs(q)
                    assert mandatory.issubset(anchors), q
                    assert emitted and all(set(pair) == mandatory for pair in emitted), (q, emitted)
                    variants += 1
            references.append((tuple(sorted(mandatory)), required(expected)))

    positive, negative = defaultdict(set), defaultdict(set)
    for q, p, n, strength in build.compile_rules(vocab, aliases, source):
        for pair in pairs(q):
            positive[pair].update(cs(p) - cs(q))
            negative[pair].update(cs(n) - cs(q))
    for pair, expected in references:
        assert positive[pair].intersection(all_values) == expected, (pair, positive[pair])
        assert not expected.intersection(negative[pair]), (pair, negative[pair])

    unqualified = ['water freezing', 'water boiling', 'water purity']
    unqualified += [f'water {scale}' for scale, _, _ in scales]
    for cue in unqualified:
        key = tuple(sorted(required(cue)))
        assert len(key) == 2
        assert not all_values.intersection(positive[key] | negative[key]), (cue, positive[key])

    retained = [('water freezing', 'ice expansion'),
                ('water boiling', 'vapor bubbles'),
                ('ice density', 'lower'),
                ('altitude boiling', 'decrease'),
                ('pressure cooker', 'higher temperature')]
    for cue, supported in retained:
        key = tuple(sorted(required(cue)))
        assert required(supported).issubset(positive[key]), (cue, positive[key])

    print(f'TEMPERATURES OK: {len(references)} phase/scale cases, {variants} compiled reference variants,')
    print(f'  {len(unqualified)} all-source unqualified-pair exclusions, {len(retained)} retained mechanisms.')
    print('Numeric aliases and distinct scales verified; no questions or accuracy scores.')


if __name__ == '__main__':
    main()

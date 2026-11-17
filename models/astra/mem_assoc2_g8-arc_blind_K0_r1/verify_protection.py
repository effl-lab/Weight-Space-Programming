"""Check protective versus hazardous source evidence, without question scoring."""
from collections import defaultdict
import build


def main():
    vocab, aliases, source = build.lexicon()
    known = set(vocab)

    def cs(text):
        words = build.TOKEN.findall(text)
        missing = set(words) - known
        assert not missing, (text, missing)
        return {aliases.get(word, word) for word in words}

    # Compiler source may contain filtered grammar, so use its actual known set.
    def source_cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}

    selectors = cs(' '.join(sorted(build.COMPARATIVE_SELECTORS)))
    negators = cs('not except')

    def pairs(q):
        cues = source_cs(q)
        anchors = source_cs(' '.join(build.ANCHOR.findall(q)))
        return build.recall_pairs(sorted(cues), negators,
                                 build.condition_guard(cues, selectors), anchors)

    conditional = [
        ('insulation electric purpose', 'insulation electric'),
        ('insulated electrical handle', 'insulated electrical'),
        ('plastic wire coating', 'plastic wire'),
        ('rubber gloves electricity', 'rubber gloves'),
        ('wire coating reason', 'wire coating'),
        ('frayed electrical insulation', 'frayed electrical'),
    ]
    for cue, required in conditional:
        matched = [rule for rule in source if source_cs(rule[0]) == cs(cue)]
        assert matched, cue
        for rule in matched:
            assert rule[3] < 0 or build.ANCHOR.findall(rule[0]), rule  # no unqualified singleton evidence
            for q, p, n, scale in build.compile_rules(vocab, aliases, [rule]):
                emitted = pairs(q)
                assert emitted and all(cs(required).issubset(pair) for pair in emitted), (cue, q)

    positive, negative = defaultdict(set), defaultdict(set)
    for q, p, n, scale in build.compile_rules(vocab, aliases, source):
        cues = source_cs(q)
        for pair in pairs(q):
            positive[pair].update(source_cs(p) - cues)
            negative[pair].update(source_cs(n) - cues)

    inhibited = [
        ('insulation electric', 'shock'),
        ('wire rubber', 'shock'),
        ('wire plastic', 'shock'),
        ('ground wire', 'shock'),
        ('earth wire', 'shock'),
        ('insulated handle', 'shock'),
        ('rubber gloves', 'shock'),
        ('wire coating', 'shock'),
        ('culture sealed', 'contamination escape'),
        ('virus outside', 'reproduction'),
        ('shoe tread', 'slipping'),
    ]
    for cue, targets in inhibited:
        key = tuple(sorted(cs(cue)))
        assert len(key) == 2, cue
        assert cs(targets).issubset(negative[key]), (cue, targets, negative[key])
        assert not cs(targets).intersection(positive[key]), (cue, targets, positive[key])

    # These superficially related outcomes must remain supported in contexts
    # where they actually occur. Opposite outcomes are not globally forbidden.
    retained = [
        ('wet outlet', 'shock'),
        ('wet hands', 'shock'),
        ('wet skin', 'shock'),
        ('frayed electrical', 'shock fire'),
        ('smooth tire', 'slipping'),
        ('virus host', 'reproduction'),
        ('culture sterile', 'contamination'),
        ('shoe tread', 'friction traction'),
        ('wire rubber', 'insulation'),
        ('culture sealed', 'protection'),
    ]
    for cue, targets in retained:
        key = tuple(sorted(cs(cue)))
        assert len(key) == 2 and cs(targets).issubset(positive[key]), (cue, targets, positive[key])

    print(f'PROTECTION OK: {len(conditional)} mandatory-condition cases,')
    print(f'  {len(inhibited)} all-source inhibited pairs, {len(retained)} retained positive-context cases.')
    print('No questions, labels, predictions, or accuracy scores used.')


if __name__ == '__main__':
    main()

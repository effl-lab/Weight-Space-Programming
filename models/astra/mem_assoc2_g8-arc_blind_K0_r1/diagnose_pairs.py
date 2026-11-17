"""Read-only explicit-sign diagnostic; never reads questions or model outputs.

An overlap is not automatically an error: different scientific contexts can share
pairs. This report does not modify facts, select weights, or measure accuracy.
Only authored signs are examined, before inferred oppositions and hash collisions.
"""
from pathlib import Path
from collections import defaultdict
import build


def main():
    vocab, aliases, source = build.lexicon()
    known = set(vocab)

    def cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}

    negators = cs('not except')
    selectors = cs(' '.join(sorted(build.COMPARATIVE_SELECTORS)))
    new_cues = set()
    for path in Path(build.ROOT).glob('facts_g8_*.txt'):
        for line in path.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                new_cues.add(line.split('|', 1)[0].strip())
    new_compiled = set(build.compile_rules(vocab, aliases,
                       [r for r in source if r[0] in new_cues]))
    slots = defaultdict(lambda: [set(), set()])
    for rule in build.compile_rules(vocab, aliases, source):
        q, p, n, scale = rule
        qs = cs(q)
        anchors = cs(' '.join(build.ANCHOR.findall(q)))
        pairs = build.recall_pairs(sorted(qs), negators,
                                  build.condition_guard(qs, selectors), anchors)
        for pair in pairs:
            for sign, text in enumerate((p, n)):
                for target in cs(text) - qs:
                    slots[(pair, target)][sign].add((q, text, rule in new_compiled))

    overlaps = []
    for (pair, target), (positive, negative) in slots.items():
        if positive and negative and any(new for _, _, new in positive | negative):
            overlaps.append((pair, target, positive, negative))
    print(f'{len(overlaps)} explicit projected-sign overlaps involving new source cues.')
    print('Overlaps are diagnostics, not automatic errors: context and ambiguity matter.')
    for pair, target, positive, negative in sorted(overlaps)[:160]:
        print('\nPAIR', ' / '.join(pair), 'TARGET', target)
        for sign, records in [('+', positive), ('-', negative)]:
            for cue, text, new in sorted(records)[:3]:
                print(sign, 'new-cue' if new else 'other-cue', cue, '|', text)


if __name__ == '__main__':
    main()

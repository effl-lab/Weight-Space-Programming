"""Structural checks on the authored facts; never reads questions or labels.

A rule's cue is a set, not an ordered sentence. Opposed rules with the same cue
set cannot encode opposite directions in the required architecture. Such cases
must be written as symmetric relationships or qualified explicitly by meaningful
concepts (for instance, heating versus cooling).
"""
import math
import re
from collections import defaultdict

TOKEN = re.compile(r"[a-z0-9']+")
# A source-only marker: each @word must remain in every emitted cue pair.
ANCHOR = re.compile(r"(?<![a-z0-9'])@([a-z0-9']+)")
# Shared with build.py so the source audit uses the compiler's exact selectors.
COMPARATIVE_SELECTORS = frozenset(
    'increase decrease more less high low fast slow above below deep shallow'.split()
)


def audit_rules(vocab, aliases, rules):
    """Return encoding errors without selecting or changing any rule or weight."""
    errors = []
    known = set(vocab)
    if len(known) != len(vocab):
        errors.append('Vocabulary contains duplicate words')
    for word in vocab:
        if not TOKEN.fullmatch(word) or word != word.lower():
            errors.append(f'Unencodable vocabulary token: {word!r}')
    # Padding/OOV IDs are supplied by build.py, not by the source vocabulary.
    def concepts(text):
        return frozenset(aliases.get(word, word)
                         for word in TOKEN.findall(text.lower()) if word in known)

    # These words distinguish scientific conditions even though some are
    # ordinarily treated as grammar. Dropping one silently changes a rule.
    selectors = {'above', 'below', 'before', 'after', 'first', 'last', 'next',
                 'not', 'no', 'except'}
    for left, right in [('above', 'below'), ('before', 'after'),
                        ('first', 'last'), ('complete', 'incomplete'),
                        ('open', 'closed'), ('increase', 'decrease'),
                        ('more', 'less'), ('high', 'low'),
                        ('fast', 'slow'), ('deep', 'shallow')]:
        if left in known and right in known and concepts(left) == concepts(right):
            errors.append(f'Distinct selectors merged: {left!r}, {right!r}')
    negators = concepts('not except')
    comparisons = concepts(' '.join(sorted(COMPARATIVE_SELECTORS)))
    if COMPARATIVE_SELECTORS - known:
        errors.append(f'Missing comparative selectors: {sorted(COMPARATIVE_SELECTORS - known)}')
    by_cue = defaultdict(list)
    for cue, positive, negative, scale in rules:
        missing_selectors = selectors.intersection(TOKEN.findall(cue)) - known
        if missing_selectors:
            errors.append(f'Distinguishing cue discarded in {cue!r}: {sorted(missing_selectors)}')
        q, p, n = map(concepts, (cue, positive, negative))
        anchor_words = ANCHOR.findall(cue)
        if cue.count('@') != len(anchor_words):
            errors.append(f'Malformed mandatory-cue marker in {cue!r}')
        anchors = concepts(' '.join(anchor_words))
        if set(anchor_words) - known:
            errors.append(f'Unknown mandatory cue in {cue!r}')
        if '@' in positive or '@' in negative:
            errors.append(f'Mandatory-cue marker used in an answer: {cue!r}')
        if anchors and (len(q) < 2 or len(anchors) > 2):
            errors.append(f'Mandatory cues cannot fit a pair: {cue!r}')
        if len(anchors | (q & negators)) > 2:
            errors.append(f'Mandatory cues plus negation exceed pair order: {cue!r}')
        present = q & comparisons
        guards = present if len(q) > 1 and len(present) == 1 else frozenset()
        required = (q & negators) or guards
        # Two anchors occupy both pair slots. If a retained comparison (or
        # higher-priority negator) lies elsewhere, no emitted pair can exist.
        if len(anchors) == 2 and required and not (anchors & required):
            errors.append(f'Mandatory cues plus retained condition exceed pair order: {cue!r}')
        if not math.isfinite(scale) or scale == 0:
            errors.append(f'Invalid strength for {cue!r}: {scale!r}')
        if not q:
            errors.append(f'Empty concept cue: {cue!r}')
        if scale < 0 and len(q) < 2:
            errors.append(f'Pair-only rule has fewer than two concepts: {cue!r}')
        if q & negators and len(q) < 2:
            # build() suppresses first-order backoff for negated conditions.
            # A lone negator therefore has neither a first-order nor a pair path.
            errors.append(f'Negated rule has no usable concept pair: {cue!r}')
        # This knowledge base requires an explicit recall, not solely an implied
        # opposite. Concepts already in q are always masked by the architecture.
        p, n = p - q, n - q
        if not (p | n):
            errors.append(f'Every explicit answer concept is masked: {cue!r}')
        if p & n:
            errors.append(f'Opposed targets within {cue!r}: {sorted(p & n)}')
        for old_cue, old_p, old_n in by_cue[q]:
            conflict = (p & old_n) | (n & old_p)
            if conflict:
                errors.append(f'Identical cue bags disagree: {old_cue!r}, {cue!r}: {sorted(conflict)}')
        by_cue[q].append((cue, p, n))
    return errors


if __name__ == '__main__':
    import build
    words, alias, facts = build.lexicon()
    problems = audit_rules(words, alias, facts)
    if problems:
        raise SystemExit('\n'.join(problems))
    compiled = build.compile_rules(words, alias, facts)
    problems = audit_rules(words, alias, compiled)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'AUDIT OK: {len(facts)} authored rules, {len(compiled)} unique lexical variants; '
          'no masked-only targets or explicit sign conflicts')

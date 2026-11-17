"""Label-free integrity/resource checks. Never selects knowledge or weights by accuracy."""
from pathlib import Path
import gc
import hashlib
import json
import resource
import time

import numpy as np
import torch
import arch
import build
from audit import audit_rules

ROOT = Path(__file__).resolve().parent
ARCH_SHA256 = '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af'


def tensor_digest(weights):
    h = hashlib.sha256()
    for name, value in sorted(weights.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(memoryview(value.numpy()).cast('B'))
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    assert hashlib.sha256((ROOT / 'arch.py').read_bytes()).hexdigest() == ARCH_SHA256
    vocab, aliases, rules = build.lexicon()
    assert vocab == (ROOT / 'vocab.txt').read_text().splitlines()
    assert not audit_rules(vocab, aliases, rules)
    assert audit_rules(vocab, aliases, [('cell', 'cell', '', 1.0)])
    assert audit_rules(vocab, aliases, [('cell', 'oxygen', 'oxygen', 1.0)])
    assert audit_rules(vocab, aliases, [('cell', 'oxygen', '', float('nan'))])
    stripped = [word for word in vocab if word != 'above']
    assert any('Distinguishing cue discarded' in msg for msg in
               audit_rules(stripped, aliases, [('ph above seven', 'base', '', 1.0)]))
    merged = dict(aliases, above=aliases['below'])
    assert any('Distinct selectors merged' in msg for msg in audit_rules(vocab, merged, []))
    assert build.recall_pairs([1, 2, 3], {1}) == ((1, 2), (1, 3))
    assert build.recall_pairs([1, 2, 3], {4}) == ((1, 2), (1, 3), (2, 3))
    # Orthographic expansion must add spellings, not meanings, and must not
    # recursively grow new apostrophe combinations on repeated calls.
    small_words = {'leaf', 'leaves', 'mass', 'nucleus', 'nuclei', "can't", '2'}
    small_aliases = {'leaves': 'leaf', 'nuclei': 'nucleus', "can't": 'not', '2': 'two'}
    roots_before = {small_aliases.get(word, word) for word in small_words}
    expanded, expanded_aliases = build.extend_surface_forms(small_words, small_aliases)
    assert {expanded_aliases.get(word, word) for word in expanded} == roots_before
    for form, root in [("leaf's", 'leaf'), ("leaves'", 'leaf'), ("mass's", 'mass'),
                       ("nucleus's", 'nucleus'), ("nuclei's", 'nucleus'),
                       ("'can't'", 'not'), ("'2'", 'two')]:
        assert expanded_aliases[form] == root
    assert "leaves's" not in expanded
    repeated_words, repeated_aliases = build.extend_surface_forms(set(expanded), dict(expanded_aliases))
    assert repeated_words == expanded and repeated_aliases == expanded_aliases
    _, overrides = build.extend_surface_forms({'leaf', "leaf's"}, {"leaf's": 'explicit'})
    assert overrides["leaf's"] == 'explicit'
    stop_snapshot = set(build.STOP)
    t = time.perf_counter()
    weights = build.build()
    build_seconds = time.perf_counter() - t
    assert set(weights) == {'concept', 'idf', 'A', 'row_ids', 'row_w', 'mix'}
    C = len(weights['idf'])
    H, R = weights['row_ids'].shape
    assert C >= 16
    assert weights['concept'].shape == (len(vocab) + 2,)
    assert weights['A'].shape == (C, C)
    assert weights['row_w'].shape == (H, R)
    assert weights['mix'].ndim == 0
    for name, value in weights.items():
        assert value.dtype == (torch.int64 if name in ('concept', 'row_ids') else torch.float32)
        assert torch.isfinite(value).all(), name
    assert weights['concept'][:2].tolist() == [0, 0]
    assert weights['idf'][0] == 0
    assert weights['idf'].min() >= 0
    for name in ('concept', 'row_ids'):
        assert 0 <= weights[name].min() <= weights[name].max() < C
    # A zero-weight memory slot is unused; target 0 cannot carry evidence.
    for start in range(0, H, 8192):
        ids = weights['row_ids'][start:start + 8192]
        vals = weights['row_w'][start:start + 8192]
        assert torch.equal(ids == 0, vals == 0)
    assert torch.count_nonzero(weights['A'][0]) == 0
    assert torch.count_nonzero(weights['A'][:, 0]) == 0

    index = {word: i + 2 for i, word in enumerate(vocab)}
    def cid(word):
        return int(weights['concept'][index[word]])
    for left, right in [('cell', 'cells'), ('organ', 'organs'), ('tooth', 'teeth'),
                        ('leaf', 'leaves'), ('root', 'roots'), ('freeze', 'freezing')]:
        assert cid(left) == cid(right), (left, right)
    for left, right in [('speed', 'velocity'), ('pollination', 'pollinator'),
                        ('engine', 'motor'), ('voltage', 'volt'),
                        ('endotherm', 'endothermic'), ('sun', 'solar'),
                        ('above', 'below'), ('first', 'last'), ('close', 'closed'),
                        ('complete', 'closed'), ('incomplete', 'open'),
                        ('solubility', 'dissolution'), ('saturated', 'concentrated'),
                        ('molarity', 'molality'), ('solute', 'solvent'),
                        ('solution', 'suspension'), ('solution', 'colloid'),
                        ('suspension', 'colloid'), ('sulfur', 'sulfide'),
                        ('capillary', 'capillarity'), ('adhesion', 'cohesion'),
                        ('phototropism', 'gravitropism'), ('hydrotropism', 'gravitropism'),
                        ('nymph', 'larva'), ('molt', 'metamorphosis')]:
        assert cid(left) != cid(right), (left, right)
    for left, right in [("earth's", 'earth'), ("plants'", 'plant'), ("'density'", 'density'),
                        ("newton's", 'newton'), ("nuclei's", 'nucleus'), ("'leaf", 'leaf'),
                        ("leaf'", 'leaf'), ("wasn't", 'not'), ("didn't", 'not'),
                        ("wouldn't", 'not'), ('behaviour', 'behavior'), ('colourful', 'colorful'),
                        ('aeroplane', 'airplane'), ('labelled', 'label'), ('oxidised', 'oxidation'),
                        ('colloidal', 'colloid'), ('sulphide', 'sulfide'),
                        ('moles', 'mole'), ('hydrated', 'hydrate'),
                        ('geotropism', 'gravitropism'), ('geotropic', 'gravitropism'),
                        ('towards', 'toward'), ('tracheal', 'trachea')]:
        assert cid(left) == cid(right), (left, right)
    assert build.TOKEN.findall("Earth's 'density'".lower()) == ["earth's", "'density'"]
    negative_contractions = ("isn't", "aren't", "wasn't", "weren't", "doesn't",
                             "didn't", "hasn't", "hadn't", "wouldn't", "shouldn't",
                             "couldn't", "mustn't", "needn't")
    for word in negative_contractions:
        stem = word[:-2]
        assert build.TOKEN.findall(word.replace("'", '’')) == [stem, 't']
        assert cid(stem) == cid(word) == cid('not')
    # Do not turn valid words or a generic single-letter fragment into negation.
    for word in ('can', 'don', 'won', 'haven', 't'):
        assert int(weights['concept'][index.get(word, 1)]) != cid('not')
    assert int(weights['concept'][index.get('t', 1)]) == 0
    assert cid('non') == cid('not')
    assert 'organses' not in index
    # The authored general property must not be inverted by the phrase
    # 'poor conductor'; a qualified graphite exception remains separate.
    assert weights['A'][cid('nonmetal'), cid('conduct')] < 0
    assert weights['A'][cid('nonmetal'), cid('insulate')] > 0
    assert weights['A'][cid('insulator'), cid('conduct')] < 0
    for neg in ('not', 'except'):
        assert torch.count_nonzero(weights['A'][cid(neg)]) == 0

    # Source-definition regressions, not scored questions. Keep denominator,
    # total-particle count, and material-phase distinctions explicit.
    def meanings(text):
        return {cid(word) for word in build.TOKEN.findall(text) if word in index}
    for cue, positive, negative, scale in rules:
        if cue == 'concentration':
            assert cid('solution') in meanings(positive)
            assert cid('solvent') not in meanings(positive)
        if cue == 'solvent':
            assert cid('liquid') not in meanings(positive)
        if cue == 'solution dilute color':
            assert cid('concentration') in meanings(positive)
            assert cid('fewer') not in meanings(positive)
    for cue, expected in [('molarity denominator', 'solution volume'),
                          ('molality denominator', 'solvent mass'),
                          ('dilution particle number', 'unchanged conserved'),
                          ('silver tarnish product', 'black sulfide')]:
        matching = [positive for q, positive, _, _ in rules if q == cue]
        assert matching and all(meanings(expected) <= meanings(p) for p in matching)

    # Definitions should work as cues, not only as answer-side vocabulary.
    for cue in ('adhesion', 'cohesion', 'capillarity', 'gravitropism',
                'hydrotropism', 'thigmotropism', 'molt', 'spiracle', 'nymph'):
        assert any(q == cue and scale > 0 for q, _, _, scale in rules), cue
        assert weights['idf'][cid(cue)] > 0, cue
        assert torch.count_nonzero(weights['A'][cid(cue)]) > 0, cue
    assert weights['A'][cid('adhesion'), cid('different')] > 0
    assert weights['A'][cid('adhesion'), cid('identical')] < 0
    assert weights['A'][cid('cohesion'), cid('identical')] > 0
    assert weights['A'][cid('cohesion'), cid('different')] < 0
    assert weights['A'][cid('capillarity'), cid('adhesion')] > 0
    # The general phenomenon is direction-neutral; qualified rules specify
    # toward/away. This guards the source definition, not prediction accuracy.
    photo_definitions = [p for q, p, _, _ in rules if q == 'phototropism']
    assert photo_definitions
    assert all(not meanings(p).intersection(meanings('toward away'))
               for p in photo_definitions)

    # Randomness is confined to test inputs; the model/compiler contains none.
    generator = torch.Generator().manual_seed(5729)
    qs = torch.randint(2, len(vocab) + 2, (8, 14), generator=generator)
    os = torch.randint(2, len(vocab) + 2, (8, 4, 9), generator=generator)
    def padded(src, length):
        dst = torch.zeros((*src.shape[:-1], length), dtype=torch.int64)
        dst[..., :src.shape[-1]] = src
        return dst
    q, o = padded(qs, 128), padded(os, 48)
    def encoded(text, length):
        tokens = build.TOKEN.findall(text.lower())[:length]
        return padded(torch.tensor([index.get(word, 1) for word in tokens], dtype=torch.int64), length)
    # Unlabeled synthetic bags: spelling variants must yield identical tensors
    # and logits, including when variants occur in options rather than questions.
    base_q = encoded('earth moon sun axis orbit', 128)[None]
    form_q = encoded("Earth's moon's sun's 'axis' 'orbit", 128)[None]
    base_o = torch.stack([encoded(text, 48) for text in
                         ['plant leaf', 'animal cell', 'oxygen atom', 'newton force']])[None]
    form_o = torch.stack([encoded(text, 48) for text in
                         ["plants' leaves'", "animal's cell's", "'oxygen' 'atom'", "newton's 'force'"]])[None]
    def same_bags_and_logits(left_q, left_o, right_q, right_o):
        for left, right in zip(arch.bags(weights, left_q, left_o),
                               arch.bags(weights, right_q, right_o)):
            assert torch.equal(left, right)
        assert torch.equal(arch.forward(weights, left_q, left_o),
                           arch.forward(weights, right_q, right_o))

    with torch.no_grad():
        same_bags_and_logits(base_q, base_o, form_q, form_o)
        # All four punctuation marks split under the fixed ASCII tokenizer.
        # These are unlabeled spelling-equivalence fixtures, not accuracy probes.
        punctuation_pairs = [(word, word.replace("'", mark))
                             for word in negative_contractions
                             for mark in ('’', '‘', 'ʼ', '＇')]
        n_forms = len(punctuation_pairs)
        plain_q = torch.stack([encoded('earth ' + word + ' moon axis orbit', 128)
                               for word, _ in punctuation_pairs])
        curly_q = torch.stack([encoded('earth ' + word + ' moon axis orbit', 128)
                               for _, word in punctuation_pairs])
        plain_o = base_o.repeat(n_forms, 1, 1)
        same_bags_and_logits(plain_q, plain_o, curly_q, plain_o)
        # Keep the question free of negation so option negators are not masked.
        plain_q = base_q.repeat(n_forms, 1)
        curly_o = plain_o.clone()
        for i, (plain, curly) in enumerate(punctuation_pairs):
            plain_o[i, 0] = encoded('plant ' + plain + ' leaf', 48)
            curly_o[i, 0] = encoded('plant ' + curly + ' leaf', 48)
        same_bags_and_logits(plain_q, plain_o, plain_q, curly_o)
        # Existing non- prefix handling also needs to retain its negator.
        stems = ('living', 'renewable', 'metal', 'vascular', 'magnetic')
        prefix_q = torch.stack([encoded('non-' + word, 128) for word in stems])
        explicit_q = torch.stack([encoded('not ' + word, 128) for word in stems])
        prefix_o = base_o.repeat(len(stems), 1, 1)
        same_bags_and_logits(prefix_q, prefix_o, explicit_q, prefix_o)
        logits = arch.forward(weights, q, o)
        assert logits.shape == (8, 4) and torch.isfinite(logits).all()
        perm = torch.tensor([2, 0, 3, 1])
        assert torch.equal(arch.forward(weights, q, o[:, perm]), logits[:, perm])
        assert torch.equal(arch.forward(weights, q.flip(-1), o.flip(-1)), logits)
        duplicate = arch.forward(weights, padded(qs.repeat(1, 2), 128),
                                 padded(os.repeat(1, 1, 2), 48))
        assert torch.equal(duplicate, logits)
        all_unknown = arch.forward(weights, torch.ones_like(q), torch.ones_like(o))
        assert torch.count_nonzero(all_unknown) == 0
        repeated_question = padded(qs[:, None, :].repeat(1, 4, 1), 48)
        assert torch.count_nonzero(arch.forward(weights, q, repeated_question)) == 0
        # Multiple batches, exercising real tensor dimensions, without labels.
        q64, o64 = q.repeat(8, 1), o.repeat(8, 1, 1)
        t = time.perf_counter()
        for _ in range(10):
            output = arch.forward(weights, q64, o64)
            assert torch.isfinite(output).all()
        ten_batch_seconds = time.perf_counter() - t

    fingerprint = tensor_digest(weights)
    # Source order and duplicate authored statements must not affect weights.
    original_sources = build.sources
    aliases0, groups0, rules0 = original_sources()
    try:
        build.sources = lambda: (dict(aliases0), list(groups0), list(reversed(rules0)) * 2)
        candidate = build.build()
        assert tensor_digest(candidate) == fingerprint
        del candidate
        gc.collect()
    finally:
        build.sources = original_sources
    assert build.STOP == stop_snapshot
    assert build.lexicon()[0] == vocab

    # Additional masked negative concepts cannot alter the compiled evidence.
    try:
        transformed = [(q, p, n + ' ' + q, s) for q, p, n, s in rules0]
        build.sources = lambda: (dict(aliases0), list(groups0), transformed)
        candidate = build.build()
        assert tensor_digest(candidate) == fingerprint
        del candidate
        gc.collect()
    finally:
        build.sources = original_sources

    # Synthetic compiler fixtures, not science questions or proposed model facts.
    # They keep the same vocabulary and exercise exact tensor compilation.
    original_lexicon = build.lexicon
    def fixture(rules):
        try:
            build.lexicon = lambda: (vocab, aliases, rules)
            return build.build()
        finally:
            build.lexicon = original_lexicon

    def row_of(a, b):
        a, b = sorted((a, b))
        return (a * 1000003 + b) % H

    def memory_value(w, pair, target):
        row = row_of(*pair)
        mask = w['row_ids'][row] == target
        return float(w['row_w'][row][mask].sum())

    neg, nucleus, membrane, bacteria = map(cid, ['not', 'nucleus', 'membrane', 'bacteria'])
    assert len({row_of(neg, nucleus), row_of(neg, membrane), row_of(nucleus, membrane)}) == 3
    candidate = fixture([('not nucleus membrane', 'bacteria', '', 1.0)])
    assert torch.count_nonzero(candidate['A']) == 0
    assert memory_value(candidate, (nucleus, membrane), bacteria) == 0
    assert memory_value(candidate, (neg, nucleus), bacteria) > 0
    assert memory_value(candidate, (neg, membrane), bacteria) > 0
    del candidate
    gc.collect()

    # Construct a storage collision algebraically. An assertion for the other
    # pair must not suppress a logically unrelated negation rule.
    anchor, target = cid('enzyme'), cid('adenosine')
    wanted_row = row_of(anchor, neg)
    excluded = {anchor, target, neg, cid('except')}
    collision = None
    for a in range(1, C):
        b = (wanted_row - a * 1000003) % H
        if a < b < C and not ({a, b} & excluded):
            collision = (a, b)
            break
    assert collision is not None
    names = {cid(word): word for word in vocab}
    left, right = (names[c] for c in collision)
    candidate = fixture([('enzyme', 'adenosine', '', 1.0),
                         (left + ' ' + right, 'adenosine', '', -0.1)])
    expected = 0.1 - 1.5 * 0.65 / float(candidate['idf'][neg])
    assert np.isclose(memory_value(candidate, (anchor, neg), target), expected)
    del candidate
    gc.collect()
    # In contrast, an explicit assertion for the very same semantic pair wins.
    candidate = fixture([('enzyme', 'adenosine', '', 1.0),
                         ('enzyme not', 'adenosine', '', -0.1)])
    assert np.isclose(memory_value(candidate, (anchor, neg), target), 0.1)
    del candidate
    gc.collect()
    # Authored conditional exceptions must stay pair-only and preserve their
    # signs. This isolated compilation tests the rule contract, not answer ranks.
    review_cues = {'stirring solubility', 'crushing solubility',
                   'catalyst equilibrium', 'catalyst reaction heat',
                   'positive tropism', 'negative tropism',
                   'positive phototropism', 'negative phototropism',
                   'positive gravitropism', 'negative gravitropism'}
    review_rules = [rule for rule in rules if rule[0] in review_cues]
    assert len(review_rules) == len(review_cues)
    assert all(scale < 0 for _, _, _, scale in review_rules)
    candidate = fixture(review_rules)
    assert torch.count_nonzero(candidate['A']) == 0
    for cue, positive, negative, _ in review_rules:
        qs = sorted(meanings(cue))
        for pair in build.recall_pairs(qs, {cid('not'), cid('except')}):
            for target in meanings(positive) - set(qs):
                assert memory_value(candidate, pair, target) > 0
            for target in meanings(negative) - set(qs):
                assert memory_value(candidate, pair, target) < 0
    del candidate
    gc.collect()
    assert build.lexicon is original_lexicon
    assert build.STOP == stop_snapshot

    byte_count = sum(value.numel() * value.element_size() for value in weights.values())
    assert byte_count < 1024 ** 3
    report = {
        'status': 'PASS',
        'rules': len(rules), 'vocabulary': len(vocab), 'concepts': C,
        'hash_rows': H, 'slots_per_row': R,
        'weights': sum(value.numel() for value in weights.values()),
        'nonzero': sum(int(value.count_nonzero()) for value in weights.values()),
        'weight_bytes': byte_count,
        'build_seconds': round(build_seconds, 3),
        'ten_batches_of_64_seconds': round(ten_batch_seconds, 3),
        'verification_seconds': round(time.perf_counter() - started, 3),
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'tensor_sha256': fingerprint,
        'architecture_sha256': ARCH_SHA256,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

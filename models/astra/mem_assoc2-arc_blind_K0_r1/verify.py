"""Non-accuracy regression checks for the authored association-memory compiler.
Run: python /srv/work/model/verify.py
No question labels, datasets, parameter selection, or model-file edits are used.
"""
import gc
import hashlib
import resource
import time

import torch
import arch
import build


def digest(weights):
    result = hashlib.sha256()
    for key in sorted(weights):
        result.update(key.encode())
        result.update(memoryview(weights[key].numpy()).cast('B'))
    return result.hexdigest()


def main():
    torch.set_num_threads(4)
    original_stop = set(build.STOP)
    weights = build.build()
    signature = digest(weights)
    tensor_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    shapes = {name: list(t.shape) for name, t in weights.items()}
    assert set(weights) == {'concept', 'idf', 'A', 'row_ids', 'row_w', 'mix'}
    assert weights['concept'][:2].tolist() == [0, 0]
    assert weights['idf'][0].item() == 0
    assert all(torch.isfinite(t).all().item() for t in weights.values())
    assert torch.all(weights['row_w'][weights['row_ids'] == 0] == 0).item()
    vocab, aliases, rules = build.lexicon()
    C = weights['idf'].numel()
    assert C >= 16
    assert weights['concept'].shape == (len(vocab) + 2,)
    assert weights['A'].shape == (C, C)
    assert weights['row_ids'].ndim == 2
    assert weights['row_ids'].shape == weights['row_w'].shape
    assert min(weights['row_ids'].shape) > 0
    assert weights['mix'].ndim == 0
    for name in ('concept', 'row_ids'):
        assert weights[name].dtype == torch.int64
        assert 0 <= int(weights[name].min()) <= int(weights[name].max()) < C
    for name in ('idf', 'A', 'row_w', 'mix'):
        assert weights[name].dtype == torch.float32
    assert torch.equal(weights['row_ids'] == 0, weights['row_w'] == 0)
    assert not build.audit_rules(vocab, aliases, rules)
    # Invalid symbolic encodings must be rejected, without testing answer labels.
    invalid_cases = [
        [('voltage', 'voltage', '', 1.0)],
        [('atom', 'electron', 'electron', 1.0)],
        [('water', 'oxygen', '', -1.0)],
        [('water', 'oxygen', '', float('nan'))],
        [('solid liquid', 'melting', 'freezing', 1.0),
         ('liquid solid', 'freezing', 'melting', 1.0)],
    ]
    for malformed in invalid_cases:
        assert build.audit_rules(vocab, aliases, malformed), malformed
    index = {word: i + 2 for i, word in enumerate(vocab)}
    def concept(word):
        return int(weights['concept'][index[word]])

    for a, b in [
        ('velocity', 'speed'), ('frog', 'tadpole'), ('groundwater', 'aquifer'),
        ('engine', 'motor'), ('thunder', 'thunderstorm'), ('conifer', 'pine'),
        ('conifer', 'evergreen'), ('nonflowering', 'gymnosperm'),
        ('chlorine', 'chloride'), ('phosphorus', 'phosphate'),
        ('reaction', 'reactant'), ('buoyancy', 'float'),
        ('electromagnet', 'electromagnetic'), ('dissolve', 'solution'),
        ('starch', 'carbohydrate'), ('glucose', 'sugar'),
        ('weather', 'weathering'), ('nucleus', 'nuclear'),
        ('sun', 'solar'), ('voltage', 'volt'), ('scratch', 'hardness'),
        ('young', 'offspring'), ('annual', 'year'),
        ('paleontology', 'paleontologist'), ('archaeology', 'archaeologist'),
        ('geology', 'geologist'), ('meteorology', 'meteorologist'),
        ('engineering', 'engineer'), ('ecology', 'ecosystem'),
    ]:
        assert concept(a) != concept(b), (a, b)
    for a, b in [('frog', 'frogs'), ('velocity', 'velocities'),
                 ('new', 'newest'), ('not', 'without')]:
        assert concept(a) == concept(b), (a, b)
    assert concept('new') != 0
    assert concept('old') != 0
    for negative_property in [('prokaryote', 'nucleus'), ('gas', 'definite'),
                              ('metamorphic', 'melting')]:
        a, b = map(concept, negative_property)
        assert weights['A'][a, b].item() < 0, negative_property
    for negator in ('not', 'except'):
        assert torch.count_nonzero(weights['A'][concept(negator)]).item() == 0
        assert weights['idf'][concept(negator)].item() > 0
    for topic in ('paleontologist', 'albedo', 'stalactite', 'annual', 'biennial',
                  'perennial', 'synapse', 'projectile', 'kilowatt', 'proxima'):
        assert torch.count_nonzero(weights['A'][concept(topic)]).item() > 0, topic
    inactive = (weights['idf'] == 0).nonzero().flatten()
    inactive = inactive[inactive > 0]
    inactive_count = inactive.numel()
    assert torch.count_nonzero(weights['A'][inactive]).item() == 0
    assert torch.count_nonzero(weights['A'][:, inactive]).item() == 0
    referenced = weights['row_ids'][weights['row_ids'] > 0]
    assert torch.all(weights['idf'][referenced] > 0).item()
    del referenced

    # Synthetic token arrays test only the specified bag architecture, not answers.
    gen = torch.Generator().manual_seed(20261115)
    q = torch.zeros((64, 128), dtype=torch.int64)
    o = torch.zeros((64, 4, 48), dtype=torch.int64)
    q[:, :20] = torch.randint(2, len(vocab) + 2, (64, 20), generator=gen)
    o[:, :, :12] = torch.randint(2, len(vocab) + 2, (64, 4, 12), generator=gen)
    with torch.no_grad():
        start = time.monotonic()
        logits = arch.forward(weights, q, o)
        for _ in range(9):
            arch.forward(weights, q, o)
        elapsed = time.monotonic() - start
        assert logits.shape == (64, 4)
        assert torch.isfinite(logits).all().item()
        permutation = [2, 0, 3, 1]
        assert torch.equal(arch.forward(weights, q, o[:, permutation]), logits[:, permutation])
        assert torch.equal(arch.forward(weights, q.flip(1), o.flip(2)), logits)
        repeated_q, repeated_o = q.clone(), o.clone()
        repeated_q[:, 20:40] = q[:, :20]
        repeated_o[:, :, 12:24] = o[:, :, :12]
        assert torch.equal(arch.forward(weights, repeated_q, repeated_o), logits)
        unknown = arch.forward(weights, torch.ones((1, 128), dtype=torch.int64),
                               torch.ones((1, 4, 48), dtype=torch.int64))
        assert torch.equal(unknown, torch.zeros((1, 4)))
        if inactive_count:
            inactive_word_ids = (weights['idf'][weights['concept']] == 0).nonzero().flatten()
            inactive_word_ids = inactive_word_ids[inactive_word_ids > 1]
            assert inactive_word_ids.numel() > 0
            inert_q = q.clone()
            inert_q[:, 20:] = inactive_word_ids[0]
            assert torch.equal(arch.forward(weights, inert_q, o), logits)
            inert_only = torch.full((1, 128), int(inactive_word_ids[0]), dtype=torch.int64)
            assert torch.equal(arch.forward(weights, inert_only, o[:1]), torch.zeros((1, 4)))
    del weights
    gc.collect()

    # Source ordering and duplicate statements must not determine any weight.
    original_sources = build.sources
    alias, groups, parsed = original_sources()
    # Adding cue concepts to a negative field cannot add usable targets. For
    # positive fields, test neutral nouns only: explicit positive contrast
    # markers intentionally infer opposites even when the marker is masked.
    neutral_nouns = {'atom', 'sun', 'water', 'earth', 'cell'}
    masked_negative = [(cue, pos, neg + ' ' + cue, scale)
                       for cue, pos, neg, scale in parsed]
    masked_positive = [
        (cue, pos + ' ' + ' '.join(
            word for word in build.TOKEN.findall(cue)
            if alias.get(word, word) in neutral_nouns), neg, scale)
        for cue, pos, neg, scale in parsed
    ]
    variants = [
        ('reversed rules', list(reversed(parsed))),
        ('duplicate rules', parsed + parsed),
        ('masked negative targets', masked_negative),
        ('masked neutral positive targets', masked_positive),
    ]
    try:
        for label, reordered in variants:
            build.sources = lambda: (dict(alias), list(groups), reordered)
            weights = build.build()
            assert digest(weights) == signature, label
            del weights
            gc.collect()
    finally:
        build.sources = original_sources
    assert build.STOP == original_stop
    print('PASS: source audit, malformed-rule rejection, tensor contract')
    print('PASS: glossary distinctions, explicit negatives, pure deterministic build')
    print('PASS: rule-order, duplicate-rule, and masked-target normalization invariance')
    print('PASS: inactive-concept isolation; disconnected concepts:', inactive_count)
    print('PASS: finite outputs, option/token-order invariance, duplicate-token invariance, unknown tie')
    print('Rules:', len(rules), 'Vocabulary:', len(vocab))
    print('Tensor shapes:', shapes)
    print('Tensor bytes:', tensor_bytes)
    print('Weight SHA256:', signature)
    print('Ten batches of 64 seconds:', round(elapsed, 3))
    print('Peak resident MiB:', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


if __name__ == '__main__':
    main()

"""Source and tensor invariants only; no question labels or accuracy evaluation.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python model/verify.py
The synthetic forward checks test masking, batching, and option permutation,
not whether any science answer wins. Nothing here selects model parameters.
"""
import hashlib
import itertools
import resource
import time
from pathlib import Path

import numpy as np
import torch

import arch
import build
from audit import audit_rules

ROOT = Path(__file__).resolve().parent
EXPECTED_ARCH = '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af'


def main():
    torch.set_num_threads(4)
    assert hashlib.sha256((ROOT / 'arch.py').read_bytes()).hexdigest() == EXPECTED_ARCH
    vocab, aliases, rules = build.lexicon()
    assert vocab == (ROOT / 'vocab.txt').read_text().splitlines()
    assert len(vocab) == len(set(vocab))
    assert not audit_rules(vocab, aliases, rules)
    compiled = build.compile_rules(vocab, aliases, rules)
    assert not audit_rules(vocab, aliases, compiled)
    # Declared spelling alternatives must be encodable on both sides. A later
    # missing token must not silently disappear from a partially valid target.
    lexical_words = set(vocab)
    for phrase, fused in build.PHRASE_ALTERNATIVES + build.SURFACE_COMPOUNDS:
        assert set(build.TOKEN.findall(phrase + ' ' + fused)).issubset(lexical_words), (phrase, fused)
    # Lexical compilation is a fixed point, not iterative knowledge inference.
    assert build.compile_rules(vocab, aliases, compiled) == compiled
    # Source order and duplicate frequency must not change compiled evidence.
    assert build.compile_rules(vocab, aliases, list(reversed(rules)) + rules) == compiled

    # Guarded conditions cannot project an unqualified subject/property pair.
    cues = ['gas', 'temperature', 'increase']
    guard = build.condition_guard(cues, {'increase', 'decrease'})
    assert guard == {'increase'}
    pairs = build.recall_pairs(cues, {'not', 'except'}, guard)
    assert len(pairs) == 2 and all('increase' in p for p in pairs)
    assert build.condition_guard(['pressure', 'high', 'low'], {'high', 'low'}) == set()
    assert build.condition_guard(['increase'], {'increase'}) == set()
    cues = ['gas', 'not', 'increase']
    pairs = build.recall_pairs(cues, {'not', 'except'}, {'increase'})
    assert all('not' in pair for pair in pairs)
    assert build.recall_pairs([1, 2, 3], set()) == tuple(itertools.combinations([1, 2, 3], 2))

    # Explicit conditions are mandatory, including when a second condition is
    # negation. Two anchors consume both slots rather than leaking third cues.
    anchored_pairs = build.recall_pairs(['gas', 'sealed', 'heat'], set(), anchors={'heat'})
    assert len(anchored_pairs) == 2 and all('heat' in pair for pair in anchored_pairs)
    assert build.recall_pairs([1, 2, 3], set(), anchors={1, 2}) == ((1, 2),)
    assert build.recall_pairs(['heat', 'not', 'gas'], {'not'}, anchors={'heat'}) == (('heat', 'not'),)
    for bad in [
        ('gas @temperature @pressure @increase', 'energy', '', 1.0),
        ('gas @unknownmandatoryword', 'energy', '', 1.0),
        ('gas', '@pressure', '', 1.0),
        ('gas @', 'pressure', '', 1.0),
        ('gas @pressure @volume increase', 'energy', '', 1.0),
        ('gas @pressure @volume not', 'energy', '', 1.0),
    ]:
        assert audit_rules(vocab, aliases, [bad]), bad
    for cue in ['gas @pressure volume increase',
                'gas @pressure @increase volume',
                'gas @pressure @volume high low',
                'gas @pressure @not increase']:
        assert not audit_rules(vocab, aliases, [(cue, 'energy', '', 1.0)]), cue
    # Aliasing away a selector would silently change the comparative guard.
    for left, right in [('increase', 'decrease'), ('more', 'less'), ('high', 'low'),
                        ('fast', 'slow'), ('above', 'below'), ('deep', 'shallow')]:
        merged_aliases = dict(aliases)
        merged_aliases[left] = aliases.get(right, right)
        assert audit_rules(vocab, merged_aliases, [('gas', 'pressure', '', 1.0)]), (left, right)
    guarded_variants = build.compile_rules(vocab, aliases,
                                          [('@foodchain energy', 'transfer', '', 1.0)])
    assert guarded_variants
    assert all(build.ANCHOR.findall(q) == ['foodchain'] for q, _, _, _ in guarded_variants)

    # All-source regressions for specifically reviewed conditional projections.
    # These inspect authored assertions, not measured logits or answer accuracy.
    known = set(vocab)
    def cs(text):
        return {aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known}
    selectors = cs('increase decrease more less high low fast slow above below deep shallow')
    negators = cs('not except')
    forbidden_projections = [
        ('gas sealed', 'increase decrease'),
        ('ramp distance', 'increase decrease'),
        ('pitch string', 'fast slow increase decrease'),
        ('periodic group', 'alkali halogen noble'),
        ('atom change', 'element ion isotope'),
        ('reaction bond', 'absorb release'),
        ('light surface', 'diffuse specular'),
        ('radiation surface', 'absorb reflect'),
        ('candle wax', 'physical chemical'),
        ('cobalt chloride', 'blue pink'),
        ('foodchain fewer', 'increase shortage'),
        ('food chain', 'increase decrease shortage'),
        ('water treatment', 'salt'),
        ('microscope field', 'increase decrease'),
        ('graph slope', 'increase decrease'),
        ('wire choosing', 'insulate'),
        ('hypothesis evidence', 'revise refute confirmation'),
        ('temperature dependent', 'germination'),
        ('blood high', 'insulin'),
        ('blood low', 'glucagon'),
        ('liquid distillation', 'boiling cooling'),
        ('red object', 'dark'),
        ('blue light', 'dark'),
        ('white paper', 'red'),
        ('antibiotic infection', 'ineffective'),
        ('population isolated', 'inbreeding drift'),
        ('habitat roads', 'migration'),
    ]
    reviewed = [(tuple(sorted(cs(cue))), cs(targets)) for cue, targets in forbidden_projections]
    for cue, pos, neg, _ in compiled:
        qs = sorted(cs(cue))
        affected = [(pair, targets) for pair, targets in reviewed
                    if set(pair).issubset(qs) and targets.intersection(cs(pos) | cs(neg))]
        if affected:
            emitted = build.recall_pairs(qs, negators, build.condition_guard(qs, selectors),
                                         cs(' '.join(build.ANCHOR.findall(cue))))
            assert not any(pair in emitted for pair, _ in affected), cue

    # Reviewed inhibitors must not be restored as positive claims elsewhere.
    signed_cases = [('vacuum gap', 'conduction convection'),
                    ('series broken', 'current'), ('heart one direction', 'backflow'),
                    ('morning shadow', 'east'), ('sunset shadow', 'west'),
                    ('solar panels emissions', 'fuel combustion'),
                    ('wind turbines emissions', 'fuel combustion'),
                    ('microbial culture closed', 'exposure'),
                    ('ocean acidification shell', 'calcification'),
                    ('acid ocean shell', 'calcification'),
                    ('material flexible', 'breaking'),
                    ('material brittle', 'plasticity'),
                    ('antibiotic virus', 'effective'),
                    ('antibiotic bacterial', 'growth'),
                    ('geographic isolation', 'migration'),
                    ('reproductive isolation', 'interbreeding'),
                    ('habitat fragmentation', 'migration'),
                    ('habitat isolated', 'migration'),
                    ('isolation migration', 'increase'),
                    ('isolation interbreeding', 'increase'),
                    ('habitat fragmentation roads', 'migration')]
    for cue, inhibited in signed_cases:
        matching = [(cs(pos), cs(neg)) for q, pos, neg, _ in compiled if cs(q) == cs(cue)]
        assert matching, cue
        targets = cs(inhibited)
        assert all(not (pos & targets) for pos, _ in matching), cue
        assert targets.issubset(set().union(*(neg for _, neg in matching))), cue

    # Viral antibiotic ineffectiveness needs both concepts, never an antibiotic
    # singleton or an unqualified antibiotic/infection pair from inherited rules.
    viral_antibiotic = cs('antibiotic virus')
    checked_antibiotic_rules = 0
    for cue, pos, _, scale in compiled:
        qs = cs(cue)
        if viral_antibiotic.issubset(qs) and cs('ineffective') & cs(pos):
            anchors = cs(' '.join(build.ANCHOR.findall(cue)))
            assert scale < 0 or anchors, cue
            emitted = build.recall_pairs(sorted(qs), negators,
                                         build.condition_guard(qs, selectors), anchors)
            assert emitted and all(set(pair) == viral_antibiotic for pair in emitted), cue
            checked_antibiotic_rules += 1
    assert checked_antibiotic_rules

    # Color descriptions must not attach other wavelengths or their absorption
    # to the observed-color mechanism. This does NOT assert no absorption: those
    # other wavelengths still can be absorbed, but the bag cannot bind their role.
    relation_cases = [('red object', 'reflect', 'absorb other'),
                      ('blue object', 'reflect', 'absorb other'),
                      ('red filter', 'transmits', 'absorb other'),
                      ('green leaf', 'reflect', 'red blue absorb')]
    for cue, supported, unbound in relation_cases:
        matching = [cs(pos) for q, pos, _, _ in compiled if cs(q) == cs(cue)]
        assert matching, cue
        assert all(not (pos & cs(unbound)) for pos in matching), cue
        assert cs(supported).issubset(set().union(*matching)), cue

    # Do not turn "reduced gene flow" into positive flow evidence or negate genes
    # themselves. Connectivity/corridor descriptions still support actual exchange.
    isolation_cases = [('geographic isolation', 'speciation', 'flow'),
                       ('reproductive isolation', 'speciation', 'flow'),
                       ('species isolated', 'divergence', 'flow'),
                       ('habitat fragmentation', 'isolated', 'flow'),
                       ('habitat isolated', 'separation', 'flow increase decrease')]
    for cue, supported, unbound in isolation_cases:
        matching = [(cs(pos), cs(neg)) for q, pos, neg, _ in compiled if cs(q) == cs(cue)]
        assert matching, cue
        assert all(not (pos & cs(unbound)) for pos, _ in matching), cue
        assert all(not (neg & cs('gene')) for _, neg in matching), cue
        assert cs(supported).issubset(set().union(*(pos for pos, _ in matching))), cue
    for cue in ['forest corridor', 'wildlife corridor', 'habitat corridor']:
        matching = [(cs(pos), cs(neg)) for q, pos, neg, _ in compiled if cs(q) == cs(cue)]
        assert matching, cue
        assert cs('gene flow').issubset(set().union(*(pos for pos, _ in matching))), cue
        assert all(not (neg & cs('gene flow migration')) for _, neg in matching), cue

    # Lexical alternatives preserve token boundaries and conditional pair-only rules.
    assert 'food chain' in build.phrase_variants('foodchain')
    assert 'foodchain' in build.phrase_variants('food chain')
    assert 'solar cell current' in build.phrase_variants('solarcell current')
    assert build.phrase_variants('seafoodchain') == ('seafoodchain',)
    pair_only = build.compile_rules(vocab, aliases, [('carbon dioxide', 'gas', '', -1.0)])
    assert all(len({aliases.get(t, t) for t in build.TOKEN.findall(q) if t in vocab}) >= 2
               for q, _, _, _ in pair_only)
    for left, right in [('stomatal', 'stomata'), ('dried', 'dry'),
                        ('dragged', 'drag'), ('co2', 'carbondioxide')]:
        assert aliases.get(left, left) == aliases.get(right, right), (left, right)
    for left, right in [('increase', 'decrease'), ('mass', 'weight'),
                        ('speed', 'velocity'), ('foodchain', 'chain')]:
        assert aliases.get(left, left) != aliases.get(right, right), (left, right)

    started = time.monotonic()
    w = build.build()
    build_seconds = time.monotonic() - started
    assert set(w) == {'concept', 'idf', 'A', 'row_ids', 'row_w', 'mix'}
    C = len(w['idf'])
    assert C >= 16
    assert w['concept'].shape == (len(vocab) + 2,)
    assert w['concept'].dtype == torch.int64
    assert w['concept'][:2].tolist() == [0, 0]
    assert int(w['concept'].min()) >= 0 and int(w['concept'].max()) < C
    assert w['A'].shape == (C, C)
    assert w['row_ids'].shape == w['row_w'].shape
    assert w['row_ids'].dtype == torch.int64
    assert w['mix'].shape == ()
    for name in ('idf', 'A', 'row_w', 'mix'):
        assert w[name].dtype == torch.float32
    for name in ('idf', 'A', 'mix'):
        assert bool(torch.isfinite(w[name]).all())
    H, R = w['row_ids'].shape
    for i in range(0, H, 32768):
        ids = w['row_ids'][i:i + 32768]
        values = w['row_w'][i:i + 32768]
        assert int(ids.min()) >= 0 and int(ids.max()) < C
        assert bool(torch.isfinite(values).all())
        assert bool((values[ids == 0] == 0).all())

    # Synthetic inputs exercise only architecture invariants, not answer quality.
    index = {word: i + 2 for i, word in enumerate(vocab)}
    def encoded(text, length):
        ids = [index.get(word, 1) for word in build.TOKEN.findall(text)][:length]
        return ids + [0] * (length - len(ids))
    q = torch.tensor([encoded('copper wire carries electric current', 128)] * 64)
    options = ['electrons move', 'sand and glass', 'water vapor', 'sugar molecules']
    o = torch.tensor([[encoded(t, 48) for t in options]] * 64)
    with torch.no_grad():
        started = time.monotonic()
        logits = arch.forward(w, q, o)
        forward_seconds = time.monotonic() - started
        assert logits.shape == (64, 4) and bool(torch.isfinite(logits).all())
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = arch.forward(w, q[:2], o[:2, permutation])
        assert torch.allclose(permuted, logits[:2, permutation], rtol=1e-5, atol=1e-6)
        unknown_q = torch.ones(1, 128, dtype=torch.int64)
        unknown_o = torch.ones(1, 4, 48, dtype=torch.int64)
        assert bool((arch.forward(w, unknown_q, unknown_o) == 0).all())
        duplicated = torch.cat([q[:1, :5], q[:1, :5], torch.zeros(1, 118, dtype=torch.int64)], 1)
        assert torch.allclose(arch.forward(w, duplicated, o[:1]), logits[:1], rtol=1e-5, atol=1e-6)
        repeated_option = q[:1, :48].reshape(1, 1, 48).repeat(1, 4, 1)
        assert bool((arch.forward(w, q[:1], repeated_option) == 0).all())

        # Resource stress only: 4,480 full-length, deterministic token patterns,
        # exceeding validation + test counts. No text labels or accuracy scores.
        varied_q = torch.arange(64 * 128).reshape(64, 128) * 131 + 7
        varied_o = torch.arange(64 * 4 * 48).reshape(64, 4, 48) * 197 + 11
        started = time.monotonic()
        for batch in range(70):
            offset = batch * 1543
            stress_q = (varied_q + offset) % len(vocab) + 2
            stress_o = (varied_o + offset) % len(vocab) + 2
            stress_logits = arch.forward(w, stress_q, stress_o)
            assert stress_logits.shape == (64, 4)
            assert bool(torch.isfinite(stress_logits).all())
        stress_seconds = time.monotonic() - started
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak_kib * 1024 < 2_000_000_000, 'Exceeded conservative decimal 2 GB process RSS limit'
    assert build_seconds + stress_seconds < 20 * 60, 'Exceeded 20-minute build/inference budget'

    print('VERIFY OK: source logic, lexical alternatives, tensor schema, finite values,')
    print('           option permutation, repeated-token invariance, question masking, OOV ties.')
    print(f'rules={len(rules)} compiled={len(compiled)} vocabulary={len(vocab)} concepts={C}')
    print(f'hash_rows={H} row_width={R} tensor_bytes={sum(t.numel() * t.element_size() for t in w.values())}')
    print(f'build_seconds={build_seconds:.3f} batch64_seconds={forward_seconds:.3f}')
    print(f'stress_4480_seconds={stress_seconds:.3f} max_rss_kib={peak_kib}')
    print('No question accuracy was measured.')


if __name__ == '__main__':
    main()

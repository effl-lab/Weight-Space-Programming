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
                   'catalyst equilibrium', 'catalyst enthalpy',
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

    # Follow-up source review: these are source/weight contracts, not questions
    # with answer labels. No ranking or accuracy is used to choose model content.
    assert cid('fossilisation') == cid('fossilization') != cid('fossil')
    for surface in ('interrupts', 'interrupted', 'interrupting', 'interruption'):
        assert cid(surface) == cid('interrupt')
    assert cid('trapped') == cid('trap')
    stripped_next = [word for word in vocab if word != 'next']
    assert any('Distinguishing cue discarded' in msg for msg in
               audit_rules(stripped_next, aliases,
                           [('experiment next', 'hypothesis', '', 1.0)]))
    # The former comparisons differed only by word order and hence collided.
    former_density_cues = {'mass same volume greater', 'volume same mass greater'}
    assert not any(q in former_density_cues for q, _, _, _ in rules)
    assert meanings('same volume heavier') != meanings('same mass larger')
    for cue, expected in [('same volume heavier', 'increase density'),
                          ('same mass larger', 'decrease density')]:
        definitions = [p for q, p, _, _ in rules if q == cue]
        assert definitions and all(meanings(expected) <= meanings(p) for p in definitions)
    # Repeated tokens cannot identify two like poles/charges in a set-valued bag.
    lossy_cues = {'north north', 'south south',
                 'positive positive charge', 'negative negative charge'}
    assert not any(q in lossy_cues for q, _, _, _ in rules)
    assert not any(q == 'flame oxygen removed' for q, _, _, _ in rules)
    encoding_cues = {
        'same charges', 'opposite charges', 'positive negative',
        'positive identical', 'negative identical', 'same poles',
        'opposite poles', 'north identical', 'south identical', 'north south',
        'electromagnet off', 'wire disconnected', 'circuit gap',
        'oxygen removed', 'flame smother', 'like charges', 'unlike charges',
        'like poles', 'unlike poles', 'interrupted circuit',
        'disconnected circuit', 'circuit discontinuity', 'switch off',
        'solenoid off', 'coil disconnected', 'moss xylem', 'moss phloem',
    }
    encoding_rules = [rule for rule in rules if rule[0] in encoding_cues]
    assert len(encoding_rules) == len(encoding_cues)
    assert all(scale < 0 and len(meanings(cue)) == 2
               for cue, _, _, scale in encoding_rules)
    candidate = fixture(encoding_rules)
    assert torch.count_nonzero(candidate['A']) == 0
    for cue, positive, negative, _ in encoding_rules:
        pair = tuple(sorted(meanings(cue)))
        for target in meanings(positive) - set(pair):
            assert memory_value(candidate, pair, target) > 0, (cue, 'positive', target)
        for target in meanings(negative) - set(pair):
            assert memory_value(candidate, pair, target) < 0, (cue, 'negative', target)
    del candidate
    gc.collect()
    for cue, inhibited in [('blackhole', 'escape'), ('moss small', 'vascular'),
                           ('sterile', 'microorganisms'), ('bacteria sterile', 'contamination')]:
        definitions = [(p, n) for q, p, n, _ in rules if q == cue]
        assert definitions
        for positive, negative in definitions:
            assert meanings(inhibited) <= meanings(negative)
            assert not meanings(inhibited).intersection(meanings(positive))
    assert weights['A'][cid('blackhole'), cid('escape')] < 0
    assert build.lexicon is original_lexicon
    assert build.STOP == stop_snapshot

    # Definition-coverage review: inspect authored definitions and their direct
    # tensor support, never multiple-choice accuracy or answer ranks.
    bridge_definitions = {
        'ammeter', 'voltmeter', 'ohmmeter', 'adsorption', 'amylase', 'pepsin',
        'endosperm', 'cuticle', 'cytoskeleton', 'chitin', 'collagen', 'lignin',
        'melanin', 'nucleoid', 'pseudopod', 'sori', 'taproot', 'thorax',
        'abdomen', 'cephalothorax', 'ammonium', 'nitrate', 'urea', 'antigen',
        'neurotransmitter', 'drag', 'thrust', 'torque', 'deceleration',
        'trajectory', 'inelastic', 'resonance', 'photon', 'positron',
        'photovoltaic', 'transistor', 'dispersion', 'scattering', 'penumbra',
        'amorphous', 'malleable', 'ductile', 'hydrophobic', 'hydrophilic',
        'miscible', 'immiscible', 'enthalpy', 'vaporization', 'moraine',
        'erratic', 'striation', 'recrystallization', 'desalination',
        'leaching', 'biofuel', 'nonbiodegradable', 'pyroclastic', 'aurora',
        'nucleotide', 'nucleoside', 'intermolecular', 'intramolecular',
    }
    for cue in bridge_definitions:
        assert any(q == cue and scale > 0 for q, _, _, scale in rules), cue
        assert weights['idf'][cid(cue)] > 0, cue
        assert torch.count_nonzero(weights['A'][cid(cue)]) > 0, cue
    for a, b in [('adsorption', 'absorption'), ('ammeter', 'ampere'),
                 ('voltmeter', 'volt'), ('ohmmeter', 'ohm'),
                 ('nucleotide', 'nucleoside'), ('hydrophobic', 'hydrophilic'),
                 ('intermolecular', 'intramolecular'), ('malleable', 'ductile'),
                 ('transform', 'transformer'), ('vaporization', 'evaporation')]:
        assert cid(a) != cid(b), (a, b)
    for a, b in [('adsorbing', 'adsorption'), ('malleability', 'malleable'),
                 ('ductility', 'ductile'), ('desalination', 'desalinate'),
                 ('leaching', 'leach'), ('coalescence', 'coalesce'),
                 ('deflection', 'deflect'), ('rearrangement', 'rearrange'),
                 ('conversion', 'convert'), ('transformation', 'transform'),
                 ('specialisation', 'specialization'),
                 ('recrystallisation', 'recrystallization')]:
        assert cid(a) == cid(b), (a, b)
    for cue, target in [('ammeter', 'current'), ('voltmeter', 'voltage'),
                        ('ohmmeter', 'resistance'), ('adsorption', 'surface'),
                        ('nucleotide', 'phosphate'), ('amylase', 'starch'),
                        ('moraine', 'glacier'), ('photovoltaic', 'electric')]:
        assert weights['A'][cid(cue), cid(target)] > 0, (cue, target)
    for cue, target in [('nucleoside', 'phosphate'), ('amorphous', 'crystal'),
                        ('recrystallization', 'melting'), ('nonmagnetic', 'attract'),
                        ('nonbiodegradable', 'biodegradable')]:
        assert weights['A'][cid(cue), cid(target)] < 0, (cue, target)
    assert not any(q in {'water vapor energy', 'atomic mass'} for q, _, _, _ in rules)
    source_contracts = {
        'element atomic mass': ('isotope weighted average', 'proton neutron total'),
        'melting heat particles': ('disorder mobility', 'separate'),
        'freezing heat particles': ('ordered arrangement release', 'slow closer'),
        'vaporization latent': ('constant temperature potential', 'kinetic'),
        'phase change temperature': ('intermolecular potential', 'bond'),
    }
    for cue, (required, excluded) in source_contracts.items():
        definitions = [p for q, p, _, _ in rules if q == cue]
        assert definitions, cue
        for positive in definitions:
            assert meanings(required) <= meanings(positive), cue
            assert not meanings(excluded).intersection(meanings(positive)), cue

    # Conditional-source review: preserve each organism/process and its condition
    # as one exact pair. The fixture inspects weights, not question predictions.
    condition_cues = {
        'plant dark', 'plant night', 'photosynthesis darkness',
        'respiration darkness', 'leaf darkness', 'leaf covered',
        'respiration night', 'photosynthesis night', 'solar night',
        'predator removed', 'prey removed', 'predator increase',
        'predator decrease', 'prey increase', 'prey decrease',
        'producer decrease', 'decomposer removed', 'sunlight blocked',
        'plants removed', 'insects removed', 'rabbits removed',
        'hawks removed', 'wolves removed', 'deer increase', 'grass decrease',
        'prey scarce', 'predator scarce', 'producer removed',
        'herbivore removed', 'prey abundant', 'trees removed', 'vegetation remove',
    }
    condition_bags = {frozenset(meanings(cue)) for cue in condition_cues}
    condition_rules = [rule for rule in rules
                       if frozenset(meanings(rule[0])) in condition_bags]
    assert condition_cues <= {q for q, _, _, _ in condition_rules}
    assert all(scale < 0 and len(meanings(cue)) == 2
               for cue, _, _, scale in condition_rules)
    former_condition_cues = {
        'food web remove predator', 'food web remove prey',
        'food chain decomposer remove', 'food web sun blocked',
        'food web plants removed', 'food web insects removed',
        'food web rabbits removed', 'food web hawks removed',
        'food web wolves removed', 'food web deer increase',
        'food web grass decrease', 'food web prey scarce',
        'food web predator scarce', 'plant dark leaves',
        'plant respiration night', 'plant photosynthesis night',
        'hill trees removed',
    }
    assert not any(q in former_condition_cues for q, _, _, _ in rules)
    for cue, positive, negative, _ in condition_rules:
        # Do not collapse opposite population changes into one unbound target bag.
        assert not meanings('increase decrease') <= meanings(positive), cue
        if cue in {'plant dark', 'leaf darkness', 'leaf covered', 'sunlight blocked'}:
            assert cid('photosynthesis') in meanings(negative), cue
            assert cid('photosynthesis') not in meanings(positive), cue
        if cue in {'photosynthesis darkness', 'photosynthesis night'}:
            assert meanings('glucose production') <= meanings(negative), cue
            assert cid('respiration') not in meanings(positive), cue
        if cue == 'decomposer removed':
            assert cid('recycling') in meanings(negative)
            assert cid('accumulation') in meanings(positive)
    candidate = fixture(condition_rules)
    assert torch.count_nonzero(candidate['A']) == 0
    expected_rows = {row_of(*sorted(meanings(q))) for q, _, _, _ in condition_rules}
    occupied_rows = set(torch.nonzero(candidate['row_ids'][:, 0], as_tuple=False)
                        .flatten().tolist())
    assert occupied_rows == expected_rows
    for cue, positive, negative, _ in condition_rules:
        pair = tuple(sorted(meanings(cue)))
        for target in meanings(positive) - set(pair):
            assert memory_value(candidate, pair, target) > 0, (cue, 'positive', target)
        for target in meanings(negative) - set(pair):
            assert memory_value(candidate, pair, target) < 0, (cue, 'negative', target)
    del candidate
    gc.collect()
    assert build.lexicon is original_lexicon
    assert build.STOP == stop_snapshot

    # Protective properties: 'prevents X' must not positively recall X merely
    # because prevention and its object shared a positive target field.
    protective_contracts = {
        'hygiene': 'disease transmission',
        'disinfection': 'pathogens', 'disinfect': 'pathogens infection',
        'inertia trajectory': 'acceleration', 'malleable': 'fracture',
        'nonflammable': 'burning combustion',
        'impervious': 'penetration infiltration',
        'impermeable': 'penetration infiltration',
        'cartilage joint': 'friction', 'heart valve': 'backflow',
        'vein valve': 'backflow', 'canning food': 'contamination',
        'vacuum flask': 'conduction convection',
        'railroad gaps': 'buckling', 'sidewalk gaps': 'cracking',
        'material waterproof': 'penetration', 'engine cooling': 'overheating',
        'leech anticoagulant': 'clotting', 'aseptic technique': 'contamination',
        'sterile dish': 'contamination microorganisms',
        'epiglottis': 'aspiration choking',
        'handwashing soap': 'pathogens transmission',
        'sterilization equipment': 'microorganisms contamination',
        'waterproof design': 'leaks', 'technology sanitation': 'disease',
        'aseptic': 'contamination', 'anticoagulant': 'clotting',
        'waterproof': 'penetration leakage',
        'sanitation': 'disease contamination', 'disinfectant': 'pathogens',
        'epiglottis swallowing': 'aspiration choking',
        'vacuum insulation': 'conduction convection',
        'expansion gap': 'buckling cracking',
    }
    # A cue can also have a general mechanism rule (e.g. railroad gaps recall
    # expansion) that says nothing about the inhibited outcome. Keep that valid
    # evidence; require at least one signed contract and no opposing assertion.
    for cue, text in protective_contracts.items():
        definitions = [rule for rule in rules if rule[0] == cue]
        inhibited = meanings(text)
        assert any(inhibited <= meanings(n) for _, _, n, _ in definitions), cue
        assert all(not inhibited.intersection(meanings(p))
                   for _, p, _, _ in definitions), cue
    protective_rules = [rule for rule in rules if rule[0] in protective_contracts
                        and meanings(protective_contracts[rule[0]]) <= meanings(rule[2])]
    assert set(protective_contracts) == {q for q, _, _, _ in protective_rules}
    for cue, positive, negative, scale in protective_rules:
        inhibited = meanings(protective_contracts[cue])
        assert inhibited <= meanings(negative), cue
        assert not inhibited.intersection(meanings(positive)), cue
        if len(meanings(cue)) == 1:
            assert scale > 0, cue
            for target in inhibited:
                assert weights['A'][next(iter(meanings(cue))), target] < 0, (cue, target)
        else:
            assert len(meanings(cue)) == 2 and scale < 0, cue
    # Inertial motion does not imply that individual balanced forces are absent.
    for cue, positive, negative, _ in protective_rules:
        if cue == 'inertia trajectory':
            assert meanings('straight constant velocity equilibrium') <= meanings(positive)
            assert cid('force') not in meanings(negative)
            assert not meanings('net force').intersection(meanings(positive))
    assert not any(q == 'petri dish sterile' for q, _, _, _ in rules)
    protective_pairs = [rule for rule in protective_rules if rule[3] < 0]
    candidate = fixture(protective_pairs)
    assert torch.count_nonzero(candidate['A']) == 0
    expected_rows = {row_of(*sorted(meanings(q))) for q, _, _, _ in protective_pairs}
    occupied_rows = set(torch.nonzero(candidate['row_ids'][:, 0], as_tuple=False)
                        .flatten().tolist())
    assert occupied_rows == expected_rows
    for cue, positive, negative, _ in protective_pairs:
        pair = tuple(sorted(meanings(cue)))
        for target in meanings(positive) - set(pair):
            assert memory_value(candidate, pair, target) > 0, (cue, 'positive', target)
        for target in meanings(negative) - set(pair):
            assert memory_value(candidate, pair, target) < 0, (cue, 'negative', target)
    del candidate
    gc.collect()
    # A negative-only definition is valid evidence, not an empty rule. Its
    # symbolic not/except inverse must reverse the sign without fitting.
    negative_only = [rule for rule in rules if rule[0] == 'nonflammable']
    assert len(negative_only) == 1 and not meanings(negative_only[0][1])
    candidate = fixture(negative_only)
    assert candidate['A'][cid('nonflammable'), cid('combustion')] < 0
    assert torch.count_nonzero(candidate['A'] > 0) == 0
    for negator in ('not', 'except'):
        assert memory_value(candidate, (cid('nonflammable'), cid(negator)),
                            cid('combustion')) > 0
    del candidate
    gc.collect()
    assert build.lexicon is original_lexicon
    assert build.STOP == stop_snapshot

    # Astronomy review: calendar hemispheres, tidal alternatives, and orbital
    # reference periods are source contracts, not multiple-choice examples.
    months = meanings('march september june december')
    for cue in ('summer solstice', 'winter solstice', 'equinox spring', 'equinox autumn'):
        definitions = [p for q, p, _, _ in rules if q == cue]
        assert definitions, cue
        assert all(not months.intersection(meanings(p)) for p in definitions), cue
    calendar_contracts = {
        'northern spring': ('march equinox', 'september'),
        'southern spring': ('september equinox', 'march'),
        'northern autumn': ('september equinox', 'march'),
        'southern autumn': ('march equinox', 'september'),
    }
    for cue, (supported, inhibited) in calendar_contracts.items():
        definitions = [rule for rule in rules if rule[0] == cue]
        assert len(definitions) == 1, cue
        _, p, n, scale = definitions[0]
        assert meanings(supported) <= meanings(p), cue
        assert meanings(inhibited) <= meanings(n), cue
        assert scale < 0 and len(meanings(cue)) == 2, cue
    old_tidal_cues = {
        'moon full new tides', 'moon quarter tides', 'sun moon right angle',
        'sun moon aligned', 'spring tides alignment', 'neap tides perpendicular',
        'spring tide new full', 'neap tide quarter',
    }
    assert not any(q in old_tidal_cues for q, _, _, _ in rules)
    tidal_cues = {'spring tide', 'neap tide', 'full tide', 'new tide',
                  'quarter tide', 'tidal perpendicular', 'tidal alignment'}
    astronomy_pairs = [rule for rule in rules
                       if rule[0] in tidal_cues | set(calendar_contracts)
                       | {'perihelion period', 'aphelion period'}]
    assert all(scale < 0 and len(meanings(q)) == 2
               for q, _, _, scale in astronomy_pairs)
    candidate = fixture(astronomy_pairs)
    assert torch.count_nonzero(candidate['A']) == 0
    expected_rows = {row_of(*sorted(meanings(q))) for q, _, _, _ in astronomy_pairs}
    occupied_rows = set(torch.nonzero(candidate['row_ids'][:, 0], as_tuple=False)
                        .flatten().tolist())
    assert occupied_rows == expected_rows
    for cue, positive, negative, _ in astronomy_pairs:
        pair = tuple(sorted(meanings(cue)))
        for target in meanings(positive) - set(pair):
            assert memory_value(candidate, pair, target) > 0, (cue, 'positive', target)
        for target in meanings(negative) - set(pair):
            assert memory_value(candidate, pair, target) < 0, (cue, 'negative', target)
    del candidate
    gc.collect()
    for cue in ('orbit farther sun', 'orbit closer sun'):
        definitions = [p for q, p, _, _ in rules if q == cue]
        assert definitions and all(cid('year') not in meanings(p) for p in definitions)
    for cue, required in [('perihelion period', 'unchanged orbital year'),
                          ('aphelion period', 'unchanged orbital year'),
                          ('ion tail', 'solar wind charged gas'),
                          ('dust tail', 'radiation pressure curved particles'),
                          ('sidereal month', 'lunar orbit stars'),
                          ('synodic month', 'lunar phase cycle sun')]:
        definitions = [p for q, p, _, _ in rules if q == cue]
        assert definitions and all(meanings(required) <= meanings(p) for p in definitions), cue
    for a, b in [('rotation', 'revolution'), ('sidereal', 'synodic'),
                 ('perihelion', 'aphelion'), ('perigee', 'apogee'),
                 ('perihelion', 'perigee'), ('retrograde', 'prograde')]:
        assert cid(a) != cid(b), (a, b)
    for a, b in [('perihelia', 'perihelion'), ('aphelia', 'aphelion'),
                 ('precessing', 'precession'), ('librating', 'libration'),
                 ('synchronized', 'synchronous'), ('synchronisation', 'synchronous')]:
        assert cid(a) == cid(b), (a, b)
    for cue, target in [('sidereal', 'stars'), ('synodic', 'sun'),
                        ('obliquity', 'tilt'), ('precession', 'axis'),
                        ('libration', 'moon'), ('annular', 'ring')]:
        assert weights['A'][cid(cue), cid(target)] > 0, (cue, target)
    assert build.lexicon is original_lexicon
    assert build.STOP == stop_snapshot

    # Reaction review: kinetics must not become unqualified thermodynamic
    # claims, and a state must remain distinct from approaching that state.
    former_reaction_cues = {
        'catalyst reaction heat', 'catalyst equilibrium time',
        'catalyst reverse reaction', 'catalyst reaction pathway',
        'reaction catalyst energy', 'catalyst after reaction',
        'reaction speeds unchanged', 'reaction releases heat', 'reaction absorbs heat',
    }
    assert not any(q in former_reaction_cues for q, _, _, _ in rules)
    reaction_pair_cues = {
        'catalyst activation', 'catalyst equilibrium', 'catalyst enthalpy',
        'catalyst equilibration', 'catalyst reverse', 'catalyst pathway',
        'catalyst energy', 'catalyst regeneration', 'catalyst consumed',
        'reaction acceleration', 'heat release', 'heat absorption',
        'bond breaking', 'bond formation', 'bond dissociation',
        'reaction exothermic', 'reaction endothermic', 'exothermic activation',
        'catalyst intermediate', 'dynamic equilibrium', 'equilibrium reactions',
        'equilibrium rates', 'equilibrium net', 'open exchange',
        'closed exchange', 'isolated exchange',
    }
    reaction_pairs = [rule for rule in rules if rule[0] in reaction_pair_cues]
    assert reaction_pair_cues == {q for q, _, _, _ in reaction_pairs}
    assert all(scale < 0 and len(meanings(q)) == 2
               for q, _, _, scale in reaction_pairs)
    reaction_contracts = {
        'bond breaking': ('absorption endothermic', 'release'),
        'bond formation': ('release exothermic', 'absorption'),
        'bond dissociation': ('absorption endothermic', 'release'),
        'heat release': ('exothermic', 'endothermic'),
        'heat absorption': ('endothermic', 'exothermic'),
        'catalyst enthalpy': ('unchanged', 'increase decrease'),
        'catalyst equilibrium': ('unchanged position', 'shift'),
        'equilibrium rates': ('forward reverse equal', 'stopped'),
        'equilibrium reactions': ('forward reverse continuing', 'stopped'),
        'closed exchange': ('energy', 'matter'),
        'isolated exchange': ('blocked absent', 'matter energy'),
    }
    for cue, (supported, inhibited) in reaction_contracts.items():
        definitions = [rule for rule in reaction_pairs if rule[0] == cue]
        assert definitions, cue
        for _, p, n, _ in definitions:
            assert meanings(supported) <= meanings(p), cue
            assert meanings(inhibited) <= meanings(n), cue
    candidate = fixture(reaction_pairs)
    assert torch.count_nonzero(candidate['A']) == 0
    expected_rows = {row_of(*sorted(meanings(q))) for q, _, _, _ in reaction_pairs}
    occupied_rows = set(torch.nonzero(candidate['row_ids'][:, 0], as_tuple=False)
                        .flatten().tolist())
    assert occupied_rows == expected_rows
    for cue, positive, negative, _ in reaction_pairs:
        pair = tuple(sorted(meanings(cue)))
        for target in meanings(positive) - set(pair):
            assert memory_value(candidate, pair, target) > 0, (cue, 'positive', target)
        for target in meanings(negative) - set(pair):
            assert memory_value(candidate, pair, target) < 0, (cue, 'negative', target)
    del candidate
    gc.collect()
    for cue, p, n, _ in rules:
        if cue == 'equilibrium concentrations':
            assert meanings('constant stable') <= meanings(p)
            assert not meanings('equal').intersection(meanings(p + ' ' + n))
        if cue == 'enthalpy':
            assert meanings('state function internal energy pressure volume') <= meanings(p)
            assert cid('heat') not in meanings(p)
        if cue in {'closed system', 'isolated system'}:
            assert not meanings('matter energy').intersection(meanings(n))
    for a, b in [('equilibration', 'equilibrium'), ('catalysis', 'catalyst'),
                 ('dissociation', 'dissolution'), ('reversible', 'irreversible')]:
        assert cid(a) != cid(b), (a, b)
    for a, b in [('equilibrating', 'equilibration'), ('dissociating', 'dissociation'),
                 ('regenerated', 'regeneration'), ('reversibility', 'reversible')]:
        assert cid(a) == cid(b), (a, b)
    for cue, target in [('catalysis', 'activation'), ('equilibration', 'equilibrium'),
                        ('dissociation', 'separation')]:
        assert weights['A'][cid(cue), cid(target)] > 0, (cue, target)
    assert build.lexicon is original_lexicon
    assert build.STOP == stop_snapshot

    # Genetics review: avoid turning common introductory cases into universal
    # claims about expression, mutation effects, or transmission routes.
    genetic_source_contracts = {
        'heterozygous': ('different alleles hybrid', 'dominant'),
        'recessive': ('allele masked', 'two homozygous'),
        'cloning plant cutting': ('asexual genotype', 'traits'),
        'mutation effect': ('neutral beneficial harmful', 'protein trait'),
        'somatic mutation': ('body cell mitotic lineage', ''),
        'mutation skin': ('somatic acquired body cell', ''),
    }
    for cue, (required, excluded) in genetic_source_contracts.items():
        definitions = [rule for rule in rules if rule[0] == cue]
        assert definitions, cue
        for _, p, n, _ in definitions:
            assert meanings(required) <= meanings(p), cue
            assert not meanings(excluded).intersection(meanings(p)), cue
            if cue in {'somatic mutation', 'mutation skin'}:
                assert not meanings('inherited offspring').intersection(meanings(n)), cue
    assert not any(q in {'adaptation sudden', 'adaptation acquired',
                         'muscle exercise inheritance'} for q, _, _, _ in rules)
    genetics_pair_cues = {
        'complete dominance', 'incomplete dominance', 'recessive homozygous',
        'recessive hemizygous', 'rare dominant', 'individual adjustment',
        'acquired change', 'exercise muscle',
    }
    genetics_pairs = [rule for rule in rules if rule[0] in genetics_pair_cues]
    assert genetics_pair_cues == {q for q, _, _, _ in genetics_pairs}
    assert all(scale < 0 and len(meanings(q)) == 2
               for q, _, _, scale in genetics_pairs)
    candidate = fixture(genetics_pairs)
    assert torch.count_nonzero(candidate['A']) == 0
    expected_rows = {row_of(*sorted(meanings(q))) for q, _, _, _ in genetics_pairs}
    occupied_rows = set(torch.nonzero(candidate['row_ids'][:, 0], as_tuple=False)
                        .flatten().tolist())
    assert occupied_rows == expected_rows
    for cue, positive, negative, _ in genetics_pairs:
        pair = tuple(sorted(meanings(cue)))
        for target in meanings(positive) - set(pair):
            assert memory_value(candidate, pair, target) > 0, (cue, 'positive', target)
        for target in meanings(negative) - set(pair):
            assert memory_value(candidate, pair, target) < 0, (cue, 'negative', target)
    del candidate
    gc.collect()
    for a, b in [('acclimatization', 'adaptation'), ('genotype', 'phenotype'),
                 ('codominance', 'dominance'), ('hemizygous', 'homozygous'),
                 ('hemizygous', 'heterozygous'), ('germline', 'somatic'),
                 ('pleiotropy', 'polygenic'), ('penetrance', 'expressivity'),
                 ('synonymous', 'missense'), ('missense', 'nonsense')]:
        assert cid(a) != cid(b), (a, b)
    for a, b in [('acclimatisation', 'acclimatization'), ('acclimation', 'acclimatization'),
                 ('codominant', 'codominance'), ('hemizygosity', 'hemizygous'),
                 ('pleiotropic', 'pleiotropy'), ('loci', 'locus'),
                 ('phenotypic', 'phenotype'), ('clonal', 'clone')]:
        assert cid(a) == cid(b), (a, b)
    for cue, target in [('acclimatization', 'individual'), ('hemizygous', 'single'),
                        ('germline', 'gametes'), ('locus', 'chromosome'),
                        ('penetrance', 'fraction'), ('expressivity', 'degree'),
                        ('codominance', 'heterozygous'), ('frameshift', 'reading')]:
        assert weights['A'][cid(cue), cid(target)] > 0, (cue, target)
    # Opposite role statements must not collapse to identical target bags.
    pleiotropy = [meanings(p) for q, p, _, _ in rules if q == 'pleiotropy']
    polygenic = [meanings(p) for q, p, _, _ in rules if q == 'polygenic']
    assert len(pleiotropy) == len(polygenic) == 1
    assert pleiotropy[0] != polygenic[0]
    assert meanings('single diverse effects') <= pleiotropy[0]
    assert meanings('multiple combined trait') <= polygenic[0]
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

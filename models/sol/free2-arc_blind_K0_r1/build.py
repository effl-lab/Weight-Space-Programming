"""Deterministically compile authored science knowledge into tensor weights.
No optimizer, fitting, question dataset, external corpus, or pretrained weights.
"""
from pathlib import Path
import re
import torch
from extension import inherited_records, extra_text, feature_phrases, compile_logic

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")
STOP = set("""a an the this that these those which what who whom whose where when why how
is are was were be been being am become becomes became becoming do does did doing done have has had having
of to in on at by for from as into onto upon about with within through
and or but than then so because if unless until while whether both either
it its it's they their them he she his her you your we our us i me my
one some such each every another other others also just very much many
can could may might will would shall should must
following shown show shows according statement statements describes describe
best most likely usually generally typically often below above given
use uses used using example examples represents represent represented
question answer choose choice choices correct appropriate
""".split())
# Clause boundaries are ignored as unigram evidence, but they must not be
# skipped when joining phrases or attaching a change to a scientific quantity.
BOUNDARY = set('and or but as than then because if unless until while whether whereas although yet however'.split())
STOP.update(BOUNDARY)


def source_lines(name):
    for raw in (ROOT / name).read_text().splitlines():
        line = raw.split('#')[0].strip()
        if line:
            yield line


def authored():
    records = []
    for path in sorted(ROOT.glob('*.tsv')):
        if path.name != 'ranks.tsv':
            for line in source_lines(path.name):
                cells = [s.strip() for s in line.split('|')]
                records.append((cells[0], cells[1], float(cells[2]) if len(cells) > 2 else 1.0))
    records.extend(inherited_records())
    return records


def lexical_source():
    aliases = {}
    originals = set(STOP)
    for line in source_lines('aliases.txt'):
        group = TOKEN.findall(line.lower())
        for word in group:
            aliases.setdefault(word, group[0])
            originals.add(word)
    for word in aliases:
        target = aliases[word]
        for step in range(8):
            target = aliases.get(target, target)
        aliases[word] = target
    for left, right, strength in authored():
        originals.update(TOKEN.findall(left.lower()))
        originals.update(TOKEN.findall(right.lower()))
    for name in ('phrases.txt', 'extra_words.txt', 'parser_words.txt', 'ranks.tsv', 'opposites.txt', 'overrides.txt'):
        for line in source_lines(name):
            originals.update(TOKEN.findall(line.lower()))
    for name in ('concepts.txt', 'quantities.txt'):
        for line in source_lines(name):
            originals.update(TOKEN.findall(line.split('|', 1)[1].lower()))
    for text in extra_text():
        originals.update(TOKEN.findall(text.lower()))
    originals.update('increase decrease high low constant unchanged same steady remain remains stay stays'.split())
    canonical = {}
    for word in sorted(originals, key=lambda s: (len(s), s)):
        target = aliases.get(word, word)
        if word not in aliases and len(word) > 3:
            candidates = []
            if word.endswith('ies'):
                candidates.append(word[:-3] + 'y')
            if word.endswith('es'):
                candidates.extend((word[:-1], word[:-2]))
            if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
                candidates.append(word[:-1])
            if word.endswith('ing'):
                candidates.extend((word[:-3], word[:-3] + 'e', word[:-4]))
            if word.endswith('ed'):
                candidates.extend((word[:-1], word[:-2], word[:-3]))
            for candidate in candidates:
                if candidate in originals and len(candidate) >= 3:
                    target = canonical.get(candidate, aliases.get(candidate, candidate))
                    break
        canonical[word] = target
    for word in STOP:
        canonical[word] = ''
    canonical.update({'weather': 'weather', 'weathering': 'weathering', 'weathered': 'weathering',
                      'laboratory': 'lab', 'laboratories': 'lab'})
    result = canonical.copy()
    for word, target in canonical.items():
        if target and word.isalpha() and len(word) > 2:
            forms = [word + 's', word + 'es', word + 'ed', word + 'ing']
            if word.endswith('e'):
                forms += [word + 'd', word[:-1] + 'ing']
            if word.endswith('y'):
                forms += [word[:-1] + 'ies', word[:-1] + 'ied']
            if len(word) > 3 and word[-1] not in 'aeiouwxy' and word[-2] in 'aeiou':
                forms += [word + word[-1] + 'ed', word + word[-1] + 'ing']
            for form in forms:
                result.setdefault(form, target)
    for number in range(10001):
        result.setdefault(str(number), '')
    return result


def compile_features(vocab, lexicon):
    """Share exact feature semantics between authored text and tensor inference."""
    concepts = sorted(set(v for v in lexicon.values() if v))
    ci = {word: i + 1 for i, word in enumerate(concepts)}
    base = len(concepts) + 1
    phrases = {}
    single = {}
    feature_count = base

    def sequence(text):
        return tuple(ci[lexicon[word]] for word in TOKEN.findall(text.lower()) if lexicon.get(word, ''))

    def register(seq, target=None):
        nonlocal feature_count
        if target is None:
            if len(seq) == 1:
                target = single.get(seq[0], seq[0])
            elif seq in phrases:
                target = phrases[seq]
            else:
                target = feature_count
                feature_count += 1
        if len(seq) == 1:
            single[seq[0]] = target
        else:
            phrases[seq] = target
        return target

    def bind(alternatives):
        # Keep equivalence groups transitive when a new group overlaps an
        # existing phrase. Reuse existing features rather than leaving stale
        # aliases pointing at an abandoned feature id.
        existing = []
        for seq in alternatives:
            if len(seq) == 1:
                existing.append(single.get(seq[0], seq[0]))
            elif seq in phrases:
                existing.append(phrases[seq])
        target = existing[0] if existing else register(alternatives[0])
        for old in set(existing) - {target}:
            for phrase, feature_id in list(phrases.items()):
                if feature_id == old:
                    phrases[phrase] = target
            for token, feature_id in list(single.items()):
                if feature_id == old:
                    single[token] = target
            if old < base:
                single[old] = target
        for seq in alternatives:
            register(seq, target)
        return target

    for line in source_lines('phrases.txt'):
        seq = sequence(line)
        if 2 <= len(seq) <= 4:
            register(seq)
    for line in source_lines('concepts.txt'):
        alternatives = [sequence(s.strip()) for s in line.split('|', 1)[1].split(';')]
        assert all(1 <= len(s) <= 4 for s in alternatives), line
        bind(alternatives)
    quantity_patterns = []
    for line in source_lines('quantities.txt'):
        nouns = [s.strip() for s in line.split('|', 1)[1].split(';')]
        states = []
        for modifiers in (('increase', 'high'), ('decrease', 'low'), ('constant', 'unchanged', 'same', 'steady')):
            patterns = []
            for noun in nouns:
                for modifier in modifiers:
                    patterns.extend((sequence(noun + ' ' + modifier), sequence(modifier + ' ' + noun)))
                if 'constant' in modifiers:
                    patterns.extend((sequence(noun + ' remains constant'), sequence(noun + ' stays constant'), sequence(noun + ' remains same')))
                    patterns.extend((sequence(noun + ' not change'), sequence('no change ' + noun), sequence('not change ' + noun)))
            assert all(2 <= len(s) <= 4 for s in patterns), line
            bind(patterns)
            states.append(patterns[0])
        quantity_patterns.append(states)
    for line in source_lines('ranks.tsv'):
        entries = line.split('|')[3]
        for entry in entries.split(';'):
            register(sequence(entry.split(':')[0].strip()))
    for name in ('opposites.txt', 'overrides.txt'):
        for line in source_lines(name):
            for text in line.split('|'):
                register(sequence(text.strip()))

    for text in feature_phrases():
        seq = sequence(text)
        assert 1 <= len(seq) <= 4, (text, seq)
        register(seq)

    def selector(text):
        seq = sequence(text)
        if len(seq) == 1:
            return single.get(seq[0], seq[0])
        return phrases[seq]

    override_pairs = [(selector(a.strip()), selector(b.strip()))
                      for a, b in (line.split('|') for line in source_lines('overrides.txt'))]
    for source, destination in list(override_pairs):
        for raw_feature, semantic_feature in single.items():
            if semantic_feature == destination and raw_feature != destination:
                override_pairs.append((source, raw_feature))

    quantity_ids = [[phrases[seq] for seq in states] for states in quantity_patterns]

    def features(text):
        seq = tuple(ci.get(lexicon.get(word, ''), base) for word in TOKEN.findall(text.lower()) if word not in STOP or word in BOUNDARY)
        active = {i for i in seq if i < base}
        active.update(single.get(i, 0) for i in seq)
        active.discard(0)
        for width in range(2, 5):
            for start in range(len(seq) - width + 1):
                phrase_id = phrases.get(seq[start:start + width], 0)
                if phrase_id:
                    active.add(phrase_id)
        for specific, embedded in override_pairs:
            if specific in active:
                active.discard(embedded)
        ambiguous = set()
        for states in quantity_ids:
            if sum(feature_id in active for feature_id in states) > 1:
                ambiguous.update(states)
        active.difference_update(ambiguous)
        return active

    codes = {}
    for seq, target in phrases.items():
        code = 0
        for part in seq:
            code = code * base + part
        codes[code] = target
    ordered = sorted(codes)
    quantity_state = torch.tensor(quantity_ids, dtype=torch.int64)
    assert all(len(set(row)) == 3 for row in quantity_state.tolist()), 'Quantity states must remain distinct'
    single_table = torch.zeros(base, dtype=torch.int64)
    for src, dst in single.items():
        single_table[src] = dst
    lex = torch.tensor([0, 0] + [ci.get(lexicon.get(word, ''), 0) for word in vocab], dtype=torch.int64)
    skip = torch.tensor([1, 0] + [int(word in STOP and word not in BOUNDARY) for word in vocab], dtype=torch.int64)
    tables = {
        'lex': lex, 'skip': skip, 'base': torch.tensor(base, dtype=torch.int64),
        'single_feature': single_table, 'quantity_state': quantity_state,
        'phrase_codes': torch.tensor(ordered + [base ** 4 + base ** 3 + base ** 2 + base + 1], dtype=torch.int64),
        'phrase_ids': torch.tensor([codes[c] for c in ordered] + [0], dtype=torch.int64),
        'override_src': torch.tensor([a for a, b in override_pairs], dtype=torch.int64),
        'override_dst': torch.tensor([b for a, b in override_pairs], dtype=torch.int64),
    }
    return tables, feature_count, features, selector, sequence


def build():
    records = authored()
    lexicon = lexical_source()
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    tables, count, features, selector, sequence = compile_features(vocab, lexicon)
    key = torch.zeros((len(records), count), dtype=torch.float32)
    value = torch.zeros_like(key)
    for i, (left, right, strength) in enumerate(records):
        key[i, list(features(left))] = 1
        value[i, list(features(right))] = 1
    strength = torch.tensor([r[2] for r in records], dtype=torch.float32)
    frequency = ((key + value) > 0).sum(0).to(torch.float32)
    idf = (torch.log((len(records) + 6.0) / (frequency + 3.0)) + 0.25).clamp(0.7, 5.5)
    idf[tables['single_feature'].shape[0]:] *= 1.35
    idf[0] = 0
    keymass = (key * idf).sum(-1).clamp_min(1)
    valuemass = (value * idf).sum(-1).clamp_min(1)
    # Opposite attributes are a penalty only when the fact unambiguously favors
    # the other member of a pair. A fact mentioning both incurs no such penalty.
    opposition = set()
    for line in source_lines('opposites.txt'):
        left, right = (selector(s.strip()) for s in line.split('|'))
        if left != right:
            opposition.add((left, right))
            opposition.add((right, left))
    for higher, lower, unchanged in tables['quantity_state'].tolist():
        for left, right in ((higher, lower), (higher, unchanged), (lower, unchanged)):
            opposition.add((left, right))
            opposition.add((right, left))
    opposition = sorted(opposition)
    clash_src = torch.tensor([a for a, b in opposition], dtype=torch.int64)
    clash_dst = torch.tensor([b for a, b in opposition], dtype=torch.int64)
    value_clash = (value[:, clash_dst] - value[:, clash_src]).clamp_min(0).t().contiguous()
    key_clash = (key[:, clash_dst] - key[:, clash_src]).clamp_min(0).t().contiguous()

    token_index = {word: i + 2 for i, word in enumerate(vocab)}
    def word_table(entries):
        out = torch.zeros(len(vocab) + 2)
        for words, number in entries:
            for word in words.split():
                assert word in token_index, ('Missing parser vocabulary', word)
                out[token_index[word]] = number
        return out
    ask = word_table([('which what', 2), ('why how where when', 1)])
    neg = word_table([('except excluding incorrect false untrue unlikely improbable least', 1)])
    least = word_table([('least', 1)])
    not_word = word_table([("not isn't aren't wasn't weren't doesn't don't didn't cannot can't never", 1)])
    scope_boundary = word_table([('that when if because why whether shows show showing indicates indicate indicating evidence proves prove explains explain happens happen', 1)])
    cubic = word_table([('cubic cubed', 1)])
    unit_bridge = word_table([('cubic of', 1)])
    per_word = word_table([('per', 1)])
    atom_words = torch.stack([word_table([(text, 1)]) for text in (
        'atomic', 'mass', 'number', 'proton protons', 'neutron neutrons', 'electron electrons'
    )], dim=-1)
    atom_context = torch.zeros(count)
    atom_context[list(features('atom atomic nucleus isotope proton neutron electron'))] = 1
    ion_word = word_table([('ion ions cation cations anion anions charge charged positive negative', 1)])
    neutral_word = word_table([('neutral uncharged', 1)])
    electron_change = word_table([('gain gains gained lose loses lost add adds added remove removes removed emit emits emitted emission decay decays decayed fission fusion', 1)])
    atom_bridge = word_table([('of is are was were be been has have had a an the equal equals exactly total number contains containing count value', 1)])
    cycle_unit = word_table([('cycle cycles oscillation oscillations vibration vibrations times', 1)])
    each_word = word_table([('each every single one 1', 1)])
    period_word = word_table([('period periods', 1)])
    period_bridge = word_table([('of is are was were be been has have had a an the in for each every single one complete full wave waves sound sounds signal signals cycle cycles oscillation oscillations vibration vibrations lasts last takes take equals exactly', 1)])
    work_forbid = word_table([('angle angles degree degrees perpendicular opposite circular centripetal', 1)])
    how_word = word_table([('how', 1)])
    long_word = word_table([('long', 1)])
    wavelength_word = word_table([('wavelength wavelengths', 1)])
    displacement_word = word_table([('displacement displace displaced displaces submerge submerged immersion immersed', 1)])
    water_word = word_table([('water', 1)])
    cylinder_word = word_table([('cylinder cylinders', 1)])
    object_word = word_table([('object objects rock rocks stone stones sample samples block blocks marble marbles pebble pebbles cube cubes', 1)])
    displacement_forbid = word_table([('float floats floating pour pours poured pouring spill spills spilled evaporate evaporates evaporation', 1)])
    volume_request_word = word_table([('volume volumes displacement displaced displace displaces', 1)])
    density_request_word = word_table([('density densities dense denser', 1)])
    volume_query = torch.zeros(count)
    volume_query[list(features('volume displacement displaced'))] = 1
    density_query = torch.zeros(count)
    density_query[list(features('density'))] = 1

    rules = [[s.strip() for s in line.split('|')] for line in source_lines('ranks.tsv')]
    rt = torch.zeros((len(rules), count))
    ru = torch.zeros_like(rt)
    rd = torch.zeros_like(rt)
    width = max(len(row[3].split(';')) for row in rules)
    entity = torch.zeros((len(rules), width), dtype=torch.int64)
    rank_number = torch.zeros((len(rules), width))
    specificity = torch.zeros_like(rank_number)
    for i, (trigger, ascending, descending, entries) in enumerate(rules):
        rt[i, list(features(trigger))] = 1
        ru[i, list(features(ascending))] = 1
        rd[i, list(features(descending))] = 1
        parsed = [entry.split(':') for entry in entries.split(';')]
        denom = max(abs(float(number)) for text, number in parsed) or 1
        for j, (text, number) in enumerate(parsed):
            entity[i, j] = selector(text.strip())
            rank_number[i, j] = float(number) / denom
            specificity[i, j] = len(sequence(text))

    number = torch.full((len(vocab) + 2,), -1.0)
    names = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
    number_words = {name: i for i, name in enumerate(names)}
    number_words.update({'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000})
    for word, idx in token_index.items():
        if word.isdigit():
            number[idx] = float(word)
        elif word in number_words:
            number[idx] = number_words[word]
    # Units: g, mL, m, s, N, J, W, V, A, ohm, Hz. Ordinary length units
    # become volumes only with an explicit cubic/cubed marker in the input.
    units = torch.zeros((len(vocab) + 2, 11))
    unit_groups = [
        [('gram grams g', 1), ('kilogram kilograms kg', 1000)],
        [('milliliter milliliters millilitre millilitres ml cm3 cc', 1), ('liter liters litre litres l', 1000), ('m3', 1000000)],
        [('meter meters metre metres m', 1), ('centimeter centimeters centimetre centimetres cm', 0.01), ('millimeter millimeters millimetre millimetres mm', 0.001), ('kilometer kilometers kilometre kilometres km', 1000)],
        [('second seconds sec s', 1), ('minute minutes min', 60), ('hour hours hr h', 3600)],
        [('newton newtons n', 1)], [('joule joules j', 1)],
        [('watt watts w', 1), ('kilowatt kilowatts kw', 1000)],
        [('volt volts v', 1)], [('amp ampere amperes amps a', 1)],
        [('ohm ohms', 1)], [('hertz hz', 1)],
    ]
    for kind, groups in enumerate(unit_groups):
        for names, scale in groups:
            for word in names.split():
                assert word in token_index, ('Missing unit vocabulary', word)
                units[token_index[word], kind] = scale
    equations = [
        ('density', 0, 1, -1), ('speed velocity', 2, 3, -1),
        ('work energy', 4, 2, 1), ('power', 5, 3, -1),
        ('voltage', 8, 9, 1), ('current', 7, 9, -1),
        ('resistance', 7, 8, -1), ('period time', 11, 10, -1),
        ('frequency', 11, 3, -1), ('time', 5, 6, -1), ('energy work', 6, 3, 1),
    ]
    math_trigger = torch.zeros((len(equations), count))
    for i, (text, left, right, op) in enumerate(equations):
        math_trigger[i, list(features(text))] = 1
    math_left = torch.tensor([e[1] for e in equations], dtype=torch.int64)
    math_right = torch.tensor([e[2] for e in equations], dtype=torch.int64)
    math_op = torch.tensor([e[3] for e in equations], dtype=torch.float32)
    math_output = torch.zeros((len(equations), 11))
    math_output[0, 0], math_output[0, 1] = 1, -1
    math_output[1, 2], math_output[1, 3] = 1, -1
    for row, kind in enumerate((5, 6, 7, 8, 9, 3, 10, 3, 5), start=2):
        math_output[row, kind] = 1
    # Each equation accepts only dimensionally matching output units. Alternate
    # forms are authored explicitly, e.g. joules or newton-meters for work.
    math_unit_power = torch.zeros((len(equations), 3, 11))
    math_unit_valid = torch.zeros((len(equations), 3))
    math_unit_power[:, 0] = math_output
    math_unit_valid[:, 0] = 1
    alternatives = {
        2: ({4: 1, 2: 1}, {6: 1, 3: 1}),
        3: ({5: 1, 3: -1}, {7: 1, 8: 1}),
        4: ({8: 1, 9: 1},),
        5: ({7: 1, 9: -1},),
        6: ({7: 1, 8: -1},),
        9: ({5: 1, 6: -1},),
        10: ({4: 1, 2: 1}, {6: 1, 3: 1}),
    }
    for equation, patterns in alternatives.items():
        for j, pattern in enumerate(patterns, start=1):
            math_unit_valid[equation, j] = 1
            for kind, exponent in pattern.items():
                math_unit_power[equation, j, kind] = exponent
    cycle_period_rule = torch.zeros(len(equations))
    cycle_frequency_rule = torch.zeros(len(equations))
    wave_speed_rule = torch.zeros(len(equations))
    duration_rule = torch.zeros(len(equations))
    force_work_rule = torch.zeros(len(equations))
    force_work_rule[2] = 1
    cycle_period_rule[7], cycle_frequency_rule[8], wave_speed_rule[1] = 1, 1, 1
    duration_rule[7], duration_rule[9] = 1, 1
    tables.update({
        'idf': idf, 'key': key, 'value': value,
        'keymass': keymass, 'valuemass': valuemass, 'strength': strength,
        'clash_src': clash_src, 'clash_dst': clash_dst, 'value_clash': value_clash, 'key_clash': key_clash,
        'neg': neg, 'least': least, 'not_word': not_word, 'scope_boundary': scope_boundary, 'ask': ask,
        'rank_trigger': rt, 'rank_up': ru, 'rank_down': rd,
        'rank_entity': entity, 'rank_number': rank_number, 'rank_specificity': specificity,
        'number': number, 'units': units, 'cubic': cubic, 'unit_bridge': unit_bridge, 'per_word': per_word,
        'atom_words': atom_words, 'atom_context': atom_context, 'ion_word': ion_word, 'neutral_word': neutral_word, 'electron_change': electron_change,
        'atom_targets': torch.tensor([0, 1, 1, 2, 3]), 'atom_bridge': atom_bridge,
        'cycle_unit': cycle_unit, 'each_word': each_word, 'period_word': period_word, 'period_bridge': period_bridge, 'wavelength_word': wavelength_word,
        'work_forbid': work_forbid, 'force_work_rule': force_work_rule,
        'cycle_period_rule': cycle_period_rule, 'cycle_frequency_rule': cycle_frequency_rule, 'wave_speed_rule': wave_speed_rule,
        'how_word': how_word, 'long_word': long_word, 'duration_rule': duration_rule,
        'displacement_word': displacement_word, 'water_word': water_word, 'cylinder_word': cylinder_word,
        'object_word': object_word, 'displacement_forbid': displacement_forbid,
        'volume_query': volume_query, 'density_query': density_query,
        'volume_request_word': volume_request_word, 'density_request_word': density_request_word,
        'math_trigger': math_trigger, 'math_left': math_left, 'math_right': math_right,
        'math_op': math_op, 'math_output': math_output,
        'math_unit_power': math_unit_power, 'math_unit_valid': math_unit_valid,
        'knobs': torch.tensor([0.8, 0.35, 0.35, 0.18, 0.08, 2.0, 1.25, 0.35, 30.0, 8.0, 0.75]),
    })
    tables.update(compile_logic(selector, count))
    return tables

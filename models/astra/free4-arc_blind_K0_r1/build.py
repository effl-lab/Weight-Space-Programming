"""Compile the authored lexicon and propositions into fixed tensor weights.

There is no fitting, dataset loading, pretrained component, or randomized weight.
Run `python build.py --vocab` once after editing the authored text to regenerate
vocab.txt. build() itself reads only the model's authored files.
"""
from pathlib import Path
import itertools
import math
import re
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCES = ('biology.txt', 'physics.txt', 'chemistry.txt', 'earth_space.txt', 'inquiry.txt', 'extra.txt', 'concepts.txt', 'extensions.txt', 'scenarios.txt', 'review.txt', 'practical_science.txt', 'life_applications.txt', 'earth_applications.txt', 'matter_applications.txt', 'inquiry_applications.txt', 'mechanisms_physics.txt', 'mechanisms_life.txt', 'mechanisms_earth.txt', 'mechanisms_matter.txt', 'mechanisms_ecology_inquiry.txt')
AUXILIARY = ('relations.txt', 'taxonomy.txt', 'properties.txt', 'comparisons.txt', 'causal.txt', 'opposites.txt', 'taxonomy_extra.txt', 'entities.txt', 'experiment_grammar.txt', 'ecology_grammar.txt', 'separation_grammar.txt', 'separation_materials.txt', 'separation_methods.txt', 'separation_rules.txt', 'formula_conditions.txt', 'force_grammar.txt', 'scope_grammar.txt', 'class_grammar.txt', 'relation_grammar.txt', 'interaction_grammar.txt')
TOKEN = re.compile(r"[a-z0-9']+")
STOP = set('a an the this that these those is are was were be being been of to for from by with as at in on into onto through than then and or but so it its their they them he she his her our we you your i me us which what where when why how who whom whose would could should may might can will shall do does did done has have had having there here such following below above example examples each any some several certain particular another other during between within about approximately usually generally typically likely best most true'.split())
# These weak words are useful in some relational rules, but should not dominate.
WEAK = set('energy water plant animal cell gas food earth sun heat temperature mass volume light force motion increase decrease high low more less same different only always all make produce help need allow process function purpose structure system part source cause result change body object substance material amount number rate'.split())
NEG_WORDS = set("no not never without cannot can't isn't aren't doesn't don't won't wasn't weren't".split())
NUMBER_WORDS = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19, 'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'sixty':60, 'seventy':70, 'eighty':80, 'ninety':90, 'hundred':100, 'thousand':1000}
EXTRA = '''reason must just merely not except least avoid false incorrect no without never neither unable cannot none zero one two three four five six seven eight nine ten eleven twelve twenty thirty sixty hundred thousand cm3 m3 cubic squared per milligram milligrams mg ampere amperes amp amps ohm ohms pascal pascals pa kelvin k mph seconds minutes hours meters kilometers milliliters liters grams kilograms newtons joules watts volts both neither either always only all constant unchanged remains remain unchanged rises falls increases decreases increasing decreasing greater smaller higher lower fewer times divided twice half third fourth double triple maximum minimum greatest smallest weakest strongest safest unsafe harmful harmless faster slower fastest slowest northward southward eastward westward rise rises rising set sets setting sunrise sunset dawn dusk cells cellular none net 365 366 273 212 32 24 27 29 23 10 100 0'''


def authored():
    lex = []
    aliases = {}
    words = set(TOKEN.findall(EXTRA)) | STOP | NEG_WORDS
    for line in ('\n'.join((ROOT / name).read_text() for name in ('lexicon.txt', 'lexicon_extra.txt', 'lexicon_new.txt', 'lexicon_mechanisms.txt'))).splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        entries = s.split()
        root = entries[0]
        lex.append((root, entries))
        for entry in entries:
            aliases[entry] = root
            words.update(TOKEN.findall(entry.replace('_', ' ')))
    lines = []
    for name in SOURCES:
        for line in (ROOT / name).read_text().splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            lines.append(s)
            words.update(TOKEN.findall(s.replace('_', ' ')))
    for name in AUXILIARY:
        for line in (ROOT / name).read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                # Reference values are weights, not extra numeric vocabulary.
                content = line.rsplit('|', 1)[0] if name == 'properties.txt' else line
                if name in ('experiment_grammar.txt', 'ecology_grammar.txt', 'separation_grammar.txt', 'scope_grammar.txt', 'class_grammar.txt', 'relation_grammar.txt', 'interaction_grammar.txt'):
                    content = line.split('|', 1)[1]
                if name == 'formula_conditions.txt':
                    content = line.split('|', 2)[2]
                words.update(TOKEN.findall(content.replace('_', ' ')))
    words.update(NUMBER_WORDS)
    words.update(str(i) for i in range(1001))
    words.update(f'{i:02d}' for i in range(100))
    words.update(f'{i:03d}' for i in range(1000))
    words.update(str(i) for i in range(1100, 10100, 100))
    words.update(('1440', '3600', '86400', '100000', '1000000', 'm', 's', 'cubed', 'cubic'))
    # Conservative deterministic inflection generation is lexicon engineering,
    # not learned morphology. Existing explicitly authored meanings win.
    bases = sorted(words)
    for word in bases:
        if len(word) < 3 or not word.isalpha() or word in STOP:
            continue
        root = aliases.get(word, word)
        forms = [word + 's']
        if word.endswith(('s', 'x', 'ch', 'sh', 'z')):
            forms.append(word + 'es')
        if word.endswith('y') and word[-2] not in 'aeiou':
            forms.append(word[:-1] + 'ies')
        if not word.endswith(('ed', 'ing', 's')):
            forms += [word + 'ed', word + 'ing']
            if word.endswith('e'):
                forms += [word + 'd', word[:-1] + 'ing']
            if len(word) > 3 and word[-1] not in 'aeiouwxy' and word[-2] in 'aeiou' and word[-3] not in 'aeiou':
                forms += [word + word[-1] + 'ed', word + word[-1] + 'ing']
        for form in forms:
            aliases.setdefault(form, root)
            words.add(form)
    # Several irregular comparative forms and common lexical variants.
    manual = {
        'unchanged':'same', 'constant':'same', 'remains':'remain',
        'greater':'high', 'greatest':'high', 'maximum':'high',
        'minimum':'low', 'fewer':'less',
        'stronger':'strong', 'strongest':'strong', 'weaker':'weak', 'weakest':'weak',
        'safest':'safe', 'safety':'safe', 'harmful':'unsafe', 'harmless':'safe',
        'quickest':'fast', 'rapidly':'fast', 'slowly':'slow',
        'h2o':'water', 'o2':'oxygen', 'co2':'carbon_dioxide',
        'unchanging':'same', 'grows':'grow',
        'largest':'large', 'smallest':'small',
    }
    manual.update({word:str(value) for word,value in NUMBER_WORDS.items()})
    aliases.update(manual)
    words.update(manual)
    return lex, aliases, words, lines


def write_vocab():
    _, _, words, _ = authored()
    (ROOT / 'vocab.txt').write_text('\n'.join(sorted(words)) + '\n')


def build():
    lex, aliases, authored_words, lines = authored()
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    vindex = {s:i+2 for i,s in enumerate(vocab)}

    def canonical(s):
        r = aliases.get(s, s)
        # Follow short authored alias chains, stopping cycles deterministically.
        seen = {s}
        for _ in range(5):
            if r in seen:
                break
            seen.add(r)
            z = aliases.get(r, r)
            if z == r:
                break
            r = z
        return r

    concepts = set()
    for word in vocab:
        if word not in STOP:
            concepts.add(canonical(word))
    phrase_specs = []
    for root, entries in lex:
        for entry in entries:
            if '_' in entry:
                parts = entry.split('_')
                if 2 <= len(parts) <= 3:
                    phrase_specs.append((tuple(canonical(p) for p in parts), canonical(root)))
                    concepts.add(canonical(root))
                    concepts.update(canonical(p) for p in parts)
    # Underscores in either assertions or ontology entries denote phrases.
    auxiliary_lines = []
    for name in AUXILIARY:
        if name in ('experiment_grammar.txt', 'ecology_grammar.txt', 'separation_grammar.txt', 'formula_conditions.txt', 'scope_grammar.txt', 'class_grammar.txt', 'relation_grammar.txt', 'interaction_grammar.txt'):
            continue
        auxiliary_lines.extend(line.strip() for line in (ROOT / name).read_text().splitlines()
                               if line.strip() and not line.lstrip().startswith('#'))
    for line in lines + auxiliary_lines:
        for word in re.findall(r"[a-z0-9_']+", line.lower()):
            if '_' in word:
                parts = word.split('_')
                if 2 <= len(parts) <= 3:
                    phrase_specs.append((tuple(canonical(p) for p in parts), canonical(word)))
                    concepts.add(canonical(word))
                    concepts.update(canonical(p) for p in parts)
    concepts.update(canonical(x) for x in ('density', 'speed', 'work', 'power', 'current', 'resistance', 'voltage'))
    concept_list = ['<null>'] + sorted(concepts)
    ci = {s:i for i,s in enumerate(concept_list)}
    d = len(ci)
    v = len(vocab) + 2
    token_concept = np.zeros(v, dtype=np.int64)
    token_raw_concept = np.zeros(v, dtype=np.int64)
    for word, wid in vindex.items():
        token_raw_concept[wid] = ci.get(canonical(word), 0)
        if word not in STOP:
            token_concept[wid] = token_raw_concept[wid]

    phrases = sorted(set(phrase_specs))
    phrase_tokens = np.zeros((len(phrases), 3), dtype=np.int64)
    phrase_mask = np.zeros((len(phrases), 3), dtype=np.bool_)
    phrase_id = np.zeros(len(phrases), dtype=np.int64)
    entity_roots = {canonical(s.strip()) for s in (ROOT / 'entities.txt').read_text().splitlines()
                    if s.strip() and not s.lstrip().startswith('#')}
    phrase_entity = np.zeros((len(phrases), 3), dtype=np.float32)
    for j, (parts, root) in enumerate(phrases):
        for k, part in enumerate(parts):
            phrase_tokens[j,k] = ci.get(part, 0)
            phrase_mask[j,k] = True
            phrase_entity[j,k] = root in entity_roots
        phrase_id[j] = ci[root]

    def encode_phrase(s, answer=False):
        ids = []
        neg = False
        for word in s.strip().lower().split():
            if word in NEG_WORDS and answer:
                neg = True
                continue
            if word in STOP:
                continue
            root = canonical(word)
            k = ci.get(root, 0)
            if k:
                ids.append(k)
        return tuple(sorted(set(ids))), (-1.0 if neg else 1.0)

    # One context can have many supported / contradicted completions.
    contexts = []
    context_index = {}
    assertions = []
    assertion_index = {}

    def add(query, answer, sign, direct, confidence=1.0):
        qids, _ = encode_phrase(query)
        aids, polarity = encode_phrase(answer, answer=True)
        if not qids or not aids:
            return
        context_key = (qids, direct)
        if context_key not in context_index:
            context_index[context_key] = len(contexts)
            contexts.append(context_key)
        ctx = context_index[context_key]
        key = (ctx, aids, sign, polarity)
        if key not in assertion_index:
            assertion_index[key] = len(assertions)
            assertions.append((ctx, aids, sign, polarity, confidence))
        else:
            position = assertion_index[key]
            old = assertions[position]
            assertions[position] = old[:-1] + (max(old[-1], confidence),)

    for line in lines:
        direct = line.startswith('!')
        reverse = line.startswith('=')
        line = line.lstrip('!=')
        parts = [p.strip() for p in line.split('|')]
        query = parts[0]
        positives = parts[1].split(';')
        negatives = parts[2].split(';') if len(parts) > 2 else []
        confidence = float(parts[3]) if len(parts) > 3 else 1.0
        variants = [' '.join(t) for t in itertools.product(*(word.split('/') for word in query.split()))]
        for qq in variants:
            for answer in positives:
                add(qq, answer, 1., direct, confidence)
                if reverse:
                    add(answer, qq, 1., direct, confidence * .85)
                elif not direct and encode_phrase(answer, answer=True)[1] > 0:
                    # Weak associative retrieval also works from a described
                    # effect to its cause/name. This is not treated as a
                    # logically certain biconditional.
                    add(answer, qq, 1., direct, confidence * .35)
            for answer in negatives:
                add(qq, answer, -1., direct, confidence)

    # Author-corpus frequency is used only to normalize word specificity;
    # there are no questions, labels, gradients, or fitted coefficients here.
    df = np.zeros(d, dtype=np.float32)
    for ids, _ in contexts:
        df[list(ids)] += 1
    feature_weight = np.clip(1. + .32 * np.log((len(contexts) + 5.) / (df + 5.)), 1., 2.8).astype(np.float32)
    for word in WEAK:
        k = ci.get(canonical(word), 0)
        feature_weight[k] *= .68
    for word in ('only', 'always', 'all', 'no', 'not', 'same', 'different', 'one', 'two', 'three', 'first', 'second', 'day', 'year'):
        k = ci.get(canonical(word), 0)
        feature_weight[k] *= .72
    feature_weight[0] = 0

    # Sparse coordinate compilation never allocates the large zero-filled
    # lexical matrices. The inference projections are the same linear maps.
    qrows, qcols, qvalues = [], [], []
    qmass = np.zeros(len(contexts), dtype=np.float32)
    qsize = np.zeros(len(contexts), dtype=np.float32)
    context_direct = np.zeros(len(contexts), dtype=np.float32)
    for j,(ids,direct) in enumerate(contexts):
        values = feature_weight[list(ids)]
        qmass[j] = values.sum()
        qrows.extend(ids)
        qcols.extend([j] * len(ids))
        qvalues.extend(values / max(qmass[j], 1e-6))
        qsize[j] = len(ids)
        context_direct[j] = direct
    qmat = torch.sparse_coo_tensor(torch.tensor([qrows, qcols]),
        torch.tensor(qvalues, dtype=torch.float32), (d, len(contexts))).coalesce()

    arows, acols, avalues = [], [], []
    amass = np.zeros(len(assertions), dtype=np.float32)
    asize = np.zeros(len(assertions), dtype=np.float32)
    acontext = np.zeros(len(assertions), dtype=np.int64)
    asign = np.zeros(len(assertions), dtype=np.float32)
    apolarity = np.zeros(len(assertions), dtype=np.float32)
    confidence = np.zeros(len(assertions), dtype=np.float32)
    context_confidence = np.zeros(len(contexts), dtype=np.float32)
    for j,(ctx,ids,sign,polarity,conf) in enumerate(assertions):
        values = feature_weight[list(ids)]
        amass[j] = values.sum()
        arows.extend(ids)
        acols.extend([j] * len(ids))
        avalues.extend(values / max(amass[j], 1e-6))
        asize[j] = len(ids)
        acontext[j] = ctx
        asign[j] = sign
        apolarity[j] = polarity
        confidence[j] = conf
        context_confidence[ctx] = max(context_confidence[ctx], conf)
    amat = torch.sparse_coo_tensor(torch.tensor([arows, acols]),
        torch.tensor(avalues, dtype=torch.float32), (d, len(assertions))).coalesce()

    # A small hand-selected, symmetric, soft semantic neighborhood. Hard
    # antonyms are deliberately kept separate.
    neighbors = np.zeros((d, 4), dtype=np.int64)
    neighbor_weights = np.zeros((d, 4), dtype=np.float32)
    soft_groups = [
        ('increase', 'high', 'more', 'rise'),
        ('decrease', 'low', 'less', 'fall'),
        ('glucose', 'sugar', 'carbohydrate'),
        ('sunlight', 'solar', 'solar_energy', 'light_energy'),
        ('heat', 'thermal_energy', 'warm', 'hot'),
        ('cold', 'cool'), ('motion', 'move', 'mechanical_energy'),
        ('kinetic', 'kinetic_energy'), ('potential', 'potential_energy'),
        ('chemical', 'chemical_energy'), ('electrical', 'electrical_energy'),
        ('cellular_respiration', 'respiration'),
        ('food_chain', 'food_web'), ('weight', 'weigh'),
        ('renewable', 'renewable_energy'), ('nonrenewable', 'nonrenewable_energy'),
        ('igneous', 'igneous_rock'), ('sedimentary', 'sedimentary_rock'), ('metamorphic', 'metamorphic_rock'),
        ('current', 'electric_current'), ('plant', 'vegetation'),
        ('nervous', 'nerve'),
        ('absorb', 'take'), ('release', 'emit'), ('produce', 'make'),
        ('heat_transfer', 'heat'), ('conduct', 'conduction'),
        ('no', 'zero', 'none'),
    ]
    for group in soft_groups:
        ids = list(dict.fromkeys(ci.get(canonical(x), 0) for x in group))
        ids = [i for i in ids if i]
        for src in ids:
            for dest in ids:
                if src != dest:
                    empty = np.where(neighbors[src] == 0)[0]
                    if len(empty) and dest not in neighbors[src]:
                        k = empty[0]
                        neighbors[src,k] = dest
                        neighbor_weights[src,k] = .72

    # Directed type inheritance is not a synonym relation. A robin implies
    # bird, but bird never implies robin. Compile a bounded transitive closure.
    parents = {}
    for line in ((ROOT / 'taxonomy.txt').read_text() + '\n' + (ROOT / 'taxonomy_extra.txt').read_text()).splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            child, parent_text = [s.strip() for s in line.split('|')]
            source = ci[canonical(child)]
            for parent in parent_text.split():
                destination = ci[canonical(parent)]
                if source != destination:
                    parents.setdefault(source, set()).add(destination)
    tax_edges = []
    for source in sorted(parents):
        visited = {source}
        frontier = {source}
        for depth in range(1, 5):
            frontier = {p for node in frontier for p in parents.get(node, ())} - visited
            for destination in sorted(frontier):
                tax_edges.append((source, destination, .90 ** depth))
            visited.update(frontier)
    tax_source = np.array([e[0] for e in tax_edges], dtype=np.int64)
    tax_target = np.array([e[1] for e in tax_edges], dtype=np.int64)
    tax_strength = np.array([e[2] for e in tax_edges], dtype=np.float32)
    class_mask = np.zeros(d, dtype=np.float32)
    class_mask[sorted({destination for values in parents.values() for destination in values})] = 1.

    qneg = np.zeros(v, dtype=np.float32)
    qexcept = np.zeros(v, dtype=np.float32)
    oneg = np.zeros(v, dtype=np.float32)
    for word,wid in vindex.items():
        if word in NEG_WORDS - {'no', 'without'} or word in {'least', 'incorrect', 'false', 'unable'}:
            qneg[wid] = 1
        if word in {'except', 'excluding'}:
            qexcept[wid] = 1
        if word in NEG_WORDS:
            oneg[wid] = 1

    question_start = np.zeros(v, dtype=np.float32)
    question_main = np.zeros(v, dtype=np.float32)
    question_explain = np.zeros(v, dtype=np.float32)
    literal_not = np.zeros(v, dtype=np.float32)
    nonnegating_follow = np.zeros(v, dtype=np.float32)
    avoid_token = np.zeros(v, dtype=np.float32)
    directive_modal = np.zeros(v, dtype=np.float32)
    purpose_to = np.zeros(v, dtype=np.float32)
    for word, wid in vindex.items():
        question_main[wid] = word in {'which', 'what'}
        question_explain[wid] = word in {'why', 'how'}
        literal_not[wid] = word == 'not'
        nonnegating_follow[wid] = word in {'only', 'just', 'merely'}
        avoid_token[wid] = word in {'avoid', 'avoids', 'avoided', 'avoiding'}
        directive_modal[wid] = word in {'should', 'must'}
        purpose_to[wid] = word == 'to'
    class_question = np.zeros(v, dtype=np.float32)
    class_link = np.zeros(v, dtype=np.float32)
    class_bridge = np.zeros(v, dtype=np.float32)
    for word, wid in vindex.items():
        class_link[wid] = word in {'is', 'are', 'was', 'were', 'be'}
        class_bridge[wid] = word in {'a', 'an', 'the', 'type', 'kind', 'example', 'of', 'classified', 'as', 'considered', 'not', 'also'}
    for word in ('which', 'what', 'why', 'how', 'where', 'when'):
        question_start[vindex[word]] = 1
    for word in ('which', 'what'):
        class_question[vindex[word]] = 1
    numeric = np.full(v, -1., dtype=np.float32)
    numeric_digit = np.zeros(v, dtype=np.float32)
    # Literal cardinal grammar: small words, tens, scales, internal conjunction.
    number_grammar = np.zeros((v, 5), dtype=np.float32)
    numeric_wrapper = np.zeros(v, dtype=np.float32)
    decimal_divisor = np.zeros(v, dtype=np.float32)
    # Base dimensions: g, ml, m, s, N, J, V, A, ohm;
    # derived dimensions: g/ml, m/s, W, m/s^2.
    units = np.zeros((v, 13), dtype=np.float32)
    unit_cues = np.zeros((v, 4), dtype=np.float32)
    unit_cues[vindex['cubic'], 0] = 1
    unit_cues[vindex['cubed'], 1] = 1
    unit_cues[vindex['per'], 2] = 1
    unit_cues[vindex['squared'], 3] = 1
    unit_map = {
        'gram': (0,1.), 'kilogram':(0,1000.), 'milligram':(0,.001),
        'milliliter':(1,1.), 'liter':(1,1000.), 'cm3':(1,1.), 'm3':(1,1000000.),
        'meter':(2,1.), 'centimeter':(2,.01), 'millimeter':(2,.001), 'kilometer':(2,1000.),
        'second':(3,1.), 'minute':(3,60.), 'hour':(3,3600.),
        'newton':(4,1.), 'joule':(5,1.), 'volt':(6,1.), 'ampere':(7,1.), 'ohm':(8,1.),
        'amps':(7,1.), 'amp':(7,1.), 'ohms':(8,1.), 'mg':(0,.001),
        'm':(2,1.), 's':(3,1.), 'watt':(11,1.),
    }
    unit_map = {canonical(word):spec for word,spec in unit_map.items()}
    for word,wid in vindex.items():
        numeric_wrapper[wid] = word in {'about', 'approximately'}
        number_grammar[wid, 0] = word in NUMBER_WORDS and NUMBER_WORDS[word] < 100
        number_grammar[wid, 1] = word in {'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'}
        number_grammar[wid, 2] = word == 'hundred'
        number_grammar[wid, 3] = word == 'thousand'
        number_grammar[wid, 4] = word == 'and'
        if word.isdigit():
            numeric_digit[wid] = 1.
            numeric[wid] = float(word)
            if len(word) <= 3:
                decimal_divisor[wid] = 10. ** len(word)
        elif word in NUMBER_WORDS:
            numeric[wid] = NUMBER_WORDS[word]
        root = canonical(word)
        spec = unit_map.get(root, unit_map.get(word))
        if spec:
            units[wid,spec[0]] = spec[1]
    formula_specs = [
        ('density', ('density',), 9, {0:1, 1:-1}, 1.),
        ('density_mass', ('mass',), 0, {9:1, 1:1}, 1.),
        ('density_volume', ('volume',), 1, {0:1, 9:-1}, 1.),
        ('travel_speed', ('speed','velocity'), 10, {2:1, 3:-1}, 1.),
        ('travel_distance', ('distance','length','far'), 2, {10:1, 3:1}, 1.),
        ('travel_time', ('time','long'), 3, {2:1, 10:-1}, 1.),
        ('work_distance', ('work','energy'), 5, {4:1, 2:1}, 1.),
        ('kinetic', ('kinetic_energy','kinetic'), 5, {0:1, 10:2}, .0005),
        ('work_distance_inverse', ('force',), 4, {5:1, 2:-1}, 1.),
        ('power', ('power',), 11, {5:1, 3:-1}, 1.),
        ('work_power', ('work','energy'), 5, {11:1, 3:1}, 1.),
        ('power_inverse', ('time','long'), 3, {5:1, 11:-1}, 1.),
        ('newton_force', ('force',), 4, {0:1, 12:1}, .001),
        ('newton_mass', ('mass',), 0, {4:1, 12:-1}, 1000.),
        ('newton_acceleration', ('accelerate',), 12, {4:1, 0:-1}, 1000.),
        ('ohm_current', ('current','electric_current'), 7, {6:1, 8:-1}, 1.),
        ('ohm_resistance', ('resistance',), 8, {6:1, 7:-1}, 1.),
        ('ohm_voltage', ('voltage',), 6, {7:1, 8:1}, 1.),
    ]
    formula_exponents = np.zeros((13, len(formula_specs)), dtype=np.float32)
    formula_output = np.zeros(len(formula_specs), dtype=np.int64)
    formula_coefficient = np.zeros(len(formula_specs), dtype=np.float32)
    formula_topics = np.zeros((len(formula_specs),3), dtype=np.int64)
    for j,(family, topics, output, exponents, coefficient) in enumerate(formula_specs):
        formula_output[j] = output
        formula_coefficient[j] = coefficient
        for dimension, power in exponents.items():
            formula_exponents[dimension,j] = power
        for k,topic in enumerate(topics):
            formula_topics[j,k] = ci.get(canonical(topic), 0)

    condition_rows = [line.strip().split('|') for line in
                      (ROOT / 'formula_conditions.txt').read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith('#')]
    formula_condition_tokens = np.zeros((v, len(condition_rows)), dtype=np.float32)
    formula_condition_map = np.zeros((len(condition_rows), len(formula_specs)), dtype=np.float32)
    formula_condition_request = np.zeros(len(condition_rows), dtype=np.float32)
    family_index = {spec[0]:j for j,spec in enumerate(formula_specs)}
    for j,(families, scope, terms) in enumerate(condition_rows):
        assert scope.strip() in {'context', 'request'}
        formula_condition_request[j] = scope.strip() == 'request'
        for family in families.split():
            formula_condition_map[j, family_index[family]] = 1.
        for word in terms.split():
            formula_condition_tokens[vindex[word], j] = 1.

    ordered_rules = []
    for line in (ROOT / 'relations.txt').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            context_text, source_text, target_text = [s.strip() for s in line.split('|')]
            context_ids, _ = encode_phrase(context_text)
            ordered_rules.append((context_ids, ci[canonical(source_text)], ci[canonical(target_text)]))
    rquery = np.zeros((d, len(ordered_rules)), dtype=np.float32)
    rsource = np.zeros(len(ordered_rules), dtype=np.int64)
    rtarget = np.zeros(len(ordered_rules), dtype=np.int64)
    for j,(context_ids, source, target) in enumerate(ordered_rules):
        rquery[list(context_ids),j] = feature_weight[list(context_ids)]
        rsource[j] = source
        rtarget[j] = target
    rquery /= np.maximum(rquery.sum(axis=0), 1e-6)[None,:]
    from_token = np.zeros(v, dtype=np.float32)
    from_token[vindex['from']] = 1

    # Comparative facts: dimensional numeric answers and named reference
    # entities share scalar properties, but an unknown option never receives
    # an invented value. Equal-valued aliases form a single evidence group.
    property_rows = []
    comparison_rows = []
    for line in (ROOT / 'properties.txt').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            attribute, entity, value = [s.strip() for s in line.split('|')]
            property_rows.append((attribute, ci[canonical(entity)], float(value)))
    for line in (ROOT / 'comparisons.txt').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            comparison_rows.append([s.strip() for s in line.split('|')])
    property_names = sorted({r[0] for r in property_rows} | {r[1] for r in comparison_rows})
    pi = {name:i for i,name in enumerate(property_names)}
    groups = {}
    entry_ids = []
    entry_groups = []
    for attribute, entity, value in property_rows:
        key = (pi[attribute], value)
        if key not in groups:
            groups[key] = len(groups)
        entry_ids.append(entity)
        entry_groups.append(groups[key])
    property_group_map = np.zeros((len(groups), len(pi)), dtype=np.float32)
    property_group_values = np.zeros_like(property_group_map)
    for (attribute,value), group in groups.items():
        property_group_map[group,attribute] = 1
        property_group_values[group,attribute] = value
    numerical_properties = {'mass':0, 'volume':1, 'length':2, 'time':3,
                            'force':4, 'energy':5, 'voltage':6, 'current':7,
                            'resistance':8, 'density':9, 'speed':10,
                            'power':11, 'acceleration':12}
    property_dimension = np.array([numerical_properties.get(name,0) for name in property_names],dtype=np.int64)
    property_numeric = np.array([name in numerical_properties for name in property_names],dtype=np.float32)
    comparison_query = np.zeros((d,len(comparison_rows)),dtype=np.float32)
    comparison_size = np.zeros(len(comparison_rows),dtype=np.float32)
    comparison_property = np.zeros(len(comparison_rows),dtype=np.int64)
    comparison_default = np.zeros(len(comparison_rows),dtype=np.float32)
    comparison_positive = np.zeros((v,len(comparison_rows)),dtype=np.float32)
    comparison_negative = np.zeros_like(comparison_positive)
    cue_expansions = {'greatest':'maximum', 'highest':'maximum',
                      'least':'minimum', 'lowest':'minimum', 'fastest':'quickest'}
    for j,(context,attribute,positive,negative,default) in enumerate(comparison_rows):
        ids,_ = encode_phrase(context)
        if not ids:
            raise ValueError('Comparison context has no recognized concepts: '+context)
        comparison_query[list(ids),j] = feature_weight[list(ids)]
        comparison_size[j] = len(ids)
        comparison_property[j] = pi[attribute]
        comparison_default[j] = float(default)
        for text, matrix in ((positive, comparison_positive),(negative, comparison_negative)):
            cues = set(text.split())
            cues.update(replacement for original,replacement in cue_expansions.items() if original in cues)
            for cue in cues:
                matrix[vindex[cue],j] = 1
    comparison_query /= np.maximum(comparison_query.sum(axis=0),1e-6)[None,:]
    attribute_pairs = sorted({(j, pi[attribute])
                              for context,attribute,_,_,_ in comparison_rows
                              for ids,_ in [encode_phrase(context)]
                              if attribute in numerical_properties and len(ids) == 1
                              for j in ids})
    comparison_attribute_ids = np.array([j for j,_ in attribute_pairs], dtype=np.int64)
    comparison_attribute_map = np.zeros((len(attribute_pairs), len(pi)), dtype=np.float32)
    for j,(_,attribute) in enumerate(attribute_pairs):
        comparison_attribute_map[j,attribute] = 1.
    comparison_not = np.zeros(v,dtype=np.float32)
    probability_words = np.zeros((v,3),dtype=np.float32)
    for word,wid in vindex.items():
        if word in (NEG_WORDS - {'no','without'}) | {'except','excluding','false','incorrect'}:
            comparison_not[wid] = 1
        probability_words[wid,0] = word in {'most','least'}
        probability_words[wid,1] = word == 'likely'
        probability_words[wid,2] = word == 'least'

    # Authored predicate oppositions. Store only endpoint ids; inference uses
    # scatter reductions rather than another dense assertion matrix.
    opposite_edges = set()
    for line in (ROOT / 'opposites.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            left, right = [ci[canonical(x.strip())] for x in line.split('|')]
            if left != right:
                opposite_edges.update(((left, right), (right, left)))
    opposite_edges = sorted(opposite_edges)

    causal_rows = []
    for line in (ROOT / 'causal.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            context_text, source, target, direction = [x.strip() for x in line.split('|')]
            causal_rows.append((context_text, ci[canonical(source)], ci[canonical(target)], float(direction)))
    causal_context = np.zeros((d, len(causal_rows)), dtype=np.float32)
    causal_bias = np.zeros(len(causal_rows), dtype=np.float32)
    causal_target_tokens = np.zeros((len(causal_rows), 3), dtype=np.int64)
    causal_target_mask = np.zeros((len(causal_rows), 3), dtype=np.bool_)
    causal_target_length = np.zeros(len(causal_rows), dtype=np.float32)
    for j, (text, source, target, direction) in enumerate(causal_rows):
        parts = concept_list[target].split('_')
        causal_target_length[j] = len(parts)
        for k, part in enumerate(parts[:3]):
            causal_target_tokens[j, k] = ci.get(canonical(part), 0)
            causal_target_mask[j, k] = True
        ids, _ = encode_phrase(text)
        if ids:
            causal_context[list(ids), j] = feature_weight[list(ids)]
        else:
            causal_bias[j] = 1.
    causal_context /= np.maximum(causal_context.sum(axis=0), 1e-6)[None, :]
    delta_tokens = np.zeros((v, 2), dtype=np.float32)
    constant_tokens = np.zeros(v, dtype=np.float32)
    delta_up = set('increase increases increased increasing raise raises raised raising rise rises rising more greater higher larger bigger longer faster stronger hotter warmer brighter louder deeper thicker rougher heavier doubled double doubles tripled triple triples adding added add expands expanded expanding grows grow grew gain gains gained'.split())
    delta_down = set('decrease decreases decreased decreasing reduce reduces reduced reducing lower lowers lowered lowering fall falls falling less fewer smaller shorter slower weaker colder cooler dimmer quieter shallower thinner smoother lighter halved half shrinks shrink shrinking contracted contracts contracting remove removes removed removing lose loses lost drop drops dropped'.split())
    for word, wid in vindex.items():
        delta_tokens[wid, 0] = word in delta_up
        delta_tokens[wid, 1] = word in delta_down
        constant_tokens[wid] = word in {'fixed', 'constant', 'unchanged', 'steady', 'same'}

    # Literal grammar cues intentionally do not use synonym canonicalization:
    # 'measured' can be a passive verb, whereas 'measurement' cannot.
    experiment_sets = {}
    for line in (ROOT / 'experiment_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            experiment_sets[name.strip()] = set(terms.split())
    experiment_weights = {}
    for name, terms in experiment_sets.items():
        table = np.zeros(v, dtype=np.float32)
        for word in terms:
            table[vindex[word]] = 1.
        experiment_weights['experiment_' + name] = table
    experiment_weights['experiment_request'] = np.stack(
        [experiment_weights['experiment_request_' + name]
         for name in ('input', 'response', 'fixed')], axis=1)
    experiment_weights['experiment_role'] = np.stack(
        [experiment_weights['experiment_' + name]
         for name in ('input', 'response', 'fixed')], axis=1)
    experiment_weights['experiment_ignore'] = np.maximum(
        experiment_weights['experiment_ignore'], (token_concept == 0).astype(np.float32))

    # Literal food-web grammar; organism identities come from authored types.
    ecology_weights = {}
    for line in (ROOT / 'ecology_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            ecology_weights['ecology_' + name.strip()] = table
    organism_classes = {ci[canonical(x)] for x in ('animal', 'plant', 'algae', 'fungus', 'bacterium')}
    organism_ids = set(organism_classes)
    for source in parents:
        frontier, seen = {source}, {source}
        for depth in range(8):
            frontier = {p for node in frontier for p in parents.get(node, ())} - seen
            seen.update(frontier)
        if seen & organism_classes:
            organism_ids.add(source)
    # Single-token names only: phrases and their component words must not be
    # mistaken for independent ecological actors.
    entity_table = np.array([k in organism_ids and k != 0 for k in token_raw_concept], dtype=np.float32)
    ecology_weights['ecology_entity'] = entity_table

    # Goal-conditioned binary-mixture separation. Physical alternatives are
    # authored directly; these tables are not inferred from questions.
    separation_weights = {}
    for line in (ROOT / 'separation_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            separation_weights['separation_' + name.strip()] = table
    material_rows = []
    method_rows = []
    rule_rows = []
    for filename, output in [('separation_materials.txt', material_rows),
                             ('separation_methods.txt', method_rows),
                             ('separation_rules.txt', rule_rows)]:
        for line in (ROOT / filename).read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                output.append([s.strip() for s in line.split('|')])
    material_names = sorted({s for _, components, _ in material_rows
                             for s in components.split() if s != '-'})
    material_index = {s:i for i,s in enumerate(material_names)}
    method_index = {row[0]:i for i,row in enumerate(method_rows)}
    entry_components = np.zeros((len(material_rows), len(material_names)), dtype=np.float32)
    entry_target = np.zeros_like(entry_components)
    for j, (name, components, target) in enumerate(material_rows):
        for component in components.split():
            if component != '-':
                entry_components[j, material_index[component]] = 1.
        if target != '-':
            entry_target[j, material_index[target]] = 1.
    # Match every occurrence, not just the first mention of each concept.
    material_patterns = set()
    for j, (name, _, _) in enumerate(material_rows):
        root = canonical(name)
        material_patterns.add(((ci[root],), j))
        for parts, phrase_root in phrases:
            if phrase_root == root:
                material_patterns.add((tuple(ci[p] for p in parts), j))
    material_patterns = sorted(material_patterns)
    material_tokens = np.zeros((len(material_patterns), 3), dtype=np.int64)
    material_mask = np.zeros_like(material_tokens, dtype=np.bool_)
    material_entries = []
    for j, (parts, entry) in enumerate(material_patterns):
        material_tokens[j, :len(parts)] = parts
        material_mask[j, :len(parts)] = True
        material_entries.append(entry)
    method_map = np.zeros((d, len(method_rows)), dtype=np.float32)
    method_parts = set()
    method_concepts = set()
    for name, terms in method_rows:
        for term in terms.split():
            method_map[ci[canonical(term)], method_index[name]] = 1.
            method_concepts.add(canonical(term))
            method_parts.update(canonical(s) for s in term.split('_'))
    allowed_option = separation_weights['separation_option_wrapper'].copy()
    for word, wid in vindex.items():
        if canonical(word) in method_concepts | method_parts:
            allowed_option[wid] = 1.
    rule_components = np.zeros((len(rule_rows), len(material_names)), dtype=np.float32)
    rule_target = np.zeros_like(rule_components)
    rule_methods = np.zeros((len(rule_rows), len(method_rows)), dtype=np.float32)
    rule_dissolved = np.zeros(len(rule_rows), dtype=np.float32)
    for j, (components, target, dissolved_conflict, methods) in enumerate(rule_rows):
        parts = components.split()
        assert len(parts) == 2 and len(set(parts)) == 2
        for part in parts:
            rule_components[j, material_index[part]] = 1.
        if target != '*':
            assert target in parts
            rule_target[j, material_index[target]] = 1.
        rule_dissolved[j] = float(dissolved_conflict)
        for spec in methods.split():
            method, score = spec.split(':')
            rule_methods[j, method_index[method]] = float(score)
    separation_weights.update({
        'separation_material_tokens':material_tokens,
        'separation_material_mask':material_mask,
        'separation_material_components':entry_components[material_entries],
        'separation_material_target':entry_target[material_entries],
        'separation_method_map':method_map,
        'separation_option_allowed':allowed_option,
        'separation_rule_components':rule_components,
        'separation_rule_target':rule_target,
        'separation_rule_methods':rule_methods,
        'separation_rule_dissolved':rule_dissolved,
        'separation_scale':np.array(1.35, dtype=np.float32),
    })

    arrays = {
        'opposite_source':np.array([a for a,b in opposite_edges], dtype=np.int64),
        'opposite_target':np.array([b for a,b in opposite_edges], dtype=np.int64),
        'causal_context':causal_context, 'causal_bias':causal_bias,
        'causal_target_tokens':causal_target_tokens, 'causal_target_mask':causal_target_mask,
        'causal_target_length':causal_target_length,
        'causal_source':np.array([x[1] for x in causal_rows], dtype=np.int64),
        'causal_source_length':np.array([len(concept_list[x[1]].split('_')) for x in causal_rows], dtype=np.float32),
        'causal_attribute_length':np.array([len(concept_list[k].split('_')) for k in sorted({x[1] for x in causal_rows} | {x[2] for x in causal_rows})], dtype=np.float32),
        'causal_target':np.array([x[2] for x in causal_rows], dtype=np.int64),
        'causal_sign':np.array([x[3] for x in causal_rows], dtype=np.float32),
        'causal_attributes':np.array(sorted({x[1] for x in causal_rows} | {x[2] for x in causal_rows}), dtype=np.int64),
        'delta_tokens':delta_tokens, 'constant_tokens':constant_tokens,
        'token_concept':token_concept, 'token_raw_concept':token_raw_concept,
        'phrase_tokens':phrase_tokens, 'phrase_mask':phrase_mask, 'phrase_id':phrase_id,
        'phrase_entity':phrase_entity,
        'feature_weight':feature_weight, 'qmat':qmat, 'qmass':qmass, 'qsize':qsize,
        'amat':amat, 'amass':amass, 'asize':asize, 'acontext':acontext,
        'asign':asign, 'apolarity':apolarity, 'confidence':confidence,
        'context_direct':context_direct, 'context_confidence':context_confidence,
        'neighbors':neighbors, 'neighbor_weights':neighbor_weights,
        'tax_source':tax_source, 'tax_target':tax_target, 'tax_strength':tax_strength,
        'class_mask':class_mask,
        'tax_classes':np.array(sorted({destination for values in parents.values() for destination in values}), dtype=np.int64),
        'qneg':qneg, 'qexcept':qexcept, 'oneg':oneg, 'question_start':question_start,
        'question_main':question_main, 'question_explain':question_explain,
        'why_token':np.array(vindex['why'], dtype=np.int64),
        'reason_bridge':np.array([ci[canonical(x)] for x in ('explain', 'reason')], dtype=np.int64),
        'literal_not':literal_not, 'nonnegating_follow':nonnegating_follow,
        'avoid_token':avoid_token, 'directive_modal':directive_modal, 'purpose_to':purpose_to,
        'class_question':class_question, 'class_link':class_link, 'class_bridge':class_bridge,
        'numeric':numeric, 'numeric_digit':numeric_digit, 'numeric_wrapper':numeric_wrapper,
        'number_grammar':number_grammar,
        'units':units, 'decimal_divisor':decimal_divisor,
        'unit_cues':unit_cues, 'unit_axes':np.eye(13,dtype=np.float32),
        'formula_exponents':formula_exponents, 'formula_output':formula_output,
        'formula_coefficient':formula_coefficient, 'formula_topics':formula_topics,
        'formula_dimensions':np.eye(13,dtype=np.float32)[formula_output],
        'formula_condition_tokens':formula_condition_tokens,
        'formula_condition_map':formula_condition_map,
        'formula_condition_request':formula_condition_request,
        'property_entry_ids':np.array(entry_ids,dtype=np.int64),
        'property_entry_groups':np.array(entry_groups,dtype=np.int64),
        'property_group_map':property_group_map, 'property_group_values':property_group_values,
        'property_dimension':property_dimension, 'property_numeric':property_numeric,
        'comparison_query':comparison_query, 'comparison_size':comparison_size,
        'comparison_attribute_ids':comparison_attribute_ids,
        'comparison_attribute_map':comparison_attribute_map,
        'comparison_attribute_rules':(comparison_query[comparison_attribute_ids] > 0).astype(np.float32),
        'comparison_property':comparison_property, 'comparison_default':comparison_default,
        'comparison_positive':comparison_positive, 'comparison_negative':comparison_negative,
        'comparison_not':comparison_not, 'probability_words':probability_words,
        'rquery':rquery, 'rsource':rsource, 'rtarget':rtarget, 'from_token':from_token,
        'interaction_ids':np.array([ci[canonical(x)] for x in ('magnet', 'charge', 'north', 'south', 'positive', 'negative', '2', 'both', 'same', 'opposite', 'attract', 'repel', 'pole', 'pair')], dtype=np.int64),
        'top_weights':np.array([1.,.20,.10,.05],dtype=np.float32),
    }
    arrays.update(experiment_weights)
    arrays.update(ecology_weights)
    arrays.update(separation_weights)

    # Explicit vector addition, not retrieval: each endpoint is a literal
    # direction and each magnitude must have an attached complete force unit.
    force_tables = {}
    for line in (ROOT / 'force_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            force_tables[name.strip()] = table
    directions = np.zeros((v, 4), dtype=np.float32)
    for j, (minus, plus) in enumerate((('left', 'right'), ('west', 'east'),
                                      ('south', 'north'), ('down', 'up'))):
        directions[:, j] = force_tables[plus] - force_tables[minus]
    for name in ('request', 'force', 'block', 'wrapper', 'bridge'):
        arrays['force_' + name] = force_tables[name]
    arrays['force_direction'] = directions
    arrays['force_scale'] = np.array(3., dtype=np.float32)
    for line in (ROOT / 'scope_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            arrays['scope_' + name.strip()] = table
    for line in (ROOT / 'class_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            arrays['class_' + name.strip()] = table
    relation_allowed = np.zeros(v, dtype=np.float32)
    for line in (ROOT / 'relation_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            arrays['relation_' + name.strip()] = table
            relation_allowed = np.maximum(relation_allowed, table)
    arrays['relation_allowed'] = relation_allowed
    for line in (ROOT / 'interaction_grammar.txt').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, terms = line.split('|')
            table = np.zeros(v, dtype=np.float32)
            for word in terms.split():
                table[vindex[word]] = 1.
            arrays['interaction_' + name.strip()] = table
    return {name:(value if isinstance(value, torch.Tensor) else
                  torch.from_numpy(np.ascontiguousarray(value)))
            for name,value in arrays.items()}


if __name__ == '__main__':
    if '--vocab' in sys.argv:
        write_vocab()
    else:
        weights = build()
        print({k:tuple(v.shape) for k,v in weights.items()})
        print('weight bytes:', sum(v.numel()*v.element_size() for v in weights.values()))

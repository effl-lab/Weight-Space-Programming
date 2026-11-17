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
SOURCES = ('biology.txt', 'physics.txt', 'chemistry.txt', 'earth_space.txt', 'inquiry.txt', 'extra.txt', 'concepts.txt')
AUXILIARY = ('relations.txt', 'taxonomy.txt', 'properties.txt', 'comparisons.txt')
TOKEN = re.compile(r"[a-z0-9']+")
STOP = set('a an the this that these those is are was were be being been of to for from by with as at in on into onto through than then and or but so it its their they them he she his her our we you your i me us which what where when why how who whom whose would could should may might can will shall do does did done has have had having there here such following below above example examples each any some several certain particular another other during between within about approximately usually generally typically likely best most true'.split())
# These weak words are useful in some relational rules, but should not dominate.
WEAK = set('energy water plant animal cell gas food earth sun heat temperature mass volume light force motion increase decrease high low more less same different only always all make produce help need allow process function purpose structure system part source cause result change body object substance material amount number rate'.split())
NEG_WORDS = set("no not never without cannot can't isn't aren't doesn't don't won't wasn't weren't".split())
NUMBER_WORDS = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'twenty':20, 'thirty':30, 'sixty':60, 'hundred':100, 'thousand':1000}
EXTRA = '''not except least avoid false incorrect no without never neither unable cannot none zero one two three four five six seven eight nine ten eleven twelve twenty thirty sixty hundred thousand cm3 m3 cubic squared per milligram milligrams mg ampere amperes amp amps ohm ohms pascal pascals pa kelvin k mph seconds minutes hours meters kilometers milliliters liters grams kilograms newtons joules watts volts both neither either always only all constant unchanged remains remain unchanged rises falls increases decreases increasing decreasing greater smaller higher lower fewer times divided twice half third fourth double triple maximum minimum greatest smallest weakest strongest safest unsafe harmful harmless faster slower fastest slowest northward southward eastward westward rise rises rising set sets setting sunrise sunset dawn dusk cells cellular none net 365 366 273 212 32 24 27 29 23 10 100 0'''


def authored():
    lex = []
    aliases = {}
    words = set(TOKEN.findall(EXTRA)) | STOP | NEG_WORDS
    for line in (ROOT / 'lexicon.txt').read_text().splitlines():
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
    for j, (parts, root) in enumerate(phrases):
        for k, part in enumerate(parts):
            phrase_tokens[j,k] = ci.get(part, 0)
            phrase_mask[j,k] = True
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

    qmat = np.zeros((d, len(contexts)), dtype=np.float32)
    qsize = np.zeros(len(contexts), dtype=np.float32)
    context_direct = np.zeros(len(contexts), dtype=np.float32)
    for j,(ids,direct) in enumerate(contexts):
        qmat[list(ids),j] = feature_weight[list(ids)]
        qsize[j] = len(ids)
        context_direct[j] = direct
    qmass = qmat.sum(axis=0)
    qmat /= np.maximum(qmass, 1e-6)[None,:]

    amat = np.zeros((d, len(assertions)), dtype=np.float32)
    amass = np.zeros(len(assertions), dtype=np.float32)
    asize = np.zeros(len(assertions), dtype=np.float32)
    acontext = np.zeros(len(assertions), dtype=np.int64)
    asign = np.zeros(len(assertions), dtype=np.float32)
    apolarity = np.zeros(len(assertions), dtype=np.float32)
    confidence = np.zeros(len(assertions), dtype=np.float32)
    context_confidence = np.zeros(len(contexts), dtype=np.float32)
    for j,(ctx,ids,sign,polarity,conf) in enumerate(assertions):
        amat[list(ids),j] = feature_weight[list(ids)]
        amass[j] = feature_weight[list(ids)].sum()
        asize[j] = len(ids)
        acontext[j] = ctx
        asign[j] = sign
        apolarity[j] = polarity
        confidence[j] = conf
        context_confidence[ctx] = max(context_confidence[ctx], conf)
    amat /= np.maximum(amass, 1e-6)[None,:]

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
    for line in (ROOT / 'taxonomy.txt').read_text().splitlines():
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

    qneg = np.zeros(v, dtype=np.float32)
    qexcept = np.zeros(v, dtype=np.float32)
    oneg = np.zeros(v, dtype=np.float32)
    for word,wid in vindex.items():
        if word in NEG_WORDS - {'no', 'without'} or word in {'least', 'avoid', 'avoided', 'incorrect', 'false', 'unable'}:
            qneg[wid] = 1
        if word in {'except', 'excluding'}:
            qexcept[wid] = 1
        if word in NEG_WORDS:
            oneg[wid] = 1

    question_start = np.zeros(v, dtype=np.float32)
    class_question = np.zeros(v, dtype=np.float32)
    for word in ('which', 'what', 'why', 'how', 'where', 'when'):
        question_start[vindex[word]] = 1
    for word in ('which', 'what'):
        class_question[vindex[word]] = 1
    numeric = np.full(v, -1., dtype=np.float32)
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
        if word.isdigit():
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
        (('density',), 9, {0:1, 1:-1}, 1.),
        (('mass',), 0, {9:1, 1:1}, 1.),
        (('volume',), 1, {0:1, 9:-1}, 1.),
        (('speed','velocity'), 10, {2:1, 3:-1}, 1.),
        (('distance','length','far'), 2, {10:1, 3:1}, 1.),
        (('time','long'), 3, {2:1, 10:-1}, 1.),
        (('work','energy'), 5, {4:1, 2:1}, 1.),
        (('kinetic_energy','kinetic'), 5, {0:1, 10:2}, .0005),
        (('force',), 4, {5:1, 2:-1}, 1.),
        (('power',), 11, {5:1, 3:-1}, 1.),
        (('work','energy'), 5, {11:1, 3:1}, 1.),
        (('time','long'), 3, {5:1, 11:-1}, 1.),
        (('force',), 4, {0:1, 12:1}, .001),
        (('mass',), 0, {4:1, 12:-1}, 1000.),
        (('accelerate',), 12, {4:1, 0:-1}, 1000.),
        (('current','electric_current'), 7, {6:1, 8:-1}, 1.),
        (('resistance',), 8, {6:1, 7:-1}, 1.),
        (('voltage',), 6, {7:1, 8:1}, 1.),
    ]
    formula_exponents = np.zeros((13, len(formula_specs)), dtype=np.float32)
    formula_output = np.zeros(len(formula_specs), dtype=np.int64)
    formula_coefficient = np.zeros(len(formula_specs), dtype=np.float32)
    formula_topics = np.zeros((len(formula_specs),3), dtype=np.int64)
    for j,(topics, output, exponents, coefficient) in enumerate(formula_specs):
        formula_output[j] = output
        formula_coefficient[j] = coefficient
        for dimension, power in exponents.items():
            formula_exponents[dimension,j] = power
        for k,topic in enumerate(topics):
            formula_topics[j,k] = ci.get(canonical(topic), 0)

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
    comparison_not = np.zeros(v,dtype=np.float32)
    probability_words = np.zeros((v,3),dtype=np.float32)
    for word,wid in vindex.items():
        if word in (NEG_WORDS - {'no','without'}) | {'except','excluding','false','incorrect'}:
            comparison_not[wid] = 1
        probability_words[wid,0] = word in {'most','least'}
        probability_words[wid,1] = word == 'likely'
        probability_words[wid,2] = word == 'least'

    arrays = {
        'token_concept':token_concept, 'token_raw_concept':token_raw_concept,
        'phrase_tokens':phrase_tokens, 'phrase_mask':phrase_mask, 'phrase_id':phrase_id,
        'feature_weight':feature_weight, 'qmat':qmat, 'qmass':qmass, 'qsize':qsize,
        'amat':amat, 'amass':amass, 'asize':asize, 'acontext':acontext,
        'asign':asign, 'apolarity':apolarity, 'confidence':confidence,
        'context_direct':context_direct, 'context_confidence':context_confidence,
        'neighbors':neighbors, 'neighbor_weights':neighbor_weights,
        'tax_source':tax_source, 'tax_target':tax_target, 'tax_strength':tax_strength,
        'tax_classes':np.array(sorted({destination for values in parents.values() for destination in values}), dtype=np.int64),
        'qneg':qneg, 'qexcept':qexcept, 'oneg':oneg, 'question_start':question_start,
        'class_question':class_question,
        'numeric':numeric, 'units':units, 'decimal_divisor':decimal_divisor,
        'unit_cues':unit_cues, 'unit_axes':np.eye(13,dtype=np.float32),
        'formula_exponents':formula_exponents, 'formula_output':formula_output,
        'formula_coefficient':formula_coefficient, 'formula_topics':formula_topics,
        'formula_dimensions':np.eye(13,dtype=np.float32)[formula_output],
        'property_entry_ids':np.array(entry_ids,dtype=np.int64),
        'property_entry_groups':np.array(entry_groups,dtype=np.int64),
        'property_group_map':property_group_map, 'property_group_values':property_group_values,
        'property_dimension':property_dimension, 'property_numeric':property_numeric,
        'comparison_query':comparison_query, 'comparison_size':comparison_size,
        'comparison_property':comparison_property, 'comparison_default':comparison_default,
        'comparison_positive':comparison_positive, 'comparison_negative':comparison_negative,
        'comparison_not':comparison_not, 'probability_words':probability_words,
        'rquery':rquery, 'rsource':rsource, 'rtarget':rtarget, 'from_token':from_token,
        'interaction_ids':np.array([ci[canonical(x)] for x in ('magnet', 'charge', 'north', 'south', 'positive', 'negative', '2', 'both', 'same', 'opposite', 'attract', 'repel', 'pole')], dtype=np.int64),
        'top_weights':np.array([1.,.20,.10,.05],dtype=np.float32),
    }
    return {name:torch.from_numpy(np.ascontiguousarray(value)) for name,value in arrays.items()}


if __name__ == '__main__':
    if '--vocab' in sys.argv:
        write_vocab()
    else:
        weights = build()
        print({k:tuple(v.shape) for k,v in weights.items()})
        print('weight bytes:', sum(v.numel()*v.element_size() for v in weights.values()))

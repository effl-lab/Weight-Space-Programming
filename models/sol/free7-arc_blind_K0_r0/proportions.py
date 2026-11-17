"""Compile hand-written dimensional power laws and ratio-language controls."""
from pathlib import Path
import re
import torch

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")

QUANTITIES = {
    'mass': 'mass', 'volume': 'volume', 'distance': 'distance separation', 'time': 'time duration',
    'force': 'force', 'speed': 'speed velocity', 'gravity': 'gravity', 'height': 'height',
    'area': 'area', 'work': 'work', 'voltage': 'voltage', 'current': 'current',
    'resistance': 'resistance', 'frequency': 'frequency', 'wavelength': 'wavelength',
    'length': 'length',
}
RATIOS = [('double doubles doubled doubling twice twofold', 2),
          ('triple triples tripled tripling thrice threefold', 3),
          ('quadruple quadruples quadrupled quadrupling fourfold', 4),
          ('half halves halved halving', 0.5), ('quarter quartered quartering', 0.25),
          ('unchanged same constant', 1)]
BRIDGE = 'a an the of is are was were be been being becomes become became its their original initial final value amount size by to in at it object objects ball cart car block sample container between two both remains remain stays stay will would could may might now factor factors times increase increases increased increasing decrease decreases decreased decreasing reduced reduces reduce rises rise rising falls fall falling drops drop dropped'
OPTION_FILLER = 'a an the of is are was were be been being as much great large original initial final value amount size by to in at it its their they will would should has have becomes become became factor factors times fold remains remain stays stay increase increases increased increasing decrease decreases decreased decreasing reduced reduces reduce'
QUALITATIVE_STATIC = 'unchanged same constant steady'
QUALITATIVE_FORBIDDEN = 'or either neither no negative zero stop stops stopped reversed reverses reversing removed remove added add heated heating cooled cooling before then'
PARSER = BRIDGE + ' factor factors times fold increase increases increased increasing decrease decreases decreased decreasing reduced reduces reduce rises rise rising falls fall falling drops drop dropped which what if when after unless not never cannot incorrect false except more less larger smaller fewer greater least most nearly almost ' + OPTION_FILLER + ' ' + QUALITATIVE_STATIC + ' ' + QUALITATIVE_FORBIDDEN


def rows():
    for raw in (ROOT / 'proportions.txt').read_text().splitlines():
        text = raw.split('#', 1)[0].strip()
        if text:
            yield [part.strip() for part in text.split('|')]


def feature_phrases():
    for target, law, context in rows():
        yield from [part.strip() for part in target.split(',')]
        yield from [part.strip() for part in context.split(',') if part.strip()]


def authored_text():
    yield from feature_phrases()
    yield from QUANTITIES.values()
    for names, ratio in RATIOS:
        yield names
    yield PARSER


def compile_proportions(selector, count, vocabulary, lexical_tables, lexicon):
    index = {word: i + 2 for i, word in enumerate(vocabulary)}
    quantity_ids = {name: i + 1 for i, name in enumerate(QUANTITIES)}
    lex = lexical_tables['lex']
    quantity_by_concept = {}
    for name, alternatives in QUANTITIES.items():
        for word in alternatives.split():
            quantity_by_concept[int(lex[index[word]])] = quantity_ids[name]
    quantity = torch.tensor([quantity_by_concept.get(int(value), 0) for value in lex], dtype=torch.int64)

    def words(text):
        table = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            assert word in index, ('Missing proportional-parser word', word)
            table[index[word]] = 1
        return table

    factor = torch.zeros(len(vocabulary) + 2)
    for names, value in RATIOS:
        canonical = {lexicon[word] for word in names.split()}
        for word, token_id in index.items():
            if lexicon.get(word) in canonical:
                factor[token_id] = value
    laws = list(rows())
    targets = torch.zeros(len(laws), count)
    contexts = torch.zeros_like(targets)
    context_needed = torch.zeros(len(laws))
    powers = torch.zeros(len(laws), len(QUANTITIES) + 1)
    option_target = torch.zeros(len(vocabulary) + 2, len(laws), dtype=torch.bool)
    for i, (target, law, context) in enumerate(laws):
        # Match exact semantic targets, not just generic words such as 'energy'.
        target_words = set()
        for text in target.split(','):
            targets[i, selector(text.strip())] = 1
            target_words.update(int(lex[index[word]]) for word in TOKEN.findall(text.lower()))
        target_words.discard(0)
        option_target[:, i] = torch.isin(lex, torch.tensor(sorted(target_words), dtype=torch.int64))
        for term in law.split():
            name, exponent = term.split(':')
            powers[i, quantity_ids[name]] = float(exponent)
        if context:
            context_needed[i] = 1
            for text in context.split(','):
                contexts[i, selector(text.strip())] = 1
    return {
        'ratio_quantity': quantity, 'ratio_factor': factor,
        'ratio_distance_id': torch.tensor(quantity_ids['distance']),
        'ratio_mass_id': torch.tensor(quantity_ids['mass']),
        'ratio_between': words('between'), 'ratio_bridge_token': torch.tensor(index['of']),
        'ratio_powers': powers, 'ratio_target_bag': targets,
        'ratio_context': contexts, 'ratio_context_needed': context_needed,
        'ratio_alignment': torch.tensor([target == 'work' for target, law, context in laws]),
        'ratio_bridge': words(BRIDGE),
        'ratio_times': words('times fold'), 'ratio_factor_word': words('factor factors'),
        'ratio_change': words('increase increases increased increasing decrease decreases decreased decreasing reduced reduces reduce rises rise rising falls fall falling drops drop dropped'),
        'ratio_down': words('decrease decreases decreased decreasing reduced reduces reduce falls fall falling drops drop dropped'),
        'ratio_up': words('increase increases increased increasing rises rise rising'),
        'ratio_option_target': option_target,
        'ratio_forbidden': words('not never cannot incorrect false except unless'),
        'ratio_option_filler': words(OPTION_FILLER),
        'ratio_comparative': words('more less larger smaller fewer greater least most nearly almost'),
        'ratio_constants': torch.tensor([7.0, 16.0, 8.0]),
        'qualitative_static': words(QUALITATIVE_STATIC),
        'qualitative_forbidden': words(QUALITATIVE_FORBIDDEN),
        'qualitative_gain': torch.tensor(6.0),
    }

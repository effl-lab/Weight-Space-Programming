"""Compile explicit scientific slots and bounded situational food-web controls.
All source statements, parser dictionaries, and constants are authored. No fitting.
"""
from pathlib import Path
import re
import torch

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")


def lines(name):
    for raw in (ROOT / name).read_text().splitlines():
        clean = raw.split('#', 1)[0].strip()
        if clean:
            yield clean


def rows(name):
    names = (name, 'slots_g5.txt') if name == 'slots.txt' else (name,)
    return [[part.strip() for part in text.split('|')] for source in names for text in lines(source)]


def atoms(expression):
    return [atom.strip() for clause in expression.split(';') for atom in clause.split(',') if atom.strip()]


def authored_text():
    for subject, relation, value, strength in rows('slots.txt'):
        yield from atoms(subject)
        yield from atoms(value)
    for relation, cues, implicit in rows('slot_relations.txt'):
        yield from atoms(cues)
    yield from lines('ecology_parser_words.txt')


def feature_phrases():
    for subject, relation, value, strength in rows('slots.txt'):
        yield from atoms(subject)
        yield from atoms(value)


def compile_relational(selector, count, vocabulary, lexical_tables):
    relation_rows = rows('slot_relations.txt')
    relation_ids = {row[0]: i for i, row in enumerate(relation_rows)}
    index = {word: i + 2 for i, word in enumerate(vocabulary)}
    # Preserve prepositions in relation cues. Dropping 'from' from 'energy from'
    # would turn every energy question into a request for an input energy.
    lex = lexical_tables['lex']
    lexical_size = int(lexical_tables['base'])
    cue_lex = torch.where(lex > 0, lex, torch.arange(len(vocabulary) + 2) + lexical_size)
    cue_lex[0] = 0
    cue_lex[1] = 0
    cue_base = int(cue_lex.amax()) + 1
    cue_lookup = {}
    cue_expressions = []
    for name, expression, implicit in relation_rows:
        clauses = []
        for clause in expression.split(';'):
            alternatives = []
            for text in atoms(clause):
                words = TOKEN.findall(text.lower())
                assert 1 <= len(words) <= 3, ('Relation cue width', text)
                code = 0
                for word in words:
                    code = code * cue_base + int(cue_lex[index[word]])
                if code not in cue_lookup:
                    cue_lookup[code] = len(cue_lookup) + 1
                alternatives.append(cue_lookup[code])
            clauses.append(alternatives)
        cue_expressions.append(clauses)
    cue_groups = [()]
    cue_group_lookup = {(): 0}
    cue_required = []
    for clauses in cue_expressions:
        required = []
        for clause in clauses:
            key = tuple(sorted(set(clause)))
            if key not in cue_group_lookup:
                cue_group_lookup[key] = len(cue_groups)
                cue_groups.append(key)
            required.append(cue_group_lookup[key])
        cue_required.append(required)
    cue_group_matrix = torch.zeros(len(cue_lookup) + 1, len(cue_groups))
    for i, group in enumerate(cue_groups):
        cue_group_matrix[list(group), i] = 1
    cue_order = sorted(cue_lookup)
    groups = [()]
    lookup = {(): 0}

    def parse(expression):
        result = []
        for clause in expression.split(';'):
            if clause.strip():
                key = tuple(sorted(set(selector(text.strip()) for text in clause.split(','))))
                if key not in lookup:
                    lookup[key] = len(groups)
                    groups.append(key)
                result.append(lookup[key])
        return result

    subjects, values, relations, strengths = [], [], [], []
    for subject, relation, value, strength in rows('slots.txt'):
        subjects.append(parse(subject))
        values.append(parse(value))
        relations.append(relation_ids[relation])
        strengths.append(float(strength))
    group_matrix = torch.zeros(len(groups), count)
    for i, group in enumerate(groups):
        group_matrix[i, list(group)] = 1

    def pad(values):
        width = max(1, max(len(row) for row in values))
        return torch.tensor([row + [0] * (width - len(row)) for row in values], dtype=torch.int64)

    def word_table(text, value=1):
        result = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            assert word in index, ('Parser vocabulary missing', word)
            result[index[word]] = value
        return result

    # Distinct named organisms stay distinct. Multiword names are deliberately
    # omitted from the bounded raw-token graph parser rather than conflated.
    named = set()
    for line in lines('taxonomy.txt'):
        category, members = line.split('|')
        if category.strip() in {'mammal', 'bird', 'reptile', 'amphibian', 'fish', 'insect', 'arachnid',
                                'crustacean', 'mollusk', 'annelid', 'cnidarian', 'echinoderm',
                                'flowering plant', 'conifer', 'fungus', 'alga', 'herbivore', 'carnivore',
                                'omnivore', 'plant', 'producer', 'scavenger'}:
            named.update(word.strip() for word in members.split(',') if len(TOKEN.findall(word.strip())) == 1)
    named.update('grass algae phytoplankton zooplankton plankton caterpillar minnow snake rabbit fox hawk owl wheat corn mouse rat beetle frog toad wolf deer lizard cricket locust krill shrimp fish bird insect plant tree shrub seal salmon trout tuna herring sardine antelope lion gazelle lynx hare coyote otter spider worm earthworm bacteria fungus mushroom mosquito aphid squirrel acorn seed leaf oak maple willow clover duckweed lettuce dragonfly kingfisher eagle robin sparrow vulture'.split())
    # Canonical IDs unify inflections; they do not identify different organisms.
    entities = sorted(set(int(lexical_tables['lex'][index[name]]) for name in named if name in index))
    entities = [value for value in entities if value > 0]
    entity_lookup = {value: i + 1 for i, value in enumerate(entities)}
    entity_table = torch.tensor([entity_lookup.get(int(value), 0) for value in lexical_tables['lex']], dtype=torch.int64)
    # Use one extra vocabulary-independent dimension for the null entity.
    entity_size = len(entities) + 1
    return {
        'slot_groups': group_matrix,
        'slot_subject': pad(subjects), 'slot_value': pad(values),
        'slot_relation': torch.tensor(relations, dtype=torch.int64),
        'slot_strength': torch.tensor(strengths),
        'slot_cue_lex': cue_lex, 'slot_cue_base': torch.tensor(cue_base),
        'slot_cue_codes': torch.tensor(cue_order + [cue_base ** 3 + 1], dtype=torch.int64),
        'slot_cue_ids': torch.tensor([cue_lookup[code] for code in cue_order] + [0], dtype=torch.int64),
        'slot_cue_groups': cue_group_matrix,
        'slot_cue_required': pad(cue_required),
        'slot_implicit': torch.tensor([0.55 * int(row[2]) for row in relation_rows]),
        'eco_entity': entity_table, 'eco_size': torch.zeros(entity_size),
        'eco_predicate': word_table('eat eats eating ate eaten consume consumes consuming consumed devour devours devoured'),
        'eco_passive': word_table('by'),
        'eco_feeds': word_table('feed feeds feeding fed'),
        'eco_on': word_table('on upon'),
        'eco_bridge': word_table('a an the is are was were be been being by on upon only mainly primarily mostly usually also both small large young adult tiny little entirely exclusively'),
        'eco_request': word_table('which what'),
        'eco_conditional': word_table('if when after unless'),
        'eco_population': word_table('population populations number numbers organisms organism ecosystem ecosystems food web chain community communities species'),
        'eco_up': word_table('increase increases increased increasing grow grows growing rise rises rising more greater abundant larger'),
        'eco_down': word_table('decrease decreases decreased decreasing decline declines declining fewer less smaller reduced reduces reduce'),
        'eco_event_up': word_table('introduced introduce introduces added add adds'),
        'eco_event_down': word_table('removed remove removes disappear disappears disappeared extinct extinction killed kill kills eliminated eliminate eliminates dies die died'),
        'eco_event_bridge': word_table('a an the population populations number numbers of in this that these those is are was were be been being will would could may might suddenly greatly rapidly sharply dramatically all many most some completely from area ecosystem food web chain'),
        'eco_query_boundary': word_table('which what why how where whether'),
        'eco_unsupported': word_table('not never cannot unlikely except incorrect false without unless no or either neither'),
        'eco_option_filler': word_table('a an the of in this that these those is are was were be been being will would could may might there it its they their population populations number numbers organisms organism species ecosystem ecosystems community communities becomes become became'),
        'eco_constants': torch.tensor([5.0, 4.0, 0.4, 7.0]),
    }

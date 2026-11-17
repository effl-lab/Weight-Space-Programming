"""Compile explicitly authored three-valued properties and taxonomic inheritance.
No datasets, fitting, search, or assumption that missing declarations are false.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
PARSER_WORDS = "and both also or either neither not no never cannot can't isn't aren't wasn't weren't doesn't don't didn't without lacks lack lacking heated heating molten melting dissolved wet burning once formerly if when unless"


def rows(name):
    for raw in (ROOT / name).read_text().splitlines():
        text = raw.split('#', 1)[0].strip()
        if text:
            yield [part.strip() for part in text.split('|')]


def atoms(expression):
    return [part.strip() for clause in expression.split(';') for part in clause.split(',') if part.strip()]


def query_rows():
    for name in ('trait_queries.txt', 'trait_queries_g5.txt', 'trait_queries_g6.txt', 'trait_queries_g7.txt'):
        for row in rows(name):
            assert len(row) in (3, 4), ('Invalid trait-query row', row)
            yield row + [''] * (4 - len(row))


def fact_rows():
    for name in ('trait_facts.txt', 'trait_facts_g5.txt', 'trait_facts_g6.txt', 'trait_facts_g7.txt'):
        for row in rows(name):
            assert len(row) == 2, ('Invalid trait-fact row', row)
            yield row


def feature_phrases():
    for name, positive, negative, forbidden in query_rows():
        yield from atoms(positive)
        yield from atoms(negative)
        yield from atoms(forbidden)
    for entity, facts in fact_rows():
        yield entity


def authored_text():
    yield from feature_phrases()
    yield PARSER_WORDS


def compile_traits(selector, sequence, count, vocabulary):
    declarations = list(query_rows())
    names = list(dict.fromkeys(row[0] for row in declarations))
    attributes = {name: i for i, name in enumerate(names)}
    group_lookup = {(): 0}
    groups = [()]

    def parse(expression):
        required = []
        for clause in expression.split(';'):
            if clause.strip():
                key = tuple(sorted(set(selector(text.strip()) for text in clause.split(','))))
                if key not in group_lookup:
                    group_lookup[key] = len(groups)
                    groups.append(key)
                required.append(group_lookup[key])
        return required

    positives, negatives, exclusions = [], [], []
    for name, positive, negative, forbidden in declarations:
        positives.append(parse(positive))
        negatives.append(parse(negative))
        exclusions.append(parse(forbidden))
    group_matrix = torch.zeros(len(groups), count)
    for i, group in enumerate(groups):
        group_matrix[i, list(group)] = 1

    def pad(entries):
        width = max(1, max(len(entry) for entry in entries))
        return torch.tensor([entry + [0] * (width - len(entry)) for entry in entries], dtype=torch.int64)

    direct = {}
    for entity, facts in fact_rows():
        for declaration in facts.split(','):
            declaration = declaration.strip()
            if declaration:
                sign, attribute = declaration[0], declaration[1:]
                assert sign in '+-0' and attribute in attributes, ('Invalid property', entity, declaration)
                value = {'+': 1, '-': -1, '0': 0}[sign]
                old = direct.setdefault(entity, {}).setdefault(attribute, value)
                assert old == value, ('Conflicting direct declarations', entity, attribute)
    parents = {}
    entities = set(direct)
    for parent, members in rows('taxonomy.txt'):
        entities.add(parent)
        for member in members.split(','):
            member = member.strip()
            entities.add(member)
            parents.setdefault(member, set()).add(parent)
    # The nearest explicit ancestor wins. Equal-priority contradictions are
    # rejected rather than resolved using an arbitrary source ordering.
    inherited = {}
    for entity in sorted(entities):
        distance = {entity: 0}
        for step in range(8):
            for node, depth in list(distance.items()):
                for parent in parents.get(node, set()):
                    distance[parent] = min(distance.get(parent, 1000), depth + 1)
        card = []
        for attribute in names:
            available = [(depth, direct[node][attribute]) for node, depth in distance.items()
                         if attribute in direct.get(node, {})]
            value = 0
            if available:
                nearest = min(depth for depth, value in available)
                values = {value for depth, value in available if depth == nearest}
                assert len(values) == 1, ('Conflicting inherited properties', entity, attribute, values)
                value = next(iter(values))
            card.append(value)
        inherited[entity] = card
    # Collapse lexical synonyms only when they agree semantically. If a named
    # entity and its spelling variant share a feature they must share a card.
    feature_cards = {}
    feature_names = {}
    feature_sequences = {}
    for entity, card in inherited.items():
        feature = selector(entity)
        if feature in feature_cards:
            previous = feature_cards[feature]
            assert all(a == b or a == 0 or b == 0 for a, b in zip(previous, card)), ('Alias property conflict', entity)
            card = [a or b for a, b in zip(previous, card)]
        feature_cards[feature] = card
        feature_names.setdefault(feature, entity)
        feature_sequences.setdefault(feature, set()).add(sequence(entity))
    # Keep every authored spelling, not only the first representative name.
    # Generic ER and rough ER can have equally long representative spellings,
    # despite a strict containment relation between their explicit aliases.
    for raw in (ROOT / 'concepts.txt').read_text().splitlines():
        clean = raw.split('#', 1)[0].strip()
        if clean:
            for text in clean.split('|', 1)[1].split(';'):
                feature = selector(text.strip())
                if feature in feature_sequences:
                    feature_sequences[feature].add(sequence(text.strip()))
    features = sorted(feature_cards)
    patterns = [()]
    pattern_index = {(): 0}
    entity_patterns = []
    owners = {}
    for i, feature in enumerate(features):
        ids = []
        for seq in sorted(feature_sequences[feature]):
            assert 1 <= len(seq) <= 4, ('Invalid entity name', feature_names[feature], seq)
            if seq not in pattern_index:
                pattern_index[seq] = len(patterns)
                patterns.append(seq)
            ids.append(pattern_index[seq])
            owners.setdefault(seq, set()).add(i)
        entity_patterns.append(ids)
    dominance = torch.zeros(len(features), len(features))
    for i, feature in enumerate(features):
        for seq in feature_sequences[feature]:
            for width in range(1, len(seq)):
                for start in range(len(seq) - width + 1):
                    for j in owners.get(seq[start:start + width], ()):
                        if i != j:
                            dominance[i, j] = 1
    assert not ((dominance > 0) & (dominance.t() > 0)).any(), 'Cyclic entity-name containment'
    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def words(text):
        tensor = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            assert word in index, ('Missing trait parser word', word)
            tensor[index[word]] = 1
        return tensor

    return {
        'trait_groups': group_matrix,
        'trait_positive': pad(positives), 'trait_negative': pad(negatives),
        'trait_exclusions': pad(exclusions),
        'trait_has_positive': torch.tensor([len(row) > 0 for row in positives]),
        'trait_has_negative': torch.tensor([len(row) > 0 for row in negatives]),
        'trait_attribute': torch.tensor([attributes[row[0]] for row in declarations], dtype=torch.int64),
        'trait_cards': torch.tensor([feature_cards[feature] for feature in features], dtype=torch.float32),
        'trait_entity': torch.tensor(features, dtype=torch.int64),
        'trait_dominance': dominance,
        'trait_patterns': torch.tensor([list(seq) + [0] * (4 - len(seq)) for seq in patterns], dtype=torch.int64),
        'trait_pattern_width': torch.tensor([len(seq) for seq in patterns], dtype=torch.int64),
        'trait_entity_patterns': pad(entity_patterns),
        'trait_conjunction': words('and both also'),
        'trait_forbidden': words(PARSER_WORDS.replace('and both also ', '')),
        'trait_gain': torch.tensor(6.0),
    }

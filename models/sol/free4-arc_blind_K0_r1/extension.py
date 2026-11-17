"""Compile additional hand-authored logical constraints and taxonomic facts.

This module only compiles declarations. No dataset, labels, optimization,
learned parameters, network services, or automatic model search are used.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def lines(name):
    for raw in (ROOT / name).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if line:
            yield line


def logical_rows():
    return [[s.strip() for s in line.split('|')]
            for name in ('logic.txt', 'logic_extra.txt') for line in lines(name)]


def atoms(expression):
    return [atom.strip().lstrip('!?') for group in expression.split(';')
            for atom in group.split(',') if atom.strip()]


def feature_phrases():
    from relational import feature_phrases as relational_phrases
    yield from relational_phrases()
    from proportions import feature_phrases as proportional_phrases
    yield from proportional_phrases()
    from traits import feature_phrases as trait_phrases
    yield from trait_phrases()
    from generation3 import claim_rows
    for row in claim_rows():
        for expression in row[:4]:
            yield from atoms(expression)
    for row in logical_rows():
        for expression in row[:3]:
            yield from atoms(expression)
    for line in lines('taxonomy.txt'):
        parent, members = line.split('|')
        yield parent.strip()
        for member in members.split(','):
            yield member.strip()


def extra_text():
    from relational import authored_text
    yield from authored_text()
    from proportions import authored_text as proportional_text
    yield from proportional_text()
    from traits import authored_text as trait_text
    yield from trait_text()
    yield from feature_phrases()
    yield from lines('properties.txt')
    yield from lines('exceptions.txt')


def inherited_records():
    parents = {}
    for line in lines('taxonomy.txt'):
        parent, members = (s.strip() for s in line.split('|'))
        for member in members.split(','):
            parents.setdefault(member.strip(), set()).add(parent)
    ancestors = {entity: set(values) for entity, values in parents.items()}
    for step in range(8):
        ancestors = {entity: values.union(*(ancestors.get(p, set()) for p in values))
                     for entity, values in ancestors.items()}
    for entity, categories in ancestors.items():
        assert entity not in categories, ('Taxonomy cycle', entity)
        assert all(parents.get(category, set()).issubset(categories) for category in categories), ('Incomplete ancestry', entity)
    exclusions = {tuple(cell.strip() for cell in line.split('|')) for line in lines('exceptions.txt')}
    properties = {}
    for line in lines('properties.txt'):
        category, request, answer = (s.strip() for s in line.split('|'))
        properties.setdefault(category, []).append((request, answer))
    for entity, category, request in exclusions:
        assert category in ancestors.get(entity, set()), (entity, category)
        assert any(prompt == request for prompt, answer in properties.get(category, [])), (category, request)
    result = []
    for category, entries in sorted(properties.items()):
        for request, answer in entries:
            result.append((category + ' ' + request, answer, 1.0))
    for entity, categories in sorted(ancestors.items()):
        for category in sorted(categories):
            # A separate membership assertion supports both directions without
            # conflating different named entities into a single lexical feature.
            result.append((entity + ' classification type group', category, 0.85))
            for request, answer in properties.get(category, []):
                if (entity, category, request) not in exclusions:
                    result.append((entity + ' ' + request, answer, 0.85))
    return result


def compile_logic(selector, count):
    group_index = {(): 1}
    groups = [(), ()]  # 0: identity/true padding; 1: false padding.

    def group_id(alternatives):
        key = tuple(sorted(set(selector(atom.strip().lstrip('!?')) for atom in alternatives)))
        if key not in group_index:
            group_index[key] = len(groups)
            groups.append(key)
        return group_index[key]

    def parse(expression):
        result = []
        for group in expression.split(';'):
            if group.strip():
                result.append(group_id(group.split(',')))
        return result

    required, forbidden, focused_required, focused_forbidden = [], [], [], []
    positive, negative, confidence = [], [], []
    for question, support, contradiction, strength in logical_rows():
        need, avoid, focus_need, focus_avoid = [], [], [], []
        for group in question.split(';'):
            group = group.strip()
            alternatives = [atom.strip() for atom in group.split(',')]
            prefixes = [atom[:len(atom) - len(atom.lstrip('!?'))] for atom in alternatives]
            assert all(set(prefix) == set(prefixes[0]) for prefix in prefixes), group
            focused = '?' in prefixes[0]
            negated = '!' in prefixes[0]
            target = (focus_avoid if negated else focus_need) if focused else (avoid if negated else need)
            target.append(group_id(alternatives))
        required.append(need)
        forbidden.append(avoid)
        focused_required.append(focus_need)
        focused_forbidden.append(focus_avoid)
        positive.append(parse(support))
        negative.append(parse(contradiction))
        confidence.append(float(strength))
    matrix = torch.zeros((len(groups), count), dtype=torch.float32)
    for i, group in enumerate(groups):
        matrix[i, list(group)] = 1

    def padded(rows, fill):
        width = max(1, max(len(row) for row in rows))
        return torch.tensor([row + [fill] * (width - len(row)) for row in rows], dtype=torch.int64)

    return {
        'logic_groups': matrix,
        'logic_required': padded(required, 0),
        'logic_forbidden': padded(forbidden, 1),
        'logic_focused_required': padded(focused_required, 0),
        'logic_focused_forbidden': padded(focused_forbidden, 1),
        'logic_positive': padded(positive, 0),
        'logic_negative': padded(negative, 0),
        'logic_has_positive': torch.tensor([len(row) > 0 for row in positive]),
        'logic_has_negative': torch.tensor([len(row) > 0 for row in negative]),
        'logic_confidence': torch.tensor(confidence),
    }

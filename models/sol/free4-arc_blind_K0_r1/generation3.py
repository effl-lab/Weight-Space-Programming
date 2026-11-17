"""Compile authored assertion schemas and literal experimental-role controls.
Compilation only: no optimization, measurements of accuracy, or external data.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def claim_rows():
    rows = []
    for raw in (ROOT / 'claims.txt').read_text().splitlines():
        text = raw.split('#', 1)[0].strip()
        if text:
            row = [part.strip() for part in text.split('|')]
            assert len(row) == 5, text
            rows.append(row)
    return rows


def compile_claims(selector, count, vocabulary):
    groups = [()]
    lookup = {(): 0}

    def parse(expression):
        indices = []
        for clause in expression.split(';'):
            if clause.strip():
                parts = [part.strip() for part in clause.split(',')]
                assert all(parts), ('Empty assertion alternative', expression)
                assert all(not part.startswith(('!', '?')) for part in parts), ('Unexpected assertion prefix', expression)
                concepts = tuple(sorted(set(selector(part) for part in parts)))
                if concepts not in lookup:
                    lookup[concepts] = len(groups)
                    groups.append(concepts)
                indices.append(lookup[concepts])
        return indices

    rows = claim_rows()
    subjects, contexts, forbidden_contexts, positives, negatives = [], [], [], [], []
    for subject, context, positive, negative, confidence in rows:
        subjects.append(parse(subject))
        required, forbidden = [], []
        for clause in context.split(';'):
            if clause.strip():
                alternatives = [part.strip() for part in clause.split(',')]
                excluded = [part.startswith('!') for part in alternatives]
                assert all(flag == excluded[0] for flag in excluded), ('Mixed context polarity', clause)
                clean = [part[1:].strip() if flag else part for part, flag in zip(alternatives, excluded)]
                target = forbidden if excluded[0] else required
                target.extend(parse(','.join(clean)))
        contexts.append(required)
        forbidden_contexts.append(forbidden)
        positives.append(parse(positive))
        negatives.append(parse(negative))
    matrix = torch.zeros(len(groups), count)
    for i, group in enumerate(groups):
        matrix[i, list(group)] = 1

    def pad(values):
        width = max(1, max(len(row) for row in values))
        return torch.tensor([row + [0] * (width - len(row)) for row in values], dtype=torch.int64)

    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def words(text):
        tensor = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            assert word in index, ('Missing raw parser word', word)
            tensor[index[word]] = 1
        return tensor

    queries = torch.zeros(2, count)
    queries[0, selector('independent variable')] = 1
    queries[1, selector('dependent variable')] = 1
    return {
        'claim_groups': matrix,
        'claim_subject': pad(subjects),
        'claim_context': pad(contexts),
        'claim_forbidden_context': pad(forbidden_contexts),
        'claim_positive': pad(positives),
        'claim_negative': pad(negatives),
        'claim_has_positive': torch.tensor([len(row) > 0 for row in positives]),
        'claim_has_negative': torch.tensor([len(row) > 0 for row in negatives]),
        'claim_strength': torch.tensor([float(row[4]) for row in rows]),
        'absence_word': words('absent missing lacks lack'),
        'option_negator': words("not no never cannot can't isn't aren't wasn't weren't doesn't don't didn't lacks lack lacking without"),
        'option_neg_boundary': words('and or but whereas although however while yet then that which because if when'),
        'option_neg_exception': words('only just merely necessarily more less fewer'),
        'option_neg_limits': torch.tensor([12, 6]),
        'numeric_bound_word': words('less fewer more greater smaller larger above below under over least most between range ranges nearly almost minimum maximum up'),
        'numeric_unsupported_unit': words('square squared cm2 mm2 m2 km2 hectare hectares acre acres celsius fahrenheit kelvin kelvins pascal pascals pa kpa kilopascal kilopascals atm atmosphere atmospheres bar bars torr psi percent percentage percentages ppm pound pounds lb lbs ounce ounces oz inch inches foot feet yard yards mile miles gallon gallons quart quarts pint pints c f k'),
        'effect_word': words('effect effects influence influences impact impacts'),
        'affect_verb': words('affect affects affected affecting influence influences'),
        'literal_of': words('of'),
        'literal_on': words('on upon'),
        'literal_how': words('how'),
        'role_boundary': words('which what why where when whether while because if using used use by with during to from compared comparing'),
        'role_query': queries,
        'role_gains': torch.tensor([0.75, 0.35, 8.0, 6.0]),
        'manipulate_verb': words('vary varies varied varying manipulate manipulates manipulated manipulating change changes changed changing adjust adjusts adjusted adjusting'),
        'measure_verb': words('measure measures measured measuring record records recorded recording count counts counted counting'),
        'passive_auxiliary': words('is are was were be been being'),
        'action_boundary': words('and or but then while when which what why how using by with during to'),
        'geometry_rectangle': words('rectangular cuboid box prism'),
        'geometry_cube': words('cube cubes'),
        'geometry_edge': words('side sides edge edges length'),
        'geometry_forbid': words('cylinder cylindrical sphere spherical cone conical radius diameter triangular hollow pyramid pyramidal'),
    }

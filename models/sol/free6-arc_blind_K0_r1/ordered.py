"""Compile authored ordered pathways; no training data or parameter estimation."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

# Declared grammar, not words learned from questions. Option fillers only
# express list structure or stage transitions; substantive extra claims fail.
ORDER_QUERY = 'sequence,order,path,route,flow,arrangement,arranged,chronological,progression,ordered'
OPTION_FILLER = 'next then finally subsequently followed follows following and into to from through toward by with of the a an stage stages step steps process processes'
# These words carry ordering semantics that a bare list verifier cannot parse;
# block them even when the ordinary lexical encoder treats them as stopwords.
OPTION_FORBIDDEN = 'before after backward backwards reverse first second third fourth fifth sixth seventh eighth ninth tenth beginning starting begins starts begin start develops develop developing becomes become becoming forms form forming changes change changing'
ANCHOR_WORDS = 'beginning starting begins starts begin start'
PARSER_WORDS = OPTION_FILLER + ' ' + OPTION_FORBIDDEN + ' ' + ANCHOR_WORDS + ' with at'


def authored_text():
    yield PARSER_WORDS


def rows():
    result = []
    for raw in (ROOT / 'order_rules.txt').read_text().splitlines():
        text = raw.split('#', 1)[0].strip()
        if text:
            parts = [s.strip() for s in text.split('|')]
            assert len(parts) == 5, text
            result.append(parts)
    return result


def atoms(text):
    return [s.strip() for group in text.split(';') for s in group.split(',') if s.strip() and s.strip() != '*']


def feature_phrases():
    yield from atoms(ORDER_QUERY)
    for required, up, down, circular, nodes in rows():
        yield from atoms(required)
        yield from atoms(up)
        yield from atoms(down)
        yield from atoms(nodes)


def compile_orders(selector, sequence, count, vocab):
    groups = [(), ()]  # zero is true padding; one is an empty/false group
    group_ids = {(): 1}

    def parse(text):
        indices = []
        for group in text.split(';'):
            values = tuple(sorted(set(selector(atom.strip()) for atom in group.split(',') if atom.strip() and atom.strip() != '*')))
            if values not in group_ids:
                group_ids[values] = len(groups)
                groups.append(values)
            indices.append(group_ids[values])
        return indices

    # Concept names may be expressed by any of the explicitly authored synonyms.
    equivalents = {}
    for raw in (ROOT / 'concepts.txt').read_text().splitlines():
        clean = raw.split('#', 1)[0].strip()
        if clean:
            alternatives = [s.strip() for s in clean.split('|', 1)[1].split(';')]
            for alternative in alternatives:
                equivalents.setdefault(selector(alternative), set()).update(alternatives)

    patterns = [()]
    pattern_ids = {(): 0}
    templates, requirements, up_groups, down_groups, defaults = [], [], [], [], []
    for required, up, down, circular, nodes in rows():
        req = parse(required)
        up_ids = parse(up.replace('*,', '').replace('*', ''))
        down_ids = parse(down)
        entries = []
        for node in nodes.split(';'):
            options = set(atoms(node))
            for text in list(options):
                options.update(equivalents.get(selector(text), ()))
            ids = []
            for text in sorted(options):
                seq = sequence(text)
                assert 1 <= len(seq) <= 4, (text, seq)
                if seq not in pattern_ids:
                    pattern_ids[seq] = len(patterns)
                    patterns.append(seq)
                ids.append(pattern_ids[seq])
            entries.append(sorted(set(ids)))
        rotations = range(len(entries)) if circular == '1' else range(1)
        for rotation in rotations:
            templates.append(entries[rotation:] + entries[:rotation])
            requirements.append(req)
            up_groups.append(up_ids)
            down_groups.append(down_ids)
            defaults.append('*' in up)

    group_matrix = torch.zeros(len(groups), count)
    for i, group in enumerate(groups):
        group_matrix[i, list(group)] = 1

    def pad(rows):
        width = max(1, max(map(len, rows)))
        return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.int64)

    entity_width = max(map(len, templates))
    alternative_width = max(len(entry) for row in templates for entry in row)
    node_patterns = torch.zeros((len(templates), entity_width, alternative_width), dtype=torch.int64)
    for i, row in enumerate(templates):
        for j, ids in enumerate(row):
            node_patterns[i, j, :len(ids)] = torch.tensor(ids)
    pattern_table = torch.tensor([list(p) + [0] * (4 - len(p)) for p in patterns], dtype=torch.int64)
    incidence = torch.zeros(len(patterns), len(templates))
    for i, row in enumerate(templates):
        for ids in row:
            incidence[ids, i] = 1
    index = {word: i + 2 for i, word in enumerate(vocab)}

    def words(text):
        table = torch.zeros(len(vocab) + 2)
        for word in text.split():
            assert word in index, ('Missing ordered parser word', word)
            table[index[word]] = 1
        return table

    query = torch.zeros(count)
    query[[selector(text) for text in atoms(ORDER_QUERY)]] = 1
    return {
        'order_query': query,
        'order_incidence': incidence,
        'order_filler': words(OPTION_FILLER),
        'order_forbidden': words(OPTION_FORBIDDEN),
        'order_anchor': words(ANCHOR_WORDS),
        'order_anchor_link': words('with at'),

        'order_groups': group_matrix,
        'order_required': pad(requirements),
        'order_up': pad(up_groups),
        'order_down': pad(down_groups),
        'order_default': torch.tensor(defaults),
        'order_patterns': pattern_table,
        'order_width': torch.tensor([len(p) for p in patterns]),
        'order_nodes': node_patterns,
        # Three named stages distinguish a true ordering task from mere mention.
        'order_threshold': torch.tensor(3),
        'order_gain': torch.tensor(16.0),
    }

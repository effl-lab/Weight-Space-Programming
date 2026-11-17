"""Compile reverse use of the explicitly authored three-valued property cards.
No questions, measured answer accuracy, fitting, or external data are used.
"""
import torch
from traits import query_rows, atoms

CUES = ('have in common', 'has in common', 'share', 'shared', 'both', 'similarity',
        'common characteristic', 'common property', 'same characteristic',
        'same property', 'similar characteristic')
FORBID = ('unlike different difference differences more less greater fewer least most '
          'than except not never neither either only before after if unless when '
          'compared compare comparing evolved ancestor related relationship')
FILLER = 'all and both also possess possesses possessing exhibit exhibits having'
QUESTION_FILLER = ('trait traits characteristic characteristics property properties '
                   'feature features statement statements similarity similarities '
                   'common shared share shares similar both and')

# Universal list requests are deliberately separate from ordinary single-name
# property questions. The grammar does not license unverified domain qualifiers
# such as "of plants", existential/disjunctive requests, or background clauses.
LIST_NOUNS = 'pair pairs group groups list lists set sets combination combinations'
LIST_UNIVERSAL = 'both all each every only exclusively'
LIST_PAIR = 'pair pairs both'
LIST_QUESTION = ('which what of at these those the following listed given a an '
                 'pair pairs group groups list lists set sets combination combinations '
                 'both all each every only exclusively member members item items '
                 'entry entries example examples things '
                 'are is be have has do does can contain contains include includes '
                 'consist consists and')
LIST_OPTION = 'a an the and'
LIST_FORBIDDEN = (FORBID.replace(' only ', ' ') +
                  ' or without lacks lacking rather instead respectively')
LIST_MEMBERSHIP = 'contain contains include includes consist consists'
LIST_POSSESSION = 'have has contain contains include includes'
IDENTITY_ATTRIBUTES = ('metal', 'element', 'compound', 'mixture', 'igneous',
                       'sedimentary', 'metamorphic', 'solid_room', 'liquid_room', 'gas_room')


def feature_phrases():
    yield from CUES


def authored_text():
    yield from CUES
    yield FORBID
    yield FILLER
    yield QUESTION_FILLER
    yield LIST_NOUNS
    yield LIST_UNIVERSAL
    yield LIST_PAIR
    yield LIST_QUESTION
    yield LIST_OPTION
    yield LIST_FORBIDDEN
    yield LIST_MEMBERSHIP
    yield LIST_POSSESSION


def compile_shared(selector, sequence, count, vocabulary, tables):
    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def words(text):
        result = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            result[index[word]] = 1
        return result

    # Option coverage is restricted to actually completed property frames.
    entries = set()
    frame_entries = set()
    declarations = list(query_rows())
    for row, (name, positive, negative, forbidden) in enumerate(declarations):
        for direction, expression in enumerate((positive, negative)):
            for text in atoms(expression):
                feature = selector(text)
                frame_entries.add((feature, direction * len(declarations) + row))
                for lexical in sequence(text):
                    entries.add((lexical, feature))
    entries = sorted(entries)
    frame_entries = sorted(frame_entries)
    frames = torch.sparse_coo_tensor(
        torch.tensor(frame_entries, dtype=torch.int64).t(), torch.ones(len(frame_entries)),
        (count, 2 * len(declarations))).coalesce()
    coverage = torch.sparse_coo_tensor(
        torch.tensor(entries, dtype=torch.int64).t(), torch.ones(len(entries)),
        (int(tables['base']) + 1, count)).coalesce()

    # Actual name occurrences, not taxonomic dominance across a whole list,
    # identify the entities. A general name is suppressed only inside the
    # same longer name span, never when mentioned separately in the question.
    pattern_entities = torch.zeros(len(tables['trait_patterns']), len(tables['trait_entity']))
    for entity, names in enumerate(tables['trait_entity_patterns'].tolist()):
        for name in names:
            if name:
                pattern_entities[name, entity] = 1
    assert (pattern_entities.sum(-1) <= 1).all(), 'Ambiguous entity name pattern'
    cue = torch.zeros(count)
    for text in CUES:
        cue[selector(text)] = 1
    return {'shared_coverage': coverage, 'shared_frames': frames, 'shared_cue': cue,
            'shared_forbidden': words(FORBID), 'shared_filler': words(FILLER),
            'shared_pattern_entities': pattern_entities,
            'shared_question_filler': words(QUESTION_FILLER),
            'group_nouns': words(LIST_NOUNS),
            'group_universal': words(LIST_UNIVERSAL),
            'group_pair': words(LIST_PAIR),
            'group_question_filler': words(LIST_QUESTION),
            'group_option_filler': words(LIST_OPTION),
            'group_forbidden': words(LIST_FORBIDDEN),
            'group_only': words('only exclusively'),
            'group_and': words('and'),
            'group_membership': words(LIST_MEMBERSHIP),
            'group_possession': words(LIST_POSSESSION),
            'group_of': words('of'),
            'group_identity': torch.tensor([
                name.endswith('_class') or name in IDENTITY_ATTRIBUTES
                for name in dict.fromkeys(row[0] for row in declarations)
            ], dtype=torch.float32)}


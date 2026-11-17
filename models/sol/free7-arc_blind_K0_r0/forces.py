"""Explicit signed-vector addition for a bounded collinear force problem.
The declarations give axes, signs, and legal measurement/list grammar.
No labels or examples are read to construct these tensors.
"""
import torch

DIRECTIONS = (
    ('right rightward rightwards', 1, 1), ('left leftward leftwards', 1, -1),
    ('east eastward eastwards', 2, 1), ('west westward westwards', 2, -1),
    ('north northward northwards', 3, 1), ('south southward southwards', 3, -1),
    ('up upward upwards', 4, 1), ('down downward downwards', 4, -1),
)
BRIDGE = ('a an the to toward towards of with in acts acting acted force forces '
          'push pushes pushing pull pulls pulling points pointing directed direction '
          'horizontal horizontally vertical vertically magnitude is are was were '
          'be being been has have exerts exert exerted applies applied newton newtons n')
FILLER = ('a an the to toward towards of with in net resultant total force forces '
          'newton newtons n direction magnitude acts acting acting points pointing')
FORBIDDEN = ('not never no except incorrect false least unlikely unless whether '
             'between range ranges minimum maximum nearly almost approximate approximately '
             'circular angle angles degree degrees diagonal diagonally perpendicular '
             'equilibrium balances balance balanced accelerate acceleration accelerates '
             'initial initially final finally change changed changing increase decreases '
             'rotate rotation torque moment momentum work energy spring compression')
QUERIES = ('net force', 'resultant force', 'combined force', 'total force')
# An exact sum is not justified by recognizing two measurements while ignoring
# arbitrary additional context (for example, an unmeasured drag force).
# Named additional force mechanisms are deliberately outside this whitelist.
QUESTION = (BRIDGE + ' and plus if on by at from it its this that these those '
            'what which how much size strength value resulting net resultant '
            'combined total sum experience experiences experienced act exert '
            'applied applying apply applies simultaneous simultaneously '
            'two three object objects body bodies block blocks box boxes '
            'crate crates cart carts car cars truck trucks wagon wagons '
            'rock rocks stone stones ball balls sled sleds rope ropes '
            'student students person people child children worker workers '
            'team teams tug tugs tugging opposite opposing oppositely '
            'floor surface flat smooth frictionless moving motion rest')


def feature_phrases():
    yield from QUERIES


def authored_text():
    yield from QUERIES
    yield BRIDGE
    yield FILLER
    yield FORBIDDEN
    yield QUESTION
    for text, axis, sign in DIRECTIONS:
        yield text


def compile_forces(selector, count, vocabulary):
    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def words(text):
        result = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            result[index[word]] = 1
        return result

    direction_axis = torch.zeros(len(vocabulary) + 2, dtype=torch.int64)
    direction_sign = torch.zeros(len(vocabulary) + 2)
    for text, axis, sign in DIRECTIONS:
        for word in text.split():
            direction_axis[index[word]] = axis
            direction_sign[index[word]] = sign
    query = torch.zeros(count)
    for text in QUERIES:
        query[selector(text)] = 1
    return {'force_axis': direction_axis, 'force_sign': direction_sign,
            'force_bridge': words(BRIDGE), 'force_filler': words(FILLER),
            'force_forbidden': words(FORBIDDEN), 'force_query': query,
            'force_count_noun': words('force forces'),
            'force_question_filler': words(QUESTION)}

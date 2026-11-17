"""Compile authored, finite partial-order comparisons; no fitting or search."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def rows(name):
    for raw in (ROOT / name).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if line:
            yield [part.strip() for part in line.split('|')]


def atoms(text):
    return [atom.strip() for clause in text.split(';') for atom in clause.split(',') if atom.strip()]


UP = 'more most greater greatest higher highest larger largest faster fastest stronger strongest maximum increase increases increasing maximize maximizes quickest rapid quickest longest hotter hottest warmer warmest'
DOWN = 'less least fewer lower lowest smaller smallest slower slowest weaker weakest minimum decrease decreases decreasing reduce reduces reducing minimize minimizes shortest colder coldest cooler coolest'
FORBID = 'not never cannot unless either neither reversed reversal double doubles doubled triple triples tripled half halves twice times factor negative zero'

# Request | numerator dimension | denominator/second dimension | exponent |
# required measurement count | contexts outside the law's elementary scope.
# Dimensions: g, mL, m, s, N, J, W, V, A, ohm, Hz, m/s, constant, kelvin.
WORK_EXCLUSIONS = 'angle angles degree degrees perpendicular opposite circular centripetal'
POTENTIAL_EXCLUSIONS = 'moon moons planet planets world worlds mars jupiter saturn venus mercury uranus neptune spring springs elastic field fields'
CANDIDATE_LAWS = [
    ('density', 0, 1, -1, 2, ''),
    ('speed,velocity', 2, 3, -1, 2, ''),
    ('acceleration', 4, 0, -1, 2, ''),
    ('work', 4, 2, 1, 2, WORK_EXCLUSIONS),
    ('power', 5, 3, -1, 2, ''),
    ('current,electric current', 7, 9, -1, 2, ''),
    ('voltage', 8, 9, 1, 2, ''),
    ('resistance', 7, 8, -1, 2, ''),
    ('energy', 6, 3, 1, 2, ''),
    ('momentum', 0, 11, 1, 2, 'relativity relativistic'),
    ('kinetic energy', 0, 11, 2, 2, 'relativity relativistic photon photons'),
    ('gravitational potential energy', 0, 2, 1, 2, POTENTIAL_EXCLUSIONS),
    ('mass', 0, 12, 0, 1, ''),
    ('volume', 1, 12, 0, 1, ''),
    ('distance,length,height,wavelength', 2, 12, 0, 1, ''),
    ('time,period', 3, 12, 0, 1, ''),
    ('force,weight', 4, 12, 0, 1, ''),
    ('energy,work,kinetic energy,gravitational potential energy', 5, 12, 0, 1, ''),
    ('power', 6, 12, 0, 1, ''),
    ('voltage', 7, 12, 0, 1, ''),
    ('current,electric current', 8, 12, 0, 1, ''),
    ('resistance', 9, 12, 0, 1, ''),
    ('frequency', 10, 12, 0, 1, ''),
    ('speed,velocity', 11, 12, 0, 1, ''),
    ('temperature,hotter,hottest,warmer,warmest,colder,coldest,cooler,coolest', 13, 12, 0, 1,
     'difference differences change changes increase increases increased increasing decrease decreases decreased decreasing rise rises rising drop drops dropped dropping reduction reductions gain gains gained loss losses falls fallen falling delta warming cooling negative minus'),
]
# Affine conversion to absolute Kelvin, not a multiplicative unit law.
TEMPERATURE_UNITS = [
    ('celsius c', 1.0, 273.15),
    ('fahrenheit f', 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    ('kelvin kelvins k', 1.0, 0.0),
]
TEMPERATURE_BRIDGE = 'degree degrees of'
SPEED_BRIDGE = 'per each every a'
SPEED_UNITS = [('mps', 1.0), ('kph', 1000.0 / 3600.0), ('kmh', 1000.0 / 3600.0), ('mph', 0.44704)]

# An absolute, nonrelativistic kinetic-energy calculation, separately guarded
# from the ordinal laws. Input mass is stored in grams; output is in joules.
KINETIC_ENERGY_UNITS = [('joule joules j', 1.0), ('kilojoule kilojoules kj', 1000.0)]
KINETIC_ENERGY_FORBIDDEN = (
    FORBID.replace(' zero', '') +
    ' relativity relativistic photon photons specific total combined collision collisions'
    ' ratio fraction percentage percent efficiency'
    ' initial final before after change changes changed changing difference differences'
    ' increase increases increased increasing decrease decreases decreased decreasing'
    ' gain gains gained lose loses lost loss losses negative minus'
)
KINETIC_OTHER_TARGETS = 'mass speed velocity momentum force work power energy distance height length volume density pressure acceleration time duration long period frequency wavelength temperature potential'


def feature_phrases():
    for request, left, right, exponent, measurements, forbidden in CANDIDATE_LAWS:
        yield from atoms(request)
    for domain, required, forbidden in rows('comparison_topics.txt'):
        yield from atoms(required)
        yield from atoms(forbidden)
    for domain, axis, positive, negative in rows('comparison_axes.txt'):
        yield from atoms(positive)
        yield from atoms(negative)


def authored_text():
    yield from feature_phrases()
    yield UP + ' ' + DOWN + ' ' + FORBID + ' likely probable probably'
    yield SPEED_BRIDGE
    yield TEMPERATURE_BRIDGE
    for units, scale, offset in TEMPERATURE_UNITS:
        yield units
    yield ' '.join(unit for unit, scale in SPEED_UNITS)
    yield KINETIC_ENERGY_FORBIDDEN
    yield KINETIC_OTHER_TARGETS
    for units, scale in KINETIC_ENERGY_UNITS:
        yield units
    for request, left, right, exponent, measurements, forbidden in CANDIDATE_LAWS:
        yield forbidden


def compile_comparisons(selector, count, vocabulary):
    topics = list(rows('comparison_topics.txt'))
    domains = {row[0]: i for i, row in enumerate(topics)}
    groups = [()]
    lookup = {(): 0}

    def parse(text):
        result = []
        for clause in text.split(';'):
            if clause.strip():
                key = tuple(sorted(set(selector(atom.strip()) for atom in clause.split(','))))
                if key not in lookup:
                    lookup[key] = len(groups)
                    groups.append(key)
                result.append(lookup[key])
        return result

    required = [parse(row[1]) for row in topics]
    forbidden = [parse(row[2]) for row in topics]
    axis_rows = list(rows('comparison_axes.txt'))
    positive = [parse(row[2]) for row in axis_rows]
    negative = [parse(row[3]) for row in axis_rows]
    matrix = torch.zeros(len(groups), count)
    for i, features in enumerate(groups):
        matrix[i, list(features)] = 1

    def pad(values):
        width = max(1, max(len(row) for row in values))
        return torch.tensor([row + [0] * (width - len(row)) for row in values], dtype=torch.int64)

    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def words(text):
        tensor = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            assert word in index, ('Missing comparison word', word)
            tensor[index[word]] = 1
        return tensor

    incidence = torch.zeros(len(axis_rows), len(topics))
    for i, row in enumerate(axis_rows):
        incidence[i, domains[row[0]]] = 1
    laws = CANDIDATE_LAWS
    numerical_queries = torch.zeros(len(laws), count)
    for i, (request, left, right, exponent, measurements, law_exclusions) in enumerate(laws):
        for name in atoms(request):
            numerical_queries[i, selector(name)] = 1
    speed_units = torch.zeros(len(vocabulary) + 2)
    for unit, scale in SPEED_UNITS:
        speed_units[index[unit]] = scale
    law_forbidden = torch.stack([words(row[5]) for row in laws], dim=-1)
    temperature_scale = torch.zeros(len(vocabulary) + 2)
    temperature_offset = torch.zeros_like(temperature_scale)
    for names, scale, offset in TEMPERATURE_UNITS:
        for unit in names.split():
            temperature_scale[index[unit]] = scale
            temperature_offset[index[unit]] = offset
    kinetic_query = torch.zeros(count)
    kinetic_query[selector('kinetic energy')] = 1
    kinetic_output_scale = torch.zeros(len(vocabulary) + 2)
    for names, scale in KINETIC_ENERGY_UNITS:
        for unit in names.split():
            kinetic_output_scale[index[unit]] = scale
    return {
        'ke_query': kinetic_query,
        'ke_keyword': torch.tensor(index['kinetic'], dtype=torch.int64),
        'ke_in_word': torch.tensor(index['in'], dtype=torch.int64),
        'ke_other_target': words(KINETIC_OTHER_TARGETS),
        'ke_forbidden': words(KINETIC_ENERGY_FORBIDDEN),
        'ke_output_scale': kinetic_output_scale,
        'ke_output_words': words('energy kinetic'),
        'ke_coefficient': torch.tensor(0.5 / 1000.0),
        'cand_query': numerical_queries,
        'cand_left': torch.tensor([row[1] for row in laws], dtype=torch.int64),
        'cand_right': torch.tensor([row[2] for row in laws], dtype=torch.int64),
        'cand_exponent': torch.tensor([row[3] for row in laws]),
        'cand_measurement_count': torch.tensor([row[4] for row in laws]),
        # Nonnegative numerators and products can be zero. Divisors must be
        # positive; the massive-body laws additionally require positive mass.
        'cand_left_zero': torch.tensor([row[0] not in ('momentum', 'kinetic energy', 'gravitational potential energy') for row in laws]),
        'cand_right_zero': torch.tensor([row[3] >= 0 for row in laws]),
        'cand_roundoff': torch.tensor(8 * torch.finfo(torch.float32).eps),
        'cand_tiny': torch.tensor(torch.finfo(torch.float32).tiny),
        'cand_scope_forbidden': words(FORBID.replace(' zero', '')),
        'cand_forbidden': law_forbidden,
        'cand_speed_bridge': words(SPEED_BRIDGE), 'cand_speed_units': speed_units,
        'cand_temperature_bridge': words(TEMPERATURE_BRIDGE),
        'cand_temperature_scale': temperature_scale,
        'cand_temperature_offset': temperature_offset,
        'cand_gain': torch.tensor(14.0),
        'cmp_groups': matrix,
        'cmp_required': pad(required), 'cmp_excluded': pad(forbidden),
        'cmp_positive': pad(positive), 'cmp_negative': pad(negative),
        'cmp_axis_domain': incidence,
        'cmp_up': words(UP), 'cmp_down': words(DOWN),
        'cmp_likely': words('likely probable probably'),
        'cmp_forbidden': words(FORBID),
        'cmp_gain': torch.tensor(7.0),
    }

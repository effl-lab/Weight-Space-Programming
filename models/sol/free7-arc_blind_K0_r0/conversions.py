"""Compile declared unit definitions and a bounded conversion grammar.
No data, optimization, or numerical answer examples are used. Unit definitions
are physical/metric conventions, not values estimated from questions.
"""
from fractions import Fraction
import torch

# dimension, multiplicative scale to the reference unit, affine offset,
# semicolon-separated raw token spellings (up to three contiguous tokens).
# References: meter, gram, second, liter, square meter, kelvin, newton,
# joule, watt, pascal, volt, ampere, ohm, hertz.
UNITS = [
    (1, 1e-9, 0, 'nanometer;nanometers;nanometre;nanometres;nm'),
    (1, 1e-6, 0, 'micrometer;micrometers;micrometre;micrometres'),
    (1, 0.001, 0, 'millimeter;millimeters;millimetre;millimetres;mm'),
    (1, 0.01, 0, 'centimeter;centimeters;centimetre;centimetres;cm'),
    (1, 0.1, 0, 'decimeter;decimeters;decimetre;decimetres;dm'),
    (1, 1, 0, 'meter;meters;metre;metres;m'),
    (1, 1000, 0, 'kilometer;kilometers;kilometre;kilometres;km'),
    (1, 0.0254, 0, 'inch;inches'),
    (1, 0.3048, 0, 'foot;feet;ft'),
    (1, 0.9144, 0, 'yard;yards;yd'),
    (1, 1609.344, 0, 'mile;miles;mi'),
    (2, 0.000001, 0, 'microgram;micrograms;mcg'),
    (2, 0.001, 0, 'milligram;milligrams;mg'),
    (2, 1, 0, 'gram;grams;g'),
    (2, 1000, 0, 'kilogram;kilograms;kg'),
    (2, 1000000, 0, 'tonne;tonnes;metric ton;metric tons'),
    (2, 453.59237, 0, 'pound;pounds;lb;lbs'),
    (2, 28.349523125, 0, 'ounce;ounces;oz'),
    (3, 0.001, 0, 'millisecond;milliseconds;ms'),
    (3, 1, 0, 'second;seconds;sec;s'),
    (3, 60, 0, 'minute;minutes;min'),
    (3, 3600, 0, 'hour;hours;hr;h'),
    (3, 86400, 0, 'day;days'),
    (3, 604800, 0, 'week;weeks'),
    (4, 0.001, 0, 'milliliter;milliliters;millilitre;millilitres;ml;cubic centimeter;cubic centimeters;cubic centimetre;cubic centimetres;centimeter cubed;centimeters cubed;centimetre cubed;centimetres cubed;cubic cm;cm cubed;cm3;cc'),
    (4, 1, 0, 'liter;liters;litre;litres;l;cubic decimeter;cubic decimeters;cubic decimetre;cubic decimetres;dm3'),
    (4, 1000, 0, 'cubic meter;cubic meters;cubic metre;cubic metres;meter cubed;meters cubed;metre cubed;metres cubed;cubic m;m cubed;m3'),
    (4, 0.000001, 0, 'cubic millimeter;cubic millimeters;cubic millimetre;cubic millimetres;cubic mm;mm cubed;mm3'),
    (5, 0.000001, 0, 'square millimeter;square millimeters;square millimetre;square millimetres;millimeter squared;millimeters squared;square mm;mm squared;mm2'),
    (5, 0.0001, 0, 'square centimeter;square centimeters;square centimetre;square centimetres;centimeter squared;centimeters squared;centimetre squared;centimetres squared;square cm;cm squared;cm2'),
    (5, 1, 0, 'square meter;square meters;square metre;square metres;meter squared;meters squared;metre squared;metres squared;square m;m squared;m2'),
    (5, 1000000, 0, 'square kilometer;square kilometers;square kilometre;square kilometres;square km;km squared;km2'),
    (5, 10000, 0, 'hectare;hectares'),
    (6, 1, 273.15, 'celsius;centigrade;c;degree celsius;degrees celsius;degree centigrade;degrees centigrade;degree c;degrees c;degrees of celsius'),
    (6, 5.0 / 9.0, 273.15 - 32 * 5.0 / 9.0, 'fahrenheit;f;degree fahrenheit;degrees fahrenheit;degree f;degrees f;degrees of fahrenheit'),
    (6, 1, 0, 'kelvin;kelvins;k;degree kelvin;degrees kelvin;degree k;degrees k'),
    (7, 1, 0, 'newton;newtons;n'),
    (7, 1000, 0, 'kilonewton;kilonewtons;kn'),
    (8, 1, 0, 'joule;joules;j'),
    (8, 1000, 0, 'kilojoule;kilojoules;kj'),
    (9, 1, 0, 'watt;watts;w'),
    (9, 1000, 0, 'kilowatt;kilowatts;kw'),
    (10, 1, 0, 'pascal;pascals;pa'),
    (10, 1000, 0, 'kilopascal;kilopascals;kpa'),
    (10, 1000000, 0, 'megapascal;megapascals'),
    (10, 100, 0, 'hectopascal;hectopascals;hpa;millibar;millibars'),
    (10, 100000, 0, 'bar;bars'),
    (10, 101325, 0, 'atmosphere;atmospheres;atm'),
    (11, 1, 0, 'volt;volts;v'),
    (11, 0.001, 0, 'millivolt;millivolts;mv'),
    (11, 1000, 0, 'kilovolt;kilovolts;kv'),
    (12, 1, 0, 'ampere;amperes;amp;amps'),
    (12, 0.001, 0, 'milliampere;milliamperes;milliamp;milliamps;ma'),
    (13, 1, 0, 'ohm;ohms'),
    (13, 1000, 0, 'kilohm;kilohms;kiloohm;kiloohms;kohm'),
    (13, 1000000, 0, 'megohm;megohms;megaohm;megaohms'),
    (14, 1, 0, 'hertz;hz'),
    (14, 1000, 0, 'kilohertz;khz'),
    (14, 1000000, 0, 'megahertz'),
]

# This whitelist admits a scalar conversion request, not arbitrary prose.
QUESTION_GRAMMAR = (
    'a an the this that it its is are was were be been being has have had '
    'how many much what which number value values in as to into of for '
    'equal equals equivalent equivalently equivalence convert converts converted '
    'converting conversion express expresses expressed expressing represents represent '
    'corresponds corresponding contain contains containing holds hold '
    'length distance height width depth thickness mass time duration volume area '
    'temperature force energy power pressure voltage current resistance frequency '
    'measurement measurements measured measures measure measuring reading reads read '
    'amount quantity quantities capacity liquid fluid water sample object container bottle tank rope '
    'wire road path line block box room air body long wide high deep thick tall '
    'weighs weight expressed written write stated given unit units '
    'approximately approximate about exactly exact precisely precise'
)
OPTION_GRAMMAR = 'a an the of is are equal equals equivalent to'
EQUIVALENCE = ('equal equals equivalent equivalently equivalence convert converts '
               'converted converting conversion express expresses expressed expressing '
               'corresponds corresponding')
ARTICLES = 'a an'
EXACT = 'exact exactly precisely precise'
# Count bounds, changes, rates, negation, and alternative representations are
# outside this one-quantity point conversion law. Missing words also fail the
# whitelist, so unknown units and appended transformations remain barriers.
FORBIDDEN = (
    'not never no cannot except incorrect false unlikely least most less more '
    'greater fewer smaller larger at between range ranges minimum maximum '
    'per each every negative minus below above under over before after '
    'initial initially final finally change changes changed changing difference '
    'differences increase increases increased increasing decrease decreases decreased '
    'decreasing cool cools cooled cooling heat heats heated heating warm warms '
    'cut cuts cutting break breaks split splits divide divides divided dividing '
    'half halves double doubles twice triple triples times factor ratio fraction '
    'percent percentage and or either neither unless if whether'
)


def authored_text():
    yield QUESTION_GRAMMAR
    yield OPTION_GRAMMAR
    yield EQUIVALENCE
    yield ARTICLES
    yield EXACT
    yield FORBIDDEN
    for kind, scale, offset, aliases in UNITS:
        yield aliases.replace(';', ' ')
        # Integer definitions and their exact reciprocal factors belong in
        # the vocabulary (e.g. the reciprocal of the authored nano scale).
        for value in (scale, offset, 1 / Fraction(str(scale))):
            if value >= 0 and value == int(value):
                yield str(int(value))


def compile_conversions(vocabulary, inherited_numbers):
    index = {word: i + 2 for i, word in enumerate(vocabulary)}
    base = len(vocabulary) + 2
    codes = {}
    for uid, (kind, scale, offset, aliases) in enumerate(UNITS, 1):
        for alias in aliases.split(';'):
            words = alias.split()
            assert 1 <= len(words) <= 3
            code = 0
            for word in words:
                code = code * base + index[word]
            previous = codes.setdefault(code, uid)
            assert previous == uid, ('Ambiguous conversion unit', alias)
    order = sorted(codes)

    def words(text):
        result = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            result[index[word]] = 1
        return result

    numbers = inherited_numbers.to(torch.float64).clone()
    for word, token_id in index.items():
        if word.isdigit():
            numbers[token_id] = float(word)
    return {
        'convert_number': numbers,
        'convert_base': torch.tensor(base, dtype=torch.int64),
        'convert_codes': torch.tensor(order + [base ** 3 + 1], dtype=torch.int64),
        'convert_ids': torch.tensor([codes[c] for c in order] + [0], dtype=torch.int64),
        'convert_dimension': torch.tensor([0] + [row[0] for row in UNITS]),
        'convert_scale': torch.tensor([1] + [row[1] for row in UNITS], dtype=torch.float64),
        'convert_offset': torch.tensor([0] + [row[2] for row in UNITS], dtype=torch.float64),
        'convert_question_grammar': words(QUESTION_GRAMMAR),
        'convert_option_grammar': words(OPTION_GRAMMAR),
        'convert_equivalence': words(EQUIVALENCE),
        'convert_article': words(ARTICLES),
        'convert_exact': words(EXACT),
        'convert_forbidden': words(FORBIDDEN),
        'convert_temperature_dimension': torch.tensor(6),
        'convert_roundoff': torch.tensor(64 * torch.finfo(torch.float64).eps, dtype=torch.float64),
    }

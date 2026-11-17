"""Authored count transformations and labelled microscope multiplication.

This compiles declarations and grammar into tensors, without training, example
questions, numerical fitting, optimization, or external resources.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

DIPLOID = 'body cell,somatic cell,diploid'
HAPLOID = 'gamete,sex cell,reproductive cell,sperm,egg cell,haploid'
DAUGHTER = 'daughter cell,new cell,resulting cell'
CHROMOSOME = 'chromosome'
MAGNIFICATION = 'magnification,magnifying power'
CELL = 'cell'
MICROSCOPE = 'microscope'
PARSER_WORDS = '''eyepiece ocular objective lens lenses magnification magnifications
power powers times x chromosomes chromosome per cell cells each in total overall
combined compound microscope magnifies magnified magnifying magnification
is are was were has have had a an the of with at for and equals equal to whose
change changes changed changing increase increases increased decrease decreases
decreased replaced replacing replacement switched switching switch ratio ratios
percent percentage fraction fractions area diameter radius estimates approximate
approximately about nearly around least most minimum maximum between range
ranges double doubled halve halved triploid tetraploid polyploid trisomy
mutation mutant abnormal cancer mitotic meiotic mitosis meiosis daughter daughters
body somatic diploid haploid gamete gametes reproductive egg eggs sperm offspring
pair pairs number count total exactly contains contain containing magnifying
somatic body cell chromosomes gamete chromosome count microscope magnification'''


def atoms(expression):
    return [word.strip() for word in expression.split(',') if word.strip()]


def feature_phrases():
    for expression in (DIPLOID, HAPLOID, DAUGHTER, CHROMOSOME, MAGNIFICATION, CELL, MICROSCOPE):
        yield from atoms(expression)


def authored_text():
    yield PARSER_WORDS
    # These strings are a mathematical vocabulary declaration, not examples.
    for value in range(1, 1001):
        yield f'{value}x x{value}'


def compile_counts(selector, count, vocabulary, lexical_tables, lexicon):
    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def feature_vector(expression):
        vector = torch.zeros(count)
        vector[[selector(text) for text in atoms(expression)]] = 1
        return vector

    def words(text, normalize=False):
        declared = set(text.split())
        # An empty canonical spelling means a grammatical stopword, not an
        # equivalence class of all skipped words or non-feature numerals.
        targets = {lexicon.get(word, word) for word in declared} - {''}
        result = torch.zeros(len(vocabulary) + 2)
        for word, idx in index.items():
            if word in declared or (normalize and lexicon.get(word, word) in targets):
                result[idx] = 1
        return result

    numeric = lexical_tables['number'].clone()
    for value in range(1, 1001):
        numeric[index[f'{value}x']] = value
        numeric[index[f'x{value}']] = value
    label = torch.zeros(len(vocabulary) + 2, dtype=torch.int64)
    label[words('eyepiece ocular', True) > 0] = 1
    label[words('objective', True) > 0] = 2
    return {
        'count_diploid': feature_vector(DIPLOID),
        'count_haploid': feature_vector(HAPLOID),
        'count_daughter': feature_vector(DAUGHTER),
        'count_chromosome': feature_vector(CHROMOSOME),
        'count_magnification': feature_vector(MAGNIFICATION),
        'count_cell': feature_vector(CELL),
        'count_microscope': feature_vector(MICROSCOPE),
        'count_chromosome_word': words('chromosome', True),
        'count_chromosome_bridge': words('is are was were has have had a an the of contains contain containing number count total exactly each'),
        'count_mitosis': words('mitosis mitotic', True),
        'count_meiosis': words('meiosis meiotic', True),
        'count_chromosome_filler': words('chromosome chromosomes per cell cells each'),
        'count_biology_forbidden': words('triploid tetraploid polyploid trisomy mutation mutant abnormal cancer ratio percent percentage fraction double doubled halve halved pair pairs', True),
        'count_magnification_number': numeric,
        'count_magnification_label': label,
        'count_magnification_bridge': words('is are was were has have had a an the of with at for equals equal to lens lenses magnification magnifications power powers x times'),
        'count_magnification_filler': words('x times magnification magnifications power powers'),
        'count_magnification_forbidden': words('change increase decrease replaced replacing replacement switched switching switch ratio percent percentage fraction area diameter radius estimate approximate about nearly around', True),
        # Biology factors are exact chromosome conservation/reduction laws.
        'count_factors': torch.tensor([0.5, 2.0, 1.0]),
        'count_numeric_gain': lexical_tables['knobs'][8].clone(),
        'count_numeric_error': lexical_tables['knobs'][9].clone(),
    }

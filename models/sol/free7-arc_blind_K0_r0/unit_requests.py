"""Compile local output-unit request grammar, never question data.

The input/output unit systems already exist in build.py. These declarations
only distinguish an explicit output-unit phrase from the default base units.
"""
import torch

CUES = 'in'
BRIDGE = 'the unit units of per cubic cubed square squared degree degrees'
EXCLUDED_SYMBOL = 'a'
ALTERNATIVE = 'and or either'


def authored_text():
    yield CUES + ' ' + BRIDGE + ' ' + EXCLUDED_SYMBOL + ' ' + ALTERNATIVE


def compile_unit_requests(vocabulary, lexical_tables):
    index = {word: i + 2 for i, word in enumerate(vocabulary)}

    def words(text):
        result = torch.zeros(len(vocabulary) + 2)
        for word in text.split():
            assert word in index, ('Missing output-unit parser word', word)
            result[index[word]] = 1
        return result

    # Joules/kilojoules were already declared for the guarded KE reader. Share
    # that exact authored conversion with the general physical-unit readers.
    units = lexical_tables['units'].clone()
    units[:, 5] = torch.maximum(units[:, 5], lexical_tables['ke_output_scale'])
    return {
        'units': units,
        'output_unit_cue': words(CUES),
        'output_unit_bridge': words(BRIDGE),
        'output_unit_alternative': words(ALTERNATIVE),
        # 'in a box' does not request amperes. Long-form amp/ampere remains
        # supported, and numeric options keep the inherited contextual A rule.
        'output_unit_excluded_symbol': words(EXCLUDED_SYMBOL),
    }

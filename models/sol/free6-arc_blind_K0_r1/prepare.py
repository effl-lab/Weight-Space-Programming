"""Regenerate vocab.txt deterministically from the authored knowledge and aliases.
This is a text compilation step, not training. No question data is read.
"""
from pathlib import Path
from build import lexical_source

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    vocabulary = sorted(lexical_source())
    (root / 'vocab.txt').write_text('\n'.join(vocabulary) + '\n')
    print(f'Wrote {len(vocabulary)} vocabulary entries.')

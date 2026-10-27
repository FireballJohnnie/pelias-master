#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Checks that Pelias's Unicode tables (generated from Python's unicodedata by tools/gen_unicode_tables.py) agree with
the Hugging Face tokenizers library itself, code point by code point. Run inside the reference venv.

    python3 tools/hf_unicode_probe.py

For every code point (surrogates excepted) it asks the library's own regex engine whether the code point is \\p{L},
\\p{M}, \\p{N} or \\s and compares with Pelias's tables (read from src/tok/unicode_tables.cpp); then it asks the
library's NFC normalizer what each code point and each canonical decomposition becomes and compares with this Python's
NFC (the tables' NFC data came from Python's unicodedata; the versions are printed so they can be checked).
Prints the counts and the first differing code points; exit 1 when anything differs (the generator then needs the
library's answers instead of Python's for those code points).
"""
import sys
import unicodedata


def main():
    import tokenizers
    from tokenizers import Regex, normalizers, pre_tokenizers
    print(f'tokenizers {tokenizers.__version__}; Python {sys.version.split()[0]} unicodedata {unicodedata.unidata_version}')

    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    text = ''.join(chr(c) for c in cps)
    import os
    import re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'tok', 'unicode_tables.cpp'),
               encoding='utf-8').read()
    print('Pelias tables: ' + src.splitlines()[0].split('from ', 1)[-1])

    def table(name):
        body = re.search(r'const CpRange ' + name + r'\[\] = \{(.*?)\n\};', src, re.S).group(1)
        out = set()
        for lo, hi in re.findall(r'\{0x([0-9A-F]+), 0x([0-9A-F]+)\}', body):
            out.update(range(int(lo, 16), int(hi, 16) + 1))
        return out

    tl, tm, tn, tw = table('kLetter'), table('kMark'), table('kNumber'), table('kWhiteSpace')
    python_class = {
        r'\p{L}': lambda c: c in tl,
        r'\p{M}': lambda c: c in tm,
        r'\p{N}': lambda c: c in tn,
        r'\s': lambda c: c in tw,
    }
    bad = 0
    for pattern, py in python_class.items():
        split = pre_tokenizers.Split(Regex(pattern), behavior='removed')
        pieces = split.pre_tokenize_str(text)
        to_char = None
        if pieces and max(e for _p, (_s, e) in pieces) > len(cps):  # byte offsets: map them to code point indices
            to_char = {}
            b = 0
            for k, c in enumerate(cps):
                to_char[b] = k
                b += len(chr(c).encode('utf-8'))
            to_char[b] = len(cps)
        kept = [False] * len(cps)
        for _piece, (start, end) in pieces:
            if to_char is not None:
                start, end = to_char[start], to_char[end]
            for k in range(start, end):
                kept[k] = True
        diffs = [c for k, c in enumerate(cps) if (not kept[k]) != py(c)]
        bad += len(diffs)
        show = ', '.join(f'U+{c:04X} (library says {"no" if py(c) else "yes"})' for c in diffs[:12])
        print(f'{pattern}: {len(diffs)} code points differ' + (f': {show}' if diffs else ''))

    nfc = normalizers.NFC()
    diffs = [c for c in cps if nfc.normalize_str(chr(c)) != unicodedata.normalize('NFC', chr(c))]
    bad += len(diffs)
    print(f'NFC of single code points: {len(diffs)} differ' + (': ' + ', '.join(f'U+{c:04X}' for c in diffs[:12]) if diffs else ''))
    seqs = []
    for c in cps:
        d = unicodedata.normalize('NFD', chr(c))
        if d != chr(c):
            seqs.append(d)
            seqs.append(d + '́')  # one more mark after it (reordering and blocking)
    diffs = [s for s in seqs if nfc.normalize_str(s) != unicodedata.normalize('NFC', s)]
    bad += len(diffs)
    print(f'NFC of {len(seqs)} decomposed sequences: {len(diffs)} differ' +
          (': ' + ', '.join(' '.join(f'U+{ord(x):04X}' for x in s) for s in diffs[:8]) if diffs else ''))
    print('RESULT: ' + ('the tables agree with the library' if bad == 0 else f'{bad} differences'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Writes src/tok/unicode_tables.cpp (standard library only).

    python3 tools/gen_unicode_tables.py [--nfc-from-crate PATH/src/tables.rs]

The tokenizer needs four facts per code point:
  - the general category class: letter (L*), mark (M*), number (N*), from the running Python's unicodedata;
  - the White_Space property (a fixed, short list from the Unicode PropList; regex \\s in the reference tokenizer);
  - the canonical combining class, the canonical decompositions and the primary composites (for NFC).

Matching the Hugging Face reference exactly needs two Unicode versions, because its two parts were built from
different ones (measured with tools/hf_unicode_probe.py on tokenizers 0.22.2):
  - its split pattern runs on Oniguruma, whose \\p{L} \\p{M} \\p{N} follow Unicode 16.0: run this with a Python
    whose unicodedata is 16.0 (Python 3.14);
  - its NFC comes from the crate unicode-normalization-alignments 0.1.12, whose tables are Unicode 9.0.0: pass that
    crate's src/tables.rs with --nfc-from-crate and the NFC data is read from it instead of from Python.
Without the option, NFC data comes from the running Python too.
"""
import os
import re
import sys
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(ROOT, 'src', 'tok', 'unicode_tables.cpp')
MAX_CP = 0x110000

# Unicode PropList.txt, White_Space (stable since Unicode 6.3)
WHITE_SPACE = [(0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085), (0x00A0, 0x00A0), (0x1680, 0x1680),
               (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]

S_BASE, S_COUNT = 0xAC00, 11172


def ranges(pred):
    out = []
    start = None
    for cp in range(MAX_CP):
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CP - 1))
    return out


def cat(cp):
    return unicodedata.category(chr(cp))


def nfc_from_python():
    """(ccc dict, {cp: full canonical decomposition}, [(first, second, composite)]) from the running Python."""
    def full(cp):
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith('<'):
            return [cp]
        out = []
        for part in d.split():
            out.extend(full(int(part, 16)))
        return out
    ccc = {cp: unicodedata.combining(chr(cp)) for cp in range(MAX_CP) if unicodedata.combining(chr(cp))}
    decomp, compose = {}, []
    for cp in range(MAX_CP):
        if S_BASE <= cp < S_BASE + S_COUNT:
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith('<'):
            continue
        decomp[cp] = full(cp)
        parts = [int(p, 16) for p in d.split()]
        if len(parts) == 2 and unicodedata.normalize('NFC', chr(parts[0]) + chr(parts[1])) == chr(cp):
            compose.append((parts[0], parts[1], cp))
    return ccc, decomp, sorted(compose), 'Python ' + sys.version.split()[0] + ' unicodedata (Unicode ' + unicodedata.unidata_version + ')'


def nfc_from_crate(path):
    """The same three tables, read from unicode-normalization(-alignments)'s generated src/tables.rs."""
    src = open(path, encoding='utf-8').read()
    ver = re.search(r'UNICODE_VERSION: \(u64, u64, u64\) = \((\d+), (\d+), (\d+)\)', src).groups()

    def block(name):
        m = re.search(r'pub\(crate\) const ' + name + r': [^=]+= &\[(.*?)\n\];', src, re.S)
        if not m:
            sys.exit(f'{name} not found in {path}')
        return m.group(1)

    def ch(s):
        return int(s, 16)

    ccc = {}
    for v in re.findall(r'0x([0-9A-Fa-f]+),', block('CANONICAL_COMBINING_CLASS_KV')):
        x = int(v, 16)
        ccc[x >> 8] = x & 0xFF
    decomp = {}
    for cp, chars in re.findall(r"\(0x([0-9A-Fa-f]+), &\[(.*?)\]\)", block('CANONICAL_DECOMPOSED_KV')):
        decomp[int(cp, 16)] = [ch(c) for c in re.findall(r"'\\u\{([0-9A-Fa-f]+)\}'", chars)]
    compose = []
    for key, c in re.findall(r"\(0x([0-9A-Fa-f]+), '\\u\{([0-9A-Fa-f]+)\}'\)", block('COMPOSITION_TABLE_KV')):
        k = int(key, 16)
        compose.append((k >> 16, k & 0xFFFF, ch(c)))
    astral = re.search(r'fn composition_table_astral.*?match \(c1, c2\) \{(.*?)_ => None', src, re.S).group(1)
    for a, b, c in re.findall(r"\('\\u\{([0-9A-Fa-f]+)\}', '\\u\{([0-9A-Fa-f]+)\}'\) => Some\('\\u\{([0-9A-Fa-f]+)\}'\)", astral):
        compose.append((ch(a), ch(b), ch(c)))
    # Hangul syllables are computed by the C++ side; the crate does not table them either
    decomp = {k: v for k, v in decomp.items() if not S_BASE <= k < S_BASE + S_COUNT}
    name = 'unicode-normalization'
    cargo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), 'Cargo.toml')
    if os.path.exists(cargo):
        txt = open(cargo, encoding='utf-8').read()
        n = re.search(r'^name = "([^"]+)"', txt, re.M)
        v = re.search(r'^version = "([^"]+)"', txt, re.M)
        name = (n.group(1) if n else name) + (' ' + v.group(1) if v else '')
    return ccc, decomp, sorted(compose), f'the crate {name} src/tables.rs (Unicode {".".join(ver)})'


def main():
    crate = None
    if '--nfc-from-crate' in sys.argv:
        crate = sys.argv[sys.argv.index('--nfc-from-crate') + 1]
    letters = ranges(lambda cp: cat(cp).startswith('L'))
    marks = ranges(lambda cp: cat(cp).startswith('M'))
    numbers = ranges(lambda cp: cat(cp).startswith('N'))
    ccc_map, decomp, compose, nfc_src = nfc_from_crate(crate) if crate else nfc_from_python()
    cat_src = 'Python ' + sys.version.split()[0] + ' unicodedata (Unicode ' + unicodedata.unidata_version + ')'

    ccc = []  # (lo, hi, class) runs of equal non-zero class
    for cp in sorted(ccc_map):
        c = ccc_map[cp]
        if not c:
            continue
        if ccc and ccc[-1][2] == c and ccc[-1][1] == cp - 1:
            ccc[-1] = (ccc[-1][0], cp, c)
        else:
            ccc.append((cp, cp, c))

    pool, index = [], []
    for cp in sorted(decomp):
        seq = decomp[cp]
        index.append((cp, len(pool), len(seq)))
        pool.extend(seq)

    def fmt_ranges(name, rs):
        body = ',\n'.join('    {0x%X, 0x%X}' % r for r in rs)
        return f'const CpRange {name}[] = {{\n{body},\n}};\nconst std::size_t {name}_count = {len(rs)};\n'

    lines = [
        '// provenance: generated by tools/gen_unicode_tables.py; categories from ' + cat_src + ';',
        '//   NFC data from ' + nfc_src,
        '// Copyright (c) 2026 John Lipka and the Pelias contributors',
        '//',
        '// Do not edit: rerun the generator. Sorted tables for binary search (tok/unicode.h).',
        '#include "tok/unicode.h"',
        '',
        'namespace pelias::tok::tables {',
        '',
        f'const char* const kUnicodeVersion = "categories {unicodedata.unidata_version}, NFC {nfc_src.split("Unicode ")[-1].rstrip(")")}";',
        '',
        fmt_ranges('kLetter', letters),
        fmt_ranges('kMark', marks),
        fmt_ranges('kNumber', numbers),
        fmt_ranges('kWhiteSpace', WHITE_SPACE),
    ]
    body = ',\n'.join('    {0x%X, 0x%X, %d}' % r for r in ccc)
    lines.append(f'const CccRange kCcc[] = {{\n{body},\n}};\nconst std::size_t kCcc_count = {len(ccc)};\n')
    body = ',\n'.join('    {0x%X, %d, %d}' % r for r in index)
    lines.append(f'const DecompEntry kDecomp[] = {{\n{body},\n}};\nconst std::size_t kDecomp_count = {len(index)};\n')
    chunks = [', '.join('0x%X' % v for v in pool[i:i + 12]) for i in range(0, len(pool), 12)]
    body = ',\n'.join('    ' + c for c in chunks)
    lines.append(f'const char32_t kDecompPool[] = {{\n{body},\n}};\nconst std::size_t kDecompPool_count = {len(pool)};\n')
    body = ',\n'.join('    {0x%X, 0x%X, 0x%X}' % r for r in compose)
    lines.append(f'const ComposePair kCompose[] = {{\n{body},\n}};\nconst std::size_t kCompose_count = {len(compose)};\n')
    lines.append('}  // namespace pelias::tok::tables')
    lines.append('')
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print(f'categories: {cat_src}: letters {len(letters)} ranges, marks {len(marks)}, numbers {len(numbers)}')
    print(f'NFC: {nfc_src}: ccc {len(ccc)} runs, decompositions {len(index)} ({len(pool)} code points), compositions {len(compose)}')


if __name__ == '__main__':
    main()

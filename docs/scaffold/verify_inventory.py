#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Verify the pelias-master tree against its inventory. Standard library only.

    python3 docs/scaffold/verify_inventory.py            (run from anywhere; the root is found from this file)
    python3 docs/scaffold/verify_inventory.py --write    (rewrite the inventory from the tree, then verify)

Checks
  1. every row of docs/scaffold/REPO_INVENTORY.tsv: the file exists and its size and SHA-256 equal the recorded ones;
  2. every file of the tree (outside .git and the git-ignored local_only folders) is listed; the inventory itself is
     the one file left out on purpose;
  3. every JSON file parses; every text file decodes as UTF-8;
  4. privacy scan of every listed text file (counts only, no value printed): local account names, the persona name,
     Windows drive and profile paths (skipped in the verbatim folders docs/inputs/ and docs/reviews/, which are reviewed by hand), a personal name
     outside the approved copyright line, Hugging Face token patterns, Windows profile paths, Linux home paths,
     e-mail addresses;
  5. profiles/local_only/ is ignored by git and nothing under it is tracked (when git is available).

Exit status 0 when every check passes, 1 otherwise.
"""
import csv, hashlib, json, os, re, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
INV = 'docs/scaffold/REPO_INVENTORY.tsv'
SKIP_DIRS = {'.git', 'build', 'out'}
LOCAL = ('profiles/local_only/', 'support/local_only/')
TEXT_EXT = {'.md', '.json', '.csv', '.tsv', '.py', '.hpp', '.txt', '.gitignore', '.gitattributes', '',
            '.h', '.cpp', '.cu', '.comp', '.idl', '.sh', '.ps1'}  # source code scanned too (2026-10-26, Batch C)
COPYRIGHT = 'Copyright (c) 2026 John Lipka and the Pelias contributors'
# Private words are stored as SHA-256 hashes of their lower-case form, so this file does not itself contain them.
HASHED_WORDS = {
    'local account name': 'c4c4da0e19c6dcde5a60a2fc6adb9eb0413fd8e8dfbd1d211d0fccf015c457fc',
    'persona name': '7deead025aea61ac476bb9bb0d13b57a7306b15adea59c1f1b74e6f0abdaf7ad',
    'personal name outside the copyright line': 'ae35b7df5e73a25fdcf982de1efd0c865d8a679411e90aa52cb64c855c074b4e',
}
PATTERNS = {
    'Hugging Face token pattern': re.compile(r'hf_[A-Za-z0-9]{30,}'),
    'Windows profile path': re.compile(r'[A-Za-z]:\\Users\\', re.I),
    'Linux home path other than /home/user and /home/claude': re.compile(r'/home/(?!user\b|claude\b)[a-z][a-z0-9_-]*/'),
    'Windows drive path': re.compile(r'\b[A-Za-z]:\\[A-Za-z]'),
    'e-mail address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}
WORD = re.compile(r'[A-Za-z0-9]+')
# The author's input files and the review record are kept word for word; they are reviewed by hand (docs/inputs/REVIEW.md,
# docs/reviews/) and excluded from these two checks only.
VERBATIM = ('docs/inputs/', 'docs/reviews/')
VERBATIM_ONLY = ('Windows drive path', 'Windows profile path')  # the persona name is checked everywhere since the 2026-10-26 redaction
ALLOWED_NAME_LINES = ('Author and maintainer:** John', COPYRIGHT)
problems, notes = [], []


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tree():
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, '/')
            if rel == INV or rel.startswith(LOCAL):
                continue
            out.append(rel)
    return out


def is_text(rel):
    return os.path.splitext(rel)[1].lower() in TEXT_EXT


def group(rel):
    return rel.split('/')[0] if '/' in rel else 'root'


def write_inventory():
    with open(os.path.join(ROOT, INV), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['path', 'group', 'bytes', 'lines', 'sha256'])
        for rel in tree():
            p = os.path.join(ROOT, rel)
            lines = ''
            if is_text(rel):
                with open(p, 'rb') as g:
                    lines = g.read().count(b'\n')
            w.writerow([rel, group(rel), os.path.getsize(p), lines, sha(p)])


def main():
    if '--write' in sys.argv:
        write_inventory()
    with open(os.path.join(ROOT, INV), encoding='utf-8', newline='') as f:
        inv = list(csv.DictReader(f, delimiter='\t'))
    listed = {r['path'] for r in inv}
    ok = 0
    for r in inv:
        p = os.path.join(ROOT, r['path'])
        if not os.path.isfile(p):
            problems.append('missing: ' + r['path'])
        elif os.path.getsize(p) != int(r['bytes']) or sha(p) != r['sha256']:
            problems.append('changed: ' + r['path'])
        else:
            ok += 1
    t = tree()
    for rel in t:
        if rel not in listed:
            problems.append('not in inventory: ' + rel)
    notes.append('inventory: %d rows, %d match' % (len(inv), ok))
    notes.append('tree: %d files outside .git and the local_only folders, 1 left out on purpose (the inventory)' % len(t))

    n_json = n_text = 0
    counts = {k: 0 for k in list(HASHED_WORDS) + list(PATTERNS)}
    for rel in t:
        p = os.path.join(ROOT, rel)
        if not is_text(rel):
            continue
        try:
            text = open(p, encoding='utf-8').read()
            n_text += 1
        except UnicodeDecodeError:
            problems.append('not UTF-8: ' + rel)
            continue
        if rel.endswith('.json'):
            try:
                json.loads(text)
                n_json += 1
            except ValueError:
                problems.append('JSON does not parse: ' + rel)
        for line in text.splitlines():
            for name, rx in PATTERNS.items():
                if name in VERBATIM_ONLY and rel.startswith(VERBATIM):
                    continue
                if rel == 'docs/scaffold/verify_inventory.py':
                    continue  # this file holds the patterns themselves
                counts[name] += len(rx.findall(line))
            allowed = any(a in line for a in ALLOWED_NAME_LINES)
            for w in WORD.findall(line):
                h = hashlib.sha256(w.lower().encode()).hexdigest()
                for name, hw in HASHED_WORDS.items():
                    if name in VERBATIM_ONLY and rel.startswith(VERBATIM):
                        continue
                    if h == hw and not (allowed and name.startswith('personal name')):
                        counts[name] += 1
    notes.append('text files: %d, all UTF-8; JSON files: %d, all parse' % (n_text, n_json))
    for name, c in counts.items():
        notes.append('privacy scan, %s: %d' % (name, c))
        if c:
            problems.append('privacy scan found %d occurrence(s): %s' % (c, name))

    try:
        inside = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--is-inside-work-tree'], capture_output=True, text=True)
        if inside.returncode != 0 or inside.stdout.strip() != 'true':
            raise FileNotFoundError
        for d in LOCAL:
            r = subprocess.run(['git', '-C', ROOT, 'check-ignore', '-q', d + 'x'], capture_output=True)
            if r.returncode != 0:
                problems.append('not ignored by git: ' + d)
        r = subprocess.run(['git', '-C', ROOT, 'ls-files'], capture_output=True, text=True)
        tracked = [x for x in r.stdout.splitlines() if x.startswith(LOCAL)]
        if tracked:
            problems.append('tracked under local_only: %d files' % len(tracked))
        notes.append('git: local_only folders ignored; %d tracked files under them' % len(tracked))
    except FileNotFoundError:
        notes.append('git: not a git checkout (or git not installed), check 5 skipped')

    for n in notes:
        print(n)
    for p in problems:
        print('PROBLEM:', p)
    print('RESULT:', 'all checks passed' if not problems else '%d problem(s)' % len(problems))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())

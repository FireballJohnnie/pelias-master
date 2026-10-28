#!/usr/bin/env python3
# provenance: original. The llama-quantize log line format and its --tensor-type-file syntax (whitespace-separated
#   `pattern=type` tokens, the pattern lower-cased and matched with std::regex_search, the first match wins, the type
#   name case-insensitive) are READ-SRC from llama.cpp tools/quantize/quantize.cpp and src/llama-quant.cpp at commit
#   2ca15f5404760548c39e7b92bd43116a09414a1a. Block sizes READ-SRC from ggml/src/ggml-common.h at the same commit.
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""PEL recipes: one exact type per tensor (docs/PELIAS_GAME_PLAN.md, 0.4).

A recipe is a TSV file, one line per tensor: `name<TAB>TYPE` (ggml type name, upper case), in file order.

  recipe.py from-log  LOG OUT.tsv                      recipe from a llama-quantize log (all tensors of the file)
  recipe.py from-gguf GGUF OUT.tsv --pelias BIN        recipe from a GGUF (pelias inspect --json)
  recipe.py edit IN.tsv OUT.tsv --set REGEX=TYPE ...   change the type of every tensor whose name matches REGEX
  recipe.py llama-file RECIPE.tsv SOURCE.gguf OUT.txt --pelias BIN
                                                       a --tensor-type-file for llama-quantize: one anchored pattern
                                                       per tensor (every tensor is named, so llama-quantize's own mix
                                                       rules and per-layer counters decide nothing)
  recipe.py check RECIPE.tsv GGUF --pelias BIN         exit 0 if every tensor of GGUF has the recipe type, else 1
  recipe.py data-bytes RECIPE.tsv SOURCE.gguf --pelias BIN
                                                       the tensor-data bytes a file made by this recipe holds
                                                       (each tensor padded to the source file's alignment)
  recipe.py diff A.tsv B.tsv                           tensors whose type differs between two recipes

Exit codes: 0 good, 1 a check or diff found differences, 2 bad input. Nothing is ever overwritten (exit 2).
Standard library only.
"""
import json
import os
import re
import subprocess
import sys

# (elements per block, bytes per block), READ-SRC ggml-common.h at 2ca15f5
BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
    "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ1_S": (256, 50), "IQ1_M": (256, 56),
    "TQ1_0": (256, 54), "TQ2_0": (256, 66), "MXFP4": (32, 17),
    "Q8_Z": (256, 264),  # Pelias-only (block_types.h)
    "Q7_Z": (256, 232),  # Pelias-only (block_types.h)
}

# A tensor's record starts with "[  N/ T] NAME - [shape], type = SRC," and ends before the next record. Other messages
# (imatrix notices on stderr) can land inside it when stdout and stderr share the log, so the result is searched for in
# the whole span: "converting to DST" if the tensor was converted, else only "size =" (kept as SRC).
LOG_HEAD = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+-\s+\[([^\]]*)\],\s+type =\s+(\S+),", re.M)
LOG_CONV = re.compile(r"converting to (\S+?)\s+\.\.")


def die(msg, code=2):
    print("recipe: " + msg, file=sys.stderr)
    sys.exit(code)


def no_overwrite(path):
    if os.path.exists(path):
        die(path + " already exists (nothing is overwritten)")


def read_recipe(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2 or parts[1] not in BLOCK:
                die("%s line %d is not `name<TAB>TYPE` with a known type" % (path, n))
            rows.append((parts[0], parts[1]))
    names = [r[0] for r in rows]
    if len(set(names)) != len(names):
        die(path + " names a tensor twice")
    return rows


def write_recipe(path, rows):
    no_overwrite(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for name, t in rows:
            f.write("%s\t%s\n" % (name, t))


def inspect(pelias, gguf):
    try:
        out = subprocess.run([pelias, "inspect", "--model", gguf, "--json"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        die("pelias inspect failed on %s: %s" % (gguf, e))
    d = json.loads(out)
    return d


def from_log(log, out):
    with open(log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    heads = list(LOG_HEAD.finditer(text))
    rows, total = [], None
    for k, m in enumerate(heads):
        idx, tot, name, _shape, src = m.groups()
        total = int(tot)
        span = text[m.end():heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        c = LOG_CONV.search(span)
        if c:
            t = c.group(1).upper()
        elif "size =" in span:
            t = src.upper()
        else:
            die("%s: tensor %s has neither a conversion nor a size in its record" % (log, name))
        if t not in BLOCK:
            die("unknown type %s for %s in %s" % (t, name, log))
        if int(idx) != len(rows) + 1:
            die("%s: tensor %s is numbered %s, expected %d (lines missing or repeated)" % (log, name, idx, len(rows) + 1))
        rows.append((name, t))
    if total is None or len(rows) != total:
        die("%s: found %d tensor records, the log announces %s" % (log, len(rows), total))
    write_recipe(out, rows)
    print("recipe: %d tensors from %s" % (len(rows), log))


def from_gguf(gguf, out, pelias):
    d = inspect(pelias, gguf)
    rows = [(t["name"], t["type"].upper()) for t in d["tensors"]]
    for name, t in rows:
        if t not in BLOCK:
            die("unknown type %s for %s" % (t, name))
    write_recipe(out, rows)
    print("recipe: %d tensors from %s" % (len(rows), gguf))


def edit(src, out, sets):
    rows = read_recipe(src)
    rules = []
    for s in sets:
        if "=" not in s:
            die("--set needs REGEX=TYPE, got " + s)
        pat, t = s.rsplit("=", 1)
        t = t.upper()
        if t not in BLOCK:
            die("unknown type " + t)
        rules.append((re.compile(pat), t, s))
    changed = 0
    hits = {r[2]: 0 for r in rules}
    new = []
    for name, t in rows:
        for rx, nt, s in rules:
            if rx.search(name):
                hits[s] += 1
                if nt != t:
                    changed += 1
                t = nt
                break
        new.append((name, t))
    for s, n in hits.items():
        if n == 0:
            die("--set %s matched no tensor" % s)
    write_recipe(out, new)
    print("recipe: %d tensors changed; matches per rule: %s" % (changed, ", ".join("%s: %d" % kv for kv in hits.items())))


def llama_file(recipe, source, out, pelias):
    rows = read_recipe(recipe)
    src = {t["name"]: t["type"].upper() for t in inspect(pelias, source)["tensors"]}
    missing = [n for n, _ in rows if n not in src]
    if missing or len(src) != len(rows):
        die("recipe and source differ in their tensor lists (%d missing, %d vs %d tensors)" % (len(missing), len(rows), len(src)))
    no_overwrite(out)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for name, t in rows:
            # every tensor gets its line, even one whose type equals the source's: llama-quantize would otherwise apply
            # its mix rule to it (a 1-D tensor that may not be quantized keeps its type whatever the line says)
            if name != name.lower() or any(c.isspace() for c in name):
                die("tensor name %r cannot be written as a lower-case, space-free pattern" % name)
            f.write("^%s$=%s\n" % (re.escape(name), t.lower()))
            n += 1
    print("recipe: %d patterns written to %s" % (n, out))


def check(recipe, gguf, pelias):
    rows = read_recipe(recipe)
    got = [(t["name"], t["type"].upper()) for t in inspect(pelias, gguf)["tensors"]]
    want = dict(rows)
    gotd = dict(got)
    bad = []
    for name, t in rows:
        if name not in gotd:
            bad.append("%s: missing from the file" % name)
        elif gotd[name] != t:
            bad.append("%s: recipe %s, file %s" % (name, t, gotd[name]))
    for name in gotd:
        if name not in want:
            bad.append("%s: in the file, not in the recipe" % name)
    if bad:
        print("recipe: %d tensors differ:" % len(bad))
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("recipe: all %d tensors have the recipe type" % len(rows))


def data_bytes(recipe, source, pelias):
    rows = read_recipe(recipe)
    d = inspect(pelias, source)
    align = int(d.get("alignment", 32))
    shapes = {t["name"]: t["shape"] for t in d["tensors"]}
    total = 0
    for i, (name, t) in enumerate(rows):
        if name not in shapes:
            die(name + " is not in the source file")
        ne = 1
        for x in shapes[name]:
            ne *= int(x)
        per, size = BLOCK[t]
        if int(shapes[name][0]) % per != 0:
            die("%s: row length %s is not a multiple of %s's block of %d" % (name, shapes[name][0], t, per))
        nb = ne // per * size
        total += nb if i == len(rows) - 1 else (nb + align - 1) // align * align
    print(total)


def diff(a, b):
    ra, rb = dict(read_recipe(a)), dict(read_recipe(b))
    names = sorted(set(ra) | set(rb))
    n = 0
    for name in names:
        if ra.get(name) != rb.get(name):
            print("%s\t%s\t%s" % (name, ra.get(name, "-"), rb.get(name, "-")))
            n += 1
    print("recipe: %d tensors differ" % n, file=sys.stderr)
    sys.exit(1 if n else 0)


def opt(args, flag):
    if flag in args:
        i = args.index(flag)
        if i + 1 >= len(args):
            die(flag + " needs a value")
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return None


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(argv) >= 2 else 2
    cmd, args = argv[1], argv[2:]
    pelias = opt(args, "--pelias")
    sets = []
    while "--set" in args:
        sets.append(opt(args, "--set"))
    need_pelias = cmd in ("from-gguf", "llama-file", "check", "data-bytes")
    if need_pelias and not pelias:
        die(cmd + " needs --pelias BIN")
    if cmd == "from-log" and len(args) == 2:
        from_log(*args)
    elif cmd == "from-gguf" and len(args) == 2:
        from_gguf(args[0], args[1], pelias)
    elif cmd == "edit" and len(args) == 2 and sets:
        edit(args[0], args[1], sets)
    elif cmd == "llama-file" and len(args) == 3:
        llama_file(args[0], args[1], args[2], pelias)
    elif cmd == "check" and len(args) == 2:
        check(args[0], args[1], pelias)
    elif cmd == "data-bytes" and len(args) == 2:
        data_bytes(args[0], args[1], pelias)
    elif cmd == "diff" and len(args) == 2:
        diff(*args)
    else:
        die("bad arguments; see --help")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

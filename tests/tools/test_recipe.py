#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Unit test of tools/recipe.py on synthetic inputs (no GGUF needed): log parsing with interleaved messages,
edit rules, diff, and the refusal to overwrite. Exit 0 on success."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "recipe.py")

LOG = """llama_quantize: quantizing 'x' to 'y' as Q4_K_M
[   1/   4] output.weight                        - [   256, 1000,      1,      1], type =    f32, 
====== llama_model_quantize_impl: did not find weights for output.weight
converting to q6_K .. load_imatrix: notice
size =   1.00 MiB ->    0.20 MiB
[   2/   4] output_norm.weight                   - [   256,      1,      1,      1], type =    f32, size =    0.001 MiB
[   3/   4] blk.0.ssm_out.weight                 - [   256,    256,      1,      1], type =   bf16, converting to q8_0 .. size =     0.12 MiB ->     0.07 MiB
[   4/   4] blk.0.ffn_up_exps.weight             - [   256,    256,      8,      1], type =   bf16, converting to iq4_xs .. size =     1.00 MiB ->     0.27 MiB
"""


def run(*args):
    return subprocess.run([sys.executable, TOOL] + list(args), capture_output=True, text=True)


def main():
    fails = 0

    def check(cond, what):
        nonlocal fails
        if not cond:
            fails += 1
            print("FAIL: " + what)

    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "q.log")
        open(log, "w").write(LOG)
        a = os.path.join(d, "a.tsv")
        r = run("from-log", log, a)
        check(r.returncode == 0, "from-log succeeds: " + r.stderr)
        rows = [l.rstrip("\n").split("\t") for l in open(a)]
        check(rows == [["output.weight", "Q6_K"], ["output_norm.weight", "F32"], ["blk.0.ssm_out.weight", "Q8_0"],
                       ["blk.0.ffn_up_exps.weight", "IQ4_XS"]], "parsed types: %r" % rows)
        check(run("from-log", log, a).returncode == 2, "refuses to overwrite")
        b = os.path.join(d, "b.tsv")
        r = run("edit", a, b, "--set", "ssm_out=Q6_K")
        check(r.returncode == 0, "edit succeeds")
        r = run("diff", a, b)
        check(r.returncode == 1 and "blk.0.ssm_out.weight\tQ8_0\tQ6_K" in r.stdout, "diff shows the one change")
        check(run("diff", a, a).returncode == 0, "identical recipes: exit 0")
        check(run("edit", a, os.path.join(d, "c.tsv"), "--set", "nomatch=Q8_0").returncode == 2, "a rule that matches nothing is refused")
        short = os.path.join(d, "short.log")
        open(short, "w").write("\n".join(LOG.splitlines()[:6]) + "\n")
        check(run("from-log", short, os.path.join(d, "s.tsv")).returncode == 2, "a log with missing records is refused")
    print("recipe tests: %s" % ("all passed" if fails == 0 else "%d failed" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

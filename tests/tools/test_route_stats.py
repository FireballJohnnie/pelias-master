#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Unit test of tools/route_stats.py on a synthetic PELROUT1 record with known answers: rank weights, reuse, the
pre-gating prediction, flips between two records and the KL split. Exit 0 on success."""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "route_stats.py")


def record(path, sels, preds, n_layer, chunks, scored, h=7):
    with open(path, "wb") as f:
        f.write(b"PELROUT1" + struct.pack("<5I", 4, chunks, scored, n_layer, 2) + struct.pack("<Q", h))
        for (s, p) in zip(sels, preds):
            f.write(struct.pack("<8h8f2f8h", s[0], s[1], -1, -1, -1, -1, -1, -1, 0.75, 0.25, 0, 0, 0, 0, 0, 0,
                                0.30, 0.10, p[0], p[1], -1, -1, -1, -1, -1, -1))


def kld(path, kls, chunks, scored, h=7):
    with open(path, "wb") as f:
        f.write(b"PELKLD01" + struct.pack("<3I", 4, chunks, scored) + struct.pack("<Q", h) + struct.pack("<Q", len(kls)))
        for k in kls:
            f.write(struct.pack("<dfffB3x", k, 0.0, 0.0, 0.0, 1))


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
        a, b, k = os.path.join(d, "a.bin"), os.path.join(d, "b.bin"), os.path.join(d, "a.kld")
        # 1 chunk, 3 tokens, 2 layers (token-major): every token picks {1,2} in layer 0 and {3,4} in layer 1;
        # layer 0 predicts {3,9} for layer 1 (half right); the last layer predicts nothing (-1)
        sels = [(1, 2), (3, 4)] * 3
        preds = [(3, 9), (-1, -1)] * 3
        record(a, sels, preds, 2, 1, 3)
        r = run("stats", a, "--hot", "1")
        check(r.returncode == 0, "stats runs: " + r.stderr)
        check("1: 0.750  2: 0.250" in r.stdout, "rank weights")
        check("previous 1 token(s): picks seen before 100.0 %" in r.stdout, "reuse of identical picks")
        check("next-layer prediction from this layer's output: 50.0 %" in r.stdout, "prediction half right")
        check("cold picks (outside the 1 most-picked experts of each layer): 50.0 %" in r.stdout, "cold share with one hot expert")
        # B differs from A in layer 1 of the middle token only
        sels_b = list(sels)
        sels_b[3] = (3, 5)
        record(b, sels_b, preds, 2, 1, 3)
        kld(k, [0.1, 0.4, 0.1], 1, 3)
        r = run("compare", a, b, "--kld", k)
        check(r.returncode == 0, "compare runs: " + r.stderr)
        check("all layers: set differs 16.7 % of token-layers" in r.stdout, "one of six token-layers differs")
        check("share of all KL carried by tokens with at least one changed layer: 66.7 %" in r.stdout, "KL share 0.4/0.6")
        record(b, sels_b, preds, 2, 1, 3, h=8)
        check(run("compare", a, b).returncode == 2, "different token hash refused")
        with open(os.path.join(d, "bad.bin"), "wb") as f:
            f.write(b"NOTROUTE" + b"\0" * 40)
        check(run("stats", os.path.join(d, "bad.bin")).returncode == 2, "bad magic refused")
    print("route_stats tests: %s" % ("all passed" if fails == 0 else "%d failed" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

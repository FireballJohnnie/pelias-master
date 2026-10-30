#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# GP3-08 gate: compares two MTP dumps (tools/oracle/mtp_oracle.cpp and `pelias mtp-logits`): per position the logits'
# max |difference| relative to max |logit|, the same top-1 token, and the hidden rows' relative L2 difference.
#   mtp_compare.py A.bin B.bin [tolerance, default 1e-4]
import struct
import sys

import numpy as np


def read(path):
    with open(path, 'rb') as f:
        n, nv, ne = struct.unpack('<3i', f.read(12))
        lg = np.frombuffer(f.read(4 * n * nv), dtype='<f4').reshape(n, nv)
        h = np.frombuffer(f.read(4 * n * ne), dtype='<f4').reshape(n, ne)
    return lg, h


def main():
    a_lg, a_h = read(sys.argv[1])
    b_lg, b_h = read(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    assert a_lg.shape == b_lg.shape and a_h.shape == b_h.shape, 'shapes differ'
    worst_l = worst_h = 0.0
    same = 0
    for p in range(a_lg.shape[0]):
        rel = float(np.max(np.abs(a_lg[p] - b_lg[p])) / max(1e-30, np.max(np.abs(a_lg[p]))))
        rh = float(np.linalg.norm(a_h[p] - b_h[p]) / max(1e-30, np.linalg.norm(a_h[p])))
        top = int(np.argmax(a_lg[p])) == int(np.argmax(b_lg[p]))
        same += top
        worst_l, worst_h = max(worst_l, rel), max(worst_h, rh)
        print(f'pos {p:3d}: logits max rel diff {rel:.3e}, hidden rel L2 {rh:.3e}, top-1 {"same" if top else "DIFFERS"}')
    ok = worst_l <= tol and worst_h <= tol and same == a_lg.shape[0]
    print(f'positions {a_lg.shape[0]}: worst logits {worst_l:.3e}, worst hidden {worst_h:.3e}, same top-1 {same}/{a_lg.shape[0]}; '
          f'tolerance {tol:g}: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

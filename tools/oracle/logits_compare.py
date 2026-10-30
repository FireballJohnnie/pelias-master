# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# Test-only: compares two logits files (int32 n, int32 n_vocab, n x n_vocab fp32; `pelias logits` and logits_oracle):
# per position the largest |difference| relative to the largest |logit| of the reference, and the top-1 token.
#   python3 logits_compare.py REF.bin GOT.bin TOL
import sys
import numpy as np
def load(p):
    h = np.fromfile(p, dtype=np.int32, count=2)
    return np.fromfile(p, dtype=np.float32, offset=8).reshape(int(h[0]), int(h[1]))
a, b, tol = load(sys.argv[1]), load(sys.argv[2]), float(sys.argv[3])
assert a.shape == b.shape, (a.shape, b.shape)
worst, same = 0.0, 0
for i in range(a.shape[0]):
    rel = float(np.max(np.abs(a[i] - b[i])) / max(1e-30, float(np.max(np.abs(a[i])))))
    worst = max(worst, rel)
    same += int(np.argmax(a[i]) == np.argmax(b[i]))
ok = worst <= tol and same == a.shape[0]
print(f"positions {a.shape[0]}: worst relative difference {worst:.3g}, same top-1 {same}/{a.shape[0]}; tolerance {tol}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)

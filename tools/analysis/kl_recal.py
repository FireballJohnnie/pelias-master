#!/usr/bin/env python3
"""Projections re-based on three MEASURED points (error left vs Unsloth's experts -> KL, WikiText): 1.0 -> 0.012808 (the
Unsloth file), 0.484 -> 0.009404 (the tiered file, orel error of tier_project_v1.txt), 0.098 -> 0.005818 (2 planes
everywhere). Piecewise-linear between the points (DERIVED); the old two-point line is kept beside for comparison.
Errors from tier_project_v1.txt, tier_vram_v1.txt and per_pick_planes_v1.txt (orel metric, 50/50 mix ranking)."""
import numpy as np
X, Y = [0.098, 0.484, 1.0], [0.005818, 0.009404, 0.012808]
pw = lambda e: float(np.interp(e, X, Y, left=None)) if e >= X[0] else Y[0] + (e - X[0]) * (Y[1] - Y[0]) / (X[1] - X[0])
old = lambda e: 0.0051 + 0.0078 * e
rows = [("tiered 22.65 GB", 0.484, 0.730), ("+1 GB block planes, always", 0.415, 0.577), ("+2 GB block planes, always", 0.361, 0.460),
        ("+1 GB planes, context-hot", 0.446, 0.645), ("tiered 23.65 GB static", 0.331, 0.532), ("tiered 24.65 GB static", 0.238, 0.385),
        ("tiered 26.93 GB static", 0.113, 0.182)]
# 2 GB block planes read for the top 4 picks: 80.8 % of the always-read benefit (w^2, per_pick_planes_v1.txt)
rows.append(("+2 GB block planes, top 4 picks", 0.484 - 0.808 * (0.484 - 0.361), 0.730 - 0.808 * (0.730 - 0.460)))
rows.append(("+1 GB block planes, top 4 picks", 0.484 - 0.808 * (0.484 - 0.415), 0.730 - 0.808 * (0.730 - 0.577)))
for name, ew, ec in rows:
    print(f'{name:34s} WikiText err {ew:.3f} -> KL {pw(ew):.4f} (old line {old(ew):.4f}); chat+code err {ec:.3f} -> KL {pw(ec):.4f} (old {old(ec):.4f})')

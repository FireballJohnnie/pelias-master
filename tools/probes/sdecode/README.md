# tools/probes/sdecode — the expert stage's decode cost per format, and the Q4_TG2 tag mix

Copyright (c) 2026 John Lipka and the Pelias contributors. Confidential, not for distribution.

R3 rebuild design (docs/R3_REBUILD_DESIGN.md), stage 0.1 and 0.2.

- `sdecode.cu`: compiles `kern/cuda/gpu_stream.cu` as a translation unit and times one warp task (2 rows x K 2048) through
  the engine's own `lane_part` / `lane_part_n` staged path with the weight bytes already in shared memory: cycles per task
  per format, 1 and 3 tokens, 1 and 8 warps per SM; Q5_K / Q4_K planes on both the 1-bit route and the dp4a route.
  Build (WSL): `nvcc -std=c++20 -O3 -arch=sm_89 -I<repo>/src -I<repo>/include sdecode.cu <repo>/src/quant/quant_cpu.cpp -o sdecode`
- `tg2_tags.py <tier.gguf>`: the histogram of the 2-bit grid tags over every Q4_TG2 tensor (read-only): the share of the
  uniform grids (tags 0 and 3: exact on dp4a) vs the log grids (tags 1 and 2: float math only).
- Runner: `D:\pelias-tests\gp\probes_s0.sh` (logs in `gp\PROBES\`).

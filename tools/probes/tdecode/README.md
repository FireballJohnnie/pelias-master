# tdecode: the 1-bit decoder on the GPU (probe)

Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.

Steps 4-5 of the 1-bit decoder plan (`docs/PELIAS_7_T_H_PLAN.md`, section 6). Target sm_89 (RTX 4070).

| Command | What it measures |
|---|---|
| `tdecode mma` | Rate of the 1-bit tensor path (`mma.sync m16n8k256 .b1 .and.popc`) against the int8 one (`m16n8k32 .s8`), 4 independent accumulators per warp |
| `tdecode gemv` | Single-token GEMV (decode) on dense shapes (8192x2048, 2048x4096, 4096x2048, 512x2048, 2048x512): Q8_Z (baseline, 8.25 bpw), Q7_ZT with the 8th bit folded into int8 (dp4a), Q7_ZT by T-decode (base dp4a + 8th bit by AND + popcount), ZH live with one shared block choice per token (H read on 0 / 25 / 100 % of input blocks, the rest filled by 0.5). Every route is checked against a double-precision host reference before it is timed; the activation prep (per-32 int8 + 8 bit planes by warp ballot) is checked against the host |

Device layouts are the engine's (repacked at load): headers (fp16 d + eight 6-bit levels), the Q7_ZT gate masks and H bits, and the ZH H plane (grouped by input block, one word per row) each in their own array.

Checked before the first run (sandbox): the kernels compile to PTX for sm_89 with clang 18 (both mma forms emitted); the integer paths (7-bit unpack with and without H, the 1-bit route, the 0.5 fill, the odd grid) emulated on the CPU against the reference on 4,000 random blocks with every edge value: 0 mismatches (`tools/experiments/tdecode_kernel_emul.py`). The first version's fold route added H to packed bytes, which carries out of a byte holding -1; caught by the emulation review and fixed before any run.

Build and run (WSL, after any other run on the machine has finished):

    export PATH=/usr/local/cuda/bin:$PATH && nvcc -std=c++17 -O3 -arch=sm_89 tdecode.cu -o tdecode && ./tdecode mma && ./tdecode gemv

## Version 2 (2026-10-28 17:4x, not yet run)

The first run (log in `data/probe_results/tdecode/`) was exact on every route, measured the 1-bit tensor path at 8.00x the int8 bit-products and the popcount 8th bit at about 2 % over folding, but every 7-bit route ran 1.4-2.8x slower than Q8_Z: the kernels gave one thread per output row (about 1.4 warps per scheduler, a latency-bound chain), and the timing was L2-warm (Q8_Z at 895 GB/s, above the 504 GB/s of VRAM). Fable's read-only review agreed and weighed the author's array idea: it holds (separate arrays by role make "read H on f of the blocks" cost exactly f x 1 bit per weight), with H kept grouped by input block.

`tdecode gemv2` runs the version 2 kernel (`gemv_w`): each lane takes 4 consecutive 32-blocks of one row (K / 128 lanes per row, shuffle reduction, 16-byte loads), activations read from global, the identity sum (2b - 127) x = 2 dp4a(b, x) - 127 sum x, H as spread bytes into dp4a or by AND + popcount, the ZH arrays with H block-major. Timing is cold: 64 MiB written between launches, median of 50. Checked before any run: device code compiles to PTX for sm_89 and the host code passes a clang syntax check; the integer paths emulated on 6,000 blocks, 0 mismatches (`tools/experiments/tdecode_kernel_emul_v2.py`).

    nvcc -std=c++17 -O3 -arch=sm_89 tdecode.cu -o tdecode && ./tdecode gemv2

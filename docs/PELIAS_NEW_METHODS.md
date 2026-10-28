# Pelias new methods tally

Unconventional or newly derived methods that are **proven to work**. Entry rule: measured on real hardware or real weights, with the result resolved (95 % interval clears zero) or measured directly (kernel throughput). Ideas that tied, missed or are untested stay in docs/PELIAS_GAME_PLAN.md, not here.

| # | Date | Method | What it does | Proof (MEASURED) | Where recorded |
|---|---|---|---|---|---|
| 1 | 2026-10-27 | Walsh-Hadamard rotation per 32 values on the dense inputs | Removes the 8-bit activation rounding cost on the dense layers | vs the same rounding unrotated: -0.001186 [-0.001913, -0.000458], t -3.23; vs no rounding: -0.000211 [-0.000586, +0.000163], no difference (R1, 25,500 tokens) | Game plan Block J, R1 |
| 2 | 2026-10-27 | Dense weights rotated per 32 from bf16, then quantized | Lowers the whole file's drift at the same size | vs unrotated twin: -0.003454 [-0.004585, -0.002323], t -6.06, **-8.9 % drift** (R2) | Game plan Block J, R2 |
| 3 | 2026-10-27 / 10-28 | Rotated file + 8-bit activations on every weight | 8-bit activations become free in a real quantized file | R2 dense q8_1 vs R2 fp32: -0.000267, not resolved; t8r all weights 8-bit vs t8r fp32: +0.000165 [-0.000148, +0.000478], not resolved. Unsloth's file pays +0.000633 [+0.000220, +0.001046] for the same switch | Game plan Block J R2; Block M price list; Block N |
| 4 | 2026-10-26 | int8 GEMM v10 with the activation quantization fused into RMSNorm | Faster than the bf16 path on every shape; removes the M=8192 slowdown (a separate memory-bound quantization pass) | 96.3 TOPS; 1.39x to 1.84x the bf16 path including norm and conversions, all 20 shapes | BUILD_LOG 2026-10-26, Batch C |
| 5 | 2026-10-26 | Grouped int8 launch for all experts (v12) | One launch for 256 experts at decode-size batches | int8 2.5x to 3.0x the bf16 kernel grouped the same way, M=64 | BUILD_LOG 2026-10-26, P-K2b |
| 6 | 2026-10-28 | _PT: word-aligned packing with the spare bits used as tags | Spare bits carry scale information instead of being wasted | 28-67 % lower rounding error than leaving the spare bits unused, real blk.20 weights (sandbox) | Game plan Block M, _PT tag test |

Methods 1-3 are one idea (rotation per 32); urm (game plan) is the test that turns it into a result against Unsloth at the same size.

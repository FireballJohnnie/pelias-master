# Batch C results (kernel-efficiency probes P-K1, P-K2, P-K2b, P-K3, P-Q10)

Raw output of `tools/probes/batch_c` on the reference machine (RTX 4070, Windows 11 native), 2026-10-26. Every run is kept, failures and superseded kernels included (ruling 2026-10-26 13:21). Each folder is one revision of the probe source (see `tools/probes/batch_c/README_BATCH_C.md`); the reading and the adopted kernels are in masterwork 17.5. The Nsight Compute reports themselves (`k2prof.ncu-repz`, 14 to 20 MB each) stay on the author's drive; their exports are here as `k2prof_raw.csv` and `k2prof_details.csv`. WSL2 results go in `wsl/` when they run.

| Folder | Local time | Probes | What it showed |
|---|---|---|---|
| `windows/r1` | 14:30 | all | P-K1 15/20 shapes at >= 0.85x cuBLASLt (own bf16 GEMM at 0.47 to 0.85 of the P-H3 peak); P-K2 int8 1.1x to 1.5x the bf16 path at 0.18 of the int8 peak; P-K3 Q4_K 487 of 506 GB/s and Q6_K 501 of 510 GB/s (memory-bound); P-Q10 e4m3 cosine >= 0.99965 (passes 0.999), e5m2 0.9986 (fails) |
| `windows/r2` | 14:41 | all | Three tile sizes and two orders: P-K1 17/20 (0.48 to 0.88 of peak, best 0.96x cuBLASLt; lm_head 0.50 to 0.93x); remaining below 0.85: attn_k M=2048 (0.72), router M=2048 / 4096 (0.846 / 0.73) |
| `windows/r3` | 14:49 | k2 | int8 v3 (exact integer-to-float, ldmatrix): 0.96x to 1.03x of v2 |
| `windows/r4` | 14:52 | k2 | Same binary as r3 (the link step was skipped); kept as a repeat of r3 |
| `windows/r4b` | 14:53 | k2 | int8 v4 (products issued before scaling): 0.96x to 0.99x of v2 |
| `windows/r5` | 14:57 | k2 | Registers and occupancy reported: v2 216 registers, one block per SM; v5 capped at 128 spilled 160 bytes, 0.54x to 0.70x of v2 |
| `windows/r6` | 15:00 | k2 | v6 (smaller warp tiles, two blocks per SM without spills): 0.68x to 0.97x of v2 |
| `windows/r7` | 15:15 | k2 | v7 (activation scale per 256, weight sub-block scale in int32): 1.06x to 1.08x of v2, cosine against the bf16 path 0.999974 |
| `windows/r8` | 15:24 | k2, k2prof (Nsight Compute) | v8 (3 or 4 load stages): 3-stage 1.17x to 1.21x of v2; the profile measured the limit (tensor busy 33 %, 2 warps per scheduler, 16.8 instructions per int8 mma) |
| `windows/r9` | 15:34 | k2, k2prof | v9 (16 warps per tile): 0.97x to 1.07x of v8; waits per instruction rose 5.6 -> 9.35 cycles |
| `windows/r10` | 15:43 | k2, k2prof | v10 (min term as one exact bf16 tensor product per super-block): best, 96.3 TOPS, 1.50x to 2.00x the bf16 GEMM; 13.4 instructions per int8 mma |
| `windows/r11` | 15:53 | k2 | v11 (v10 with 16 warps): 0.88x to 0.95x of v10. Activation conversions with a cold L2: RMSNorm fused with the int8 quantization costs the same as RMSNorm to bf16; int8 path including norm and conversions 1.39x to 1.84x the bf16 path on all 20 shapes |
| `windows/r12` | 16:08 | k2b | Split-K (at best 1.04x), bf16 output (0.92x to 0.99x), MoE experts 256 x M=64 grouped (int8 2.5x to 3.0x the bf16 kernel grouped the same way), layer-order L2 reuse (+0.6 to +1.2 % for the whole slice) |

Labels: every number in these files is MEASURED on the reference machine, except fields named `*_DERIVED` or `*_derived`. Weights and activations are synthetic (seeded); the Q4_K layout and formulas follow llama.cpp (READ-SRC, masterwork 5.3).

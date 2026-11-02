# Pelias: open items after the endgame

Confidential, not for distribution.

Written 2026-11-01 19:45 EST (author: "put those into a markdown laundry list in order, grouped by a, b, c ... numbered 1, 2, 3").
The endgame (the token kernel, the MTP fix, the final sweep) is in docs/PELIAS_GAME_PLAN.md under ENDGAME. Everything below comes
after it, in the order to take it. Labels as in the game plan: MEASURED, DERIVED, ESTIMATE, UNMEASURED.

## A. Engine speed, still open

1. **12k context decode.** Gate PG7: decode >= 55 tok/s at 12k. KV q8_0 (B6, author ruled 10-31 15:09): 491 MB of fp32 K / V per token at 12k becomes ~123 MB (DERIVED). Attention split over positions (the current output kernel walks every position serially per thread). Decode at 12k is not measured since the reader era.
2. **GP1-09, the 4096 base.** Re-measures the KV q8 and DeltaNet int8-row checkpoint costs at longer context.
3. **Prefill track.** Gate PG7: 1,670 tok/s at 12k (today 46 tok/s). Prompt tokens grouped by expert; each non-bank expert streamed once by the copy engines into a double buffer while int8 tensor-core tiles run the previous group. DERIVED: ~740 tok/s at 512-token chunks, ~2,970 at 2,048. Makes the prompt independent of the bank size.
4. **Q5_K / Q6_K int8 prompt tiles.** The open part of the tensor-core prompt path (Q4_K and Q4_TG2 done, Q4_TG2 at 1.1e-7, MEASURED).
5. **MSI step.** Memory overclock above +1000 in +100 steps, read-rate probe at each step (stop where GB/s stops rising); "CUDA - Force P2 State" = Off (decode sits in P2 at 11,251 vs 11,501 MHz, MEASURED); Hardware-Accelerated GPU Scheduling on vs off, measured; one gate at the chosen setting.
6. **Native Windows build and measurement (Line F, GP4-04).** No VM layer, 0.07 s/GiB pinning, BAR1 flags, honours no-sysmem-fallback; the tier build ran 5.0-5.3 ms per token faster there (MEASURED).
7. **Start-up.** 2 MB arena registration (36x faster, MEASURED); a resident pin host (dlopen the engine) for test cycles.
8. **Bank / cache sweep (GP4-02).** Config only; replay says bank 40 + cache 33 is the best split at today's VRAM.

## B. Infinite context loop (GP3-14c)

1. **C1 page recall.** Sandbox on existing q / k dumps: share of attention mass in the top 1/8 and 1/32 of pages (the 1-bit page gate kept 72.6 % at 1/8 vs Quest 62.0 %, MEASURED).
2. **Tiers.** VRAM: page summaries and 1-bit key codes, a bank of hot pages; RAM: every q8 KV page and the DeltaNet checkpoints up to the probe's budget; NVMe: overflow only (U11 prefix store, CRC32 keys).
3. **Each step.** Score summaries, then candidate pages' codes, pull the top pages' KV on a copy engine, attend exactly.
4. **Positions (way 1, default).** Keys stored before RoPE; pulled pages placed at fresh positions inside the trained window. Way 2: YaRN / NTK scaling as a toggle. Way 3: longer-context fine-tuning (trainer).
5. **Tests in order.** Re-placed pages vs full attention (paired KL on text that fits the window); YaRN x2 kld at short context; needles past 262,144 (PG7 needles 12/12 to 120k, extended).
6. **Sizes (DERIVED).** 1M tokens = ~10.9 GB of q8 KV, codes ~0.8 GB, summaries ~12.5 MB; a 64-page pull ~45 MB: under 2 ms from RAM, ~15 ms from NVMe.

## C. Accuracy (the free-accuracy goal)

1. **Planes in the engine, top 4 picks, on the CPU.** 0.008800 KL (-6.4 % vs tiered, MEASURED on the CPU reference), 0 VRAM; GPU first, exact CPU late; off the token path, 0 % speed cost to stay on.
2. **CPU accuracy check.** Sampled, L3-resident experts only, may skip, never waits (THE PLAN rule, 18:05-18:07).
3. **Activation coin flips.** GPU first, exact CPU late (standing rule 10-31 01:16).
4. **Tie band vs block error energy.** B = 0.1 against the top 10 % / 20 % of 32-blocks, against +0.001009 KL.
5. **The "stuck 1 %".** Places that hold 23-26 % of the KL on WikiText (7_T_H plan).
6. **Head sketch recall.** Top-1-in-top-64 of the 2-bit sketch from existing logits (speed and accuracy both depend on it).

## D. Files and evidence

1. **Line B.** A held-out chat + code text (B4; the calibration text cannot be used), tier_v1 scored on it, the hot set re-profiled.
2. **Line E.** The combined file from BF16 (rotated Q7_ZT dense, tiered experts, v2 dense and head planes) and the Pelias-made scoring base (B9 / GP2-12).
3. **Group 3 proof.** A stock third-party MoE GGUF through the bank / cache + reader on WikiText.
4. **Novelty ledger and PDF revisions.** Patent first: nothing public before filing. **Scope later (author 2026-11-02 01:54-01:55):** revise every document as necessary with the results since 2026-10-31 (v18-v54d engine runs, planes in the speed path, the gate line, GPU-only decode, the WSL two-copy finding, the plane-data budget): Pelias_7bit_system_and_1bit_decoder (rev4 -> rev5, make_7bit_pdf_rev4.py), Pelias_planes_encoding_decoding (rev2 -> rev3, make_planes_pdf_rev2.py), Pelias_pathways (v7 -> v8, make_pathways_pdf_v7.py; whole-project line; check whether a separate private pathways doc exists), Pelias_how_the_speedup_works and Pelias_Section21_calibrated_gains (rev5 -> rev6, make_pdfs_rev5.py), docs/PELIAS_RESIDUAL_PLANES.md (new section). With it, a patent pass: every new method each document describes gets an entry in the invention-disclosure package (what it is, document and section, dated conception line and who conceived it, Claude's adopted implementation choices, measured result, prior art from the novelty ledger or "not yet searched"). First estimate 120-160k tokens, 2-3 h (ESTIMATE); to be scoped with the author.
5. **GP2 docs.** GP2-02 / 02b status corrections, GP2-07 parameter conflicts, GP2-08 data port, GP2-13 namespace, GP2-14 version string; Appendix B open: B2, B8.

## E. Product

1. **Server and web interface (GP3-15 / 15b).** Port 8062.
2. **Small-card layout.** A locked int4 expert core plus free 1-bit planes.
3. **GP3-03 AWQ, GP3-04 GPTQ, GP3-05 data pipeline.**
4. **GP3-13b idle-work scheduler.**
5. **GP3-16, masterwork steps 13-17.**

## F. Trainer phase

1. **GP3-12 backward / LoRA, GP3-12b parity.**
2. **GP3-13 repair training.**
3. **Mode Q and GP1-08 (re-quantize the same bytes); GP2-03.**
4. **B12 LoRA scaling.**

## G. Release (GP4)

1. **GP4-01 benchmark.**
2. **GP4-02 sweeps.** Bank size, Lane D, threads / poll, huge pages, live hit rate.
3. **GP4-03 acceptance.** Restore the persona criterion.
4. **GP4-04 hardware.**
5. **GP4-05 gates and trademark check; GP4-05b future tests.**
6. **GP4-06 freeze.**

## H. Parked

1. **V100 32 GB test machine.** R580 + CUDA 12.9 (Volta dropped in CUDA 13).
2. **A shared view of RAM between WSL and Windows.** Outside this project's scope.
3. **The wormhole DAX tail test.** After Pelias-master 1.0.
4. **The 8 / 12 / 16-thread CPU sweep.** Until another CPU sweep is needed.

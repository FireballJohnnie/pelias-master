# Fable review: R3 fix 6 (c1/c2/c3) and the OS-overhead hypothesis (2026-10-29, read-only)

Geometry (DERIVED from run_stream, 736 warps): DeltaNet stages 6144 / 2048 / 4096 / 2048 tasks = 23.7 / 7.9 / 10.2 / 6.0 MB = 8.35 / 2.78 / 5.57 / 2.78 rounds; last partial round 256 / 576 / 416 / 576 warps busy (attention stage 0: 192); head 169 rounds. Per boundary over (b) path 2 (MEASURED): grid sync 7.73 us, counter 8.38, + 1-task prefetch 6.79, + whole share 13.54.

## 1. The coordinator's reading: partly right, and it misses the largest piece

- **The partial last round costs ~0.4 us, not 5.4.** The last round is never small (192-576 of 736 warps). The ceiling log shows 0.75 MB in flight still gives ~380 GB/s and 3.0 MB gives 494 (MEASURED), so a 192-warp round (0.74 MB) runs at ~380: loss (1/380 - 1/472) x 0.74 MB = 0.4 us (DERIVED). The real costs are the drain (in-flight bytes decaying to zero over about one loaded latency) and the ramp after release (736 warps re-run the 11-probe search, then wait one DRAM latency while the queue refills): ~2-3 us together (DERIVED). c2 removed exactly one latency: c1 - c2 = 1.59 us per boundary (MEASURED).
- **Missing piece: format serialization.** Standalone rates (MEASURED, f6b): Q7_ZT 467, Q4_TG2 path 2 318, Q5_K 425, Q6_K 460 GB/s; their sum over the token is 5.25 ms, yet (b) is 4.82 (MEASURED): in (b) warps drift across job boundaries and Q4_TG2's compute hides under Q7_ZT traffic. A barrier forbids that, so stage 2 runs at ~320-350, not 472. Upper bound: 0.44 ms = 2.7 us per boundary (DERIVED: sum of bytes/standalone - bytes/472). A third of c2's 6.8 us, and no barrier work removes it; only a memory-bound Q4_TG2 does. Test alone: (b) on a format-sorted table (no dependencies in (b)); if it rises toward 5.25 ms, confirmed.
- **Counter contention:** c1 - cg = 0.65 us per boundary (MEASURED). 736 lane-0 spinners (an L2 round trip each ~0.5 us; __nanosleep(32) is a hint) plus 736 same-address atomics load one L2 slice that also serves 1/N of the last tasks' lines; cg polls per block (92). Measure with a counter-only, empty-stage kernel (the c0 analogue). Fix: the last arriver writes a release flag replicated per SM (46 lines) and warps poll their SM's line.
- **c3 is the head:** each warp prefetches its whole head share, 417 MB into a 36 MB L2 consumed from the front: re-read 417 MB / 472 GB/s = 0.88 of the measured 1.08 ms (DERIVED). The rest (~1.3 us per boundary) is 8-9 extra lookups per warp before the wait (the last arriver delays the release by that) and 24 MB prefetch bursts queued with the last demand loads.
- **Code and timing: no bug.** Epochs are right (warm-up = 1, reps 2-8, &ep read at launch, memset between kernels); every t in [begin, end) is one task, so the target is exact; fence + atomic + acquire + __syncwarp is a valid chain; tb2 as base is consistent. Notes: r3_staged still syncs after the head (0.5 us); `__ldg` on xq/xs/xsum is the non-coherent path, wrong once a stage writes what the next reads in the same kernel (plain or `.cg` loads after the acquire); print min/max beside the median.
- **Peak correction:** the memory clock under load is 11,251 MHz in every CSV (11,501 only idle) (MEASURED): peak during runs is 540 GB/s, not 552; the 494 ceiling is 91.5 % of it, (b) 87 %.

## 2. The OS-overhead hypothesis, for R3

Device-side rows are OS-independent (MEASURED, f2 WSL vs Windows, same source): (b) path 2 4.546 vs 4.563 ms, (c) 5.938 vs 5.831, Q7_ZT-only 0.573 vs 0.572, Q4_TG2 path 1 0.774 vs 0.770, bare sync 0.52 vs 0.52 us, in-kernel SM clock 2.774 GHz on both. The 8 % Q6_K and 3 % Q5_K gaps are a compiler effect: Windows is nvcc 13.4 (122/126 registers), WSL 13.0 (124) (BUILD_LOG 10:30), and only those two formats decode through memcpy-to-byte-arrays. Verify with `cuobjdump -sass | md5sum`, or ship one cubin.

What the OS does touch, and how to see it on any machine:
- **Submission:** (a) - (a') = 2.9 ms native, 4.8 ms WSL over 1,091 launches = 2.7 / 4.4 us per launch; graph replay 19.2 ms on both; the GPU-side cost per small launch is 17.6 us (19.2 ms / 1,091, DERIVED), larger than either OS path. One launch per token removes all three. Record WSL/native, HAGS (HwSchMode), `cudaDevAttrTccDriver`.
- **Other GPU clients under WDDM** (compositor, browser): compute preemption can stall a 5 ms kernel by a frame's work (~0.1-0.5 ms); invisible to a median of 7, visible as outliers. Record min/max of >= 20 reps; the engine tolerates it rather than designing around it.
- **Exec timeout:** `cudaDevAttrKernelExecTimeout` = 1 means a per-launch limit (TDR 2 s on WDDM); cap tokens per persistent launch from the measured ms per token.
- **Power / thermal:** 0x4 appears only with the fast kernels at 180-187 W of 200 and costs 15-45 MHz of 2,775 (MEASURED, CSVs); 0x20 once at 48 C on Windows, memory junction unlogged (overclocked GDDR6X: log it). Only compute-bound phases (Q4_TG2 stage 2, the head) follow the SM clock; the probe's clock64/globaltimer ratio already measures it per run; print GB/s as a fraction of NVML's live memory clock x bus width, not the stock attribute (10.501 GHz).
- **Not OS-dependent:** barrier cost, drain/ramp, format serialization, L2 behaviour, cooperative residency (consumer GPUs time-slice contexts, never share SMs).

## 3. Routes to close 4.82 -> 5.91 ms (ranked; test each alone)

1. **Rolling one-task-ahead L2 prefetch, always on** (prefetch t + 736 before computing t). Doubles bytes in flight (2.6 -> 5.3 MB; the ceiling wanted 3 MB), fills drain and ramp with no cap to tune, and is what makes Q4_TG2 memory-bound. Expected: -2 to -3 us per boundary (0.3-0.5 ms) plus up to 0.34 ms of stage-2 serialization (DERIVED); c2 -> ~5.1-5.3 ms, (b) possibly 4.82 -> ~4.6-4.7. Tests: Q4_TG2-only row (target >= 450), then (b), then c2.
2. **Incremental job lookup:** carry `lo` across a warp's tasks (t monotonic, jobs >= 256 tasks: <= 3 sequential L1 probes instead of 11 dependent ones) in run_task and prefetch_task. Required by 1; a few % of (b) alone.
3. **Per-expert counters at gate/up -> down** (item 7): 40 of 160 global barriers become 8 counters each; the last gate/up warp of an expert does silu(g) x u and the int8 quant. ~40 x 3 us = 0.12 ms, more if Q4_TG2 and Q5_K overlap. The other three per layer are real all-to-all points.
4. **Per-SM release lines** (section 1): ~0.65 us x 160 = 0.1 ms.
5. **Byte cap:** the hole is ~472 GB/s x 3 us = 1.4 MB per boundary = ~2 KB per warp = one task (DERIVED): c2 already is the cap; whole-share has no headroom. Superseded by 1.
6. **Tail splitting / work stealing:** a 0.4 us effect; skip. `__launch_bounds__(256, 3)`: cheap to try, likely spills.

## 4. Other things that look off

- Soak 518 GB/s on the 4-layer cycle beats the 494 ceiling; 40-distinct gives 479: ~5 % L2 reuse, so ceiling claims must cite --distinct (done from 10:46).
- Windows 0x24 at 48 C GPU points to a memory-temperature limit on overclocked GDDR6X; a long soak could drop the memory clock mid-run. Add memory temperature to the CSV.

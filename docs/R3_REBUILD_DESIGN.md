# R3 REBUILD — DESIGN (item 6; written 2026-11-06 17:45–18:1x EST from a read of gpu_stream.cu, moe_dev.cuh, the engine's step paths and the v55–v60q measurements)

Copyright (c) 2026 John Lipka and the Pelias contributors. Confidential, not for distribution.

A design, not code. Every number carries its label (MEASURED / DERIVED / READ-SRC / ESTIMATE / DECISION / UNMEASURED); a measurement on the PC beats any derivation here. The rule from the game plan holds: one engine path, variants are template parameters, the loser leaves; GP5 byte for byte and 128/128 are the gates of every stage that does not intend to change numerics. "Right, but not llama right": nothing here imports another engine's structure; it follows what our own stamps and counters say.

## 0. Where the time goes today (MEASURED v60p/q, fresh VM, 3 bits, WSL)

| Part of one trunk token step (11.53 ms median) | ms | Label / source |
|---|---|---|
| Fused MoE half, 40 layers: A norm + router + shared gate | 0.46 | MEASURED v60q A2s (GPU clock, block 0) |
| B picks + grouping + plan (block 0) | 0.85 | MEASURED |
| **C expert stage** | **4.19** | MEASURED; its byte floor is ~0.9 ms (8 picks × 40 layers × 3 matrices × 1,048,576 values × ~0.44 B ≈ 0.44 GB at ~500 GB/s, DERIVED) → runs at ~105 GB/s effective |
| D combine + residual | 0.03 | MEASURED |
| DeltaNet dense half (L3), 30 layers: 0 norm 0.23 / 1 qkv + gate + β/α 2.01 / 2 conv + l2 + recurrence 0.61 / 3 ssm_out 0.59 | 3.44 | MEASURED; the dense weights' byte floor ≈ 3.2 ms (trunk dense ≈ 1.6 GB, DERIVED from the 2.01 GB dense + head) → already near its bytes |
| Attention layers (10), head, residuals, gaps between ~N graph nodes | ≈ 3.4 (= 11.53 − 5.53 − 3.44 + the dn-half layers' share…) | DERIVED remainder; N UNMEASURED (instrument: `cudaGraphGetNodes` count on the graph note, one line) |

Verify-3 step (MEASURED v60q A2s): MoE half 14.07 ms = A 0.74 / B 1.60 / **C 11.67** / D 0.05 — the expert stage for three tokens costs 2.8× the one-token stage although every weight row is read once (`lane_part_n<M, NT>`, v47) → the per-token work *after* the load is what costs, not the bytes (§1c).

Expert stage per-format cost (MEASURED v60q counting arm, cycles per warp task = 16 blocks of 256 values, one token):

| Format (bytes per task at 3 bits) | cycles / task | share of task cycles | cycles per byte vs Q8_0 |
|---|---|---|---|
| Q8_0 (4,352 B, the shared expert, VRAM) | 4,906 | 6.1 % | 1.0× |
| Q6_K stored (3,360 B) | 6,647 | 6.1 % | 1.75× |
| **Q4_TG2 planes (1,792 B)** | **12,964** | **31.8 %** | **6.4×** |
| **Q5_K planes (1,792 B)** | **8,409** | **55.6 %** | **4.2×** |
| silu × up (kEpi, one per expert) | 26,247 | 0.4 % | — |
| staged (host RAM over PCIe) tasks | 20,644 | 8.1 % of task cycles (3.5 % of tasks) | — |

Warp time (MEASURED): running tasks 57.2 %, **waiting for a group 32.2 %**, rest (job lookup, signals) 10.5 %; 18.8 tasks per warp; two resident blocks per SM did not move the wait (v60q A1s2: −0.04 ms).

MTP block per draft (MEASURED v60q): input 0.12–0.17 + eh_proj 0.07–0.11 + attention 0.23–0.43 + MoE half 0.16–0.19 = 0.63–0.90 ms, all parts scaling together across arms while the bandwidth-bound head holds 0.37–0.39 → launch-latency-bound (its bytes are worth ~0.2 ms). READ-SRC: `mtp_run` is not under a CUDA graph (the trunk's `run()` captures `layers()` only; the embedding gather and the position copy stay outside and the layers read the position from the fixed `pos_d`), so the draft's ~15 launches are eager launches on WSL/dxg.

## 1. What the source says (READ-SRC), and what it means

**a. The expert stage is decode-bound, not byte-bound.** Q4_TG2 planes cost 6.4× Q8_0's cycles per byte, Q5_K planes 4.2×; together 87 % of the stage's task cycles. Inside `lane_part_n` the Q4_TG2(P) path decodes through a float LUT in shared memory — per weight per token one `s_lut` load (replicated, conflict-free) + `i8f` (the int8→float of the activation, redone per weight) + an FMA; the codes are unpacked once per lane part (`tg2p_codes8`) but the LUT work is per token. The Q5_KP / Q4_KP path in the fused kernel takes the **1-bit route** (`J.xb` set: `plane_dot32` = AND + popcount per plane per token); the dp4a route exists in the same branch (`q5p_word` → `__dp4a`) and is selected only when no activation planes are passed. E2/E3 MEASURED on the head sketch that `popc` runs at a quarter of the INT32 rate on this card and the dp4a form of the same integers was faster (0.50 → 0.37 ms): the same question stands for the expert stage and has never been A/B'd on today's kernel (v55f measured the 1-bit route exact; its speed comparison predates v55f3/v57).

**b. The 32 % group-wait is structural, not occupancy.** Tasks are dealt statically in table order (`t = w0; t += warps`): gate u and up u for every expert (2U × nt_gu tasks), then U silu × up tasks, then U × nt_dn down tasks; a warp reaching a down task of expert u spins (`ld.acquire` + `__nanosleep`) until expert u's 2 × nt_gu gate/up tasks are complete. With 368 warps and 18.8 tasks each, the first warps to finish gate/up land on down tasks whose groups are still running. v55f's pipelined *table order* lost (+54 % waits, prefill 121 → 84 tok/s) because dealing stayed static; v60q showed two resident blocks per SM do not hide the wait either. Nothing has yet tried **dynamic dealing** (a warp takes the next ready task from a counter), which changes who waits, not the order of the table.

**c. Weights-once already exists; per-token decode is the multi-token cost.** `lane_part_n<M, NT>` loads each lane part's weight words once and loops the tokens inside — but the loop body is the LUT (Q4_TG2P) or popcount-per-plane (Q5_KP) work above, so three tokens cost ~3× the decode. C 11.67 ms for 3 tokens vs 4.19 for one is this. **Decode once, dot many** is the form: unpack to int8 words once per lane part, then one `dp4a` chain per token (8 dp4a per 32 values) — exactly what Q8_0 and stored Q4_K / Q5_K already do, and why they cost 1.0–1.75× per byte.

**d. Q4_TG2's grids decide how far "decode once" goes exactly (READ-SRC `quant_cpu.cpp` Q4TGGrids, v2):** grid 0 = (i − 7)/8 (uniform, exact zero), grid 3 = 0.6 × grid 0 (uniform) → integer codes × a per-tag float scale: **exact on dp4a** (code − 7 as int8, scale 1/8 or 0.075 applied once per 8-value group); grids 1 and 2 = log grids (1.3^m / 1.3^7, 1.45^m / 1.45^7, grid 1's smallest positive level = 0) → **not integer-representable**: exact only through the float LUT (or eight FMAs with the levels in registers). So an exact fast path exists for tags 0 and 3; tags 1 and 2 keep float math. **The tag mix of the tier file is UNMEASURED** — a host tool over the file's Q4_TG2 blocks (minutes) decides how much of the 31.8 % the exact dp4a path recovers. Replacing the log grids by integer grids is a *format* decision (accuracy changes; P4's gate), parked here, not taken.

**e. The copy pipeline is worth 1.03 ms per token (MEASURED v60q) and keeps 1 block per SM** (82,944 B of shared per block). Any rebuild of the stage keeps the pipeline (or something that delivers the same prefetch) — the stage reads ~1.8–4.4 KB per task from VRAM or PCIe, and the measured difference says latency hiding matters more than occupancy.

**f. The dense side is near its bytes.** L3 stage 1 (qkv + gate + β/α) 2.01 ms over 30 layers reads the DeltaNet projections (Q8_0 / Q6_K) at roughly the dense ceiling; the dense half is not where the rebuild's ms are. It gets the op program's launch savings and nothing structural.

**g. The MTP block is launch-bound and uncaptured** (§0). The trunk's graphs already show what capture does on launch-bound steps (v26: 3–6.6× on those kernels; 0.716 µs per kernel in a replay vs eager launches).

## 2. The rebuild, in stages (each its own commit and gate; nothing skipped, nothing assumed)

### Stage 0 — measure, and take the two cheap exact wins (no kernel surgery)
0.1 **Decode microbench** (a `tools/probes` program, host-run by the author; no engine change): each stage format's lane-part body with the weight bytes already in shared memory, 1 and 3 tokens, 2/3/4 bits for the plane formats, both routes for Q5_KP/Q4_KP (1-bit vs dp4a), the Q4_TG2P LUT path vs a dp4a-by-tag path on synthetic uniform-tag data — **cycles per 256 values per token with no DRAM in the way**. This is the number the stage's cycles-per-task table lacks: how much of 12,964 / 8,409 is decode. Prediction (DERIVED from the per-byte ratios): decode is ≥ 60 % of the Q4_TG2P task and ≥ 50 % of the Q5_KP task at one token; at three tokens the decode share rises toward 80 %.
0.2 **Tag histogram** of the tier file's Q4_TG2 groups (host tool): the share of tags 0/3 (exact dp4a) vs 1/2 (float). UNMEASURED today; ESTIMATE from the quantizer's choice rule (nearest grid by error): the log grids win on heavy-tailed groups, the uniform ones elsewhere — no number until measured.
0.3 **Graph node count** on the engine's graph note (`cudaGraphGetNodes`; one line) — N launches per trunk step, so the op program's launch saving is a number, not a guess.
0.4 **The MTP draft under a CUDA graph** (engine; the cheap exact win). As the trunk: the id gather stays outside (it reads the rotating ring slot), the position goes through the fixed `pos_d`, the chain from the strided copies to the pick is captured per key (h_prev buffer ∈ {h_d, hn_d}, output index 0..3, sketch on/off; ≤ 8 graphs, no host sync inside — verified by reading `mtp_run`: none). The host-side stamp clearing stays outside the graph. Gate: 128/128, kept 59/68 & 76/102, GP5 untouched. **Prediction (DERIVED from §0 and the trunk's 0.7 µs per replayed kernel): block 0.63–0.90 → ~0.25–0.35 ms; draft 1.14 → ~0.75–0.85; M1 +1.5–2 %, M2 +2.5–3 % (→ ~92–93 tok/s); and the arm-to-arm swing of the block should collapse, which is the test of the launch-latency reading itself.**
0.5 **Fused-path route A/B for the plane formats** (engine knob for one gate, loser leaves): the stage without activation bit planes (`J.xb` null → the dp4a route) vs today's 1-bit route. Same integers by construction (both branches exist; GP5 byte for byte is the check). Prediction (DERIVED from E2/E3's popcount finding, ESTIMATE on the stage): Q5_K planes 8,409 → ~6,000–7,000 cycles per task, C 4.19 → ~3.7–3.9 ms (−0.3..−0.5), and the 3-token C by more; if the 1-bit route wins instead, that is the record and the design keeps it.

### Stage 1 — the expert stage: decode once, dot many; dynamic dealing (exactness-preserving)
1.1 **Decode once per lane part** for every format: unpack the row part's codes to int8 words in registers once (Q5_KP/Q4_KP: `q5p_word` once; Q4_TG2P uniform tags: `code − 7` bytes once; stored formats already are words), then per token the dp4a chain + the per-sub-block scale math. Q4_TG2P log tags keep the exact float form, but with the activations converted once per sub-block (not per weight) and the level table in registers (16 floats per tag, two tags) instead of a shared LUT load per weight. Gate: GP5 byte for byte (the integers and the float sums are the same operations in the same order — the design keeps the summation order per sub-block; where a reorder is unavoidable the unit test "4/5 bits = stored" and the rung "one token = verify bit for bit" decide). Prediction (from 0.1's measured decode share; ESTIMATE until then): Q4_TG2P 12,964 → ~7,000–8,000 cycles per task, Q5_KP → ~5,500–6,500; **C 4.19 → ~2.8–3.2 ms one token; verify-3 C 11.67 → ~6–7 ms** (the second and third tokens pay dp4a only).
1.2 **Dynamic dealing.** Tasks taken from a per-launch atomic counter in table order (gate/up first, then epi, then down) so that the warps that would spin on a down group instead take the next unfinished gate/up task — the dependency structure is unchanged (no deadlock: the lowest unfinished task can always run), only the assignment moves. Counter contention: one atomic per task per warp (6,900 per layer) is small next to a task's 5–13k cycles; the FABLE_R3_FIX6 note (counter contention, per-SM release lines) applies — counters on their own 128-B lines (the barrier-counter padding item), per-SM sub-counters if the probe says so. Prediction (ESTIMATE): group-wait 32 % → 10–15 %; C −0.4..−0.8 ms on top of 1.1. Measured by the existing counting kernel; if the wait does not fall, the wait is the silu×up step's latency (one warp per expert, 26k cycles) and the next form is splitting kEpi across warps.
1.3 **Fix 3:** the `s_lut` rule (stored-grid jobs and planed jobs in one plan; today the plan refuses the mix — keep the refusal or carry two grids: decided by 1.1, which removes the LUT for uniform tags) and the Q6_K 16-byte repack at open (`kQ6KS` 210-B 2-byte-aligned → `kQ6K` 208 B + d6 side array; 2.7 % of expert bytes at 1.75× cycles per byte; −0.05..−0.1 ms DERIVED). Bank bytes +0.
1.4 The pipeline stays (1.e); the `(256,2)` instantiation and the `moe_blocks` / `rx_pipe` knobs leave here (measured losers, v60q).
Gate for stage 1 as a whole: T (all rungs), GP5 byte for byte, 128/128, the counting arm's per-format cycles and wait share, A1s/A2s tok/s. **Target on the record (DERIVED): one-token step 11.53 → ~9.5–10.0 ms (100–105 tok/s); verify-2 17.5 → ~13–14 ms; M2 decode phase 1.20× → ~1.3×.**

### Stage 2 — the op program (THE FINAL FORM's one cooperative launch per step)
One persistent cooperative launch per token step walks an op list (norm, product, attention, MoE half, residual, head) with tickets (the gdone pattern generalized), so the ~N launches of a step become one and the inter-kernel drain/fill gaps go. First customer: **the MTP block** (its ~15 launches are the worst case per byte); second: the trunk step. Prediction (DERIVED, after 0.3 gives N): the trunk saves the gaps (ESTIMATE 2–5 µs each × N ≈ 0.3–0.7 ms per step) plus whatever 0.4 did not already recover for the MTP block. Gate: GP5 byte for byte, 128/128. Risk: the dense products and the attention at one block per SM lose the per-kernel grid shapes they have now — the op program must keep each op's measured block shape (the fused MoE half already shows how: phases A/B/C in one launch with grid syncs).

### Stage 3 — the dense half (after 2)
Near its bytes (1.f). Items that remain are measured small: ssm_out 0.59 and conv/l2/recurrence 0.61 are kernels with fixed costs per layer; the op program removes their launch gaps. No structural change planned until the stamps say otherwise.

## 3. Order of runs and what each decides
1. 0.1 + 0.2 (host-run probes, ~20 min of machine time) → the decode share and the tag mix fix 1.1's predictions as numbers.
2. 0.3 + 0.4 + 0.5 in one engine commit (I), gate v60r: T, GP5, A1s, A2s, + the route A/B arm → the MTP block under a graph (exactness + the swing test), the route decision, N.
3. Stage 1 as two commits (J: decode once + Fix 3; K: dynamic dealing), each gated; the counting arm reads the per-format cycles and the wait.
4. Stage 2 (L): the MTP block as one launch, then the trunk step.
5. Then C4's CPU plane-dot bench and the accuracy pass, as the game plan lists.

## 4. What is not known, and the instrument for each
- The decode share of a task per format → 0.1. The tag mix → 0.2. N launches per step → 0.3.
- Whether the launch-latency reading of the block swing is right → 0.4 (the swing collapses under a graph, or it does not).
- Whether dynamic dealing cuts the wait or merely moves it → 1.2's counting arm.
- Whether the Q4_TG2 log tags are a large enough share to justify a format decision → 0.2, then P4.
- The lottery's second probe state (SM streaming 23.4 / page lines 18–20) → unrelated to the above; the floor stays 0 until a 2–3× slow arm shows a number.

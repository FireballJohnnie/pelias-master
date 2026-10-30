# Pelias — everything done in the 2026-10-30 chat (about 09:36–16:35, America/New_York)

Written at the author's request (16:33): "please, please, please write out everything even if it's in bullet points
that we've done from this chat into a markdown". Companion to `docs/handoff/HANDOFF_2026-10-30_1430.md` (the formal
handoff; use the `_1430b` copy on C:). Every number keeps its label: **MEASURED** (run on the PC or in the sandbox),
**DERIVED** (computed from measured inputs), **ESTIMATE**, **UNMEASURED**. Full run records: `logs/BUILD_LOG.md`
(2026-10-30 entries); plans: `docs/PELIAS_GAME_PLAN.md`; planes: `docs/PELIAS_RESIDUAL_PLANES.md` (sections 6.x).

---

## 0. READ THIS FIRST — open mind, no tunnel vision (the author, 14:31 and 16:33)

> "Emphasize to keep an open mind. We wouldn't be here if it wasn't for that. This is exploratory and new and we are
> in the final steps. We have made proof that our theories have worked." (14:31)

> "i need the other chat to understand as much as possible how much we need to keep an open mind and not be tunnel
> visioned. i don't want this going the fable way and undoing all of our work" (16:33)

> "there has been a lot of pushback thus far." (16:35)

**On pushback:** Honest disagreement is still wanted. Give it as **a test that would settle it**, not as a refusal or
a lecture. "I think X won't help; here's a 10-minute sandbox check that shows it either way" is right. "That won't work"
followed by steering back to the conventional route is wrong. It has cost this project time, and several "won't help"
calls were overturned by measurement (rule 2). When unsure, measure first and argue second.

**What this means in practice for the next conversation — rules, not suggestions:**

1. **Do not undo, "simplify", or redo work that has passed its gate.** Everything in sections 2–9 below was measured
   and pushed. If something looks odd, read the BUILD_LOG entry and the test first, then *ask the author* before
   changing it. No rewrites of passing code on a hunch. No reverting methods because they are unfamiliar.
2. **The author's intuition has beaten the derivations more than once — treat his predictions as serious hypotheses.**
   The clearest example (verified 16:28 from BUILD_LOG / plan docs): on pure 7-bit (Q7_Z) the derivation said a
   1-bit plane would not help much (predicted whole-model cost about +0.0002 KL; MEASURED +0.001122, 5.6x worse than
   derived). The author predicted the extra bit would roughly double the accuracy and make it far more representative of its size.
(That is his prediction as he recalls it; the logs don't record his exact words.) MEASURED: the gated
   8th bit (Q7_ZT) has 1.9x less weight error than pure 7-bit (0.000060 vs 0.000113), and the full urm_ZT2 file landed
   at **8-bit parity** (KL 0.012536, +0.000134 vs 8-bit, t 0.86 = no difference) while being **236.4 MB smaller than
   Unsloth's file**. The derivation had predicted about 0.01282; the measured result beat it.
3. **A miss is information, not a verdict.** When a run misses its prediction, find *why* and look for another route
   before writing the idea off. Examples from today: the fast CPU path missed its 2x on kld → cause found (row decoded
   once per four tokens) → batch kernels → row blocks → experts 1.36x. The NVFP4 file gave NaN → cause found
   (per-weight `.scale` ignored) → fixed → 0.1334 vs llama.cpp 0.1345. BitNet logits missed 1e-4 → cause understood
   (int8 rounding of every BitLinear input sets a KL floor of about 0.0017 between *any* two summation orders) → Pelias
   sits at that floor, top-1 14/14.
4. **Little is published for what Pelias builds. Measured behaviour on this machine is the guide, not convention.**
   llama.cpp is read as a *specification* and used as a *math checker* (test-only oracles). Do not pull Pelias's
   methods back toward how llama.cpp or anyone else does things. Pelias already beats llama.cpp on the same file where
   its methods differ (Qwen3-1.7B Q8_0: exact activations KLD 0.002263 vs llama.cpp 0.00367, −38 %, t −16.28).
5. **Parked ≠ dead.** The small-card layout, GP3-14c infinite context, D-1B dense 1-bit, more planes on any file,
   int4/int8 residual ladders: all parked by the author's ordering, not rejected. "The only dead ends are dead ends."
6. **Reviews check math; they do not rebuild.** If a second model or a reviewer is used, it reviews read-only and
   reports; nothing it says is applied without the author's okay, and nothing measured is overturned by a derivation.
7. **Measured beats derived, always.** When a derivation disagrees with a measured result, the derivation is wrong
   (or its assumptions are) — record the miss, don't argue with the measurement.
8. **We are in the final steps.** The pre-build list is complete. Next is the GPU kernels into `src` and the first
   engine build — not re-litigating the methods that got us here.

**Proof the theories work (all MEASURED on the author's PC, 2026-10-30):**
- Usage-tiered experts: KL **0.009404** vs Unsloth's same-size file **0.012808** (−26.6 %, t −12.41), same bytes
  (22,663,406,784 vs 22,663,387,424), lower on 94 of 100 chunks; GPU token time +0.3–1.3 %.
- + 2 GB worst-error block planes: **0.008747** (−31.7 % vs Unsloth); top-4 picks only: **0.008800**, keeping 92 % of
  the gain at half the plane reads.
- MTP block vs llama.cpp: 9.4e-6 (tiny) / 9.1e-6 (real file), top-1 13/13.
- Qwen3, Qwen2, Llama families; BitNet b1.58 2B4T; FP8 and NVFP4 files: all run, all match their references.
- The whole engine builds on Windows MSVC /W4 /WX with 0 warnings, 84 tests pass, Windows vs WSL logits 2.6e-6.

---

## 1. Timeline at a glance

| Local time | What |
|---|---|
| 05:3x–08:5x (before this chat, same day) | Unsloth-size analysis, tiered experts built and MEASURED (−26.6 %), R3 speed test, plane builder options |
| 09:09–10:36 | Planes chain A1 / A2 / A2-top4 MEASURED; 1-bit + planes, 7-bit + plane, planes as int arrays (sandbox MEASURED) |
| 09:40–10:31 | Author's 1-bit / planes / small-card / ladder ideas, recorded in the game plan |
| 10:53–10:57 | Small-card moved after GP3-14; **build first** decision |
| 11:0x–12:0x | GP3-10 CPU kernels, MTP (GP3-08), all 30 decoders, int8 path, clang/ASan fixes |
| 12:01–12:16 | GP3-14c infinite-context loop designed (after the first build) |
| 12:25–12:37 | Tools check, D:\bitnet, model downloads (15.47 GB) |
| 12:30–13:24 | PC chain-3, batch fix, row blocks (v3 / v3b / v3d) |
| 12:59–13:00 | "do everything you need to do"; accuracy checks before new ideas after the build |
| 13:2x–14:28 | Dense families, Qwen3-1.7B, BitNet, NVFP4 / FP8, Windows MSVC — all five mass-appeal items done |
| 14:0x–14:30 | Q&A (FP8 error source, unified bus), handoff written, backups |
| 16:28 | 7-bit vs 7+1 verified from the records |
| 16:33 | This file |

---

## 2. Residual planes on the tiered file — chain block_tierplanes_v1 (MEASURED)

- **Build d898233:** 72 tests, 0 failures, 3 skipped (17 s).
- **Planes A1 (1,000 MB budget):** 29,411,764 of 125,829,120 expert blocks kept (23.37 %); importance-weighted residual
  7849.08 → 6420.49 (0.818; prediction 0.6–0.8: just missed); 254 s; file 1,157,315,040 bytes.
- **Planes A2 (2,000 MB):** 58,823,529 blocks (46.75 %); 7849.08 → 4796.04 (0.611; prediction 0.45–0.65: met); 288 s;
  file 2,157,315,360 bytes.
- The ~157 MB over budget = masks (1 byte/row) + row offsets (4 bytes/row); per-expert offsets would cut it to about
  32 MB (later).
- **kld A1 every pick:** KL 0.008963 ± 0.000330 (pred. 0.0088, range 0.0082–0.0093: in range); same top 95.847 %;
  vs tiered −4.7 % (t −2.26); vs Unsloth −30.0 % (t −12.27). 1489 s.
- **kld A2 every pick:** KL **0.008747** ± 0.000385 (pred. 0.0083, range 0.0077–0.0089: in range); same top 96.106 %;
  vs tiered −7.0 % (t −4.09); vs Unsloth −31.7 % (t −12.81). 1505 s.
- **kld A2 top-4 picks:** KL **0.008800** ± 0.000355; same top 96.027 %; vs tiered −6.4 % (t −2.88); vs Unsloth
  −31.3 % (t −11.18); keeps 92 % of the every-pick gain at half the plane reads (80.8 % projected). 1606 s.
  **Winner (author 10:53: "i think we have a winner").**
- Second GB buys about half the first (−2.3 % vs −4.7 %, DERIVED).
- All three predictions in range, each slightly above its point.
- "Top 4" explained to the author: planes are read only for each token's 4 highest-weighted of 8 routed experts —
  same side-file size on disk/RAM, half the plane reads per token.
- Author restarted the PC after this (map server likely timed out).

## 3. Planes on 7-bit, 1-bit and coarse bases; planes as integer arrays (sandbox, MEASURED, blk.20)

- **7-bit + plane:** Q7_Z 1.133e-4 (7.25 bpw) → + 1 plane 3.262e-5 (8.31; **−71 %**, even-noise theory said −75 %);
  Q8_Z 2.854e-5; Q8_0 2.806e-5; Q7_ZT 6.089e-5 (7.53); **Q7_ZT + 1 plane 1.980e-5 (8.59: −29 % vs Q8_0 at about the
  same size).** The gated 8th bit is the cheapest bit.
- **1-bit sign base + planes (experts):** 0.366 (1.06 bpw) → +1 0.116 (2.13) → +2 0.0372 (3.19) → +3 0.0130 (4.25):
  no blow-up, each plane about 1/3 of the error. vs K-quants at the same bits 1.8–2.8x worse (Q2_K 0.066 @ 2.63,
  Q3_K 0.020 @ 3.44, Q4_K 0.0047 @ 4.5).
- Q4_TG2 0.00418 vs Q4_K 0.00468 (−11 %); + 1 plane both about 0.0016.
- **Planes as one integer array (author 09:33):** k planes with locked scales s, s/2, s/4 … = bits of one k-bit integer,
  readable by int4/int8 tensor cores. **Locked-4 (int4, 4.125 bpw) 0.01051 vs 4 free planes (4.25) 0.01299 — the
  integer layout loses nothing and wins 19 %.** Locked-4 + 1 free = 0.00281 (5.19); + 2 free = 0.00071 (6.25).
- Author's question "would this make 1-bit the new meta?": recorded; 1-bit + planes keeps 1-bit VRAM if the planes
  stay in RAM (side file) and are added on the CPU.

## 4. Ideas recorded from the author (09:40–10:31), all in the game plan

- **Small-card layout:** locked int4 expert core + free 1-bit planes; top bit + non-expert parts in VRAM (about 7 GB,
  DERIVED), the rest in a RAM side file added on the CPU in parallel. 8 GB card core + 2 free: 0.00071 weight error
  (MEASURED sandbox), 61–78 tok/s on a 5800X3D (DERIVED); 12 GB card core + 3 free about 0.00018 (DERIVED, UNMEASURED),
  130–167 tok/s (DERIVED).
- Offered by the machine probe for **GPUs of at least 8 GB** (author's correction: "at least", not "only 8 GB"); the
  probe reads VRAM, RAM, CPU and suggests the plane count.
- **User may pick more planes while RAM allows** (64 / 128 / 192 GB machines and up), capped at core + 5 free (about
  9.4 bpw, about 1.1e-5, DERIVED) — cap to be checked by a +4 vs +5 kld.
- **Planes on any file** (1-bit, tiered, Unsloth, any n-bit), plane size shown.
- **The ladder (author 10:20):** n-bit base + 1–3 free 1-bit planes → optional int4 residual (+ 1–2 free planes) →
  optional int8 residual. Same RAM (slightly less), about the same CPU time (memory-bound), lighter compute.
  Residual estimates on the 1-bit base: + int4 about 0.0038 at 5.19 bpw, + int8 about 1e-5 at 9.19 bpw (DERIVED,
  UNMEASURED).
- **Dense 1-bit (D-1B):** the side file can stay in RAM on dense models too; 1-bit 70B / 27B speed discussed as
  DERIVED only (no public 1-bit versions of those). Author 10:31: "Scratch the dense theory for now, that'll be a test
  to explore after the build."
- Accuracy options become **GUI toggles** (masterwork 11.5), each with its measured cost and gain.
- **Three ways to get the accuracy** (09:17): run a premade file as is; add a plane side file (original weights needed
  once; shareable for that exact file); or make a Pelias tiered quant from the original.

## 5. Decisions on order

- 10:53: small-card layout → after GP3-14. "Did we run it at all yet on rotated?" answered from the records.
- 10:55–10:57: **Build first.** No more tests before the first engine build; the GPU goes online and accuracy testing
  resumes on the engine (kld in minutes instead of 25). Parked: block_tierplanes_v2.sh (A2 top-4 + dense plane + 3
  head planes, ready) and the split-token grade.
- 11:13: MTP back before the first build ("can we do the mtp so you can add that to the build?").
- 11:20–11:24: every readable type gets a fast CPU decoder; and **five mass-appeal items** before the build:
  (1) fused tensors, (2) dense model families, (3) FP8 / NVFP4, (4) Windows MSVC build proven, (5) int8-activation
  CPU path.
- 12:59: "i told ya do everything you need to do to get this running" (blanket go for the pre-build list).
- 13:00: **after the build, the accuracy checks come first, before the new ideas.**
- 12:16: "dude... this engine... lets build".

## 6. GP3-10 CPU kernels (sandbox + PC)

- `kern/cpu_dot.{h,cpp}`: `dot_row` (one weight row · up to 4 activation rows straight from storage, planes included),
  32 FMA lanes + fixed reduce tree; every weight bit-identical to `quant::dequantize_row`; AVX2 = portable bit for bit.
- `--fast-cpu` routes every matmul through it; `pelias bench-dot` measures GB/s per type.
- **All 30 (later 32) readable types have their own AVX2 decoder** (IQ2_*, IQ3_*, IQ1_*, TQ1_0/TQ2_0, Q7_Z/Q7_ZT,
  K-quants, IQ4, MXFP4, F16/BF16, later I2_S and NVFP4). Sandbox speedups vs reference 2.3–10.9x.
- **Int8-activation path** (`ActQ8x32`, `dot_row_q8`, maddubs/madd, planes in integers) for 10 types; others take the
  float path on the same rounded inputs. Within 2e-6 of fp64.
- **clang / ASan fixes (12:0x):** a sign-conversion in the Q5_0/Q5_1 int decoder; a Q7_ZT block with a corrupt gate mask
  read past the block in every Q7_ZT reader — all readers now take at most two H words.
- **PC chain-3 (c79229d, 12:30–12:37, MEASURED):** 80 tests pass. membw 41.6 GB/s. Decode GB/s float → int8 @ 16 thr:
  Q4_K 22.2 → 28.1; Q6_K 38.0 → 36.1; Q8_0 35.4 → 38.6; IQ4_XS 23.9 → 28.7; Q4_TG2 16.8 → 17.0; IQ2_XXS 10.4 → 10.7;
  Q4_K reference path 8.5 (kernels 2.7x). 27 GB/s gate met by Q8_0, Q6_K, Q5_K, Q4_K int8, IQ4_XS int8; not by
  Q4_TG2 or IQ2_XXS (open). **kld 5 chunks: 117 s vs 116 s — the "2x faster" prediction missed** → cause found.
  MTP on the real file: 9.1e-6, top-1 13/13 PASS.
- **Batch fix (cdfdb72):** `dot_row_n` / `dot_row_q8_n` (row decoded once per batch); `PELIAS_PROF=1` phase timer.
  PC: 8 tokens 1.07–1.40x; kld experts 1.21x; dense projections slower than reference (cause: reference reuses each
  input across 16 rows).
- **Row blocks (35d3e48):** `dot_rows_n` / `dot_rows_q8_n`, blocks of 16 rows, accumulators in registers.
  PC: 64 tokens Q4_K 1.66x, IQ4_XS int8 1.61x, Q4_TG2 1.99x; kld warm chunk 16.20 → 14.45 s (1.12x), **experts 1.36x**
  (met); shared expert / attention / DeltaNet still slower (open). KL bit-identical to v3 (blocks give the same bits).
- Script name issues on the PC (v3b name collision, v3c lost names to `$` expansion) → v3d; nothing overwritten.

## 7. GP3-08 MTP (sandbox + PC)

- `Qwen35MoeMtp` from blk.<n_layer>; `mtp_forward` as llama.cpp's graph_mtp (read as spec); `generate_greedy_mtp`
  with rollback; `pelias generate --mtp-draft N`. Fused tensors read (attn_qkv, ffn_gate_up_exps).
- MEASURED: drafting = plain greedy for draft 0/1/2/4 on both paths; echo block keeps 18/18; vs llama.cpp tiny 9.4e-6,
  real file 9.1e-6, top-1 13/13.

## 8. The five mass-appeal items (all done by 14:28)

### 8.1 Dense families (GP3-09)
- qwen3 / qwen2 / llama spec, NEOX vs adjacent rope, qk-norm, qkv-bias, tied output, rope_freqs; `pelias logits`;
  `generate --chat` (ChatML, Llama 3 headers). Tokenizers qwen2 and llama-bpe (llama-bpe 47/47 after ignore_merges).
- Tiny models vs llama.cpp ≤ 7.5e-7, top-1 20/20.
- **Qwen3-1.7B on the PC (MEASURED):** BF16 PPL 17.0594; Q8_0 llama.cpp KLD 0.00367 @ chunk 20 (0.003790 all 100);
  **Pelias with llama.cpp's rounding 0.003617; Pelias exact activations 0.002263 (−38 %)**. Chat answer sensible
  (Rayleigh scattering). Decode 4.19 tok/s (open: thread start-up per call suspected).
- Needed: venv-convert (torch 2.15.0+cpu, transformers 5.22.1) at C:\AI\pelias-tests\venv-convert.

### 8.2 BitNet b1.58 2B4T
- I2_S (type 36) decoded from the real file vs its BF16 original: 0 = −1, 1 = 0, 2 = +1 (97.98–99.43 % agree); scale =
  first fp32 of the 32-byte tail = mean |w| (2.29311 vs 2.29312).
- Sub-norms, relu², per-token int8 absmax rounding, NEOX rope; llama-bpe default; BitNet chat template.
- **PC (MEASURED):** top-1 **14/14** vs transformers (fp32, same ternary weights); KL 0.0016 (reference) / 0.0018
  (fast) — at the int8-rounding floor (Pelias reference vs its own fast path 0.0018). Chat: correct Rayleigh answer,
  stopped at <|eot_id|>; 4.02 tok/s. Needed `accelerate` in the venv (v1b).

### 8.3 FP8 and NVFP4 (GP3-11)
- FP8 folder → llama.cpp converter (dequantized to BF16). **Pelias 0.00862 vs llama.cpp 0.00863.** (FP8 KLD is
  ~2.5x Q8_0's because e4m3 keeps 3 mantissa bits — explained to the author at 14:01: not the bit-array decoder; the
  decoder is exact, the error is in the FP8 file's own rounding.)
- NVFP4 (type 40): own decoder + AVX2; NaN at first → per-weight `.scale` tensors bound → **Pelias 0.1334 vs llama.cpp
  0.1345** (−0.8 %). gguf-py has no NVFP4 quantizer → own encoder in make_tiny_dense.py.
- Open: a Pelias-native FP8 / compressed-tensors safetensors reader.

### 8.4 Windows MSVC
- membw gcc attribute → `hal/cpuid.h`; C4244 in tokenizer → narrowed; text-mode test file doubled CR → "wbx";
  one stale D: copy (v1c) → v1d with md5 checks.
- **v1f (c03789b): Visual Studio 2026 /W4 /WX, 56 s, 0 warnings; 84 tests pass on Windows; Windows vs WSL logits
  2.6e-6 (reference) / 6.9e-7 (fast), top-1 14/14.**

### 8.5 Int8-activation CPU path — see section 6.

## 9. GP3-14c infinite context (designed 12:01–12:16, after the first build)

- Author's question: what did we solve for the context cache; can we use spare unified bandwidth and 1-bit on both
  sides of the KV; how to increase positions.
- Design: keys stored before RoPE; VRAM holds 1-bit key-code page summaries + hot pages; **RAM holds every KV page (q8)
  and DeltaNet checkpoints first; NVMe only as overflow** (author 12:15); each step scores summaries, then 1-bit codes,
  pulls top pages to VRAM on a copy engine, attends exactly; pulled pages placed at fresh positions inside the trained
  window (way 1), RoPE scaling toggle (way 2), long-context fine-tune (way 3).
- Sizes (DERIVED): 1M tokens ≈ 10.9 GB q8 KV, codes ≈ 0.8 GB, summaries ≈ 12.5 MB; a 64-page cold pull ≈ 45 MB,
  ≈ 15 ms from NVMe, < 2 ms from RAM.

## 10. Q&A answered (14:0x–16:28)

- **"Is that because of the bit array decoder?"** No — the decoders are exact (bit for bit vs reference); the FP8 /
  NVFP4 error is the files' own rounding; Pelias matches llama.cpp on them.
- **"Are any of these models only working because of the unified bus?"** No — every model today runs on the CPU path
  (fast kernels); the unified bus is what the GPU engine (next) uses for speed and for the plane/tier layouts.
- **"What would you rather?"** → a new conversation with a full handoff.
- **"Did the pure 7-bit vs 7+1 happen?"** Yes, verified (section 0, rule 2).
- Folders / info for the MCP in the next chat: given (see handoff section 2).

## 11. Saved and backed up

- Repo: FireballJohnnie/pelias-master, `main` = **6751b04** (pushed) + this file's commit. WSL clone at c03789b.
- Handoff: `docs/handoff/HANDOFF_2026-10-30_1430.md` (sections 0–9); C: copies `_1430` and **`_1430b` (use b)**.
- Backups: `C:\AI\cl-sndbx-bkp\stuart-creator\2026-10-28\` — every gp script (block_cpukern_v3…v3d, block_dense_v1/v1b,
  block_bitnet_v1/v1b, block_fp_v1/v1b, block_win_v1…v1f, i2s_check.py), bundles up to
  `pelias-master_main_d898233-6751b04.bundle`, `sandbox_2026-10-30_test_logs.tar.gz`; INDEX.md and RUNBOOK.md lines.
- PC logs: `D:\pelias-tests\gp\{TIER,KERN,DENSE,BITNET,FP,WIN}\`; large outputs `C:\AI\pelias-tests\{DENSE,win}`;
  models `C:\AI\models\pelias-new` (15.47 GB downloaded 12:35–12:37) and `D:\bitnet`.

## 12. What comes next (in order)

**Update at commit time (16:4x):** the next chat has already moved on and pushed. It made the queued doc updates
(147862f, df1da5d, 17e9595), put the GP3-10 GPU kernels into `src` (341a29a), and ran GPU rung 1 on the RTX 4070:
11/11, rel L2 <= 2.7e-07, sanitizer 0 errors (fa74fd0). It added GPU decoders for all 32 types, with rel L2 <= 1.8e-07
(64ca815, b4fa7c9). It also proved the Windows GPU build with VS 2026 + CUDA 13.4: 0 warnings, rung 1 11/11 identical
to WSL (a9c0627). Items 1–2 below are therefore partly done; see BUILD_LOG.


1. Ask the author once whether the queued PDF/MD updates come before the GPU work (handoff section 9: speedup +
   Section 21 rev4, 7-bit/1-bit decoder rev3, pathways v5, project overview; masterwork 10/20/21, PELIAS_NEW_METHODS,
   PELIAS_VS_EXISTING_FRAMEWORKS, PELIAS_7_T_H_PLAN, README, PELIAS_SYSTEM_MASTER).
2. GP3-10 GPU kernels into `src` (the measured R3 probe kernels) → GP3-12 inference rungs → **GP3-14 first engine
   build** (WSL + Windows).
3. Parked accuracy checks first (block_tierplanes_v2.sh, split-token grade, the rest).
4. Then the new ideas: small-card layout, planes on any file / ladder, GP3-14c, D-1B dense 1-bit.
5. Open items: native FP8 reader; CPU decode ~4 tok/s; CPU batch dense projections; Q4_TG2 / IQ2_XXS below 27 GB/s;
   BitNet KL base from transformers.

**And again: keep an open mind. Build on what is measured; don't undo it.**

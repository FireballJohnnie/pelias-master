# PELIAS (system)

**One ground-up program that probes the machine, takes a local GGUF or safetensors model, coaches and runs training or quantization, writes the resulting GGUF, and serves it on port 8062 through its own engine and server.**

> **PRE-RELEASE, PRIVATE. Updated 2026-10-30.** Pelias reads the model, quantizes it with its own formats (`pelias quantize`, including usage-tiered experts), runs the whole model on the CPU (an exact reference path and a fast AVX2 path) and scores it against the original (`pelias kld`). Every item on the list before the first engine build is done and measured on the PC: the MTP draft head, fast CPU code for 32 storage types, dense Qwen3 / Qwen2 / Llama models, BitNet b1.58, FP8 / NVFP4 files and the Windows (MSVC) build. All of it is ground-up code. Next: the GPU kernels go into the product, then the first engine test build. Every number below carries a label. The raw results are in `data/probe_results/` and `logs/BUILD_LOG.md`.

> **How to read the numbers in this repository.** Every result marked MEASURED was measured on one machine: the author's PC (RTX 4070 12 GB with its memory overclocked, Ryzen 7 5800X3D, 64 GB DDR4-3200, Windows 11 + WSL2). Every figure for other hardware is DERIVED: a proposal, with its formula and inputs written down, so anyone with that hardware can run the same benchmark and report what they measure. Where a measurement disagrees with a derived figure, the measurement wins and the derivation gets fixed.

**Author and maintainer:** John Lipka. Developed with AI assistance under his direction. Sibling repositories (records of the design and measurements): pelias-trainer, pelias-engine.

## Origin

In the author's words: "this started as an effort for me to better create my own llm, when i found this... accidentally. everything said we couldn't train an moe using my type of system, we found a way. then we saw the hardware ceiling limit when running that quantized llm, and we found a way to unify the bus to make it run even better from the way we found to train it... and then even more!" The full timeline, from the first training folder of September 2026 to this repository, is Section 20 of the masterwork.

## The goal

Beat the best published 4-bit file of Qwen3.6-35B-A3B on accuracy, at the same size or smaller, then run it faster on one consumer PC. The bar is Unsloth's MTP UD-Q4_K_M: KL 0.012864 on llama.cpp's GPU path (MEASURED). The PC has an RTX 4070 12 GB, a Ryzen 7 5800X3D and 64 GB of dual-channel DDR4-3200.

## Where it stands

### Accuracy: the best file so far

| File | KL vs the original (lower is better) | Size vs Unsloth | Label |
|---|---|---|---|
| Unsloth MTP UD-Q4_K_M (the bar) | 0.012864 | — | MEASURED |
| urm_ZT2: dense Q7_ZT | 0.012536 | -236 MB | MEASURED (100 chunks) |
| **urm_TG2: dense Q7_ZT + expert gate/up Q4_TG2** | **0.012056 (-0.000808 vs Unsloth, t -1.98)** | **-236 MB** | MEASURED (100 chunks) |

urm_TG2 is lower than Unsloth at a smaller size. Its 95 % interval, [-0.001618, +0.000002], just touches zero (t -1.98 at 100 chunks), so the gap is not yet called resolved; more chunks or a second text settle it.

### Accuracy at the same size: usage-tiered experts and residual planes (2026-10-30)

Scored by Pelias against the original (BF16) model, WikiText-2, 100 chunks, paired token by token. The baseline here is Unsloth's MTP UD-Q4_K_M read by Pelias (0.012808), so these rows compare with each other, not with the llama.cpp-path figure above.

| File | KL vs the original | Size | Label |
|---|---|---|---|
| Unsloth MTP UD-Q4_K_M, read by Pelias | 0.012808 | 22,663,387,424 bytes | MEASURED |
| **Usage-tiered experts** (each of the 10,240 experts in its own type by use and measured error: Q4_TG2 floor, Q5_K, Q6_K, Q8_0) | **0.009404 (-26.6 %, t -12.41, settled)** | 22,663,406,784 bytes (+19,360: the type lists) | MEASURED |
| Tiered + 1 GB of residual sign planes on the worst-error blocks | 0.008963 (-4.7 % vs tiered, t -2.26) | + 1 GB side file, read by the CPU | MEASURED |
| **Tiered + 2 GB of such planes** | **0.008747 (-31.7 % vs Unsloth, t -12.81)** | + 2 GB side file | MEASURED |
| Tiered + 2 GB, planes read only for each token's top 4 experts | 0.008800 (92 % of the gain at half the plane reads) | + 2 GB side file | MEASURED |
| Unsloth's file + 2 whole-file planes | 0.005818 (-54.6 %, t -12.25) | + 10.2 GB side file | MEASURED |

A user gets this one of three ways: run a premade file as it is, add a plane side file built once for that exact file (the file stays portable), or build a Pelias tiered file from the original weights (Pelias only). Details: `docs/PELIAS_RESIDUAL_PLANES.md`.

### Methods proven so far

| Method | Result | Label |
|---|---|---|
| Rotation per 32 values (dense weights and their inputs) | File error -8.9 %; 8-bit activations then cost nothing | MEASURED (whole model) |
| Q8_Z, the author's zip scales | Same accuracy as Q8_0, 61 MB smaller | MEASURED (whole model) |
| **Q7_ZT: 7 bits plus an 8th bit gated per 32 values** | urm_ZT2 below Unsloth at -236 MB (table above); the 8th bit is stored on 2 of every 8 blocks of 32 values | MEASURED (whole model) |
| Light vs heavy 8th bit, rotated vs not | Rotated + light is best in every case. A missing heavy bit costs about 340-360x; a missing light bit costs about 1.4-1.5x | MEASURED (sandbox, real weights) |
| The 1-bit decoder (8th bit added by AND + popcount on idle integer units) | Bit-identical to 8-bit across the whole model; GPU kernel 5.7-6.9 % faster than 8-bit | MEASURED |
| **Q4_TG2: the author's bit tags with an exact 0 (expert gate/up)** | Weighted error -10 to -11 % per tensor vs imatrix Q4_K, same bytes | MEASURED |
| The miss map (where the model gets stuck) | The same hard tokens repeat across every file (47-66 % vs 1 % by chance); number and math tokens get stuck at 2x the base rate | MEASURED |
| The author's H gate on KV pages | 72.6 % of attention mass kept in 1/8 of the keys, vs 62.0 % (Quest pages) and 42.6 % (recent window) | MEASURED (512-token chunks) |
| **Usage-tiered experts** | -26.6 % KL at the public file's exact size (table above) | MEASURED (whole model) |
| **Residual sign planes** (side file; worst-error blocks; top-4 picks) | -54.6 % on the public file; -31.7 % with 2 GB on the tiered file; 92 % of it at half the reads | MEASURED (whole model) |
| Planes locked into one int4 core | 19 % less error than 4 free planes (0.01051 vs 0.01299) | MEASURED (sandbox, real weights) |
| Exact activations on a Q8_0 file (Qwen3-1.7B) | KL 0.002263 vs llama.cpp's 0.00367 on the same file (-38 %; first 20 chunks, not paired) | MEASURED |

**Candidate file (PEL7):** dense Q7_ZT, rotated, with the light 8th bit; expert gate/up Q4_TG2; expert down at Unsloth's types; the MTP input projection rotated.

### Speed: measured so far

| What | Result | Label |
|---|---|---|
| Unified-bus probes (doorbell, replay, overlap, two DRAM readers) | CPU/GPU hand-off 42-78x faster; tier build 31.8 -> 53.5 tok/s | MEASURED |
| System RAM read rate (`pelias membw`) | DDR4 saturates at 2 threads (47.8 GB/s, 93 % of 51.2); no lag felt by the user up to 4 reader threads | MEASURED |
| GPU read ceiling | 494 GB/s on one long read: 98 % of the stock 504 GB/s, 89.5 % of this card's overclocked memory (about 552 GB/s) | MEASURED |
| One launch per matrix (how engines read today) | 150-352 GB/s per matrix size | MEASURED |
| R3 v1, one token's weights (2.27 GB) in one streaming launch | 8.79 ms (259 GB/s) vs 20.81 ms one launch per matrix; with 160 dependency syncs 12.57 ms | MEASURED |
| **R3 v2, the same token on the real token path (40 distinct layers)** | **5.685 ms**; pure streaming 4.75 ms (about 478 GB/s); the usage-tiered types add +0.3 to +1.3 % | MEASURED |
| CPU kernels (`pelias bench-dot`, 16 threads, float / int8 activations) | Q8_0 35 / 39 GB/s, Q6_K 38 / 36, Q5_K 31 / 27, Q4_K 22 / 28, IQ4_XS 24 / 29 vs 8.5 for the reference path; Q4_TG2 17 and IQ2_XXS 10-11 still below the 27 GB/s gate | MEASURED |

**Where the rest of the speed was:** an independent review of R3 v1 (`docs/reviews/FABLE_R3_V1_REVIEW_2026-10-29.md`) traces about 4 of the 8.79 ms to one lookup in the Q4_TG2 decode. Without it, the other formats already stream at about 375-405 GB/s (DERIVED). R3 v2 removed them one at a time, measuring each, down to the 5.685 ms above; the remaining ~0.93 ms to pure streaming is measured and explained. At the 494 GB/s ceiling a token's weights take 4.60 ms (DERIVED). Projected speeds once R3 is in the engine are in the game plan (DERIVED, not measured).

### Before vs now: speed on the same PC

The first measured speeds of this model on this machine, before any of the work above. The first four rows are the stock llama.cpp server with a Q8_0 KV cache; the fifth is the author's tier build in a patched llama.cpp. Records are in the pelias-engine repository, 2026-10-19 to 10-22.:

| When | Setup | Generation, tok/s (short / 4K / 12K tokens) | Label |
|---|---|---|---|
| 2026-10-19 23:17 | Q5_K_M, all experts on the CPU, 32K context: **the very first run** | 29.2 / 28.5 / 27.0 | MEASURED |
| 2026-10-20 02:00 | Q4_K_M, all experts on the CPU | 32.1 / 30.7 / 30.4 | MEASURED |
| 2026-10-20 02:55 | Q4_K_M, 128K context, `-ub 2048` | 31.8 / 31.5 / 31.3 | MEASURED |
| 2026-10-20 03:20 | Q4_K_M, best stock layout (`--n-cpu-moe 28`: 12 layers of experts on the GPU) | 41.1 / 40.2 / 38.8 | MEASURED |
| 2026-10-22 | The author's tier build (hot experts on the GPU, bypass 32) | 53.5 / 53.0 / 49.1 | MEASURED |
| next | Pelias engine: unified bus rungs 1-3 (bus speed + doorbells, one-layer-ahead prediction, L3 repeats + draft prefetch) | 89-103 / 112 / 116; 119 with today's ~3.2 ms per-token overhead; 193 memory floor (about 211 at this card's overclocked memory), reachable only with GPU-side work such as R3 | DERIVED (game plan, Fable's corrected figures) |

**The first targets, set 2026-10-20 to 10-21:**
- A custom engine would need about 47 tok/s to be 15 % faster than the best stock layout.
- The 12K-context target was at least 55 tok/s; the best tier run reached 46.8, so it failed at that point.
- The hardware-ceiling model of that day put the limit at 88-92 tok/s (DERIVED).

The newer projections sit above that ceiling because they rest on measurements taken since then: bus speed, doorbells, the hot set and the 494 GB/s read. The arithmetic is in the game plan. They are projections until the engine runs and its timers decide.

### Running now and next

1. **Done (2026-10-30, MEASURED on the PC): the list before the first engine build.** GP3-08 the MTP draft head (real file, fast vs reference 9.1e-6, top-1 13 of 13); GP3-10 CPU kernels (32 types, int8-activation path, batch kernels); fused gate/up and q/k/v tensors; GP3-09 dense Qwen3 / Qwen2 / Llama with their tokenizers and `generate --chat`; BitNet b1.58 2B4T (top-1 14 of 14 vs transformers); GP3-11 FP8 (via the converter) and NVFP4 GGUF files; the Windows MSVC build (/W4 /WX, 0 warnings, 84 tests; Windows vs WSL logits within 2.6e-6 of the largest).
2. **Next:** GP3-10 GPU kernels into the product (the measured R3 streaming kernels, RMSNorm, IMRoPE, attention, DeltaNet, router, top-k; rung 1 against an fp64 reference with zero compute-sanitizer errors), GP3-12 inference rungs (one layer, then the full forward pass on the GPU), GP3-14 the first engine test build on WSL and Windows.
3. **Right after the first build:** the parked accuracy checks (a dense plane and three output-head planes on the tiered file, the split-token grade), before any new idea (small-card layout, infinite context, dense 1-bit + planes).
4. **Open for the engine:** CPU decode 4.0-4.2 tok/s on 1.7-2.4 B models (thread start-up per product suspected); dense projections on the CPU batch path; Q4_TG2 / IQ2_XXS CPU decode below the gate; a native FP8 / compressed-tensors safetensors reader.

## Other cards: proposed numbers, waiting for your benchmarks

Pelias was built and measured on one 12 GB consumer card. The levers behind its numbers aren't specific to that card:
- reading weights at 89.5 % of the card's peak in one long stream;
- a quantizer that beats the best published 4-bit file at a smaller size;
- a bus that keeps the hot experts in VRAM and lets the CPU handle the rest.

The table carries the same arithmetic over to other cards, including older ones and retired data-centre cards. These are proposals, not results (DERIVED; the formula and every input are in masterwork 21.8). They exist so people with these cards can benchmark them and replace each row with a measurement:

| Card | VRAM | What it runs (Pelias formats) | tok/s, DERIVED |
|---|---|---|---|
| Tesla V100 32GB (data centre, 2017) | 32 GB | Qwen3.6-35B-A3B, whole model in VRAM | 117-301 |
| Tesla V100 16GB | 16 GB | the same 35B: dense + 12 GiB of hot experts, the rest from RAM | 89-162 |
| Instinct MI50 32GB / MI60 | 32 GB | the same 35B, whole model in VRAM | 126-336 |
| Tesla P40 (data centre, 2016) | 24 GB | the same 35B, whole model in VRAM | 58-128 |
| RTX 3060 12GB | 12 GB | the same 35B through the bus | 53-98 |
| RTX 2060 | 6 GB | the same 35B through the bus | 46-77 |
| GTX 1060 6GB (2016) | 6 GB | a small MoE (16B total, 2.4B active class) through the bus | 41-78 |
| GTX 1050 Ti 4GB (2016) | 4 GB | a small MoE (7B total, 1.3B active class) through the bus | 45-90 |
| *For scale: this PC today* | 12 GB | *the 35B: stock llama.cpp best layout 41.1, the author's tier build 53.5 (MEASURED)* | — |

**How to read a range:**
- **Low end:** the streaming kernel as measured today (R3 v1) plus today's per-token overhead.
- **High end:** the kernel at the measured read ceiling with most of the overhead moved onto the GPU.
- **"Through the bus"** means the CPU reads the experts that don't fit in VRAM from system RAM, so the PC needs RAM for the rest of the model. Masterwork 21.9 lists what the PC needs.

On the oldest cards, dense models are limited to what fits in 4-6 GB. Small MoE models are the more interesting case on paper: only a slice of the weights is read per token, so the arithmetic puts models several times larger than the card's VRAM within reach.

**To turn a row into a measurement:** run the steps in masterwork 21.8.10 on your card and open an issue with the log. Measured rows replace derived ones, with the card, the date and the submitter's setup.

## Key documents

| Document | What it is |
|---|---|
| [`docs/PELIAS_GAME_PLAN.md`](docs/PELIAS_GAME_PLAN.md) | The plan of record: every run, result and decision, in order |
| [`docs/PELIAS_7_T_H_PLAN.md`](docs/PELIAS_7_T_H_PLAN.md) | The 7-bit / T / H / 1-bit decoder line |
| [`docs/PELIAS_NEW_METHODS.md`](docs/PELIAS_NEW_METHODS.md) | Methods proven to work, with their numbers |
| [`docs/PELIAS_VS_EXISTING_FRAMEWORKS.md`](docs/PELIAS_VS_EXISTING_FRAMEWORKS.md) | Plain-language summary and comparison with existing frameworks |
| [`docs/PELIAS_SYSTEM_MASTERWORK.md`](docs/PELIAS_SYSTEM_MASTERWORK.md) | The full design specification |
| [`docs/reviews/`](docs/reviews/) | Independent reviews and what was done with each finding (latest: the R3 v1 review) |
| [`logs/BUILD_LOG.md`](logs/BUILD_LOG.md) | Dated log of every build, run and correction |

## What it will be

| Piece | One line |
|---|---|
| probe | Detects CPU, GPU, RAM, drives, OS and ReBAR; measures what matters (`pelias membw` is the first piece); uses published values only until it has measured |
| plan | Fits the work under the measured VRAM ceiling and offers two to four plans; recommends a bigger plan when it finds more room than an imported profile assumed (and waits for the user's OK) |
| intake / coach | Imports parameters from json, txt, log, csv, md (and `.py`, parsed, never run); coaches the training set for the clients that will call the model |
| quantize / train / export | Own bit-exact quantizers; layer-major training with the core trained through the exact quantizer; frozen bytes proven unchanged |
| engine + server | The unified bus (GPU hot bank, CPU cold experts, doorbell handoff, dual DRAM readers, R3 streaming decode) behind an OpenAI-compatible server on 127.0.0.1:**8062** |
| web interface | Served by Pelias itself: lithic style, the Pelias logo, the spear progress bar, the bus view |
| web tools | Search, fetch and read for the model; FlareSolverr as the one optional outside process |
| installers | Windows 10/11, and Linux (WSL first, native capable) |

## Building

Everything needed to build is in this repository: C++20, CMake 3.24+ and Ninja; no third-party libraries. The CUDA toolkit is needed only for the CUDA presets and the GPU probes. Every step is built and tested on Linux (gcc, clang and gcc with AddressSanitizer, in the sandbox) and on the author's WSL2. The CPU build (no CUDA) is also proven on Windows with Visual Studio 2026 (MSVC 14.51, /W4 /WX, 0 warnings, 84 tests; c03789b, 2026-10-30), configured as below; the R3 probe was built natively on Windows (see below). Other setups are untested.

```
git clone https://github.com/FireballJohnnie/pelias-master.git && cd pelias-master
# host-only (any machine, no GPU): build, unit tests, the corrupt-input gate
cmake --preset host && cmake --build --preset host && ctest --preset host
# WSL2 / Linux with CUDA (nvcc at /usr/local/cuda/bin/nvcc)
cmake --preset wsl && cmake --build --preset wsl && ctest --preset wsl
# Windows: from an x64 Native Tools Command Prompt (where cl -> Hostx64\x64)
cmake --preset windows && cmake --build --preset windows && ctest --preset windows
# Windows, CPU only, Visual Studio generator (the configuration proven on 2026-10-30)
cmake -S . -B build-msvc -G "Visual Studio 18 2026" -A x64 -DPELIAS_CUDA=OFF && cmake --build build-msvc --config Release --parallel
build-msvc\Release\pelias_unit.exe --data <empty folder>
```

Options (add to the configure line):
- `-DPELIAS_CPU_AVX2=ON`: the CPU reference pass with AVX2 and FMA (gcc / clang; identical logits, faster; the author's runs use it). `-march=native` was measured slower and is not offered.
- `-DPELIAS_CUDA_ARCHITECTURES=86`: the CUDA architecture of your card (default 89, the author's RTX 4070; only 89 has been run).

The binary is `build/<preset>/pelias`. A first run on your own GGUF file (Qwen3.6-35B-A3B class, `qwen35moe`; dense `qwen3`, `qwen2`, `llama` and BitNet `bitnet-b1.58` files also run; Pelias never downloads models):

```
build/host/pelias inspect --model model.gguf
build/host/pelias generate --model model.gguf --prompt "The capital of France is" --temp 0 --n-predict 32
```

`generate` runs the fp32 reference pass on the CPU: it is the correctness reference that the engine is checked against, so expect it to be slow. `--fast-cpu` (on `kld`, `logits`, `generate`) uses the AVX2 kernels instead: the same products, another order of the sum. `--chat` wraps the prompt in the model family's chat format. The engine (GP3-14) will be the fast path.

**Commands** (`pelias --help` lists every option):

| Command | Does |
|---|---|
| `pelias inspect --model F`, `pelias device`, `pelias plan --model F`, `pelias key --model F` | Read a model, list the devices, plan a fit, key a file |
| `pelias generate --model F --prompt TEXT` | Text from the CPU reference pass: prefill, then one token per step on the carried KV cache and DeltaNet state. `--temp 0` is greedy; `--top-k`, `--top-p`, `--min-p`, the penalties and `--seed` follow the parameter dictionary; `--ids-out` writes the token ids; `--no-mmap` reads the model into RAM first |
| `pelias generate --model F --prompt-list P.txt --out-dir DIR` | Every line of P.txt as a prompt, on the model (and planes) loaded once |
| `pelias teacher --model ORIGINAL.gguf --ids A.ids [--ids B.ids ...] --out-dir DIR` | The original model's own top-1 / top-2 at every position of decoded text; `tools/split_grade.py` grades the GP3-07 splits with it (who is right: Pelias, llama.cpp or neither) |
| `pelias tokenize --model F --text TEXT` | The model's tokenizer (equal to llama.cpp's, token for token, on the P-D1 test files; `logs/BUILD_LOG.md`) |
| `pelias quantize --model IN --out OUT [--recipe R.tsv] [--imatrix M.gguf] [--rotate32 GROUPS]` | Quantize with Pelias's own formats (incl. Q8_Z, Q7_ZT, Q4_TG2) |
| `pelias quantize --tier ALLOC.tsv --model BASE --original ORIG --out OUT [--imatrix M.gguf]` | Usage-tiered experts: each routed expert re-quantized from the original into the type `tools/tier_alloc.py` gives it (Pelias reads the result) |
| `pelias convert --model FOLDER --out OUT.gguf [--type BF16\|Q4_K_M\|Q6_K\|Q8_0]` and `pelias verify --model A --reference B` | Hugging Face folder to GGUF; compare two files tensor by tensor and byte for byte |
| `pelias logits --model F --ids 1,2,3 --out OUT.bin [--fast-cpu] [--decode]` and `pelias mtp-logits ...` | Every position's logits (and the MTP block's) for the oracle checks in `tools/oracle/` |
| `pelias bench-dot [--type T] [--int8] [--tokens N] [--planes K]` | GB/s of the CPU kernels for one storage type |
| `pelias imatrix --model F --file TEXT --out M.gguf` | Importance matrix from the fp32 reference pass |
| `pelias kld --model F --base BASE.bin` | Score a file against the original. `--dump-kld`, `--route-dump`, `--dump-codes` and `--dump-qk` write per-token data; `--kv-store` / `--state-store` test KV and DeltaNet-state compaction |
| `pelias kld-compare --a A.bin --b B.bin` | Paired comparison of two scored files (difference, interval, verdict) |
| `pelias planes --model F --base ORIGINAL.gguf --out P.gguf [--k 2] [--exact-head]` | Residual sign planes of any model file against its original weights (each plane cuts the weight error about 3x); `--k-rules` per-tensor counts, `--select PART --budget-mb N` planes only on the worst-error blocks; `kld` and `generate` read them with `--planes P.gguf`, `--plane-k K`, `--plane-picks N` (top N picks only), `--head-rescore N`, and `generate --planes-mode all\|prefill\|start:N` |
| `pelias membw` | System RAM read rate by threads, page size and prefetch, and the lag the user would feel |

**GPU probes** (stand-alone, one `.cu` each; the build line is at the top of each file):

```
# WSL / Linux
nvcc -std=c++17 -O3 -arch=sm_89 tools/probes/r3/r3.cu -o r3
# Windows (x64 Native Tools prompt): MSVC needs its standard preprocessor for the CUDA headers
nvcc -std=c++17 -O3 -arch=sm_89 -Xcompiler /Zc:preprocessor tools/probes/r3/r3.cu -o r3.exe
```

Use your card's `-arch` (sm_70 V100, sm_75 Turing, sm_80 / sm_86 Ampere, sm_89 Ada, sm_120 Blackwell). Only sm_89 has been run. `r3 check` must pass before any timing; `r3 stream --distinct` is the 40-layer timing used in masterwork 21.6. Results from other cards are welcome as issues (see "Other cards" above).

**Checks:**
- `python3 tools/check_module_deps.py`: module boundaries.
- `python3 docs/scaffold/verify_inventory.py`: inventory and privacy scan.
- `python3 tools/greedy_compare.py FOLDER`: the GP3-07 gate's comparison of `pelias generate --ids-out` with llama-server output (prompts in `tools/gates/gp3_07_prompts.txt`).

## Repository map

| Path | Holds |
|---|---|
| `src/` | The product code: GGUF / safetensors I/O, tokenizer, quantizers, the reference model, the OS and device layers, the `pelias` commands |
| `tests/` | Unit tests (run by `ctest`) |
| `tools/probes/` | GPU and CPU probes: `tdecode` (1-bit decoder, read ceiling), `r3` (streaming decode kernel), `gfx_probe`, `e16`, `batch_c` |
| `tools/oracle/` | Test-only references, never linked into the product: llama.cpp logits and MTP oracles, tiny test models, tokenizer vectors, BitNet checks |
| `tools/experiments/` | Sandbox checks behind each result (named in the game plan); `older_cards/` computes masterwork 21.8 |
| `tools/` | Recipe tool, imatrix merge, route statistics, table generators, UI texture baker |
| `data/probe_results/` | Measured results kept with the code (run logs, miss logs, per-token KL summaries) |
| `docs/` | The game plan, method and comparison papers, the masterwork, build notes, reviews, the author's input files (`docs/inputs/`, unchanged) |
| `profiles/reference_pelias/` | Import files for the reference machine and model |
| `profiles/local_only/`, `support/local_only/` | The author's private material (ignored by git, never pushed) |
| `resources/ui/`, `support/ui_reference/` | Logo, spear stages and textures; the lithic interface specification |
| `logs/` | `BUILD_LOG.md`, `DECISIONS_LOG.md` |
| `THIRD_PARTY_NOTICES.md` | Every outside work the product or its build relies on, with its licence |

## Rules

- Every number carries a label: MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION or AUTHOR-STATED. No made-up numbers. Predictions are written down before each run.
- No llama.cpp or other outside code in the product. llama.cpp is an external, test-only comparison.
- Never truncate training data. The VRAM ceiling is measured at start and enforced. No model downloads.
- Dataset text, persona and private equations never enter this repository.
- Nothing is pushed without the author's word.

## Licence

Apache-2.0 (`LICENSE`). Copyright (c) 2026 John Lipka and the Pelias contributors.

# PELIAS (system)

**One ground-up program that probes the machine, takes a local GGUF or safetensors model, coaches and runs training or quantization, writes the resulting GGUF, and serves it on port 8062 through its own engine and server.**

> **PRE-RELEASE, PRIVATE. Updated 2026-10-29.** Milestone 1 runs: Pelias reads the model, quantizes it with its own formats (`pelias quantize`), runs the whole model on the CPU reference path and scores it against the original (`pelias kld`), all ground-up. The GPU engine is next (R3, the streaming decode kernel, is in progress). The plan of record is [`docs/PELIAS_GAME_PLAN.md`](docs/PELIAS_GAME_PLAN.md); the design is [`docs/PELIAS_SYSTEM_MASTERWORK.md`](docs/PELIAS_SYSTEM_MASTERWORK.md). Every number below is labelled; the probe and run results are in `data/probe_results/` and `logs/BUILD_LOG.md`.

**Author and maintainer:** John Lipka. Developed with AI assistance under his direction. Sibling repositories (records of the design and measurements): pelias-trainer, pelias-engine.

## Origin

In the author's words: "this started as an effort for me to better create my own llm, when i found this... accidentally. everything said we couldn't train an moe using my type of system, we found a way. then we saw the hardware ceiling limit when running that quantized llm, and we found a way to unify the bus to make it run even better from the way we found to train it... and then even more!" The full timeline, from the first training folder of September 2026 to this repository, is Section 20 of the masterwork.

## Where it stands (2026-10-29)

The goal of the first build: beat Unsloth's best 4-bit file of the same model (Qwen3.6-35B-A3B, MTP UD-Q4_K_M, KL 0.012864 on llama.cpp's GPU path) at the same size or smaller, then run it faster on one consumer PC (RTX 4070 12 GB, Ryzen 7 5800X3D, dual-channel DDR4).

| What | Result | Label |
|---|---|---|
| Rotation per 32 values (dense weights and activations only) | File error -8.9 %; 8-bit activations cost nothing | MEASURED (whole model) |
| Q8_Z, the author's zip scales | Same accuracy as Q8_0, -61 MB | MEASURED (whole model) |
| **Q7_ZT: 7 bits + the 8th bit gated per 32 values (light bit, rotated)** | **urm_ZT2: KL 0.012536 vs Unsloth 0.012864 at -236 MB** (lower, not yet statistically resolved) | MEASURED (whole model) |
| The 1-bit decoder (8th bit added by AND + popcount on idle integer units) | Bit-identical to 8-bit across the whole model; GPU kernel 5.7-6.9 % faster than 8-bit | MEASURED |
| Light vs heavy 8th bit | Rotated + light best in every case; a missing heavy bit is ~360x worse, a missing light bit ~1.4x | MEASURED (sandbox, real weights) |
| **Q4_TG2: the author's bit tags with an exact 0 (expert gate/up), in urm_TG2** | **KL 0.012056 vs Unsloth 0.012864 (-0.000808, t -1.98) at -236 MB**; -10 to -11 % weighted error per tensor | MEASURED (whole model) |
| The miss map (where the model gets stuck) | Hard tokens repeat across every file (47-66 % vs 1 %); number / math tokens stuck at 2x the base rate | MEASURED |
| The author's H gate on KV pages | 72.6 % of attention mass in 1/8 of the keys vs 62.0 % (Quest pages), 42.6 % (recent window) | MEASURED (512-token chunks) |
| Unified bus probes (doorbell, replay, overlap, two DRAM readers) | 42-78x faster CPU/GPU hand-off; 93 % of DDR4 peak with both readers; tier build 31.8 -> 53.5 tok/s | MEASURED |
| R3, the streaming decode kernel | One launch per matrix reads at 150-352 GB/s; one long stream at 494 GB/s (98 % of the card): ~3 ms per token to gain | MEASURED ceiling; kernel in test |

**Candidate file (PEL7):** dense Q7_ZT rotated with the light 8th bit, expert gate/up Q4_TG2, expert down at Unsloth's types, the MTP input projection rotated.

**Key documents:** [`docs/PELIAS_GAME_PLAN.md`](docs/PELIAS_GAME_PLAN.md) (plan, every run and decision), [`docs/PELIAS_7_T_H_PLAN.md`](docs/PELIAS_7_T_H_PLAN.md) (the 7-bit / T / H / 1-bit decoder line), [`docs/PELIAS_NEW_METHODS.md`](docs/PELIAS_NEW_METHODS.md) (methods proven to work), [`docs/PELIAS_VS_EXISTING_FRAMEWORKS.md`](docs/PELIAS_VS_EXISTING_FRAMEWORKS.md) (plain-language summary and comparison), [`docs/reviews/`](docs/reviews/) (independent reviews, latest: Fable's math check of the engine plan).

## What it will be

| Piece | One line |
|---|---|
| probe | Detects CPU, GPU, RAM, drives, OS and ReBAR; measures what matters; uses published values only until it has measured |
| plan | Fits the work under the measured VRAM ceiling and offers two to four plans; recommends a bigger plan when it finds more room than an imported profile assumed (and waits for the user's OK) |
| intake / coach | Imports parameters from json, txt, log, csv, md (and `.py`, parsed, never run); coaches the training set for the clients that will call the model |
| quantize / train / export | Own bit-exact quantizers; layer-major training with the core trained through the exact quantizer; frozen bytes proven unchanged |
| engine + server | The unified bus (GPU hot bank, CPU cold experts, doorbell handoff, dual DRAM readers) behind an OpenAI-compatible server on 127.0.0.1:**8062** |
| web interface | Served by Pelias itself: lithic style, the Pelias logo, the spear progress bar |
| web tools | Search, fetch and read for the model; FlareSolverr as the one optional outside process |
| installers | Windows 10/11, and Linux (WSL first, native capable) |

## Repository map

| Path | Holds |
|---|---|
| `docs/PELIAS_SYSTEM_MASTERWORK.md` | The specification: requirements, product, boundary, detection, models, flow, parameters, training, quantizing, engine, server, web tools, installers, scaffold, order of operations, gates, numbers, conflicts, sources, history, and future hypotheses beyond LLMs (Section 21) |
| `docs/PELIAS_SYSTEM_MASTER.md` | The one-page scaffold and audited numbers of 2026-10-25 (two toolchain rows corrected) |
| `docs/BUILD_NOTES.md` | How to build, per platform (filled as the build proceeds) |
| `docs/inputs/` | The author's input files, unchanged, and `REVIEW.md` |
| `docs/reviews/` | Independent reviews and what was done with each finding |
| `THIRD_PARTY_NOTICES.md` | Every outside work the product or its build relies on, with its licence |
| `docs/scaffold/` | `REPO_INVENTORY.tsv` and `verify_inventory.py` |
| `profiles/reference_pelias/` | Import files for the author's machine and model |
| `profiles/local_only/` | The author's private training profile (ignored by git) |
| `resources/ui/` | Logo, spear stages, baked spear textures |
| `support/ui_reference/` | The lithic interface header of 2026-10-24 (specification for the web interface) |
| `tools/ui/` | The spear texture baker (offline tool) |
| `docs/PELIAS_GAME_PLAN.md` | The plan of record: every run, result and decision, in order |
| `docs/PELIAS_7_T_H_PLAN.md`, `docs/PELIAS_NEW_METHODS.md`, `docs/PELIAS_VS_EXISTING_FRAMEWORKS.md` | The 7-bit line, the proven-methods tally, the plain-language comparison |
| `src/` | The product code: GGUF / safetensors I/O, quantizers (incl. Q8_Z, Q7_ZT, Q4_TG2), the reference model, `pelias` commands |
| `tools/experiments/` | Sandbox checks behind each result (scripts named in the game plan) |
| `tools/probes/` | GPU and CPU probes: `tdecode` (1-bit decoder, read ceiling), `r3` (streaming decode kernel) |
| `data/probe_results/` | Measured results kept with the code (miss logs, per-token KL summaries) |
| `logs/` | `BUILD_LOG.md`, `DECISIONS_LOG.md` |

## Rules

Every number carries a label (MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED). No llama.cpp or other outside code in the product; llama.cpp is an external, test-only comparison. Never truncate training data. The VRAM ceiling is measured at start and enforced. No model downloads. Dataset text, persona and private equations never enter this repository. Nothing is pushed without the author's word.

## Licence

Apache-2.0 (`LICENSE`). Copyright (c) 2026 John Lipka and the Pelias contributors.


## Building (step 2 onward)

No third-party libraries: C++20, CMake 3.24+, Ninja; the CUDA toolkit only for `-DPELIAS_CUDA=ON`.

```
# host-only (any machine, no GPU): build, unit tests, the corrupt-input gate
cmake --preset host && cmake --build --preset host && ctest --preset host
# WSL2 / Linux with CUDA
cmake --preset wsl && cmake --build --preset wsl && ctest --preset wsl
# Windows: from an x64 Native Tools Command Prompt (where cl -> Hostx64\x64)
cmake --preset windows && cmake --build --preset windows && ctest --preset windows
```

Then `build/<preset>/pelias inspect --model <file.gguf>`, `pelias plan --model <file.gguf> [--bank-gib N] [--ceiling-gib C]`, `pelias device`, `pelias key --model <file>`. Milestone 1: `pelias quantize --model IN.gguf --out OUT.gguf [--recipe R.tsv] [--imatrix M.gguf] [--rotate32 GROUPS]`, `pelias imatrix`, `pelias kld --model F --base BASE.bin` (with `--dump-kld`, `--route-dump`, `--dump-codes`, `--dump-qk`), `pelias kld-compare`; `pelias --help` lists every option. Repository checks: `python3 tools/check_module_deps.py` (module boundaries) and `python3 docs/scaffold/verify_inventory.py` (inventory and privacy scan).

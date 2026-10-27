# PELIAS-SYSTEM: master scaffold and order of operations

2026-10-25. One place for the tie-ins of the whole system: every part, the names it is made of, the order they are built in (Sections 1 to 3), every data number and parameter with its label and source (Section 4), and what a draft build needs installed on WSL2 and Windows 11 (Section 6). No code. Nothing is pushed. No `pelias-system` directory and no GitHub repository exist yet; this file is the only thing created for it.

> **Copy in pelias-master (2026-10-26).** As delivered on 2026-10-25, with two toolchain rows of Section 6 corrected (CMake 4.4.3 and Ninja 1.13.2 on Windows were in the record of 2026-10-23 01:09; see `PELIAS_SYSTEM_MASTERWORK.md` 18.1, C19). The holds described in Sections 3 and 5 were lifted by the author on 2026-10-26; the current order of work is `PELIAS_SYSTEM_MASTERWORK.md` Section 15.

## 0. How to read this

- **Built from:** the trainer master and notes (local head `953c9b1`), the engine blueprint, scaffold and specification extracts (local commit `ba74954`), and the server notes (master Part 6 and the server runbook). Where a fact already lives in one of them, this page names the place and does not repeat it.
- **Marks:** *exists* = the name is in a repository or on the drive today. *planned* = the name is only in a plan. *DECISION (proposed)* = not ruled by the author.
- **The author's order of work (AUTHOR-STATED):** (1) trainer comparison and scripts; (2) a hard stop until the trainer is "top tier"; (3) make it universal; (4) the engine, **HELD** since 2026-10-24 06:07; (5) unify.
- **Rules that apply to every step:** a step starts only when the gate before it is met; the author runs every GPU program; nothing is pushed until the author says "push".

## 1. The system on one page

| Part | What it is | Today | Where its notes are |
|---|---|---|---|
| `pelias-trainer` | Trains the model: bf16 files in, a trained GGUF and its manifest out | *exists*, own repository; framework only (file names, build and dependency checks; nothing computes except the configuration defaults, the ingest plan, the export decision and the interface drawing layer). Local head `953c9b1`, 3 commits not pushed | `docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`, `docs/PELIAS_TRAINER_MASTER_NOTES.md` |
| `pelias-engine` | Runs the model fast on one 12 GB card plus system RAM | *exists*, own repository; records only (documents, benchmark logs, patches, support archive); **no source code**, HELD. Local commit `ba74954`, not pushed | `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`, `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, `docs/PELIAS_ENGINE_LINEAGE.md`, `docs/spec/` |
| `pelias-server` | The network front of the engine (HTTP, streaming, tool calls). **Not a separate repository:** it is the `server/` part of the engine plus the shared tokenizer | *planned*. Until it exists, the interim server is the patched llama.cpp tier build, started by two launcher files | Master Part 6.12 to 6.14; `RUNBOOK_SERVER_GROUNDUP.md`; `HANDOFF_20261024_SERVER_GROUNDUP.md` |
| Shared base | `libpelias_substrate` (hardware layer) and `libpelias_core` (model files, kernels, comparators): the code both applications use once | *planned*, DECISION (proposed). Where its source lives is not ruled. The trainer's early stages write it first | Master 6.4 |
| Test oracle | llama.cpp commit `2ca15f540` (b11399), built unchanged | External and read-only. Used only under `tests/oracle/` for reference logits and kernel parity. Never linked, never started by the product | Engine blueprint (verification ladder); trainer master Chapter 10 |
| Interface | Dear ImGui drawing layer (MIT) | Trainer only. Header `pelias_lithic_ui.hpp` *exists*; the rest is *planned* | Trainer `src/README.md` |

**Fixed facts shared by all parts** (each is stated once, in its home document): model Qwen3.6-35B-A3B; machine RTX 4070 12 GB, Ryzen 7 5800X3D, WSL2 and Windows-native (native Linux held); VRAM ceiling 10.7 GiB (the 7.5 GiB bank is derived, open decision O1); no code from llama.cpp in the product (trainer decision D37).

## 2. The scaffold: every name

### 2.1 System level

| Name | Kind | Mark |
|---|---|---|
| `PELIAS_SYSTEM_MASTER.md` | This file | *exists* (not in any repository) |
| `pelias-trainer/` | Repository | *exists* |
| `pelias-engine/` | Repository (contains the server) | *exists* |
| `pelias-system/` | The combined directory or repository, when unified | *planned*; not created; DECISION (proposed) at the last step |
| `PELIAS_PROPRIETARY_MASTER_SERVER.md` | The long master: history, derivations, server specification (local archive, not for push) | *exists* on the drive |

### 2.2 `pelias-trainer` (exists: 228 tracked files)

**Root:** `.gitattributes` `.gitignore` `CITATIONS.md` `CMakeLists.txt` `CMakePresets.json` `LICENSE` `README.md` `README_PLAIN_ENGLISH.md` `VERSION`

**Other folders:**

| Folder | Files |
|---|---|
| `cmake/` | `PeliasModule.cmake` `PeliasOptions.cmake` |
| `include/pelias/` | `status.h` `version.h.in` |
| `docs/` | `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` `PELIAS_TRAINER_MASTER_NOTES.md` |
| `resources/help/` | `00_example.txt` `README.txt` |
| `resources/sheets/` | `README.md` `clients.csv` `parameters.csv` `persona.csv` |
| `tests/` | `CMakeLists.txt`; `unit/` (`README.md` `defaults_tests.cc`); `smoke/` (`framework_smoke.cc`); `parity/` `oracle/` `probes/` (each `README.md`) |
| `tools/` | `check_module_deps.py` `check_resources.py` |
| `benchmarks/` | `RESULTS.md`; `raw/` with `gguf-export/` `training-run-0.2/` `training-run-second-phase/` |
| `third_party/` | `README.md` (the oracle pin only; never linked) |

**`src/` modules** (file stems; each has `.h` and `.cc` or `.cu`, plus a `CMakeLists.txt`; `src/README.md` holds the one-line role and the allowed dependencies of each):

| Module | Files |
|---|---|
| `hal` | `arena` `device` `event` `pinned_pool` `stream` `sync_audit` `vram_governor` |
| `plan` | `memory_plan` `probe_constants` `roofline_solver` |
| `io` | `block_types` `gguf_reader` `gguf_writer` `safetensors` `sha256` |
| `config` | `trainer_config` |
| `ingest` | `ingest_plan` |
| `tok` | `chat_template` `tokenizer` |
| `data` | `audit` `dataset` `masks` `packer` |
| `model` | `frozen_store` `lora` `spec` `tape` `trainable_core` |
| `stream` | `boundary_store` `expert_streamer` |
| `kern/gemm_gguf` | `dequant_ref` `dgrad_t` `grouped_tiles` `mma_bf16` `mma_int8` |
| `kern/route` | `combine` `router_topk` `tile_map` |
| `kern/elementwise` | `cast` `gated_norm` `residual` `rmsnorm` `swiglu` |
| `kern/attn` | `attn_bwd` `attn_fwd` `attn_ref_blocked` `rope` |
| `kern/deltanet` | `chunk_bwd` `chunk_out` `chunk_prep` `chunk_state` `conv1d` `gates` `l2norm` `seq_ref` |
| `kern/loss`, `kern/optim` | `ce_chunk`; `adamw` `grad_norm` `lora_grad` |
| `sched` | `head` `layer_bwd` `layer_fwd` `mtp` `step` |
| `train` | `checkpoint` `metrics` `optimizer` `schedule` `trainer` |
| `export` | `export_adapter` `export_core` `splice` |
| `verify` | `canary` `dump_compare` `gate_runner` `kl` |
| `cli` | `main` |
| *Universal-use modules (built after the top-tier gate):* | |
| `project` | `model_key` `project_file` `run_folder` |
| `startup` | `machine_profile` `quick_pass` `vram_scan` |
| `intake` | `checks` `clients` `import` `sheets` |
| `help` | `help_catalog` |
| `warn` | `client_matrix` `fit` `loops` `works` |
| `flow` | `expert` `plan_card` `proposal` `summary` |
| `serve` | `serve_command` (prints the server command for a finished GGUF) |
| `ui` | `pelias_lithic_ui.hpp` |

**Planned, not yet in the repository** (trainer master 5.2 and 12): `tools/oracle_llama/` (test-only program that writes reference dumps; never linked), one `tests/probes/P-xx/` directory per probe, and in `docs/` a model map, format notes and a results ledger.

**What the trainer writes out** (the hand-off files, section 3.2): `pelias_base_0step.gguf`, an adapter GGUF, `merged-core.gguf`, `layout_manifest.json`, the `PELIAS_HOT_SET` file.

### 2.3 `pelias-engine`

**Exists** (records; at local commit `ba74954`):

| Folder or file | What it holds |
|---|---|
| `README.md` `LICENSE` `CITATIONS.md` `.gitignore` `.gitattributes` | Root |
| `docs/` | `PELIAS_ENGINE_LINUX_BLUEPRINT.md` `PELIAS_ENGINE_MASTER_SCAFFOLD.md` `PELIAS_ENGINE_LINEAGE.md` |
| `docs/spec/` | `PROPRIETARY_RUNTIME_SPEC.md` `OPEN_DECISIONS_AND_ORDER.md` `TRAINER_CONTRACT.md` `PROBE_REGISTRY.md` `WAVE0_AND_CEILING.md` `PATCHES_AND_TOOLS.md` `ENGINE_QUEUE_AND_MEASURABLES.md` `MACHINE_AND_DRIFT.md` `RECONCILIATION.md` `RESEARCH_UPDATE_AUDIT.md` `SYNC_REGISTER.md` |
| `docs/scaffold/` | `REPO_INVENTORY.tsv` `verify_inventory.py` |
| `benchmarks/` | `RESULTS.md`; `raw/` (the logged runs) |
| `patches/stage2-tier-cuda/` | The stage-2 patches on llama.cpp (record only) |
| `support/` | `README.md` `MANIFEST.tsv` `tier_build/` `archive/c_ai/`; `local_only/` is git-ignored (withheld material) |

**Planned code tree** (engine blueprint, "Proposed scaffolding and file names"; the letter is the phase that first creates it):

| Folder | Names |
|---|---|
| root | `README.md` `LICENSE` `CMakeLists.txt` `CMakePresets.json` `.clang-format` |
| `cmake/` | `pelias_options.cmake` `probe_flags.cmake` (E2) |
| `docs/` | `BLUEPRINT.md` `RUNBOOK.md` `decisions/` `floors/` (E0) |
| `include/pelias/` | `status.h` `log.h` `trace.h` `timer.h` (E2) |
| `include/pelias/model/` | `gguf.h` `tensor_map.h` `expert_table.h` `arch.h` (E3) |
| `include/pelias/mem/` | `host_alloc.h` `vram_plan.h` `probe.h` (E2, E11) |
| `include/pelias/cpu/` | `expert.h` `threadpool.h` (E4) |
| `include/pelias/gpu/` | `bank.h` `zerocopy.cuh` `handoff.cuh` `expert.cuh` `spine.cuh` (E5 to E7) |
| `include/pelias/engine/` | `layer.h` `decode_loop.h` `prefill.h` `kvcache.h` `state.h` `sampler.h` (E6 to E9) |
| `include/pelias/server/` | `http.h` `tokenizer.h` `chat_template.h` `tools.h` `api_openai.h` (E10) |
| `src/cpu/` | `expert_q4k_avx2.cpp` `expert_q6k_avx2.cpp` `threadpool.cpp` (E4) |
| `src/gpu/` | `expert_q4k.cu` `expert_q6k.cu` `bank.cu` `zerocopy.cu` `handoff.cu` (E5, E6) |
| `src/gpu/spine/` | `attention.cu` `deltanet.cu` `norm.cu` `router.cu` `lm_head.cu` (E7) |
| `src/mem/` | `host_alloc_wsl.cpp` `host_alloc_win32.cpp` `probe.cpp` `vram_plan.cpp` (E2, E11); `host_alloc_linux.cpp` is held with native Linux |
| `src/model/`, `src/` | `arch_moe.cpp` (E3); `main.cpp` with the commands serve, bench, trace, probe (E8) |
| `tools/` (Python, offline) | `collect_trace.py` `export_hot_set.py` `kl_gate.py` `speed.py` `repack_gguf.py` (later) |
| `bench/` | `ram_bw.cpp` `handoff.cu` `dual_bus.cu` `pin_modes.cu` `overlap.cu` `wide_load.cu` (E0, E1) |
| `tests/` | `unit/` `parity/` `oracle/` (the external llama.cpp harness lives only here) |
| `profiles/` | Hot-set files and launcher profiles (E5, E10) |
| `scripts/` | `run_floors.sh` `run_base_cell.sh` `run_sweep.sh` (E1) |

### 2.4 The server (inside the engine, plus the shared tokenizer)

| Name | Role | Mark |
|---|---|---|
| `server/http.h` `tools.h` `api_openai.h` | HTTP/1.1 and Server-Sent Events over raw sockets, OpenAI-compatible endpoints, tool-call parser (master 6.12) | *planned* (E10) |
| `server/tokenizer.h` `chat_template.h` | Native byte-level tokenizer and chat-template renderer; the same code is used by the trainer's `tok/` (master 6.13) | *planned* (written once, early, by trainer stage T2) |
| `os/net` | Sockets: POSIX on WSL2, Winsock2 on Windows (master 6.5) | *planned* |
| llama-server catalogue | The table of what llama-server does, row by row, that becomes the server's specification (master 6.14); no file name chosen | *planned*; not written; waits on open decision O8 |
| `start_pelias_server.bat` `stop_pelias_server.bat` | Interim: start and stop the Windows tier-build server for the author's model | *exists* on the drive (`2026-10-24\win_bat\`) |
| `RUNBOOK_SERVER_GROUNDUP.md` `HANDOFF_20261024_SERVER_GROUNDUP.md` | The server build order and the state at hand-off | *exists* at the root of the drive backup folder |
| `ENGINE_HOLD_QUEUE.md` | The hold and the queue of held engine tasks | *exists* on the drive (`2026-10-24\engine_hold\`) |

### 2.5 Shared base: names by layer (master 6.4; all *planned*, DECISION proposed)

| Layer | Module | Who uses it | First built |
|---|---|---|---|
| `libpelias_substrate` | `os/file_map` `os/threads` `os/large_pages` | Both | T1, E3 / E4 / E11 |
| | `os/net` | Engine (server) | E10 |
| | `hal/device` `hal/stream` `hal/event` `hal/arena` `hal/vram_governor` `hal/sync_audit` | Both | T1, E2 |
| | `hal/pinned_pool` with `host_alloc_wsl` and `host_alloc_win32` | Both | T1, E2, E11 |
| | `plan/` (`probe_constants` `roofline_solver` `memory_plan`) | Both | T1, E11 |
| | `bus/doorbell_handoff` `bus/cpu_threadpool` | Engine | E6 / E4 |
| | `bus/stage_streamer` | Trainer | T10 |
| | `bank/hot_set_solver` `bank/hot_set_io` | Both | E5, T12 |
| | `quant/` (Q4_K, Q6_K, Q8_0, q8_1, Q8_K block formats and quantizers) | Both | T1b, T3, E4 |
| `libpelias_core` | `io/gguf_reader` `io/gguf_writer` `io/gguf_splice` `io/sha256` `io/json` `io/safetensors` | Both (writer: trainer) | T1, T1b, T12, E3 |
| | `tok/` | Both | T2, E10 |
| | `kern/` spine family (norms, attention, DeltaNet, router, lm_head) | Both | T4 to T6, E7 |
| | `kern/` expert and matrix family | Engine (hot and cold branch), trainer (GEMM, LoRA) | E4, E5, T3 |
| | `model/` | Both (stores: trainer) | E3, T7 |
| | `verify/` | Both | E3, T12 |

## 3. Order of operations, from the ground up

### 3.1 The ladder (master 7.1; the engine and trainer columns name the same step in each plan)

Each step starts only when the gate of the step before it is met. Trainer steps are not held. Steps marked HELD wait for the author's word.

| # | Build | Trainer | Engine | Gate that closes it | Status today |
|---|---|---|---|---|---|
| 1 | Measure the machine; the author signs the switches | T0 | E0, E1 | Switch summary signed; the gap between the modelled ceiling and the live speed explained | Wave 0 ran on WSL2 and Windows-native. The gap split is DERIVED, not isolated. Not run: P-M4, the Batch C kernel probes (source not finished), P-E10, P-E11 |
| 2 | Foundations: device layer, VRAM governor, arenas, file mapping, GGUF reader, memory solver, test harness | T1 | E2, E3 (reader) | Lists all 753 tensors; a plan above 10.7 GiB is refused; clean build on WSL2 and Windows-native | Trainer framework builds host-only; `io/gguf_reader` is a stub. Engine: no code |
| 3 | Block formats and quantizers, then the base file from the bf16 shards | T1b | E3, E4 | P-E6 byte-identical to the oracle; ingest output equals the reference (P-X6) | Done 2026-10-27: P-E6 and P-X6 passed (P-X6: byte-identical except 13 ssm_a values, 1 ulp, by design) |
| 4 | Tokenizer, chat-template renderer, data pipeline | T2 | E10 (reuse) | P-D1 and P-D2: zero mismatches | Started 2026-10-27: tokenizer built (`tok/`); P-D1 against llama.cpp passed in the sandbox (0 mismatches, 2,455,727 tokens); P-D1 against Hugging Face: first run 3 of 7 files mismatched (NFC data version), tables rebuilt to the reference's own versions, rerun pending; renderer and data pipeline not started |
| 5 | Kernel families, each with a slow reference first | T3 to T6 | E4, E5, E7 | Rung 1 tolerances; kernel parity | Not started |
| 6 | The external oracle harness and the project's own comparators | T12 (verify) | E3 | Oracle reproduces the stock noise floor (KL 0.017353 at batch 16, 0.018286 at batch 1024) | Not started |
| 7 | One layer, a tiny model, the full forward pass | T7, T8 | E6, E7 | Rung 2 parity; rung 3 KL within the stock noise | Not started |
| 8 | Backward pass, LoRA, optimizer; streamer and scheduler | T9, T10 | E5, E6 | Rung 4; zero allocations in the step loop | Not started |
| 9 | Training loop, evaluation, checkpoints; export, verify and hand-off to the engine | T11, T12 | E3 (K3 check), E8 | Rungs 6 to 8; hash proof; KL gates K1, K2, K3 | Not started |
| 10 | Performance, optional training modes | T13, T14 | | Targets met or the gap explained by measurement | Not started |
| | **Top-tier gate (the author's order, item 2), then make it universal:** the `project` `startup` `intake` `help` `warn` `flow` `ui` modules | after the top-tier gate | | The author's word | Names exist; nothing built |
| 11 | Decode loop, prefill, long context (CPU and GPU expert kernels, threadpool, doorbell, bank) | | E4 to E9 | At least the llama.cpp tier's speed, then 55.0 tok/s at 12k tokens | **HELD** |
| 12 | The server: sockets, HTTP, streaming, tool calls; the llama-server catalogue first | `serve` (at unify) | E10 | Drop-in for the launcher; capability probes; a Visual Studio session works | **HELD**; catalogue not written |
| 13 | Start-up probe, plan card, capability detection | shared `startup` | E11 | Same code base passes the step 11 and 12 gates on WSL2 and Windows-native | **HELD** |
| 14 | Gated extras; hardening; a second model through the plug-in interface | (step 10) | E12, E13 | Each mode's own pre-declared gate | **HELD** |
| 15 | **Unify:** one shared base, one set of probes, one release; the trainer's `serve` starts the proprietary server | after both | after both | Both applications pass their ladders on the shared base | Not started |

The detail behind each step is in its home document: trainer stages T0 to T14 in trainer master Chapter 12; engine phases E0 to E13 in the blueprint ("Order of operations") and the scaffold (section 6); the server's eleven build steps in `RUNBOOK_SERVER_GROUNDUP.md` (from master 6.15).

### 3.2 The tie-ins: what passes from one part to the next

| From | To | What moves | Checked by |
|---|---|---|---|
| Original bf16 shards | Trainer, Stage A (T1b) | Converted to `pelias_base_0step.gguf` | P-X6 against the P-X0 reference |
| Trainer (T12) | Engine (E3) | `merged-core.gguf` with `layout_manifest.json`; optionally in the expert-last layout (experts on 2 MiB boundaries, a `pelias.pad` entry) | The reader parses file and manifest identically; gate K3 |
| Trainer (T12) | Engine (E5) | `PELIAS_HOT_SET` file (resident experts per layer) | Slot-table loader accepts it; P-E14 |
| Engine measurements | Trainer | Named inputs released at each engine milestone (E0 timers, E3 noise floor, E5 wide-load result, E8 and E9 VRAM plan) | `docs/spec/TRAINER_CONTRACT.md` (the two-way contract, trainer master 16.7) |
| Measured machine rates | Both (`plan/`) | `probe_constants.json` | The plan card is printed before any allocation |
| llama.cpp `2ca15f540` | Both (tests only) | Reference logits and kernel parity data | KL gates K1 (trainer), K2 (export), K3 (engine) |
| Shared `tok/` | Trainer data pipeline and the server | One tokenizer and one chat-template renderer | P-D1 and P-D2 |
| Shared `io/gguf_reader` | Trainer `ingest`, `export`, `verify`; engine loader | One file reader | Step 2 gate |
| Trainer `serve` module | Engine server | The command that starts the server for a finished GGUF | At unify |
| VRAM governor | Both | The 10.7 GiB ceiling | A plan above it is refused |

## 4. Data and parameters

Every row carries its label and the place it comes from. A number that does not exist yet says *unmeasured* and names the probe that will measure it; nothing here is estimated by this page. Where two sources print slightly different roundings of one value, both are given. **Source keys:** TM = trainer master (`PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`); TN = trainer notes; EB = engine blueprint; WC = `docs/spec/WAVE0_AND_CEILING.md`; M = the long master. **Labels:** MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED. "A / B" means WSL2 / Windows-native.

### 4.1 Machine and toolchain

| Item | Value | Label (source) |
|---|---|---|
| GPU | RTX 4070 12 GB (AD104, `sm_89`); 46 SMs; 5,888 CUDA cores; 184 tensor cores; 36 MiB L2; 504 GB/s; 200 W; boost 2,475 MHz | PUBLISHED (TM 2.1) |
| GPU overclock | Memory +1000 MHz; core about 2,775 to 2,790 MHz at 965 mV; no ECC | AUTHOR-STATED (TM 2.1) |
| Shared memory, copy engines | 100 KiB per SM; 99 KiB (101,376 B) per block opt-in; 5 async copy engines | PUBLISHED; MEASURED, P-H1 (WC) |
| CPU | Ryzen 7 5800X3D; 8 cores, 16 threads; AVX2 only (no AVX-512); 96 MB L3 | AUTHOR-STATED / PUBLISHED (TM 2.1; EB) |
| RAM | 4 × 16 GB DDR4-3200; 51.2 GB/s peak; 63.9 GiB visible on Windows | PUBLISHED; MEASURED (TM 2.1) |
| Operating systems | Windows 11 Pro, build 26300; WSL2 kernel 6.18.54.1 with a 48 GB RAM cap and 48 GB swap on D: | AUTHOR-STATED; MEASURED (TM 2.1) |
| Toolchain | Driver 617.42 (CUDA 13.4 capable); `nvcc` 13.0.88 on WSL2 and 13.4.92 on Windows; Visual Studio 18 Professional; avoid toolkit 13.2 | MEASURED; PUBLISHED (EB; TM 2.1) |
| Windows settings | Hardware GPU scheduling off; display timeout at default (2 s); power plan Balanced; Resizable BAR on (BAR1 total 16,384 MiB); fixed 32 GiB page file | MEASURED (TM 2.1; WC) |
| **VRAM ceiling** | **10.7 GiB**, absolute (author ruling 2026-10-23, D26; 10.9 before it). Ruled 2026-10-24 (D28): measured at start as the smaller of CUDA free and NVML free after the context exists, minus a margin; 10.7 stays this machine's value until the margin is measured | AUTHOR-STATED / DECISION (EB; TM 15) |
| Free VRAM (P-H8) | Total 11.99 GiB. CUDA free after the context: 10.83 / 10.85 GiB; NVML free 10.95 / 10.85. Context costs 154 / 138 MiB. Windows holds 914 / 1,031 MiB before the process starts. Idle after a fresh WSL restart: 579 MiB | MEASURED (WC; TM 2.1) |
| Pinned memory | 32 GiB (34.4 GB) held on both platforms; the limit is above what was tried. Allocation 0.76 to 1.0 s per GiB on A, 0.069 s per GiB on B. `RLIMIT_MEMLOCK` 64 MiB does not stop `cudaHostAlloc` | MEASURED, P-H7 (WC) |

### 4.2 The model (header of the served file)

| Item | Value | Label (source) |
|---|---|---|
| Model and file | Qwen3.6-35B-A3B; architecture `qwen35moe`; Q4_K_M, no imatrix; 753 tensors; 21,713,463,296 bytes (20.22 GiB); alignment 32; tensor data starts at byte 10,991,104 | READ-SRC (TM 2.2, 2.3) |
| Blocks and size | 41 blocks (40 trunk + MTP block `blk.40`; `nextn_predict_layers` 1); embedding 2,048; vocabulary 248,320; context length 262,144 | READ-SRC (TM 2.2) |
| Layer pattern | Full attention every 4th layer (3, 7, 11, …, 39: 10 layers); DeltaNet in the other 30; `blk.40` is a full-attention block | READ-SRC (TM 2.2) |
| Full attention | 16 query heads; 2 KV heads; head dim 256; RoPE dimension 64; sections [11, 11, 10, 0]; base 10,000,000 | READ-SRC (TM 2.2) |
| DeltaNet | Conv kernel 4; `group_count` 16; `time_step_rank` 32; state size 128; inner size 4,096 | READ-SRC (TM 2.2) |
| Mixture of experts | 256 experts, 8 used per token; expert FFN 512; shared-expert FFN 512; RMS epsilon 1e-6 | READ-SRC (TM 2.2) |
| Parameters | One expert 3,145,728; one layer 805,306,368; 40 layers 32,212,254,720; active per token 2.94 B (experts 1.007 B, shared experts 125.8 M, DeltaNet projections 755.0 M in + 251.7 M out, attention 272.6 M, routers 21.0 M, lm_head 508.6 M) | DERIVED (TM 2.3) |
| Work per token | 20.39 GFLOP at a window of 8,192 tokens; 19.64 GFLOP at 6,144 | DERIVED (TM 13.2) |

### 4.3 Byte map: the sizes the plans are built from

| Item | Value | Label (source) |
|---|---|---|
| Block sizes | Q4_K 144 B; Q6_K 210 B; Q8_0 34 B; q8_1 36 B; Q8_K 292 B | READ-SRC (M 6.4, 6.11) |
| Per expert (bytes) | Gate 589,824; up 589,824; down 589,824 (Q4_K) or 860,160 (Q6_K) | READ-SRC / DERIVED (TM 5.4) |
| One expert layer | 432 MiB (down Q4_K) or 498 MiB (down Q6_K); gate + up 288 MiB | DERIVED (TM 2.3) |
| Layers with Q6_K down | 0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39 (20 layers) and the MTP block 40; the other 20 trunk layers use Q4_K | READ-SRC (TM 2.3) |
| All routed experts | 40 trunk layers 18,600 MiB (18.164 GiB); with the MTP layer 19,098 MiB (18.65 GiB) | DERIVED (TM 2.3) |
| Q6_K 420-byte rows | 20 trunk layers plus `blk.40`; the zero-copy kernel reads them at 9.59 GB/s against 26.2 GB/s for the checksum kernel | READ-SRC; MEASURED (WC 3.5) |
| Tensor classes (MiB) | `token_embd` Q4_K 272.8; `output` Q6_K 397.9; gate and up experts 5,904.0 each; down experts 4,410.0 (Q6_K) + 2,880.0 (Q4_K); routers (F32) 82.0; shared gate and up 23.1 each; shared down 17.2 + 11.2; DeltaNet `attn_qkv` 183.8 + 144.0; DeltaNet `attn_gate` 135.0; `ssm_out` 135.0; `ssm_alpha` and `ssm_beta` 1.1 each; `ssm_conv1d` 3.8; attention `attn_q` 99.0, `attn_k` 6.2, `attn_v` 5.7 + 2.2, `attn_output` 49.5; MTP `eh_proj` 4.5 | READ-SRC (TM 2.3) |
| LoRA targets | 310 modules (10 × 4 attention, 30 × 5 DeltaNet, 40 × 3 shared expert); rank 16 gives 21,166,080 parameters | READ-SRC / DERIVED (TM 2.3) |
| Router tensors | 41 tensors of 2,048 × 256, F32; 21,495,808 parameters | DERIVED (TM 7.12) |

**The core (the trainable spine), parameters and MiB by type** (TM 2.3, 9.4; DERIVED; the MTP block adds 47.2 M parameters):

| Class | Parameters | Served mix | All Q8_0 | All Q6_K | bf16 |
|---|---|---|---|---|---|
| Shared experts | 125.8 M | 72.7 | 127.5 | 98.4 | 240.0 |
| DeltaNet in (qkv + z) | 755.0 M | 462.8 | 765.0 | 590.6 | 1,440.0 |
| DeltaNet alpha and beta | 3.9 M | 2.1 | 4.0 | 3.1 | 7.5 |
| DeltaNet out | 251.7 M | 135.0 | 255.0 | 196.9 | 480.0 |
| Full attention | 272.6 M | 147.8 | 276.2 | 213.3 | 520.0 |
| **Total** | **1,409.0 M** | **820.4** (820.3 in TM 9.4) | **1,427.7** (+607) | **1,102.3** (+282) | **2,687.5** |

### 4.4 Measured machine floors (Wave 0, both platforms; A / B)

| Probe | Result | Label (source) |
|---|---|---|
| P-H3 tensor rates (sustained) | bf16 65.2; fp16 64.9; int8 259.5; fp8-e4m3 129.8 T(FL)OPS; SM clock 2,760 to 2,775 MHz; 99.8 % of the burst rate | MEASURED (WC) |
| P-H10 canary | 10,000 GEMM runs, 0 mismatches, 1 distinct hash | MEASURED (WC) |
| P-H4 soak, 30 min | 17,685 / 17,700 cycles, 0 mismatches; SM clock median 2,760 MHz; 43 to 55 / 49 to 56 °C; power median about 107 W of 200 W | MEASURED (WC) |
| P-H2 bandwidth | VRAM read 523 / 534, write 448 / 454 GB/s; host DRAM read: 1 thread 37.1 / 37.9, best (4 threads) 43.7 / 44.8 GB/s | MEASURED (WC) |
| P-H5 transfers (256 MiB) | Pinned H2D and D2H 26.3 GB/s each way; pageable H2D 10.3 / 11.8, D2H 10.8 / 11.8; zero-copy kernel read 26.3; 4 KiB copy 73 / 31 µs, 1 MiB copy 113 / 72 µs | MEASURED (WC) |
| P-H6 contention | H2D alone 26.1; with 4 / 8 / 16 CPU reader threads 26.0 / 25.3 / 23.6 (A); CPU + H2D together 46.6 to 48.7 GB/s (A) | MEASURED (WC) |
| P-M3 host hold | 24 GiB pinned + 4 GiB pageable held 30 min: 0 mismatches, 0 MiB swapped in, H2D 25.8 / 26.0 GB/s | MEASURED (WC) |
| P-H9 graphs and flush | Eager launch 4.70 / 2.18 µs per kernel; graph replay 0.716 / 0.707 µs; `cudaStreamQuery` flush on a busy stream median 76.6 / 8.05 µs per call | MEASURED (WC) |
| P-H12 overlap | Ordinary submission: all-three efficiency 0.47 / 0.52 (many short kernels), 0.00 (one long kernel). With a flush after each enqueue: 0.92 on both; H2D + D2H pair 0.84 | MEASURED (WC) |
| P-H13 display timeout | Stages of 2, 5 and 30 s, a 10 s queued set and a 10 s graph all completed with no errors | MEASURED (WC) |
| Engine floors (microbenchmarks 02 to 07, 2026-10-20) | CPU alone 43.8 to 45.9 GB/s; zero-copy checksum kernel 26.2; zero-copy compute kernel 9.59; PCIe alone 26.1; combined CPU + zero-copy 47.62 (26.56 CPU + 21.06 GPU = 93.0 % of 51.2) | MEASURED (EB; WC) |
| Handoff | Stream sync 94 µs per layer; resident-kernel doorbell 13.3 µs median, 18.7 µs p99; flag alone 3.4 µs | MEASURED (EB; WC) |

### 4.5 The ceiling model and the recorded speeds (all from the patched llama.cpp tier build)

**Ceiling inputs** (EB "Where the numbers stand"; WC 3.4):

| Input | Value | Label |
|---|---|---|
| GPU spine per token | 7.73 ms (a 7.4 ms figure is open; shown as a range until the per-token timers close it) | ESTIMATE / assumed |
| Hot bank | 7.5 GiB bank (4,170 experts incl. 40 zero slots = 72.6 MiB, as the source prints it); byte-weighted hit rate 0.8273 (0.83 in the blueprint), from 84,503 fit and 36,657 held-out decode steps | DERIVED (simulated) |
| Cold read time per token | 2.86 ms at 26.2 GB/s; 3.33 ms at 9.59 GB/s, assuming perfect overlap | DERIVED |
| Handoff in the model | 0.25 ms per token (about 6.25 µs per layer) | ASSUMED |
| **Ceiling** | **10.87 ms per token = 92.0 tok/s** (26.2 GB/s kernel); **11.34 ms = 88.2 tok/s** (9.59 GB/s kernel); 88.2 to 90.9 under the 7.4 ms ruling | DERIVED |

**Recorded runs** (tok/s; ms per token; gap to the ceiling):

| Run | Context | Speed | ms per token | Gap to ceiling |
|---|---|---|---|---|
| WSL2 Cell A (7.0 GiB bank, bypass 32, `-ub 4096`) | short | 37.9 | 26.39 | 15.05 to 15.52 ms |
| WSL2 Cell A | 12k | 42.1 | 23.75 | 12.41 to 12.88 ms |
| Windows-native Cell A (two launches) | short | 47.3 / 46.8 | 21.14 / 21.37 | 9.80 to 10.50 ms |
| Windows-native Cell A (two launches) | 12k | 53.7 / 54.1 | 18.62 / 18.48 | 7.14 to 7.75 ms |
| WSL2 best cell (7.5 GiB bank, bypass 32, `-ub 2048`) | short / 4k | 53.5 / 53.0 | 18.69 / 18.87 | 7.35 to 7.82 ms |
| WSL2 best cell | 12k | 49.0 | 20.41 | 9.07 to 9.54 ms |

All MEASURED, gap DERIVED (WC 3.4). Over all launches and load modes: 44 to 54 tok/s short and 43 to 49 at 12k (WSL2) (EB). The old 57.0 figure is retired. No run of a non-llama server exists, so the cause of the gap is not isolated (UNVERIFIED).

**Where the WSL2 time goes** (DERIVED, one launch; WC 3.5): CPU cold-expert compute 10.1 ms against 2.86 ms modelled, with zero overlap (parts 30.5 ms against 30.3 ms wall); GPU compute 6.6 ms; stream-sync handoff and activation copies 4.6 to 5.0 ms or more (model 0.25 ms); copy-back and launch 2.8 ms (copy-back 2.3 + launch 0.5). Any-miss share: 63.9 % of layer-steps (25.56 of 40 layers per token) have at least one cold expert; 36.1 % have none; mean 2.16 misses per missing layer (MEASURED trace). The Windows-native tier build, same cell: 5.25 and 5.02 ms per token less than WSL2 at short context (24.8 % and 23.5 % faster) and 5.13 and 5.27 ms less at 12k (27.6 % and 28.5 % faster); two launches against one, so not yet a claim.

**Other recorded engine numbers:**

| Item | Value | Label (source) |
|---|---|---|
| Hit rate, live, 7.0 GiB bank | 0.7753 to 0.7793 at 1 token; 0.733 at 2 to 31 tokens; simulated 0.806 | MEASURED / simulated (WC; TM 2.7) |
| Hit rate by layer and domain | Layers 0 to 2 lowest (0.61 to 0.65), layer 20 highest (0.90 to 0.92); at the 6.1 GiB bank: agent 0.790, code 0.753, persona 0.799, reasoning 0.674 | MEASURED (TM 2.7, 7.12) |
| Simulated hit rate by bank size | 0.43 / 0.54 / 0.63 / 0.70 / 0.74 / 0.76 / 0.85 at 2 / 3 / 4 / 5 / 5.6 / 6 / 8 GiB; about +1 tok/s per +0.01 of hit rate | DERIVED (TM 7.12) |
| Peak VRAM by bank size | 7.0 GiB bank at `-ub 4096`: 10.7 GiB; 7.5 GiB: 10.40 to 10.44 (default micro-batch), 10.67 to 10.69 (`-ub 2048`); 8.0 GiB: 11.0 to 11.19; 8.5 GiB: 11.4 | MEASURED, whole card (TM 2.7; EB) |
| Headroom under 10.7 at the 7.5 GiB bank | 0.26 to 0.30 GiB (default micro-batch); 0.01 to 0.03 GiB (`-ub 2048`); less a 0.1 GiB margin: 0.16 GiB and none (−0.09) | DERIVED (TM 2.7, 9.4) |
| Prompt reading at 12k | 1,670 tok/s (tier); 2,238 (stock, load mode none); load-mode effect 863 to 929 against 1,667 to 1,680 tok/s | MEASURED (EB) |
| Multi-token prediction | Draft 2/3 at a 7.0 GiB bank: 12k decode 51.5 to 52.9 tok/s against 45.4 base | MEASURED (TM 2.7) |
| Accuracy floor of the served file | KL 0.017353 (batch 16) and 0.018286 (batch 1024) between two stock llama.cpp runs; reruns are deterministic; `tier8_nobank` 0.0017; `tier8` 0.0106 (meaning of the columns to be re-verified). Strict limit KL 0.005 / top-1 99 % is reported for information only (H6 ruling 2026-10-22: the float-spread reading governs) | MEASURED; AUTHOR-STATED (EB; TM 2.7) |
| Launch-to-launch spread | 5 % to 21 % | MEASURED (EB) |

### 4.6 Memory plans (trainer) and expert traffic

**Trainer device memory, window 8,192 tokens, step up to 32 k tokens** (TM 5.3):

| Item | GiB | Label |
|---|---|---|
| CUDA context and driver | 0.60 | ESTIMATE |
| Frozen core base (served bytes) | 0.80 | READ-SRC |
| `lm_head` (Q6_K) | 0.39 | READ-SRC |
| Token embedding (Q4_K) | 0.27 | READ-SRC |
| Routers (F32) and small F32 tensors | 0.09 | READ-SRC |
| Expert slots, 2 × 498 MiB | 0.97 | READ-SRC |
| LoRA fp32 masters, gradients and Adam states (r = 16) | 0.32 | DERIVED |
| LoRA bf16 compute copies | 0.04 | DERIVED |
| Layer-boundary tensors (current, next, gradient; fp32; 32 k tokens) | 0.75 | DERIVED |
| Per-layer working set | 2.50 | ESTIMATE (P-M2 measures) |
| Cross-entropy chunk (256 positions × 248,320) | 0.65 | MEASURED in v0.2 |
| Routing tape and index buffers | 0.02 | DERIVED |
| Safety margin | 0.50 | DECISION |
| **Total** | **7.90** | |
| **Headroom under 10.7 GiB** | **2.8**; with the Mode Q `W_eff` copy (+0.8 to 1.1) 1.7 to 2.0 | DERIVED |
| Mode L only | +0.32 (router copy, gradient, Adam states) | DERIVED |

Ways the headroom is spent, in order: a third expert slot +0.49 GiB; LoRA rank 32 +0.32 GiB, rank 64 +0.96 GiB; each resident expert layer +0.42 to 0.49 GiB.

**Trainer host memory** (TM 5.3): pinned expert store 18.65 GiB (DERIVED); pinned boundary store 5.0 GiB at 16 k tokens, 10.0 GiB at 32 k (fp32); non-pinned about 4 GiB (ESTIMATE); total about 28 GiB at 16 k and 33 GiB at 32 k per step; the WSL2 cap is 48 GiB.

**Expert traffic** (TM 5.4, 2.6): the new streamer moves 79 fetches, about 35.8 GiB per step regardless of the tokens in the step (DERIVED). v0.2 moved 72 fetches of 432 MiB, 30.38 GiB per micro-batch, about 121.5 GiB per optimizer step (MEASURED, DERIVED). The traffic time is 2 × 18.16 GiB = 39.0 GB at 26.1 GB/s = 1.49 s per step (DERIVED).

### 4.7 Trainer parameters

**The baseline being replaced and the parity recipe** (TM 2.6, 7.1; READ-SRC / MEASURED):

| Setting | Value |
|---|---|
| Optimizer | AdamW, β = (0.9, 0.999), ε = 1e-8, weight decay 0.0 |
| Gradient clip | Global norm 1.0 |
| Schedule | Linear warm-up over 5 % of the steps, then cosine to 0 |
| Learning rate | 1e-4 (run 1); 5e-5 (second phase) |
| Epochs, seed | 2; 3407 |
| LoRA | Rank 16, α 32 (scale 2), dropout 0.0, 310 modules |
| Documents per optimizer step | 4 (batch 1, accumulation 4), about 4.9 k processed tokens per step |
| Maximum sample length | 8,192 tokens; longer is a hard failure (author policy) |
| Cross-entropy | 256-position chunks |
| v0.2 run 1 | 616 steps, 202.7 min; median 247.85 tok/s (p10 88.4, p90 598.1); 4.55 s per micro-batch (token budget 4,096); stream wait 26.8 %; peak 8.02 GiB VRAM and 20.0 GiB host; largest gradient norm 2.82 (clipped to 1.0) |
| v0.2 second phase | 114 steps, 37.9 min; median 56.45 tok/s; token budget 8,192; stream wait 29.5 %; peak 9.36 GiB VRAM |
| Data (counts only) | 1,300 rows (1,232 train, 68 eval); 987,440 assistant tokens; 2 epochs; about 3.01 M processed tokens over two epochs (DERIVED) |

**New-trainer defaults** (TM 9.4; `src/config`; DECISION, rulings D31, D32, D34): core mode Mode Q (trained through the exact quantizer at the served types); `Δ_max` = 0 (no extra bytes); byte-budget solver off; all type-raise toggles off; planner automatic; overrides only in Advanced/Developer mode. Loss: step-wide token normalization, weight 1 on tokens the model must generate when served (D8). LoRA: fp32 masters, bf16 copies; initialization Kaiming-uniform (a = √5), B = 0; options off by default: rank 32 or 64, LoRA on `lm_head`, trainable shared-expert gate (40 × 2,048 = 81,920 parameters) and norms (about 175 k parameters). MTP co-training weight λ = 0.1 to 0.3 (to be ablated). Checkpoints: four registries, atomic writes, the last three kept. Canary kernel every 50 steps. Stop conditions: a plan above 10.7 GiB or free VRAM below the start threshold is refused; sustained throttle over 60 s checkpoints and stops; a non-finite gradient stops the run.

**Mode L (optional, gated, off by default; TM 7.12):** router fine-tune of 21,495,808 parameters in fp32; starting router learning rate 2×10⁻⁵ (DECISION as a starting value only); loss weights λ_trust and λ_res swept in P-L4; cost 0.21 % of the step's FLOPs and +0.32 GiB of VRAM.

### 4.8 Trainer speed model and targets (TM 1.4, 13)

| Item | Value | Label |
|---|---|---|
| Peak tensor rates, stock clocks | bf16 58.3 TFLOPS; int8 233 TOPS (fp8 116.6) | DERIVED from published RTX 4090 rows |
| Peak tensor rates, overclock | bf16 65.4 and int8 261 derived; **measured 65.2 and 259.5** | DERIVED; MEASURED (P-H3) |
| Int8-eligible share of FLOPs | 52 % at a window of 8,192 | DERIVED |
| GEMM efficiency ε; other-work factor f | ε 0.3 to 0.6 (ε − 0.1 for int8); f 0.75 (0.6 pessimistic); ε unmeasured until P-K1 to P-K3 | ESTIMATE |
| Predicted tok/s (window 8,192, stock, f = 0.75), bf16 / int8 forward | ε 0.3: 643 / 956; ε 0.4: 858 / 1,318; ε 0.5: 1,072 / 1,675; ε 0.6: 1,287 / 2,031 | DERIVED |
| Ceiling (ε = 1, f = 1) | 2,859 tok/s bf16-only or 4,708 mixed (stock); 3,206 / 5,279 overclocked | DERIVED |
| **Targets** | Minimum viable **600 tok/s** (2.4× v0.2); target **1,200** (4.8×); stretch **2,000** (8×) | DECISION |
| Time for the v0.2 run | 83.7 / 41.9 / 25.1 min at 600 / 1,200 / 2,000 tok/s, against 202.7 min | DERIVED |

### 4.9 Gates and thresholds

**Engine phase gates** (EB; each starts only when the one before it is met):

| Phase | Gate |
|---|---|
| G3 (decode speed) | At least 55.0 tok/s at 12k tokens |
| E3 | Oracle reproduces the stock floor (KL 0.017353 at batch 16, 0.018286 at batch 1024); a trainer-exported GGUF and its manifest parse identically; K3 holds |
| E4 | CPU expert throughput at or above llama.cpp's path (about 27 GB/s effective, DERIVED); output matches the oracle |
| E5 | Zero-copy compute kernel reaches 26.2 GB/s (from 9.59); bank experts match the oracle |
| E6 | Layer latency within the E0 model; handoff p99 no worse than 18.7 µs |
| E8 | At least the tier's result, then G3 (55.0 tok/s at 12k), then the ceiling table |
| E9 | Prompt reading at least 1,670 tok/s at 12k (aimed at 2,238); peak VRAM within the ceiling |
| E10 | Persona probe not-pass 12 of 111 or fewer (baseline 9); needles 12/12 to 120k; a Visual Studio Copilot session works |
| E12 | Zero-copy third branch: GPU rate at least 15 GB/s and combined at least 1.25× the CPU alone; LRU only if the live hit rate is well below 0.83 |
| Context | Contexts up to 131,072 in E9 |

**Trainer rungs and kernel tolerances** (TM 10.2, 12.4, 6.11; DECISION, initial): rung 2 relative L2 within 2× the llama.cpp CPU-against-GPU difference; rung 3 truncated KL ≤ 1.0 × the float spread on 64 held-out documents; rung 4 directional-derivative match to 1e-8 relative in fp64 and LoRA-gradient cosine ≥ 0.99999 against PyTorch on a 4-layer slice; rung 5 PCIe stall ≤ 5 %, zero allocations in the step loop; rung 7 KL(file ‖ trainer reference) ≤ 1.0 × the float spread; rung 8 regression ≤ 2 % and VRAM ≤ 10.7 GiB at the 7.5 GiB bank. Dequantizers and quantizers bit-exact; bf16 and int8 GEMMs relative L2 ≤ 1e-6; router identical top-8 sets; attention forward ≤ 5e-4, backward ≤ 2e-3; DeltaNet forward ≤ 1e-4 per layer-window (fp64 ≤ 1e-5), backward ≤ 1e-3; cross-entropy loss ≤ 1e-6; AdamW ≤ 1e-6.

**Mode L gates** (TM 7.12; DECISION): G-L1 held-out CE ≤ 1.01 × the reference run; G-L2 hit rate at least +0.03 over the reference; G-L3 no class more than 0.01 below the reference; G-L4 decode faster than the reference by more than run-to-run noise (3 launches per side).

### 4.10 The interim server's launch parameters (as written in the start file of 2026-10-24; the last record says the file had not been run)

| Group | Setting |
|---|---|
| Address | Host 127.0.0.1, port 8012, the model alias (in the author's local profile), chat template on (`--jinja`) |
| Context and slots | `-c 32768`; `-np 1`; `-n -1`; `--no-context-shift` |
| Placement | `-ngl 99`; `--n-cpu-moe 40`; `--fit off`; `--load-mode none` |
| Attention and cache | `--flash-attn on`; key and value cache type `q8_0` |
| Threads and batches | `-t 6`; `-tb 8`; `-b 4096`; `-ub 4096` |
| Sampling | `--temp 1.0`; `--top-p 0.95`; `--top-k 20`; `--min-p 0.0`; `--reasoning-format deepseek` |
| Logging | `-lv 4` |
| Tier environment | `PELIAS_HOT_SET` (hot-set file for the 7.0 GiB bank); `PELIAS_TIER_BYPASS_N` 32 (cell A; 9 is named the production setting in a sweep script and has never been run); `PELIAS_TIER_MAX_MIB` 7300 (6400 in that unrun bypass-9 sweep); `PELIAS_TIER_VERBOSE` 1; `PELIAS_TIER_STATS_FILE` (written on a clean stop) |
| Toolchain on the path | CUDA 13.4 `bin\x64`, for that window only |

Build these settings belong to: the patched llama.cpp tier build on commit `2ca15f540` (b11399); the mapping of each setting to the proprietary server's parameters is master 6.14 (still to be written).

## 5. What is not settled (so this order is not read as permission)

- **The engine hold** (steps 11 to 14) stands until the author lifts it. Open decision O13: when it lifts, and whether engine work may start before the trainer is "top tier".
- **Server decisions:** O8 (add the llama-server catalogue to the hold queue), O6 (may client request formats be looked up), O10 (special-token text in user content), O11 (restore the original decode-loop kill criterion: fall back to a llama.cpp patch series), O12 (does "WSL2 and Windows-native, native Linux held" apply to the engine). All in `docs/spec/OPEN_DECISIONS_AND_ORDER.md`, 7.16.
- **Not pushed:** the trainer's 3 local commits (`62b2174` `707fd56` `953c9b1`) and the engine's local commit `ba74954`.
- **Not written yet:** the Batch C kernel probes (source not finished), the llama-server catalogue, the `pelias-system` directory itself.
- **Not searched:** the author's proposals folder and the older archive drive (open item O18; both searched on 2026-10-26, see PELIAS_SYSTEM_MASTERWORK.md 19.2 and 20).
- **Where a name or number in this page and a home document disagree**, the home document wins; this page is only the index.

## 6. Draft build: what is needed on WSL2 and Windows 11

**What the draft build is:** the trainer's framework (the host-only build and the CUDA build, with their tests), the shared base as it gets written, and the Wave 0 probe program. The engine and server drafts use the same tools (the engine stays HELD; native Linux is held), so they add nothing to install. The author runs every install and every GPU program.

**Where the facts come from (all records, none of them today's state):** the WSL toolchain survey of 2026-10-18; the WSL census of 2026-10-23; the Windows facts run of 2026-10-23; the configure log of the Windows tier build; the trainer's `CMakeLists.txt`, `CMakePresets.json` and `cmake/PeliasOptions.cmake`. Labels: the installed versions are MEASURED on the dates recorded; the CUDA 13.2 warning is AUTHOR-STATED; the two NVIDIA WSL notes in 6.3 are PUBLISHED (NVIDIA's CUDA on WSL guide); the remaining judgements are marked where they appear. To refresh the records, the read-only checkers already exist: `run_wsl_all.sh census` and `run_windows_all.ps1 -Mode facts` (they install nothing and say what is missing).

### 6.1 What the draft needs, and what the records show

| Need | Why | WSL2 (recorded) | Windows 11 (recorded) | Missing? |
|---|---|---|---|---|
| C++20 compiler | `CMakeLists.txt` requires C++20 for host and CUDA code; warnings on | g++ and gcc 15.2.0 (Ubuntu 15.2.0-16ubuntu1); also g++-13, clang 21 | MSVC 19.44.35229.0 in Visual Studio 18 Professional (toolset 14.44.35207), with `VsDevCmd.bat` | No |
| CMake 3.28 or newer | `cmake_minimum_required` 3.28; presets are version 6 | 4.2.3 (system); 4.4.4 also sits in the project venv | 4.4.3 (MEASURED 2026-10-23 01:09, the author's PowerShell; corrected 2026-10-26, it was missed on 2026-10-25) | No |
| Ninja | Every preset names the Ninja generator | 1.13.2 | 1.13.2 (MEASURED 2026-10-23 01:09; corrected 2026-10-26) | No |
| CUDA toolkit with `nvcc` for `sm_89`, C++20 | `PELIAS_CUDA` on; architecture 89 only | `cuda-toolkit-13-0`, `nvcc` 13.0.88 | CUDA 13.4, `nvcc` 13.4.92 (host compiler MSVC) | No |
| NVIDIA driver | The GPU, from both sides | WSL2 uses the Windows driver through `/dev/dxg`; libraries in `/usr/lib/wsl/lib`; the census shows kernel-mode version 617.42 and CUDA 13.4 | Driver 617.42 (32.0.16.1742) | No |
| Git | Clone, commit | 2.53.0 | 2.55.0.windows.5 | No |
| Python 3 (standard library only) | The two framework checks `check_module_deps.py` and `check_resources.py`; the tests register only if Python 3 is found | python3.11 and python3.13 (apt); 3.12.15 in the project's venv | **Not recorded** | **Windows: yes, or those two checks are skipped** |
| Kernel check tools | Rung 1 asks for a clean `compute-sanitizer` run; Nsight for tile tuning later | `compute-sanitizer` and `nsys` present; Nsight Compute packages installed 2026-10-17, but the 2026-10-23 census reported `ncu` as missing | Not checked by the facts run | WSL: path of `ncu`. Windows: unconfirmed |
| x64 Native Tools prompt | The `windows-release` preset must be run from it | Not applicable | `VsDevCmd.bat` present | No |
| Windows SDK (Winsock2) | Only for the server's sockets, later | Not applicable (POSIX sockets) | Expected with the Visual Studio C++ workload; not checked, version not recorded | Not for the framework; confirm before the server step |
| ccache (optional) | Faster rebuilds | 4.12.3 | "ccache not found" in the tier build | Optional |

**Result by the records:** on WSL2 nothing is missing for the host-only and CUDA builds of the framework. On Windows 11 the compiler, CMake, Ninja, CUDA 13.4 and Git are all recorded as working; the one gap is Python 3, and it only affects two checks.

### 6.2 Missing or unconfirmed (the short list)

1. **Windows: Python 3** is not recorded. Install it, or accept that the dependency check and the resource check do not run there.
2. **Windows: Windows SDK version, `compute-sanitizer` and Nsight** were not checked by the facts run (the Ninja version, 1.13.2, is in the toolchain record of 2026-10-23 01:09; corrected 2026-10-26). The next facts run can list them.
3. **WSL: `ncu`** was installed as a package on 2026-10-17 but the census reported it missing. Whether Nsight Compute works under WSL2 at all is UNVERIFIED: NVIDIA's WSL guide (PUBLISHED) lists the Nsight Systems command line and CUPTI tracing but does not name Nsight Compute. Settle it only when tile tuning starts.
4. **Disk space** on C: and D: is not recorded. The sizes that matter: the original bf16 safetensors 71.90 GB (66.97 GiB, supplied by the user; the trainer never downloads models); the cached `pelias_base_0step.gguf` that Stage A writes (stock Q4_K_M type map, no imatrix; its size is not recorded, and the served file today is 21,713,463,296 bytes, so plan for about the same, ESTIMATE); the export wants about 21 GB free on its destination. Files on the NTFS mount read at about 0.5 GB/s from WSL (engine measurement), so copy a file to the WSL disk once if a step reads it repeatedly.
5. **An unchanged build of llama.cpp `2ca15f540`** (the test oracle) is needed on each platform before step 6 of the ladder (the oracle harness: trainer T12 verify, engine E3) and the parity steps after it. WSL has the patched tier tree and the stock-run logs; the Windows tree on record is the patched one. Not needed for the framework build.
6. **The window and graphics backend for Dear ImGui** is not chosen (author decision). Not needed until the interface implementation is written; the header-only `ui` module builds without it.

### 6.3 Install list, names only (what a clean machine would need)

| Platform | Names |
|---|---|
| WSL2 (Ubuntu) | `build-essential`, `cmake` (3.28 or newer), `ninja-build`, `git`, `python3` with `python3-venv`, `ccache` (optional), `cuda-keyring` and `cuda-toolkit-13-0` (NVIDIA's WSL-Ubuntu repository). PUBLISHED, from NVIDIA's guide: do not install any NVIDIA Linux driver inside WSL 2, and do not choose the `cuda`, `cuda-<version>` or `cuda-drivers` meta-packages there |
| Windows 11 | NVIDIA driver 617.42 or newer; Visual Studio 18 Professional with the "Desktop development with C++" workload (MSVC v14.44 toolset and a Windows SDK); CMake 4.x; Ninja on the path; Git for Windows; CUDA Toolkit 13.4 (avoid 13.2, AUTHOR-STATED); Python 3 (the missing one); ccache (optional) |

### 6.4 Needed later, by stage (not for the draft)

| When | What |
|---|---|
| Stage A (T1b) | The original bf16 safetensors (71.90 GB) supplied by the user, and the disk space above |
| Tokenizer parity (T2), the oracle harness (step 6), parity rungs 2 and 3 (T7, T8) | The unchanged llama.cpp `2ca15f540` build per platform; on WSL the Python oracles (test-only, never imported by the product): PyTorch 2.14.1+cu130, transformers 5.17.0, safetensors 0.8.0, tokenizers 0.23.3, numpy 2.5.4, Jinja2 3.1.6, recorded in the WSL project venv (Python 3.12.15) on 2026-10-18 |
| Tile tuning, performance (T13) | Nsight Compute (`ncu`) and Nsight Systems; which platform runs them is UNVERIFIED (see 6.2, item 3) |
| Server (E10, step 12) | No new build tool is recorded as needed: sockets come from the operating system (Winsock2 on Windows, so confirm the Windows SDK then). Clients named in the engine spec: the Visual Studio 2026 Copilot "bring your own model" provider (untested when last recorded on 2026-10-21), the author's gauntlet and speed tools, a web UI, odysseus-ai and Ollama-style clients |
| Interface | Dear ImGui source (MIT, per the trainer notes) plus the window and graphics backend the author has not yet chosen |

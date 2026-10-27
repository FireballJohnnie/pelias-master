# PELIAS SYSTEM MASTERWORK

**One installable, ground-up program that probes the machine, takes a local GGUF or safetensors model, coaches and runs training or quantization, writes the resulting GGUF, and serves it on port 8062 through its own engine and server.**

| | |
|---|---|
| Document | `docs/PELIAS_SYSTEM_MASTERWORK.md`, pelias-master |
| Written | 2026-10-26 |
| Status | **Specification only. No product code exists.** Reviewed by Claude Fable on 2026-10-26 (step R0); its 19 findings are folded in (Section 18.1, C21 to C24; 18.2, N8 to N11; and the sections they name). The build starts on the author's go. |
| Copyright | Copyright (c) 2026 John Lipka and the Pelias contributors |
| Licence | Apache-2.0 (the same standard text as pelias-trainer and pelias-engine) |
| Built from | pelias-trainer (local head `953c9b1`), pelias-engine (local commit `ba74954`), the server runbook and handoff of 2026-10-24, the long master `PELIAS_PROPRIETARY_MASTER_SERVER.md` (the 2026-10-25 version, sha256 `c935a4a4…`, local only; the engine specification files cite `56634f43…`, the 2026-10-24 version), `PELIAS_SYSTEM_MASTER.md` (2026-10-25), the author's attachments of 2026-10-26 (listed in 19.2), the author's drives (working folder, proposals folder, FlareSolverr folder; exact paths in the git-ignored `profiles/local_only/LOCAL_SOURCES.md`), and published sources (19.3) |

---

## Why this exists, and why hasn't it been done before?

**How it started (AUTHOR-STATED, 2026-10-26 12:11):** "this started as an effort for me to better create my own llm, when i found this... accidentally. everything said we couldn't train an moe using my type of system, we found a way. then we saw the hardware ceiling limit when running that quantized llm, and we found a way to unify the bus to make it run even better from the way we found to train it... and then even more!"

**The question this project keeps running into:** a consumer PC's CPU, GPU and memory spend much of their time *waiting on each other*, and the waiting can be measured and cut. On the reference machine a CPU-to-GPU hand-off through the usual path took 58 to 94 µs, and through a resident doorbell 0.8 to 13 µs (MEASURED, 21.2). Speed, latency and timings get enormous attention, so why has the gap *between* the parts gone unmeasured for so long?

**What we presume is the reason, as unbelievable as it sounds** (the author's reading, AUTHOR-STATED 2026-10-26 13:28, with what the records show):

| Reason | In the author's words | What the records show |
|---|---|---|
| **An assumed limit** | "nobody owns the gap (assumed limit)" | CPU, GPU, operating-system, driver and framework teams each tune their own layer; the waiting sits between layers, where no team is measured on it. Benchmarks test parts alone (TFLOPS, GB/s), never the gap. The graphics specifications do not promise that the GPU sees a CPU write while a shader runs, so careful engineers stay out; the first P-G0b run (2026-10-26 13:29; repeats pending) found that it works on this machine in some memories and not others, which is only knowable by measuring |
| **Intentional ignorance** | "intentional ignorance" | The industry builds for data centres, where the model fits in the accelerator's own memory and the gap barely matters; on a consumer PC the standing answer is a bigger card. Nobody selling cards is paid to make a 12 GB card do 20 GB of work |
| **Time wasted finding it with the methods at hand** | "too much time to figure it out by using the methods at hand" | Tools hide copies and syncs on purpose, and stacked tools conflicted with one another into a choke point. The engine's own timers found zero CPU/GPU overlap and 4.6 to 5.0 ms of hand-off per token (MEASURED, E0) that no tool in the stack had reported |

The author's summary: "It was a mixture of assumed limit, intentional ignorance, and time waste to figure it out. People know about these things being out, but not in the way we're doing it. There aren't programs written for it."

**Developed independently (AUTHOR-STATED, 2026-10-26 13:36):** the work below was not a source for Pelias; it is listed as related work and for comparison. What Pelias did build on stays credited: llama.cpp (the patched tier build behind the measured tok/s, and the GGUF layouts read from its source; THIRD_PARTY_NOTICES.md) and the papers the trainer master cites (19.3).

**What already exists, stated plainly (prior work, checked 2026-10-26):** keeping hot MoE experts on the GPU and cold ones on the CPU is known and open source (KTransformers, HybriMoE, an RTX 4090 collaborative-inference paper with code, llama.cpp's `--n-cpu-moe`; PowerInfer did a hot/cold split for dense models); fine-tuning a 30B-class MoE on one consumer card is claimed by KTransformers (about 24 GB) and by USAF (12 GB, no benchmarks published; UNVERIFIED); persistent GPU kernels exist in data-centre research (S9 to S15). **What we did not find:** a per-token CPU-GPU doorbell inside a consumer LLM runtime, the hand-off gap measured across CUDA, Direct3D 12 and Vulkan, or one ground-up program that probes, trains, quantizes and serves with every number labelled by its source. Not finding it in a search is not proof it does not exist, so the claim stays: **the first open, ground-up system we know of** that does this. The first benchmark (16.4) includes the prior work above as comparisons.

---

## 0. How to read this

### 0.1 Labels

Every number and claim carries one of the project's eight labels.

| Label | Meaning |
|---|---|
| **MEASURED** | Run on the reference machine; the raw file or ledger exists |
| **READ-SRC** | Read from a file header or from source code (the served GGUF, the safetensors shards, llama.cpp `2ca15f540`) |
| **PUBLISHED** | Vendor documentation, a standard or a paper; the link is in Section 19 |
| **DERIVED** | Arithmetic from labelled inputs; the inputs are named |
| **ESTIMATE** | Judgement; always paired with the probe that replaces it |
| **UNVERIFIED** | A hypothesis or outside claim; not relied on until its probe passes |
| **DECISION** | A design choice with its reason; "(proposed)" means the author has not ruled |
| **AUTHOR-STATED** | Said by the author; no log attached |

The author's rule applies to every page: **no made-up numbers.** A value for hardware that was never measured is a PUBLISHED starting default with a source, and the start-up probe replaces it with a MEASURED value on the user's machine. A value nobody has measured is written *unmeasured*, together with the probe that will measure it.

### 0.2 What changed on 2026-10-26 (AUTHOR-STATED, the author's message of 11:44 and the answers of 11:52)

| # | Ruling | Effect on the earlier plan |
|---|---|---|
| A1 | Build the whole system now: "I know we have more than enough needed to build this from the ground up, and i want to see it happen" | The engine hold (2026-10-24 06:07, open decision O13) and the "top tier first" stop are lifted for this build. Recorded as AUTHOR-STATED; the order of work is Section 15 |
| A2 | One standalone installer for Linux (WSL first, also native Linux) and one for Windows 10/11 | Native Linux is no longer held (O12 answered). Windows 10 joins Windows 11; its driver limit is checked in 13.4 |
| A3 | Ground up, C++20/CUDA; no llama.cpp or other outside work inside; llama.cpp only to compare against | Confirms D37 and sync R2. The oracle stays external and test-only |
| A4 | The system detects CPU, GPU, RAM, drives and anything else relevant by itself, using published values for hardware never tested here | Section 4. Answer to question 1 of 11:52: "if it is possible, i want the system to account for that itself" (GPU vendor included) |
| A5 | The interface is a web page served by Pelias itself (answer 2: "yes") | Dear ImGui is not used in the product (open decision O9 closed) |
| A6 | Encrypted web connections may use what the operating system provides (answer 3: "yes") | Section 3.2, row B4 |
| A7 | Search the author's proposals folder and use everything attached (answer 4) | Section 19.2 |
| A8 | Port **8062** for `pelias-server`: inside the 80xx API range, avoiding 8012 (the llama.cpp interim server), 62 = P16 + E5 + L12 + I9 + A1 + S19 | Every earlier "8012" for the Pelias server becomes 8062; 8012 stays the llama.cpp comparison server |
| A9 | Web tools of its own; SearXNG may be a base; FlareSolverr is the one piece of outside code allowed if it cannot be recreated | Section 12 |
| A10 | Import files made for the author's own machine, model and parameters; if the system later finds a higher ceiling, it recommends a better choice and waits for the user's OK | Section 7.6 and `profiles/` |
| A11 | Everything lives in the pelias-master directory, prepared for git like pelias-trainer and pelias-engine | This repository |
| A12 | A pause for Claude Fable to review the markdowns before the build is queued; its findings are folded in, then the build continues | Section 15, step R0 |
| A13 | The first build is benchmarked: the Pelias 4-bit file against the official and Unsloth 4-bit files of the same model and against the unquantized model; speed against the ceiling (2026-10-26 12:05) | 16.4; step 16 |
| A14 | Future hypotheses beyond LLMs (2026-10-26 12:19) | Section 21 |

### 0.3 Standing rules (unchanged, in force for every step)

1. **Ground up** (D37). No llama.cpp, ggml, libllama, PyTorch, PEFT, Triton, Unsloth, or third-party JSON, HTTP, regex or template code in the product. The exceptions are only those in Section 3.2.
2. **Never truncate** a training row. A row longer than the window is a hard failure with its row id.
3. **Hard VRAM ceiling**, measured at start (D26, D28): the smaller of CUDA free and NVML free after the GPU context exists, minus a margin. 10.7 GiB on the reference machine.
4. **No allocation inside a hot loop** (training step, decode token).
5. **No model downloads.** The user gives a local path.
6. **Provenance line** at the top of every source file; nothing AGPL or LGPL is copied.
7. **Privacy.** Dataset text, persona and identity strings, the author's private equations, local profile paths, serials and tokens never enter a repository file or a log. Logs carry counts, ids, lengths and pass or fail.
8. **The user runs every program that touches the GPU.** Pushes happen only on the author's word.

---

## 1. The vision and the requirements

The author's message of 2026-10-26 11:44, broken into numbered requirements so every later section can point back to one. Words in quotes are the author's.

| # | Requirement | Where it is met |
|---|---|---|
| R1 | One "masterwork, ground up, finalized markdown" for the Pelias-System, built from all the work on pelias-trainer, pelias-engine and pelias-server: the unified bus, training of MoE and non-MoE models, and the server | This document |
| R2 | A standalone installer for Linux (WSL focused, also without WSL) and one for Windows 10/11 | 13 |
| R3 | Wholly ground up in C++20 and CUDA, unless a faster way exists to unify everything in an installable program | 3.1 (the answer: C++20 with CUDA for NVIDIA; a second ground-up GPU backend in Vulkan compute for other vendors) |
| R4 | No llama.cpp or other outside work inside; llama.cpp may be used to benchmark that Pelias behaves the same; no dependency beyond what the project creates | 3 |
| R5 | Follow the master scaffold; fill gaps from the server, engine and trainer work | 14, 18 |
| R6 | Light probes detect the CPU and activate the right paths; unknown CPUs get published values | 4.2 |
| R7 | The same for the GPU | 4.3 |
| R8 | The same for memory (DDR4 or DDR5, channels, speed) | 4.4 |
| R9 | The same for drives (NVMe, SATA, others) | 4.5 |
| R10 | The same for anything else relevant, and for MoE and non-MoE models, quantized or not | 4.6, 5 |
| R11 | The user gives a path to a GGUF or safetensors model pulled from Hugging Face | 5.1, 6.1 |
| R12 | Probes analyse the system and the model and choose the best settings | 4.1, 6.2 |
| R13 | The user sees what the probes found and which settings they activated, can adjust them, and can add their own in an advanced menu | 6.3 |
| R14 | Import all parameters from json, txt, log, csv or md | 7.3 |
| R15 | Coaching for the user's own training parameters, accounting for the agent identifiers used by IDEs, Odysseus-ai, web UIs and similar programs | 6.5, 8.5 |
| R16 | The user enters their programs in a CSV-like table, or imports csv, log, py, txt or md; the system builds the parameters and keeps the jsonl for training | 6.5, 7.3, 8.4 |
| R17 | The system decides whether the plan is possible and makes a final recommendation | 6.6 |
| R18 | Three paths: quantize a full model or its safetensors; train a quantized model; train and quantize a full model | 6.7, 8, 9 |
| R19 | The probes find the VRAM that can hold the model. For MoE, keep all active tensors in VRAM during training "until all are trained"; for non-MoE, the best case for that model | 8.2, 8.3 |
| R20 | "Near 0 drift", so the engine reads the result at near-identical quality to the unquantized model | 8.6 |
| R21 | The output GGUF (quantized, trained, or trained and quantized) is written | 9.4 |
| R22 | The server loads the GGUF on port 8062 | 11 |
| R23 | Engine and server act as one: best isolation and inference method, the user's hardware used as "a near perfect unified bus system" | 10 |
| R24 | Resizable BAR built into the Windows and Linux builds so it can be tested | 10.6 |
| R25 | Near-ceiling speed | 10.7 |
| R26 | Its own web tools for the model, ground up; SearXNG may be a base; FlareSolverr is the one outside piece allowed if it cannot be recreated | 12 |
| R27 | Anything not listed: decide the best course from all the information | 18 |
| R28 | Import files tailored to the author's machine, model and existing parameters (newest versions on the drive) | 7.5, `profiles/` |
| R29 | If the system finds a higher ceiling than an imported file assumes, it recommends a choice for those parameters, which the user must accept before continuing | 7.6 |
| R30 | Everything in pelias-master, with logs and build notes, ready for git | 14.4, `logs/`, `docs/` |
| R31 | A Fable review before the build is queued | 15 |
| R32 | Benchmark the first build against published 4-bit files and the unquantized model, and its speed against the ceiling | 16.4 |
| R33 | Record as future hypotheses: the bus unification outside LLMs, kept tuned by a tiny CPU model or by code alone; its use across fields with on-paper gains; consumer PCs doing data-centre work, less power, longer hardware life; gaming for players and developers; ballpark end-to-end gains | 21 |

---

## 2. The product

### 2.1 One program, one installer per platform

The user installs one program, `pelias`. It has a command line and a local web interface; both drive the same core. The pieces:

| Piece | What it does | Built from |
|---|---|---|
| **probe** | Light start-up probes of CPU, GPU, RAM, drives, OS and model; writes a machine profile and a model key | Trainer U1, U2, U4, U6; the Wave 0 probe program; engine E11 |
| **plan** | Turns probe results and the chosen path into a memory plan that fits the ceiling, and two to four options on a plan card | Trainer `plan/roofline_solver`, U3; engine E9 planner |
| **intake / coach** | Reads the user's parameters and data (sheets, imports, clients), checks them, and coaches the user to a training set | Trainer U7, U8, U9, U10 |
| **quantize** | The project's own bit-exact block quantizers; safetensors to GGUF; re-quantization to the variant chosen | Trainer Stage A (T1b, D33); `quant/` |
| **train** | Layer-major LoRA training with the core trained through the exact quantizer (Mode Q) | Trainer T2 to T14 |
| **export** | Writes the GGUF (merged core, optional adapter, layout manifest, hot set) and proves frozen bytes unchanged | Trainer T12; `io/gguf_writer`, `gguf_splice` |
| **engine** | Token-by-token inference: GPU spine and hot bank, CPU cold experts, doorbell handoff, dual bus | Engine E2 to E9 |
| **server** | HTTP/1.1 and server-sent events on 127.0.0.1:8062, OpenAI-compatible API, tool calls, reasoning split | Engine E10, the server runbook |
| **web UI** | The local interface, served by the server; lithic style, the Pelias logo, the spear progress bar | Trainer `ui/` header (now a web page, A5) |
| **web tools** | Search, fetch and read for the model; challenge pages through an optional FlareSolverr process | The author's earlier web-tools script; Section 12 |

### 2.2 How the pieces meet

```text
            ┌──────────── probe ────────────┐
 machine ──►│ CPU GPU RAM drives OS ReBAR  │──► machine profile (MEASURED, cached by machine_state_sha)
 model path►│ GGUF / safetensors header     │──► model key (SHA-256 + ModelSpec)
            └──────────────┬───────────────┘
                           ▼
 imported profile ──► plan (fit + worth) ──► plan card (2–4 options, one marked default) ──► user OK
                           │
          ┌────────────────┼──────────────────────┐
          ▼                ▼                      ▼
   path Q: quantize   path T: train quantized   path TQ: train full model, then quantize
          │                │                      │
          └────────► export GGUF + layout_manifest + hot set ◄┘
                           │
                           ▼
                engine + server on 127.0.0.1:8062 ──► clients (IDE, Odysseus-ai, web UI, web tools)
```

---

## 3. The ground-up boundary

### 3.1 Language and the GPU question (answer to R3)

**C++20 for everything on the host; CUDA C++ for NVIDIA GPUs.** DECISION. The measured floors (Section 17) come from C++ and CUDA programs, and the doorbell handoff, zero-copy reads and graph replay that the unified bus needs are CUDA features that were measured on this machine. No faster language exists for the host code; the speed comes from the kernels and the loop order, not the language.

**For other GPU vendors (A4): a second backend in Vulkan compute.** DECISION (proposed). The reason: one Vulkan backend covers AMD, Intel and older NVIDIA cards on both Windows and Linux, and the Vulkan loader ships with every GPU driver, so the installer adds nothing. HIP (AMD ROCm) and Level Zero (Intel) are vendor runtimes that the user would have to install; they stay optional later backends. What Vulkan cannot match on every card is decided by probe, not assumed: the five hardware requirements H1 to H5 of the universal design (zero-copy host reads, microsecond flag visibility, a resident waiting kernel, sub-block dot products, async copy engines; 4.7) are tested on each machine and a missing one switches to its fallback (for example the doorbell falls back to stream synchronization, which was measured at 94.4 µs per layer against 13.34 µs; MEASURED).

**Two costs of the Vulkan backend, stated plainly** (Fable review, 2026-10-26): (1) Vulkan runs SPIR-V, so the build needs a shader compiler. DECISION (proposed): glslang (Khronos), used at build time only, never shipped, listed in `THIRD_PARTY_NOTICES.md`; it was built from source and used successfully for the graphics probe on 2026-10-26. (2) It is a second full set of kernels (GEMM, attention, DeltaNet, quantizers, backward). It is therefore scheduled after the first benchmark (15.2, step 16). Inside WSL2 there is no NVIDIA Vulkan path (only an experimental Mesa route over D3D12), so the Vulkan backend and the ReBAR route A are Windows-native and native-Linux features.

**CPU-only mode** is a later step (15.2, step 14): the cold-expert AVX kernels do not cover the spine (attention, DeltaNet, norms), so CPU-only needs its own spine kernels, which are not in the first tree. Its speed is unmeasured; the plan card will say so.

### 3.2 What the product may use that is not ours

| # | Item | Why it is allowed | How it is loaded |
|---|---|---|---|
| B1 | The operating system (POSIX on Linux and WSL2; Win32 and Winsock2 on Windows) | It is the platform | Only inside `src/os/` and `src/hal/host_alloc_*` (boundary check BC6) |
| B2 | The GPU driver and its runtime: CUDA driver API and runtime for NVIDIA; the Vulkan loader for the Vulkan backend; NVML for telemetry | Nothing can reach the GPU without them | The CUDA runtime is linked statically into the program (DECISION (proposed); the CUDA toolkit licence permits redistributing the static runtime, to be checked against the licence text before release, UNVERIFIED); NVML and the Vulkan loader are opened at run time (`dlopen` / `LoadLibrary`) so the program starts without them |
| B3 | C++20 standard library and CPU intrinsics | Part of the language and the compiler | Compiler |
| B4 | **Encrypted connections (HTTPS) for the web tools** | Writing cryptography from scratch is a security risk; ruled A6 | Windows: the system's Schannel / WinHTTP. Linux and WSL: the system's own OpenSSL library, opened at run time; if it is missing, web tools say so and the rest of the program works |
| B5 | **FlareSolverr** (MIT licence, version 3.5.2 found on the drive) | Ruled A9: the one outside piece allowed if it cannot be recreated | An optional separate process on 127.0.0.1:8191, never linked; Section 12.4 |
| B6 | Unicode character tables | Public standard data for the tokenizer's NFC normalization | Compiled into static tables by our own generator; the Unicode data licence notice goes in `THIRD_PARTY_NOTICES.md` |
| B7 | Build-time tools and headers (compilers, CMake, Ninja, glslang for SPIR-V, Khronos Vulkan headers, Microsoft DirectX headers) | Needed to build, not to run | Never shipped; listed in `THIRD_PARTY_NOTICES.md` with their licences |

**Licence note:** the quantizers and type rules are re-expressed from ggml's MIT-licensed source so that the bytes match (rule 6). No file is copied, but because the arithmetic follows that source exactly, `THIRD_PARTY_NOTICES.md` carries the llama.cpp/ggml MIT notice.

**Never in the product:** llama.cpp, ggml, libllama, `common`, PyTorch, transformers, PEFT, Triton, Unsloth, Python at run time, any JSON, HTTP, regex or template library, Dear ImGui, SearXNG code (AGPL-3.0 licence: used as a design reference only, never copied), cuBLAS (the measured route is our own GEMM; cuBLASLt stays a test-only comparison, D-row in 18.3).

### 3.3 Test-only material (never linked, never started by the product)

- llama.cpp `2ca15f540` (b11399), built unchanged, in `tests/oracle/`: logits, KL, top-1 and kernel parity references.
- A test-only PyTorch slice in `tools/oracle_torch/` for backward gradients.
- An fp64 CPU reference of every operation in `tests/ref/`.

Release gate PG10 proves the product runs with `tests/oracle/` and `third_party/` deleted.

### 3.4 Enforcement (boundary checks, from the trainer and engine plans)

| Check | What it proves |
|---|---|
| BC1 | The module graph is acyclic; no product module includes from `tests/` or `third_party/` |
| BC2 | No symbol from ggml, llama or any forbidden library in any product binary's link map |
| BC3 | Product files include only repository, standard-library, GPU-runtime, intrinsics and (inside `os/`) OS headers |
| BC4 | Line 1 of every source file is a provenance line; the copyright line follows |
| BC5 | The product runs with the oracles removed (PG10) |
| BC6 | Operating-system headers only in `src/os/` and `src/hal/host_alloc_*` |
| BC7 (new) | The installer payload holds no file outside the product list (13.5) |


---

## 4. Hardware detection (R6 to R10, R12)

### 4.1 How detection works

1. **Read what the machine reports exactly** (no guessing needed): CPU vendor, family, model, core and thread counts, cache sizes and instruction sets from CPUID; GPU vendor, name, compute capability or Vulkan device properties, VRAM, SM or compute-unit count, PCIe link width and generation, BAR size, from the GPU runtime and NVML (or the Vulkan and PCI interfaces); RAM total, module type and speed from the OS (SMBIOS tables on both platforms); drive model, bus and link from the OS; OS, build, WSL version and kernel.
2. **Look up starting defaults** for numbers that are only known by measurement (bandwidths, latencies, the free-VRAM budget) in the published tables of 4.2 to 4.5. These are labelled PUBLISHED or DERIVED and are only used until step 3 finishes.
3. **Measure with light probes** (the quick pass, U1): free VRAM after the GPU context, pinned PCIe rate, whether copies need the flush (D27), a short tensor-core burst, host RAM free, and the read rate of the model file's drive. Slow probes (30-minute soak, display timeout, pinned limit) run once per machine and are cached under `machine_state_sha` (GPU, driver, OS and clock state). The quick pass's own running time is unmeasured; it is measured when built.
4. **Plan from the measured values.** The planner always takes the lower of the quick and the cached value; the VRAM governor stays the hard stop.

Every value on the detected-settings screen (6.3) shows where it came from: MEASURED (this machine, date), PUBLISHED default (source) or USER-FORCED (Expert mode).

### 4.2 CPU

| What is read | How it is used |
|---|---|
| Instruction sets (CPUID): AVX2, FMA, F16C, AVX-512 (F, BW, VNNI, BF16), AVX-VNNI, AMX, AVX10 | Picks the cold-expert and CPU-only kernel path: `avx512` > `avx_vnni` > `avx2` > portable C++. The kernels for every path exist in the binary; the choice is made once at start. Exact by CPUID, so no published default is needed |
| Physical cores, SMT siblings, hybrid core classes (Intel P/E), CCDs and the V-Cache CCD (AMD), NUMA nodes | Thread pinning policy `cpu.pin_mode`; decode thread on its own core; cold workers one per physical core (`physical_1t`); E-cores excluded by default; on multi-socket machines the pinned store and workers bind to the node nearest the GPU |
| L2 and L3 sizes | Cold-row tile sizes; the non-temporal prefetch switch is offered only where a large victim L3 exists (an A/B test, E-A3, UNVERIFIED) |

**Published starting points for CPUs never tested here** (the probe replaces each):

| Family | Vector path | Source | Label |
|---|---|---|---|
| AMD Zen 2, Zen 3 (incl. Ryzen 7 5800X3D) | AVX2 + FMA + F16C; no AVX-512 | Reference machine census; AMD Family 19h guide | MEASURED (reference) / PUBLISHED |
| AMD Ryzen 7000, 8000, 9000 desktop; Ryzen AI 300/400 parts | AVX-512 | List of CPUs with AVX-512 (19.3, S5) | PUBLISHED |
| Intel 11th-gen desktop (Rocket Lake), Core X-series 2017–2019 | AVX-512 | S5 | PUBLISHED |
| Any other CPU | Whatever CPUID reports | CPUID | exact |

Reference measurements (MEASURED, Wave 0): host DRAM read 37.1 / 37.9 GB/s with 1 thread and 43.7 to 45.81 / 44.8 GB/s with 4 threads (WSL2 / Windows, P-H2); 46.4 GB/s for the CPU alone with 4 readers in the Windows P-H6 run log; 5 to 7 decode threads beat 10 on the stock build. The default for an unknown CPU is `physical cores − 1` decode workers (the decode thread keeps one core), replaced by the E4 thread sweep when it runs (E-A1). DECISION (proposed).

### 4.3 GPU

| Vendor and family | Backend | What is decided by probe |
|---|---|---|
| NVIDIA, compute capability 7.5 to 12.x (RTX 20, 30, 40, 50 series and their workstation cards; 19.3 S1) | CUDA (the reference backend) | Tile sizes per capability; tensor-core data types (bf16 and int8 from 8.0 up, fp8 from 8.9 up); resident doorbell kernel (H2, H3); graph replay; flush need (WDDM) |
| AMD Radeon RX 7000 (RDNA3, gfx1100 to gfx1102) and RX 9000 (RDNA4, gfx1200, gfx1201) | Vulkan compute (proposed; Windows-native and native Linux only); HIP later | Whether host-visible memory reads and system-scope atomics are fast enough for the doorbell; otherwise stream-sync fallback |
| Intel Arc (Alchemist, Battlemage) | Vulkan compute (proposed) | Same as AMD |
| No usable GPU, or a GPU below the minimum | CPU only | Everything on the CPU kernels |

**Published starting defaults for GPUs never tested here** (VRAM is read exactly; bandwidth is a default until P-H2 measures it):

| Card | VRAM | Bus | Published bandwidth | Source |
|---|---|---|---|---|
| GeForce RTX 5090 | 32 GB GDDR7 | 512-bit | 1,792 GB/s | S2 |
| GeForce RTX 5080 | 16 GB GDDR7 | 256-bit | 960 GB/s | S2 |
| GeForce RTX 5070 Ti | 16 GB GDDR7 | 256-bit | 896 GB/s | S2 |
| GeForce RTX 5070 | 12 GB GDDR7 | 192-bit | 672 GB/s | S2 |
| GeForce RTX 5060 Ti | 16 or 8 GB GDDR7 | 128-bit | 448 GB/s | S2 |
| GeForce RTX 5060 | 8 GB GDDR7 | 128-bit | 448 GB/s | S2 |
| GeForce RTX 5050 | 8 GB GDDR6 | 128-bit | 320 GB/s | S2 |
| GeForce RTX 4070 (reference) | 12 GB | 192-bit | 504 GB/s; **MEASURED 523.3 / 534.0 GB/s read** with the memory overclock | Ada whitepaper (records); P-H2 |
| Other RTX 40 cards | as read | as read | not on the vendor page fetched; the probe measures it | S2 (bandwidth not listed) |
| Intel Arc B580 / B570 | 12 GB / 10 GB | 192 / 160-bit | 456 / 380 GB/s; PCIe 4.0 x8 | S6 |
| AMD RX 7000 / 9000 | as read | as read | not fetched; the probe measures it | S3, S4 (support lists only) |

The compute-capability families are PUBLISHED (S1): 12.0 RTX 50 series and RTX PRO Blackwell; 8.9 RTX 40 series and RTX Ada; 8.6 RTX 30 series and RTX A-series; 7.5 RTX 20 series and Quadro RTX. AMD's own runtime supports RX 7000 and RX 9000 on Windows (HIP SDK 7.2) and on Linux (ROCm 7.14, Ubuntu 24.04.4 and 22.04.5 among others); RDNA2 (RX 6000) is not supported on Windows (S3, S4). The Vulkan backend does not depend on those lists.

**The VRAM ceiling on any card** is the rule of D28, not a table: the smaller of CUDA (or Vulkan) free and NVML free after the context exists, minus the margin.

**Serving working set in the planner** (corrected 2026-10-26 19:0x): the 2.50 GiB of 8.8 is the training working set; the serving planner uses 1.00 GiB, a DERIVED upper bound from the 7.5 GiB-bank peak of 17.4 on the process-only basis below (10.69 GiB whole card − 914 MiB Windows − 7.5 GiB bank − 1,326 MiB of weights on the card = 1,026 MiB, rounded up), until P-E4 measures the engine's own peak.

**Which reading the ceiling is** (decision 7, ruled 2026-10-24; trainer master 16.3.1): the governor enforces the **process-only** reading (what this process may allocate after the context exists; 10.7 GiB on the reference). The engine's recorded peaks (10.40 to 10.44 GiB, 10.67 to 10.69 GiB, 11.0 to 11.19 GiB) are **whole-card** readings that include about 0.9 to 1.0 GiB held by Windows (MEASURED, P-H8: 914 / 1,031 MiB before the process starts). Until P-E4 re-measures the Pelias engine's own peaks, the whole-card figures stay the stricter planning figures. The difference matters: read as process-only, the same peaks leave about 1.0 to 1.3 GiB of room instead of 0.0 to 0.3 GiB (DERIVED, trainer master 16.3.1). Every plan, profile and recommendation states which basis it uses. The margin on the reference machine is 0.13 GiB (10.83 − 0.13 = 10.7 GiB, AUTHOR-STATED ruling with MEASURED inputs). On an unknown card the margin starts at the same 0.13 GiB and the disturbance test (desktop activity while a plan is held) sets it; until then the plan card says "margin not yet measured on this machine".

### 4.4 Memory (DDR4, DDR5, channels)

| What is read | Use |
|---|---|
| Module type, rated speed (MT/s), count, populated channels (SMBIOS) | Starting default for the host-read rate |
| Free RAM, page-lock limit (`RLIMIT_MEMLOCK` on Linux), WSL memory cap (`.wslconfig`), page file or swap | The pinned pool size and the fallback chain (pinned → registered → pageable staging) |

**Starting default** (DERIVED): peak = MT/s × 8 bytes × channels. A DDR4 or DDR5 channel is 64 bits wide either way (a DDR5 module splits it into two 32-bit sub-channels, which does not change the product). Reference: 3,200 × 8 × 2 = 51.2 GB/s (PUBLISHED in the records), against a measured 43.7 to 45.9 GB/s CPU read (MEASURED), so the reference reaches 85 % to 90 % of peak (DERIVED). For another machine the default host-read rate is the peak times 0.85, labelled DERIVED from the reference ratio, and P-H2 replaces it. Examples (DERIVED): DDR5-6000 dual channel 96.0 GB/s peak; DDR4-3600 dual channel 57.6 GB/s; DDR5-5600 quad channel 179.2 GB/s.

Why it matters: the cold experts are read from RAM every token. The engine's "dual bus" finding (CPU 26.56 + GPU zero-copy 21.06 = 47.62 GB/s together, 93.0 % of the 51.2 GB/s peak, MEASURED) says the two readers share one DRAM. On a machine with more DRAM bandwidth than PCIe bandwidth, the GPU's zero-copy share can grow; the cold split `gpu.cold_share_fraction` is therefore probe-derived (the reference value 0.44 = 21.06 / 47.62, DERIVED), never a constant.

### 4.5 Drives

| What is read | Use |
|---|---|
| Drive bus (NVMe, SATA, USB, network), PCIe generation and width of an NVMe link, file system, whether the path is a WSL view of a Windows drive | A plan-card warning and the load plan |
| Sequential read rate of the model file (P-M4, a short timed read) | Load-time estimate; "copy the model to a faster place" proposal |

**Starting defaults** (DERIVED from published link rates; the probe replaces them):

| Drive | Link ceiling | Derivation |
|---|---|---|
| NVMe PCIe 3.0 x4 | 3.94 GB/s | 8 GT/s × 4 lanes × 128/130 ÷ 8 |
| NVMe PCIe 4.0 x4 | 7.88 GB/s | 16 GT/s × 4 × 128/130 ÷ 8 |
| NVMe PCIe 5.0 x4 | 15.75 GB/s | 32 GT/s × 4 × 128/130 ÷ 8 |
| SATA III SSD | 0.60 GB/s | 6 Gb/s × 8/10 ÷ 8 |
| Hard disk, USB, network share | none set | read by the probe only; the plan card shows "measured on first load" |

**Reference machine** (MEASURED): the 20.22 GiB model read from the Windows drives inside WSL2 at 0.17 to 0.52 GB/s (87 to 119 s); from the WSL2 ext4 disk at 2.99 to 3.06 GB/s cold (7.1 s) and 0.25 s warm. The D: drive (WD SN580) is linked at PCIe 3.0 x2, about 1.97 GB/s (DERIVED). **Rule (DECISION):** on WSL2, a model path under `/mnt/<drive>` raises a warning and a proposal to copy the file once to the WSL disk or read it once into the pinned store; the warning threshold `io.warn_slow_mount_gbps` defaults to 0.80 GB/s (from the parameter dictionary, which places it above the measured 0.17 to 0.52 GB/s mount rates).

### 4.6 Operating system and everything else relevant

| Item | Read from | Use |
|---|---|---|
| OS: Windows 10, Windows 11 (build), WSL2 (version, kernel, `.wslconfig` limits), native Linux (kernel, distribution) | OS | Which host-memory backend (`host_alloc_win32`, `host_alloc_wsl`, `host_alloc_linux`), the timer resolution call on Windows, the WSL memory cap |
| Hardware GPU scheduling (HAGS), display timeout (TDR), whether the GPU drives the display | Registry (Windows); the probe from WSL asks the Windows side | Resident-kernel time limit; the display-timeout probe P-H13 (reference: 2, 5 and 30 s kernels passed on both platforms, MEASURED) |
| Resizable BAR and BAR1 size | PCI configuration and NVML | Section 10.6 |
| Transparent huge pages, large-page privilege | `/sys/kernel/mm/transparent_hugepage`, Windows `SeLockMemoryPrivilege` | `os.large_pages` stays off until P-E11 shows a gain of at least 5 % end to end (dictionary rule) |
| Power plan, clocks, temperature | OS and NVML | The watchdog uses the SM clock and temperature, because NVML's throttle field read "idle" under full load on both platforms (MEASURED, P-H4) |
| Other programs holding VRAM | NVML and the Windows performance counters | The close-these-apps proposal (U6); the tool never closes a program itself |

### 4.7 The five hardware requirements of the unified bus (any vendor)

| # | Requirement | NVIDIA (reference) | If missing |
|---|---|---|---|
| H1 | The GPU reads pinned host memory directly (zero-copy) | `cudaHostAllocMapped`; 26.3 GB/s MEASURED | Cold experts on the CPU only |
| H2 | A CPU write becomes visible to a running GPU kernel within microseconds | 3.36 µs flag, MEASURED | `stream_sync` handoff (94.4 µs, MEASURED) |
| H3 | A kernel can wait resident for that flag | Doorbell 13.34 µs median, 18.71 µs p99, MEASURED | `stream_sync` |
| H4 | Sub-block integer or bf16 dot products | Tensor cores: int8 259.5 TOPS, bf16 65.2 TFLOPS sustained, MEASURED | Plain FMA kernels (slower; speed unmeasured) |
| H5 | Separate copy engines for overlap | 5 engines; 0.92 overlap with the flush, MEASURED | Serial copies; the planner prices them |

The universal specification also lists unified-memory APUs (a GPU sharing the CPU's RAM): there the hot bank size becomes 0 and every expert is read in place. UNVERIFIED on any machine here; offered only when H1 is measured on such a device.

---

## 5. Models (R10, R11)

### 5.1 What the user gives

A local path to either:
- a **GGUF file** (any quantization in 5.3), or
- a **safetensors folder** (the original Hugging Face layout: `config.json`, `tokenizer.json`, the chat template, and one or more `.safetensors` shards).

The program never downloads (rule 5). If the folder came from Hugging Face, nothing more is needed. Reading uses the project's own memory-mapped readers (`io/gguf_reader`, `io/safetensors`); a damaged file is refused with the exact tensor name and byte offset (gate PG2: 1,500 mutated files rejected without a crash).

### 5.2 What is read from the model, and the model key

| Read | From | Use |
|---|---|---|
| Architecture name, layer count, layer kinds, attention and recurrent sizes, expert count, experts per token, expert width, shared experts, vocabulary, context length | GGUF metadata or `config.json` | `ModelSpec`, the architecture plug-in, the memory plan |
| Every tensor's name, shape, type, offset | GGUF tensor table or safetensors header | Byte map; which tensors are experts, spine, embedding, head |
| Tokenizer and chat template | GGUF `tokenizer.*` keys or `tokenizer.json` and the template file | `tok/` (own BPE and template renderer) |
| SHA-256 of the file | Read once, cached by path, size and modification time | The **model key** (U4): per-model facts (hot sets, hit rates, expert bytes) are stored under it |

**Two reference files** (not to be confused): (1) the **served file**, the author's fine-tuned export (`test-model` in the engine ledger), on which every hot set, hit rate, peak and the float spread were measured; (2) the **clean base** that Stage A writes from the bf16 shards (`pelias_base_0step.gguf`), which the benchmark (16.4) and P-X6 use. The hot sets and the float spread transfer from (1) to (2) only as an assumption: step 10 re-profiles the hot set on the clean base and P-Q4b re-measures the spread. Both files are identified by their model keys.

**Reference model** (READ-SRC, the served file): Qwen3.6-35B-A3B, architecture `qwen35moe`, Q4_K_M without imatrix, 753 tensors, 21,713,463,296 bytes (20.22 GiB); 40 trunk layers (30 Gated DeltaNet, 10 full attention at 3, 7, …, 39) plus the multi-token-prediction block `blk.40`; 256 experts, 8 per token, expert width 512, one shared expert; vocabulary 248,320; context 262,144. The bf16 source: 26 shards, 1,045 tensors, 71,903,645,408 bytes (MEASURED from the shard headers). The full byte map is in Section 17.

### 5.3 Quantization formats (READ-SRC from `ggml/src/ggml-common.h` at commit `2ca15f5404760548c39e7b92bd43116a09414a1a`, sizes DERIVED)

Block sizes are read from the reference header's size checks; bits per weight = bytes × 8 ÷ elements.

| Type | Bytes per block | Elements | Bits per weight |
|---|---|---|---|
| Q4_0 | 18 | 32 | 4.5 |
| Q4_1 | 20 | 32 | 5.0 |
| Q5_0 | 22 | 32 | 5.5 |
| Q5_1 | 24 | 32 | 6.0 |
| Q8_0 | 34 | 32 | 8.5 |
| Q2_K | 84 | 256 | 2.625 |
| Q3_K | 110 | 256 | 3.4375 |
| Q4_K | 144 | 256 | 4.5 |
| Q5_K | 176 | 256 | 5.5 |
| Q6_K | 210 | 256 | 6.5625 |
| IQ1_S / IQ1_M | 50 / 56 | 256 | 1.5625 / 1.75 |
| IQ2_XXS / IQ2_XS / IQ2_S | 66 / 74 / 82 | 256 | 2.0625 / 2.3125 / 2.5625 |
| IQ3_XXS / IQ3_S | 98 / 110 | 256 | 3.0625 / 3.4375 |
| IQ4_NL | 18 | 32 | 4.5 |
| IQ4_XS | 136 | 256 | 4.25 |
| MXFP4 | 17 | 32 | 4.25 |
| TQ1_0 / TQ2_0 | 54 / 66 | 256 | 1.6875 / 2.0625 |
| Activation types: q8_1 (GPU), Q8_K (CPU) | 36 / 292 | 32 / 256 | 9.0 / 9.125 |

"Q4_K_M", "Q5_K_M" and similar names are **mixes**, not single types: a rule assigns a type per tensor (for the reference file: experts gate and up Q4_K; expert down Q6_K in 21 blocks and Q4_K in 20; `lm_head` Q6_K; embedding Q4_K; routers F32; READ-SRC). Pelias writes these mixes with its own rule tables; the stock rule for each mix is read from the reference source as a specification and re-expressed in original code (the same way as the quantizers, rule 6).

**Build order of format support** (DECISION (proposed)): Q4_K, Q6_K, Q8_0, F16, BF16, F32 first (the reference file and its trained export need only these, plus q8_1 and Q8_K for activations); then Q5_K, Q3_K, Q2_K, Q4_0 to Q5_1 (quantize targets users ask for); IQ and TQ types and MXFP4 last (they need importance data or special kernels). Each format enters only after gate P-E6 (bit-identical to the reference quantizer on every tensor) passes for it. **2-bit and 3-bit support is requested by the author (2026-10-26 19:40, O19):** Q2_K and Q3_K (no importance data needed) come first, then the 2- and 3-bit IQ types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), which need importance data (an imatrix) to quantize well. A GGUF using a format not yet supported is refused at load with the format named.

Published quality notes for choosing a variant (shown to the user as guidance, not as a promise for their model): a 2026 guide reports perplexity increases on a 7B model of +0.0004 (Q8_0), +0.0044 (Q6_K), +0.0142 (Q5_K_M), +0.0535 (Q4_K_M), +0.2437 (Q3_K_M), +0.8698 (Q2_K) (S7, PUBLISHED, secondary source); a paper on Llama-3.1-8B reports +0.24 (Q4_K_M), +0.03 (Q6_K), +0.01 (Q8_0) (arXiv 2601.14277, in the trainer master). The two disagree in size; the user's own model is measured by the K-gates (8.6).

### 5.4 MoE and non-MoE ("dense") models

| | MoE (reference: `qwen35moe`) | Dense |
|---|---|---|
| What is always on the GPU | Spine (attention or DeltaNet, norms, router, shared expert), embedding, `lm_head`, the hot bank | As many whole layers as fit under the ceiling |
| What streams or runs elsewhere | Cold experts: read from pinned RAM by the CPU (AVX kernels) and, optionally, by the GPU over PCIe (zero-copy) | The remaining layers: streamed layer by layer over PCIe for training; for inference either streamed or run on the CPU, chosen by the planner from measured rates. A dense model reads every non-resident byte per token, so its decode time per token is at least (non-resident bytes) ÷ (host read rate): for example 40 GB of non-resident weights ÷ 47.6 GB/s ≈ 0.84 s per token (DERIVED); the plan card shows this bound |
| Training trainable set | The "core" (spine projections and shared experts) through LoRA, trained through the exact quantizer | The same projections (attention and MLP) through LoRA |
| Status | The design was derived and partly measured for this model | Dense support is new in this document. It was a non-goal of the trainer master (3.2: "no evidence the design transfers"). Its speed is unmeasured; the first dense test model is chosen at step 13 of the order (15.2) and the plan card says "unmeasured" until then |

Architecture plug-ins (`model/arch_*`): each names its layer kinds, tensor-name map, norm conventions (for example the `+1` folding of the reference model's norms) and template quirks. The first plug-in is `qwen35moe`. The plug-in interface is gated by E13: a second model must run through it.

### 5.5 Quantized or not

| Input | Paths offered |
|---|---|
| bf16 / fp16 / fp32 safetensors | Q (quantize), TQ (train then quantize) |
| A quantized GGUF | T (train the quantized model; frozen tensors stay byte-identical), Q (re-quantize to another variant: warned, because quantizing an already-quantized file loses more than quantizing from the original; the warning names the original-precision route) |
| An unquantized GGUF (F16, BF16, F32) | Q and TQ |

---

## 6. The user's flow, screen by screen (R11 to R18)

The same flow runs in the web interface and on the command line (`pelias plan`, `pelias train`, and so on). Every screen has Back. Nothing is written until the final summary is confirmed (U10).

### 6.1 Start: model and mode

The user chooses **Standard** or **Expert** mode (U11) and gives the model path. Optionally an import file (7.3).

### 6.2 Probe

The quick pass runs (4.1); slow probes run only if the machine is new or changed (`machine_state_sha`). If GPU memory is held by other programs, the close-these-apps list appears (U6): the user chooses all, some or none; the tool never closes anything itself.

### 6.3 Detected settings: see, adjust, add (R13)

One table per group (CPU, GPU, memory, drive, OS, model). Each row: the setting, its value, **why** it was chosen ("MEASURED 26.3 GB/s pinned copy, 2026-10-23" or "PUBLISHED default for RTX 5070, replaced at first probe"), and what it switched on (for example "cold experts on 7 pinned CPU workers, AVX2 path").

- **Adjust:** any value can be changed within its allowed range (the dictionary of 7.2). The change is checked against physical limits at once.
- **Advanced menu (Expert mode only):** "+ Add variable" searches the whole parameter dictionary and adds any key, including per-layer keys (for example `bank.layer_capacity.<0..39>`). Every forced value is labelled USER-FORCED, shown beside the measured one, tried for real for a moment (allocate and touch) so a wrong number fails in seconds, and repeated on the final summary.

### 6.4 Import (R14)

Drop in a file (json, txt, log, csv, md; `.py` read without running it). The import report lists every line read, every key matched, every key not recognized, and every value outside its range. Imported values become **proposals**, not settings, until accepted (U10).

### 6.5 Coach: what the model should learn, and for which clients (R15, R16)

Only for paths T and TQ. Three sheets, each editable in a table in the interface or imported:

| Sheet | Columns | Example rows |
|---|---|---|
| **parameters** (what the model learns) | `id, name, tier, description, answer kind, answer, language, depends on` (D30; the author's own list uses `id, name, tier, description`, 330 rows in 11 tiers, READ-SRC) | 3 to 4 erasable rows marked EXAMPLE |
| **persona** | assistant name, one-line role, system-prompt text, optional operator name | 1 EXAMPLE row |
| **clients** (the programs that will call the model, R15) | name, system-prompt style, roles used, tool schema, whether reasoning is sent back, request source label (CAPTURED, PUBLISHED, APPROXIMATION, UNVERIFIED) | Section 8.5 |

The coach then: checks the sheets (duplicates, empty or vague descriptions, answers that parse, every parameter covered); proposes rows that are missing (knowledge rows from descriptions; the base model may draft answers only if the user allows it, open decision O3); builds the training jsonl with the model's own chat template and never truncates (rows over the window stop the build with their ids); keeps the jsonl, its checksum and the audit with the project (R16).

### 6.6 Feasibility and recommendation (R17)

The plan card shows two to four plans that fit. Each plan shows: fit (measured headroom under the ceiling), predicted speed (DERIVED from measured rates, or "unmeasured" with the probe that will measure it), quality risk (which K-gate covers it), disk and time (measured or unmeasured), and its effect on the output file and the engine. One plan is marked **recommended**, with the reason. If nothing fits, the card says why and what would make it fit (close apps, smaller context, a smaller model variant). Nothing starts until the user picks a plan.

### 6.7 The three paths (R18)

| Path | Input | What runs | Output |
|---|---|---|---|
| **Q** quantize | Safetensors folder or GGUF | Own quantizers, per-tensor type rule of the chosen variant | `<model>-<variant>.gguf` + report |
| **T** train a quantized model | A quantized GGUF + the coached data | Layer-major training; frozen tensors byte-identical; the core trained through the exact quantizer (Mode Q) | Trained GGUF at the same variant, optional adapter GGUF, layout manifest, hot set |
| **TQ** train and quantize a full model | Safetensors folder + data | Stage A (own quantization to the chosen variant, written once as the base) then path T on that base | Trained GGUF at the chosen variant; the base GGUF is kept for later runs |

### 6.8 Final summary and start

One screen (U10): model key, machine numbers and margin, the chosen plan, data counts per family, every change made, every open warning, every forced value, disk and time, the files to be written, and how to stop. Start works only after the user confirms the open warnings. During the run: the spear progress bar, a plain traffic light, the telemetry panel, and a warning window with its actions (11.5).

### 6.9 Serve

After the run (or for any GGUF the user points at), `pelias serve` starts the engine and server on 127.0.0.1:8062 with the plan's settings (Section 11).

---

## 7. Parameters

### 7.1 One dictionary

Every setting has one canonical key, a type, an allowed range, a default on the reference machine, its label and source, and the module that uses it. The dictionary is the single source for the interface, the importer, the command line, the profiles and the documentation. Its first version is the author's attached dictionary (`docs/inputs/pelias-parameter-dictionary.md`) and parameter sheet (`docs/inputs/pelias_server_master_parameters.csv`), reconciled in 18.1; the build turns it into `resources/parameters/dictionary.json` with a check that every key is used by a module and every module key is in the dictionary.

Key groups: `system.*`, `probe.*`, `plan.*`, `hal.*`, `os.*`, `vram.*`, `host.*`, `model.*`, `io.*`, `bank.*`, `handoff.*`, `cpu.*`, `gpu.*`, `kv.*`, `prefill.*`, `engine.*`, `generation.*`, `server.*`, `api.*`, `sampling.*`, `tok.*`, `ingest.*`, `data.*`, `train.*`, `lora.*`, `core.*`, `mode_l.*`, `mode_b.*`, `export.*`, `warn.*`, `web.*` (new, Section 12), and indexed per-layer keys in Expert mode.

### 7.2 Where a value comes from, and which wins

How the ceiling settings combine: `vram.ceiling_mode = probe_auto` (default) measures the ceiling at start; `vram.ceiling_override_gib` is a **cap** that can only lower it (an imported 10.7 never raises a smaller measured value); `user_forced` (Expert mode) is the only way to set a higher number, and it is labelled and tried for real (6.3).

Order of precedence, lowest first: built-in default → PUBLISHED hardware default → cached slow-probe value → imported profile (accepted) → quick-probe value where the probe measures the same quantity (a measurement beats an imported number for a hardware fact) → user adjustment → USER-FORCED (Expert). Exception: an imported value that is **lower** than the measured limit (for example a smaller bank) is kept, because it is a choice, not a fact. The detected-settings screen shows the winning source for each row.

### 7.3 Import formats (R14, R16)

| Format | What is read | Never |
|---|---|---|
| `.json` | A profile object (7.4) or a flat key/value object | — |
| `.txt`, `.log` | `key = value` and `key: value` lines; also the probe program's own log lines (`P-H8 ... cuda_free_gib=10.83`) | Free prose is listed as "not used" |
| `.csv` | A sheet with a header row (parameters, persona, clients, or `key,value[,note]`) | — |
| `.md` | Tables (header row names the columns), `key = value` lines, numbered lists (`Var N: NAME - text` becomes a parameter row draft), headings as tiers | — |
| `.py` | Parsed only: the list of `{id, name, tier, description}` and simple constant assignments (`MAX_TOKENS = 4096`) | **Never executed** |

Every imported row is DRAFT with its source file and line until accepted. UTF-8 and UTF-16 are both read. The report lists lines read, rows found, duplicates, and lines not used (the coverage-ledger idea of the author's chunked-reading rows).

### 7.4 Profile file layout

```json
{
  "profile_schema": "pelias.profile/1",
  "profile_name": "…",
  "applies_to": { "machine_state_sha": "…", "model_key": "…" },
  "provenance": [ { "value_key": "…", "label": "MEASURED", "source": "…" } ],
  "hardware_snapshot": { "…": "…" },
  "settings": { "canonical.key": "value" },
  "plan_basis": { "vram_ceiling_gib": 10.7, "assumed_free_after_context_gib": 10.83, "margin_gib": 0.13 },
  "recommend_on_higher_ceiling": true
}
```

`plan_basis` records the ceiling the profile was tuned for, so the system can compare it with what it measures now (7.6).

### 7.5 The author's own profiles (R28)

Written for this repository from the newest versions on the drive (sources in 19.2):

| File | Holds | In git? |
|---|---|---|
| `profiles/reference_pelias/pelias_machine_profile.json` | The reference machine: measured hardware facts and the tuned engine, server and trainer settings for it | Yes (hardware facts already appear in the trainer and engine repositories; no serials, paths or names) |
| `profiles/reference_pelias/qwen36_35b_a3b_q4km_model_profile.json` | The reference model: byte map summary, layer pattern, hot-bank sizes and measured peaks, tuned sampling and template settings | Yes |
| `profiles/reference_pelias/pelias_reference_settings.csv` | The same settings as one flat sheet for the CSV importer, each with label and source | Yes |
| `profiles/local_only/author_training_profile.json` | The author's training set-up: the builder's settings (row ceiling, seed, eval share, families), the location of the 330-parameter registry and the dataset files, and the v0.2 recipe | **No** (ignored by git; the parameter descriptions and persona are private) |

### 7.6 Higher-ceiling recommendation (R29)

When a profile is imported, the system compares the profile's `plan_basis` with what it has just measured.

1. **Measured ceiling higher than the profile's.** On the reference machine this is expected on the very first run: the profiles' `plan_basis` is the whole-card reading (it includes Windows' share of VRAM), while the governor measures the process-only reading, which leaves about 1.0 to 1.3 GiB more room (4.3, decision 7). A leaner allocator in the Pelias engine than in the llama.cpp tier would add more (unmeasured, P-E4). The planner re-solves the plan with the new ceiling. Every setting that changes is listed as a **recommendation**: the old value, the new value, why, what it buys (DERIVED from measured curves, for example the hot-bank hit-rate curve: 0.43 / 0.54 / 0.63 / 0.70 / 0.74 / 0.76 / 0.85 at 2 / 3 / 4 / 5 / 5.6 / 6 / 8 GiB, and about +1 tok/s per +0.01 hit rate, DERIVED in the trainer master 7.12), and what it risks. Candidates, in the order the planner tries them: a larger hot bank (inference), a third expert slot (+0.49 GiB, training), a longer context, a raised core type (only if the engine supports it, P-E5). No threshold is invented: a recommendation appears exactly when a plan that did not fit under the old ceiling fits under the new one.
2. **Measured ceiling lower:** the governor refuses the imported plan and the planner proposes the largest plan that fits, listed the same way.
3. **The user decides:** accept all, accept some, or keep the imported values. Nothing continues until the user answers (R29). The answer is saved in the project file.

The same rule applies when nothing is imported: the plan card compares against the reference profile's basis only if the user asks for that comparison.


---

## 8. Training (paths T and TQ; R15 to R20)

The training design is the trainer master's (`pelias-trainer/docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`, chapters 4 to 12), unchanged in substance. This section states it for the whole system and adds the dense-model route.

### 8.1 Principles (trainer master 3.1)

Train what is served (the base weights are the GGUF bytes the engine will serve); splice, do not re-derive (frozen tensors are copied byte for byte and proven by hash); spend bits where behaviour lives (the spine, not the experts); reorganize the loop before optimizing the kernel; two implementations of everything (a slow obviously-correct reference and a fast version admitted only when it matches); measure, then plan; no hidden state; fail loudly; reversible stages; Python only in tests.

### 8.2 Keeping the active tensors in VRAM until they are trained (R19, MoE)

The author's requirement: "isolate and keep all active tensors within VRAM during training until all are trained". The layer-major scheduler (D5, D7, D13) does exactly this:

1. The **core** (the always-active tensors: attention or DeltaNet projections, shared experts; 1,409.0 M parameters, 820.3 MiB at the served types, DERIVED) stays resident in VRAM for the whole run, with its LoRA state (rank 16: 21,166,080 parameters, 323.0 MiB of fp32 masters, gradients and Adam states, DERIVED).
2. For each layer in turn, that layer's **routed experts** are brought into a VRAM slot once, and **every micro-batch of the step passes through that layer** before the next layer is loaded. The routing of each token is recorded on a tape so the backward pass touches the same experts.
3. Forward runs layers 0 to 39, backward runs 39 to 0, reusing the resident layer 39 at the turn: **79 fetches, about 35.8 GiB per step, about 1.47 s at 26.1 GB/s, regardless of how many tokens the step holds** (DERIVED: 35.8 GiB = 38.4 GB). The trainer master also uses the 80-fetch bound, 2 × 18.164 GiB = 39.0 GB = 1.49 s, for its thresholds: the transfer hides behind compute once a step holds about 1,495 / 2,242 / 2,989 tokens at 1,000 / 1,500 / 2,000 tokens per second of compute (DERIVED on that bound; 1,473 / 2,210 / 2,947 on the 79-fetch figure); the default minimum is 3,072 tokens per step (`data.min_step_tokens`).
4. Three streams (compute, expert copies, boundary copies) are flushed with `cudaStreamQuery` so WDDM runs them in parallel: 0.92 overlap measured with a flush after each enqueue, against 0.47 / 0.52 without (MEASURED, P-H12); D27 applies it once per multi-millisecond batch, because a flush on a busy stream costs 76.6 / 8.05 µs (MEASURED, P-H9).

The old trainer v0.2 instead streamed each layer once per micro-batch: 72 fetches, 30.38 GiB per micro-batch, about 121.5 GiB per optimizer step (MEASURED, DERIVED).

### 8.3 Dense models (R19, non-MoE)

The same scheduler with the layer itself as the streamed unit: the planner keeps as many whole layers resident as fit under the ceiling (with LoRA state, boundaries and working set accounted), streams the rest layer by layer in both directions, and requires each step to hold enough tokens to hide the stream (the same arithmetic as 8.2, with the dense layer bytes). DECISION (proposed). Speed and the right minimum step size are unmeasured until the first dense model is tested (15.2, step 13).

### 8.4 Data (R16)

- Input: chat-format JSONL (`messages[]`) built by the coach (6.5) or supplied by the user.
- Rendered with the model's own chat template by the project's renderer; tokenized by the project's tokenizer (gates P-D1 and P-D2: zero mismatches against the reference).
- Labels: weight 1 on exactly the tokens the model must generate when served; step-wide token normalization (D8), which fixes v0.2's per-micro-batch weighting (a short reply got up to about 87 times the per-token weight of a long one, DERIVED).
- **Never truncate.** The training window is 8,192 tokens (`data.max_seq_len`); a longer row stops the run before any GPU time with its row id. The author's own builder already holds rows to 4,096 tokens and **stops** (exit code 3) when a row does not fit; it never cuts one (READ-SRC, the builder's header).
- Known trap for the template renderer: `tojson` must format exactly like the reference (spacing and escaping of Python's `json.dumps`), or tool definitions render differently; P-D2 includes it.
- Reasoning is passed as a separate field (`reasoning_content`), never left inside the answer, because the template splits on `</think>` when it is missing; an early audit found 82 of 3,000 turns damaged by that split (MEASURED, master).
- Audits run before training (counts only): over-length rows, assistant text found verbatim in the render, every parameter present in the training split, reasoning share, duplicates, tool-call schemas, held-out share (the author's builder uses 7 %, READ-SRC).

### 8.5 Clients and agent identifiers (R15)

The model must keep its identity and format under whatever program calls it (U8). The clients sheet holds one row per front end. Starting rows (each labelled; CAPTURED rows replace them as soon as a real request is logged by the server's structure-only request log):

| Client | Roles and system prompt | Tools | Reasoning returned? | Label |
|---|---|---|---|---|
| Visual Studio 2026 Copilot "bring your own model" (OpenAI provider at `http://localhost:8062/v1`) | IDE-style system prompt, developer role possible | IDE tool schema | Not known | APPROXIMATION (the author's builder carries Copilot-style prompts); the provider was untested when last recorded (2026-10-21) |
| Odysseus-ai | Not captured | Not captured | Not captured | UNVERIFIED (capture one request) |
| The author's web UI client | Not captured | Not captured | Not captured | UNVERIFIED (capture one request) |
| Ollama-style clients (optional alias port 11434, off by default) | Plain chat | Optional | No | APPROXIMATION |
| No system prompt / generic one-line prompt | — | — | — | APPROXIMATION (from the author's builder: seven system-prompt kinds and seven identity kinds, READ-SRC) |

Only what reaches the messages and the tool list can be trained on; HTTP headers never reach the model. Looking up the clients' published request formats waits for the author's go-ahead (open decision O6).

### 8.6 Near-zero drift (R20): what it means and how it is proven

What the author asked: the trained, quantized file should behave as close to the unquantized model as possible, so the engine reads it "at near identical performance". How the design gets there, and how it is checked:

| Step | What is done | Proof |
|---|---|---|
| 1 | The frozen tensors (all 123 routed expert tensors, embedding, `lm_head`, norms, routers) are **copied byte for byte** into the output | P-X1: SHA-256 of every frozen tensor equals the base file's (123 of 123 expert tensors) |
| 2 | The core is trained **through the exact quantizer** (Mode Q, D23, D31): every step rebuilds the effective weight `W_eff = Q(deq(W_base) + (α/r)·B·A)` with the project's bit-exact quantizer, runs the forward pass on `W_eff`, and passes gradients straight through to the fp32 LoRA masters. The exported core is the final step's `W_eff` at the served types (+0 bytes) | K2 below; P-E8 (does Mode Q close the merge gap, and at what cost) |
| 3 | The own quantizers are **bit-identical** to the reference | P-E6: zero byte differences on all 753 tensors |
| 4 | The forward pass matches the served arithmetic | K1 below |

The three KL gates are relative to the **measured float spread** of the served file, not a round number (author ruling H6): two stock llama.cpp configurations already differ by KL 0.017353 (batch 16) and 0.018286 (batch 1024) on the 16 × 1,024-token set (MEASURED).

| Gate | Compares | Pass |
|---|---|---|
| K1 | The trainer's forward pass against llama.cpp on the same GGUF | KL ≤ 1.0 × the float spread of the base file |
| K2 | The exported GGUF against the trainer's own reference | KL ≤ 1.0 × the float spread of the new file (P-Q4b re-measures it) |
| K3 | The Pelias engine against llama.cpp on the exported GGUF | KL ≤ 1.0 × the new float spread |

**Honest limit:** "near-zero drift" is the target these gates enforce. Whether Mode Q reaches it on this model is UNVERIFIED until P-E7 (do the served weights sit exactly on their grid), P-E8 and K2 run. If a gate fails, the escalation ladder of the trainer master (4.8) applies: raise the core type within the byte budget (D22), keep an unmerged adapter, or roll back. The design never claims "0.00 % drift" for the merged core; only spliced bytes are identical.

For path TQ the comparison includes the quantization itself: the K-gates compare against the bf16 original through the test-only oracles, and the user sees the measured KL next to the published guidance of 5.3.

### 8.7 Recipe and defaults

The first run reproduces the v0.2 recipe (so it can be compared): AdamW (0.9, 0.999), ε 1e-8, weight decay 0, clip 1.0, 5 % warm-up then cosine, learning rate 1e-4 (5e-5 second phase), 2 epochs, seed 3407, LoRA rank 16 and α 32 on 310 modules (READ-SRC, v0.2). New defaults: Mode Q on, Δ_max 0, LoRA on attention, DeltaNet and shared experts; optional (off): rank 32 or 64, LoRA on `lm_head`, trainable shared-expert gates and norms, MTP co-training, Mode L (router locality, gated by G-L1 to G-L4), Mode B or N (gated by P-E12). Deterministic mode: two runs with one seed are bitwise equal (P-T4), and a resumed run equals an uninterrupted one (P-T11).

### 8.8 Memory plan and speed targets (reference machine)

Device plan at a window of 8,192 tokens and up to 32 k tokens per step: 7.90 GiB base, 8.70 to 9.00 GiB with the Mode Q `W_eff` copy, leaving 1.70 to 2.00 GiB under 10.7 (DERIVED; working set 2.50 GiB is an ESTIMATE that P-M2 measures). Host: pinned expert store 18.65 GiB plus a boundary store of 5.0 GiB (16 k tokens) or 10.0 GiB (32 k), about 28 to 33 GiB in total under the 48 GiB WSL2 cap (DERIVED; 32 GiB pinned was held on both platforms, MEASURED). Speed targets: 600 tok/s minimum, 1,200 target, 2,000 stretch (DECISION), against v0.2's 247.85 (MEASURED); the prediction is 643 to 2,031 tok/s depending on GEMM efficiency, which is unmeasured until the Batch C probes P-K1 to P-K3 run (DERIVED). GEMM efficiency is now MEASURED on Windows (17.5: own bf16 GEMM 0.48 to 0.88 of the P-H3 peak; int8 path 0.25 to 0.37 of the int8 peak, 1.39x to 1.84x the bf16 path including norm and conversions); recomputing the prediction from it is open (O12).

---

## 9. Quantizing (paths Q and TQ)

### 9.1 Stage A: from safetensors to the base GGUF (D33)

Reads the shards one tensor at a time through the memory map (largest tensor in flight 1,024 MiB on the reference model, DERIVED); applies the architecture plug-in's conversions (for the reference model: strip `model.language_model.`, drop the 333 vision tensors (0.832 GiB), split the fused `gate_up_proj` at row 512, permute the DeltaNet V-heads to the tiled order (P-S3), fold `+1` into every RMSNorm weight except `ssm_norm` (P-S2), `ssm_a = −exp(A_log)`, keep `mtp.*` as `blk.40`; READ-SRC); quantizes with the project's own quantizers by the chosen variant's type rule; writes the GGUF (753 tensors for the reference model). Disk: the export needs about 21 GB free on the destination and checks before starting (trainer master 9.1); it writes to a temporary name and renames only after the hash proof.

### 9.2 Re-quantizing an existing GGUF

Allowed with a warning (5.5). Frozen layout options: `export.layout = expert_last_2mib` puts every expert tensor after a `pelias.pad` entry so each starts on a 2 MiB boundary at zero byte cost (all 123 expert tensors of the reference file are exact multiples of 2 MiB, READ-SRC).

### 9.3 Variant choice

The plan card lists the variants the planner can fit (5.3 table), with size (DERIVED from the type rule and tensor shapes), the published quality guidance, and whether the engine has kernels for every type in it (a variant the engine cannot run is marked "quantize only, cannot serve yet").

### 9.4 Outputs (R21)

Every run writes to a new timestamped folder; nothing is overwritten. Possible files: `<name>-<variant>.gguf` (path Q); `merged-core.gguf`, `adapter.gguf` (optional), `layout_manifest.json`, `PELIAS_HOT_SET` file re-profiled from the trained model's routing tape, `export_report.json` (hashes, K-gate results, settings), and the training jsonl with its checksum and audit (paths T, TQ).

---

## 10. The engine: the unified bus (R23 to R25)

The engine design is the blueprint's (phases E0 to E13; `pelias-engine/docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`) and the runtime specification's (`docs/spec/PROPRIETARY_RUNTIME_SPEC.md`).

### 10.1 One token, one layer

For each of the 40 MoE layers: the GPU runs the spine (attention or DeltaNet, norms, router); the router picks 8 experts; the slot table splits them into **hot** (resident in the VRAM bank, computed on the GPU) and **cold** (in pinned RAM, computed by pinned CPU workers with AVX kernels, optionally a share read by the GPU over PCIe); the two branches run at the same time and meet through the **doorbell**; the merge feeds the next layer.

### 10.2 The doorbell (bus/doorbell_handoff)

Cache-line-separated request and reply mailboxes in mapped pinned memory; acquire/release ordering (plain `MOV` on x86); the GPU publishes the cold request, launches the hot branch without waiting, then waits with bounded back-off for the CPU's reply; a timeout falls back to stream synchronization. Measured: 13.34 µs median, 18.71 µs p99 against 94.4 µs for stream synchronization (MEASURED, WSL2). 36.1 % of layer-steps have no cold expert at all and skip the wait (MEASURED trace).

### 10.3 The hot bank

A static, water-filled set of experts in VRAM per layer (84 to 161 per layer in the 7.5 GiB solution), chosen to maximize the byte-weighted hit rate (0.8273 simulated on held-out decode steps; live 0.775 to 0.779 for the 7.0 GiB bank, MEASURED). One all-zero slot per layer (72.6 MiB in total) means no kernel ever receives a negative expert id: the crash class of the prototype (a −1 id widened to 4,294,967,295) cannot occur. The bank is re-profiled from a trained model's routing tape at export.

### 10.4 The dual bus

CPU and GPU read the same DRAM at once: 26.56 + 21.06 = 47.62 GB/s, 93.0 % of the 51.2 GB/s peak (MEASURED). The GPU's zero-copy share is a third branch (E12). Its original gate, "combined at least 1.25 times the CPU alone", **cannot pass on this machine**: 1.25 × 45.9 = 57.4 GB/s is above the 51.2 GB/s physical peak of dual-channel DDR4-3200 (the engine ledger records this, RESULTS §1). The reason for the branch is not more bandwidth but less CPU work: the E0 timeline measured 10.1 ms of CPU cold-expert compute per token against 2.86 ms modelled (DERIVED split), so moving a share of the rows to the GPU frees CPU time. **Restated gate** (DECISION (proposed)): with the branch on, the wide-load kernel reaches at least 15 GB/s (target 26.2 GB/s from 9.59 GB/s; P-E10 not run), the combined rate is at least 1.0 times the CPU alone, and decode is faster than with the branch off by more than run-to-run spread over 3 launches per side.

### 10.5 Isolation (the "best isolation method")

| What is isolated | How |
|---|---|
| Memory | Arenas with lifetimes (model, bank, session, prefill, token, pinned host); all planned before token 1; zero allocations in the decode loop |
| The GPU's work | CUDA graphs for the static spine (graph replay 0.716 / 0.707 µs per kernel against 4.70 / 2.18 µs eager, MEASURED) |
| CPU workers | One per physical core, pinned, never sharing a core with the decode thread |
| OS differences | Only in `src/os/` and the host-alloc backends |
| The user's other programs | The VRAM governor's measured ceiling; the clean-stop guard |

### 10.6 Resizable BAR (R24)

Recorded facts (MEASURED): ReBAR is on in the BIOS; BAR1 total 16,384 MiB; inside WSL2 BAR1 used reads 0 MiB, on Windows-native 16,355 MiB (meaning not examined). No ReBAR on/off comparison exists.

What ReBAR could give: the CPU writing directly into VRAM (write-combining, CPU-write-only) instead of the GPU pulling from pinned RAM. The routes found in the author's ReBAR guide (attachment of 2026-10-22, UNVERIFIED until tested): (A) Vulkan memory that is both device-local and host-visible, mapped by the CPU, then shared with CUDA through external-memory interop; (B) a D3D12 heap with a write-combining CPU page property, shared with CUDA the same way. The kernel route used by `gdrcopy` is locked on GeForce cards by the proprietary driver (the guide's reading of NVIDIA's documentation).

**What is built** (DECISION): a switch `hal.rebar_path = off | auto_probe | on` (default `auto_probe`) in both the Windows and the Linux builds; a probe **P-E15** (new) that maps a host-visible device-local buffer through route A (and B on Windows), measures CPU write bandwidth into it with non-temporal AVX stores and the GPU-side visibility latency, and compares with the pinned zero-copy path; and the on/off test from the Wave 0 record (switch ReBAR off in the BIOS, BAR1 should read 256 MiB, rerun P-H2, P-H5, P-H6 and P-H12 tagged `rebar_off`). `auto_probe` turns the path on only when P-E15 shows a gain over the measured zero-copy route. Inside WSL2 the route depends on whether the WSL Vulkan or D3D12 driver exposes such memory: unmeasured.

### 10.7 Speed (R25)

The modelled ceiling on the reference machine is 10.87 ms per token = 92.0 tok/s (26.2 GB/s cold reads) to 11.34 ms = 88.2 tok/s (9.59 GB/s) (DERIVED; spine time 7.73 ms is assumed). Every recorded run used the patched llama.cpp tier build: WSL2 reference Cell A 37.9 short and 42.1 at 12 k; Windows-native Cell A 47.3 / 46.8 short and 53.7 / 54.1 at 12 k (two launches); the best WSL2 cell 53.5 / 53.0 / 49.0 at short / 4 k / 12 k; the blueprint's summary over the WSL2 launches and load modes it lists, 44 to 54 short and 43 to 49 at 12 k (MEASURED). The measured WSL2 time split shows zero CPU/GPU overlap and 4.6 to 5.0 ms or more of handoff per token against 0.25 ms modelled (DERIVED from E0 timers). The first run of the Pelias engine is the measurement that finally isolates the gap. Gates: first at least the tier's result, then G3 (at least 55.0 tok/s at 12 k), then the ceiling table. Prompt reading at least 1,670 tok/s at 12 k, aimed at 2,238 (MEASURED stock).

---

## 11. The server on port 8062, and the web interface (R22)

### 11.1 Address and endpoints

`server.bind_address` 127.0.0.1, `server.port` **8062** (A8). On Windows the socket is exclusive (`SO_EXCLUSIVEADDRUSE`); if 8062 is taken, start-up fails at once with a clear message.

**Protection against other web pages** (Fable review): any web page the user visits can send requests to 127.0.0.1. So: requests whose `Host` header is not the bound address and port are refused; every `/api/*` call (start a probe, import a file, start a job, start or stop the solver) needs a session token that the served interface receives and that is checked together with the `Origin` header; `/v1/*` follows `server.api_key` when it is set and refuses browser `Origin`s other than the interface's own unless `server.cors_allow_origin` names them. Gate PG14. Endpoints: `POST /v1/chat/completions` (streaming and not), `GET /v1/models`, `GET /health` (state, plan, hit counters, VRAM), `GET /` and `/ui` (the web interface), `/api/*` (the interface's own calls: probe results, settings, plan card, imports, runs, logs), and an optional Ollama-style listener on 11434 (off by default).

### 11.2 Request life cycle (runtime specification 6.12, unchanged)

Accept → parse (headers up to `server.max_header_bytes` 16,384, body up to `server.max_body_bytes` 32 MiB, chunked bodies) → JSON → render the chat template and tokenize → context check (no silent context shift: an over-long prompt returns HTTP 400) → prefix cache (DeltaNet state snapshots, 62.9 MiB each in the records; DERIVED: 30 layers × 32 heads × 128 × 128 × 4 B = 60.0 MiB of state plus 3 × 8,192 × 4 B × 30 = 2.8 MiB of convolution history) → chunked prefill → decode and sample → reasoning split (`reasoning_content`) and tool-call parsing (`<tool_call><function=…><parameter=…>` to OpenAI `tool_calls`) → server-sent events with a `: ping` heartbeat every 15 s during long prefills → `finish_reason`, usage, `[DONE]` → a structure-only request log (roles, tool count, unknown field names, token counts, latencies; never message text).

Unknown request fields are ignored and **counted by name** in the log; that count is how the llama-server catalogue (O8) is closed from real traffic instead of guessed.

### 11.3 Defaults (from the launcher the clients use today; 18.1 lists the conflicts settled)

Context 32,768 (131,072 supported after E9), one slot, no context shift, unlimited `max_tokens` up to the context, KV cache `q8_0`, sampling temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, repeat penalty 1.0, presence penalty 0.0 (READ-SRC from the 2026-10-24 launcher, the best-cell launch record and the author's Modelfile), reasoning format `deepseek` with the field `reasoning_content`, thinking on by default (the launcher's `{"enable_thinking":true}`), seed 20261019 (dictionary).

### 11.4 Comparing with llama.cpp (R4)

The llama.cpp interim server keeps port 8012. A replay tool sends each captured request to both servers and compares response structure, finish reasons and event shapes (not message text); the K3 gate compares logits. Neither server ever starts the other.

### 11.5 The web interface (A5; R13)

A single page served by Pelias itself (HTML, CSS and JavaScript written for this project, no outside library), so it works the same on Windows and Linux, in any browser, with nothing to install. It is the web port of the lithic design already written as `ui/pelias_lithic_ui.hpp` (2026-10-24, kept in `support/ui_reference/` as the specification):

| Element | Specification (from the header and the art) |
|---|---|
| Palette | Matte slate canvas `#262A31`; deep slate wells `#1D2026`; marble light / mid / shadow `#C6CCD4` / `#899099` / `#4E535B`; fissure `#202328`; weathered taupe `#8F8578`; core bloom cyan `#C4F7FF`; Aegean voxel cyan `#6CD4FF`; holo wireframe cyan `#3898EC` at 45 %; lightning yellow `#FFD600`; thermal amber `#FFB300`; crimson fault `#E53935`; the spear's particles at a measured hue near 185° |
| Logo | `resources/ui/source/logo_full.png` (2,757 × 1,530): a stone head dissolving into cyan voxels above the lettering ΠELIAS |
| Spear progress bar | Two baked textures of the same size (`spear_stone.png`, `spear_holo.png`, 1,600 × 208, layout in `spear_layout.json`: padding 20 px, texture u from 0.0125 to 0.9875), cut at the progress point with a ragged edge: stone behind, glowing hologram ahead. Reference stages `s0`, `s10`, `s50`, `s100_full.png`. The baker `tools/ui/bake_spear_textures.py` (offline, not part of the product) remakes the textures |
| Headings | Chiselled Greek-style lettering for page titles, milestones and the percent read-out; ordinary font for everything interactive |
| Run states | Idle, running, paused on a warning, complete, stopped by a fault |
| Warning window | The problem, its numbers, and the actions confirm, deny, try again (and stop) |
| Telemetry | Live speed, VRAM, temperatures, clocks, hit rate, log lines; the page polls `/api/telemetry` and never blocks the engine |

### 11.6 Two programs or one

`pelias serve` runs engine and server in one process (the doorbell needs the engine's memory). The interface's long jobs (probe, quantize, train) run in a worker process started by the server, so the interface stays responsive. DECISION (proposed).

**One GPU owner at a time** (Fable review): serving holds the bank and the KV cache (up to the ceiling) and training needs 7.90 to 9.00 GiB of VRAM, and on WSL2 the two pinned stores (18.65 GiB and 28 to 33 GiB) together exceed the 48 GB memory cap. So the server unloads the model (or refuses the job) before a quantize or training worker starts, and the governor checks free VRAM and free pinned RAM across all Pelias processes before any of them allocates. The interface shows which job owns the GPU.

---

## 12. Web tools for the model (R26)

### 12.1 What already exists (READ-SRC, the author's earlier web-tools script and dataset builder)

A search through a local SearXNG instance (`http://127.0.0.1:8888/search`), page fetches through FlareSolverr (`http://127.0.0.1:8191/v1`, a 60,000 ms challenge window), HTML-to-text extraction that skips script, style and similar tags, paging of long text (6,000 characters per page) with a coverage ledger, a jail for file paths, a permissions registry (domain, download, microphone: always allow, allow once, deny), and redaction of card numbers (Luhn-checked), e-mail addresses and API keys before anything is sent out.

### 12.2 What Pelias builds (ground up, in `src/web/`)

| Tool | What it does |
|---|---|
| `web.search` | Metasearch: sends the query to configured public search pages, parses the result pages, merges and de-duplicates. Design taken from SearXNG's idea (engines, merging, scoring); **no SearXNG code** (AGPL-3.0) |
| `web.fetch` | HTTP/1.1 client with HTTPS through the OS (B4), redirects, size and time limits, robots and rate limits per domain |
| `web.read` | HTML to plain text; page slicing with a coverage ledger, so the model can page through long documents |
| `web.solve` | Pages behind a browser challenge: hands the URL to FlareSolverr if it is running (B5); otherwise reports that the page needs it |
| Guards | The permissions registry, redaction before sending, a domain allow-list in Standard mode, and every request logged by structure only. `web.fetch` refuses loopback, private, link-local and LAN addresses (and names that resolve to them), so a fetched page cannot steer the model into calling 127.0.0.1:8062, the solver on 8191 or the user's network; only USER-FORCED in Expert mode lifts it. Gate PG12 tests it |

The tools are offered to the model as ordinary tool calls (`<tool_call>` → server → tool → result back into the conversation), and the clients sheet (8.5) carries their schema so the model is trained to use them.

### 12.3 FlareSolverr (B5)

Found on the drive: FlareSolverr 3.5.2 and a newer master copy, MIT licence, Python with Selenium and `undetected_chromedriver` (READ-SRC, `requirements.txt`), plus a prebuilt Windows executable and a Linux archive. It works by driving a real Chrome browser, so the challenge is solved by the browser itself.

**Use (DECISION):** an optional separate process, started by the user or by `pelias web solver start` from the prebuilt binary, reached only on 127.0.0.1:8191, never linked, never required. If it is not running, `web.solve` says so and everything else works.

### 12.4 Replacing it with our own code later (the forward route)

FlareSolverr's essential mechanism is "let a real browser load the page, then read the result". Pelias can do the same with its own code: a small client for the browser's DevTools protocol (a WebSocket and JSON, both of which the project writes anyway) that drives the browser already on the machine (Microsoft Edge ships with Windows; on Linux and WSL a Chromium-based browser if installed). DECISION (proposed) for after the core system works; measurable by one test: the same list of challenge pages through FlareSolverr and through the own driver, success counted. Until then FlareSolverr stays the exception the author allowed. Use of the tools must respect each site's terms; the domain allow-list and rate limits are on by default.

---

## 13. The installers (R2)

### 13.1 What is installed

| Item | Windows 10 / 11 | Linux (WSL2 or native) |
|---|---|---|
| Program | `pelias.exe` (one executable: command line, server, engine, trainer, quantizer) | `pelias` (one executable) |
| Resources | Web interface files, logo and spear textures, Unicode tables, the parameter dictionary, help texts, example sheets | Same |
| Data folder | `%LOCALAPPDATA%\Pelias\` (machine profile, caches, projects, logs) | `~/.local/share/pelias/` |
| Start menu / launcher | Start-menu entry for the web interface; a console entry; an optional "start server" entry that keeps its window open (the author's interim-launcher rule) | A `.desktop` entry where a desktop exists; the command line otherwise |
| Not installed | NVIDIA driver (the user's; the installer checks the version and says what is needed), FlareSolverr (optional, B5), any model | Same |

### 13.2 Requirements checked by the installer (no silent installs)

GPU driver present and new enough for the program's CUDA runtime (the reference: driver 617.42 supports CUDA 13.4, MEASURED); OS version; free disk for the program; on WSL2, the `.wslconfig` memory cap is read and shown. A missing requirement is reported with the exact step to fix it; the installer installs nothing outside its own files.

### 13.3 How the installers are built

Built by the project's own CMake targets on the build machines (the toolchain of Section 17.1: MSVC 19.44 with nvcc 13.4.92 on Windows; GCC 15.2.0 with nvcc 13.0.88 on WSL2). Packaging: Windows, a self-extracting installer program written for the project (no third-party installer framework in the product; building with the platform's own tools is a build-time choice, not a run-time dependency, decided at step 14 of the order); Linux, a `.tar.gz` with an install script and a `.deb` package. DECISION (proposed).

### 13.4 Compatibility targets

CUDA architectures in one binary: SASS for sm_75, sm_80, sm_86, sm_89, sm_90 and sm_120, plus PTX for compute_75 so that newer cards can compile it at load (DECISION (proposed); the reference build is sm_89 only today; PTX built for a higher capability cannot run on a lower one, which is why the PTX is the lowest one).

**Windows 10 and the driver** (PUBLISHED, S8): NVIDIA releases its final Game Ready and Studio driver for Windows 10 in October 2026; from November 2026 new packages require Windows 11, and Windows 10 gets quarterly security releases through October 2029. Whether that final Windows 10 driver runs a program built with the CUDA 13.4 toolkit is **UNVERIFIED**. Check before step 15: the minimum driver of the 13.4 toolkit (its release notes) against the version of the final Windows 10 driver. Fallback if it does not: build the Windows installer with the 13.0 toolkit (the one already used on WSL2), keep to the 13.0 API and ship SASS for every listed architecture so nothing depends on compiling PTX with an older driver. Windows 10 is tested only when a Windows 10 machine is available (unmeasured).

### 13.5 Release check (BC7)

The installer's file list is compared against the product list; no oracle, test, Python or third-party file may be in it. Uninstall removes the program and leaves the data folder unless the user asks.


---

## 14. The file scaffold (R5, R30)

### 14.1 Where the code lives

**pelias-master is the system repository.** The product code is written here, in one tree, from the first line. pelias-trainer and pelias-engine stay as the design and measurement records they are (their documents, ledgers, raw logs and probe archives are referenced, not copied). DECISION (proposed); it follows the author's order "(5) unify" by starting unified, which the lifted holds (A1) allow.

### 14.2 The tree

Two layers of shared libraries and one program. Every directory below is planned unless marked *exists*. The step that first creates each piece is in brackets (Section 15.2).

```text
pelias-master/
├─ README.md  LICENSE  CITATIONS.md  THIRD_PARTY_NOTICES.md  VERSION          (exists: README, LICENSE, CITATIONS)
├─ CMakeLists.txt  CMakePresets.json  .clang-format  .gitignore  .gitattributes [2]
├─ cmake/  PeliasOptions.cmake  PeliasModule.cmake  ProbeFlags.cmake            [2]
├─ docs/                                                                       (exists)
│  ├─ PELIAS_SYSTEM_MASTERWORK.md      this document
│  ├─ PELIAS_SYSTEM_MASTER.md          the one-page scaffold and numbers of 2026-10-25 (Section 6 corrected)
│  ├─ BUILD_NOTES.md                   how to build, per platform; filled as the build proceeds
│  ├─ inputs/                          the author's attachments of 2026-10-25/26, kept as given, with REVIEW.md
│  └─ scaffold/                        REPO_INVENTORY.tsv, verify_inventory.py
├─ logs/  BUILD_LOG.md  DECISIONS_LOG.md                                       (exists)
├─ profiles/                                                                    (exists)
│  ├─ reference_pelias/                the author's machine and model import files
│  └─ local_only/                      private training profile (ignored by git)
├─ resources/
│  ├─ ui/  source/ (logo, spear stages)  spear/ (baked textures, layout)       (exists)
│  ├─ webui/  index.html  pelias.css  pelias.js                                 [11]
│  ├─ parameters/  dictionary.json                                              [12]
│  ├─ sheets/  parameters.csv  persona.csv  clients.csv  (EXAMPLE rows)          [12]
│  └─ help/  one text per setting                                               [12]
├─ include/pelias/  status.h  version.h.in  log.h  trace.h  timer.h            [2]
├─ src/
│  ├─ os/        file_map  net  threads  large_pages  clock  signals  tls (OS HTTPS, B4)        [2, 14]
│  ├─ hal/       device  stream  event  arena  vram_governor  pinned_pool  sync_audit
│  │             host_alloc_wsl  host_alloc_win32  host_alloc_linux
│  │             backend_cuda/  backend_vulkan/  backend_cpu/                                  [2, 13]
│  ├─ plan/      probe_constants  roofline_solver  memory_plan  recommend (7.6)                [2, 12]
│  ├─ bus/       doorbell_handoff  cpu_threadpool  stage_streamer  graph_capture               [8, 10]
│  ├─ bank/      hot_set_solver  hot_set_io                                                    [10]
│  ├─ quant/     block_types  quant_cpu  quant_gpu  type_rules (variant mixes)                 [3]
│  ├─ io/        sha256  json  gguf_reader  gguf_writer  gguf_splice  safetensors  profile_io  [2, 3, 9, 12]
│  ├─ tok/       utf8  unicode_data  normalizer  pretokenizer  bpe  added_tokens  tokenizer
│  │             template_ast  template_eval  chat_template                                    [4]
│  ├─ model/     spec  arch_qwen35moe  arch_dense  frozen_store  trainable_core  lora  tape    [3, 7, 13]
│  ├─ kern/      gemm_gguf/  route/  elementwise/  attn/  deltanet/  loss/  optim/             [5]
│  ├─ cpu/       expert kernels: avx2, avx512, avx_vnni, portable                              [10, 13]
│  ├─ gpu/       bank  expert_q4k  expert_q6k  zerocopy (wide load)  rebar (P-E15)             [10, 15]
│  ├─ engine/    kvcache  state  layer  decode_loop  prefill  sampler                           [10]
│  ├─ server/    http  sse  api_openai  tools  ollama_alias  ui_routes  request_log            [11]
│  ├─ web/       search  fetch  read  solve_client (FlareSolverr)  permissions  redact         [14]
│  ├─ ingest/    stage_a                                                                        [3]
│  ├─ data/      dataset  masks  packer  audit                                                  [4, 9]
│  ├─ stream/  sched/  train/  export/  verify/                                                 [6–9]
│  ├─ project/   model_key  project_file  run_folder                                            [12]
│  ├─ startup/   quick_pass  machine_profile  vram_scan  detect_cpu  detect_gpu  detect_mem  detect_disk  detect_os [12]
│  ├─ intake/    sheets  import  clients  checks  coach                                         [12]
│  ├─ warn/  help/  flow/                                                                        [12]
│  └─ cli/       main.cc (pelias: probe, plan, quantize, train, export, verify, serve, bench, web) [2]
├─ installer/    windows/ (self-extracting installer)  linux/ (tar.gz + install script, deb)   [14]
├─ bench/        floors (ram_bw, handoff, dual_bus, pin_modes, wide_load, rebar)              [1, 15]
├─ tools/        check_module_deps.py  check_resources.py  unicode generator  ui/bake_spear_textures.py (exists) [2]
├─ tests/        unit/  smoke/  parity/  probes/  ref/ (fp64)  oracle/ (llama.cpp, test only)  [2 on]
├─ support/      ui_reference/pelias_lithic_ui.hpp (exists)  probe sources and results as they are added
└─ third_party/  (empty; any entry needs an entry in THIRD_PARTY_NOTICES.md and is test-only)
```

### 14.3 Module dependencies (acyclic; boundary check BC1)

`os` → none; `hal` → os; `quant`, `bank` → hal; `plan` → hal, io (io added at step 2: the planner reads the tensor table); `bus` → os, hal, quant; `io` → os, hal, quant; `tok` → io (for `io/json`, so there is one JSON parser); `kern` → hal, quant; `model` → hal, quant, io; `ingest` → hal, quant, io; `data` → io, tok; `stream` → hal, bus, model; `sched` → hal, kern, model, stream; `train` → hal, data, model, sched; `export` → hal, quant, io, model, bank; `verify` → hal, io, quant; `cpu` → os, hal, quant, bus; `gpu` → hal, quant, bank, bus, kern; `engine` → os, hal, plan, bus, bank, quant, io, model, kern, cpu, gpu; `server` → os, io, tok, engine, web, flow (the interface's `/api/*` calls and the model's tool calls); `web` → os, io; `project` → io; `startup` → os, hal, plan, io, project; `intake` → io, tok, data; `warn` → io, data, verify, intake; `flow` → plan, startup, project, intake, warn, verify, help; `cli` → everything above. (From the universal specification 4.3, with `web` added and `ui` removed: the interface is now files served by `server`.)

### 14.4 What this repository holds today (R30)

| Path | What |
|---|---|
| `docs/PELIAS_SYSTEM_MASTERWORK.md` | This document |
| `docs/PELIAS_SYSTEM_MASTER.md` | The system scaffold and numbers of 2026-10-25, with Section 6's two corrected toolchain rows (18.1, C19) |
| `docs/inputs/` | The author's attachments of 2026-10-25/26 that feed this design, unchanged, and `REVIEW.md` (what was checked, what conflicts) |
| `docs/scaffold/` | `REPO_INVENTORY.tsv` and `verify_inventory.py` (standard-library check of every file's size and hash, the privacy scan, and the git-ignore rule) |
| `profiles/reference_pelias/` | The author's machine and model import files |
| `profiles/local_only/` | The author's private training profile (ignored by git) |
| `resources/ui/` | The logo, the spear stages, the baked spear textures and their layout |
| `support/ui_reference/` | The lithic interface header of 2026-10-24 (the specification for the web interface) |
| `tools/ui/` | The spear texture baker |
| `tools/probes/gfx_probe/` | The graphics probe P-G0 / P-G0b (Direct3D 12 and Vulkan) |
| `tools/probes/batch_c/` | The Batch C kernel probes P-K1, P-K2, P-K2b, P-K3, P-Q10 (with the Wave 0 framework files they build on) |
| `data/probe_results/` | Every probe result file, failures included: `P-G0/`, `batch_c/`, `P-M4/`, `P-E16/`, each with a README |
| `src/`, `include/pelias/`, `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, `VERSION` | The product tree from step 2 on (os, hal, quant, io, plan, cli; 14.2) |
| `tests/` | The own test harness, unit tests, the corrupt-input gate (`pelias_mutate`); `tests/data/` is written at test time |
| `tools/check_module_deps.py` | Boundary checks BC1, BC3, BC4, BC6 (3.4) |
| `logs/` | `BUILD_LOG.md` (every step, dated) and `DECISIONS_LOG.md` (every ruling, dated, with its source) |

---

## 15. Order of operations (R31)

### 15.1 The routine for every step

1. One sentence: what is built and which gate closes it.
2. Code written, built host-only and tested in the sandbox; anything that fails is fixed before hand-over.
3. The author runs every GPU program on WSL2 and on Windows (and native Linux when available); results go to the author's results folder; each number is labelled MEASURED with its run.
4. A failed gate stops that branch; the reason goes in `logs/BUILD_LOG.md`.
5. Backups to the author's dated backup folder with INDEX.md and RUNBOOK.md updated (previous versions saved first; exact locations in `profiles/local_only/LOCAL_SOURCES.md`).
6. Commits stay local until the author says "push".

### 15.2 The steps

| Step | Build | Gate that closes it | From the plans |
|---|---|---|---|
| **R0** | **Fable review of this document and the profiles; findings folded in; the author says go** (A12) | The author's word | — |
| 1 | Close the measurement gaps: run the Batch C kernel probes (P-K1 to P-K3, P-Q10; rewritten 2026-10-26 13:5x because the 01:50 source never compiled; done on Windows in twelve revisions and on WSL2 with the final one, 17.5), the P-M4 file-read probe (done on both platforms 2026-10-26, 17.2), P-E16 (done 2026-10-26, 21.2), P-M5 (deferred to step 10, O16), the ReBAR on/off pair (made optional by the author 2026-10-26 17:29: ReBAR support is built in and the startup probe picks each path on the user's machine, 21.7 ruling of 13:21; the ReBAR-off path is validated on a later build day as an adjustment, not a rewrite, O17; the driver's per-program rBAR setting does not shrink BAR1, so only the BIOS switch is a real "off") (BIOS switch, P-H2/H5/H6/H12 rerun), the graphics probe P-G0 (done 2026-10-26, 21.2.1) and its follow-ups P-G0b, P-E16, P-M5 (21.7) | Results recorded on both platforms | T0, E0, E1 |
| 2 | Foundations: `os/`, `hal/` (CUDA backend), `plan/`, `io/` (reader, sha256, json, safetensors), `cli/` skeleton, build files, tests, boundary checks (stages A and B built and tested host-only 2026-10-26 evening, `docs/BUILD_NOTES.md`; stage C, the gate on both platforms, is the author's run) | Lists every tensor with its type: 753 in the served file (40 trunk blocks plus the MTP head `blk.40`, `block_count` 41, `nextn_predict_layers` 1, READ-SRC 2026-10-26); the Unsloth file has 733 (`block_count` 40, no MTP head) and served the first stage C runs; the planner counts the MTP head (O18); a plan above the ceiling is refused; corrupt-input gate; clean build on WSL2 and Windows | T1, E2, E3 (reader) |
| 3 | `quant/` (Q4_K, Q6_K, Q8_0, activations, F16/BF16/F32) and Stage A | P-E6 bit-identical on all tensors; Stage A output equals the reference (P-X6) | T1b, E3 |
| 4 | `tok/` tokenizer and template renderer; `data/` | P-D1, P-D2: zero mismatches (including `tojson` formatting) | T2, E10 (reuse) |
| 5 | `kern/` spine and expert kernels, slow references first | Rung 1 tolerances; kernel parity | T3 to T6, E4, E5, E7 |
| 6 | `verify/` and the external oracle harness | The oracle reproduces the float spread (0.017353 / 0.018286) | T12 (verify), E3 |
| 7 | One layer, a tiny model, the full forward pass | Rung 2 parity; rung 3 KL within the float spread (K1) | T7, T8, E6, E7 |
| 8 | Backward, LoRA, Mode Q; streamer and layer-major scheduler | Rung 4; zero allocations in the step loop; 79 fetches per step | T9, T10 |
| 9 | Training loop, evaluation, checkpoints, export | Rungs 5 to 7; hash proof; K2 | T11, T12 |
| 10 | Engine: CPU cold kernels, GPU bank, doorbell, decode loop, prefill; then MTP drafting with the head `blk.40`, built on the plain decode loop once that passes (required, author 2026-10-26 18:58; O18); hot set re-profiled on the clean base; P-E4 (NVML used before start and at peak, so both ceiling readings are measured); the ceiling read again before and during weight loading (it moved 0.5 GiB with a browser open, 2026-10-26) | E4 to E9 gates; first at least the tier's speed, then G3 (55.0 tok/s at 12 k); K3; MTP: drafted decode gives the same tokens as plain decode under greedy and stays inside the same KL gates under sampling, and its acceptance rate and tok/s with and without MTP are recorded | E4 to E9 |
| 11 | Server on 8062 and the web interface, with the request protections of 11.1 and one GPU owner at a time (11.6) | Drop-in for the interim launcher; client replay; a Visual Studio session works; PG14 | E10, PG8 |
| 12 | **The first benchmark** (16.4): quality F0, F1, F2, F5 (F3, F4 when the author supplies them), the bf16 reference through P-Q11, speed against the ceiling, energy through P-W1 | The table of results; no invented pass mark | new (A13) |
| 13 | Start-up probes, detection, plan card, recommendation (7.6), intake, coach, flow | The same build passes steps 10 and 11 on WSL2 and Windows; the author's profiles import cleanly | U1 to U11, E11 |
| 14 | Universality, part a: AVX-512 and AVX-VNNI paths, CPU-only mode (with its own spine kernels), dense plug-in and the first dense model, more quant formats | A second model (MoE) runs through the plug-in; a dense model passes K1 | E13 |
| 15 | Web tools; installers for Windows 10/11 and Linux (after the Windows 10 driver check of 13.4) | PG11, PG12; BC7; install and uninstall on a clean machine | new |
| 16 | Universality, part b: the Vulkan backend; speed work and gated extras (zero-copy third branch with the restated gate of 10.4, ReBAR path and P-E15, Mode L, LRU; MTP moved to step 10 as required, O18) | Each extra's own gate | T13, T14, E12 |
| 17 | Release gates; the benchmark repeated on the release build | PG1 to PG14 | — |

Steps 2 to 9 follow the trainer's order because the engine reuses everything they build; step 10 can start in parallel after step 7 if the author wants both moving (the engine needs the spine kernels and the reader, not the training loop). The first benchmark (step 12) comes as soon as engine and server exist, because it is the author's stated goal for this build (A13); it is repeated on the release build (step 17). (Order revised after the Fable review of 2026-10-26.)

### 15.3 What the author runs, and what Claude does not do

The author runs every GPU program, every install, and every command that touches PowerShell. Claude writes code, builds host-only, tests on the CPU, reads results, and logs. Nothing is pushed or deleted without the author's word.

---

## 16. Verification and gates

### 16.1 The ladder (trainer master 10.2, universal specification 10.3)

| Rung | Scope | Pass |
|---|---|---|
| 1 | Each kernel | Tolerances of 16.2 against the fp64 reference; zero `compute-sanitizer` errors |
| 2 | One layer forward | Hidden states within the single-layer float spread of the oracle; router top-8 agreement ≥ 99.5 % |
| 3 | Full forward | K1 |
| 4 | Backward | fp64 finite differences and the test-only PyTorch slice (cosine ≥ 0.99999) |
| 5 | Scheduler and governor | Zero allocations in the hot loop; 79 fetches per step; peak ≤ the ceiling; stall ≤ 5 % at ≥ 16 k tokens per step (trainer T10, P-T7; 3,072 is the default minimum step, not the stall gate) |
| 6 | Training | 32-step overfit cuts loss by ≥ 90 %; same seed, bitwise-equal runs; resume equals uninterrupted; at least v0.2's eval loss on its recipe |
| 7 | Export and serve | P-X1 hash proof; K2; K3; engine peak ≤ the ceiling; client matrix (U9) |

### 16.2 Kernel tolerances (DECISION, initial; tightened after first results)

Each tolerance names its reference, because the two source documents measure against different things. **Trainer master 6.11 (governs):** quantizers, dequantizers and the Mode Q rebuild bit-exact; bf16 and int8 GEMMs relative L2 ≤ 1e-6 against an fp64 GEMM of the **same rounded inputs** (so only the accumulation order differs); router identical top-8 sets; DeltaNet forward ≤ 1e-4 per layer-window (fp64 ≤ 1e-5), backward ≤ 1e-3; attention forward ≤ 5e-4, backward ≤ 2e-3; cross-entropy ≤ 1e-6; AdamW ≤ 1e-6. **Universal specification 10.2 (for information):** the same kernels against the fp64 reference of the **unrounded** operation, which is why its GEMM limits are looser (≤ 1e-3 bf16, ≤ 2e-3 int8). Both are initial DECISIONs, tightened after the first results.

### 16.3 Release gates

PG1 boundary and hygiene (BC1 to BC7, provenance, privacy scan). PG2 corrupt inputs (1,500 mutated files). PG3 governor refuses over-ceiling plans. PG4 tokenizer and template. PG5 bit-exact quantizers and hash-proven splice. PG6 determinism and resume. PG7 engine speed and memory (peak ≤ ceiling, G3, prefill ≥ 1,670 tok/s at 12 k, needles 12/12 to 120 k). PG8 client replay (Copilot, Odysseus-ai, web UI, Ollama alias). PG9 builds on WSL2 and Windows with warnings as errors. PG10 runs with the oracles deleted. **New:** PG11 installers (clean install, run, uninstall on Windows and Linux); PG12 web tools (search, fetch, read, solve-through-FlareSolverr, permissions, redaction); PG13 universal detection (a machine profile from a second, different machine matches its hardware); PG14 local-API protection (Host and Origin checks, session token on `/api/*`, `web.fetch` refuses private and loopback addresses). These numbers follow the universal specification; `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` in the engine numbers its gates differently (its PG3 is the tokenizer).

### 16.4 The first benchmark (the author's goal for this build, 2026-10-26 12:05, AUTHOR-STATED)

The author's aim: with this first build, measure the Pelias 4-bit file against the 4-bit files published for the same model (the official upload and Unsloth's), against the unquantized model, and measure the engine's speed against the ceiling. "Proof is in the pudding. We make the pudding."

**What is expected, stated honestly:**

| Claim | On paper | What decides it |
|---|---|---|
| The Pelias base (Stage A, stock Q4_K_M type map, no imatrix) equals a stock Q4_K_M | **Byte-identical** to llama.cpp's own quantization of the same bf16 shards, by design (P-E6, P-X6). That makes it a correctness proof, not a quality gain | P-X6 hash comparison |
| Pelias beats a published 4-bit file | Not for the untrained base: an imatrix or "dynamic" upload may score better than a plain Q4_K_M. A gain must come from (a) Mode Q training, (b) raising chosen core classes within the byte budget (D22), or (c) a calibration step, which is a non-goal today (trainer master 3.2) and would be reopened only by the author | The benchmark below, per variant |
| Near the unquantized model | The target of the K-gates (8.6) | The bf16 reference (next table), and published scores of the bf16 model |
| Speed near the 88.2 to 92.0 tok/s ceiling | DERIVED only. The bus floors are MEASURED (47.62 GB/s dual bus, 13.34 µs doorbell), but every recorded tok/s came from the patched llama.cpp build; the Pelias engine has not run | Step 10 gates; the speed tool at short, 4 k and 12 k, 3 launches per side |

**The bf16 reference on this machine (new probe P-Q11):** the layer-major streaming built for training can run the full bf16 model layer by layer from the safetensors shards (71,903,645,408 bytes) through a 12 GB card, so the true unquantized logits can be computed here, slowly, instead of relying only on published scores. Its time per pass is unmeasured (it is bounded by the drive's read rate, 2.99 to 3.06 GB/s on the WSL2 disk, MEASURED, and by compute). DECISION (proposed).

**Files compared** (all served by the same Pelias engine, and by the llama.cpp oracle as a cross-check):

| # | File | Source |
|---|---|---|
| F0 | bf16 (reference logits through P-Q11) | The author's shards |
| F1 | Pelias base Q4_K_M | Stage A |
| F2 | Stock Q4_K_M of the same shards | llama.cpp oracle, test only |
| F3 | Official 4-bit upload of the model, if one exists | Hugging Face, supplied by the author (the program never downloads) |
| F4 | Unsloth's 4-bit upload(s) | Hugging Face, supplied by the author |
| F5 | Pelias trained (Mode Q), same variant | Path T |
| F6 | Pelias with raised core classes within the budget | D22 option |

**Measures:** KL and top-1 agreement against F0 on the 16 × 1,024-token set and on a held-out set; perplexity; the published task benchmarks of the model card where an open harness exists (scores compared only when settings match the card; otherwise reported as "not comparable"); the author's own gauntlet and client matrix; decode and prefill speed at short, 4 k and 12 k with spread over 3 launches; peak VRAM. No pass mark is invented for the comparisons: the table is the result.

---

## 17. Numbers appendix (the reference machine and model)

Everything here is copied from the audited sources; the full set (491 numbers, each found in the trainer, engine or master files when audited on 2026-10-25, backup RUNBOOK entry 12) is Section 4 of `docs/PELIAS_SYSTEM_MASTER.md`.

### 17.1 Machine and toolchain

| Item | Value | Label |
|---|---|---|
| GPU | GeForce RTX 4070 12 GB (AD104, sm_89), 46 SMs, 5,888 CUDA cores, 184 tensor cores, 36 MiB L2, 192-bit, 504 GB/s stock, 200 W | PUBLISHED |
| GPU overclock | Memory +1000 MHz (11,501 MHz reported under load), core held at 2,760 to 2,775 MHz under load at 965 mV; no ECC | AUTHOR-STATED; MEASURED (P-H3, P-H4) |
| CPU | Ryzen 7 5800X3D, 8 cores, 16 threads, AVX2 (no AVX-512), 96 MB L3 | PUBLISHED; MEASURED (census) |
| RAM | 4 × 16 GB DDR4-3200 (G.Skill F4-3200C16-16GVK), dual channel, 51.2 GB/s peak | PUBLISHED; MEASURED (Windows facts run) |
| Board, bus | PCIe 4.0 x16; Resizable BAR on, BAR1 16,384 MiB | MEASURED |
| Drives | C: Samsung NVMe (Windows and the WSL2 disk); D: WD SN580 NVMe at PCIe 3.0 x2 (archives, WSL2 swap) | MEASURED (HWiNFO, records) |
| Windows | Windows 11 Pro build 26300; HAGS off; display timeout default (2 s); power plan Balanced; fixed 32 GiB page file | MEASURED |
| WSL2 | Ubuntu, kernel 6.18.54.1; `.wslconfig`: memory 48 GB, processors 16, swap 48 GB on D:, `transparent_hugepage=always swiotlb=65536` | MEASURED |
| Toolchain, WSL2 | Driver 617.42 (CUDA 13.4 user mode); nvcc 13.0.88; GCC 15.2.0; Clang 21; CMake 4.2.3; Ninja 1.13.2; Git 2.53.0 | MEASURED |
| Toolchain, Windows | nvcc 13.4.92; MSVC 19.44.35229.0 (Visual Studio 18 Professional, toolset 14.44.35207); CMake 4.4.3; Ninja 1.13.2; Git 2.55.0.windows.5 | MEASURED (2026-10-23 01:09, the author's PowerShell; configure log) |
| Avoid | CUDA toolkit 13.2 | AUTHOR-STATED |

### 17.2 Measured floors (WSL2 / Windows)

| Probe | Result |
|---|---|
| Free VRAM after the context (P-H8) | CUDA 10.83 / 10.85 GiB; context 154 / 138 MiB; **ceiling 10.7 GiB** (D26, D28) |
| Tensor rates, sustained (P-H3) | bf16 65.2 TFLOPS; fp16 64.9; int8 259.5 TOPS; fp8 129.8 |
| VRAM bandwidth (P-H2) | Read 523.3 / 534.0 GB/s; write 447.5 / 454.0 |
| Host DRAM read (P-H2, 02, P-H6) | 1 thread 37.1 / 37.9 GB/s; 4 threads 43.7 to 45.81 / 44.8 (P-H2); CPU alone with 4 readers 46.4 on Windows (P-H6 run log) |
| PCIe (P-H5) | Pinned 26.3 GB/s both ways; zero-copy read 26.3; 4 KiB copy 73 / 31 µs |
| Dual bus (04) | CPU 26.56 + GPU zero-copy 21.06 = 47.62 GB/s (93.0 % of 51.2) |
| Handoff (03) | Stream sync 94.4 µs; doorbell 13.34 µs median, 18.71 µs p99; flag 3.36 µs |
| Doorbell round trip, one resident GPU thread (P-E16, 2026-10-26) | 2.40 / 2.40 µs median (p99 2.70 / 2.73) with the words in pinned system RAM; 2.30 with either word in CPU-visible VRAM (Windows); launch and synchronize 30.5 / 34.0 |
| Pinned memory (P-H7) | 32 GiB held on both; allocation 0.76 to 1.0 / 0.069 s per GiB |
| Graphs (P-H9) | Replay 0.716 / 0.707 µs per kernel; eager 4.70 / 2.18 |
| Overlap (P-H12) | 0.47 / 0.52 unflushed; 0.92 flushed; copy pair 0.84 |
| Model file read, 22.13 GB Q4_K_M on the C: NVMe (P-M4, 2026-10-26) | Windows: unbuffered 2.51 GB/s (1 MiB reads) and 3.08 (16 MiB), file cache 12.0 / 12.4, memory-mapped 4.6 to 4.7; WSL2 through `/mnt/c` (virtiofs): 1.54 to 1.59 cold and warm alike (the bridge is the limit; the Windows cache is outside WSL's control), memory-mapped 1.22. Load time for the file: about 7 s Windows cold, 14 s WSL2, 2 s from the Windows cache (DERIVED from these). Single-threaded synchronous reads; a loader with several reads in flight is unmeasured. The earlier 1.9 to 2.1 GB/s cold read was the D: drive (PCIe 3.0 x2; author, 16:28) |
| Long hold, soak, canary, display timeout | All passed with zero mismatches and zero errors |

### 17.3 Model

753 tensors; 21,713,463,296 bytes; alignment 32; data at byte 10,991,104. Expert layer 432 MiB (down Q4_K) or 498 MiB (down Q6_K: layers 0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39 and `blk.40`). All routed experts: 18,600 MiB (18.164 GiB) for the 40 trunk layers, 19,098 MiB (18.65 GiB) with `blk.40`. Core: 1,409.0 M parameters, 820.3 MiB served, 1,102.3 MiB all Q6_K, 1,427.7 MiB all Q8_0, 2,687.5 MiB bf16. Active parameters per token 2.94 B. bf16 source 71,903,645,408 bytes in 26 shards (READ-SRC, DERIVED, MEASURED as labelled in the sources).

### 17.4 Engine record

Ceiling 92.0 to 88.2 tok/s (DERIVED). Recorded (tier build): WSL2 Cell A 37.9 short / 42.1 at 12 k; Windows Cell A 47.3, 46.8 short / 53.7, 54.1 at 12 k; WSL2 best cell 53.5 / 53.0 / 49.0. Bank peaks (whole card): 7.5 GiB bank 10.40 to 10.44 GiB at the default micro-batch and 10.67 to 10.69 at `-ub 2048`; 8.0 GiB bank 11.0 to 11.19 (over the ceiling). Prompt 1,670 tok/s at 12 k (tier), 2,238 (stock). MTP draft 2/3: 51.5 to 52.9 tok/s at 12 k against 45.4.

### 17.5 Batch C kernel results (2026-10-26, Windows native and WSL2)

Raw files, every revision: `data/probe_results/batch_c/` (README there lists each run). The table gives Windows; WSL2 ran the final source (revision 12) the same day and matched within the run-to-run spread: P-K1 17/20 shapes (best 0.98x cuBLASLt), int8 v10 0.98x to 1.01x of Windows per shape and 1.45x to 1.84x the bf16 path including norm and conversions, P-K2b and P-Q10 the same; P-K3 452 to 460 GB/s at a memory clock of 10,251 MHz against 11,251 MHz in the Windows P-K3 runs (the ratio of the clocks, 0.91, matches the ratio of the plain reads, 0.92), so the platform difference there is the clock, not WSL2. Source: `tools/probes/batch_c/`. Weights and activations synthetic; Q4_K arithmetic from llama.cpp (READ-SRC, 5.3); peaks from P-H3 at the author's overclock (bf16 65.2 TFLOPS, int8 259.5 TOPS, fp8 129.8).

| Probe | Result (MEASURED) | Rule | Outcome |
|---|---|---|---|
| P-K1 bf16 GEMM (Q4_K dequantized to a bf16 scratch, own tiled mma kernel) | 0.48 to 0.88 of the bf16 peak; 0.72x to 0.96x cuBLASLt; >= 0.85x on 17 of 20 shapes (r2). Below: attn_k M=2048 0.72, router M=2048 0.846, M=4096 0.73 | Adopt own kernel at >= 0.85x; reopen if efficiency < 0.35 | Own kernel adopted on 17 shapes; the three small ones keep the C16 rule |
| P-K2 int8 path (Q4_K unpacked to int8, mma s8) | Best kernel v10 (r10/r11): 96.3 TOPS best, 0.25 to 0.37 of the int8 peak, 1.50x to 2.00x the bf16 GEMM, 1.39x to 1.84x including norm and conversions on all 20 shapes; correct on all shapes (max relative error 3.2e-6 against fp32 on the same inputs); cosine against the bf16 path 0.999974 | Mode F default at >= 1.3x and KL no worse (P-Q6) | Speed rule met on all 20 shapes; KL still needs P-Q6 |
| P-K2b (r12) | Split-K on the low-tile shapes: at best 1.04x; bf16 output 0.92x to 0.99x; MoE experts 256 x M=64 in one grouped launch: int8 39.2 to 43.3 TOPS, 2.5x to 3.0x the bf16 kernel grouped the same way (4.7x to 5.0x with the bf16 path's weight dequantization), 2.5x to 2.8x faster than one expert at a time; layer order norm -> router -> q/k/v -> gate/up: L2 reuse of the shared input +0.6 to +1.2 % for the whole slice; int8 slice about 1.7x the bf16 slice | — | Grouped launch adopted for experts; no split-K; fp32 output |
| P-K3 streamed dequantization | Q4_K 487 of 501 to 506 GB/s, Q6_K 496 to 501 of 510 GB/s (96 to 98 % of a plain read) | ALU-bound below 60 % | Memory-bound: in-register dequantization costs no bandwidth |
| P-Q10 fp8 data gradient | e4m3 cosine 0.99965 to 0.99976 (>= 0.999); e5m2 0.9986 to 0.9991 (< 0.999); speed 1.99x from the peaks (DERIVED) | Adopt at >= 0.999 and >= 1.3x | e4m3 passes; e5m2 does not |

Why the int8 path stops near a third of its peak (MEASURED with Nsight Compute on r8 to r10, attn_qkv M=4096): tensor busy = 16 cycles per m16n8k32 / (instructions per mma x cycles between a warp's instructions / warps per scheduler). The split is close to an accounting identity; the cause comes from what moved the speed. Fewer instructions per mma moved it (v2 24.7 -> v7 17.4 -> v8 16.8 -> v10 13.4; 27.7 % -> 37.7 % busy); more warps did not (v9 and v11: each warp's waits grew, mostly at barriers and the shared-memory queue). The remaining instructions are the exact Q4_K sub-block scale (a 6-bit integer per 32 weights, applied as an integer multiply-add per accumulator) and its bookkeeping. GeForce Ada runs bf16 and fp16 with fp32 accumulation at half rate, so int8 is the only full-rate path, and the Q4_K scales do not fit inside an int8 operand.

Adopted for the engine (DECISION 2026-10-26, author's go on each revision): dense layers use int8 v10 (3 stages, fp32 output); MoE experts at small M use v12's 64x128 tile in one grouped launch; the activation quantization is fused into the RMSNorm that produces the activations (no separate pass: a separate pass costs a full read at 272 to 431 GB/s cold, the fused norm costs the same as a norm to bf16). Open: O12 to O14 (18.2).

---

## 18. Conflicts settled, decisions open (R27)

### 18.1 Conflicts between the sources, and how each is settled

Sources compared: the reference profile JSON (P), the parameter dictionary (D), the server parameter sheet CSV (S), the universal and ground-up specifications (U), the launch records and Modelfile (L, READ-SRC), the measured records (M).

| # | Item | What the sources say | Settled as | Why |
|---|---|---|---|---|
| C1 | Server port | P, D, U: 8012; S: 8080; L: 8012 (llama.cpp) | **8062** | A8 |
| C2 | Decode threads | P, L: 6; S: 7 (physical − 1) | Reference profile 6, labelled "measured on the llama.cpp build"; the Pelias engine's own count (the decode thread plus one cold worker per remaining physical core = 7 on the 5800X3D) is re-measured by the E-A1 sweep; unknown CPUs start at physical − 1 | Every recorded run used 6 (L); the Pelias threading model differs |
| C3 | Worker spin (`cpu.poll_level`) | D, P: 50; S: 100 | 100 on the reference, DECISION (proposed); A/B at step 10 | The measured doorbell latency assumes spinning workers; neither value was measured |
| C4 | Non-temporal prefetch | D, P: off; S: on | Off | Hypothesis E-A3, UNVERIFIED |
| C5 | Thinking default | L, D, P: on; S: off | On | The launcher's `{"enable_thinking":true}` (L) |
| C6 | Default `max_tokens` | L, D: −1 (unlimited); S: 256 | −1 | L |
| C7 | Timeouts | D: read 60 s, idle 300 s; S: one 300,000 ms value | D's split | Two different timers |
| C8 | Host loading | S: `load_mode none`; P: `host_alloc` | `host_alloc` (pinned pool); a probe compares at step 12 | "none" is a llama.cpp option; its 2× prefill effect has no isolated cause (P-E11) |
| C9 | Hot bank and micro-batch | Windows launcher: 7.0 GiB (`MAX_MIB` 7300), `-ub 4096`; best WSL2 cell: 7.5 GiB (`MAX_MIB` 7800), `-ub 2048`; P: 7.5 / 2048 | Reference profile 7.5 GiB with `-ub 2048`; 7.0 / 4096 kept as the alternative | Best recorded cell. Its whole-card peak of 10.67 to 10.69 GiB is compared with the whole-card planning figure, not with the process-only 10.7 GiB ceiling (4.3, decision 7); O1 (is 7.5 a ruling?) stays open |
| C10 | Context range | S: up to 131,072; D: up to 262,144 | Default 32,768; 131,072 after E9; the model's 262,144 accepted but marked unmeasured | READ-SRC and E9 gate |
| C11 | GPU zero-copy cold share | D, P: 0.44 when on; S: also "0.60 = Pass 1 default" | 0.44, and the third branch off until the E12 gate | DERIVED from measured rates |
| C12 | Pinned pool | P: auto; S: max 32 GiB | Auto, capped at the measured 32 GiB | P-H7 |
| C13 | Interface | U: Dear ImGui (O9) | Web interface | A5 |
| C14 | Kernel tolerances | U 10.2 vs trainer master 6.11 | The trainer's definitions govern; the universal ones are reported for information | They use different references (16.2) |
| C15 | GPU backends | U: CUDA, HIP, SYCL/Level Zero, Vulkan | CUDA; Vulkan proposed; HIP and Level Zero later | 3.1 |
| C16 | cuBLASLt | U: allowed if own GEMM < 0.85×; the Batch C source links it (MEASURED 2026-10-26: own bf16 GEMM >= 0.85x on 17 of 20 shapes, 17.5) | Test-only comparison; not in the product | R4 ("no additional dependencies") |
| C17 | "Avoid CUDA 13.2" label | U: PUBLISHED; records: AUTHOR-STATED | AUTHOR-STATED | Trainer master |
| C18 | Training length | D: 8,192 window; the author's builder: rows ≤ 4,096 | Both: window 8,192; the author's rows are ≤ 4,096 | Different limits |
| C20 | The port inside the training data | The author's dataset (built 2026-10-19) was made with `--server-port 8012` (READ-SRC, audit `options.server_port`) | Rebuild with `--server-port 8062` before the next training run; the Pelias coach writes `server.port` into the data automatically | A8: the trained model should name the port it is served on |
| C19 | Windows CMake and Ninja | `PELIAS_SYSTEM_MASTER.md` Section 6 (2026-10-25): "4.4 series", "Ninja version not recorded" | CMake 4.4.3 and Ninja 1.13.2 (MEASURED 2026-10-23 01:09) | That record was missed on 2026-10-25; corrected in this repository's copy |
| C21 | Zero-copy third-branch gate | Blueprint E12: combined ≥ 1.25 × CPU alone | Restated (10.4): GPU share ≥ 15 GB/s, combined ≥ 1.0 ×, decode faster with the branch on over 3 launches | 1.25 × 45.9 = 57.4 GB/s exceeds the 51.2 GB/s physical peak (engine ledger, RESULTS §1) |
| C22 | The VRAM ceiling basis | Profiles compared whole-card peaks with the process-only ceiling | Every plan states its basis; profiles carry `basis` | Decision 7, ruled 2026-10-24 (4.3) |
| C23 | Expert-stream time per step | 1.49 s beside 79 fetches / 35.8 GiB | 1.47 s for 79 fetches; 1.49 s is the trainer master's 80-fetch bound (8.2) | Arithmetic |
| C24 | 46.4 GB/s Windows host read | Labelled P-H2 | It is the CPU-alone reading of the Windows P-H6 run log (4 readers); P-H2 best is 44.8 | Raw run log |

### 18.2 Decisions still open

| # | Question | Proposed answer | Needed by |
|---|---|---|---|
| O1 | Is the 7.5 GiB bank a ruling, or the best step under the ceiling? | Derived; the planner may choose any size that fits (7.6) | Step 10 |
| O2 | Build Mode L after its probes? | After P-L1, P-L2, P-E2, P-E3 | Step 15 |
| O3 | What a blank parameter answer does | Knowledge-only rows; base-model drafts only with the user's OK | Step 12 |
| O4 | May Expert mode switch off the clean-stop guard? | Limits only | Step 12 |
| O5 | May Claude read the author's verification script? | — | Step 12 |
| O6 | May Claude look up the clients' published request formats? | — | Step 11 |
| O7 | Support-bundle wording | Counts only | Step 12 |
| O8 | The llama-server catalogue | Answered by design: unknown fields counted from real traffic, plus a read-only catalogue for the replay tests | Step 11 |
| O10 | Special-token text inside user content | Pass and count | Step 11 |
| O11 | The E8 fallback to a llama.cpp patch series | Superseded by A1 and A3; at E8 the author decides with the timers in hand | Step 10 |
| N1 | Vulkan as the second backend | Yes (3.1) | Step 13 |
| N2 | pelias-master as the code repository | Yes (14.1) | Step 2 |
| N3 | Installer packaging | Own self-extracting installer; `.deb` and `.tar.gz` | Step 14 |
| N4 | Own browser driver to replace FlareSolverr | After the core works | After step 15 |
| N5 | First dense test model | The author chooses | Step 13 |
| N6 | CUDA architecture list and the static-runtime licence check | sm_75 to sm_120 plus PTX; licence read before release | Step 14 |
| N7 | A Windows 10 test machine | — | Step 15 |
| N8 | Local-API protection (11.1, 12.2) | As written; PG14 | Step 11 |
| N9 | One GPU owner at a time (11.6) | Server unloads before a job; governor checks across processes | Step 11 |
| N10 | Windows 10 and the CUDA 13.4 driver minimum | Check the toolkit's minimum driver against the final Windows 10 driver; fall back to the 13.0 toolkit for Windows | Step 15 |
| N11 | SPIR-V compiler for the Vulkan backend | glslang at build time only | Step 16 |
| O12 | Recompute the 8.8 speed prediction with the measured GEMM efficiencies (17.5) | Redo the 8.8 derivation with P-K1 and P-K2 as inputs | Step 10 |
| O13 | expert down (K = 512) runs at 0.27 to 0.29 of the int8 peak: per-tile fill and drain over only 8 K steps (split-K, 2 stages and bf16 output measured, none helped) | A persistent kernel that loads the next tile while the current one drains; probe first | Step 5 |
| O14 | Grouped expert launch with real routing (unequal rows per expert) | Measure with the router's actual counts once the engine routes | Step 7 |
| O15 | CUDA doorbell ordering (partly answered 2026-10-26: each ordering step costs about 1 us; store only 1.2 us, release store 2.0) | Measure the two untested correct forms (volatile poll with release store; fence then store) before the engine's doorbell is written | Step 10 |
| O17 | ReBAR-off path validated on real hardware (BIOS switch, or a native-Linux driver setting) | Optional: on a later build day, run the startup probe and the doorbell / upload paths with ReBAR off and adjust | After step 10 |
| O20 | A "Pelias mix": a 4-bit variant meant to be MORE accurate than the stock Q4_K_M (author, 2026-10-26 20:30: the benchmark's question is accuracy, not size) | Built on the proven stock quantizer (P-E6): candidates are higher types for the most sensitive tensors within a stated byte budget, importance-weighted quantization (needs a calibration step, reopening the trainer master 3.2 non-goal by the author's word), or both. Judged only by measurement: KL divergence and top-1 agreement against the original bf16 model on the same text, next to the stock Q4_K_M, the official and the Unsloth 4-bit files (16.4), with file size stated | After stage C of step 3 (the original weights converted) |
| O20 | A "Pelias mix": a 4-bit variant meant to be MORE accurate than the stock Q4_K_M (author, 2026-10-26 20:31: the benchmark's question is accuracy) | Built on the proven stock quantizer (P-E6 passed 2026-10-26: byte-identical to the reference on the real 753-tensor model). Candidates: higher types for the most sensitive tensors within a stated byte budget, importance-weighted quantization (needs a calibration step, reopening the trainer master 3.2 non-goal by the author's word), or both. Judged only by measurement: KL divergence and top-1 agreement against the original bf16 model on the same text, next to the stock Q4_K_M, the official and the Unsloth 4-bit files (16.4), file size stated | After stage C of step 3 (the original weights converted) |
| O19 | 2-bit and 3-bit formats (author, 2026-10-26 19:40: "add 2bit and 3bit support") | Q2_K and Q3_K quantizers and dequantizers by the stage A method (re-expressed from the reference, byte-identical through the oracle on synthetic rows and through P-E6), then the engine kernels for them; the IQ2 and IQ3 types after an importance-data step exists | After step 3 (quantizers); kernels with the engine's later format work |
| O18 | MTP (multi-token prediction) head `blk.40`: DECIDED by the author 2026-10-26 18:58, required in the engine (it adds speed and leaves the chosen tokens to the trunk, so accuracy is kept). Already MEASURED on the tier build (17.4): MTP draft 2/3 gave 51.5 to 52.9 tok/s at 12 k against 45.4 without, 1.13 to 1.17x. Open parts: Pelias's own acceptance rate and gain (step 10 measures them against that record); whether the head's routed experts (498 MiB, 17.3) earn bank space by hits like any layer's (the planner puts them in the expert pool and keeps the rest of the head always resident; its size comes from `pelias plan` on the served file); whether MTP co-training (Section 8, off by default) is needed after a fine-tune | Measure in step 10 against plain decode; turn on co-training only if a fine-tuned model's acceptance rate drops (measured) | Step 10 |
| O16 | P-M5 (RAM busy % and bytes served per level) | Deferred to step 10 (author, 2026-10-26 17:26): the engine counts its own bytes against the measured 46.4 GB/s host-read ceiling; no third-party counter tool | Step 10 |

---

## 19. Sources

### 19.1 Project documents

pelias-trainer: `docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`, `docs/PELIAS_TRAINER_MASTER_NOTES.md`, `README.md`. pelias-engine: `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`, `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, `docs/PELIAS_ENGINE_LINEAGE.md`, `docs/spec/` (11 files), `benchmarks/RESULTS.md`, `support/archive/` (probe results and launch records). Backups: `HANDOFF_20261024_SERVER_GROUNDUP.md`, `RUNBOOK_SERVER_GROUNDUP.md`, `PELIAS_PROPRIETARY_MASTER_SERVER.md` (local only), `PELIAS_SYSTEM_MASTER.md`.

### 19.2 The author's attachments and drive files used for this document

Attachments of 2026-10-26: `pelias_reference_profile.json`, `pelias-parameter-dictionary.md`, `pelias_server_master_parameters.csv`, `pelias-universal-build-master.md`, `pelias_ground_up_master_specification.md`, and copies of the documents in 19.1. Earlier attachment: `PCIe_BAR1_Host-to-VRAM_Architecture_and_Verification_Guide.md` (2026-10-22). From the author's drives (exact locations in the git-ignored `profiles/local_only/LOCAL_SOURCES.md`): the 330-parameter registry and its builder (version 2.0.0, 2026-10-19); the Windows interim launcher (2026-10-24); the best-cell launch record; the author's model file for the earlier serving setup; the earlier web-tools script; the interface work of 2026-10-24 (logo, spear stages, baked textures, the lithic header, the texture baker); FlareSolverr 3.5.2 and master; the nine trainer proposal files of 2026-10-21 (already folded into the trainer master 2.9); the Batch C probe source (2026-10-26 01:50, not yet reviewed or run).

### 19.3 Published sources

- S1: NVIDIA, CUDA GPU Compute Capability, https://developer.nvidia.com/cuda-gpus (fetched 2026-10-26)
- S2: NVIDIA, GeForce graphics cards comparison, https://www.nvidia.com/en-zz/geforce/graphics-cards/compare/ (fetched 2026-10-26)
- S3: AMD, ROCm/HIP SDK system requirements (Windows), https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html (fetched 2026-10-26)
- S4: AMD, ROCm system requirements (Linux), https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html (fetched 2026-10-26)
- S5: List of CPUs with AVX-512 instructions, https://www.eatyourbytes.com/list-of-cpus-with-avx512/ (fetched 2026-10-26; secondary source)
- S6: Tom's Hardware, Intel announces the Arc B580 and Arc B570, https://tomshardware.com/pc-components/gpus/intel-announces-the-arc-b580-and-arc-b570-gpus (fetched 2026-10-26)
- S7: GGUF quantization guide, https://tonisagrista.com/blog/2026/quantization (fetched 2026-10-26; secondary source)
- S8: Guru3D, NVIDIA ends Windows 10 Game Ready driver support in October 2026, https://guru3d.com/story/nvidia-ends-windows-10-game-ready-driver-support-in-october-2026/ (fetched 2026-10-26)
- S9: KTransformers, CPU/GPU heterogeneous inference and fine-tuning for MoE models, https://github.com/kvcache-ai/ktransformers (fetched 2026-10-26)
- S10: Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems, arXiv 2512.16473, code github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference (fetched 2026-10-26)
- S11: HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference, arXiv 2504.05897, https://huggingface.co/papers/2504.05897 (fetched 2026-10-26)
- S12: PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU, arXiv 2312.12456 (fetched 2026-10-26)
- S13: USAF sparse MoE fine-tuning, Hugging Face forums post of 2026-07-05, https://discuss.huggingface.co/t/if-your-gpu-can-run-inference-it-is-now-also-capable-of-performing-fine-tuning/177456, code github.com/tsuyu122/usaf (fetched 2026-10-26; claims not benchmarked)
- S14: Qwen3 MoE `--n-cpu-moe` on 12 GB VRAM, https://aiweekly.co/alerts/qwen3-moe-cpu-expert-offload-doubles-decode-speed-on-12gb-vram (fetched 2026-10-26; secondary source)
- S15: Blink: CPU-free LLM inference by delegating the serving stack to GPU and SmartNIC, arXiv 2604.07609 (search result 2026-10-26)
- llama.cpp `ggml/src/ggml-common.h` at `2ca15f5404760548c39e7b92bd43116a09414a1a` (block sizes, read 2026-10-26)
- NVIDIA, CUDA on WSL User Guide, https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html
- The papers cited in the trainer master 2.8 and the engine blueprint (arXiv 2608.18261, 2605.27081, 2406.06484, 2412.06464, 2411.09009, 2312.03732, 2408.11743, 2601.14277)

---

## 20. How we got here: start to present

**In the author's words** (2026-10-26 12:11, AUTHOR-STATED): "this started as an effort for me to better create my own llm, when i found this... accidentally. everything said we couldn't train an moe using my type of system, we found a way. then we saw the hardware ceiling limit when running that quantized llm, and we found a way to unify the bus to make it run even better from the way we found to train it... and then even more!"

Dates up to 2026-10-17 are **file modification dates on the author's drives** (exact folders in the git-ignored `profiles/local_only/LOCAL_SOURCES.md`) (MEASURED from the file system; a file's date shows when it was last written, not necessarily when the work began). From 2026-10-18 the timeline is the long master's Part 8.2, where every line has its record. The author's account of working alone, then with Gemini and local models, then with Claude, is AUTHOR-STATED (2026-10-26).

| When | What | Source |
|---|---|---|
| 2026-09-19 to 09-20 | The first training folder of the assistant project: an alignment JSONL, an Unsloth LoRA training script, an export script, serving model files, and a first "AIOS" script | The author's older archive drive |
| 2026-10-01 | An "Architect" model file | The author's older archive drive |
| 2026-10-04 | A llama.cpp clone on the drive | The author's main drive |
| 2026-10-09 to 10-13 | The AIOS blueprints, directives, packaging guides and training orders (versions 1.0 and 1.1) | The author's older archive drive |
| 2026-10-14 | The name **Pelias** first appears in a file name: `1.0 Pelias_Engine_Master_Specification.md` and `1.0 Pelias_Creator_Training.md` | The author's older archive drive |
| 2026-10-17 | The creator project folder: WSL environment, checkpoints, the original parameter script and the 330-parameter registry | The author's main and archive drives |
| 2026-10-18, 23:16 | The first handoff to a reviewing model: fine-tune on one 12 GB card without truncation; a GPU-resident spine with 4-bit experts streamed from pinned RAM | Long master 8.2, 8.3 |
| 2026-10-18, 23:59 | "if we have to write our own dang program from the ground up and call it Pelias, i'm down for it." | Long master 8.3 |
| 2026-10-19 | Trainer v0.1 and v0.2 written; `create_parameters_v2.py` builds the 1,300-row set; the first logged training run (08:52) | Long master 8.2; trainer RESULTS |
| 2026-10-20 | The 5-bit and 4-bit files exported; Q4_K_M chosen; engine hardware floors measured (microbenchmarks 02 to 07); the first tier blueprint; the two private repositories created | Long master 8.2 |
| 2026-10-21 | Nine trainer proposal files (C++/CUDA plans and reviews); the tier built into llama.cpp; the 9-or-more-token crash | The author's proposals folder; long master 8.2 |
| 2026-10-22 | The crash fixed; the engine blueprint (E0 to E13); the E0 timeline and hit counter; author rulings on the ceiling, the accuracy gate and the patches | Long master 8.2 |
| 2026-10-23 | Windows-native runs; the trainer master; the outside research update; the Wave 0 probes on both platforms; the 10.7 GiB ceiling | Long master 8.2 |
| 2026-10-24 | Engine hold (06:07); WSL and Windows versions (06:21); final micro-batch and export rulings (07:39); "ground up means the ground up" (08:15); the long master requested (09:09); the interim launchers and the server runbook | Long master 8.2; handoff of 2026-10-24 |
| 2026-10-25 | The engine records consolidated (1,559-row manifest); the system master with its numbers and the draft-build list; a Batch C probe source written in the early hours of 10-26 (not yet run) | Backup INDEX entries 10 to 13 |
| 2026-10-26 | The author lifts the holds and asks for this masterwork: one ground-up, universal, installable system, port 8062, Fable review before the build, and a first benchmark against the published 4-bit files. Afternoon: build step 1 starts; Batch C rewritten and run on Windows in twelve revisions, the int8 limit measured and the kernels adopted | Section 0.2; 17.5 |

The thread through all of it, in the long master's words: the project began as a **training** problem, the measurements of that problem raised the **inference** question, and the llama.cpp work was the way to learn what a ground-up server must do. Every speed recorded so far came from the patched llama.cpp build; the first Pelias engine run is the experiment that answers the question the whole project has been asking.

---

## 21. Future hypotheses: the unified bus beyond LLMs (author, 2026-10-26 12:19; rebuilt 13:1x with the graphics probe P-G0; updated 17:4x with build step 1: Batch C, P-M4, P-E16)

**Status: hypotheses, not results, except where a row says MEASURED.** Nothing in this section is part of the first build. Every "on paper" increase is DERIVED only from numbers this project measured on the reference machine. For any field other than the LLM, a factor carries over **only if** that field's work has the same bottleneck; each row says what that condition is and how to test it. The author's statements are AUTHOR-STATED. Ruling 2026-10-26 12:52 (AUTHOR-STATED): each field below gets its own scripted probe, run against a stock baseline with unchanged output, after pelias-master is built and running; until then every non-LLM row stays DERIVED.

### 21.1 The three hypotheses

| # | Hypothesis (the author's) | What the records already show | How to test it |
|---|---|---|---|
| H-U1 | The unification of the system bus can be recreated outside LLM use | Nothing in the bus is specific to language models. The doorbell, the hot bank, zero-copy reads, the dual DRAM readers, graph replay and the flush rule were each measured as a **hardware** property by stand-alone programs (microbenchmarks 02 to 07, Wave 0), before any model ran on them. P-G0 (21.2.1) then measured the same primitives through Direct3D 12 and Vulkan, two interfaces no LLM code uses | Port one non-LLM workload (21.3) to `libpelias_substrate` (Layer 0 only: `os/`, `hal/`, `plan/`, `bus/`, `bank/`) and measure it against its own best conventional version on the same machine |
| H-U2 | A very small model running on the CPU could keep the bus tuned (predict what to keep resident, when to hand off), costing almost nothing next to the speed it gains | The residency choice is the lever: in simulation a **recency cache** of experts held 0.86 of the chosen bytes against 0.77 for a fixed set of the same size (DERIVED, simulated on inference traces; trainer README); on the engine's curve about +1 tok/s per +0.01 of hit rate (DERIVED). A predictor already exists as a test tool (`predictor_eval.py` in the engine archive) | The rule a predictor must pass: its CPU time per decision must be smaller than the miss time it saves, measured in the decode loop. Compare three residency policies on one trace and one live run: the static water-fill (today), a coded recency or frequency rule, and the tiny model |
| H-U3 | It can work without any model, if the system is coded to use the bus properly | The current design is already model-free: the hot set is a water-fill solver, the split is a measured ratio, the plan is a solver over probe results. The live hit rate (0.775 to 0.779, 7.0 GiB bank) is close to the simulated 0.806 (MEASURED) | Same test as H-U2. If a coded rule matches the tiny model, H-U3 holds and the model is not needed |

**With and without a model, in one line:** the speed comes from the primitives; a tiny model (H-U2) can only help a machine reach its ceiling sooner on unfamiliar work or hardware, never raise it, and every suggestion it makes is measured before it is kept.

### 21.2 The primitives, and what each one gained here

#### 21.2.0 Through CUDA (the engine's path)

| Primitive | Before | After | Factor | Label |
|---|---|---|---|---|
| CPU and GPU handing work to each other (microbenchmarks 02 / 03, WSL2, 2026-10-20) | 94.4 µs (operating-system stream sync) | 13.34 µs (resident doorbell in system RAM, median; 18.71 µs p99; several mailboxes and back-off per hand-off) | **7.1× lower latency** | DERIVED from MEASURED |
| CPU and GPU handing work to each other, one resident GPU thread (P-E16, 2026-10-26; same run, same machine) | Launch and synchronize: 30.5 µs (Windows) / 34.0 µs (WSL2) median | Doorbell round trip 2.40 µs on both (system-RAM flag; VRAM 2.30). Reply ordering (Windows, P-E16 r4): 2.00 µs with a release store (the cheapest correct way to publish data), 1.20 µs for a plain signal | **12.7× (Windows) / 14.2× (WSL2)** at 2.40 µs; the flag's location changes it by about 4 % | DERIVED from MEASURED |
| Multiplying by 4-bit (Q4_K) weights on the tensor cores (Batch C P-K2, 20 real layer shapes) | bf16 path: weights dequantized to bf16 each call, then a bf16 GEMM (GeForce runs bf16 with fp32 sums at half rate) | int8 path: weights unpacked in registers, activations int8 with the quantization fused into the RMSNorm, own kernel v10 | **1.39× to 1.84×** including norm and conversions (1.50× to 2.00× for the GEMM alone); output cosine to the bf16 path 0.99997 | MEASURED |
| Many small expert multiplies at once (MoE experts, 256 experts × 64 tokens, P-K2b) | One expert at a time: at most 16 to 32 work units for 46 SMs | One grouped launch for all experts (1,024 to 4,096 work units) | **2.5× to 2.8×** over one at a time (launch overhead not counted); grouped int8 **2.5× to 3.0×** the bf16 kernel grouped the same way (4.7× to 5.0× with the bf16 path's dequantization) | MEASURED |
| Removing a pass over memory (activation quantization fused into its producer, P-K2 r11, cold L2) | A separate quantization pass: a full read and write at 272 to 431 GB/s | Fused into the RMSNorm: costs the same as a norm to bf16 | The pass disappears; the int8 path moves from 0.87× to 1.83× of bf16 (separate) to 1.39× to 1.84× (fused), including norm and conversions | MEASURED |
| Data the CPU decodes in registers on the way to the GPU (P-K3) | Plain read of Q4_K / Q6_K bytes | Read and dequantized in registers | 96 % to 98 % of the plain read: 4-bit and 6-bit storage costs no bandwidth to unpack | MEASURED |
| Loading a 22.13 GB model file (P-M4, C: NVMe) | WSL2 through `/mnt/c`: 1.54 to 1.59 GB/s (about 14 s) | Windows native: 2.51 to 3.13 GB/s cold (about 7 s, one read in flight), 12.0 to 12.7 GB/s from the file cache (about 2 s) | **about 2× (cold) to 8× (cached) faster on Windows**; several reads in flight unmeasured | MEASURED / DERIVED |
| Launching many small GPU jobs | 4.70 µs (WSL2) / 2.18 µs (Windows) per kernel | 0.716 / 0.707 µs (graph replay) | **6.6× / 3.1× lower overhead** | DERIVED from MEASURED |
| Copies and compute at the same time | 0.47 (WSL2) / 0.52 (Windows) of ideal overlap | 0.92 (with the flush rule) | **1.96× / 1.77× more overlap** | DERIVED from MEASURED |
| Data in RAM that the GPU needs (skewed access) | An equivalent GPU-only design where every byte crosses PCIe (stock llama.cpp instead computes cold experts on the CPU, so this row is not the stock comparison; the last row is) | The hot 7.5 GiB of 18.6 GiB stays in VRAM; byte hit 0.8273, so 17.27 % crosses | **5.8× less host traffic** (= 1 / (1 − 0.8273)) | DERIVED from a simulated hit rate |
| Order of a streamed loop | 121.5 GiB per training step (v0.2) | 35.8 GiB per step (layer-major) | **3.4× less PCIe traffic** | DERIVED |
| Two processors reading one RAM | CPU alone 45.9 GB/s; GPU zero-copy alone 26.2 GB/s | Both at once 47.62 GB/s | **1.04× over the CPU alone, 1.82× over the GPU alone** | DERIVED from MEASURED |
| Working set larger than VRAM | A 20.22 GiB model on a card with 11.99 GiB | Runs interactively (37.9 to 54.1 tok/s across the recorded tier-build cells) | **1.69× the card's memory** in use at interactive speed | MEASURED |
| End to end, against stock software on the same machine | Stock llama.cpp: all experts on the CPU 31.8 / 31.5 / 31.3 tok/s; best stock layout (`--n-cpu-moe 28`) 41.1 / 40.2 / 38.8 | The hot-bank tier: best WSL2 run 53.5 / 53.0 / 49.0 | **+68 %** over all-CPU, **+30 %** over the best stock layout (short context); matched-run figures in the engine README: +5 % to +7 % (6.1 GiB bank), up to +17 % at 12 k and +28 % short (7.5 GiB bank, single launches) | MEASURED / DERIVED |

#### 21.2.1 Through Direct3D 12 and Vulkan (probes P-G0 and P-G0b, MEASURED 2026-10-26 13:14 to 13:33, three runs)

Machine: the reference RTX 4070 (12,010 MiB), NVIDIA driver 621.01 (Vulkan 1.4.357; Direct3D driver 32.0.16.2101), Windows build 10.0.26300. Resizable BAR on: Vulkan exposes all 12,010 MiB as CPU-visible VRAM, and the Direct3D 12 GPU upload heap is supported. Raw JSON: `data/probe_results/P-G0/` (runs at 17:14:13Z with `gfx_probe` 0.1.0, 17:29:37Z with 0.2.0, 17:33:41Z with 0.2.1; UTC). Each cell is the range of the run medians; the factor is DERIVED from them. The Windows timer counts in 0.1 µs steps, so sub-microsecond medians are read to that step.

| Test | Vulkan without → with | Direct3D 12 without → with | Factor | Compared with CUDA (21.2.0) |
|---|---|---|---|---|
| Submit 1,000 tiny jobs: each recorded and submitted alone → one recorded list replayed | 30.2 to 31.5 → 0.84 to 0.85 µs per job | 40.8 to 47.0 → 0.81 to 0.82 µs per job | **36× to 37× / 50× to 57×** | 3.1× to 6.6× (a single graphics submit costs far more than a CUDA launch) |
| Same, jobs pre-recorded, submitted one by one → replayed (the fair game comparison: engines record anyway) | 30.6 to 31.5 → 0.84 to 0.85 µs | 34.2 to 34.6 → 0.81 to 0.82 µs | **36× to 37× / 42×** | |
| CPU → GPU → CPU hand-off: fence round trip → doorbell flag in VRAM through the BAR (best read style per run) | 58.3 to 58.6 → **0.8 to 1.4 µs** | 62.4 (spin; 69.8 to 70.5 by event) → **0.8 to 1.3 µs** | **42× to 73× / 48× to 78×** | 7.1× (94.4 → 13.34 µs with the flag in system RAM) |
| Same, doorbell flag in system RAM (P-G0b: every memory type with plain, atomic and, on Vulkan, memory-model reads) | **never completes** with any type or read: each run stops at round 0 or 1 of 1,000 | **write-back: never completes** (round 0 or 1); **write-combined: completes 1,000 of 1,000, 0.8 to 1.4 µs** (0.8 to 0.9 µs with the best read) | none / **69× to 78×** (write-combined) | see O-G1 below |
| Upload 64 MiB from application memory: staging copy → CPU write straight into VRAM | 12.5 to 12.6 → 22.5 to 22.7 GB/s | 12.6 to 12.7 → 22.7 GB/s | **1.78× to 1.82× / 1.79× to 1.80×** | 2.23× (pinned against pageable memory) |
| 256 MiB copy and a matched compute job: one after the other → copy and compute queues at once | 20.45 to 20.48 → 10.33 to 10.39 ms | 20.45 to 20.47 → 10.32 to 10.34 ms | **1.97× to 1.98× / 1.98×** (ideal 1.99×) | 1.96× / 1.77× |

What P-G0 and P-G0b add, stated plainly:
- **The flag's location matters more than the API.** The doorbell answered in 0.8 to 1.4 µs with its flag in VRAM (both APIs) or in write-combined system RAM (Direct3D 12), against 13.34 µs for the CUDA doorbell with its flag in system RAM. Hypothesis H-G1: moving the CUDA doorbell's flag into VRAM through the BAR cuts the engine's hand-off by a similar order. Test **P-E16 (DECISION (proposed))**: allocate the flag in CPU-visible VRAM (a Vulkan `DEVICE_LOCAL | HOST_VISIBLE` allocation exported to CUDA by external-memory interop) and rerun microbenchmark 02. The flag is 4 KiB, so it should also fit the 256 MiB BAR window cards have without Resizable BAR (UNVERIFIED; P-E16 runs with ReBAR on and off). The Direct3D 12 write-combined result shows a system-RAM doorbell needs no BAR at all where the driver maps it that way. **P-E16 result (2026-10-26, MEASURED, `data/probe_results/P-E16/`): H-G1 is not supported as stated.** A single CUDA doorbell round trip takes 2.40 us median (p99 2.70) with both words in pinned system RAM on Windows and 2.40 (p99 2.73) on WSL2, and 2.30 (p99 2.50 to 2.60) with either or both words in CPU-visible VRAM: the location changes it by about 4 %. Unlike Vulkan and Direct3D 12 write-back memory, CUDA's system-RAM flag completes every ring. The 13.34 us of microbenchmark 03 came from that benchmark's heavier hand-off (several mailboxes, back-off), not from the flag's location, so the engine's hand-off floor is about 2.4 us, 12.7x below launch and synchronize (30.5 us Windows, 34.0 WSL2). CUDA's 2.3 us is still about twice P-G0's 0.8 to 1.4 us; the ordering around the reply turned out to be the cost (O15, r4): fence-store-fence 2.40 us, store-fence 1.40, store only 1.20 (inside P-G0's range), acquire load with release store 2.00. A reply that publishes data needs its ordering before the flag, so the cheapest correct form measured is the release store at 2.0 us; plain signals can use the 1.2 us store. On Windows the doorbell kernel's launch must be flushed (`cudaStreamQuery`), or the driver holds it.
- **O-G1, cause (DERIVED from the MEASURED pattern; the driver's mapping itself is undocumented, so UNVERIFIED at that level).** Every failing variant stops at round 0 or 1: the GPU reads the flag from RAM once, then keeps answering from a cached copy that the CPU's later writes never reach. Only mappings the driver keeps out of the GPU's cache work: VRAM on both APIs, and the Direct3D 12 write-combined custom heap. Atomic and memory-model reads do not get past the cached copy. On this Vulkan driver no system-RAM memory type behaves that way. In 0.2.0 a 1-round run printed as a result (the JSON was right); 0.2.1 counts a variant only when all rounds complete.
- **CUDA behaves differently (P-E16).** A CUDA kernel's plain volatile reads of pinned system RAM see the CPU's writes every time, so CUDA needs neither VRAM nor the BAR for its doorbell: 2.40 µs in system RAM, 2.30 in VRAM. What costs time is the ordering around the reply, about 1 µs per ordering step. Graphics APIs need the flag in VRAM (or Direct3D 12 write-combined RAM); CUDA does not.
- **Copy and compute already overlap almost perfectly** through dedicated copy queues on this machine (1.98× of 1.99×), so for games the overlap gain applies only to engines that do not use a copy queue.
- **A spinning doorbell occupies one GPU thread** for as long as it waits; its power cost is unmeasured (P-W1 covers it).

#### 21.2.2 Two honest limits

**The RAM's own peak does not grow.** 3,200 MT/s × 8 bytes × 2 channels = 51.2 GB/s is set by the memory clock and the channel count; the two readers together already reach 93.0 % of it. What the bus changes is **how much useful work each trip to RAM does and when the trips happen**: hot data served from VRAM (about 504 GB/s, PUBLISHED) or the CPU's cache instead of RAM, 4-bit weights carrying four times the model per byte, each byte moved once, reads started ahead of need, no idle gaps during hand-offs (the engine's timers found zero CPU/GPU overlap and 4.6 to 5.0 ms of hand-off per token, MEASURED), and reads in long sequential runs instead of scattered ones. Neither the useful bandwidth across VRAM, cache and RAM nor the fraction of each token the RAM spends busy has been measured. Test **P-M5**: RAM busy % and model bytes served per second from each level, with and without the bus. Deferred to step 10 by the author (2026-10-26 17:26, O16): the engine counts its own bytes per level against the measured 46.4 GB/s host-read ceiling, with no third-party counter tool.

**The full Pelias engine has not run**: the end-to-end rows come from the patched llama.cpp build.

### 21.3 Where it could apply, and the on-paper increase for each

The column "Condition" is the bottleneck the field must share for the factor to carry over; "Factor" is a primitive's value from 21.2, never a new number. Where a factor depends on that field's own hit rate h, host traffic falls to (1 − h) of the original, a factor of 1 / (1 − h); our h was 0.8273.

| Field | What maps onto the bus | Condition | On-paper increase | Test |
|---|---|---|---|---|
| Recommendation systems | Embedding tables much larger than VRAM, with a few popular rows: hot rows in the VRAM bank, cold rows read by the CPU at the same time | Lookups are skewed (a power-law popularity) and the job is transfer-bound | Host traffic 1 / (1 − h); 5.8× at our h | Replay a public click log through Layer 0 |
| Vector search and retrieval (RAG) over indexes larger than VRAM | Hot clusters resident; cold clusters scanned by the CPU's vector units in parallel; doorbell joins results; 4-bit or int8 vectors scored on the tensor cores | Queries concentrate on part of the index | 1 / (1 − h) less traffic; 12.7× to 14.2× faster CPU/GPU joins than launch-and-synchronize (CUDA doorbell, P-E16; the flag in VRAM adds about 4 %); quantized scoring 1.39× to 1.84× over bf16 where scoring is a GEMM (P-K2, UNVERIFIED for this workload) | Index of 2 to 5 times VRAM, recall held equal |
| Databases and analytics | GPU scans and joins on tables that stay in RAM (zero-copy), copies overlapped with compute | Scans are transfer-bound and split into many chunks | Up to 1.96× (WSL2) / 1.77× (Windows) more overlap; 1.98× through dedicated copy queues (P-G0) | A standard analytics query set at a size above VRAM |
| Graph analytics | Frontier-based traversal passes work between CPU and GPU many times | Many fine-grained hand-offs per iteration | 12.7× to 14.2× lower hand-off latency (CUDA doorbell against launch and synchronize, P-E16); 42× to 78× through Vulkan / Direct3D 12 with the flag in VRAM (P-G0) | Breadth-first search on a graph larger than VRAM |
| Scientific simulation, CFD, FEA, out-of-core solvers | Sweeps over a domain larger than VRAM; reordering the sweep so each block is loaded once per pass (the layer-major idea) | The sweep is transfer-bound and its order can be changed | Traffic per pass falls by the loop's reuse ratio; 3.4× in our training loop | Out-of-core stencil or sparse solve |
| Genomics | Reference index in RAM, hot regions resident, alignment on GPU and CPU together | Access concentrates on part of the index | 1 / (1 − h); both processors busy | Read alignment against a human reference |
| Rendering, 3D, game asset streaming | Hot textures and meshes resident; CPU writes new assets straight into VRAM through ReBAR | Scenes larger than VRAM; streaming-bound | 1.78× / 1.79× faster per upload (P-G0, MEASURED: 12.6 to 12.7 → 22.5 to 22.7 GB/s) | Virtual-texture streaming benchmark, ReBAR on and off |
| Video and image pipelines | CPU decodes, GPU processes, results handed back per frame | Many small hand-offs and launches per frame | 12.7× to 14.2× lower hand-off (CUDA doorbell, P-E16), 3.1× to 6.6× lower launch overhead (CUDA graphs); 36× to 42× lower per-submit cost through Direct3D 12 / Vulkan replay | A multi-stage frame pipeline at a fixed frame rate |
| Financial risk and Monte Carlo | Thousands of short kernels per batch | Launch-bound | 3.1× to 6.6× lower launch overhead (graph replay) | A pricing batch of many small paths |
| Any quantized-model inference or fine-tuning on a GeForce card (beyond this model) | 4-bit weights multiplied on the int8 tensor path; many small expert or adapter multiplies in one grouped launch; quantization fused into the producer | GEMM-bound work on cards that run bf16 with fp32 sums at half rate (GeForce) | 1.39× to 1.84× per layer over the bf16 path; 2.5× to 3.0× for grouped small multiplies (P-K2, P-K2b) | Batch C on a second GeForce card; a second model through the same kernels (E13) |
| Compression, hashing, storage | CPU and GPU work on the same buffer in RAM at once | Memory-bound on the host | Up to 1.82× over the GPU alone (both readers) | Hash or compress a RAM-resident set larger than VRAM |
| Robotics and real-time control | GPU perception and CPU control exchange results every cycle | Per-cycle hand-off latency matters | 40 hand-offs: 1.22 ms by launch and synchronize against 0.096 ms by the CUDA doorbell (0.080 ms with a release store, 0.048 ms for plain signals; P-E16, Windows); 2.33 ms (Vulkan fence) against 0.032 ms (Vulkan doorbell in VRAM) (DERIVED, 40 × the medians) | A control loop with a GPU perception stage |
| Home and small-office servers | One consumer PC serving models or jobs that today need a larger card or a rented GPU | VRAM, not compute, is what stops the job | Working sets above VRAM run at interactive speed: 1.69× the card's memory measured here; for sparse access the limit becomes host RAM (64 GB here against the card's 12 GB, 5.3×, DERIVED; UNVERIFIED for any workload but MoE) | The client matrix and speed tool on a second machine |
| Data centres | Cheaper or older cards running work that now needs high-memory accelerators | Jobs bound by VRAM size rather than compute or memory bandwidth | No number: a data-centre comparison needs published power and throughput figures that are not in these records | The energy probe below, then a published comparison |

### 21.4 "Any consumer PC as a small data centre", power, and hardware life (AUTHOR-STATED vision)

The author's view: the unification could, in theory, turn a consumer PC into a data centre for its owner; if many users ran their own work this way, data centres would need less power and fewer machines, the demand on all computer hardware would ease, and recent CPUs and GPUs would stay useful longer.

What the records support today, stated plainly:
- **Measured:** one consumer PC (12 GB card, 64 GB RAM) runs a 35-billion-parameter mixture-of-experts model whose file is 1.69 times its VRAM, at 37.9 to 54.1 tok/s across the recorded cells, with CPU and GPU reading the same RAM at 93 % of its peak; the same card's graphics interfaces hand work between CPU and GPU in under 1.5 µs (P-G0) and CUDA in 2.0 to 2.4 µs (P-E16); its 4-bit multiply path runs 1.39× to 1.84× faster than the bf16 path and 2.5× to 3.0× for grouped expert work (Batch C).
- **Not measured:** energy. The 30-minute soak recorded about 107 W at the GPU (MEASURED, P-H4), but no wall-power reading exists for the whole machine while decoding, so no energy per token can be stated, and no comparison with a data centre can be made yet.
- **Not measured:** hardware life. The argument (a card with less memory stays useful for larger work) is a reasoned hypothesis; it becomes a number only when the list of models and jobs each card class can run, with and without the bus, is measured.

**Measurement pathway (probe P-W1, DECISION (proposed)):** measure whole-machine wall power with a plug-in power meter (the user reads it; nothing to install) during the speed tool's runs, giving joules per token at short, 4 k and 12 k, for the tier build now and the Pelias engine later, plus the idle cost of a spinning doorbell; then compare with published energy figures of data-centre inference for the same model class, sources cited. Only after that can "less power" carry a number.

### 21.5 Gaming (author, 2026-10-26 12:24; numbers from P-G0)

**Status:** the building blocks are now MEASURED through Direct3D 12 and Vulkan on the reference machine (21.2.1). No game has been run on the bus, so no fps gain is claimed; the rows below are the measured savings applied to a frame budget, with the condition each needs.

**The frame budget** (DERIVED): 16.67 ms per frame at 60 fps, 6.94 ms at 144 fps, 4.17 ms at 240 fps. A saving turns into fps only if the saved time was on the frame's critical path.

#### 21.5.1 For players (fps, frame pacing, latency)

| Where time goes in a frame | What the bus offers | Measured saving (P-G0) | Condition | Label |
|---|---|---|---|---|
| CPU and GPU waiting on each other (readbacks, occlusion queries, GPU-driven culling results) | Doorbell flag in VRAM (or write-combined RAM on Direct3D 12) instead of a fence wait | 56.9 to 57.8 µs per sync point (Vulkan, 58.3 to 58.6 → 0.8 to 1.4); 61.1 to 61.6 µs (Direct3D 12, 62.4 → 0.8 to 1.3): 0.34 % to 0.37 % of a 60 fps frame, 0.82 % to 0.89 % of a 144 fps frame, per sync point (three runs) | The game has blocking CPU/GPU sync points on its critical path; how many per frame is unmeasured per title; the doorbell keeps one GPU thread busy | DERIVED from MEASURED |
| Many separate submissions per frame | Record once, replay as one list | 29.8 µs per submission merged (Vulkan, 30.6 → 0.85); 33.8 µs (Direct3D 12, 34.6 → 0.82) | Engines that submit many separate lists per frame; well-written engines already submit few, so their gain is small | DERIVED from MEASURED |
| Stutter when new assets arrive | CPU writes straight into VRAM (ReBAR / GPU upload heap) instead of a staging copy | A 64 MiB upload: 5.31 → 2.98 ms (Vulkan), 5.29 → 2.96 ms (Direct3D 12) | Streaming-bound titles; ReBAR on; Windows 11 24H2 or later for the Direct3D 12 heap | DERIVED from MEASURED |
| Streaming while rendering | Copy queue and compute queue at once | 1.98× of an ideal 1.99× (both APIs) | Engines that do not already use a dedicated copy queue | MEASURED |
| VRAM too small for the settings the player wants | Hot set in VRAM, the rest read from RAM on demand | A working set up to 1.69× the card's memory ran at interactive speed for the LLM (MEASURED); for a game UNVERIFIED, because a frame touches far more of its data each frame than a token touches of the model | Assets with skewed use per frame | UNVERIFIED |
| An AI character or assistant running next to the game | The LLM takes only the VRAM the game leaves; experts stay in RAM; the governor's measured ceiling stops it from starving the game | With the measured hit curve, a 2 GiB bank gives a hit rate of 0.43, 4 GiB 0.63, 6 GiB 0.76 (DERIVED, simulated, reference model); about +1 tok/s per +0.01 of hit rate | The game leaves VRAM free; the CPU cores used by the cold experts are not the game's main threads | DERIVED |
| The AI character's own GPU work (reading the prompt, answering) | 4-bit weights on the int8 tensor path; small expert batches in one grouped launch | 1.39× to 1.84× per layer over the bf16 path; 2.5× to 3.0× for grouped expert work (Batch C) | GeForce cards (bf16 with fp32 sums at half rate); prompt-sized batches | MEASURED (kernels); in-game UNVERIFIED |
| Loading the AI model when the game starts | Read the model file natively on Windows, from the file cache when it is warm | 22.13 GB in about 7 s cold and about 2 s cached on Windows, about 14 s through WSL2 (P-M4) | Model on an NVMe drive; enough RAM to keep it cached | DERIVED from MEASURED |
| A game handing work to CUDA (physics, AI, video) | CUDA doorbell instead of launch and synchronize | 30.5 → 2.0 to 2.4 µs per hand-off (P-E16): 0.17 % of a 60 fps frame saved per hand-off, 0.41 % at 144 fps | The game uses CUDA next to its graphics API; the waiting GPU thread stays busy | DERIVED from MEASURED |

What turns this into fps numbers (probe P-G1, DECISION (proposed)): a small test scene in the Vulkan backend that (1) counts sync points and submissions per frame, (2) streams a world larger than VRAM, and (3) runs with and without the bus primitives, recording average fps, 1 % low fps and frame-time variance on the reference machine.

#### 21.5.2 For game developers

| Developer task | What the bus or the trainer offers | On-paper effect | Label |
|---|---|---|---|
| A local coding assistant on the development PC (the author's own use: Visual Studio 2026 with the model on port 8062) | A 35B-class model on a 12 GB card at interactive speed, no second GPU, no cloud | 37.9 to 54.1 tok/s across the recorded tier-build cells; the Pelias engine aims higher (10.7) | MEASURED (tier); target DERIVED |
| Training game-specific models (NPC dialogue, quest text, tools) on the developer's own data | The layer-major trainer: a 35B MoE trained on one 12 GB card | v0.2 trained at 247.85 tok/s (MEASURED); targets 600 to 2,000 tok/s (DECISION) | MEASURED / DECISION |
| Engine work: fewer, larger submissions; GPU-driven readbacks without stalls | The P-G0 numbers as a budget: about 30 to 34 µs per extra submission and 57 to 61 µs per fence round trip on this machine | Measured costs to design against, per API | MEASURED |
| Asset build pipelines (texture compression, mesh processing, lighting bakes) on data larger than VRAM | CPU and GPU working on the same data in RAM at once | Up to 1.82× over the GPU alone on memory-bound steps (dual readers); loop reordering cut traffic 3.4× in the trainer | DERIVED; condition: memory-bound and reorderable |
| Testing on smaller player cards | The VRAM governor can run a plan under a forced lower ceiling (Expert mode, USER-FORCED) | Emulates an 8 GB card's memory limit on a 12 GB card for memory-fit testing (not its speed) | DECISION (proposed) |
| Profiling CPU/GPU hand-offs | The per-token timers and hand-off counters of the engine, reusable as a frame timeline; `gfx_probe` as a per-machine API cost check | Same instruments that found zero CPU/GPU overlap and 4.6 to 5.0 ms of hand-off per token in the LLM (DERIVED, E0) | MEASURED method |
| Writing CPU/GPU signalling code | Measured costs to design against: about 1 µs per ordering step in a CUDA reply (2.40 / 2.00 / 1.20 µs, P-E16); Windows holds a resident kernel's launch until it is flushed; graphics APIs need the flag in VRAM, CUDA does not | Correct and fast signalling chosen by measurement, not by guess | MEASURED |
| Shipping quantized models with a game or tool | Q4_K weights read and unpacked at 96 % to 98 % of plain read bandwidth (P-K3); the int8 tensor path for the multiplies (P-K2) | 4-bit storage without a bandwidth cost; 1.39× to 1.84× per layer over bf16 on GeForce | MEASURED |

### 21.6 End-to-end gains, calibrated by the measured LLM result (author's questions, 2026-10-26 12:29 to 12:43)

A primitive speeds up only the part of a job it fixes, so the end-to-end gain follows Amdahl's law: gain = 1 ÷ ((1 − f) + f ÷ k) − 1, where k is the primitive's factor and f is the share of the job's time spent in that bottleneck.

**Calibration.** f is unmeasured for every field but one. For the LLM, k = 1 / (1 − 0.775) = 4.444 (the live hit rate), and the measured runs give f back-solved: 0.269 (38.8 → 49.0 tok/s), 0.299 (41.1 → 53.5), 0.466 (31.3 → 49.0) and 0.522 (31.8 → 53.5), so **27 % to 52 %**. The table uses those two ends for every field as the best measured guide until each field measures its own f (ruling 12:52). "Whole job" is k itself: the gain if the entire job were that bottleneck, an upper bound only. All values DERIVED; the anchor row reproduces the measured runs by construction.

| Area | What the bus fixes | Factor k | Share 27 % | Share 52 % | Source of k |
|---|---|---|---|---|---|
| LLM inference (anchor; the shares are back-solved from these MEASURED runs) | Hot bank, live hit 0.775 | 4.44× | +26 % | +68 % | 21.2 |
| Recommendation systems | Host traffic | 5.79× | +29 % | +75 % | 21.2 |
| Vector search / RAG | Traffic; CPU/GPU joins | 5.79× to 14.2× | +29 % to +34 % | +75 % to +94 % | 21.2 (P-E16 for the joins) |
| Databases / analytics | Copy/compute overlap | 1.77× to 1.96× | +13 % to +15 % | +29 % to +34 % | 21.2 |
| Graph analytics | Hand-off | 12.7× to 14.2× | +33 % to +34 % | +92 % to +94 % | 21.2 (P-E16) |
| Scientific simulation | Loop order | 3.39× | +24 % | +58 % | 21.2 |
| Genomics | Host traffic | 5.79× | +29 % | +75 % | 21.2 |
| Video pipelines | Launch to hand-off | 3.08× to 14.2× | +22 % to +34 % | +54 % to +94 % | 21.2 (P-E16 for the hand-off) |
| Financial Monte Carlo | Launch | 3.08× to 6.56× | +22 % to +30 % | +54 % to +79 % | 21.2 |
| Compression / hashing, vs the GPU alone | Two readers | 1.82× | +14 % | +31 % | 21.2 |
| Compression / hashing, vs the CPU alone | Two readers | 1.04× | +1 % | +2 % | 21.2 |
| Quantized-model GEMMs on GeForce (prompt reading, fine-tuning), per layer | int8 path with fused quantization | 1.39× to 1.84× | +8 % to +14 % | +17 % to +31 % | 21.2 (P-K2) |
| MoE expert work, grouped | One grouped launch, int8 | 2.5× to 3.0× (against bf16 grouped) | +19 % to +22 % | +45 % to +53 % | 21.2 (P-K2b) |
| Gaming: asset streaming | CPU writes straight into VRAM | 1.78× to 1.79× | +13 % to +14 % | +30 % | 21.2.1 |
| Gaming: streaming while rendering | Copy and compute queues at once | 1.98× | +15 % | +35 % | 21.2.1 |
| Gaming: submission-heavy frames | Separate submits merged into one replayed list | 36.2× to 42.2× | +36 % | +102 % to +103 % | 21.2.1 |
| Gaming: CPU/GPU sync points | Doorbell in VRAM (or write-combined RAM on D3D12) instead of a fence | 41.9× to 78× | +36 % | +103 % to +105 % | 21.2.1 |

**Full stack for the LLM, on paper** (hot bank + doorbell + overlap + zero-copy share; ceiling 88.2 to 92 tok/s, DERIVED): +177 % to +189 % over stock all-CPU (31.8, MEASURED), +115 % to +124 % over the best stock layout (41.1, MEASURED), +65 % to +72 % over the best tier run so far (53.5, MEASURED).

**Reading the gaming rows.** A k of 36× to 78× means the bottleneck all but disappears, so the gain is set by f alone (1 / (1 − f) − 1, +37 % at 27 %, +108 % at 52 %). A game's share of frame time in submissions or sync waits is usually far below the LLM's 27 % to 52 %, so the frame-time arithmetic below is the better guide until P-G1 measures f per title. Rows with more than one primitive are not stacked.

**Gaming, frame-time arithmetic** (P-G0 savings, Vulkan / Direct3D 12; fps gain = 1 ÷ (1 − saving ÷ frame) − 1):

| Situation | 60 fps | 144 fps |
|---|---|---|
| 1 / 5 / 10 blocking CPU-GPU syncs per frame, doorbell in VRAM instead | +0.35 / +1.8 / +3.6 % (Vulkan); +0.37 / +1.9 / +3.8 % (D3D12) | +0.83 / +4.3 / +9.0 % (Vulkan); +0.89 / +4.6 / +9.7 % (D3D12) |
| 5 / 20 / 50 separate submissions per frame merged into one replayed list | +0.9 / +3.7 / +9.8 % (Vulkan); +1.0 / +4.2 / +11.3 % (D3D12) | +2.2 / +9.4 / +27.3 % (Vulkan); +2.5 / +10.8 / +32.2 % (D3D12) |
| A frame that also uploads a 64 MiB texture, written straight into VRAM instead of staged | that frame 11.9 % faster (both APIs) | that frame 23.5 % / 23.6 % faster |

The previous version of this table used the 7.08× of microbenchmark 03 for every hand-off row; P-E16 replaced it with the same-run comparison against launch and synchronize (12.7× to 14.2×). The two new rows are kernel-level factors, so their shares need their own measurement (prompt-reading time is mostly GEMM, so f there is likely above the LLM decode shares; UNVERIFIED until the engine's timers run).

For scale, the end-to-end gains measured so far: the LLM hot-bank tier +30 % over the best stock layout and +68 % over all-CPU on the same machine, and Windows-native 23.5 % to 28.5 % faster than WSL2 on the same cell (two launches against one, not yet a claim; engine ledger).

### 21.7 Where this goes in the order of work

1. Done 2026-10-26: P-G0 (21.2.1).
2. Small probes that need no engine: P-G0b done (O-G1 cause, 21.2.1); P-E16 done (ReBAR on; 21.2.0, 21.2.1; the ReBAR-off run is O17); P-M5 deferred to step 10 (O16); P-W1 (wall power) still open. Build step 1 also measured the kernel primitives now in 21.2.0 (Batch C, P-M4).
3. After the first benchmark (16.4) and the release gates: extract Layer 0 (`libpelias_substrate`) as a stand-alone library with its own examples; then, per the 12:52 ruling, one scripted probe per field of 21.3 and 21.5, each against a stock baseline with unchanged output, measuring that field's own f and k (proposed first: vector search over an index larger than VRAM, the access pattern closest to the hot-bank case); P-G1 for games; H-U2 against H-U3 on the LLM workload, where the traces already exist.

No step here starts before the author says so.

# PELIAS UNIVERSAL MASTER SPECIFICATION (`pelias-universal-build-master.md`)

**A 100% Proprietary, Ground-Up C++20 and Multi-Vendor GPU Architecture for the Unified Pelias Runtime: Substrate, Core, Inference Engine, Native HTTP/SSE Server, and Layer-Major LoRA/Mode-Q Trainer.**

| Field | Value |
|---|---|
| **Document** | `pelias-universal-build-master.md` — Unified Ground-Up & Multi-Vendor Master Specification v1.0 |
| **Date** | 2026-10-25 |
| **Copyright** | Copyright (c) 2026 John Lipka and the Pelias contributors |
| **Licence** | Apache-2.0 |
| **Status** | **100% Build-Ready Architectural Specification. Zero source code included.** Wave 0 hardware probes are measured on WSL2 and Windows-native. Engine execution phases (`E2`–`E13`) are HELD per author ruling (`2026-10-24 06:07`) until lifted; shared Layer 0 and Layer 1 foundation modules needed by the trainer are built first. |
| **Target Model** | Qwen3.6-35B-A3B (`qwen35moe` architecture, 40 trunk layers `blk.0`–`blk.39` + 1 MTP block `blk.40`, 753 GGUF tensors / 1,045 `bf16` safetensors across 26 shards) |
| **Reference Hardware** | NVIDIA GeForce RTX 4070 12 GB (`AD104-250`, `sm_89`) + AMD Ryzen 7 5800X3D (8C/16T, AVX2) + 64 GB (`4x16 GB`) DDR4-3200 + PCIe 4.0 x16 |
| **Target Platforms** | Dual-target primary build from one codebase (`[sync R1]`): WSL2 (Ubuntu 26.04.1 LTS, kernel `6.18.54.1`, `nvcc 13.0.88`, GCC `15.2.0`) and Windows 11 Pro Native (build `26300`, `nvcc 13.4.92`, MSVC `19.44`). Multi-vendor/OS portability (`CUDA`, `HIP/ROCm`, `SYCL/Level Zero`, `Vulkan 1.3`; `AVX2`, `AVX-512`, `AMX`; Native Linux) abstracted in Layer 0 (`os/`, `hal/`). |
| **Runtime Mandate** | **100% Proprietary Ground-Up (`D37`, `[sync R2]`).** Zero runtime files, headers, libraries, or code from external AI or utility frameworks (`llama.cpp`, `ggml`, `libllama`, `common`, PyTorch, PEFT, Triton, Unsloth, or third-party JSON, HTTP, Regex, or Jinja libraries). |

---

## 1. Governance, Honesty System, Standing Rules, and the Proprietary Boundary

### 1.1 The Eight Honesty Labels

Every numerical value, architectural assertion, and performance target in this specification carries one of the following eight labels. Where a quantity is not yet measured, it is explicitly marked as `UNVERIFIED` or `ESTIMATE` and paired with the verification probe that replaces it with a measurement.

| Label | Definition |
|---|---|
| **MEASURED** | Measured directly on the reference hardware and recorded in a verified project ledger or raw probe output file. |
| **READ-SRC** | Read directly from the binary header of the served `Q4_K_M` GGUF, the 26 `bf16` safetensors shard headers, or reference specification sources (`2ca15f540`). Exact as a structural description of what the data or reference is. |
| **PUBLISHED** | Taken from primary vendor documentation (NVIDIA, AMD, Intel, Microsoft, JEDEC) or peer-reviewed academic literature. |
| **DERIVED** | Computed via scripted arithmetic from labelled inputs. All inputs and formulas are explicitly stated. |
| **ESTIMATE** | Engineering judgement. Never relied upon as fact; always paired with a named verification probe that replaces it with a measurement. |
| **UNVERIFIED** | A hypothesis or external claim that has not yet been tested on the reference hardware. Not relied upon until its probe passes. |
| **DECISION** | A design choice or author ruling with a recorded engineering reason. Reversible unless marked locked; items marked `DECISION (proposed)` await final author confirmation. |
| **AUTHOR-STATED** | A figure, observation, or requirement stated by the author where no raw log file is attached; paired with a verification probe where empirical. |

**Units and Conventions:**
* Binary vs. Decimal: `GiB = 2^30 bytes`, `MiB = 2^20 bytes`, `KiB = 2^10 bytes`, `GB = 10^9 bytes`, `MB = 10^6 bytes`.
* Training Throughput (`tok/s`): Defined as *processed tokens per second* (every token passing through the network, whether assistant or prompt), matching the baseline ledger convention (`P-T10`: `202.7 min * 247.85 tok/s = 3.01 M` processed tokens vs. `1.97 M` assistant tokens over 2 epochs).
* Inference Throughput (`tok/s`): Separated strictly into *prompt prefill* (`tok/s` at depth) and *token decode* (`tok/s` at short/`ctx0`, `4k`, and `12k` context depths).

### 1.2 The Author's Standing Rules

1. **Strictly Ground-Up (`D37`, `[sync R2]`):** No part of the Pelias runtime is a wrapper around PyTorch, PEFT, Unsloth, TRL, Triton, `llama.cpp`, `ggml`, `libllama`, or `common`. Every runtime module—GGUF reader/writer, safetensors reader, SHA-256, JSON parser/serializer, block quantizers/dequantizers, byte-level BPE tokenizer, chat-template subset renderer, raw-socket HTTP/1.1 and SSE server, arena allocator, doorbell coordinator, SIMD threadpool, and GPU spine/expert/LoRA kernels—is original code written for this project.
2. **Never Truncate:** Any training sample whose tokenized length exceeds the maximum training window (`8,192` tokens) is a hard failure with an audit report listing the offending row IDs. Silent truncation is forbidden.
3. **Hard VRAM Ceiling (`D26`, `D28`, `[sync R3]`):** Enforced in code by `hal/vram_governor`. The ceiling is dynamically derived at process startup from the quick probe: `min(CUDA free, NVML free)` after GPU context creation, minus a safety margin. On the reference machine, **`10.7 GiB`** is the governing threshold (`10.83 GiB` free minus `0.13 GiB` margin) until the disturbance test sets the final margin. Any memory plan exceeding the ceiling is refused before the first allocation.
4. **Zero Allocations in Hot Loops:** All device and host memory is pre-allocated in lifetime-scoped arenas (`hal/arena`, `hal/pinned_pool`) before step 1 (trainer) or token 1 (engine). Calling `cudaMalloc`, `cudaFree`, `new`, or `malloc` inside the training step loop or token decode loop is a fatal error caught by debug hooks (`P-M5`).
5. **No Model Downloads:** The system never fetches models from the internet. The user supplies local paths to the original `bf16` safetensors shards (for Stage A ingest) or an existing GGUF.
6. **Provenance and Clean-Room Hygiene:** The first line of every source file must state: `// provenance: from the spec: <name>`, `// provenance: adapted from <X>, licence <Y>`, or `// provenance: original`, followed by `Copyright (c) 2026 John Lipka and the Pelias contributors`. Where Pelias bytes must equal `ggml` block layouts or quantization arithmetic for GGUF interoperability, the arithmetic is re-expressed in original code after reading the specification, and credited in `CITATIONS.md`. Zero AGPL or LGPL code (e.g., Unsloth) is ever copied.
7. **Privacy Rails:** Dataset text, persona/identity strings, private mathematical equations, local user-profile paths, machine serials, and credentials (such as Hugging Face tokens) never enter any repository file, log, or report. Dataset audits and verification reports emit counts, row IDs, lengths, and pass/fail flags only.
8. **Execution Discipline:** Every step is gated by the verification rung beneath it. The user executes all GPU-touching binaries on the host machine. Git pushes occur only on the author's explicit command.

### 1.3 The Proprietary Boundary and Enforcement Checks (`BC1`–`BC6`)

| Classification | Permitted / Forbidden Components | Governing Rule |
|---|---|---|
| **Proprietary Product Code** | `os/`, `hal/`, `plan/`, `bus/`, `bank/`, `quant/`, `io/`, `tok/`, `kern/`, `model/`, `engine/`, `cpu/`, `gpu/`, `server/`, `config/`, `ingest/`, `data/`, `stream/`, `sched/`, `train/`, `export/`, `verify/`, `project/`, `startup/`, `intake/`, `help/`, `warn/`, `flow/`, `serve/`, `cli/` | 100% original C++20 and GPU kernel code written to this specification. |
| **Allowed System Dependencies** | GPU Runtime & Driver API (`cudart`/`cuda`, or `HIP`/`Level Zero`/`Vulkan` per backend); NVML / `amd_smi` / `zes` (loaded dynamically via `dlopen`/`LoadLibrary` so binaries start without it); C++20 Standard Library; x86-64 AVX2/FMA/F16C/AVX-512/AMX compiler intrinsics; OS kernel interfaces (POSIX on WSL2/Linux; Win32 & Winsock2 on Windows); `cuBLASLt` (optional, dense `bf16` only, retained only if `P-K1` shows hand-written GEMMs `< 0.85x cuBLASLt`). | Listed in `THIRD_PARTY_NOTICES.md` with exact versions; never redistributed. |
| **Open UI Dependency (`O9`)** | Dear ImGui (MIT Licence) in `src/ui/` (trainer interface only; never in `pelias-engine` or `server/`). | Author decides (`O9`): keep for UI only, write custom immediate-mode layer, or ship headless first. |
| **Static Data Tables** | Unicode Character Database (UCD) general-category and NFC canonical-composition tables in `tok/unicode_data`. | Public standard data compiled into static C++ tables by an original offline generator tool. |
| **Strictly Forbidden in Product** | `llama.cpp`, `ggml`, `libllama`, `common`, PyTorch, `transformers`, PEFT, Triton, Unsloth, `nlohmann/json` or any JSON library, `cpp-httplib` or any HTTP library, PCRE/`std::regex` or any regex library, Jinja2Cpp or any template library, Python interpreter at runtime. | Enforced on every build by Boundary Checks `BC1`–`BC6`. |
| **External Read-Only Test Oracles** | Pinned, untouched `llama.cpp` commit `2ca15f540` (`b11399`) in `tests/oracle/` and `tools/oracle_llama/`; test-only Python/PyTorch `fp32` autograd slice in `tools/oracle_torch/`; `gguf-py`. | Executed as separate processes during verification gates only. Never linked into, started by, or required at runtime by any product binary. |

* **`BC1` (Acyclic Module Graph):** `tools/check_module_deps.py` verifies on every build that the `#include` and CMake link graph is strictly acyclic and that no product module includes headers from `tests/` or `third_party/`.
* **`BC2` (Link-Map Symbol Audit):** Every product binary's linker map is scanned on every build to guarantee zero symbols from `ggml`, `llama`, or forbidden libraries.
* **`BC3` (Header Allow-List):** Product translation units may include only repository headers, C++20 standard library headers, GPU/telemetry headers, CPU intrinsics headers, and OS headers (restricted by `BC6`).
* **`BC4` (Provenance Header Audit):** `tools/check_resources.py` verifies that line 1 of every source and header file carries a valid provenance tag and copyright notice.
* **`BC5` (Oracle-Free Execution Gate):** Release gate (`PG10`) verifying that `pelias-engine` and `pelias-train` start, probe, plan, load, train, export, decode, and serve HTTP/SSE requests when `tests/oracle/` and `third_party/` are completely removed from disk.
* **`BC6` (OS Isolation Rule):** POSIX (`sys/socket.h`, `sys/mman.h`, `pthread.h`, `unistd.h`) and Win32 (`windows.h`, `winsock2.h`, `ws2tcpip.h`) headers are forbidden outside `src/os/` and `src/hal/host_alloc_{wsl,win32,linux}.cc`.

---

## 2. Reference Hardware Fact Base & Measured Wave 0 Floors

### 2.1 Reference Machine Specifications (`PELIAS`)

| Subsystem | Specification | Label & Source |
|---|---|---|
| **GPU Hardware** | NVIDIA GeForce RTX 4070 12 GB (`AD104-250`, `sm_89`): 46 SMs, 5,888 CUDA cores, 184 4th-Gen Tensor Cores, `36 MiB` L2 cache (`24.75 MiB` max persisting window), 192-bit GDDR6X bus, `504 GB/s` stock bandwidth, `200 W` default power limit (`216 W` max), `2,475 MHz` rated boost, `100 KiB` L1/shared memory per SM (`101,376 B = 99 KiB` max opt-in dynamic shared memory per block), 5 async copy engines. | `PUBLISHED` (Ada Whitepaper) & `MEASURED` (`P-H1`, HWiNFO64) |
| **GPU Overclock** | MSI Afterburner: Memory `+1000 MHz` (`nvidia-smi` reports `11,251–11,501 MHz` vs. `10,501 MHz` stock = `23,003.9 MT/s` effective vs. `21,004.0 MT/s` stock -> `540 GB/s` theoretical pin bandwidth); Core V/F curve locked at `2,779 MHz` @ `965 mV` (sustains `2,760–2,775 MHz` under full load). **No ECC.** | `AUTHOR-STATED` & `MEASURED` (HWiNFO64 v8.55, Afterburner, `P-H3`, `P-H4`) |
| **CPU Hardware** | AMD Ryzen 7 5800X3D (Family `19h` Zen 3): 8 physical cores, 16 threads, `4.54 GHz` boost, `4 MiB` L2 (`8 x 512 KiB`), `96 MiB` shared 3D V-Cache **victim L3 cache**, 64-entry L1 DTLB, 2,048-entry L2 DTLB. ISA: **AVX2, FMA, F16C, BMI2 only** (no AVX-512, no AVX-VNNI, no AMX). | `PUBLISHED` (AMD Family 19h SOG) & `MEASURED` (`P-H1`, census) |
| **System DRAM** | `64 GB` (`4 x 16 GB` G.Skill `F4-3200C16-16GVK`, dual-rank Samsung dies), dual-channel DDR4-3200 (`1,597.1 MHz`, timings 16-18-18-38, Command Rate 1T, XMP enabled, no ECC). Theoretical peak bandwidth: **`51.2 GB/s`**. | `PUBLISHED` (JEDEC) & `MEASURED` (HWiNFO64) |
| **Motherboard & Bus** | GIGABYTE B550M GAMING X WIFI6 (BIOS F5c, SMU 56.78.0); PCIe 4.0 x16 @ `16 GT/s` (`31.5 GB/s` raw per direction), max payload size `256 B`, ASPM off, Resizable BAR enabled (`16,384 MiB` BAR1). | `MEASURED` (HWiNFO64, `P-H8`) |
| **Storage Drives** | `C:` Samsung `MZVLB1T0HBLR-000H1` `954 GB` NVMe (PCIe 3.0 x4 @ `8 GT/s`, holds Windows OS and WSL2 `ext4.vhdx`); `D:` WD Blue `SN580` `932 GB` NVMe (PCIe 4.0 drive linked at PCIe 3.0 x2 @ `8 GT/s`, raw ceiling `~1.97 GB/s` [`DERIVED`], holds archives and WSL2 swap file). | `MEASURED` & `DERIVED` (HWiNFO64) |
| **Host OS (Windows 11)** | Windows 11 Pro 26H2 (build `26300.9711`), Hyper-V active, HVCI off, Hardware-Accelerated GPU Scheduling (HAGS) off (`HwSchMode = 1`), TDR at Windows default (`2 s`), Power Plan Balanced, fixed `32 GiB` page file. NVIDIA Profile Inspector: `CUDA - Force P2 State = Off`, `Power management = Prefer maximum performance`, `Prefer no system memory fallback = On`. | `MEASURED` & `AUTHOR-STATED` (`MACHINE_STATE.md`) |
| **Guest OS (WSL2)** | WSL `3.0.4.0`, Ubuntu 26.04.1 LTS, kernel `6.18.54.1-microsoft-standard-WSL2`, `/dev/dxg` WDDM paravirtualization. `.wslconfig`: `memory=48GB`, `processors=16`, `swap=48GB` on `D:`, `kernelCommandLine=transparent_hugepage=always swiotlb=65536`, `virtiofs=true`, `autoMemoryReclaim=disabled`, `sparseVhd=false`. `RLIMIT_MEMLOCK = 64 MiB`. | `MEASURED` (`census.txt`, `MACHINE_STATE.md`) |
| **Compilers & CUDA** | NVIDIA Driver `617.42` (CUDA 13.4 UMD). **WSL2:** `nvcc 13.0.88`, GCC `15.2.0`, Clang `21`, CMake `4.2.3`, Ninja `1.13.2`. **Windows Native:** `nvcc 13.4.92`, MSVC `19.44.35229.0` (Visual Studio 2026 / 18 Professional, toolset `14.44.35207`), CMake `4.4.3`, Ninja `1.13.2`. **Avoid CUDA 13.2** (`PUBLISHED`). | `MEASURED` (Wave 0 logs) |

### 2.2 Complete Measured Hardware Floors (Wave 0 & Microbenchmarks `02`–`07`)

| Probe / Microbench | Measured Quantity | WSL2 Result | Windows-Native Result | Architectural Consequence |
|---|---|---|---|---|
| **`P-H8` (Free VRAM & Context)** | Total VRAM / Pre-context used / Context cost / **Post-context CUDA Free** / Post-context NVML Free | `11.99 GiB` total / `914–1,070 MiB` used / `153.7 MiB` ctx / **`10.830 GiB` CUDA free** / `10.73–10.95 GiB` NVML free | `11.99 GiB` total / `1,015–1,031 MiB` used / `138 MiB` ctx / **`10.846 GiB` CUDA free** / `10.85–10.87 GiB` NVML free | CUDA free is a fixed driver budget (`10.83 GiB` across 21 engine logs + 4 probe runs). Sets the **`10.7 GiB` ceiling (`D26`)** and `min(CUDA, NVML)` startup rule (`D28`). |
| **`P-H1` (Device Limits)** | SMs / Max opt-in shared memory per block / Async copy engines / Max persisting L2 / KernelExecTimeout | `46` SMs / `101,376 B` (`99 KiB`) / `5` engines / `24.75 MiB` / `1` (on) | `46` SMs / `101,376 B` (`99 KiB`) / `5` engines / `24.75 MiB` / `1` (on) | Locks `kern/` tile constants (`<= 96 KiB` shared memory per CTA). |
| **`P-H2` & `02` (VRAM & DRAM BW)** | VRAM Read / Write / Copy kernel / D2D `cudaMemcpy`; Host AVX2 DRAM Read (1T / 4T best / 7T / `2 MiB` THP) | VRAM: **`523.3`** / `447.5` / `473.3` / `494.5 GB/s`. DRAM: `37.1` (1T), **`43.7–45.81 GB/s`** (4T, `4 KiB`), `45.89` (7T), `44.86` (`2 MiB` THP) | VRAM: **`534.0`** / `454.0` / `481.0` / `504.0 GB/s`. DRAM: `37.9` (1T), **`44.8–46.4 GB/s`** (4T) | Memory OC delivers `> 504 GB/s` stock VRAM read. `2 MiB` pages change CPU read by `< 1%`. |
| **`P-H3` (Sustained Tensor Peaks)** | `60 s` sustained MMA: `bf16` (`m16n8k16` `fp32`-acc) / `fp16` / `int8` (`m16n8k32` `s32`-acc) / `fp8-e4m3` @ `2,760–2,775 MHz` | **`bf16` `65.2 TFLOPS`**, `fp16` `64.9 TFLOPS`, **`int8` `259.5 TOPS`**, `fp8` `129.8 TOPS` (`99.8%` of burst; `0.511 / 2.044 T-ops/SM/GHz`) | **`bf16` `65.2 TFLOPS`**, `fp16` `64.9 TFLOPS`, **`int8` `259.5 TOPS`**, `fp8` `129.8 TOPS` (`99.8%` of burst) | Replaces derived OC peaks (`65.4 / 261.4`). Proves `int8` is `3.98x` `bf16` on `sm_89`. Stock-clock peaks remain `DERIVED` (`58.3 TFLOPS` `bf16`, `233.2 TOPS` `int8`). |
| **`P-H10` & `P-H4` (Canary & `30 min` Soak)** | `10,000` canary GEMMs (`2048^3`); `30 min` thermal/clock stability soak (`~17,700` cycles) | `0` mismatches, `1` hash, error/tol `<= 0.01`. Soak: `0` mismatches, SM clock flat `2,760.0 MHz` median, `43–55 C`, `~107 W`. | `0` mismatches, `1` hash, error/tol `<= 0.01`. Soak: `0` mismatches, SM clock flat `2,760.0 MHz` median, `49–56 C`, `~106.7 W` (mem flat `11,501 MHz`). | Overclock is stable for multi-hour training. NVML throttle field falsely reports `gpu_idle` (`0x1`) under full load; watchdog uses SM clock + temp (`15.3`). |
| **`P-H5` (PCIe DMA & Zero-Copy)** | `256 MiB` Pinned H2D / Pinned D2H / Pageable H2D / Pageable D2H / Zero-Copy Read / `4 KiB` & `1 MiB` latency | Pinned: **`26.3 GB/s`** H2D & D2H; Pageable: `10.3` / `10.8 GB/s`; Zero-copy: **`26.3 GB/s`**; Latency: `73 us` (`4 KiB`), `113 us` (`1 MiB`) | Pinned: **`26.3 GB/s`** H2D & D2H; Pageable: `11.8` / `11.8 GB/s`; Zero-copy: **`26.3 GB/s`**; Latency: `31 us` (`4 KiB`), `72 us` (`1 MiB`) | Confirms `26.1–26.3 GB/s` PCIe 4.0 x16 floor. Windows small-copy latency is `2.3x` lower than WSL2. |
| **`04` & `P-H6` (Dual-Bus Contention)** | Simultaneous CPU AVX2 cold read + GPU PCIe zero-copy read / Simultaneous CPU read + H2D DMA | **CPU `26.56 GB/s` + GPU ZC `21.06 GB/s` = `47.62 GB/s`** (`93.0%` of `51.2 GB/s` peak, `1.04x` CPU alone). CPU + H2D: `46.6–49.96 GB/s` (H2D keeps `87–100%`). | Unloaded H2D `26.1 GB/s`; with 4/8/16 CPU reader threads, H2D retains `84–100%` (`21.9–26.1 GB/s`). | Proves the **Unified Dual-Bus discovery**: CPU and GPU read system DRAM simultaneously without halving each other's bandwidth. |
| **`03` & Pass 1 (Doorbell Handoff)** | Per-layer `cudaStreamSynchronize` vs. Resident-kernel shared-memory doorbell ping-pong | Stream sync: **`94.40 us` median** (`123.01 us` p99). Doorbell (`03`, `4 KiB`): **`13.34 us` median, `18.71 us` p99** (`3.36 us` flag only). Block-256 test (`test_handoff`): **`5.05 us` median, `6.17 us` p99**. | Unmeasured on Windows-native (`E1` floor test). | Replaces `3.76–5.1 ms/token` OS sync overhead with sub-`20 us/layer` lock-free flag coordination (`bus/doorbell_handoff`). |
| **`07` & `P-H7` (Pinned Memory Cap)** | Maximum `cudaHostAlloc` pinned allocation up to `32 GiB` (`34.4 GB`) & pointer identity vs. `cudaHostRegister` | **`32 GiB` pinned & held** (`0 MiB` swap-in, `0.76–1.0 s/GiB` alloc). `cudaHostAlloc` host ptr `=` dev ptr; `cudaHostRegister` dev ptr `!=` host ptr. | **`32 GiB` pinned & held** (`0 MiB` swap-in, `0.069 s/GiB` alloc — `12x` faster than WSL2). | Closes Risk `R7`: the `18.65 GiB` expert store + `5–10 GiB` boundary store fit in `cudaHostAlloc` memory on both platforms. |
| **`P-M3` (`30 min` Host Hold)** | Hold `24 GiB` pinned + `4 GiB` pageable for `30 min` with continuous DMA & `10 ms` sleep heartbeat | `0` mismatches, `0 MiB` swap-in, H2D median `25.8 GB/s`; sleep overshoot median `0.082 ms`, p99 `0.375 ms`, max `1.70 ms`. | `0` mismatches, `0 MiB` swap-in, H2D median `26.03 GB/s`; sleep overshoot median `6.14 ms`, p99 `6.48 ms` (default `15.6 ms` Win32 timer tick). | Proves long-run host memory stability under WSL2's `48 GB` cap and Windows native. |
| **`P-H9` (CUDA Graphs & Flush Cost)** | 100-kernel eager launch vs. CUDA graph replay (`10^5` replays); `cudaStreamQuery` flush cost | Eager `2.25–4.70 us/kernel` vs. **Graph `0.716 us/kernel`** (`68–85%` saving). Flush: `0.098 us` idle, **`76.6 us` busy stream**. | Eager `2.18–2.32 us/kernel` vs. **Graph `0.707 us/kernel`** (`68%` saving). Flush: `0.091 us` idle, **`8.05 us` busy stream** (`9.5x` faster). | Locks CUDA graph adoption for Engine spine (`E8`) and Trainer layers (`6.9`). |
| **`P-H12` (3-Stream Copy/Compute Overlap)** | Concurrent H2D (`S_w`) + D2H (`S_x`) + Compute (`S_compute`) across 4 submission patterns | Unflushed default: **`0.47`** all-three (`0.00` H2D+D2H). **Flushed (`cudaStreamQuery` per batch): `0.92` all-three, `0.84` H2D+D2H pair**. | Unflushed default: **`0.52`** all-three (`0.00` H2D+D2H). **Flushed (`cudaStreamQuery` per batch): `0.92` all-three, `0.84` H2D+D2H pair**. | Proves WDDM batches cross-stream work unless flushed. Locks `D27`: flush once per multi-ms batch; roofline solver uses `0.84` pair / `0.92` all-three efficiency. |
| **`P-H13` (Display Timeout / TDR)** | `2 s`, `5 s`, `30 s` continuous spin kernels, `10 s` queued set, `10 s` graph with desktop active | All 5 stages PASS (`30.001 s` wall), `0` errors, `0` visual lag/flicker. | All 5 stages PASS (`29.9998 s` wall), `0` errors, `0` `nvlddmkm` or WHEA events, `0` visual lag. | Hardware compute preemption prevents TDR resets up to `30 s` at `~60 W`. Resident doorbell kernels remain scoped to 1 token (`~18–26 ms`). |
| **File Read Speeds (`P-M4` / Runbook)** | Sequential read of `20.22 GiB` GGUF from Windows 9p mount (`/mnt/c`, `/mnt/d`) vs. WSL2 `ext4` disk | `/mnt/c` & `/mnt/d` (9p): **`0.17–0.52 GB/s`** (`87–119 s` load). **WSL2 `ext4`: `2.99–3.06 GB/s` cold (`7.1 s`), `0.25 s` warm**. | `P-M4` standalone binary awaits `PELIAS_GGUF` path. | Models accessed inside WSL2 must reside on the WSL2 `ext4` virtual disk (`~/models/`) or be read once into the pinned store. |

---

## 3. Model Architecture, Exact Byte Maps, and Mathematical Specifications

### 3.1 Model Structural Constants (`Qwen3.6-35B-A3B` / `qwen35moe`, `READ-SRC`)

| Key / Parameter | Exact Value | Architectural Note |
|---|---|---|
| **Architecture & Block Count** | `qwen35moe`; `block_count = 41` (`40` trunk layers `blk.0`–`blk.39` + `1` MTP block `blk.40`); `nextn_predict_layers = 1` | Stock `llama-server` ignores `blk.40` with 20 warnings at startup; Pelias treats `blk.40` as a first-class block for speculative decoding (`E12`) and MTP co-training (`D12`). |
| **Dimensions & Vocabulary** | Hidden dim `d_model = 2,048`; Vocabulary `V = 248,320` (BPE vocab `248,044` + padding); Max context `262,144`; RMSNorm `eps = 1e-6` | Embedding (`token_embd.weight`) and output head (`output.weight`) are **untied**. |
| **Hybrid Layer Pattern** | `full_attention_interval = 4` -> **10 Full GQA Attention layers** at `l in {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}` (`(l+1) mod 4 == 0`); **30 Gated DeltaNet layers** at all other trunk indices; `blk.40` is Full Attention + MoE | Verified from `qwen35moe.cpp` graph builder and per-layer tensor presence (`F2`). |
| **Full GQA Attention** | `H_q = 16` query heads, `H_kv = 2` KV heads (8:1 GQA), head dim `d_h = 256`; `attn_q` projects `2,048 -> 8,192` (**interleaved query + output gate** per head); per-head `attn_q_norm` and `attn_k_norm` (`256`-wide); **IMRoPE** on first `64` dims (`32` pairs), sections `[11, 11, 10, 0]`, base `theta_0 = 10,000,000` | Attention output is gated by `sigma(gate_h)` before `attn_output` (`4,096 -> 2,048`). |
| **Gated DeltaNet ("SSM")** | Causal depthwise `conv1d` kernel `K_conv = 4` across `8,192` channels (`q 2,048 | k 2,048 | v 4,096`); `group_count` `H_k = 16` (`q, k` heads of dim `128`); `time_step_rank` `H_v = 32` (`v` heads of dim `128`); state size `128 x 128` per `v`-head; inner size `4,096`; gated RMSNorm `ssm_norm` (`128`-wide) | `k`-heads are tiled `16 -> 32` so `v`-head `j` uses `k`-head `(j mod 16)`. |
| **Mixture of Experts (MoE)** | `E = 256` routed experts per layer, top-`k = 8` active per token, expert FFN intermediate dim `d_exp = 512`; `1` always-on shared expert per layer (`d_shexp = 512`) gated by a single scalar gate `sigma(g_sh . u)` (`ffn_gate_inp_shexp`, `2,048`-wide `F32`) | Total active parameters per token: **`2.94 B`** (experts `1.007 B` + shared experts `125.8 M` + DeltaNet in `755.0 M` / out `251.7 M` + attention `272.6 M` + routers `21.0 M` + `lm_head` `508.6 M`). |

### 3.2 Complete Tensor and Byte Maps: Served `Q4_K_M` GGUF and Source `bf16` Safetensors

#### 3.2.1 Served `Q4_K_M` GGUF Inventory (`READ-SRC` & `DERIVED`)
* **File Totals:** `753` tensors; `21,713,463,296 bytes` (`20.22 GiB`); `general.alignment = 32`; tensor data starts at byte `10,991,104`; every tensor offset is `32-byte` aligned; all `123` expert tensors are `16-byte` aligned and exact multiples of `2 MiB` (`144 MiB` or `210 MiB`).

| GGUF Tensor Name | GGUF Shape (Inner Dim First) | Quantization Types & Layer Distribution | Count | Total MiB | Role & Training Status |
|---|---|---|---:|---:|---|
| `token_embd.weight` | `2048 x 248320` | `Q4_K` | 1 | `272.8` | Input embedding; frozen, gather + dequant only, byte-spliced. |
| `output.weight` | `2048 x 248320` | `Q6_K` | 1 | `397.9` | Untied `lm_head`; frozen (`dgrad` only; optional LoRA in `T-A8`). |
| `output_norm.weight` | `2048` | `F32` (`+1` folded) | 1 | `0.008` | Final RMSNorm; frozen (optional `fp32` trainable). |
| `blk.{N}.attn_norm.weight` | `2048` | `F32` (`+1` folded) | 41 | `0.3` | Pre-attention / pre-DeltaNet RMSNorm; frozen (optional `fp32` trainable). |
| `blk.{N}.post_attention_norm.weight` | `2048` | `F32` (`+1` folded) | 41 | `0.3` | Pre-MoE RMSNorm; frozen (optional `fp32` trainable). |
| `blk.{N}.ffn_gate_exps.weight` | `2048 x 512 x 256` | `Q4_K` (`144 MiB/layer`) | 41 | `5,904.0` | Routed expert gate projections; **frozen, streamed, byte-spliced**. |
| `blk.{N}.ffn_up_exps.weight` | `2048 x 512 x 256` | `Q4_K` (`144 MiB/layer`) | 41 | `5,904.0` | Routed expert up projections; **frozen, streamed, byte-spliced**. |
| `blk.{N}.ffn_down_exps.weight` | `512 x 2048 x 256` | **`Q6_K` x 21** (`210 MiB/layer`), **`Q4_K` x 20** (`144 MiB/layer`) | 41 | `4,410.0 + 2,880.0` | Routed expert down projections; **frozen, streamed, byte-spliced**. |
| `blk.{N}.ffn_gate_inp.weight` | `2048 x 256` | `F32` (`2.0 MiB/layer`) | 41 | `82.0` | MoE Router (`21,495,808` params); frozen default (**trainable `fp32` in Mode L**). |
| `blk.{N}.ffn_gate_inp_shexp.weight` | `2048` | `F32` | 41 | `0.3` | Shared-expert scalar gate vector; frozen (optional `fp32` trainable in `T-A5`). |
| `blk.{N}.ffn_gate_shexp.weight` | `2048 x 512` | `Q4_K` | 41 | `23.1` | Shared expert gate; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ffn_up_shexp.weight` | `2048 x 512` | `Q4_K` | 41 | `23.1` | Shared expert up; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ffn_down_shexp.weight` | `512 x 2048` | `Q6_K` x 21, `Q4_K` x 20 | 41 | `17.2 + 11.2` | Shared expert down; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_qkv.weight` | `2048 x 8192` | `Q6_K` x 14, `Q4_K` x 16 | 30 | `183.8 + 144.0` | DeltaNet joint `q, k, v`; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_gate.weight` | `2048 x 4096` | `Q4_K` | 30 | `135.0` | DeltaNet output gate `z`; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_alpha.weight` | `2048 x 32` | `Q4_K` | 30 | `1.1` | DeltaNet decay projection; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_beta.weight` | `2048 x 32` | `Q4_K` | 30 | `1.1` | DeltaNet write-strength projection; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_out.weight` | `4096 x 2048` | `Q4_K` | 30 | `135.0` | DeltaNet output projection; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_conv1d.weight` | `4 x 8192` | `F32` | 30 | `3.8` | DeltaNet 4-tap depthwise conv; frozen. |
| `blk.{N}.ssm_a`, `ssm_dt.bias`, `ssm_norm.weight` | `32` / `32` / `128` | `F32` (`ssm_norm` is plain, no `+1`) | 30 each | `< 0.1` | DeltaNet decay base (`-exp(A_log)`), bias, gated RMSNorm; frozen. |
| `blk.{N}.attn_q.weight` | `2048 x 8192` | `Q4_K` | 11 | `99.0` | Full Attention joint query + gate; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_k.weight` | `2048 x 512` | `Q4_K` | 11 | `6.2` | Full Attention key; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_v.weight` | `2048 x 512` | `Q6_K` x 7, `Q4_K` x 4 | 11 | `5.7 + 2.2` | Full Attention value; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_output.weight` | `4096 x 2048` | `Q4_K` | 11 | `49.5` | Full Attention output; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_q_norm.weight`, `attn_k_norm.weight` | `256` each | `F32` (`+1` folded) | 11 each | `< 0.1` | Full Attention per-head RMSNorms; frozen. |
| `blk.40.nextn.eh_proj.weight` | `4096 x 2048` | `Q4_K` | 1 | `4.5` | MTP embedding+hidden fusion projection; optional LoRA (`D12`). |
| `blk.40.nextn.{enorm,hnorm,shared_head_norm}.weight` | `2048` each | `F32` (`+1` folded) | 1 each | `< 0.1` | MTP norms; frozen. |

**Exact Layer-Index Rules for Mixed Quantization (`READ-SRC`):**
* **21 Blocks with `Q6_K` `ffn_down_exps` and `ffn_down_shexp` (`498 MiB` expert layer = `288 MiB` gate+up `Q4_K` + `210 MiB` down `Q6_K`; `2,039,808 B/expert`):** Trunk layers `0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39`, plus MTP block `40`.
* **20 Trunk Layers with `Q4_K` `ffn_down_exps` and `ffn_down_shexp` (`432 MiB` expert layer = `288 MiB` gate+up `Q4_K` + `144 MiB` down `Q4_K`; `1,769,472 B/expert`):** Trunk layers `5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33`.
* **14 DeltaNet Layers with `Q6_K` `attn_qkv`:** Layers `0, 1, 2, 4, 10, 13, 16, 22, 25, 28, 34, 36, 37, 38` (the other 16 DeltaNet layers use `Q4_K`).
* **7 Full-Attention Blocks with `Q6_K` `attn_v`:** Trunk layers `3, 7, 19, 31, 35, 39`, plus MTP block `40` (the other 4 full-attention layers `11, 15, 23, 27` use `Q4_K`).
* **Total Routed Expert Bytes:** 40 trunk layers = `20 x 432 MiB + 20 x 498 MiB = 18,600 MiB = 19,503,513,600 B` = **`18.164 GiB`** (`32,212,254,720` params; fleet average `1,904,640 B/expert`; `8 x 40 = 320` active expert calls/token = `0.6095 GB/token`). Including `blk.40` (`498 MiB`) = `19,098 MiB` = **`18.65 GiB`** pinned in host RAM.

#### 3.2.2 Source `bf16` Safetensors Checkpoint Layout (`MEASURED` from Shard Headers, `2026-10-24`)
Read from the 26 shards of the archived `unsloth/Qwen3.6-35B-A3B` checkpoint: **26 shards (`1.10–4.00 GB` each), 1,045 tensors, all `BF16`, `71,903,645,408` tensor bytes (`66.965 GiB = 71.90 GB`)**.

| Checkpoint Class | Tensors | Parameters | Bytes | GiB | Stage A (`ingest`) Handling Rule (`D33`) |
|---|---:|---:|---:|---:|---|
| **Routed Experts (40 trunk layers)** (`mlp.experts.gate_up_proj` `[256, 1024, 2048]` + `down_proj` `[256, 2048, 512]`) | 80 | `32,212,254,720` | `64,424,509,440` | `60.000` | Fused `gate_up_proj` (`1,024 MiB/layer`) is sliced along `dim = -2` at row `512` into separate `ffn_gate_exps` (`0..511`) and `ffn_up_exps` (`512..1023`), then block-quantized to `Q4_K`/`Q6_K`. |
| **Trunk Spine & Routers** | 611 | `1,431,237,248` | `2,862,474,496` | `2.666` | Apply name map (`model.language_model.` stripped), V-head tiled permutation (`P-S3`), norm `+1` folding (`P-S2`), `ssm_a = -exp(A_log)`, and stock `Q4_K_M` type map. |
| **Input Embedding** | 1 | `508,559,360` | `1,017,118,720` | `0.947` | Quantize to `Q4_K` (`token_embd.weight`). |
| **Output Head (`lm_head`, untied)** | 1 | `508,559,360` | `1,017,118,720` | `0.947` | Quantize to `Q6_K` (`output.weight`). |
| **MTP Block (`mtp.*`)** | 19 | `844,640,768` | `1,689,281,536` | `1.573` | Kept by default as `blk.40` (2 fused expert tensors split to 3 GGUF expert tensors + 17 spine/norm tensors = 20 GGUF tensors). |
| **Vision Tower (`model.visual.*`)** | 333 | `446,571,248` | `893,142,496` | `0.832` | **Dropped** (text-only GGUF: `1,045 - 333 = 712` source tensors -> `753` GGUF tensors after splitting 41 `gate_up_proj` tensors into `82`). |
| **Total** | **1,045** | **`35,951,822,704`** | **`71,903,645,408`** | **`66.965`** | Streamed one tensor at a time via `mmap` (max in-flight tensor `1,024 MiB`). |

#### 3.2.3 Trainable Core Byte Budget Table (40 Trunk Layers, `1,409.0 M` Parameters, 310 LoRA Modules)

| Core Tensor Class | Module Count | Parameters | Served Mix (`MiB`) | All `Q6_K` (`MiB` / Delta) | All `Q8_0` (`MiB` / Delta) | `bf16` (`MiB`) |
|---|---:|---:|---:|---:|---:|---:|
| **Shared Experts** (`gate`, `up`, `down` x 40) | 120 | `125.8 M` | `72.7` | `98.4` (`+25.7`) | `127.5` (`+54.8`) | `240.0` |
| **DeltaNet In-Projections** (`qkv` + `gate` x 30) | 60 | `755.0 M` | `462.8` | `590.6` (`+127.8`) | `765.0` (`+302.2`) | `1,440.0` |
| **DeltaNet Decay/Write** (`alpha`, `beta` x 30) | 60 | `3.9 M` | `2.1` | `3.1` (`+1.0`) | `4.0` (`+1.9`) | `7.5` |
| **DeltaNet Out-Projection** (`ssm_out` x 30) | 30 | `251.7 M` | `135.0` | `196.9` (`+61.9`) | `255.0` (`+120.0`) | `480.0` |
| **Full Attention** (`q`, `k`, `v`, `output` x 10) | 40 | `272.6 M` | `147.8` | `213.3` (`+65.5`) | `276.2` (`+128.4`) | `520.0` |
| **Total Trunk Core** | **310** | **`1,409.0 M`** | **`820.3`** (`820.4`) | **`1,102.3`** (**`+282.0`**) | **`1,427.7`** (**`+607.4`**) | **`2,687.5`** |

*(Note: MTP `blk.40` adds `47.2 M` core parameters when `D12` is active. At rank `r = 16`, the 310 trunk LoRA modules contain **`21,166,080` trainable parameters**; at `16 B/param` for `fp32` master + `fp32` grad + `fp32` Adam `m, v`, LoRA state consumes **`323.0 MiB`** [`0.32 GiB`], plus `40.4 MiB` for `bf16` compute copies.)*

### 3.3 Mathematical Specification of the Forward and Backward Graph

Let $x \in \mathbb{R}^{T \times 2048}$ be the residual hidden state stream (`fp32` between graph nodes). Per-token RMSNorm with weight vector $w$ (where $w = 1 + w_{\text{HF}}$ is pre-folded for all norms except `ssm_norm`) is:
$$\text{RMS}(x; w) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot w, \quad \epsilon = 10^{-6}$$

1. **Trunk Layer Structure ($l = 0 \dots 39$):**
   $$a = \text{RMS}(x; w_{\text{attn\_norm}, l}), \quad m = \begin{cases} \text{DeltaNet}_l(a) & \text{if } (l+1) \bmod 4 \neq 0 \\ \text{Attention}_l(a) & \text{if } (l+1) \bmod 4 = 0 \end{cases}$$
   $$h = x + m, \quad u = \text{RMS}(h; w_{\text{post\_attn\_norm}, l}), \quad y = h + \text{MoE}_l(u)$$
2. **Mixture-of-Experts Sub-Layer (`MoE`):**
   * **Router (`F32`):** Logits $z = u W_r^\top \in \mathbb{R}^{T \times 256}$ computed in `fp32` FMA (no TF32). Select top-8 expert indices $S = \text{argtopk}_8(z)$ per token (ties broken to lower expert index, `P-S8`). Normalized weights over $e \in S$ use an 8-way `fp32` softmax: $w_e = \exp(z_e) / \sum_{e' \in S} \exp(z_{e'})$.
   * **Expert & Shared-Expert FFN:** $E(u) = W_{\text{down}} (\text{SiLU}(u W_{\text{gate}}^\top) \odot (u W_{\text{up}}^\top))$, and $\text{MoE}(u) = \sum_{e \in S} w_e E_e(u) + \sigma(u \cdot g_{\text{sh}}) E_{\text{sh}}(u)$.
   * **Backward (`R3`, `E2`, `G5/G6`):** $dw_e = \langle dy, E_e(u) \rangle$ in `fp32`; $dz_e = w_e (dw_e - \sum_{e' \in S} w_{e'} dw_{e'})$ for $e \in S$; $du \mathrel{+}= \sum_{e \in S} dz_e W_r[e, :]$. For SwiGLU with $h = \text{SiLU}(g) \odot p$: $dp = dh \odot \text{SiLU}(g)$ and $dg = dh \odot p \odot \sigma(g)(1 + g(1 - \sigma(g)))$.
3. **Full GQA Attention Sub-Layer ($l \in \{3, 7, \dots, 39\}$ and `blk.40`):**
   * $[q \mid \text{gate}] = a W_q^\top \in \mathbb{R}^{T \times 8192}$ (per head $h \in \{0\dots 15\}$, rows $[512h, 512h+256)$ are $q_h$ and $[512h+256, 512h+512)$ are $\text{gate}_h$). Keys $k = a W_k^\top \in \mathbb{R}^{T \times 512}$ and values $v = a W_v^\top \in \mathbb{R}^{T \times 512}$.
   * Apply per-head RMSNorm (`w_q_norm`, `w_k_norm`) and IMRoPE (`A1`, sections `[11, 11, 10, 0]`, base $\theta_0 = 10^7$) to the first 64 dims of each head, resetting position $p$ to 0 at packed document boundaries (`cu_seqlens`).
   * Causal GQA with KV head $g = \lfloor h/8 \rfloor$: $o_h = \text{softmax}_{\text{causal}}(q_h k_g^\top / 16) v_g$, gated output $o'_h = o_h \odot \sigma(\text{gate}_h)$, projection $m = \text{concat}(o'_0 \dots o'_{15}) W_o^\top$.
4. **Gated DeltaNet Sub-Layer (30 Trunk Layers):**
   * $\text{qkv}_{\text{raw}} = a W_{\text{qkv}}^\top \in \mathbb{R}^{T \times 8192}$, $z = a W_z^\top \in \mathbb{R}^{T \times 4096}$, $\beta = \sigma(a W_\beta^\top) \in (0, 1)^{T \times 32}$, $g = \text{ssm\_a} \odot \text{softplus}(a W_\alpha^\top + b_{\text{dt}}) \in (-\infty, 0]^{T \times 32}$.
   * $\text{qkv}' = \text{SiLU}(\text{CausalConv1D}_{k=4}(\text{qkv}_{\text{raw}}; w_{\text{conv}}))$, split into $q, k \in \mathbb{R}^{T \times 16 \times 128}$ and $v \in \mathbb{R}^{T \times 32 \times 128}$. L2-normalize $q$ (scaled by $1/\sqrt{128}$) and $k$ per head; tile $q, k$ heads $16 \rightarrow 32$ (`j mod 16`).
   * **Sequential Recurrence (`D7`):** With $\alpha_t = \exp(g_t)$ and $S_0 = 0$ at document start: $r_t = v_t - \alpha_t (k_t S_{t-1})$, $S_t = \alpha_t S_{t-1} + \beta_t (k_t^\top r_t)$, $o_t = q_t S_t$.
   * **Chunked WY/UT Form (`D3`–`D5`, $C = 64$):** With $g_{\text{cs}, i} = \sum_{j \le i} g_j$, $D_{ij} = \exp(g_{\text{cs}, i} - g_{\text{cs}, j})$ for $j \le i$, $N_{ij} = \beta_i (k_i \cdot k_j) D_{ij}$ for $j < i$, and $T = (I + N)^{-1}$: $u = T(\beta \odot v)$, $k_{\text{cd}} = T(\beta \odot \exp(g_{\text{cs}}) \odot k)$, $v_{\text{new}} = u - k_{\text{cd}} S$, $o = (q \odot \exp(g_{\text{cs}})) S + \text{kq} \cdot v_{\text{new}}$, $S \leftarrow \exp(g_{\text{last}}) S + \text{kg}^\top v_{\text{new}}$.
   * **Gated RMSNorm & Output (`E3`):** $o' = \text{RMS}_{\text{plain}}(o; w_{\text{ssm\_norm}}) \odot \text{SiLU}(z)$, $m = o' W_{\text{out}}^\top$.
   * **Backward (`D6`, `D7`):** With $G = dS_t + q_t^\top do_t$: $dq_t = do_t S_t^\top$, $d\beta_t = k_t G r_t^\top$, $dr = \beta_t k_t G$, $dv_t = dr$, $dk_t = \beta_t (G r_t^\top)^\top - \alpha_t (dr S_{t-1}^\top)$, $d\alpha_t = \langle G, S_{t-1} \rangle - \langle dr, k_t S_{t-1} \rangle$, $dg_t = \alpha_t d\alpha_t$, $dS_{t-1} = \alpha_t G - \alpha_t k_t^\top dr$.
5. **Step-Wide Loss Normalization (`L1`, `D8`):**
   $$L_{\text{step}} = \frac{\sum_{t \in \text{step}} w_t \cdot \text{CE}_t}{\sum_{t \in \text{step}} w_t}$$
   evaluated in 256-position chunks (`242.5 MiB` `fp32` buffer) where $w_t = 1$ on assistant generation tokens (after the generation prompt through closing `<|im_end|>`) and $w_t = 0$ on all other tokens.

---

## 4. Complete System Scaffolding & Acyclic Module Dependency Graph

### 4.1 Unified Four-Layer Architecture
* **Layer 0: `libpelias_substrate`** — Domain-agnostic unified-bus hardware abstraction (`os/`, `hal/`, `plan/`, `bus/`, `bank/`, `quant/`).
* **Layer 1: `libpelias_core`** — Shared model I/O, byte-splice exporter, native tokenizer & template renderer, spine forward kernels, and verification comparators (`io/`, `tok/`, `kern/` [spine], `model/` [spec], `verify/`).
* **Layer 2A: `pelias-engine`** — Token-major dual-bus inference engine and raw-socket HTTP/SSE server (`engine/`, `cpu/`, `gpu/`, `server/`, `cli/`).
* **Layer 2B: `pelias-trainer`** — Layer-major $K$-window streaming LoRA/Mode-Q trainer and universal-use workflow (`config/`, `ingest/`, `data/`, `model/` [stores/LoRA/tape], `stream/`, `kern/` [trainer GEMM/bwd/loss/optim], `sched/`, `train/`, `export/`, `project/`, `startup/`, `intake/`, `help/`, `warn/`, `flow/`, `serve/`, `ui/`, `cli/`).

### 4.2 Complete Directory and File Scaffolding

```text
pelias/
├─ CMakeLists.txt  CMakePresets.json  .clang-format  .gitignore  .gitattributes
├─ VERSION  LICENSE (Apache-2.0)  CITATIONS.md  THIRD_PARTY_NOTICES.md
├─ README.md  README_PLAIN_ENGLISH.md  pelias-universal-build-master.md
├─ cmake/
│  ├─ PeliasOptions.cmake          Build options: GPU backend (CUDA/HIP/SYCL/VK), arch, host-only tests (T0/E2)
│  ├─ PeliasModule.cmake           Static library target registration and link-dependency enforcement (T0/E2)
│  └─ ProbeFlags.cmake             Feature checks: AVX2/AVX-512/AMX, FMA, GPU toolkit version, OS backend (E2)
├─ include/pelias/
│  ├─ status.h  version.h.in       Status codes, RAII result wrappers, version macros (T0/E2)
│  └─ log.h  trace.h  timer.h      Structured logging, debug partition traces, GPU event timers (E2)
├─ src/
│  ├─ os/                          [Layer 0: Shared OS Isolation Layer — Only place with OS headers (BC6)]
│  │  ├─ file_map.{h,cc}           POSIX mmap/madvise vs. Win32 CreateFileMappingW/MapViewOfFile (T1/E3)
│  │  ├─ net.{h,cc}                POSIX sys/socket.h + poll vs. Winsock2 + WSAPoll + SO_EXCLUSIVEADDRUSE (E10)
│  │  ├─ threads.{h,cc}            Thread creation, P/E-core & CCD topology, affinity pinning, clock, signals (E4)
│  │  └─ large_pages.{h,cc}        Optional 2 MiB THP / Win32 MEM_LARGE_PAGES / Linux hugetlbfs wrapper (E11)
│  ├─ hal/                         [Layer 0: Shared Hardware Abstraction & Memory Governor]
│  │  ├─ device.{h,cc}             Device selection, attribute queries, dynamic NVML/amd_smi/zes loader (T1/E2)
│  │  ├─ stream.{h,cc}             Stream wrapper with priority and StreamQuery flush helper (D27) (T1/E2)
│  │  ├─ event.{h,cc}              Event wrapper for timing and cross-stream ordering (T1/E2)
│  │  ├─ arena.{h,cu}              Lifetime-scoped bump arena allocator; zero-malloc loop enforcement (T1/E2)
│  │  ├─ vram_governor.{h,cc}      Hard 10.7 GiB ceiling enforcement, pre-allocation plan check, polling (T1/E2)
│  │  ├─ pinned_pool.{h,cc}        Unified pinned host memory pool (cudaHostAlloc/hipHostMalloc) + fallback (T1/E2)
│  │  ├─ host_alloc_wsl.cc         WSL2 pinned allocation backend & /proc/meminfo AnonHugePages logger (E2/E11)
│  │  ├─ host_alloc_win32.cc       Windows-native pinned allocation backend & perf counter reader (E2/E11)
│  │  ├─ host_alloc_linux.cc       Native Linux pinned allocation & /dev/cpu_dma_latency backend (E1/E11)
│  │  └─ sync_audit.{h,cc}         Debug stream-ownership verifier after every kernel launch (T1)
│  ├─ plan/                        [Layer 0: Shared Roofline & Memory Planner]
│  │  ├─ probe_constants.{h,cc}    Reader/writer for probe_constants.json and machine_state_sha (T1/E11)
│  │  ├─ roofline_solver.{h,cc}    Solves window size T, micro-batches K, slot count S, N_hide, cold split (T1/E11)
│  │  └─ memory_plan.{h,cc}        Computes exact arena capacities for Engine and Trainer before allocation (T1/E9)
│  ├─ bus/                         [Layer 0: Unified-Bus Coordination & Streaming]
│  │  ├─ doorbell_handoff.{h,cuh,cu} Cache-line-isolated GPU<->CPU mailbox, acquire/release flags, fallback (E6)
│  │  ├─ cpu_threadpool.{h,cc}     Physical-core-pinned SIMD worker pool polling the doorbell (E4)
│  │  └─ stage_streamer.{h,cu}     Trainer 3-stream DMA engine (S_compute, S_w H2D slots, S_x boundaries) (T10)
│  ├─ bank/                        [Layer 0: Shared Hot-Bank Optimizer]
│  │  ├─ hot_set_solver.{h,cc}     Water-filling (byte-hit h) and Zero-Miss (Z_l) residency solver (E5/T12)
│  │  └─ hot_set_io.{h,cc}         Byte-identical C++ reader/writer for PELIAS_HOT_SET profiles (P-E14) (E5/T12)
│  ├─ quant/                       [Layer 0: Shared Proprietary Block Quantizers & Dequantizers]
│  │  ├─ block_types.h             Exact byte layouts: Q4_K (144B), Q6_K (210B), Q8_0 (34B), q8_1 (36B), Q8_K (292B)
│  │  ├─ quant_cpu.{h,cc}          Bit-exact CPU reference quantizers/dequantizers (P-E6, P-Q1) (T1b/E4)
│  │  └─ quant_gpu.{h,cuh,cu}      Bit-exact GPU dequantizers (G8), q8_1 quantizer (G9), Mode Q requantizer (GQ)
│  ├─ io/                          [Layer 1: Shared Model & Data I/O]
│  │  ├─ sha256.{h,cc}             Native SHA-256 implementation for model keys and tensor splice proofs (T1/E3)
│  │  ├─ json.{h,cc}               Native zero-dependency JSON parser and serializer (T1/E3)
│  │  ├─ gguf_reader.{h,cc}        Memory-mapped GGUF v3 parser, bounds checker, typed TensorView (T1/E3)
│  │  ├─ gguf_writer.{h,cc}        GGUF v3 writer with --layout expert-last and pelias.pad 2 MiB alignment (T1b/T12)
│  │  ├─ gguf_splice.{h,cc}        Byte-splice exporter with per-tensor SHA-256 verification & manifest (T12)
│  │  └─ safetensors.{h,cc}        Memory-mapped safetensors shard header & payload reader for Stage A (T1b)
│  ├─ tok/                         [Layer 1: Shared Native Tokenizer & Chat-Template Engine]
│  │  ├─ utf8.{h,cc}               UTF-8 decoder, validator, encoder, and streaming incomplete-sequence buffer
│  │  ├─ unicode_data.{h,cc}       Static UCD general-category, canonical combining class, and NFC tables
│  │  ├─ normalizer.{h,cc}         Native NFC Unicode normalizer (T2/E10)
│  │  ├─ pretokenizer.{h,cc}       Hand-written state-machine matcher for Qwen split regex + byte mapping
│  │  ├─ bpe.{h,cc}                248,044-vocab / 247,587-merge rank table, BPE merge engine, piece cache
│  │  ├─ added_tokens.{h,cc}       33 added / 21 special token splitter and policy handler (O10)
│  │  ├─ tokenizer.{h,cc}          Public encode, decode, and streaming detokenizer interface (P-D1)
│  │  ├─ template_ast.{h,cc}       Lexer and AST parser for the 158-line Jinja chat-template subset
│  │  ├─ template_eval.{h,cc}      Subset evaluator (filters, tests, string methods, loop vars, tojson)
│  │  └─ chat_template.{h,cc}      Public chat renderer with reasoning_content split protection (P-D2)
│  ├─ model/                       [Layer 1 & 2B: Model Specification & Trainer Stores]
│  │  ├─ spec.{h,cc}               ModelSpec parsed from GGUF metadata; layer topology & expert byte tables
│  │  ├─ arch_moe.{h,cc}           Model-plugin interface for Qwen3.6 and other MoE/dense architectures (E3/E13)
│  │  ├─ frozen_store.{h,cc}       Trainer read-only mapped/pinned store for experts, embedding, lm_head, routers
│  │  ├─ trainable_core.{h,cc}     Trainer isolated core weights and Mode Q effective weight (W_eff) buffers
│  │  ├─ lora.{h,cu}               310-module LoRA fp32 masters, bf16 compute copies, Kaiming init
│  │  └─ tape.{h,cu}               Forward routing tape (40 layers x N tokens x top-8 IDs & weights) + histogram
│  ├─ kern/                        [Layer 1 & 2B: GPU Kernels — Depends ONLY on hal/ and quant/ (BC1)]
│  │  ├─ gemm_gguf/                dequant_ref.cu, mma_bf16.cu, mma_int8.cu, grouped_tiles.cu, dgrad_t.cu
│  │  ├─ route/                    router_topk.cu, tile_map.cu, combine.cu
│  │  ├─ elementwise/              rmsnorm.cu, swiglu.cu, gated_norm.cu, residual.cu, cast.cu
│  │  ├─ attn/                     rope.cu, attn_fwd.cu, attn_bwd.cu, attn_ref_blocked.cu
│  │  ├─ deltanet/                 gates.cu, conv1d.cu, l2norm.cu, chunk_prep.cu, chunk_state.cu, chunk_out.cu, chunk_bwd.cu, seq_ref.cu
│  │  ├─ loss/                     ce_chunk.cu (256-position chunked fp32 CE + Mode L losses)
│  │  └─ optim/                    adamw.cu, grad_norm.cu, lora_grad.cu
│  ├─ cpu/                         [Layer 2A: Engine Cold-Expert CPU Path]
│  │  ├─ expert_q4k_avx2.cc        AVX2/AVX-512/AMX Q4_K gate/up + SwiGLU + Q4_K down cold-expert kernel (E4)
│  │  └─ expert_q6k_avx2.cc        AVX2/AVX-512/AMX Q6_K down-projection cold-expert kernel (E4)
│  ├─ gpu/                         [Layer 2A: Engine Hot-Bank & Zero-Copy GPU Path]
│  │  ├─ bank.{h,cu}               Static VRAM hot-bank loader, variable per-layer slot tables, zero-slot init (E5)
│  │  ├─ expert_q4k.cu             Hot-branch Q4_K gate/up/down decode and small-batch kernel (E5)
│  │  ├─ expert_q6k.cu             Hot-branch Q6_K down-projection kernel (E5)
│  │  └─ zerocopy.cu               16-byte (uint4) shared-memory-staged PCIe zero-copy cold expert kernel (P-E10)
│  ├─ engine/                      [Layer 2A: Engine Execution, State, and Sampler]
│  │  ├─ kvcache.{h,cu}            q8_0 KV cache manager for the 10 full-attention layers (up to 131,072 ctx)
│  │  ├─ state.{h,cu}              30-layer DeltaNet recurrent state store and 62.9 MiB prefix-cache snapshots
│  │  ├─ layer.{h,cu}              Single MoE layer coordinator: spine -> router -> hot/cold -> doorbell -> merge
│  │  ├─ decode_loop.{h,cu}        CUDA-graphed 40-layer token decode loop, hit counters, per-token timers
│  │  ├─ prefill.{h,cu}            Chunked prefill executor with bypass threshold rule (E9)
│  │  └─ sampler.{h,cu}            Counter-RNG sampler chain: temperature, top-k, top-p, min-p, penalties
│  ├─ server/                      [Layer 2A: Native Raw-Socket HTTP/1.1 & SSE Server]
│  │  ├─ http.{h,cc}               State-machine HTTP/1.1 parser, chunked transfer decoder, SSE event writer
│  │  ├─ tools.{h,cc}              Native XML <tool_call><function=...><parameter=...> parser to OpenAI JSON
│  │  └─ api_openai.{h,cc}         /v1/chat/completions, /v1/models, reasoning_content splitter, request logger
│  ├─ config/                      trainer_config.{h,cc} (Defaults: Mode Q on, delta_max_bytes = 0, Expert overrides)
│  ├─ ingest/                      ingest_plan.{h,cc} (Stage A: 26 bf16 shards -> pelias_base_0step.gguf, D33)
│  ├─ data/                        dataset.{h,cc}, masks.{h,cc}, packer.{h,cc}, audit.{h,cc}
│  ├─ sched/                       layer_fwd.cu, layer_bwd.cu, head.cu, mtp.cu, step.{h,cu}
│  ├─ train/                       schedule.{h,cc}, optimizer.{h,cc}, checkpoint.{h,cc}, metrics.{h,cc}, trainer.{h,cc}
│  ├─ export/                      export_adapter.{h,cc}, export_core.{h,cc}
│  ├─ verify/                      canary.{h,cu}, kl.{h,cc}, dump_compare.{h,cc}, gate_runner.{h,cc}
│  ├─ project/                     model_key.{h,cc}, project_file.{h,cc}, run_folder.{h,cc} (U4)
│  ├─ startup/                     quick_pass.{h,cc}, machine_profile.{h,cc}, vram_scan.{h,cc} (U1, U2, U6)
│  ├─ intake/                      sheets.{h,cc}, import.{h,cc}, clients.{h,cc}, checks.{h,cc} (U7, U8)
│  ├─ help/                        help_catalog.{h,cc} (U10)
│  ├─ warn/                        fit.{h,cc}, loops.{h,cc}, works.{h,cc}, client_matrix.{h,cc} (U9)
│  ├─ flow/                        plan_card.{h,cc}, proposal.{h,cc}, expert.{h,cc}, summary.{h,cc} (U3, U10, U11)
│  ├─ serve/                       serve_command.{h,cc} (Unification bridge)
│  ├─ ui/                          pelias_lithic_ui.{hpp,cpp} (Lithic-to-Voxel UI with Dynamic Variable Drawer)
│  └─ cli/                         main_train.cc, main_engine.cc
├─ resources/                      sheets/ (persona.csv, parameters.csv, clients.csv), help/ (.txt), ui/ (.png/.json)
├─ profiles/                       Exported PELIAS_HOT_SET profiles & reference machine import profiles
├─ bench/                          Standalone hardware floor microbenchmarks (ram_bw, handoff, dual_bus, pin_modes, wide_load)
├─ tools/                          check_module_deps.py, check_resources.py, unicode table generator, oracle runners
└─ tests/                          unit/, smoke/, parity/, probes/, oracle/ (never linked into product)

```

### 4.3 Single-Sourced Acyclic Module Dependency Graph (`BC1`)

* **`os`**: `[]` | **`hal`**: `["os"]` | **`quant`**: `["hal"]` | **`plan`**: `["hal"]` | **`bus`**: `["os", "hal", "quant"]` | **`bank`**: `["hal"]`
* **`io`**: `["os", "hal", "quant"]` | **`tok`**: `[]` | **`kern`**: `["hal", "quant"]` | **`model`**: `["hal", "quant", "io"]`
* **`config`**: `[]` | **`ingest`**: `["hal", "quant", "io", "config"]` | **`data`**: `["io", "tok"]` | **`stream`**: `["hal", "bus", "model"]`
* **`sched`**: `["hal", "kern", "model", "stream"]` | **`train`**: `["hal", "config", "data", "model", "sched"]`
* **`export`**: `["hal", "quant", "io", "config", "model", "bank"]` | **`verify`**: `["hal", "io", "quant"]`
* **`cpu`**: `["os", "hal", "quant", "bus"]` | **`gpu`**: `["hal", "quant", "bank", "bus", "kern"]`
* **`engine`**: `["os", "hal", "plan", "bus", "bank", "quant", "io", "model", "kern", "cpu", "gpu"]` | **`server`**: `["os", "io", "tok", "engine"]`
* **`project`**: `["io"]` | **`startup`**: `["os", "hal", "plan", "io", "project"]` | **`intake`**: `["io", "tok", "data"]` | **`help`**: `[]`
* **`warn`**: `["io", "data", "verify", "intake"]` | **`flow`**: `["config", "plan", "startup", "project", "intake", "warn", "verify", "help"]`
* **`serve`**: `["project"]` | **`ui`**: `[]` | **`cli`**: `["config", "ingest", "train", "export", "verify", "flow", "serve", "engine", "server"]`

---

## 5. Proprietary Subsystem Specifications

1. **OS Isolation Layer (`src/os/`):** Encapsulates POSIX (`mmap`, `sys/socket.h`, `poll`, `pthread_setaffinity_np`, `clock_gettime`) vs. Win32/Winsock2 (`CreateFileMappingW`, `WSAPoll`, `SO_EXCLUSIVEADDRUSE`, `SetThreadAffinityMask`, `QueryPerformanceCounter`, `timeBeginPeriod(1)`). Supports partial unmapping of the `18.16 GiB` expert file region after pinning so host page cache is freed.
2. **VRAM Governor & Arenas (`src/hal/`):**
* **Engine Arenas:** `model` (`~1.54 GiB` spine/embd/lm_head/routers), `bank` (`7.5 GiB` = `4,170` experts + `40` all-zero slots = `72.6 MiB` so **no kernel ever receives a `-1` expert ID**), `session` (`q8_0` KV cache + `62.9 MiB` DeltaNet snapshots), `prefill`, `token`, and `host_pinned` (`18.164 GiB` or `18.65 GiB` via `cudaHostAlloc`).
* **Trainer Memory Plan (`T = 8,192`, up to `32k` step tokens):** Base trainer = **`7.90 GiB`** (`0.60` ctx + `0.80` core + `0.39` lm_head + `0.27` embd + `0.09` routers + `0.97` 2x498 MiB expert slots + `0.32` LoRA fp32 + `0.04` LoRA bf16 + `0.75` boundaries + `2.50` working set + `0.65` CE chunk + `0.02` tape + `0.50` safety margin). Adding Mode Q `W_eff` (`+0.80–1.10 GiB`) gives **`8.70–9.00 GiB`** (**`1.70–2.00 GiB` headroom** under `10.7 GiB`). Host pinned RAM = **`23.65–28.65 GiB`**.


3. **GGUF Reader, Writer & Zero-Cost `2 MiB` Alignment (`src/io/`):** 12-step bounds-checked parser (`io/gguf_reader`). In `io/gguf_writer`, `--layout expert-last` places all non-expert tensors first, inserts a `< 2 MiB` `pelias.pad` metadata string so the first expert tensor starts on an exact `2 MiB` (`2,097,152 B`) boundary, and writes all 123 expert tensors contiguously at **zero tensor byte overhead**.
4. **Block Quantizers (`src/quant/`):** Original C++20/GPU implementations of `Q4_K` (`144 B` / 256 elems), `Q6_K` (`210 B` / 256 elems), `Q8_0` (`34 B` / 32 elems), `q8_1` (`36 B` / 32 elems), and `Q8_K` (`292 B` / 256 elems), verified bit-identical to reference by `P-E6`, `P-Q1`, and `P-Q2`.
5. **Doorbell Coordinator (`src/bus/doorbell_handoff`):** Cache-line-isolated (`64 B`) request and reply mailboxes in `cudaHostAllocMapped` memory. Uses `acquire`/`release` (`MOV` on x86, avoiding `LOCK XCHG` per AMD Family 19h SOG §2.15, `G7`) and bounded `__nanosleep` backoff scoped to a single token (`~18–26 ms`) with automatic `stream_sync` timeout fallback.
6. **Native BPE Tokenizer & Chat-Template Interpreter (`src/tok/`):** Reads `248,044` vocab entries, `247,587` merges, and `33` added tokens (`21` special) from GGUF metadata; applies NFC normalization, hand-written state-machine pre-tokenization, byte-level BPE, and AST evaluation of the 158-line Qwen Jinja subset with explicit `reasoning_content` split protection (`P-D1`, `P-D2`).
7. **Native Raw-Socket HTTP/1.1 & SSE Server (`src/server/`):** Serves `/v1/chat/completions` and `/v1/models` on `127.0.0.1:8012` (with optional Ollama `:11434` alias), SSE `: ping` heartbeats during prefill, DeltaNet recurrent-state prefix snapshots (`62.9 MiB/slot`), `reasoning_content` splitting, native XML `<tool_call>`-to-OpenAI JSON parsing, and structural-only request telemetry.

---

## 6. Trainer Precision Plan, Layer-Major Scheduler, Stage A Ingest & Optional Modes

1. **Stage A Ingest (`src/ingest/`, `D33`):** Streams the 26 `bf16` safetensors shards (`71.90 GB`), drops `model.visual.*` (`333` tensors), slices fused `gate_up_proj` `[256, 1024, 2048]` at row `512`, applies V-head tiled permutation (`P-S3`) and norm `+1` folding (`P-S2`), quantizes with `quant/quant_cpu` to the stock `Q4_K_M` map, and writes `pelias_base_0step.gguf` (`753` tensors, `20.22 GiB`).
2. **Mode Q Merge-Aware Training (`D23`, `D31`, `D32` — Default):** Rebuilds $W_{\text{eff}} = Q_T(\text{deq}(W_{\text{base}}) + \frac{\alpha}{r} B A)$ once per optimizer step using the bit-exact quantizer `GQ`, runs forward/recompute directly on $W_{\text{eff}}$, backpropagates via STE to `fp32` masters $A, B$, and exports the final step's $W_{\text{eff}}$ bytes into `merged-core.gguf` at **`+0 extra bytes`** (`Delta_max = 0`) alongside `adapter.gguf`.
3. **Layer-Major $K$-Window Scheduler (`D5`, `D7`, `D13`, `D27`):** Sweeps layers `0..39` forward (recording `Tape[40][K]`) and `39..0` backward (reusing resident layer 39 at turnaround -> **`79 fetches = 35.8 GiB/step = 1.49 s/step`** at `26.1 GB/s`), flushing each stream batch via `StreamQuery` (`D27`, `0.92` overlap) so PCIe transfers are 100% hidden once a step holds `>= 3,072` tokens.
4. **Optional Gated Modes:** **Mode L / L-Z / L-C (`D21`)** trains the 41 `F32` routers (`+0.32 GiB`, `+0.21%` FLOPs) to favor resident bank experts $R_l$, gated on `G-L1`–`G-L4`. **Mode B / Mode N (`D24`)** aligns `Q8_K` (256-elem CPU) and `q8_1` (32-elem GPU) activation quantization, gated on `P-E12`. **MTP Co-Training (`D12`)** trains `blk.40` with loss weight $\lambda_{\text{mtp}}$, gated on `P-S5` and `P-T8`.

---

## 7. Engine Speed-to-Ceiling Attribution & Trainer Roofline

* **Engine Ceiling (`7.1`):** $T_{\text{ceiling}} = T_{\text{spine}} (7.73\text{ ms}) + (1 - 0.8273)\frac{608\text{ MB}}{BW_{\text{cold}}} + 40 \times 6.17\text{ }\mu\text{s} = \mathbf{10.87\text{ ms}}$ (**`92.0 tok/s`** at `26.2 GB/s` zero-copy) to **`11.34 ms`** (**`88.2 tok/s`** at `9.59 GB/s` narrow compute kernel).
* **Verified Engine Runs (Patched `llama.cpp` Tier Build; `57.0 tok/s` Retired):**
* *WSL2 Reference Cell A (`7.0 GiB`, bypass 32, `-ub 4096`):* **`37.9 tok/s`** (`26.39 ms`) short; **`42.1 tok/s`** (`23.75 ms`) at `12k`.
* *Windows-Native Cell A (`7.0 GiB`, bypass 32, `-ub 4096`):* **`47.3 / 46.8 tok/s`** (`21.14 / 21.37 ms`) short; **`53.7 / 54.1 tok/s`** (`18.62 / 18.48 ms`) at `12k`.
* *WSL2 Best Recorded Cell (`7.5 GiB`, bypass 32, `-ub 2048`, gauntlet):* **`53.5 / 53.0 / 49.0 tok/s`** at short / `4k` / `12k` (`1,680.3 tok/s` prefill with `--load-mode none`).
* *WSL2 MTP Speculative (`draft 2/3`, `7.0 GiB` bank):* **`51.5–52.9 tok/s`** at `12k`.


* **Trainer Speed Targets (`7.3`):** Minimum viable **`600 tok/s`** (`2.4x` v0.2.1's `247.85 tok/s`), Target **`1,200 tok/s`** (`4.8x`), Stretch **`2,000 tok/s`** (`8.1x`).

---

## 8. Interdependencies, Invalidation Matrix & Two-Way Contract

See Section 8 of `pelias-master.md` (incorporated intact: Directed Dependency Chain `8.1`, 13-Row Invalidation Matrix `8.2`, and Two-Way Engine-Trainer Milestone Contract `8.3` across `E0`–`E12`).

---

## 9. Master Order of Operations

1. **Phase I:** Wave 0 hardware probes (DONE on WSL2 & Win11) + Batch C kernel probes (`P-K1`–`P-K3`, `P-Q10`) + `P-M4` model read speed -> Author signs T0 switch summary.
2. **Phase II:** Ground-up `pelias-trainer` execution (`T1` Foundations -> `T1b` Stage A Ingest -> `T2` Native Tokenizer/Template/Data -> `T3`–`T6` GEMM/Route/Attn/DeltaNet Kernels -> `T7`–`T9` Forward/Backward/LoRA/Mode-Q Parity -> `T10`–`T12` 3-Stream Scheduler, Checkpoints, Byte-Splice Exporter -> `T13`–`T14` Speed & Optional Modes).
3. **Phase III:** Universal-Use & Lithic-to-Voxel UI (`U1`–`U11`: `project/`, `startup/`, `intake/`, `help/`, `warn/`, `flow/`, `ui/`).
4. **Phase IV:** Ground-up `pelias-engine` & native `server/` (`E0`–`E13`), **HELD** until lifted by the author (`O13`), reusing Layer 0 and Layer 1 modules built in Phase II.
5. **Phase V:** Unification (`serve/serve_command`, shared `probe_constants.json`, single cohesive release passing `PG1`–`PG10` with `tests/oracle/` removed).

---

## 10. Verification Ladders, Oracles, Kernel Tolerances, and Three-Tier KL Gates

### 10.1 The Three Oracles and Conflict Resolution Rules (`10.1`, `10.6`)

1. **Oracle A — External Pinned `llama.cpp` (`2ca15f540` / `b11399` in `tests/oracle/` and `tools/oracle_llama/`):** The served truth for forward-pass semantics, per-layer activation dumps (`eval-callback` on the clean `P-X0` baseline GGUF, `D36`), and logit KL divergence. Never linked into product binaries.
2. **Oracle B — Proprietary `fp64` CPU Reference (`tests/ref/`):** Unambiguous double-precision CPU implementation of every forward and backward operation (sequential DeltaNet, materialized attention, exact block dequantization). Checked first against central finite differences in `fp64` ($\frac{f(\theta + \epsilon u) - f(\theta - \epsilon u)}{2\epsilon}$ matching analytic directional derivatives to $\le 10^{-8}$ relative error, `P-T1`).
3. **Oracle C — Test-Only PyTorch `fp32` Autograd Slice (`tools/oracle_torch/`):** Independent 4-layer slice with inverse V-head and norm `+1` permutations to cross-check backward LoRA gradients (`P-T2`).
4. **Conflict Resolution:** Forward conventions (`+1` norm, RoPE pairing, conv tap order, top-k ties) -> **Oracle A wins** (float spread between legal Oracle A batch configs defines $\sigma_{\text{float}}$). Backward gradients -> **Oracle B (verified by `fp64` finite differences) wins**. Fast GPU kernels -> judged against **Oracle B on identical inputs**.

### 10.2 Rung-1 Kernel Tolerance Targets (`6.11` — Restored & Complete)

Every kernel in `src/kern/`, `src/cpu/`, and `src/gpu/` must pass these tolerances against Oracle B across normal shapes, tile-boundary $\pm 1$ shapes, $M_e \in \{0, 1, 7, 8, 63, 64, 65\}$, all-tokens-to-one-expert, and zero-tokens-to-all-other-experts (the negative-ID / hole crash test), with `compute-sanitizer --tool memcheck,racecheck,initcheck,synccheck` reporting `0` errors:

| Kernel ID & Family | Operation | Max Relative L2 Error (vs. `fp64` Oracle B) | Max Elementwise Abs Error / Exactness Rule |
| --- | --- | --- | --- |
| **`G8`, `quant_cpu**` | `Q4_K`, `Q6_K`, `Q8_0`, `q8_1`, `Q8_K` dequant & quant | **`0.0` (Bit-exact)** | **0 ULP (`100%` bit-identical to `ggml` reference across all 753 tensors, `P-Q1`, `P-Q2`, `P-E6`)** |
| **`G1`, `G3**` | Dense & Grouped MoE `bf16` MMA Forward GEMM (`fp32` accum) | `<= 1.0e-3` | `<= 5.0e-3` (tested across $M_e \in \{0, 1, 7, 8, 63, 64, 65\}$) |
| **`G2`, `G4**` | Dense & Grouped MoE `int8` (`q8_1`) Serve-Faithful Forward GEMM | `<= 2.0e-3` vs. `G8`; `<= 1.0e-5` vs. Oracle A MMQ | Matches `ggml` MMQ `int32`-accum + `fp32`-scale path (`P-Q8`) |
| **`G5`, `G6`, `G7**` | Transposed-dequant `dgrad` GEMM & LoRA `dA, dB` epilogue | `<= 1.0e-3` | Cosine similarity `>= 0.99999` vs. `fp64` finite-diff; bitwise deterministic across runs (`O3`) |
| **`GQ`** | Mode Q Effective Weight Rebuild ($W_{\text{eff}} = Q_T(\text{deq}(W) + \Delta W)$) | **`0.0` (Bit-exact)** | Bit-identical to `quant_cpu` reference quantizer (`P-E6`, `P-E8`) |
| **`R1`, `R2`, `R3**` | Router `fp32` FMA projection, top-8 argmax, tile map, combine fwd/bwd | `<= 1.0e-6` | **`100%` identical top-8 expert ID sets** on synthetic & tie-seeded inputs (`P-K6`, `P-Q12`) |
| **`E1`–`E5**` | RMSNorm (`+1` & plain), SwiGLU, Gated RMSNorm, L2-norm, Residual/Cast | `<= 1.0e-5` (fwd) / `<= 5.0e-5` (bwd) | `0` NaN/Inf on zero/denormal/outlier inputs |
| **`A1`–`A4**` | IMRoPE `[11,11,10,0]`, FlashAttention-2 GQA (`d=256`) fwd & bwd | `<= 5.0e-4` (fwd) / `<= 1.0e-3` (bwd) | `0` register spills in `ptxas -v` (`P-K7`); `cu_seqlens` document-boundary isolation exact (`P-D3`) |
| **`D1`–`D7**` | Gated DeltaNet gates, `conv1d`, chunked WY/UT (`C=64`) fwd & reverse bwd | `<= 5.0e-4` (fwd vs. `D7` `fp32`) / `<= 1.0e-3` (bwd vs. `D7` `fp64`) | `T = (I+N)^{-1}` condition number monitored (`P-S7`, `P-K8`–`P-K10`) |
| **`L1`** | 256-position chunked `fp32` Cross-Entropy + log-sum-exp + Mode L losses | `<= 1.0e-6` | Step-wide denominator exact across arbitrary window splits $K$ (`P-K11`, `P-T5`) |
| **`O1`, `O2`, `O3**` | Fused `fp32` AdamW, two-level global grad norm/clip, tree-reduced LoRA grad | `<= 1.0e-6` | **Bitwise identical (`0` bit diffs)** across repeated runs and `K=1 vs. K=2 vs. K=4` (`P-T4`, `P-T5`) |

### 10.3 The Seven-Rung Verification Ladder (`10.2`)

| Rung | Scope | Passing Criterion | Governing Probes |
| --- | --- | --- | --- |
| **Rung 1** | Individual Kernels (`kern/`, `cpu/`, `gpu/`) | Passes all Section 10.2 tolerances against Oracle B + `0` `compute-sanitizer` errors. | `P-Q1`, `P-Q2`, `P-K4`–`P-K12` |
| **Rung 2** | Single Sub-Layer & Single Layer Forward (`sched/layer_fwd`, `engine/layer`) | Per-layer hidden state dumps match Oracle A (`eval-callback` on `P-X0`) within the measured single-layer float spread; router top-8 agreement `>= 99.5%`. | `P-S2`–`P-S8`, `P-Q8` |
| **Rung 3** | Full 40-Layer Forward Parity | Full-model teacher-forced logits on the 16x1,024-token eval set satisfy **Gate `K1**` ($\text{KL} \le 1.0 \times \sigma_{\text{float}}$, top-1 agreement within `0.5%` of Oracle A). | `P-Q4a`, `P-Q5`, `P-Q6` |
| **Rung 4** | Single-Layer & 4-Layer Backward Gradients (`sched/layer_bwd`) | Analytic gradients match `fp64` central finite differences (`<= 1e-4` rel error) and Oracle C (`tools/oracle_torch/`) with cosine similarity `>= 0.99999`; routing tape guarantees `0` router flips during recompute. | `P-T1`, `P-T2`, `P-T5`, `P-D3` |
| **Rung 5** | 3-Stream Layer-Major Scheduler & Arena Governor | `0` dynamic allocations in hot loop (`P-M5`); exactly `79` expert fetches/step; peak VRAM `<= 10.7 GiB` (`P-M1`, `P-M2`); stream stall `<= 5%` of step time at `N >= 3,072` tokens (`P-T7`). | `P-M1`–`P-M5`, `P-T6`, `P-T7` |
| **Rung 6** | Training Loop, Determinism, Convergence & Resume | 32-step micro-overfit reduces loss by `>= 90%` (`P-T3`); two runs with same seed are **bitwise identical** (`P-T4`); step 10->20 resumed from step-10 checkpoint is **bitwise identical** to uninterrupted run (`P-T11`); matches/beats v0.2.1 eval loss on same recipe (`P-T9`). | `P-T3`, `P-T4`, `P-T9`, `P-T11` |
| **Rung 7** | Byte-Splice Export, Engine Serve & Client Matrix | All 123 frozen expert tensors match base GGUF SHA-256 (`P-X1`); exported GGUF passes **Gate `K2**` (`P-X2`) and **Gate `K3**` (`P-X5`); engine peaks `<= 10.7 GiB` with `<= 2%` decode regression; passes loop/retention/client-matrix checks (`U8`, `U9`). | `P-X1`–`P-X7`, `P-E14` |

### 10.4 The Three-Tier KL Divergence Gates (`K1`, `K2`, `K3` — Author Ruling `H6`)

Because stock `llama.cpp` at batch 16 and batch 1024 differs from stock `llama.cpp` at batch 8 by a measured floating-point reduction spread of $\sigma_{\text{float}} = \mathbf{0.017353}$ (batch 16) and $\mathbf{0.018286}$ (batch 1024) on the 16x1,024-token set (`MEASURED`, Engine Ledger §8.3), every KL gate is anchored to $\sigma_{\text{float}}$ (with strict `KL <= 0.005` / `top-1 >= 99%` logged alongside for information):

1. **Gate `K1` (Trainer Forward vs. Oracle A on Base GGUF):** $\text{KL}(\text{Trainer}_{0\text{-step}} \parallel \text{Oracle A}) \le 1.0 \times \sigma_{\text{float, base}}$.
2. **Gate `K2` (Exported Merged GGUF vs. Trainer In-Memory Reference):** Under Mode Q (`D31`), $W_{\text{eff}}$ is already quantized in forward, so $\text{KL}(\text{Exported GGUF} \parallel \text{Trainer } W_{\text{eff}}) \le 1.0 \times \sigma_{\text{float, new}}$ (zero merge-step erase).
3. **Gate `K3` (`pelias-engine` Dual-Bus Serve vs. Oracle A on Exported GGUF):** $\text{KL}(\text{pelias-engine} \parallel \text{Oracle A}) \le 1.0 \times \sigma_{\text{float, new}}$ (measured by `P-Q4b` on the newly exported file).

### 10.5 Pre-Flight and Release Gates (`PG1`–`PG10`)

* **`PG1` (Boundary & Hygiene Gate):** `BC1`–`BC6` pass (`0` forbidden headers/symbols, `100%` provenance lines, `0` private strings/paths/tokens).
* **`PG2` (Corrupt-Input Gate):** `io/gguf_reader`, `io/safetensors`, `io/json`, and `data/dataset` cleanly reject 1,500 mutated/truncated inputs without crashing or leaking memory.
* **`PG3` (Governor Gate):** `hal/vram_governor` refuses any synthetic plan exceeding `10.7 GiB` (or the probe-derived ceiling) before allocating a single byte.
* **`PG4` (Tokenizer & Template Gate):** `P-D1` (`0` token ID diffs across full dataset + 10,000 multilingual/code/edge strings), `P-D2` (`0` assistant round-trip diffs, `<think>` split protected), and `P-D5` (`0` loss-mask diffs).
* **`PG5` (Bit-Exact Quantizer & Splice Gate):** `P-E6` (`0` byte diffs across all 753 tensors) and `P-X1` (`123/123` expert tensor SHA-256 matches).
* **`PG6` (Determinism & Resume Gate):** `P-T4` and `P-T11` pass with `0` bit differences in weights, gradients, and Adam moments.
* **`PG7` (Engine Speed & Memory Gate):** Peak VRAM `<= 10.7 GiB`, 12k decode clears Gate `G3` (`>= 55.0 tok/s`), 12k prefill `>= 1,670 tok/s`, and needles `12/12` up to `120k` context.
* **`PG8` (Client Replay & Tool-Call Gate):** Replays recorded Visual Studio 2026 Copilot, `odysseus-ai`, Ollama-alias, and Web UI sessions (`clients.csv`) with `100%` valid SSE framing, `reasoning_content` separation, and XML tool-call JSON extraction.
* **`PG9` (Cross-Platform Parity Gate):** Both WSL2 and Windows 11 Native builds compile cleanly with `-Wall -Wextra -Werror` / `/W4 /WX` and pass `PG1`–`PG8`.
* **`PG10` (Oracle-Free Standalone Gate, `BC5`):** With `tests/oracle/` and `third_party/` deleted from disk, `pelias-train` and `pelias-engine` execute end-to-end without error.

---

## 11. Universal Multi-Vendor, Multi-OS & Guided Intake Architecture

### 11.1 The Five Universal Hardware Requirements (`R1`–`R5`) and Automatic Fallbacks

Tested automatically by `pelias probe` (`startup/quick_pass`) on any machine:

* **`R1` (Zero-Copy Host Reads):** `cudaHostAllocMapped` (NVIDIA), `hipHostMallocMapped` (AMD ROCm), `zeMemAllocHost` (Intel Level Zero), or `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_CACHED_BIT` (Vulkan). On Unified Memory APUs (Ryzen AI / Halo Strix / Core Ultra / Grace), `bank.size_gib` is set to `0` and all `18.65 GiB` experts are mapped zero-copy in place with zero H2D copies.
* **`R2` (Sub-20-us Flag Visibility) & `R3` (Resident Waiting Kernel):** System-scope acquire/release atomics (`ld.global.acquire.sys` / `__hip_atomic_load` / `sycl::atomic_ref`). **Universal Fallback:** If a target GPU/driver fails `R2` or `R3`, `handoff.mode` automatically falls back from `doorbell` to `stream_sync` and prices the exact latency into the roofline plan card.
* **`R4` (Sub-Block Dot Products):** NVIDIA Tensor Cores (`mma.sync`), AMD Matrix/WMMA (`v_wmma` / `v_dot4_i32_i8`), Intel XMX DPAS, or Vulkan `VK_KHR_cooperative_matrix`.
* **`R5` (Async Copy Engines):** Explicit H2D + D2H + Compute overlap with automatic `StreamQuery` batch flushing (`D27`).

### 11.2 Universal CPU SIMD & Topology Matrix (`src/cpu/`, `src/os/threads`)

* **AMD Zen 2 / Zen 3 / 5800X3D:** `AVX2 + FMA + F16C` (`Q8_K` 256-elem cold activations); `physical_1t` pinning (1 worker per physical core to avoid Zen 3 50/50 SMT Load Queue split, `G6`); `acquire`/`release` (`MOV`, zero `LOCK` before FP, `G7`); optional `_MM_HINT_NTA` to protect 96 MB victim L3 (`G8`).
* **AMD Zen 4 / Zen 5:** `AVX-512` (`VNNI` + `BF16`); dual-CCD awareness pins workers to the 3D V-Cache CCD first.
* **Intel 12th–15th Gen Hybrid (P+E Cores):** `AVX2 + AVX-VNNI` (`VPDPBUSD`); `os/threads` reads core efficiency classes and pins decode and primary cold workers to **P-cores (`1T`)** first (`cpu.use_ecores = false` default).
* **Workstation / Multi-Socket Xeon & Threadripper:** `AVX-512 + AMX` or quad/eight-channel `AVX2`; binds the pinned expert store and worker pool to the NUMA node attached to the GPU's PCIe root complex.

### 11.3 Three-Path Universal Configuration (`src/intake/`, `src/flow/`, `src/ui/`)

1. **Path A — Universal Non-Executing File Importer (`--import <file>`):** Accepts `.txt` INI/key-value profiles, `.json` profiles, `.csv` sheets (`persona.csv`, `parameters.csv`, `clients.csv`), `.log`/`.tsv` probe outputs, `.py` parameter scripts (parsed via static AST only, never executed), and `.md` blueprints. Every imported training row enters as `DRAFT`; every hardware/parameter override is checked against physical limits before Start unlocks.
2. **Path B — 4-Step Guided Selector (Zero-File Startup):** Auto-detects hardware via `quick_pass` + `vram_scan` (`U6`, proposing closeable GPU apps) -> selects local model or built-in Model Preset -> presents plain-English Quick-Adjust sliders -> displays the 2–4 option **Plan Card (`U3`)**.
3. **Path C — Dynamic Variable Injection Drawer (`[ + Add Variable ]` in Expert Mode, `U11`):** Allows live search and injection of any parameter from the Master Parameter Dictionary into the open window, badges overrides as **`USER-FORCED`**, runs a 2-second real-allocation pre-test, and repeats all active cautions on the final confirmation summary (`flow/summary`).

```

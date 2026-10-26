# PELIAS MASTER SPECIFICATION (`pelias-master.md`)

> **Redacted for publication (2026-10-26):** the private model alias and the author's drive paths are replaced by `<model alias>` and `<drive path>`; everything else is word for word. The unredacted copy is kept local-only.

**A 100% Proprietary, Ground-Up C++20 and CUDA Architecture for the Unified Pelias Runtime: Substrate, Core, Inference Engine, Native Server, and LoRA/QAT Trainer.**

| Field | Value |
|---|---|
| **Document** | `pelias-master.md` — Ground-Up Master Specification v1.0 |
| **Date** | 2026-10-25 |
| **Copyright** | Copyright (c) 2026 John Lipka and the Pelias contributors |
| **Licence** | Apache-2.0 (Proposed) |
| **Status** | **Master Architectural Specification. Zero code included.** Wave 0 hardware probes are measured on WSL2 and Windows-native. Engine execution phases (`E2`–`E13`) are HELD per author ruling (`2026-10-24 06:07`) until lifted; shared foundation modules needed by the trainer are built first. |
| **Target Model** | Qwen3.6-35B-A3B (`qwen35moe` architecture, 40 trunk layers + 1 MTP block `blk.40`, 753 GGUF tensors / 1,045 `bf16` safetensors) |
| **Reference Hardware** | NVIDIA GeForce RTX 4070 12 GB (`AD104`, `sm_89`) + AMD Ryzen 7 5800X3D (8C/16T, AVX2) + 64 GB DDR4-3200 + PCIe 4.0 x16 |
| **Target Platforms** | Dual-target build from one codebase (`[sync R1]`): WSL2 (Ubuntu 26.04.1 LTS, kernel 6.18.54.1, `nvcc 13.0.88`, GCC 15.2.0) and Windows 11 Pro Native (build 26300, `nvcc 13.4.92`, MSVC 19.44). Native Linux is held as a separate entity for later. |
| **Runtime Mandate** | **100% Proprietary Ground-Up (`D37`, `[sync R2]`).** Zero runtime files, headers, libraries, or code from external AI or utility frameworks (`llama.cpp`, `ggml`, `libllama`, `common`, PyTorch, PEFT, Triton, Unsloth, or third-party JSON, HTTP, Regex, or Jinja libraries). |

---

## 1. Governance, Honesty System, Standing Rules, and the Proprietary Boundary

### 1.1 The Eight Honesty Labels

Every numerical value, architectural assertion, and performance target in this specification carries one of the following eight labels. Where a quantity is not yet measured, it is explicitly marked as `UNVERIFIED` or `ESTIMATE` and paired with the verification probe that measures it.

| Label | Definition |
|---|---|
| **MEASURED** | Measured directly on the reference hardware and recorded in a verified project ledger or raw probe output file. |
| **READ-SRC** | Read directly from the binary header of the served `Q4_K_M` GGUF, the 26 `bf16` safetensors shard headers, or reference specification sources (`2ca15f540`). Exact as a structural description of what the data or reference is. |
| **PUBLISHED** | Taken from primary vendor documentation (NVIDIA, AMD, Microsoft, JEDEC) or peer-reviewed academic literature. |
| **DERIVED** | Computed via scripted arithmetic from labelled inputs. All inputs and formulas are explicitly stated. |
| **ESTIMATE** | Engineering judgement. Never relied upon as fact; always paired with a named verification probe that replaces it with a measurement. |
| **UNVERIFIED** | A hypothesis or external claim that has not yet been tested on the reference hardware. Not relied upon until its probe passes. |
| **DECISION** | A design choice or author ruling with a recorded engineering reason. Reversible unless marked locked; items marked `DECISION (proposed)` await final author confirmation. |
| **AUTHOR-STATED** | A figure, observation, or requirement stated by the author where no raw log file is attached; paired with a verification probe where empirical. |

**Units and Conventions:**
* Binary vs. Decimal: $\text{GiB} = 2^{30}\text{ bytes}$, $\text{MiB} = 2^{20}\text{ bytes}$, $\text{KiB} = 2^{10}\text{ bytes}$, $\text{GB} = 10^9\text{ bytes}$, $\text{MB} = 10^6\text{ bytes}$.
* Training Throughput ($\text{tok/s}$): Defined as *processed tokens per second* (every token passing through the network, whether assistant or prompt), matching the baseline ledger convention (`P-T10`: $202.7\text{ min} \times 247.85\text{ tok/s} \approx 3.01\text{ M}$ processed tokens vs. $1.97\text{ M}$ assistant tokens over 2 epochs).
* Inference Throughput ($\text{tok/s}$): Separated strictly into *prompt prefill* ($\text{tok/s}$ at depth) and *token decode* ($\text{tok/s}$ at short/`ctx0`, $4\text{k}$, and $12\text{k}$ context depths).

### 1.2 The Author's Standing Rules

1. **Strictly Ground-Up (`D37`, `[sync R2]`):** No part of the Pelias runtime is a wrapper around PyTorch, PEFT, Unsloth, TRL, Triton, `llama.cpp`, `ggml`, `libllama`, or `common`. Every runtime module—GGUF reader/writer, safetensors reader, SHA-256, JSON parser/serializer, block quantizers/dequantizers, byte-level BPE tokenizer, chat-template subset renderer, raw-socket HTTP/1.1 and SSE server, arena allocator, doorbell coordinator, AVX2 threadpool, and CUDA spine/expert/LoRA kernels—is original code written for this project.
2. **Never Truncate:** Any training sample whose tokenized length exceeds the maximum training window ($8,192$ tokens) is a hard failure with an audit report listing the offending row IDs. Silent truncation is forbidden.
3. **Hard VRAM Ceiling (`D26`, `D28`, `[sync R3]`):** Enforced in code by `hal/vram_governor`. The ceiling is dynamically derived at process startup from the quick probe: $\min(\text{CUDA free}, \text{NVML free})$ after CUDA context creation, minus a safety margin. On the reference machine, $10.7\text{ GiB}$ is the governing threshold ($10.83\text{ GiB}$ free minus $0.13\text{ GiB}$ margin) until the disturbance test sets the final margin. Any memory plan exceeding the ceiling is refused before the first allocation.
4. **Zero Allocations in Hot Loops:** All device and host memory is pre-allocated in lifetime-scoped arenas (`hal/arena`, `hal/pinned_pool`) before step 1 (trainer) or token 1 (engine). Calling `cudaMalloc`, `cudaFree`, `new`, or `malloc` inside the training step loop or token decode loop is a fatal error caught by debug hooks (`P-M5`).
5. **No Model Downloads:** The system never fetches models from the internet. The user supplies local paths to the original `bf16` safetensors shards (for Stage A ingest) or an existing GGUF.
6. **Provenance and Clean-Room Hygiene:** The first line of every source file must state: `// provenance: from the spec: <name>`, `// provenance: adapted from <X>, licence <Y>`, or `// provenance: original`, followed by `Copyright (c) 2026 John Lipka and the Pelias contributors`. Where Pelias bytes must equal `ggml` block layouts or quantization arithmetic for GGUF interoperability, the arithmetic is re-expressed in original code after reading the specification, and credited in `CITATIONS.md`. Zero AGPL or LGPL code (e.g., Unsloth) is ever copied.
7. **Privacy Rails:** Dataset text, persona/identity strings, private mathematical equations, local user-profile paths, machine serials, and credentials (such as Hugging Face tokens) never enter any repository file, log, or report. Dataset audits and verification reports emit counts, row IDs, lengths, and pass/fail flags only.
8. **Execution Discipline:** Every step is gated by the verification rung beneath it. The user executes all GPU-touching binaries on the host machine. Git pushes occur only on the author's explicit command.

### 1.3 The Proprietary Boundary and Enforcement Checks

| Classification | Permitted / Forbidden Components | Governing Rule |
|---|---|---|
| **Proprietary Product Code** | `os/`, `hal/`, `plan/`, `bus/`, `bank/`, `quant/`, `io/`, `tok/`, `kern/`, `model/`, `engine/`, `cpu/`, `gpu/`, `server/`, `config/`, `ingest/`, `data/`, `stream/`, `sched/`, `train/`, `export/`, `verify/`, `project/`, `startup/`, `intake/`, `help/`, `warn/`, `flow/`, `serve/`, `cli/` | 100% original C++20 and CUDA code written to this specification. |
| **Allowed System Dependencies** | CUDA Runtime & Driver API (`cudart`, `cuda`); NVML (loaded dynamically at runtime via `dlopen`/`LoadLibrary` so binaries start without it); C++20 Standard Library; x86-64 AVX2/FMA/F16C compiler intrinsics; OS kernel interfaces (POSIX on WSL2; Win32 & Winsock2 on Windows); `cuBLASLt` (optional, dense `bf16` only, retained only if `P-K1` shows hand-written GEMMs $< 0.85 \times \text{cuBLASLt}$). | Listed in `THIRD_PARTY_NOTICES.md` with exact versions; never redistributed. |
| **Open UI Dependency (`O9`)** | Dear ImGui (MIT Licence) in `src/ui/` (trainer interface only; never in `pelias-engine` or `server/`). | Author decides (`O9`): keep for UI only, write custom immediate-mode layer, or ship headless first. |
| **Static Data Tables** | Unicode Character Database (UCD) general-category and NFC canonical-composition tables in `tok/unicode_data`. | Public standard data compiled into static C++ tables by an original offline generator tool. |
| **Strictly Forbidden in Product** | `llama.cpp`, `ggml`, `libllama`, `common`, PyTorch, `transformers`, PEFT, Triton, Unsloth, `nlohmann/json` or any JSON library, `cpp-httplib` or any HTTP library, PCRE/std::regex or any regex library, Jinja2Cpp or any template library, Python interpreter at runtime. | Enforced on every build by Boundary Checks `BC1`–`BC6`. |
| **External Read-Only Test Oracles** | Pinned, untouched `llama.cpp` commit `2ca15f540` (`b11399`) in `tests/oracle/` and `tools/oracle_llama/`; test-only Python/PyTorch `fp32` autograd slice in `tools/oracle_torch/`; `gguf-py`. | Executed as separate processes during verification gates only. Never linked into, started by, or required at runtime by any product binary. |

**Mandatory Build-Time and Release-Time Boundary Checks (`DECISION`):**
* **`BC1` (Acyclic Module Graph):** `tools/check_module_deps.py` verifies on every build that the `#include` and CMake link graph is strictly acyclic and that no product module includes headers from `tests/` or `third_party/`.
* **`BC2` (Link-Map Symbol Audit):** Every product binary's linker map is scanned on every build to guarantee zero symbols from `ggml`, `llama`, or forbidden libraries.
* **`BC3` (Header Allow-List):** Product translation units may include only repository headers, C++20 standard library headers, CUDA/NVML headers, x86 intrinsics headers, and OS headers (restricted by `BC6`).
* **`BC4` (Provenance Header Audit):** `tools/check_resources.py` verifies that line 1 of every source and header file carries a valid provenance tag and copyright notice.
* **`BC5` (Oracle-Free Execution Gate):** Release gate (`PG10`) verifying that `pelias-engine` and `pelias-train` start, probe, plan, load, train, export, decode, and serve HTTP/SSE requests when `tests/oracle/` and `third_party/` are completely removed from disk.
* **`BC6` (OS Isolation Rule):** POSIX (`sys/socket.h`, `sys/mman.h`, `pthread.h`, `unistd.h`) and Win32 (`windows.h`, `winsock2.h`, `ws2tcpip.h`) headers are forbidden outside `src/os/` and `src/hal/host_alloc_{wsl,win32}.cc`.

---

## 2. Hardware Fact Base, Measured Wave 0 Floors, and Toolchain

### 2.1 Reference Machine Specifications

| Subsystem | Specification | Label & Source |
|---|---|---|
| **GPU Hardware** | NVIDIA GeForce RTX 4070 12 GB (`AD104-250`, `sm_89`): 46 SMs, 5,888 CUDA cores, 184 4th-Gen Tensor Cores, $36\text{ MiB}$ L2 cache ($24.75\text{ MiB}$ max persisting window), 192-bit GDDR6X bus, $504\text{ GB/s}$ stock bandwidth, $200\text{ W}$ default power limit ($216\text{ W}$ max), $2,475\text{ MHz}$ rated boost, $100\text{ KiB}$ L1/shared memory per SM ($101,376\text{ B} = 99\text{ KiB}$ max opt-in dynamic shared memory per block), 5 async copy engines. | `PUBLISHED` (Ada Whitepaper) & `MEASURED` (`P-H1`, HWiNFO64) |
| **GPU Overclock** | MSI Afterburner: Memory $+1000\text{ MHz}$ (`nvidia-smi` reports $11,251\text{--}11,501\text{ MHz}$ vs. $10,501\text{ MHz}$ stock = $23,003.9\text{ MT/s}$ effective vs. $21,004.0\text{ MT/s}$ stock $\rightarrow 540\text{ GB/s}$ theoretical pin bandwidth); Core V/F curve locked at $2,779\text{ MHz}$ @ $965\text{ mV}$ (sustains $2,760\text{--}2,775\text{ MHz}$ under full load). **No ECC.** | `AUTHOR-STATED` & `MEASURED` (HWiNFO64 v8.55, Afterburner, `P-H3`, `P-H4`) |
| **CPU Hardware** | AMD Ryzen 7 5800X3D (Family `19h` Zen 3): 8 physical cores, 16 threads, $4.54\text{ GHz}$ boost, $4\text{ MiB}$ L2 ($8 \times 512\text{ KiB}$), $96\text{ MiB}$ shared 3D V-Cache **victim L3 cache**, 64-entry L1 DTLB, 2,048-entry L2 DTLB. ISA: **AVX2, FMA, F16C, BMI2 only** (no AVX-512, no AVX-VNNI, no AMX). | `PUBLISHED` (AMD Family 19h SOG) & `MEASURED` (`P-H1`, census) |
| **System DRAM** | $64\text{ GB}$ ($4 \times 16\text{ GB}$ G.Skill `F4-3200C16-16GVK`, dual-rank Samsung dies), dual-channel DDR4-3200 ($1,597.1\text{ MHz}$, timings 16-18-18-38, Command Rate 1T, XMP enabled, no ECC). Theoretical peak bandwidth: **$51.2\text{ GB/s}$**. | `PUBLISHED` (JEDEC) & `MEASURED` (HWiNFO64) |
| **Motherboard & Bus** | GIGABYTE B550M GAMING X WIFI6 (BIOS F5c, SMU 56.78.0); PCIe 4.0 x16 @ $16\text{ GT/s}$ ($31.5\text{ GB/s}$ raw per direction), max payload size $256\text{ B}$, ASPM off, Resizable BAR enabled ($16,384\text{ MiB}$ BAR1). | `MEASURED` (HWiNFO64, `P-H8`) |
| **Storage Drives** | `C:` Samsung `MZVLB1T0HBLR-000H1` $954\text{ GB}$ NVMe (PCIe 3.0 x4 @ $8\text{ GT/s}$, holds Windows OS and WSL2 `ext4.vhdx`); `D:` WD Blue `SN580` $932\text{ GB}$ NVMe (PCIe 4.0 drive linked at PCIe 3.0 x2 @ $8\text{ GT/s}$, raw ceiling $\approx 1.97\text{ GB/s}$ [`DERIVED`], holds archives and WSL2 swap file). | `MEASURED` & `DERIVED` (HWiNFO64, PowerShell `Get-PhysicalDisk`) |
| **Host OS (Windows 11)** | Windows 11 Pro 26H2 (build `26300.9711`), Hyper-V active, HVCI off, Hardware-Accelerated GPU Scheduling (HAGS) off (`HwSchMode = 1`), TDR at Windows default ($2\text{ s}$), Power Plan Balanced, fixed $32\text{ GiB}$ page file. NVIDIA Profile Inspector: `CUDA - Force P2 State = Off`, `Power management = Prefer maximum performance`, `Prefer no system memory fallback = On`. | `MEASURED` & `AUTHOR-STATED` (Windows facts run, `MACHINE_STATE.md`) |
| **Guest OS (WSL2)** | WSL `3.0.4.0`, Ubuntu 26.04.1 LTS, kernel `6.18.54.1-microsoft-standard-WSL2`, `/dev/dxg` WDDM paravirtualization. `.wslconfig`: `memory=48GB`, `processors=16`, `swap=48GB` on `D:`, `kernelCommandLine=transparent_hugepage=always swiotlb=65536`, `virtiofs=true`, `autoMemoryReclaim=disabled`, `sparseVhd=false`. `RLIMIT_MEMLOCK = 64 MiB` ($65,536\text{ KiB}$). | `MEASURED` (`census.txt`, `MACHINE_STATE.md`) |
| **Compilers & CUDA** | NVIDIA Driver `617.42` (CUDA 13.4 UMD). **WSL2:** `nvcc 13.0.88`, GCC `15.2.0`, Clang `21`, CMake `4.2.3`, Ninja `1.13.2`. **Windows Native:** `nvcc 13.4.92`, MSVC `19.44.35229.0` (Visual Studio 18 Professional, toolset `14.44.35207`), CMake `4.4.3`, Ninja `1.13.2`. **Avoid CUDA 13.2** (`PUBLISHED` Unsloth warning). | `MEASURED` (Wave 0 logs) |

### 2.2 Complete Measured Hardware Floors (Wave 0 & Microbenchmarks `02`–`07`)

All measurements below were executed on the reference machine (`2026-10-20` to `2026-10-24`) with the overclock active and no competing GPU workloads. "WSL2 / Win11" denotes side-by-side platform measurements.

| Probe / Microbench | Measured Quantity | WSL2 Result | Windows-Native Result | Architectural Consequence |
|---|---|---|---|---|
| **`P-H8` (Free VRAM & Context)** | Total VRAM / Pre-context used / Context cost / **Post-context CUDA Free** / Post-context NVML Free | $11.99\text{ GiB}$ total / $914\text{--}1,070\text{ MiB}$ used / $153.7\text{ MiB}$ ctx / **$10.830\text{ GiB}$ CUDA free** / $10.73\text{--}10.95\text{ GiB}$ NVML free | $11.99\text{ GiB}$ total / $1,015\text{--}1,031\text{ MiB}$ used / $138\text{ MiB}$ ctx / **$10.846\text{ GiB}$ CUDA free** / $10.85\text{--}10.87\text{ GiB}$ NVML free | CUDA free is a fixed driver budget ($10.83\text{ GiB}$ across 21 engine logs + 4 probe runs). Sets the **$10.7\text{ GiB}$ ceiling (`D26`)** and $\min(\text{CUDA}, \text{NVML})$ startup rule (`D28`). |
| **`P-H1` (Device Limits)** | SMs / Max opt-in shared memory per block / Async copy engines / Max persisting L2 / KernelExecTimeout | $46$ SMs / $101,376\text{ B}$ ($99\text{ KiB}$) / $5$ engines / $24.75\text{ MiB}$ / $1$ (on) | $46$ SMs / $101,376\text{ B}$ ($99\text{ KiB}$) / $5$ engines / $24.75\text{ MiB}$ / $1$ (on) | Locks `kern/` tile constants ($\le 96\text{ KiB}$ shared memory per CTA). |
| **`P-H2` & `02` (VRAM & DRAM BW)** | VRAM Read / Write / Copy kernel / D2D `cudaMemcpy`; Host AVX2 DRAM Read (1T / 4T best / 7T / $2\text{ MiB}$ THP) | VRAM: **$523.3$** / $447.5$ / $473.3$ / $494.5\text{ GB/s}$. DRAM: $37.1$ (1T), **$43.7\text{--}45.81\text{ GB/s}$** (4T, $4\text{ KiB}$), $45.89$ (7T), $44.86$ ($2\text{ MiB}$ THP) | VRAM: **$534.0$** / $454.0$ / $481.0$ / $504.0\text{ GB/s}$. DRAM: $37.9$ (1T), **$44.8\text{--}46.4\text{ GB/s}$** (4T) | Memory OC delivers $> 504\text{ GB/s}$ stock VRAM read. $2\text{ MiB}$ pages change CPU read by $< 1\%$. |
| **`P-H3` (Sustained Tensor Peaks)** | $60\text{ s}$ sustained MMA: `bf16` (`m16n8k16` `fp32`-acc) / `fp16` / `int8` (`m16n8k32` `s32`-acc) / `fp8-e4m3` @ $2,760\text{--}2,775\text{ MHz}$ | **`bf16` $65.2\text{ TFLOPS}$**, `fp16` $64.9\text{ TFLOPS}$, **`int8` $259.5\text{ TOPS}$**, `fp8` $129.8\text{ TOPS}$ ($99.8\%$ of burst; $0.511 / 2.044\text{ T-ops/SM/GHz}$) | **`bf16` $65.2\text{ TFLOPS}$**, `fp16` $64.9\text{ TFLOPS}$, **`int8` $259.5\text{ TOPS}$**, `fp8` $129.8\text{ TOPS}$ ($99.8\%$ of burst) | Replaces derived OC peaks ($65.4 / 261.4$). Proves `int8` is $3.98\times$ `bf16` on `sm_89`. Stock-clock peaks remain `DERIVED` ($58.3\text{ TFLOPS}$ `bf16`, $233.2\text{ TOPS}$ `int8`). |
| **`P-H10` & `P-H4` (Canary & $30\text{ min}$ Soak)** | $10,000$ canary GEMMs ($2048^3$); $30\text{ min}$ thermal/clock stability soak ($\sim 17,700$ cycles) | $0$ mismatches, $1$ hash, error/tol $\le 0.01$. Soak: $0$ mismatches, SM clock flat $2,760.0\text{ MHz}$ median, $43\text{--}55^\circ\text{C}$, $\sim 107\text{ W}$. | $0$ mismatches, $1$ hash, error/tol $\le 0.01$. Soak: $0$ mismatches, SM clock flat $2,760.0\text{ MHz}$ median, $49\text{--}56^\circ\text{C}$, $\sim 106.7\text{ W}$ (mem flat $11,501\text{ MHz}$). | Overclock is stable for multi-hour training. NVML throttle field falsely reports `gpu_idle` ($0\text{x}1$) under full load on both stacks; watchdog must use SM clock + temp (`15.3`). |
| **`P-H5` (PCIe DMA & Zero-Copy)** | $256\text{ MiB}$ Pinned H2D / Pinned D2H / Pageable H2D / Pageable D2H / Zero-Copy Read / $4\text{ KiB}$ & $1\text{ MiB}$ latency | Pinned: **$26.3\text{ GB/s}$** H2D & D2H; Pageable: $10.3$ / $10.8\text{ GB/s}$; Zero-copy: **$26.3\text{ GB/s}$**; Latency: $73\text{ }\mu\text{s}$ ($4\text{ KiB}$), $113\text{ }\mu\text{s}$ ($1\text{ MiB}$) | Pinned: **$26.3\text{ GB/s}$** H2D & D2H; Pageable: $11.8$ / $11.8\text{ GB/s}$; Zero-copy: **$26.3\text{ GB/s}$**; Latency: $31\text{ }\mu\text{s}$ ($4\text{ KiB}$), $72\text{ }\mu\text{s}$ ($1\text{ MiB}$) | Confirms $26.1\text{--}26.3\text{ GB/s}$ PCIe 4.0 x16 floor. Windows small-copy latency is $2.3\times$ lower than WSL2. |
| **`04` & `P-H6` (Dual-Bus Contention)** | Simultaneous CPU AVX2 cold read + GPU PCIe zero-copy read / Simultaneous CPU read + H2D DMA | **CPU $26.56\text{ GB/s}$ + GPU ZC $21.06\text{ GB/s} = \mathbf{47.62\text{ GB/s}}$** ($93.0\%$ of $51.2\text{ GB/s}$ peak, $1.04\times$ CPU alone). CPU + H2D: $46.6\text{--}49.96\text{ GB/s}$ (H2D keeps $87\text{--}100\%$). | Unloaded H2D $26.1\text{ GB/s}$; with 4/8/16 CPU reader threads, H2D retains $84\text{--}100\%$ ($21.9\text{--}26.1\text{ GB/s}$). | Proves the **Unified Dual-Bus discovery**: CPU and GPU read system DRAM simultaneously without halving each other's bandwidth. |
| **`03` & Pass 1 (Doorbell Handoff)** | Per-layer `cudaStreamSynchronize` vs. Resident-kernel shared-memory doorbell ping-pong | Stream sync: **$94.40\text{ }\mu\text{s}$ median** ($123.01\text{ }\mu\text{s}$ p99). Doorbell (`03`, $4\text{ KiB}$): **$13.34\text{ }\mu\text{s}$ median, $18.71\text{ }\mu\text{s}$ p99** ($3.36\text{ }\mu\text{s}$ flag only). Block-256 test (`test_handoff`): **$5.05\text{ }\mu\text{s}$ median, $6.17\text{ }\mu\text{s}$ p99**. | Unmeasured on Windows-native (`E1` floor test). | Replaces $3.76\text{--}5.1\text{ ms/token}$ OS sync overhead with sub-$20\text{ }\mu\text{s/layer}$ lock-free flag coordination (`bus/doorbell_handoff`). |
| **`07` & `P-H7` (Pinned Memory Cap)** | Maximum `cudaHostAlloc` pinned allocation up to $32\text{ GiB}$ ($34.4\text{ GB}$) & pointer identity vs. `cudaHostRegister` | **$32\text{ GiB}$ pinned & held** ($0\text{ MiB}$ swap-in, $0.76\text{--}1.0\text{ s/GiB}$ alloc). `cudaHostAlloc` host ptr $=$ dev ptr; `cudaHostRegister` dev ptr $\neq$ host ptr (constant offset per run). | **$32\text{ GiB}$ pinned & held** ($0\text{ MiB}$ swap-in, $0.069\text{ s/GiB}$ alloc — $12\times$ faster than WSL2). | Closes Risk `R7`: the $18.65\text{ GiB}$ expert store + $5\text{--}10\text{ GiB}$ boundary store fit in `cudaHostAlloc` memory on both platforms. |
| **`P-M3` ($30\text{ min}$ Host Hold)** | Hold $24\text{ GiB}$ pinned + $4\text{ GiB}$ pageable for $30\text{ min}$ with continuous DMA & $10\text{ ms}$ sleep heartbeat | $0$ mismatches, $0\text{ MiB}$ swap-in, H2D median $25.8\text{ GB/s}$; sleep overshoot median $0.082\text{ ms}$, p99 $0.375\text{ ms}$, max $1.70\text{ ms}$. | $0$ mismatches, $0\text{ MiB}$ swap-in, H2D median $26.03\text{ GB/s}$; sleep overshoot median $6.14\text{ ms}$, p99 $6.48\text{ ms}$ (default $15.6\text{ ms}$ Win32 timer tick). | Proves long-run host memory stability under WSL2's $48\text{ GB}$ cap and Windows native. |
| **`P-H9` (CUDA Graphs & Flush Cost)** | 100-kernel eager launch vs. CUDA graph replay ($10^5$ replays); `cudaStreamQuery` flush cost | Eager $2.25\text{--}4.70\text{ }\mu\text{s/kernel}$ vs. **Graph $0.716\text{ }\mu\text{s/kernel}$** ($68\text{--}85\%$ saving). Flush: $0.098\text{ }\mu\text{s}$ idle, **$76.6\text{ }\mu\text{s}$ busy stream**. | Eager $2.18\text{--}2.32\text{ }\mu\text{s/kernel}$ vs. **Graph $0.707\text{ }\mu\text{s/kernel}$** ($68\%$ saving). Flush: $0.091\text{ }\mu\text{s}$ idle, **$8.05\text{ }\mu\text{s}$ busy stream** ($9.5\times$ faster). | Locks CUDA graph adoption for Engine spine (`E8`) and Trainer layers (`6.9`). |
| **`P-H12` (3-Stream Copy/Compute Overlap)** | Concurrent H2D (`S_w`) + D2H (`S_x`) + Compute (`S_compute`) across 4 submission patterns | Unflushed default: **$0.47$** all-three ($0.00$ H2D+D2H). **Flushed (`cudaStreamQuery` per batch): $0.92$ all-three, $0.84$ H2D+D2H pair**. | Unflushed default: **$0.52$** all-three ($0.00$ H2D+D2H). **Flushed (`cudaStreamQuery` per batch): $0.92$ all-three, $0.84$ H2D+D2H pair**. | Proves WDDM batches cross-stream work unless flushed. Locks `D27`: flush once per multi-ms batch; roofline solver uses $0.84$ pair / $0.92$ all-three efficiency. |
| **`P-H13` (Display Timeout / TDR)** | $2\text{ s}$, $5\text{ s}$, $30\text{ s}$ continuous spin kernels, $10\text{ s}$ queued set, $10\text{ s}$ graph with desktop active | All 5 stages PASS ($30.001\text{ s}$ wall), $0$ errors, $0$ visual lag/flicker. | All 5 stages PASS ($29.9998\text{ s}$ wall), $0$ errors, $0$ `nvlddmkm` or WHEA events, $0$ visual lag. | Hardware compute preemption prevents TDR resets up to $30\text{ s}$ at $\sim 60\text{ W}$. Resident doorbell kernels remain scoped to 1 token ($\sim 18\text{--}26\text{ ms}$) for absolute safety. |
| **File Read Speeds (`P-M4` / Runbook)** | Sequential read of $20.22\text{ GiB}$ GGUF from Windows 9p mount (`/mnt/c`, `/mnt/d`) vs. WSL2 `ext4` disk | `/mnt/c` & `/mnt/d` (9p `msize=65536`): **$0.17\text{--}0.52\text{ GB/s}$** ($87\text{--}119\text{ s}$ load; `dd` $0.94\text{--}1.6\text{ GB/s}$). **WSL2 `ext4`: $2.99\text{--}3.06\text{ GB/s}$ cold ($7.1\text{ s}$), $0.25\text{ s}$ warm ($1.5\text{ s}$ full tracer init)**. | `P-M4` standalone binary skipped (awaiting `PELIAS_GGUF` path). | Models accessed inside WSL2 must reside on the WSL2 `ext4` virtual disk (`~/models/`) or be read once into the pinned store. |

---

## 3. Model Architecture, Exact Byte Maps, and Mathematical Specifications

### 3.1 Model Structural Constants (`Qwen3.6-35B-A3B` / `qwen35moe`, `READ-SRC`)

| Key / Parameter | Exact Value | Architectural Note |
|---|---|---|
| **Architecture & Block Count** | `qwen35moe`; `block_count = 41` ($40$ trunk layers `blk.0`–`blk.39` + $1$ MTP block `blk.40`); `nextn_predict_layers = 1` | Stock `llama-server` ignores `blk.40` with 20 warnings at startup; Pelias treats `blk.40` as a first-class block for speculative decoding (`E12`) and MTP co-training (`D12`). |
| **Dimensions & Vocabulary** | Hidden dim $d_{\text{model}} = 2,048$; Vocabulary $V = 248,320$ (BPE vocab $248,044$ + padding); Max context $262,144$; RMSNorm $\epsilon = 9.99999974752427 \times 10^{-7}$ ($10^{-6}$ in `fp32`) | Embedding (`token_embd.weight`) and output head (`output.weight`) are **untied**. |
| **Hybrid Layer Pattern** | `full_attention_interval = 4` $\rightarrow$ **10 Full GQA Attention layers** at $l \in \{3, 7, 11, 15, 19, 23, 27, 31, 35, 39\}$ (where $(l+1) \bmod 4 = 0$); **30 Gated DeltaNet layers** at all other trunk indices; `blk.40` is Full Attention + MoE | Verified from `qwen35moe.cpp` graph builder and per-layer tensor presence (`F2`). |
| **Full GQA Attention** | $H_q = 16$ query heads, $H_{kv} = 2$ KV heads (8:1 GQA), head dim $d_h = 256$ (key and value length $256$); `attn_q` projects $2,048 \rightarrow 8,192$ (**interleaved query + output gate** per head); per-head `attn_q_norm` and `attn_k_norm` ($256$-wide); **IMRoPE** on first $64$ dims ($32$ pairs), sections $[11, 11, 10, 0]$, base $\theta_0 = 10,000,000$ | Attention output is gated by $\sigma(\text{gate}_h)$ before `attn_output` ($4,096 \rightarrow 2,048$). |
| **Gated DeltaNet ("SSM")** | Causal depthwise `conv1d` kernel $K_{\text{conv}} = 4$ across $8,192$ channels ($q\text{ }2,048 \mid k\text{ }2,048 \mid v\text{ }4,096$); `group_count` $H_k = 16$ ($q, k$ heads of dim $128$); `time_step_rank` $H_v = 32$ ($v$ heads of dim $128$); state size $128 \times 128$ per $v$-head; inner size $4,096$; gated RMSNorm `ssm_norm` ($128$-wide) | $k$-heads are tiled $16 \rightarrow 32$ so $v$-head $j$ uses $k$-head $(j \bmod 16)$. |
| **Mixture of Experts (MoE)** | $E = 256$ routed experts per layer, top-$k = 8$ active per token, expert FFN intermediate dim $d_{\text{exp}} = 512$; $1$ always-on shared expert per layer ($d_{\text{shexp}} = 512$) gated by a single scalar gate $\sigma(g_{\text{sh}} \cdot u)$ (`ffn_gate_inp_shexp`, $2,048$-wide `F32`) | Total active parameters per token: **$2.94\text{ B}$** (experts $1.007\text{ B}$ + shared experts $125.8\text{ M}$ + DeltaNet in $755.0\text{ M}$ / out $251.7\text{ M}$ + attention $272.6\text{ M}$ + routers $21.0\text{ M}$ + `lm_head` $508.6\text{ M}$). |

### 3.2 Complete Tensor and Byte Maps: Served `Q4_K_M` GGUF and Source `bf16` Safetensors

#### 3.2.1 Served `Q4_K_M` GGUF Inventory (`READ-SRC` & `DERIVED`)
* **File Totals:** $753$ tensors; $21,713,463,296\text{ bytes}$ ($20.22\text{ GiB}$); `general.alignment = 32`; tensor data starts at byte $10,991,104$; every tensor offset is $32\text{-byte}$ aligned; all $123$ expert tensors are $16\text{-byte}$ aligned and exact multiples of $2\text{ MiB}$ ($144\text{ MiB}$ or $210\text{ MiB}$).

| GGUF Tensor Name | GGUF Shape (Inner Dim First) | Quantization Types & Layer Distribution | Count | Total MiB | Role & Training Status |
|---|---|---|---:|---:|---|
| `token_embd.weight` | $2048 \times 248320$ | `Q4_K` | 1 | $272.8$ | Input embedding; frozen, gather + dequant only, byte-spliced. |
| `output.weight` | $2048 \times 248320$ | `Q6_K` | 1 | $397.9$ | Untied `lm_head`; frozen (`dgrad` only; optional LoRA in `T-A8`). |
| `output_norm.weight` | $2048$ | `F32` ($+1$ folded) | 1 | $0.008$ | Final RMSNorm; frozen (optional `fp32` trainable). |
| `blk.{N}.attn_norm.weight` | $2048$ | `F32` ($+1$ folded) | 41 | $0.3$ | Pre-attention / pre-DeltaNet RMSNorm; frozen (optional `fp32` trainable). |
| `blk.{N}.post_attention_norm.weight` | $2048$ | `F32` ($+1$ folded) | 41 | $0.3$ | Pre-MoE RMSNorm; frozen (optional `fp32` trainable). |
| `blk.{N}.ffn_gate_exps.weight` | $2048 \times 512 \times 256$ | `Q4_K` ($144\text{ MiB/layer}$) | 41 | $5,904.0$ | Routed expert gate projections; **frozen, streamed, byte-spliced**. |
| `blk.{N}.ffn_up_exps.weight` | $2048 \times 512 \times 256$ | `Q4_K` ($144\text{ MiB/layer}$) | 41 | $5,904.0$ | Routed expert up projections; **frozen, streamed, byte-spliced**. |
| `blk.{N}.ffn_down_exps.weight` | $512 \times 2048 \times 256$ | **`Q6_K` $\times 21$** ($210\text{ MiB/layer}$), **`Q4_K` $\times 20$** ($144\text{ MiB/layer}$) | 41 | $4,410.0 + 2,880.0$ | Routed expert down projections; **frozen, streamed, byte-spliced**. |
| `blk.{N}.ffn_gate_inp.weight` | $2048 \times 256$ | `F32` ($2.0\text{ MiB/layer}$) | 41 | $82.0$ | MoE Router ($21,495,808$ params); frozen default (**trainable `fp32` in Mode L**). |
| `blk.{N}.ffn_gate_inp_shexp.weight` | $2048$ | `F32` | 41 | $0.3$ | Shared-expert scalar gate vector; frozen (optional `fp32` trainable in `T-A5`). |
| `blk.{N}.ffn_gate_shexp.weight` | $2048 \times 512$ | `Q4_K` | 41 | $23.1$ | Shared expert gate; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ffn_up_shexp.weight` | $2048 \times 512$ | `Q4_K` | 41 | $23.1$ | Shared expert up; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ffn_down_shexp.weight` | $512 \times 2048$ | `Q6_K` $\times 21$, `Q4_K` $\times 20$ | 41 | $17.2 + 11.2$ | Shared expert down; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_qkv.weight` | $2048 \times 8192$ | `Q6_K` $\times 14$, `Q4_K` $\times 16$ | 30 | $183.8 + 144.0$ | DeltaNet joint $q, k, v$; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_gate.weight` | $2048 \times 4096$ | `Q4_K` | 30 | $135.0$ | DeltaNet output gate $z$; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_alpha.weight` | $2048 \times 32$ | `Q4_K` | 30 | $1.1$ | DeltaNet decay projection; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_beta.weight` | $2048 \times 32$ | `Q4_K` | 30 | $1.1$ | DeltaNet write-strength projection; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_out.weight` | $4096 \times 2048$ | `Q4_K` | 30 | $135.0$ | DeltaNet output projection; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.ssm_conv1d.weight` | $4 \times 8192$ | `F32` | 30 | $3.8$ | DeltaNet 4-tap depthwise conv; frozen. |
| `blk.{N}.ssm_a`, `ssm_dt.bias`, `ssm_norm.weight` | $32$ / $32$ / $128$ | `F32` (`ssm_norm` is plain, no $+1$) | 30 each | $< 0.1$ | DeltaNet decay base ($-\exp(A_{\text{log}})$), bias, gated RMSNorm; frozen. |
| `blk.{N}.attn_q.weight` | $2048 \times 8192$ | `Q4_K` | 11 | $99.0$ | Full Attention joint query + gate; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_k.weight` | $2048 \times 512$ | `Q4_K` | 11 | $6.2$ | Full Attention key; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_v.weight` | $2048 \times 512$ | `Q6_K` $\times 7$, `Q4_K` $\times 4$ | 11 | $5.7 + 2.2$ | Full Attention value; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_output.weight` | $4096 \times 2048$ | `Q4_K` | 11 | $49.5$ | Full Attention output; **Trainable Core (LoRA + Mode Q)**. |
| `blk.{N}.attn_q_norm.weight`, `attn_k_norm.weight` | $256$ each | `F32` ($+1$ folded) | 11 each | $< 0.1$ | Full Attention per-head RMSNorms; frozen. |
| `blk.40.nextn.eh_proj.weight` | $4096 \times 2048$ | `Q4_K` | 1 | $4.5$ | MTP embedding+hidden fusion projection; optional LoRA (`D12`). |
| `blk.40.nextn.{enorm,hnorm,shared_head_norm}.weight` | $2048$ each | `F32` ($+1$ folded) | 1 each | $< 0.1$ | MTP norms; frozen. |

**Exact Layer-Index Rules for Mixed Quantization (`READ-SRC`):**
* **21 Blocks with `Q6_K` `ffn_down_exps` and `ffn_down_shexp` ($498\text{ MiB}$ expert layer = $288\text{ MiB}$ gate+up `Q4_K` + $210\text{ MiB}$ down `Q6_K`; $2,039,808\text{ B/expert}$):** Trunk layers $0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39$, plus MTP block $40$.
* **20 Trunk Layers with `Q4_K` `ffn_down_exps` and `ffn_down_shexp` ($432\text{ MiB}$ expert layer = $288\text{ MiB}$ gate+up `Q4_K` + $144\text{ MiB}$ down `Q4_K`; $1,769,472\text{ B/expert}$):** Trunk layers $5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33$.
* **14 DeltaNet Layers with `Q6_K` `attn_qkv`:** Layers $0, 1, 2, 4, 10, 13, 16, 22, 25, 28, 34, 36, 37, 38$ (the other 16 DeltaNet layers use `Q4_K`).
* **7 Full-Attention Blocks with `Q6_K` `attn_v`:** Trunk layers $3, 7, 19, 31, 35, 39$, plus MTP block $40$ (the other 4 full-attention layers $11, 15, 23, 27$ use `Q4_K`).
* **Total Routed Expert Bytes:** 40 trunk layers = $20 \times 432\text{ MiB} + 20 \times 498\text{ MiB} = 18,600\text{ MiB} = 19,503,513,600\text{ B} = \mathbf{18.164\text{ GiB}}$ ($32,212,254,720$ params; fleet average $1,904,640\text{ B/expert}$; $8 \times 40 = 320$ active expert calls/token $= 0.6095\text{ GB/token}$). Including `blk.40` ($498\text{ MiB}$) = $19,098\text{ MiB} = \mathbf{18.65\text{ GiB}}$ pinned in host RAM.

#### 3.2.2 Source `bf16` Safetensors Checkpoint Layout (`MEASURED` from Shard Headers, `2026-10-24`)
Read from the 26 shards of the archived `unsloth/Qwen3.6-35B-A3B` checkpoint (`2ab40a9a...`): **26 shards ($1.10\text{--}4.00\text{ GB}$ each), 1,045 tensors, all `BF16`, $71,903,645,408\text{ tensor bytes}$ ($66.965\text{ GiB} = 71.90\text{ GB}$)**.

| Checkpoint Class | Tensors | Parameters | Bytes | GiB | Stage A (`ingest`) Handling Rule (`D33`) |
|---|---:|---:|---:|---:|---|
| **Routed Experts (40 trunk layers)** (`mlp.experts.gate_up_proj` $[256, 1024, 2048]$ + `down_proj` $[256, 2048, 512]$) | 80 | $32,212,254,720$ | $64,424,509,440$ | $60.000$ | Fused `gate_up_proj` ($1,024\text{ MiB/layer}$) is sliced along `dim = -2` at row $512$ into separate `ffn_gate_exps` (rows $0\dots 511$) and `ffn_up_exps` (rows $512\dots 1023$), then block-quantized to `Q4_K`/`Q6_K`. |
| **Trunk Spine & Routers** | 611 | $1,431,237,248$ | $2,862,474,496$ | $2.666$ | Apply name map (`model.language_model.` prefix stripped), V-head tiled permutation (`P-S3`), norm $+1$ folding (`P-S2`), `ssm_a = -\exp(A_{\text{log}})`, and stock `Q4_K_M` type map. |
| **Input Embedding** | 1 | $508,559,360$ | $1,017,118,720$ | $0.947$ | Quantize to `Q4_K` (`token_embd.weight`). |
| **Output Head (`lm_head`, untied)** | 1 | $508,559,360$ | $1,017,118,720$ | $0.947$ | Quantize to `Q6_K` (`output.weight`). |
| **MTP Block (`mtp.*`)** | 19 | $844,640,768$ | $1,689,281,536$ | $1.573$ | Kept by default as `blk.40` (2 fused expert tensors split to 3 GGUF expert tensors + 17 spine/norm tensors = 20 GGUF tensors). |
| **Vision Tower (`model.visual.*`)** | 333 | $446,571,248$ | $893,142,496$ | $0.832$ | **Dropped** (text-only GGUF: $1,045 - 333 = 712$ source tensors $\rightarrow 753$ GGUF tensors after splitting 41 `gate_up_proj` tensors into $41 \times 2 = 82$). |
| **Total** | **1,045** | **$35,951,822,704$** | **$71,903,645,408$** | **$66.965$** | Streamed one tensor at a time via `mmap` (max in-flight tensor $1,024\text{ MiB}$). |

#### 3.2.3 Trainable Core Byte Budget Table (40 Trunk Layers, $1,409.0\text{ M}$ Parameters, 310 LoRA Modules)

| Core Tensor Class | Module Count | Parameters | Served Mix (`MiB`) | All `Q6_K` (`MiB` / $\Delta$) | All `Q8_0` (`MiB` / $\Delta$) | `bf16` (`MiB`) |
|---|---:|---:|---:|---:|---:|---:|
| **Shared Experts** (`gate`, `up`, `down` $\times 40$) | 120 | $125.8\text{ M}$ | $72.7$ | $98.4$ ($+25.7$) | $127.5$ ($+54.8$) | $240.0$ |
| **DeltaNet In-Projections** (`qkv` + `gate` $\times 30$) | 60 | $755.0\text{ M}$ | $462.8$ | $590.6$ ($+127.8$) | $765.0$ ($+302.2$) | $1,440.0$ |
| **DeltaNet Decay/Write** (`alpha`, `beta` $\times 30$) | 60 | $3.9\text{ M}$ | $2.1$ | $3.1$ ($+1.0$) | $4.0$ ($+1.9$) | $7.5$ |
| **DeltaNet Out-Projection** (`ssm_out` $\times 30$) | 30 | $251.7\text{ M}$ | $135.0$ | $196.9$ ($+61.9$) | $255.0$ ($+120.0$) | $480.0$ |
| **Full Attention** (`q`, `k`, `v`, `output` $\times 10$) | 40 | $272.6\text{ M}$ | $147.8$ | $213.3$ ($+65.5$) | $276.2$ ($+128.4$) | $520.0$ |
| **Total Trunk Core** | **310** | **$1,409.0\text{ M}$** | **$820.3$** ($820.4$) | **$1,102.3$** (**$+282.0$**) | **$1,427.7$** (**$+607.4$**) | **$2,687.5$** |

*(Note: MTP `blk.40` adds $47.2\text{ M}$ core parameters when `D12` is active. At rank $r = 16$, the 310 trunk LoRA modules contain exactly **$21,166,080$ trainable parameters**; at $16\text{ B/param}$ for `fp32` master + `fp32` grad + `fp32` Adam $m, v$, LoRA state consumes **$323.0\text{ MiB}$** [$0.32\text{ GiB}$], plus $40.4\text{ MiB}$ for `bf16` compute copies.)*

### 3.3 Mathematical Specification of the Forward and Backward Graph

Let $x \in \mathbb{R}^{T \times 2048}$ be the residual hidden state stream (`fp32` between graph nodes). Per-token RMSNorm with weight vector $w$ (where $w = 1 + w_{\text{HF}}$ is pre-folded for all norms except `ssm_norm`) is:
$$\text{RMS}(x; w) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot w, \quad \epsilon = 10^{-6}$$

#### 3.3.1 Trunk Layer Structure ($l = 0 \dots 39$)
$$a = \text{RMS}(x; w_{\text{attn\_norm}, l}), \quad m = \begin{cases} \text{DeltaNet}_l(a) & \text{if } (l+1) \bmod 4 \neq 0 \\ \text{Attention}_l(a) & \text{if } (l+1) \bmod 4 = 0 \end{cases}$$
$$h = x + m, \quad u = \text{RMS}(h; w_{\text{post\_attn\_norm}, l}), \quad y = h + \text{MoE}_l(u)$$

#### 3.3.2 Mixture-of-Experts Sub-Layer (`MoE`)
1. **Router (`F32`):** Logits $z = u W_r^\top \in \mathbb{R}^{T \times 256}$ computed in `fp32` FMA (no TF32). Select top-8 expert indices $S = \text{argtopk}_8(z)$ per token (ties broken to lower expert index, `P-S8`). Because the 256-way denominator cancels during renormalization, the normalized weights over $e \in S$ are computed directly via an 8-way `fp32` softmax:
   $$w_e = \frac{\exp(z_e)}{\sum_{e' \in S} \exp(z_{e'})}, \quad e \in S$$
2. **Expert & Shared-Expert FFN:** For any expert $e$ (or shared expert $\text{sh}$), with SwiGLU activation $\text{SiLU}(g) = g \cdot \sigma(g)$:
   $$E(u) = W_{\text{down}} \left( \text{SiLU}(u W_{\text{gate}}^\top) \odot (u W_{\text{up}}^\top) \right)$$
   $$\text{MoE}(u) = \sum_{e \in S} w_e E_e(u) + \sigma(u \cdot g_{\text{sh}}) E_{\text{sh}}(u)$$
3. **Backward (`R3`, `E2`, `G5/G6`):** For selected experts $e \in S$, $d w_e = \langle dy, E_e(u) \rangle$ in `fp32`; router logit gradient $d z_e = w_e \left(d w_e - \sum_{e' \in S} w_{e'} d w_{e'}\right)$ for $e \in S$ and $0$ for $e \notin S$; $du \mathrel{+}= \sum_{e \in S} d z_e W_r[e, :]$. For SwiGLU with $h = \text{SiLU}(g) \odot p$: $dp = dh \odot \text{SiLU}(g)$ and $dg = dh \odot p \odot \sigma(g)(1 + g(1 - \sigma(g)))$.

#### 3.3.3 Full GQA Attention Sub-Layer ($l \in \{3, 7, \dots, 39\}$ and `blk.40`)
1. **Interleaved Query + Gate Projection:** $[q \mid \text{gate}] = a W_q^\top \in \mathbb{R}^{T \times 8192}$. For query head $h \in \{0 \dots 15\}$, rows $[512h, 512h+256)$ form $q_h \in \mathbb{R}^{T \times 256}$ and rows $[512h+256, 512h+512)$ form $\text{gate}_h \in \mathbb{R}^{T \times 256}$. Keys and values are $k = a W_k^\top \in \mathbb{R}^{T \times 512}$ ($2 \times 256$) and $v = a W_v^\top \in \mathbb{R}^{T \times 512}$ ($2 \times 256$).
2. **Per-Head Norm & IMRoPE:** $q_h \leftarrow \text{RMS}(q_h; w_{q\text{\_norm}})$, $k_g \leftarrow \text{RMS}(k_g; w_{k\text{\_norm}})$. Apply IMRoPE (`A1`, sections $[11, 11, 10, 0]$, base $\theta_0 = 10^7$, $\theta_i = \theta_0^{-2i/64}$) to the first $64$ dimensions of each head, with position $p$ resetting to $0$ at each packed document boundary (`cu_seqlens`).
3. **Causal GQA & Output Gate:** With scale $s = 1/\sqrt{256} = 1/16$ and KV head $g = \lfloor h/8 \rfloor$:
   $$o_h = \text{softmax}_{\text{causal}}\left( \frac{q_h k_g^\top}{\sqrt{256}} \right) v_g, \quad o'_h = o_h \odot \sigma(\text{gate}_h), \quad m = \text{concat}(o'_0 \dots o'_{15}) W_o^\top$$

#### 3.3.4 Gated DeltaNet Sub-Layer (30 Trunk Layers)
1. **Projections & Gates (`D1`, `D2`):**
   $$\text{qkv}_{\text{raw}} = a W_{\text{qkv}}^\top \in \mathbb{R}^{T \times 8192}, \quad z = a W_z^\top \in \mathbb{R}^{T \times 4096}$$
   $$\beta = \sigma(a W_\beta^\top) \in (0, 1)^{T \times 32}, \quad g = \text{ssm\_a} \odot \text{softplus}(a W_\alpha^\top + b_{\text{dt}}) \in (-\infty, 0]^{T \times 32}$$
   $$\text{qkv}' = \text{SiLU}\left(\text{CausalConv1D}_{k=4}(\text{qkv}_{\text{raw}}; w_{\text{conv}})\right)$$
   Split $\text{qkv}'$ into $q \in \mathbb{R}^{T \times 16 \times 128}$, $k \in \mathbb{R}^{T \times 16 \times 128}$, $v \in \mathbb{R}^{T \times 32 \times 128}$. Normalize per head:
   $$q \leftarrow \frac{1}{\sqrt{128}} \frac{q}{\sqrt{\|q\|_2^2 + \epsilon}}, \quad k \leftarrow \frac{k}{\sqrt{\|k\|_2^2 + \epsilon}}$$
   Tile $q, k$ heads $16 \rightarrow 32$ in GGUF tiled order ($v$-head $j \in \{0 \dots 31\}$ uses $q, k$ head $j \bmod 16$).
2. **Sequential Recurrence Form (`D7` — Engine Decode & Trainer Rung-2 Oracle):** For each $v$-head and token $t$, with state $S_t \in \mathbb{R}^{128 \times 128}$ ($S_0 = 0$ at document start) and decay $\alpha_t = \exp(g_t)$:
   $$r_t = v_t - \alpha_t (k_t S_{t-1}), \quad S_t = \alpha_t S_{t-1} + \beta_t (k_t^\top r_t), \quad o_t = q_t S_t$$
3. **Chunked WY/UT Form (`D3`–`D5` — Engine Prefill & Trainer Forward, Chunk Size $C = 64$):** Within each 64-token chunk (aligned to document starts so no chunk crosses a document boundary), let $g_{\text{cs}, i} = \sum_{j \le i} g_j$ and $D_{ij} = \exp(g_{\text{cs}, i} - g_{\text{cs}, j})$ for $j \le i$:
   $$N_{ij} = \beta_i (k_i \cdot k_j) D_{ij} \text{ for } j < i \text{ (strictly lower triangular)}, \quad T = (I + N)^{-1}$$
   $$u = T (\beta \odot v), \quad k_{\text{cd}} = T (\beta \odot \exp(g_{\text{cs}}) \odot k)$$
   $$\text{kq}_{ij} = (q_i \cdot k_j) D_{ij} \text{ for } j \le i, \quad \text{kg}_i = k_i \odot \exp(g_{\text{last}} - g_{\text{cs}, i})$$
   Across chunks with incoming state $S$:
   $$v_{\text{new}} = u - k_{\text{cd}} S, \quad o = (q \odot \exp(g_{\text{cs}})) S + \text{kq} \cdot v_{\text{new}}, \quad S \leftarrow \exp(g_{\text{last}}) S + \text{kg}^\top v_{\text{new}}$$
4. **Gated RMSNorm & Output (`E3`):** Per $v$-head ($128$-wide, plain RMSNorm without $+1$):
   $$o' = \text{RMS}_{\text{plain}}(o; w_{\text{ssm\_norm}}) \odot \text{SiLU}(z), \quad m = o' W_{\text{out}}^\top$$
5. **Sequential Backward Oracle (`D7` `fp64`) & Chunked Reverse-State Backward (`D6`):** With $G = dS_t + q_t^\top d o_t$ and $\alpha_t = \exp(g_t)$:
   $$dq_t = d o_t S_t^\top, \quad d\beta_t = k_t G r_t^\top, \quad dr = \beta_t k_t G, \quad dv_t = dr$$
   $$dk_t = \beta_t (G r_t^\top)^\top - \alpha_t (dr S_{t-1}^\top), \quad d\alpha_t = \langle G, S_{t-1} \rangle - \langle dr, k_t S_{t-1} \rangle, \quad dg_t = \alpha_t d\alpha_t$$
   $$dS_{t-1} = \alpha_t G - \alpha_t k_t^\top dr$$
   In the chunked backward (`D6`), chunk-boundary states $S_c$ ($256\text{ MiB/window}$ in `fp32`) and $T$ matrices ($64\text{ MiB/window}$ in `fp32`) saved during the layer's forward recompute are swept in reverse chunk order to compute $dv_{\text{new}}$, $dk_{\text{cd}}$, $d\text{kq}$, $d\text{kg}$, $dT$, and $d(I+N) = -T^\top dT T^\top$.

#### 3.3.5 Loss Function and Step-Wide Token Normalization (`L1`, `D8`)
To eliminate the v0.2 trainer's per-micro-batch normalization bug (which over-weighted a 40-token reply by $\sim 87\times$ relative to a 3,500-token reply), loss is normalized across the **entire optimizer step** over all $K$ windows:
$$L_{\text{step}} = \frac{\sum_{t \in \text{step}} w_t \cdot \text{CE}_t}{\sum_{t \in \text{step}} w_t}$$
where $w_t = 1$ on assistant generation tokens (after the generation prompt, including the closing `<|im_end|>`) and $w_t = 0$ on system, developer, user, tool-result, generation-prompt, opening `<think>`, and padding tokens. Logits are materialized only in **256-position chunks** ($\le 256 \times 248,320 \times 4\text{ B} = 242.5\text{ MiB}$ `fp32` buffer) for positions where $w_t > 0$.

---

## 4. Complete System Scaffolding, Module Hierarchy, and Dependency Graph

### 4.1 The Unified Four-Layer Architecture

Pelias is architected into four strict layers so that `pelias-engine` (inference + server) and `pelias-trainer` (ingest + LoRA/QAT fine-tuning + export) share one hardware abstraction substrate and one model/kernel core without forward-pass or quantization drift:
* **Layer 0: `libpelias_substrate`** — Domain-agnostic unified-bus hardware abstraction (`os/`, `hal/`, `plan/`, `bus/`, `bank/`, `quant/`).
* **Layer 1: `libpelias_core`** — Shared model I/O, byte-splice exporter, native tokenizer & template renderer, spine forward kernels, and verification comparators (`io/`, `tok/`, `kern/` [spine], `model/` [spec], `verify/`).
* **Layer 2A: `pelias-engine`** — Token-major dual-bus inference engine and raw-socket HTTP/SSE server (`engine/`, `cpu/`, `gpu/`, `server/`, `cli/`).
* **Layer 2B: `pelias-trainer`** — Layer-major $K$-window streaming LoRA/Mode-Q trainer and universal-use workflow (`config/`, `ingest/`, `data/`, `model/` [stores/LoRA/tape], `stream/`, `kern/` [trainer GEMM/bwd/loss/optim], `sched/`, `train/`, `export/`, `project/`, `startup/`, `intake/`, `help/`, `warn/`, `flow/`, `serve/`, `ui/`, `cli/`).

### 4.2 Complete Directory and File Scaffolding

Every file in the ground-up architecture is listed below with its creation stage (`T0`–`T14` for trainer stages; `E0`–`E13` for engine phases) and its exact responsibility.

```text
pelias/
├─ CMakeLists.txt  CMakePresets.json  .clang-format  .gitignore  .gitattributes
├─ VERSION  LICENSE (Apache-2.0)  CITATIONS.md  THIRD_PARTY_NOTICES.md
├─ README.md  README_PLAIN_ENGLISH.md  pelias-master.md
├─ cmake/
│  ├─ PeliasOptions.cmake          Build options: CUDA arch 89, host-only test build, WSL/Win32 flags (T0/E2)
│  ├─ PeliasModule.cmake           Static library target registration and link-dependency enforcement (T0/E2)
│  └─ ProbeFlags.cmake             Feature checks: AVX2, FMA, CUDA toolkit version, OS backend (E2)
├─ include/pelias/
│  ├─ status.h  version.h.in       Status codes, RAII result wrappers, version macros (T0/E2)
│  └─ log.h  trace.h  timer.h      Structured logging, debug partition traces, CUDA event timers (E2)
├─ src/
│  ├─ os/                          [Layer 0: Shared Operating-System Isolation Layer — Only place with OS headers]
│  │  ├─ file_map.{h,cc}           POSIX mmap/madvise vs. Win32 CreateFileMappingW/MapViewOfFile (T1/E3)
│  │  ├─ net.{h,cc}                POSIX sys/socket.h + poll vs. Winsock2 + WSAPoll + SO_EXCLUSIVEADDRUSE (E10)
│  │  ├─ threads.{h,cc}            Thread creation, physical-core topology, affinity pinning, monotonic clock, signals (E4)
│  │  └─ large_pages.{h,cc}        Optional 2 MiB THP / Win32 MEM_LARGE_PAGES allocation wrapper (E11)
│  ├─ hal/                         [Layer 0: Shared Hardware Abstraction & Memory Governor]
│  │  ├─ device.{h,cc}             Device selection, attribute queries, dynamic NVML loader (T1/E2)
│  │  ├─ stream.{h,cc}             CUDA stream wrapper with priority and cudaStreamQuery flush helper (T1/E2)
│  │  ├─ event.{h,cc}              CUDA event wrapper for timing and cross-stream ordering (T1/E2)
│  │  ├─ arena.{h,cu}              Lifetime-scoped bump arena allocator; zero-malloc loop enforcement (T1/E2)
│  │  ├─ vram_governor.{h,cc}      Hard 10.7 GiB ceiling enforcement, pre-allocation plan check, NVML polling (T1/E2)
│  │  ├─ pinned_pool.{h,cc}        Unified pinned host memory pool (cudaHostAlloc) + pageable fallback chain (T1/E2)
│  │  ├─ host_alloc_wsl.cc         WSL2 pinned allocation backend & /proc/meminfo AnonHugePages logger (E2/E11)
│  │  ├─ host_alloc_win32.cc       Windows-native pinned allocation backend & performance counter reader (E2/E11)
│  │  └─ sync_audit.{h,cc}         Debug stream-ownership verifier after every kernel launch (T1)
│  ├─ plan/                        [Layer 0: Shared Roofline & Memory Planner]
│  │  ├─ probe_constants.{h,cc}    Reader/writer for probe_constants.json and machine_state_sha (T1/E11)
│  │  ├─ roofline_solver.{h,cc}    Solves window size T, micro-batches K, slot count S, N_hide, and cold split (T1/E11)
│  │  └─ memory_plan.{h,cc}        Computes exact arena capacities for Engine and Trainer before allocation (T1/E9)
│  ├─ bus/                         [Layer 0: Unified-Bus Coordination & Streaming]
│  │  ├─ doorbell_handoff.{h,cuh,cu} Cache-line-isolated GPU<->CPU mailbox, acquire/release flags, sync fallback (E6)
│  │  ├─ cpu_threadpool.{h,cc}     Physical-core-pinned AVX2 worker pool polling the doorbell (E4)
│  │  └─ stage_streamer.{h,cu}     Trainer 3-stream DMA engine (S_compute, S_w H2D slots, S_x boundaries) (T10)
│  ├─ bank/                        [Layer 0: Shared Hot-Bank Optimizer]
│  │  ├─ hot_set_solver.{h,cc}     Water-filling (byte-hit h) and Zero-Miss (Z_l) residency solver (E5/T12)
│  │  └─ hot_set_io.{h,cc}         Byte-identical C++ reader/writer for PELIAS_HOT_SET profiles (P-E14) (E5/T12)
│  ├─ quant/                       [Layer 0: Shared Proprietary Block Quantizers & Dequantizers]
│  │  ├─ block_types.h             Exact byte layouts: Q4_K (144B), Q6_K (210B), Q8_0 (34B), q8_1 (36B), Q8_K (292B) (T1b/E3)
│  │  ├─ quant_cpu.{h,cc}          Bit-exact CPU reference quantizers/dequantizers (P-E6, P-Q1) (T1b/E4)
│  │  └─ quant_gpu.{h,cuh,cu}      Bit-exact GPU dequantizers (G8), q8_1 quantizer (G9), Mode Q requantizer (GQ) (T3/E5)
│  ├─ io/                          [Layer 1: Shared Model & Data I/O]
│  │  ├─ sha256.{h,cc}             Native SHA-256 implementation for model keys and tensor splice proofs (T1/E3)
│  │  ├─ json.{h,cc}               Native zero-dependency JSON parser and serializer (T1/E3)
│  │  ├─ gguf_reader.{h,cc}        Memory-mapped GGUF v3 parser, bounds checker, typed TensorView, layout verifier (T1/E3)
│  │  ├─ gguf_writer.{h,cc}        GGUF v3 writer with --layout expert-last and pelias.pad 2 MiB alignment (T1b/T12)
│  │  ├─ gguf_splice.{h,cc}        Byte-splice exporter with per-tensor SHA-256 verification and layout_manifest.json (T12)
│  │  └─ safetensors.{h,cc}        Memory-mapped safetensors shard header & payload reader for Stage A ingest (T1b)
│  ├─ tok/                         [Layer 1: Shared Native Tokenizer & Chat-Template Engine]
│  │  ├─ utf8.{h,cc}               UTF-8 decoder, validator, encoder, and streaming incomplete-sequence buffer (T2/E10)
│  │  ├─ unicode_data.{h,cc}       Static UCD general-category, canonical combining class, and NFC tables (T2/E10)
│  │  ├─ normalizer.{h,cc}         Native NFC Unicode normalizer (T2/E10)
│  │  ├─ pretokenizer.{h,cc}       Hand-written state-machine matcher for Qwen split regex + byte-level mapping (T2/E10)
│  │  ├─ bpe.{h,cc}                248,044-vocab / 247,587-merge rank table, BPE merge engine, and piece cache (T2/E10)
│  │  ├─ added_tokens.{h,cc}       33 added / 21 special token splitter and policy handler (O10) (T2/E10)
│  │  ├─ tokenizer.{h,cc}          Public encode, decode, and streaming detokenizer interface (P-D1) (T2/E10)
│  │  ├─ template_ast.{h,cc}       Lexer and AST parser for the 158-line Jinja chat-template subset (T2/E10)
│  │  ├─ template_eval.{h,cc}      Subset evaluator (filters, tests, string methods, loop vars, tojson) (T2/E10)
│  │  └─ chat_template.{h,cc}      Public chat renderer with reasoning_content split protection (P-D2) (T2/E10)
│  ├─ model/                       [Layer 1 & 2B: Model Specification & Trainer Stores]
│  │  ├─ spec.{h,cc}               ModelSpec parsed from GGUF metadata; layer topology & expert byte tables (T1/E3)
│  │  ├─ arch_moe.{h,cc}           Model-plugin interface for Qwen3.6 and other MoE architectures (E3/E13)
│  │  ├─ frozen_store.{h,cc}       Trainer read-only mapped/pinned store for experts, embedding, lm_head, routers (T7)
│  │  ├─ trainable_core.{h,cc}     Trainer isolated core weights and Mode Q effective weight (W_eff) buffers (T7/T9)
│  │  ├─ lora.{h,cu}               310-module LoRA fp32 masters, bf16 compute copies, Kaiming init (T7/T9)
│  │  └─ tape.{h,cu}               Forward routing tape (40 layers x N tokens x top-8 IDs & weights) + histogram (T8/T9)
│  ├─ kern/                        [Layer 1 & 2B: CUDA Kernels — Depends ONLY on hal/ and plain pointer/shape structs]
│  │  ├─ gemm_gguf/
│  │  │  ├─ dequant_ref.cu         G8 bit-exact fp32 dequantization oracle kernel (T3)
│  │  │  ├─ mma_bf16.cu            G1 dense & G3 grouped MoE forward GEMM (swap-AB, in-register dequant, SwiGLU) (T3)
│  │  │  ├─ mma_int8.cu            G2 dense & G4 grouped serve-faithful int8 MMA forward GEMM (T3)
│  │  │  ├─ grouped_tiles.cu       Device-side MoE tile consumer and per-assignment scatter (T3)
│  │  │  └─ dgrad_t.cu             G5 dense & G6 grouped transposed-dequant data-gradient GEMM + G7 LoRA epilogue (T3)
│  │  ├─ route/
│  │  │  ├─ router_topk.cu         R1 fp32 FMA router projection, deterministic top-8 argmax, 8-way softmax (T4/E6)
│  │  │  ├─ tile_map.cu            R2 block-parallel counting sort and device-side expert tile map builder (T4)
│  │  │  └─ combine.cu             R3 deterministic fixed-order expert combine fwd/bwd + router weight grad (T4/E6)
│  │  ├─ elementwise/
│  │  │  ├─ rmsnorm.cu             E1 RMSNorm (+1 folded) and per-head RMSNorm (A6) fwd/bwd (T4/E7)
│  │  │  ├─ swiglu.cu              E2 SwiGLU and A5 attention output-gate fwd/bwd (T4/E7)
│  │  │  ├─ gated_norm.cu          E3 DeltaNet gated RMSNorm (ssm_norm, plain) fwd/bwd (T4/E7)
│  │  │  ├─ residual.cu            E5 fp32 residual add and Q4_K embedding row gather + dequant (T4/E7)
│  │  │  └─ cast.cu                E5 fp32 <-> bf16 casts (T4/E7)
│  │  ├─ attn/
│  │  │  ├─ rope.cu                A1 IMRoPE [11,11,10,0] rotation fwd/bwd with document-start reset (T5/E7)
│  │  │  ├─ attn_fwd.cu            A2 FlashAttention-2-style GQA (8:1, head-dim 256) forward kernel (T5/E7)
│  │  │  ├─ attn_bwd.cu            A3 Fused GQA head-dim 256 backward kernel (deterministic dQ pass) (T5)
│  │  │  └─ attn_ref_blocked.cu    A4 Blocked materialized-score fp32 reference & fallback attention fwd/bwd (T5/E7)
│  │  ├─ deltanet/
│  │  │  ├─ gates.cu               D1 beta = sigma(W_b a) and g = ssm_a * softplus(W_a a + dt_bias) fwd/bwd (T6/E7)
│  │  │  ├─ conv1d.cu              D2 4-tap causal depthwise conv1d + SiLU + E4 L2-norm epilogue fwd/bwd (T6/E7)
│  │  │  ├─ l2norm.cu              E4 Standalone per-head L2-normalization fwd/bwd (T6/E7)
│  │  │  ├─ chunk_prep.cu          D3 Chunk C=64 cumulative decay and (I+N)^{-1} forward substitution (T6/E9)
│  │  │  ├─ chunk_state.cu         D4 Inter-chunk state recurrence sweep (128x128 fp32 state) (T6/E9)
│  │  │  ├─ chunk_out.cu           D5 Intra/inter-chunk output combination (T6/E9)
│  │  │  ├─ chunk_bwd.cu           D6 Reverse-chunk state sweep and triangular-solve adjoint backward (T6)
│  │  │  └─ seq_ref.cu             D7 Warp-per-column sequential fp32 (Decode/Oracle) & fp64 reference fwd/bwd (T6/E7)
│  │  ├─ loss/
│  │  │  └─ ce_chunk.cu            L1 256-position chunked fp32 cross-entropy loss, log-sum-exp, and Mode L losses (T4)
│  │  └─ optim/
│  │     ├─ adamw.cu               O1 Fused fp32 AdamW update and bf16 compute-copy refresh (T9)
│  │     ├─ grad_norm.cu           O2 Deterministic two-level global gradient norm and clip factor (T9)
│  │     └─ lora_grad.cu           O3 Deterministic tree-reduction for LoRA dA and dB across windows (T9)
│  ├─ cpu/                         [Layer 2A: Engine Cold-Expert CPU Path]
│  │  ├─ expert_q4k_avx2.cc        AVX2 Q4_K gate/up + SwiGLU + Q4_K down cold-expert kernel (E4)
│  │  └─ expert_q6k_avx2.cc        AVX2 Q6_K down-projection cold-expert kernel (E4)
│  ├─ gpu/                         [Layer 2A: Engine Hot-Bank & Zero-Copy GPU Path]
│  │  ├─ bank.{h,cu}               Static VRAM hot-bank loader, variable per-layer slot tables, zero-slot init (E5)
│  │  ├─ expert_q4k.cu             Hot-branch Q4_K gate/up/down decode and small-batch kernel (E5)
│  │  ├─ expert_q6k.cu             Hot-branch Q6_K down-projection kernel (E5)
│  │  └─ zerocopy.cu               16-byte (uint4) shared-memory-staged PCIe zero-copy cold expert kernel (P-E10) (E5/E12)
│  ├─ engine/                      [Layer 2A: Engine Execution, State, and Sampler]
│  │  ├─ kvcache.{h,cu}            q8_0 KV cache manager for the 10 full-attention layers (up to 131,072 ctx) (E7/E9)
│  │  ├─ state.{h,cu}              30-layer DeltaNet recurrent state store and 62.9 MiB prefix-cache snapshots (E7/E10)
│  │  ├─ layer.{h,cu}              Single MoE layer coordinator: spine -> router -> hot/cold split -> doorbell -> merge (E6)
│  │  ├─ decode_loop.{h,cu}        CUDA-graphed 40-layer token decode loop, hit counters, per-token timers (E8)
│  │  ├─ prefill.{h,cu}            Chunked prefill executor with bypass threshold rule (E9)
│  │  └─ sampler.{h,cu}            Counter-RNG sampler chain: temperature, top-k, top-p, min-p, penalties (E8/E10)
│  ├─ server/                      [Layer 2A: Native Raw-Socket HTTP/1.1 & SSE Server]
│  │  ├─ http.{h,cc}               State-machine HTTP/1.1 parser, chunked transfer decoder, SSE event writer (E10)
│  │  ├─ tools.{h,cc}              Native XML <tool_call><function=...><parameter=...> parser to OpenAI JSON (E10)
│  │  └─ api_openai.{h,cc}         /v1/chat/completions, /v1/models, reasoning_content splitter, request logger (E10)
│  ├─ config/                      [Layer 2B: Trainer Configuration]
│  │  └─ trainer_config.{h,cc}     Defaults: Mode Q on, delta_max_bytes = 0, promotions off, Expert overrides (T1b)
│  ├─ ingest/                      [Layer 2B: Stage A Base Model Converter]
│  │  └─ ingest_plan.{h,cc}        Stage A: 26 bf16 safetensors shards -> pelias_base_0step.gguf (D33) (T1b)
│  ├─ data/                        [Layer 2B: Trainer Dataset Pipeline]
│  │  ├─ dataset.{h,cc}            JSONL conversation reader and schema validator (T2)
│  │  ├─ masks.{h,cc}              Assistant generation span mask builder and independent marker verifier (P-D5) (T2)
│  │  ├─ packer.{h,cc}             64-slot chunk-aligned best-fit-decreasing window packer (<= 8,192 slots) (T2)
│  │  └─ audit.{h,cc}              Pre-training audits A-LEN, A-TOK, A-TPL, A-MASK, A-DUP, A-SPECIAL, A-FAMILY (T2)
│  ├─ sched/                       [Layer 2B: Trainer Layer-Major Scheduler]
│  │  ├─ layer_fwd.cu              Single layer forward across K windows + routing tape recording (T7/T8)
│  │  ├─ layer_bwd.cu              Single layer boundary recompute (using tape) + backward + LoRA grad accum (T7/T9)
│  │  ├─ head.cu                   Output RMSNorm + chunked lm_head + L1 cross-entropy fwd/bwd (T8/T9)
│  │  ├─ mtp.cu                    Optional blk.40 MTP forward/backward step (D12) (T8/T14)
│  │  └─ step.{h,cu}               Enqueue-only layer-major step coordinator (0..39 fwd -> loss -> 39..0 bwd) (T10)
│  ├─ train/                       [Layer 2B: Trainer Loop & Checkpointing]
│  │  ├─ schedule.{h,cc}           Linear warm-up (5%) + cosine decay learning-rate schedule (T11)
│  │  ├─ optimizer.{h,cc}          Host coordinator for O2 grad norm/clip, O1 AdamW, and Mode Q W_eff rebuild (T9/T11)
│  │  ├─ checkpoint.{h,cc}         Atomic 4-registry checkpoint writer/reader with SHA-256 manifest (T11)
│  │  ├─ metrics.{h,cc}            Step-wide JSONL telemetry and per-layer expert usage accumulator (T11)
│  │  └─ trainer.{h,cc}            Top-level training, evaluation, and bitwise-deterministic resume driver (T11)
│  ├─ export/                      [Layer 2B: Trainer Exporter]
│  │  ├─ export_adapter.{h,cc}     Writes unmerged GGUF LoRA adapter (adapter.gguf) (T12)
│  │  └─ export_core.{h,cc}        Writes merged-core.gguf (Mode Q copy or D22 byte-budget solver override) (T12)
│  ├─ verify/                      [Layer 1 & 2B: Verification & Gates]
│  │  ├─ canary.{h,cu}             Periodic 50-step GEMM/reduction soft-error canary and state checksum (T11)
│  │  ├─ kl.{h,cc}                 Truncated top-256 + lumped-tail KL divergence and top-1 comparator (T8/T12/E3)
│  │  ├─ dump_compare.{h,cc}       Per-layer tensor dump comparator against Oracle A and Oracle B (T7/T8/E3)
│  │  └─ gate_runner.{h,cc}        Executes G1-G7 and PG1-PG10 gates; emits verify_<gate>.json (T12/E8)
│  ├─ project/                     [Layer 2B: Universal-Use Project Management (U4)]
│  │  ├─ model_key.{h,cc}          Computes and caches SHA-256 + ModelSpec key; isolates per-model facts (Post-T12)
│  │  ├─ project_file.{h,cc}       Saves/loads user setup, forced overrides, and hashes (Post-T12)
│  │  └─ run_folder.{h,cc}         Creates timestamped non-overwriting run directories (Post-T12)
│  ├─ startup/                     [Layer 0/2B: Universal-Use Startup Probe (U1, U2, U6)]
│  │  ├─ quick_pass.{h,cc}         Startup self-probe (free VRAM, pinned BW, flush need, tensor burst) (Post-T12/E11)
│  │  ├─ machine_profile.{h,cc}    Caches slow probes by GPU/driver/OS/clock hash (Post-T12/E11)
│  │  └─ vram_scan.{h,cc}          Scans per-process GPU memory and proposes closeable apps (U6) (Post-T12)
│  ├─ intake/                      [Layer 2B: Universal-Use Parameter & Client Intake (U7, U8)]
│  │  ├─ sheets.{h,cc}             CSV reader/writer for persona.csv, parameters.csv, clients.csv + EXAMPLE check (Post-T12)
│  │  ├─ import.{h,cc}             Non-executing AST/rule importer for .py, .csv, .json, .md, .txt into DRAFT rows (Post-T12)
│  │  ├─ clients.{h,cc}            Client-style prompt/tool generator with CAPTURED/PUBLISHED/APPROXIMATION tags (Post-T12)
│  │  └─ checks.{h,cc}             Counts-only intake audits (coverage, AST parse, length ceiling, duplicates) (Post-T12)
│  ├─ help/                        [Layer 2B: Universal-Use Help Catalog (U10)]
│  │  └─ help_catalog.{h,cc}       One-sentence setting explanations and openable .txt help files (Post-T12)
│  ├─ warn/                        [Layer 2B: Universal-Use Safeguards (U9)]
│  │  ├─ fit.{h,cc}                Pre-training coverage/diversity and mid-training eval-vs-train gap warnings (Post-T12)
│  │  ├─ loops.{h,cc}              Post-training greedy/sampled n-gram repetition and EOS stop-token checker (Post-T12)
│  │  ├─ works.{h,cc}              Per-parameter held-out question and base-retention checker (Post-T12)
│  │  └─ client_matrix.{h,cc}      Evaluates trained model across all clients.csv front-end styles (Post-T12)
│  ├─ flow/                        [Layer 2B: Universal-Use Confirmation Loop & Expert Mode (U3, U10, U11)]
│  │  ├─ plan_card.{h,cc}          Generates 2-4 feasible training plans with fit, speed, and gate risk (Post-T12)
│  │  ├─ proposal.{h,cc}           Propose -> user confirm -> apply to copy -> re-test loop (capped at 1 retry) (Post-T12)
│  │  ├─ expert.{h,cc}             Expert mode USER-FORCED value tracker and real-allocation pre-test (Post-T12)
│  │  └─ summary.{h,cc}            Final pre-Start summary screen requiring explicit warning acknowledgement (Post-T12)
│  ├─ serve/                       [Layer 2B: Unification Bridge]
│  │  └─ serve_command.{h,cc}      Emits or launches the native Pelias server command for a finished GGUF (Unify)
│  ├─ ui/                          [Layer 2B: Lithic-to-Voxel Interface]
│  │  └─ pelias_lithic_ui.{hpp,cpp} Pre-baked dual-texture spear progress bar, chiselled glyphs, telemetry grid (Post-T12)
│  └─ cli/
│     ├─ main_train.cc             pelias-train CLI (probe, ingest, audit-data, dump, train, eval, export, verify) (T0+)
│     └─ main_engine.cc            pelias-engine CLI (serve, bench, trace, probe, repack) (E2/E8)
├─ resources/
│  ├─ sheets/                      persona.csv, parameters.csv, clients.csv (with 3-4 erasable EXAMPLE rows each)
│  ├─ help/                        Plain-English setting descriptions (.txt)
│  └─ ui/                          Pre-baked spear_stone.png, spear_holo.png, spear_layout.json, logo_full.png
├─ profiles/                       Exported PELIAS_HOT_SET profiles
├─ bench/                          Standalone hardware floor microbenchmarks (ram_bw, handoff, dual_bus, pin_modes, wide_load)
├─ tools/                          check_module_deps.py, check_resources.py, unicode table generator, oracle runners
└─ tests/
   ├─ unit/                        CPU-only & fp64 reference unit tests for every kernel and module
   ├─ smoke/                       Framework and arena smoke tests
   ├─ parity/                      Per-layer and full-model dump comparison harnesses
   ├─ probes/                      Executable Wave 0 and stage probes (P-H, P-M, P-K, P-Q, P-S, P-D, P-T, P-X, P-L, P-E)
   └─ oracle/                      External, read-only llama.cpp 2ca15f540 and PyTorch fp32 slice harnesses (never linked)
```

### 4.3 Single-Sourced Acyclic Module Dependency Graph

Enforced by `tools/check_module_deps.py` (`BC1`). A module on the left may only `#include` headers from or link against the modules listed on the right:

* **`os`**: `[]` (no internal dependencies)
* **`hal`**: `["os"]`
* **`quant`**: `["hal"]`
* **`plan`**: `["hal"]`
* **`bus`**: `["os", "hal", "quant"]`
* **`bank`**: `["hal"]`
* **`io`**: `["os", "hal", "quant"]`
* **`tok`**: `[]` (pure C++20 + static UCD tables; reads raw strings/arrays passed from `io`)
* **`kern`**: `["hal", "quant"]` (**Critical Rule:** `kern` never includes `model`, `io`, `sched`, or `engine`; it takes only plain structs of device pointers and shapes)
* **`model`**: `["hal", "quant", "io"]`
* **`config`**: `[]`
* **`ingest`**: `["hal", "quant", "io", "config"]`
* **`data`**: `["io", "tok"]`
* **`stream`**: `["hal", "bus", "model"]`
* **`sched`**: `["hal", "kern", "model", "stream"]`
* **`train`**: `["hal", "config", "data", "model", "sched"]`
* **`export`**: `["hal", "quant", "io", "config", "model", "bank"]`
* **`verify`**: `["hal", "io", "quant"]`
* **`cpu`**: `["os", "hal", "quant", "bus"]`
* **`gpu`**: `["hal", "quant", "bank", "bus", "kern"]`
* **`engine`**: `["os", "hal", "plan", "bus", "bank", "quant", "io", "model", "kern", "cpu", "gpu"]`
* **`server`**: `["os", "io", "tok", "engine"]`
* **`project`**: `["io"]`
* **`startup`**: `["os", "hal", "plan", "io", "project"]`
* **`intake`**: `["io", "tok", "data"]`
* **`help`**: `[]`
* **`warn`**: `["io", "data", "verify", "intake"]`
* **`flow`**: `["config", "plan", "startup", "project", "intake", "warn", "verify", "help"]`
* **`serve`**: `["project"]`
* **`ui`**: `[]`
* **`cli`**: `["config", "ingest", "train", "export", "verify", "flow", "serve", "engine", "server"]`

---

## 5. Proprietary Subsystem Specifications

### 5.1 Operating-System Layer (`src/os/`)

One codebase compiles into two native binaries (`[sync R1]`). All OS-specific calls are encapsulated inside `src/os/` and `src/hal/host_alloc_{wsl,win32}.cc`:

| OS Service | WSL2 Build (POSIX) | Windows-Native Build (Win32 / Winsock2) | Behavioural Specification |
|---|---|---|---|
| **File Open & Size** | `open(O_RDONLY)`, `fstat` | `CreateFileW(GENERIC_READ, FILE_SHARE_READ, ...)`, `GetFileSizeEx` | Accepts UTF-8 paths everywhere; converts to UTF-16 via `MultiByteToWideChar` inside the Win32 backend. |
| **Read-Only File Map** | `mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)`, `munmap` | `CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL)`, `MapViewOfFile(..., FILE_MAP_READ, ...)`, `UnmapViewOfFile` | Zero-copy read-only views. Supports partial unmapping (or dual-view mapping on Win32) to release the $18.16\text{ GiB}$ expert region after pinning while keeping the header mapped. |
| **Memory Advice** | `madvise(MADV_SEQUENTIAL / MADV_WILLNEED / MADV_DONTNEED)` | `PrefetchVirtualMemory` | Advisory only; never alters numerical results. |
| **Sockets Lifecycle** | `socket`, `bind`, `listen`, `accept`, `poll`, `recv`, `send(..., MSG_NOSIGNAL)`, `shutdown`, `close` | `WSAStartup`, `socket`, `setsockopt(SO_EXCLUSIVEADDRUSE)`, `bind`, `listen`, `accept`, `WSAPoll`, `recv`, `send`, `shutdown`, `closesocket`, `WSACleanup` | Default bind `127.0.0.1:8012`. `SO_EXCLUSIVEADDRUSE` on Windows prevents port hijacking; if port `8012` is already bound, startup fails immediately with a clear error. |
| **Threads & Affinity** | `pthread_create`, `pthread_setaffinity_np`, `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` | `CreateThread`, `SetThreadAffinityMask` / `SetThreadSelectedCpuSets`, `GetLogicalProcessorInformationEx` | Pins the decode thread to Core 0 and cold-expert workers to distinct physical cores (1 thread per physical core in `1T` mode vs. SMT in `E-A1`). |
| **Monotonic Clock** | `clock_gettime(CLOCK_MONOTONIC)` | `QueryPerformanceCounter` / `QueryPerformanceFrequency` | Used for host-side token arrival and split telemetry; GPU kernel timing always uses `cudaEventRecord`. |
| **Signal Handling** | `sigaction(SIGINT, SIGTERM)` | `SetConsoleCtrlHandler` | **First signal:** stop accepting requests, finish or abort current token, flush `tier_stats.json` and request log, close socket, exit 0. **Second signal:** immediate exit. |

### 5.2 VRAM Governor, Arenas, and Memory Plans (`src/hal/`, `src/plan/`)

#### 5.2.1 Engine Arena Lifetimes (`pelias-engine`)
1. **`model` Arena (Process Lifetime):** Holds resident spine weights ($\sim 0.80\text{ GiB}$), `token_embd` ($0.27\text{ GiB}$), `output.weight` ($0.39\text{ GiB}$), `F32` routers ($0.08\text{ GiB}$), norms, and the per-layer bank slot lookup table.
2. **`bank` Arena (Process Lifetime):** Holds the water-filled static hot bank ($7.5\text{ GiB} = 4,170$ experts across 40 layers, $84\text{--}161$ experts/layer, including **one all-zero slot per layer** $= 72.6\text{ MiB}$ total).
3. **`session` Arena (Slot Lifetime):** Holds the `q8_0` KV cache for the 10 full-attention layers and the $128 \times 128$ `fp32` recurrent states for the 30 DeltaNet layers (plus $62.9\text{ MiB/snapshot}$ prefix-cache snapshots).
4. **`prefill` Arena (Prompt Lifetime):** Scratch activation buffers scaled to `prefill.micro_batch` (`-ub`).
5. **`token` Arena (Token Lifetime):** Per-token scratch buffers reset at the end of every token step.
6. **`host_pinned` Pool (Process Lifetime):** Allocated via `cudaHostAlloc` ($18.164\text{ GiB}$ trunk experts or $18.65\text{ GiB}$ with `blk.40`, plus cache-line-aligned doorbell mailboxes).

**The Zero-Slot No-Op Invariant (Eliminating the Negative-ID Crash Class):**
In the prototype, passing $-1$ expert IDs into `ggml`'s CUDA quantizer caused `const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;` to promote $-1$ to unsigned $4,294,967,295$, reading $8.8\text{ TB}$ out of bounds on chunks $\ge 9$ tokens (`quantize_signed_id.patch`). In Pelias, **no kernel ever receives a negative expert ID**. The partition step emits dense, valid-only index lists (`hot_list` with `n_hot`, `cold_list` with `n_cold`), and any padded or unrouted slot in a fixed-width tile points to the layer's **all-zero bank slot**, contributing exact zeros without branching.

#### 5.2.2 Trainer Device and Host Memory Plan (`pelias-trainer`, Window $T = 8,192$, Step up to $32\text{k}$ Tokens)

| Memory Component | Size (`GiB`) | Label & Derivation |
|---|---:|---|
| CUDA Context & Driver Overhead | $0.60$ | `ESTIMATE` (`P-H8` measured $0.15\text{ GiB}$ context; $0.60\text{ GiB}$ budget provides conservative padding; verified by `P-M1`). |
| Frozen Core Base Weights (Served Mix) | $0.80$ | `READ-SRC` ($820.3\text{ MiB}$). |
| Untied `lm_head` (`Q6_K`) | $0.39$ | `READ-SRC` ($397.9\text{ MiB}$). |
| Token Embedding (`Q4_K`) | $0.27$ | `READ-SRC` ($272.8\text{ MiB}$). |
| Routers (`F32` $\times 41$) & Small `F32` Tensors | $0.09$ | `READ-SRC` ($82.0\text{ MiB}$ routers + $< 8\text{ MiB}$ norms/conv/gates). |
| Streaming Expert Device Slots ($2 \times 498\text{ MiB}$) | $0.97$ | `READ-SRC` (sized to the largest `Q6_K`-down layer). |
| LoRA `fp32` Masters + Grads + Adam $m, v$ ($r = 16$) | $0.32$ | `DERIVED` ($21,166,080 \times 16\text{ B} = 323.0\text{ MiB}$). |
| LoRA `bf16` Compute Copies | $0.04$ | `DERIVED` ($21,166,080 \times 2\text{ B} = 40.4\text{ MiB}$). |
| Resident Layer-Boundary Tensors (cur, next, grad; `fp32`, $32\text{k}$ tok) | $0.75$ | `DERIVED` ($3 \times 32,768 \times 2,048 \times 4\text{ B} = 768\text{ MiB}$). |
| Per-Layer Working Set ($T = 8,192$ fwd + recompute + bwd peak) | $2.50$ | `ESTIMATE` (DeltaNet fwd $930\text{ MiB}$ + MoE fwd $550\text{ MiB}$ + MoE bwd $700\text{ MiB} \approx 2,180\text{ MiB}$; measured by `P-M2`). |
| Cross-Entropy Chunk ($256 \times 248,320$ `fp32` logits + workspace) | $0.65$ | `MEASURED` in v0.2 ($242.5\text{ MiB}$ logits + grad/workspace). |
| Routing Tape & Tile-Map Index Buffers | $0.02$ | `DERIVED`. |
| Safety Margin | $0.50$ | `DECISION`. |
| **Base Trainer Total** | **$7.90$** | Leaves **$2.80\text{ GiB}$ headroom** under $10.7\text{ GiB}$. |
| **Mode Q Effective Weight Copy (`W_eff`, Default `D31`)** | $+0.80\text{--}1.10$ | `ESTIMATE` (`W_eff` in served `Q4_K`/`Q6_K` formats). **Total with Mode Q: $8.70\text{--}9.00\text{ GiB}$ ($\mathbf{1.70\text{--}2.00\text{ GiB}}$ headroom)**. |
| *Optional Mode L Trainable Router (`fp32` copy + grad + Adam $m,v$)* | *$+0.32$* | `DERIVED` ($21.50\text{ M} \times 16\text{ B} = 328\text{ MiB}$; fits easily inside headroom). |
| **Host Pinned RAM (`cudaHostAlloc`)** | **$23.65\text{--}28.65$** | `DERIVED`: $18.65\text{ GiB}$ pinned expert store + $5.0\text{ GiB}$ ($16\text{k}$ tok) or $10.0\text{ GiB}$ ($32\text{k}$ tok) `fp32` boundary store (+ $\sim 4\text{ GiB}$ pageable runtime $= 28\text{--}33\text{ GiB}$ total under $48\text{ GiB}$ WSL2 cap; `P-H7` proved $32\text{ GiB}$ pinned). |

### 5.3 Memory-Mapped GGUF Reader, Writer, and Byte-Splice Exporter (`src/io/`)

**12-Step Order of Operations for `io/gguf_reader`:**
1. **Path & Medium Check:** Warn on the plan card if a GGUF on a Windows 9p mount (`/mnt/c`, `/mnt/d`) is opened from WSL2 without being cached on `ext4` or pinned RAM.
2. **Open & Size Validation:** Reject any file smaller than the 24-byte GGUF v3 header.
3. **Read-Only Map:** Map via `os/file_map` (`PROT_READ` / `PAGE_READONLY`).
4. **Header Parse:** Verify magic `GGUF` (`0x46554747`), version $3$, `tensor_count` ($753$), and `metadata_kv_count` on a little-endian host.
5. **Metadata KV Parse with Bounds Checks:** Every string length and array count is checked against remaining file bytes before advancing the cursor. Extract `general.architecture`, `general.alignment` (default $32$), `block_count` ($41$), expert parameters, `tokenizer.ggml.*`, `tokenizer.chat_template`, and optional `pelias.*` keys.
6. **Tensor Info Table Parse:** For each of the 753 tensors, read name, dimensions, `ggml_type`, and relative offset. Compute exact byte size using 64-bit overflow-checked multiplication against `quant/block_types.h`. Reject any tensor whose offset is not a multiple of `alignment`, whose end exceeds file size, or whose byte range overlaps another tensor.
7. **Data Start & Tensor Map:** Compute `data_start = align_up(header_end, alignment)` ($10,991,104$ in the served file) and build the zero-copy `TensorView` map.
8. **Layout & Alignment Audit (`P-E9`):** Verify that all 123 expert tensors are $16\text{-byte}$ aligned; log $4\text{ KiB}$ and $2\text{ MiB}$ alignment status; if `layout_manifest.json` is present, verify every offset and hash.
9. **Optional Full-File SHA-256 (`U4`):** Compute and cache the model key (`SHA-256` + `ModelSpec` dimensions).
10. **Build `ModelSpec`:** Populate layer types, expert byte tables ($432\text{ MiB}$ vs. $498\text{ MiB}$), and core target lists.
11. **Arena Upload & Pinned Copy:** Upload spine tensors to the GPU `model` arena; copy routed experts into the `host_pinned` pool; upload resident hot-bank experts to the `bank` arena; unmap the $18.16\text{ GiB}$ expert file range so host page cache is freed.
12. **Loud Failure:** Any parse or alignment violation returns an explicit `pelias::Status` containing the exact tensor name and byte offset.

**Zero-Cost $2\text{ MiB}$ Alignment in `io/gguf_writer` (`--layout expert-last`, `F12`):**
Because every one of the 123 expert tensors is an exact multiple of $2\text{ MiB}$ ($144\text{ MiB}$ for `Q4_K` or $210\text{ MiB}$ for `Q6_K`), placing all non-expert tensors first in the GGUF, inserting a metadata padding string `pelias.pad` of $< 2\text{ MiB}$ in the header so the first expert tensor starts on an exact $2\text{ MiB}$ ($2,097,152\text{ B}$) boundary, and writing the 123 expert tensors contiguously guarantees that **every single expert tensor is $2\text{ MiB}$-aligned at zero tensor byte overhead**.

### 5.4 Proprietary Block Quantizers and Dequantizers (`src/quant/`)

Re-expressed in original C++20/CUDA code from the `ggml` block specification (`READ-SRC`) and verified bit-for-bit by `P-Q1`, `P-Q2`, and `P-E6`:

| Format | Block Size | Elements | Byte Layout | Exact Dequantization Formula (`fp32`) |
|---|---:|---:|---|---|
| **`Q4_K`** | $144\text{ B}$ | $256$ | `d` (`f16`, $2\text{ B}$), `dmin` (`f16`, $2\text{ B}$), `scales[12]` (packed 6-bit scales/mins for 8 sub-blocks of 32), `qs[128]` (4 chunks of $32\text{ B}$: low nibbles = sub-block $2c$, high nibbles = sub-block $2c+1$). For $j < 4$: $\text{sc}_j = \text{scales}[j] \mathrel{\&} 63$, $m_j = \text{scales}[j+4] \mathrel{\&} 63$. For $j \ge 4$: $\text{sc}_j = (\text{scales}[j+4] \mathrel{\&} 15) \mid ((\text{scales}[j-4] \gg 6) \ll 4)$, $m_j = (\text{scales}[j+4] \gg 4) \mid ((\text{scales}[j] \gg 6) \ll 4)$. | $y = (d \cdot \text{sc}_j) \cdot q - (\text{dmin} \cdot m_j)$ |
| **`Q6_K`** | $210\text{ B}$ | $256$ | `ql[128]` (low 4 bits), `qh[64]` (high 2 bits), `scales[16]` (`int8` scales for 16 sub-blocks of 16), `d` (`f16`, $2\text{ B}$). Per 128 values ($l \in [0, 31]$): $q_l = (\text{ql}[l] \mathrel{\&} 15) \mid (((\text{qh}[l] \gg 0) \mathrel{\&} 3) \ll 4)$; $q_{l+32} = (\text{ql}[l+32] \mathrel{\&} 15) \mid (((\text{qh}[l] \gg 2) \mathrel{\&} 3) \ll 4)$; $q_{l+64} = (\text{ql}[l] \gg 4) \mid (((\text{qh}[l] \gg 4) \mathrel{\&} 3) \ll 4)$; $q_{l+96} = (\text{ql}[l+32] \gg 4) \mid (((\text{qh}[l] \gg 6) \mathrel{\&} 3) \ll 4)$. | $y = d \cdot \text{scales}_{\lfloor i/16 \rfloor} \cdot (q_i - 32)$ |
| **`Q8_0`** | $34\text{ B}$ | $32$ | `d` (`f16`, $2\text{ B}$), `qs[32]` (`int8`). | $y = d \cdot q_i$ (`d = amax / 127.0f`, `q = roundf(x / d)`). |
| **`q8_1`** | $36\text{ B}$ | $32$ | `d` (`f16`, $2\text{ B}$), `s` (`f16`, $2\text{ B}$, stores $d \sum_{i=0}^{31} q_i$), `qs[32]` (`int8`). MMQ groups 4 blocks ($128$ values, $144\text{ B}$). | GPU activation block: `Q4_K` sub-block dot uses $s_x \left(d \cdot \text{sc}_j \sum q_w q_x - \text{dmin} \cdot m_j \sum q_x\right)$. |
| **`Q8_K`** | $292\text{ B}$ | $256$ | `d` (`f32`, $4\text{ B}$), `qs[256]` (`int8`), `bsums[16]` (`int16` sums of 16-element sub-blocks). | CPU AVX2 activation block (`F14`). |

### 5.5 Doorbell Coordinator (`src/bus/doorbell_handoff`)

Coordinates parallel GPU hot-expert and CPU cold-expert execution across all 40 MoE layers without OS synchronization:
1. **Cache-Line-Isolated Mailboxes:** Allocated in `host_pinned` (`cudaHostAllocMapped`). For each layer $l$, the GPU-to-CPU request mailbox and the CPU-to-GPU reply flag occupy separate $64\text{-byte}$ cache lines to prevent false sharing.
2. **Seven-Step Per-Layer Execution Sequence:**
   * **(a)** The GPU finishes the `fp32` router (`R1`) and bank slot-table lookup, writing the cold expert count $n_{\text{cold}} \in [0, 8]$, cold expert IDs, routing weights, and input activation into the pinned payload buffer.
   * **(b)** The GPU publishes the monotonically increasing `(token_seq, layer_idx)` word into the request doorbell using **release** memory ordering (`E-A2`).
   * **(c)** The GPU immediately launches the hot-expert branch (`gpu/expert_q4k/q6k.cu`) on the same stream without waiting. (If $n_{\text{cold}} = 0$, steps d–e are bypassed immediately; $36.1\%$ of layer-steps have $0$ misses per `P-E1`).
   * **(d)** Pinned CPU worker threads spinning on the request doorbell with **acquire** ordering (`MOV` on x86, avoiding `LOCK XCHG` per AMD Family 19h SOG $\S 2.15$, `G7`) read the payload and execute their assigned cold expert row slices in AVX2 (`cpu/expert_q4k/q6k_avx2.cc`).
   * **(e)** The last CPU worker to finish atomic-decrements the barrier counter, writes the combined CPU partial vector (`fp32`) to the reply buffer, and publishes `(token_seq, layer_idx)` to the reply doorbell with **release** ordering.
   * **(f)** The GPU merge kernel polls the reply doorbell with acquire ordering and bounded `__nanosleep` backoff, then adds the CPU partial to the GPU hot-branch output.
   * **(g)** The merged layer output feeds layer $l+1$.
3. **Timeout & Fallback Safety:** Resident doorbell polling is scoped strictly to a single token ($\sim 18\text{--}26\text{ ms}$). If the reply flag does not arrive within `handoff.timeout_us`, the kernel exits with a timeout status and the host completes the layer via `stream_sync` fallback.

### 5.6 Native Byte-Level BPE Tokenizer and Chat-Template Renderer (`src/tok/`)

Built from the exact census of the checkpoint's `tokenizer.json` and `chat_template.jinja` (`2026-10-24`, `MEASURED` counts):
* **Tokenizer Specification (`P-D1`):**
  * **Vocabulary & Merges:** Read directly from GGUF metadata (`tokenizer.ggml.*`, model `gpt2`): **$248,044$ vocabulary entries**, **$247,587$ BPE merges**, and **$33$ added tokens** ($21$ special tokens, including `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`).
  * **Pipeline:** (1) Split input on special/added tokens when `parse_special = true`; (2) apply **NFC** Unicode normalization (`tok/normalizer`, controlled by `tok.apply_nfc` verified in `P-D1`); (3) split via the hand-written pre-tokenizer state machine matching case-insensitive English contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), runs of Unicode letters/marks (`\p{L}`, `\p{M}`), single digits (`\p{N}`), runs of punctuation/symbols, newline runs, and whitespace with a `(?!\S)` look-ahead; (4) map UTF-8 bytes through the 256-entry `gpt2` byte-to-unicode table; (5) apply BPE merge ranks iteratively using a priority queue / piece cache; (6) during SSE streaming decode, hold back incomplete multi-byte UTF-8 sequences in `tok/utf8` until the next token completes the codepoint.
* **Chat-Template Subset Interpreter (`P-D2`):**
  * **Scope ($158$ lines, $8,057\text{ B}$):** Lexes and evaluates the exact Jinja subset used by the Qwen template: statements `set` (26 occurrences), `if`/`elif`/`else` (30/7/9), `for` (6), `macro` (1); 41 expression blocks; filters `trim`, `length`, `tojson`, `safe`, `string`; tests `string`, `iterable`, `mapping`, `defined`, `none`, `undefined`, `sequence`; string methods `startswith`, `endswith`, `split`, `rstrip`, `lstrip`; `namespace()`, `raise_exception()`, and `loop.{index0, first, last, previtem, nextitem}`. Any unsupported construct fails loudly at load time.
  * **The `</think>` Split Trap:** Because the Qwen template splits assistant messages on `</think>` when `reasoning_content` is omitted (which silently corrupted 82 of 3,000 turns in early audits), `tok/chat_template` always passes `reasoning_content` explicitly as a separate field and round-trip verifies every assistant turn during data prep.

### 5.7 Native Raw-Socket HTTP/1.1 and Server-Sent Events (SSE) Server (`src/server/`)

Replaces `llama-server` (`[sync R2]`) so that Visual Studio 2026 Copilot (`http://localhost:8012/v1`), `odysseus-ai`, web UIs, Ollama-style clients, and the Pelias benchmark suites connect with zero external server dependencies.

**10-Step Request Lifecycle:**
1. **Accept & Parse:** `os/net` accepts connections on `127.0.0.1:8012` (`server.bind_address`, `server.port`). `server/http` parses the HTTP/1.1 request line, headers (case-insensitive, up to `server.max_header_bytes`), and body (`Content-Length` up to `server.max_body_bytes` or `Transfer-Encoding: chunked`), handling `Expect: 100-continue`.
2. **JSON Validation & Field Telemetry:** `io/json` parses the request body. Known fields (`model`, `messages`, `tools`, `tool_choice`, `stream`, `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens` / `n_predict`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `chat_template_kwargs`) are extracted; **unknown fields are ignored and counted by name in the structural request log** to close the client catalogue (`6.14`).
3. **Template Render & Tokenize:** `tok/chat_template` renders `messages[]` (supporting roles `system`, `developer`, `user`, `assistant`, `tool`) and `tools[]` with `add_generation_prompt = true`, and `tok/tokenizer` encodes the prompt into token IDs.
4. **Context Overflow Check:** With context-shift disabled (`engine.context_overflow = stop`, matching `--no-context-shift`), any prompt exceeding `kv.context_tokens` returns HTTP `400` with an OpenAI-formatted JSON error object.
5. **Recurrent-State Snapshot Prefix Cache (`engine/state`):** Matches the longest common token prefix with cached conversation state. Because Gated DeltaNet recurrent states ($30 \times 32 \times 128 \times 128 \times 4\text{ B} = 60.0\text{ MiB}$ + conv history $= \mathbf{62.9\text{ MiB/slot}}$) cannot be sliced backward like an attention KV cache, the cache restores the nearest saved recurrent-state checkpoint boundary ($\sim 0.5\text{ ms}$ restore time) and prefills only the delta tokens.
6. **Chunked Prefill:** `engine/prefill` processes remaining prompt tokens in chunks of `prefill.micro_batch` (using the dual hot/cold path for chunks $<$ `prefill.bypass_tokens` and the large-batch path above it).
7. **Token Decode & Sampling:** `engine/decode_loop` executes the 40 MoE layers per token and `engine/sampler` selects the next token.
8. **Reasoning & XML Tool-Call Splitting (`server/tools`, `server/api_openai`):**
   * Tokens generated before `</think>` are emitted in the `reasoning_content` delta field (`api.reasoning_field = reasoning_content`, matching `--reasoning-format deepseek`), never mixed into `content`.
   * Native XML tool calls (`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`) are parsed into OpenAI `tool_calls` array deltas with JSON-serialized `arguments` typed against the request's tool schema.
9. **SSE Streaming:** Each token delta is flushed immediately over the socket as `data: {...}\n\n` with `Content-Type: text/event-stream`. If `send` detects a client disconnect, the mailbox abort word (`6.6`) is set and the decode slot is freed immediately.
10. **Completion & Telemetry:** Emits `finish_reason` (`stop`, `length`, or `tool_calls`), token usage statistics, and final `data: [DONE]\n\n`, then writes structural telemetry (zero message text) to the request log.

**Mapping of Current Launcher Flags and Environment Variables to Proprietary Parameters (`6.14`):**

| Legacy `llama-server` Flag / Env Var | Current Value | Proprietary Pelias Parameter | Default & Scope |
|---|---|---|---|
| `-m <path>` | `<model alias>-Q4_K_M.gguf` | `model.path` | Required path to served GGUF. |
| `--alias` | `<model alias>` | `model.alias` | Returned in `/v1/models` and response objects. |
| `--host`, `--port` | `127.0.0.1`, `8012` | `server.bind_address`, `server.port` | `127.0.0.1`, `8012`. |
| `--jinja` | on | Built-in (`tok/chat_template`) | Always active. |
| `-np` | `1` | `engine.slots` | `1` concurrent decode slot. |
| `-c` | `32768` / `131072` | `kv.context_tokens` | `32768` (`131072` supported in `E9`). |
| `-n` | `-1` | `generation.max_tokens` | `-1` (unlimited up to `kv.context_tokens`). |
| `--no-context-shift` | on | `engine.context_overflow` | `stop` (never evict prefix silently). |
| `-ngl 99`, `--n-cpu-moe 40`, `--fit off` | all spine on GPU, cold experts on CPU | `plan/memory_plan` | Automatic: spine + hot bank in VRAM, cold experts in pinned RAM. |
| `--load-mode none` | anonymous host memory | `host.pinned_pool` (`cudaHostAlloc`) | Automatic via `E11` startup probe. |
| `--flash-attn on` | on | `kern/attn/attn_fwd.cu` | Always active. |
| `--cache-type-k`, `--cache-type-v` | `q8_0` | `kv.type` | `q8_0`. |
| `-t 6`, `-tb 8` | 6 decode / 8 batch threads | `cpu.threads`, `cpu.threads_batch` | Auto-configured by `E4` physical-core probe (`E-A1`). |
| `-b 4096`, `-ub 2048` / `4096` | batch / micro-batch size | `prefill.batch`, `prefill.micro_batch` | Engine setting (`D31`, `D34`). |
| `--temp 1.0`, `--top-p 0.95`, `--top-k 20`, `--min-p 0.0` | vendor thinking defaults | `sampling.{temp, top_p, top_k, min_p}` | `1.0`, `0.95`, `20`, `0.0` (overridable per request). |
| `--reasoning-format deepseek` | `reasoning_content` field | `api.reasoning_field` | `reasoning_content`. |
| `--chat-template-kwargs '{"enable_thinking":true}'` | thinking enabled | `template.enable_thinking` | `true` (overridable per request). |
| `PELIAS_HOT_SET` | `pelias_hot_set_b7.0.txt` / `b7.5` | `bank.hot_set_file` | Path to `PELIAS_HOT_SET` profile. |
| `PELIAS_TIER_BYPASS_N` | `32` (Cell A) / `9` (prod) | `prefill.bypass_tokens` | `32` (dual path for chunks $< 32$). |
| `PELIAS_TIER_MAX_MIB` | `7300` / `7680` | `bank.size_gib` | `7.5` GiB (`O1`, governed by `hal/vram_governor`). |
| `PELIAS_TIER_VERBOSE`, `PELIAS_TIER_STATS_FILE` | `1`, `tier_stats.json` | `log.level`, `stats.file` | Written atomically on clean stop. |

---

## 6. Trainer Precision Plan, Layer-Major Scheduler, Stage A Ingest, and Optional Modes

### 6.1 Two-Stage Ingest and Training Pipeline (`D1`, `D9`, `D33`)

1. **Stage A (`src/ingest/` — `bf16` Safetensors $\rightarrow$ `pelias_base_0step.gguf`):**
   * Reads the user's 26 `bf16` safetensors shards ($71.90\text{ GB}$, `4.2`) one tensor at a time via `os/file_map`.
   * Applies exact converter transformations: strips `model.language_model.`, drops `model.visual.*` (333 tensors), splits fused `mlp.experts.gate_up_proj` $[256, 1024, 2048]$ at row $512$, permutes DeltaNet V-heads to GGUF tiled order (`P-S3`), folds $+1$ into RMSNorm weights except `ssm_norm` (`P-S2`), computes `ssm_a = -exp(A_log)`, and retains `mtp.*` as `blk.40` (753 total GGUF tensors).
   * Quantizes tensors using Pelias's own bit-exact port of the reference block quantizers (`quant/quant_cpu`, `P-E6`) following the stock `Q4_K_M` type map (no imatrix).
   * Writes `pelias_base_0step.gguf` ($\sim 20.22\text{ GiB}$, cached on disk) with source shard SHA-256 provenance keys and benchmarks it against the `P-X0` reference and full `bf16` model (`P-X6`, `P-X7`).
2. **Stage B (Training on Exact Served Bytes):**
   * Loads `pelias_base_0step.gguf`. The 123 routed expert tensors ($18.65\text{ GiB}$ with `blk.40`) are pinned in host RAM, never updated, and **byte-spliced verbatim** into the exported GGUF with a per-tensor SHA-256 proof (`P-X1`).

### 6.2 Mode Q (Merge-Aware Training Through the Exact Quantizer — Default `D23`, `D31`, `D32`)

To solve the **Delta vs. Step 4-Bit Merge Erase Problem (`F4`)**—where merging a small LoRA update $\Delta W = \frac{\alpha}{r} B A$ into a `Q4_K` weight (step size $\approx 0.320\sigma$, rounding noise $9.2\%\sigma$) erases updates smaller than half a step—while respecting the engine's tight VRAM budget ($\Delta_{\max} = 0.16\text{ GiB}$ at default micro-batch and $-0.09\text{ GiB}$ at `-ub 2048` on the whole-card reading of the $10.7\text{ GiB}$ ceiling):
1. **Effective Weight Construction (`GQ` kernel):** Once per optimizer step, for every trainable core tensor $W_{\text{base}}$ of served type $T \in \{\text{Q4\_K}, \text{Q6\_K}\}$:
   $$W_{\text{eff}} = Q_T\left( \text{deq}(W_{\text{base}}) + \frac{\alpha}{r} B A \right)$$
   where $Q_T$ is Pelias's bit-exact implementation of the `ggml` reference quantizer (`P-E6`).
2. **Forward & Recompute:** Execute the standard `G1`–`G4` quantized GEMMs directly on the packed bytes of $W_{\text{eff}}$ (no separate LoRA epilogue branch is needed in forward).
3. **Straight-Through Estimator (STE) Backward:** The data-gradient uses $W_{\text{eff}}$. Using $\frac{\partial L}{\partial \Delta W} := \frac{\partial L}{\partial W_{\text{eff}}}$, the gradients to the `fp32` master weights $A, B$ are the standard low-rank products:
   $$dB = \frac{\alpha}{r} (dY)^\top (X A^\top), \quad dA = \frac{\alpha}{r} (B^\top dY^\top) X$$
   evaluated at the quantized forward activations. The `fp32` masters $A, B$ accumulate sub-step updates across steps until weights cross quantization thresholds.
4. **Zero-Drift Export (`E-F`):** At export, the final step's $W_{\text{eff}}$ bytes are written directly into `merged-core.gguf` at the native served types ($\mathbf{\Delta_{\max} = 0\text{ extra bytes}}$). What is exported is bit-for-bit what the training loss saw. An unmerged `adapter.gguf` (`E-A`) is also emitted alongside for information.
5. **Byte-Budget Solver Override (`D22`, `D32`):** Promoting core tensor classes to `Q6_K` ($+282.0\text{ MiB}$) or `Q8_0` ($+607.4\text{ MiB}$) via the `P-Q3` retention solver is off by default and selectable in Advanced/Developer (Expert) mode (`U11`) for GPUs with $> 12\text{ GB}$ VRAM or smaller banks.

### 6.3 The Layer-Major $K$-Window Scheduler and 3-Stream Streamer (`D5`, `D7`, `D13`, `D27`)

In `pelias-trainer 0.2.1`, streaming 36 non-resident expert layers per micro-batch moved $30.38\text{ GiB}$ in 72 fetches per micro-batch ($\sim 121.5\text{ GiB}$ per optimizer step at `grad_accum = 4`), stalling the GPU for $26.8\text{--}29.5\%$ of step time. Pelias replaces token-major micro-batching with **Layer-Major $K$-Window Scheduling**:

```text
STEP(N tokens packed into K windows w_1..w_K of T <= 8,192 slots, routing tape Tape[40][K]):
  1. Gather & dequant Q4_K token embeddings for w_1..w_K -> x_cur[1..K] (fp32, resident in VRAM)
  2. ── FORWARD SWEEP (l = 0 .. 39) ──
     For l = 0 to 39:
       • S_w prefetches expert layer (l + 1) from pinned host RAM into free slot (1 of 2 x 498 MiB slots) + flush (D27)
       • S_x copies x_cur[1..K] to host pinned boundary_store[l] (D2H) + flush (D27)
       • S_compute waits on E_load(l), then for k = 1 to K:
           layer_fwd(l, w_k, x_cur[k] -> x_next[k], records top-8 IDs & weights into Tape[l][k])
       • S_compute records E_done(l) so slot(l) can be reused; swap(x_cur, x_next) + flush (D27)
  3. ── HEAD & CHUNKED CROSS-ENTROPY LOSS ──
     For k = 1 to K, for each 256-position chunk c with active assistant tokens (w_t > 0):
       • h = RMS(x_cur[k][c]; output_norm); logits = h * W_lm^T (Q6_K, fp32 output)
       • loss_c, dlogits = SoftmaxCE(logits, labels, w_t / N_assistant_tokens_in_step)
       • dh = dlogits * W_lm (G5 dgrad); dx_cur[k][c] = RMS_bwd(dh)
     [Optional D12: Run blk.40 MTP forward & backward using its own 498 MiB expert layer]
  4. ── BACKWARD SWEEP (l = 39 .. 0) ──
     (Turnaround optimization: layer 39 is still resident in its VRAM slot from forward, saving 1 fetch -> 79 fetches/step)
     For l = 39 down to 0:
       • S_w prefetches expert layer (l - 1) into free slot + flush (D27)
       • S_x prefetches boundary_store[l - 1] (H2D) into x_in[1..K] + flush (D27)
       • S_compute waits on E_load(l) and E_boundary(l), then for k = 1 to K:
           layer_bwd(l, w_k, x_in[k], dx_cur[k] -> dx_next[k], Tape[l][k]):
             - Recomputes layer l forward from x_in[k] using the EXACT top-8 expert IDs from Tape[l][k] (zero router flip)
             - Backpropagates MoE -> post_attn_norm -> DeltaNet/Attention -> attn_norm
             - Accumulates LoRA dA, dB in fp32 (deterministic tree reduction O3)
       • swap(dx_cur, dx_next) + flush (D27)
  5. ── OPTIMIZER STEP & TELEMETRY ──
     • O2 global grad-norm & clip (1.0) -> O1 fused fp32 AdamW -> GQ Mode Q rebuild of W_eff
     • Single host synchronization per step; log step-wide metrics & canary check (every 50 steps)
```

**Why PCIe Traffic is $100\%$ Hidden (`F7`):**
Total expert traffic per optimizer step is fixed at $79\text{ fetches} = 35.8\text{ GiB}$ ($2 \times 18.164\text{ GiB} - 0.486\text{ GiB}$ turnaround saving), taking **$1.49\text{ s/step}$** at $26.1\text{ GB/s}$ regardless of $N$. With boundary traffic included ($3.0\text{ GiB}$ at $4.9\text{k}$ tokens, $10.0\text{ GiB}$ at $16\text{k}$, $20.0\text{ GiB}$ at $32\text{k}$) and the measured `P-H12` flushed overlap efficiency ($0.92$ all-three, $0.84$ bidirectional DMA), PCIe transfers are completely hidden behind GPU compute once a step holds $\ge 1,495 / 2,242 / 2,989$ tokens at $1,000 / 1,500 / 2,000\text{ tok/s}$ (packer enforces a minimum step size of $3,072$ tokens).

### 6.4 Optional Gated Training Modes (`Mode L / L-Z / L-C`, `Mode B / N`, `MTP`)

1. **Mode L — Locality-Aware Router Training (`D21`, `7.12`, `16.5 B1`):**
   * **Objective:** Train the 41 `F32` routers (`ffn_gate_inp`, $21,495,808$ params, $+0.32\text{ GiB}$ VRAM, $+0.21\%$ FLOPs) alongside the core LoRA from checkpoint `C0` so more top-8 selections land inside the engine's static VRAM hot set $R_l$, without inference-time clamping (`[sync 1]`).
   * **Composite Loss (`B1.2`):**
     $$L = L_{\text{CE}} + \lambda_{\text{trust}} \text{KL}(p \parallel p^0) + \lambda_{\text{res}} L_{\text{res}} + \lambda_{\text{zm}} L_{\text{zm}} + \lambda_{\text{KD}} L_{\text{KD}}$$
     where $p^0$ is the frozen base router distribution; $L_{\text{res}} = -\sum_l w_l \text{mean}_t \log\left(\sum_{e \in R_l} p_{l,t}(e) + \epsilon\right)$ on assistant tokens; $L_{\text{zm}} = \sum_{l \in T} w_l \text{mean}_t \text{softplus}\left(\frac{\delta - \mu_{l,t}}{\tau}\right)\tau$ is the **zero-miss hinge (`Mode L-Z`)** on the margin $\mu_{l,t} = z_{l,t}^{(8)}[R_l] - \max_{e \notin R_l} z_{l,t}(e)$; and $L_{\text{KD}}$ is truncated top-256 KL distillation against `C0`.
   * **Clamped-Forward Absorb Variant (`Mode L-C`, `B1.3`):** Stage A masks out-of-bank experts in forward so core LoRA adapts to resident experts; Stage B removes the mask and trains $W_r$ to select $R_l$ naturally.
   * **Pre-Declared Gates:** Ships only if `G-L1` ($\text{CE} \le 1.01 \times \text{CE}(C_0)$), `G-L2` ($\Delta h \ge +0.03$), `G-L3` (domain breadth $\ge C_0 - 0.01$), and `G-L4` (measured engine decode speedup over 3 launches) all pass.
2. **Mode B & Mode N — Dual-Branch Bimodal Numerics (`D24`, `16.5 B5`):**
   * Both CPU and GPU K-quant kernels accumulate integer dots in `int32` with `fp32` scales (`P-E13`), but CPU cold experts quantize activations in $256$-element `Q8_K` blocks while GPU hot experts use $32$-element `q8_1` blocks (`F14`).
   * **Mode B:** Quantizes activations in the trainer's forward pass using `q8_1` for experts in $R_l$ and `Q8_K` for experts outside $R_l$.
   * **Mode N:** Penalizes per-256-block activation outlier ratio $\rho = \max|x| / \text{rms}(x)$ above the 95th percentile $\rho_0$: $L_{\text{out}} = \lambda_{\text{out}} \text{mean}(\text{relu}(\rho - \rho_0)^2)$.
   * **Gate:** Built only if `P-E12` shows Pearson $r \ge 0.3$ between $\rho$ and CPU/GPU expert output divergence and the top $5\%$ of $\rho$ carry $\ge 50\%$ of excess KL.

---

## 7. Engine Speed-to-Ceiling Attribution and Trainer Performance Model

### 7.1 Engine Analytical Speed Ceiling and Verified Run Ledger

The analytical ceiling model (`2026-10-20`, updated per the `2026-10-22` author ruling) sums three terms across the 40 trunk MoE layers:
$$T_{\text{ceiling}} = T_{\text{spine}} + (1 - h) \frac{B_{\text{active}}}{BW_{\text{cold}}} + 40 \times t_{\text{handoff}}$$
With $T_{\text{spine}} = 7.73\text{ ms}$ ($7.4\text{ ms}$ open split discrepancy), $h = 0.8273$ ($7.5\text{ GiB}$ bank, $4,170$ experts), $B_{\text{active}} = 608\text{ MB/token}$, $BW_{\text{cold}} = 47.62\text{ GB/s}$ ($26.56\text{ GB/s}$ CPU + $21.06\text{ GB/s}$ GPU zero-copy; or $36.15\text{ GB/s}$ with today's $9.59\text{ GB/s}$ compute kernel), and $t_{\text{handoff}} = 6.17\text{ }\mu\text{s}$ ($0.247\text{ ms/token}$):
* **Ceiling with $26.2\text{ GB/s}$ zero-copy kernel:** $7.73 + 2.86 + 0.25 = \mathbf{10.87\text{ ms/token}}$ (**$92.0\text{ tok/s}$**).
* **Ceiling with today's $9.59\text{ GB/s}$ compute kernel:** $7.73 + 3.33 + 0.25 = \mathbf{11.34\text{ ms/token}}$ (**$88.2\text{ tok/s}$**; range **$88.2\text{--}90.9\text{ tok/s}$** under the $7.4\text{ ms}$ ruling).

**All Six Verified Engine Benchmark Runs (`MEASURED` on Patched `llama.cpp` Tier Build; `57.0 tok/s` Retired):**

| Recorded Run & Configuration | Client | Context | Decode (`tok/s`) | Token Time (`ms`) | Gap to Ceiling (`10.87–11.34 ms`) | Per-Layer Gap (`\mu s`) |
|---|---|---|---:|---:|---|---|
| **1. WSL2 Reference Cell A** ($7.0\text{ GiB}$ bank, bypass 32, `-ub 4096`, timer off) | Streaming (`9.2`) | Short (`ctx0`) | **$37.9$** | $26.39$ (p50 $26.33$) | $15.05\text{--}15.52\text{ ms}$ | $376\text{--}388\text{ }\mu\text{s}$ |
| **2. WSL2 Reference Cell A** ($7.0\text{ GiB}$ bank, bypass 32, `-ub 4096`, timer off) | Streaming (`9.2`) | $12\text{k}$ | **$42.1$** | $23.75$ (p50 $23.34$) | $12.41\text{--}12.88\text{ ms}$ | $310\text{--}322\text{ }\mu\text{s}$ |
| **3. Windows-Native Cell A (`r1 / r2`)** ($7.0\text{ GiB}$ bank, bypass 32, `-ub 4096`) | Streaming (`9.4`) | Short (`ctx0`) | **$47.3 / 46.8$** | $21.14 / 21.37$ (p50 $21.53$) | $9.80\text{--}10.50\text{ ms}$ | $245\text{--}262\text{ }\mu\text{s}$ |
| **4. Windows-Native Cell A (`r1 / r2`)** ($7.0\text{ GiB}$ bank, bypass 32, `-ub 4096`) | Streaming (`9.4`) | $12\text{k}$ | **$53.7 / 54.1$** | $18.62 / 18.48$ (p50 $18.28$) | $7.14\text{--}7.75\text{ ms}$ | $179\text{--}194\text{ }\mu\text{s}$ |
| **5. WSL2 Best Recorded Cell** ($7.5\text{ GiB}$ bank, bypass 32, `-ub 2048`) | Gauntlet (`8.2`) | Short / $4\text{k}$ | **$53.5 / 53.0$** | $18.69 / 18.87$ | $7.35\text{--}7.82\text{ ms}$ | $184\text{--}196\text{ }\mu\text{s}$ |
| **6. WSL2 Best Recorded Cell** ($7.5\text{ GiB}$ bank, bypass 32, `-ub 2048`) | Gauntlet (`8.2`) | $12\text{k}$ | **$49.0$** | $20.41$ | $9.07\text{--}9.54\text{ ms}$ | $227\text{--}239\text{ }\mu\text{s}$ |
| *WSL2 MTP Speculative (`draft 2/3`, $7.0\text{ GiB}$ bank, single/pooled runs)* | Gauntlet (`W/X`) | $12\text{k}$ | *$51.5\text{--}52.9$* | *$18.90\text{--}19.42$* | *$7.56\text{--}8.55\text{ ms}$* | *$189\text{--}214\text{ }\mu\text{s}$* |

### 7.2 Ten-Row Gap Attribution Register (`DERIVED` from `E0` Timeline)

On WSL2 Reference Cell A, internal timer-free token time is **$24.6\text{ ms}$** ($26.3\text{ ms}$ client gap minus $1.7\text{ ms}$ HTTP/sampler overhead), exceeding the $10.87\text{ ms}$ model by $+13.73\text{ ms}$:

| # | Gap Term | Measured / Derived Quantity | Governing Label | Proprietary Module that Resolves It |
|---|---|---|---|---|
| 1 | **CPU Cold Compute & Zero Overlap** | $10.1\text{ ms}$ (`ctx0`) / $6.6\text{ ms}$ ($12\text{k}$) vs. $2.86\text{ ms}$ model ($+7.24\text{ ms}$), executing **strictly serially ($0\%$ overlap)** with $6.6\text{ ms}$ GPU compute ($30.5\text{ ms}$ sum vs. $30.3\text{ ms}$ wall). | `DERIVED` (`D4`) | `bus/doorbell_handoff` + `bus/cpu_threadpool` (`E4`, `E6`) running hot GPU and cold CPU branches in parallel. |
| 2 | **Stream-Sync & Copy Handoff** | $\ge 5.0\text{ ms}$ (`ctx0`) / $\ge 4.6\text{ ms}$ ($12\text{k}$) timer-free ($5.1\text{ ms}$ in split) vs. $0.25\text{ ms}$ model ($+4.85\text{ ms}$). | `DERIVED` lower bound | `bus/doorbell_handoff` ($13.3\text{ }\mu\text{s}$ median / $18.7\text{ }\mu\text{s}$ p99 resident doorbell = $0.53\text{--}0.75\text{ ms/token}$). |
| 3 | **Copy-Back & Kernel Launch** | $2.8\text{ ms/token}$ ($2.3\text{ ms}$ copy-back + $0.5\text{ ms}$ launch) vs. $0.0\text{ ms}$ model. | `DERIVED` | `engine/decode_loop` static CUDA graphs (`E8`, `P-H9` $68\text{--}85\%$ launch saving) + pinned zero-copy merge. |
| 4 | **Win11 vs. WSL2 Delta** | Windows-native is $5.02\text{--}5.27\text{ ms/token}$ ($+24\text{--}28\%$) faster on Cell A. | `MEASURED` / `DERIVED` | Dual-target build (`[sync R1]`, `P-H11`, `E1`). |
| 5 | **Any-Miss Layer Share** | $63.9\%$ of layer-steps ($25.56 / 40$ layers) have $\ge 1$ cold miss (mean $2.16$ misses/missing layer); $36.1\%$ have $0$ misses (CPU bus idle). | `MEASURED` trace (`P-E1`) | `bank/hot_set_solver` zero-miss share objective $Z_l$ (`P-E2`) + Trainer Mode L (`D21`). |
| 6 | **82 Splits/Token (`41 CPU + 41 CUDA`)** | `E0` logged 82 scheduler splits/token. Whether split 41 is `blk.40` (MTP) is open (`D5`). | `MEASURED` (82) / `UNVERIFIED` (`blk.40`) | `engine/decode_loop` executes exactly 40 trunk layers unless MTP speculation is enabled. |
| 7 | **`Q6_K` $420\text{ B}$ Row Narrow Loads** | Zero-copy compute kernel reads at $9.59\text{ GB/s}$ ($13.31\text{ GB/s}$ with `Q4_K` wide loads) vs. $26.2\text{ GB/s}$ checksum kernel (`F12`). | `MEASURED` | `gpu/zerocopy.cu` $16\text{-byte}$ (`uint4`) shared-memory stream staging (`P-E10`). |
| 8 | **Live vs. Simulated Hit Rate** | Live $7.0\text{ GiB}$ bank hit rate is $0.7753\text{--}0.7793$ vs. $0.806$ simulated ($-0.03 \rightarrow +0.38\text{ ms/token}$). | `MEASURED` (`9.1`) | Re-profiling `PELIAS_HOT_SET` from live workload traces (`9.5`). |
| 9 | **Load-Mode Prefill Swing** | WSL2 `mmap` prefill at $12\text{k}$ is $863\text{--}929\text{ tok/s}$ vs. $1,667\text{--}1,680\text{ tok/s}$ with `--load-mode none` ($2,255\text{ tok/s}$ at `-ub 4096`). | `MEASURED` | `hal/pinned_pool` anonymous pinned host memory + `P-E11` `AnonHugePages` probe. |
| 10 | **Unaccounted `llama.cpp` Overhead** | All recorded runs used the patched `llama.cpp` tier build; no non-llama server run exists yet. | `UNVERIFIED` | Ground-up `pelias-engine` + `server/` (`D37`, `[sync R2]`). |

### 7.3 Trainer Roofline Model and Sensitivity Table (`1.4`, `13`)

With training work $F(T) = 20.39\text{ GFLOP/token}$ at $T = 8,192$ ($19.64\text{ GFLOP}$ at $T = 6,144$), `int8`-eligible share $52\%$ ($10.68\text{ GFLOP}$), non-GEMM overhead factor $f \in \{0.60, 0.75\}$, and GEMM efficiency $\epsilon \in [0.3, 0.6]$ ($\epsilon_8 = \epsilon - 0.1$):

| Scenario ($T = 8,192$) | $\epsilon = 0.3$ (`bf16` / `int8`) | $\epsilon = 0.4$ (`bf16` / `int8`) | $\epsilon = 0.5$ (`bf16` / `int8`) | $\epsilon = 0.6$ (`bf16` / `int8`) | $\epsilon = 1.0, f = 1.0$ Ceiling |
|---|---|---|---|---|---|
| **Stock Clocks ($58.3 / 233.2\text{ T-ops}$), $f = 0.75$** | $643$ / $956\text{ tok/s}$ | $858$ / $1,318\text{ tok/s}$ | $1,072$ / $1,675\text{ tok/s}$ | $1,287$ / $2,031\text{ tok/s}$ | $2,859$ / $4,708\text{ tok/s}$ |
| **Stock Clocks ($58.3 / 233.2\text{ T-ops}$), $f = 0.60$** | $515$ / $765\text{ tok/s}$ | $686$ / $1,054\text{ tok/s}$ | $858$ / $1,340\text{ tok/s}$ | $1,029$ / $1,625\text{ tok/s}$ | — |
| **Measured OC ($65.2 / 259.5\text{ T-ops}$), $f = 0.75$** | $721$ / $1,072\text{ tok/s}$ | $962$ / $1,477\text{ tok/s}$ | $1,202$ / $1,878\text{ tok/s}$ | $1,443$ / $2,277\text{ tok/s}$ | $3,206$ / $5,279\text{ tok/s}$ |

* **Declared Trainer Speed Targets (`DECISION`, re-based after Batch C probes `P-K1`–`P-K3`):**
  * **Minimum Viable:** **$600\text{ tok/s}$** ($2.4\times$ v0.2.1's $247.85\text{ tok/s}$ median $\rightarrow 83.7\text{ min}$ vs. $202.7\text{ min}$ for $3.01\text{ M}$ tokens).
  * **Target:** **$1,200\text{ tok/s}$** ($4.8\times$ v0.2.1 $\rightarrow 41.9\text{ min}$).
  * **Stretch:** **$2,000\text{ tok/s}$** ($8.1\times$ v0.2.1 $\rightarrow 25.1\text{ min}$).

---

## 8. Interdependencies, Invalidation Matrix, and Two-Way Engine-Trainer Contract

### 8.1 Directed Dependency Chain

1. **Byte-Budget & Export Chain:** Served GGUF byte map (`3.2.1`) + Engine bank size ($7.5\text{ GiB}$) + Measured VRAM ceiling ($10.7\text{ GiB}$, `P-H8`) + Engine non-bank peak (`P-E4`) $\rightarrow$ VRAM headroom $\Delta_{\max}$ $\rightarrow$ Mode Q default ($\Delta_{\max} = 0$, `D31`) & Byte-Budget Solver override (`D22`, `D32`) $\rightarrow$ `merged-core.gguf` + `layout_manifest.json`.
2. **Locality & Bank Chain:** Routing trace (`P-E1`: $121,160$ tokens, $h = 0.8273$, any-miss $= 0.639$) + Per-layer miss-cost regression (`P-E2`: $c_0$ vs. $c_1$) + Router logit margins (`P-E3`) $\rightarrow$ Hot-set objective ($h$ vs. $Z_l$) & Mode L / L-Z loss weights $\rightarrow$ Re-profiled `PELIAS_HOT_SET` (`P-E14`).
3. **Streaming & Roofline Chain:** Routed expert bytes ($18.164\text{ GiB} \times 2$) + Pinned PCIe bandwidth ($26.1\text{--}26.3\text{ GB/s}$, `P-H5`) + Flushed 3-stream overlap efficiency ($0.92$, `P-H12`, `D27`) $\rightarrow$ Fixed $1.49\text{ s/step}$ transfer time $\rightarrow$ Compute-hiding token threshold $N_{\text{hide}} \in [1,495, 2,989]$ $\rightarrow$ Layer-major $K$-window scheduler (`D5`).
4. **Accuracy & Parity Chain:** Stage A conversion (`P-X6` vs. `P-X0`) + Bit-exact quantizers (`P-E6`) + Float noise floor $\sigma_{\text{float}}$ (`P-Q4a/b`) + Bimodal outlier audit (`P-E12`, `P-E13`) $\rightarrow$ Three-tier KL gates (`K1`, `K2`, `K3`).

### 8.2 Complete Invalidation Matrix

| If This Input Changes... | ...Then Re-Compute / Re-Run | Reason |
|---|---|---|
| **1. Engine bank size ($7.5\text{ GiB}$) or non-bank VRAM (`P-E4`)** | $\Delta_{\max}$ (`16.3.1`), `D22` override solver, Mode L exchange rate, and `PELIAS_HOT_SET` capacity ($4,170$ experts). | Moves both spare VRAM under $10.7\text{ GiB}$ and resident expert count. |
| **2. Engine prefill micro-batch (`-ub`)** | Engine peak VRAM ($10.40\text{--}10.44$ default vs. $10.67\text{--}10.69\text{ GiB}$ at `-ub 2048` vs. `P-E4` at `-ub 4096`) and prefill $\text{tok/s}$. | Scratch buffers scale with `-ub`; trainer is decoupled by `D31`/`D34`. |
| **3. Pinned oracle commit (`2ca15f540`)** | Layer rule (`F2`), `Q6_K` layout (`F12`), accumulator audit (`P-E13`), and quantizer parity (`P-E6`). | Reference specifications are read from commit `2ca15f540`. |
| **4. `P-E2` per-layer miss cost ($c_0$ vs. $c_1$)** | `bank/hot_set_solver` objective (byte-hit $h$ vs. zero-miss share $Z_l$) and Mode L layer weights $c_l$. | If fixed trip cost $c_0 \gg c_1$, eliminating 1-miss layers beats raising mean $h$. |
| **5. `P-E10` wide-load zero-copy rate ($9.59 \rightarrow 26.2\text{ GB/s}$)** | Ceiling cold-read term ($3.33\text{ ms} \rightarrow 2.86\text{ ms}$), `E12` third-branch gate ($\ge 15\text{ GB/s}$), and `Q6_K` $420 \rightarrow 432\text{ B}$ row padding. | Decides PCIe compute read rate and whether `repack` pads `Q6_K` rows. |
| **6. Measured PCIe stream bandwidth ($26.1\text{--}26.3\text{ GB/s}$)** | $1.49\text{ s/step}$ stream time and $N_{\text{hide}}$ token thresholds (`5.5`, `13.4`). | $t_{\text{stream}} = 2 B_{\text{experts}} / BW_{\text{H2D}}$. |
| **7. GPU clock settings (OC vs. stock)** | `P-H3` sustained tensor peaks ($65.2 / 259.5$ vs. $58.3 / 233.2\text{ T-ops}$) and roofline tables (`7.3`). | Tensor Core throughput scales linearly with SM clock. |
| **8. VRAM ceiling ($10.7\text{ GiB}$) or disturbance margin** | `hal/vram_governor` limit, arena capacities (`5.2`), and `P-X5` engine acceptance gate. | Every arena is budgeted against the governor's ceiling. |
| **9. GPU driver (`617.42`) or submission pattern** | `P-H12` overlap efficiency ($0.92$ / $0.84$), `P-H9` flush cost, and `D27` flush rule. | Cross-stream overlap on WDDM depends on queue flushing. |
| **10. Engine adds `Q8_0` weight kernels (`P-E5`)** | Allowed target types in the `D22`/`D32` override solver. | Engine must have kernels for every exported `ggml_type`. |
| **11. Target OS selection (`WSL2` vs. `Win11`)** | Doorbell floor (`03`), small-copy latency (`P-H5`), flush cost (`P-H9`), and default platform (`P-H11`). | Host-side driver costs differ by $2\text{--}12\times$ between WSL2 and Win11. |
| **12. Newly trained GGUF weights** | Model key (`U4`), $\sigma_{\text{float}}$ (`P-Q4b`), routing trace (`P-E1`), and `PELIAS_HOT_SET` profile (`R18`). | New weights shift both logit rounding noise and router frequencies. |
| **13. Proprietary runtime replaces patched `llama.cpp` (`D37`)** | All decode speed claims (`7.1`), gap attribution (`7.2`), server catalogue (`5.7`), and `K3` gate subject. | Isolates Pelias performance from inherited `llama.cpp` overhead. |

### 8.3 The Two-Way Engine $\leftrightarrow$ Trainer Milestone Contract (`[sync 9]`)

| Engine Milestone | What `pelias-engine` Releases to `pelias-trainer` | What `pelias-trainer` Releases / What `pelias-engine` Verifies |
|---|---|---|
| **`E0` (Per-Layer Timers)** | `P-E2` ($c_0, c_1$ per miss & layer type) $\rightarrow$ layer weights $c_l$ in Mode L (`B1.2`). | — |
| **`E1 / E11` (Platform Floors)** | Side-by-side WSL2 and Windows-native floor table $\rightarrow$ platform default (`P-H11`). | — |
| **`E3` (Reader & Oracle)** | Reproduction of stock KL floor (`0.017353` batch 16 / `0.018286` batch 1024, `P-Q4a`). | `merged-core.gguf` and `layout_manifest.json` parse identically in `io/gguf_reader`; `K3` holds. |
| **`E5` (Bank & Wide-Load)** | `P-E10` (`uint4` staging) & `P-E11` ($2\text{ MiB}$ pages) $\rightarrow$ manifest flags & optional pack spec. | `bank/hot_set_io` output matches `export_hot_set.py` byte-for-byte (`P-E14`) and loads into `gpu/bank`. |
| **`E7` (Full 40-Layer Spine)** | `P-E5` supported weight type matrix (`Q4_K`, `Q6_K`, `F32`; optional `Q8_0`) & `P-E13` accumulator audit. | Trainer exports only supported core types (`D31` default = served types). |
| **`E8 / E9` (Decode & Planner)** | True native engine non-bank VRAM footprint (`P-E4`) at shipped prefill micro-batch. | Exported GGUF peaks $\le 10.7\text{ GiB}$ at the $7.5\text{ GiB}$ bank with $\le 2\%$ decode regression (`P-X5`). |
| **`E12` (Gated Extras)** | MTP draft acceptance rate (`P-T8`) and live bank hit rate. | Mode L candidates and re-profiled hot sets evaluated by `G-L4` and `G-Z4`. |

---

## 9. Master Order of Operations

### 9.1 Standing Governing Order of Work (`AUTHOR-STATED`)
1. **Phase I:** Trainer comparison and Wave 0 / Batch C measurement scripts.
2. **Phase II:** Hard stop on engine coding until `pelias-trainer` reaches "top tier" (all Wave 0 probes run, switches signed off, stages `T0`–`T14` verified).
3. **Phase III:** Make `pelias-trainer` universal (`U1`–`U11` modules and Lithic-to-Voxel UI).
4. **Phase IV:** `pelias-engine` and native `server/` execution (`E0`–`E13`), **HELD** until lifted by the author (`O13`). *(Note: All shared Layer 0 and Layer 1 modules—`os`, `hal`, `plan`, `quant`, `io`, `tok`, `kern` spine, `verify`, `bank`—are built during Trainer stages `T1`–`T12` and reused directly by the engine.)*
5. **Phase V:** Unify into a single cohesive release where `pelias-train` and `pelias-engine` run on the shared substrate and core.

### 9.2 Unified 15-Step Ground-Up Execution Ladder (`7.1`)

| Step | What Is Built or Decided | Trainer Stage | Engine Phase | Gate That Closes It | Current Status |
|---:|---|---|---|---|---|
| **1** | Measure hardware floors; run Batch C kernel probes; sign off switch summary. | `T0` | `E0`, `E1` | Wave 0 hardware set done (`2.2`); Batch C (`P-K1`–`P-K3`, `P-Q10`) & `P-M4` run; author signs T0 switch summary. | Wave 0 hardware PASS on WSL2 & Win11; Batch C source next. |
| **2** | Foundations: `os/` (`file_map`, `threads`, `clock`), `hal/` (`device`, `stream`, `event`, `arena`, `vram_governor`, `pinned_pool`), `io/` (`gguf_reader`, `sha256`, `json`), `plan/`. | `T1` | `E2`, `E3` (reader) | Lists all 753 tensors with types/offsets; refuses any plan $> 10.7\text{ GiB}$ (`P-M1`); rejects corrupt GGUFs (`PG2`); clean build on WSL2 & Win11; `BC1`–`BC4`, `BC6` pass. | Host-only framework builds (`4/4` ctest); full implementation next. |
| **3** | Block formats & CPU quantizers (`quant/`), `io/safetensors`, `io/gguf_writer`, and Stage A Ingest (`ingest/`: `bf16` shards $\rightarrow$ `pelias_base_0step.gguf`). | `T1b` | `E3`, `E4` | `P-E6` 100% bit-exact vs. oracle; `P-X6` Stage A output matches `P-X0` reference tensor-for-tensor; `P-X7` benchmarked. | `bf16` shard headers measured (`3.2.2`); `ingest` skeleton ready. |
| **4** | Native byte-level BPE tokenizer, Jinja-subset chat-template renderer (`tok/`), and dataset packer/audits (`data/`). | `T2` | `E10` (reuse) | `P-D1` (0 token mismatches), `P-D2` (0 round-trip failures), `P-D5` (0 mask mismatches); audits `A-LEN` to `A-FAMILY` pass. | Census complete (`5.6`); implementation in `T2`. |
| **5** | Quantized GEMM family (`kern/gemm_gguf/`: `G8` dequant, `G9` `q8_1`, `GQ` Mode Q requantizer, `G1`/`G3` `bf16`, `G2`/`G4` `int8`, `G5`/`G6` transposed `dgrad`, `G7` LoRA). | `T3` | `E4`, `E5` | Rung-1 tolerances (`10.2`) & `compute-sanitizer` clean; `P-Q2`, `P-Q8`, `P-Q11`, `P-K4`, `P-K5`, `P-K12` pass. | Not started. |
| **6** | Elementwise, router, and chunked loss kernels (`kern/elementwise/`, `kern/route/`, `kern/loss/`). | `T4` | `E6`, `E7` | Rung-1 tolerances; `P-K6` router identical top-8 sets (`P-Q12`); `P-K11` chunked CE loss rel. error $\le 10^{-6}$. | Not started. |
| **7** | Full GQA Attention kernels (`kern/attn/`: `A1` IMRoPE, `A2` fused fwd, `A3` fused bwd at head-dim 256, `A4` blocked ref). | `T5` | `E7` | Rung-1 tolerances; `P-S4` RoPE convention match; `P-K7` 0 register spills; `P-Q7` internal precision locked. | Not started. |
| **8** | Gated DeltaNet kernels (`kern/deltanet/`: `D1` gates, `D2` conv1d, `D3` chunk prep, `D4` state sweep, `D5` out, `D6` chunk bwd, `D7` seq ref). | `T6` | `E7`, `E9` | Rung-1 tolerances; `P-S7` chunked fwd matches `fp32` sequential oracle; `P-K8`–`P-K10` bwd $\le 10^{-3}$ vs. `fp64`. | Not started. |
| **9** | Single layer & 40-layer forward parity vs. served truth; backward sweep with `Tape`, LoRA, AdamW, and Mode Q step. | `T7`–`T9` | `E6`, `E7` | Rung 2 per-layer parity; Rung 3 full forward $\text{KL} \le 1.0 \times \sigma_{\text{float}}$ (`P-Q5`, `P-Q6`); Rung 4 finite-diff (`P-T1`) & Oracle C cosine $\ge 0.99999$ (`P-T2`); `P-T5` $K$-invariance; `P-D3` packing equivalence. | Not started. |
| **10** | 3-Stream Layer-Major Scheduler (`stream/`, `sched/`), Training Loop, 4-Registry Checkpoints (`train/`), Byte-Splice Exporter (`export/`), and `bank/hot_set_io`. | `T10`–`T12` | `E3` (`K3`), `E5` (`P-E14`) | Rung 5 ($0$ allocations in loop `P-M5`, 79 fetches, stall $\le 5\%$ `P-T7`); Rung 6 (`P-T3` overfit, `P-T4` bitwise determinism, `P-T9` v0.2 parity, `P-T11` bitwise resume); Rung 7 (`P-X1` hash proof, `P-X2` `K2` KL gate, `P-E14` hot-set parity). | Not started. |
| **11** | Trainer Performance Engineering, Optional Modes (MTP `D12`, Mode L `D21`, Mode B/N `D24`), Ablations `T-A1`–`T-A13`, and Universal-Use / UI (`U1`–`U11`). | `T13`, `T14`, Universal | — | Speed targets (`600`/`1,200`/`2,000 tok/s`) met or explained (`P-T12`); optional modes pass pre-declared gates; universal intake/warnings/UI verified. | UI header exists; rest post-`T12`. |
| **12** | **Engine Cold/Hot Branches & Doorbell:** `bus/cpu_threadpool`, `cpu/` AVX2 kernels, `gpu/` bank & `uint4` zero-copy kernels, `bus/doorbell_handoff`, 1-layer end-to-end. | — | `E4`–`E6` | `E4`: CPU throughput $\ge 27\text{ GB/s}$ effective; `E5`: zero-copy compute reaches $26.2\text{ GB/s}$ (`P-E10`); `E6`: doorbell p99 $\le 18.7\text{ }\mu\text{s}$ and layer output within oracle noise floor. | **HELD (`O13`)**. |
| **13** | **Engine Full Spine, Decode Loop, and Long-Context Prefill:** `engine/` (`kvcache`, `state`, `decode_loop`, `prefill`, `sampler`). | — | `E7`–`E9` | `E7`: `K3` $\text{KL} \le \sigma_{\text{float}}$; `E8`: decode beats tier build, clears Gate `G3` ($\ge 55.0\text{ tok/s}$ at $12\text{k}$), approaches ceiling; `E9`: prefill $\ge 1,670\text{ tok/s}$ at $12\text{k}$ (target $2,238$), context to $131,072$, VRAM $\le 10.7\text{ GiB}$. | **HELD (`O13`)**. |
| **14** | **Native HTTP/SSE Server & Capability Probe:** `os/net`, `server/` (`http`, `tools`, `api_openai`), `llama-server` catalogue (`O8`), `E11` startup probe, `E12` extras, `E13` generality. | `serve/` | `E10`–`E13` | Drop-in replacement for launcher; persona not-pass $\le 12/111$, needles $12/12$ to $120\text{k}$, VS 2026 Copilot session works, client replay (`PG8`) passes on WSL2 & Win11; second MoE runs via `arch_moe`. | **HELD (`O13`)**. |
| **15** | **Unification:** Single cohesive release (`pelias`), shared probe constants, in-process or direct hand-off from `pelias-train` to `pelias-engine`. | Unify | Unify | Both applications pass all verification rungs and `PG1`–`PG10` with `tests/oracle/` absent (`BC5`). | Final phase. |

---

## 10. Verification Ladders, Oracles, Kernel Tolerances, and Three-Tier KL Gates

### 10.1 The Three Oracles and Conflict Resolution Rules (`10.1`, `10.6`)

1. **Oracle A — External Pinned `llama.cpp` (`2ca15f540` / `b11399` in `tests/oracle/` and `tools/oracle_llama/`):**
   * **Role:** The served truth for forward-pass semantics, per-layer activation dumps (`eval-callback` on the clean `P-X0` baseline GGUF, `D36`), and logit KL divergence. Never linked into product binaries.
2. **Oracle B — Proprietary `fp64` CPU Reference (`tests/ref/`):**
   * **Role:** Unambiguous double-precision CPU implementation of every forward and backward operation (sequential DeltaNet, materialized attention, exact block dequantization). Checked first against central finite differences in `fp64` ($\frac{f(\theta + \epsilon u) - f(\theta - \epsilon u)}{2\epsilon}$ matching analytic directional derivatives to $\le 10^{-8}$ relative error, `P-T1`).
3. **Oracle C — Test-Only PyTorch `fp32` Autograd Slice (`tools/oracle_torch/`):**
   * **Role:** Independent 4-layer slice with inverse V-head and norm $+1$ permutations to cross-check backward LoRA gradients (`P-T2`).
4. **Who Wins a Disagreement:**
   * *Forward conventions (norm $+1$, RoPE pairing, conv tap order, top-k ties):* **Oracle A wins.** Disagreements between legal float configurations of Oracle A define the float noise floor $\sigma_{\text{float}}$ (`P-Q4`), not a bug.
   * *Backward gradients:* **Oracle B (verified by `fp64` finite differences) wins.**
   * *Fast CUDA kernels:* Judged against **Oracle B on identical inputs** so kernel precision is never confounded with model float spread.

### 10.2 Rung-1 Kernel Tolerance Targets (`6.11`)

Every kernel in `src/kern/`, `src/cpu/`, and `src/gpu/` must pass these tolerances against Oracle B across normal shapes, tile-boundary $\pm 1$ shapes, $M_e \in \{0, 1, 7, 8, 63, 64, 65\}$, all-tokens-to-
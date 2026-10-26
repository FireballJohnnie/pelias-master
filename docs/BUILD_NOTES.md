# Build notes

**Status 2026-10-26 (evening): step 2 (Foundations) stages A and B built and tested host-only; the stage C gate (both platforms, the real model file) is the author's run.** This file grows step by step as the build proceeds (Masterwork Section 15). Each step adds: what was built, the exact commands used on each platform, the tool versions, what the author ran, and where the results are.

## Toolchains on record (the reference machine)

| Platform | Compiler | CUDA | CMake | Ninja | Git | Label |
|---|---|---|---|---|---|---|
| WSL2 (Ubuntu, kernel 6.18.54.1) | GCC 15.2.0 (also Clang 21) | nvcc 13.0.88 | 4.2.3 | 1.13.2 | 2.53.0 | MEASURED (census 2026-10-23) |
| Windows 11 Pro (build 26300) | MSVC 19.44.35229.0, Visual Studio 18 Professional, toolset 14.44.35207 | nvcc 13.4.92 | 4.4.3 | 1.13.2 | 2.55.0.windows.5 | MEASURED (2026-10-23 01:09; configure log) |
| Driver | 617.42 (CUDA 13.4 user mode) | | | | | MEASURED |
| Avoid | CUDA toolkit 13.2 | | | | | AUTHOR-STATED |

Missing or unconfirmed for a build (from `PELIAS_SYSTEM_MASTER.md` Section 6): Python 3 on Windows (only for two repository checks); the Windows SDK version; `compute-sanitizer` and Nsight on Windows; `ncu` on WSL2; free disk space on both sides.

## Rules for every build step

- Host-only builds and CPU tests run in Claude's sandbox; every GPU build and run is done by the author on WSL2 and on Windows.
- Windows builds run from the x64 Native Tools prompt (Visual Studio 18).
- Warnings are errors in release (`-Wall -Wextra -Werror` / `/W4 /WX`), gate PG9.
- The boundary checks BC1 to BC7 run on every build.

## Steps

### Step 2, Foundations (2026-10-26, stages A and B)

**What was built:** `include/pelias/` (status, log, timer, version), `src/os/` (file_map, clock, dynlib, status, log), `src/hal/` (device, nvml loaded at run time, vram_governor, `backend_cuda/cuda_device.cpp` with the CUDA runtime, `backend_cpu/no_cuda_device.cpp` for host-only builds), `src/quant/block_types` (every GGUF type's block size, READ-SRC), `src/io/` (sha256, json, gguf_reader, safetensors), `src/plan/` (probe_constants, memory_plan), `src/cli/main.cc` (`pelias version | device | inspect | key | plan`), CMake (`CMakeLists.txt`, `cmake/`, `CMakePresets.json`: `host`, `wsl`, `windows`), the test harness and tests (`tests/`), `tools/check_module_deps.py` (BC1, BC3, BC4, BC6).

**Sandbox results (host-only, no GPU):** GCC 13.3 release with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror`: clean. Clang 18 release: clean. GCC debug with AddressSanitizer and UndefinedBehaviorSanitizer: clean; all 13 unit tests pass under it; the mutation gate (1,500 and 3,000 random mutations of a GGUF and a safetensors file, seeds 7, 99, 3) refuses every damaged file it detects with a Status and never accepts a tensor outside the file, with no sanitizer report. The CUDA backend file is syntax-checked with clang against the CUDA 13 headers (it needs nvcc and a GPU to run). The reader was checked against the real model file's header (the first 16 MiB of the Unsloth Q4_K_M file, read on the author's machine at 17:54): 733 tensors, 54 keys, data at byte 10,990,048, every tensor's name, shape, type and offset equal to an independent parse, and data start plus padded data = 22,134,528,992 = the file's size (P-M4).

**Decisions recorded at this step:** `plan -> io` added to the module graph of 14.3 (the planner reads the tensor table; io does not depend on plan, so the graph stays acyclic). The GGUF reader's `require_data = false` mode (and `inspect --header-only`) reads a header whose data is cut off. The working-set term of the plan is the masterwork 8.8 ESTIMATE (2.50 GiB) until P-M2 measures it, and the plan card says so.

**Observation (DERIVED from the tensor table), corrected 2026-10-26 19:0x:** on the Unsloth file the always-resident part is 2.38 GiB (Q8_0 spine 1.49 GiB, Q8_0 embedding 515 MiB, Q6_K head 398 MiB); on the served file it is 1.56 GiB (embedding 272.8 MiB, head 397.9 MiB, spine 905.0 MiB, MTP head 23.4 MiB, MEASURED by `pelias plan` 19:03). The planner first used the masterwork 8.8 working set of 2.50 GiB, which is the TRAINING figure (8,192-token window, up to 32 k tokens per step); serving now uses 1.00 GiB, a DERIVED upper bound from the tier build's measured 7.5 GiB-bank peak (10.69 GiB whole card, minus Windows' 914 MiB, the bank and the weights on the card; the steps are in `src/plan/probe_constants.h`). With it, at a 10.72 GiB ceiling: served file, 7.5 GiB bank fits (peak 10.06 GiB), largest bank 8.16 GiB (DERIVED: 10.72 − 1.56 − 1.00); Unsloth file, 7.5 GiB refused by 163 MiB, largest 7.34 GiB. P-E4 (step 10) measures the engine's own peak and replaces the figure.

**Commands (sandbox):**

```
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release -DPELIAS_CUDA=OFF && cmake --build build/host && ctest --test-dir build/host
python3 tools/check_module_deps.py && python3 docs/scaffold/verify_inventory.py
```

**Stage C (the author, both platforms):** see the RUNBOOK entry of 2026-10-26 and `README.md`; the gate is: clean configure and build with `-DPELIAS_CUDA=ON` on WSL2 (GCC, nvcc 13.0) and Windows (MSVC x64 tools, nvcc 13.4) with warnings as errors; `pelias inspect` on the Unsloth file lists every tensor (733 in that file; the served file of 17.3 has 753) with its type and the count equals the header's; `pelias plan` refuses a bank above the ceiling (exit 3) and proposes the largest that fits; `pelias_mutate` passes; `pelias device` reports the measured ceiling.

# Citations (system)

**Status: draft, private repository.** Sources used by `docs/PELIAS_SYSTEM_MASTERWORK.md`, with their role. No outside code is part of the product (Masterwork Section 3). The trainer's and the engine's own citation files list the sources of their designs and are not repeated here.

| Role | Meaning |
|---|---|
| **Specification read** | Read to re-express a format or behaviour in original code; nothing copied |
| **Published value** | A vendor or reference number used as a starting default until the program measures the real value |
| **Optional outside process** | Started separately, never linked, never required |
| **Design reference** | An idea or interface looked at; no code used |

## Specification read

* **llama.cpp / ggml**, ggml-org. `ggml/src/ggml-common.h` at commit `2ca15f5404760548c39e7b92bd43116a09414a1a` (tag b11399): block layouts and sizes of every GGUF type (read 2026-10-26). MIT licence. Also the external, test-only oracle (`tests/oracle/`).
* **GGUF file format**, ggml-org. <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.

## Published values (fetched 2026-10-26)

* NVIDIA, *CUDA GPU Compute Capability*. <https://developer.nvidia.com/cuda-gpus>.
* NVIDIA, *GeForce graphics cards comparison*. <https://www.nvidia.com/en-zz/geforce/graphics-cards/compare/>.
* AMD, *System requirements (Windows), HIP SDK 7.2*. <https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html>.
* AMD, *System requirements (Linux), ROCm*. <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>.
* *List of CPUs with AVX-512 instructions* (secondary source). <https://www.eatyourbytes.com/list-of-cpus-with-avx512/>.
* Tom's Hardware, *Intel announces the Arc B580 and Arc B570*. <https://tomshardware.com/pc-components/gpus/intel-announces-the-arc-b580-and-arc-b570-gpus>.
* *GGUF quantization guide* (secondary source, perplexity guidance). <https://tonisagrista.com/blog/2026/quantization>.
* NVIDIA, *CUDA on WSL User Guide*. <https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html>.

## Optional outside process

* **FlareSolverr** 3.5.2, Diego Heras and contributors. MIT licence. <https://github.com/FlareSolverr/FlareSolverr>. Runs as its own process on 127.0.0.1:8191 when the user starts it (Masterwork 12.3).

## Design reference

* **SearXNG** (metasearch). AGPL-3.0: used for the idea only; no code is copied (Masterwork 12.2).
* **Dear ImGui** (MIT): the 2026-10-24 interface header was written for it; the product's interface is now a web page and does not use it.

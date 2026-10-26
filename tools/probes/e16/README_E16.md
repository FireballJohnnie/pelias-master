# P-E16: the CUDA doorbell with its flag in VRAM

Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.

Written 2026-10-26 for build step 1 (masterwork 15.2, 21.2 hypothesis H-G1). P-G0 measured a CPU-to-GPU-to-CPU hand-off of 0.8 to 1.4 us on Vulkan and Direct3D 12 with the flag in CPU-visible VRAM, against 13.34 us for the CUDA doorbell with its flag in system RAM (microbenchmark 03). P-E16 asks whether CUDA gets the same by moving its flag into VRAM.

How: one resident GPU thread and the CPU ring each other 20,000 times per placement (`--rounds`), 1,000 rings per kernel launch (`--batch`, keeps every launch far below the display-timeout limit). The CPU writes ring n to the request word; the GPU thread waits for it (volatile loads), writes n to the reply word (volatile store between system-scope fences); the CPU waits for that. Each word sits either in pinned system RAM mapped into the GPU, or in CPU-visible VRAM: a Vulkan `DEVICE_LOCAL | HOST_VISIBLE` block (reached by the CPU through the PCIe BAR, write-combined, `sfence` after each write) exported to CUDA by external-memory interop. Four placements plus launch-and-synchronize as the reference. A ring not answered within 250 ms is recorded as "does not complete". The first ring of each launch is left out of the timing.

Platforms: Windows native runs all placements. Under WSL2 the Vulkan loader finds no Vulkan device with the CUDA GPU's identity (no NVIDIA Vulkan driver there), so it records the system-RAM placement and the reference only. Native Linux would use opaque file descriptors (same code). ReBAR state is inferred from the size of the CPU-visible VRAM heap (about 256 MiB without ReBAR).

Files: `e16_kernels.cu` / `.h` (GPU side), `vk_extmem.h` (Vulkan block exported to CUDA), `main_e16.cpp`; it builds on the Wave 0 framework (`cudautil.h`, `hostlib.h`, `nvml_dyn.h`, `kernels.h`, linked with `kernels.obj` / `kernels.o`) and the Vulkan-Headers vendored with gfx_probe (`../gfx_probe/third/Vulkan-Headers/include` in this repository). The framework files are in `../batch_c/`; copy them next to these files, or add `-I../batch_c`, and build `kernels.cu` from there first.

Revisions: r1's GPU side used inline-PTX acquire/release; it and every other file failed to build in a 32-bit (x86) Developer PowerShell window (`cudafe++` 0xC0000005), which was the real cause; r2 uses volatile loads and system-scope fences; r3 flushes the launch with `cudaStreamQuery`; r4 adds four ways for the GPU to post its reply (open item O15): fence-store-fence, store-fence, store only, acquire load with release store (without it Windows held the launch and no ring was answered: `data/probe_results/P-E16/windows_r2_no_flush`).

Windows (a 64-bit Developer Command Prompt: `where cl` must show `Hostx64\x64`; run `vcvars64.bat` otherwise; in the probe folder):

```
set PATH=%CUDA_PATH%\bin\x64;%PATH% && nvcc -std=c++20 -O2 -arch=sm_89 -c e16_kernels.cu -o build\e16_kernels.obj && nvcc -std=c++20 -O2 -c main_e16.cpp -o build\main_e16.obj -I ..\gfx_probe\third\Vulkan-Headers\include -Xcompiler "/EHsc /D_CRT_SECURE_NO_WARNINGS" && nvcc -arch=sm_89 build\main_e16.obj build\e16_kernels.obj build\kernels.obj -o build\pelias_e16.exe
build\pelias_e16.exe --out DIR
```

WSL2 / Linux:

```
export PATH=/usr/local/cuda/bin:$PATH && nvcc -std=c++20 -O2 -arch=sm_89 -c e16_kernels.cu -o build_wsl/e16_kernels.o && g++ -std=c++20 -O2 -I/usr/local/cuda/include -I../gfx_probe/third/Vulkan-Headers/include -c main_e16.cpp -o build_wsl/main_e16.o && nvcc -arch=sm_89 build_wsl/main_e16.o build_wsl/e16_kernels.o build/kernels.o -ldl -lpthread -o build_wsl/pelias_e16
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./build_wsl/pelias_e16 --out DIR
```

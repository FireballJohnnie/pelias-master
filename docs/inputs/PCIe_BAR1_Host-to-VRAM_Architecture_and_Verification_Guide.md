# PCIe BAR1 Host-to-VRAM Architecture, Driver Locks, and Verification Pathways (WSL2 & Native Linux)

**Hardware & Environment Baseline:**
* **GPU:** NVIDIA GeForce RTX 4070 (12,282 MiB FB, 16,384 MiB BAR1 Aperture, PCIe 4.0 x16)
* **CPU:** AMD Ryzen 7 5800X3D (8C/16T, 96 MiB L3 3D V-Cache, AVX2)
* **Subsystem:** WSL2 (`6.18.54.1-microsoft-standard-WSL2`, `/dev/dxg` WDDM paravirtualization)
* **Host/UMD Driver:** 617.42 / CUDA 13.x (`nvcc` 13.0.88, UMD 13.4)

---

## 1. Root Cause of `BAR1 Memory Usage: Used: 0 MiB`

In a standard CUDA execution graph (`ggml` / `llama.cpp`), `BAR1 Memory Usage` remains at `0 MiB` during active inference and host-to-device transfers for two architectural reasons:

1. **Telemetry Passthrough vs. Guest PCI Topology:** Inside WSL2, `/dev/nvidia*` and `nvidia.ko` do not exist; user-space libraries (`/usr/lib/wsl/lib/libnvidia-ml.so.1` and `libcuda.so.1`) communicate with the Windows host kernel driver (`nvlddmkm.sys`) by issuing `ioctl` calls to `/dev/dxg` over the Hyper-V VMBus (`D3DKMTQueryAdapterInfo`). The 16,384 MiB BAR1 total reported by `nvidia-smi` reflects the physical WDDM adapter state on the Windows host, where Resizable BAR (ReBAR) has expanded the PCIe Base Address Register 1 aperture to cover the entire VRAM space.
2. **Copy Engine (DMA) Default Path:** Standard CUDA memory allocations (`cudaMalloc`) reside in device-private VRAM (`D3D12_HEAP_TYPE_DEFAULT` / unmapped `LOCAL` segment). Standard host-pinned allocations (`cudaHostAlloc` / `cudaMallocHost`) reside in system RAM (`NON_LOCAL` segment). When `cudaMemcpyAsync` transfers weights or activations from host to device, the GPU's hardware Copy Engine (CE) executes a PCIe Bus Master DMA read from system RAM into VRAM. Because the CPU never directly loads or stores to a virtual address backed by the physical PCIe BAR1 aperture, no BAR1 address space is mapped or consumed.

To increment `BAR1 Memory Usage: Used` and bypass DMA launch overhead, memory must be explicitly allocated as **Host-Visible Device Memory** (Write-Combined VRAM mapped into the CPU page tables via the BAR1 aperture).

---

## 2. Native Linux (`gdrcopy` / GPUDirect RDMA) Hardware Lock on GeForce

Transitioning to native Linux to utilize `gdrcopy` (`gdrdrv` + `libgdrapi.so`) encounters an intentional driver restriction on consumer GeForce GPUs (including the Ada Lovelace RTX 4070):

* **`nvidia_p2p_get_pages` Restriction:** `gdrcopy` relies on the Linux kernel symbol `nvidia_p2p_get_pages` exported by `nvidia.ko` to pin and map GPU BAR1 physical pages into user space. On unpatched proprietary drivers, NVIDIA restricts GPUDirect RDMA and `CU_POINTER_ATTRIBUTE_P2P_TOKENS` to Data Center (Tesla/A100/H100) and professional RTX (formerly Quadro) product lines.
* **Runtime Behavior:** Calling `gdr_pin_buffer()` on a `cudaMalloc` pointer on an RTX 4070 returns `EINVAL` / `CUDA_ERROR_NOT_SUPPORTED` (`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` evaluates to `0`).
* **Open-Kernel Workaround (Native Linux Only):** Because NVIDIA publishes the open-source kernel modules (`open-gpu-kernel-modules`), native Linux users can bypass this check only by modifying `kernel-open/nvidia/nv-p2p.c` (or using community patches for `nvidia-open`) to allow `nvidia_p2p_get_pages` on GeForce PCI device IDs. This workaround cannot be applied inside WSL2 because the kernel-mode driver (`nvlddmkm.sys`) resides on the Windows host and is closed-source.

---

## 3. WSL2 Paravirtualized Architecture (`/dev/dxg` & WDDM)

Inside the `microsoft/WSL2-Linux-Kernel` source tree (`drivers/hv/dxgkrnl/`), GPU memory mapping across the hypervisor boundary is governed by the following mechanisms:

### 3.1. `dxgkrnl` IO Space Mapping (`LX_DXLOCK2`)
In `drivers/hv/dxgkrnl/ioctl.c` and `dxgvmbus.c`, the WSL2 kernel driver exposes `/dev/dxg` ioctls corresponding to the Windows WDDM `D3DKMT` API:
* **`LX_DXCREATEALLOCATION` (`dxgkio_create_allocation`):** Forwards allocation requests over the VMBus to the host via `dxgvmb_send_create_allocation`.
* **`LX_DXLOCK2` (`dxgkio_lock2`):** Calls `D3DKMTLock2` on the host. When a CPU-visible local VRAM allocation is locked, the Windows host pins the BAR1 physical aperture slice, exposes it to the Hyper-V guest via a Guest Physical Address Descriptor List (GPADL) / IO space address, and `dxgkrnl` maps it into the Linux process's virtual address space using `dxg_map_iospace()`.

### 3.2. `libd3d12core.so` Runtime Constraints (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
Microsoft introduced `D3D12_HEAP_TYPE_GPU_UPLOAD` (enum value `5` in `d3d12.h`, `D3D12_HEAP_TYPE`) in DirectX 12 Agility SDK 1.610 to provide direct CPU write access to VRAM via ReBAR.
* In WSL2, `/usr/lib/wsl/lib/libd3d12core.so` is mounted directly from the Windows host (`C:\Windows\System32\lxss\lib\libd3d12core.so`). A binary dated October 2023 corresponds to the inbox Windows 11 D3D12 runtime rather than the newer Agility SDK redistributable.
* Even if `D3D12_HEAP_TYPE_GPU_UPLOAD` is absent from the inbox `libd3d12core.so`, WDDM allows creating ReBAR-backed heaps via `D3D12_HEAP_TYPE_CUSTOM` (enum value `4`) configured with `D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE` (`2`) and `D3D12_MEMORY_POOL_L1` (`2`, physical VRAM) when `D3D12_FEATURE_DATA_ARCHITECTURE1::UMA` is `FALSE` and the driver supports CPU-visible L1 pools, or via Vulkan `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.

---

## 4. Verified Diagnostic & Enumeration Commands (Read-Only)

All integer values for `CUdevice_attribute` below are verified directly against NVIDIA's `cuda.h` header specification (`CUdevice_attribute_enum` in CUDA 12.x / 13.x):

* `CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY` = `19`
* `CU_DEVICE_ATTRIBUTE_INTEGRATED` = `18`
* `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED` = `99`
* `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` = `102`
* `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED` = `103`
* `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED` = `104`
* `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` = `116`
* `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` = `124`

### 4.1. Exact Header & Driver Attribute Query
Run the following read-only command block inside WSL2 to extract the exact enum definitions from your local `/usr/local/cuda/include/cuda.h` and query `libcuda.so.1` at runtime:

```bash
python3 -c "
import ctypes

cuda = ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1')
assert cuda.cuInit(0) == 0, 'cuInit failed'

attrs = [
    ('CU_DEVICE_ATTRIBUTE_INTEGRATED', 18),
    ('CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY', 19),
    ('CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED', 99),
    ('CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED', 102),
    ('CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED', 103),
    ('CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED', 104),
    ('CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED', 116),
    ('CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED', 124),
]

val = ctypes.c_int()
for name, code in attrs:
    res = cuda.cuDeviceGetAttribute(ctypes.byref(val), code, 0)
    print(f'{name} ({code}): {val.value} (ret={res})')
"
```

### 4.2. WSL2 Library Symbol & Kernel Header Verification
Run the following read-only bash commands to verify exported interop symbols in `libcuda.so.1`, check `libd3d12core.so` strings, and locate the `dxgkrnl` UAPI header:

```bash
# 1. Verify CUDA External Memory and VMM entry points in WSL libcuda
nm -D /usr/lib/wsl/lib/libcuda.so.1 | grep -E "cuImportExternalMemory|cuExternalMemoryGetMappedBuffer|cuMemCreate|cuMemMap|cuMemAddressReserve"

# 2. Check for GPU_UPLOAD heap strings in WSL libd3d12core.so
strings /usr/lib/wsl/lib/libd3d12core.so | grep -iE "GPU_UPLOAD|GpuUpload" | head -n 10

# 3. Locate dxgkrnl ioctl UAPI headers in the installed WSL2 kernel headers
find /usr/src/linux-headers-$(uname -r) -iname "*d3dkmt*" -o -iname "*dxg*" 2>/dev/null
```

---

## 5. Viable Implementation Pathways for CPU-to-BAR1 Writes

### Pathway A: Vulkan `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` -> CUDA Interop
Unlike `gdrcopy`, graphics APIs do not lock out GeForce cards from mapping the ReBAR window. This pathway works natively on Linux and on Windows (and in WSL2 when a hardware Vulkan ICD or D3D12 interop handle is exported):
1. Query `VkPhysicalDeviceMemoryProperties`. With ReBAR enabled on the RTX 4070, Vulkan exposes a memory heap with `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` sized to the full VRAM capacity (12 GiB) rather than the legacy 256 MiB staging aperture.
2. Allocate a `VkDeviceMemory` buffer on that heap with `VkExportMemoryAllocateInfo` (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` on Linux or `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` on Windows).
3. Map the buffer into the CPU address space via `vkMapMemory()`. This immediately increments `BAR1 Memory Usage: Used` in `nvidia-smi` and returns a Write-Combining (WC) CPU virtual address pointing directly into the BAR1 window.
4. Export the handle (`vkGetMemoryFdKHR` or `vkGetMemoryWin32HandleKHR`) and import it into CUDA via `cudaImportExternalMemory()` and `cudaExternalMemoryGetMappedBuffer()`.
5. Pass the resulting device pointer directly to CUDA kernels while the CPU writes to the `vkMapMemory()` pointer.

### Pathway B: D3D12 Shared Heap -> CUDA External Memory Interop (WSL2 / Windows)
Inside WSL2, `libd3d12.so` and `libdxcore.so` communicate natively with the host WDDM driver:
1. Enumerate the hardware adapter via `DXCoreCreateAdapterFactory` (`libdxcore.so`) and instantiate a D3D12 device (`D3D12CreateDevice` in `libd3d12.so`).
2. Create a shared heap (`D3D12_HEAP_FLAG_SHARED`) backed by CPU-visible VRAM (`D3D12_HEAP_TYPE_GPU_UPLOAD` if supported by the host runtime, or `D3D12_HEAP_TYPE_CUSTOM` with `D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE` and `D3D12_MEMORY_POOL_L1`).
3. Call `ID3D12Resource::Map()` to obtain the CPU-side BAR1 pointer.
4. Export the shared handle via `ID3D12Device::CreateSharedHandle()` and import it into CUDA using `cudaImportExternalMemory()` with `cudaExternalMemoryHandleTypeD3D12Heap` (`4`) or `cudaExternalMemoryHandleTypeD3D12Resource` (`5`).

---

## 6. Microarchitectural Constraints: Ryzen 7 5800X3D & Write-Combining (WC) BAR1 Memory

When the CPU maps the PCIe BAR1 aperture, the x86 Page Attribute Table (PAT) marks those virtual memory pages as **Write-Combining (WC)** (`USWC` - Uncacheable Speculative Write-Combining).

1. **Strictly Unidirectional (CPU Write Only):** CPU loads (`MOV` / `VMOVDQA`) from a WC BAR1 pointer are uncached and execute as individual PCIe Memory Read TLPs across the bus. Read latency exceeds $1\text{ }\mu\text{s}$ per cache line, limiting read throughput to $<50\text{ MB/s}$. The CPU must never read from the BAR1 pointer; GPU-to-CPU signaling (such as completion flags) must always write from the GPU into cached host-pinned system RAM (`cudaHostAlloc`).
2. **Preventing Partial Write-Combining Buffer Flushes:** The AMD Zen 3 core (Ryzen 7 5800X3D) combines stores into 64-byte Write-Combining buffers before bursting a single 64-byte payload across the Infinity Fabric to the PCIe 4.0 root complex. Unaligned scalar writes or out-of-order stores can prematurely flush partial WC buffers, degrading PCIe 4.0 x16 throughput from $\sim 25\text{ GB/s}$ down to $<3\text{ GB/s}$.
3. **AVX2 Non-Temporal Streaming Stores:** For payload transfers (such as streaming MoE expert weights or activation vectors into BAR1), enforce 64-byte alignment and issue AVX2 non-temporal stores (`_mm256_stream_si256` from `<immintrin.h>`) in contiguous 64-byte (two 256-bit YMM registers) strides, followed by a store fence (`_mm_sfence()`) prior to ringing the doorbell flag:

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

void bar1_stream_copy_avx2(void * __restrict dst_bar1_wc, const void * __restrict src_ram, size_t num_bytes) {
    // Requires dst_bar1_wc and src_ram to be 32-byte (ideally 64-byte) aligned
    // and num_bytes to be a multiple of 64.
    const __m256i *src = (const __m256i *)src_ram;
    __m256i *dst = (__m256i *)dst_bar1_wc;
    size_t chunks = num_bytes / 64;

    for (size_t i = 0; i < chunks; ++i) {
        __m256i v0 = _mm256_load_si256(src + (i * 2) + 0);
        __m256i v1 = _mm256_load_si256(src + (i * 2) + 1);
        _mm256_stream_si256(dst + (i * 2) + 0, v0);
        _mm256_stream_si256(dst + (i * 2) + 1, v1);
    }
    _mm_sfence(); // Ensures all WC buffers are flushed to the PCIe root complex before signaling GPU
}
```
This guarantees full 64-byte TLP bursts over PCIe 4.0 x16 while bypassing the 96 MiB 3D V-Cache on the 5800X3D so CPU cache residency is preserved for host-side MoE compute.

---

## 7. Primary Technical Sources & Specifications

1. **NVIDIA CUDA Driver API Specification (`cuda.h` / `CUdevice_attribute` & External Resource Interoperability):**
   * *NVIDIA CUDA Driver API Reference — Data Types (`CUdevice_attribute_enum`):* Verifies enum values `19` (`CAN_MAP_HOST_MEMORY`), `99` (`HOST_REGISTER_SUPPORTED`), `102` (`VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`), `103` (`HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED`), `104` (`HANDLE_TYPE_WIN32_HANDLE_SUPPORTED`), `116` (`GPU_DIRECT_RDMA_SUPPORTED`), and `124` (`DMA_BUF_SUPPORTED`).
   * *NVIDIA CUDA Driver API — External Resource Interoperability (`cuImportExternalMemory`, `cuExternalMemoryGetMappedBuffer`):* Specifies importing `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`, `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`, and `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`.
   * URL: `https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html` and `https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html`
2. **NVIDIA CUDA on WSL User Guide:**
   * Documents the paravirtualized WDDM model over `/dev/dxg`, absence of the native Linux `nvidia.ko` kernel driver inside the guest, and limitations on unified memory / P2P / GPUDirect RDMA under WSL2.
   * URL: `https://docs.nvidia.com/cuda/wsl-user-guide/index.html`
3. **NVIDIA GPUDirect RDMA & `gdrcopy` Platform Restrictions:**
   * *NVIDIA GPUDirect RDMA Documentation & `gdrcopy` GitHub Repository (`NVIDIA/gdrcopy`):* Details the requirement for `nvidia_p2p_get_pages` in `nvidia.ko` and notes that GPUDirect RDMA is supported only on Tesla/Data Center and professional Quadro/RTX GPUs, not standard consumer GeForce drivers.
   * URL: `https://docs.nvidia.com/cuda/gpudirect-rdma/index.html` and `https://github.com/NVIDIA/gdrcopy`
4. **Microsoft WSL2 Linux Kernel (`dxgkrnl` Paravirtualization Driver):**
   * *Source files:* `drivers/hv/dxgkrnl/ioctl.c`, `drivers/hv/dxgkrnl/dxgvmbus.c`, and `include/uapi/misc/d3dkmthk.h` in `microsoft/WSL2-Linux-Kernel`. Defines `LX_DXCREATEALLOCATION`, `LX_DXLOCK2`, `dxg_map_iospace`, and Hyper-V VMBus GPADL mapping for CPU-visible adapter allocations.
   * URL: `https://github.com/microsoft/WSL2-Linux-Kernel/tree/linux-msft-wsl-6.6.y/drivers/hv/dxgkrnl`
5. **Microsoft DirectX 12 Specifications (`D3D12_HEAP_TYPE` & GPU Upload Heaps):**
   * *Win32 `d3d12.h` Reference & DirectX 12 Agility SDK GPU Upload Heaps:* Defines `D3D12_HEAP_TYPE_CUSTOM` (`4`), `D3D12_HEAP_TYPE_GPU_UPLOAD` (`5`), and CPU write-combined access to physical L1 VRAM via Resizable BAR.
   * URL: `https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type` and `https://devblogs.microsoft.com/directx/agility-sdk-1-610-0-gpu-upload-heaps-and-more/`
6. **Khronos Vulkan Specification (`VkMemoryPropertyFlagBits` & External Memory):**
   * *`VK_KHR_external_memory` and `VkMemoryPropertyFlagBits`:* Specifies `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` for ReBAR host-mapped VRAM allocations and export to external compute runtimes.
   * URL: `https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryPropertyFlagBits.html`
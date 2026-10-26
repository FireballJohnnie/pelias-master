// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// e16_kernels.cu - P-E16 GPU side. Target sm_89.
// Revision 2: the first version (ld.acquire.sys / st.release.sys as inline PTX with "memory" clobbers, kernels in a
// top-level anonymous namespace) crashed cudafe++ on Windows (0xC0000005). This version uses only constructs the Wave 0
// and Batch C kernels already compile with: volatile loads and stores plus __threadfence_system(), the %globaltimer read
// written exactly as in kernels.cu, and the anonymous namespace nested inside a named one.
#include "e16_kernels.h"

namespace e16 {
namespace {

__device__ __forceinline__ unsigned long long gtime_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Waits on the request word with volatile loads (not cached in L1; each poll reads L2 for VRAM, or crosses PCIe for
// system RAM), then publishes the reply with a volatile store followed by a system-scope fence.
__global__ void k_bell(const uint32_t* req, uint32_t* rep, uint32_t first, uint32_t count, unsigned long long wait_ns, uint32_t* status) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const volatile uint32_t* rq = req;
  volatile uint32_t* rp = rep;
  for (uint32_t n = first; n != first + count; ++n) {
    const unsigned long long t0 = gtime_ns();
    uint32_t spins = 0;
    while (*rq != n) {
      if ((++spins & 255u) == 0 && gtime_ns() - t0 > wait_ns) {
        *status = n;
        return;
      }
    }
    __threadfence_system();
    *rp = n;
    __threadfence_system();
  }
  *status = 0xFFFFFFFFu;
}

__global__ void k_empty() {}

}  // namespace

cudaError_t launch_bell(const uint32_t* req, uint32_t* rep, uint32_t first, uint32_t count, uint64_t wait_ns, uint32_t* status, cudaStream_t s) {
  k_bell<<<1, 32, 0, s>>>(req, rep, first, count, static_cast<unsigned long long>(wait_ns), status);
  return cudaGetLastError();
}
cudaError_t launch_empty(cudaStream_t s) {
  k_empty<<<1, 32, 0, s>>>();
  return cudaGetLastError();
}

}  // namespace e16

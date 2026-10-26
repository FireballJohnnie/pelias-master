// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// e16_kernels.h - P-E16: the GPU side of the doorbell round trip (host-callable wrappers; e16_kernels.cu needs nvcc).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace e16 {
// One GPU thread answers `count` rings: for n = first .. first + count - 1 it waits until *req == n (acquire, system
// scope), then writes n to *rep (release, system scope). It gives up when one wait exceeds wait_ns of GPU time and
// writes the ring it was waiting for to *status; on success *status = 0xFFFFFFFF. status is device memory.
cudaError_t launch_bell(const uint32_t* req, uint32_t* rep, uint32_t first, uint32_t count, uint64_t wait_ns, uint32_t* status, cudaStream_t s);
// An empty kernel, for the launch-and-synchronize reference.
cudaError_t launch_empty(cudaStream_t s);
}  // namespace e16

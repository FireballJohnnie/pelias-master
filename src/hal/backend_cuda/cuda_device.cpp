// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// cuda_device.cpp - the CUDA runtime's view of the device. Compiled only when the CUDA backend is built in; the
// runtime is linked statically (3.2 B2). Creating the context here is what makes the free-memory reading the
// "after the context exists" value the ceiling rule needs.
#include <cuda_runtime.h>

#include <cstdio>

#include "hal/device.h"

namespace pelias::hal {

bool cuda_built_in() { return true; }

bool cuda_query(int index, DeviceInfo& out, std::string& pci_bus_id) {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    out.note = std::string("CUDA runtime: ") + cudaGetErrorString(e);
    return false;
  }
  if (index < 0 || index >= n) {
    out.note = "CUDA device " + std::to_string(index) + " not present (" + std::to_string(n) + " device(s))";
    return false;
  }
  if ((e = cudaSetDevice(index)) != cudaSuccess || (e = cudaFree(nullptr)) != cudaSuccess) {
    out.note = std::string("CUDA context: ") + cudaGetErrorString(e);
    return false;
  }
  cudaDeviceProp p{};
  if ((e = cudaGetDeviceProperties(&p, index)) != cudaSuccess) {
    out.note = std::string("CUDA properties: ") + cudaGetErrorString(e);
    return false;
  }
  out.name = p.name;
  out.cc_major = p.major;
  out.cc_minor = p.minor;
  out.sm_count = p.multiProcessorCount;
  out.vram_total = p.totalGlobalMem;
  size_t free_b = 0, total_b = 0;
  if ((e = cudaMemGetInfo(&free_b, &total_b)) != cudaSuccess) {
    out.note = std::string("CUDA memory info: ") + cudaGetErrorString(e);
    return false;
  }
  out.vram_free_cuda = free_b;
  if (total_b) out.vram_total = total_b;
  char bus[32] = {};
  if (cudaDeviceGetPCIBusId(bus, sizeof bus, index) == cudaSuccess) pci_bus_id = bus;
  return true;
}

}  // namespace pelias::hal

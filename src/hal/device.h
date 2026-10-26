// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// device.h - what the program knows about the GPU at start-up: the CUDA device (when the CUDA backend is built in and a
// device exists) and NVML's view of the same card (loaded at run time; optional). The VRAM ceiling rule of 0.3 (3)
// uses both readings. A build without CUDA, or a machine without a device, reports `available = false` and the rest of
// the program (inspect, plan on a stated ceiling) still works.
#pragma once

#include <cstdint>
#include <string>

#include "pelias/status.h"

namespace pelias::hal {

struct DeviceInfo {
  bool available = false;        // a CUDA device was opened and its context created
  std::string backend;           // "cuda" or "none"
  std::string name;
  int index = 0;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  std::uint64_t vram_total = 0;        // bytes, from the runtime
  std::uint64_t vram_free_cuda = 0;    // bytes free as the CUDA runtime reports after the context exists
  bool nvml_available = false;
  std::uint64_t vram_free_nvml = 0;    // bytes free as NVML reports (whole card)
  std::uint64_t vram_used_nvml = 0;
  std::string driver_version;          // from NVML when available
  std::string note;                    // why something is unavailable (user-readable, no paths)
};

// Opens device `index` (creates its context) and reads both views. Never throws.
DeviceInfo query_device(int index = 0);

// True when this build carries the CUDA backend.
bool cuda_built_in();

}  // namespace pelias::hal

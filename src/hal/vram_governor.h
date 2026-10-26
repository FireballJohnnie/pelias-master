// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// vram_governor.h - the hard VRAM ceiling (standing rule 3; D26, D28): the smaller of CUDA free and NVML free after the
// context exists, minus a margin. The margin on the reference machine is 0.13 GiB (profiles' plan_basis: 10.83 GiB free
// after the context, 10.7 GiB ceiling). Every plan is checked against this number before anything is allocated.
#pragma once

#include <cstdint>

#include "hal/device.h"

namespace pelias::hal {

inline constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
inline constexpr std::uint64_t kDefaultMarginBytes = 139586437;  // 0.13 GiB (10.83 - 10.7 on the reference, DERIVED)

struct Ceiling {
  std::uint64_t bytes = 0;         // the ceiling
  std::uint64_t basis_free = 0;    // the reading it came from
  const char* basis = "none";      // "cuda", "nvml", "stated"
  std::uint64_t margin = 0;
};

// From measured readings. If NVML is unavailable the CUDA reading alone is used; if no device, bytes = 0.
Ceiling ceiling_from_device(const DeviceInfo& d, std::uint64_t margin = kDefaultMarginBytes);
// From a number the user states (for planning without a device, or for emulating a smaller card; 21.5.2).
Ceiling ceiling_stated(std::uint64_t bytes);

}  // namespace pelias::hal

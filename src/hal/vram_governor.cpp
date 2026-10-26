// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "hal/vram_governor.h"

#include <algorithm>

namespace pelias::hal {

Ceiling ceiling_from_device(const DeviceInfo& d, std::uint64_t margin) {
  Ceiling c;
  c.margin = margin;
  if (!d.available) return c;
  std::uint64_t free = d.vram_free_cuda;
  c.basis = "cuda";
  if (d.nvml_available && d.vram_free_nvml < free) {
    free = d.vram_free_nvml;
    c.basis = "nvml";
  }
  c.basis_free = free;
  c.bytes = free > margin ? free - margin : 0;
  return c;
}

Ceiling ceiling_stated(std::uint64_t bytes) {
  Ceiling c;
  c.bytes = bytes;
  c.basis_free = bytes;
  c.basis = "stated";
  return c;
}

}  // namespace pelias::hal

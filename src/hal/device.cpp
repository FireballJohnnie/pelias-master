// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "hal/device.h"

#include "hal/nvml.h"

namespace pelias::hal {

// Implemented in backend_cuda/cuda_device.cpp when PELIAS_CUDA is on; the stub below otherwise.
bool cuda_query(int index, DeviceInfo& out, std::string& pci_bus_id);

DeviceInfo query_device(int index) {
  DeviceInfo d;
  d.index = index;
  std::string pci;
  if (!cuda_built_in()) {
    d.backend = "none";
    d.note = "this build has no CUDA backend";
  } else if (!cuda_query(index, d, pci)) {
    d.backend = "none";
    if (d.note.empty()) d.note = "no usable CUDA device";
  } else {
    d.backend = "cuda";
    d.available = true;
  }
  const NvmlMemory m = nvml_memory(pci);
  d.nvml_available = m.ok;
  if (m.ok) {
    d.vram_free_nvml = m.free;
    d.vram_used_nvml = m.used;
    if (d.vram_total == 0) d.vram_total = m.total;
  } else if (!m.note.empty()) {
    d.note += (d.note.empty() ? "" : "; ") + m.note;
  }
  d.driver_version = m.driver_version;
  return d;
}

}  // namespace pelias::hal

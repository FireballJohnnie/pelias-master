// provenance: original (the NVML entry points and struct layout are the public NVML API, loaded by name at run time;
//   no NVML header is included and nothing is linked)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// nvml.h - NVML opened at run time for the card's memory and driver version. Every call is optional: when the library
// or a function is missing the result says so and the program continues.
#pragma once

#include <cstdint>
#include <string>

namespace pelias::hal {

struct NvmlMemory {
  bool ok = false;
  std::uint64_t total = 0, free = 0, used = 0;
  std::string driver_version;
  std::string note;
};

// Which NVML device to read. CUDA and NVML may number cards differently (CUDA's default order is fastest first, NVML's
// is by PCI bus), so only the PCI bus id identifies the same card in both. Without a PCI match, the one card is used
// only when NVML sees exactly one; otherwise none (the ceiling then rests on CUDA alone, and the note says so).
enum class NvmlPick { by_pci, only_device, none };
NvmlPick nvml_pick(bool pci_matched, unsigned device_count);

// Reads memory of the device whose PCI bus id matches `pci_bus_id` (as the CUDA runtime formats it); see nvml_pick.
NvmlMemory nvml_memory(const std::string& pci_bus_id);

}  // namespace pelias::hal

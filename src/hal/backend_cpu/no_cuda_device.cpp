// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// no_cuda_device.cpp - the stand-in when the CUDA backend is not built (host-only builds, the sandbox).
#include "hal/device.h"

namespace pelias::hal {
bool cuda_built_in() { return false; }
bool cuda_query(int, DeviceInfo& out, std::string&) {
  out.note = "this build has no CUDA backend";
  return false;
}
}  // namespace pelias::hal

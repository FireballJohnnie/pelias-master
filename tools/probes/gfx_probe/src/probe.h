// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// probe.h: the two halves of gfx_probe. Each returns one JSON object as text and prints a short summary.
#pragma once

#include <string>

#include "common.h"

namespace gp {

std::string run_vulkan(const Settings& s, int device_index);
std::string run_d3d12(const Settings& s, int adapter_index);

}  // namespace gp

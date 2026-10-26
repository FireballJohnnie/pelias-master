// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// clock.h - wall-clock time as an ISO-8601 string (for result files) and a monotonic nanosecond counter.
#pragma once

#include <cstdint>
#include <string>

namespace pelias::os {
std::string iso_now();          // local time with offset, e.g. 2026-10-26T17:53:00-0400
std::uint64_t monotonic_ns();   // steady clock, nanoseconds
std::string platform_name();    // "windows-native", "wsl2" or "linux"
}  // namespace pelias::os

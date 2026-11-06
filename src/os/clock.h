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
struct PageFaults { std::uint64_t major = 0, minor = 0; };  // v58c: the process's page faults so far (getrusage on Linux)
bool page_faults_available();   // true on Linux / WSL2
PageFaults page_faults();       // zeros where not available
// v60 E2 (author 13:29: "what's getting read vs what should be ignored ... if it's accidentally hooked or folded in to the wrong
// process"): the process's CPU time so far (user + system, all threads) and its live thread count, read at phase boundaries
// so each phase's CPU seconds against its wall seconds says how many cores it kept busy (Linux: getrusage + /proc/self/status;
// Windows: GetProcessTimes; threads -1 where not available)
struct CpuTime { double user_s = 0.0, sys_s = 0.0; int threads = -1; };
bool cpu_time_available();
CpuTime cpu_time();
}  // namespace pelias::os

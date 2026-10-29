// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// membw.h - the machine probe's memory and threads check (author 2026-10-29): how fast the CPU reads system RAM with
// 1..N threads, with 4 KB or 2 MB pages and with or without software prefetch, and what that load does to the user's
// own work on the same machine: a separate "foreground" thread wakes every millisecond, does a small fixed piece of
// work, and records how late it woke and how much slower the work ran than on an idle machine.
// The readers are pinned one per physical core first, then to the second thread of each core (the order the engine's
// cold-expert readers would use). Under WSL2 the CPUs are virtual (Hyper-V), so pinning picks virtual CPUs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pelias::os {

struct MemBwOptions {
  std::size_t bytes = std::size_t{1} << 30;  // the buffer read by all threads together (well past the L3)
  std::vector<int> threads{1, 2, 4, 6, 8, 12, 16};
  int reps = 5;                              // median of this many timed passes per cell
  int passes = 3;                            // full reads of the buffer per timed pass
  bool foreground = true;                    // measure the user-side effect on the cells with huge = false, prefetch = false
};

struct MemBwCell {
  int threads = 0;
  bool huge = false;      // 2 MB pages asked for (madvise MADV_HUGEPAGE)
  bool prefetch = false;  // software prefetch 1 KiB ahead
  double gbps = 0.0;      // median read rate, GB/s (1e9 bytes)
  // foreground thread while the readers run (only on the plain cells): wake-up lateness p50 / p99 in microseconds, and
  // the fixed work's time relative to the idle machine (1.00 = no slowdown)
  double wake_p50_us = -1.0, wake_p99_us = -1.0, work_ratio = -1.0;
};

struct MemBwResult {
  std::vector<MemBwCell> cells;
  int logical_cpus = 0, physical_cores = 0;
  std::string thp_mode;               // /sys/kernel/mm/transparent_hugepage/enabled ("" if absent)
  long long huge_kib_seen = -1;       // AnonHugePages (kB) while the 2 MB buffer was in use, -1 if unknown
  double idle_wake_p50_us = -1.0, idle_wake_p99_us = -1.0, idle_work_us = -1.0;
  std::string note;                   // platform remarks (e.g. no huge pages on this platform)
};

// Runs the sweep. Fails only if the buffer cannot be allocated.
bool membw_run(const MemBwOptions& opt, MemBwResult& out, std::string& error);

}  // namespace pelias::os

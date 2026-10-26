// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// timer.h - monotonic wall-clock time for measurements (the OS clock lives in os/clock; this is the header-only use).
#pragma once

#include <chrono>

namespace pelias {

class Timer {
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

 public:
  void reset() { t0_ = std::chrono::steady_clock::now(); }
  [[nodiscard]] double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
  [[nodiscard]] double ms() const { return seconds() * 1e3; }
};

}  // namespace pelias

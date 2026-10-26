// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/clock.h"

#include <chrono>
#include <cstdio>
#include <ctime>

#ifndef _WIN32
#include <fstream>
#include <string>
#endif

namespace pelias::os {

std::string iso_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tmv{};
#ifdef _WIN32
  localtime_s(&tmv, &t);
#else
  localtime_r(&t, &tmv);
#endif
  char buf[64];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S%z", &tmv);
  return buf;
}

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string platform_name() {
#ifdef _WIN32
  return "windows-native";
#else
  std::ifstream f("/proc/sys/kernel/osrelease");
  std::string rel;
  if (f && std::getline(f, rel) && (rel.find("microsoft") != std::string::npos || rel.find("WSL") != std::string::npos)) return "wsl2";
  return "linux";
#endif
}

}  // namespace pelias::os

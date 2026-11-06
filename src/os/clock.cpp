// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/clock.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

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

namespace pelias::os {
bool page_faults_available() {
#if defined(__linux__)
  return true;
#else
  return false;
#endif
}
PageFaults page_faults() {
  PageFaults p;
#if defined(__linux__)
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    p.major = static_cast<std::uint64_t>(ru.ru_majflt);
    p.minor = static_cast<std::uint64_t>(ru.ru_minflt);
  }
#endif
  return p;
}
// v60 E2
bool cpu_time_available() {
#if defined(__linux__) || defined(_WIN32)
  return true;
#else
  return false;
#endif
}
CpuTime cpu_time() {
  CpuTime c;
#if defined(__linux__)
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    c.user_s = static_cast<double>(ru.ru_utime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_utime.tv_usec);
    c.sys_s = static_cast<double>(ru.ru_stime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_stime.tv_usec);
  }
  std::ifstream f("/proc/self/status");
  std::string line;
  while (f && std::getline(f, line))
    if (line.rfind("Threads:", 0) == 0) {
      c.threads = std::atoi(line.c_str() + 8);
      break;
    }
#elif defined(_WIN32)
  FILETIME ct, et, kt, ut;
  if (GetProcessTimes(GetCurrentProcess(), &ct, &et, &kt, &ut)) {
    auto secs = [](const FILETIME& ft) {
      ULARGE_INTEGER u;
      u.LowPart = ft.dwLowDateTime;
      u.HighPart = ft.dwHighDateTime;
      return static_cast<double>(u.QuadPart) * 1e-7;  // 100 ns units
    };
    c.user_s = secs(ut);
    c.sys_s = secs(kt);
  }
#endif
  return c;
}
}  // namespace pelias::os

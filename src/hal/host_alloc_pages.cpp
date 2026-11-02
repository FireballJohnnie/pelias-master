// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// host_alloc_pages.cpp - see host_alloc_pages.h. OS calls: madvise, the thread affinity, and (v46 arena probe) mmap /
// munmap of an anonymous arena and direct reads of the model file into it.
#include "hal/host_alloc_pages.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace pelias::hal {

bool advise_small_pages(void* p, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_NOHUGEPAGE)
  return madvise(p, bytes, MADV_NOHUGEPAGE) == 0;
#else
  (void)p;
  (void)bytes;
  return true;
#endif
}

bool advise_huge_pages(void* p, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  return madvise(p, bytes, MADV_HUGEPAGE) == 0;
#else
  (void)p;
  (void)bytes;
  return true;
#endif
}

bool pin_this_thread(int cpu) {
#if defined(__linux__)
  if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<std::size_t>(cpu), &set);
  return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#elif defined(_WIN32)
  // v58 (author 15:09: on Windows the GPU's threads can be pinned): processor group 0, logical CPUs 0-63
  if (cpu < 0 || cpu >= 64) return false;
  return SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu) != 0;
#else
  (void)cpu;
  return false;
#endif
}

std::string cpu_core_map() {
  std::string out;
#if defined(_WIN32)
  DWORD len = 0;
  (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len == 0) return out;
  std::string buf(static_cast<std::size_t>(len), '\0');
  auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) return out;
  int core = 0;
  for (DWORD off = 0; off < len;) {
    const auto* p = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
    if (p->Relationship == RelationProcessorCore) {
      out += "core " + std::to_string(core++) + ":";
      for (WORD gi = 0; gi < p->Processor.GroupCount; ++gi) {
        const GROUP_AFFINITY& ga = p->Processor.GroupMask[gi];
        for (int b = 0; b < 64; ++b)
          if ((ga.Mask >> b) & 1u) out += " " + std::to_string(64 * static_cast<int>(ga.Group) + b);
      }
      out += p->Processor.Flags == LTP_PC_SMT ? " (SMT)\n" : "\n";
    }
    if (p->Size == 0) break;
    off += p->Size;
  }
#elif defined(__linux__)
  std::vector<std::pair<std::string, std::string>> cores;  // (package:core, cpus)
  for (int c = 0; c < 1024; ++c) {
    std::ifstream fc("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_id");
    if (!fc) break;
    std::string id, pk;
    fc >> id;
    std::ifstream fp("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id");
    if (fp) fp >> pk;
    const std::string key = pk + ":" + id;
    bool found = false;
    for (auto& kv : cores)
      if (kv.first == key) {
        kv.second += " " + std::to_string(c);
        found = true;
      }
    if (!found) cores.emplace_back(key, std::to_string(c));
  }
  for (std::size_t i = 0; i < cores.size(); ++i) out += "core " + std::to_string(i) + ": " + cores[i].second + "\n";
#endif
  return out;
}

void* huge_arena_from_file(const char* path, unsigned long long offset, std::size_t bytes, bool* direct_used, std::string* why) {
#if defined(__linux__)
  if (direct_used) *direct_used = false;
  constexpr std::size_t kHuge = std::size_t{2} << 20;
  if (bytes == 0 || bytes % kHuge != 0 || offset % 4096 != 0) {
    if (why) *why = "bytes must be a multiple of 2 MB and the offset of 4 KB";
    return nullptr;
  }
  // reserve 2 MB more and start at a 2 MB boundary (the tail and head slack stay unmapped-in-use, released on free)
  const std::size_t span = bytes + kHuge;
  void* raw = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) {
    if (why) *why = std::string("mmap: ") + std::strerror(errno);
    return nullptr;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(raw);
  const std::uintptr_t al = (base + kHuge - 1) & ~(std::uintptr_t{kHuge} - 1);
  if (al > base) munmap(raw, al - base);
  const std::uintptr_t end = base + span, aend = al + bytes;
  if (end > aend) munmap(reinterpret_cast<void*>(aend), end - aend);
  void* p = reinterpret_cast<void*>(al);
#if defined(MADV_HUGEPAGE)
  madvise(p, bytes, MADV_HUGEPAGE);
#endif
  auto* dst = static_cast<char*>(p);
  int fd = open(path, O_RDONLY | O_DIRECT);
  bool direct = fd >= 0;
  if (fd < 0) fd = open(path, O_RDONLY);
  if (fd < 0) {
    if (why) *why = std::string("open: ") + std::strerror(errno);
    munmap(p, bytes);
    return nullptr;
  }
  std::size_t done = 0;
  while (done < bytes) {
    const std::size_t want = bytes - done < (std::size_t{64} << 20) ? bytes - done : (std::size_t{64} << 20);
    const ssize_t got = pread(fd, dst + done, want, static_cast<off_t>(offset + done));
    if (got < 0 && direct && done == 0 && errno == EINVAL) {  // the filesystem refused O_DIRECT: read through the cache instead
      close(fd);
      fd = open(path, O_RDONLY);
      direct = false;
      if (fd < 0) break;
      continue;
    }
    if (got <= 0) break;
    done += static_cast<std::size_t>(got);
  }
  if (fd >= 0) close(fd);
  if (done != bytes) {
    if (why) *why = "short read of the file into the arena";
    munmap(p, bytes);
    return nullptr;
  }
  if (direct_used) *direct_used = direct;
  return p;
#else
  (void)path;
  (void)offset;
  (void)bytes;
  if (direct_used) *direct_used = false;
  if (why) *why = "the arena probe is Linux only (Windows: VirtualAlloc MEM_LARGE_PAGES, not built yet)";
  return nullptr;
#endif
}

void huge_arena_free(void* p, std::size_t bytes) {
#if defined(__linux__)
  if (p) munmap(p, bytes);
#else
  (void)p;
  (void)bytes;
#endif
}

std::string smaps_rollup_line(const char* key) {
  std::ifstream f("/proc/self/smaps_rollup");
  std::string l;
  while (std::getline(f, l))
    if (l.rfind(key, 0) == 0) return l;
  return std::string();
}

}  // namespace pelias::hal

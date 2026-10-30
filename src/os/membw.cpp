// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/membw.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#define PELIAS_MEMBW_LINUX 1
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PELIAS_MEMBW_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define PELIAS_TARGET_AVX2_FN  // MSVC compiles AVX2 intrinsics in any function
#else
#define PELIAS_TARGET_AVX2_FN __attribute__((target("avx2")))
#endif
#endif

namespace pelias::os {
namespace {

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

// CPUs ordered one per physical core first, then the second thread of each core (Linux sysfs topology); identity order
// elsewhere
std::vector<int> cpu_order(int& cores) {
  const int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  std::vector<int> order;
  cores = n;
#if defined(PELIAS_MEMBW_LINUX)
  std::map<std::pair<int, int>, std::vector<int>> by_core;  // (package, core) -> cpus
  for (int c = 0; c < n; ++c) {
    int core = c, pkg = 0;
    std::ifstream fc("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_id");
    std::ifstream fp("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id");
    if (fc) fc >> core;
    if (fp) fp >> pkg;
    by_core[{pkg, core}].push_back(c);
  }
  cores = static_cast<int>(by_core.size());
  for (std::size_t k = 0;; ++k) {
    bool any = false;
    for (auto& [key, cpus] : by_core)
      if (k < cpus.size()) {
        order.push_back(cpus[k]);
        any = true;
      }
    if (!any) break;
  }
#endif
  if (order.empty())
    for (int c = 0; c < n; ++c) order.push_back(c);
  return order;
}

void pin_to(int cpu) {
#if defined(PELIAS_MEMBW_LINUX)
  cpu_set_t set;
  CPU_ZERO(&set);
  const std::size_t c = static_cast<std::size_t>(cpu);
  CPU_SET(c, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#else
  (void)cpu;
#endif
}

#if defined(PELIAS_MEMBW_X86)
PELIAS_TARGET_AVX2_FN std::uint64_t read_avx2(const unsigned char* p, std::size_t n, bool pf) {
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
  for (std::size_t i = 0; i + 128 <= n; i += 128) {
    if (pf) _mm_prefetch(reinterpret_cast<const char*>(p + i + 1024), _MM_HINT_T0);
    a0 = _mm256_xor_si256(a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i)));
    a1 = _mm256_xor_si256(a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 32)));
    a2 = _mm256_xor_si256(a2, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 64)));
    a3 = _mm256_xor_si256(a3, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 96)));
  }
  const __m256i x = _mm256_xor_si256(_mm256_xor_si256(a0, a1), _mm256_xor_si256(a2, a3));
  alignas(32) std::uint64_t w[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(w), x);
  return w[0] ^ w[1] ^ w[2] ^ w[3];
}
#endif

std::uint64_t read_plain(const unsigned char* p, std::size_t n) {
  std::uint64_t a = 0, b = 0;
  for (std::size_t i = 0; i + 16 <= n; i += 16) {
    std::uint64_t x, y;
    std::memcpy(&x, p + i, 8);
    std::memcpy(&y, p + i + 8, 8);
    a ^= x;
    b ^= y;
  }
  return a ^ b;
}

#if defined(PELIAS_MEMBW_X86)
// AVX2 usable (the os layer depends on nothing, so this is its own copy of hal/cpuid.h's check)
bool cpu_has_avx2() {
#if defined(_MSC_VER) && !defined(__clang__)
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1) || (_xgetbv(0) & 6) != 6) return false;
  __cpuidex(r, 7, 0);
  return (r[1] >> 5) & 1;
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#endif
}
#endif

std::uint64_t read_slice(const unsigned char* p, std::size_t n, bool pf) {
#if defined(PELIAS_MEMBW_X86)
  static const bool avx2 = cpu_has_avx2();
  if (avx2) return read_avx2(p, n, pf);
#endif
  (void)pf;
  return read_plain(p, n);
}

struct Buffer {
  unsigned char* p = nullptr;  // 2 MB aligned start
  std::size_t n = 0;           // usable bytes
  void* raw = nullptr;         // the allocation as made
  std::size_t raw_n = 0;
};

bool alloc_buffer(std::size_t n, bool huge, Buffer& b) {
  const std::size_t align = std::size_t{2} << 20;
  n = (n + align - 1) / align * align;
#if defined(PELIAS_MEMBW_LINUX)
  void* raw = mmap(nullptr, n + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) return false;
  b.raw = raw;
  b.raw_n = n + align;
  b.p = reinterpret_cast<unsigned char*>((reinterpret_cast<std::uintptr_t>(raw) + align - 1) / align * align);
  (void)madvise(b.p, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
#else
  (void)huge;
  b.raw = std::malloc(n + align);
  if (!b.raw) return false;
  b.raw_n = n + align;
  b.p = reinterpret_cast<unsigned char*>((reinterpret_cast<std::uintptr_t>(b.raw) + align - 1) / align * align);
#endif
  b.n = n;
  std::memset(b.p, 0x5a, n);  // touch every page so the reads hit RAM, not first-touch faults
  return true;
}

void free_buffer(Buffer& b) {
#if defined(PELIAS_MEMBW_LINUX)
  (void)munmap(b.raw, b.raw_n);
#else
  std::free(b.raw);
#endif
  b = Buffer{};
}

long long anon_huge_kib() {
#if defined(PELIAS_MEMBW_LINUX)
  std::ifstream f("/proc/meminfo");
  std::string k;
  long long v;
  std::string unit;
  while (f >> k >> v) {
    std::getline(f, unit);
    if (k == "AnonHugePages:") return v;
  }
#endif
  return -1;
}

// the user stand-in: wakes every 1 ms, then does a fixed small job (a 64 KiB checksum and a little integer work)
struct Fore {
  std::vector<double> late_us, work_us;
};

void foreground(std::atomic<bool>& stop, Fore& out, int samples) {
  std::vector<unsigned char> buf(64 * 1024, 1);
  for (int s = 0; s < samples && !stop.load(std::memory_order_relaxed); ++s) {
    const auto want = Clock::now() + std::chrono::microseconds(1000);
    std::this_thread::sleep_until(want);
    const auto woke = Clock::now();
    std::uint64_t h = 1469598103934665603ull;
    for (int rep = 0; rep < 4; ++rep)
      for (std::size_t i = 0; i < buf.size(); i += 8) {
        std::uint64_t x;
        std::memcpy(&x, buf.data() + i, 8);
        h = (h ^ x) * 1099511628211ull;
      }
    buf[h & 0xffff] ^= 1;
    const auto done = Clock::now();
    out.late_us.push_back(us_since(want, woke));
    out.work_us.push_back(us_since(woke, done));
  }
}

double pct(std::vector<double> v, double p) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())))];
}

double run_cell(const Buffer& b, int threads, bool pf, int passes, const std::vector<int>& order) {
  const std::size_t slice = b.n / static_cast<std::size_t>(threads) / 128 * 128;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<double> took(static_cast<std::size_t>(threads), 0.0);
  std::atomic<std::uint64_t> sink{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      pin_to(order[static_cast<std::size_t>(t) % order.size()]);
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      const auto t0 = Clock::now();
      std::uint64_t acc = 0;
      for (int p = 0; p < passes; ++p) acc ^= read_slice(b.p + static_cast<std::size_t>(t) * slice, slice, pf);
      took[static_cast<std::size_t>(t)] = us_since(t0, Clock::now());
      sink.fetch_xor(acc);
    });
  while (ready.load() < threads) std::this_thread::yield();
  go.store(true, std::memory_order_release);
  for (auto& th : pool) th.join();
  const double us = *std::max_element(took.begin(), took.end());
  return static_cast<double>(slice) * threads * passes / (us * 1e3);  // GB/s
}

}  // namespace

bool membw_run(const MemBwOptions& opt, MemBwResult& out, std::string& error) {
  out = MemBwResult{};
  out.logical_cpus = static_cast<int>(std::thread::hardware_concurrency());
  const std::vector<int> order = cpu_order(out.physical_cores);
#if defined(PELIAS_MEMBW_LINUX)
  {
    std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
    std::getline(f, out.thp_mode);
  }
#else
  out.note = "huge pages and pinning are measured on Linux / WSL only in this version";
#endif
  // the idle machine: the foreground job alone
  {
    std::atomic<bool> stop{false};
    Fore f;
    foreground(stop, f, 400);
    out.idle_wake_p50_us = pct(f.late_us, 0.5);
    out.idle_wake_p99_us = pct(f.late_us, 0.99);
    out.idle_work_us = pct(f.work_us, 0.5);
  }
  for (const bool huge : {false, true}) {
    Buffer b;
    if (!alloc_buffer(opt.bytes, huge, b)) {
      error = "could not allocate the test buffer";
      return false;
    }
    if (huge) out.huge_kib_seen = anon_huge_kib();
    for (const bool pf : {false, true})
      for (const int th : opt.threads) {
        if (th < 1) continue;
        MemBwCell c;
        c.threads = th;
        c.huge = huge;
        c.prefetch = pf;
        const bool fg = opt.foreground && !huge && !pf;
        std::atomic<bool> stop{false};
        Fore f;
        std::thread fore;
        if (fg) fore = std::thread([&] { foreground(stop, f, 1 << 30); });
        std::vector<double> rates;
        for (int r = 0; r < opt.reps; ++r) rates.push_back(run_cell(b, th, pf, opt.passes, order));
        if (fg) {
          stop.store(true);
          fore.join();
          c.wake_p50_us = pct(f.late_us, 0.5);
          c.wake_p99_us = pct(f.late_us, 0.99);
          const double w = pct(f.work_us, 0.5);
          c.work_ratio = out.idle_work_us > 0 && w > 0 ? w / out.idle_work_us : -1.0;
        }
        std::sort(rates.begin(), rates.end());
        c.gbps = rates[rates.size() / 2];
        out.cells.push_back(c);
      }
    free_buffer(b);
  }
  return true;
}

}  // namespace pelias::os

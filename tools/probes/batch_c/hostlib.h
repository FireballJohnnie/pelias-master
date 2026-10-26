// hostlib.h - host-only helpers for the pelias probe suite (no CUDA in this file).
// C++20. Linux (WSL2 / native) and Windows. Everything OS-specific is in this file.
// Rules followed: RAII, no raw new/delete, no C-style casts, all warnings on.
#pragma once

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Implemented in cpu_bw.cpp (compiled with AVX2 enabled).
uint64_t cpu_read_sum_avx2(const void* p, size_t bytes);  // p 32-byte aligned, bytes multiple of 256
bool cpu_has_avx2();

namespace pp {

inline constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
inline constexpr double kMiB = 1024.0 * 1024.0;

// ---------------------------------------------------------------- time
inline double now_s() {
  using C = std::chrono::steady_clock;
  return std::chrono::duration<double>(C::now().time_since_epoch()).count();
}

inline std::string iso_now() {
  std::time_t t = std::time(nullptr);
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

// ---------------------------------------------------------------- JSON writer
class JW {
  std::string s_;
  bool first_ = true;

  void sep() {
    if (!first_) s_ += ',';
    first_ = false;
  }
  static void esc(std::string& o, std::string_view v) {
    for (const char ch : v) {
      const auto u = static_cast<unsigned char>(ch);
      switch (ch) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
          if (u < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", u);
            o += b;
          } else {
            o += ch;
          }
      }
    }
  }
  void key(std::string_view k) {
    sep();
    s_ += '"';
    esc(s_, k);
    s_ += "\":";
  }

 public:
  static std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    std::snprintf(b, sizeof b, "%.10g", v);
    return b;
  }

  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  JW& kv(std::string_view k, T v) {
    key(k);
    if constexpr (std::is_same_v<T, bool>) {
      s_ += v ? "true" : "false";
    } else if constexpr (std::is_floating_point_v<T>) {
      s_ += num(static_cast<double>(v));
    } else {
      s_ += std::to_string(v);
    }
    return *this;
  }
  JW& kv(std::string_view k, const std::string& v) {
    key(k);
    s_ += '"';
    esc(s_, v);
    s_ += '"';
    return *this;
  }
  JW& kv(std::string_view k, const char* v) { return kv(k, std::string(v)); }
  JW& kv_null(std::string_view k) {
    key(k);
    s_ += "null";
    return *this;
  }
  JW& kv_raw(std::string_view k, const std::string& raw) {
    key(k);
    s_ += raw;
    return *this;
  }
  JW& kv_obj(std::string_view k, const JW& o) { return kv_raw(k, o.obj()); }
  JW& arr_num(std::string_view k, const std::vector<double>& v) {
    key(k);
    s_ += '[';
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) s_ += ',';
      s_ += num(v[i]);
    }
    s_ += ']';
    return *this;
  }
  JW& arr_str(std::string_view k, const std::vector<std::string>& v) {
    key(k);
    s_ += '[';
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) s_ += ',';
      s_ += '"';
      esc(s_, v[i]);
      s_ += '"';
    }
    s_ += ']';
    return *this;
  }
  JW& arr_obj(std::string_view k, const std::vector<JW>& v) {
    key(k);
    s_ += '[';
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) s_ += ',';
      s_ += v[i].obj();
    }
    s_ += ']';
    return *this;
  }
  [[nodiscard]] std::string body() const { return s_; }
  [[nodiscard]] std::string obj() const { return "{" + s_ + "}"; }
  [[nodiscard]] bool empty() const { return s_.empty(); }
};

// ---------------------------------------------------------------- stats
inline double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
inline double vmean(const std::vector<double>& v) {
  if (v.empty()) return std::nan("");
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
inline double vmin(const std::vector<double>& v) {
  return v.empty() ? std::nan("") : *std::min_element(v.begin(), v.end());
}
inline double vmax(const std::vector<double>& v) {
  return v.empty() ? std::nan("") : *std::max_element(v.begin(), v.end());
}
inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}
inline JW stat_obj(const std::vector<double>& v) {
  JW j;
  j.kv("n", v.size()).kv("min", vmin(v)).kv("median", median(v)).kv("mean", vmean(v)).kv("max", vmax(v));
  return j;
}

// ---------------------------------------------------------------- args
class Args {
  std::map<std::string, std::string> kv_;
  std::vector<std::string> pos_;

 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string t = argv[i];
      if (t.rfind("--", 0) == 0) {
        const std::string k = t.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
          kv_[k] = argv[++i];
        } else {
          kv_[k] = "1";
        }
      } else {
        pos_.push_back(t);
      }
    }
  }
  [[nodiscard]] bool has(const std::string& k) const { return kv_.count(k) != 0; }
  [[nodiscard]] std::string get(const std::string& k, const std::string& d = "") const {
    const auto it = kv_.find(k);
    return it == kv_.end() ? d : it->second;
  }
  [[nodiscard]] double num(const std::string& k, double d) const {
    const auto it = kv_.find(k);
    return it == kv_.end() ? d : std::strtod(it->second.c_str(), nullptr);
  }
  [[nodiscard]] long long inum(const std::string& k, long long d) const {
    const auto it = kv_.find(k);
    return it == kv_.end() ? d : std::strtoll(it->second.c_str(), nullptr, 10);
  }
  [[nodiscard]] const std::vector<std::string>& pos() const { return pos_; }
};

// ---------------------------------------------------------------- text / process helpers
inline std::string read_text(const std::string& path, size_t max_bytes = 1u << 20) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::string s(max_bytes, '\0');
  f.read(s.data(), static_cast<std::streamsize>(max_bytes));
  s.resize(static_cast<size_t>(f.gcount()));
  return s;
}

inline std::string run_cmd(const std::string& cmd, size_t max_bytes = 1u << 20) {
#ifdef _WIN32
  FILE* p = _popen(cmd.c_str(), "r");
#else
  FILE* p = popen(cmd.c_str(), "r");
#endif
  if (!p) return {};
  std::string out;
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof buf, p)) > 0 && out.size() < max_bytes) out.append(buf, n);
#ifdef _WIN32
  _pclose(p);
#else
  pclose(p);
#endif
  return out;
}

inline std::string trim(std::string s) {
  const auto ws = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && ws(static_cast<unsigned char>(s.back()))) s.pop_back();
  size_t i = 0;
  while (i < s.size() && ws(static_cast<unsigned char>(s[i]))) ++i;
  return s.substr(i);
}

inline bool path_exists(const std::string& p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec);
}

// ---------------------------------------------------------------- dynamic libraries
inline void* lib_open(std::initializer_list<const char*> names) {
  for (const char* n : names) {
#ifdef _WIN32
    if (HMODULE h = LoadLibraryA(n)) return reinterpret_cast<void*>(h);
#else
    if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
#endif
  }
  return nullptr;
}
inline void* lib_sym(void* h, const char* name) {
  if (!h) return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(h), name));
#else
  return dlsym(h, name);
#endif
}
inline void lib_close(void* h) {
  if (!h) return;
#ifdef _WIN32
  FreeLibrary(reinterpret_cast<HMODULE>(h));
#else
  dlclose(h);
#endif
}

// ---------------------------------------------------------------- host memory
struct HostMem {
  uint64_t total_kib = 0, avail_kib = 0, swap_total_kib = 0, swap_free_kib = 0;
  uint64_t anon_huge_kib = 0, swapin_pages = 0;  // Linux only
  [[nodiscard]] double avail_gib() const { return static_cast<double>(avail_kib) / (1024.0 * 1024.0); }
  [[nodiscard]] double total_gib() const { return static_cast<double>(total_kib) / (1024.0 * 1024.0); }
};

inline HostMem host_mem() {
  HostMem m;
#ifdef _WIN32
  MEMORYSTATUSEX s{};
  s.dwLength = sizeof s;
  if (GlobalMemoryStatusEx(&s)) {
    m.total_kib = s.ullTotalPhys / 1024;
    m.avail_kib = s.ullAvailPhys / 1024;
    m.swap_total_kib = s.ullTotalPageFile > s.ullTotalPhys ? (s.ullTotalPageFile - s.ullTotalPhys) / 1024 : 0;
    m.swap_free_kib = s.ullAvailPageFile > s.ullAvailPhys ? (s.ullAvailPageFile - s.ullAvailPhys) / 1024 : 0;
  }
#else
  std::ifstream f("/proc/meminfo");
  std::string line;
  while (std::getline(f, line)) {
    char key[64];
    unsigned long long v = 0;
    if (std::sscanf(line.c_str(), "%63[^:]: %llu", key, &v) == 2) {
      const std::string k = key;
      if (k == "MemTotal") m.total_kib = v;
      else if (k == "MemAvailable") m.avail_kib = v;
      else if (k == "SwapTotal") m.swap_total_kib = v;
      else if (k == "SwapFree") m.swap_free_kib = v;
      else if (k == "AnonHugePages") m.anon_huge_kib = v;
    }
  }
  std::ifstream g("/proc/vmstat");
  while (std::getline(g, line)) {
    char key[64];
    unsigned long long v = 0;
    if (std::sscanf(line.c_str(), "%63s %llu", key, &v) == 2 && std::string(key) == "pswpin") {
      m.swapin_pages = v;
      break;
    }
  }
#endif
  return m;
}

// ---------------------------------------------------------------- platform facts
// RLIMIT_MEMLOCK in bytes (0 = unlimited or unknown; Windows has no such limit).
inline uint64_t memlock_limit_bytes() {
#ifdef _WIN32
  return 0;
#else
  struct rlimit rl {};
  if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0) return 0;
  return rl.rlim_cur == RLIM_INFINITY ? 0 : static_cast<uint64_t>(rl.rlim_cur);
#endif
}

inline bool is_wsl() {
#ifdef _WIN32
  return false;
#else
  std::string v = read_text("/proc/version", 4096);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v.find("microsoft") != std::string::npos || v.find("wsl") != std::string::npos;
#endif
}

inline std::string platform_name() {
#ifdef _WIN32
  return "windows-native";
#else
  return is_wsl() ? "wsl2" : "linux-native";
#endif
}

inline std::string cpu_model() {
#ifdef _WIN32
  const char* e = std::getenv("PROCESSOR_IDENTIFIER");
  return e ? e : "unknown";
#else
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) return trim(line.substr(pos + 1));
    }
  }
  return "unknown";
#endif
}

inline std::string os_text() {
#ifdef _WIN32
  return trim(run_cmd("cmd /c ver"));
#else
  std::string pretty;
  std::ifstream f("/etc/os-release");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("PRETTY_NAME=", 0) == 0) {
      pretty = line.substr(12);
      pretty.erase(std::remove(pretty.begin(), pretty.end(), '"'), pretty.end());
    }
  }
  return pretty + " | kernel " + trim(read_text("/proc/sys/kernel/osrelease", 256));
#endif
}

inline unsigned hw_threads() { return std::max(1u, std::thread::hardware_concurrency()); }

// Where does this path live? (Linux: longest matching mount in /proc/mounts; Windows: volume info.)
struct FsInfo {
  std::string mount, fstype, device;
};
inline FsInfo fs_info(const std::string& path) {
  FsInfo r;
  std::error_code ec;
  const std::string full = std::filesystem::weakly_canonical(path, ec).string();
#ifdef _WIN32
  char vol[MAX_PATH] = {0};
  char fsn[MAX_PATH] = {0};
  if (GetVolumePathNameA(full.c_str(), vol, MAX_PATH)) {
    r.mount = vol;
    r.device = vol;
    GetVolumeInformationA(vol, nullptr, 0, nullptr, nullptr, nullptr, fsn, MAX_PATH);
    r.fstype = fsn;
  }
#else
  std::ifstream f("/proc/mounts");
  std::string line;
  size_t best = 0;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string dev, mnt, fst;
    ss >> dev >> mnt >> fst;
    if (mnt.empty()) continue;
    const bool prefix = full.rfind(mnt, 0) == 0 && (mnt == "/" || full.size() == mnt.size() || full[mnt.size()] == '/');
    if (prefix && mnt.size() >= best) {
      best = mnt.size();
      r.mount = mnt;
      r.fstype = fst;
      r.device = dev;
    }
  }
#endif
  return r;
}

// ---------------------------------------------------------------- aligned memory (RAII)
struct AlignedBuf {
  void* p = nullptr;
  size_t bytes = 0;
  AlignedBuf() = default;
  AlignedBuf(size_t align, size_t n) { alloc(align, n); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
  AlignedBuf(AlignedBuf&& o) noexcept : p(o.p), bytes(o.bytes) {
    o.p = nullptr;
    o.bytes = 0;
  }
  AlignedBuf& operator=(AlignedBuf&& o) noexcept {
    if (this != &o) {
      release();
      p = o.p;
      bytes = o.bytes;
      o.p = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
  ~AlignedBuf() { release(); }
  bool alloc(size_t align, size_t n) {
    release();
#ifdef _WIN32
    p = _aligned_malloc(n, align);
#else
    if (posix_memalign(&p, align, n) != 0) p = nullptr;
#endif
    bytes = p ? n : 0;
    return p != nullptr;
  }
  void release() {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
    p = nullptr;
    bytes = 0;
  }
  [[nodiscard]] unsigned char* u8() const { return static_cast<unsigned char*>(p); }
};

// Touch every page: write one byte per 4 KiB, then (optionally) fill the whole range.
inline void touch_pages(void* p, size_t bytes) {
  auto* b = static_cast<volatile unsigned char*>(p);
  for (size_t i = 0; i < bytes; i += 4096) b[i] = static_cast<unsigned char>(i >> 12);
}

// Deterministic 64-bit pattern fill, so a later checksum pass can be compared with the first.
inline void fill_pattern(void* p, size_t bytes, uint64_t seed) {
  auto* w = static_cast<uint64_t*>(p);
  const size_t n = bytes / 8;
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (size_t i = 0; i < n; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    w[i] = x;
  }
}

// ---------------------------------------------------------------- file read paths (P-M4)
struct ReadResult {
  bool ok = false;
  std::string err;
  double secs = 0;
  uint64_t bytes = 0;
  uint64_t sum = 0;
  [[nodiscard]] double gbps() const { return secs > 0 ? static_cast<double>(bytes) / secs / 1e9 : 0.0; }
};

inline bool is_root() {
#ifdef _WIN32
  return false;
#else
  return geteuid() == 0;
#endif
}

#ifndef _WIN32
// cold = ask the kernel to drop this file's cached pages first (works without root for clean pages).
inline ReadResult read_file_seq(const std::string& path, size_t chunk, uint64_t limit, bool cold) {
  ReadResult r;
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    r.err = "open failed";
    return r;
  }
  if (cold) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  AlignedBuf buf(4096, chunk);
  if (!buf.p) {
    close(fd);
    r.err = "alloc failed";
    return r;
  }
  const double t0 = now_s();
  uint64_t total = 0, sum = 0;
  while (limit == 0 || total < limit) {
    size_t want = chunk;
    if (limit != 0 && limit - total < want) want = static_cast<size_t>(limit - total);
    const ssize_t n = read(fd, buf.p, want);
    if (n < 0) {
      r.err = "read failed";
      close(fd);
      return r;
    }
    if (n == 0) break;
    const auto* w = static_cast<const uint64_t*>(buf.p);
    for (size_t i = 0; i < static_cast<size_t>(n) / 8; i += 64) sum ^= w[i];
    total += static_cast<uint64_t>(n);
  }
  r.secs = now_s() - t0;
  r.bytes = total;
  r.sum = sum;
  r.ok = true;
  close(fd);
  return r;
}

inline ReadResult read_file_mmap(const std::string& path, uint64_t limit, bool cold) {
  ReadResult r;
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    r.err = "open failed";
    return r;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    close(fd);
    r.err = "fstat failed";
    return r;
  }
  uint64_t len = static_cast<uint64_t>(st.st_size);
  if (limit != 0 && limit < len) len = limit;
  if (cold) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  void* m = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m == MAP_FAILED) {
    close(fd);
    r.err = "mmap failed";
    return r;
  }
  madvise(m, len, MADV_SEQUENTIAL);
  const double t0 = now_s();
  uint64_t sum = 0;
  const auto* b = static_cast<const unsigned char*>(m);
  for (uint64_t i = 0; i < len; i += 4096) sum ^= b[i];
  r.secs = now_s() - t0;
  r.bytes = len;
  r.sum = sum;
  r.ok = true;
  munmap(m, len);
  close(fd);
  return r;
}

inline bool drop_caches_if_root() {
  if (!is_root()) return false;
  std::ofstream f("/proc/sys/vm/drop_caches");
  if (!f) return false;
  f << "3\n";
  return static_cast<bool>(f);
}
#else
inline ReadResult read_file_seq(const std::string& path, size_t chunk, uint64_t limit, bool cold) {
  ReadResult r;
  const DWORD flags = FILE_FLAG_SEQUENTIAL_SCAN | (cold ? FILE_FLAG_NO_BUFFERING : 0);
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    r.err = "CreateFile failed";
    return r;
  }
  AlignedBuf buf(4096, chunk);
  if (!buf.p) {
    CloseHandle(h);
    r.err = "alloc failed";
    return r;
  }
  const double t0 = now_s();
  uint64_t total = 0, sum = 0;
  while (limit == 0 || total < limit) {
    DWORD n = 0;
    if (!ReadFile(h, buf.p, static_cast<DWORD>(chunk), &n, nullptr)) {
      r.err = "ReadFile failed";
      CloseHandle(h);
      return r;
    }
    if (n == 0) break;
    const auto* w = static_cast<const uint64_t*>(buf.p);
    for (size_t i = 0; i < n / 8; i += 64) sum ^= w[i];
    total += n;
  }
  r.secs = now_s() - t0;
  r.bytes = total;
  r.sum = sum;
  r.ok = true;
  CloseHandle(h);
  return r;
}
inline ReadResult read_file_mmap(const std::string& path, uint64_t limit, bool cold) {
  (void)cold;
  ReadResult r;
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    r.err = "CreateFile failed";
    return r;
  }
  LARGE_INTEGER sz{};
  GetFileSizeEx(h, &sz);
  uint64_t len = static_cast<uint64_t>(sz.QuadPart);
  if (limit != 0 && limit < len) len = limit;
  HANDLE map = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!map) {
    CloseHandle(h);
    r.err = "CreateFileMapping failed";
    return r;
  }
  const void* v = MapViewOfFile(map, FILE_MAP_READ, 0, 0, static_cast<SIZE_T>(len));
  if (!v) {
    CloseHandle(map);
    CloseHandle(h);
    r.err = "MapViewOfFile failed";
    return r;
  }
  const double t0 = now_s();
  uint64_t sum = 0;
  const auto* b = static_cast<const unsigned char*>(v);
  for (uint64_t i = 0; i < len; i += 4096) sum ^= b[i];
  r.secs = now_s() - t0;
  r.bytes = len;
  r.sum = sum;
  r.ok = true;
  UnmapViewOfFile(v);
  CloseHandle(map);
  CloseHandle(h);
  return r;
}
inline bool drop_caches_if_root() { return false; }
#endif

// ---------------------------------------------------------------- CPU bandwidth workers
// Read `bytes` of `p` once per call; returns checksum. Bytes must be a multiple of 256.
inline uint64_t cpu_read_once(const void* p, size_t bytes) { return cpu_read_sum_avx2(p, bytes); }

}  // namespace pp

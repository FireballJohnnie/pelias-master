// cudautil.h - CUDA RAII wrappers, run context, result writer. Plain C++ (needs cuda_runtime.h, no kernel syntax).
#pragma once

#include <cuda_runtime.h>

#include <cctype>
#include <functional>
#include <set>
#include <stdexcept>

#include "hostlib.h"
#include "kernels.h"
#include "nvml_dyn.h"

namespace pp {

// ---------------------------------------------------------------- errors
struct CudaError : std::runtime_error {
  cudaError_t code;
  CudaError(cudaError_t c, const std::string& m) : std::runtime_error(m), code(c) {}
};

inline std::string cuda_msg(cudaError_t e) {
  const char* s = cudaGetErrorString(e);
  return s ? s : "unknown CUDA error";
}

// Hard check: throws; the probe's top-level handler writes a failure result.
#define PP_CK(call)                                                                                              \
  do {                                                                                                           \
    const cudaError_t pp_e_ = (call);                                                                            \
    if (pp_e_ != cudaSuccess)                                                                                    \
      throw ::pp::CudaError(pp_e_, std::string(#call) + " -> " + ::pp::cuda_msg(pp_e_) + " (" + __FILE__ + ":" + \
                                       std::to_string(__LINE__) + ")");                                          \
  } while (0)

// Soft check: returns the error text ("" when fine) and clears the sticky-free error state.
inline std::string soft_err(cudaError_t e) {
  if (e == cudaSuccess) return "";
  (void)cudaGetLastError();
  return cuda_msg(e);
}

// ---------------------------------------------------------------- RAII
class Stream {
  cudaStream_t s_ = nullptr;

 public:
  Stream() { PP_CK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (s_) cudaStreamDestroy(s_);
  }
  operator cudaStream_t() const { return s_; }  // NOLINT: deliberate implicit conversion
  void sync() const { PP_CK(cudaStreamSynchronize(s_)); }
};

class Event {
  cudaEvent_t e_ = nullptr;

 public:
  Event() { PP_CK(cudaEventCreate(&e_)); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (e_) cudaEventDestroy(e_);
  }
  operator cudaEvent_t() const { return e_; }  // NOLINT
  void record(cudaStream_t s) const { PP_CK(cudaEventRecord(e_, s)); }
  void sync() const { PP_CK(cudaEventSynchronize(e_)); }
  static double ms(const Event& a, const Event& b) {
    float m = 0;
    PP_CK(cudaEventElapsedTime(&m, a.e_, b.e_));
    return static_cast<double>(m);
  }
};

class DevBuf {
  void* p_ = nullptr;
  size_t n_ = 0;

 public:
  DevBuf() = default;
  explicit DevBuf(size_t n) { PP_CK(cudaMalloc(&p_, n)); n_ = n; }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }
  ~DevBuf() { release(); }
  void release() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    n_ = 0;
  }
  cudaError_t try_alloc(size_t n) {
    release();
    const cudaError_t e = cudaMalloc(&p_, n);
    if (e != cudaSuccess) { p_ = nullptr; return e; }
    n_ = n;
    return cudaSuccess;
  }
  template <class T = void> [[nodiscard]] T* as() const { return static_cast<T*>(p_); }
  [[nodiscard]] size_t bytes() const { return n_; }
};

// Page-locked host memory from cudaHostAlloc.
class PinnedBuf {
  void* p_ = nullptr;
  size_t n_ = 0;

 public:
  PinnedBuf() = default;
  PinnedBuf(const PinnedBuf&) = delete;
  PinnedBuf& operator=(const PinnedBuf&) = delete;
  PinnedBuf(PinnedBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  PinnedBuf& operator=(PinnedBuf&& o) noexcept {
    if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }
  ~PinnedBuf() { release(); }
  void release() {
    if (p_) cudaFreeHost(p_);
    p_ = nullptr;
    n_ = 0;
  }
  cudaError_t try_alloc(size_t n, unsigned flags = cudaHostAllocDefault) {
    release();
    const cudaError_t e = cudaHostAlloc(&p_, n, flags);
    if (e != cudaSuccess) { p_ = nullptr; return e; }
    n_ = n;
    return cudaSuccess;
  }
  [[nodiscard]] void* ptr() const { return p_; }
  [[nodiscard]] size_t bytes() const { return n_; }
  [[nodiscard]] unsigned char* u8() const { return static_cast<unsigned char*>(p_); }
};

// Ordinary aligned memory registered with cudaHostRegister.
class RegisteredBuf {
  AlignedBuf mem_;
  bool reg_ = false;

 public:
  RegisteredBuf() = default;
  RegisteredBuf(const RegisteredBuf&) = delete;
  RegisteredBuf& operator=(const RegisteredBuf&) = delete;
  ~RegisteredBuf() { release(); }
  void release() {
    if (reg_ && mem_.p) cudaHostUnregister(mem_.p);
    reg_ = false;
    mem_.release();
  }
  cudaError_t try_alloc(size_t n) {
    release();
    if (!mem_.alloc(4096, n)) return cudaErrorMemoryAllocation;
    touch_pages(mem_.p, n);
    const cudaError_t e = cudaHostRegister(mem_.p, n, cudaHostRegisterDefault);
    if (e != cudaSuccess) { mem_.release(); return e; }
    reg_ = true;
    return cudaSuccess;
  }
  [[nodiscard]] void* ptr() const { return mem_.p; }
  [[nodiscard]] size_t bytes() const { return mem_.bytes; }
};

// ---------------------------------------------------------------- run context + result
struct Ctx {
  std::string out_dir = "./pelias_probe_out";
  std::string tag;
  std::string machine_sha = "unrecorded";
  std::string cmd;
  bool quick = false;
  Nvml nvml;
  int sm_count = 0, cc_major = 0, cc_minor = 0;
  std::string dev_name;
  size_t vram_total = 0;
  double t_start = now_s();
};

struct Result {
  std::string probe, label = "MEASURED", status = "ok", error, notes, headline;
  int pass = -1;  // -1 = record only (JSON null), 0 = fail, 1 = pass
  JW inputs, values, thresholds;
  std::vector<std::string> raw;
};

inline void init_cuda(Ctx& c) {
  PP_CK(cudaSetDevice(0));
  PP_CK(cudaFree(nullptr));  // creates the context
  cudaDeviceProp p{};
  PP_CK(cudaGetDeviceProperties(&p, 0));
  c.dev_name = p.name;
  c.sm_count = p.multiProcessorCount;
  c.cc_major = p.major;
  c.cc_minor = p.minor;
  c.vram_total = p.totalGlobalMem;
}

inline std::string unique_path(const std::string& dir, const std::string& stem, const std::string& ext) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::string p = (std::filesystem::path(dir) / (stem + ext)).string();
  for (int i = 2; std::filesystem::exists(p, ec) && i < 1000; ++i)
    p = (std::filesystem::path(dir) / (stem + "_" + std::to_string(i) + ext)).string();  // never overwrite
  return p;
}

inline std::string write_result(const Ctx& c, const Result& r, const std::string& stem) {
  JW j;
  j.kv("probe", r.probe).kv("date", iso_now()).kv("machine_state_sha", c.machine_sha).kv("label", r.label);
  j.kv("status", r.status);
  if (!r.error.empty()) j.kv("error", r.error);
  j.kv("platform", platform_name()).kv("tag", c.tag).kv("quick", c.quick).kv("duration_s", now_s() - c.t_start);
  j.kv_obj("inputs", r.inputs).kv_obj("values", r.values).kv_obj("thresholds", r.thresholds);
  if (r.pass < 0) j.kv_null("pass");
  else j.kv("pass", r.pass == 1);
  j.arr_str("raw", r.raw).kv("notes", r.notes);
  const std::string path = unique_path(c.out_dir, stem + (c.tag.empty() ? "" : "_" + c.tag), ".json");
  std::ofstream f(path, std::ios::binary);
  f << j.obj() << "\n";
  const char* verdict = r.status == "skipped" ? "SKIP " : (r.status != "ok" ? "ERROR" : (r.pass < 0 ? "REC  " : (r.pass == 1 ? "PASS " : "FAIL ")));
  std::printf("[%s] %-6s %s\n        -> %s\n", verdict, r.probe.c_str(), r.headline.c_str(), path.c_str());
  std::fflush(stdout);
  return path;
}

// ---------------------------------------------------------------- shared helpers
// An explicit --name always wins; otherwise quick mode picks the small value.
inline double qnum(const Ctx& c, const Args& a, const char* name, double quick_v, double full_v) {
  return a.has(name) ? a.num(name, full_v) : (c.quick ? quick_v : full_v);
}

inline int gpu_blocks(const Ctx& c, int per_sm) { return std::max(1, c.sm_count * per_sm); }

// Time one enqueue-and-wait on a stream with events. Returns milliseconds.
inline double time_stream_ms(cudaStream_t s, const std::function<void()>& enqueue) {
  Event a, b;
  a.record(s);
  enqueue();
  b.record(s);
  b.sync();
  return Event::ms(a, b);
}

inline double gbps(double bytes, double ms) { return ms > 0 ? bytes / (ms * 1e-3) / 1e9 : 0.0; }

// Pick an iteration count so one tensor-kernel launch lasts about target_ms.
inline unsigned calibrate_mma_iters(pk::Mma k, int blocks, float* sink, cudaStream_t s, double target_ms) {
  unsigned iters = 2000;
  for (int round = 0; round < 3; ++round) {
    const double ms = time_stream_ms(s, [&] { PP_CK(pk::launch_mma(k, blocks, iters, sink, s)); });
    const double scale = std::clamp(target_ms / std::max(ms, 1e-3), 0.05, 2000.0);
    const double next = std::clamp(static_cast<double>(iters) * scale, 100.0, 4.0e9);
    iters = static_cast<unsigned>(next);
    if (std::fabs(scale - 1.0) < 0.05) break;
  }
  return iters;
}

// Threads that read slices of one big host buffer (AVX2 loads) until told to stop.
struct CpuRun {
  double secs = 0;
  uint64_t bytes = 0;
  [[nodiscard]] double gbps() const { return secs > 0 ? static_cast<double>(bytes) / secs / 1e9 : 0.0; }
};

inline CpuRun cpu_read_threads(const unsigned char* base, size_t bytes, int n, int max_passes, const std::atomic<bool>* stop,
                               std::atomic<uint64_t>* progress = nullptr) {
  CpuRun r;
  if (n <= 0 || !cpu_has_avx2()) return r;
  constexpr size_t kChunk = 8u << 20;  // progress is reported every 8 MiB
  const size_t slice = (bytes / static_cast<size_t>(n)) / 256 * 256;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<uint64_t> got(static_cast<size_t>(n), 0);
  std::vector<double> fin(static_cast<size_t>(n), 0.0);
  std::vector<std::thread> th;
  for (int i = 0; i < n; ++i) {
    th.emplace_back([&, i] {
      const unsigned char* p = base + static_cast<size_t>(i) * slice;
      volatile uint64_t sink = 0;
      ready.fetch_add(1);
      while (!go.load()) std::this_thread::yield();
      int pass = 0;
      bool halt = false;
      while (!halt && (max_passes <= 0 || pass < max_passes)) {
        for (size_t off = 0; off < slice; off += kChunk) {
          if (stop && stop->load()) { halt = true; break; }
          const size_t len = std::min(kChunk, slice - off);
          sink = sink + cpu_read_sum_avx2(p + off, len);
          got[static_cast<size_t>(i)] += len;
          if (progress) progress->fetch_add(len, std::memory_order_relaxed);
        }
        ++pass;
      }
      fin[static_cast<size_t>(i)] = now_s();
    });
  }
  while (ready.load() < n) std::this_thread::yield();
  const double t0 = now_s();
  go = true;
  for (auto& t : th) t.join();
  r.secs = *std::max_element(fin.begin(), fin.end()) - t0;
  r.bytes = std::accumulate(got.begin(), got.end(), uint64_t{0});
  return r;
}

}  // namespace pp

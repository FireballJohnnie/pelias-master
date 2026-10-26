// nvml_dyn.h - NVML loaded at run time (no nvml.h, no link dependency) + a background sampler.
// Works with the WSL2 library (/usr/lib/wsl/lib/libnvidia-ml.so.1) and nvml.dll on Windows.
// Every call is optional: an unsupported query becomes "no value", never a crash.
#pragma once

#include "hostlib.h"

namespace pp {

// Opaque handle and the few structs we use (layouts from NVIDIA's public NVML API).
using NvmlHandle = void*;
struct NvmlMemory {
  unsigned long long total = 0, free = 0, used = 0;
};
struct NvmlBar1 {
  unsigned long long total = 0, free = 0, used = 0;
};
struct NvmlUtil {
  unsigned gpu = 0, memory = 0;
};

inline std::string throttle_names(unsigned long long m) {
  static const struct {
    unsigned long long bit;
    const char* name;
  } tbl[] = {{0x1, "gpu_idle"},         {0x2, "applications_clocks_setting"}, {0x4, "sw_power_cap"},
             {0x8, "hw_slowdown"},     {0x10, "sync_boost"},                 {0x20, "sw_thermal_slowdown"},
             {0x40, "hw_thermal_slowdown"}, {0x80, "hw_power_brake_slowdown"}, {0x100, "display_clock_setting"}};
  std::string s;
  for (const auto& e : tbl) {
    if (m & e.bit) {
      if (!s.empty()) s += '|';
      s += e.name;
    }
  }
  return s.empty() ? "none" : s;
}
inline constexpr unsigned long long kThermalBits = 0x20ull | 0x40ull;  // sw + hw thermal slowdown

struct Sample {
  double t = 0;  // steady-clock seconds (pp::now_s)
  unsigned sm_mhz = 0, mem_mhz = 0, temp_c = 0, util_gpu = 0;
  double power_w = -1;
  unsigned long long throttle = 0;
  bool have_throttle = false;
  double vram_used_mib = -1;
};

class Nvml {
  void* lib_ = nullptr;
  NvmlHandle dev_ = nullptr;
  bool ok_ = false;

  using FInit = int (*)();
  using FHandle = int (*)(unsigned, NvmlHandle*);
  using FClock = int (*)(NvmlHandle, int, unsigned*);
  using FU1 = int (*)(NvmlHandle, unsigned*);
  using FTemp = int (*)(NvmlHandle, int, unsigned*);
  using FU64 = int (*)(NvmlHandle, unsigned long long*);
  using FMem = int (*)(NvmlHandle, NvmlMemory*);
  using FBar1 = int (*)(NvmlHandle, NvmlBar1*);
  using FName = int (*)(NvmlHandle, char*, unsigned);
  using FDrv = int (*)(char*, unsigned);
  using FUtil = int (*)(NvmlHandle, NvmlUtil*);

  FInit shutdown_ = nullptr;
  FClock clock_ = nullptr, maxclock_ = nullptr;
  FTemp temp_ = nullptr;
  FU1 power_ = nullptr, plimit_ = nullptr, plimit_enf_ = nullptr, pcie_gen_ = nullptr, pcie_w_ = nullptr,
      pcie_gen_max_ = nullptr, pcie_w_max_ = nullptr, pstate_ = nullptr;
  FU64 thr_ = nullptr;
  FMem mem_ = nullptr;
  FBar1 bar1_ = nullptr;
  FName name_ = nullptr;
  FDrv drv_ = nullptr;
  FUtil util_ = nullptr;

  template <class F>
  F sym(const char* n) const {
    return reinterpret_cast<F>(lib_sym(lib_, n));
  }

 public:
  Nvml() = default;
  Nvml(const Nvml&) = delete;
  Nvml& operator=(const Nvml&) = delete;
  ~Nvml() { close(); }

  bool open() {
    if (ok_) return true;
#ifdef _WIN32
    lib_ = lib_open({"nvml.dll", "C:\\Windows\\System32\\nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"});
#else
    lib_ = lib_open({"libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1", "libnvidia-ml.so"});
#endif
    if (!lib_) return false;
    const auto init = sym<FInit>("nvmlInit_v2");
    const auto byidx = sym<FHandle>("nvmlDeviceGetHandleByIndex_v2");
    if (!init || !byidx || init() != 0 || byidx(0, &dev_) != 0) {
      lib_close(lib_);
      lib_ = nullptr;
      return false;
    }
    shutdown_ = sym<FInit>("nvmlShutdown");
    clock_ = sym<FClock>("nvmlDeviceGetClockInfo");
    maxclock_ = sym<FClock>("nvmlDeviceGetMaxClockInfo");
    temp_ = sym<FTemp>("nvmlDeviceGetTemperature");
    power_ = sym<FU1>("nvmlDeviceGetPowerUsage");
    plimit_ = sym<FU1>("nvmlDeviceGetPowerManagementLimit");
    plimit_enf_ = sym<FU1>("nvmlDeviceGetEnforcedPowerLimit");
    pcie_gen_ = sym<FU1>("nvmlDeviceGetCurrPcieLinkGeneration");
    pcie_w_ = sym<FU1>("nvmlDeviceGetCurrPcieLinkWidth");
    pcie_gen_max_ = sym<FU1>("nvmlDeviceGetMaxPcieLinkGeneration");
    pcie_w_max_ = sym<FU1>("nvmlDeviceGetMaxPcieLinkWidth");
    pstate_ = sym<FU1>("nvmlDeviceGetPerformanceState");
    thr_ = sym<FU64>("nvmlDeviceGetCurrentClocksEventReasons");
    if (!thr_) thr_ = sym<FU64>("nvmlDeviceGetCurrentClocksThrottleReasons");
    mem_ = sym<FMem>("nvmlDeviceGetMemoryInfo");
    bar1_ = sym<FBar1>("nvmlDeviceGetBAR1MemoryInfo");
    name_ = sym<FName>("nvmlDeviceGetName");
    drv_ = sym<FDrv>("nvmlSystemGetDriverVersion");
    util_ = sym<FUtil>("nvmlDeviceGetUtilizationRates");
    ok_ = true;
    return true;
  }
  void close() {
    if (ok_ && shutdown_) shutdown_();
    ok_ = false;
    lib_close(lib_);
    lib_ = nullptr;
  }
  [[nodiscard]] bool ok() const { return ok_; }

  [[nodiscard]] std::string name() const {
    char b[96] = {0};
    return (ok_ && name_ && name_(dev_, b, sizeof b) == 0) ? b : "";
  }
  [[nodiscard]] std::string driver() const {
    char b[96] = {0};
    return (ok_ && drv_ && drv_(b, sizeof b) == 0) ? b : "";
  }
  bool mem(NvmlMemory& m) const { return ok_ && mem_ && mem_(dev_, &m) == 0; }
  bool bar1(NvmlBar1& m) const { return ok_ && bar1_ && bar1_(dev_, &m) == 0; }
  bool clock(int type, unsigned& mhz) const { return ok_ && clock_ && clock_(dev_, type, &mhz) == 0; }  // 0 gfx, 1 sm, 2 mem
  bool max_clock(int type, unsigned& mhz) const { return ok_ && maxclock_ && maxclock_(dev_, type, &mhz) == 0; }
  bool power_limit_w(double& w) const {
    unsigned mw = 0;
    if (ok_ && plimit_enf_ && plimit_enf_(dev_, &mw) == 0) { w = mw / 1000.0; return true; }
    if (ok_ && plimit_ && plimit_(dev_, &mw) == 0) { w = mw / 1000.0; return true; }
    return false;
  }
  bool pcie(unsigned& gen, unsigned& width, unsigned& gen_max, unsigned& width_max) const {
    bool any = false;
    any |= ok_ && pcie_gen_ && pcie_gen_(dev_, &gen) == 0;
    any |= ok_ && pcie_w_ && pcie_w_(dev_, &width) == 0;
    any |= ok_ && pcie_gen_max_ && pcie_gen_max_(dev_, &gen_max) == 0;
    any |= ok_ && pcie_w_max_ && pcie_w_max_(dev_, &width_max) == 0;
    return any;
  }
  bool pstate(unsigned& p) const { return ok_ && pstate_ && pstate_(dev_, &p) == 0; }

  Sample sample() const {
    Sample s;
    s.t = now_s();
    if (!ok_) return s;
    unsigned v = 0;
    if (clock(1, v)) s.sm_mhz = v;
    if (clock(2, v)) s.mem_mhz = v;
    if (temp_ && temp_(dev_, 0, &v) == 0) s.temp_c = v;
    if (power_ && power_(dev_, &v) == 0) s.power_w = v / 1000.0;
    unsigned long long t = 0;
    if (thr_ && thr_(dev_, &t) == 0) {
      s.throttle = t;
      s.have_throttle = true;
    }
    NvmlUtil u;
    if (util_ && util_(dev_, &u) == 0) s.util_gpu = u.gpu;
    NvmlMemory m;
    if (mem(m)) s.vram_used_mib = static_cast<double>(m.used) / kMiB;
    return s;
  }
};

// Safety stop shared by every long probe: set when the card gets too hot.
inline std::atomic<bool> g_abort{false};
inline std::string g_abort_reason;
inline std::mutex g_abort_mu;
inline constexpr unsigned kAbortTempC = 87;

inline void request_abort(const std::string& why) {
  std::lock_guard<std::mutex> lk(g_abort_mu);
  if (!g_abort.exchange(true)) g_abort_reason = why;
}
inline std::string abort_reason() {
  std::lock_guard<std::mutex> lk(g_abort_mu);
  return g_abort_reason;
}

// Background sampler. start() ... stop() returns every sample taken.
class Sampler {
  const Nvml& nv_;
  std::thread th_;
  std::atomic<bool> run_{false};
  mutable std::mutex mu_;
  std::vector<Sample> v_;

 public:
  explicit Sampler(const Nvml& nv) : nv_(nv) {}
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;
  ~Sampler() { stop(); }

  void start(int interval_ms = 200) {
    if (run_.exchange(true) || !nv_.ok()) {
      return;
    }
    th_ = std::thread([this, interval_ms] {
      while (run_.load()) {
        const Sample s = nv_.sample();
        {
          std::lock_guard<std::mutex> lk(mu_);
          v_.push_back(s);
        }
        if (s.temp_c >= kAbortTempC) request_abort("GPU temperature reached " + std::to_string(s.temp_c) + " C");
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
      }
    });
  }
  std::vector<Sample> stop() {
    run_ = false;
    if (th_.joinable()) th_.join();
    std::lock_guard<std::mutex> lk(mu_);
    return v_;
  }
  [[nodiscard]] std::vector<Sample> snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return v_;
  }
};

// Samples with t0 <= t < t1.
inline std::vector<Sample> window(const std::vector<Sample>& all, double t0, double t1) {
  std::vector<Sample> r;
  for (const auto& s : all)
    if (s.t >= t0 && s.t < t1) r.push_back(s);
  return r;
}

// Summary of a window of samples. Values that were never readable are null.
inline JW summarize(const std::vector<Sample>& w) {
  JW j;
  j.kv("samples", w.size());
  if (w.empty()) return j;
  std::vector<double> sm, mem, tmp, pw;
  unsigned long long orm = 0;
  size_t thermal = 0, have_thr = 0;
  std::map<std::string, size_t> bits;
  for (const auto& s : w) {
    if (s.sm_mhz) sm.push_back(s.sm_mhz);
    if (s.mem_mhz) mem.push_back(s.mem_mhz);
    if (s.temp_c) tmp.push_back(s.temp_c);
    if (s.power_w >= 0) pw.push_back(s.power_w);
    if (s.have_throttle) {
      ++have_thr;
      orm |= s.throttle;
      if (s.throttle & kThermalBits) ++thermal;
      for (unsigned long long b = 1; b <= 0x100; b <<= 1)
        if (s.throttle & b) ++bits[throttle_names(b)];
    }
  }
  if (!sm.empty()) j.kv_obj("sm_clock_mhz", stat_obj(sm));
  if (!mem.empty()) j.kv_obj("mem_clock_mhz", stat_obj(mem));
  if (!tmp.empty()) j.kv_obj("temp_c", stat_obj(tmp));
  if (!pw.empty()) j.kv_obj("power_w", stat_obj(pw));
  if (have_thr) {
    j.kv("throttle_reasons_seen", throttle_names(orm));
    j.kv("thermal_throttle_fraction", static_cast<double>(thermal) / static_cast<double>(have_thr));
    JW frac;
    for (const auto& [name, n] : bits) frac.kv(name, static_cast<double>(n) / static_cast<double>(have_thr));
    j.kv_obj("throttle_fraction_by_reason", frac);
  } else {
    j.kv_null("throttle_reasons_seen");
  }
  return j;
}

// Decimated time series (at most max_points) for plotting later.
inline JW series(const std::vector<Sample>& all, double t_ref, size_t max_points = 400) {
  JW j;
  std::vector<double> t, sm, tmp, pw, thr;
  const size_t step = std::max<size_t>(1, all.size() / max_points);
  for (size_t i = 0; i < all.size(); i += step) {
    t.push_back(all[i].t - t_ref);
    sm.push_back(all[i].sm_mhz);
    tmp.push_back(all[i].temp_c);
    pw.push_back(all[i].power_w);
    thr.push_back(static_cast<double>(all[i].throttle));
  }
  j.arr_num("t_s", t).arr_num("sm_mhz", sm).arr_num("temp_c", tmp).arr_num("power_w", pw).arr_num("throttle_mask", thr);
  return j;
}

}  // namespace pp

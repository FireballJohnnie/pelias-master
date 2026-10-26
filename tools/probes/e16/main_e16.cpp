// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// main_e16.cpp - pelias_e16: probe P-E16, the CUDA doorbell with its flag in VRAM (masterwork 21.2, hypothesis H-G1).
//   pelias_e16 [--out DIR] [--rounds 20000] [--batch 1000] [--no-vulkan]
// A resident GPU thread and a CPU thread ring each other: the CPU writes ring n to the request word, the GPU sees it and
// writes n to the reply word, the CPU sees that. One round trip per ring, timed on the CPU. The request and reply words
// are each placed in pinned system RAM (mapped into the GPU) or in CPU-visible VRAM (Vulkan DEVICE_LOCAL | HOST_VISIBLE,
// reached by the CPU through the PCIe BAR, imported into CUDA), giving four placements, plus the launch-and-synchronize
// path as the reference. A placement whose ring is not answered within 250 ms is recorded as not completing (as the
// system-RAM flag of P-G0 did on Vulkan and Direct3D 12 write-back memory). Writes one JSON into --out.
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

#include "cudautil.h"
#include "e16_kernels.h"
#include "vk_extmem.h"

namespace {

using namespace pp;
using clk = std::chrono::steady_clock;

struct Loc {
  const char* name;
  void* host;      // CPU address
  uint32_t* dev;   // GPU address
};

inline void store(void* p, uint32_t v) {
  *static_cast<volatile uint32_t*>(p) = v;
  _mm_sfence();  // drain write-combining buffers (the BAR mapping is write-combined)
}
inline uint32_t load(const void* p) { return *static_cast<const volatile uint32_t*>(p); }

struct Stats {
  double median = 0, p99 = 0, mean = 0, min = 0, max = 0;
  size_t n = 0;
};
Stats stats(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.n = v.size();
  s.median = v[v.size() / 2];
  s.p99 = v[std::min(v.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(v.size())))];
  s.min = v.front();
  s.max = v.back();
  double t = 0;
  for (double x : v) t += x;
  s.mean = t / static_cast<double>(v.size());
  return s;
}
JW stats_json(const Stats& s) {
  JW j;
  j.kv("rounds_timed", static_cast<double>(s.n)).kv("median_us", s.median).kv("p99_us", s.p99).kv("mean_us", s.mean).kv("min_us", s.min).kv("max_us", s.max);
  return j;
}

// One placement: returns the timed rounds (first ring of each batch left out: it also waits for the kernel to start).
JW run_bell(const Loc& req, const Loc& rep, int rounds, int batch, uint32_t* status_dev, cudaStream_t s, Stats& out) {
  JW j;
  j.kv("request_word", req.name).kv("reply_word", rep.name);
  store(req.host, 0);
  store(rep.host, 0);
  std::vector<double> us;
  us.reserve(static_cast<size_t>(rounds));
  uint32_t n = 1;
  int done = 0;
  std::string failure;
  const auto patience = std::chrono::milliseconds(250);
  while (done < rounds && failure.empty()) {
    const int count = std::min(batch, rounds - done);
    PP_CK(cudaMemsetAsync(status_dev, 0, 4, s));
    PP_CK(e16::launch_bell(req.dev, rep.dev, n, static_cast<uint32_t>(count), 200000000ull, status_dev, s));
    // Windows (WDDM) batches launches and may hold this one until something flushes the queue; a query flushes it
    // without waiting (revision 3: without it no ring was ever answered on Windows, in any placement).
    (void)cudaStreamQuery(s);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    for (int i = 0; i < count; ++i, ++n) {
      const auto t0 = clk::now();
      store(req.host, n);
      uint32_t spins = 0;
      bool ok = true;
      while (load(rep.host) != n) {
        if ((++spins & 1023u) == 0 && clk::now() - t0 > patience) {
          ok = false;
          break;
        }
      }
      const auto t1 = clk::now();
      if (!ok) {
        failure = "ring " + std::to_string(n) + " not answered within 250 ms";
        break;
      }
      if (i > 0) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    if (!failure.empty()) {
      store(req.host, 0xFFFFFFF0u);  // nothing to answer; the GPU thread gives up after its own 200 ms
    }
    PP_CK(cudaStreamSynchronize(s));
    uint32_t st = 0;
    PP_CK(cudaMemcpy(&st, status_dev, 4, cudaMemcpyDeviceToHost));
    if (failure.empty() && st != 0xFFFFFFFFu) failure = "GPU thread stopped waiting at ring " + std::to_string(st);
    done += count;
  }
  out = stats(us);
  j.kv("completes", failure.empty()).kv("rounds_requested", rounds).kv_obj("round_trip", stats_json(out));
  if (!failure.empty()) j.kv("failure", failure);
  return j;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a(argc, argv);
  Ctx c;
  c.out_dir = a.get("out", "./pelias_e16_out");
  c.tag = a.get("tag", "");
  for (int i = 0; i < argc; ++i) c.cmd += std::string(i ? " " : "") + argv[i];
  const int rounds = static_cast<int>(a.inum("rounds", 20000));
  const int batch = static_cast<int>(a.inum("batch", 1000));
  Result r;
  r.probe = "P-E16";
  try {
    init_cuda(c);
    cudaDeviceProp prop{};
    PP_CK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t s;
    PP_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    uint32_t* status_dev = nullptr;
    PP_CK(cudaMalloc(&status_dev, 4));
    // pinned system RAM, mapped into the GPU: request at byte 0, reply at byte 128 (separate cache lines)
    void* sys = nullptr;
    PP_CK(cudaHostAlloc(&sys, 4096, cudaHostAllocMapped));
    std::memset(sys, 0, 4096);
    void* sys_dev = nullptr;
    PP_CK(cudaHostGetDevicePointer(&sys_dev, sys, 0));
    const Loc sys_req{"pinned system RAM", sys, static_cast<uint32_t*>(sys_dev)};
    const Loc sys_rep{"pinned system RAM", static_cast<char*>(sys) + 128, reinterpret_cast<uint32_t*>(static_cast<char*>(sys_dev) + 128)};

    // reference: launch an empty kernel and synchronize
    std::vector<double> ls;
    for (int i = 0; i < 2000; ++i) {
      const auto t0 = clk::now();
      PP_CK(e16::launch_empty(s));
      PP_CK(cudaStreamSynchronize(s));
      if (i >= 50) ls.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
    }
    const Stats sl = stats(ls);
    std::printf("  launch + synchronize        median %8.2f us  p99 %8.2f us\n", sl.median, sl.p99);
    std::fflush(stdout);

    std::vector<JW> rows;
    Stats st;
    JW base = run_bell(sys_req, sys_rep, rounds, batch, status_dev, s, st);
    const double base_median = st.median;
    rows.push_back(base);
    std::printf("  request sys  / reply sys    median %8.2f us  p99 %8.2f us\n", st.median, st.p99);
    std::fflush(stdout);

    e16::VkVram vk;
    std::string why;
    JW vkj;
    const bool have_vram = !a.has("no-vulkan") && vk.open(reinterpret_cast<const unsigned char*>(prop.uuid.bytes), 4096, why);
    if (have_vram) {
      std::memset(vk.host, 0, 4096);
      _mm_sfence();
      const Loc v_req{"CPU-visible VRAM (BAR)", vk.host, static_cast<uint32_t*>(vk.dev)};
      const Loc v_rep{"CPU-visible VRAM (BAR)", static_cast<char*>(vk.host) + 256, reinterpret_cast<uint32_t*>(static_cast<char*>(vk.dev) + 256)};
      vkj.kv("available", true).kv("vulkan_device", vk.vk_device_name).kv("host_visible_vram_heap_mib", static_cast<double>(vk.host_visible_vram_heap_bytes) / 1048576.0);
      vkj.kv("host_coherent", vk.coherent).kv("resizable_bar_inferred_on", vk.host_visible_vram_heap_bytes > (uint64_t{512} << 20));
      const Loc* pairs[3][2] = {{&v_req, &sys_rep}, {&sys_req, &v_rep}, {&v_req, &v_rep}};
      const char* labels[3] = {"request VRAM / reply sys  ", "request sys  / reply VRAM ", "request VRAM / reply VRAM "};
      for (int k = 0; k < 3; ++k) {
        JW row = run_bell(*pairs[k][0], *pairs[k][1], rounds, batch, status_dev, s, st);
        if (st.n) row.kv("median_over_system_ram_both", st.median / base_median).kv("speedup_vs_system_ram_both", base_median / st.median);
        rows.push_back(row);
        std::printf("  %s median %8.2f us  p99 %8.2f us%s\n", labels[k], st.median, st.p99, st.n ? "" : "  (did not complete)");
        std::fflush(stdout);
      }
    } else {
      vkj.kv("available", false).kv("reason", a.has("no-vulkan") ? std::string("--no-vulkan") : why);
      std::printf("  CPU-visible VRAM: unavailable (%s)\n", a.has("no-vulkan") ? "--no-vulkan" : why.c_str());
    }
    r.inputs.kv("rounds_per_placement", rounds).kv("rounds_per_kernel_launch", batch).kv("patience_ms", 250).kv("gpu", prop.name);
    r.inputs.kv("cpu_timer", "std::chrono::steady_clock").kv("gpu_side", "one thread; volatile loads; reply = volatile store between __threadfence_system()");
    r.values.kv_obj("launch_and_synchronize", stats_json(sl)).arr_obj("doorbell", rows).kv_obj("cpu_visible_vram", vkj);
    r.headline = "system RAM both " + std::to_string(base_median) + " us median" + (have_vram ? "; VRAM placements recorded" : "; VRAM unavailable");
    r.inputs.kv("launch_flush", "cudaStreamQuery right after each launch (WDDM batching)");
    r.notes = "Masterwork 21.2 H-G1. The first ring of each kernel launch is left out (it also waits for the launch). ReBAR state is inferred from the size of the CPU-visible VRAM heap.";
    write_result(c, r, "e16_doorbell");
    vk.close();
    PP_CK(cudaFreeHost(sys));
    PP_CK(cudaFree(status_dev));
    PP_CK(cudaStreamDestroy(s));
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    r.status = "error";
    r.error = ex.what();
    r.headline = ex.what();
    write_result(c, r, "e16_doorbell_error");
    return 2;
  }
}

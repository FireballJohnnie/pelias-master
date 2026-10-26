// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// main_c.cpp - pelias_batch_c: Batch C kernel-efficiency probes (trainer master 11.3, ruling D35).
//   pelias_batch_c <k1|k2|k2b|k3|q10|all|k2prof|list> [--quick] [--out DIR] [--tag NAME] [--ceiling-gib 10.7] [--cooldown-s 30]
//                  [--m N] [--bf16-peak T] [--int8-peak T] [--fp8-peak T] [--k3-mib N]
// Every probe writes one JSON into --out (never overwrites). "all" runs k1, k2, k3, q10 with a cooldown between them and
// writes batch_c_summary.json. Exit code: 0 ok, 2 error, 3 a probe failed its threshold.
#include <cstdio>
#include <thread>

#include "probes_c.h"

namespace {

using namespace pp;

void write_failure(const Ctx& c, const std::string& sub, const std::string& what) {
  Result r;
  r.probe = "BATCH-C " + sub;
  r.status = "error";
  r.error = what;
  r.headline = what;
  r.notes = "The probe stopped with an error before finishing; nothing here is a measurement.";
  write_result(c, r, sub + "_error");
}

void usage() {
  std::puts(
      "pelias_batch_c <command> [--quick] [--out DIR] [--tag NAME]\n"
      "  k1    P-K1  bf16 path (Q4_K -> bf16 scratch + tiled mma GEMM) against the P-H3 peak and cuBLASLt\n"
      "  k2    P-K2  int8 path (Q4_K unpacked in registers, mma s8, fp32 epilogue) against the int8 peak and the bf16 path\n"
      "  k2b   P-K2b v12: split-K on low-tile shapes, bf16 output, MoE experts at small M (grouped), layer-order L2 test\n"
      "  k3    P-K3  streaming Q4_K / Q6_K dequantization against a plain read of the same bytes\n"
      "  q10   P-Q10 fp8 (e4m3 / e5m2) data-gradient cosine against fp32\n"
      "  all   k1, k2, k2b, k3, q10 with --cooldown-s seconds between them (default 30)\n"
      "  k2prof  one launch of each GEMM kernel on one shape (--shape NAME, --m N) for Nsight Compute; not part of all\n"
      "options: --ceiling-gib 10.7  --m N (one row count)  --experts 256  --expert-m 64  --bf16-peak 65.2  --int8-peak 259.5  --fp8-peak 129.8  --k3-mib 1024");
}

using Fn = int (*)(Ctx&, const Args&);
struct Entry {
  const char* name;
  Fn fn;
};
const Entry kTable[] = {{"k1", p_k1}, {"k2", p_k2}, {"k2b", p_k2b}, {"k3", p_k3}, {"q10", p_q10}};
const Entry kExtra[] = {{"k2prof", p_k2prof}};  // not part of "all"

int run_one(Ctx& c, const Args& a, const Entry& e) {
  try {
    std::printf("== %s\n", e.name);
    std::fflush(stdout);
    return e.fn(c, a);
  } catch (const CudaError& ex) {
    std::fprintf(stderr, "CUDA error: %s\n", ex.what());
    (void)cudaGetLastError();
    write_failure(c, e.name, std::string("CUDA error: ") + ex.what());
    return 2;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    write_failure(c, e.name, ex.what());
    return 2;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args a(argc, argv);
  if (a.pos().empty() || a.has("help")) {
    usage();
    return a.pos().empty() ? 1 : 0;
  }
  Ctx c;
  const std::string sub = a.pos()[0];
  c.out_dir = a.get("out", "./pelias_batch_c_out");
  c.tag = a.get("tag", "");
  c.machine_sha = a.get("machine-sha", "unrecorded");
  c.quick = a.has("quick");
  for (int i = 0; i < argc; ++i) c.cmd += std::string(i ? " " : "") + argv[i];
  if (sub == "list") {
    for (const auto& e : kTable) std::printf("%s\n", e.name);
    return 0;
  }
  (void)c.nvml.open();
  if (sub != "all") {
    for (const auto& e : kTable)
      if (sub == e.name) return run_one(c, a, e);
    for (const auto& e : kExtra)
      if (sub == e.name) return run_one(c, a, e);
    std::fprintf(stderr, "unknown command '%s'\n", sub.c_str());
    usage();
    return 1;
  }
  const double cooldown = a.num("cooldown-s", 30.0);
  std::vector<JW> parts;
  int worst = 0;
  for (size_t i = 0; i < std::size(kTable); ++i) {
    if (i > 0 && cooldown > 0) {
      std::printf("   cooldown %.0f s\n", cooldown);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long long>(cooldown * 1000)));
    }
    JW p;
    p.kv("probe", kTable[i].name).kv_obj("state_at_start", state_now(c));
    c.t_start = now_s();
    const int rc = run_one(c, a, kTable[i]);
    p.kv("exit_code", rc).kv("seconds", now_s() - c.t_start);
    parts.push_back(p);
    worst = std::max(worst, rc);
    if (g_abort.load()) break;
  }
  Result r;
  r.probe = "BATCH-C";
  r.values.arr_obj("parts", parts).kv("cooldown_s", cooldown);
  r.pass = -1;
  r.headline = "exit codes per probe recorded; worst " + std::to_string(worst);
  r.notes = g_abort.load() ? "ABORTED: " + abort_reason() : "";
  write_result(c, r, "batch_c_summary");
  return worst;
}

// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// probes_c.h - Batch C (trainer master 11.3, ruling D35): P-K1, P-K2, P-K3, P-Q10 as one run.
//   P-K1  bf16 path: Q4_K weights dequantized to a bf16 scratch, then a tiled mma.sync GEMM; efficiency against the
//         P-H3 bf16 peak and against cuBLASLt on the same data (cuBLASLt is a test-only reference, never shipped).
//   P-K2  int8 path: activations to int8 per row per 32, Q4_K weights unpacked to int8 in registers, mma.sync s8 with a
//         per-sub-block fp32 epilogue; efficiency against the P-H3 int8 peak, speed against the bf16 path.
//   P-K3  streaming in-register dequantization of Q4_K and Q6_K (no mma) against a plain read of the same bytes.
//   P-Q10 fp8 (e4m3, e5m2) data gradients: cosine of dX = dY * W with dY rounded to fp8 per row, against fp32.
// Every sub-test refuses an allocation that would take the whole card above the VRAM ceiling (10.7 GiB, judged on the
// whole-card reading, so memory held by other programs counts against it), and records NVML clocks, temperature, power
// and VRAM before and after. Shapes are the model's real ones (trainer master 11.3).
#pragma once

#include <cublasLt.h>

#include <algorithm>
#include <iterator>
#include <functional>
#include <memory>
#include <vector>

#include "cudautil.h"
#include "kernels_c.h"
#include "kernels_tg2.h"

namespace pp {

// ---------------------------------------------------------------- shared
struct Shape {
  const char* name;
  int K, N;  // forward: Y[M,N] = X[M,K] * W[N,K]^T
};
// Trainer master 11.3: DeltaNet attn_qkv 2048->8192, attn_gate 2048->4096, ssm_out 4096->2048; attention attn_q
// 2048->8192 (joint q+gate), attn_k/attn_v 2048->512, attn_o 4096->2048; shared expert 2048<->512; routed expert gate/up
// 2048->512, down 512->2048; router 2048->256; lm_head 2048->248320. Equal shapes are measured once.
inline const Shape kShapes[] = {
    {"attn_qkv / attn_q (2048->8192)", 2048, 8192}, {"attn_gate (2048->4096)", 2048, 4096},
    {"ssm_out / attn_o (4096->2048)", 4096, 2048},  {"attn_k / attn_v / expert gate,up (2048->512)", 2048, 512},
    {"expert down (512->2048)", 512, 2048},         {"router (2048->256)", 2048, 256},
    {"lm_head (2048->248320)", 2048, 248320},
};

#define PP_LT(call)                                                                                     \
  do {                                                                                                  \
    const cublasStatus_t st_ = (call);                                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                                                   \
      throw std::runtime_error(std::string(#call) + " -> cuBLASLt status " + std::to_string(static_cast<int>(st_))); \
  } while (0)

inline double ceiling_gib(const Args& a) { return a.num("ceiling-gib", 10.7); }

// Whole-card VRAM in use now (CUDA's view: total - free).
inline double card_used_gib() {
  size_t f = 0, t = 0;
  PP_CK(cudaMemGetInfo(&f, &t));
  return static_cast<double>(t - f) / kGiB;
}

// True when `need` more bytes keep the whole card at or below the ceiling.
inline bool fits(const Args& a, double need_bytes, JW& why) {
  const double used = card_used_gib(), need = need_bytes / kGiB, cap = ceiling_gib(a);
  why.kv("card_used_gib_before", used).kv("need_gib", need).kv("ceiling_gib", cap);
  return used + need <= cap;
}

inline JW state_now(const Ctx& c) {
  JW j;
  const Sample s = c.nvml.sample();
  if (c.nvml.ok()) {
    j.kv("sm_mhz", s.sm_mhz).kv("mem_mhz", s.mem_mhz).kv("temp_c", s.temp_c).kv("power_w", s.power_w);
    if (s.have_throttle) j.kv("throttle", throttle_names(s.throttle));
  }
  j.kv("card_used_gib", card_used_gib());
  return j;
}

// Per-call milliseconds with a cold L2: before each call a 64 MiB buffer is overwritten (larger than the 36 MB L2), so
// the call's inputs come from VRAM as they would in a model where other work ran in between. Median of `reps` calls.
inline double time_cold_ms(cudaStream_t s, const DevBuf& flush, size_t flush_bytes, const std::function<void()>& enqueue, int reps = 11) {
  enqueue();
  PP_CK(cudaStreamSynchronize(s));
  std::vector<double> v;
  for (int i = 0; i < reps; ++i) {
    PP_CK(cudaMemsetAsync(flush.as(), i & 0xFF, flush_bytes, s));
    Event a, b;
    a.record(s);
    enqueue();
    b.record(s);
    b.sync();
    v.push_back(Event::ms(a, b));
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Per-call milliseconds: two warm-ups, then enough calls for at least min_ms of work (at least 3, at most 2000).
inline double time_ms(cudaStream_t s, const std::function<void()>& enqueue, double min_ms = 150.0) {
  enqueue();
  enqueue();
  PP_CK(cudaStreamSynchronize(s));
  const double one = time_stream_ms(s, enqueue);
  const int n = static_cast<int>(std::clamp(std::ceil(min_ms / std::max(one, 1e-3)), 3.0, 2000.0));
  Event a, b;
  a.record(s);
  for (int i = 0; i < n; ++i) enqueue();
  b.record(s);
  b.sync();
  return Event::ms(a, b) / n;
}

// max |x - ref| / max |ref|, both on the GPU.
inline double rel_max_err(const float* x, const float* ref, size_t n, cudaStream_t s, double* abs_out = nullptr) {
  DevBuf out(2 * sizeof(unsigned));
  PP_CK(cudaMemsetAsync(out.as(), 0, 2 * sizeof(unsigned), s));
  PP_CK(pkc::max_abs_diff(x, ref, n, out.as<unsigned>(), s));
  unsigned h[2] = {0, 0};
  PP_CK(cudaMemcpyAsync(h, out.as(), sizeof h, cudaMemcpyDeviceToHost, s));
  PP_CK(cudaStreamSynchronize(s));
  float d, m;
  std::memcpy(&d, &h[0], 4);
  std::memcpy(&m, &h[1], 4);
  if (abs_out) *abs_out = d;
  return m > 0 ? static_cast<double>(d) / m : static_cast<double>(d);
}

struct Cos {
  double cosine = 0, rel_l2 = 0;
};
inline Cos cosine(const float* x, const float* ref, size_t n, cudaStream_t s) {
  DevBuf out(3 * sizeof(double));
  PP_CK(cudaMemsetAsync(out.as(), 0, 3 * sizeof(double), s));
  PP_CK(pkc::dot3(x, ref, n, out.as<double>(), s));
  double h[3] = {0, 0, 0};
  PP_CK(cudaMemcpyAsync(h, out.as(), sizeof h, cudaMemcpyDeviceToHost, s));
  PP_CK(cudaStreamSynchronize(s));
  Cos r;
  r.cosine = (h[1] > 0 && h[2] > 0) ? h[0] / std::sqrt(h[1] * h[2]) : 0.0;
  r.rel_l2 = h[2] > 0 ? std::sqrt(std::max(0.0, h[1] - 2 * h[0] + h[2]) / h[2]) : 0.0;
  return r;
}

// GPU dequantization of the first `nb` blocks against the host reference (ggml's formulas). Returns max rel error.
inline double check_q4k_dequant(const pkc::BlockQ4K* w, size_t nb, cudaStream_t s) {
  DevBuf out(nb * pkc::kQK * sizeof(float));
  PP_CK(pkc::dequant_q4k_f32(w, out.as<float>(), nb, s));
  std::vector<pkc::BlockQ4K> hb(nb);
  std::vector<float> gpu(nb * pkc::kQK), host(nb * pkc::kQK);
  PP_CK(cudaMemcpyAsync(hb.data(), w, nb * sizeof(pkc::BlockQ4K), cudaMemcpyDeviceToHost, s));
  PP_CK(cudaMemcpyAsync(gpu.data(), out.as(), gpu.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
  PP_CK(cudaStreamSynchronize(s));
  double md = 0, mx = 0;
  for (size_t i = 0; i < nb; ++i) pkc::dequant_q4k_host(hb[i], host.data() + i * pkc::kQK);
  for (size_t i = 0; i < host.size(); ++i) {
    md = std::max(md, static_cast<double>(std::fabs(gpu[i] - host[i])));
    mx = std::max(mx, static_cast<double>(std::fabs(host[i])));
  }
  return mx > 0 ? md / mx : md;
}
inline double check_q6k_dequant(const pkc::BlockQ6K* w, size_t nb, cudaStream_t s) {
  DevBuf out(nb * pkc::kQK * sizeof(float));
  PP_CK(pkc::dequant_q6k_f32(w, out.as<float>(), nb, s));
  std::vector<pkc::BlockQ6K> hb(nb);
  std::vector<float> gpu(nb * pkc::kQK), host(nb * pkc::kQK);
  PP_CK(cudaMemcpyAsync(hb.data(), w, nb * sizeof(pkc::BlockQ6K), cudaMemcpyDeviceToHost, s));
  PP_CK(cudaMemcpyAsync(gpu.data(), out.as(), gpu.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
  PP_CK(cudaStreamSynchronize(s));
  double md = 0, mx = 0;
  for (size_t i = 0; i < nb; ++i) pkc::dequant_q6k_host(hb[i], host.data() + i * pkc::kQK);
  for (size_t i = 0; i < host.size(); ++i) {
    md = std::max(md, static_cast<double>(std::fabs(gpu[i] - host[i])));
    mx = std::max(mx, static_cast<double>(std::fabs(host[i])));
  }
  return mx > 0 ? md / mx : md;
}

// cuBLASLt bf16 x bf16 -> fp32 GEMM, C[M,N] = A[M,K] * B[N,K]^T, with the heuristic's best algorithm and a workspace.
class LtGemm {
  cublasLtHandle_t h_ = nullptr;
  cublasLtMatmulDesc_t op_ = nullptr;
  cublasLtMatrixLayout_t la_ = nullptr, lb_ = nullptr, lc_ = nullptr;
  cublasLtMatmulPreference_t pref_ = nullptr;
  cublasLtMatmulHeuristicResult_t heur_{};
  DevBuf ws_;
  size_t ws_bytes_ = 0;

 public:
  LtGemm(int M, int N, int K, size_t ws_bytes) : ws_(ws_bytes), ws_bytes_(ws_bytes) {
    PP_LT(cublasLtCreate(&h_));
    PP_LT(cublasLtMatmulDescCreate(&op_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    PP_LT(cublasLtMatmulDescSetAttribute(op_, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    PP_LT(cublasLtMatmulDescSetAttribute(op_, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    // column-major view: C^T[N,M] = (B as K x N, transposed) * (A as K x M)
    PP_LT(cublasLtMatrixLayoutCreate(&la_, CUDA_R_16BF, K, N, K));
    PP_LT(cublasLtMatrixLayoutCreate(&lb_, CUDA_R_16BF, K, M, K));
    PP_LT(cublasLtMatrixLayoutCreate(&lc_, CUDA_R_32F, N, M, N));
    PP_LT(cublasLtMatmulPreferenceCreate(&pref_));
    PP_LT(cublasLtMatmulPreferenceSetAttribute(pref_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_, sizeof ws_bytes_));
    int found = 0;
    PP_LT(cublasLtMatmulAlgoGetHeuristic(h_, op_, la_, lb_, lc_, lc_, pref_, 1, &heur_, &found));
    if (found < 1) throw std::runtime_error("cuBLASLt found no algorithm for this shape");
  }
  LtGemm(const LtGemm&) = delete;
  LtGemm& operator=(const LtGemm&) = delete;
  ~LtGemm() {
    if (pref_) cublasLtMatmulPreferenceDestroy(pref_);
    if (lc_) cublasLtMatrixLayoutDestroy(lc_);
    if (lb_) cublasLtMatrixLayoutDestroy(lb_);
    if (la_) cublasLtMatrixLayoutDestroy(la_);
    if (op_) cublasLtMatmulDescDestroy(op_);
    if (h_) cublasLtDestroy(h_);
  }
  void run(const uint16_t* A, const uint16_t* B, float* C, cudaStream_t s) {
    const float alpha = 1.f, beta = 0.f;
    PP_LT(cublasLtMatmul(h_, op_, &alpha, B, la_, A, lb_, &beta, C, lc_, C, lc_, &heur_.algo, ws_.as(), ws_bytes_, s));
  }
};

inline std::vector<int> m_list(const Ctx& c, const Args& a) {
  if (a.has("m")) return {static_cast<int>(a.inum("m", 2048))};
  return c.quick ? std::vector<int>{2048} : std::vector<int>{2048, 4096, 8192};
}

inline std::string fmt(const char* f, double x) {
  char b[64];
  std::snprintf(b, sizeof b, f, x);
  return b;
}

// Every valid tile size and output order of the own bf16 GEMM, timed; optionally each checked against `ref`.
struct GemmVariant {
  int cfg = -1, raster = -1;
  double ms = 1e30, rel_err = -1;
};
inline std::vector<GemmVariant> sweep_bf16(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K, cudaStream_t s,
                                           const float* ref) {
  std::vector<GemmVariant> out;
  for (int cfg = 0; cfg < pkc::kGemmCfgs; ++cfg)
    for (int raster = 0; raster < 2; ++raster) {
      const cudaError_t e = pkc::gemm_bf16_tiled(A, B, C, M, N, K, cfg, raster, s);
      if (e == cudaErrorInvalidValue) {  // the shape does not divide into this tile; nothing was launched
        (void)cudaGetLastError();
        continue;
      }
      PP_CK(e);
      GemmVariant v;
      v.cfg = cfg;
      v.raster = raster;
      v.ms = time_ms(s, [&] { PP_CK(pkc::gemm_bf16_tiled(A, B, C, M, N, K, cfg, raster, s)); }, 80.0);
      if (ref) v.rel_err = rel_max_err(C, ref, static_cast<size_t>(M) * N, s);
      out.push_back(v);
    }
  return out;
}
inline const GemmVariant& fastest(const std::vector<GemmVariant>& v) {
  return *std::min_element(v.begin(), v.end(), [](const GemmVariant& a, const GemmVariant& b) { return a.ms < b.ms; });
}

// ================================================================ P-K1
inline int p_k1(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-K1";
  init_cuda(c);
  const double peak = a.num("bf16-peak", 65.2);  // MEASURED P-H3 sustained bf16, at the author's overclock (trainer 11.12)
  const size_t ws_bytes = 32u << 20;
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  std::vector<JW> rows;
  bool all_correct = true, any_low = false;
  double min_eps = 1e9, max_ratio = 0;
  int adopted = 0, measured = 0;
  double dequant_check = -1, scratch_check = -1;
  for (const Shape& sh : kShapes) {
    for (const int M : m_list(c, a)) {
      if (g_abort.load()) break;
      JW row, why;
      row.kv("shape", sh.name).kv("M", M).kv("K", sh.K).kv("N", sh.N);
      const size_t nblk = static_cast<size_t>(sh.N) * (sh.K / pkc::kQK);
      const double need = static_cast<double>(M) * sh.K * 2 + nblk * pkc::kQ4KBytes + static_cast<double>(sh.N) * sh.K * 2 +
                          2.0 * M * sh.N * 4 + ws_bytes;
      if (!fits(a, need, why)) {
        row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
        rows.push_back(row);
        continue;
      }
      DevBuf A(static_cast<size_t>(M) * sh.K * 2), Wq(nblk * pkc::kQ4KBytes), Wb(static_cast<size_t>(sh.N) * sh.K * 2),
          C(static_cast<size_t>(M) * sh.N * 4), Cr(static_cast<size_t>(M) * sh.N * 4);
      PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(M) * sh.K, 1000u + sh.K + M, s));
      PP_CK(pkc::fill_q4k(Wq.as<pkc::BlockQ4K>(), nblk, 2000u + sh.N, s));
      if (dequant_check < 0) {
        dequant_check = check_q4k_dequant(Wq.as<pkc::BlockQ4K>(), std::min<size_t>(nblk, 64), s);
        // the bf16 scratch itself: the first 64 blocks against the fp32 dequantization (bf16 keeps 8 bits: <= 2^-8 relative)
        PP_CK(pkc::dequant_q4k_to_bf16(Wq.as<pkc::BlockQ4K>(), Wb.as<uint16_t>(), sh.N, sh.K, s));
        DevBuf f1(64 * pkc::kQK * 4), f2(64 * pkc::kQK * 4);
        PP_CK(pkc::bf16_to_f32(Wb.as<uint16_t>(), f1.as<float>(), 64 * pkc::kQK, s));
        PP_CK(pkc::dequant_q4k_f32(Wq.as<pkc::BlockQ4K>(), f2.as<float>(), 64, s));
        scratch_check = rel_max_err(f1.as<float>(), f2.as<float>(), 64 * pkc::kQK, s);
      }
      LtGemm lt(M, sh.N, sh.K, ws_bytes);
      const double t_deq = time_ms(s, [&] { PP_CK(pkc::dequant_q4k_to_bf16(Wq.as<pkc::BlockQ4K>(), Wb.as<uint16_t>(), sh.N, sh.K, s)); });
      const double t_lt = time_ms(s, [&] { lt.run(A.as<uint16_t>(), Wb.as<uint16_t>(), Cr.as<float>(), s); });
      const std::vector<GemmVariant> vars = sweep_bf16(A.as<uint16_t>(), Wb.as<uint16_t>(), C.as<float>(), M, sh.N, sh.K, s, Cr.as<float>());
      const GemmVariant& best = fastest(vars);
      double rel = 0, base_ms = 0;
      std::vector<JW> vt;
      for (const auto& v : vars) {
        rel = std::max(rel, v.rel_err);
        if (v.cfg == 0 && v.raster == 0) base_ms = v.ms;
        JW j;
        j.kv("tile", pkc::gemm_cfg_name(v.cfg)).kv("order", v.raster ? "grouped8" : "rows").kv("ms", v.ms).kv("max_rel_err_vs_cublaslt", v.rel_err);
        vt.push_back(j);
      }
      const double t_own = best.ms;
      const bool correct = rel <= 1e-3;
      const double flop = 2.0 * M * sh.N * sh.K;
      const double tf_own = flop / (t_own * 1e-3) / 1e12, tf_lt = flop / (t_lt * 1e-3) / 1e12;
      const double eps = tf_own / peak, ratio = tf_own / tf_lt;
      const double deq_gbs = static_cast<double>(nblk) * pkc::kQ4KBytes / (t_deq * 1e-3) / 1e9;
      row.kv("status", "ok").kv("dequant_ms", t_deq).kv("dequant_q4k_read_gbs", deq_gbs).kv("own_gemm_ms", t_own).kv("cublaslt_ms", t_lt);
      row.kv("best_tile", pkc::gemm_cfg_name(best.cfg)).kv("best_order", best.raster ? "grouped8" : "rows");
      if (base_ms > 0) row.kv("tflops_first_design_128x128_rows", flop / (base_ms * 1e-3) / 1e12).kv("speedup_over_first_design", base_ms / t_own);
      row.arr_obj("variants", vt);
      row.kv("own_tflops", tf_own).kv("cublaslt_tflops", tf_lt).kv("own_epsilon_vs_peak", eps).kv("cublaslt_epsilon_vs_peak", tf_lt / peak);
      row.kv("own_over_cublaslt", ratio).kv("own_tflops_incl_dequant", flop / ((t_own + t_deq) * 1e-3) / 1e12);
      row.kv("max_rel_err_vs_cublaslt_all_variants", rel).kv("correct", correct);
      row.kv("adopt_own_kernel_at_0p85", ratio >= 0.85);
      rows.push_back(row);
      ++measured;
      all_correct = all_correct && correct;
      if (eps < 0.35) any_low = true;
      min_eps = std::min(min_eps, eps);
      max_ratio = std::max(max_ratio, ratio);
      if (ratio >= 0.85) ++adopted;
      std::printf("  P-K1 %-46s M=%-5d own %6.1f TFLOPS (eps %.2f, %s %s)  cuBLASLt %6.1f  ratio %.2f  %s\n", sh.name, M, tf_own, eps,
                  pkc::gemm_cfg_name(best.cfg), best.raster ? "grouped8" : "rows", tf_lt, ratio, correct ? "ok" : "MISMATCH");
      std::fflush(stdout);
    }
  }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("bf16_peak_tflops", peak).kv("peak_source", "P-H3 sustained bf16, MEASURED at the author's overclock (trainer master 11.12)");
  r.inputs.kv("tiles", "128x128, 128x64, 64x64 (BK 32, 8 warps, mma.sync m16n8k16, cp.async 2 stages); orders: rows, grouped by 8 along M; the fastest per shape is reported").kv("cublaslt_workspace_mib", 32);
  r.inputs.kv("weights", "synthetic Q4_K blocks (random scales and quants), dequantized to a bf16 scratch").kv("activations", "synthetic bf16, gaussian");
  r.values.arr_obj("runs", rows).kv("measured", measured).kv("min_epsilon", measured ? min_eps : std::nan(""));
  r.values.kv("best_own_over_cublaslt", max_ratio).kv("shapes_adopting_own_kernel", adopted).kv("q4k_dequant_max_rel_err_vs_host", dequant_check).kv("bf16_scratch_max_rel_err_vs_fp32", scratch_check);
  r.values.kv("reopen_tile_design_epsilon_below_0p35", any_low).kv_obj("state_before", before).kv_obj("state_after", state_now(c));
  r.values.kv_obj("nvml_whole_run", summarize(all));
  r.thresholds.kv("adopt_own_kernel_if_own_over_cublaslt_at_least", 0.85).kv("reopen_tile_design_if_epsilon_below", 0.35);
  r.thresholds.kv("correct_if_max_rel_err_below", 1e-3).kv("plan_assumed_epsilon", "0.4 to 0.6");
  r.pass = (all_correct && !any_low && measured > 0 && dequant_check >= 0 && dequant_check < 1e-5 && scratch_check >= 0 && scratch_check <= 1.0 / 256) ? 1 : 0;
  r.headline = "min eps " + fmt("%.2f", measured ? min_eps : 0.0) + ", best own/cuBLASLt " + fmt("%.2f", max_ratio) + ", " +
               std::to_string(adopted) + "/" + std::to_string(measured) + " shapes adopt own kernel" + (all_correct ? "" : ", OUTPUT MISMATCH");
  r.notes = "cuBLASLt is a test-only reference (THIRD_PARTY_NOTICES: not shipped). Efficiency is against the overclocked P-H3 peak; at stock clocks pass --bf16-peak with the stock measurement. " +
            std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "k1_bf16");
  return r.pass == 1 ? 0 : 3;
}

// ================================================================ P-K2
inline int p_k2(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-K2";
  init_cuda(c);
  {  // compiler and occupancy facts for every GEMM kernel (registers, spills, shared memory, blocks resident per SM)
    static const char* names[] = {"bf16 128x128", "bf16 128x64", "bf16 64x64", "int8 v2", "int8 v4", "int8 v5",
                                  "int8 v6 w32x32", "int8 v6 w64x16", "int8 v6 64x64", "int8 v7", "int8 v8 3-stage", "int8 v8 4-stage",
                                  "int8 v9 2-stage", "int8 v9 3-stage", "int8 v10 2-stage", "int8 v10 3-stage",
                                  "int8 v11 2-stage", "int8 v11 3-stage"};
    std::vector<JW> ki;
    for (int k = 0; k < 18; ++k) {
      int regs = 0, local = 0, smem = 0, bps = 0;
      const cudaError_t e = pkc::kernel_info(k, &regs, &local, &smem, &bps);
      JW j;
      j.kv("kernel", names[k]);
      if (e == cudaSuccess) {
        j.kv("registers_per_thread", regs).kv("spill_local_bytes", local).kv("static_shared_bytes", smem).kv("blocks_per_sm_at_256_threads", bps);
        std::printf("  kernel %-16s %3d registers, %4d spill bytes, %6d shared bytes, %d blocks/SM\n", names[k], regs, local, smem, bps);
      } else {
        (void)cudaGetLastError();
        j.kv("error", cuda_msg(e));
      }
      ki.push_back(j);
    }
    r.values.arr_obj("kernel_resources", ki);
  }
  const double peak8 = a.num("int8-peak", 259.5);  // MEASURED P-H3 sustained int8 TOPS, overclock
  const double peak16 = a.num("bf16-peak", 65.2);
  const double vram_gbs = a.num("vram-gbs-published", 504.2);  // RTX 4070, PUBLISHED (NVIDIA spec page, masterwork S2)
  constexpr size_t kFlushBytes = 64ull << 20;
  DevBuf flush(kFlushBytes);
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  std::vector<JW> rows;
  bool all_correct = true;
  int measured = 0, fast = 0;
  double min_eps8 = 1e9;
  int fast7 = 0, fast8 = 0, fast9 = 0, fast10 = 0;
  double v10_min = 1e9, v10_max = 0, v10_min_eps = 1e9, v10_max_eps = 0, v10_min_bf = 1e9, v10_max_bf = 0;
  double fz_min = 1e9, fz_max = 0, sep_min = 1e9, sep_max = 0;
  double v11_min = 1e9, v11_max = 0, v11_min_eps = 1e9, v11_max_eps = 0, v11_min_bf = 1e9, v11_max_bf = 0;
  int fast11 = 0;
  int fast_fz = 0;
  double v7_min = 1e9, v7_max = 0, v7_min_cos = 1, v8_min = 1e9, v8_max = 0, v9_min = 1e9, v9_max = 0, v9_min_eps = 1e9, v9_max_eps = 0;
  for (const Shape& sh : kShapes) {
    for (const int M : m_list(c, a)) {
      if (g_abort.load()) break;
      JW row, why;
      row.kv("shape", sh.name).kv("M", M).kv("K", sh.K).kv("N", sh.N);
      const size_t nblk = static_cast<size_t>(sh.N) * (sh.K / pkc::kQK);
      const int ref_rows = 16, ref_cols = std::min(sh.N, 1024);
      const double need = static_cast<double>(M) * sh.K * 3 + static_cast<double>(M) * (sh.K / 32) * 12 + nblk * (pkc::kQ4KBytes + 64.0) +
                          static_cast<double>(sh.N) * sh.K * 2 + 2.0 * M * sh.N * 4 + ref_rows * ref_cols * 4.0 +
                          static_cast<double>(M) * sh.K + static_cast<double>(M) * (sh.K / 32) * 8 + ref_rows * ref_cols * 4.0 + nblk * 16.0 + nblk * 68.0 + nblk * 32.0 +
                          static_cast<double>(M) * (sh.K / pkc::kQK) * 32 +  // v7 / v8 / v9 / v10 inputs
                          static_cast<double>(M) * sh.K * 3 + static_cast<double>(M) * (sh.K / 32) * 8 + static_cast<double>(M) * (sh.K / pkc::kQK) * 36 +
                          sh.K * 2.0 + static_cast<double>(kFlushBytes);  // revision 11: norm outputs, cold-L2 flush buffer
      if (!fits(a, need, why)) {
        row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
        rows.push_back(row);
        continue;
      }
      DevBuf A(static_cast<size_t>(M) * sh.K * 2), a8(static_cast<size_t>(M) * sh.K), sa(static_cast<size_t>(M) * (sh.K / 32) * 4),
          s8(static_cast<size_t>(M) * (sh.K / 32) * 4), sas(static_cast<size_t>(M) * (sh.K / 32) * 4), Wq(nblk * pkc::kQ4KBytes),
          wdsc(nblk * 8 * 4), wdmn(nblk * 8 * 4),
          Wb(static_cast<size_t>(sh.N) * sh.K * 2),
          C8(static_cast<size_t>(M) * sh.N * 4), Cb(static_cast<size_t>(M) * sh.N * 4), Cref(static_cast<size_t>(ref_rows) * ref_cols * 4);
      PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(M) * sh.K, 1000u + sh.K + M, s));
      PP_CK(pkc::fill_q4k(Wq.as<pkc::BlockQ4K>(), nblk, 2000u + sh.N, s));
      const double t_q = time_ms(s, [&] { PP_CK(pkc::quantize_rows_q8(A.as<uint16_t>(), a8.as<int8_t>(), sa.as<float>(), s8.as<int>(), sas.as<float>(), M, sh.K, s)); });
      // Every int8 kernel version in both tile orders, each timed and each checked against fp32 on the same quantized inputs:
      // v2 (revision 2), v4 (revision 4), v5 (registers capped), v6 (smaller warp tiles, three variants).
      const double t_prep = time_ms(s, [&] { PP_CK(pkc::prep_q4k_scales(Wq.as<pkc::BlockQ4K>(), wdsc.as<float>(), wdmn.as<float>(), sh.N, sh.K, s)); });
      struct Cand {
        std::string name;
        std::function<cudaError_t(int)> run;
        double ms = 1e30, err = -1;
        int raster = 0;
      };
      std::vector<Cand> cands;
      cands.push_back({"v2", [&](int r) { return pkc::gemm_q8_q4k(a8.as<int8_t>(), sa.as<float>(), s8.as<int>(), Wq.as<pkc::BlockQ4K>(), C8.as<float>(), M, sh.N, sh.K, r, s); }});
      cands.push_back({"v4", [&](int r) { return pkc::gemm_q8_q4k_v4(a8.as<int8_t>(), sa.as<float>(), sas.as<float>(), Wq.as<pkc::BlockQ4K>(), C8.as<float>(), M, sh.N, sh.K, r, s); }});
      cands.push_back({"v5", [&](int r) {
                         return pkc::gemm_q8_q4k_v5(a8.as<int8_t>(), sa.as<float>(), sas.as<float>(), Wq.as<pkc::BlockQ4K>(), wdsc.as<float>(), wdmn.as<float>(),
                                                    C8.as<float>(), M, sh.N, sh.K, r, s);
                       }});
      for (int v = 0; v < pkc::kV6Variants; ++v)
        cands.push_back({pkc::v6_name(v), [&, v](int r) {
                           return pkc::gemm_q8_q4k_v6(a8.as<int8_t>(), sa.as<float>(), sas.as<float>(), Wq.as<pkc::BlockQ4K>(), wdsc.as<float>(),
                                                      wdmn.as<float>(), C8.as<float>(), M, sh.N, sh.K, v, r, s);
                         }});
      PP_CK(pkc::ref_q8_q4k_rows(a8.as<int8_t>(), sa.as<float>(), Wq.as<pkc::BlockQ4K>(), Cref.as<float>(), ref_rows, ref_cols, sh.K, s));
      auto worst_err = [&] {
        double w = 0;
        for (int rr = 0; rr < ref_rows; ++rr)
          w = std::max(w, rel_max_err(C8.as<float>() + static_cast<size_t>(rr) * sh.N, Cref.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
        return w;
      };
      std::vector<JW> cj;
      const Cand* best = nullptr;
      double worst = 0;
      for (auto& cd : cands) {
        for (int raster = 0; raster < 2; ++raster) {
          const cudaError_t e = cd.run(raster);
          if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
          PP_CK(e);
          const double ms = time_ms(s, [&] { PP_CK(cd.run(raster)); }, 80.0);
          if (ms < cd.ms) { cd.ms = ms; cd.raster = raster; }
        }
        if (cd.ms >= 1e29) continue;
        PP_CK(cd.run(0));
        cd.err = worst_err();
        worst = std::max(worst, cd.err);
        if (!best || cd.ms < best->ms) best = &cd;
        JW j;
        j.kv("kernel", cd.name).kv("ms", cd.ms).kv("order", cd.raster ? "grouped8" : "rows").kv("over_v2", cands[0].ms / cd.ms).kv("max_rel_err_vs_fp32", cd.err);
        cj.push_back(j);
      }
      const double t_i8 = best->ms;
      const double t_deq = time_ms(s, [&] { PP_CK(pkc::dequant_q4k_to_bf16(Wq.as<pkc::BlockQ4K>(), Wb.as<uint16_t>(), sh.N, sh.K, s)); });
      const GemmVariant bfb = fastest(sweep_bf16(A.as<uint16_t>(), Wb.as<uint16_t>(), Cb.as<float>(), M, sh.N, sh.K, s, nullptr));
      const double t_bf = bfb.ms;
      PP_CK(best->run(best->raster));
      const bool correct = worst <= 1e-3;
      // how far the int8 path's output is from the bf16 path's output on the same activations and weights
      const Cos cs = cosine(C8.as<float>(), Cb.as<float>(), static_cast<size_t>(M) * sh.N, s);
      // v7 (revision 7): activations with one scale per 256, weight sub-block scale applied in integers, d * sa once per
      // super-block. Own quantization (timed), both tile orders, checked against fp32 on its own quantized inputs, and the
      // cosine against the bf16 path shows what the coarser activation scale costs in precision.
      DevBuf a87(static_cast<size_t>(M) * sh.K), sa7(static_cast<size_t>(M) * (sh.K / pkc::kQK) * 4), s327(static_cast<size_t>(M) * (sh.K / 32) * 4),
          x7(static_cast<size_t>(M) * (sh.K / 32) * 4), Cref7(static_cast<size_t>(ref_rows) * ref_cols * 4);
      const double t_q7 = time_ms(s, [&] {
        PP_CK(pkc::quantize_rows_q8k(A.as<uint16_t>(), a87.as<int8_t>(), sa7.as<float>(), s327.as<int>(), x7.as<float>(), M, sh.K, s));
      });
      auto run7 = [&](int rr) {
        return pkc::gemm_q8k_q4k_v7(a87.as<int8_t>(), sa7.as<float>(), x7.as<float>(), Wq.as<pkc::BlockQ4K>(), wdmn.as<float>(), C8.as<float>(), M, sh.N, sh.K, rr, s);
      };
      double t7 = 1e30;
      int raster7 = 0;
      for (int raster = 0; raster < 2; ++raster) {
        const cudaError_t e = run7(raster);
        if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
        PP_CK(e);
        const double ms = time_ms(s, [&] { PP_CK(run7(raster)); }, 80.0);
        if (ms < t7) { t7 = ms; raster7 = raster; }
      }
      PP_CK(pkc::ref_q8k_q4k_rows(a87.as<int8_t>(), sa7.as<float>(), Wq.as<pkc::BlockQ4K>(), Cref7.as<float>(), ref_rows, ref_cols, sh.K, s));
      PP_CK(run7(0));
      double err7 = 0;
      for (int rr = 0; rr < ref_rows; ++rr)
        err7 = std::max(err7, rel_max_err(C8.as<float>() + static_cast<size_t>(rr) * sh.N, Cref7.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
      const Cos cs7 = cosine(C8.as<float>(), Cb.as<float>(), static_cast<size_t>(M) * sh.N, s);
      const bool ok7 = err7 <= 1e-3;
      const double op7 = 2.0 * M * sh.N * sh.K;
      const double tops7 = op7 / (t7 * 1e-3) / 1e12;
      JW j7;
      j7.kv("ms", t7).kv("order", raster7 ? "grouped8" : "rows").kv("over_v2", cands[0].ms / t7).kv("over_best_earlier", t_i8 / t7);
      j7.kv("tops", tops7).kv("epsilon_vs_peak", tops7 / peak8).kv("over_bf16_gemm_only", t_bf / t7).kv("over_bf16_incl_conversions", (t_bf + t_deq) / (t7 + t_q7));
      j7.kv("quantize_activations_ms", t_q7).kv("max_rel_err_vs_fp32_on_same_quantized_inputs", err7).kv("correct", ok7);
      j7.kv("cosine_vs_bf16_path", cs7.cosine).kv("rel_l2_vs_bf16_path", cs7.rel_l2).kv("cosine_per32_path_vs_bf16_path", cs.cosine);
      all_correct = all_correct && ok7;
      if (t_bf / t7 >= 1.3) ++fast7;
      v7_min = std::min(v7_min, cands[0].ms / t7);
      v7_max = std::max(v7_max, cands[0].ms / t7);
      v7_min_cos = std::min(v7_min_cos, cs7.cosine);
      // v8 (revision 8): v7's arithmetic and inputs, 3 or 4 load stages, one barrier per step, weight record prepared once.
      DevBuf rec8(nblk * 16);
      const double t_rec = time_ms(s, [&] { PP_CK(pkc::prep_q4k_rec(Wq.as<pkc::BlockQ4K>(), rec8.as(), sh.N, sh.K, s)); });
      std::vector<JW> j8s;
      double t8best = 1e30;
      std::string n8best;
      for (int v = 0; v < pkc::kV8Variants; ++v) {
        auto run8 = [&](int rr) {
          return pkc::gemm_q8k_q4k_v8(a87.as<int8_t>(), sa7.as<float>(), x7.as<float>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wdmn.as<float>(), C8.as<float>(), M,
                                      sh.N, sh.K, v, rr, s);
        };
        double t8 = 1e30;
        int raster8 = 0;
        for (int raster = 0; raster < 2; ++raster) {
          const cudaError_t e = run8(raster);
          if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
          PP_CK(e);
          const double ms = time_ms(s, [&] { PP_CK(run8(raster)); }, 80.0);
          if (ms < t8) { t8 = ms; raster8 = raster; }
        }
        JW j8;
        j8.kv("kernel", pkc::v8_name(v));
        if (t8 >= 1e29) {
          j8.kv("status", "not run");
          j8s.push_back(j8);
          continue;
        }
        PP_CK(run8(0));
        double err8 = 0;
        for (int rr = 0; rr < ref_rows; ++rr)
          err8 = std::max(err8, rel_max_err(C8.as<float>() + static_cast<size_t>(rr) * sh.N, Cref7.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
        const Cos cs8 = cosine(C8.as<float>(), Cb.as<float>(), static_cast<size_t>(M) * sh.N, s);
        const bool ok8 = err8 <= 1e-3;
        const double tops8 = op7 / (t8 * 1e-3) / 1e12;
        j8.kv("ms", t8).kv("order", raster8 ? "grouped8" : "rows").kv("over_v2", cands[0].ms / t8).kv("over_v7", t7 / t8).kv("tops", tops8);
        j8.kv("epsilon_vs_peak", tops8 / peak8).kv("over_bf16_gemm_only", t_bf / t8).kv("over_bf16_incl_conversions", (t_bf + t_deq) / (t8 + t_q7));
        j8.kv("max_rel_err_vs_fp32_on_same_quantized_inputs", err8).kv("correct", ok8).kv("cosine_vs_bf16_path", cs8.cosine);
        j8s.push_back(j8);
        all_correct = all_correct && ok8;
        if (ok8 && t8 < t8best) { t8best = t8; n8best = pkc::v8_name(v); }
        std::printf("       %-10s %6.1f TOPS (eps8 %.2f) = %.2fx v2, %.2fx v7, %.2fx bf16  %s\n", pkc::v8_name(v), tops8, tops8 / peak8, cands[0].ms / t8, t7 / t8,
                    t_bf / t8, ok8 ? "ok" : "MISMATCH");
      }
      if (t8best < 1e29) {
        if (t_bf / t8best >= 1.3) ++fast8;
        v8_min = std::min(v8_min, cands[0].ms / t8best);
        v8_max = std::max(v8_max, cands[0].ms / t8best);
      }
      // v9 (revision 9): v8's arithmetic, 16 warps per tile (4 per scheduler), weight scales by column.
      DevBuf scT9(nblk * 32), dmnT9(nblk * 32), dT9(nblk * 4);
      const double t_p9 = time_ms(s, [&] { PP_CK(pkc::prep_q4k_v9(Wq.as<pkc::BlockQ4K>(), scT9.as<int>(), dmnT9.as<float>(), dT9.as<float>(), sh.N, sh.K, s)); });
      std::vector<JW> j9s;
      double t9best = 1e30;
      for (int v = 0; v < pkc::kV9Variants; ++v) {
        auto run9 = [&](int rr) {
          return pkc::gemm_q8k_q4k_v9(a87.as<int8_t>(), sa7.as<float>(), x7.as<float>(), Wq.as<pkc::BlockQ4K>(), scT9.as<int>(), dmnT9.as<float>(), dT9.as<float>(),
                                      C8.as<float>(), M, sh.N, sh.K, v, rr, s);
        };
        double t9 = 1e30;
        int raster9 = 0;
        for (int raster = 0; raster < 2; ++raster) {
          const cudaError_t e = run9(raster);
          if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
          PP_CK(e);
          const double ms = time_ms(s, [&] { PP_CK(run9(raster)); }, 80.0);
          if (ms < t9) { t9 = ms; raster9 = raster; }
        }
        JW j9;
        j9.kv("kernel", pkc::v9_name(v));
        if (t9 >= 1e29) {
          j9.kv("status", "not run");
          j9s.push_back(j9);
          continue;
        }
        PP_CK(run9(0));
        double err9 = 0;
        for (int rr = 0; rr < ref_rows; ++rr)
          err9 = std::max(err9, rel_max_err(C8.as<float>() + static_cast<size_t>(rr) * sh.N, Cref7.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
        const Cos cs9 = cosine(C8.as<float>(), Cb.as<float>(), static_cast<size_t>(M) * sh.N, s);
        const bool ok9 = err9 <= 1e-3;
        const double tops9 = op7 / (t9 * 1e-3) / 1e12;
        j9.kv("ms", t9).kv("order", raster9 ? "grouped8" : "rows").kv("over_v2", cands[0].ms / t9).kv("over_v7", t7 / t9).kv("over_best_v8", t8best / t9).kv("tops", tops9);
        j9.kv("epsilon_vs_peak", tops9 / peak8).kv("over_bf16_gemm_only", t_bf / t9).kv("over_bf16_incl_conversions", (t_bf + t_deq) / (t9 + t_q7));
        j9.kv("max_rel_err_vs_fp32_on_same_quantized_inputs", err9).kv("correct", ok9).kv("cosine_vs_bf16_path", cs9.cosine);
        j9s.push_back(j9);
        all_correct = all_correct && ok9;
        if (ok9 && t9 < t9best) t9best = t9;
        std::printf("       %-10s %6.1f TOPS (eps8 %.2f) = %.2fx v2, %.2fx v8, %.2fx bf16  %s\n", pkc::v9_name(v), tops9, tops9 / peak8, cands[0].ms / t9,
                    t8best / t9, t_bf / t9, ok9 ? "ok" : "MISMATCH");
      }
      if (t9best < 1e29) {
        if (t_bf / t9best >= 1.3) ++fast9;
        v9_min = std::min(v9_min, cands[0].ms / t9best);
        v9_max = std::max(v9_max, cands[0].ms / t9best);
        const double e9 = op7 / (t9best * 1e-3) / 1e12 / peak8;
        v9_min_eps = std::min(v9_min_eps, e9);
        v9_max_eps = std::max(v9_max_eps, e9);
      }
      // v10 (revision 10): v8's integer main term; min term once per super-block as one exact bf16 product of digits.
      DevBuf dig10(static_cast<size_t>(M) * (sh.K / pkc::kQK) * 32), wd10(nblk * 32);
      const double t_q10 = time_ms(s, [&] {
        PP_CK(pkc::quantize_rows_q8k_dig(A.as<uint16_t>(), a87.as<int8_t>(), sa7.as<float>(), s327.as<int>(), x7.as<float>(), dig10.as<uint16_t>(), M, sh.K, s));
      });
      const double t_p10 = time_ms(s, [&] { PP_CK(pkc::prep_q4k_dig(Wq.as<pkc::BlockQ4K>(), wd10.as<uint16_t>(), sh.N, sh.K, s)); });
      std::vector<JW> j10s;
      double t10best = 1e30;
      for (int v = 0; v < pkc::kV10Variants; ++v) {
        auto run10 = [&](int rr) {
          return pkc::gemm_q8k_q4k_v10(a87.as<int8_t>(), sa7.as<float>(), dig10.as<uint16_t>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wd10.as<uint16_t>(), C8.as<float>(), M,
                                       sh.N, sh.K, v, rr, s);
        };
        double t10 = 1e30;
        int raster10 = 0;
        for (int raster = 0; raster < 2; ++raster) {
          const cudaError_t e = run10(raster);
          if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
          PP_CK(e);
          const double ms = time_ms(s, [&] { PP_CK(run10(raster)); }, 80.0);
          if (ms < t10) { t10 = ms; raster10 = raster; }
        }
        JW j10;
        j10.kv("kernel", pkc::v10_name(v));
        if (t10 >= 1e29) {
          j10.kv("status", "not run");
          j10s.push_back(j10);
          continue;
        }
        PP_CK(run10(0));
        double err10 = 0;
        for (int rr = 0; rr < ref_rows; ++rr)
          err10 = std::max(err10, rel_max_err(C8.as<float>() + static_cast<size_t>(rr) * sh.N, Cref7.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
        const Cos cs10 = cosine(C8.as<float>(), Cb.as<float>(), static_cast<size_t>(M) * sh.N, s);
        const bool ok10 = err10 <= 1e-3;
        const double tops10 = op7 / (t10 * 1e-3) / 1e12;
        j10.kv("ms", t10).kv("order", raster10 ? "grouped8" : "rows").kv("over_v2", cands[0].ms / t10).kv("over_best_v8", t8best / t10).kv("over_best_v9", t9best / t10);
        j10.kv("tops", tops10).kv("epsilon_vs_peak", tops10 / peak8).kv("over_bf16_gemm_only", t_bf / t10).kv("over_bf16_incl_conversions", (t_bf + t_deq) / (t10 + t_q10));
        j10.kv("quantize_activations_ms", t_q10).kv("max_rel_err_vs_fp32_on_same_quantized_inputs", err10).kv("correct", ok10).kv("cosine_vs_bf16_path", cs10.cosine);
        j10s.push_back(j10);
        all_correct = all_correct && ok10;
        if (ok10 && t10 < t10best) t10best = t10;
        std::printf("       %-11s %6.1f TOPS (eps8 %.2f) = %.2fx v2, %.2fx v8, %.2fx bf16  %s\n", pkc::v10_name(v), tops10, tops10 / peak8, cands[0].ms / t10,
                    t8best / t10, t_bf / t10, ok10 ? "ok" : "MISMATCH");
      }
      if (t10best < 1e29) {
        if (t_bf / t10best >= 1.3) ++fast10;
        v10_min = std::min(v10_min, cands[0].ms / t10best);
        v10_max = std::max(v10_max, cands[0].ms / t10best);
        const double e10 = op7 / (t10best * 1e-3) / 1e12 / peak8;
        v10_min_eps = std::min(v10_min_eps, e10);
        v10_max_eps = std::max(v10_max_eps, e10);
        v10_min_bf = std::min(v10_min_bf, t_bf / t10best);
        v10_max_bf = std::max(v10_max_bf, t_bf / t10best);
      }
      // v11 (revision 11): v10's arithmetic with 16 warps per tile.
      std::vector<JW> j11s;
      double t11best = 1e30;
      for (int v = 0; v < pkc::kV11Variants; ++v) {
        auto run11 = [&](int rr) {
          return pkc::gemm_q8k_q4k_v11(a87.as<int8_t>(), sa7.as<float>(), dig10.as<uint16_t>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wd10.as<uint16_t>(), C8.as<float>(), M,
                                       sh.N, sh.K, v, rr, s);
        };
        double t11 = 1e30;
        int raster11 = 0;
        for (int raster = 0; raster < 2; ++raster) {
          const cudaError_t e = run11(raster);
          if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
          PP_CK(e);
          const double ms = time_ms(s, [&] { PP_CK(run11(raster)); }, 80.0);
          if (ms < t11) { t11 = ms; raster11 = raster; }
        }
        JW j11;
        j11.kv("kernel", pkc::v11_name(v));
        if (t11 >= 1e29) {
          j11.kv("status", "not run");
          j11s.push_back(j11);
          continue;
        }
        PP_CK(run11(0));
        double err11 = 0;
        for (int rr = 0; rr < ref_rows; ++rr)
          err11 = std::max(err11, rel_max_err(C8.as<float>() + static_cast<size_t>(rr) * sh.N, Cref7.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
        const Cos cs11 = cosine(C8.as<float>(), Cb.as<float>(), static_cast<size_t>(M) * sh.N, s);
        const bool ok11 = err11 <= 1e-3;
        const double tops11 = op7 / (t11 * 1e-3) / 1e12;
        j11.kv("ms", t11).kv("order", raster11 ? "grouped8" : "rows").kv("over_v2", cands[0].ms / t11).kv("over_best_v10", t10best / t11).kv("tops", tops11);
        j11.kv("epsilon_vs_peak", tops11 / peak8).kv("over_bf16_gemm_only", t_bf / t11);
        j11.kv("max_rel_err_vs_fp32_on_same_quantized_inputs", err11).kv("correct", ok11).kv("cosine_vs_bf16_path", cs11.cosine);
        j11s.push_back(j11);
        all_correct = all_correct && ok11;
        if (ok11 && t11 < t11best) t11best = t11;
        std::printf("       %-11s %6.1f TOPS (eps8 %.2f) = %.2fx v2, %.2fx v10, %.2fx bf16  %s\n", pkc::v11_name(v), tops11, tops11 / peak8, cands[0].ms / t11,
                    t10best / t11, t_bf / t11, ok11 ? "ok" : "MISMATCH");
      }
      if (t11best < 1e29) {
        if (t_bf / t11best >= 1.3) ++fast11;
        v11_min = std::min(v11_min, cands[0].ms / t11best);
        v11_max = std::max(v11_max, cands[0].ms / t11best);
        const double e11 = op7 / (t11best * 1e-3) / 1e12 / peak8;
        v11_min_eps = std::min(v11_min_eps, e11);
        v11_max_eps = std::max(v11_max_eps, e11);
        v11_min_bf = std::min(v11_min_bf, t_bf / t11best);
        v11_max_bf = std::max(v11_max_bf, t_bf / t11best);
      }
      const double t_int8 = std::min(t10best, t11best);  // fastest correct int8 kernel of this shape
      // Revision 11: activation conversion with a cold L2, separate pass against quantization fused into the RMSNorm that
      // produces the activations. The bf16 path needs norm -> bf16 anyway, so the int8 path's extra cost is
      // (norm -> int8) - (norm -> bf16) when fused, and (norm -> bf16) + quantize when separate.
      JW cv;
      if (t_int8 < 1e29) {
        DevBuf gam(static_cast<size_t>(sh.K) * 2), Y(static_cast<size_t>(M) * sh.K * 2), a8f(static_cast<size_t>(M) * sh.K),
            saf(static_cast<size_t>(M) * (sh.K / pkc::kQK) * 4), s32f(static_cast<size_t>(M) * (sh.K / 32) * 4), x32f(static_cast<size_t>(M) * (sh.K / 32) * 4),
            digf(static_cast<size_t>(M) * (sh.K / pkc::kQK) * 32);
        PP_CK(pkc::fill_bf16(gam.as<uint16_t>(), sh.K, 3000u + sh.K, s));
        const double t_nb = time_cold_ms(s, flush, kFlushBytes, [&] { PP_CK(pkc::rmsnorm_bf16(A.as<uint16_t>(), gam.as<uint16_t>(), Y.as<uint16_t>(), M, sh.K, s)); });
        const double t_nq = time_cold_ms(s, flush, kFlushBytes, [&] {
          PP_CK(pkc::rmsnorm_q8k(A.as<uint16_t>(), gam.as<uint16_t>(), a8f.as<int8_t>(), saf.as<float>(), s32f.as<int>(), x32f.as<float>(), digf.as<uint16_t>(), M, sh.K, s));
        });
        const double t_qs = time_cold_ms(s, flush, kFlushBytes, [&] {
          PP_CK(pkc::quantize_rows_q8k_dig(Y.as<uint16_t>(), a87.as<int8_t>(), sa7.as<float>(), s327.as<int>(), x7.as<float>(), dig10.as<uint16_t>(), M, sh.K, s));
        });
        const double t_10c = time_cold_ms(s, flush, kFlushBytes, [&] {
          PP_CK(pkc::gemm_q8k_q4k_v10(a87.as<int8_t>(), sa7.as<float>(), dig10.as<uint16_t>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wd10.as<uint16_t>(), C8.as<float>(), M,
                                      sh.N, sh.K, 1, 0, s));
        });
        // fused against separate on the first rows: int8 values and scales
        const int cr = std::min(M, 64);
        std::vector<int8_t> h1(static_cast<size_t>(cr) * sh.K), h2(h1.size());
        std::vector<float> f1(static_cast<size_t>(cr) * (sh.K / pkc::kQK)), f2(f1.size());
        PP_CK(cudaMemcpy(h1.data(), a8f.as(), h1.size(), cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(h2.data(), a87.as(), h2.size(), cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(f1.data(), saf.as(), f1.size() * 4, cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(f2.data(), sa7.as(), f2.size() * 4, cudaMemcpyDeviceToHost));
        size_t ndiff = 0;
        int maxd = 0;
        for (size_t i = 0; i < h1.size(); ++i) {
          const int d = std::abs(static_cast<int>(h1[i]) - static_cast<int>(h2[i]));
          ndiff += d != 0;
          maxd = std::max(maxd, d);
        }
        double maxs = 0;
        for (size_t i = 0; i < f1.size(); ++i) maxs = std::max(maxs, std::fabs(f1[i] - f2[i]) / std::max(std::fabs(static_cast<double>(f2[i])), 1e-30));
        const double qbytes = static_cast<double>(M) * sh.K * 3 + static_cast<double>(M) * (sh.K / 32) * 8 + static_cast<double>(M) * (sh.K / pkc::kQK) * 36;
        const double extra_fused = t_nq - t_nb, extra_sep = t_qs;
        const double r_fz = (t_nb + t_deq + t_bf) / (t_nq + t_int8), r_sep = (t_nb + t_deq + t_bf) / (t_nb + t_qs + t_int8);
        cv.kv("timing", "cold L2: 64 MiB overwritten before each call, median of 11");
        cv.kv("norm_to_bf16_ms", t_nb).kv("norm_to_int8_fused_ms", t_nq).kv("separate_quantize_ms", t_qs).kv("separate_quantize_gbs", qbytes / (t_qs * 1e-3) / 1e9);
        cv.kv("separate_quantize_over_published_vram_bw", qbytes / (t_qs * 1e-3) / 1e9 / vram_gbs).kv("v10_gemm_cold_ms", t_10c);
        cv.kv("int8_extra_conversion_ms_fused", extra_fused).kv("int8_extra_conversion_ms_separate", extra_sep);
        cv.kv("int8_over_bf16_incl_norm_and_conversions_fused", r_fz).kv("int8_over_bf16_incl_norm_and_conversions_separate", r_sep);
        cv.kv("fused_vs_separate_int8_values_differing_first_rows", static_cast<double>(ndiff) / h1.size()).kv("fused_vs_separate_max_int8_step", maxd);
        cv.kv("fused_vs_separate_max_rel_scale_diff", maxs).kv("note", "the separate path rounds the norm output to bf16 before quantizing; the fused path quantizes the fp32 value, so single-step differences are expected");
        if (r_fz >= 1.3) ++fast_fz;
        fz_min = std::min(fz_min, r_fz);
        fz_max = std::max(fz_max, r_fz);
        sep_min = std::min(sep_min, r_sep);
        sep_max = std::max(sep_max, r_sep);
        std::printf("       conversions (cold L2): norm->bf16 %.3f ms, norm->int8 fused %.3f ms (extra %+.3f), separate quantize %.3f ms (%.0f GB/s); int8 path incl norm: %.2fx fused, %.2fx separate\n",
                    t_nb, t_nq, extra_fused, t_qs, qbytes / (t_qs * 1e-3) / 1e9, r_fz, r_sep);
      }
      const double op = 2.0 * M * sh.N * sh.K;
      const double tops = op / (t_i8 * 1e-3) / 1e12, tf_bf = op / (t_bf * 1e-3) / 1e12;
      const double speed = t_bf / t_i8, speed_all = (t_bf + t_deq) / (t_i8 + t_q);
      row.kv("status", "ok").kv("quantize_activations_ms", t_q).kv("int8_gemm_ms", t_i8).kv("bf16_dequant_ms", t_deq).kv("bf16_gemm_ms", t_bf);
      row.kv("int8_tops", tops).kv("int8_epsilon_vs_peak", tops / peak8).kv("bf16_tflops", tf_bf).kv("bf16_epsilon_vs_peak", tf_bf / peak16);
      row.kv("int8_over_bf16_gemm_only", speed).kv("int8_over_bf16_incl_conversions", speed_all);
      row.kv("int8_kernel", best->name).kv("int8_order", best->raster ? "grouped8" : "rows").arr_obj("int8_kernels", cj);
      row.kv("weight_scale_prep_ms_once_per_weight", t_prep);
      row.kv("bf16_best_tile", pkc::gemm_cfg_name(bfb.cfg)).kv("bf16_best_order", bfb.raster ? "grouped8" : "rows");
      row.kv("max_rel_err_vs_fp32_on_same_quantized_inputs", worst).kv("correct", correct);
      row.kv("cosine_int8_vs_bf16_path", cs.cosine).kv("rel_l2_int8_vs_bf16_path", cs.rel_l2);
      row.kv_obj("v7", j7).arr_obj("v8", j8s).kv("v8_weight_record_prep_ms_once_per_weight", t_rec).arr_obj("v9", j9s).kv("v9_weight_scale_prep_ms_once_per_weight", t_p9);
      row.arr_obj("v10", j10s).kv("v10_weight_digit_prep_ms_once_per_weight", t_p10).kv_obj("conversions_cold_l2", cv).arr_obj("v11", j11s);
      rows.push_back(row);
      ++measured;
      all_correct = all_correct && correct;
      min_eps8 = std::min(min_eps8, tops / peak8);
      if (speed >= 1.3) ++fast;
      std::printf("  P-K2 %-46s M=%-5d int8 %6.1f TOPS (eps8 %.2f, best %s = %.2fx v2)  bf16 %6.1f  speed %.2fx  %s\n", sh.name, M, tops, tops / peak8,
                  best->name.c_str(), cands[0].ms / best->ms, tf_bf, speed, correct ? "ok" : "MISMATCH");
      std::printf("       v7 %6.1f TOPS (eps8 %.2f) = %.2fx v2, %.2fx bf16  cosine vs bf16 %.6f (per-32 path %.6f)  %s\n", tops7, tops7 / peak8,
                  cands[0].ms / t7, t_bf / t7, cs7.cosine, cs.cosine, ok7 ? "ok" : "MISMATCH");
      std::fflush(stdout);
    }
  }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("int8_peak_tops", peak8).kv("bf16_peak_tflops", peak16).kv("peak_source", "P-H3 sustained, MEASURED at the author's overclock (trainer master 11.12)");
  r.inputs.kv("design", "activations int8 per row per 32 (scale and sum); Q4_K quants unpacked to int8 in registers; mma.sync m16n8k32 s8; per-sub-block fp32 epilogue d*sc*acc - dmin*m*sum");
  r.inputs.kv("v4", "revision 3: exact magic-number int32->fp32 instead of I2F, min term from the precomputed scale*sum, 4-bit words loaded once per K step for both sub-blocks, ldmatrix.x4 for A");
  r.inputs.kv("v7", "revision 7: activations int8 with one scale per 256 (Q8_K-style layout, written from scratch); sum_j sc_j * dot_j in int32 (one IMAD per accumulator per sub-block), d * sa applied once per super-block; min term one FFMA per accumulator per sub-block from (sa * sum) x (dmin * m)");
  r.inputs.kv("v8", "revision 8: v7's arithmetic with 3 or 4 cp.async stages in dynamic shared memory, one barrier per K step, and a 16-byte weight record (d, eight 6-bit scales) per weight row and super-block prepared once per weight");
  r.inputs.kv("v9", "revision 9: v8's arithmetic with 16 warps per 128x128 tile (512 threads, warp tile 32x32, registers capped at 128), weight scales by column so each pair comes in one load; 2 or 3 stages");
  r.inputs.kv("v10", "revision 10: v8's integer main term (1 mma + 4 IMAD per mma); the min term once per super-block as one bf16 m16n8k16 product of exact digits (activation sums as 64 hi + lo, weight mins as 64 m and m), then acc -= sa256 * dmin * that; 2 or 3 stages");
  r.inputs.kv("v11", "revision 11: v10's arithmetic with 16 warps per 128x128 tile (512 threads, warp tile 32x32, registers capped at 128), load work spread over all warps; 2 or 3 stages");
  r.inputs.kv("conversions", "revision 11: activation conversion timed with a cold L2 (64 MiB overwritten before each call); RMSNorm to bf16 against RMSNorm fused with the int8 quantization; ratios include norm, weight dequant (bf16 path) and the fastest correct int8 kernel");
  r.inputs.kv("weights", "synthetic Q4_K").kv("q6k", "not in this revision (P-K2b; its sub-blocks of 16 need m16n8k16 s8)");
  r.values.arr_obj("runs", rows).kv("measured", measured).kv("min_int8_epsilon", measured ? min_eps8 : std::nan("")).kv("shapes_at_least_1p3x", fast);
  r.values.kv("v7_over_v2_min", measured ? v7_min : std::nan("")).kv("v7_over_v2_max", measured ? v7_max : std::nan(""));
  r.values.kv("v7_min_cosine_vs_bf16_path", measured ? v7_min_cos : std::nan("")).kv("v7_shapes_at_least_1p3x_bf16", fast7);
  r.values.kv("v8_best_over_v2_min", measured ? v8_min : std::nan("")).kv("v8_best_over_v2_max", measured ? v8_max : std::nan("")).kv("v8_shapes_at_least_1p3x_bf16", fast8);
  r.values.kv("v9_best_over_v2_min", measured ? v9_min : std::nan("")).kv("v9_best_over_v2_max", measured ? v9_max : std::nan(""));
  r.values.kv("v9_best_eps8_min", measured ? v9_min_eps : std::nan("")).kv("v9_best_eps8_max", measured ? v9_max_eps : std::nan("")).kv("v9_shapes_at_least_1p3x_bf16", fast9);
  r.values.kv("v10_best_over_v2_min", measured ? v10_min : std::nan("")).kv("v10_best_over_v2_max", measured ? v10_max : std::nan(""));
  r.values.kv("v10_best_eps8_min", measured ? v10_min_eps : std::nan("")).kv("v10_best_eps8_max", measured ? v10_max_eps : std::nan(""));
  r.values.kv("v10_best_over_bf16_min", measured ? v10_min_bf : std::nan("")).kv("v10_best_over_bf16_max", measured ? v10_max_bf : std::nan("")).kv("v10_shapes_at_least_1p3x_bf16", fast10);
  r.values.kv("v11_best_over_v2_min", measured ? v11_min : std::nan("")).kv("v11_best_over_v2_max", measured ? v11_max : std::nan(""));
  r.values.kv("v11_best_eps8_min", measured ? v11_min_eps : std::nan("")).kv("v11_best_eps8_max", measured ? v11_max_eps : std::nan(""));
  r.values.kv("v11_best_over_bf16_min", measured ? v11_min_bf : std::nan("")).kv("v11_best_over_bf16_max", measured ? v11_max_bf : std::nan("")).kv("v11_shapes_at_least_1p3x_bf16", fast11);
  r.values.kv("int8_incl_norm_fused_over_bf16_min", measured ? fz_min : std::nan("")).kv("int8_incl_norm_fused_over_bf16_max", measured ? fz_max : std::nan(""));
  r.values.kv("int8_incl_norm_separate_over_bf16_min", measured ? sep_min : std::nan("")).kv("int8_incl_norm_separate_over_bf16_max", measured ? sep_max : std::nan(""));
  r.values.kv("int8_incl_norm_fused_shapes_at_least_1p3x", fast_fz);
  r.values.kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.thresholds.kv("mode_f_default_if_speed_at_least", 1.3).kv("and_kl_not_worse", "P-Q6 (separate)").kv("plan_assumed_epsilon8", "epsilon - 0.1");
  r.thresholds.kv("correct_if_max_rel_err_below", 1e-3);
  r.pass = all_correct && measured > 0 ? -1 : 0;  // record only when correct (adoption needs P-Q6); a wrong kernel fails
  r.headline = "min eps8 " + fmt("%.2f", measured ? min_eps8 : 0.0) + ", " + std::to_string(fast) + "/" + std::to_string(measured) + " shapes >= 1.3x the bf16 path; v7 " + fmt("%.2f", measured ? v7_min : 0.0) + "-" +
               fmt("%.2f", measured ? v7_max : 0.0) + "x v2, " + std::to_string(fast7) + "/" + std::to_string(measured) + " >= 1.3x bf16; v8 " + fmt("%.2f", measured ? v8_min : 0.0) + "-" + fmt("%.2f", measured ? v8_max : 0.0) + "x v2, " +
               std::to_string(fast8) + "/" + std::to_string(measured) + " >= 1.3x bf16; v9 " + fmt("%.2f", measured ? v9_min : 0.0) + "-" +
               fmt("%.2f", measured ? v9_max : 0.0) + "x v2 (eps8 " + fmt("%.2f", measured ? v9_min_eps : 0.0) + "-" + fmt("%.2f", measured ? v9_max_eps : 0.0) + "), " +
               std::to_string(fast9) + "/" + std::to_string(measured) + " >= 1.3x bf16; v10 " + fmt("%.2f", measured ? v10_min : 0.0) + "-" +
               fmt("%.2f", measured ? v10_max : 0.0) + "x v2 (eps8 " + fmt("%.2f", measured ? v10_min_eps : 0.0) + "-" + fmt("%.2f", measured ? v10_max_eps : 0.0) + "), " +
               std::to_string(fast10) + "/" + std::to_string(measured) + " >= 1.3x bf16; incl norm, fused " + fmt("%.2f", measured ? fz_min : 0.0) + "-" +
               fmt("%.2f", measured ? fz_max : 0.0) + "x, " + std::to_string(fast_fz) + "/" + std::to_string(measured) + " >= 1.3x; v11 " + fmt("%.2f", measured ? v11_min : 0.0) + "-" +
               fmt("%.2f", measured ? v11_max : 0.0) + "x v2 (eps8 " + fmt("%.2f", measured ? v11_min_eps : 0.0) + "-" + fmt("%.2f", measured ? v11_max_eps : 0.0) + ")" +
               (all_correct ? "" : ", OUTPUT MISMATCH");
  r.notes = "Mode F (int8) is adopted as default only at >= 1.3x per layer window AND KL no worse (P-Q6). " +
            std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "k2_int8");
  return all_correct ? 0 : 3;
}

// ================================================================ P-K2b (revision 12): split-K, bf16 output, M = 64
// grouped experts, and the layer-order L2 test
namespace k2b {

struct W8 {  // Q4_K weights of `rows` weight rows plus v10 / v12's per-weight preparations
  DevBuf q, rec, dig;
  W8(int rows, int K, unsigned seed, cudaStream_t s)
      : q(static_cast<size_t>(rows) * (K / pkc::kQK) * pkc::kQ4KBytes), rec(static_cast<size_t>(rows) * (K / pkc::kQK) * 16),
        dig(static_cast<size_t>(rows) * (K / pkc::kQK) * 32) {
    const size_t nblk = static_cast<size_t>(rows) * (K / pkc::kQK);
    PP_CK(pkc::fill_q4k(q.as<pkc::BlockQ4K>(), nblk, seed, s));
    PP_CK(pkc::prep_q4k_rec(q.as<pkc::BlockQ4K>(), rec.as(), rows, K, s));
    PP_CK(pkc::prep_q4k_dig(q.as<pkc::BlockQ4K>(), dig.as<uint16_t>(), rows, K, s));
  }
};
struct A8 {  // int8 activations of `rows` rows with one scale per 256, per-32 sums and digits
  DevBuf a8, sa, s32, x32, dig;
  A8(int rows, int K)
      : a8(static_cast<size_t>(rows) * K), sa(static_cast<size_t>(rows) * (K / pkc::kQK) * 4), s32(static_cast<size_t>(rows) * (K / 32) * 4),
        x32(static_cast<size_t>(rows) * (K / 32) * 4), dig(static_cast<size_t>(rows) * (K / pkc::kQK) * 32) {}
  void quantize(const DevBuf& A, int rows, int K, cudaStream_t s) {
    PP_CK(pkc::quantize_rows_q8k_dig(A.as<uint16_t>(), a8.as<int8_t>(), sa.as<float>(), s32.as<int>(), x32.as<float>(), dig.as<uint16_t>(), rows, K, s));
  }
};
inline double w8_bytes(int rows, int K) { return static_cast<double>(rows) * (K / pkc::kQK) * (pkc::kQ4KBytes + 48.0); }
inline double a8_bytes(int rows, int K) { return static_cast<double>(rows) * K * 1.0 + static_cast<double>(rows) * (K / 32) * 8 + static_cast<double>(rows) * (K / pkc::kQK) * 36; }

// worst relative error of the first `rows` rows of C (row stride N, fp32 or bf16) against an fp32 reference (stride rc)
inline double err_rows(const void* C, bool bf, int N, const DevBuf& ref, int rows, int rc, cudaStream_t s) {
  double w = 0;
  if (bf) {
    DevBuf f(static_cast<size_t>(rows) * N * 4);
    PP_CK(pkc::bf16_to_f32(static_cast<const uint16_t*>(C), f.as<float>(), static_cast<size_t>(rows) * N, s));
    for (int r = 0; r < rows; ++r) w = std::max(w, rel_max_err(f.as<float>() + static_cast<size_t>(r) * N, ref.as<float>() + static_cast<size_t>(r) * rc, rc, s));
  } else {
    for (int r = 0; r < rows; ++r)
      w = std::max(w, rel_max_err(static_cast<const float*>(C) + static_cast<size_t>(r) * N, ref.as<float>() + static_cast<size_t>(r) * rc, rc, s));
  }
  return w;
}

}  // namespace k2b

inline int p_k2b(Ctx& c, const Args& a) {
  using namespace k2b;
  Result r;
  r.probe = "P-K2b";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  int sms = 0;
  PP_CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  constexpr size_t kFlushBytes = 64ull << 20;
  DevBuf flush(kFlushBytes);
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  bool all_correct = true;
  const int ref_rows = 16;
  auto tops = [](double M, double N, double K, double ms) { return 2.0 * M * N * K / (ms * 1e-3) / 1e12; };
  auto best_of_rasters = [&](const std::function<cudaError_t(int)>& run, int& raster_out) {
    double best = 1e30;
    for (int raster = 0; raster < 2; ++raster) {
      const cudaError_t e = run(raster);
      if (e == cudaErrorInvalidValue) { (void)cudaGetLastError(); continue; }
      PP_CK(e);
      const double ms = time_ms(s, [&] { PP_CK(run(raster)); }, 80.0);
      if (ms < best) { best = ms; raster_out = raster; }
    }
    return best;
  };

  // ---- A: split-K on the low-tile shapes (router, attn_k), and expert down for comparison
  std::vector<JW> rowsA;
  for (const int si : {5, 3, 4}) {
    const Shape& sh = kShapes[si];
    for (const int M : m_list(c, a)) {
      if (g_abort.load()) break;
      JW row, why;
      row.kv("shape", sh.name).kv("M", M).kv("K", sh.K).kv("N", sh.N);
      const int rc = std::min(sh.N, 1024);
      const double need = static_cast<double>(M) * sh.K * 2 + a8_bytes(M, sh.K) + w8_bytes(sh.N, sh.K) + static_cast<double>(M) * sh.N * 4 * 9 + ref_rows * rc * 4.0 +
                          static_cast<double>(kFlushBytes);
      if (!fits(a, need, why)) {
        row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
        rowsA.push_back(row);
        continue;
      }
      DevBuf A(static_cast<size_t>(M) * sh.K * 2), C(static_cast<size_t>(M) * sh.N * 4), Cref(static_cast<size_t>(ref_rows) * rc * 4),
          part(pkc::splitk_partial_bytes(M, sh.N, 1, 8) + 16);
      PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(M) * sh.K, 1000u + sh.K + M, s));
      W8 w(sh.N, sh.K, 2000u + sh.N, s);
      A8 x(M, sh.K);
      x.quantize(A, M, sh.K, s);
      PP_CK(pkc::ref_q8k_q4k_rows(x.a8.as<int8_t>(), x.sa.as<float>(), w.q.as<pkc::BlockQ4K>(), Cref.as<float>(), ref_rows, rc, sh.K, s));
      int rz = 0;
      const double t10 = best_of_rasters(
          [&](int rr) {
            return pkc::gemm_q8k_q4k_v10(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C.as<float>(), M,
                                         sh.N, sh.K, 1, rr, s);
          },
          rz);
      std::vector<JW> vj;
      double tbest = 1e30;
      std::string nbest;
      for (const int v : {1, 3}) {
        for (const int sp : {1, 2, 4, 8}) {
          if ((sh.K / pkc::kQK) % sp) continue;
          int rr8 = 0;
          auto run = [&](int rr) {
            return pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C.as(), M,
                                         sh.N, sh.K, 1, sp, 0, part.as<float>(), v, rr, s);
          };
          const double t = best_of_rasters(run, rr8);
          if (t >= 1e29) continue;
          PP_CK(run(0));
          const double e = err_rows(C.as(), false, sh.N, Cref, ref_rows, rc, s);
          const bool ok = e <= 1e-3;
          all_correct = all_correct && ok;
          const int tiles = (M / pkc::v12_tile_m(v)) * (sh.N / 128) * sp;
          const int waves = (tiles + sms - 1) / sms;
          JW j;
          j.kv("kernel", pkc::v12_name(v)).kv("splits", sp).kv("work_units", tiles).kv("waves", waves).kv("sm_fill_derived", static_cast<double>(tiles) / (waves * sms));
          j.kv("ms_incl_reduction", t).kv("tops", tops(M, sh.N, sh.K, t)).kv("over_v10", t10 / t).kv("max_rel_err_vs_fp32", e).kv("correct", ok);
          vj.push_back(j);
          if (ok && t < tbest) {
            tbest = t;
            nbest = std::string(pkc::v12_name(v)) + " split " + std::to_string(sp);
          }
        }
      }
      row.kv("status", "ok").kv("v10_ms", t10).kv("v10_tops", tops(M, sh.N, sh.K, t10)).kv("v10_tiles", (M / 128) * (sh.N / 128)).arr_obj("v12", vj);
      row.kv("best", nbest).kv("best_ms", tbest).kv("best_tops", tops(M, sh.N, sh.K, tbest)).kv("best_eps8", tops(M, sh.N, sh.K, tbest) / peak8).kv("best_over_v10", t10 / tbest);
      rowsA.push_back(row);
      std::printf("  split-K %-44s M=%-5d v10 %5.1f TOPS (%d tiles) -> best %s %5.1f TOPS = %.2fx v10\n", sh.name, M, tops(M, sh.N, sh.K, t10), (M / 128) * (sh.N / 128),
                  nbest.c_str(), tops(M, sh.N, sh.K, tbest), t10 / tbest);
      std::fflush(stdout);
    }
  }

  // ---- B: bf16 output against fp32 output (v12 128x128 3-stage, no split)
  std::vector<JW> rowsB;
  for (int si = 0; si < 6; ++si) {  // every shape but lm_head
    const Shape& sh = kShapes[si];
    for (const int M : m_list(c, a)) {
      if (g_abort.load()) break;
      JW row, why;
      row.kv("shape", sh.name).kv("M", M).kv("K", sh.K).kv("N", sh.N);
      const int rc = std::min(sh.N, 1024);
      const double need = static_cast<double>(M) * sh.K * 2 + a8_bytes(M, sh.K) + w8_bytes(sh.N, sh.K) + static_cast<double>(M) * sh.N * 6 + ref_rows * rc * 4.0 +
                          static_cast<double>(ref_rows) * sh.N * 4 + static_cast<double>(kFlushBytes);
      if (!fits(a, need, why)) {
        row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
        rowsB.push_back(row);
        continue;
      }
      DevBuf A(static_cast<size_t>(M) * sh.K * 2), C(static_cast<size_t>(M) * sh.N * 4), Ch(static_cast<size_t>(M) * sh.N * 2), Cref(static_cast<size_t>(ref_rows) * rc * 4);
      PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(M) * sh.K, 1000u + sh.K + M, s));
      W8 w(sh.N, sh.K, 2000u + sh.N, s);
      A8 x(M, sh.K);
      x.quantize(A, M, sh.K, s);
      PP_CK(pkc::ref_q8k_q4k_rows(x.a8.as<int8_t>(), x.sa.as<float>(), w.q.as<pkc::BlockQ4K>(), Cref.as<float>(), ref_rows, rc, sh.K, s));
      double t[2];
      double e[2];
      for (int ob = 0; ob < 2; ++ob) {
        void* dst = ob ? Ch.as() : C.as();
        int rr = 0;
        auto run = [&](int rg) {
          return pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), dst, M, sh.N,
                                       sh.K, 1, 1, ob, nullptr, 1, rg, s);
        };
        t[ob] = best_of_rasters(run, rr);
        PP_CK(run(0));
        e[ob] = err_rows(dst, ob == 1, sh.N, Cref, ref_rows, rc, s);
      }
      const bool ok = e[0] <= 1e-3 && e[1] <= 4e-3;  // bf16 output: one rounding of at most 2^-9 relative
      all_correct = all_correct && ok;
      const double out_mib32 = static_cast<double>(M) * sh.N * 4 / 1048576.0;
      row.kv("status", "ok").kv("fp32_out_ms", t[0]).kv("bf16_out_ms", t[1]).kv("bf16_out_speedup", t[0] / t[1]).kv("fp32_output_mib", out_mib32);
      row.kv("fp32_out_err", e[0]).kv("bf16_out_err", e[1]).kv("correct", ok).kv("bf16_out_tops", tops(M, sh.N, sh.K, t[1]));
      rowsB.push_back(row);
      std::printf("  bf16-out %-44s M=%-5d fp32 out %.3f ms, bf16 out %.3f ms = %.2fx (fp32 output %.0f MiB)  %s\n", sh.name, M, t[0], t[1], t[0] / t[1], out_mib32,
                  ok ? "ok" : "MISMATCH");
      std::fflush(stdout);
    }
  }

  // ---- C: MoE experts at small M: one launch for all experts (grouped) against one expert with split-K, and the bf16
  // kernel grouped the same way
  const int E = static_cast<int>(a.inum("experts", 256));
  const int Me = static_cast<int>(a.inum("expert-m", 64));
  std::vector<JW> rowsC;
  for (const int si : {3, 4}) {  // expert gate,up (2048->512), expert down (512->2048)
    const Shape& sh = kShapes[si];
    if (g_abort.load()) break;
    JW row, why;
    const std::string nm = si == 3 ? "expert gate,up (2048->512)" : "expert down (512->2048)";
    row.kv("shape", nm).kv("experts", E).kv("M_per_expert", Me).kv("K", sh.K).kv("N", sh.N).kv("routing", "equal rows per expert (synthetic)");
    const int rows = E * Me, wrows = E * sh.N, rc = std::min(sh.N, 1024);
    const double need = static_cast<double>(rows) * sh.K * 2 + a8_bytes(rows, sh.K) + w8_bytes(wrows, sh.K) + static_cast<double>(wrows) * sh.K * 2 +
                        static_cast<double>(rows) * sh.N * 4 * 2 + pkc::splitk_partial_bytes(Me, sh.N, 1, 8) + pkc::splitk_partial_bytes(Me, sh.N, E, 2) +
                        ref_rows * rc * 4.0 + static_cast<double>(kFlushBytes);
    if (Me % 64 || !fits(a, need, why)) {
      row.kv("status", Me % 64 ? "skipped: M per expert must be a multiple of 64" : "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
      rowsC.push_back(row);
      continue;
    }
    DevBuf A(static_cast<size_t>(rows) * sh.K * 2), C(static_cast<size_t>(rows) * sh.N * 4), Cb(static_cast<size_t>(rows) * sh.N * 4),
        Wb(static_cast<size_t>(wrows) * sh.K * 2), Cref(static_cast<size_t>(ref_rows) * rc * 4), part(pkc::splitk_partial_bytes(Me, sh.N, 1, 8) + 16),
        partG(pkc::splitk_partial_bytes(Me, sh.N, E, 2) + 16);
    PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(rows) * sh.K, 7000u + sh.K, s));
    W8 w(wrows, sh.K, 8000u + sh.N, s);
    A8 x(rows, sh.K);
    x.quantize(A, rows, sh.K, s);
    const int bpr = sh.K / pkc::kQK;
    std::vector<JW> gj;
    double tg = 1e30;
    std::string ng;
    for (const int v : {2, 3}) {
      for (const int sp : {1, 2}) {
        if (bpr % sp) continue;
        int rr = 0;
        auto run = [&](int rg) {
          return pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C.as(), Me,
                                       sh.N, sh.K, E, sp, 0, sp > 1 ? partG.as<float>() : nullptr, v, rg, s);
        };
        const double t = best_of_rasters(run, rr);
        if (t >= 1e29) continue;
        PP_CK(run(0));
        double e = 0;
        for (const int z : {0, E - 1}) {
          PP_CK(pkc::ref_q8k_q4k_rows(x.a8.as<int8_t>() + static_cast<size_t>(z) * Me * sh.K, x.sa.as<float>() + static_cast<size_t>(z) * Me * bpr,
                                      w.q.as<pkc::BlockQ4K>() + static_cast<size_t>(z) * sh.N * bpr, Cref.as<float>(), ref_rows, rc, sh.K, s));
          e = std::max(e, err_rows(C.as<float>() + static_cast<size_t>(z) * Me * sh.N, false, sh.N, Cref, ref_rows, rc, s));
        }
        const bool ok = e <= 1e-3;
        all_correct = all_correct && ok;
        JW j;
        j.kv("kernel", pkc::v12_name(v)).kv("splits", sp).kv("work_units", (Me / 64) * (sh.N / 128) * E * sp).kv("ms", t).kv("tops", tops(rows, sh.N, sh.K, t));
        j.kv("max_rel_err_vs_fp32_experts_first_last", e).kv("correct", ok);
        gj.push_back(j);
        if (ok && t < tg) {
          tg = t;
          ng = std::string(pkc::v12_name(v)) + " split " + std::to_string(sp);
        }
      }
    }
    // one expert alone with split-K (what a per-expert launch would see)
    std::vector<JW> sj;
    double t1best = 1e30;
    for (const int sp : {1, 2, 4, 8}) {
      if (bpr % sp) continue;
      int rr = 0;
      auto run = [&](int rg) {
        return pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C.as(), Me, sh.N,
                                     sh.K, 1, sp, 0, sp > 1 ? part.as<float>() : nullptr, 3, rg, s);
      };
      const double t = best_of_rasters(run, rr);
      if (t >= 1e29) continue;
      t1best = std::min(t1best, t);
      JW j;
      j.kv("splits", sp).kv("work_units", (Me / 64) * (sh.N / 128) * sp).kv("ms", t).kv("tops", tops(Me, sh.N, sh.K, t)).kv("all_experts_one_by_one_ms_derived", t * E);
      sj.push_back(j);
    }
    // bf16 kernel grouped the same way (64x64 tile), with the per-call weight dequantization of the bf16 path
    const double t_deq = time_ms(s, [&] { PP_CK(pkc::dequant_q4k_to_bf16(w.q.as<pkc::BlockQ4K>(), Wb.as<uint16_t>(), wrows, sh.K, s)); });
    int rb = 0;
    const double tb = best_of_rasters([&](int rg) { return pkc::gemm_bf16_batched(A.as<uint16_t>(), Wb.as<uint16_t>(), Cb.as<float>(), Me, sh.N, sh.K, E, 2, rg, s); }, rb);
    const Cos cs = cosine(C.as<float>(), Cb.as<float>(), static_cast<size_t>(rows) * sh.N, s);
    row.kv("status", "ok").arr_obj("int8_grouped", gj).kv("int8_grouped_best", ng).kv("int8_grouped_best_ms", tg).kv("int8_grouped_best_tops", tops(rows, sh.N, sh.K, tg));
    row.kv("int8_grouped_best_eps8", tops(rows, sh.N, sh.K, tg) / peak8).arr_obj("int8_one_expert_split_k", sj).kv("int8_one_expert_best_ms", t1best);
    row.kv("int8_one_expert_best_tops", tops(Me, sh.N, sh.K, t1best)).kv("grouped_over_one_by_one", t1best * E / tg);
    row.kv("bf16_grouped_ms", tb).kv("bf16_grouped_tflops", tops(rows, sh.N, sh.K, tb)).kv("bf16_dequant_all_experts_ms", t_deq);
    row.kv("int8_over_bf16_gemm_only", tb / tg).kv("int8_over_bf16_incl_dequant", (tb + t_deq) / tg).kv("cosine_int8_vs_bf16_path", cs.cosine);
    rowsC.push_back(row);
    std::printf("  experts %-28s %d x M=%d: int8 grouped %s %5.1f TOPS; one expert best %5.1f TOPS (%.1fx slower one by one); bf16 grouped %5.1f -> int8 %.2fx (%.2fx incl dequant)\n",
                nm.c_str(), E, Me, ng.c_str(), tops(rows, sh.N, sh.K, tg), tops(Me, sh.N, sh.K, t1best), t1best * E / tg, tops(rows, sh.N, sh.K, tb), tb / tg, (tb + t_deq) / tg);
    std::fflush(stdout);
  }

  // ---- D: layer order and L2: norm -> router -> q/k/v -> gate/up on one shared input (K = 2048), int8 against bf16
  // activations. Each step timed inside the sequence (cache cleared only before the norm) and alone (cache cleared
  // right before it). The weights are first touched in both, so the difference is what the shared input gains.
  std::vector<JW> rowsD;
  const int Kd = 2048;
  const int Nd[3] = {256, 8192, 512};
  const char* nd[3] = {"router (2048->256)", "attn_qkv (2048->8192)", "expert gate,up (2048->512)"};
  for (const int M : m_list(c, a)) {
    if (g_abort.load()) break;
    JW row, why;
    row.kv("M", M).kv("K", Kd).kv("order", "norm -> router -> q/k/v -> gate/up").kv("int8_activation_mib", M * static_cast<double>(Kd) / 1048576.0);
    row.kv("bf16_activation_mib", M * static_cast<double>(Kd) * 2 / 1048576.0).kv("l2_mb_published", 36);
    double need = static_cast<double>(M) * Kd * 4 + a8_bytes(M, Kd) + Kd * 2.0 + static_cast<double>(kFlushBytes);
    for (int i = 0; i < 3; ++i) need += w8_bytes(Nd[i], Kd) + static_cast<double>(Nd[i]) * Kd * 2 + static_cast<double>(M) * Nd[i] * 4;
    if (!fits(a, need, why)) {
      row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
      rowsD.push_back(row);
      continue;
    }
    DevBuf X(static_cast<size_t>(M) * Kd * 2), Y(static_cast<size_t>(M) * Kd * 2), gam(Kd * 2);
    PP_CK(pkc::fill_bf16(X.as<uint16_t>(), static_cast<size_t>(M) * Kd, 9000u + M, s));
    PP_CK(pkc::fill_bf16(gam.as<uint16_t>(), Kd, 9100u, s));
    A8 x(M, Kd);
    std::vector<std::unique_ptr<W8>> ws;
    std::vector<DevBuf> wb, cs;
    for (int i = 0; i < 3; ++i) {
      ws.emplace_back(std::make_unique<W8>(Nd[i], Kd, 9200u + Nd[i], s));
      wb.emplace_back(static_cast<size_t>(Nd[i]) * Kd * 2);
      PP_CK(pkc::dequant_q4k_to_bf16(ws[i]->q.as<pkc::BlockQ4K>(), wb[i].as<uint16_t>(), Nd[i], Kd, s));
      cs.emplace_back(static_cast<size_t>(M) * Nd[i] * 4);
    }
    // step 0 = norm, 1..3 = the three GEMMs; path 0 = int8 (fused norm, v10 3-stage), path 1 = bf16 (norm to bf16, bf16 128x128)
    auto step = [&](int path, int i) {
      if (path == 0) {
        if (i == 0) {
          PP_CK(pkc::rmsnorm_q8k(X.as<uint16_t>(), gam.as<uint16_t>(), x.a8.as<int8_t>(), x.sa.as<float>(), x.s32.as<int>(), x.x32.as<float>(), x.dig.as<uint16_t>(), M, Kd, s));
        } else {
          const W8& w = *ws[i - 1];
          PP_CK(pkc::gemm_q8k_q4k_v10(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(),
                                      cs[i - 1].as<float>(), M, Nd[i - 1], Kd, 1, 1, s));
        }
      } else {
        if (i == 0) {
          PP_CK(pkc::rmsnorm_bf16(X.as<uint16_t>(), gam.as<uint16_t>(), Y.as<uint16_t>(), M, Kd, s));
        } else {
          PP_CK(pkc::gemm_bf16_tiled(Y.as<uint16_t>(), wb[i - 1].as<uint16_t>(), cs[i - 1].as<float>(), M, Nd[i - 1], Kd, 0, 1, s));
        }
      }
    };
    std::vector<JW> paths;
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 4; ++i) step(path, i);  // warm-up
      PP_CK(cudaStreamSynchronize(s));
      constexpr int kReps = 11;
      std::vector<double> seq[4], cold[4];
      for (int rep = 0; rep < kReps; ++rep) {
        PP_CK(cudaMemsetAsync(flush.as(), rep & 0xFF, kFlushBytes, s));
        Event ev[5];
        ev[0].record(s);
        for (int i = 0; i < 4; ++i) {
          step(path, i);
          ev[i + 1].record(s);
        }
        ev[4].sync();
        for (int i = 0; i < 4; ++i) seq[i].push_back(Event::ms(ev[i], ev[i + 1]));
        for (int i = 0; i < 4; ++i) {
          PP_CK(cudaMemsetAsync(flush.as(), (rep + i) & 0xFF, kFlushBytes, s));
          Event e0, e1;
          e0.record(s);
          step(path, i);
          e1.record(s);
          e1.sync();
          cold[i].push_back(Event::ms(e0, e1));
        }
      }
      JW pj;
      pj.kv("path", path == 0 ? "int8 (norm fused with quantization, v10 3-stage)" : "bf16 (norm to bf16, bf16 128x128)");
      std::vector<JW> sj;
      double tot_seq = 0, tot_cold = 0;
      for (int i = 0; i < 4; ++i) {
        std::sort(seq[i].begin(), seq[i].end());
        std::sort(cold[i].begin(), cold[i].end());
        const double ms_seq = seq[i][kReps / 2], ms_cold = cold[i][kReps / 2];
        tot_seq += ms_seq;
        tot_cold += ms_cold;
        JW j;
        j.kv("step", i == 0 ? "norm" : nd[i - 1]).kv("ms_in_sequence", ms_seq).kv("ms_alone_cold", ms_cold).kv("sequence_gain", ms_cold / ms_seq - 1.0);
        sj.push_back(j);
      }
      pj.arr_obj("steps", sj).kv("total_in_sequence_ms", tot_seq).kv("total_alone_cold_ms", tot_cold).kv("sequence_gain", tot_cold / tot_seq - 1.0);
      paths.push_back(pj);
      std::printf("  layer order M=%-5d %-5s: in sequence %.3f ms, each alone with a cold cache %.3f ms (gain %+.1f %%)\n", M, path == 0 ? "int8" : "bf16", tot_seq, tot_cold,
                  100.0 * (tot_cold / tot_seq - 1.0));
      std::fflush(stdout);
    }
    row.kv("status", "ok").arr_obj("paths", paths);
    rowsD.push_back(row);
  }

  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("int8_peak_tops", peak8).kv("sm_count", sms).kv("experts", E).kv("expert_m", Me).kv("weights", "synthetic Q4_K").kv("activations", "synthetic bf16");
  r.inputs.kv("kernels", "v12 = v10's arithmetic with split-K (fp32 partials and a reduction kernel), grouped problems, a 64x128 tile and bf16 output; v10 3-stage as reference");
  r.values.arr_obj("split_k", rowsA).arr_obj("bf16_output", rowsB).arr_obj("grouped_experts", rowsC).arr_obj("layer_order_l2", rowsD);
  r.values.kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.thresholds.kv("correct_if_max_rel_err_below", 1e-3).kv("bf16_output_correct_if_below", 4e-3);
  r.pass = all_correct ? -1 : 0;
  r.headline = std::string(all_correct ? "all outputs correct" : "OUTPUT MISMATCH") + "; split-K, bf16 output, grouped experts and layer order recorded";
  r.notes = "Grouped experts use equal rows per expert; real routing gives unequal counts (engine step). " + std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "k2b_v12");
  return all_correct ? 0 : 3;
}

// ================================================================ P-K2 profile run (for Nsight Compute)
// Runs each GEMM kernel once on one shape (--shape substring, default the first; --m, default 4096) with no timing loops,
// so a profiler sees one launch of each: bf16 128x128, int8 v2, v7, v8 3-stage, v8 4-stage. Outputs are checked so a
// profile of a wrong kernel is flagged. Writes no timings: the profiler is the measurement.
inline int p_k2prof(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-K2 profile run";
  init_cuda(c);
  const std::string want = a.get("shape", kShapes[0].name);
  const Shape* shp = nullptr;
  for (const Shape& x : kShapes)
    if (std::string(x.name).find(want) != std::string::npos) { shp = &x; break; }
  if (!shp) throw std::runtime_error("no shape matches --shape " + want);
  const Shape& sh = *shp;
  const int M = static_cast<int>(a.inum("m", 4096));
  Stream s;
  const size_t nblk = static_cast<size_t>(sh.N) * (sh.K / pkc::kQK);
  const size_t mk = static_cast<size_t>(M) * sh.K, mkb = static_cast<size_t>(M) * (sh.K / 32);
  const int ref_rows = 16, ref_cols = std::min(sh.N, 1024);
  JW why;
  const double need = mk * 4.0 + mkb * 28.0 + nblk * (pkc::kQ4KBytes + 180.0) + static_cast<double>(M) * (sh.K / pkc::kQK) * 32 + static_cast<double>(sh.N) * sh.K * 2 + 2.0 * M * sh.N * 4;
  if (!fits(a, need, why)) throw std::runtime_error("shape would cross the VRAM ceiling");
  DevBuf A(mk * 2), a8(mk), sa(mkb * 4), s8(mkb * 4), sas(mkb * 4), a87(mk), sa7(static_cast<size_t>(M) * (sh.K / pkc::kQK) * 4), s327(mkb * 4), x7(mkb * 4),
      Wq(nblk * pkc::kQ4KBytes), wdsc(nblk * 32), wdmn(nblk * 32), rec8(nblk * 16), scT9(nblk * 32), dmnT9(nblk * 32), dT9(nblk * 4), wd10(nblk * 32),
      dig10(static_cast<size_t>(M) * (sh.K / pkc::kQK) * 32), Wb(static_cast<size_t>(sh.N) * sh.K * 2), C(static_cast<size_t>(M) * sh.N * 4),
      Cref(static_cast<size_t>(ref_rows) * ref_cols * 4), Cref7(static_cast<size_t>(ref_rows) * ref_cols * 4);
  PP_CK(pkc::fill_bf16(A.as<uint16_t>(), mk, 1000u + sh.K + M, s));
  PP_CK(pkc::fill_q4k(Wq.as<pkc::BlockQ4K>(), nblk, 2000u + sh.N, s));
  PP_CK(pkc::quantize_rows_q8(A.as<uint16_t>(), a8.as<int8_t>(), sa.as<float>(), s8.as<int>(), sas.as<float>(), M, sh.K, s));
  PP_CK(pkc::quantize_rows_q8k_dig(A.as<uint16_t>(), a87.as<int8_t>(), sa7.as<float>(), s327.as<int>(), x7.as<float>(), dig10.as<uint16_t>(), M, sh.K, s));
  PP_CK(pkc::prep_q4k_dig(Wq.as<pkc::BlockQ4K>(), wd10.as<uint16_t>(), sh.N, sh.K, s));
  PP_CK(pkc::prep_q4k_scales(Wq.as<pkc::BlockQ4K>(), wdsc.as<float>(), wdmn.as<float>(), sh.N, sh.K, s));
  PP_CK(pkc::prep_q4k_rec(Wq.as<pkc::BlockQ4K>(), rec8.as(), sh.N, sh.K, s));
  PP_CK(pkc::prep_q4k_v9(Wq.as<pkc::BlockQ4K>(), scT9.as<int>(), dmnT9.as<float>(), dT9.as<float>(), sh.N, sh.K, s));
  PP_CK(pkc::dequant_q4k_to_bf16(Wq.as<pkc::BlockQ4K>(), Wb.as<uint16_t>(), sh.N, sh.K, s));
  PP_CK(pkc::ref_q8_q4k_rows(a8.as<int8_t>(), sa.as<float>(), Wq.as<pkc::BlockQ4K>(), Cref.as<float>(), ref_rows, ref_cols, sh.K, s));
  PP_CK(pkc::ref_q8k_q4k_rows(a87.as<int8_t>(), sa7.as<float>(), Wq.as<pkc::BlockQ4K>(), Cref7.as<float>(), ref_rows, ref_cols, sh.K, s));
  auto err = [&](const DevBuf& ref) {
    double w = 0;
    for (int rr = 0; rr < ref_rows; ++rr)
      w = std::max(w, rel_max_err(C.as<float>() + static_cast<size_t>(rr) * sh.N, ref.as<float>() + static_cast<size_t>(rr) * ref_cols, ref_cols, s));
    return w;
  };
  std::vector<JW> ks;
  auto one = [&](const char* name, const std::function<cudaError_t()>& fn, const DevBuf* ref) {
    PP_CK(fn());
    PP_CK(cudaStreamSynchronize(s));
    JW j;
    j.kv("kernel", name);
    if (ref) {
      const double e = err(*ref);
      j.kv("max_rel_err_vs_fp32", e).kv("correct", e <= 1e-3);
      std::printf("  ran %-16s err %.1e %s\n", name, e, e <= 1e-3 ? "ok" : "MISMATCH");
    } else {
      std::printf("  ran %-16s\n", name);
    }
    ks.push_back(j);
  };
  one("bf16 128x128", [&] { return pkc::gemm_bf16_tiled(A.as<uint16_t>(), Wb.as<uint16_t>(), C.as<float>(), M, sh.N, sh.K, 0, 1, s); }, nullptr);
  one("int8 v2", [&] { return pkc::gemm_q8_q4k(a8.as<int8_t>(), sa.as<float>(), s8.as<int>(), Wq.as<pkc::BlockQ4K>(), C.as<float>(), M, sh.N, sh.K, 1, s); }, &Cref);
  one("int8 v7", [&] { return pkc::gemm_q8k_q4k_v7(a87.as<int8_t>(), sa7.as<float>(), x7.as<float>(), Wq.as<pkc::BlockQ4K>(), wdmn.as<float>(), C.as<float>(), M, sh.N, sh.K, 1, s); },
      &Cref7);
  for (int v = 0; v < pkc::kV8Variants; ++v)
    one(pkc::v8_name(v),
        [&] {
          return pkc::gemm_q8k_q4k_v8(a87.as<int8_t>(), sa7.as<float>(), x7.as<float>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wdmn.as<float>(), C.as<float>(), M, sh.N,
                                      sh.K, v, 1, s);
        },
        &Cref7);
  for (int v = 0; v < pkc::kV9Variants; ++v)
    one(pkc::v9_name(v),
        [&] {
          return pkc::gemm_q8k_q4k_v9(a87.as<int8_t>(), sa7.as<float>(), x7.as<float>(), Wq.as<pkc::BlockQ4K>(), scT9.as<int>(), dmnT9.as<float>(), dT9.as<float>(),
                                      C.as<float>(), M, sh.N, sh.K, v, 1, s);
        },
        &Cref7);
  for (int v = 0; v < pkc::kV11Variants; ++v)
    one(pkc::v11_name(v),
        [&] {
          return pkc::gemm_q8k_q4k_v11(a87.as<int8_t>(), sa7.as<float>(), dig10.as<uint16_t>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wd10.as<uint16_t>(), C.as<float>(), M,
                                       sh.N, sh.K, v, 1, s);
        },
        &Cref7);
  for (int v = 0; v < pkc::kV10Variants; ++v)
    one(pkc::v10_name(v),
        [&] {
          return pkc::gemm_q8k_q4k_v10(a87.as<int8_t>(), sa7.as<float>(), dig10.as<uint16_t>(), Wq.as<pkc::BlockQ4K>(), rec8.as(), wd10.as<uint16_t>(), C.as<float>(), M,
                                       sh.N, sh.K, v, 1, s);
        },
        &Cref7);
  r.inputs.kv("shape", sh.name).kv("M", M).kv("K", sh.K).kv("N", sh.N).kv("order", "grouped8");
  r.values.arr_obj("kernels", ks);
  r.pass = -1;
  r.headline = "one launch of each GEMM kernel on " + std::string(sh.name) + " M=" + std::to_string(M) + " for the profiler";
  r.notes = "No timings here: Nsight Compute is the measurement. Kernel names in the profile: k_gemm_bf16_t, k_gemm_q8_q4k, k_gemm_q8k_q4k_v7, k_gemm_q8k_q4k_v8 (3 then 4 stages), k_gemm_q8k_q4k_v9 (2 then 3 stages), k_gemm_q8k_q4k_v11 (2 then 3 stages), k_gemm_q8k_q4k_v10 (2 then 3 stages).";
  write_result(c, r, "k2prof");
  return 0;
}

// ================================================================ P-K3
inline int p_k3(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-K3";
  init_cuda(c);
  const double mib = a.num("k3-mib", c.quick ? 256 : 1024);
  const double published_bw = a.num("vram-gbs-published", 504.2);  // RTX 4070, PUBLISHED (NVIDIA spec page, masterwork S2)
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c), why;
  const size_t n4 = static_cast<size_t>(mib * kMiB / pkc::kQ4KBytes) / 32 * 32;
  const size_t n6 = static_cast<size_t>(mib * kMiB / pkc::kQ6KBytes) / 32 * 32;
  const double need = static_cast<double>(n4) * pkc::kQ4KBytes + static_cast<double>(n6) * pkc::kQ6KBytes;
  if (!fits(a, need, why)) {
    r.status = "skipped";
    r.values.kv_obj("cap", why);
    r.headline = "skipped: would cross the VRAM ceiling";
    write_result(c, r, "k3_dequant");
    return 0;
  }
  DevBuf w4(n4 * pkc::kQ4KBytes), w6(n6 * pkc::kQ6KBytes);
  const int blocks = gpu_blocks(c, 4);
  DevBuf sink(static_cast<size_t>(blocks) * 256 * sizeof(float)), usink(sizeof(unsigned long long));
  PP_CK(pkc::fill_q4k(w4.as<pkc::BlockQ4K>(), n4, 31u, s));
  PP_CK(pkc::fill_q6k(w6.as<pkc::BlockQ6K>(), n6, 37u, s));
  const double e4 = check_q4k_dequant(w4.as<pkc::BlockQ4K>(), 64, s);
  const double e6 = check_q6k_dequant(w6.as<pkc::BlockQ6K>(), 64, s);
  const double b4 = static_cast<double>(n4) * pkc::kQ4KBytes, b6 = static_cast<double>(n6) * pkc::kQ6KBytes;
  const double t_r4 = time_ms(s, [&] { PP_CK(pk::launch_read(w4.as(), static_cast<size_t>(b4 / 16), usink.as<unsigned long long>(), gpu_blocks(c, 8), 256, s)); });
  const double t_d4 = time_ms(s, [&] { PP_CK(pkc::stream_dequant_q4k(w4.as<pkc::BlockQ4K>(), n4, sink.as<float>(), blocks, s)); });
  const double t_r6 = time_ms(s, [&] { PP_CK(pk::launch_read(w6.as(), static_cast<size_t>(b6 / 16), usink.as<unsigned long long>(), gpu_blocks(c, 8), 256, s)); });
  const double t_d6 = time_ms(s, [&] { PP_CK(pkc::stream_dequant_q6k(w6.as<pkc::BlockQ6K>(), n6, sink.as<float>(), gpu_blocks(c, 8), s)); });
  auto gbs = [](double bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; };
  const double r4 = gbs(b4, t_r4), d4 = gbs(b4, t_d4), r6 = gbs(b6, t_r6), d6 = gbs(b6, t_d6);
  auto verdict = [](double frac) { return frac < 0.60 ? "ALU-bound (below 60 % of the plain read)" : "memory-bound (at or above 60 % of the plain read)"; };
  JW q4, q6;
  q4.kv("bytes", b4).kv("plain_read_gbs", r4).kv("dequant_stream_gbs", d4).kv("dequant_over_plain_read", d4 / r4).kv("dequant_over_published_bw", d4 / published_bw);
  q4.kv("weights_per_second_g", d4 / pkc::kQ4KBytes * pkc::kQK).kv("verdict", verdict(d4 / r4)).kv("dequant_max_rel_err_vs_host", e4);
  q6.kv("bytes", b6).kv("plain_read_gbs", r6).kv("dequant_stream_gbs", d6).kv("dequant_over_plain_read", d6 / r6).kv("dequant_over_published_bw", d6 / published_bw);
  q6.kv("weights_per_second_g", d6 / pkc::kQ6KBytes * pkc::kQK).kv("verdict", verdict(d6 / r6)).kv("dequant_max_rel_err_vs_host", e6);
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("buffer_mib_each", mib).kv("published_vram_gbs", published_bw).kv("blocks_q4k", blocks).kv("threads_q4k", 256).kv("threads_q6k", 128);
  r.values.kv_obj("q4_k", q4).kv_obj("q6_k", q6).kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.thresholds.kv("alu_bound_if_below_fraction_of_plain_read", 0.60).kv("dequant_correct_if_rel_err_below", 1e-5);
  r.pass = (e4 < 1e-5 && e6 < 1e-5) ? -1 : 0;
  r.headline = "Q4_K " + fmt("%.0f", d4) + " of " + fmt("%.0f", r4) + " GB/s (" + fmt("%.0f", 100 * d4 / r4) + " %), Q6_K " + fmt("%.0f", d6) + " of " +
               fmt("%.0f", r6) + " GB/s (" + fmt("%.0f", 100 * d6 / r6) + " %)";
  r.notes = "The plain read of the same bytes in the same run is the bandwidth reference; the published figure is listed for comparison. " +
            std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "k3_dequant");
  std::printf("  P-K3 %s\n", r.headline.c_str());
  return r.pass == 0 ? 3 : 0;
}

// ================================================================ P-Q10
inline int p_q10(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-Q10";
  init_cuda(c);
  const int M = static_cast<int>(a.inum("m", 2048));
  const double fp8_peak = a.num("fp8-peak", 129.8), bf16_peak = a.num("bf16-peak", 65.2);  // MEASURED P-H3 (overclock)
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  std::vector<JW> rows;
  bool e4_ok = true, e5_ok = true;
  int measured = 0;
  double min_e4 = 1, min_e5 = 1;
  for (const Shape& sh : kShapes) {
    if (sh.N > 16384 || g_abort.load()) continue;  // lm_head: no fp8 data gradient is planned for it
    for (int dist = 0; dist < 2; ++dist) {
      JW row, why;
      row.kv("shape", sh.name).kv("M", M).kv("K", sh.K).kv("N", sh.N).kv("dY_distribution", dist == 0 ? "gaussian" : "heavy-tailed (gaussian x exp(gaussian))");
      const double need = 4.0 * (2.0 * M * sh.N + static_cast<double>(sh.N) * sh.K + 2.0 * M * sh.K) + 2.0 * sh.N * sh.K;
      if (!fits(a, need, why)) {
        row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
        rows.push_back(row);
        continue;
      }
      DevBuf dY(static_cast<size_t>(M) * sh.N * 4), dYq(static_cast<size_t>(M) * sh.N * 4), Wh(static_cast<size_t>(sh.N) * sh.K * 2),
          W(static_cast<size_t>(sh.N) * sh.K * 4), ref(static_cast<size_t>(M) * sh.K * 4), out(static_cast<size_t>(M) * sh.K * 4);
      PP_CK(pkc::fill_f32(dY.as<float>(), static_cast<size_t>(M) * sh.N, 500u + sh.N + dist, dist, 1e-3f, s));
      PP_CK(pkc::fill_bf16(Wh.as<uint16_t>(), static_cast<size_t>(sh.N) * sh.K, 600u + sh.K, s));
      PP_CK(pkc::bf16_to_f32(Wh.as<uint16_t>(), W.as<float>(), static_cast<size_t>(sh.N) * sh.K, s));
      PP_CK(pkc::gemm_f32_nn(dY.as<float>(), W.as<float>(), ref.as<float>(), M, sh.N, sh.K, s));
      for (int kind = 0; kind < 2; ++kind) {
        PP_CK(pkc::fp8_roundtrip_rows(dY.as<float>(), dYq.as<float>(), M, sh.N, kind, s));
        PP_CK(pkc::gemm_f32_nn(dYq.as<float>(), W.as<float>(), out.as<float>(), M, sh.N, sh.K, s));
        const Cos cs = cosine(out.as<float>(), ref.as<float>(), static_cast<size_t>(M) * sh.K, s);
        const char* key = kind == 0 ? "e4m3" : "e5m2";
        row.kv(std::string(key) + "_cosine", cs.cosine).kv(std::string(key) + "_rel_l2", cs.rel_l2);
        if (kind == 0) { min_e4 = std::min(min_e4, cs.cosine); e4_ok = e4_ok && cs.cosine >= 0.999; }
        else { min_e5 = std::min(min_e5, cs.cosine); e5_ok = e5_ok && cs.cosine >= 0.999; }
      }
      row.kv("status", "ok");
      rows.push_back(row);
      ++measured;
    }
  }
  const std::vector<Sample> all = sampler.stop();
  const double speed = fp8_peak / bf16_peak;
  r.inputs.kv("M", M).kv("scaling", "per row of dY: amax / 448 (e4m4) or amax / 57344 (e5m2)").kv("weights", "bf16, gaussian").kv("dY_scale", 1e-3);
  r.inputs.kv("fp8_peak_tflops", fp8_peak).kv("bf16_peak_tflops", bf16_peak).kv("peak_source", "P-H3 sustained, MEASURED at the author's overclock (trainer master 11.12)");
  r.values.arr_obj("runs", rows).kv("measured", measured).kv("min_cosine_e4m3", min_e4).kv("min_cosine_e5m2", min_e5);
  r.values.kv("e4m3_all_at_least_0p999", e4_ok).kv("e5m2_all_at_least_0p999", e5_ok).kv("speed_fp8_over_bf16_from_peaks_DERIVED", speed);
  r.values.kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.thresholds.kv("adopt_only_at_cosine_at_least", 0.999).kv("and_speed_at_least", 1.3).kv("status_in_plan", "measured only; not planned");
  r.pass = -1;
  r.headline = "min cosine e4m4 " + fmt("%.6f", min_e4) + ", e5m2 " + fmt("%.6f", min_e5) + "; fp8/bf16 peak ratio " + fmt("%.2f", speed) + " (DERIVED)";
  r.notes = "Synthetic gradients: a real-gradient repeat needs the trainer (step 8). The products are computed in fp32 from the fp8-rounded values, which "
            "is what an fp8 mma with fp32 accumulation computes up to summation order. " +
            std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "q10_fp8");
  std::printf("  P-Q10 %s\n", r.headline.c_str());
  return 0;
}

// ================================================================ O13: persistent v13 against v12 (2026-10-30)
// Expert down (512->2048) and, as the K = 2048 control, expert gate/up (2048->512): dense at M = 2048 and 4096, and
// grouped (--experts 256 x --expert-m 64, one launch). Every v12 variant (splits 1) and every v13 variant (persistent,
// at the resident block count and, for the 64-row tile, also capped at 1 block per SM), best of both tile orders. Each v13
// result must equal the same v12 variant's bit for bit (max |diff| 0) and stay within 1e-3 of fp32 on the first rows.
inline int p_o13(Ctx& c, const Args& a) {
  using namespace k2b;
  Result r;
  r.probe = "O13";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  const int ref_rows = 16;
  const int E = static_cast<int>(a.inum("experts", 256));
  const int Me = static_cast<int>(a.inum("expert-m", 64));
  auto tops = [](double M, double N, double K, double ms) { return 2.0 * M * N * K / (ms * 1e-3) / 1e12; };
  bool all_ok = true;
  std::vector<JW> rows;
  double best_gain_down = 0.0;
  struct Case {
    int si, M, batch;
  };
  std::vector<Case> cases;
  for (const int si : {4, 3})
    for (const int M : {2048, 4096}) cases.push_back({si, M, 1});
  for (const int si : {4, 3}) cases.push_back({si, Me, E});
  std::vector<JW> infos;
  for (int fam = 0; fam < 3; ++fam)
    for (int v = 0; v < pkc::kV12Variants; ++v) {
      int regs = 0, loc = 0, sm = 0, bps = 0;
      PP_CK(pkc::v12_13_info(fam, v, &regs, &loc, &sm, &bps));
      const char* fn = fam == 0 ? "v12" : (fam == 1 ? "v13" : "v13b");
      JW j;
      j.kv("family", fn).kv("variant", v).kv("registers", regs).kv("local_spill_bytes", loc).kv("shared_bytes", sm).kv("blocks_per_sm", bps);
      infos.push_back(j);
      std::printf("    %-4s variant %d (%s): %3d registers, %4d spill bytes, %6d shared bytes, %d blocks per SM\n", fn, v, pkc::v12_name(v) + 4, regs, loc, sm, bps);
    }
  std::fflush(stdout);
  for (const Case& cs : cases) {
    if (g_abort.load()) break;
    const Shape& sh = kShapes[cs.si];
    const int M = cs.M, B = cs.batch, rowsA = M * B, wrows = sh.N * B, rc = std::min(sh.N, 1024);
    JW row, why;
    row.kv("shape", sh.name).kv("M", M).kv("problems", B).kv("K", sh.K).kv("N", sh.N);
    const double need = static_cast<double>(rowsA) * sh.K * 2 + a8_bytes(rowsA, sh.K) + w8_bytes(wrows, sh.K) + static_cast<double>(rowsA) * sh.N * 4 * 2 +
                        ref_rows * rc * 4.0;
    if (!fits(a, need, why)) {
      row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
      rows.push_back(row);
      continue;
    }
    DevBuf A(static_cast<size_t>(rowsA) * sh.K * 2), C12(static_cast<size_t>(rowsA) * sh.N * 4), C13(static_cast<size_t>(rowsA) * sh.N * 4),
        Cref(static_cast<size_t>(ref_rows) * rc * 4);
    PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(rowsA) * sh.K, 7000u + sh.K + M, s));
    W8 w(wrows, sh.K, 8000u + sh.N, s);
    A8 x(rowsA, sh.K);
    x.quantize(A, rowsA, sh.K, s);
    PP_CK(pkc::ref_q8k_q4k_rows(x.a8.as<int8_t>(), x.sa.as<float>(), w.q.as<pkc::BlockQ4K>(), Cref.as<float>(), ref_rows, rc, sh.K, s));
    std::vector<JW> kj;
    double t12 = 1e30, t13 = 1e30;
    std::string n12, n13;
    for (int v = 0; v < pkc::kV12Variants; ++v) {
      if (M % pkc::v12_tile_m(v)) continue;
      // v12, splits 1, best tile order
      double tv12 = 1e30;
      int rg12 = 0;
      for (int rg = 0; rg < 2; ++rg) {
        auto run = [&] { PP_CK(pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(),
                                                     C12.as(), M, sh.N, sh.K, B, 1, 0, nullptr, v, rg, s)); };
        const double t = time_ms(s, run, 80.0);
        if (t < tv12) { tv12 = t; rg12 = rg; }
      }
      PP_CK(pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C12.as(), M,
                                  sh.N, sh.K, B, 1, 0, nullptr, v, rg12, s));
      JW j12;
      j12.kv("kernel", pkc::v12_name(v)).kv("raster", rg12).kv("blocks", (M / pkc::v12_tile_m(v)) * (sh.N / 128) * B).kv("ms", tv12).kv("tops", tops(rowsA, sh.N, sh.K, tv12));
      j12.kv("eps8", tops(rowsA, sh.N, sh.K, tv12) / peak8);
      kj.push_back(j12);
      if (tv12 < t12) { t12 = tv12; n12 = pkc::v12_name(v); }
      // v13 at the resident count, and capped at 1 block per SM for the 64-row tile (2 resident)
      for (const int fam : {1, 2})
      for (const int cap : {0, 1}) {
        if (cap == 1 && pkc::v12_tile_m(v) != 64) continue;
        const bool hb = fam == 2;
        const std::string kname = std::string(pkc::v13_name(v)) + (hb ? " (v13b)" : "");
        double tv13 = 1e30;
        int rg13 = 0, grid = 0;
        for (int rg = 0; rg < 2; ++rg) {
          auto run = [&] { PP_CK(pkc::gemm_q8k_q4k_v13(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(),
                                                       w.dig.as<uint16_t>(), C13.as(), M, sh.N, sh.K, B, 0, v, rg, cap, &grid, s, hb)); };
          const double t = time_ms(s, run, 80.0);
          if (t < tv13) { tv13 = t; rg13 = rg; }
        }
        PP_CK(cudaMemsetAsync(C13.as(), 0xFF, static_cast<size_t>(rowsA) * sh.N * 4, s));  // NaN fill: a tile not written shows up
        // the same variant and tile order as v12's best, for the bit-for-bit check
        PP_CK(pkc::gemm_q8k_q4k_v13(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C13.as(), M,
                                    sh.N, sh.K, B, 0, v, rg12, cap, &grid, s, hb));
        double absd = 0;
        (void)rel_max_err(C13.as<float>(), C12.as<float>(), static_cast<size_t>(rowsA) * sh.N, s, &absd);
        const Cos cc = cosine(C13.as<float>(), C12.as<float>(), static_cast<size_t>(rowsA) * sh.N, s);
        const bool same = absd == 0.0 && std::isfinite(cc.cosine) && cc.cosine > 0.999999;
        const double eref = err_rows(C13.as(), false, sh.N, Cref, ref_rows, rc, s);
        const bool ok = same && eref <= 1e-3;
        all_ok = all_ok && ok;
        JW j13;
        j13.kv("kernel", kname).kv("blocks_per_sm_cap", cap).kv("grid", grid).kv("raster", rg13).kv("ms", tv13).kv("tops", tops(rowsA, sh.N, sh.K, tv13));
        j13.kv("eps8", tops(rowsA, sh.N, sh.K, tv13) / peak8).kv("over_v12_same_variant", tv12 / tv13).kv("max_abs_diff_vs_v12", absd).kv("bit_equal_v12", same);
        j13.kv("max_rel_err_vs_fp32_first_rows", eref).kv("correct", ok);
        kj.push_back(j13);
        if (ok && tv13 < t13) { t13 = tv13; n13 = kname + (cap ? " (1 block per SM)" : ""); }
        std::printf("    %-34s M=%4d x%3d  %-37s cap %d grid %4d: %.4f ms %6.1f TOPS (%.3f of peak) vs v12 %.4f ms -> %.3fx; |diff| %g; vs fp32 %.2e%s\n",
                    sh.name, M, B, kname.c_str(), cap, grid, tv13, tops(rowsA, sh.N, sh.K, tv13), tops(rowsA, sh.N, sh.K, tv13) / peak8, tv12, tv12 / tv13, absd, eref,
                    ok ? "" : "  FAIL");
        std::fflush(stdout);
      }
    }
    const double gain = t12 / t13;
    if (cs.si == 4) best_gain_down = std::max(best_gain_down, gain);
    row.kv("status", "ok").arr_obj("kernels", kj).kv("v12_best", n12).kv("v12_best_ms", t12).kv("v12_best_eps8", tops(rowsA, sh.N, sh.K, t12) / peak8);
    row.kv("v13_best", n13).kv("v13_best_ms", t13).kv("v13_best_eps8", tops(rowsA, sh.N, sh.K, t13) / peak8).kv("v13_over_v12_best", gain);
    rows.push_back(row);
    std::printf("  %-34s M=%4d x%3d: v12 best %s %.3f of peak; v13 best %s %.3f of peak; %.3fx\n", sh.name, M, B, n12.c_str(), tops(rowsA, sh.N, sh.K, t12) / peak8,
                n13.c_str(), tops(rowsA, sh.N, sh.K, t13) / peak8, gain);
    std::fflush(stdout);
  }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("int8_peak_tops", peak8).kv("experts", E).kv("expert_m", Me).kv("weights_activations", "synthetic (fill_q4k / fill_bf16), as P-K2b");
  r.values.arr_obj("kernel_info", infos).arr_obj("cases", rows).kv("all_v13_bit_equal_v12_and_correct", all_ok).kv("best_gain_expert_down", best_gain_down);
  r.values.kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.thresholds.kv("o13_goal_eps8_above", 0.29).kv("adopt_if_faster_and_bit_equal", true);
  r.pass = all_ok ? -1 : 0;
  r.headline = std::string(all_ok ? "v13 / v13b bit-equal to v12 everywhere" : "v13 / v13b MISMATCH") + "; best expert-down gain " + fmt("%.3f", best_gain_down) + "x";
  r.notes = std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "o13_persistent");
  std::printf("  O13 %s\n", r.headline.c_str());
  return all_ok ? 0 : 3;
}

// ================================================================ B: per-32 activation scales in the prompt GEMM (2026-10-30)
// v12 (one activation scale per 256, the 10-26 kernels) against v12x8 (one per 32: the adopted gpu256x8 format) on the
// dense shapes that run (attn_qkv 2048->8192, ssm_out 4096->2048, gate/up 2048->512, expert down 512->2048) at M 2048
// and 4096, and grouped (256 experts x M 64: expert gate/up and down). Each kernel checked against fp32 of its own
// rounded inputs (16 rows); both compared with the bf16 path (Q4_K dequantized to bf16, bf16 activations, fp32
// accumulate) as the accuracy view: what the 8-bit activations cost against unrounded activations. Speed: best of every
// variant and both tile orders, warm, as P-K2b.
inline int p_act32(Ctx& c, const Args& a) {
  using namespace k2b;
  Result r;
  r.probe = "ACT32";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  const int ref_rows = 16;
  const int E = static_cast<int>(a.inum("experts", 256));
  const int Me = static_cast<int>(a.inum("expert-m", 64));
  auto tops = [](double M, double N, double K, double ms) { return 2.0 * M * N * K / (ms * 1e-3) / 1e12; };
  std::vector<JW> infos;
  for (int v = 0; v < pkc::kV12Variants; ++v) {
    int regs = 0, loc = 0, sm = 0, bps = 0;
    PP_CK(pkc::v12x8_info(v, &regs, &loc, &sm, &bps));
    JW j;
    j.kv("kernel", pkc::v12x8_name(v)).kv("registers", regs).kv("local_spill_bytes", loc).kv("shared_bytes", sm).kv("blocks_per_sm", bps);
    infos.push_back(j);
    std::printf("    %-24s %3d registers, %4d spill bytes, %6d shared bytes, %d blocks per SM\n", pkc::v12x8_name(v), regs, loc, sm, bps);
  }
  struct Case {
    int si, M, batch;
  };
  std::vector<Case> cases;
  for (const int si : {0, 2, 3, 4})
    for (const int M : {2048, 4096}) cases.push_back({si, M, 1});
  for (const int si : {3, 4}) cases.push_back({si, Me, E});
  bool all_ok = true;
  std::vector<JW> rows;
  for (const Case& cs : cases) {
    if (g_abort.load()) break;
    const Shape& sh = kShapes[cs.si];
    const int M = cs.M, B = cs.batch, rowsA = M * B, wrows = sh.N * B, rc = std::min(sh.N, 1024), bpr = sh.K / pkc::kQK;
    JW row, why;
    row.kv("shape", sh.name).kv("M", M).kv("problems", B).kv("K", sh.K).kv("N", sh.N);
    const double need = static_cast<double>(rowsA) * sh.K * 2 + 2 * a8_bytes(rowsA, sh.K) + w8_bytes(wrows, sh.K) + static_cast<double>(wrows) * bpr * 32 +
                        static_cast<double>(wrows) * sh.K * 2 + static_cast<double>(rowsA) * sh.N * 4 * 3 + ref_rows * rc * 4.0 * 2;
    if (!fits(a, need, why)) {
      row.kv("status", "skipped: would cross the VRAM ceiling").kv_obj("cap", why);
      rows.push_back(row);
      continue;
    }
    DevBuf A(static_cast<size_t>(rowsA) * sh.K * 2), C256(static_cast<size_t>(rowsA) * sh.N * 4), C32(static_cast<size_t>(rowsA) * sh.N * 4),
        Cbf(static_cast<size_t>(rowsA) * sh.N * 4), Wb(static_cast<size_t>(wrows) * sh.K * 2), R256(static_cast<size_t>(ref_rows) * rc * 4),
        R32(static_cast<size_t>(ref_rows) * rc * 4);
    PP_CK(pkc::fill_bf16(A.as<uint16_t>(), static_cast<size_t>(rowsA) * sh.K, 7100u + sh.K + M, s));
    W8 w(wrows, sh.K, 8100u + sh.N, s);
    DevBuf wd8(static_cast<size_t>(wrows) * bpr * 32);
    PP_CK(pkc::prep_q4k_dig8(w.q.as<pkc::BlockQ4K>(), wd8.as<uint16_t>(), wrows, sh.K, s));
    A8 x(rowsA, sh.K);  // per 256 (v12's)
    x.quantize(A, rowsA, sh.K, s);
    DevBuf a32(static_cast<size_t>(rowsA) * sh.K), s32(static_cast<size_t>(rowsA) * (sh.K / 32) * 4), m32(static_cast<size_t>(rowsA) * (sh.K / 32) * 4),
        x32(static_cast<size_t>(rowsA) * (sh.K / 32) * 4), d32(static_cast<size_t>(rowsA) * bpr * 32);
    PP_CK(pkc::quantize_rows_q8(A.as<uint16_t>(), a32.as<int8_t>(), s32.as<float>(), m32.as<int>(), x32.as<float>(), rowsA, sh.K, s));
    PP_CK(pkc::dig_x32(x32.as<float>(), d32.as<uint16_t>(), rowsA, sh.K, s));
    // references of each kernel's own inputs (first rows of problem 0), and the bf16 path for all rows
    PP_CK(pkc::ref_q8k_q4k_rows(x.a8.as<int8_t>(), x.sa.as<float>(), w.q.as<pkc::BlockQ4K>(), R256.as<float>(), ref_rows, rc, sh.K, s));
    PP_CK(pkc::ref_q8_q4k_rows(a32.as<int8_t>(), s32.as<float>(), w.q.as<pkc::BlockQ4K>(), R32.as<float>(), ref_rows, rc, sh.K, s));
    PP_CK(pkc::dequant_q4k_to_bf16(w.q.as<pkc::BlockQ4K>(), Wb.as<uint16_t>(), wrows, sh.K, s));
    PP_CK(pkc::gemm_bf16_batched(A.as<uint16_t>(), Wb.as<uint16_t>(), Cbf.as<float>(), M, sh.N, sh.K, B, M % 128 ? 2 : 0, 0, s));
    double t256 = 1e30, t32 = 1e30;
    std::string n256, n32;
    std::vector<JW> kj;
    for (int fam = 0; fam < 2; ++fam)
      for (int v = 0; v < pkc::kV12Variants; ++v) {
        if (M % pkc::v12_tile_m(v)) continue;
        double best = 1e30;
        int rgb = 0;
        for (int rg = 0; rg < 2; ++rg) {
          auto run = [&] {
            if (fam == 0)
              PP_CK(pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(),
                                          C256.as(), M, sh.N, sh.K, B, 1, 0, nullptr, v, rg, s));
            else
              PP_CK(pkc::gemm_q8x32_q4k_v12x8(a32.as<int8_t>(), s32.as<float>(), d32.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), wd8.as<uint16_t>(),
                                              C32.as<float>(), M, sh.N, sh.K, B, v, rg, s));
          };
          const double t = time_ms(s, run, 80.0);
          if (t < best) { best = t; rgb = rg; }
        }
        const char* nm = fam == 0 ? pkc::v12_name(v) : pkc::v12x8_name(v);
        DevBuf& Co = fam == 0 ? C256 : C32;
        if (fam == 1) PP_CK(cudaMemsetAsync(C32.as(), 0xFF, static_cast<size_t>(rowsA) * sh.N * 4, s));
        if (fam == 0)
          PP_CK(pkc::gemm_q8k_q4k_v12(x.a8.as<int8_t>(), x.sa.as<float>(), x.dig.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), w.dig.as<uint16_t>(), C256.as(),
                                      M, sh.N, sh.K, B, 1, 0, nullptr, v, rgb, s));
        else
          PP_CK(pkc::gemm_q8x32_q4k_v12x8(a32.as<int8_t>(), s32.as<float>(), d32.as<uint16_t>(), w.q.as<pkc::BlockQ4K>(), w.rec.as(), wd8.as<uint16_t>(),
                                          C32.as<float>(), M, sh.N, sh.K, B, v, rgb, s));
        const double eref = err_rows(Co.as(), false, sh.N, fam == 0 ? R256 : R32, ref_rows, rc, s);
        const Cos cb = cosine(Co.as<float>(), Cbf.as<float>(), static_cast<size_t>(rowsA) * sh.N, s);
        const bool ok = eref <= 1e-3 && std::isfinite(cb.rel_l2);
        all_ok = all_ok && ok;
        JW j;
        j.kv("kernel", nm).kv("raster", rgb).kv("ms", best).kv("tops", tops(rowsA, sh.N, sh.K, best)).kv("eps8", tops(rowsA, sh.N, sh.K, best) / peak8);
        j.kv("max_rel_err_vs_fp32_own_inputs", eref).kv("rel_l2_vs_bf16_path", cb.rel_l2).kv("correct", ok);
        kj.push_back(j);
        std::printf("    %-34s M=%4d x%3d  %-24s %.4f ms %6.1f TOPS (%.3f of peak); vs fp32 own inputs %.2e; vs bf16 path rel L2 %.3e%s\n", sh.name, M, B, nm, best,
                    tops(rowsA, sh.N, sh.K, best), tops(rowsA, sh.N, sh.K, best) / peak8, eref, cb.rel_l2, ok ? "" : "  FAIL");
        std::fflush(stdout);
        if (ok && fam == 0 && best < t256) { t256 = best; n256 = nm; }
        if (ok && fam == 1 && best < t32) { t32 = best; n32 = nm; }
      }
    row.kv("status", "ok").arr_obj("kernels", kj).kv("per256_best", n256).kv("per256_best_eps8", tops(rowsA, sh.N, sh.K, t256) / peak8);
    row.kv("per32_best", n32).kv("per32_best_eps8", tops(rowsA, sh.N, sh.K, t32) / peak8).kv("per32_over_per256_speed", t256 / t32);
    rows.push_back(row);
    std::printf("  %-34s M=%4d x%3d: per-256 best %.3f of peak, per-32 best %.3f of peak: per-32 speed %.3fx of per-256\n", sh.name, M, B,
                tops(rowsA, sh.N, sh.K, t256) / peak8, tops(rowsA, sh.N, sh.K, t32) / peak8, t256 / t32);
    std::fflush(stdout);
  }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("int8_peak_tops", peak8).kv("experts", E).kv("expert_m", Me).kv("weights_activations", "synthetic (fill_q4k / fill_bf16)");
  r.values.arr_obj("kernel_info", infos).arr_obj("cases", rows).kv("all_correct", all_ok);
  r.values.kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.pass = all_ok ? -1 : 0;
  r.headline = std::string(all_ok ? "per-256 and per-32 kernels correct" : "a kernel FAILED its check") + "; per-32 speed ratios in the cases";
  r.notes = std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "act32");
  std::printf("  ACT32 %s\n", r.headline.c_str());
  return all_ok ? 0 : 3;
}

// ================================================================ D: one MoE layer slice on real routing (2026-10-30 21:0x)
// The author's "merge now" (20:55): A (norm fused with the rounding), B (per-32 activation scales) and C (O14: a work
// list for unequal expert counts) in one run, each piece timed on its own so the cost of each stays visible.
// Routing from a PELROUT1 file (pelias kld --route-dump): one chunk's scored tokens, the 8 picks and weights per token, for
// a few layers. Per layer: RMSNorm + rounding -> gather by expert -> gate|up (2048 -> 2 x 512 per expert) -> silu x up +
// rounding -> down (512 -> 2048) -> weighted combine, for activation scales per 256 (v12 math) and per 32 (v12x8 math).
// GEMMs three ways: the work list (only the experts that have rows, each padded to 64), the grouped launch padded to the
// largest count (every expert, as v12 runs today), and an equal-count ideal with the same total rows. Weights and
// activations synthetic (the layer's real routing only). Checks: the work-list GEMM equals the padded launch bit for bit on
// every real row; both formats within 1e-3 of fp32 of their own inputs on 16 rows of the first active expert.
namespace dslice {
struct Routes {
  unsigned n_ctx = 0, chunks = 0, scored = 0, n_layer = 0, n_used = 0;
  std::vector<int16_t> sel;
  std::vector<float> w;
};
inline Routes read_routes(const std::string& path) {
  Routes r;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (b.size() < 36 || std::memcmp(b.data(), "PELROUT1", 8) != 0) throw std::runtime_error(path + " is not a PELROUT1 routing record");
  unsigned h[5];
  std::memcpy(h, b.data() + 8, sizeof h);
  r.n_ctx = h[0];
  r.chunks = h[1];
  r.scored = h[2];
  r.n_layer = h[3];
  r.n_used = h[4];
  if (r.n_used != 8) throw std::runtime_error("routing record: n_used must be 8");
  const size_t recs = static_cast<size_t>(r.chunks) * r.scored * r.n_layer;
  if (b.size() < 36 + recs * 72) throw std::runtime_error("routing record shorter than its header says");
  r.sel.resize(recs * 8);
  r.w.resize(recs * 8);
  for (size_t i = 0; i < recs; ++i) {
    std::memcpy(&r.sel[i * 8], b.data() + 36 + i * 72, 16);
    std::memcpy(&r.w[i * 8], b.data() + 36 + i * 72 + 16, 32);
  }
  return r;
}
}  // namespace dslice

inline int p_moe(Ctx& c, const Args& a) {
  using namespace k2b;
  Result r;
  r.probe = "MOE-SLICE";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  const std::string rpath = a.get("routes", "");
  if (rpath.empty()) throw std::runtime_error("moe: --routes FILE (a pelias kld --route-dump record) is required");
  const dslice::Routes R = dslice::read_routes(rpath);
  const int chunk = static_cast<int>(a.inum("chunk", 0));
  std::vector<int> layers;
  {
    std::stringstream ss(a.get("layers", "0,20,39"));
    std::string tok;
    while (std::getline(ss, tok, ',')) layers.push_back(std::stoi(tok));
  }
  std::vector<int> nchs;
  {
    std::stringstream ss(a.get("nchunks", "1,8"));
    std::string tok;
    while (std::getline(ss, tok, ',')) nchs.push_back(std::stoi(tok));
  }
  int nmax = 1;
  for (const int n : nchs) nmax = std::max(nmax, n);
  if (chunk < 0 || static_cast<unsigned>(chunk + nmax) > R.chunks) throw std::runtime_error("moe: --chunk / --nchunks out of range");
  const int Tmax = static_cast<int>(R.scored) * nmax, E = 256, H = 2048, F = 512, NU = 8;
  constexpr int kBMt = 64;
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  auto tops = [](double macs, double ms) { return 2.0 * macs / (ms * 1e-3) / 1e12; };
  // weights: experts' gate|up (E x 2F rows of H) and down (E x H rows of F), with v12's records / digits and v12x8's digits
  JW why;
  const double wneed = w8_bytes(E * 2 * F, H) + w8_bytes(E * H, F) + static_cast<double>(E) * 2 * F * (H / 256) * 32 + static_cast<double>(E) * H * (F / 256) * 32;
  if (!fits(a, wneed + 3.0 * (1ull << 30), why)) throw std::runtime_error("moe: would cross the VRAM ceiling");
  W8 wgu(E * 2 * F, H, 9100u, s), wdn(E * H, F, 9200u, s);
  DevBuf gu8(static_cast<size_t>(E) * 2 * F * (H / 256) * 32), dn8(static_cast<size_t>(E) * H * (F / 256) * 32);
  PP_CK(pkc::prep_q4k_dig8(wgu.q.as<pkc::BlockQ4K>(), gu8.as<uint16_t>(), E * 2 * F, H, s));
  PP_CK(pkc::prep_q4k_dig8(wdn.q.as<pkc::BlockQ4K>(), dn8.as<uint16_t>(), E * H, F, s));
  DevBuf X(static_cast<size_t>(Tmax) * H * 2), G(static_cast<size_t>(H) * 2), Xn(static_cast<size_t>(Tmax) * H * 2);
  PP_CK(pkc::fill_bf16(X.as<uint16_t>(), static_cast<size_t>(Tmax) * H, 9300u, s));
  PP_CK(pkc::fill_bf16(G.as<uint16_t>(), H, 9400u, s));
  bool all_ok = true;
  std::vector<JW> rows;
  for (const int nch : nchs)
  for (const int L : layers) {
    if (g_abort.load()) break;
    if (L < 0 || static_cast<unsigned>(L) >= R.n_layer) continue;
    const int T = static_cast<int>(R.scored) * nch;  // nch consecutive chunks = one prompt of T tokens
    // routing of this chunk and layer
    std::vector<int> cnt(E, 0), sel(static_cast<size_t>(T) * NU);
    std::vector<float> wt(static_cast<size_t>(T) * NU);
    for (int t = 0; t < T; ++t) {
      const size_t rec = (static_cast<size_t>(chunk) * R.scored + t) * R.n_layer + L;
      for (int k = 0; k < NU; ++k) {
        const int e = R.sel[rec * 8 + k];
        if (e < 0 || e >= E) throw std::runtime_error("moe: expert index out of range in the routing record");
        sel[static_cast<size_t>(t) * NU + k] = e;
        wt[static_cast<size_t>(t) * NU + k] = R.w[rec * 8 + k];
        ++cnt[e];
      }
    }
    int active = 0, maxc = 0;
    std::vector<int> off(E, 0);
    int P = 0;
    for (int e = 0; e < E; ++e) {
      off[e] = P;
      if (cnt[e]) ++active;
      maxc = std::max(maxc, cnt[e]);
      P += (cnt[e] + kBMt - 1) / kBMt * kBMt;
    }
    const int Me = (maxc + kBMt - 1) / kBMt * kBMt, Pp = Me * E;
    const int Mi = ((T * NU + E - 1) / E + kBMt - 1) / kBMt * kBMt;  // equal-count ideal, same total rows
    std::vector<int> src(P, -1), srcp(Pp, -1), prow(static_cast<size_t>(T) * NU), fill(E, 0);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < NU; ++k) {
        const int e = sel[static_cast<size_t>(t) * NU + k], i = fill[e]++;
        src[off[e] + i] = t;
        srcp[e * Me + i] = t;
        prow[static_cast<size_t>(t) * NU + k] = off[e] + i;
      }
    auto units_for = [&](int N) {
      std::vector<int> u;
      for (int e = 0; e < E; ++e)
        for (int nt = 0; nt < N / 128; ++nt)
          for (int mt = 0; mt < (cnt[e] + kBMt - 1) / kBMt; ++mt) {
            u.push_back(e);
            u.push_back(off[e] + mt * kBMt);
            u.push_back(nt * 128);
            u.push_back(0);
          }
      return u;
    };
    const std::vector<int> u1 = units_for(2 * F), u2 = units_for(H);
    DevBuf dsrc(static_cast<size_t>(P) * 4), dsrcp(static_cast<size_t>(Pp) * 4), dprow(static_cast<size_t>(T) * NU * 4), dwt(static_cast<size_t>(T) * NU * 4),
        du1(u1.size() * 4), du2(u2.size() * 4);
    PP_CK(cudaMemcpyAsync(dsrc.as(), src.data(), src.size() * 4, cudaMemcpyHostToDevice, s));
    PP_CK(cudaMemcpyAsync(dsrcp.as(), srcp.data(), srcp.size() * 4, cudaMemcpyHostToDevice, s));
    PP_CK(cudaMemcpyAsync(dprow.as(), prow.data(), prow.size() * 4, cudaMemcpyHostToDevice, s));
    PP_CK(cudaMemcpyAsync(dwt.as(), wt.data(), wt.size() * 4, cudaMemcpyHostToDevice, s));
    PP_CK(cudaMemcpyAsync(du1.as(), u1.data(), u1.size() * 4, cudaMemcpyHostToDevice, s));
    PP_CK(cudaMemcpyAsync(du2.as(), u2.data(), u2.size() * 4, cudaMemcpyHostToDevice, s));
    const double macs1 = static_cast<double>(T) * NU * 2 * F * H, macs2 = static_cast<double>(T) * NU * H * F;  // useful work only
    JW row;
    row.kv("layer", L).kv("chunks", nch).kv("tokens", T).kv("active_experts", active).kv("max_rows_per_expert", maxc).kv("worklist_rows", P).kv("padded_rows", Pp);
    row.kv("ideal_rows_per_expert", Mi).kv("units_gate_up", static_cast<int>(u1.size() / 4)).kv("units_down", static_cast<int>(u2.size() / 4));
    std::printf("  layer %2d: %d tokens, %d of %d experts active, rows per expert max %d; work list %d rows, padded %d rows (x%.2f), ideal %d per expert\n", L, T, active,
                E, maxc, P, Pp, static_cast<double>(Pp) / P, Mi);
    std::fflush(stdout);
    const int Pmax = std::max(P, std::max(Pp, Mi * E));
    std::vector<float> y256;
    for (const int per32 : {0, 1}) {
      if (g_abort.load()) break;
      const int nsa = per32 ? H / 32 : H / 256;
      // token-side rounded activations
      DevBuf ta8(static_cast<size_t>(T) * H), tsa(static_cast<size_t>(T) * (H / 32) * 4), ts32(static_cast<size_t>(T) * (H / 32) * 4),
          tx32(static_cast<size_t>(T) * (H / 32) * 4), tdig(static_cast<size_t>(T) * (H / 256) * 32);
      auto norm = [&] {
        if (per32) {
          PP_CK(pkc::rmsnorm_q8x32(X.as<uint16_t>(), G.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), tx32.as<float>(), T, H, s));
          PP_CK(pkc::dig_x32(tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
        } else {
          PP_CK(pkc::rmsnorm_q8k(X.as<uint16_t>(), G.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), ts32.as<int>(), tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
        }
      };
      auto norm2 = [&] {  // the two-pass form (norm to bf16, then round), for A's comparison
        PP_CK(pkc::rmsnorm_bf16(X.as<uint16_t>(), G.as<uint16_t>(), Xn.as<uint16_t>(), T, H, s));
        if (per32) {
          PP_CK(pkc::quantize_rows_q8(Xn.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), ts32.as<int>(), tx32.as<float>(), T, H, s));
          PP_CK(pkc::dig_x32(tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
        } else {
          PP_CK(pkc::quantize_rows_q8k_dig(Xn.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), ts32.as<int>(), tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
        }
      };
      // packed (by expert) activations: work-list layout and padded layout share buffers sized to the larger
      DevBuf pa8(static_cast<size_t>(Pmax) * H), psa(static_cast<size_t>(Pmax) * nsa * 4), pdig(static_cast<size_t>(Pmax) * (H / 256) * 32),
          H1(static_cast<size_t>(Pmax) * 2 * F * 4), H1p(static_cast<size_t>(Pp) * 2 * F * 4), qa8(static_cast<size_t>(Pmax) * F),
          qsa(static_cast<size_t>(Pmax) * (F / 32) * 4), qs32(static_cast<size_t>(Pmax) * (F / 32) * 4), qx32(static_cast<size_t>(Pmax) * (F / 32) * 4),
          qdig(static_cast<size_t>(Pmax) * (F / 256) * 32), O(static_cast<size_t>(Pmax) * H * 4), Y(static_cast<size_t>(T) * H * 4);
      auto gather = [&](const DevBuf& sidx, int rowsP) {
        PP_CK(pkc::gather_q8(sidx.as<int>(), rowsP, H, nsa, (H / 256) * 16, ta8.as<int8_t>(), tsa.as<float>(), tdig.as<uint16_t>(), pa8.as<int8_t>(), psa.as<float>(),
                             pdig.as<uint16_t>(), s));
      };
      const pkc::BlockQ4K* Wg = wgu.q.as<pkc::BlockQ4K>();
      const pkc::BlockQ4K* Wd = wdn.q.as<pkc::BlockQ4K>();
      int v1 = 3, v2 = 3;  // 64-row tile, 3 stages (the engine's adopted expert tile), both GEMMs
      auto gemm1 = [&](float* out) {
        if (per32)
          PP_CK(pkc::gemm_list_v12x8(static_cast<int>(u1.size() / 4), du1.as(), pa8.as<int8_t>(), psa.as<float>(), pdig.as<uint16_t>(), Wg, wgu.rec.as(), gu8.as<uint16_t>(),
                                     out, P, 2 * F, H, v1, s));
        else
          PP_CK(pkc::gemm_list_v12(static_cast<int>(u1.size() / 4), du1.as(), pa8.as<int8_t>(), psa.as<float>(), pdig.as<uint16_t>(), Wg, wgu.rec.as(), wgu.dig.as<uint16_t>(),
                                   out, P, 2 * F, H, v1, s));
      };
      auto gemm1_grouped = [&](int Mrows, float* out) {  // every expert, Mrows each (padded or ideal)
        if (per32)
          PP_CK(pkc::gemm_q8x32_q4k_v12x8(pa8.as<int8_t>(), psa.as<float>(), pdig.as<uint16_t>(), Wg, wgu.rec.as(), gu8.as<uint16_t>(), out, Mrows, 2 * F, H, E, v1, 0, s));
        else
          PP_CK(pkc::gemm_q8k_q4k_v12(pa8.as<int8_t>(), psa.as<float>(), pdig.as<uint16_t>(), Wg, wgu.rec.as(), wgu.dig.as<uint16_t>(), out, Mrows, 2 * F, H, E, 1, 0,
                                      nullptr, v1, 0, s));
      };
      auto act = [&] {
        PP_CK(pkc::silu_q8(H1.as<float>(), P, F, per32, qa8.as<int8_t>(), qsa.as<float>(), qs32.as<int>(), qx32.as<float>(), qdig.as<uint16_t>(), s));
        if (per32) PP_CK(pkc::dig_x32(qx32.as<float>(), qdig.as<uint16_t>(), P, F, s));
      };
      auto gemm2 = [&] {
        if (per32)
          PP_CK(pkc::gemm_list_v12x8(static_cast<int>(u2.size() / 4), du2.as(), qa8.as<int8_t>(), qsa.as<float>(), qdig.as<uint16_t>(), Wd, wdn.rec.as(), dn8.as<uint16_t>(),
                                     O.as<float>(), P, H, F, v2, s));
        else
          PP_CK(pkc::gemm_list_v12(static_cast<int>(u2.size() / 4), du2.as(), qa8.as<int8_t>(), qsa.as<float>(), qdig.as<uint16_t>(), Wd, wdn.rec.as(), wdn.dig.as<uint16_t>(),
                                   O.as<float>(), P, H, F, v2, s));
      };
      auto gemm2_grouped = [&](int Mrows) {
        if (per32)
          PP_CK(pkc::gemm_q8x32_q4k_v12x8(qa8.as<int8_t>(), qsa.as<float>(), qdig.as<uint16_t>(), Wd, wdn.rec.as(), dn8.as<uint16_t>(), O.as<float>(), Mrows, H, F, E, v2, 0, s));
        else
          PP_CK(pkc::gemm_q8k_q4k_v12(qa8.as<int8_t>(), qsa.as<float>(), qdig.as<uint16_t>(), Wd, wdn.rec.as(), wdn.dig.as<uint16_t>(), O.as(), Mrows, H, F, E, 1, 0, nullptr,
                                      v2, 0, s));
      };
      auto comb = [&] { PP_CK(pkc::combine_rows(O.as<float>(), dprow.as<int>(), dwt.as<float>(), T, NU, H, Y.as<float>(), s)); };
      // one full pass (work list), then the checks
      norm();
      gather(dsrc, P);
      gemm1(H1.as<float>());
      act();
      gemm2();
      comb();
      PP_CK(cudaStreamSynchronize(s));
      std::vector<float> h1(static_cast<size_t>(P) * 2 * F), y(static_cast<size_t>(T) * H);
      PP_CK(cudaMemcpy(h1.data(), H1.as(), h1.size() * 4, cudaMemcpyDeviceToHost));
      PP_CK(cudaMemcpy(y.data(), Y.as(), y.size() * 4, cudaMemcpyDeviceToHost));
      // fp32 check of gate|up: 16 rows of the first active expert, its own inputs
      int e0 = 0;
      while (e0 < E && cnt[e0] < 16) ++e0;
      double eref = 0.0;
      if (e0 < E) {
        DevBuf Rf(static_cast<size_t>(16) * 1024 * 4);
        const size_t bpr = H / 256;
        if (per32)
          PP_CK(pkc::ref_q8_q4k_rows(pa8.as<int8_t>() + static_cast<size_t>(off[e0]) * H, psa.as<float>() + static_cast<size_t>(off[e0]) * nsa,
                                     Wg + static_cast<size_t>(e0) * 2 * F * bpr, Rf.as<float>(), 16, 1024, H, s));
        else
          PP_CK(pkc::ref_q8k_q4k_rows(pa8.as<int8_t>() + static_cast<size_t>(off[e0]) * H, psa.as<float>() + static_cast<size_t>(off[e0]) * nsa,
                                      Wg + static_cast<size_t>(e0) * 2 * F * bpr, Rf.as<float>(), 16, 1024, H, s));
        eref = err_rows(H1.as<float>() + static_cast<size_t>(off[e0]) * 2 * F, false, 2 * F, Rf, 16, 1024, s);
      }
      // the padded grouped launch: same rows, bit for bit
      gather(dsrcp, Pp);
      gemm1_grouped(Me, H1p.as<float>());
      PP_CK(cudaStreamSynchronize(s));
      std::vector<float> h1p(static_cast<size_t>(Pp) * 2 * F);
      PP_CK(cudaMemcpy(h1p.data(), H1p.as(), h1p.size() * 4, cudaMemcpyDeviceToHost));
      bool same = true;
      for (int e = 0; e < E && same; ++e)
        for (int i = 0; i < cnt[e] && same; ++i)
          same = std::memcmp(&h1[static_cast<size_t>(off[e] + i) * 2 * F], &h1p[static_cast<size_t>(e * Me + i) * 2 * F], static_cast<size_t>(2 * F) * 4) == 0;
      bool finite = true;
      for (const float v : y) finite = finite && std::isfinite(v);
      const bool ok = same && finite && eref <= 1e-3;
      all_ok = all_ok && ok;
      // timings (warm, each stage on its own)
      gather(dsrc, P);
      const double t_norm = time_ms(s, norm, 60.0), t_norm2 = time_ms(s, norm2, 60.0);
      norm();
      const double t_gather = time_ms(s, [&] { gather(dsrc, P); }, 60.0);
      const double t_g1 = time_ms(s, [&] { gemm1(H1.as<float>()); }, 60.0);
      const double t_act = time_ms(s, act, 60.0);
      const double t_g2 = time_ms(s, gemm2, 60.0);
      const double t_comb = time_ms(s, comb, 60.0);
      gather(dsrcp, Pp);
      const double t_g1p = time_ms(s, [&] { gemm1_grouped(Me, H1p.as<float>()); }, 60.0);
      const double t_g2p = time_ms(s, [&] { gemm2_grouped(Me); }, 60.0);
      const double t_g1i = time_ms(s, [&] { gemm1_grouped(Mi, H1p.as<float>()); }, 60.0);
      const double t_g2i = time_ms(s, [&] { gemm2_grouped(Mi); }, 60.0);
      gather(dsrc, P);
      const double t_all = time_ms(s, [&] { norm(); gather(dsrc, P); gemm1(H1.as<float>()); act(); gemm2(); comb(); }, 100.0);
      const double sum_st = t_norm + t_gather + t_g1 + t_act + t_g2 + t_comb;
      const char* fmtn = per32 ? "per-32" : "per-256";
      JW j;
      j.kv("format", fmtn).kv("ms_norm_fused", t_norm).kv("ms_norm_two_pass", t_norm2).kv("ms_gather", t_gather).kv("ms_gate_up_worklist", t_g1).kv("ms_silu_round", t_act);
      j.kv("ms_down_worklist", t_g2).kv("ms_combine", t_comb).kv("ms_stages_sum", sum_st).kv("ms_whole_slice", t_all);
      j.kv("ms_gate_up_padded", t_g1p).kv("ms_down_padded", t_g2p).kv("ms_gate_up_ideal", t_g1i).kv("ms_down_ideal", t_g2i);
      j.kv("eps8_gate_up_worklist", tops(macs1, t_g1) / peak8).kv("eps8_down_worklist", tops(macs2, t_g2) / peak8);
      j.kv("worklist_bit_equal_padded", same).kv("max_rel_err_vs_fp32_first_expert", eref).kv("output_finite", finite).kv("correct", ok);
      row.kv_obj(per32 ? "per32" : "per256", j);
      std::printf("    %-7s norm fused %.4f (two-pass %.4f) | gather %.4f | gate|up list %.4f (padded %.4f, ideal %.4f; %.3f of peak) | silu+round %.4f | down list %.4f "
                  "(padded %.4f, ideal %.4f; %.3f of peak) | combine %.4f | whole slice %.4f ms; list = padded bit for bit: %s; vs fp32 %.2e%s\n",
                  fmtn, t_norm, t_norm2, t_gather, t_g1, t_g1p, t_g1i, tops(macs1, t_g1) / peak8, t_act, t_g2, t_g2p, t_g2i, tops(macs2, t_g2) / peak8, t_comb, t_all,
                  same ? "yes" : "NO", eref, ok ? "" : "  FAIL");
      std::fflush(stdout);
      if (!per32) {
        y256 = y;
      } else if (y256.size() == y.size()) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
          const double d = static_cast<double>(y[i]) - y256[i];
          num += d * d;
          den += static_cast<double>(y256[i]) * y256[i];
        }
        row.kv("y_rel_l2_per32_vs_per256", den > 0 ? std::sqrt(num / den) : 0.0);
        std::printf("    layer output, per-32 vs per-256: relative L2 %.3e (the difference the activation format makes)\n", den > 0 ? std::sqrt(num / den) : 0.0);
      }
    }
    rows.push_back(row);
  }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("routes", rpath).kv("first_chunk", chunk).kv("tokens_per_chunk", static_cast<int>(R.scored)).kv("experts", E).kv("hidden", H).kv("expert_ffn", F).kv("int8_peak_tops", peak8);
  r.inputs.kv("weights_activations", "synthetic (fill_q4k / fill_bf16); routing real");
  r.values.arr_obj("layers", rows).kv("all_correct", all_ok).kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.pass = all_ok ? -1 : 0;
  r.headline = std::string(all_ok ? "all checks passed" : "a check FAILED") + "; per-stage times per layer and format in the JSON";
  r.notes = std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "moe_slice");
  std::printf("  MOE-SLICE %s\n", r.headline.c_str());
  return all_ok ? 0 : 3;
}

// ================================================================ D v2 (2026-10-30 21:2x): tails, warp-vector, folded combine
// Per-32 format throughout (the adopted one). Per layer of real routing:
//   schedules (gate|up and down): S0 64-row tiles, every expert padded to 64 (D v1's work list); S1 64-row tiles for the
//   whole 64-row bodies + 16-row tiles for the tails (experts padded to 16); S2 = S1 with experts of 1-8 rows on the
//   warp-vector path. Checks: S1 = S0 bit for bit on every real row (same per-element arithmetic); the warp-vector rows
//   within 1e-3 of S0 (different summation order).
//   combine (down output into the token rows): C0 store + the separate combine kernel (D v1); C1 folded, red.add.f32;
//   C2 folded, fixed point int32 (scale 2^k from the measured |down output| bound: max|O| x 8 x 2^k < 2^31) + a
//   conversion pass; C3 folded, fixed point int64 (scale 2^k for 2^62 headroom) + conversion. Checks: each against C0
//   (relative L2, max abs); C2 / C3 run twice must give the same bits (determinism); C2's overflow counter.
inline int p_moe2(Ctx& c, const Args& a) {
  using namespace k2b;
  Result r;
  r.probe = "MOE-SLICE-V2";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  const std::string rpath = a.get("routes", "");
  if (rpath.empty()) throw std::runtime_error("moe2: --routes FILE is required");
  const dslice::Routes R = dslice::read_routes(rpath);
  const int chunk = static_cast<int>(a.inum("chunk", 0));
  std::vector<int> layers, nchs;
  {
    std::stringstream ss(a.get("layers", "0,20,39"));
    std::string tok;
    while (std::getline(ss, tok, ',')) layers.push_back(std::stoi(tok));
    std::stringstream s2(a.get("nchunks", "1,8"));
    while (std::getline(s2, tok, ',')) nchs.push_back(std::stoi(tok));
  }
  int nmax = 1;
  for (const int n : nchs) nmax = std::max(nmax, n);
  if (chunk < 0 || static_cast<unsigned>(chunk + nmax) > R.chunks) throw std::runtime_error("moe2: --chunk / --nchunks out of range");
  const int Tmax = static_cast<int>(R.scored) * nmax, E = 256, H = 2048, F = 512, NU = 8;
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  auto tops = [](double macs, double ms) { return 2.0 * macs / (ms * 1e-3) / 1e12; };
  JW why;
  const double wneed = w8_bytes(E * 2 * F, H) + w8_bytes(E * H, F) + static_cast<double>(E) * 2 * F * (H / 256) * 32 + static_cast<double>(E) * H * (F / 256) * 32;
  if (!fits(a, wneed + 3.0 * (1ull << 30), why)) throw std::runtime_error("moe2: would cross the VRAM ceiling");
  W8 wgu(E * 2 * F, H, 9100u, s), wdn(E * H, F, 9200u, s);
  DevBuf gu8(static_cast<size_t>(E) * 2 * F * (H / 256) * 32), dn8(static_cast<size_t>(E) * H * (F / 256) * 32);
  PP_CK(pkc::prep_q4k_dig8(wgu.q.as<pkc::BlockQ4K>(), gu8.as<uint16_t>(), E * 2 * F, H, s));
  PP_CK(pkc::prep_q4k_dig8(wdn.q.as<pkc::BlockQ4K>(), dn8.as<uint16_t>(), E * H, F, s));
  DevBuf X(static_cast<size_t>(Tmax) * H * 2), G(static_cast<size_t>(H) * 2);
  PP_CK(pkc::fill_bf16(X.as<uint16_t>(), static_cast<size_t>(Tmax) * H, 9300u, s));
  PP_CK(pkc::fill_bf16(G.as<uint16_t>(), H, 9400u, s));
  std::vector<JW> infos;
  for (const int bmt : {64, 16}) {
    int regs = 0, loc = 0, sm = 0, bps = 0;
    PP_CK(pkc::moe_tile_info(bmt, &regs, &loc, &sm, &bps));
    JW j;
    j.kv("tile_rows", bmt).kv("registers", regs).kv("local_spill_bytes", loc).kv("shared_bytes", sm).kv("blocks_per_sm", bps);
    infos.push_back(j);
    std::printf("    moe tile %2d rows: %3d registers, %4d spill bytes, %6d shared bytes, %d blocks per SM\n", bmt, regs, loc, sm, bps);
  }
  bool all_ok = true;
  std::vector<JW> rows;
  for (const int nch : nchs)
    for (const int L : layers) {
      if (g_abort.load()) break;
      if (L < 0 || static_cast<unsigned>(L) >= R.n_layer) continue;
      const int T = static_cast<int>(R.scored) * nch;
      std::vector<int> cnt(E, 0), sel(static_cast<size_t>(T) * NU);
      std::vector<float> wt(static_cast<size_t>(T) * NU);
      for (int t = 0; t < T; ++t) {
        const size_t rec = (static_cast<size_t>(chunk) * R.scored + t) * R.n_layer + L;
        for (int k = 0; k < NU; ++k) {
          const int e = R.sel[rec * 8 + k];
          if (e < 0 || e >= E) throw std::runtime_error("moe2: expert index out of range");
          sel[static_cast<size_t>(t) * NU + k] = e;
          wt[static_cast<size_t>(t) * NU + k] = R.w[rec * 8 + k];
          ++cnt[e];
        }
      }
      // two layouts: experts padded to 64 (S0) and to 16 (S1 / S2)
      auto layout = [&](int pad, std::vector<int>& off, std::vector<int>& src, std::vector<int>& prow, std::vector<int>& rtok, std::vector<float>& rwv) {
        off.assign(E, 0);
        int P = 0;
        for (int e = 0; e < E; ++e) {
          off[e] = P;
          P += (cnt[e] + pad - 1) / pad * pad;
        }
        src.assign(P, -1);
        rtok.assign(P, -1);
        rwv.assign(P, 0.f);
        prow.assign(static_cast<size_t>(T) * NU, 0);
        std::vector<int> fill(E, 0);
        for (int t = 0; t < T; ++t)
          for (int k = 0; k < NU; ++k) {
            const int e = sel[static_cast<size_t>(t) * NU + k], pr = off[e] + fill[e]++;
            src[pr] = t;
            rtok[pr] = t;
            rwv[pr] = wt[static_cast<size_t>(t) * NU + k];
            prow[static_cast<size_t>(t) * NU + k] = pr;
          }
        return P;
      };
      std::vector<int> off64, src64, prow64, rtok64, off16, src16, prow16, rtok16;
      std::vector<float> rw64, rw16;
      const int P64 = layout(64, off64, src64, prow64, rtok64, rw64), P16 = layout(16, off16, src16, prow16, rtok16, rw16);
      // unit tables: n-tile major inside an expert (its weight tile stays in L2 across the row tiles)
      auto units = [&](int N, int mode, std::vector<int>& u64, std::vector<int>& u16, std::vector<int>& uv) {
        u64.clear();
        u16.clear();
        uv.clear();
        for (int e = 0; e < E; ++e) {
          if (!cnt[e]) continue;
          if (mode == 0) {
            for (int nt = 0; nt < N / 128; ++nt)
              for (int mt = 0; mt < (cnt[e] + 63) / 64; ++mt) u64.insert(u64.end(), {e, off64[e] + mt * 64, nt * 128, 0});
            continue;
          }
          if (mode == 2 && cnt[e] <= 8) {
            uv.insert(uv.end(), {e, off16[e], cnt[e], 0});
            continue;
          }
          const int body = cnt[e] / 64, tail = (cnt[e] - body * 64 + 15) / 16;
          for (int nt = 0; nt < N / 128; ++nt) {
            for (int mt = 0; mt < body; ++mt) u64.insert(u64.end(), {e, off16[e] + mt * 64, nt * 128, 0});
            for (int tt = 0; tt < tail; ++tt) u16.insert(u16.end(), {e, off16[e] + body * 64 + tt * 16, nt * 128, 0});
          }
        }
      };
      struct UT {
        std::vector<int> u64, u16, uv;
        DevBuf d64, d16, dv;
      };
      UT ut[2][3];  // [gemm 0 gate|up / 1 down][schedule]
      for (int gm = 0; gm < 2; ++gm)
        for (int sc = 0; sc < 3; ++sc) {
          UT& x = ut[gm][sc];
          units(gm == 0 ? 2 * F : H, sc, x.u64, x.u16, x.uv);
          x.d64 = DevBuf(std::max<size_t>(16, x.u64.size() * 4));
          x.d16 = DevBuf(std::max<size_t>(16, x.u16.size() * 4));
          x.dv = DevBuf(std::max<size_t>(16, x.uv.size() * 4));
          if (!x.u64.empty()) PP_CK(cudaMemcpyAsync(x.d64.as(), x.u64.data(), x.u64.size() * 4, cudaMemcpyHostToDevice, s));
          if (!x.u16.empty()) PP_CK(cudaMemcpyAsync(x.d16.as(), x.u16.data(), x.u16.size() * 4, cudaMemcpyHostToDevice, s));
          if (!x.uv.empty()) PP_CK(cudaMemcpyAsync(x.dv.as(), x.uv.data(), x.uv.size() * 4, cudaMemcpyHostToDevice, s));
        }
      auto up = [&](const auto& v) {
        DevBuf d(std::max<size_t>(16, v.size() * 4));
        PP_CK(cudaMemcpyAsync(d.as(), v.data(), v.size() * 4, cudaMemcpyHostToDevice, s));
        return d;
      };
      DevBuf dsrc64 = up(src64), dsrc16 = up(src16), dprow64 = up(prow64), dprow16 = up(prow16), drt64 = up(rtok64), drt16 = up(rtok16), drw64 = up(rw64),
             drw16 = up(rw16), dwt = up(wt);
      int vec_experts = 0, vec_rows = 0;
      for (int e = 0; e < E; ++e)
        if (cnt[e] && cnt[e] <= 8) {
          ++vec_experts;
          vec_rows += cnt[e];
        }
      JW row;
      row.kv("layer", L).kv("chunks", nch).kv("tokens", T).kv("rows_pad64", P64).kv("rows_pad16", P16).kv("experts_1_to_8_rows", vec_experts).kv("rows_in_1_to_8", vec_rows);
      for (int sc = 0; sc < 3; ++sc) {
        JW j;
        j.kv("gate_up_units_64", static_cast<int>(ut[0][sc].u64.size() / 4)).kv("gate_up_units_16", static_cast<int>(ut[0][sc].u16.size() / 4));
        j.kv("gate_up_units_vec", static_cast<int>(ut[0][sc].uv.size() / 4));
        row.kv_obj(std::string("units_S") + std::to_string(sc), j);
      }
      std::printf("  layer %2d, %4d tokens: rows padded to 64 %d, to 16 %d (real %d); experts with 1-8 rows %d (%d rows)\n", L, T, P64, P16, T * NU, vec_experts, vec_rows);
      std::fflush(stdout);
      // token side: fused norm + per-32 rounding, digits
      DevBuf ta8(static_cast<size_t>(T) * H), tsa(static_cast<size_t>(T) * (H / 32) * 4), tx32(static_cast<size_t>(T) * (H / 32) * 4),
          tdig(static_cast<size_t>(T) * (H / 256) * 32);
      PP_CK(pkc::rmsnorm_q8x32(X.as<uint16_t>(), G.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), tx32.as<float>(), T, H, s));
      PP_CK(pkc::dig_x32(tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
      const int Pm = std::max(P64, P16);
      DevBuf pa8(static_cast<size_t>(Pm) * H), psa(static_cast<size_t>(Pm) * (H / 32) * 4), pdig(static_cast<size_t>(Pm) * (H / 256) * 32),
          H1(static_cast<size_t>(Pm) * 2 * F * 4), qa8(static_cast<size_t>(Pm) * F), qsa(static_cast<size_t>(Pm) * (F / 32) * 4),
          qs32(static_cast<size_t>(Pm) * (F / 32) * 4), qx32(static_cast<size_t>(Pm) * (F / 32) * 4), qdig(static_cast<size_t>(Pm) * (F / 256) * 32),
          O(static_cast<size_t>(Pm) * H * 4), Y0(static_cast<size_t>(T) * H * 4), Yf(static_cast<size_t>(T) * H * 4), Yi(static_cast<size_t>(T) * H * 4),
          Yl(static_cast<size_t>(T) * H * 8), Yc(static_cast<size_t>(T) * H * 4), ovf(16);
      auto gather = [&](const DevBuf& sidx, int P) {
        PP_CK(pkc::gather_q8(sidx.as<int>(), P, H, H / 32, (H / 256) * 16, ta8.as<int8_t>(), tsa.as<float>(), tdig.as<uint16_t>(), pa8.as<int8_t>(), psa.as<float>(),
                             pdig.as<uint16_t>(), s));
      };
      auto args1 = [&] {
        pkc::MoeArgs A;
        A.a8 = pa8.as<int8_t>();
        A.sa32 = psa.as<float>();
        A.adig = pdig.as<uint16_t>();
        A.w = wgu.q.as<pkc::BlockQ4K>();
        A.rec = wgu.rec.as();
        A.wdig8 = gu8.as<uint16_t>();
        A.N = 2 * F;
        A.K = H;
        A.C = H1.as<float>();
        return A;
      };
      auto args2 = [&](int sc, int epi, float scale) {
        pkc::MoeArgs A;
        A.a8 = qa8.as<int8_t>();
        A.sa32 = qsa.as<float>();
        A.adig = qdig.as<uint16_t>();
        A.w = wdn.q.as<pkc::BlockQ4K>();
        A.rec = wdn.rec.as();
        A.wdig8 = dn8.as<uint16_t>();
        A.N = H;
        A.K = F;
        A.C = O.as<float>();
        A.rtok = (sc == 0 ? drt64 : drt16).as<int>();
        A.rw = (sc == 0 ? drw64 : drw16).as<float>();
        A.Yf = Yf.as<float>();
        A.Yi = Yi.as<int>();
        A.Yl = Yl.as<unsigned long long>();
        A.scale = scale;
        A.ovf = ovf.as<unsigned>();
        (void)epi;
        return A;
      };
      auto gemm = [&](int gm, int sc, int epi, const pkc::MoeArgs& A) {
        UT& x = ut[gm][sc];
        PP_CK(pkc::moe_tiles(64, epi, static_cast<int>(x.u64.size() / 4), x.d64.as(), A, s));
        if (!x.u16.empty()) PP_CK(pkc::moe_tiles(16, epi, static_cast<int>(x.u16.size() / 4), x.d16.as(), A, s));
        if (!x.uv.empty()) PP_CK(pkc::moe_vec(epi, static_cast<int>(x.uv.size() / 4), x.dv.as(), A, s));
      };
      auto act = [&](int P) {
        PP_CK(pkc::silu_q8(H1.as<float>(), P, F, 1, qa8.as<int8_t>(), qsa.as<float>(), qs32.as<int>(), qx32.as<float>(), qdig.as<uint16_t>(), s));
        PP_CK(pkc::dig_x32(qx32.as<float>(), qdig.as<uint16_t>(), P, F, s));
      };
      // ---- schedules: full pass per schedule (separate combine), outputs kept for the checks
      std::vector<std::vector<float>> h1s(3), os(3), ys(3);
      double t_g1[3], t_g2[3], t_sl[3];
      for (int sc = 0; sc < 3; ++sc) {
        const int P = sc == 0 ? P64 : P16;
        const DevBuf& sidx = sc == 0 ? dsrc64 : dsrc16;
        const DevBuf& pr = sc == 0 ? dprow64 : dprow16;
        PP_CK(cudaMemsetAsync(H1.as(), 0, static_cast<size_t>(Pm) * 2 * F * 4, s));
        PP_CK(cudaMemsetAsync(O.as(), 0, static_cast<size_t>(Pm) * H * 4, s));
        gather(sidx, P);
        gemm(0, sc, 0, args1());
        act(P);
        gemm(1, sc, 0, args2(sc, 0, 1.f));
        PP_CK(pkc::combine_rows(O.as<float>(), pr.as<int>(), dwt.as<float>(), T, NU, H, Y0.as<float>(), s));
        PP_CK(cudaStreamSynchronize(s));
        h1s[sc].resize(static_cast<size_t>(P) * 2 * F);
        os[sc].resize(static_cast<size_t>(P) * H);
        ys[sc].resize(static_cast<size_t>(T) * H);
        PP_CK(cudaMemcpy(h1s[sc].data(), H1.as(), h1s[sc].size() * 4, cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(os[sc].data(), O.as(), os[sc].size() * 4, cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(ys[sc].data(), Y0.as(), ys[sc].size() * 4, cudaMemcpyDeviceToHost));
        t_g1[sc] = time_ms(s, [&] { gemm(0, sc, 0, args1()); }, 60.0);
        act(P);
        t_g2[sc] = time_ms(s, [&] { gemm(1, sc, 0, args2(sc, 0, 1.f)); }, 60.0);
        t_sl[sc] = time_ms(s, [&] {
          PP_CK(pkc::rmsnorm_q8x32(X.as<uint16_t>(), G.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), tx32.as<float>(), T, H, s));
          PP_CK(pkc::dig_x32(tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
          gather(sidx, P);
          gemm(0, sc, 0, args1());
          act(P);
          gemm(1, sc, 0, args2(sc, 0, 1.f));
          PP_CK(pkc::combine_rows(O.as<float>(), pr.as<int>(), dwt.as<float>(), T, NU, H, Y0.as<float>(), s));
        }, 100.0);
      }
      // checks: S1 = S0 bit for bit on every real row (gate|up and down); S2's vector rows within 1e-3 of S0 (max rel per row)
      bool s1same = true;
      double vec_err = 0.0;
      for (int e = 0; e < E; ++e)
        for (int i = 0; i < cnt[e]; ++i) {
          const float* a0 = &h1s[0][static_cast<size_t>(off64[e] + i) * 2 * F];
          const float* a1 = &h1s[1][static_cast<size_t>(off16[e] + i) * 2 * F];
          const float* a2 = &h1s[2][static_cast<size_t>(off16[e] + i) * 2 * F];
          const float* o0 = &os[0][static_cast<size_t>(off64[e] + i) * H];
          const float* o1 = &os[1][static_cast<size_t>(off16[e] + i) * H];
          s1same = s1same && std::memcmp(a0, a1, static_cast<size_t>(2 * F) * 4) == 0 && std::memcmp(o0, o1, static_cast<size_t>(H) * 4) == 0;
          if (cnt[e] <= 8) {
            double mx = 0.0, md = 0.0;
            for (int cc = 0; cc < 2 * F; ++cc) {
              mx = std::max(mx, static_cast<double>(std::fabs(a0[cc])));
              md = std::max(md, static_cast<double>(std::fabs(a2[cc] - a0[cc])));
            }
            if (mx > 0) vec_err = std::max(vec_err, md / mx);
          }
        }
      const bool vec_ok = vec_err <= 1e-3;
      // ---- combine variants on the S2 schedule (the down GEMM with each epilogue), against C0 = S2's separate combine
      const int sb = 2;
      const DevBuf& pr16 = dprow16;
      act(P16);
      gather(dsrc16, P16);
      gemm(0, sb, 0, args1());
      act(P16);
      gemm(1, sb, 0, args2(sb, 0, 1.f));
      PP_CK(pkc::combine_rows(O.as<float>(), pr16.as<int>(), dwt.as<float>(), T, NU, H, Y0.as<float>(), s));
      // the fixed-point scales from the measured bound of one contribution (|O| x max route weight)
      double maxO = 0.0;
      PP_CK(cudaStreamSynchronize(s));
      {
        std::vector<float> hO(static_cast<size_t>(P16) * H);
        PP_CK(cudaMemcpy(hO.data(), O.as(), hO.size() * 4, cudaMemcpyDeviceToHost));
        for (int p = 0; p < P16; ++p)
          if (rtok16[p] >= 0)
            for (int cc = 0; cc < H; ++cc) maxO = std::max(maxO, static_cast<double>(std::fabs(hO[static_cast<size_t>(p) * H + cc]) * rw16[p]));
      }
      const int k32 = maxO > 0 ? static_cast<int>(std::floor(std::log2(268435455.0 / maxO))) : 20;
      const int k64 = maxO > 0 ? std::min(60, static_cast<int>(std::floor(std::log2(5.7e17 / maxO)))) : 40;
      const float sc32 = std::ldexp(1.f, k32), sc64 = std::ldexp(1.f, std::min(k64, 100));
      std::vector<float> y0(static_cast<size_t>(T) * H);
      PP_CK(cudaMemcpy(y0.data(), Y0.as(), y0.size() * 4, cudaMemcpyDeviceToHost));
      auto fold = [&](int epi) {
        if (epi == 1) PP_CK(cudaMemsetAsync(Yf.as(), 0, static_cast<size_t>(T) * H * 4, s));
        if (epi == 2) PP_CK(cudaMemsetAsync(Yi.as(), 0, static_cast<size_t>(T) * H * 4, s));
        if (epi == 3) PP_CK(cudaMemsetAsync(Yl.as(), 0, static_cast<size_t>(T) * H * 8, s));
        gemm(1, sb, epi, args2(sb, epi, epi == 2 ? sc32 : sc64));
        if (epi == 2) PP_CK(pkc::fix_to_f32(Yi.as<int>(), nullptr, static_cast<size_t>(T) * H, 1.f / sc32, Yc.as<float>(), s));
        if (epi == 3) PP_CK(pkc::fix_to_f32(nullptr, Yl.as<unsigned long long>(), static_cast<size_t>(T) * H, 1.f / sc64, Yc.as<float>(), s));
      };
      const double t_c0 = time_ms(s, [&] {
        gemm(1, sb, 0, args2(sb, 0, 1.f));
        PP_CK(pkc::combine_rows(O.as<float>(), pr16.as<int>(), dwt.as<float>(), T, NU, H, Y0.as<float>(), s));
      }, 60.0);
      std::vector<JW> cj;
      bool comb_ok = true;
      double t_c[4] = {t_c0, 0, 0, 0};
      for (int epi = 1; epi <= 3; ++epi) {
        PP_CK(cudaMemsetAsync(ovf.as(), 0, 16, s));
        fold(epi);
        PP_CK(cudaStreamSynchronize(s));
        std::vector<float> ya(static_cast<size_t>(T) * H), yb(static_cast<size_t>(T) * H);
        PP_CK(cudaMemcpy(ya.data(), (epi == 1 ? Yf : Yc).as(), ya.size() * 4, cudaMemcpyDeviceToHost));
        unsigned nov = 0;
        PP_CK(cudaMemcpy(&nov, ovf.as(), 4, cudaMemcpyDeviceToHost));
        fold(epi);
        PP_CK(cudaStreamSynchronize(s));
        PP_CK(cudaMemcpy(yb.data(), (epi == 1 ? Yf : Yc).as(), yb.size() * 4, cudaMemcpyDeviceToHost));
        const bool repeat_same = std::memcmp(ya.data(), yb.data(), ya.size() * 4) == 0;
        double num = 0.0, den = 0.0, mabs = 0.0, ymax = 0.0;
        for (size_t i = 0; i < ya.size(); ++i) {
          const double d = static_cast<double>(ya[i]) - y0[i];
          num += d * d;
          den += static_cast<double>(y0[i]) * y0[i];
          mabs = std::max(mabs, std::fabs(d));
          ymax = std::max(ymax, static_cast<double>(std::fabs(y0[i])));
        }
        const double rl2 = den > 0 ? std::sqrt(num / den) : 0.0;
        t_c[epi] = time_ms(s, [&] { fold(epi); }, 60.0);
        const bool ok = rl2 <= 1e-4 && (epi == 1 || (repeat_same && nov == 0));
        comb_ok = comb_ok && ok;
        const char* nm = epi == 1 ? "fold f32 red" : (epi == 2 ? "fold int32 fixed" : "fold int64 fixed");
        JW j;
        j.kv("combine", nm).kv("ms_down_plus_combine", t_c[epi]).kv("rel_l2_vs_separate_combine", rl2).kv("max_abs_diff", mabs).kv("max_abs_output", ymax);
        j.kv("repeat_bit_equal", repeat_same).kv("overflow_count", static_cast<long long>(nov)).kv("scale_log2", epi == 2 ? k32 : (epi == 3 ? k64 : 0)).kv("ok", ok);
        cj.push_back(j);
        std::printf("    %-17s down+combine %.4f ms (separate %.4f: %.2fx) | vs separate: rel L2 %.2e, max |diff| %.2e (max |y| %.2e) | repeat bit-equal %s%s%s\n", nm,
                    t_c[epi], t_c0, t_c0 / t_c[epi], rl2, mabs, ymax, repeat_same ? "yes" : "no",
                    epi == 2 ? (std::string(" | scale 2^") + std::to_string(k32) + ", overflows " + std::to_string(nov)).c_str() : "", ok ? "" : "  FAIL");
        std::fflush(stdout);
      }
      const bool ok = s1same && vec_ok && comb_ok;
      all_ok = all_ok && ok;
      const double macs1 = static_cast<double>(T) * NU * 2 * F * H, macs2 = static_cast<double>(T) * NU * H * F;
      const char* sn[3] = {"S0 pad-64", "S1 64 + 16 tails", "S2 S1 + vector 1-8"};
      std::vector<JW> sj;
      for (int sc = 0; sc < 3; ++sc) {
        JW j;
        j.kv("schedule", sn[sc]).kv("ms_gate_up", t_g1[sc]).kv("ms_down", t_g2[sc]).kv("ms_whole_slice", t_sl[sc]).kv("eps8_gate_up", tops(macs1, t_g1[sc]) / peak8);
        j.kv("eps8_down", tops(macs2, t_g2[sc]) / peak8);
        sj.push_back(j);
        std::printf("    %-20s gate|up %.4f ms (%.3f of peak) | down %.4f ms | whole slice %.4f ms (%.2fx of S0)\n", sn[sc], t_g1[sc], tops(macs1, t_g1[sc]) / peak8, t_g2[sc],
                    t_sl[sc], t_sl[0] / t_sl[sc]);
      }
      std::printf("    checks: S1 = S0 bit for bit %s; vector rows vs S0 max rel %.2e%s\n", s1same ? "yes" : "NO", vec_err, ok ? "" : "  FAIL");
      std::fflush(stdout);
      row.arr_obj("schedules", sj).arr_obj("combine", cj).kv("s1_bit_equal_s0", s1same).kv("vector_max_rel_vs_s0", vec_err).kv("ok", ok);
      rows.push_back(row);
    }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("routes", rpath).kv("first_chunk", chunk).kv("tokens_per_chunk", static_cast<int>(R.scored)).kv("int8_peak_tops", peak8);
  r.inputs.kv("format", "per-32 (gpu256x8)").kv("weights_activations", "synthetic; routing real");
  r.values.arr_obj("kernel_info", infos).arr_obj("layers", rows).kv("all_ok", all_ok).kv_obj("state_before", before).kv_obj("state_after", state_now(c));
  r.values.kv_obj("nvml_whole_run", summarize(all));
  r.pass = all_ok ? -1 : 0;
  r.headline = std::string(all_ok ? "all checks passed" : "a check FAILED") + "; schedules and combine variants per layer in the JSON";
  r.notes = std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "moe_slice_v2");
  std::printf("  MOE-SLICE-V2 %s\n", r.headline.c_str());
  return all_ok ? 0 : 3;
}

// ================================================================ S3 (2026-10-30 21:3x, author 21:30): no padding + the GPU work list
// Per layer of real routing, per-32 format: S0 (D v1: every expert padded to 64, host-built list, separate combine) as
// the reference and baseline; S3 at cut 8 / 16 / 24 / 32: the work list built on the GPU (match_any / ballot / popc),
// experts' rows unpadded, 64-row tiles for experts above the cut, warp-vector below, the down projection folded into the
// combine with fixed-point int32 reductions (deterministic) + conversion. Checks: the GPU list's counts equal the host's;
// tile-path rows = S0 bit for bit (gate|up and down); vector rows within 1e-3 of S0; the int32 fold within 1e-4 of S0's
// combine, the same bits on repeat, no overflow. Timed: the GPU list build vs the host build + copy + sync; the whole
// slice, S3 at each cut against S0.
inline int p_moe3(Ctx& c, const Args& a) {
  using namespace k2b;
  Result r;
  r.probe = "MOE-SLICE-S3";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  const std::string rpath = a.get("routes", "");
  if (rpath.empty()) throw std::runtime_error("moe3: --routes FILE is required");
  const dslice::Routes R = dslice::read_routes(rpath);
  const int chunk = static_cast<int>(a.inum("chunk", 0));
  std::vector<int> layers, nchs, cuts;
  {
    std::string tok;
    std::stringstream s1(a.get("layers", "0,20,39"));
    while (std::getline(s1, tok, ',')) layers.push_back(std::stoi(tok));
    std::stringstream s2(a.get("nchunks", "1,8"));
    while (std::getline(s2, tok, ',')) nchs.push_back(std::stoi(tok));
    std::stringstream s3(a.get("cuts", "8,16,24,32"));
    while (std::getline(s3, tok, ',')) cuts.push_back(std::stoi(tok));
  }
  std::vector<int> orders;  // 0: unit in blockIdx.x (S3); 1: column tile in blockIdx.x (S3b)
  {
    std::string tok;
    std::stringstream s4(a.get("orders", "0"));
    while (std::getline(s4, tok, ',')) orders.push_back(std::stoi(tok));
  }
  int nmax = 1;
  for (const int n : nchs) nmax = std::max(nmax, n);
  if (chunk < 0 || static_cast<unsigned>(chunk + nmax) > R.chunks) throw std::runtime_error("moe3: --chunk / --nchunks out of range");
  const int Tmax = static_cast<int>(R.scored) * nmax, E = 256, H = 2048, F = 512, NU = 8;
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  JW why;
  const double wneed = w8_bytes(E * 2 * F, H) + w8_bytes(E * H, F) + static_cast<double>(E) * 2 * F * (H / 256) * 32 + static_cast<double>(E) * H * (F / 256) * 32;
  if (!fits(a, wneed + 3.0 * (1ull << 30), why)) throw std::runtime_error("moe3: would cross the VRAM ceiling");
  W8 wgu(E * 2 * F, H, 9100u, s), wdn(E * H, F, 9200u, s);
  DevBuf gu8(static_cast<size_t>(E) * 2 * F * (H / 256) * 32), dn8(static_cast<size_t>(E) * H * (F / 256) * 32);
  PP_CK(pkc::prep_q4k_dig8(wgu.q.as<pkc::BlockQ4K>(), gu8.as<uint16_t>(), E * 2 * F, H, s));
  PP_CK(pkc::prep_q4k_dig8(wdn.q.as<pkc::BlockQ4K>(), dn8.as<uint16_t>(), E * H, F, s));
  DevBuf X(static_cast<size_t>(Tmax) * H * 2), G(static_cast<size_t>(H) * 2);
  PP_CK(pkc::fill_bf16(X.as<uint16_t>(), static_cast<size_t>(Tmax) * H, 9300u, s));
  PP_CK(pkc::fill_bf16(G.as<uint16_t>(), H, 9400u, s));
  bool all_ok = true;
  std::vector<JW> rows;
  for (const int nch : nchs)
    for (const int L : layers) {
      if (g_abort.load()) break;
      if (L < 0 || static_cast<unsigned>(L) >= R.n_layer) continue;
      const int T = static_cast<int>(R.scored) * nch, NP = T * NU;
      std::vector<int> cnt(E, 0);
      std::vector<int16_t> sel16(static_cast<size_t>(NP));
      std::vector<float> wt(static_cast<size_t>(NP));
      for (int t = 0; t < T; ++t) {
        const size_t rec = (static_cast<size_t>(chunk) * R.scored + t) * R.n_layer + L;
        for (int k = 0; k < NU; ++k) {
          const int e = R.sel[rec * 8 + k];
          if (e < 0 || e >= E) throw std::runtime_error("moe3: expert index out of range");
          sel16[static_cast<size_t>(t) * NU + k] = static_cast<int16_t>(e);
          wt[static_cast<size_t>(t) * NU + k] = R.w[rec * 8 + k];
          ++cnt[e];
        }
      }
      // ---- S0 (reference / baseline): pad 64, host list
      std::vector<int> off64(E, 0), src64, prow64(static_cast<size_t>(NP)), u1, u2;
      int P64 = 0;
      for (int e = 0; e < E; ++e) {
        off64[e] = P64;
        P64 += (cnt[e] + 63) / 64 * 64;
      }
      src64.assign(P64, -1);
      {
        std::vector<int> fill(E, 0);
        for (int i = 0; i < NP; ++i) {
          const int e = sel16[i], pr = off64[e] + fill[e]++;
          src64[pr] = i / NU;
          prow64[i] = pr;
        }
      }
      const auto t_host0 = std::chrono::steady_clock::now();
      for (int e = 0; e < E; ++e)
        for (int mt = 0; mt < (cnt[e] + 63) / 64; ++mt) {
          for (int nt = 0; nt < 2 * F / 128; ++nt) u1.insert(u1.end(), {e, off64[e] + mt * 64, nt * 128, 0});
          for (int nt = 0; nt < H / 128; ++nt) u2.insert(u2.end(), {e, off64[e] + mt * 64, nt * 128, 0});
        }
      DevBuf du1(u1.size() * 4), du2(u2.size() * 4), dsrc64(src64.size() * 4), dprow64(prow64.size() * 4), dwt(wt.size() * 4), dsel(sel16.size() * 2);
      PP_CK(cudaMemcpyAsync(du1.as(), u1.data(), u1.size() * 4, cudaMemcpyHostToDevice, s));
      PP_CK(cudaMemcpyAsync(du2.as(), u2.data(), u2.size() * 4, cudaMemcpyHostToDevice, s));
      PP_CK(cudaMemcpyAsync(dsrc64.as(), src64.data(), src64.size() * 4, cudaMemcpyHostToDevice, s));
      PP_CK(cudaMemcpyAsync(dprow64.as(), prow64.data(), prow64.size() * 4, cudaMemcpyHostToDevice, s));
      PP_CK(cudaStreamSynchronize(s));
      const double t_host_list = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_host0).count() * 1e3;  // build + copy + sync (ms)
      PP_CK(cudaMemcpyAsync(dwt.as(), wt.data(), wt.size() * 4, cudaMemcpyHostToDevice, s));
      PP_CK(cudaMemcpyAsync(dsel.as(), sel16.data(), sel16.size() * 2, cudaMemcpyHostToDevice, s));
      const int Pm = std::max(P64, NP + 64);
      DevBuf ta8(static_cast<size_t>(T) * H), tsa(static_cast<size_t>(T) * (H / 32) * 4), tx32(static_cast<size_t>(T) * (H / 32) * 4),
          tdig(static_cast<size_t>(T) * (H / 256) * 32), pa8(static_cast<size_t>(Pm) * H), psa(static_cast<size_t>(Pm) * (H / 32) * 4),
          pdig(static_cast<size_t>(Pm) * (H / 256) * 32), H1(static_cast<size_t>(Pm) * 2 * F * 4), qa8(static_cast<size_t>(Pm) * F),
          qsa(static_cast<size_t>(Pm) * (F / 32) * 4), qs32(static_cast<size_t>(Pm) * (F / 32) * 4), qx32(static_cast<size_t>(Pm) * (F / 32) * 4),
          qdig(static_cast<size_t>(Pm) * (F / 256) * 32), O(static_cast<size_t>(Pm) * H * 4), Y0(static_cast<size_t>(T) * H * 4), Yi(static_cast<size_t>(T) * H * 4),
          Yc(static_cast<size_t>(T) * H * 4), ovf(16);
      PP_CK(cudaMemsetAsync(pa8.as(), 0, static_cast<size_t>(Pm) * H, s));
      PP_CK(cudaMemsetAsync(psa.as(), 0, static_cast<size_t>(Pm) * (H / 32) * 4, s));
      PP_CK(cudaMemsetAsync(pdig.as(), 0, static_cast<size_t>(Pm) * (H / 256) * 32, s));
      PP_CK(cudaMemsetAsync(qa8.as(), 0, static_cast<size_t>(Pm) * F, s));
      PP_CK(cudaMemsetAsync(qsa.as(), 0, static_cast<size_t>(Pm) * (F / 32) * 4, s));
      PP_CK(cudaMemsetAsync(qdig.as(), 0, static_cast<size_t>(Pm) * (F / 256) * 32, s));
      auto norm = [&] {
        PP_CK(pkc::rmsnorm_q8x32(X.as<uint16_t>(), G.as<uint16_t>(), ta8.as<int8_t>(), tsa.as<float>(), tx32.as<float>(), T, H, s));
        PP_CK(pkc::dig_x32(tx32.as<float>(), tdig.as<uint16_t>(), T, H, s));
      };
      auto gather = [&](const int* sidx, int P) {
        PP_CK(pkc::gather_q8(sidx, P, H, H / 32, (H / 256) * 16, ta8.as<int8_t>(), tsa.as<float>(), tdig.as<uint16_t>(), pa8.as<int8_t>(), psa.as<float>(),
                             pdig.as<uint16_t>(), s));
      };
      auto act = [&](int P) {
        PP_CK(pkc::silu_q8(H1.as<float>(), P, F, 1, qa8.as<int8_t>(), qsa.as<float>(), qs32.as<int>(), qx32.as<float>(), qdig.as<uint16_t>(), s));
        PP_CK(pkc::dig_x32(qx32.as<float>(), qdig.as<uint16_t>(), P, F, s));
      };
      auto A1 = [&] {
        pkc::MoeArgs A;
        A.a8 = pa8.as<int8_t>();
        A.sa32 = psa.as<float>();
        A.adig = pdig.as<uint16_t>();
        A.w = wgu.q.as<pkc::BlockQ4K>();
        A.rec = wgu.rec.as();
        A.wdig8 = gu8.as<uint16_t>();
        A.N = 2 * F;
        A.K = H;
        A.C = H1.as<float>();
        return A;
      };
      auto A2 = [&](const int* rt, const float* rwp, float scale) {
        pkc::MoeArgs A;
        A.a8 = qa8.as<int8_t>();
        A.sa32 = qsa.as<float>();
        A.adig = qdig.as<uint16_t>();
        A.w = wdn.q.as<pkc::BlockQ4K>();
        A.rec = wdn.rec.as();
        A.wdig8 = dn8.as<uint16_t>();
        A.N = H;
        A.K = F;
        A.C = O.as<float>();
        A.rtok = rt;
        A.rw = rwp;
        A.Yi = Yi.as<int>();
        A.scale = scale;
        A.ovf = ovf.as<unsigned>();
        return A;
      };
      // S0 pass (reference)
      auto s0_slice = [&] {
        norm();
        gather(dsrc64.as<int>(), P64);
        PP_CK(pkc::moe_tiles(64, 0, static_cast<int>(u1.size() / 4), du1.as(), A1(), s));
        act(P64);
        PP_CK(pkc::moe_tiles(64, 0, static_cast<int>(u2.size() / 4), du2.as(), A2(nullptr, nullptr, 1.f), s));
        PP_CK(pkc::combine_rows(O.as<float>(), dprow64.as<int>(), dwt.as<float>(), T, NU, H, Y0.as<float>(), s));
      };
      s0_slice();
      PP_CK(cudaStreamSynchronize(s));
      std::vector<float> h1s0(static_cast<size_t>(P64) * 2 * F), os0(static_cast<size_t>(P64) * H), y0(static_cast<size_t>(T) * H);
      PP_CK(cudaMemcpy(h1s0.data(), H1.as(), h1s0.size() * 4, cudaMemcpyDeviceToHost));
      PP_CK(cudaMemcpy(os0.data(), O.as(), os0.size() * 4, cudaMemcpyDeviceToHost));
      PP_CK(cudaMemcpy(y0.data(), Y0.as(), y0.size() * 4, cudaMemcpyDeviceToHost));
      double maxO = 0.0;
      for (int i = 0; i < NP; ++i) {
        const float* o = &os0[static_cast<size_t>(prow64[i]) * H];
        for (int cc = 0; cc < H; ++cc) maxO = std::max(maxO, static_cast<double>(std::fabs(o[cc]) * wt[i]));
      }
      const int k32 = maxO > 0 ? static_cast<int>(std::floor(std::log2(268435455.0 / maxO))) : 20;
      const float sc32 = std::ldexp(1.f, k32);
      const double t_s0 = time_ms(s, s0_slice, 100.0);
      int active = 0;
      for (int e = 0; e < E; ++e) active += cnt[e] ? 1 : 0;
      JW row;
      row.kv("layer", L).kv("chunks", nch).kv("tokens", T).kv("picks", NP).kv("active_experts", active).kv("rows_pad64", P64).kv("ms_s0_slice", t_s0);
      row.kv("ms_host_list_build_copy_sync", t_host_list).kv("int32_scale_log2", k32);
      std::printf("  layer %2d, %4d tokens: %d of %d experts active; S0 (pad-64, host list, separate combine) whole slice %.4f ms; host list build + copy + sync %.4f ms\n", L, T,
                  active, E, t_s0, t_host_list);
      std::fflush(stdout);
      // ---- S3 per cut
      DevBuf dcnt(E * 4), doff(E * 4), dcur(E * 4), dmeta(16), dsrc(static_cast<size_t>(NP) * 4), drt(static_cast<size_t>(NP + 64) * 4),
          drw(static_cast<size_t>(NP + 64) * 4), dprow(static_cast<size_t>(NP) * 4);
      const int max_tiles = NP / 64 + E;
      DevBuf dut(static_cast<size_t>(max_tiles) * 16), duv(static_cast<size_t>(E) * 16);
      std::vector<JW> cj;
      for (const int ord : orders)
      for (const int cut : cuts) {
        if (g_abort.load()) break;
        const bool colx = ord == 1;
        auto build = [&] {
          PP_CK(pkc::route_list_gpu(dsel.as<int16_t>(), dwt.as<float>(), T, NU, cut, dcnt.as<int>(), doff.as<int>(), dcur.as<int>(), dut.as(), duv.as(), dmeta.as<int>(),
                                    dsrc.as<int>(), drt.as<int>(), drw.as<float>(), dprow.as<int>(), s));
        };
        auto gemm1 = [&] {
          PP_CK(pkc::s3_tiles(0, max_tiles, dmeta.as<int>(), dut.as(), A1(), s, colx));
          PP_CK(pkc::s3_vec(0, cut, E, dmeta.as<int>() + 1, duv.as(), A1(), s));
        };
        auto gemm2 = [&](int epi) {
          const pkc::MoeArgs A = A2(drt.as<int>(), drw.as<float>(), sc32);
          PP_CK(pkc::s3_tiles(epi, max_tiles, dmeta.as<int>(), dut.as(), A, s, colx));
          PP_CK(pkc::s3_vec(epi, cut, E, dmeta.as<int>() + 1, duv.as(), A, s));
        };
        auto fold = [&] {
          PP_CK(cudaMemsetAsync(Yi.as(), 0, static_cast<size_t>(T) * H * 4, s));
          gemm2(2);
          PP_CK(pkc::fix_to_f32(Yi.as<int>(), nullptr, static_cast<size_t>(T) * H, 1.f / sc32, Yc.as<float>(), s));
        };
        auto s3_slice = [&] {
          norm();
          build();
          gather(dsrc.as<int>(), NP);
          gemm1();
          act(NP);
          fold();
        };
        // one pass with the store epilogue for the row checks, then the fold
        PP_CK(cudaMemsetAsync(ovf.as(), 0, 16, s));
        norm();
        build();
        gather(dsrc.as<int>(), NP);
        gemm1();
        act(NP);
        gemm2(0);
        PP_CK(cudaStreamSynchronize(s));
        int meta[4] = {0, 0, 0, 0};
        PP_CK(cudaMemcpy(meta, dmeta.as(), 16, cudaMemcpyDeviceToHost));
        std::vector<int> prow3(static_cast<size_t>(NP));
        std::vector<float> h1(static_cast<size_t>(NP) * 2 * F), o3(static_cast<size_t>(NP) * H);
        PP_CK(cudaMemcpy(prow3.data(), dprow.as(), prow3.size() * 4, cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(h1.data(), H1.as(), h1.size() * 4, cudaMemcpyDeviceToHost));
        PP_CK(cudaMemcpy(o3.data(), O.as(), o3.size() * 4, cudaMemcpyDeviceToHost));
        int exp_t = 0, exp_v = 0;
        for (int e = 0; e < E; ++e) {
          if (cnt[e] > cut) exp_t += (cnt[e] + 63) / 64;
          else if (cnt[e]) ++exp_v;
        }
        const bool list_ok = meta[0] == exp_t && meta[1] == exp_v && meta[2] == active && meta[3] == NP;
        bool tile_same = true;
        double vec_err = 0.0;
        std::vector<int> seen(static_cast<size_t>(NP), 0);
        bool perm_ok = true;
        for (int i = 0; i < NP; ++i) {
          const int p3 = prow3[i];
          if (p3 < 0 || p3 >= NP || seen[p3]++) {
            perm_ok = false;
            break;
          }
          const int e = sel16[i];
          const float* a0 = &h1s0[static_cast<size_t>(prow64[i]) * 2 * F];
          const float* a3 = &h1[static_cast<size_t>(p3) * 2 * F];
          const float* b0 = &os0[static_cast<size_t>(prow64[i]) * H];
          const float* b3 = &o3[static_cast<size_t>(p3) * H];
          if (cnt[e] > cut) {
            tile_same = tile_same && std::memcmp(a0, a3, static_cast<size_t>(2 * F) * 4) == 0 && std::memcmp(b0, b3, static_cast<size_t>(H) * 4) == 0;
          } else {
            double mx = 0.0, md = 0.0;
            for (int cc = 0; cc < 2 * F; ++cc) {
              mx = std::max(mx, static_cast<double>(std::fabs(a0[cc])));
              md = std::max(md, static_cast<double>(std::fabs(a3[cc] - a0[cc])));
            }
            if (mx > 0) vec_err = std::max(vec_err, md / mx);
          }
        }
        // the int32 fold: within 1e-4 of S0's combine, same bits on repeat, no overflow
        fold();
        PP_CK(cudaStreamSynchronize(s));
        std::vector<float> ya(static_cast<size_t>(T) * H), yb(static_cast<size_t>(T) * H);
        PP_CK(cudaMemcpy(ya.data(), Yc.as(), ya.size() * 4, cudaMemcpyDeviceToHost));
        fold();
        PP_CK(cudaStreamSynchronize(s));
        PP_CK(cudaMemcpy(yb.data(), Yc.as(), yb.size() * 4, cudaMemcpyDeviceToHost));
        unsigned nov = 0;
        PP_CK(cudaMemcpy(&nov, ovf.as(), 4, cudaMemcpyDeviceToHost));
        const bool rep = std::memcmp(ya.data(), yb.data(), ya.size() * 4) == 0;
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ya.size(); ++i) {
          const double d = static_cast<double>(ya[i]) - y0[i];
          num += d * d;
          den += static_cast<double>(y0[i]) * y0[i];
        }
        const double rl2 = den > 0 ? std::sqrt(num / den) : 0.0;
        const bool ok = list_ok && perm_ok && tile_same && vec_err <= 1e-3 && rl2 <= 1e-4 && rep && nov == 0;
        all_ok = all_ok && ok;
        const double t_build = time_ms(s, build, 40.0), t_g1 = time_ms(s, gemm1, 60.0);
        act(NP);
        const double t_fold = time_ms(s, fold, 60.0), t_sl = time_ms(s, s3_slice, 100.0);
        JW j;
        j.kv("grid_order", colx ? "column tile in x (S3b)" : "unit in x (S3)").kv("cut", cut).kv("tile_units", meta[0]).kv("vector_units", meta[1]).kv("ms_gpu_list_build", t_build).kv("ms_gate_up", t_g1).kv("ms_down_int32_fold", t_fold);
        j.kv("ms_whole_slice", t_sl).kv("over_s0", t_s0 / t_sl).kv("list_counts_match_host", list_ok).kv("rows_permutation_ok", perm_ok).kv("tile_rows_bit_equal_s0", tile_same);
        j.kv("vector_max_rel_vs_s0", vec_err).kv("fold_rel_l2_vs_s0", rl2).kv("fold_repeat_bit_equal", rep).kv("int32_overflows", static_cast<long long>(nov)).kv("ok", ok);
        cj.push_back(j);
        std::printf("    %-3s cut %2d: %4d tile + %3d vector units | GPU list %.4f ms | gate|up %.4f | down + int32 fold %.4f | whole slice %.4f ms (%.2fx of S0) | list = host %s, "
                    "tile rows = S0 bit for bit %s, vector %.1e, fold vs S0 %.1e, repeat %s, overflows %u%s\n",
                    colx ? "S3b" : "S3", cut, meta[0], meta[1], t_build, t_g1, t_fold, t_sl, t_s0 / t_sl, list_ok ? "yes" : "NO", tile_same ? "yes" : "NO", vec_err, rl2, rep ? "same" : "DIFF",
                    nov, ok ? "" : "  FAIL");
        std::fflush(stdout);
      }
      row.arr_obj("s3", cj);
      rows.push_back(row);
    }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("routes", rpath).kv("first_chunk", chunk).kv("tokens_per_chunk", static_cast<int>(R.scored)).kv("int8_peak_tops", peak8).kv("format", "per-32");
  r.values.arr_obj("layers", rows).kv("all_ok", all_ok).kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.pass = all_ok ? -1 : 0;
  r.headline = std::string(all_ok ? "all checks passed" : "a check FAILED") + "; S3 per cut against S0 per layer in the JSON";
  r.notes = std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "moe_slice_s3");
  std::printf("  MOE-SLICE-S3 %s\n", r.headline.c_str());
  return all_ok ? 0 : 3;
}


// ================================================================ TG2: the int8 prompt path for Q4_TG2 (2026-10-30 22:2x)
// Author 22:17 "go ahead with the five-point probe": Q4_TG2's grids are fp32 values (1.3 / 1.45 log grids), so the Q4_K
// int8 trick is not exact. Five tile variants (kernels_tg2.h) on 64-row tiles, each against fp64 of the dequantized
// weights and rounded activations (relative L2, the src gate 1e-6), timed; the warp-vector path (1-8 rows) with the exact
// fp32 table next to the Q4_K vector kernel; the Q4_K tile with the v10 bf16 two-digit min term against v10b's fp32 FMAs
// (speed; accuracy against fp64 too). Weights and activations synthetic (random codes, tags, levels); shapes of the real
// model (expert gate/up 2048 -> 512, down 512 -> 2048, a dense 2048 x 2048).
namespace tg2p {
inline float h2f_host(uint16_t h) {  // fp16 bits to float (normal numbers and zero; the generator makes only those)
  const int e = (h >> 10) & 31, m = h & 1023;
  if (e == 0) return std::ldexp(static_cast<float>(m), -24) * ((h & 0x8000) ? -1.f : 1.f);
  const float v = std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, e - 15);
  return (h & 0x8000) ? -v : v;
}
struct Lcg {
  uint64_t x;
  explicit Lcg(uint64_t s) : x(s * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t next() {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(x >> 33);
  }
};
struct Acts {
  int rows = 0, K = 0;
  std::vector<int8_t> q;
  std::vector<float> xs;
  std::vector<int> sum;
};
inline Acts make_acts(int rows, int K, uint64_t seed) {
  Acts a;
  a.rows = rows;
  a.K = K;
  Lcg r(seed);
  a.q.resize(static_cast<size_t>(rows) * K);
  a.xs.resize(static_cast<size_t>(rows) * (K / 32));
  a.sum.resize(a.xs.size());
  for (auto& v : a.q) v = static_cast<int8_t>(static_cast<int>(r.next() % 255u) - 127);
  for (size_t i = 0; i < a.xs.size(); ++i) {
    a.xs[i] = 0.005f + 0.015f * static_cast<float>(r.next() % 1000u) / 1000.0f;
    int s = 0;
    for (int k = 0; k < 32; ++k) s += a.q[i * 32 + k];
    a.sum[i] = s;
  }
  return a;
}
inline std::vector<uint8_t> make_tg2(size_t nb, uint64_t seed) {
  std::vector<uint8_t> w(nb * 144);
  Lcg r(seed);
  for (size_t i = 0; i < nb; ++i) {
    uint8_t* b = &w[i * 144];
    const uint16_t d = static_cast<uint16_t>(0x2000u + (r.next() & 1023u));  // 2^-7 .. 2^-6
    b[0] = static_cast<uint8_t>(d & 0xFF);
    b[1] = static_cast<uint8_t>(d >> 8);
    uint64_t lv = 0;
    for (int j = 0; j < 8; ++j) lv |= static_cast<uint64_t>(1u + r.next() % 63u) << (6 * j);
    for (int k = 0; k < 6; ++k) b[2 + k] = static_cast<uint8_t>(lv >> (8 * k));
    for (int k = 8; k < 144; ++k) b[k] = static_cast<uint8_t>(r.next() & 0xFF);
  }
  return w;
}
inline std::vector<uint8_t> make_q4k(size_t nb, uint64_t seed) {
  std::vector<uint8_t> w(nb * 144);
  Lcg r(seed);
  for (size_t i = 0; i < nb; ++i) {
    uint8_t* b = &w[i * 144];
    const uint16_t d = static_cast<uint16_t>(0x1C00u + (r.next() & 1023u)), dm = static_cast<uint16_t>(0x1800u + (r.next() & 1023u));
    b[0] = static_cast<uint8_t>(d & 0xFF);
    b[1] = static_cast<uint8_t>(d >> 8);
    b[2] = static_cast<uint8_t>(dm & 0xFF);
    b[3] = static_cast<uint8_t>(dm >> 8);
    for (int k = 4; k < 144; ++k) b[k] = static_cast<uint8_t>(r.next() & 0xFF);
  }
  return w;
}
inline void q4k_sm(int j, const uint8_t* q, int& d, int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
// fp64 reference of one output: row `row` of the activations, weight row `wrow` (expert-major)
inline double ref_tg2(const Acts& a, int row, const std::vector<uint8_t>& w, size_t wrow, const float g[4][16]) {
  const int K = a.K, bpr = K / 256;
  double acc = 0;
  for (int sb = 0; sb < bpr; ++sb) {
    const uint8_t* b = &w[(wrow * bpr + sb) * 144];
    const float d = h2f_host(static_cast<uint16_t>(b[0] | (b[1] << 8)));
    uint64_t lv = 0;
    for (int k = 0; k < 6; ++k) lv |= static_cast<uint64_t>(b[2 + k]) << (8 * k);
    for (int j = 0; j < 8; ++j) {
      volatile float dl = d * static_cast<float>((lv >> (6 * j)) & 63u);
      const float s = dl / 63.0f;
      double dot = 0;
      for (int i = 0; i < 32; ++i) {
        const int idx = 32 * j + i;
        const int code = (b[16 + idx / 2] >> (4 * (idx & 1))) & 15, tag = (b[8 + j] >> (2 * (i >> 3))) & 3;
        dot += static_cast<double>(g[tag][code]) * a.q[static_cast<size_t>(row) * K + sb * 256 + idx];
      }
      acc += static_cast<double>(a.xs[static_cast<size_t>(row) * (K / 32) + sb * 8 + j]) * static_cast<double>(s) * dot;
    }
  }
  return acc;
}
inline double ref_q4k(const Acts& a, int row, const std::vector<uint8_t>& w, size_t wrow) {
  const int K = a.K, bpr = K / 256;
  double acc = 0;
  for (int sb = 0; sb < bpr; ++sb) {
    const uint8_t* b = &w[(wrow * bpr + sb) * 144];
    const float d = h2f_host(static_cast<uint16_t>(b[0] | (b[1] << 8))), dmin = h2f_host(static_cast<uint16_t>(b[2] | (b[3] << 8)));
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      q4k_sm(j, b + 4, sc, m);
      const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(m);
      long long qq = 0, qs = 0;
      for (int i = 0; i < 32; ++i) {
        const int qw = (b[16 + (j / 2) * 32 + i] >> (4 * (j & 1))) & 15, qx = a.q[static_cast<size_t>(row) * K + sb * 256 + 32 * j + i];
        qq += static_cast<long long>(qw) * qx;
        qs += qx;
      }
      acc += static_cast<double>(a.xs[static_cast<size_t>(row) * (K / 32) + sb * 8 + j]) *
             (static_cast<double>(d1) * static_cast<double>(qq) - static_cast<double>(m1) * static_cast<double>(qs));
    }
  }
  return acc;
}
struct Unit {
  int e, row, rows;
};
// relative L2 of C (device, rows x N) against fp64 on the sampled rows (first 16 rows of the first unit, last 16 of the last)
template <class Ref>
double rel_l2(const DevBuf& C, int N, const std::vector<Unit>& units, Ref ref, cudaStream_t s) {
  std::vector<std::pair<int, int>> pick;  // (row, expert)
  const Unit& u0 = units.front();
  const Unit& u1 = units.back();
  for (int i = 0; i < std::min(16, u0.rows); ++i) pick.push_back({u0.row + i, u0.e});
  if (units.size() > 1)
    for (int i = std::max(0, u1.rows - 16); i < u1.rows; ++i) pick.push_back({u1.row + i, u1.e});
  std::vector<float> y(static_cast<size_t>(N));
  double num = 0, den = 0;
  std::vector<double> tn(static_cast<size_t>(N / 128 + 1), 0.0), td(tn.size(), 0.0), rn, rd;
  for (const auto& [row, e] : pick) {
    PP_CK(cudaMemcpyAsync(y.data(), C.as<float>() + static_cast<size_t>(row) * N, static_cast<size_t>(N) * 4, cudaMemcpyDeviceToHost, s));
    PP_CK(cudaStreamSynchronize(s));
    double a = 0, b = 0;
    for (int n = 0; n < N; ++n) {
      const double r = ref(row, static_cast<size_t>(e) * N + n);
      num += (y[n] - r) * (y[n] - r);
      den += r * r;
      tn[static_cast<size_t>(n / 128)] += (y[n] - r) * (y[n] - r);
      td[static_cast<size_t>(n / 128)] += r * r;
      a += (y[n] - r) * (y[n] - r);
      b += r * r;
    }
    rn.push_back(a);
    rd.push_back(b);
  }
  const double e = den > 0 ? std::sqrt(num / den) : 0.0;
  if (!(e <= 1e-6)) {  // where the error sits: per column tile of 128 and per checked row
    std::printf("      diag: per column tile:");
    for (size_t t = 0; t < tn.size(); ++t)
      if (td[t] > 0) std::printf(" %zu:%.1e", t, std::sqrt(tn[t] / td[t]));
    std::printf("\n      diag: per checked row:");
    for (size_t i = 0; i < pick.size(); ++i) std::printf(" %d:%.1e", pick[i].first, rd[i] > 0 ? std::sqrt(rn[i] / rd[i]) : 0.0);
    std::printf("\n");
  }
  return e;
}
}  // namespace tg2p

inline int p_tg2(Ctx& c, const Args& a) {
  using namespace tg2p;
  Result r;
  r.probe = "TG2";
  init_cuda(c);
  const double peak8 = a.num("int8-peak", 259.5);
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(200);
  JW before = state_now(c);
  float grid[4][16];
  ptg::tg2_grids(grid);
  constexpr double kGate = 1e-6;
  std::vector<JW> infos;
  for (int v = 0; v < ptg::kTg2Vars; ++v) {
    int regs = 0, spill = 0, sm = 0, bps = 0;
    PP_CK(ptg::tg2_tile_info(v, &regs, &spill, &sm, &bps));
    JW j;
    j.kv("kernel", ptg::var_name(v)).kv("registers", regs).kv("local_spill_bytes", spill).kv("shared_bytes", sm).kv("blocks_per_sm", bps);
    infos.push_back(j);
    std::printf("    %-30s %3d registers, %4d spill bytes, %6d shared bytes, %d blocks per SM\n", ptg::var_name(v), regs, spill, sm, bps);
  }
  for (int f = 0; f < 2; ++f) {
    int regs = 0, spill = 0, sm = 0, bps = 0;
    PP_CK(ptg::q4k_tile_info(f == 1, &regs, &spill, &sm, &bps));
    JW j;
    const char* nm = f ? "q4k tile, fp32 min (v10b)" : "q4k tile, bf16 min (v10)";
    j.kv("kernel", nm).kv("registers", regs).kv("local_spill_bytes", spill).kv("shared_bytes", sm).kv("blocks_per_sm", bps);
    infos.push_back(j);
    std::printf("    %-30s %3d registers, %4d spill bytes, %6d shared bytes, %d blocks per SM\n", nm, regs, spill, sm, bps);
  }
  // fixed tables, uploaded once
  std::vector<DevBuf> luts;
  for (int v = 0; v < ptg::kTg2Vars; ++v) {
    uint32_t t[64];
    ptg::tg2_table(v, t);
    luts.emplace_back(sizeof t);
    PP_CK(cudaMemcpy(luts.back().as(), t, sizeof t, cudaMemcpyHostToDevice));
  }
  struct Case {
    const char* name;
    int E, N, K, units, rows_per_unit;  // tile cases: rows_per_unit 64; vector cases 1-8
    bool vec;
  };
  std::vector<Case> cases = {{"expert gate/up grouped (64 experts x 64 rows)", 64, 512, 2048, 64, 64, false},
                             {"expert down grouped (64 experts x 64 rows)", 64, 2048, 512, 64, 64, false},
                             {"dense 2048 x 2048, 2048 rows", 1, 2048, 2048, 32, 64, false}};
  for (const int rows : {1, 4, 8}) {
    cases.push_back({"vector gate/up (64 experts)", 64, 512, 2048, 64, rows, true});
    cases.push_back({"vector down (64 experts)", 64, 2048, 512, 64, rows, true});
  }
  if (c.quick) cases = {cases[0], cases[3]};
  bool all_ok = true;
  std::vector<JW> rows_out;
  for (const Case& cs : cases) {
    if (g_abort.load()) break;
    const int bpr = cs.K / 256;
    const size_t nb = static_cast<size_t>(cs.E) * cs.N * bpr;
    std::vector<Unit> units;
    std::vector<int4> uh;
    for (int u = 0; u < cs.units; ++u) {
      const Unit un{cs.E == 1 ? 0 : u % cs.E, u * cs.rows_per_unit, cs.rows_per_unit};
      units.push_back(un);
      uh.push_back(make_int4(un.e, un.row, un.rows, 0));
    }
    const int arows = cs.units * cs.rows_per_unit + 64;
    const Acts ac = make_acts(arows, cs.K, 9100u + static_cast<uint64_t>(cs.N) + static_cast<uint64_t>(cs.rows_per_unit));
    const std::vector<uint8_t> wt = make_tg2(nb, 9200u + static_cast<uint64_t>(cs.K)), wk = make_q4k(nb, 9300u + static_cast<uint64_t>(cs.K));
    DevBuf dq(ac.q.size()), dxs(ac.xs.size() * 4), dsum(ac.sum.size() * 4), dU(uh.size() * sizeof(int4)), dwt(wt.size()), dwk(wk.size()), dsc(nb * 32),
        drec(nb * 16), dwb(nb * 32), dwf(nb * 32), dab(ac.xs.size() * 4), daf(ac.xs.size() * 4),
        dC(static_cast<size_t>(arows) * cs.N * 4);
    PP_CK(cudaMemcpy(dq.as(), ac.q.data(), ac.q.size(), cudaMemcpyHostToDevice));
    PP_CK(cudaMemcpy(dxs.as(), ac.xs.data(), ac.xs.size() * 4, cudaMemcpyHostToDevice));
    PP_CK(cudaMemcpy(dsum.as(), ac.sum.data(), ac.sum.size() * 4, cudaMemcpyHostToDevice));
    PP_CK(cudaMemcpy(dU.as(), uh.data(), uh.size() * sizeof(int4), cudaMemcpyHostToDevice));
    PP_CK(cudaMemcpy(dwt.as(), wt.data(), wt.size(), cudaMemcpyHostToDevice));
    PP_CK(cudaMemcpy(dwk.as(), wk.data(), wk.size(), cudaMemcpyHostToDevice));
    // pageable cudaMemcpy may return before its DMA lands, and stream s is non-blocking: wait for the uploads before
    // the prep kernels read them (TG2 v1's dense Q4_K 0.507, 2026-10-30, did not reproduce in the next run)
    PP_CK(cudaDeviceSynchronize());
    PP_CK(ptg::tg2_prep(dwt.as(), dsc.as<float>(), nb, s));
    PP_CK(ptg::q4k_prep(dwk.as(), drec.as<uint4>(), dwb.as(), dwf.as<float>(), nb, s));
    PP_CK(ptg::q4k_act_min(dxs.as<float>(), dsum.as<int>(), ac.xs.size(), dab.as(), daf.as<float>(), s));
    s.sync();
    const double mac = 2.0 * cs.units * cs.rows_per_unit * static_cast<double>(cs.N) * cs.K;
    const double wbytes = static_cast<double>(std::min(cs.E, cs.units)) * cs.N * bpr * 144.0;
    auto ref_t = [&](int row, size_t wrow) { return ref_tg2(ac, row, wt, wrow, grid); };
    auto ref_k = [&](int row, size_t wrow) { return ref_q4k(ac, row, wk, wrow); };
    JW row;
    row.kv("case", cs.name).kv("experts", cs.E).kv("N", cs.N).kv("K", cs.K).kv("units", cs.units).kv("rows_per_unit", cs.rows_per_unit);
    std::vector<JW> kj;
    auto record = [&](const char* nm, const std::function<void()>& run, bool is_tg2, int v) {
      const double ms = time_ms(s, run, 80.0);
      PP_CK(cudaMemsetAsync(dC.as(), 0xFF, static_cast<size_t>(arows) * cs.N * 4, s));
      run();
      PP_CK(cudaStreamSynchronize(s));
      const double e = is_tg2 ? rel_l2(dC, cs.N, units, ref_t, s) : rel_l2(dC, cs.N, units, ref_k, s);
      const double tops = mac / (ms * 1e-3) / 1e12, gbs = wbytes / (ms * 1e-3) / 1e9;
      const bool pass = std::isfinite(e) && e <= kGate;
      JW j;
      j.kv("kernel", nm).kv("ms", ms).kv("tops", tops).kv("eps8", tops / peak8).kv("weight_gb_s", gbs).kv("rel_l2_vs_fp64", e).kv("passes_1e-6", pass);
      if (v >= 0) j.kv("variant", v);
      kj.push_back(j);
      std::printf("    %-44s %-30s %8.4f ms %6.1f TOPS (%.3f of peak) %6.1f GB/s weights; vs fp64 rel L2 %.3e %s\n", cs.name, nm, ms, tops, tops / peak8, gbs, e,
                  pass ? "<= 1e-6" : "> 1e-6");
      std::fflush(stdout);
      return ms;
    };
    char lbl[64];
    if (!cs.vec) {
      const double tb = record("q4k tile, bf16 min (v10)", [&] {
        PP_CK(ptg::q4k_tile(false, dq.as<int8_t>(), dxs.as<float>(), dab.as(), dwk.as(), drec.as<uint4>(), dwb.as(), dU.as<int4>(), cs.units, cs.N, cs.K, dC.as<float>(), s));
      }, false, -1);
      const double tf = record("q4k tile, fp32 min (v10b)", [&] {
        PP_CK(ptg::q4k_tile(true, dq.as<int8_t>(), dxs.as<float>(), daf.as(), dwk.as(), drec.as<uint4>(), dwf.as(), dU.as<int4>(), cs.units, cs.N, cs.K, dC.as<float>(), s));
      }, false, -1);
      row.kv("q4k_fp32min_over_bf16min_time", tf / tb);
      std::printf("  %-44s min term: fp32 FMAs take %.3fx the time of the bf16 mma\n", cs.name, tf / tb);
      for (int v = 0; v < ptg::kTg2Vars; ++v) {
        const double tv = record(ptg::var_name(v), [&] {
          PP_CK(ptg::tg2_tile(v, dq.as<int8_t>(), dxs.as<float>(), dwt.as(), dsc.as<float>(), luts[v].as<uint32_t>(), dU.as<int4>(), cs.units, cs.N, cs.K, dC.as<float>(), s));
        }, true, v);
        std::snprintf(lbl, sizeof lbl, "time_over_q4k_v10b_%d", v);
        row.kv(lbl, tv / tf);
        std::printf("  %-44s %-30s %.3fx the Q4_K v10b tile time\n", cs.name, ptg::var_name(v), tv / tf);
      }
    } else {
      const double tk = record("q4k vector", [&] {
        PP_CK(ptg::q4k_vec(dq.as<int8_t>(), dxs.as<float>(), dwk.as(), dU.as<int4>(), cs.units, cs.N, cs.K, dC.as<float>(), s));
      }, false, -1);
      const double tt = record("tg2 vector, exact fp32 table", [&] {
        PP_CK(ptg::tg2_vec(dq.as<int8_t>(), dxs.as<float>(), dwt.as(), dsc.as<float>(), luts[ptg::kF32Lut].as<uint32_t>(), dU.as<int4>(), cs.units, cs.N, cs.K,
                           dC.as<float>(), s));
      }, true, ptg::kF32Lut);
      row.kv("tg2_vec_over_q4k_vec_time", tt / tk);
      std::printf("  %-44s rows %d: tg2 exact vector %.3fx the Q4_K vector time\n", cs.name, cs.rows_per_unit, tt / tk);
    }
    row.arr_obj("kernels", kj);
    rows_out.push_back(row);
  }
  const std::vector<Sample> all = sampler.stop();
  r.inputs.kv("int8_peak_tops", peak8).kv("gate_rel_l2", kGate).kv("weights_activations", "synthetic (random codes, tags, levels; int8 activations, per-32 scales)");
  r.values.arr_obj("kernel_info", infos).arr_obj("cases", rows_out);
  r.values.kv_obj("state_before", before).kv_obj("state_after", state_now(c)).kv_obj("nvml_whole_run", summarize(all));
  r.pass = -1;
  r.headline = "TG2 variants timed and checked against fp64 (gate 1e-6 relative L2); min-term sweep recorded";
  r.notes = std::string(g_abort.load() ? "ABORTED: " + abort_reason() : "");
  write_result(c, r, "tg2");
  std::printf("  TG2 %s\n", r.headline.c_str());
  return all_ok ? 0 : 3;
}

}  // namespace pp

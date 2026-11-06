// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Confidential, not for distribution.
//
// tools/probes/sdecode/sdecode.cu - the expert stage's DECODE cost per format, with the weight bytes already in shared
// memory (R3 rebuild design, stage 0.1; 2026-11-06). It compiles the engine's own kernel bodies (kern/cuda/gpu_stream.cu is
// included as a translation unit, so lane_part / lane_part_n are the exact code the fused MoE half runs) and times one
// warp task (32 lanes x 128 values = 2 rows of K = 2048, nB = 8 blocks per row - the gate / up shape) read through the
// staged path (M = 3: weights from the warp's shared buffer, activations through L2) for NT = 1 and 3 tokens, at 1 and 8
// warps per SM. No DRAM is in the way, so cycles per task here = the decode (and issue) cost the stage's measured cycles per
// task (v60q counting arm: Q8_0 4,906 / Q6_K 6,647 / Q5_K planes 8,409 / Q4_TG2 planes 12,964) include on top of the bytes.
// Routes: Q5_K / Q4_K planes with the activation bit planes (the 1-bit route, as the fused kernel runs today) and without
// (the dp4a route of the same integers). Nothing here writes a file or touches the engine.
// Build (WSL): nvcc -std=c++20 -O3 -arch=sm_89 -I<repo>/src -I<repo>/include sdecode.cu <repo>/src/quant/quant_cpu.cpp -o sdecode
#include "kern/cuda/gpu_stream.cu"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace pelias::kern::gpu {  // the symbols gpu_stream.cu takes from gpu_ops.cu (unused by the bench)
cudaStream_t cur_stream_raw() { return nullptr; }
bool rd3_group_launch(const ExpertPtr*, const ExpertPtr*, const std::int32_t*, int, int, const ReaderCache*, std::int64_t, std::int64_t, int*, ExpertPtr*,
                      ExpertPtr*, Stream) {
  return false;
}
bool rd3_combine_launch(const float*, const float*, const float*, int, std::int64_t, std::int64_t, float*, const std::int32_t*, Stream) { return false; }
void stamp_launch(unsigned long long*, Stream) {}
// gpu_ops.cu's allocation trio, as the stream plan's host side uses them (plain CUDA calls here)
void* dalloc(std::size_t n) {
  void* p = nullptr;
  return cudaMalloc(&p, n) == cudaSuccess ? p : nullptr;
}
void dfree(void* p) {
  if (p) cudaFree(p);
}
bool to_device(void* dst, const void* src, std::size_t n, Stream) { return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice) == cudaSuccess; }
}  // namespace pelias::kern::gpu

using namespace pelias::kern::gpu;

// (a named namespace: nvcc's host stubs cannot tell this file's anonymous namespace from gpu_stream.cu's, 18:19 build)
namespace sdecode_probe {
constexpr int kK = 2048, kNB = kK / 256, kLpr = kK / 128, kRpw = 32 / kLpr;  // the gate / up shape: 2 rows per task

// one warp task per active warp, reps times; wbytes of weights copied into the warp's shared buffer first
// SACT: the activations (3 tokens: int8, scales, sums, bit planes) copied into the block's shared memory and read through
// M = 5, as the engine's gate / up jobs read them (v57 sact); otherwise M = 3 (activations through L2, as the down jobs)
template <int NT, bool SACT>
__global__ void __launch_bounds__(256) k_bench(DJob J, const std::uint8_t* __restrict__ wsrc, int wbytes, int reps, int wpb, unsigned long long* cycles,
                                                float* sink) {
  lut_setup_rx();
  const int w = static_cast<int>(threadIdx.x / 32), lane = static_cast<int>(threadIdx.x & 31);
  constexpr int M = SACT ? 5 : 3;
  if constexpr (SACT) {  // after the 8 warps' buffers: xq (3 x K), xs (3 x K / 32 floats), xsum (3 x K / 32 ints), xb (3 x K / 4 words)
    std::uint8_t* base = reinterpret_cast<std::uint8_t*>(s_lut) + kLutRxBytes + 8LL * wbytes;
    const int nH = 3 * kK, nb3 = nH / 32;
    std::int8_t* q8 = reinterpret_cast<std::int8_t*>(base);
    float* xs = reinterpret_cast<float*>(base + nH);
    int* xm = reinterpret_cast<int*>(base + nH + 4 * nb3);
    std::uint32_t* xb = reinterpret_cast<std::uint32_t*>(base + nH + 8 * nb3);
    for (int i = static_cast<int>(threadIdx.x); i < nH; i += 256) q8[i] = J.xq[i];
    for (int i = static_cast<int>(threadIdx.x); i < nb3; i += 256) {
      xs[i] = J.xs[i];
      xm[i] = J.xsum[i];
    }
    if (J.xb)
      for (int i = static_cast<int>(threadIdx.x); i < nH / 4; i += 256) xb[i] = J.xb[i];
    if (threadIdx.x == 0) {
      s_rx_xq0 = q8;
      s_rx_xs0 = xs;
      s_rx_xsum0 = xm;
      s_rx_xb0 = J.xb ? xb : nullptr;
      s_rx_gxq = J.xq;
    }
    __syncthreads();
  }
  if (w >= wpb) return;
  std::uint8_t* wb = reinterpret_cast<std::uint8_t*>(s_lut) + kLutRxBytes + static_cast<long long>(w) * wbytes;
  for (int i = lane * 16; i < wbytes; i += 32 * 16) *reinterpret_cast<int4*>(wb + i) = *reinterpret_cast<const int4*>(wsrc + i);
  __syncwarp();
  const int r = lane / J.lpr, part = lane % J.lpr;
  // with SACT the jobs' pointers are the shared copies (rxp_compute: bq / bs / bm = s_rx_*0 when J.xq == s_rx_gxq); the 1-bit
  // route finds its planes from the offset of xq[t] against s_rx_xq0, as the engine does
  const std::int8_t* xq[4];
  const float* xs[4];
  const int* xsum[4];
  const std::int8_t* bq = SACT ? s_rx_xq0 : J.xq;
  const float* bs = SACT ? s_rx_xs0 : J.xs;
  const int* bm = SACT ? s_rx_xsum0 : J.xsum;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int u = t < NT ? t : 0;
    xq[t] = bq + u * J.xqs;
    xs[t] = bs + u * J.xss;
    xsum[t] = bm + u * J.xss;
  }
  float s = 0.f;
  const long long c0 = clock64();
  for (int rep = 0; rep < reps; ++rep) {
    float acc[NT > 1 ? NT : 1];
    if constexpr (NT == 1) acc[0] = lane_part<M, kOnlyRx>(J, r, part, xq[0], xs[0], xsum[0], false, wb, 0);
    else lane_part_n<M, NT, kOnlyRx>(J, r, part, xq, xs, xsum, false, acc, wb, 0);
#pragma unroll
    for (int t = 0; t < (NT > 1 ? NT : 1); ++t) s += acc[t];
    // the next rep's inputs differ by a lane-dependent amount the compiler cannot fold (keeps every rep's work live)
    xq[0] += (s > 1e30f) ? 32 : 0;
  }
  const long long c1 = clock64();
  if (lane == 0) atomicAdd(cycles, static_cast<unsigned long long>(c1 - c0));
  sink[blockIdx.x * 256 + threadIdx.x] = s;
}

struct Case {
  const char* name;
  int fmt, pb, route;  // route: 0 dp4a / stored, 1 the 1-bit route (xb)
};

std::uint16_t f16_one() { return 0x3c00; }

// synthetic rows of one task in the layout the staged path reads (rows packed at the plane prefix, or stored rows)
std::vector<std::uint8_t> make_rows(int fmt, int pb, std::mt19937& rng) {
  std::vector<std::uint8_t> v;
  auto rnd = [&](std::size_t n) {
    std::vector<std::uint8_t> b(n);
    for (auto& x : b) x = static_cast<std::uint8_t>(rng());
    return b;
  };
  auto put16 = [](std::vector<std::uint8_t>& b, std::size_t at, std::uint16_t x) {
    b[at] = static_cast<std::uint8_t>(x & 0xff);
    b[at + 1] = static_cast<std::uint8_t>(x >> 8);
  };
  if (fmt == kQ4TG2P || fmt == kQ5KP || fmt == kQ4KP) {
    const std::size_t pf = static_cast<std::size_t>(plane_pf(fmt, kNB, pb));
    v = rnd(pf * kRpw);
    for (int r = 0; r < kRpw; ++r)
      for (int b = 0; b < kNB; ++b) {
        put16(v, r * pf + 16 * b, f16_one());  // d (and dmin for Q5_K / Q4_K)
        if (fmt != kQ4TG2P) put16(v, r * pf + 16 * b + 2, f16_one());
      }
  } else {
    const std::size_t blk = static_cast<std::size_t>(rx_blk(fmt)), rowb = blk * kNB;
    v = rnd(rowb * kRpw);
    for (int r = 0; r < kRpw; ++r)
      for (int b = 0; b < kNB; ++b) {
        const std::size_t at = r * rowb + b * blk;
        if (fmt == kQ6KS) put16(v, at + 208, f16_one());
        else if (fmt == kQ8_0)
          for (int i = 0; i < 8; ++i) put16(v, at + 256 + 2 * i, f16_one());
        else {
          put16(v, at, f16_one());
          put16(v, at + 2, f16_one());
        }
      }
  }
  while (v.size() % 16) v.push_back(0);
  return v;
}
}  // namespace sdecode_probe
using namespace sdecode_probe;

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 200, rounds = argc > 2 ? std::atoi(argv[2]) : 5;
  int dev = 0, sms = 0, clk = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  std::printf("sdecode: %d SMs, clock attr %d kHz; one warp task = %d rows x K %d (nB %d, lpr %d); reps %d x rounds %d (median)\n", sms, clk, kRpw, kK, kNB, kLpr,
              reps, rounds);
  if (!grid_ready()) {
    std::printf("grid upload failed\n");
    return 1;
  }
  // activations: 3 tokens x K int8, their scales (1.0) and sums, and the bit planes (act_bits32 layout: 8 words per 32 values)
  std::mt19937 rng(20261106);
  const int ntok = 3, nb = kK / 32;
  std::vector<std::int8_t> hq(static_cast<std::size_t>(ntok) * kK);
  std::vector<float> hs(static_cast<std::size_t>(ntok) * nb, 1.0f);
  std::vector<int> hsum(static_cast<std::size_t>(ntok) * nb, 0);
  std::vector<std::uint32_t> hb(static_cast<std::size_t>(ntok) * kK / 4, 0u);
  for (int t = 0; t < ntok; ++t)
    for (int j = 0; j < nb; ++j) {
      int sm = 0;
      for (int l = 0; l < 32; ++l) {
        const int qv = static_cast<int>(rng() % 255) - 127;
        hq[static_cast<std::size_t>(t) * kK + 32 * j + l] = static_cast<std::int8_t>(qv);
        sm += qv;
        const unsigned u = static_cast<unsigned>(qv + 128);
        for (int k = 0; k < 8; ++k)
          if ((u >> k) & 1u) hb[static_cast<std::size_t>(t) * kK / 4 + 8 * j + k] |= 1u << l;
      }
      hsum[static_cast<std::size_t>(t) * nb + j] = sm;
    }
  std::int8_t* dq = nullptr;
  float* ds = nullptr;
  int* dsum = nullptr;
  std::uint32_t* db = nullptr;
  unsigned long long* dcyc = nullptr;
  float* dsink = nullptr;
  cudaMalloc(&dq, hq.size());
  cudaMalloc(&ds, hs.size() * 4);
  cudaMalloc(&dsum, hsum.size() * 4);
  cudaMalloc(&db, hb.size() * 4);
  cudaMalloc(&dcyc, 8);
  cudaMalloc(&dsink, static_cast<std::size_t>(sms) * 256 * 4);
  cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(ds, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsum, hsum.data(), hsum.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
  const Case cases[] = {{"Q8_0 (shared expert, VRAM)", kQ8_0, 4, 0},       {"Q6_K stored (210 B, 2-byte aligned)", kQ6KS, 4, 0},
                        {"Q4_K stored (dp4a over nibbles)", kQ4K, 4, 0},     {"Q5_K stored (dp4a)", kQ5K, 5, 0},
                        {"Q4_TG2 planes 4 bits (LUT)", kQ4TG2P, 4, 0},       {"Q4_TG2 planes 3 bits (LUT)", kQ4TG2P, 3, 0},
                        {"Q4_TG2 planes 2 bits (LUT)", kQ4TG2P, 2, 0},       {"Q5_K planes 3 bits, 1-bit route", kQ5KP, 3, 1},
                        {"Q5_K planes 3 bits, dp4a route", kQ5KP, 3, 0},     {"Q5_K planes 4 bits, 1-bit route", kQ5KP, 4, 1},
                        {"Q5_K planes 4 bits, dp4a route", kQ5KP, 4, 0},     {"Q5_K planes 5 bits, 1-bit route", kQ5KP, 5, 1},
                        {"Q5_K planes 5 bits, dp4a route", kQ5KP, 5, 0},     {"Q4_K planes 3 bits, 1-bit route", kQ4KP, 3, 1},
                        {"Q4_K planes 3 bits, dp4a route", kQ4KP, 3, 0},     {"Q4_K planes 4 bits, 1-bit route", kQ4KP, 4, 1},
                        {"Q4_K planes 4 bits, dp4a route", kQ4KP, 4, 0}};
  std::printf("%-40s %4s %5s | %9s %9s | %9s %9s | %9s %9s | %s\n", "format / route", "bits", "bytes", "1w L2 1", "1w L2 3", "8w L2 1", "8w L2 3",
              "8w sh 1", "8w sh 3", "cycles per warp task, median of rounds; Nw = warps per SM; L2 = activations through L2 (the down jobs), sh = in shared memory (the gate / up jobs); 1 / 3 = tokens");
  for (const Case& c : cases) {
    const std::vector<std::uint8_t> rows = make_rows(c.fmt, c.pb, rng);
    std::uint8_t* dw = nullptr;
    cudaMalloc(&dw, rows.size());
    cudaMemcpy(dw, rows.data(), rows.size(), cudaMemcpyHostToDevice);
    const int wbytes = static_cast<int>(rows.size());
    DJob J{};
    J.fmt = c.fmt;
    J.n = kRpw;
    J.K = kK;
    J.lpr = kLpr;
    J.task0 = 0;
    J.ntask = 1;
    J.w = dw;  // (unused by the staged path, which reads wb)
    J.pb = c.pb;
    J.rs = c.fmt == kQ4TG2P || c.fmt == kQ5KP || c.fmt == kQ4KP ? plane_pf(c.fmt, kNB, c.pb) : static_cast<long long>(rx_blk(c.fmt)) * kNB;
    J.ntok = ntok;
    J.xq = dq;
    J.xs = ds;
    J.xsum = dsum;
    J.xb = c.route ? db : nullptr;
    J.xqs = kK;
    J.xss = nb;
    J.y = dsink;
    J.ys = 0;
    if (!rx_set_grid_bits(c.fmt == kQ4TG2P ? c.pb : 4)) {
      std::printf("grid bits failed\n");
      return 1;
    }
    double res[3][2];
    const int sact_bytes = 3 * kK + 8 * (3 * kK / 32) + 3 * kK;  // xq | xs | xsum | xb for 3 tokens
    for (int wi = 0; wi < 3; ++wi) {
      const int wpb = wi == 0 ? 1 : 8;
      const bool sact = wi == 2;
      const int smem = std::max(kLutRxBytes + 8 * wbytes + (sact ? sact_bytes : 0), 60 * 1024);  // >= 60 KB: one block per SM
      for (int nt = 0; nt < 2; ++nt) {
        std::vector<double> per;
        for (int rd = 0; rd < rounds; ++rd) {
          cudaMemset(dcyc, 0, 8);
          if (nt == 0 && !sact) {
            cudaFuncSetAttribute(k_bench<1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            k_bench<1, false><<<sms, 256, smem>>>(J, dw, wbytes, reps, wpb, dcyc, dsink);
          } else if (!sact) {
            cudaFuncSetAttribute(k_bench<3, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            k_bench<3, false><<<sms, 256, smem>>>(J, dw, wbytes, reps, wpb, dcyc, dsink);
          } else if (nt == 0) {
            cudaFuncSetAttribute(k_bench<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            k_bench<1, true><<<sms, 256, smem>>>(J, dw, wbytes, reps, wpb, dcyc, dsink);
          } else {
            cudaFuncSetAttribute(k_bench<3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            k_bench<3, true><<<sms, 256, smem>>>(J, dw, wbytes, reps, wpb, dcyc, dsink);
          }
          const cudaError_t e = cudaDeviceSynchronize();
          if (e != cudaSuccess) {
            std::printf("%s: %s\n", c.name, cudaGetErrorString(e));
            return 1;
          }
          unsigned long long cyc = 0;
          cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost);
          per.push_back(static_cast<double>(cyc) / (static_cast<double>(sms) * wpb * reps));
        }
        std::sort(per.begin(), per.end());
        res[wi][nt] = per[per.size() / 2];
      }
    }
    std::printf("%-40s %4d %5d | %9.0f %9.0f | %9.0f %9.0f | %9.0f %9.0f |\n", c.name, c.pb, wbytes, res[0][0], res[0][1], res[1][0], res[1][1], res[2][0],
                res[2][1]);
    cudaFree(dw);
  }
  std::printf("done\n");
  return 0;
}

// provenance: original (moved from tools/probes/batch_c kernels_c.cu: V12x8Stage, k_moe_tile<3, 64, EPI, true, true>,
// k_moe_vec, k_route_count / k_route_build / k_route_place, k_gather_q8, k_fix_to_f32, k_prep_q4k_rec, k_prep_q4k_dig8,
// k_dig_x32; MEASURED 2026-10-30 21:05-21:40, gp/MOE; the device arithmetic unchanged. New here: the dense mode of the
// tile kernel (units computed from blockIdx), int32 expert ids from router_topk, and silu x up rounded with gpu_ops's
// own expressions (silu_d, act_q8x32's loops), so the down projection's input is the reference's. Q4_TG2 (2026-10-30
// 23:3x): the TG2 probe's int8 x 3 tile and exact-table vector kernels (tools/probes/batch_c/kernels_tg2.cu, MEASURED
// 22:31, locked 23:22) with this file's epilogues and unit modes)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/cuda/gpu_prompt.cu - the int8 prompt path behind kern/gpu_prompt.h.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "kern/gpu_decode.h"
#include "kern/gpu_prompt.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::kern::gpu {
namespace {

thread_local std::string p_err;
bool pfail(const std::string& why) {
  p_err = why;
  return false;
}
bool pck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  return pfail(std::string(what) + ": " + cudaGetErrorString(e));
}
cudaStream_t pst(Stream s) { return static_cast<cudaStream_t>(s); }

constexpr int kQK = 256, kBN = 128, kTile = 64, kVecMax = 8, kStages = 3;
constexpr int kQ8Lda = 64 + 16, kQsLd = 32 + 16, kMinLd = 12;  // kMinLd: 8 floats + 4 (48-byte rows, float4 reads without bank conflicts)

struct BlockQ4K {
  std::uint16_t d, dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K is 144 bytes");

__device__ __forceinline__ float h2f(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ void scale_min_k4_d(int j, const std::uint8_t* q, int& d, int& m) {  // get_scale_min_k4 (ggml-quants.c)
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
__device__ __forceinline__ std::uint32_t ld32(const void* p) { return *reinterpret_cast<const std::uint32_t*>(p); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float silu_p(float x) { return quo(x, add(1.0f, expf(-x))); }  // gpu_ops.cu silu_d

// ------------------------------------------------------------------------------------------------ weight side tables

__global__ void k_prep_rec(const std::uint8_t* __restrict__ w, int bstride, uint4* __restrict__ rec, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = *reinterpret_cast<const BlockQ4K*>(w + i * static_cast<std::size_t>(bstride));  // d, dmin, scales: Q4_K and Q5_K alike
    std::uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      if (j < 4) lo |= static_cast<std::uint32_t>(sc) << (8 * j);
      else hi |= static_cast<std::uint32_t>(sc) << (8 * (j - 4));
    }
    rec[i] = make_uint4(__float_as_uint(h2f(b.d)), lo, hi, __float_as_uint(h2f(b.dmin)));
  }
}
// the weight side of the min term in fp32: dmin x m_j per 32-block (fp16 x 6-bit integer, exact in fp32), [rows][k / 32]
__global__ void k_prep_dig(const std::uint8_t* __restrict__ w, int bstride, float* __restrict__ wd, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = *reinterpret_cast<const BlockQ4K*>(w + i * static_cast<std::size_t>(bstride));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      wd[i * 8 + j] = h2f(b.dmin) * static_cast<float>(m);
    }
  }
}
// the activation side of the min term in fp32: scale x float(sum) per 32-block, [rows][k / 32]
__global__ void k_act_digits(const float* __restrict__ scale, const int* __restrict__ sum, std::size_t n32, float* __restrict__ dig) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n32; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    dig[i] = scale[i] * static_cast<float>(sum[i]);
}

// ------------------------------------------------------------------------------------------------ the tile kernel

// F: 0 = Q4_K (144-byte blocks), 1 = Q5_K (176: d, dmin, scales as Q4_K, then qh[32] = the 5th bit of value l of sub-block
// j at bit j of byte l, then qs[128] as Q4_K's)
template <int F>
struct alignas(16) StageT {
  std::int8_t As[kTile][kQ8Lda];
  std::uint8_t Qs[kBN][kQsLd];
  uint4 Wr[kBN];
  float S2[kTile][2];
  float Am[kTile][kMinLd];  // activation min factors of the super-block (8 used)
  float Bm[kBN][kMinLd];    // weight min factors dmin x m_j (8 used)
  std::uint8_t Qh[kBN][F == 1 ? kQsLd : 16];  // Q5_K: the 32 high-bit bytes of the super-block
};
template <int F>
constexpr int kSmemT = kStages * static_cast<int>(sizeof(StageT<F>));
constexpr int kSmem = kSmemT<0>;
template <int F>
constexpr int kBlk = F == 1 ? 176 : 144;
template <int F>
constexpr int kQsOff = F == 1 ? 48 : 16;

struct Epi {
  float* C = nullptr;  // EPI 0: rows into C (N wide)
  const int* rtok = nullptr;
  const float* rw = nullptr;
  float* Yf = nullptr;
  int* Yi = nullptr;
  unsigned long long* Yl = nullptr;
  float scale = 1.f;
  unsigned* ovf = nullptr;
};

template <int EPI>
__device__ __forceinline__ void epi_store(int row, int col, float v, int N, const Epi& E) {
  if constexpr (EPI == 0) {
    E.C[static_cast<std::size_t>(row) * N + col] = v;
  } else {
    const int tok = E.rtok[row];
    if (tok < 0) return;
    const float x = v * E.rw[row];
    const std::size_t idx = static_cast<std::size_t>(tok) * N + col;
    if constexpr (EPI == 1) {
      atomicAdd(E.Yf + idx, x);
    } else if constexpr (EPI == 2) {
      const float f = x * E.scale;
      if (fabsf(f) > 268435455.f) atomicAdd(E.ovf, 1u);  // 2^31 / 8: eight addends could overflow
      atomicAdd(E.Yi + idx, __float2int_rn(f));
    } else {
      atomicAdd(E.Yl + idx, static_cast<unsigned long long>(__float2ll_rn(x * E.scale)));
    }
  }
}

// 64-row tiles, 3 stages, column tile in blockIdx.x. kDense: the row tile is blockIdx.y (rows `nrows_total`); else the
// unit is blockIdx.y of a device unit table (expert, first row, valid rows, 0) with a device count (extra blocks leave).
template <int EPI, bool kDense, int F = 0>
__global__ void __launch_bounds__(256, 2) k_tile(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const float* __restrict__ adig,
                                                const std::uint8_t* __restrict__ w, const uint4* __restrict__ rec, const float* __restrict__ wdig8,
                                                const int4* __restrict__ units, const int* __restrict__ nunits, int nrows_total, int N, int K, Epi E) {
  constexpr int WMW = 2, WNW = 4, NI = 4, WR = 32, MI = 2;
  extern __shared__ __align__(16) std::uint8_t p_smem[];
  using Stage = StageT<F>;
  Stage* stg = reinterpret_cast<Stage*>(p_smem);
  const int ublk = static_cast<int>(blockIdx.y), cblk = static_cast<int>(blockIdx.x);
  int ex = 0, bm, vrows;
  if constexpr (kDense) {
    bm = ublk * kTile;
    vrows = min(kTile, nrows_total - bm);
  } else {
    if (ublk >= *nunits) return;
    const int4 u = units[ublk];
    ex = u.x;
    bm = u.y;
    vrows = u.z;
  }
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp / WNW, wn = warp % WNW;
  const int bpr = K / kQK, n32 = K / 32;
  w += static_cast<std::size_t>(ex) * N * bpr * kBlk<F>;
  rec += static_cast<std::size_t>(ex) * N * bpr;
  wdig8 += static_cast<std::size_t>(ex) * N * bpr * 8;
  const int bn = cblk * kBN;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);
  auto load = [&](int stage, int k0) {
    Stage& T = stg[stage];
    {
      const int r = tid >> 2, c = (tid & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<std::size_t>(bm + r) * K + k0 + c);
    }
    const int sb = k0 / kQK;
    {
      const int r = tid >> 1, h = tid & 1;
      const std::uint8_t* blk = w + (static_cast<std::size_t>(bn + r) * bpr + sb) * kBlk<F>;
      cp_async16(&T.Qs[r][h * 16], blk + kQsOff<F> + ((k0 % kQK) / 64) * 32 + h * 16);
      if constexpr (F == 1) cp_async16(&T.Qh[r][h * 16], blk + 16 + h * 16);
    }
    if (tid < 128) {
      cp_async16(&T.Wr[tid], rec + static_cast<std::size_t>(bn + tid) * bpr + sb);
    } else if (tid < 128 + kTile) {
      const int r2 = tid - 128;
      cp_async8(&T.S2[r2][0], sa32 + static_cast<std::size_t>(bm + r2) * n32 + k0 / 32);
    }
    if (((k0 + 64) % kQK) == 0) {
      const int r = tid >> 1, h = tid & 1;
      if (r < kTile) cp_async16(&T.Am[r][h * 4], adig + (static_cast<std::size_t>(bm + r) * bpr + sb) * 8 + h * 4);
      cp_async16(&T.Bm[r][h * 4], wdig8 + (static_cast<std::size_t>(bn + r) * bpr + sb) * 8 + h * 4);
    }
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  constexpr float kMagic = 12582912.f;  // 1.5 x 2^23: Q4_K |is x c| <= 63 x 32 x 127 x 15 < 2^22, exact (Q5_K: < 2^23, I2F)
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nk = kt + kStages - 1;
      if (nk < kt_n) load(nk % kStages, nk * 64);
      cp_async_commit();
    }
    const Stage& T = stg[kt % kStages];
    const int k0 = kt * 64;
    const int j0 = (k0 % kQK) / 32;
    std::uint32_t bw[NI][2], hw[NI][2];
    float dc0[NI], dc1[NI];
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int c = wn * NI * 8 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
      if constexpr (F == 1) {
        hw[ni][0] = ld32(&T.Qh[c][4 * t]);
        hw[ni][1] = ld32(&T.Qh[c][16 + 4 * t]);
      }
      const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
      dc0[ni] = __uint_as_float(T.Wr[col0].x);
      dc1[ni] = __uint_as_float(T.Wr[col0 + 1].x);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      std::uint32_t b[NI][2];
      int is0[NI], is1[NI];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        if constexpr (F == 1) {  // the 5th bit of sub-block j0 + h
          b[ni][0] |= ((hw[ni][0] >> (j0 + h)) & 0x01010101u) << 4;
          b[ni][1] |= ((hw[ni][1] >> (j0 + h)) & 0x01010101u) << 4;
        }
        const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
        is0[ni] = reinterpret_cast<const std::uint8_t*>(&T.Wr[col0])[4 + j0 + h];
        is1[ni] = reinterpret_cast<const std::uint8_t*>(&T.Wr[col0 + 1])[4 + j0 + h];
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        std::uint32_t a[4];
        ldsm_x4(a, &T.As[wm * WR + mi * 16 + lm_row][h * 32 + lm_col]);
        const int r0 = wm * WR + mi * 16 + g;
        const float s0 = T.S2[r0][h], s1 = T.S2[r0 + 8][h];
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          float p0, p1, p2, p3;
          if constexpr (F == 1) {  // |is x c| <= 63 x 32 x 127 x 31 < 2^23: exact as a float
            p0 = static_cast<float>(is0[ni] * c4[0]);
            p1 = static_cast<float>(is1[ni] * c4[1]);
            p2 = static_cast<float>(is0[ni] * c4[2]);
            p3 = static_cast<float>(is1[ni] * c4[3]);
          } else {
            p0 = __int_as_float(is0[ni] * c4[0] + 0x4B400000) - kMagic;
            p1 = __int_as_float(is1[ni] * c4[1] + 0x4B400000) - kMagic;
            p2 = __int_as_float(is0[ni] * c4[2] + 0x4B400000) - kMagic;
            p3 = __int_as_float(is1[ni] * c4[3] + 0x4B400000) - kMagic;
          }
          acc[mi][ni][0] = fmaf(s0 * dc0[ni], p0, acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(s0 * dc1[ni], p1, acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(s1 * dc0[ni], p2, acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(s1 * dc1[ni], p3, acc[mi][ni][3]);
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {  // the min term of the super-block in fp32: sum_j (scale x sum)_j x (dmin x m_j), 8 FMAs per output
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r0 = wm * WR + mi * 16 + g, r1 = r0 + 8;
        const float4 a0l = *reinterpret_cast<const float4*>(&T.Am[r0][0]), a0h = *reinterpret_cast<const float4*>(&T.Am[r0][4]);
        const float4 a1l = *reinterpret_cast<const float4*>(&T.Am[r1][0]), a1h = *reinterpret_cast<const float4*>(&T.Am[r1][4]);
        const float a0[8] = {a0l.x, a0l.y, a0l.z, a0l.w, a0h.x, a0h.y, a0h.z, a0h.w};
        const float a1[8] = {a1l.x, a1l.y, a1l.z, a1l.w, a1h.x, a1h.y, a1h.z, a1h.w};
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
          const float4 b0l = *reinterpret_cast<const float4*>(&T.Bm[col0][0]), b0h = *reinterpret_cast<const float4*>(&T.Bm[col0][4]);
          const float4 b1l = *reinterpret_cast<const float4*>(&T.Bm[col0 + 1][0]), b1h = *reinterpret_cast<const float4*>(&T.Bm[col0 + 1][4]);
          const float b0[8] = {b0l.x, b0l.y, b0l.z, b0l.w, b0h.x, b0h.y, b0h.z, b0h.w};
          const float b1[8] = {b1l.x, b1l.y, b1l.z, b1l.w, b1h.x, b1h.y, b1h.z, b1h.w};
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[mi][ni][0] = fmaf(-a0[j], b0[j], acc[mi][ni][0]);
            acc[mi][ni][1] = fmaf(-a0[j], b1[j], acc[mi][ni][1]);
            acc[mi][ni][2] = fmaf(-a1[j], b0[j], acc[mi][ni][2]);
            acc[mi][ni][3] = fmaf(-a1[j], b1[j], acc[mi][ni][3]);
          }
        }
      }
    }
  }
  cp_async_wait<0>();
  (void)WMW;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int lr = wm * WR + mi * 16 + g, r0 = bm + lr;
      const int c0 = bn + wn * NI * 8 + ni * 8 + 2 * t;
      if (lr < vrows) {
        epi_store<EPI>(r0, c0, acc[mi][ni][0], N, E);
        epi_store<EPI>(r0, c0 + 1, acc[mi][ni][1], N, E);
      }
      if (lr + 8 < vrows) {
        epi_store<EPI>(r0 + 8, c0, acc[mi][ni][2], N, E);
        epi_store<EPI>(r0 + 8, c0 + 1, acc[mi][ni][3], N, E);
      }
    }
}

// the warp-vector path for experts with 1-8 rows: one warp per output column, lanes over the 32-wide sub-blocks
template <int EPI, int F = 0>
__global__ void __launch_bounds__(256) k_vec(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const std::uint8_t* __restrict__ w,
                                             const int4* __restrict__ units, const int* __restrict__ nunits, int N, int K, Epi E) {
  if (static_cast<int>(blockIdx.x) >= *nunits) return;
  const int4 u = units[blockIdx.x];
  const int lane = threadIdx.x & 31, n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const int bpr = K / kQK, n32 = K / 32, rows = u.z;
  const std::uint8_t* wr = w + (static_cast<std::size_t>(u.x) * N + n) * bpr * kBlk<F>;
  float acc[kVecMax];
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) acc[r] = 0.f;
  for (int jj = lane; jj < n32; jj += 32) {
    const std::uint8_t* bb = wr + static_cast<std::size_t>(jj >> 3) * kBlk<F>;
    const BlockQ4K& b = *reinterpret_cast<const BlockQ4K*>(bb);  // d, dmin, scales (Q4_K and Q5_K alike)
    const std::uint8_t* qs = bb + kQsOff<F>;
    const int j = jj & 7;
    int sc, m;
    scale_min_k4_d(j, b.scales, sc, m);
    const float dsc = h2f(b.d) * static_cast<float>(sc), dm = h2f(b.dmin) * static_cast<float>(m);
    const uint4 qa = *reinterpret_cast<const uint4*>(qs + (j >> 1) * 32), qb = *reinterpret_cast<const uint4*>(qs + (j >> 1) * 32 + 16);
    std::uint32_t qw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const int sh = 4 * (j & 1);
#pragma unroll
    for (int q = 0; q < 8; ++q) qw[q] = (qw[q] >> sh) & 0x0F0F0F0Fu;
    if constexpr (F == 1) {  // the 5th bit: qh byte l, bit j
      const uint4 ha = *reinterpret_cast<const uint4*>(bb + 16), hb = *reinterpret_cast<const uint4*>(bb + 32);
      const std::uint32_t hw[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
#pragma unroll
      for (int q = 0; q < 8; ++q) qw[q] |= ((hw[q] >> j) & 0x01010101u) << 4;
    }
#pragma unroll
    for (int r = 0; r < kVecMax; ++r) {
      if (r >= rows) break;
      const std::int8_t* xr = a8 + static_cast<std::size_t>(u.y + r) * K + jj * 32;
      const uint4 xa = *reinterpret_cast<const uint4*>(xr), xb = *reinterpret_cast<const uint4*>(xr + 16);
      const int xw[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y), static_cast<int>(xa.z), static_cast<int>(xa.w),
                         static_cast<int>(xb.x), static_cast<int>(xb.y), static_cast<int>(xb.z), static_cast<int>(xb.w)};
      int isum = 0, xsum = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        isum = __dp4a(static_cast<int>(qw[q]), xw[q], isum);
        xsum = __dp4a(xw[q], 0x01010101, xsum);
      }
      const float xs = sa32[static_cast<std::size_t>(u.y + r) * n32 + jj];
      acc[r] = fmaf(xs, dsc * static_cast<float>(isum) - dm * static_cast<float>(xsum), acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
  if (lane == 0)
    for (int r = 0; r < rows; ++r) epi_store<EPI>(u.y + r, n, acc[r], N, E);
}

// ------------------------------------------------------------------------------------------------ Q4_TG2 (locked 2026-10-30 23:22)
// Block (144 bytes per 256): fp16 d, six bytes of eight 6-bit levels, eight tag bytes (byte j: the 2-bit grid tag of each
// 8 values of sub-block j), 128 bytes of 4-bit codes (value i: byte i / 2, low nibble for even i). Weight = s_j x
// grid[tag][code], s_j = fl(fl(d x lv_j) / 63) (quant q8_z_scale). Tiles: each grid value as a 24-bit fixed-point number
// G = round(g x (2^23 - 1)) in three bytes (s8 high, u8 mid, u8 low), three int8 mmas per 32-block, combined in fp32
// (the high sum |.| < 2^20 and the others < 2^21: exact floats); 1-8 rows: the exact fp32 grid table and fp32 FMAs.
constexpr int kTgBytes = 144;
constexpr float kTgFix = 8388607.0f;

__device__ __forceinline__ std::uint32_t ld16(const void* p) { return *reinterpret_cast<const std::uint16_t*>(p); }
__device__ __forceinline__ void mma_s8u8(int (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int D>
__device__ __forceinline__ std::uint32_t pick4(std::uint32_t e0, std::uint32_t e1, std::uint32_t e2, std::uint32_t e3) {  // byte D of each
  constexpr unsigned sel = static_cast<unsigned>(D | ((D + 4) << 4));
  return __byte_perm(__byte_perm(e0, e1, sel), __byte_perm(e2, e3, sel), 0x5410);
}
__device__ __forceinline__ float i8f(std::uint32_t w, int b) {  // signed byte b of w, exact (magic number 2^23)
  return __uint_as_float(__byte_perm(w ^ 0x80808080u, 0x4B000000u, 0x7540u | static_cast<unsigned>(b))) - 8388736.0f;
}

__global__ void k_tg2_prep(const std::uint8_t* __restrict__ w, float* __restrict__ sc, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint8_t* b = w + i * kTgBytes;
    const float d = h2f(static_cast<std::uint16_t>(b[0] | (b[1] << 8)));
    std::uint64_t lv = 0;
#pragma unroll
    for (int k = 0; k < 6; ++k) lv |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
#pragma unroll
    for (int j = 0; j < 8; ++j) sc[i * 8 + j] = __fdiv_rn(__fmul_rn(d, static_cast<float>((lv >> (6 * j)) & 63u)), 63.0f);
  }
}

struct alignas(16) TgStage {
  std::int8_t As[kTile][kQ8Lda];
  std::uint8_t Qs[kBN][kQsLd];  // 32 code bytes of the 64-value step
  std::uint8_t Tg[kBN][8];      // the eight tag bytes of the super-block
  float Sc[kBN][2];             // s_j of the step's two sub-blocks
  float S2[kTile][2];           // activation scales of the step's two sub-blocks
};
constexpr int kTgSmem = kStages * static_cast<int>(sizeof(TgStage));

template <int EPI, bool kDense>
__global__ void __launch_bounds__(256, 2) k_tg2_tile(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const std::uint8_t* __restrict__ w,
                                                    const float* __restrict__ sc, const std::uint32_t* __restrict__ tables, const int4* __restrict__ units,
                                                    const int* __restrict__ nunits, int nrows_total, int N, int K, Epi E) {
  constexpr int NI = 4, WR = 32, MI = 2;
  extern __shared__ __align__(16) std::uint8_t p_smem[];
  __shared__ std::uint32_t lut[64];
  TgStage* stg = reinterpret_cast<TgStage*>(p_smem);
  const int tid = threadIdx.x;
  if (tid < 64) lut[tid] = tables[tid];
  const int ublk = static_cast<int>(blockIdx.y), cblk = static_cast<int>(blockIdx.x);
  int ex = 0, bm, vrows;
  if constexpr (kDense) {
    bm = ublk * kTile;
    vrows = min(kTile, nrows_total - bm);
  } else {
    if (ublk >= *nunits) return;
    const int4 u = units[ublk];
    ex = u.x;
    bm = u.y;
    vrows = u.z;
  }
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, wm = warp / 4, wn = warp % 4;
  const int bpr = K / kQK, n32 = K / 32, bn = cblk * kBN;
  const std::uint8_t* wb = w + static_cast<std::size_t>(ex) * N * bpr * kTgBytes;
  const float* scb = sc + static_cast<std::size_t>(ex) * N * bpr * 8;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);
  auto load = [&](int stage, int k0) {
    TgStage& T = stg[stage];
    const int sb = k0 / kQK, j0 = (k0 % kQK) / 32;
    {
      const int r = tid >> 2, c = (tid & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<std::size_t>(bm + r) * K + k0 + c);
    }
    {
      const int r = tid >> 1, h = tid & 1;
      cp_async16(&T.Qs[r][h * 16], wb + (static_cast<std::size_t>(bn + r) * bpr + sb) * kTgBytes + 16 + (k0 % kQK) / 2 + h * 16);
    }
    if (tid < kBN) {
      cp_async8(&T.Tg[tid][0], wb + (static_cast<std::size_t>(bn + tid) * bpr + sb) * kTgBytes + 8);
    } else {
      const int r = tid - kBN;
      cp_async8(&T.Sc[r][0], scb + (static_cast<std::size_t>(bn + r) * bpr + sb) * 8 + j0);
    }
    if (tid < kTile) cp_async8(&T.S2[tid][0], sa32 + static_cast<std::size_t>(bm + tid) * n32 + k0 / 32);
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  constexpr float inv = 1.0f / kTgFix;
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nk = kt + kStages - 1;
      if (nk < kt_n) load(nk % kStages, nk * 64);
      cp_async_commit();
    }
    const TgStage& T = stg[kt % kStages];
    const int j0 = ((kt * 64) % kQK) / 32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int j = j0 + h;
      std::uint32_t b[NI][3][2];  // [column group][digit: low, mid, high][k half]
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * NI * 8 + ni * 8 + g;
        const unsigned tb = T.Tg[c][j];
        const std::uint32_t w0 = ld16(&T.Qs[c][16 * h + 2 * t]), w1 = ld16(&T.Qs[c][16 * h + 8 + 2 * t]);
        const unsigned ta = (tb >> (2 * (t >> 1))) & 3u, tz = (tb >> (2 * (2 + (t >> 1)))) & 3u;
        std::uint32_t e[4], f[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          e[i] = lut[ta * 16 + ((w0 >> (4 * i)) & 15u)];
          f[i] = lut[tz * 16 + ((w1 >> (4 * i)) & 15u)];
        }
        b[ni][0][0] = pick4<0>(e[0], e[1], e[2], e[3]);
        b[ni][0][1] = pick4<0>(f[0], f[1], f[2], f[3]);
        b[ni][1][0] = pick4<1>(e[0], e[1], e[2], e[3]);
        b[ni][1][1] = pick4<1>(f[0], f[1], f[2], f[3]);
        b[ni][2][0] = pick4<2>(e[0], e[1], e[2], e[3]);
        b[ni][2][1] = pick4<2>(f[0], f[1], f[2], f[3]);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r0 = wm * WR + mi * 16 + g;
        const float xs0 = T.S2[r0][h], xs1 = T.S2[r0 + 8][h];
        std::uint32_t a[4];
        ldsm_x4(a, &T.As[wm * WR + mi * 16 + lm_row][h * 32 + lm_col]);
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
          const float s0 = T.Sc[col0][h], s1 = T.Sc[col0 + 1][h];
          int dh[4] = {0, 0, 0, 0}, dm[4] = {0, 0, 0, 0}, dl[4] = {0, 0, 0, 0};
          mma_s8(dh, a, b[ni][2]);
          mma_s8u8(dm, a, b[ni][1]);
          mma_s8u8(dl, a, b[ni][0]);
          float D[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) D[q] = fmaf(static_cast<float>(dh[q]), 65536.0f, fmaf(static_cast<float>(dm[q]), 256.0f, static_cast<float>(dl[q])));
          acc[mi][ni][0] = fmaf(xs0 * s0 * inv, D[0], acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(xs0 * s1 * inv, D[1], acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(xs1 * s0 * inv, D[2], acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(xs1 * s1 * inv, D[3], acc[mi][ni][3]);
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int lr = wm * WR + mi * 16 + g, r0 = bm + lr;
      const int c0 = bn + wn * NI * 8 + ni * 8 + 2 * t;
      if (lr < vrows) {
        epi_store<EPI>(r0, c0, acc[mi][ni][0], N, E);
        epi_store<EPI>(r0, c0 + 1, acc[mi][ni][1], N, E);
      }
      if (lr + 8 < vrows) {
        epi_store<EPI>(r0 + 8, c0, acc[mi][ni][2], N, E);
        epi_store<EPI>(r0 + 8, c0 + 1, acc[mi][ni][3], N, E);
      }
    }
}

// 1-8 rows, exact: one warp per output column, lanes over the 32-blocks; weights through the fp32 grid table once per
// 32-block, then 32 fp32 FMAs per row
template <int EPI>
__global__ void __launch_bounds__(256) k_tg2_vec(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const std::uint8_t* __restrict__ w,
                                                 const float* __restrict__ sc, const std::uint32_t* __restrict__ tables, const int4* __restrict__ units,
                                                 const int* __restrict__ nunits, int N, int K, Epi E) {
  __shared__ float lutf[64];
  if (threadIdx.x < 64) lutf[threadIdx.x] = __uint_as_float(tables[64 + threadIdx.x]);
  __syncthreads();
  if (static_cast<int>(blockIdx.x) >= *nunits) return;
  const int4 u = units[blockIdx.x];
  const int lane = threadIdx.x & 31, n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const int bpr = K / kQK, n32 = K / 32, rows = u.z;
  const std::size_t wrow = static_cast<std::size_t>(u.x) * N + n;
  const std::uint8_t* wr = w + wrow * bpr * kTgBytes;
  float acc[kVecMax];
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) acc[r] = 0.f;
  for (int jj = lane; jj < n32; jj += 32) {
    const std::uint8_t* b = wr + static_cast<std::size_t>(jj >> 3) * kTgBytes;
    const int j = jj & 7;
    const float s = sc[(wrow * bpr + static_cast<std::size_t>(jj >> 3)) * 8 + static_cast<std::size_t>(j)];
    const unsigned tb = b[8 + j];
    const uint4 cq = *reinterpret_cast<const uint4*>(b + 16 + j * 16);
    const std::uint32_t cw[4] = {cq.x, cq.y, cq.z, cq.w};
    float wv[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) wv[i] = lutf[((tb >> (2 * (i >> 3))) & 3u) * 16 + ((cw[i >> 3] >> (4 * (i & 7))) & 15u)];
#pragma unroll
    for (int r = 0; r < kVecMax; ++r) {
      if (r >= rows) break;
      const std::int8_t* xr = a8 + static_cast<std::size_t>(u.y + r) * K + jj * 32;
      const uint4 xa = *reinterpret_cast<const uint4*>(xr), xb = *reinterpret_cast<const uint4*>(xr + 16);
      const std::uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) sum = fmaf(wv[i], i8f(xw[i >> 2], i & 3), sum);
      acc[r] = fmaf(sa32[static_cast<std::size_t>(u.y + r) * n32 + jj] * s, sum, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
  if (lane == 0)
    for (int r = 0; r < rows; ++r) epi_store<EPI>(u.y + r, n, acc[r], N, E);
}

// ------------------------------------------------------------------------------------------------ Q6_K (2026-10-30 23:5x)
// Block (210 bytes per 256; ggml block_q6_K): ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] (int8, one per 16
// values), d (fp16). Sub-block j (32 values): half n = j / 4, jj = j % 4; value l: low = (ql[n * 64 + (jj & 1) * 32 + l]
// >> (4 * (jj >> 1))) & 15, high = (qh[n * 32 + l] >> (2 * jj)) & 3, q = (low | high << 4) - 32 in -32..31 (exact int8,
// no min term); weight = d x scales[2 j + (l >= 16)] x q. Tiles: two int8 m16n8k16 mmas per 32-block (one per scale),
// combined in int32 (|.| < 2^24: exact as a float). 210-byte blocks are not 16-byte aligned: each step loads the 16-byte
// aligned windows around the 64 ql and 32 qh bytes it needs (ends inside the block; the tensor start must be 16-byte
// aligned) and keeps the window offsets. Side table: per block 16 int8 scales + fp32 d (32 bytes).
constexpr int kQ6Bytes = 210;
__device__ __forceinline__ void mma_s8_k16(int (&d)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ std::uint32_t ld32_2(const std::uint8_t* p) {  // 4 bytes at a 2-byte aligned address
  return static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(p)) | (static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(p + 2)) << 16);
}
// 4 values of sub-block j at l .. l + 3 (ql / qh pointers already at the half's first byte), as int8 q - 32
__device__ __forceinline__ std::uint32_t q6_word(const std::uint8_t* ql, const std::uint8_t* qh, int j, int l) {
  const int jj = j & 3;
  const std::uint32_t lo = (ld32_2(ql + (jj & 1) * 32 + l) >> (4 * (jj >> 1))) & 0x0F0F0F0Fu;
  const std::uint32_t hi = (ld32_2(qh + l) >> (2 * jj)) & 0x03030303u;
  return __vsub4(lo | (hi << 4), 0x20202020u);
}

__global__ void k_q6_prep(const std::uint8_t* __restrict__ w, std::uint8_t* __restrict__ rec, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint8_t* b = w + i * kQ6Bytes;
    std::uint8_t* r = rec + i * 32;
#pragma unroll
    for (int k = 0; k < 16; ++k) r[k] = b[192 + k];
    const float d = h2f(static_cast<std::uint16_t>(b[208] | (b[209] << 8)));
    *reinterpret_cast<float*>(r + 16) = d;
    *reinterpret_cast<float*>(r + 20) = 0.f;
    *reinterpret_cast<uint2*>(r + 24) = make_uint2(0u, 0u);
  }
}

struct alignas(16) Q6Stage {
  std::int8_t As[kTile][kQ8Lda];
  std::uint8_t Ql[kBN][80];  // aligned window around the 64 ql bytes of the step's 128-half
  std::uint8_t Qh[kBN][48];  // aligned window around its 32 qh bytes
  std::int8_t Sc[kBN][16];   // the block's 16 scales
  float Dd[kBN][2];          // d (and padding)
  float S2[kTile][2];
  std::uint8_t Off[kBN][2];  // window offsets (ql, qh)
};
constexpr int kQ6Smem = kStages * static_cast<int>(sizeof(Q6Stage));

template <int EPI, bool kDense>
__global__ void __launch_bounds__(256, 2) k_q6_tile(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const std::uint8_t* __restrict__ w,
                                                   const std::uint8_t* __restrict__ rec, const int4* __restrict__ units, const int* __restrict__ nunits,
                                                   int nrows_total, int N, int K, Epi E) {
  constexpr int NI = 4, WR = 32, MI = 2;
  extern __shared__ __align__(16) std::uint8_t p_smem[];
  Q6Stage* stg = reinterpret_cast<Q6Stage*>(p_smem);
  const int ublk = static_cast<int>(blockIdx.y), cblk = static_cast<int>(blockIdx.x);
  int ex = 0, bm, vrows;
  if constexpr (kDense) {
    bm = ublk * kTile;
    vrows = min(kTile, nrows_total - bm);
  } else {
    if (ublk >= *nunits) return;
    const int4 u = units[ublk];
    ex = u.x;
    bm = u.y;
    vrows = u.z;
  }
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, wm = warp / 4, wn = warp % 4;
  const int bpr = K / kQK, n32 = K / 32, bn = cblk * kBN;
  const std::size_t base_blk = static_cast<std::size_t>(ex) * N * bpr;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);
  auto load = [&](int stage, int k0) {
    Q6Stage& T = stg[stage];
    const int sb = k0 / kQK, n = (k0 % kQK) / 128;
    {
      const int r = tid >> 2, c = (tid & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<std::size_t>(bm + r) * K + k0 + c);
    }
    {
      const int c = tid >> 1, half = tid & 1;
      const std::size_t bi = base_blk + static_cast<std::size_t>(bn + c) * bpr + sb;
      const std::uint8_t* B = w + bi * kQ6Bytes;
      const std::uint8_t* a1 = B + n * 64;
      const std::uint8_t* a2 = B + 128 + n * 32;
      const std::uint8_t* f1 = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(a1) & ~static_cast<std::uintptr_t>(15));
      const std::uint8_t* f2 = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(a2) & ~static_cast<std::uintptr_t>(15));
      if (half == 0) {
#pragma unroll
        for (int q = 0; q < 4; ++q) cp_async16(&T.Ql[c][q * 16], f1 + q * 16);
        T.Off[c][0] = static_cast<std::uint8_t>(a1 - f1);
        cp_async16(&T.Sc[c][0], rec + bi * 32);
      } else {
        cp_async16(&T.Ql[c][64], f1 + 64);
#pragma unroll
        for (int q = 0; q < 3; ++q) cp_async16(&T.Qh[c][q * 16], f2 + q * 16);
        T.Off[c][1] = static_cast<std::uint8_t>(a2 - f2);
        cp_async8(&T.Dd[c][0], rec + bi * 32 + 16);
      }
    }
    if (tid < kTile) cp_async8(&T.S2[tid][0], sa32 + static_cast<std::size_t>(bm + tid) * n32 + k0 / 32);
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nk = kt + kStages - 1;
      if (nk < kt_n) load(nk % kStages, nk * 64);
      cp_async_commit();
    }
    const Q6Stage& T = stg[kt % kStages];
    const int j0 = ((kt * 64) % kQK) / 32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int j = j0 + h;
      std::uint32_t b[NI][2];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * NI * 8 + ni * 8 + g;
        const std::uint8_t* ql = &T.Ql[c][T.Off[c][0]];
        const std::uint8_t* qh = &T.Qh[c][T.Off[c][1]];
        b[ni][0] = q6_word(ql, qh, j, 4 * t);
        b[ni][1] = q6_word(ql, qh, j, 16 + 4 * t);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r0 = wm * WR + mi * 16 + g;
        const float xs0 = T.S2[r0][h], xs1 = T.S2[r0 + 8][h];
        std::uint32_t a[4];
        ldsm_x4(a, &T.As[wm * WR + mi * 16 + lm_row][h * 32 + lm_col]);
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
          int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
          mma_s8_k16(c0, a[0], a[1], b[ni][0]);  // values 0-15 of the sub-block
          mma_s8_k16(c1, a[2], a[3], b[ni][1]);  // values 16-31
          const int sl0 = T.Sc[col0][2 * j], sh0 = T.Sc[col0][2 * j + 1], sl1 = T.Sc[col0 + 1][2 * j], sh1 = T.Sc[col0 + 1][2 * j + 1];
          const float d0 = T.Dd[col0][0], d1 = T.Dd[col0 + 1][0];
          acc[mi][ni][0] = fmaf(xs0 * d0, static_cast<float>(sl0 * c0[0] + sh0 * c1[0]), acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(xs0 * d1, static_cast<float>(sl1 * c0[1] + sh1 * c1[1]), acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(xs1 * d0, static_cast<float>(sl0 * c0[2] + sh0 * c1[2]), acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(xs1 * d1, static_cast<float>(sl1 * c0[3] + sh1 * c1[3]), acc[mi][ni][3]);
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int lr = wm * WR + mi * 16 + g, r0 = bm + lr;
      const int c0 = bn + wn * NI * 8 + ni * 8 + 2 * t;
      if (lr < vrows) {
        epi_store<EPI>(r0, c0, acc[mi][ni][0], N, E);
        epi_store<EPI>(r0, c0 + 1, acc[mi][ni][1], N, E);
      }
      if (lr + 8 < vrows) {
        epi_store<EPI>(r0 + 8, c0, acc[mi][ni][2], N, E);
        epi_store<EPI>(r0 + 8, c0 + 1, acc[mi][ni][3], N, E);
      }
    }
}

// 1-8 rows: one warp per output column, lanes over the 32-blocks, dp4a per 16-value half
template <int EPI>
__global__ void __launch_bounds__(256) k_q6_vec(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const std::uint8_t* __restrict__ w,
                                                const std::uint8_t* __restrict__ rec, const int4* __restrict__ units, const int* __restrict__ nunits, int N,
                                                int K, Epi E) {
  if (static_cast<int>(blockIdx.x) >= *nunits) return;
  const int4 u = units[blockIdx.x];
  const int lane = threadIdx.x & 31, n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const int bpr = K / kQK, n32 = K / 32, rows = u.z;
  const std::size_t b0 = (static_cast<std::size_t>(u.x) * N + n) * bpr;
  float acc[kVecMax];
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) acc[r] = 0.f;
  for (int jj = lane; jj < n32; jj += 32) {
    const std::size_t bi = b0 + static_cast<std::size_t>(jj >> 3);
    const std::uint8_t* B = w + bi * kQ6Bytes;
    const int j = jj & 7, hn = j >> 2;
    const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(rec + bi * 32);
    const float d = *reinterpret_cast<const float*>(rec + bi * 32 + 16);
    const int slo = sc[2 * j], shi = sc[2 * j + 1];
    std::uint32_t qw[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) qw[q] = q6_word(B + hn * 64, B + 128 + hn * 32, j, 4 * q);
#pragma unroll
    for (int r = 0; r < kVecMax; ++r) {
      if (r >= rows) break;
      const std::int8_t* xr = a8 + static_cast<std::size_t>(u.y + r) * K + jj * 32;
      const uint4 xa = *reinterpret_cast<const uint4*>(xr), xb = *reinterpret_cast<const uint4*>(xr + 16);
      const int xw[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y), static_cast<int>(xa.z), static_cast<int>(xa.w),
                         static_cast<int>(xb.x), static_cast<int>(xb.y), static_cast<int>(xb.z), static_cast<int>(xb.w)};
      int lo = 0, hi = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) lo = __dp4a(static_cast<int>(qw[q]), xw[q], lo);
#pragma unroll
      for (int q = 4; q < 8; ++q) hi = __dp4a(static_cast<int>(qw[q]), xw[q], hi);
      acc[r] = fmaf(sa32[static_cast<std::size_t>(u.y + r) * n32 + jj] * d, static_cast<float>(slo * lo + shi * hi), acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
  if (lane == 0)
    for (int r = 0; r < rows; ++r) epi_store<EPI>(u.y + r, n, acc[r], N, E);
}

// ------------------------------------------------------------------------------------------------ the work list (S3)

__global__ void k_route_count(const std::int32_t* __restrict__ sel, int n, int* __restrict__ cnt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned live = __ballot_sync(0xffffffffu, i < n);
  if (i >= n) return;
  const int e = sel[i];
  const unsigned grp = __match_any_sync(live, e);
  if (static_cast<int>(threadIdx.x & 31) == __ffs(static_cast<int>(grp)) - 1) atomicAdd(cnt + e, __popc(grp));
}
__device__ __forceinline__ int warp_incl_scan(int v, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += x;
  }
  return v;
}
// one warp: lane l owns experts 8l..8l+7 (n_expert <= 256); meta = tile units, vector units, active experts, rows
__global__ void k_route_build(const int* __restrict__ cnt, int n_expert, int* __restrict__ off, int* __restrict__ cursor, int4* __restrict__ ut,
                              int4* __restrict__ uv, int* __restrict__ meta) {
  const int lane = threadIdx.x & 31;
  int c[8], rows = 0, nt = 0, nv = 0;
  unsigned act = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = lane * 8 + i;
    c[i] = e < n_expert ? cnt[e] : 0;
    rows += c[i];
    if (c[i]) act |= 1u << i;
    if (c[i] > kVecMax) nt += (c[i] >> 6) + ((c[i] & 63) ? 1 : 0);
    else if (c[i]) ++nv;
  }
  const int r_in = warp_incl_scan(rows, lane), t_in = warp_incl_scan(nt, lane), v_in = warp_incl_scan(nv, lane);
  const int a_in = warp_incl_scan(__popc(act), lane);
  int r0 = r_in - rows, t0 = t_in - nt, v0 = v_in - nv;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = lane * 8 + i;
    if (e < n_expert) {
      off[e] = r0;
      cursor[e] = r0;
    }
    if (c[i] > kVecMax) {
      for (int m = 0; m < c[i]; m += kTile) ut[t0++] = make_int4(e, r0 + m, min(kTile, c[i] - m), 0);
    } else if (c[i]) {
      uv[v0++] = make_int4(e, r0, c[i], 0);
    }
    r0 += c[i];
  }
  if (lane == 31) {
    meta[0] = t_in;
    meta[1] = v_in;
    meta[2] = a_in;
    meta[3] = r_in;
    off[n_expert] = r_in;  // the end of the last expert (gpu_ops' moe_matmul reads off[e + 1])
  }
}
__global__ void k_route_place(const std::int32_t* __restrict__ sel, const float* __restrict__ wt, int n, int n_used, int* __restrict__ cursor,
                              int* __restrict__ src, int* __restrict__ rtok, float* __restrict__ rw, int* __restrict__ prow) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int e = sel[i], p = atomicAdd(cursor + e, 1), t = i / n_used;
  src[p] = t;
  rtok[p] = t;
  rw[p] = wt[i];
  prow[i] = p;
}
// packed row p = token src[p]: K int8, K / 32 scales, K / 32 min factors
__global__ void k_gather(const int* __restrict__ src, int P, int K, const std::int8_t* __restrict__ q, const float* __restrict__ sc,
                         const float* __restrict__ dg, std::int8_t* __restrict__ oq, float* __restrict__ osc, float* __restrict__ odg) {
  const int p = blockIdx.x;
  if (p >= P) return;
  const int r = src[p];
  for (int i = threadIdx.x; i < K / 16; i += blockDim.x)
    reinterpret_cast<uint4*>(oq + static_cast<std::size_t>(p) * K)[i] = reinterpret_cast<const uint4*>(q + static_cast<std::size_t>(r) * K)[i];
  for (int i = threadIdx.x; i < K / 32; i += blockDim.x) osc[static_cast<std::size_t>(p) * (K / 32) + i] = sc[static_cast<std::size_t>(r) * (K / 32) + i];
  for (int i = threadIdx.x; i < K / 32; i += blockDim.x) odg[static_cast<std::size_t>(p) * (K / 32) + i] = dg[static_cast<std::size_t>(r) * (K / 32) + i];
}
// silu(gate) x up, rounded per 32 exactly as act_q8x32 rounds (one thread per 32-block), for the down projection
__global__ void k_silu_q8x32(const float* __restrict__ hg, const float* __restrict__ hu, std::int64_t rows, std::int64_t F, std::int8_t* __restrict__ q,
                             float* __restrict__ scale, int* __restrict__ sum) {
  const std::int64_t nb = F / 32;
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= rows * nb) return;
  const std::int64_t t = id / nb, b = id % nb;
  float v[32];
  for (int i = 0; i < 32; ++i) v[i] = mul(silu_p(hg[t * F + 32 * b + i]), hu[t * F + 32 * b + i]);
  float a = 0.0f;
  for (int j = 0; j < 32; ++j) a = fmaxf(a, fabsf(v[j]));
  const float sc = a > 0.0f ? quo(a, 127.0f) : 1.0f;
  const float inv = quo(1.0f, sc);
  int s = 0;
  for (int j = 0; j < 32; ++j) {
    const float qv = fminf(127.0f, fmaxf(-127.0f, rintf(mul(v[j], inv))));
    q[t * F + 32 * b + j] = static_cast<std::int8_t>(qv);
    s += static_cast<int>(qv);
  }
  scale[t * nb + b] = sc;
  sum[t * nb + b] = s;
}
__global__ void k_fix(const int* __restrict__ yi, const unsigned long long* __restrict__ yl, std::size_t n, float inv, float* __restrict__ y) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    y[i] = yi ? static_cast<float>(yi[i]) * inv : static_cast<float>(static_cast<double>(static_cast<long long>(yl[i])) * static_cast<double>(inv));
}

// exact paths: rows of q x scale as floats; the identity row map; the down rows folded into y by the combine epilogue
__global__ void k_deq_rows(const std::int8_t* __restrict__ q, const float* __restrict__ sc, std::size_t n, float* __restrict__ x) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    x[i] = static_cast<float>(q[i]) * sc[i / 32];
}
__global__ void k_iota(int* __restrict__ v, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v[i] = i;
}
template <int EPI>
__global__ void k_fold(const float* __restrict__ yd, const int* __restrict__ nrows, int N, Epi E) {
  const std::size_t n = static_cast<std::size_t>(*nrows) * static_cast<std::size_t>(N);
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    epi_store<EPI>(static_cast<int>(i / static_cast<std::size_t>(N)), static_cast<int>(i % static_cast<std::size_t>(N)), yd[i], N, E);
}

unsigned grid_of(std::size_t n, int per, unsigned cap = 65535) {
  const std::size_t b = (n + static_cast<std::size_t>(per) - 1) / static_cast<std::size_t>(per);
  return static_cast<unsigned>(b < 1 ? 1 : (b > cap ? cap : b));
}

template <int EPI, bool kDense>
bool tile_launch(dim3 grid, const std::int8_t* a8, const float* sa32, const float* adig, const PromptWeights& W, const int4* units, const int* nunits,
                 int nrows, int N, int K, const Epi& E, Stream s) {
  if (W.type == static_cast<std::uint32_t>(quant::Type::q4_tg2)) {
    static const cudaError_t once_tg = cudaFuncSetAttribute(k_tg2_tile<EPI, kDense>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTgSmem);
    if (!pck(once_tg, "prompt tile (Q4_TG2): shared memory opt-in")) return false;
    k_tg2_tile<EPI, kDense><<<grid, 256, kTgSmem, pst(s)>>>(a8, sa32, static_cast<const std::uint8_t*>(W.w), static_cast<const float*>(W.rec),
                                                            static_cast<const std::uint32_t*>(W.dig), units, nunits, nrows, N, K, E);
    return pck(cudaGetLastError(), "prompt tile (Q4_TG2)");
  }
  if (W.type == static_cast<std::uint32_t>(quant::Type::q6_k)) {
    static const cudaError_t once6 = cudaFuncSetAttribute(k_q6_tile<EPI, kDense>, cudaFuncAttributeMaxDynamicSharedMemorySize, kQ6Smem);
    if (!pck(once6, "prompt tile (Q6_K): shared memory opt-in")) return false;
    k_q6_tile<EPI, kDense><<<grid, 256, kQ6Smem, pst(s)>>>(a8, sa32, static_cast<const std::uint8_t*>(W.w), static_cast<const std::uint8_t*>(W.rec), units,
                                                           nunits, nrows, N, K, E);
    return pck(cudaGetLastError(), "prompt tile (Q6_K)");
  }
  if (W.type == static_cast<std::uint32_t>(quant::Type::q5_k)) {
    static const cudaError_t once5 = cudaFuncSetAttribute(k_tile<EPI, kDense, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemT<1>);
    if (!pck(once5, "prompt tile (Q5_K): shared memory opt-in")) return false;
    k_tile<EPI, kDense, 1><<<grid, 256, kSmemT<1>, pst(s)>>>(a8, sa32, adig, static_cast<const std::uint8_t*>(W.w), static_cast<const uint4*>(W.rec),
                                                              static_cast<const float*>(W.dig), units, nunits, nrows, N, K, E);
    return pck(cudaGetLastError(), "prompt tile (Q5_K)");
  }
  static const cudaError_t once = cudaFuncSetAttribute(k_tile<EPI, kDense>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (!pck(once, "prompt tile: shared memory opt-in")) return false;
  k_tile<EPI, kDense><<<grid, 256, kSmem, pst(s)>>>(a8, sa32, adig, static_cast<const std::uint8_t*>(W.w), static_cast<const uint4*>(W.rec),
                                                    static_cast<const float*>(W.dig), units, nunits, nrows, N, K, E);
  return pck(cudaGetLastError(), "prompt tile");
}
template <int EPI>
bool vec_launch(dim3 grid, const std::int8_t* a8, const float* sa32, const PromptWeights& W, const int4* units, const int* nunits, int N, int K, const Epi& E,
                Stream s) {
  if (W.type == static_cast<std::uint32_t>(quant::Type::q4_tg2)) {
    k_tg2_vec<EPI><<<grid, 256, 0, pst(s)>>>(a8, sa32, static_cast<const std::uint8_t*>(W.w), static_cast<const float*>(W.rec),
                                             static_cast<const std::uint32_t*>(W.dig), units, nunits, N, K, E);
    return pck(cudaGetLastError(), "prompt vector (Q4_TG2)");
  }
  if (W.type == static_cast<std::uint32_t>(quant::Type::q6_k)) {
    k_q6_vec<EPI><<<grid, 256, 0, pst(s)>>>(a8, sa32, static_cast<const std::uint8_t*>(W.w), static_cast<const std::uint8_t*>(W.rec), units, nunits, N, K, E);
    return pck(cudaGetLastError(), "prompt vector (Q6_K)");
  }
  if (W.type == static_cast<std::uint32_t>(quant::Type::q5_k)) {
    k_vec<EPI, 1><<<grid, 256, 0, pst(s)>>>(a8, sa32, static_cast<const std::uint8_t*>(W.w), units, nunits, N, K, E);
    return pck(cudaGetLastError(), "prompt vector (Q5_K)");
  }
  k_vec<EPI><<<grid, 256, 0, pst(s)>>>(a8, sa32, static_cast<const std::uint8_t*>(W.w), units, nunits, N, K, E);
  return pck(cudaGetLastError(), "prompt vector");
}
// one stack's copy of the work list: the units of its experts (emap[e] >= 0), expert index rewritten to its matrix index,
// in list order (one thread: at most a few hundred units)
__global__ void k_units_remap(const int4* __restrict__ ut, const int4* __restrict__ uv, const int* __restrict__ meta, const int* __restrict__ emap,
                              int4* __restrict__ ut2, int4* __restrict__ uv2, int* __restrict__ meta2) {
  int nt = 0, nv = 0;
  for (int i = 0; i < meta[0]; ++i) {
    const int4 u = ut[i];
    const int x = emap[u.x];
    if (x >= 0) ut2[nt++] = make_int4(x, u.y, u.z, u.w);
  }
  for (int i = 0; i < meta[1]; ++i) {
    const int4 u = uv[i];
    const int x = emap[u.x];
    if (x >= 0) uv2[nv++] = make_int4(x, u.y, u.z, u.w);
  }
  meta2[0] = nt;
  meta2[1] = nv;
}
// expert GEMM over the device lists (tiles + vectors) with epilogue EPI; a stack with an emap runs over its own copy
template <int EPI>
bool experts(const MoeScratch& m, const std::int8_t* a8, const float* sa32, const float* adig, const PromptWeights& W, int N, int K, const Epi& E,
             std::int64_t picks, Stream s) {
  const int max_tiles = static_cast<int>(picks / kTile) + m.n_expert;
  const int4* ut = static_cast<const int4*>(m.units_t);
  const int4* uv = static_cast<const int4*>(m.units_v);
  const int* nt = m.meta;
  const int* nv = m.meta + 1;
  if (W.emap) {
    k_units_remap<<<1, 1, 0, pst(s)>>>(ut, uv, m.meta, W.emap, static_cast<int4*>(m.units_t2), static_cast<int4*>(m.units_v2), m.meta2);
    if (!pck(cudaGetLastError(), "moe_prompt: stack units")) return false;
    ut = static_cast<const int4*>(m.units_t2);
    uv = static_cast<const int4*>(m.units_v2);
    nt = m.meta2;
    nv = m.meta2 + 1;
  }
  // v22 (the v21 per-operation timeline: the bank's experts took 0.54 ms per layer at one token): the grids were sized for
  // every expert (vector: n_expert blocks across; tiles: picks / 64 + n_expert), almost all exiting at once. An expert has
  // at most `tokens` rows (a token picks an expert once), so with tokens <= kVecMax there are no tile units at all, and
  // the vector units are at most min(n_expert, picks). Same units, same kernels, same values; fewer empty blocks.
  const std::int64_t toks = m.n_used > 0 ? picks / m.n_used : picks;
  const int vec_x = static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(m.n_expert, picks)));
  const int tile_y = static_cast<int>(std::min<std::int64_t>(max_tiles, picks / kTile + std::min<std::int64_t>(m.n_expert, picks / (kVecMax + 1) + 1)));
  if (toks > kVecMax && !tile_launch<EPI, false>(dim3(static_cast<unsigned>(N / kBN), static_cast<unsigned>(tile_y)), a8, sa32, adig, W, ut, nt, 0, N, K, E, s))
    return false;
  return vec_launch<EPI>(dim3(static_cast<unsigned>(vec_x), static_cast<unsigned>((N + 7) / 8)), a8, sa32, W, uv, nv, N, K, E, s);
}

// one expert stack over the GPU work list: the int8 kernels for Q4_K / Q5_K / Q6_K / Q4_TG2, else the exact path (gpu_ops'
// moe_matmul on float rows, same work list: slots off[e] .. off[e + 1], rows read through `srcmap`, written in packed
// order; EPI 0 writes E.C directly, the folded epilogues go through m.yd and k_fold)
bool int8_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::q4_k) || t == static_cast<std::uint32_t>(quant::Type::q5_k) ||
         t == static_cast<std::uint32_t>(quant::Type::q6_k) || t == static_cast<std::uint32_t>(quant::Type::q4_tg2);
}
template <int EPI>
bool run_stack(const MoeScratch& m, const PromptWeights& W, const std::int8_t* a8, const float* sa32, const float* adig, const float* xf, const int* srcmap,
               int N, int K, const Epi& E, std::int64_t picks, Stream s) {
  if (int8_type(W.type)) return experts<EPI>(m, a8, sa32, adig, W, N, K, E, picks, s);
  if (!m.exact) return pfail("moe_prompt: a stack without an int8 step needs moe_scratch_alloc(..., exact = true)");
  if (W.emap) return pfail("moe_prompt: stacks with an expert map must have an int8 type");
  const std::uint64_t rb = static_cast<std::uint64_t>(K / static_cast<int>(block_values(W.type))) * block_bytes(W.type);
  float* out = EPI == 0 ? E.C : m.yd;
  if (!moe_matmul(W.type, W.w, rb * static_cast<std::uint64_t>(N), rb, N, K, m.n_expert, m.off, srcmap, m.iota, xf, K, out, N, static_cast<int>(picks), s))
    return pfail(std::string("moe_prompt: exact path: ") + last_error());
  if constexpr (EPI != 0) {
    k_fold<EPI><<<grid_of(static_cast<std::size_t>(picks) * static_cast<std::size_t>(N), 256, 4096), 256, 0, pst(s)>>>(m.yd, m.meta + 3, N, E);
    return pck(cudaGetLastError(), "moe_prompt: exact fold");
  }
  return true;
}

std::size_t al(std::size_t x) { return (x + 255) / 256 * 256; }

}  // namespace

const char* prompt_last_error() { return p_err.c_str(); }

bool prompt_type_ok(std::uint32_t type, std::int64_t k) {
  return (type == static_cast<std::uint32_t>(quant::Type::q4_k) || type == static_cast<std::uint32_t>(quant::Type::q5_k) ||
          type == static_cast<std::uint32_t>(quant::Type::q6_k) || type == static_cast<std::uint32_t>(quant::Type::q4_tg2)) &&
         k > 0 && k % kQK == 0;
}

std::size_t prompt_table_bytes(std::uint32_t type, std::int64_t rows, std::int64_t k) {
  if (!prompt_type_ok(type, k) || rows <= 0) return 0;
  const std::size_t nb = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k / kQK);
  if (type == static_cast<std::uint32_t>(quant::Type::q6_k)) return al(nb * 32);
  if (type == static_cast<std::uint32_t>(quant::Type::q4_tg2)) return al(nb * 32) + al(128 * 4);
  return al(nb * 16) + al(nb * 32);  // Q4_K, Q5_K
}

bool prompt_prepare_at(std::uint32_t type, const void* w_dev, std::int64_t rows, std::int64_t k, void* mem, PromptWeights* out, Stream s) {
  if (!out || !w_dev || rows <= 0 || rows % kBN != 0) return pfail("prompt_prepare: bad arguments (rows a multiple of 128)");
  *out = PromptWeights{};
  if (!prompt_type_ok(type, k)) {  // no int8 step: the exact path (moe_prompt) if the GPU decodes the type
    if (!type_supported(type) || k <= 0 || k % 32 != 0 || k % static_cast<std::int64_t>(block_values(type)) != 0)
      return pfail("prompt_prepare: type " + std::to_string(type) + " has neither an int8 prompt step nor a GPU decoder for rows of " + std::to_string(k));
    out->type = type;
    out->rows = rows;
    out->k = k;
    out->w = w_dev;
    return true;
  }
  if (!mem || (reinterpret_cast<std::uintptr_t>(mem) & 255u) != 0) return pfail("prompt_prepare: the side-table memory must be given, 256-byte aligned");
  const std::size_t nb = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k / kQK);
  auto* m8 = static_cast<std::uint8_t*>(mem);
  out->type = type;
  out->rows = rows;
  out->k = k;
  out->w = w_dev;
  if (type == static_cast<std::uint32_t>(quant::Type::q6_k)) {  // rec: 16 int8 scales + fp32 d per 256; no min term
    if ((reinterpret_cast<std::uintptr_t>(w_dev) & 15u) != 0) return pfail("prompt_prepare: Q6_K data must start 16-byte aligned");
    k_q6_prep<<<grid_of(nb, 256, 4096), 256, 0, pst(s)>>>(static_cast<const std::uint8_t*>(w_dev), m8, nb);
    out->rec = m8;
    return pck(cudaGetLastError(), "prompt_prepare (Q6_K)");
  }
  if (type == static_cast<std::uint32_t>(quant::Type::q4_tg2)) {  // rec: s_j per 32 (fp32); dig: 64 digit words + 64 fp32 grid values
    std::uint32_t host[128];
    for (int t = 0; t < 4; ++t)
      for (int c = 0; c < 16; ++c) {
        const float gv = quant::q4_tg_grid(t, 2)[c];
        const long long G = std::llround(static_cast<double>(gv) * static_cast<double>(kTgFix));
        host[t * 16 + c] = static_cast<std::uint32_t>(G) & 0xFFFFFFu;  // bytes low, mid, high (two's complement)
        std::memcpy(&host[64 + t * 16 + c], &gv, 4);
      }
    out->rec = m8;
    out->dig = m8 + al(nb * 32);
    k_tg2_prep<<<grid_of(nb, 256, 4096), 256, 0, pst(s)>>>(static_cast<const std::uint8_t*>(w_dev), static_cast<float*>(out->rec), nb);
    if (!pck(cudaGetLastError(), "prompt_prepare (Q4_TG2)") || !to_device(out->dig, host, sizeof host, s)) return pfail("prompt_prepare (Q4_TG2): side tables");
    return true;
  }
  out->rec = m8;
  out->dig = m8 + al(nb * 16);
  const int bstride = type == static_cast<std::uint32_t>(quant::Type::q5_k) ? kBlk<1> : kBlk<0>;
  k_prep_rec<<<grid_of(nb, 256, 4096), 256, 0, pst(s)>>>(static_cast<const std::uint8_t*>(w_dev), bstride, static_cast<uint4*>(out->rec), nb);
  k_prep_dig<<<grid_of(nb, 256, 4096), 256, 0, pst(s)>>>(static_cast<const std::uint8_t*>(w_dev), bstride, static_cast<float*>(out->dig), nb);
  return pck(cudaGetLastError(), "prompt_prepare");
}

bool prompt_prepare(std::uint32_t type, const void* w_dev, std::int64_t rows, std::int64_t k, PromptWeights* out, Stream s) {
  const std::size_t bytes = prompt_table_bytes(type, rows, k);
  void* mem = bytes ? dalloc(bytes) : nullptr;
  if (bytes && !mem) return pfail("prompt_prepare: device memory");
  if (!prompt_prepare_at(type, w_dev, rows, k, mem, out, s)) {
    dfree(mem);
    if (out) *out = PromptWeights{};
    return false;
  }
  out->owned = mem != nullptr;
  return true;
}

void prompt_release(PromptWeights* w) {
  if (!w) return;
  if (w->owned) dfree(w->rec);  // rec is the start of the one side-table block (dig inside it)
  *w = PromptWeights{};
}

bool act_digits(const float* scale, const int* sum, std::int64_t rows, std::int64_t k, float* dig, Stream s) {
  if (!scale || !sum || !dig || rows < 0 || k % kQK != 0) return pfail("act_digits: bad arguments");
  if (rows == 0) return true;
  const std::size_t n32 = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k / 32);
  k_act_digits<<<grid_of(n32, 256, 4096), 256, 0, pst(s)>>>(scale, sum, n32, dig);
  return pck(cudaGetLastError(), "act_digits");
}

bool prompt_gemm(const PromptWeights& w, std::int64_t n_out, const std::int8_t* q, const float* scale, const float* dig, std::int64_t rows, float* y,
                 Stream s) {
  if (!prompt_type_ok(w.type, w.k)) return pfail("prompt_gemm: no int8 step for this type (use gpu_ops' matmul on the rounded floats)");
  if (!w.w || !w.rec || !q || !scale || !dig || !y || n_out <= 0 || n_out % kBN != 0 || n_out > w.rows || rows < 0 || rows > 64 * 65535)
    return pfail("prompt_gemm: bad arguments (n_out a multiple of 128, within the weight)");
  if (rows == 0) return true;
  Epi E;
  E.C = y;
  const dim3 grid(static_cast<unsigned>(n_out / kBN), static_cast<unsigned>((rows + kTile - 1) / kTile));
  return tile_launch<0, true>(grid, q, scale, dig, w, nullptr, nullptr, static_cast<int>(rows), static_cast<int>(n_out), static_cast<int>(w.k), E, s);
}

bool moe_scratch_alloc(std::int64_t max_tokens, int n_used, int n_expert, std::int64_t hidden, std::int64_t ffn, MoeScratch* out, bool exact) {
  if (!out || max_tokens <= 0 || n_used <= 0 || n_expert <= 0 || n_expert > 256 || hidden % kQK != 0 || ffn % kQK != 0 || hidden % kBN != 0 || ffn % kBN != 0)
    return pfail("moe_scratch_alloc: bad shape (n_expert <= 256; hidden and ffn multiples of 256)");
  *out = MoeScratch{};
  const std::size_t np = static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(n_used), pr = np + kTile, H = static_cast<std::size_t>(hidden),
                    F = static_cast<std::size_t>(ffn), T = static_cast<std::size_t>(max_tokens);
  const std::size_t xe = exact ? 1 : 0;
  const std::size_t sizes[] = {al(256 * 4), al(257 * 4), al(256 * 4), al(16),  // cnt, off (n_expert + 1), cursor, meta
                               al((np / kTile + 256) * 16), al(256 * 16),     // units
                               al(np * 4), al(pr * 4), al(np * 4), al(pr * 4),  // src, rtok, prow, rw
                               al(pr * H), al(pr * (H / 32) * 4), al(pr * (H / 32) * 4),  // pq, psc, pdig
                               al(pr * F * 4), al(pr * F * 4),  // hg, hu
                               al(pr * F), al(pr * (F / 32) * 4), al(pr * (F / 32) * 4), al(pr * (F / 32) * 4),  // q2, sc2, sum2, dig2
                               al(T * H * 8), al(16),  // yfix (int64 capable), ovf
                               al(xe * T * H * 4), al(xe * pr * F * 4), al(xe * pr * H * 4), al(xe * pr * 4),  // exact: xf, x2f, yd, iota
                               al((np / kTile + 256) * 16), al(256 * 16), al(16)};  // a stack's units (emap): tile, vector, counts
  std::size_t total = 0;
  for (const std::size_t z : sizes) total += z;
  auto* base = static_cast<std::uint8_t*>(dalloc(total));
  if (!base) return pfail("moe_scratch_alloc: device memory");
  if (!dzero(base, total)) {
    dfree(base);
    return pfail("moe_scratch_alloc: clear");
  }
  std::uint8_t* p = base;
  int i = 0;
  auto take = [&]() {
    std::uint8_t* r = p;
    p += sizes[i++];
    return static_cast<void*>(r);
  };
  out->cnt = static_cast<int*>(take());
  out->off = static_cast<int*>(take());
  out->cursor = static_cast<int*>(take());
  out->meta = static_cast<int*>(take());
  out->units_t = take();
  out->units_v = take();
  out->src = static_cast<int*>(take());
  out->rtok = static_cast<int*>(take());
  out->prow = static_cast<int*>(take());
  out->rw = static_cast<float*>(take());
  out->pq = static_cast<std::int8_t*>(take());
  out->psc = static_cast<float*>(take());
  out->pdig = static_cast<float*>(take());
  out->hg = static_cast<float*>(take());
  out->hu = static_cast<float*>(take());
  out->q2 = static_cast<std::int8_t*>(take());
  out->sc2 = static_cast<float*>(take());
  out->sum2 = static_cast<int*>(take());
  out->dig2 = static_cast<float*>(take());
  out->yfix = take();
  out->ovf = static_cast<unsigned*>(take());
  out->xf = static_cast<float*>(take());
  out->x2f = static_cast<float*>(take());
  out->yd = static_cast<float*>(take());
  out->iota = static_cast<int*>(take());
  out->units_t2 = take();
  out->units_v2 = take();
  out->meta2 = static_cast<int*>(take());
  out->exact = exact;
  if (exact) {
    k_iota<<<grid_of(pr, 256, 4096), 256>>>(out->iota, static_cast<int>(pr));
    if (!pck(cudaGetLastError(), "moe_scratch_alloc: row map") || !pck(cudaDeviceSynchronize(), "moe_scratch_alloc: row map")) {
      dfree(base);
      *out = MoeScratch{};
      return false;
    }
  } else {
    out->xf = out->x2f = out->yd = nullptr;
    out->iota = nullptr;
  }
  out->mem = base;
  out->max_tokens = max_tokens;
  out->hidden = hidden;
  out->ffn = ffn;
  out->n_used = n_used;
  out->n_expert = n_expert;
  return true;
}

void moe_scratch_free(MoeScratch* m) {
  if (!m) return;
  dfree(m->mem);
  *m = MoeScratch{};
}

bool moe_prompt_multi(MoeStacks gate, MoeStacks up, MoeStacks down, const std::int8_t* q, const float* scale, const float* dig, std::int64_t tokens,
                      const std::int32_t* sel, const float* wsel, MoeScratch& m, MoeCombine combine, int scale_log2, float* y, Stream s) {
  if (!m.mem || tokens < 0 || tokens > m.max_tokens || !q || !scale || !dig || !sel || !wsel || !y) return pfail("moe_prompt: bad arguments");
  const std::int64_t H = m.hidden, F = m.ffn, E = m.n_expert;
  if (gate.n < 1 || up.n < 1 || down.n < 1 || gate.n > 4 || up.n > 4 || down.n > 4 || !gate.w || !up.w || !down.w)
    return pfail("moe_prompt: 1-4 stacks per projection");
  auto shape_ok = [&](const MoeStacks& st, std::int64_t n_out, std::int64_t k) {
    for (int i = 0; i < st.n; ++i) {
      const PromptWeights& W = st.w[i];
      const std::int64_t nm = W.n_mat > 0 ? W.n_mat : E;
      if (W.rows != nm * n_out || W.k != k || (st.n > 1 && !W.emap)) return false;
    }
    return true;
  };
  if (!shape_ok(gate, F, H) || !shape_ok(up, F, H) || !shape_ok(down, H, F)) return pfail("moe_prompt: the expert stacks do not match the scratch shape");
  auto any_exact = [](const MoeStacks& st) {
    for (int i = 0; i < st.n; ++i)
      if (!int8_type(st.w[i].type)) return true;
    return false;
  };
  if (tokens == 0) return true;
  const int np = static_cast<int>(tokens * m.n_used);
  cudaStream_t cs = pst(s);
  // (2) the work list on the device
  if (!pck(cudaMemsetAsync(m.cnt, 0, 256 * sizeof(int), cs), "moe_prompt: clear counts")) return false;
  k_route_count<<<grid_of(static_cast<std::size_t>(np), 256), 256, 0, cs>>>(sel, np, m.cnt);
  k_route_build<<<1, 32, 0, cs>>>(m.cnt, static_cast<int>(E), m.off, m.cursor, static_cast<int4*>(m.units_t), static_cast<int4*>(m.units_v), m.meta);
  k_route_place<<<grid_of(static_cast<std::size_t>(np), 256), 256, 0, cs>>>(sel, wsel, np, m.n_used, m.cursor, m.src, m.rtok, m.rw, m.prow);
  k_gather<<<static_cast<unsigned>(np), 128, 0, cs>>>(m.src, np, static_cast<int>(H), q, scale, dig, m.pq, m.psc, m.pdig);
  if (!pck(cudaGetLastError(), "moe_prompt: work list")) return false;
  // (3) / (4) gate and up
  Epi eg, eu;
  eg.C = m.hg;
  eu.C = m.hu;
  if ((any_exact(gate) || any_exact(up)) && m.exact) {  // the exact path reads the token rows as floats (q x scale)
    k_deq_rows<<<grid_of(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(H), 256, 4096), 256, 0, cs>>>(q, scale, static_cast<std::size_t>(tokens * H), m.xf);
    if (!pck(cudaGetLastError(), "moe_prompt: float rows")) return false;
  }
  for (int i = 0; i < gate.n; ++i)
    if (!run_stack<0>(m, gate.w[i], m.pq, m.psc, m.pdig, m.xf, m.src, static_cast<int>(F), static_cast<int>(H), eg, np, s)) return false;
  for (int i = 0; i < up.n; ++i)
    if (!run_stack<0>(m, up.w[i], m.pq, m.psc, m.pdig, m.xf, m.src, static_cast<int>(F), static_cast<int>(H), eu, np, s)) return false;
  k_silu_q8x32<<<grid_of(static_cast<std::size_t>(np) * static_cast<std::size_t>(F / 32), 128, 0x7fffffff), 128, 0, cs>>>(m.hg, m.hu, np, F, m.q2, m.sc2, m.sum2);
  const std::size_t n32 = static_cast<std::size_t>(np) * static_cast<std::size_t>(F / 32);
  k_act_digits<<<grid_of(n32, 256, 4096), 256, 0, cs>>>(m.sc2, m.sum2, n32, m.dig2);
  if (any_exact(down) && m.exact)
    k_deq_rows<<<grid_of(static_cast<std::size_t>(np) * static_cast<std::size_t>(F), 256, 4096), 256, 0, cs>>>(m.q2, m.sc2, static_cast<std::size_t>(np) * static_cast<std::size_t>(F), m.x2f);
  if (!pck(cudaGetLastError(), "moe_prompt: silu and rounding")) return false;
  // (5) down folded into the combine
  const std::size_t ny = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(H);
  Epi ed;
  ed.rtok = m.rtok;
  ed.rw = m.rw;
  ed.ovf = m.ovf;
  ed.scale = std::ldexp(1.0f, scale_log2);
  if (combine == MoeCombine::kFloat) {
    if (!pck(cudaMemsetAsync(y, 0, ny * sizeof(float), cs), "moe_prompt: clear y")) return false;
    ed.Yf = y;
    for (int i = 0; i < down.n; ++i)
      if (!run_stack<1>(m, down.w[i], m.q2, m.sc2, m.dig2, m.x2f, m.iota, static_cast<int>(H), static_cast<int>(F), ed, np, s)) return false;
    return true;
  }
  const bool w64 = combine == MoeCombine::kFixed64;
  if (!pck(cudaMemsetAsync(m.yfix, 0, ny * (w64 ? 8 : 4), cs), "moe_prompt: clear accumulators")) return false;
  if (w64) {
    ed.Yl = static_cast<unsigned long long*>(m.yfix);
    for (int i = 0; i < down.n; ++i)
      if (!run_stack<3>(m, down.w[i], m.q2, m.sc2, m.dig2, m.x2f, m.iota, static_cast<int>(H), static_cast<int>(F), ed, np, s)) return false;
  } else {
    ed.Yi = static_cast<int*>(m.yfix);
    for (int i = 0; i < down.n; ++i)
      if (!run_stack<2>(m, down.w[i], m.q2, m.sc2, m.dig2, m.x2f, m.iota, static_cast<int>(H), static_cast<int>(F), ed, np, s)) return false;
  }
  k_fix<<<grid_of(ny, 256, 4096), 256, 0, cs>>>(w64 ? nullptr : static_cast<const int*>(m.yfix), w64 ? static_cast<const unsigned long long*>(m.yfix) : nullptr, ny,
                                                std::ldexp(1.0f, -scale_log2), y);
  return pck(cudaGetLastError(), "moe_prompt: conversion");
}

bool moe_prompt(const PromptWeights& gate, const PromptWeights& up, const PromptWeights& down, const std::int8_t* q, const float* scale, const float* dig,
                std::int64_t tokens, const std::int32_t* sel, const float* wsel, MoeScratch& m, MoeCombine combine, int scale_log2, float* y, Stream s) {
  return moe_prompt_multi(MoeStacks{&gate, 1}, MoeStacks{&up, 1}, MoeStacks{&down, 1}, q, scale, dig, tokens, sel, wsel, m, combine, scale_log2, y, s);
}

}  // namespace pelias::kern::gpu

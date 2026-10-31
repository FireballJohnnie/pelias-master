// provenance: original (moved from tools/probes/batch_c kernels_c.cu: V12x8Stage, k_moe_tile<3, 64, EPI, true, true>,
// k_moe_vec, k_route_count / k_route_build / k_route_place, k_gather_q8, k_fix_to_f32, k_prep_q4k_rec, k_prep_q4k_dig8,
// k_dig_x32; MEASURED 2026-10-30 21:05-21:40, gp/MOE; the device arithmetic unchanged. New here: the dense mode of the
// tile kernel (units computed from blockIdx), int32 expert ids from router_topk, and silu x up rounded with gpu_ops's
// own expressions (silu_d, act_q8x32's loops), so the down projection's input is the reference's)
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

__global__ void k_prep_rec(const BlockQ4K* __restrict__ w, uint4* __restrict__ rec, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = w[i];
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
__global__ void k_prep_dig(const BlockQ4K* __restrict__ w, float* __restrict__ wd, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = w[i];
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

struct alignas(16) Stage {
  std::int8_t As[kTile][kQ8Lda];
  std::uint8_t Qs[kBN][kQsLd];
  uint4 Wr[kBN];
  float S2[kTile][2];
  float Am[kTile][kMinLd];  // activation min factors of the super-block (8 used)
  float Bm[kBN][kMinLd];    // weight min factors dmin x m_j (8 used)
};
constexpr int kSmem = kStages * static_cast<int>(sizeof(Stage));

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
template <int EPI, bool kDense>
__global__ void __launch_bounds__(256, 2) k_tile(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const float* __restrict__ adig,
                                                const BlockQ4K* __restrict__ w, const uint4* __restrict__ rec, const float* __restrict__ wdig8,
                                                const int4* __restrict__ units, const int* __restrict__ nunits, int nrows_total, int N, int K, Epi E) {
  constexpr int WMW = 2, WNW = 4, NI = 4, WR = 32, MI = 2;
  extern __shared__ __align__(16) std::uint8_t p_smem[];
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
  w += static_cast<std::size_t>(ex) * N * bpr;
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
      const std::uint8_t* blk = reinterpret_cast<const std::uint8_t*>(w + static_cast<std::size_t>(bn + r) * bpr + sb);
      cp_async16(&T.Qs[r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
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
  constexpr float kMagic = 12582912.f;  // 1.5 x 2^23: |is x c| <= 63 x 32 x 127 x 15 < 2^22, exact
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
    std::uint32_t bw[NI][2];
    float dc0[NI], dc1[NI];
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int c = wn * NI * 8 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
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
          const float p0 = __int_as_float(is0[ni] * c4[0] + 0x4B400000) - kMagic;
          const float p1 = __int_as_float(is1[ni] * c4[1] + 0x4B400000) - kMagic;
          const float p2 = __int_as_float(is0[ni] * c4[2] + 0x4B400000) - kMagic;
          const float p3 = __int_as_float(is1[ni] * c4[3] + 0x4B400000) - kMagic;
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
template <int EPI>
__global__ void __launch_bounds__(256) k_vec(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const BlockQ4K* __restrict__ w,
                                             const int4* __restrict__ units, const int* __restrict__ nunits, int N, int K, Epi E) {
  if (static_cast<int>(blockIdx.x) >= *nunits) return;
  const int4 u = units[blockIdx.x];
  const int lane = threadIdx.x & 31, n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const int bpr = K / kQK, n32 = K / 32, rows = u.z;
  const BlockQ4K* wr = w + (static_cast<std::size_t>(u.x) * N + n) * bpr;
  float acc[kVecMax];
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) acc[r] = 0.f;
  for (int jj = lane; jj < n32; jj += 32) {
    const BlockQ4K& b = wr[jj >> 3];
    const int j = jj & 7;
    int sc, m;
    scale_min_k4_d(j, b.scales, sc, m);
    const float dsc = h2f(b.d) * static_cast<float>(sc), dm = h2f(b.dmin) * static_cast<float>(m);
    const uint4 qa = *reinterpret_cast<const uint4*>(b.qs + (j >> 1) * 32), qb = *reinterpret_cast<const uint4*>(b.qs + (j >> 1) * 32 + 16);
    const std::uint32_t qw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const int sh = 4 * (j & 1);
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
        isum = __dp4a(static_cast<int>((qw[q] >> sh) & 0x0F0F0F0Fu), xw[q], isum);
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

unsigned grid_of(std::size_t n, int per, unsigned cap = 65535) {
  const std::size_t b = (n + static_cast<std::size_t>(per) - 1) / static_cast<std::size_t>(per);
  return static_cast<unsigned>(b < 1 ? 1 : (b > cap ? cap : b));
}

template <int EPI, bool kDense>
bool tile_launch(dim3 grid, const std::int8_t* a8, const float* sa32, const float* adig, const PromptWeights& W, const int4* units, const int* nunits,
                 int nrows, int N, int K, const Epi& E, Stream s) {
  static const cudaError_t once = cudaFuncSetAttribute(k_tile<EPI, kDense>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (!pck(once, "prompt tile: shared memory opt-in")) return false;
  k_tile<EPI, kDense><<<grid, 256, kSmem, pst(s)>>>(a8, sa32, adig, static_cast<const BlockQ4K*>(W.w), static_cast<const uint4*>(W.rec),
                                                    static_cast<const float*>(W.dig), units, nunits, nrows, N, K, E);
  return pck(cudaGetLastError(), "prompt tile");
}
template <int EPI>
bool vec_launch(dim3 grid, const std::int8_t* a8, const float* sa32, const PromptWeights& W, const int4* units, const int* nunits, int N, int K, const Epi& E,
                Stream s) {
  k_vec<EPI><<<grid, 256, 0, pst(s)>>>(a8, sa32, static_cast<const BlockQ4K*>(W.w), units, nunits, N, K, E);
  return pck(cudaGetLastError(), "prompt vector");
}
// expert GEMM over the device lists (tiles + vectors) with epilogue EPI
template <int EPI>
bool experts(const MoeScratch& m, const std::int8_t* a8, const float* sa32, const float* adig, const PromptWeights& W, int N, int K, const Epi& E,
             std::int64_t picks, Stream s) {
  const int max_tiles = static_cast<int>(picks / kTile) + m.n_expert;
  const int4* ut = static_cast<const int4*>(m.units_t);
  const int4* uv = static_cast<const int4*>(m.units_v);
  return tile_launch<EPI, false>(dim3(static_cast<unsigned>(N / kBN), static_cast<unsigned>(max_tiles)), a8, sa32, adig, W, ut, m.meta, 0, N, K, E, s) &&
         vec_launch<EPI>(dim3(static_cast<unsigned>(m.n_expert), static_cast<unsigned>((N + 7) / 8)), a8, sa32, W, uv, m.meta + 1, N, K, E, s);
}

std::size_t al(std::size_t x) { return (x + 255) / 256 * 256; }

}  // namespace

const char* prompt_last_error() { return p_err.c_str(); }

bool prompt_type_ok(std::uint32_t type, std::int64_t k) { return type == static_cast<std::uint32_t>(quant::Type::q4_k) && k > 0 && k % kQK == 0; }

bool prompt_prepare(std::uint32_t type, const void* w_dev, std::int64_t rows, std::int64_t k, PromptWeights* out, Stream s) {
  if (!out || !w_dev || rows <= 0 || rows % kBN != 0) return pfail("prompt_prepare: bad arguments (rows a multiple of 128)");
  if (!prompt_type_ok(type, k)) return pfail("prompt_prepare: the int8 prompt path reads Q4_K with k a multiple of 256");
  *out = PromptWeights{};
  const std::size_t nb = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k / kQK);
  void* rec = dalloc(nb * 16);
  void* dg = dalloc(nb * 32);
  if (!rec || !dg) {
    dfree(rec);
    dfree(dg);
    return pfail("prompt_prepare: device memory");
  }
  k_prep_rec<<<grid_of(nb, 256, 4096), 256, 0, pst(s)>>>(static_cast<const BlockQ4K*>(w_dev), static_cast<uint4*>(rec), nb);
  k_prep_dig<<<grid_of(nb, 256, 4096), 256, 0, pst(s)>>>(static_cast<const BlockQ4K*>(w_dev), static_cast<float*>(dg), nb);
  if (!pck(cudaGetLastError(), "prompt_prepare")) {
    dfree(rec);
    dfree(dg);
    return false;
  }
  out->type = type;
  out->rows = rows;
  out->k = k;
  out->w = w_dev;
  out->rec = rec;
  out->dig = dg;
  return true;
}

void prompt_release(PromptWeights* w) {
  if (!w) return;
  dfree(w->rec);
  dfree(w->dig);
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
  if (!w.w || !w.rec || !q || !scale || !dig || !y || n_out <= 0 || n_out % kBN != 0 || n_out > w.rows || rows < 0 || rows > 64 * 65535)
    return pfail("prompt_gemm: bad arguments (n_out a multiple of 128, within the weight)");
  if (rows == 0) return true;
  Epi E;
  E.C = y;
  const dim3 grid(static_cast<unsigned>(n_out / kBN), static_cast<unsigned>((rows + kTile - 1) / kTile));
  return tile_launch<0, true>(grid, q, scale, dig, w, nullptr, nullptr, static_cast<int>(rows), static_cast<int>(n_out), static_cast<int>(w.k), E, s);
}

bool moe_scratch_alloc(std::int64_t max_tokens, int n_used, int n_expert, std::int64_t hidden, std::int64_t ffn, MoeScratch* out) {
  if (!out || max_tokens <= 0 || n_used <= 0 || n_expert <= 0 || n_expert > 256 || hidden % kQK != 0 || ffn % kQK != 0 || hidden % kBN != 0 || ffn % kBN != 0)
    return pfail("moe_scratch_alloc: bad shape (n_expert <= 256; hidden and ffn multiples of 256)");
  *out = MoeScratch{};
  const std::size_t np = static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(n_used), pr = np + kTile, H = static_cast<std::size_t>(hidden),
                    F = static_cast<std::size_t>(ffn), T = static_cast<std::size_t>(max_tokens);
  const std::size_t sizes[] = {al(256 * 4), al(256 * 4), al(256 * 4), al(16),  // cnt, off, cursor, meta
                               al((np / kTile + 256) * 16), al(256 * 16),     // units
                               al(np * 4), al(pr * 4), al(np * 4), al(pr * 4),  // src, rtok, prow, rw
                               al(pr * H), al(pr * (H / 32) * 4), al(pr * (H / 32) * 4),  // pq, psc, pdig
                               al(pr * F * 4), al(pr * F * 4),  // hg, hu
                               al(pr * F), al(pr * (F / 32) * 4), al(pr * (F / 32) * 4), al(pr * (F / 32) * 4),  // q2, sc2, sum2, dig2
                               al(T * H * 8), al(16)};  // yfix (int64 capable), ovf
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

bool moe_prompt(const PromptWeights& gate, const PromptWeights& up, const PromptWeights& down, const std::int8_t* q, const float* scale, const float* dig,
                std::int64_t tokens, const std::int32_t* sel, const float* wsel, MoeScratch& m, MoeCombine combine, int scale_log2, float* y, Stream s) {
  if (!m.mem || tokens < 0 || tokens > m.max_tokens || !q || !scale || !dig || !sel || !wsel || !y) return pfail("moe_prompt: bad arguments");
  const std::int64_t H = m.hidden, F = m.ffn, E = m.n_expert;
  if (gate.rows != E * F || up.rows != E * F || down.rows != E * H || gate.k != H || up.k != H || down.k != F)
    return pfail("moe_prompt: the expert stacks do not match the scratch shape");
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
  if (!experts<0>(m, m.pq, m.psc, m.pdig, gate, static_cast<int>(F), static_cast<int>(H), eg, np, s)) return false;
  if (!experts<0>(m, m.pq, m.psc, m.pdig, up, static_cast<int>(F), static_cast<int>(H), eu, np, s)) return false;
  k_silu_q8x32<<<grid_of(static_cast<std::size_t>(np) * static_cast<std::size_t>(F / 32), 128, 0x7fffffff), 128, 0, cs>>>(m.hg, m.hu, np, F, m.q2, m.sc2, m.sum2);
  const std::size_t n32 = static_cast<std::size_t>(np) * static_cast<std::size_t>(F / 32);
  k_act_digits<<<grid_of(n32, 256, 4096), 256, 0, cs>>>(m.sc2, m.sum2, n32, m.dig2);
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
    return experts<1>(m, m.q2, m.sc2, m.dig2, down, static_cast<int>(H), static_cast<int>(F), ed, np, s);
  }
  const bool w64 = combine == MoeCombine::kFixed64;
  if (!pck(cudaMemsetAsync(m.yfix, 0, ny * (w64 ? 8 : 4), cs), "moe_prompt: clear accumulators")) return false;
  if (w64) {
    ed.Yl = static_cast<unsigned long long*>(m.yfix);
    if (!experts<3>(m, m.q2, m.sc2, m.dig2, down, static_cast<int>(H), static_cast<int>(F), ed, np, s)) return false;
  } else {
    ed.Yi = static_cast<int*>(m.yfix);
    if (!experts<2>(m, m.q2, m.sc2, m.dig2, down, static_cast<int>(H), static_cast<int>(F), ed, np, s)) return false;
  }
  k_fix<<<grid_of(ny, 256, 4096), 256, 0, cs>>>(w64 ? nullptr : static_cast<const int*>(m.yfix), w64 ? static_cast<const unsigned long long*>(m.yfix) : nullptr, ny,
                                                std::ldexp(1.0f, -scale_log2), y);
  return pck(cudaGetLastError(), "moe_prompt: conversion");
}

}  // namespace pelias::kern::gpu

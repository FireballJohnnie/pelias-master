// provenance: original (Q4_K / Q6_K arithmetic from the spec: llama.cpp ggml-quants.c dequantize_row_q4_K / _q6_K)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kernels_c.cu - Batch C kernels (P-K1, P-K2, P-K3, P-Q10). Target sm_89 (Ada). nvcc only.
// The mma.sync fragment layouts are the ones the canary GEMM of kernels.cu uses (that canary passed on the reference
// GPU, P-H10 / P-H4); the int8 m16n8k32 layout follows the same pattern with four 8-bit values per register, and every
// kernel here is checked against a reference inside its probe.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

#include "kernels_c.h"

namespace pkc {
namespace {

// ---------------------------------------------------------------- small helpers
__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
__device__ __forceinline__ float u01(uint32_t r) { return (static_cast<float>(r >> 8) + 0.5f) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float gauss(uint32_t seed, size_t i, uint32_t salt) {
  const uint32_t a = mix32(seed ^ static_cast<uint32_t>(i * 2654435761u) ^ salt);
  const uint32_t b = mix32(a + 0x9E3779B9u + static_cast<uint32_t>(i >> 32));
  return sqrtf(-2.0f * logf(u01(a))) * cospif(2.0f * u01(b));
}
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __bfloat162float(__ushort_as_bfloat16(b)); }

// get_scale_min_k4 (ggml-quants.c)
__device__ __forceinline__ void scale_min_k4_d(int j, const uint8_t* q, int& d, int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

__device__ __forceinline__ uint32_t ld32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
// Exact int -> float for |x| < 2^22 on the full-rate integer and fp32 pipes (I2F runs at a quarter of that rate on Ada).
// An m16n8k32 int8 x 4-bit sum is at most 32 * 127 * 15 = 60,960.
__device__ __forceinline__ float i2f_fast(int x) { return __int_as_float(0x4B400000 + x) - 12582912.0f; }

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------- data generation
__global__ void k_fill_q4k(BlockQ4K* w, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    BlockQ4K& b = w[i];
    const uint32_t r0 = mix32(seed ^ static_cast<uint32_t>(i) * 0x9E3779B1u);
    b.d = f2h(0.002f + 0.018f * u01(r0));
    b.dmin = f2h(0.010f * u01(mix32(r0 + 1u)));
    for (int k = 0; k < 12; ++k) b.scales[k] = static_cast<uint8_t>(mix32(r0 + 2u + k) >> 24);
    for (int k = 0; k < 128; k += 4) {
      const uint32_t v = mix32(r0 + 100u + k);
      b.qs[k] = static_cast<uint8_t>(v);
      b.qs[k + 1] = static_cast<uint8_t>(v >> 8);
      b.qs[k + 2] = static_cast<uint8_t>(v >> 16);
      b.qs[k + 3] = static_cast<uint8_t>(v >> 24);
    }
  }
}
__global__ void k_fill_q6k(BlockQ6K* w, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    BlockQ6K& b = w[i];
    const uint32_t r0 = mix32(seed ^ static_cast<uint32_t>(i) * 0x85EBCA77u);
    for (int k = 0; k < 128; ++k) b.ql[k] = static_cast<uint8_t>(mix32(r0 + k) >> 24);
    for (int k = 0; k < 64; ++k) b.qh[k] = static_cast<uint8_t>(mix32(r0 + 200u + k) >> 24);
    for (int k = 0; k < 16; ++k) b.scales[k] = static_cast<int8_t>(static_cast<int>(mix32(r0 + 300u + k) >> 25) - 64);
    b.d = f2h(0.0005f + 0.0015f * u01(mix32(r0 + 400u)));
  }
}
__global__ void k_fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    p[i] = f2bf(0.5f * gauss(seed, i, 0x1234u));
}
__global__ void k_fill_f32(float* p, size_t n, uint32_t seed, int dist, float scale) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    float v = gauss(seed, i, 0x77u);
    if (dist == 1) v *= expf(gauss(seed, i, 0x3141u));  // heavy tails
    p[i] = v * scale;
  }
}

// ---------------------------------------------------------------- Q4_K -> bf16 scratch (P-K1 dequant step)
// Rows of W are contiguous, so block blk's 256 outputs start at blk * 256. One warp takes 32 blocks: it copies them to
// shared memory with 16-byte loads, then each lane writes 8 consecutive bf16 of every block (one 16-byte store).
template <int kWarps>
__global__ void __launch_bounds__(kWarps * 32) k_dequant_q4k_bf16(const BlockQ4K* __restrict__ w, uint16_t* __restrict__ out, size_t nblocks) {
  __shared__ __align__(16) uint8_t sm[kWarps][32 * kQ4KBytes];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t groups = nblocks / 32;
  const int j = lane >> 2, l0 = (lane & 3) * 8, shift = (j & 1) * 4;
  for (size_t gi = static_cast<size_t>(blockIdx.x) * kWarps + warp; gi < groups; gi += static_cast<size_t>(gridDim.x) * kWarps) {
    const uint4* src = reinterpret_cast<const uint4*>(w + gi * 32);
    uint4* dst = reinterpret_cast<uint4*>(sm[warp]);
    for (int i = lane; i < 32 * kQ4KBytes / 16; i += 32) dst[i] = src[i];
    __syncwarp();
    for (int b = 0; b < 32; ++b) {
      const BlockQ4K& blk = *reinterpret_cast<const BlockQ4K*>(sm[warp] + b * kQ4KBytes);
      int sc, m;
      scale_min_k4_d(j, blk.scales, sc, m);
      const float d = h2f(blk.d) * sc, mn = h2f(blk.dmin) * m;
      const uint8_t* q = blk.qs + (j >> 1) * 32 + l0;
      uint32_t p[4];
#pragma unroll
      for (int l = 0; l < 4; ++l)
        p[l] = static_cast<uint32_t>(f2bf(d * ((q[2 * l] >> shift) & 0xF) - mn)) |
               (static_cast<uint32_t>(f2bf(d * ((q[2 * l + 1] >> shift) & 0xF) - mn)) << 16);
      *reinterpret_cast<uint4*>(out + (gi * 32 + b) * kQK + j * 32 + l0) = make_uint4(p[0], p[1], p[2], p[3]);
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------- tile order
// raster 0: N fastest (row by row of output tiles). raster 1: grouped, 8 tiles along M per group, so a weight tile is
// reused by 8 neighbouring blocks while it is still in L2 (the order Triton's matmul tutorial uses).
__device__ __forceinline__ void tile_coords(int raster, int npm, int npn, int& pm, int& pn) {
  const int pid = static_cast<int>(blockIdx.x);
  if (raster == 0) {
    pm = pid / npn;
    pn = pid % npn;
  } else {
    constexpr int G = 8;
    const int width = G * npn, group = pid / width, first = group * G;
    const int gs = min(npm - first, G);
    pm = first + (pid % width) % gs;
    pn = (pid % width) / gs;
  }
}

// ---------------------------------------------------------------- P-K1: tiled bf16 GEMM, cp.async two-stage pipeline
// 256 threads = 8 warps (WM along M x WN along N), each warp MI x NI mma tiles of 16 x 8; block tile BM x BN x 32.
//   cfg 0: 128 x 128 (2 x 4 warps, 4 x 4 tiles)   cfg 1: 128 x 64 (4 x 2, 2 x 4)   cfg 2: 64 x 64 (2 x 4, 2 x 2)
constexpr int kBK = 32, kLdsH = kBK + 8;  // row of 40 halves = 80 bytes (16-byte aligned)

template <int WM, int WN, int MI, int NI>
__global__ void __launch_bounds__(256) k_gemm_bf16_t(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
                                                     float* __restrict__ C, int M, int N, int K, int raster, size_t sA = 0,
                                                     size_t sB = 0, size_t sC = 0) {
  static_assert(WM * WN == 8, "8 warps");
  A += blockIdx.y * sA;  // batch (revision 12, grouped experts): one problem per blockIdx.y
  B += blockIdx.y * sB;
  C += blockIdx.y * sC;
  constexpr int BM = WM * MI * 16, BN = WN * NI * 8;
  __shared__ __align__(16) uint16_t As[2][BM][kLdsH];
  __shared__ __align__(16) uint16_t Bs[2][BN][kLdsH];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp / WN, wn = warp % WN;
  int pm, pn;
  tile_coords(raster, M / BM, N / BN, pm, pn);
  const int bm = pm * BM, bn = pn * BN;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

  const int kt_n = K / kBK;
  auto load = [&](int stage, int k0) {
    for (int idx = tid; idx < BM * 4; idx += 256) {
      const int r = idx >> 2, c = (idx & 3) * 8;
      cp_async16(&As[stage][r][c], A + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    for (int idx = tid; idx < BN * 4; idx += 256) {
      const int r = idx >> 2, c = (idx & 3) * 8;
      cp_async16(&Bs[stage][r][c], B + static_cast<size_t>(bn + r) * K + k0 + c);
    }
  };
  load(0, 0);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    const int st = kt & 1;
    if (kt + 1 < kt_n) {
      load(st ^ 1, (kt + 1) * kBK);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t a[MI][4], b[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r0 = wm * (MI * 16) + mi * 16 + g;
        a[mi][0] = ld32(&As[st][r0][kk + 2 * t]);
        a[mi][1] = ld32(&As[st][r0 + 8][kk + 2 * t]);
        a[mi][2] = ld32(&As[st][r0][kk + 2 * t + 8]);
        a[mi][3] = ld32(&As[st][r0 + 8][kk + 2 * t + 8]);
      }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * (NI * 8) + ni * 8 + g;
        b[ni][0] = ld32(&Bs[st][c][kk + 2 * t]);
        b[ni][1] = ld32(&Bs[st][c][kk + 2 * t + 8]);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], a[mi], b[ni]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int r0 = bm + wm * (MI * 16) + mi * 16 + g;
      const int c0 = bn + wn * (NI * 8) + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

constexpr int kBM = 128, kBN = 128;  // tile of the int8 kernel

// ---------------------------------------------------------------- P-K2: activations to int8 per row per 32
// Rows are contiguous, so group gidx (row m, sub-block kb) starts at gidx * 32. Four threads per group, 8 values each
// (one 16-byte load); max and sum are shared inside the four lanes with shuffles.
__global__ void k_quantize_rows_q8(const uint16_t* __restrict__ A, int8_t* __restrict__ a8, float* __restrict__ sa,
                                   int* __restrict__ s8, float* __restrict__ sas, size_t groups) {
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < groups * 4; t += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const unsigned mask = __activemask();  // whole quads are active together (groups * 4 and the stride are multiples of 4)
    const size_t gidx = t >> 2;
    const int part = static_cast<int>(t & 3);
    const uint4 v = *reinterpret_cast<const uint4*>(A + gidx * 32 + part * 8);
    const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
    float f[8];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = bf2f(static_cast<uint16_t>(wv[i] & 0xFFFFu));
      f[2 * i + 1] = bf2f(static_cast<uint16_t>(wv[i] >> 16));
      amax = fmaxf(amax, fmaxf(fabsf(f[2 * i]), fabsf(f[2 * i + 1])));
    }
    amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    const float inv = 1.f / scale;
    int sum = 0;
    uint32_t packed[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q = max(-127, min(127, __float2int_rn(f[i] * inv)));
      sum += q;
      packed[i >> 2] |= (static_cast<uint32_t>(q) & 0xFFu) << ((i & 3) * 8);
    }
    sum += __shfl_xor_sync(mask, sum, 1);
    sum += __shfl_xor_sync(mask, sum, 2);
    *reinterpret_cast<uint2*>(a8 + gidx * 32 + part * 8) = make_uint2(packed[0], packed[1]);
    if (part == 0) {
      sa[gidx] = scale;
      s8[gidx] = sum;
      sas[gidx] = scale * static_cast<float>(sum);  // the activation side of the min term, for the v3 kernel
    }
  }
}

// ---------------------------------------------------------------- P-K2: int8 x Q4_K GEMM
// Block tile 128 x 128, K step 64 (one 64-weight chunk of a Q4_K super-block = sub-blocks 2c and 2c+1, which share the
// same 32 qs bytes: low nibbles = 2c, high nibbles = 2c+1). Two cp.async stages.
constexpr int kQ8Lda = 64 + 16;  // int8 row of the A tile (80 bytes)
constexpr int kQsLd = 32 + 16;   // raw qs bytes per weight row (48 bytes)

__global__ void __launch_bounds__(256) k_gemm_q8_q4k(const int8_t* __restrict__ a8, const float* __restrict__ sa,
                                                     const int* __restrict__ s8, const BlockQ4K* __restrict__ w,
                                                     float* __restrict__ C, int M, int N, int K, int raster) {
  __shared__ __align__(16) int8_t As[2][kBM][kQ8Lda];
  __shared__ __align__(16) uint8_t Qs[2][kBN][kQsLd];
  __shared__ __align__(16) uint8_t Hd[2][kBN][16];  // d, dmin, scales[12] of the current super-block
  __shared__ __align__(16) float Sa[2][kBM][2];
  __shared__ __align__(16) int S8[2][kBM][2];
  __shared__ float Dsc[2][kBN], Dmn[2][kBN];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

  const int kt_n = K / 64;
#define PKC_LOAD_Q8(stage, k0)                                                                                   \
  {                                                                                                              \
    _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) {                                                           \
      const int idx_ = tid + i_ * 256, r_ = idx_ >> 2, c_ = (idx_ & 3) * 16;                                     \
      cp_async16(&As[stage][r_][c_], a8 + static_cast<size_t>(bm + r_) * K + (k0) + c_);                        \
    }                                                                                                            \
    {                                                                                                            \
      const int r_ = tid >> 1, h_ = tid & 1;                                                                     \
      const uint8_t* blk_ = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r_) * bpr + (k0) / kQK); \
      cp_async16(&Qs[stage][r_][h_ * 16], blk_ + 16 + (((k0) % kQK) / 64) * 32 + h_ * 16);                       \
      if (tid < 128) {                                                                                           \
        const uint8_t* hb_ = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + tid) * bpr + (k0) / kQK); \
        cp_async16(&Hd[stage][tid][0], hb_);                                                                     \
        cp_async8(&Sa[stage][tid][0], sa + static_cast<size_t>(bm + tid) * nkb + (k0) / 32);                     \
      } else {                                                                                                   \
        const int r2_ = tid - 128;                                                                               \
        cp_async8(&S8[stage][r2_][0], s8 + static_cast<size_t>(bm + r2_) * nkb + (k0) / 32);                     \
      }                                                                                                          \
    }                                                                                                            \
  }
  PKC_LOAD_Q8(0, 0);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    const int st = kt & 1;
    const int k0 = kt * 64;
    if (kt + 1 < kt_n) {
      PKC_LOAD_Q8(st ^ 1, (kt + 1) * 64);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    if (tid < 128) {  // the two sub-block scales of this chunk, per weight row
      const uint8_t* hb = Hd[st][tid];
      const float d = h2f(static_cast<uint16_t>(hb[0] | (hb[1] << 8)));
      const float dm = h2f(static_cast<uint16_t>(hb[2] | (hb[3] << 8)));
      const int c = (k0 % kQK) / 64;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int sc, m;
        scale_min_k4_d(2 * c + h, hb + 4, sc, m);
        Dsc[h][tid] = d * sc;
        Dmn[h][tid] = dm * m;
      }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t a[4][4], b[4][2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r0 = wm * 64 + mi * 16 + g;
        a[mi][0] = ld32(&As[st][r0][h * 32 + 4 * t]);
        a[mi][1] = ld32(&As[st][r0 + 8][h * 32 + 4 * t]);
        a[mi][2] = ld32(&As[st][r0][h * 32 + 16 + 4 * t]);
        a[mi][3] = ld32(&As[st][r0 + 8][h * 32 + 16 + 4 * t]);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int c = wn * 32 + ni * 8 + g;
        const uint32_t w0 = ld32(&Qs[st][c][4 * t]), w1 = ld32(&Qs[st][c][16 + 4 * t]);
        b[ni][0] = h ? ((w0 >> 4) & 0x0F0F0F0Fu) : (w0 & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((w1 >> 4) & 0x0F0F0F0Fu) : (w1 & 0x0F0F0F0Fu);
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
        const float fa0 = Sa[st][r0][h], fa1 = Sa[st][r1][h];
        const float sm0 = static_cast<float>(S8[st][r0][h]), sm1 = static_cast<float>(S8[st][r1][h]);
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a[mi], b[ni]);
          const int col0 = wn * 32 + ni * 8 + 2 * t, col1 = col0 + 1;
          const float d0 = Dsc[h][col0], d1 = Dsc[h][col1], m0 = Dmn[h][col0], m1 = Dmn[h][col1];
          acc[mi][ni][0] += fa0 * (d0 * static_cast<float>(c4[0]) - m0 * sm0);
          acc[mi][ni][1] += fa0 * (d1 * static_cast<float>(c4[1]) - m1 * sm0);
          acc[mi][ni][2] += fa1 * (d0 * static_cast<float>(c4[2]) - m0 * sm1);
          acc[mi][ni][3] += fa1 * (d1 * static_cast<float>(c4[3]) - m1 * sm1);
        }
      }
    }
    __syncthreads();
  }
#undef PKC_LOAD_Q8
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 64 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v4: the same arithmetic, products batched before the epilogue
// Revision 3 (v3) measured the same speed as v2 (0.96x to 1.03x), so the conversion was not the limit. v4 keeps v3's
// changes and issues the 16 products of a sub-block before any of them is scaled. v3's changes against v2: the int32 -> fp32 conversion uses the exact magic-number
// form instead of I2F; the min term uses the activation-side product scale * sum precomputed by the quantizer
// (acc += sa * d * c - sas * m, two FMAs and one multiply); the four-bit words of B are loaded once per K step and serve
// both sub-blocks; A fragments come from ldmatrix.x4 (one instruction instead of four 32-bit loads).
__global__ void __launch_bounds__(256) k_gemm_q8_q4k_v4(const int8_t* __restrict__ a8, const float* __restrict__ sa,
                                                        const float* __restrict__ sas, const BlockQ4K* __restrict__ w,
                                                        float* __restrict__ C, int M, int N, int K, int raster) {
  __shared__ __align__(16) int8_t As[2][kBM][kQ8Lda];
  __shared__ __align__(16) uint8_t Qs[2][kBN][kQsLd];
  __shared__ __align__(16) uint8_t Hd[2][kBN][16];
  __shared__ __align__(16) float Sa[2][kBM][2];
  __shared__ __align__(16) float Sas[2][kBM][2];
  __shared__ float Dsc[2][kBN], Dmn[2][kBN];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  // ldmatrix row and byte offset of this lane inside a 16 x 32-byte A fragment
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&As[stage][r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int r = tid >> 1, h = tid & 1;
    const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + k0 / kQK);
    cp_async16(&Qs[stage][r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    if (tid < 128) {
      cp_async16(&Hd[stage][tid][0], reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + tid) * bpr + k0 / kQK));
      cp_async8(&Sa[stage][tid][0], sa + static_cast<size_t>(bm + tid) * nkb + k0 / 32);
    } else {
      const int r2 = tid - 128;
      cp_async8(&Sas[stage][r2][0], sas + static_cast<size_t>(bm + r2) * nkb + k0 / 32);
    }
  };
  const int kt_n = K / 64;
  load(0, 0);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    const int st = kt & 1;
    const int k0 = kt * 64;
    if (kt + 1 < kt_n) {
      load(st ^ 1, (kt + 1) * 64);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    if (tid < 128) {
      const uint8_t* hb = Hd[st][tid];
      const float d = h2f(static_cast<uint16_t>(hb[0] | (hb[1] << 8)));
      const float dm = h2f(static_cast<uint16_t>(hb[2] | (hb[3] << 8)));
      const int c = (k0 % kQK) / 64;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int sc, m;
        scale_min_k4_d(2 * c + h, hb + 4, sc, m);
        Dsc[h][tid] = d * sc;
        Dmn[h][tid] = dm * m;
      }
    }
    __syncthreads();
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&Qs[st][c][4 * t]);
      bw[ni][1] = ld32(&Qs[st][c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t a[4][4], b[4][2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) ldsm_x4(a[mi], &As[st][wm * 64 + mi * 16 + lm_row][h * 32 + lm_col]);
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
      }
      float dsc[4][2], dmn[4][2];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        dsc[ni][0] = Dsc[h][col0];
        dsc[ni][1] = Dsc[h][col0 + 1];
        dmn[ni][0] = Dmn[h][col0];
        dmn[ni][1] = Dmn[h][col0 + 1];
      }
      // v4: all 16 int8 products of this sub-block first (independent, so the tensor pipe overlaps them), then the
      // scale epilogue on all 16 (v3 consumed each product right after its mma and waited on it).
      int c4[4][4][4];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
          for (int k = 0; k < 4; ++k) c4[mi][ni][k] = 0;
          mma_s8(c4[mi][ni], a[mi], b[ni]);
        }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
        const float a10 = Sa[st][r0][h], a11 = Sa[st][r1][h];
        const float a20 = Sas[st][r0][h], a21 = Sas[st][r1][h];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          acc[mi][ni][0] = fmaf(a10, dsc[ni][0] * i2f_fast(c4[mi][ni][0]), fmaf(-a20, dmn[ni][0], acc[mi][ni][0]));
          acc[mi][ni][1] = fmaf(a10, dsc[ni][1] * i2f_fast(c4[mi][ni][1]), fmaf(-a20, dmn[ni][1], acc[mi][ni][1]));
          acc[mi][ni][2] = fmaf(a11, dsc[ni][0] * i2f_fast(c4[mi][ni][2]), fmaf(-a21, dmn[ni][0], acc[mi][ni][2]));
          acc[mi][ni][3] = fmaf(a11, dsc[ni][1] * i2f_fast(c4[mi][ni][3]), fmaf(-a21, dmn[ni][1], acc[mi][ni][3]));
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 64 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v5: two blocks per SM, weight scales prepared once
// v3 and v4 measured the same speed as v2, so neither the conversion nor the product latency inside a warp was the limit.
// v5 targets occupancy and the per-step bubbles: registers capped at 128 (__launch_bounds__(256, 2)) so two blocks share
// an SM; the per-sub-block weight scales d*sc and dmin*m are prepared once per weight (k_prep_q4k_scales) and streamed
// like the activation scales, which removes the serial unpacking by half the threads and one barrier per step.
__global__ void k_prep_q4k_scales(const BlockQ4K* __restrict__ w, float* __restrict__ dsc, float* __restrict__ dmn, size_t nblocks) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < nblocks; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = w[i];
    const float d = h2f(b.d), dm = h2f(b.dmin);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      dsc[i * 8 + j] = d * sc;
      dmn[i * 8 + j] = dm * m;
    }
  }
}

__global__ void __launch_bounds__(256, 2) k_gemm_q8_q4k_v5(const int8_t* __restrict__ a8, const float* __restrict__ sa,
                                                           const float* __restrict__ sas, const BlockQ4K* __restrict__ w,
                                                           const float* __restrict__ wdsc, const float* __restrict__ wdmn,
                                                           float* __restrict__ C, int M, int N, int K, int raster) {
  __shared__ __align__(16) int8_t As[2][kBM][kQ8Lda];
  __shared__ __align__(16) uint8_t Qs[2][kBN][kQsLd];
  __shared__ __align__(16) float Sa[2][kBM][2];
  __shared__ __align__(16) float Sas[2][kBM][2];
  __shared__ __align__(16) float Dsc[2][kBN][2];
  __shared__ __align__(16) float Dmn[2][kBN][2];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&As[stage][r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int r = tid >> 1, h = tid & 1;
    const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + k0 / kQK);
    cp_async16(&Qs[stage][r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    const int kb = k0 / 32;
    if (tid < 128) {
      cp_async8(&Sa[stage][tid][0], sa + static_cast<size_t>(bm + tid) * nkb + kb);
      cp_async8(&Dsc[stage][tid][0], wdsc + static_cast<size_t>(bn + tid) * nkb + kb);
    } else {
      const int r2 = tid - 128;
      cp_async8(&Sas[stage][r2][0], sas + static_cast<size_t>(bm + r2) * nkb + kb);
      cp_async8(&Dmn[stage][r2][0], wdmn + static_cast<size_t>(bn + r2) * nkb + kb);
    }
  };
  const int kt_n = K / 64;
  load(0, 0);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    const int st = kt & 1;
    if (kt + 1 < kt_n) {
      load(st ^ 1, (kt + 1) * 64);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&Qs[st][c][4 * t]);
      bw[ni][1] = ld32(&Qs[st][c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &As[st][wm * 64 + mi * 16 + lm_row][h * 32 + lm_col]);
        const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
        const float a10 = Sa[st][r0][h], a11 = Sa[st][r1][h];
        const float a20 = Sas[st][r0][h], a21 = Sas[st][r1][h];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          const int col0 = wn * 32 + ni * 8 + 2 * t;
          const float d0 = Dsc[st][col0][h], d1 = Dsc[st][col0 + 1][h];
          const float m0 = Dmn[st][col0][h], m1 = Dmn[st][col0 + 1][h];
          acc[mi][ni][0] = fmaf(a10, d0 * i2f_fast(c4[0]), fmaf(-a20, m0, acc[mi][ni][0]));
          acc[mi][ni][1] = fmaf(a10, d1 * i2f_fast(c4[1]), fmaf(-a20, m1, acc[mi][ni][1]));
          acc[mi][ni][2] = fmaf(a11, d0 * i2f_fast(c4[2]), fmaf(-a21, m0, acc[mi][ni][2]));
          acc[mi][ni][3] = fmaf(a11, d1 * i2f_fast(c4[3]), fmaf(-a21, m1, acc[mi][ni][3]));
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 64 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v6: smaller warp tiles, no register cap
// Revision 5 measured why: v2 and v4 need 216 and 223 registers per thread, so one block fits an SM (bf16: 119, two
// blocks); capping v5 at 128 spilled 160 bytes per thread and ran 0.54x to 0.70x of v2. v6 asks the question directly:
// the same arithmetic as v5 (weight scales prepared once, fast exact conversion, ldmatrix, 4-bit words once per step)
// with smaller warp tiles, so fewer accumulators per thread and more blocks per SM without spilling.
//   variant 0: 128 x 64 tile, warps 4 x 2, warp tile 32 x 32 (MI 2, NI 4)
//   variant 1: 128 x 64 tile, warps 2 x 4, warp tile 64 x 16 (MI 4, NI 2)
//   variant 2:  64 x 64 tile, warps 2 x 4, warp tile 32 x 16 (MI 2, NI 2)
template <int WM, int WN, int MI, int NI>
__global__ void __launch_bounds__(256) k_gemm_q8_q4k_v6(const int8_t* __restrict__ a8, const float* __restrict__ sa,
                                                        const float* __restrict__ sas, const BlockQ4K* __restrict__ w,
                                                        const float* __restrict__ wdsc, const float* __restrict__ wdmn,
                                                        float* __restrict__ C, int M, int N, int K, int raster) {
  static_assert(WM * WN == 8, "8 warps");
  constexpr int BM = WM * MI * 16, BN = WN * NI * 8;
  __shared__ __align__(16) int8_t As[2][BM][kQ8Lda];
  __shared__ __align__(16) uint8_t Qs[2][BN][kQsLd];
  __shared__ __align__(16) float Sa[2][2][BM][2];  // [stage][scale or scale*sum][row][sub-block]
  __shared__ __align__(16) float Dw[2][2][BN][2];  // [stage][d*sc or dmin*m][row][sub-block]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp / WN, wn = warp % WN;
  int pm, pn;
  tile_coords(raster, M / BM, N / BN, pm, pn);
  const int bm = pm * BM, bn = pn * BN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
    for (int idx = tid; idx < BM * 4; idx += 256) {
      const int r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&As[stage][r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    for (int idx = tid; idx < BN * 2; idx += 256) {
      const int r = idx >> 1, h = idx & 1;
      const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + k0 / kQK);
      cp_async16(&Qs[stage][r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    }
    const int kb = k0 / 32;
    for (int idx = tid; idx < BM * 2; idx += 256) {
      const int which = idx / BM, r = idx % BM;
      cp_async8(&Sa[stage][which][r][0], (which ? sas : sa) + static_cast<size_t>(bm + r) * nkb + kb);
    }
    for (int idx = tid; idx < BN * 2; idx += 256) {
      const int which = idx / BN, r = idx % BN;
      cp_async8(&Dw[stage][which][r][0], (which ? wdmn : wdsc) + static_cast<size_t>(bn + r) * nkb + kb);
    }
  };
  const int kt_n = K / 64;
  load(0, 0);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    const int st = kt & 1;
    if (kt + 1 < kt_n) {
      load(st ^ 1, (kt + 1) * 64);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    uint32_t bw[NI][2];
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int c = wn * (NI * 8) + ni * 8 + g;
      bw[ni][0] = ld32(&Qs[st][c][4 * t]);
      bw[ni][1] = ld32(&Qs[st][c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[NI][2];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &As[st][wm * (MI * 16) + mi * 16 + lm_row][h * 32 + lm_col]);
        const int r0 = wm * (MI * 16) + mi * 16 + g, r1 = r0 + 8;
        const float a10 = Sa[st][0][r0][h], a11 = Sa[st][0][r1][h];
        const float a20 = Sa[st][1][r0][h], a21 = Sa[st][1][r1][h];
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          const int col0 = wn * (NI * 8) + ni * 8 + 2 * t;
          const float d0 = Dw[st][0][col0][h], d1 = Dw[st][0][col0 + 1][h];
          const float m0 = Dw[st][1][col0][h], m1 = Dw[st][1][col0 + 1][h];
          acc[mi][ni][0] = fmaf(a10, d0 * i2f_fast(c4[0]), fmaf(-a20, m0, acc[mi][ni][0]));
          acc[mi][ni][1] = fmaf(a10, d1 * i2f_fast(c4[1]), fmaf(-a20, m1, acc[mi][ni][1]));
          acc[mi][ni][2] = fmaf(a11, d0 * i2f_fast(c4[2]), fmaf(-a21, m0, acc[mi][ni][2]));
          acc[mi][ni][3] = fmaf(a11, d1 * i2f_fast(c4[3]), fmaf(-a21, m1, acc[mi][ni][3]));
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int r0 = bm + wm * (MI * 16) + mi * 16 + g;
      const int c0 = bn + wn * (NI * 8) + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v7: integer-domain epilogue, activation scale per 256
// Revisions 3 to 6 measured why every per-32 design lands at about 0.27 of the int8 peak: each m16n8k32 product (about
// 16 tensor cycles per SM sub-partition, DERIVED from P-H3) is followed by about 20 issued instructions of fp32 scale
// arithmetic, so the tensor pipe waits on the issue port. v7 moves that work into integers: activations are quantized
// with one scale per 256 values (the Q4_K super-block; the layout of llama.cpp's Q8_K, written here from scratch), so
// inside a super-block the weight sub-block scale sc_j (a 6-bit integer) is applied with one integer multiply-add per
// accumulator, and the floats d * sa are applied once per super-block. The min term keeps one fp32 FMA per accumulator per
// sub-block from precomputed products (sa * sum over the sub-block) x (dmin * m_j).
//   per-sub-block cost per accumulator: 1 IMAD + 1 FFMA (per-32 designs: about 5); exact in int32 (|sum sc_j * dot_j|
//   <= 8 * 63 * 60,960 = 30.7 M).

// Activations per row per 256: scale = amax / 127; a8; per-32 sums (int) and x32 = scale * sum (float). One warp per
// (row, super-block): each lane 8 values; the 4 lanes of a quad make one 32-value sub-block.
// One warp quantizes 256 values (8 per lane, f[] already in fp32) to int8 with one scale per 256, and writes the per-32
// integer sums, x32 = scale * sum and (v10) the bf16 digits of each sum.
__device__ __forceinline__ void q8k_warp(const float (&f)[8], int lane, size_t gidx, int8_t* __restrict__ a8, float* __restrict__ sa256,
                                         int* __restrict__ s32, float* __restrict__ x32, uint16_t* __restrict__ dig) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(f[i]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float scale = amax > 0.f ? amax / 127.f : 1.f;
  const float inv = 1.f / scale;
  int sum = 0;
  uint32_t packed[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = max(-127, min(127, __float2int_rn(f[i] * inv)));
    sum += q;
    packed[i >> 2] |= (static_cast<uint32_t>(q) & 0xFFu) << ((i & 3) * 8);
  }
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  *reinterpret_cast<uint2*>(a8 + gidx * 256 + lane * 8) = make_uint2(packed[0], packed[1]);
  if ((lane & 3) == 0) {
    s32[gidx * 8 + (lane >> 2)] = sum;
    x32[gidx * 8 + (lane >> 2)] = scale * static_cast<float>(sum);
    if (dig) {  // v10: sum = 64 * hi + lo, hi in [-64, 63], lo in [0, 63]; both exact in bf16
      const int hi = (sum >= 0) ? (sum >> 6) : -((-sum + 63) >> 6);
      const int lo = sum - 64 * hi;
      dig[gidx * 16 + (lane >> 2)] = f2bf(static_cast<float>(hi));
      dig[gidx * 16 + 8 + (lane >> 2)] = f2bf(static_cast<float>(lo));
    }
  }
  if (lane == 0) sa256[gidx] = scale;
}

__device__ __forceinline__ void bf16x8_to_f32(const uint4 v, float (&f)[8]) {
  const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = bf2f(static_cast<uint16_t>(wv[i] & 0xFFFFu));
    f[2 * i + 1] = bf2f(static_cast<uint16_t>(wv[i] >> 16));
  }
}

__global__ void k_quantize_rows_q8k(const uint16_t* __restrict__ A, int8_t* __restrict__ a8, float* __restrict__ sa256,
                                    int* __restrict__ s32, float* __restrict__ x32, uint16_t* __restrict__ dig, size_t groups) {
  const size_t warps = (static_cast<size_t>(gridDim.x) * blockDim.x) >> 5;
  const int lane = threadIdx.x & 31;
  for (size_t gidx = (blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x) >> 5; gidx < groups; gidx += warps) {
    float f[8];
    bf16x8_to_f32(*reinterpret_cast<const uint4*>(A + gidx * 256 + lane * 8), f);
    q8k_warp(f, lane, gidx, a8, sa256, s32, x32, dig);
  }
}

// ---------------------------------------------------------------- activation quantization fused into the producer
// Revision 10 showed the separate quantization pass is memory-bound once the activations leave the 36 MB L2 (436 to 454
// GB/s at 32 MiB and up, about 90 % of the 504 GB/s published VRAM bandwidth); below that its timings were L2-warm. A
// separate pass over memory cannot beat that, so the int8 path quantizes inside the kernel that produces the activations.
// The producer before every Q4_K GEMM of a layer's input is the RMSNorm, so: one block per row; the sum of squares, then
// y = x * rsqrt(mean(x^2) + eps) * gamma, written either as bf16 (what the bf16 path needs) or directly as int8 with one
// scale per 256 and the per-32 sums and digits (what v10 needs). The row's second read hits L1/L2 (at most 8 KiB).
template <bool kQ8>
__global__ void __launch_bounds__(256) k_rmsnorm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ gamma, int K, float eps,
                                                 uint16_t* __restrict__ Y, int8_t* __restrict__ a8, float* __restrict__ sa256,
                                                 int* __restrict__ s32, float* __restrict__ x32, uint16_t* __restrict__ dig) {
  __shared__ float red[8];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint16_t* x = X + static_cast<size_t>(row) * K;
  float ss = 0.f;
  for (int c = tid * 8; c < K; c += 2048) {
    float f[8];
    bf16x8_to_f32(*reinterpret_cast<const uint4*>(x + c), f);
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(f[i], f[i], ss);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rinv = rsqrtf(tot / static_cast<float>(K) + eps);
  for (int c = tid * 8; c < K; c += 2048) {  // warp-uniform: K is a multiple of 256, one warp covers 256 values
    float f[8], gm[8];
    bf16x8_to_f32(*reinterpret_cast<const uint4*>(x + c), f);
    bf16x8_to_f32(*reinterpret_cast<const uint4*>(gamma + c), gm);
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = f[i] * rinv * gm[i];
    if constexpr (kQ8) {
      q8k_warp(f, lane, static_cast<size_t>(row) * (K / 256) + c / 256, a8, sa256, s32, x32, dig);
    } else {
      uint32_t p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = static_cast<uint32_t>(f2bf(f[2 * i])) | (static_cast<uint32_t>(f2bf(f[2 * i + 1])) << 16);
      *reinterpret_cast<uint4*>(Y + static_cast<size_t>(row) * K + c) = make_uint4(p[0], p[1], p[2], p[3]);
    }
  }
}

__global__ void __launch_bounds__(256) k_gemm_q8k_q4k_v7(const int8_t* __restrict__ a8, const float* __restrict__ sa256,
                                                         const float* __restrict__ x32, const BlockQ4K* __restrict__ w,
                                                         const float* __restrict__ wdmn, float* __restrict__ C, int M, int N,
                                                         int K, int raster) {
  __shared__ __align__(16) int8_t As[2][kBM][kQ8Lda];
  __shared__ __align__(16) uint8_t Qs[2][kBN][kQsLd];
  __shared__ __align__(16) uint8_t Hd[2][kBN][16];
  __shared__ __align__(16) float Xr[2][kBM][2];   // sa256 * sum over each of the two sub-blocks, per row
  __shared__ __align__(16) float Dm[2][kBN][2];   // dmin * m_j of the two sub-blocks, per weight row
  __shared__ __align__(16) float Sr[2][kBM];      // sa256 of the current super-block, per row
  __shared__ int Isc[2][kBN];                     // integer sub-block scales sc_j of this step
  __shared__ float Dd[kBN];                       // d of the current super-block, per weight row
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[4][4][4];
  int ai[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        acc[i][j][k] = 0.f;
        ai[i][j][k] = 0;
      }
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&As[stage][r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int r = tid >> 1, h = tid & 1;
    const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + k0 / kQK);
    cp_async16(&Qs[stage][r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    const int kb = k0 / 32;
    if (tid < 128) {
      cp_async16(&Hd[stage][tid][0], reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + tid) * bpr + k0 / kQK));
      cp_async8(&Xr[stage][tid][0], x32 + static_cast<size_t>(bm + tid) * nkb + kb);
    } else {
      const int r2 = tid - 128;
      cp_async8(&Dm[stage][r2][0], wdmn + static_cast<size_t>(bn + r2) * nkb + kb);
      // sa256 of this super-block, one float per row (4-byte copy)
      const unsigned s4 = static_cast<unsigned>(__cvta_generic_to_shared(&Sr[stage][r2]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s4), "l"(sa256 + static_cast<size_t>(bm + r2) * bpr + k0 / kQK));
    }
  };
  const int kt_n = K / 64;
  load(0, 0);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    const int st = kt & 1;
    const int k0 = kt * 64;
    if (kt + 1 < kt_n) {
      load(st ^ 1, (kt + 1) * 64);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    if (tid < 128) {  // integer sub-block scales of this step; d of the super-block
      const uint8_t* hb = Hd[st][tid];
      const int c = (k0 % kQK) / 64;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int sc, m;
        scale_min_k4_d(2 * c + h, hb + 4, sc, m);
        Isc[h][tid] = sc;
      }
      Dd[tid] = h2f(static_cast<uint16_t>(hb[0] | (hb[1] << 8)));
    }
    __syncthreads();
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&Qs[st][c][4 * t]);
      bw[ni][1] = ld32(&Qs[st][c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
      int is0[4], is1[4];
      float dm0[4], dm1[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        is0[ni] = Isc[h][col0];
        is1[ni] = Isc[h][col0 + 1];
        dm0[ni] = Dm[st][col0][h];
        dm1[ni] = Dm[st][col0 + 1][h];
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &As[st][wm * 64 + mi * 16 + lm_row][h * 32 + lm_col]);
        const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
        const float x0 = Xr[st][r0][h], x1 = Xr[st][r1][h];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          ai[mi][ni][0] += is0[ni] * c4[0];
          ai[mi][ni][1] += is1[ni] * c4[1];
          ai[mi][ni][2] += is0[ni] * c4[2];
          ai[mi][ni][3] += is1[ni] * c4[3];
          acc[mi][ni][0] = fmaf(-x0, dm0[ni], acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(-x0, dm1[ni], acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(-x1, dm0[ni], acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(-x1, dm1[ni], acc[mi][ni][3]);
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {  // end of a super-block: apply sa256 * d once, in fp32
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        const float d0 = Dd[col0], d1 = Dd[col0 + 1];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
          const float s0 = Sr[st][r0], s1 = Sr[st][r1];
          acc[mi][ni][0] = fmaf(s0 * d0, static_cast<float>(ai[mi][ni][0]), acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(s0 * d1, static_cast<float>(ai[mi][ni][1]), acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(s1 * d0, static_cast<float>(ai[mi][ni][2]), acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(s1 * d1, static_cast<float>(ai[mi][ni][3]), acc[mi][ni][3]);
          ai[mi][ni][0] = ai[mi][ni][1] = ai[mi][ni][2] = ai[mi][ni][3] = 0;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 64 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v8: v7's arithmetic behind a deeper load queue
// Revision 7 measured v7 at 1.06x to 1.08x of v2, so the epilogue was about 7 % of the gap. DERIVED from that run: a
// 64-deep int8 step is about 1,024 tensor cycles per SM sub-partition (256 m16n8k32 per 128x128 tile, 4 sub-partitions,
// about 16 cycles each) against about 4,096 for the bf16 kernel's 32-deep step, and the measured int8 step took about
// 3,440 cycles: with two stages at one block per SM, the math of one step is too short to cover the load of the next.
// v8 keeps v7's arithmetic and changes only the pipeline: S stages (3 or 4) in dynamic shared memory, one barrier per
// step (v7: three), and the per-step weight header decode replaced by a 16-byte record per weight row and super-block
// (d as fp32 and the eight 6-bit scales as bytes), prepared once per weight like v5's scales.
struct alignas(16) V8Stage {
  int8_t As[kBM][kQ8Lda];
  uint8_t Qs[kBN][kQsLd];
  uint4 Wr[kBN];   // d (fp32 bits), sc[8] (bytes), unused
  float2 Xr[kBM];  // sa256 * sum over each of the two sub-blocks of this step, per row
  float2 Dm[kBN];  // dmin * m of the two sub-blocks, per weight row
  float Sr[kBM];   // sa256 of the super-block, per row
};

__global__ void k_prep_q4k_rec(const BlockQ4K* __restrict__ w, uint4* __restrict__ rec, size_t nblocks) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < nblocks; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = w[i];
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      if (j < 4) lo |= static_cast<uint32_t>(sc) << (8 * j);
      else hi |= static_cast<uint32_t>(sc) << (8 * (j - 4));
    }
    rec[i] = make_uint4(__float_as_uint(h2f(b.d)), lo, hi, __float_as_uint(h2f(b.dmin)));  // .w: dmin (v10; v8 ignores it)
  }
}

template <int S>
__global__ void __launch_bounds__(256) k_gemm_q8k_q4k_v8(const int8_t* __restrict__ a8, const float* __restrict__ sa256,
                                                         const float* __restrict__ x32, const BlockQ4K* __restrict__ w,
                                                         const uint4* __restrict__ rec, const float* __restrict__ wdmn,
                                                         float* __restrict__ C, int M, int N, int K, int raster) {
  extern __shared__ __align__(16) uint8_t v8_smem[];
  V8Stage* stg = reinterpret_cast<V8Stage*>(v8_smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[4][4][4];
  int ai[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        acc[i][j][k] = 0.f;
        ai[i][j][k] = 0;
      }
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
    V8Stage& T = stg[stage];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int r = tid >> 1, h = tid & 1;
    const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + k0 / kQK);
    cp_async16(&T.Qs[r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    const int kb = k0 / 32;
    if (tid < 128) {
      cp_async16(&T.Wr[tid], rec + static_cast<size_t>(bn + tid) * bpr + k0 / kQK);
      cp_async8(&T.Xr[tid], x32 + static_cast<size_t>(bm + tid) * nkb + kb);
    } else {
      const int r2 = tid - 128;
      cp_async8(&T.Dm[r2], wdmn + static_cast<size_t>(bn + r2) * nkb + kb);
      const unsigned s4 = static_cast<unsigned>(__cvta_generic_to_shared(&T.Sr[r2]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s4), "l"(sa256 + static_cast<size_t>(bm + r2) * bpr + k0 / kQK));
    }
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < S - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<S - 2>();
    __syncthreads();  // stage kt % S is complete for everyone, and stage (kt - 1) % S is no longer being read
    {
      const int nk = kt + S - 1;
      if (nk < kt_n) load(nk % S, nk * 64);
      cp_async_commit();
    }
    const V8Stage& T = stg[kt % S];
    const int k0 = kt * 64;
    const int j0 = (k0 % kQK) / 32;
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
      int is0[4], is1[4];
      float dm0[4], dm1[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        is0[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0])[4 + j0 + h];
        is1[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0 + 1])[4 + j0 + h];
        dm0[ni] = reinterpret_cast<const float*>(&T.Dm[col0])[h];
        dm1[ni] = reinterpret_cast<const float*>(&T.Dm[col0 + 1])[h];
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &T.As[wm * 64 + mi * 16 + lm_row][h * 32 + lm_col]);
        const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
        const float x0 = reinterpret_cast<const float*>(&T.Xr[r0])[h], x1 = reinterpret_cast<const float*>(&T.Xr[r1])[h];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          ai[mi][ni][0] += is0[ni] * c4[0];
          ai[mi][ni][1] += is1[ni] * c4[1];
          ai[mi][ni][2] += is0[ni] * c4[2];
          ai[mi][ni][3] += is1[ni] * c4[3];
          acc[mi][ni][0] = fmaf(-x0, dm0[ni], acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(-x0, dm1[ni], acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(-x1, dm0[ni], acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(-x1, dm1[ni], acc[mi][ni][3]);
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {  // end of a super-block: apply sa256 * d once, in fp32
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        const float d0 = __uint_as_float(T.Wr[col0].x), d1 = __uint_as_float(T.Wr[col0 + 1].x);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
          const float s0 = T.Sr[r0], s1 = T.Sr[r1];
          acc[mi][ni][0] = fmaf(s0 * d0, static_cast<float>(ai[mi][ni][0]), acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(s0 * d1, static_cast<float>(ai[mi][ni][1]), acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(s1 * d0, static_cast<float>(ai[mi][ni][2]), acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(s1 * d1, static_cast<float>(ai[mi][ni][3]), acc[mi][ni][3]);
          ai[mi][ni][0] = ai[mi][ni][1] = ai[mi][ni][2] = ai[mi][ni][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 64 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v9: v8 with 16 warps per tile
// Nsight Compute on revision 8 (k2prof, attn_qkv M=4096) MEASURED why every int8 version sat near 0.3 of the int8 peak:
// tensor busy = 16 cycles per m16n8k32 / (instructions per mma x cycles between a warp's instructions / warps per
// scheduler). v8: 16.8 x 5.6 / 2 = 47 cycles -> 34 % predicted, 33.2 % measured (v2, v7, v8-4 also within one point).
// All int8 versions ran 2 warps per scheduler (216 to 252 registers -> one block of 8 warps per SM). v9 keeps v8's
// arithmetic and pipeline and puts 16 warps on the same 128x128 tile (512 threads, warp tile 32x32, registers capped at
// 128 by __launch_bounds__(512, 1)), so each scheduler has 4 warps. The weight scales of a step are laid out by column
// ([sub-block][column]) so the two columns a thread needs come in one 8-byte load each (sc, dmin*m) instead of four.
struct alignas(16) V9Stage {
  int8_t As[kBM][kQ8Lda];
  uint8_t Qs[kBN][kQsLd];
  int Sc[2][kBN];     // 6-bit sub-block scales of this step's two sub-blocks, by column
  float Dm[2][kBN];   // dmin * m of the two sub-blocks, by column
  float D[kBN];       // d of the super-block, by column
  float2 Xr[kBM];     // sa256 * sum over each of the two sub-blocks, per row
  float Sr[kBM];      // sa256 of the super-block, per row
};

// Weight scales by column: scT[kb * N + n] (int), dmnT[kb * N + n] = dmin * m, dT[sb * N + n] = d. Once per weight.
__global__ void k_prep_q4k_v9(const BlockQ4K* __restrict__ w, int* __restrict__ scT, float* __restrict__ dmnT, float* __restrict__ dT,
                              int N, int K) {
  const int bpr = K / kQK;
  const size_t nblocks = static_cast<size_t>(N) * bpr;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < nblocks; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int n = static_cast<int>(i / bpr), sb = static_cast<int>(i % bpr);
    const BlockQ4K& b = w[i];
    const float dm = h2f(b.dmin);
    dT[static_cast<size_t>(sb) * N + n] = h2f(b.d);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      const size_t kb = static_cast<size_t>(sb) * 8 + j;
      scT[kb * N + n] = sc;
      dmnT[kb * N + n] = dm * m;
    }
  }
}

template <int S>
__global__ void __launch_bounds__(512, 1) k_gemm_q8k_q4k_v9(const int8_t* __restrict__ a8, const float* __restrict__ sa256,
                                                            const float* __restrict__ x32, const BlockQ4K* __restrict__ w,
                                                            const int* __restrict__ scT, const float* __restrict__ dmnT,
                                                            const float* __restrict__ dT, float* __restrict__ C, int M, int N, int K,
                                                            int raster) {
  extern __shared__ __align__(16) uint8_t v9_smem[];
  V9Stage* stg = reinterpret_cast<V9Stage*>(v9_smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;  // warp tile: rows wm*32 .. +31, columns wn*32 .. +31
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK, nkb = K / 32;
  float acc[2][4][4];
  int ai[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        acc[i][j][k] = 0.f;
        ai[i][j][k] = 0;
      }
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
    V9Stage& T = stg[stage];
    {
      const int r = tid >> 2, c = (tid & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int kb = k0 / 32, sb = k0 / kQK;
    if (tid < 256) {
      const int r = tid >> 1, h = tid & 1;
      const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + sb);
      cp_async16(&T.Qs[r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
      if (tid < 128) {
        cp_async8(&T.Xr[tid], x32 + static_cast<size_t>(bm + tid) * nkb + kb);
      } else {
        const int r2 = tid - 128;
        const unsigned s4 = static_cast<unsigned>(__cvta_generic_to_shared(&T.Sr[r2]));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s4), "l"(sa256 + static_cast<size_t>(bm + r2) * bpr + sb));
      }
    } else if (tid < 320) {
      const int q = tid - 256, h = q >> 5, c = (q & 31) * 4;
      cp_async16(&T.Sc[h][c], scT + static_cast<size_t>(kb + h) * N + bn + c);
    } else if (tid < 384) {
      const int q = tid - 320, h = q >> 5, c = (q & 31) * 4;
      cp_async16(&T.Dm[h][c], dmnT + static_cast<size_t>(kb + h) * N + bn + c);
    } else if (tid < 416) {
      const int c = (tid - 384) * 4;
      cp_async16(&T.D[c], dT + static_cast<size_t>(sb) * N + bn + c);
    }
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < S - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<S - 2>();
    __syncthreads();
    {
      const int nk = kt + S - 1;
      if (nk < kt_n) load(nk % S, nk * 64);
      cp_async_commit();
    }
    const V9Stage& T = stg[kt % S];
    const int k0 = kt * 64;
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
      int2 is[4];
      float2 dm[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        is[ni] = *reinterpret_cast<const int2*>(&T.Sc[h][col0]);
        dm[ni] = *reinterpret_cast<const float2*>(&T.Dm[h][col0]);
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &T.As[wm * 32 + mi * 16 + lm_row][h * 32 + lm_col]);
        const int r0 = wm * 32 + mi * 16 + g, r1 = r0 + 8;
        const float x0 = reinterpret_cast<const float*>(&T.Xr[r0])[h], x1 = reinterpret_cast<const float*>(&T.Xr[r1])[h];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          ai[mi][ni][0] += is[ni].x * c4[0];
          ai[mi][ni][1] += is[ni].y * c4[1];
          ai[mi][ni][2] += is[ni].x * c4[2];
          ai[mi][ni][3] += is[ni].y * c4[3];
          acc[mi][ni][0] = fmaf(-x0, dm[ni].x, acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(-x0, dm[ni].y, acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(-x1, dm[ni].x, acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(-x1, dm[ni].y, acc[mi][ni][3]);
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {  // end of a super-block: apply sa256 * d once, in fp32
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        const float2 dd = *reinterpret_cast<const float2*>(&T.D[col0]);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const int r0 = wm * 32 + mi * 16 + g, r1 = r0 + 8;
          const float s0 = T.Sr[r0], s1 = T.Sr[r1];
          acc[mi][ni][0] = fmaf(s0 * dd.x, static_cast<float>(ai[mi][ni][0]), acc[mi][ni][0]);
          acc[mi][ni][1] = fmaf(s0 * dd.y, static_cast<float>(ai[mi][ni][1]), acc[mi][ni][1]);
          acc[mi][ni][2] = fmaf(s1 * dd.x, static_cast<float>(ai[mi][ni][2]), acc[mi][ni][2]);
          acc[mi][ni][3] = fmaf(s1 * dd.y, static_cast<float>(ai[mi][ni][3]), acc[mi][ni][3]);
          ai[mi][ni][0] = ai[mi][ni][1] = ai[mi][ni][2] = ai[mi][ni][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 32 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v10: the min term on the tensor core, exactly
// Nsight Compute on revision 9 MEASURED that more warps only raise each warp's wait (v9: 4 warps per scheduler, 19.0
// instructions per mma, 9.35 cycles between a warp's instructions -> 36 % predicted, 34.9 % measured), so the lever left
// is instructions per mma. Of v8's 16.8, the min term costs 4 FFMA per mma plus 1.5 shared loads (x, dmin*m).
// v10 removes the min term from the inner loop. Per row and super-block, sum_j m_j * bsum_j (bsum_j the integer sum of a
// sub-block's int8 activations, m_j the weight's 6-bit min) is one bf16 m16n8k16 product, exact:
//   activation side, 16 bf16: hi_0..7, lo_0..7 with bsum_j = 64 * hi_j + lo_j (|hi| <= 64, 0 <= lo <= 63)
//   weight side,     16 bf16: 64 * m_0..7, m_0..7                               (<= 4,032: six significant bits)
// every value and product is an integer below 2^24 (the sum is at most 8 x 63 x 4,064 = 2.05 M), so the fp32 tensor-core
// result is the exact integer. Applied once per super-block: acc -= sa256 * dmin * that. Inner loop per mma: 1 mma + 4
// IMAD (v8: 1 mma + 4 IMAD + 4 FFMA). Base: v8 (8 warps, 128x128 tile); 2 or 3 stages.
constexpr int kDigLd = 24;  // 16 bf16 + 8 padding per row: 48-byte rows, conflict-free 4-byte fragment loads
struct alignas(16) V10Stage {
  int8_t As[kBM][kQ8Lda];
  uint8_t Qs[kBN][kQsLd];
  uint4 Wr[kBN];                // d, sc[8], dmin
  float Sr[kBM];                // sa256 of the super-block, per row
  uint16_t Ad[kBM][kDigLd];     // activation digits of the super-block (loaded on its last step only)
  uint16_t Bd[kBN][kDigLd];     // weight digits of the super-block (loaded on its last step only)
};

// Weight digits per weight row and super-block: [64 m_0 .. 64 m_7, m_0 .. m_7] as bf16. Once per weight.
__global__ void k_prep_q4k_dig(const BlockQ4K* __restrict__ w, uint16_t* __restrict__ wd, size_t nblocks) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < nblocks; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = w[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      wd[i * 16 + j] = f2bf(static_cast<float>(64 * m));
      wd[i * 16 + 8 + j] = f2bf(static_cast<float>(m));
    }
  }
}

template <int S>
__global__ void __launch_bounds__(256) k_gemm_q8k_q4k_v10(const int8_t* __restrict__ a8, const float* __restrict__ sa256,
                                                          const uint16_t* __restrict__ adig, const BlockQ4K* __restrict__ w,
                                                          const uint4* __restrict__ rec, const uint16_t* __restrict__ wdig,
                                                          float* __restrict__ C, int M, int N, int K, int raster) {
  extern __shared__ __align__(16) uint8_t v10_smem[];
  V10Stage* stg = reinterpret_cast<V10Stage*>(v10_smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK;
  float acc[4][4][4];
  int ai[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        acc[i][j][k] = 0.f;
        ai[i][j][k] = 0;
      }
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
    V10Stage& T = stg[stage];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int sb = k0 / kQK;
    {
      const int r = tid >> 1, h = tid & 1;
      const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + sb);
      cp_async16(&T.Qs[r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    }
    if (tid < 128) {
      cp_async16(&T.Wr[tid], rec + static_cast<size_t>(bn + tid) * bpr + sb);
    } else {
      const int r2 = tid - 128;
      const unsigned s4 = static_cast<unsigned>(__cvta_generic_to_shared(&T.Sr[r2]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s4), "l"(sa256 + static_cast<size_t>(bm + r2) * bpr + sb));
    }
    if (((k0 + 64) % kQK) == 0) {  // last step of a super-block: its digits (32 bytes per row, two 16-byte halves)
      const int r = tid >> 1, h = tid & 1;
      cp_async16(&T.Ad[r][h * 8], adig + (static_cast<size_t>(bm + r) * bpr + sb) * 16 + h * 8);
      cp_async16(&T.Bd[r][h * 8], wdig + (static_cast<size_t>(bn + r) * bpr + sb) * 16 + h * 8);
    }
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < S - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<S - 2>();
    __syncthreads();
    {
      const int nk = kt + S - 1;
      if (nk < kt_n) load(nk % S, nk * 64);
      cp_async_commit();
    }
    const V10Stage& T = stg[kt % S];
    const int k0 = kt * 64;
    const int j0 = (k0 % kQK) / 32;
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
      int is0[4], is1[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        is0[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0])[4 + j0 + h];
        is1[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0 + 1])[4 + j0 + h];
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &T.As[wm * 64 + mi * 16 + lm_row][h * 32 + lm_col]);
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          ai[mi][ni][0] += is0[ni] * c4[0];
          ai[mi][ni][1] += is1[ni] * c4[1];
          ai[mi][ni][2] += is0[ni] * c4[2];
          ai[mi][ni][3] += is1[ni] * c4[3];
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {  // end of a super-block: d * sa256 on the integer sum, minus dmin * sa256 * sum_j m_j bsum_j
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r0 = wm * 64 + mi * 16 + g, r1 = r0 + 8;
        uint32_t ad[4];
        ad[0] = ld32(&T.Ad[r0][2 * t]);
        ad[1] = ld32(&T.Ad[r1][2 * t]);
        ad[2] = ld32(&T.Ad[r0][2 * t + 8]);
        ad[3] = ld32(&T.Ad[r1][2 * t + 8]);
        const float s0 = T.Sr[r0], s1 = T.Sr[r1];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int cb = wn * 32 + ni * 8 + g;  // the weight row this lane supplies to the min product
          uint32_t bd[2];
          bd[0] = ld32(&T.Bd[cb][2 * t]);
          bd[1] = ld32(&T.Bd[cb][2 * t + 8]);
          float mn[4] = {0.f, 0.f, 0.f, 0.f};
          mma_bf16(mn, ad, bd);
          const int col0 = wn * 32 + ni * 8 + 2 * t;
          const uint4 w0 = T.Wr[col0], w1 = T.Wr[col0 + 1];
          const float d0 = __uint_as_float(w0.x), d1 = __uint_as_float(w1.x);
          const float m0 = __uint_as_float(w0.w), m1 = __uint_as_float(w1.w);
          acc[mi][ni][0] = fmaf(s0 * d0, static_cast<float>(ai[mi][ni][0]), fmaf(-s0 * m0, mn[0], acc[mi][ni][0]));
          acc[mi][ni][1] = fmaf(s0 * d1, static_cast<float>(ai[mi][ni][1]), fmaf(-s0 * m1, mn[1], acc[mi][ni][1]));
          acc[mi][ni][2] = fmaf(s1 * d0, static_cast<float>(ai[mi][ni][2]), fmaf(-s1 * m0, mn[2], acc[mi][ni][2]));
          acc[mi][ni][3] = fmaf(s1 * d1, static_cast<float>(ai[mi][ni][3]), fmaf(-s1 * m1, mn[3], acc[mi][ni][3]));
          ai[mi][ni][0] = ai[mi][ni][1] = ai[mi][ni][2] = ai[mi][ni][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 64 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2 v11: v10 with 16 warps per tile
// Nsight Compute on revision 10 (k2prof, attn_qkv M=4096): v10 3-stage 13.4 instructions per int8 mma (v8 16.8),
// 6.2 cycles between a warp's instructions, 2 warps per scheduler -> 13.4 x 6.2 / 2 = 41 cycles per mma, 39 % predicted,
// 37.7 % int8 tensor busy measured (plus 4.7 % for the bf16 min product). v9 tried 16 warps on v8's arithmetic and lost
// most of it to barrier and shared-memory-queue waits (1.67 and 1.73 cycles per instruction), driven by v8's per-mma
// shared loads; v10 has a third fewer of those (18.9 M against 31.5 M shared loads). v11 = v10's arithmetic with v9's
// layout: 512 threads, warp tile 32x32, registers capped at 128, the load work spread evenly over the 16 warps.
template <int S>
__global__ void __launch_bounds__(512, 1) k_gemm_q8k_q4k_v11(const int8_t* __restrict__ a8, const float* __restrict__ sa256,
                                                             const uint16_t* __restrict__ adig, const BlockQ4K* __restrict__ w,
                                                             const uint4* __restrict__ rec, const uint16_t* __restrict__ wdig,
                                                             float* __restrict__ C, int M, int N, int K, int raster) {
  extern __shared__ __align__(16) uint8_t v11_smem[];
  V10Stage* stg = reinterpret_cast<V10Stage*>(v11_smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;  // rows wm*32 .. +31, columns wn*32 .. +31
  int pm, pn;
  tile_coords(raster, M / kBM, N / kBN, pm, pn);
  const int bm = pm * kBM, bn = pn * kBN;
  const int bpr = K / kQK;
  float acc[2][4][4];
  int ai[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        acc[i][j][k] = 0.f;
        ai[i][j][k] = 0;
      }
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
    V10Stage& T = stg[stage];
    {
      const int r = tid >> 2, c = (tid & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int sb = k0 / kQK;
    if (tid < 256) {
      const int r = tid >> 1, h = tid & 1;
      const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + sb);
      cp_async16(&T.Qs[r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    } else if (tid < 384) {
      const int r = tid - 256;
      cp_async16(&T.Wr[r], rec + static_cast<size_t>(bn + r) * bpr + sb);
    } else {
      const int r2 = tid - 384;
      const unsigned s4 = static_cast<unsigned>(__cvta_generic_to_shared(&T.Sr[r2]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s4), "l"(sa256 + static_cast<size_t>(bm + r2) * bpr + sb));
    }
    if (((k0 + 64) % kQK) == 0) {
      const int q = tid & 255, r = q >> 1, h = q & 1;
      if (tid < 256) cp_async16(&T.Ad[r][h * 8], adig + (static_cast<size_t>(bm + r) * bpr + sb) * 16 + h * 8);
      else cp_async16(&T.Bd[r][h * 8], wdig + (static_cast<size_t>(bn + r) * bpr + sb) * 16 + h * 8);
    }
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < S - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<S - 2>();
    __syncthreads();
    {
      const int nk = kt + S - 1;
      if (nk < kt_n) load(nk % S, nk * 64);
      cp_async_commit();
    }
    const V10Stage& T = stg[kt % S];
    const int k0 = kt * 64;
    const int j0 = (k0 % kQK) / 32;
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
      int is0[4], is1[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        is0[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0])[4 + j0 + h];
        is1[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0 + 1])[4 + j0 + h];
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &T.As[wm * 32 + mi * 16 + lm_row][h * 32 + lm_col]);
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          ai[mi][ni][0] += is0[ni] * c4[0];
          ai[mi][ni][1] += is1[ni] * c4[1];
          ai[mi][ni][2] += is0[ni] * c4[2];
          ai[mi][ni][3] += is1[ni] * c4[3];
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r0 = wm * 32 + mi * 16 + g, r1 = r0 + 8;
        uint32_t ad[4];
        ad[0] = ld32(&T.Ad[r0][2 * t]);
        ad[1] = ld32(&T.Ad[r1][2 * t]);
        ad[2] = ld32(&T.Ad[r0][2 * t + 8]);
        ad[3] = ld32(&T.Ad[r1][2 * t + 8]);
        const float s0 = T.Sr[r0], s1 = T.Sr[r1];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int cb = wn * 32 + ni * 8 + g;
          uint32_t bd[2];
          bd[0] = ld32(&T.Bd[cb][2 * t]);
          bd[1] = ld32(&T.Bd[cb][2 * t + 8]);
          float mn[4] = {0.f, 0.f, 0.f, 0.f};
          mma_bf16(mn, ad, bd);
          const int col0 = wn * 32 + ni * 8 + 2 * t;
          const uint4 w0 = T.Wr[col0], w1 = T.Wr[col0 + 1];
          const float d0 = __uint_as_float(w0.x), d1 = __uint_as_float(w1.x);
          const float m0 = __uint_as_float(w0.w), m1 = __uint_as_float(w1.w);
          acc[mi][ni][0] = fmaf(s0 * d0, static_cast<float>(ai[mi][ni][0]), fmaf(-s0 * m0, mn[0], acc[mi][ni][0]));
          acc[mi][ni][1] = fmaf(s0 * d1, static_cast<float>(ai[mi][ni][1]), fmaf(-s0 * m1, mn[1], acc[mi][ni][1]));
          acc[mi][ni][2] = fmaf(s1 * d0, static_cast<float>(ai[mi][ni][2]), fmaf(-s1 * m0, mn[2], acc[mi][ni][2]));
          acc[mi][ni][3] = fmaf(s1 * d1, static_cast<float>(ai[mi][ni][3]), fmaf(-s1 * m1, mn[3], acc[mi][ni][3]));
          ai[mi][ni][0] = ai[mi][ni][1] = ai[mi][ni][2] = ai[mi][ni][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * 32 + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0) * N + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + static_cast<size_t>(r0 + 8) * N + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

// ---------------------------------------------------------------- P-K2b v12: v10 with split-K, a 64-row tile, grouped
// problems and a bf16 output option (revision 12)
// Revision 11 left two measured losses outside the per-mma cost: (1) low-tile shapes idle SMs: router and attn_k at
// M = 2048 / 4096 make 32 or 64 tiles of 128x128 for 46 SMs, about 70 % busy (DERIVED; measured 68 to 72 % of the big
// shapes); (2) expert down (K = 512) writes 16 MiB of fp32 results against about 1.6 MiB of inputs at M = N = 2048.
// v12 = v10's arithmetic with: split-K (blockIdx.y = split, each a whole number of super-blocks, fp32 partials summed by
// a second kernel), a 64-row tile (warp tile 32x32, for M = 64 experts), grouped problems (blockIdx.z = expert: own rows
// of A, own weights, own rows of C; one launch for all experts), and bf16 output (half the result bytes).
template <int BMT>
struct alignas(16) V12Stage {
  int8_t As[BMT][kQ8Lda];
  uint8_t Qs[kBN][kQsLd];
  uint4 Wr[kBN];
  float Sr[BMT];
  uint16_t Ad[BMT][kDigLd];
  uint16_t Bd[kBN][kDigLd];
};

template <int S, int BMT>
__global__ void __launch_bounds__(256, BMT == 64 ? 2 : 1) k_gemm_q8k_q4k_v12(const int8_t* __restrict__ a8, const float* __restrict__ sa256,
                                                                              const uint16_t* __restrict__ adig, const BlockQ4K* __restrict__ w,
                                                                              const uint4* __restrict__ rec, const uint16_t* __restrict__ wdig,
                                                                              void* __restrict__ Cout, int M, int N, int K, int raster, int out_bf16) {
  constexpr int MI = BMT / 32;  // 16-row fragments per warp (warp rows BMT / 2)
  extern __shared__ __align__(16) uint8_t v12_smem[];
  V12Stage<BMT>* stg = reinterpret_cast<V12Stage<BMT>*>(v12_smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int bpr = K / kQK;
  const int splits = static_cast<int>(gridDim.y), split = static_cast<int>(blockIdx.y), z = static_cast<int>(blockIdx.z);
  a8 += static_cast<size_t>(z) * M * K;
  sa256 += static_cast<size_t>(z) * M * bpr;
  adig += static_cast<size_t>(z) * M * bpr * 16;
  w += static_cast<size_t>(z) * N * bpr;
  rec += static_cast<size_t>(z) * N * bpr;
  wdig += static_cast<size_t>(z) * N * bpr * 16;
  int pm, pn;
  tile_coords(raster, M / BMT, N / kBN, pm, pn);
  const int bm = pm * BMT, bn = pn * kBN;
  const int kspan = K / splits, kbeg = split * kspan;
  float acc[MI][4][4];
  int ai[MI][4][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        acc[i][j][k] = 0.f;
        ai[i][j][k] = 0;
      }
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);

  auto load = [&](int stage, int k0) {
    V12Stage<BMT>& T = stg[stage];
#pragma unroll
    for (int i = 0; i < BMT / 64; ++i) {
      const int idx = tid + i * 256, r = idx >> 2, c = (idx & 3) * 16;
      cp_async16(&T.As[r][c], a8 + static_cast<size_t>(bm + r) * K + k0 + c);
    }
    const int sb = k0 / kQK;
    {
      const int r = tid >> 1, h = tid & 1;
      const uint8_t* blk = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(bn + r) * bpr + sb);
      cp_async16(&T.Qs[r][h * 16], blk + 16 + ((k0 % kQK) / 64) * 32 + h * 16);
    }
    if (tid < 128) {
      cp_async16(&T.Wr[tid], rec + static_cast<size_t>(bn + tid) * bpr + sb);
    } else if (tid < 128 + BMT) {
      const int r2 = tid - 128;
      const unsigned s4 = static_cast<unsigned>(__cvta_generic_to_shared(&T.Sr[r2]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s4), "l"(sa256 + static_cast<size_t>(bm + r2) * bpr + sb));
    }
    if (((k0 + 64) % kQK) == 0) {
      const int r = tid >> 1, h = tid & 1;
      if (r < BMT) cp_async16(&T.Ad[r][h * 8], adig + (static_cast<size_t>(bm + r) * bpr + sb) * 16 + h * 8);
      cp_async16(&T.Bd[r][h * 8], wdig + (static_cast<size_t>(bn + r) * bpr + sb) * 16 + h * 8);
    }
  };
  const int kt_n = kspan / 64;
#pragma unroll
  for (int st = 0; st < S - 1; ++st) {
    if (st < kt_n) load(st, kbeg + st * 64);
    cp_async_commit();
  }
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<S - 2>();
    __syncthreads();
    {
      const int nk = kt + S - 1;
      if (nk < kt_n) load(nk % S, kbeg + nk * 64);
      cp_async_commit();
    }
    const V12Stage<BMT>& T = stg[kt % S];
    const int k0 = kbeg + kt * 64;
    const int j0 = (k0 % kQK) / 32;
    uint32_t bw[4][2];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = wn * 32 + ni * 8 + g;
      bw[ni][0] = ld32(&T.Qs[c][4 * t]);
      bw[ni][1] = ld32(&T.Qs[c][16 + 4 * t]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b[4][2];
      int is0[4], is1[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b[ni][0] = h ? ((bw[ni][0] >> 4) & 0x0F0F0F0Fu) : (bw[ni][0] & 0x0F0F0F0Fu);
        b[ni][1] = h ? ((bw[ni][1] >> 4) & 0x0F0F0F0Fu) : (bw[ni][1] & 0x0F0F0F0Fu);
        const int col0 = wn * 32 + ni * 8 + 2 * t;
        is0[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0])[4 + j0 + h];
        is1[ni] = reinterpret_cast<const uint8_t*>(&T.Wr[col0 + 1])[4 + j0 + h];
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        uint32_t a[4];
        ldsm_x4(a, &T.As[wm * (BMT / 2) + mi * 16 + lm_row][h * 32 + lm_col]);
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          int c4[4] = {0, 0, 0, 0};
          mma_s8(c4, a, b[ni]);
          ai[mi][ni][0] += is0[ni] * c4[0];
          ai[mi][ni][1] += is1[ni] * c4[1];
          ai[mi][ni][2] += is0[ni] * c4[2];
          ai[mi][ni][3] += is1[ni] * c4[3];
        }
      }
    }
    if (((k0 + 64) % kQK) == 0) {
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r0 = wm * (BMT / 2) + mi * 16 + g, r1 = r0 + 8;
        uint32_t ad[4];
        ad[0] = ld32(&T.Ad[r0][2 * t]);
        ad[1] = ld32(&T.Ad[r1][2 * t]);
        ad[2] = ld32(&T.Ad[r0][2 * t + 8]);
        ad[3] = ld32(&T.Ad[r1][2 * t + 8]);
        const float s0 = T.Sr[r0], s1 = T.Sr[r1];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int cb = wn * 32 + ni * 8 + g;
          uint32_t bd[2];
          bd[0] = ld32(&T.Bd[cb][2 * t]);
          bd[1] = ld32(&T.Bd[cb][2 * t + 8]);
          float mn[4] = {0.f, 0.f, 0.f, 0.f};
          mma_bf16(mn, ad, bd);
          const int col0 = wn * 32 + ni * 8 + 2 * t;
          const uint4 w0 = T.Wr[col0], w1 = T.Wr[col0 + 1];
          const float d0 = __uint_as_float(w0.x), d1 = __uint_as_float(w1.x);
          const float m0 = __uint_as_float(w0.w), m1 = __uint_as_float(w1.w);
          acc[mi][ni][0] = fmaf(s0 * d0, static_cast<float>(ai[mi][ni][0]), fmaf(-s0 * m0, mn[0], acc[mi][ni][0]));
          acc[mi][ni][1] = fmaf(s0 * d1, static_cast<float>(ai[mi][ni][1]), fmaf(-s0 * m1, mn[1], acc[mi][ni][1]));
          acc[mi][ni][2] = fmaf(s1 * d0, static_cast<float>(ai[mi][ni][2]), fmaf(-s1 * m0, mn[2], acc[mi][ni][2]));
          acc[mi][ni][3] = fmaf(s1 * d1, static_cast<float>(ai[mi][ni][3]), fmaf(-s1 * m1, mn[3], acc[mi][ni][3]));
          ai[mi][ni][0] = ai[mi][ni][1] = ai[mi][ni][2] = ai[mi][ni][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();
  // output: one split writes the final result (fp32 or bf16); several write fp32 partials [split][z][M][N]
  const size_t mn = static_cast<size_t>(M) * N;
  const size_t zoff = (splits > 1 ? (static_cast<size_t>(split) * gridDim.z + z) : static_cast<size_t>(z)) * mn;
  const bool bf = out_bf16 && splits == 1;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r0 = bm + wm * (BMT / 2) + mi * 16 + g;
      const int c0 = bn + wn * 32 + ni * 8 + 2 * t;
      const size_t i0 = zoff + static_cast<size_t>(r0) * N + c0, i1 = i0 + static_cast<size_t>(8) * N;
      if (bf) {
        uint16_t* C = static_cast<uint16_t*>(Cout);
        *reinterpret_cast<uint32_t*>(C + i0) = static_cast<uint32_t>(f2bf(acc[mi][ni][0])) | (static_cast<uint32_t>(f2bf(acc[mi][ni][1])) << 16);
        *reinterpret_cast<uint32_t*>(C + i1) = static_cast<uint32_t>(f2bf(acc[mi][ni][2])) | (static_cast<uint32_t>(f2bf(acc[mi][ni][3])) << 16);
      } else {
        float* C = static_cast<float*>(Cout);
        *reinterpret_cast<float2*>(C + i0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<float2*>(C + i1) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    }
}

// Split-K: out[i] = sum over splits of part[s * n + i], as fp32 or bf16; n a multiple of 4.
__global__ void k_splitk_reduce(const float* __restrict__ part, int splits, size_t n, void* __restrict__ out, int out_bf16) {
  const size_t n4 = n / 4;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n4; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    float4 a = reinterpret_cast<const float4*>(part)[i];
    for (int sp = 1; sp < splits; ++sp) {
      const float4 b = reinterpret_cast<const float4*>(part + static_cast<size_t>(sp) * n)[i];
      a.x += b.x;
      a.y += b.y;
      a.z += b.z;
      a.w += b.w;
    }
    if (out_bf16) {
      reinterpret_cast<uint2*>(out)[i] = make_uint2(static_cast<uint32_t>(f2bf(a.x)) | (static_cast<uint32_t>(f2bf(a.y)) << 16),
                                                    static_cast<uint32_t>(f2bf(a.z)) | (static_cast<uint32_t>(f2bf(a.w)) << 16));
    } else {
      reinterpret_cast<float4*>(out)[i] = a;
    }
  }
}

// fp32 reference: C[r, n] for r < rows, n < ncols, from the dequantized int8 activations and dequantized Q4_K weights.
__global__ void k_ref_q8_q4k(const int8_t* __restrict__ a8, const float* __restrict__ sa, const BlockQ4K* __restrict__ w,
                             float* __restrict__ C, int rows, int ncols, int K, int ga) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (n >= ncols || r >= rows) return;
  const int bpr = K / kQK, nga = K / ga;
  float s = 0.f;
  for (int bk = 0; bk < bpr; ++bk) {
    const BlockQ4K& b = w[static_cast<size_t>(n) * bpr + bk];
    const float d = h2f(b.d), dm = h2f(b.dmin);
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      const float ds = d * sc, mn = dm * m;
      const float as = sa[static_cast<size_t>(r) * nga + (bk * kQK + j * 32) / ga];
      const int8_t* x = a8 + static_cast<size_t>(r) * K + bk * kQK + j * 32;
      const uint8_t* q = b.qs + (j >> 1) * 32;
      const int shift = (j & 1) * 4;
      for (int l = 0; l < 32; ++l) s += (as * static_cast<float>(x[l])) * (ds * static_cast<float>((q[l] >> shift) & 0xF) - mn);
    }
  }
  C[static_cast<size_t>(r) * ncols + n] = s;
}

// ---------------------------------------------------------------- P-K3: streaming dequantization (no mma)
// One warp takes 32 consecutive blocks: the warp copies them into shared memory with 16-byte loads, then every lane
// dequantizes 8 consecutive weights of each block (conflict-free reads of one block at a time).
template <int kWarps>
__global__ void __launch_bounds__(kWarps * 32) k_stream_q4k(const BlockQ4K* __restrict__ w, size_t nblocks, float* sink) {
  __shared__ __align__(16) uint8_t sm[kWarps][32 * kQ4KBytes];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t groups = nblocks / 32;
  const size_t gstride = static_cast<size_t>(gridDim.x) * kWarps;
  float s = 0.f;
  for (size_t gi = static_cast<size_t>(blockIdx.x) * kWarps + warp; gi < groups; gi += gstride) {
    const uint4* src = reinterpret_cast<const uint4*>(w + gi * 32);
    uint4* dst = reinterpret_cast<uint4*>(sm[warp]);
    for (int i = lane; i < 32 * kQ4KBytes / 16; i += 32) dst[i] = src[i];
    __syncwarp();
    const int j = lane >> 2, l0 = (lane & 3) * 8, shift = (j & 1) * 4;
    for (int b = 0; b < 32; ++b) {
      const BlockQ4K& blk = *reinterpret_cast<const BlockQ4K*>(sm[warp] + b * kQ4KBytes);
      int sc, m;
      scale_min_k4_d(j, blk.scales, sc, m);
      const float d = h2f(blk.d) * sc, mn = h2f(blk.dmin) * m;
      const uint8_t* q = blk.qs + (j >> 1) * 32 + l0;
#pragma unroll
      for (int l = 0; l < 8; ++l) s += d * static_cast<float>((q[l] >> shift) & 0xF) - mn;
    }
    __syncwarp();
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int kWarps>
__global__ void __launch_bounds__(kWarps * 32) k_stream_q6k(const BlockQ6K* __restrict__ w, size_t nblocks, float* sink) {
  __shared__ __align__(16) uint8_t sm[kWarps][32 * kQ6KBytes + 16];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t groups = nblocks / 32;
  const size_t gstride = static_cast<size_t>(gridDim.x) * kWarps;
  float s = 0.f;
  // 32 blocks of 210 bytes = 6,720 bytes = 420 aligned 16-byte loads (32 * 210 is a multiple of 16)
  for (size_t gi = static_cast<size_t>(blockIdx.x) * kWarps + warp; gi < groups; gi += gstride) {
    const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(w) + gi * 32 * kQ6KBytes);
    uint4* dst = reinterpret_cast<uint4*>(sm[warp]);
    for (int i = lane; i < 32 * kQ6KBytes / 16; i += 32) dst[i] = src[i];
    __syncwarp();
    const int e0 = lane * 8, half = e0 >> 7, qd = (e0 & 127) >> 5, l0 = e0 & 31, is = l0 >> 4;
    for (int b = 0; b < 32; ++b) {
      const uint8_t* base = sm[warp] + b * kQ6KBytes;
      const uint8_t* ql = base + half * 64 + ((qd & 1) ? 32 : 0);
      const uint8_t* qh = base + 128 + half * 32;
      const int8_t sc = static_cast<int8_t>(base[192 + half * 8 + is + 2 * qd]);
      const float d = h2f(static_cast<uint16_t>(base[208] | (base[209] << 8))) * sc;
      const int lshift = (qd >= 2) ? 4 : 0, hshift = 2 * qd;
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int q = (((ql[l0 + l] >> lshift) & 0xF) | (((qh[l0 + l] >> hshift) & 3) << 4)) - 32;
        s += d * static_cast<float>(q);
      }
    }
    __syncwarp();
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void k_dequant_q4k_f32(const BlockQ4K* w, float* out, size_t nblocks) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= nblocks) return;
  const BlockQ4K& b = w[i];
  for (int j = 0; j < 8; ++j) {
    int sc, m;
    scale_min_k4_d(j, b.scales, sc, m);
    const float d = h2f(b.d) * sc, mn = h2f(b.dmin) * m;
    const uint8_t* q = b.qs + (j >> 1) * 32;
    const int shift = (j & 1) * 4;
    for (int l = 0; l < 32; ++l) out[i * kQK + j * 32 + l] = d * static_cast<float>((q[l] >> shift) & 0xF) - mn;
  }
}
__global__ void k_dequant_q6k_f32(const BlockQ6K* w, float* out, size_t nblocks) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= nblocks) return;
  const uint8_t* base = reinterpret_cast<const uint8_t*>(&w[i]);
  const float d0 = h2f(static_cast<uint16_t>(base[208] | (base[209] << 8)));
  for (int e = 0; e < kQK; ++e) {
    const int half = e >> 7, qd = (e & 127) >> 5, l = e & 31, is = l >> 4;
    const uint8_t* ql = base + half * 64 + ((qd & 1) ? 32 : 0);
    const uint8_t* qh = base + 128 + half * 32;
    const int8_t sc = static_cast<int8_t>(base[192 + half * 8 + is + 2 * qd]);
    const int q = (((ql[l] >> ((qd >= 2) ? 4 : 0)) & 0xF) | (((qh[l] >> (2 * qd)) & 3) << 4)) - 32;
    out[i * kQK + e] = d0 * sc * static_cast<float>(q);
  }
}

// ---------------------------------------------------------------- P-Q10: fp8 round trip per row, fp32 GEMM
__global__ void __launch_bounds__(256) k_fp8_roundtrip(const float* __restrict__ x, float* __restrict__ y, int N, int kind) {
  __shared__ float red[256];
  const float* row = x + static_cast<size_t>(blockIdx.x) * N;
  float* out = y + static_cast<size_t>(blockIdx.x) * N;
  float amax = 0.f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) amax = fmaxf(amax, fabsf(row[i]));
  red[threadIdx.x] = amax;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float fmax_fmt = kind == 0 ? 448.f : 57344.f;
  const float scale = red[0] > 0.f ? red[0] / fmax_fmt : 1.f;
  const float inv = 1.f / scale;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v;
    if (kind == 0) v = static_cast<float>(__nv_fp8_e4m3(row[i] * inv));
    else v = static_cast<float>(__nv_fp8_e5m2(row[i] * inv));
    out[i] = v * scale;
  }
}

constexpr int kT = 16;
__global__ void k_gemm_f32_nn(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[kT][kT + 1], Bs[kT][kT + 1];
  const int r = blockIdx.y * kT + threadIdx.y, c = blockIdx.x * kT + threadIdx.x;
  float s = 0.f;
  for (int k0 = 0; k0 < N; k0 += kT) {
    As[threadIdx.y][threadIdx.x] = A[static_cast<size_t>(r) * N + k0 + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[static_cast<size_t>(k0 + threadIdx.y) * K + c];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kT; ++k) s = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], s);
    __syncthreads();
  }
  (void)M;
  C[static_cast<size_t>(r) * K + c] = s;
}

__global__ void k_bf16_to_f32(const uint16_t* in, float* out, size_t n) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    out[i] = bf2f(in[i]);
}

// ---------------------------------------------------------------- comparisons
__global__ void k_max_abs_diff(const float* a, const float* b, size_t n, unsigned* out) {
  float md = 0.f, mb = 0.f;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float d = fabsf(a[i] - b[i]);
    md = (d != d) ? 3.0e38f : fmaxf(md, d);  // NaN counts as a huge difference
    mb = fmaxf(mb, fabsf(b[i]));
  }
  for (int o = 16; o > 0; o >>= 1) {
    md = fmaxf(md, __shfl_down_sync(0xffffffffu, md, o));
    mb = fmaxf(mb, __shfl_down_sync(0xffffffffu, mb, o));
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(out, __float_as_uint(md));  // non-negative floats order like their bits
    atomicMax(out + 1, __float_as_uint(mb));
  }
}

__global__ void __launch_bounds__(256) k_dot3(const float* a, const float* b, size_t n, double* out) {
  __shared__ double r[3][256];
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const double x = a[i], y = b[i];
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  r[0][threadIdx.x] = ab;
  r[1][threadIdx.x] = aa;
  r[2][threadIdx.x] = bb;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      for (int k = 0; k < 3; ++k) r[k][threadIdx.x] += r[k][threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < 3; ++k) atomicAdd(out + k, r[k][0]);
}

inline int grid_for(size_t n, int threads, int cap = 4096) {
  const size_t b = (n + threads - 1) / threads;
  return static_cast<int>(b < 1 ? 1 : (b > static_cast<size_t>(cap) ? cap : b));
}

}  // namespace

// ================================================================ host wrappers
cudaError_t fill_q4k(BlockQ4K* w, size_t n, uint32_t seed, cudaStream_t s) {
  k_fill_q4k<<<grid_for(n, 256), 256, 0, s>>>(w, n, seed);
  return cudaGetLastError();
}
cudaError_t fill_q6k(BlockQ6K* w, size_t n, uint32_t seed, cudaStream_t s) {
  k_fill_q6k<<<grid_for(n, 256), 256, 0, s>>>(w, n, seed);
  return cudaGetLastError();
}
cudaError_t fill_bf16(uint16_t* p, size_t n, uint32_t seed, cudaStream_t s) {
  k_fill_bf16<<<grid_for(n, 256), 256, 0, s>>>(p, n, seed);
  return cudaGetLastError();
}
cudaError_t fill_f32(float* p, size_t n, uint32_t seed, int dist, float scale, cudaStream_t s) {
  k_fill_f32<<<grid_for(n, 256), 256, 0, s>>>(p, n, seed, dist, scale);
  return cudaGetLastError();
}
cudaError_t dequant_q4k_to_bf16(const BlockQ4K* w, uint16_t* wout, int N, int K, cudaStream_t s) {
  const size_t nblocks = static_cast<size_t>(N) * (K / kQK);
  if (nblocks % 32) return cudaErrorInvalidValue;
  const size_t groups = nblocks / 32;
  k_dequant_q4k_bf16<8><<<static_cast<int>(std::min<size_t>((groups + 7) / 8, 8192)), 256, 0, s>>>(w, wout, nblocks);
  return cudaGetLastError();
}
cudaError_t gemm_bf16_tiled(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K, int cfg, int raster, cudaStream_t s) {
  if (K % kBK) return cudaErrorInvalidValue;
  switch (cfg) {
    case 0:
      if (M % 128 || N % 128) return cudaErrorInvalidValue;
      k_gemm_bf16_t<2, 4, 4, 4><<<(M / 128) * (N / 128), 256, 0, s>>>(A, B, C, M, N, K, raster);
      break;
    case 1:
      if (M % 128 || N % 64) return cudaErrorInvalidValue;
      k_gemm_bf16_t<4, 2, 2, 4><<<(M / 128) * (N / 64), 256, 0, s>>>(A, B, C, M, N, K, raster);
      break;
    case 2:
      if (M % 64 || N % 64) return cudaErrorInvalidValue;
      k_gemm_bf16_t<2, 4, 2, 2><<<(M / 64) * (N / 64), 256, 0, s>>>(A, B, C, M, N, K, raster);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
cudaError_t quantize_rows_q8(const uint16_t* A, int8_t* a8, float* sa, int* s8, float* sas, int M, int K, cudaStream_t s) {
  if (K % 32) return cudaErrorInvalidValue;
  const size_t groups = static_cast<size_t>(M) * (K / 32);
  k_quantize_rows_q8<<<grid_for(groups * 4, 256, 1 << 16), 256, 0, s>>>(A, a8, sa, s8, sas, groups);
  return cudaGetLastError();
}
cudaError_t gemm_q8_q4k(const int8_t* a8, const float* sa, const int* s8, const BlockQ4K* w, float* C, int M, int N, int K,
                        int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK) return cudaErrorInvalidValue;
  k_gemm_q8_q4k<<<(M / kBM) * (N / kBN), 256, 0, s>>>(a8, sa, s8, w, C, M, N, K, raster);
  return cudaGetLastError();
}
cudaError_t gemm_q8_q4k_v4(const int8_t* a8, const float* sa, const float* sas, const BlockQ4K* w, float* C, int M, int N, int K,
                           int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK) return cudaErrorInvalidValue;
  k_gemm_q8_q4k_v4<<<(M / kBM) * (N / kBN), 256, 0, s>>>(a8, sa, sas, w, C, M, N, K, raster);
  return cudaGetLastError();
}
cudaError_t prep_q4k_scales(const BlockQ4K* w, float* dsc, float* dmn, int N, int K, cudaStream_t s) {
  const size_t nblocks = static_cast<size_t>(N) * (K / kQK);
  k_prep_q4k_scales<<<grid_for(nblocks, 256, 1 << 16), 256, 0, s>>>(w, dsc, dmn, nblocks);
  return cudaGetLastError();
}
cudaError_t gemm_q8_q4k_v5(const int8_t* a8, const float* sa, const float* sas, const BlockQ4K* w, const float* dsc, const float* dmn,
                           float* C, int M, int N, int K, int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK) return cudaErrorInvalidValue;
  k_gemm_q8_q4k_v5<<<(M / kBM) * (N / kBN), 256, 0, s>>>(a8, sa, sas, w, dsc, dmn, C, M, N, K, raster);
  return cudaGetLastError();
}
cudaError_t gemm_q8_q4k_v6(const int8_t* a8, const float* sa, const float* sas, const BlockQ4K* w, const float* dsc, const float* dmn,
                           float* C, int M, int N, int K, int variant, int raster, cudaStream_t s) {
  if (K % kQK) return cudaErrorInvalidValue;
  switch (variant) {
    case 0:
      if (M % 128 || N % 64) return cudaErrorInvalidValue;
      k_gemm_q8_q4k_v6<4, 2, 2, 4><<<(M / 128) * (N / 64), 256, 0, s>>>(a8, sa, sas, w, dsc, dmn, C, M, N, K, raster);
      break;
    case 1:
      if (M % 128 || N % 64) return cudaErrorInvalidValue;
      k_gemm_q8_q4k_v6<2, 4, 4, 2><<<(M / 128) * (N / 64), 256, 0, s>>>(a8, sa, sas, w, dsc, dmn, C, M, N, K, raster);
      break;
    case 2:
      if (M % 64 || N % 64) return cudaErrorInvalidValue;
      k_gemm_q8_q4k_v6<2, 4, 2, 2><<<(M / 64) * (N / 64), 256, 0, s>>>(a8, sa, sas, w, dsc, dmn, C, M, N, K, raster);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
cudaError_t quantize_rows_q8k(const uint16_t* A, int8_t* a8, float* sa256, int* s32, float* x32, int M, int K, cudaStream_t s) {
  if (K % kQK) return cudaErrorInvalidValue;
  const size_t groups = static_cast<size_t>(M) * (K / kQK);
  k_quantize_rows_q8k<<<grid_for(groups * 32, 256, 1 << 16), 256, 0, s>>>(A, a8, sa256, s32, x32, nullptr, groups);
  return cudaGetLastError();
}
cudaError_t gemm_q8k_q4k_v7(const int8_t* a8, const float* sa256, const float* x32, const BlockQ4K* w, const float* dmn, float* C,
                            int M, int N, int K, int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK) return cudaErrorInvalidValue;
  k_gemm_q8k_q4k_v7<<<(M / kBM) * (N / kBN), 256, 0, s>>>(a8, sa256, x32, w, dmn, C, M, N, K, raster);
  return cudaGetLastError();
}
cudaError_t prep_q4k_rec(const BlockQ4K* w, void* rec, int N, int K, cudaStream_t s) {
  if (K % kQK) return cudaErrorInvalidValue;
  const size_t n = static_cast<size_t>(N) * (K / kQK);
  k_prep_q4k_rec<<<grid_for(n, 256), 256, 0, s>>>(w, static_cast<uint4*>(rec), n);
  return cudaGetLastError();
}
namespace {
template <int S>
const void* v8_fn() { return reinterpret_cast<const void*>(&k_gemm_q8k_q4k_v8<S>); }
template <int S>
cudaError_t v8_ready() {  // dynamic shared memory above 48 KiB needs an opt-in, once per kernel
  static cudaError_t once = cudaFuncSetAttribute(k_gemm_q8k_q4k_v8<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(S * sizeof(V8Stage)));
  return once;
}
}  // namespace
int v8_stages(int variant) { return variant == 0 ? 3 : 4; }
int v8_smem_bytes(int variant) { return static_cast<int>(v8_stages(variant) * sizeof(V8Stage)); }
cudaError_t gemm_q8k_q4k_v8(const int8_t* a8, const float* sa256, const float* x32, const BlockQ4K* w, const void* rec, const float* dmn,
                            float* C, int M, int N, int K, int variant, int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK || variant < 0 || variant > 1) return cudaErrorInvalidValue;
  const dim3 grid((M / kBM) * (N / kBN));
  const uint4* r = static_cast<const uint4*>(rec);
  if (variant == 0) {
    const cudaError_t e = v8_ready<3>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v8<3><<<grid, 256, 3 * sizeof(V8Stage), s>>>(a8, sa256, x32, w, r, dmn, C, M, N, K, raster);
  } else {
    const cudaError_t e = v8_ready<4>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v8<4><<<grid, 256, 4 * sizeof(V8Stage), s>>>(a8, sa256, x32, w, r, dmn, C, M, N, K, raster);
  }
  return cudaGetLastError();
}
cudaError_t prep_q4k_v9(const BlockQ4K* w, int* scT, float* dmnT, float* dT, int N, int K, cudaStream_t s) {
  if (K % kQK) return cudaErrorInvalidValue;
  const size_t n = static_cast<size_t>(N) * (K / kQK);
  k_prep_q4k_v9<<<grid_for(n, 256), 256, 0, s>>>(w, scT, dmnT, dT, N, K);
  return cudaGetLastError();
}
namespace {
template <int S>
const void* v9_fn() { return reinterpret_cast<const void*>(&k_gemm_q8k_q4k_v9<S>); }
template <int S>
cudaError_t v9_ready() {
  static cudaError_t once = cudaFuncSetAttribute(k_gemm_q8k_q4k_v9<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(S * sizeof(V9Stage)));
  return once;
}
}  // namespace
int v9_stages(int variant) { return variant == 0 ? 2 : 3; }
int v9_smem_bytes(int variant) { return static_cast<int>(v9_stages(variant) * sizeof(V9Stage)); }
cudaError_t gemm_q8k_q4k_v9(const int8_t* a8, const float* sa256, const float* x32, const BlockQ4K* w, const int* scT, const float* dmnT,
                            const float* dT, float* C, int M, int N, int K, int variant, int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK || variant < 0 || variant > 1) return cudaErrorInvalidValue;
  const dim3 grid((M / kBM) * (N / kBN));
  if (variant == 0) {
    const cudaError_t e = v9_ready<2>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v9<2><<<grid, 512, 2 * sizeof(V9Stage), s>>>(a8, sa256, x32, w, scT, dmnT, dT, C, M, N, K, raster);
  } else {
    const cudaError_t e = v9_ready<3>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v9<3><<<grid, 512, 3 * sizeof(V9Stage), s>>>(a8, sa256, x32, w, scT, dmnT, dT, C, M, N, K, raster);
  }
  return cudaGetLastError();
}
cudaError_t quantize_rows_q8k_dig(const uint16_t* A, int8_t* a8, float* sa256, int* s32, float* x32, uint16_t* dig, int M, int K, cudaStream_t s) {
  if (K % kQK) return cudaErrorInvalidValue;
  const size_t groups = static_cast<size_t>(M) * (K / kQK);
  k_quantize_rows_q8k<<<grid_for(groups * 32, 256, 1 << 16), 256, 0, s>>>(A, a8, sa256, s32, x32, dig, groups);
  return cudaGetLastError();
}
cudaError_t prep_q4k_dig(const BlockQ4K* w, uint16_t* wd, int N, int K, cudaStream_t s) {
  if (K % kQK) return cudaErrorInvalidValue;
  const size_t n = static_cast<size_t>(N) * (K / kQK);
  k_prep_q4k_dig<<<grid_for(n, 256), 256, 0, s>>>(w, wd, n);
  return cudaGetLastError();
}
namespace {
template <int S>
const void* v10_fn() { return reinterpret_cast<const void*>(&k_gemm_q8k_q4k_v10<S>); }
template <int S>
cudaError_t v10_ready() {
  static cudaError_t once = cudaFuncSetAttribute(k_gemm_q8k_q4k_v10<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(S * sizeof(V10Stage)));
  return once;
}
}  // namespace
int v10_stages(int variant) { return variant == 0 ? 2 : 3; }
int v10_smem_bytes(int variant) { return static_cast<int>(v10_stages(variant) * sizeof(V10Stage)); }
cudaError_t gemm_q8k_q4k_v10(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             float* C, int M, int N, int K, int variant, int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK || variant < 0 || variant > 1) return cudaErrorInvalidValue;
  const dim3 grid((M / kBM) * (N / kBN));
  const uint4* r = static_cast<const uint4*>(rec);
  if (variant == 0) {
    const cudaError_t e = v10_ready<2>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v10<2><<<grid, 256, 2 * sizeof(V10Stage), s>>>(a8, sa256, adig, w, r, wdig, C, M, N, K, raster);
  } else {
    const cudaError_t e = v10_ready<3>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v10<3><<<grid, 256, 3 * sizeof(V10Stage), s>>>(a8, sa256, adig, w, r, wdig, C, M, N, K, raster);
  }
  return cudaGetLastError();
}
cudaError_t rmsnorm_bf16(const uint16_t* X, const uint16_t* gamma, uint16_t* Y, int M, int K, cudaStream_t s) {
  if (K % 256) return cudaErrorInvalidValue;
  k_rmsnorm<false><<<M, 256, 0, s>>>(X, gamma, K, 1e-6f, Y, nullptr, nullptr, nullptr, nullptr, nullptr);
  return cudaGetLastError();
}
cudaError_t rmsnorm_q8k(const uint16_t* X, const uint16_t* gamma, int8_t* a8, float* sa256, int* s32, float* x32, uint16_t* dig, int M, int K,
                        cudaStream_t s) {
  if (K % 256) return cudaErrorInvalidValue;
  k_rmsnorm<true><<<M, 256, 0, s>>>(X, gamma, K, 1e-6f, nullptr, a8, sa256, s32, x32, dig);
  return cudaGetLastError();
}
namespace {
template <int S>
const void* v11_fn() { return reinterpret_cast<const void*>(&k_gemm_q8k_q4k_v11<S>); }
template <int S>
cudaError_t v11_ready() {
  static cudaError_t once = cudaFuncSetAttribute(k_gemm_q8k_q4k_v11<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(S * sizeof(V10Stage)));
  return once;
}
}  // namespace
int v11_smem_bytes(int variant) { return static_cast<int>((variant == 0 ? 2 : 3) * sizeof(V10Stage)); }
cudaError_t gemm_q8k_q4k_v11(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             float* C, int M, int N, int K, int variant, int raster, cudaStream_t s) {
  if (M % kBM || N % kBN || K % kQK || variant < 0 || variant > 1) return cudaErrorInvalidValue;
  const dim3 grid((M / kBM) * (N / kBN));
  const uint4* r = static_cast<const uint4*>(rec);
  if (variant == 0) {
    const cudaError_t e = v11_ready<2>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v11<2><<<grid, 512, 2 * sizeof(V10Stage), s>>>(a8, sa256, adig, w, r, wdig, C, M, N, K, raster);
  } else {
    const cudaError_t e = v11_ready<3>();
    if (e != cudaSuccess) return e;
    k_gemm_q8k_q4k_v11<3><<<grid, 512, 3 * sizeof(V10Stage), s>>>(a8, sa256, adig, w, r, wdig, C, M, N, K, raster);
  }
  return cudaGetLastError();
}
cudaError_t gemm_bf16_batched(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K, int batch, int cfg, int raster, cudaStream_t s) {
  if (K % kBK || batch < 1) return cudaErrorInvalidValue;
  const size_t sA = static_cast<size_t>(M) * K, sB = static_cast<size_t>(N) * K, sC = static_cast<size_t>(M) * N;
  switch (cfg) {
    case 0:
      if (M % 128 || N % 128) return cudaErrorInvalidValue;
      k_gemm_bf16_t<2, 4, 4, 4><<<dim3((M / 128) * (N / 128), batch), 256, 0, s>>>(A, B, C, M, N, K, raster, sA, sB, sC);
      break;
    case 1:
      if (M % 128 || N % 64) return cudaErrorInvalidValue;
      k_gemm_bf16_t<4, 2, 2, 4><<<dim3((M / 128) * (N / 64), batch), 256, 0, s>>>(A, B, C, M, N, K, raster, sA, sB, sC);
      break;
    case 2:
      if (M % 64 || N % 64) return cudaErrorInvalidValue;
      k_gemm_bf16_t<2, 4, 2, 2><<<dim3((M / 64) * (N / 64), batch), 256, 0, s>>>(A, B, C, M, N, K, raster, sA, sB, sC);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
namespace {
template <int S, int BMT>
const void* v12_fn() { return reinterpret_cast<const void*>(&k_gemm_q8k_q4k_v12<S, BMT>); }
template <int S, int BMT>
cudaError_t v12_ready() {
  static cudaError_t once = cudaFuncSetAttribute(k_gemm_q8k_q4k_v12<S, BMT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(S * sizeof(V12Stage<BMT>)));
  return once;
}
template <int S, int BMT>
cudaError_t v12_launch(dim3 grid, const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const uint4* rec,
                       const uint16_t* wdig, void* C, int M, int N, int K, int raster, int out_bf16, cudaStream_t s) {
  const cudaError_t e = v12_ready<S, BMT>();
  if (e != cudaSuccess) return e;
  k_gemm_q8k_q4k_v12<S, BMT><<<grid, 256, S * sizeof(V12Stage<BMT>), s>>>(a8, sa256, adig, w, rec, wdig, C, M, N, K, raster, out_bf16);
  return cudaGetLastError();
}
}  // namespace
const char* v12_name(int variant) {
  static const char* n[] = {"v12 128x128 2-stage", "v12 128x128 3-stage", "v12 64x128 2-stage", "v12 64x128 3-stage"};
  return (variant >= 0 && variant < kV12Variants) ? n[variant] : "?";
}
int v12_tile_m(int variant) { return variant < 2 ? 128 : 64; }
size_t splitk_partial_bytes(int M, int N, int batch, int splits) { return splits > 1 ? static_cast<size_t>(splits) * batch * M * N * 4 : 0; }
cudaError_t gemm_q8k_q4k_v12(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             void* C, int M, int N, int K, int batch, int splits, int out_bf16, float* partial, int variant, int raster, cudaStream_t s) {
  const int bm = v12_tile_m(variant);
  if (M % bm || N % kBN || K % kQK || batch < 1 || splits < 1 || (K / kQK) % splits || variant < 0 || variant >= kV12Variants) return cudaErrorInvalidValue;
  if (splits > 1 && !partial) return cudaErrorInvalidValue;
  const dim3 grid((M / bm) * (N / kBN), splits, batch);
  const uint4* r = static_cast<const uint4*>(rec);
  void* dst = splits > 1 ? static_cast<void*>(partial) : C;
  cudaError_t e;
  switch (variant) {
    case 0: e = v12_launch<2, 128>(grid, a8, sa256, adig, w, r, wdig, dst, M, N, K, raster, out_bf16, s); break;
    case 1: e = v12_launch<3, 128>(grid, a8, sa256, adig, w, r, wdig, dst, M, N, K, raster, out_bf16, s); break;
    case 2: e = v12_launch<2, 64>(grid, a8, sa256, adig, w, r, wdig, dst, M, N, K, raster, out_bf16, s); break;
    default: e = v12_launch<3, 64>(grid, a8, sa256, adig, w, r, wdig, dst, M, N, K, raster, out_bf16, s); break;
  }
  if (e != cudaSuccess || splits == 1) return e;
  const size_t n = static_cast<size_t>(batch) * M * N;
  k_splitk_reduce<<<grid_for(n / 4, 256), 256, 0, s>>>(partial, splits, n, C, out_bf16);
  return cudaGetLastError();
}
cudaError_t kernel_info(int which, int* regs, int* local_bytes, int* static_smem, int* blocks_per_sm) {
  const void* f = nullptr;
  switch (which) {
    case 0: f = reinterpret_cast<const void*>(&k_gemm_bf16_t<2, 4, 4, 4>); break;
    case 1: f = reinterpret_cast<const void*>(&k_gemm_bf16_t<4, 2, 2, 4>); break;
    case 2: f = reinterpret_cast<const void*>(&k_gemm_bf16_t<2, 4, 2, 2>); break;
    case 3: f = reinterpret_cast<const void*>(&k_gemm_q8_q4k); break;
    case 4: f = reinterpret_cast<const void*>(&k_gemm_q8_q4k_v4); break;
    case 5: f = reinterpret_cast<const void*>(&k_gemm_q8_q4k_v5); break;
    case 6: f = reinterpret_cast<const void*>(&k_gemm_q8_q4k_v6<4, 2, 2, 4>); break;
    case 7: f = reinterpret_cast<const void*>(&k_gemm_q8_q4k_v6<2, 4, 4, 2>); break;
    case 8: f = reinterpret_cast<const void*>(&k_gemm_q8_q4k_v6<2, 4, 2, 2>); break;
    case 9: f = reinterpret_cast<const void*>(&k_gemm_q8k_q4k_v7); break;
    case 10: f = v8_fn<3>(); break;
    case 11: f = v8_fn<4>(); break;
    case 12: f = v9_fn<2>(); break;
    case 13: f = v9_fn<3>(); break;
    case 14: f = v10_fn<2>(); break;
    case 15: f = v10_fn<3>(); break;
    case 16: f = v11_fn<2>(); break;
    case 17: f = v11_fn<3>(); break;
    case 18: f = v12_fn<2, 128>(); break;
    case 19: f = v12_fn<3, 128>(); break;
    case 20: f = v12_fn<2, 64>(); break;
    case 21: f = v12_fn<3, 64>(); break;
    default: return cudaErrorInvalidValue;
  }
  int dyn = 0, threads = 256;
  if (which == 10 || which == 11) {  // the opt-in must be set before the occupancy query
    const cudaError_t e = which == 10 ? v8_ready<3>() : v8_ready<4>();
    if (e != cudaSuccess) return e;
    dyn = v8_smem_bytes(which - 10);
  }
  if (which == 12 || which == 13) {
    const cudaError_t e = which == 12 ? v9_ready<2>() : v9_ready<3>();
    if (e != cudaSuccess) return e;
    dyn = v9_smem_bytes(which - 12);
    threads = 512;
  }
  if (which == 14 || which == 15) {
    const cudaError_t e = which == 14 ? v10_ready<2>() : v10_ready<3>();
    if (e != cudaSuccess) return e;
    dyn = v10_smem_bytes(which - 14);
  }
  if (which == 16 || which == 17) {
    const cudaError_t e = which == 16 ? v11_ready<2>() : v11_ready<3>();
    if (e != cudaSuccess) return e;
    dyn = v11_smem_bytes(which - 16);
    threads = 512;
  }
  if (which >= 18 && which <= 21) {
    cudaError_t e = cudaSuccess;
    switch (which) {
      case 18: e = v12_ready<2, 128>(); dyn = 2 * sizeof(V12Stage<128>); break;
      case 19: e = v12_ready<3, 128>(); dyn = 3 * sizeof(V12Stage<128>); break;
      case 20: e = v12_ready<2, 64>(); dyn = 2 * sizeof(V12Stage<64>); break;
      default: e = v12_ready<3, 64>(); dyn = 3 * sizeof(V12Stage<64>); break;
    }
    if (e != cudaSuccess) return e;
  }
  cudaFuncAttributes fa{};
  cudaError_t e = cudaFuncGetAttributes(&fa, f);
  if (e != cudaSuccess) return e;
  *regs = fa.numRegs;
  *local_bytes = static_cast<int>(fa.localSizeBytes);
  *static_smem = static_cast<int>(fa.sharedSizeBytes) + dyn;  // static plus dynamic
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, f, threads, dyn);
}
cudaError_t ref_q8_q4k_rows(const int8_t* a8, const float* sa, const BlockQ4K* w, float* C, int rows, int ncols, int K,
                            cudaStream_t s) {
  k_ref_q8_q4k<<<dim3((ncols + 127) / 128, rows), 128, 0, s>>>(a8, sa, w, C, rows, ncols, K, 32);
  return cudaGetLastError();
}
cudaError_t ref_q8k_q4k_rows(const int8_t* a8, const float* sa256, const BlockQ4K* w, float* C, int rows, int ncols, int K, cudaStream_t s) {
  k_ref_q8_q4k<<<dim3((ncols + 127) / 128, rows), 128, 0, s>>>(a8, sa256, w, C, rows, ncols, K, 256);
  return cudaGetLastError();
}
cudaError_t stream_dequant_q4k(const BlockQ4K* w, size_t nblocks, float* sink, int blocks, cudaStream_t s) {
  k_stream_q4k<8><<<blocks, 256, 0, s>>>(w, nblocks, sink);
  return cudaGetLastError();
}
cudaError_t stream_dequant_q6k(const BlockQ6K* w, size_t nblocks, float* sink, int blocks, cudaStream_t s) {
  k_stream_q6k<4><<<blocks, 128, 0, s>>>(w, nblocks, sink);
  return cudaGetLastError();
}
cudaError_t dequant_q4k_f32(const BlockQ4K* w, float* out, size_t nblocks, cudaStream_t s) {
  k_dequant_q4k_f32<<<grid_for(nblocks, 64), 64, 0, s>>>(w, out, nblocks);
  return cudaGetLastError();
}
cudaError_t dequant_q6k_f32(const BlockQ6K* w, float* out, size_t nblocks, cudaStream_t s) {
  k_dequant_q6k_f32<<<grid_for(nblocks, 64), 64, 0, s>>>(w, out, nblocks);
  return cudaGetLastError();
}
cudaError_t fp8_roundtrip_rows(const float* x, float* y, int M, int N, int kind, cudaStream_t s) {
  k_fp8_roundtrip<<<M, 256, 0, s>>>(x, y, N, kind);
  return cudaGetLastError();
}
cudaError_t gemm_f32_nn(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  if (M % kT || N % kT || K % kT) return cudaErrorInvalidValue;
  k_gemm_f32_nn<<<dim3(K / kT, M / kT), dim3(kT, kT), 0, s>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}
cudaError_t bf16_to_f32(const uint16_t* in, float* out, size_t n, cudaStream_t s) {
  k_bf16_to_f32<<<grid_for(n, 256), 256, 0, s>>>(in, out, n);
  return cudaGetLastError();
}
cudaError_t max_abs_diff(const float* a, const float* b, size_t n, unsigned* out, cudaStream_t s) {
  k_max_abs_diff<<<grid_for(n, 256, 2048), 256, 0, s>>>(a, b, n, out);
  return cudaGetLastError();
}
cudaError_t dot3(const float* a, const float* b, size_t n, double* out, cudaStream_t s) {
  k_dot3<<<grid_for(n, 256, 1024), 256, 0, s>>>(a, b, n, out);
  return cudaGetLastError();
}

}  // namespace pkc

// provenance: original (the Q4_K kernels are kern/cuda/gpu_prompt.cu's k_tile / k_vec / k_prep_* at 87bbddb (bf16
// two-digit min term) and ef9fb0b (fp32 min term), store-only epilogue; the Q4_TG2 kernels are new, 2026-10-30 22:2x)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kernels_tg2.cu - see kernels_tg2.h.
#include "kernels_tg2.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstring>

namespace ptg {
namespace {

constexpr int kStages = 3, kQ8Lda = 64 + 16, kQsLd = 32 + 16, kMinLdB = 48;  // kMinLdB: 48-byte rows (24 bf16 / 12 fp32)

struct BlockQ4K {
  std::uint16_t d, dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K is 144 bytes");
constexpr int kTgBytes = 144;  // Q4_TG2: d (2), levels (6), tags (8), codes (128)

__device__ __forceinline__ float h2f(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ std::uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bf2f(std::uint16_t b) { return __bfloat162float(__ushort_as_bfloat16(b)); }
__device__ __forceinline__ std::uint32_t ld32(const void* p) { return *reinterpret_cast<const std::uint32_t*>(p); }
__device__ __forceinline__ std::uint32_t ld16(const void* p) { return *reinterpret_cast<const std::uint16_t*>(p); }
__device__ __forceinline__ void scale_min_k4_d(int j, const std::uint8_t* q, int& d, int& m) {  // get_scale_min_k4 (ggml-quants.c)
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
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
__device__ __forceinline__ void mma_s8u8(int (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// two signed int8 (the low 16 bits of v) to an fp16 pair, exact: 0x6400 | (x + 128) is 1024 + x + 128 in fp16
__device__ __forceinline__ std::uint32_t i8pair_h2(std::uint32_t v) {
  const std::uint32_t p = __byte_perm(v ^ 0x8080u, 0x64646464u, 0x4140);
  const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&p), __float2half2_rn(1152.0f));
  return *reinterpret_cast<const std::uint32_t*>(&h);
}
// byte d of each of e0..e3 into one word (e0's in byte 0)
template <int D>
__device__ __forceinline__ std::uint32_t pick4(std::uint32_t e0, std::uint32_t e1, std::uint32_t e2, std::uint32_t e3) {
  constexpr unsigned sel = static_cast<unsigned>(D | ((D + 4) << 4));
  return __byte_perm(__byte_perm(e0, e1, sel), __byte_perm(e2, e3, sel), 0x5410);
}
// signed int8 byte b of w to float, exact (magic number 2^23)
__device__ __forceinline__ float i8f(std::uint32_t w, int b) {
  return __uint_as_float(__byte_perm(w ^ 0x80808080u, 0x4B000000u, 0x7540u | static_cast<unsigned>(b))) - 8388736.0f;
}

// ------------------------------------------------------------------------------------------------ Q4_TG2 side tables

__global__ void k_tg2_prep(const std::uint8_t* __restrict__ w, float* __restrict__ sc, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint8_t* b = w + i * kTgBytes;
    const float d = h2f(static_cast<std::uint16_t>(b[0] | (b[1] << 8)));
    std::uint64_t lv = 0;
#pragma unroll
    for (int k = 0; k < 6; ++k) lv |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
#pragma unroll
    for (int j = 0; j < 8; ++j) sc[i * 8 + j] = __fdiv_rn(__fmul_rn(d, static_cast<float>((lv >> (6 * j)) & 63u)), 63.0f);  // quant q8_z_scale
  }
}

// ------------------------------------------------------------------------------------------------ Q4_TG2 tensor-core tiles

struct alignas(16) TgStage {
  std::int8_t As[kTile][kQ8Lda];
  std::uint8_t Qs[kBN][kQsLd];  // 32 code bytes of the 64-value step
  std::uint8_t Tg[kBN][8];      // the eight tag bytes of the super-block
  float Sc[kBN][2];             // s_j of the step's two sub-blocks
  float S2[kTile][2];           // activation scales of the step's two sub-blocks
};
constexpr int kTgSmem = kStages * static_cast<int>(sizeof(TgStage));

struct TgArgs {
  const std::int8_t* a8;
  const float* sa32;
  const std::uint8_t* w;
  const float* sc;
  const std::uint32_t* lut;
  const int4* units;
  int nunits, N, K;
  float* C;
};

__device__ __forceinline__ void tg_load(TgStage& T, const TgArgs& A, int ex, int bm, int bn, int k0, int tid) {
  const int bpr = A.K / kQK, n32 = A.K / 32, sb = k0 / kQK, j0 = (k0 % kQK) / 32;
  const std::uint8_t* wb = A.w + static_cast<std::size_t>(ex) * A.N * bpr * kTgBytes;
  {
    const int r = tid >> 2, c = (tid & 3) * 16;
    cp_async16(&T.As[r][c], A.a8 + static_cast<std::size_t>(bm + r) * A.K + k0 + c);
  }
  {
    const int r = tid >> 1, h = tid & 1;
    cp_async16(&T.Qs[r][h * 16], wb + (static_cast<std::size_t>(bn + r) * bpr + sb) * kTgBytes + 16 + (k0 % kQK) / 2 + h * 16);
  }
  if (tid < kBN) {
    cp_async8(&T.Tg[tid][0], wb + (static_cast<std::size_t>(bn + tid) * bpr + sb) * kTgBytes + 8);
  } else {
    const int r = tid - kBN;
    cp_async8(&T.Sc[r][0], A.sc + ((static_cast<std::size_t>(ex) * A.N + bn + r) * bpr + sb) * 8 + j0);
  }
  if (tid < kTile) cp_async8(&T.S2[tid][0], A.sa32 + static_cast<std::size_t>(bm + tid) * n32 + k0 / 32);
}

template <int V>
__global__ void __launch_bounds__(256, 2) k_tg2_tile(TgArgs A) {
  constexpr int NI = 4, WR = 32, MI = 2;
  constexpr int ND = V == kI8x3 ? 3 : (V == kF16 ? 1 : 2);  // digits
  constexpr bool kF16v = V == kF16 || V == kF16HiLo;
  extern __shared__ __align__(16) std::uint8_t g_smem[];
  __shared__ std::uint32_t lut[64];
  TgStage* stg = reinterpret_cast<TgStage*>(g_smem);
  const int tid = threadIdx.x;
  if (tid < 64) lut[tid] = A.lut[tid];
  if (static_cast<int>(blockIdx.y) >= A.nunits) return;
  const int4 u = A.units[blockIdx.y];
  const int ex = u.x, bm = u.y, vrows = u.z, bn = static_cast<int>(blockIdx.x) * kBN;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, wm = warp / 4, wn = warp % 4;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);
  const int kt_n = A.K / 64;
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < kt_n) tg_load(stg[st], A, ex, bm, bn, st * 64, tid);
    cp_async_commit();
  }
  const float inv = V == kI8x2 ? 1.0f / 32767.0f : (V == kI8x3 ? 1.0f / 8388607.0f : 1.0f);
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nk = kt + kStages - 1;
      if (nk < kt_n) tg_load(stg[nk % kStages], A, ex, bm, bn, nk * 64, tid);
      cp_async_commit();
    }
    const TgStage& T = stg[kt % kStages];
    const int j0 = ((kt * 64) % kQK) / 32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int j = j0 + h;
      // B fragments of the 4 column groups: [ni][kk or 0][digit][2]
      std::uint32_t b[NI][kF16v ? 2 : 1][ND][2];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * NI * 8 + ni * 8 + g;
        const unsigned tb = T.Tg[c][j];
        if constexpr (kF16v) {
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const unsigned q0 = T.Qs[c][16 * h + 8 * kk + t], q1 = T.Qs[c][16 * h + 8 * kk + t + 4];
            const unsigned t0 = (tb >> (4 * kk)) & 3u, t1 = (tb >> (4 * kk + 2)) & 3u;
            const std::uint32_t e0 = lut[t0 * 16 + (q0 & 15u)], e1 = lut[t0 * 16 + (q0 >> 4)];
            const std::uint32_t f0 = lut[t1 * 16 + (q1 & 15u)], f1 = lut[t1 * 16 + (q1 >> 4)];
            b[ni][kk][0][0] = __byte_perm(e0, e1, 0x5410);
            b[ni][kk][0][1] = __byte_perm(f0, f1, 0x5410);
            if constexpr (ND == 2) {
              b[ni][kk][1][0] = __byte_perm(e0, e1, 0x7632);
              b[ni][kk][1][1] = __byte_perm(f0, f1, 0x7632);
            }
          }
        } else {
          const std::uint32_t w0 = ld16(&T.Qs[c][16 * h + 2 * t]), w1 = ld16(&T.Qs[c][16 * h + 8 + 2 * t]);
          const unsigned ta = (tb >> (2 * (t >> 1))) & 3u, tz = (tb >> (2 * (2 + (t >> 1)))) & 3u;
          std::uint32_t e[4], f[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            e[i] = lut[ta * 16 + ((w0 >> (4 * i)) & 15u)];
            f[i] = lut[tz * 16 + ((w1 >> (4 * i)) & 15u)];
          }
          b[ni][0][0][0] = pick4<0>(e[0], e[1], e[2], e[3]);
          b[ni][0][0][1] = pick4<0>(f[0], f[1], f[2], f[3]);
          b[ni][0][1][0] = pick4<1>(e[0], e[1], e[2], e[3]);
          b[ni][0][1][1] = pick4<1>(f[0], f[1], f[2], f[3]);
          if constexpr (ND == 3) {
            b[ni][0][2][0] = pick4<2>(e[0], e[1], e[2], e[3]);
            b[ni][0][2][1] = pick4<2>(f[0], f[1], f[2], f[3]);
          }
        }
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r0 = wm * WR + mi * 16 + g;
        const float xs0 = T.S2[r0][h], xs1 = T.S2[r0 + 8][h];
        std::uint32_t a[kF16v ? 2 : 1][4];
        if constexpr (kF16v) {
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const int kb = 32 * h + 16 * kk + 2 * t;
            a[kk][0] = i8pair_h2(ld16(&T.As[r0][kb]));
            a[kk][1] = i8pair_h2(ld16(&T.As[r0 + 8][kb]));
            a[kk][2] = i8pair_h2(ld16(&T.As[r0][kb + 8]));
            a[kk][3] = i8pair_h2(ld16(&T.As[r0 + 8][kb + 8]));
          }
        } else {
          ldsm_x4(a[0], &T.As[wm * WR + mi * 16 + lm_row][h * 32 + lm_col]);
        }
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
          const float s0 = T.Sc[col0][h], s1 = T.Sc[col0 + 1][h];
          float D[4];
          if constexpr (kF16v) {
            float dd[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
              if constexpr (ND == 2) mma_f16(dd, a[kk], b[ni][kk][1]);  // low digit first (smaller terms)
              mma_f16(dd, a[kk], b[ni][kk][0]);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) D[q] = dd[q];
          } else if constexpr (ND == 2) {  // 16-bit fixed point: s8 high byte, u8 low byte, combined exactly in int32
            int dh[4] = {0, 0, 0, 0}, dl[4] = {0, 0, 0, 0};
            mma_s8(dh, a[0], b[ni][0][1]);
            mma_s8u8(dl, a[0], b[ni][0][0]);
#pragma unroll
            for (int q = 0; q < 4; ++q) D[q] = static_cast<float>(dh[q] * 256 + dl[q]);
          } else {  // 24-bit fixed point: s8, u8, u8; |dh| < 2^20 and the others < 2^21: exact floats, combined in fp32
            int dh[4] = {0, 0, 0, 0}, dm[4] = {0, 0, 0, 0}, dl[4] = {0, 0, 0, 0};
            mma_s8(dh, a[0], b[ni][0][2]);
            mma_s8u8(dm, a[0], b[ni][0][1]);
            mma_s8u8(dl, a[0], b[ni][0][0]);
#pragma unroll
            for (int q = 0; q < 4; ++q) D[q] = fmaf(static_cast<float>(dh[q]), 65536.0f, fmaf(static_cast<float>(dm[q]), 256.0f, static_cast<float>(dl[q])));
          }
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
      const int lr = wm * WR + mi * 16 + g;
      const int c0 = bn + wn * NI * 8 + ni * 8 + 2 * t;
      if (lr < vrows) {
        A.C[static_cast<std::size_t>(bm + lr) * A.N + c0] = acc[mi][ni][0];
        A.C[static_cast<std::size_t>(bm + lr) * A.N + c0 + 1] = acc[mi][ni][1];
      }
      if (lr + 8 < vrows) {
        A.C[static_cast<std::size_t>(bm + lr + 8) * A.N + c0] = acc[mi][ni][2];
        A.C[static_cast<std::size_t>(bm + lr + 8) * A.N + c0 + 1] = acc[mi][ni][3];
      }
    }
}

// exact path: weights through the fp32 table into shared memory (x s_j), activations x scale, fp32 FMAs on CUDA cores;
// 2 raw stages + one decoded buffer; each thread 4 rows x 8 columns
constexpr int kWfLd = kBN + 4, kAfLd = kTile + 4;
struct alignas(16) F32Dec {
  float Wf[64][kWfLd];
  float Af[64][kAfLd];
};
constexpr int kF32Smem = 2 * static_cast<int>(sizeof(TgStage)) + static_cast<int>(sizeof(F32Dec));

__global__ void __launch_bounds__(256, 1) k_tg2_f32(TgArgs A) {
  extern __shared__ __align__(16) std::uint8_t g_smem[];
  __shared__ float lutf[64];
  TgStage* stg = reinterpret_cast<TgStage*>(g_smem);
  F32Dec& Dc = *reinterpret_cast<F32Dec*>(g_smem + 2 * sizeof(TgStage));
  const int tid = threadIdx.x;
  if (tid < 64) lutf[tid] = __uint_as_float(A.lut[tid]);
  if (static_cast<int>(blockIdx.y) >= A.nunits) return;
  const int4 u = A.units[blockIdx.y];
  const int ex = u.x, bm = u.y, vrows = u.z, bn = static_cast<int>(blockIdx.x) * kBN;
  const int ty = tid >> 4, tx = tid & 15;
  float acc[4][8];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  const int kt_n = A.K / 64;
  tg_load(stg[0], A, ex, bm, bn, 0, tid);
  cp_async_commit();
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<0>();
    __syncthreads();
    if (kt + 1 < kt_n) tg_load(stg[(kt + 1) & 1], A, ex, bm, bn, (kt + 1) * 64, tid);
    cp_async_commit();
    const TgStage& T = stg[kt & 1];
    const int j0 = ((kt * 64) % kQK) / 32;
    {  // decode the weights: thread = (column, sub-block)
      const int c = tid >> 1, h = tid & 1;
      const float s = T.Sc[c][h];
      const unsigned tb = T.Tg[c][j0 + h];
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        const unsigned code = (static_cast<unsigned>(T.Qs[c][16 * h + i / 2]) >> (4 * (i & 1))) & 15u;
        const unsigned tg = (tb >> (2 * (i >> 3))) & 3u;
        Dc.Wf[32 * h + i][c] = s * lutf[tg * 16 + code];
      }
    }
    {  // the activations: thread = (row, 16 values)
      const int r = tid >> 2, k0 = (tid & 3) * 16;
      const float xs = T.S2[r][k0 / 32];
#pragma unroll
      for (int i = 0; i < 16; ++i) Dc.Af[k0 + i][r] = xs * static_cast<float>(T.As[r][k0 + i]);
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < 64; ++k) {
      const float4 a = *reinterpret_cast<const float4*>(&Dc.Af[k][ty * 4]);
      const float4 w0 = *reinterpret_cast<const float4*>(&Dc.Wf[k][tx * 4]);
      const float4 w1 = *reinterpret_cast<const float4*>(&Dc.Wf[k][64 + tx * 4]);
      const float av[4] = {a.x, a.y, a.z, a.w}, wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int jx = 0; jx < 8; ++jx) acc[i][jx] = fmaf(av[i], wv[jx], acc[i][jx]);
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int lr = ty * 4 + i;
    if (lr >= vrows) continue;
#pragma unroll
    for (int jx = 0; jx < 8; ++jx) {
      const int c = bn + (jx < 4 ? tx * 4 + jx : 64 + tx * 4 + jx - 4);
      A.C[static_cast<std::size_t>(bm + lr) * A.N + c] = acc[i][jx];
    }
  }
}

// warp-vector path for 1-8 rows, exact: one warp per output column, lanes over the 32-blocks; weights decoded through
// the fp32 table once per 32-block, then 32 FMAs per row (activations to float exactly by the magic number)
__global__ void __launch_bounds__(256) k_tg2_vec(TgArgs A) {
  __shared__ float lutf[64];
  if (threadIdx.x < 64) lutf[threadIdx.x] = __uint_as_float(A.lut[threadIdx.x]);
  __syncthreads();
  if (static_cast<int>(blockIdx.x) >= A.nunits) return;
  const int4 u = A.units[blockIdx.x];
  const int lane = threadIdx.x & 31, n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= A.N) return;
  const int bpr = A.K / kQK, n32 = A.K / 32, rows = u.z;
  const std::size_t wrow = static_cast<std::size_t>(u.x) * A.N + n;
  const std::uint8_t* wr = A.w + wrow * bpr * kTgBytes;
  float acc[kVecMax];
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) acc[r] = 0.f;
  for (int jj = lane; jj < n32; jj += 32) {
    const std::uint8_t* b = wr + static_cast<std::size_t>(jj >> 3) * kTgBytes;
    const int j = jj & 7;
    const float s = A.sc[(wrow * bpr + (jj >> 3)) * 8 + j];
    const unsigned tb = b[8 + j];
    const uint4 cq = *reinterpret_cast<const uint4*>(b + 16 + j * 16);
    const std::uint32_t cw[4] = {cq.x, cq.y, cq.z, cq.w};
    float wv[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) wv[i] = lutf[((tb >> (2 * (i >> 3))) & 3u) * 16 + ((cw[i >> 3] >> (4 * (i & 7))) & 15u)];
#pragma unroll
    for (int r = 0; r < kVecMax; ++r) {
      if (r >= rows) break;
      const std::int8_t* xr = A.a8 + static_cast<std::size_t>(u.y + r) * A.K + jj * 32;
      const uint4 xa = *reinterpret_cast<const uint4*>(xr), xb = *reinterpret_cast<const uint4*>(xr + 16);
      const std::uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) sum = fmaf(wv[i], i8f(xw[i >> 2], i & 3), sum);
      acc[r] = fmaf(A.sa32[static_cast<std::size_t>(u.y + r) * n32 + jj] * s, sum, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kVecMax; ++r) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
  if (lane == 0)
    for (int r = 0; r < rows; ++r) A.C[static_cast<std::size_t>(u.y + r) * A.N + n] = acc[r];
}

// ------------------------------------------------------------------------------------------------ Q4_K (gpu_prompt's kernels)

__global__ void k_q4k_prep(const BlockQ4K* __restrict__ w, uint4* __restrict__ rec, std::uint16_t* __restrict__ wb, float* __restrict__ wf, std::size_t nb) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < nb; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const BlockQ4K& b = w[i];
    std::uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      scale_min_k4_d(j, b.scales, sc, m);
      if (j < 4) lo |= static_cast<std::uint32_t>(sc) << (8 * j);
      else hi |= static_cast<std::uint32_t>(sc) << (8 * (j - 4));
      wb[i * 16 + j] = f2bf(static_cast<float>(m));  // v10: digits (m, m)
      wb[i * 16 + 8 + j] = f2bf(static_cast<float>(m));
      wf[i * 8 + j] = h2f(b.dmin) * static_cast<float>(m);  // v10b: dmin x m
    }
    rec[i] = make_uint4(__float_as_uint(h2f(b.d)), lo, hi, __float_as_uint(h2f(b.dmin)));
  }
}
__global__ void k_q4k_act_min(const float* __restrict__ scale, const int* __restrict__ sum, std::size_t n32, std::uint16_t* __restrict__ db, float* __restrict__ df) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n32; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const float x = scale[i] * static_cast<float>(sum[i]);
    const std::uint16_t hi = f2bf(x);
    db[(i >> 3) * 16 + (i & 7)] = hi;
    db[(i >> 3) * 16 + 8 + (i & 7)] = f2bf(x - bf2f(hi));
    df[i] = x;
  }
}

struct alignas(16) KStage {
  std::int8_t As[kTile][kQ8Lda];
  std::uint8_t Qs[kBN][kQsLd];
  uint4 Wr[kBN];
  float S2[kTile][2];
  std::uint8_t Am[kTile][kMinLdB];
  std::uint8_t Bm[kBN][kMinLdB];
};
constexpr int kKSmem = kStages * static_cast<int>(sizeof(KStage));

template <bool F32>
__global__ void __launch_bounds__(256, 2) k_q4k_tile(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const void* __restrict__ amin,
                                                    const BlockQ4K* __restrict__ w, const uint4* __restrict__ rec, const void* __restrict__ wmin,
                                                    const int4* __restrict__ units, int nunits, int N, int K, float* __restrict__ C) {
  constexpr int WNW = 4, NI = 4, WR = 32, MI = 2;
  extern __shared__ __align__(16) std::uint8_t g_smem[];
  KStage* stg = reinterpret_cast<KStage*>(g_smem);
  if (static_cast<int>(blockIdx.y) >= nunits) return;
  const int4 u = units[blockIdx.y];
  const int ex = u.x, bm = u.y, vrows = u.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp / WNW, wn = warp % WNW;
  const int bpr = K / kQK, n32 = K / 32;
  w += static_cast<std::size_t>(ex) * N * bpr;
  rec += static_cast<std::size_t>(ex) * N * bpr;
  const std::uint8_t* wminb = static_cast<const std::uint8_t*>(wmin) + static_cast<std::size_t>(ex) * N * bpr * 32;
  const std::uint8_t* aminb = static_cast<const std::uint8_t*>(amin);
  const int bn = static_cast<int>(blockIdx.x) * kBN;
  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_col = 16 * (lane >> 4);
  auto load = [&](int stage, int k0) {
    KStage& T = stg[stage];
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
    if (((k0 + 64) % kQK) == 0) {  // 32 bytes per row and super-block in both forms (16 bf16 or 8 fp32)
      const int r = tid >> 1, h = tid & 1;
      if (r < kTile) cp_async16(&T.Am[r][h * 16], aminb + (static_cast<std::size_t>(bm + r) * bpr + sb) * 32 + h * 16);
      cp_async16(&T.Bm[r][h * 16], wminb + (static_cast<std::size_t>(bn + r) * bpr + sb) * 32 + h * 16);
    }
  };
  const int kt_n = K / 64;
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < kt_n) load(st, st * 64);
    cp_async_commit();
  }
  constexpr float kMagic = 12582912.f;
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nk = kt + kStages - 1;
      if (nk < kt_n) load(nk % kStages, nk * 64);
      cp_async_commit();
    }
    const KStage& T = stg[kt % kStages];
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
    if (((k0 + 64) % kQK) == 0) {
      if constexpr (!F32) {  // v10: one bf16 mma per super-block, digits (hi, lo) x (m, m), then x dmin
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
          const int r0 = wm * WR + mi * 16 + g, r1 = r0 + 8;
          const std::uint16_t* ar0 = reinterpret_cast<const std::uint16_t*>(T.Am[r0]);
          const std::uint16_t* ar1 = reinterpret_cast<const std::uint16_t*>(T.Am[r1]);
          std::uint32_t ad[4] = {ld32(ar0 + 2 * t), ld32(ar1 + 2 * t), ld32(ar0 + 2 * t + 8), ld32(ar1 + 2 * t + 8)};
#pragma unroll
          for (int ni = 0; ni < NI; ++ni) {
            const int cb = wn * NI * 8 + ni * 8 + g;
            const std::uint16_t* br = reinterpret_cast<const std::uint16_t*>(T.Bm[cb]);
            std::uint32_t bd[2] = {ld32(br + 2 * t), ld32(br + 2 * t + 8)};
            float mn[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16(mn, ad, bd);
            const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
            const float m0 = __uint_as_float(T.Wr[col0].w), m1 = __uint_as_float(T.Wr[col0 + 1].w);
            acc[mi][ni][0] = fmaf(-m0, mn[0], acc[mi][ni][0]);
            acc[mi][ni][1] = fmaf(-m1, mn[1], acc[mi][ni][1]);
            acc[mi][ni][2] = fmaf(-m0, mn[2], acc[mi][ni][2]);
            acc[mi][ni][3] = fmaf(-m1, mn[3], acc[mi][ni][3]);
          }
        }
      } else {  // v10b: 8 fp32 FMAs per output
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
          const int r0 = wm * WR + mi * 16 + g, r1 = r0 + 8;
          const float* ar0 = reinterpret_cast<const float*>(T.Am[r0]);
          const float* ar1 = reinterpret_cast<const float*>(T.Am[r1]);
          const float4 a0l = *reinterpret_cast<const float4*>(ar0), a0h = *reinterpret_cast<const float4*>(ar0 + 4);
          const float4 a1l = *reinterpret_cast<const float4*>(ar1), a1h = *reinterpret_cast<const float4*>(ar1 + 4);
          const float a0[8] = {a0l.x, a0l.y, a0l.z, a0l.w, a0h.x, a0h.y, a0h.z, a0h.w};
          const float a1[8] = {a1l.x, a1l.y, a1l.z, a1l.w, a1h.x, a1h.y, a1h.z, a1h.w};
#pragma unroll
          for (int ni = 0; ni < NI; ++ni) {
            const int col0 = wn * NI * 8 + ni * 8 + 2 * t;
            const float* b0p = reinterpret_cast<const float*>(T.Bm[col0]);
            const float* b1p = reinterpret_cast<const float*>(T.Bm[col0 + 1]);
            const float4 b0l = *reinterpret_cast<const float4*>(b0p), b0h = *reinterpret_cast<const float4*>(b0p + 4);
            const float4 b1l = *reinterpret_cast<const float4*>(b1p), b1h = *reinterpret_cast<const float4*>(b1p + 4);
            const float b0[8] = {b0l.x, b0l.y, b0l.z, b0l.w, b0h.x, b0h.y, b0h.z, b0h.w};
            const float b1[8] = {b1l.x, b1l.y, b1l.z, b1l.w, b1h.x, b1h.y, b1h.z, b1h.w};
#pragma unroll
            for (int jx = 0; jx < 8; ++jx) {
              acc[mi][ni][0] = fmaf(-a0[jx], b0[jx], acc[mi][ni][0]);
              acc[mi][ni][1] = fmaf(-a0[jx], b1[jx], acc[mi][ni][1]);
              acc[mi][ni][2] = fmaf(-a1[jx], b0[jx], acc[mi][ni][2]);
              acc[mi][ni][3] = fmaf(-a1[jx], b1[jx], acc[mi][ni][3]);
            }
          }
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int lr = wm * WR + mi * 16 + g;
      const int c0 = bn + wn * NI * 8 + ni * 8 + 2 * t;
      if (lr < vrows) {
        C[static_cast<std::size_t>(bm + lr) * N + c0] = acc[mi][ni][0];
        C[static_cast<std::size_t>(bm + lr) * N + c0 + 1] = acc[mi][ni][1];
      }
      if (lr + 8 < vrows) {
        C[static_cast<std::size_t>(bm + lr + 8) * N + c0] = acc[mi][ni][2];
        C[static_cast<std::size_t>(bm + lr + 8) * N + c0 + 1] = acc[mi][ni][3];
      }
    }
}

__global__ void __launch_bounds__(256) k_q4k_vec(const std::int8_t* __restrict__ a8, const float* __restrict__ sa32, const BlockQ4K* __restrict__ w,
                                                 const int4* __restrict__ units, int nunits, int N, int K, float* __restrict__ C) {
  if (static_cast<int>(blockIdx.x) >= nunits) return;
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
    for (int r = 0; r < rows; ++r) C[static_cast<std::size_t>(u.y + r) * N + n] = acc[r];
}

unsigned grid_of(std::size_t n, int per, unsigned cap = 4096) {
  const std::size_t b = (n + static_cast<std::size_t>(per) - 1) / static_cast<std::size_t>(per);
  return static_cast<unsigned>(b < 1 ? 1 : (b > cap ? cap : b));
}

template <typename F>
cudaError_t info_of(F f, int smem_dyn, int* regs, int* spill, int* smem, int* bps) {
  cudaFuncAttributes at{};
  cudaError_t e = cudaFuncGetAttributes(&at, f);
  if (e != cudaSuccess) return e;
  e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dyn);
  if (e != cudaSuccess) return e;
  int b = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, f, 256, static_cast<std::size_t>(smem_dyn));
  if (e != cudaSuccess) return e;
  *regs = at.numRegs;
  *spill = static_cast<int>(at.localSizeBytes);
  *smem = static_cast<int>(at.sharedSizeBytes) + smem_dyn;
  *bps = b;
  return cudaSuccess;
}

}  // namespace

const char* var_name(int v) {
  switch (v) {
    case kF32Lut: return "tg2 fp32 lookup (CUDA cores)";
    case kF16: return "tg2 fp16 mma";
    case kF16HiLo: return "tg2 fp16 hi+lo (2 mma)";
    case kI8x2: return "tg2 int8 x2 (16-bit fixed)";
    case kI8x3: return "tg2 int8 x3 (24-bit fixed)";
    default: return "?";
  }
}

void tg2_grids(float g[4][16]) {  // quant_cpu.cpp Q4TGGrids, v2 (h)
  float g0[4][16];
  for (int i = 0; i < 16; ++i) g0[0][i] = (static_cast<float>(i) - 7.5f) / 7.5f;
  const double ratios[2] = {1.3, 1.45};
  for (int t = 0; t < 2; ++t) {
    const double top = std::pow(ratios[t], 7.0);
    for (int m = 0; m < 8; ++m) {
      const float v = static_cast<float>(std::pow(ratios[t], static_cast<double>(m)) / top);
      g0[1 + t][8 + m] = v;
      g0[1 + t][7 - m] = -v;
    }
  }
  for (int i = 0; i < 16; ++i) g[0][i] = (static_cast<float>(i) - 7.0f) / 8.0f;
  for (int i = 0; i < 16; ++i) g[1][i] = g0[1][i];
  g[1][8] = 0.0f;
  for (int i = 0; i < 16; ++i) g[2][i] = g0[2][i];
  for (int i = 0; i < 16; ++i) g[3][i] = 0.6f * g[0][i];
}

float tg2_fixed_scale(int v) { return v == kI8x2 ? 32767.0f : (v == kI8x3 ? 8388607.0f : 1.0f); }

void tg2_table(int v, std::uint32_t out[64]) {
  float g[4][16];
  tg2_grids(g);
  for (int t = 0; t < 4; ++t)
    for (int c = 0; c < 16; ++c) {
      const float x = g[t][c];
      std::uint32_t e = 0;
      if (v == kF32Lut) {
        std::memcpy(&e, &x, 4);
      } else if (v == kF16 || v == kF16HiLo) {
        const __half hi = __float2half_rn(x);
        const __half lo = __float2half_rn(x - __half2float(hi));
        std::uint16_t hb, lb;
        std::memcpy(&hb, &hi, 2);
        std::memcpy(&lb, &lo, 2);
        e = hb | (v == kF16HiLo ? static_cast<std::uint32_t>(lb) << 16 : 0u);
      } else {
        const long long G = std::llround(static_cast<double>(x) * static_cast<double>(tg2_fixed_scale(v)));
        const std::uint32_t u = static_cast<std::uint32_t>(G) & (v == kI8x2 ? 0xFFFFu : 0xFFFFFFu);  // two's complement bytes
        e = u;  // bytes lo, (mid,) hi: the top byte read as s8, the others as u8
      }
      out[t * 16 + c] = e;
    }
}

cudaError_t tg2_prep(const void* w, float* sc, std::size_t nblocks, cudaStream_t s) {
  k_tg2_prep<<<grid_of(nblocks, 256), 256, 0, s>>>(static_cast<const std::uint8_t*>(w), sc, nblocks);
  return cudaGetLastError();
}

cudaError_t tg2_tile(int v, const std::int8_t* a8, const float* sa32, const void* w, const float* sc, const std::uint32_t* lut, const int4* units, int nunits,
                     int N, int K, float* C, cudaStream_t s) {
  if (N % kBN || K % kQK || nunits <= 0) return cudaErrorInvalidValue;
  TgArgs A{a8, sa32, static_cast<const std::uint8_t*>(w), sc, lut, units, nunits, N, K, C};
  const dim3 grid(static_cast<unsigned>(N / kBN), static_cast<unsigned>(nunits));
  cudaError_t e = cudaSuccess;
  switch (v) {
    case kF32Lut:
      e = cudaFuncSetAttribute(k_tg2_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, kF32Smem);
      if (e == cudaSuccess) k_tg2_f32<<<grid, 256, kF32Smem, s>>>(A);
      break;
    case kF16:
      e = cudaFuncSetAttribute(k_tg2_tile<kF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTgSmem);
      if (e == cudaSuccess) k_tg2_tile<kF16><<<grid, 256, kTgSmem, s>>>(A);
      break;
    case kF16HiLo:
      e = cudaFuncSetAttribute(k_tg2_tile<kF16HiLo>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTgSmem);
      if (e == cudaSuccess) k_tg2_tile<kF16HiLo><<<grid, 256, kTgSmem, s>>>(A);
      break;
    case kI8x2:
      e = cudaFuncSetAttribute(k_tg2_tile<kI8x2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTgSmem);
      if (e == cudaSuccess) k_tg2_tile<kI8x2><<<grid, 256, kTgSmem, s>>>(A);
      break;
    case kI8x3:
      e = cudaFuncSetAttribute(k_tg2_tile<kI8x3>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTgSmem);
      if (e == cudaSuccess) k_tg2_tile<kI8x3><<<grid, 256, kTgSmem, s>>>(A);
      break;
    default: return cudaErrorInvalidValue;
  }
  return e != cudaSuccess ? e : cudaGetLastError();
}

cudaError_t tg2_vec(const std::int8_t* a8, const float* sa32, const void* w, const float* sc, const std::uint32_t* lut, const int4* units, int nunits, int N,
                    int K, float* C, cudaStream_t s) {
  if (N % 8 || K % kQK || nunits <= 0) return cudaErrorInvalidValue;
  TgArgs A{a8, sa32, static_cast<const std::uint8_t*>(w), sc, lut, units, nunits, N, K, C};
  k_tg2_vec<<<dim3(static_cast<unsigned>(nunits), static_cast<unsigned>(N / 8)), 256, 0, s>>>(A);
  return cudaGetLastError();
}

cudaError_t tg2_tile_info(int v, int* regs, int* spill, int* smem, int* bps) {
  switch (v) {
    case kF32Lut: return info_of(k_tg2_f32, kF32Smem, regs, spill, smem, bps);
    case kF16: return info_of(k_tg2_tile<kF16>, kTgSmem, regs, spill, smem, bps);
    case kF16HiLo: return info_of(k_tg2_tile<kF16HiLo>, kTgSmem, regs, spill, smem, bps);
    case kI8x2: return info_of(k_tg2_tile<kI8x2>, kTgSmem, regs, spill, smem, bps);
    case kI8x3: return info_of(k_tg2_tile<kI8x3>, kTgSmem, regs, spill, smem, bps);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t q4k_prep(const void* w, uint4* rec, void* wmin_bf16, float* wmin_f32, std::size_t nblocks, cudaStream_t s) {
  k_q4k_prep<<<grid_of(nblocks, 256), 256, 0, s>>>(static_cast<const BlockQ4K*>(w), rec, static_cast<std::uint16_t*>(wmin_bf16), wmin_f32, nblocks);
  return cudaGetLastError();
}
cudaError_t q4k_act_min(const float* scale, const int* sum, std::size_t n32, void* amin_bf16, float* amin_f32, cudaStream_t s) {
  k_q4k_act_min<<<grid_of(n32, 256), 256, 0, s>>>(scale, sum, n32, static_cast<std::uint16_t*>(amin_bf16), amin_f32);
  return cudaGetLastError();
}
cudaError_t q4k_tile(bool f32min, const std::int8_t* a8, const float* sa32, const void* amin, const void* w, const uint4* rec, const void* wmin, const int4* units,
                     int nunits, int N, int K, float* C, cudaStream_t s) {
  if (N % kBN || K % kQK || nunits <= 0) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(N / kBN), static_cast<unsigned>(nunits));
  cudaError_t e;
  if (f32min) {
    e = cudaFuncSetAttribute(k_q4k_tile<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kKSmem);
    if (e == cudaSuccess) k_q4k_tile<true><<<grid, 256, kKSmem, s>>>(a8, sa32, amin, static_cast<const BlockQ4K*>(w), rec, wmin, units, nunits, N, K, C);
  } else {
    e = cudaFuncSetAttribute(k_q4k_tile<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kKSmem);
    if (e == cudaSuccess) k_q4k_tile<false><<<grid, 256, kKSmem, s>>>(a8, sa32, amin, static_cast<const BlockQ4K*>(w), rec, wmin, units, nunits, N, K, C);
  }
  return e != cudaSuccess ? e : cudaGetLastError();
}
cudaError_t q4k_vec(const std::int8_t* a8, const float* sa32, const void* w, const int4* units, int nunits, int N, int K, float* C, cudaStream_t s) {
  if (N % 8 || K % kQK || nunits <= 0) return cudaErrorInvalidValue;
  k_q4k_vec<<<dim3(static_cast<unsigned>(nunits), static_cast<unsigned>(N / 8)), 256, 0, s>>>(a8, sa32, static_cast<const BlockQ4K*>(w), units, nunits, N, K, C);
  return cudaGetLastError();
}
cudaError_t q4k_tile_info(bool f32min, int* regs, int* spill, int* smem, int* bps) {
  return f32min ? info_of(k_q4k_tile<true>, kKSmem, regs, spill, smem, bps) : info_of(k_q4k_tile<false>, kKSmem, regs, spill, smem, bps);
}

}  // namespace ptg

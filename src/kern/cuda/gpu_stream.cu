// provenance: original (moved from tools/probes/r3/r3.cu, the measured R3 probe; the device math is the probe's)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/cuda/gpu_stream.cu - the R3 streaming kernel behind kern/gpu_stream.h. lane_part is the probe's lane_part<2> (grid
// path 2), run_task its route-2 form (incremental lookup), k_stream its r3_stream2<2, 1> and k_staged its r3_grouped<2>
// (item 7: per-expert counters inside a stage, a counter barrier per warp between stages). What changed in moving: the
// template parameters fixed at the measured choices, the job struct's pointers typed for the engine, the counters and the
// table in one device buffer per plan, the launch and its checks on the host side.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "kern/gpu_stream.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::kern::gpu {
namespace {

enum Fmt : int { kQ7ZT = 0, kQ4TG2 = 1, kQ5K = 2, kQ6K = 3, kQ8_0 = 4, kQ4K = 5, kIQ4XS = 6 };

struct DJob {
  int fmt, n, K, lpr;      // format, rows, inputs, lanes per row (K / 128)
  long long task0, ntask;  // first warp task of this job in the table, warp tasks (32 / lpr rows each)
  const std::uint8_t* w;
  const std::uint64_t* hdr;
  const std::uint8_t* mask;
  const std::uint32_t* hb;
  const std::uint16_t* d6;
  const std::int8_t* xq;
  const float* xs;
  const int* xsum;
  float* y;
};

__constant__ float c_grid[4][16];  // the Q4_TG2 grids (quant::q4_tg_grid(tag, 2)), uploaded once

thread_local std::string s_err;
bool sfail(const std::string& why) {
  s_err = why;
  return false;
}
bool sck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  return sfail(std::string(what) + ": " + cudaGetErrorString(e));
}
cudaStream_t sst(Stream s) { return static_cast<cudaStream_t>(s); }

int fmt_of(std::uint32_t type) {
  switch (type) {
    case static_cast<std::uint32_t>(quant::Type::q7_zt): return kQ7ZT;
    case static_cast<std::uint32_t>(quant::Type::q4_tg2): return kQ4TG2;
    case static_cast<std::uint32_t>(quant::Type::q5_k): return kQ5K;
    case static_cast<std::uint32_t>(quant::Type::q6_k): return kQ6K;
    case static_cast<std::uint32_t>(quant::Type::q8_0): return kQ8_0;
    case static_cast<std::uint32_t>(quant::Type::q4_k): return kQ4K;
    case static_cast<std::uint32_t>(quant::Type::iq4_xs): return kIQ4XS;
    default: return -1;
  }
}

__device__ __forceinline__ int4 ld_cs4(const void* p) {
  int4 r;
  asm volatile("ld.global.cs.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ int sx8(const int* xw, int i) {  // int8 value i of a 128-value activation slice in 32 words
  return static_cast<int>(static_cast<std::int8_t>((xw[i >> 2] >> (8 * (i & 3))) & 0xff));
}

extern __shared__ float s_lut[];  // the replicated grid: 2,048 floats (8 KB), lane L reads s_lut[L + 32 * (tag * 16 + code)]
constexpr int kLutBytes = 2048 * 4;

__device__ __forceinline__ float i8f(const int* xw, int i) {  // int8 -> float by the magic number (exact for int8)
  const unsigned b = (static_cast<unsigned>(xw[i >> 2]) >> (8 * (i & 3))) & 0xffu;
  return __int_as_float(static_cast<int>((b ^ 0x80u) | 0x4B000000u)) - 8388736.f;
}

// IQ4_XS: four 4-bit codes (one per byte, 0..15) -> the four int8 levels kvalues_iq4nl[code], packed for __dp4a. The 16
// levels sit in four words; __byte_perm picks by the low 3 bits of each code from (T0, T1) and (T2, T3), bit 3 chooses
// between the two (checked on the host for all 65,536 code combinations, 2026-10-30)
__device__ __forceinline__ int iq4_levels(std::uint32_t v) {
  constexpr std::uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;  // kvalues_iq4nl 0-3, 4-7, 8-11, 12-15
  const std::uint32_t x = v & 0x07070707u, t = x | (x >> 4);
  const std::uint32_t sel = (t & 0xffu) | ((t >> 4) & 0x0f00u) | ((t >> 8) & 0xf000u);
  const std::uint32_t lo = __byte_perm(T0, T1, sel), hi = __byte_perm(T2, T3, sel), m = ((v >> 3) & 0x01010101u) * 0xffu;
  return static_cast<int>((lo & ~m) | (hi & m));
}

// one lane's part (128 values = 4 sub-blocks of 32) of one row; returns the partial dot product
__device__ float lane_part(const DJob& J, int r, int part) {
  const int B = part >> 1, half = part & 1, nB = J.K / 256;
  const long long rb = static_cast<long long>(r) * nB + B;
  const int jg0 = B * 8 + half * 4;  // first 32-block of the part in the row
  int xw[32];
  {
    const int4* x4 = reinterpret_cast<const int4*>(J.xq + jg0 * 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int4 t = __ldg(x4 + i);
      xw[4 * i] = t.x;
      xw[4 * i + 1] = t.y;
      xw[4 * i + 2] = t.z;
      xw[4 * i + 3] = t.w;
    }
  }
  float acc = 0.f;
  if (J.fmt == kQ7ZT) {
    std::uint32_t wd[29];
    const std::uint8_t* bp = J.w + rb * 224 + half * 112;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      const int4 t = ld_cs4(bp + 16 * i);
      wd[4 * i] = static_cast<std::uint32_t>(t.x);
      wd[4 * i + 1] = static_cast<std::uint32_t>(t.y);
      wd[4 * i + 2] = static_cast<std::uint32_t>(t.z);
      wd[4 * i + 3] = static_cast<std::uint32_t>(t.w);
    }
    wd[28] = 0u;
    const std::uint64_t h = J.hdr[rb];
    const unsigned m = J.mask[rb];
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h & 0xffffu)));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      int pb = 0;
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        std::uint32_t packed = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int i = 32 * jj + 4 * g + t, bit = 7 * i;
          packed |= (__funnelshift_r(wd[bit >> 5], wd[(bit >> 5) + 1], static_cast<unsigned>(bit & 31)) & 127u) << (8 * t);
        }
        pb = __dp4a(static_cast<int>(packed), xw[8 * jj + g], pb);
      }
      int hx = 0;
      if ((m >> j) & 1u) {
        const std::uint32_t H = J.hb[rb * 2 + __popc(m & ((1u << j) - 1u))];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const std::uint32_t nib = (H >> (4 * g)) & 0xfu;
          hx = __dp4a(static_cast<int>((nib * 0x00204081u) & 0x01010101u), xw[8 * jj + g], hx);
        }
      }
      const int P = 2 * pb - 127 * __ldg(J.xsum + jg) + hx;  // sum (2b + h - 127) x
      const float s = (d * static_cast<float>((h >> (16 + 6 * j)) & 63u)) / 63.f;
      acc += s * __ldg(J.xs + jg) * static_cast<float>(P);
    }
  } else if (J.fmt == kQ4TG2) {
    const std::uint8_t* bp = J.w + rb * 144;
    const int4 hd = ld_cs4(bp);
    int4 c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = ld_cs4(bp + 16 + half * 64 + 16 * i);
    const std::uint32_t cw[16] = {static_cast<std::uint32_t>(c[0].x), static_cast<std::uint32_t>(c[0].y), static_cast<std::uint32_t>(c[0].z),
                                  static_cast<std::uint32_t>(c[0].w), static_cast<std::uint32_t>(c[1].x), static_cast<std::uint32_t>(c[1].y),
                                  static_cast<std::uint32_t>(c[1].z), static_cast<std::uint32_t>(c[1].w), static_cast<std::uint32_t>(c[2].x),
                                  static_cast<std::uint32_t>(c[2].y), static_cast<std::uint32_t>(c[2].z), static_cast<std::uint32_t>(c[2].w),
                                  static_cast<std::uint32_t>(c[3].x), static_cast<std::uint32_t>(c[3].y), static_cast<std::uint32_t>(c[3].z),
                                  static_cast<std::uint32_t>(c[3].w)};
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x), h1 = static_cast<std::uint32_t>(hd.y);
    const std::uint64_t lv = (static_cast<std::uint64_t>(h0 >> 16)) | (static_cast<std::uint64_t>(h1) << 16);  // 6 level bytes at 2..7
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const std::uint64_t tags =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(hd.z)) | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(hd.w)) << 32);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      const unsigned tb = static_cast<unsigned>((tags >> (8 * j)) & 0xffu);
      float sum = 0.f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const unsigned tg = (tb >> (2 * g)) & 3u;
        const std::uint32_t word = cw[jj * 4 + g];  // 8 codes = values 32 jj + 8 g .. + 7 of the part
        const float* lr = s_lut + (threadIdx.x & 31) + 32 * 16 * tg;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float wv = lr[32 * ((word >> (4 * k)) & 15u)];
          sum += wv * i8f(xw, 32 * jj + 8 * g + k);
        }
      }
      const float s = (d * static_cast<float>((lv >> (6 * j)) & 63u)) / 63.f;
      acc += s * __ldg(J.xs + jg) * sum;
    }
  } else if (J.fmt == kQ5K) {
    const std::uint8_t* bp = J.w + rb * 176;
    const int4 hd = ld_cs4(bp);
    const int4 qa = ld_cs4(bp + 16), qb4 = ld_cs4(bp + 32);
    int4 ql[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) ql[i] = ld_cs4(bp + 48 + half * 64 + 16 * i);
    std::uint8_t hb[4], qh[32], qlb[64];
    memcpy(hb, &hd, 4);
    memcpy(qh, &qa, 16);
    memcpy(qh + 16, &qb4, 16);
    memcpy(qlb, ql, 64);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hb[0] | (hb[1] << 8))));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hb[2] | (hb[3] << 8))));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      // the 12 scale bytes sit in header words y, z, w (bytes 4..15); read by shifts, so nothing goes to local memory
      const std::uint32_t s0 = static_cast<std::uint32_t>(hd.y), s1 = static_cast<std::uint32_t>(hd.z), s2 = static_cast<std::uint32_t>(hd.w);
      auto qb = [&](int k) { return static_cast<int>(((k < 4 ? s0 : (k < 8 ? s1 : s2)) >> (8 * (k & 3))) & 0xffu); };
      int sc, mn;
      if (j < 4) {
        sc = qb(j) & 63;
        mn = qb(j + 4) & 63;
      } else {
        sc = (qb(j + 4) & 0xF) | ((qb(j - 4) >> 6) << 4);
        mn = (qb(j + 4) >> 4) | ((qb(j) >> 6) << 4);
      }
      const std::uint8_t* q = qlb + 32 * (jj >> 1);
      int sq = 0;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const int lo = (j & 1) ? (q[l] >> 4) : (q[l] & 0xF);
        sq += (lo + (((qh[l] >> j) & 1) << 4)) * sx8(xw, 32 * jj + l);
      }
      acc += __ldg(J.xs + jg) *
             (d * static_cast<float>(sc) * static_cast<float>(sq) - dmin * static_cast<float>(mn) * static_cast<float>(__ldg(J.xsum + jg)));
    }
  } else if (J.fmt == kQ8_0) {  // repacked per 256: qs[256] (int8) | d[8] (fp16); the half's 128 values and 4 d's
    const std::uint8_t* bp = J.w + rb * 272;
    const uint2 dd = *reinterpret_cast<const uint2*>(bp + 256 + half * 8);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int jg = jg0 + jj;
      const int4 qa = ld_cs4(bp + half * 128 + 32 * jj), qb = ld_cs4(bp + half * 128 + 32 * jj + 16);
      int P = 0;
      P = __dp4a(qa.x, xw[8 * jj], P);
      P = __dp4a(qa.y, xw[8 * jj + 1], P);
      P = __dp4a(qa.z, xw[8 * jj + 2], P);
      P = __dp4a(qa.w, xw[8 * jj + 3], P);
      P = __dp4a(qb.x, xw[8 * jj + 4], P);
      P = __dp4a(qb.y, xw[8 * jj + 5], P);
      P = __dp4a(qb.z, xw[8 * jj + 6], P);
      P = __dp4a(qb.w, xw[8 * jj + 7], P);
      const std::uint32_t dw = (jj < 2 ? dd.x : dd.y) >> (16 * (jj & 1));
      const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dw & 0xffffu)));
      acc += d * __ldg(J.xs + jg) * static_cast<float>(P);
    }
  } else if (J.fmt == kQ4K) {  // as stored (144 bytes): d | dmin | scales[12] | qs[128]; Q5_K without qh
    const std::uint8_t* bp = J.w + rb * 144;
    const int4 hd = ld_cs4(bp);
    int4 ql[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) ql[i] = ld_cs4(bp + 16 + half * 64 + 16 * i);
    const std::uint32_t qw[16] = {static_cast<std::uint32_t>(ql[0].x), static_cast<std::uint32_t>(ql[0].y), static_cast<std::uint32_t>(ql[0].z),
                                  static_cast<std::uint32_t>(ql[0].w), static_cast<std::uint32_t>(ql[1].x), static_cast<std::uint32_t>(ql[1].y),
                                  static_cast<std::uint32_t>(ql[1].z), static_cast<std::uint32_t>(ql[1].w), static_cast<std::uint32_t>(ql[2].x),
                                  static_cast<std::uint32_t>(ql[2].y), static_cast<std::uint32_t>(ql[2].z), static_cast<std::uint32_t>(ql[2].w),
                                  static_cast<std::uint32_t>(ql[3].x), static_cast<std::uint32_t>(ql[3].y), static_cast<std::uint32_t>(ql[3].z),
                                  static_cast<std::uint32_t>(ql[3].w)};
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 >> 16)));
    const std::uint32_t s0 = static_cast<std::uint32_t>(hd.y), s1 = static_cast<std::uint32_t>(hd.z), s2 = static_cast<std::uint32_t>(hd.w);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      auto qb = [&](int k) { return static_cast<int>(((k < 4 ? s0 : (k < 8 ? s1 : s2)) >> (8 * (k & 3))) & 0xffu); };
      int sc, mn;
      if (j < 4) {
        sc = qb(j) & 63;
        mn = qb(j + 4) & 63;
      } else {
        sc = (qb(j + 4) & 0xF) | ((qb(j - 4) >> 6) << 4);
        mn = (qb(j + 4) >> 4) | ((qb(j) >> 6) << 4);
      }
      // sub-block j: 32 nibbles of bytes 32 (jj / 2) .. +31 of the half, low nibbles for even j, high for odd
      const int sh = 4 * (j & 1);
      int sq = 0;
#pragma unroll
      for (int g = 0; g < 8; ++g) sq = __dp4a(static_cast<int>((qw[8 * (jj >> 1) + g] >> sh) & 0x0F0F0F0Fu), xw[8 * jj + g], sq);
      acc += __ldg(J.xs + jg) *
             (d * static_cast<float>(sc) * static_cast<float>(sq) - dmin * static_cast<float>(mn) * static_cast<float>(__ldg(J.xsum + jg)));
    }
  } else if (J.fmt == kIQ4XS) {  // repacked per 256 (144 bytes): d | scales_h | scales_l[4] | 8 zero bytes | qs[128]
    const std::uint8_t* bp = J.w + rb * 144;
    const int4 hd = ld_cs4(bp);
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x), sl = static_cast<std::uint32_t>(hd.y);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const std::uint32_t sh = h0 >> 16;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int ib = half * 4 + jj, jg = jg0 + jj;
      const int ls = static_cast<int>(((sl >> (4 * ib)) & 0xfu) | (((sh >> (2 * ib)) & 3u) << 4));
      const int4 qv = ld_cs4(bp + 16 + 16 * ib);  // 16 bytes: value l (low nibble of byte l) and l + 16 (high nibble)
      const std::uint32_t qw[4] = {static_cast<std::uint32_t>(qv.x), static_cast<std::uint32_t>(qv.y), static_cast<std::uint32_t>(qv.z),
                                   static_cast<std::uint32_t>(qv.w)};
      int P = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        P = __dp4a(iq4_levels(qw[w] & 0x0F0F0F0Fu), xw[8 * jj + w], P);
        P = __dp4a(iq4_levels((qw[w] >> 4) & 0x0F0F0F0Fu), xw[8 * jj + 4 + w], P);
      }
      acc += d * static_cast<float>(ls - 32) * __ldg(J.xs + jg) * static_cast<float>(P);
    }
  } else {  // kQ6K
    const std::uint8_t* bp = J.w + rb * 208;
    int4 l4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) l4[i] = ld_cs4(bp + half * 64 + 16 * i);
    const int4 h4a = ld_cs4(bp + 128 + half * 32), h4b = ld_cs4(bp + 128 + half * 32 + 16);
    const uint2 s2 = *reinterpret_cast<const uint2*>(bp + 192 + half * 8);
    std::uint8_t ql[64], qh[32];
    std::int8_t sc[8];
    memcpy(ql, l4, 64);
    memcpy(qh, &h4a, 16);
    memcpy(qh + 16, &h4b, 16);
    memcpy(sc, &s2, 8);
    const float d = __half2float(__ushort_as_half(J.d6[rb]));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int jg = jg0 + p;
      int s0 = 0, s1 = 0;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        int q;
        if (p == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
        else if (p == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
        else if (p == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
        else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        const int v = (q - 32) * sx8(xw, 32 * p + l);
        if (l < 16) s0 += v;
        else s1 += v;
      }
      acc += d * __ldg(J.xs + jg) * static_cast<float>(sc[2 * p] * s0 + sc[2 * p + 1] * s1);
    }
  }
  return acc;
}

// route 2: a warp's t only grows, so the job is found by walking forward from the last one
__device__ __forceinline__ int find_job(const DJob* __restrict__ jobs, int njobs, long long t, int* hint) {
  int lo = *hint;
  while (lo + 1 < njobs && jobs[lo + 1].task0 <= t) ++lo;
  *hint = lo;
  return lo;
}

__device__ __forceinline__ void run_task(const DJob* __restrict__ jobs, int njobs, long long t, int* hint) {
  const DJob& J = jobs[find_job(jobs, njobs, t, hint)];
  const int lane = static_cast<int>(threadIdx.x & 31), rpw = 32 / J.lpr;
  const int r = static_cast<int>((t - J.task0) * rpw) + lane / J.lpr, part = lane % J.lpr;
  float acc = r < J.n ? lane_part(J, r, part) : 0.f;
  for (int o = J.lpr / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (r < J.n && part == 0) J.y[r] = acc;
}

__device__ __forceinline__ void lut_setup() {
  for (int i = static_cast<int>(threadIdx.x); i < 2048; i += static_cast<int>(blockDim.x)) s_lut[i] = c_grid[(i >> 9) & 3][(i >> 5) & 15];
  __syncthreads();
}

__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(unsigned long long* p, unsigned long long v) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// every job at once (r3_stream2<2, 1>)
__global__ void __launch_bounds__(256) k_stream(const DJob* __restrict__ jobs, int njobs, long long t_end) {
  lut_setup();
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  int hint = 0;
  for (long long t = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; t < t_end; t += warps)
    run_task(jobs, njobs, t, &hint);
}

// the token (r3_grouped<2>): stages with a counter barrier per warp between them; inside a stage a job may wait for a
// group's tasks (acquire) and its own tasks may signal a group (release). done[] and gdone[] are never reset: run number
// epoch (1-based) waits for epoch x the counts.
__global__ void __launch_bounds__(256) k_staged(const DJob* __restrict__ jobs, int njobs, const long long* __restrict__ stage_end, int nstages,
                                                unsigned long long* done, const int* __restrict__ grp, const int* __restrict__ dep,
                                                const long long* __restrict__ gtasks, unsigned long long* gdone, unsigned long long epoch) {
  lut_setup();
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  const long long w0 = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = static_cast<int>(threadIdx.x & 31);
  int hint = 0;
  long long begin = 0;
  for (int s = 0; s < nstages; ++s) {
    const long long end = stage_end[s];
    unsigned long long mine = 0;
    for (long long t = begin + w0; t < end; t += warps) {
      int j = hint;
      while (j + 1 < njobs && jobs[j + 1].task0 <= t) ++j;
      const int dg = dep[j];
      if (dg >= 0) {
        if (lane == 0) {
          const unsigned long long target = epoch * static_cast<unsigned long long>(gtasks[dg]);
          while (ld_acquire_u64(gdone + dg) < target) __nanosleep(32);
        }
        __syncwarp();
      }
      run_task(jobs, njobs, t, &hint);
      const int sg = grp[j];
      if (sg >= 0) {
        __syncwarp();  // the lanes' outputs are ordered before lane 0's release
        if (lane == 0) red_release_add(gdone + sg, 1ull);
      }
      ++mine;
    }
    if (s + 1 == nstages) break;
    __syncwarp();
    if (lane == 0) {
      __threadfence();  // this warp's outputs are visible before its count is
      if (mine) atomicAdd(done + s, mine);
      const unsigned long long target = epoch * static_cast<unsigned long long>(end - begin);
      while (ld_acquire_u64(done + s) < target) __nanosleep(32);
    }
    __syncwarp();
    begin = end;
  }
}

bool grid_ready() {
  static bool ready = false;
  if (ready) return true;
  float g[4][16];
  for (int t = 0; t < 4; ++t)
    for (int i = 0; i < 16; ++i) g[t][i] = quant::q4_tg_grid(t, 2)[i];
  if (!sck(cudaMemcpyToSymbol(c_grid, g, sizeof g), "stream: grid upload")) return false;
  ready = true;
  return true;
}

std::uint16_t rd16h(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

template <class T>
bool up(const std::vector<T>& v, void** out) {
  *out = dalloc(v.size() * sizeof(T));
  if (!*out) return sfail("stream_upload: device memory");
  return to_device(*out, v.data(), v.size() * sizeof(T));
}

constexpr std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace

bool stream_type_ok(std::uint32_t type, std::int64_t k) {
  return fmt_of(type) >= 0 && (k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096);
}

bool stream_upload(std::uint32_t type, const void* rows, std::int64_t n, std::int64_t k, StreamWeights* out) {
  if (!out || !rows || n <= 0 || n > 0x7fffffff) return sfail("stream_upload: bad arguments");
  if (!stream_type_ok(type, k)) return sfail("stream_upload: the streaming kernel reads Q7_ZT, Q4_TG2, Q5_K, Q6_K, Q8_0, Q4_K, IQ4_XS with k 256-4096 (powers of two)");
  *out = StreamWeights{};
  out->type = type;
  out->n = n;
  out->k = k;
  const std::int64_t nb = n * (k / 256);
  const auto* in = static_cast<const std::uint8_t*>(rows);
  const int f = fmt_of(type);
  if (f == kQ7ZT) {  // GGUF Q7_ZT (241 bytes) -> base, header, mask, H (r3_formats.h q7zt_repack)
    std::vector<std::uint8_t> base(static_cast<std::size_t>(nb) * 224), mask(static_cast<std::size_t>(nb));
    std::vector<std::uint64_t> hdr(static_cast<std::size_t>(nb));
    std::vector<std::uint32_t> hb(static_cast<std::size_t>(nb) * 2);
    for (std::int64_t b = 0; b < nb; ++b) {
      const std::uint8_t* p = in + 241 * b;
      std::uint64_t h = rd16h(p);
      for (int i = 0; i < 6; ++i) h |= static_cast<std::uint64_t>(p[2 + i]) << (16 + 8 * i);
      hdr[static_cast<std::size_t>(b)] = h;
      mask[static_cast<std::size_t>(b)] = p[8];
      std::memcpy(base.data() + 224 * b, p + 9, 224);
      for (int s = 0; s < 2; ++s)
        hb[static_cast<std::size_t>(2 * b + s)] = static_cast<std::uint32_t>(p[233 + 4 * s]) | static_cast<std::uint32_t>(p[234 + 4 * s]) << 8 |
                                                  static_cast<std::uint32_t>(p[235 + 4 * s]) << 16 | static_cast<std::uint32_t>(p[236 + 4 * s]) << 24;
    }
    if (!up(base, &out->w) || !up(hdr, &out->hdr) || !up(mask, &out->mask) || !up(hb, &out->hb)) {
      stream_release(out);
      return false;
    }
    out->bytes = static_cast<std::uint64_t>(nb) * (224 + 8 + 1 + 8);
  } else if (f == kQ6K) {  // GGUF Q6_K (210 bytes: ql, qh, scales, d) -> 208 bytes + d
    std::vector<std::uint8_t> w(static_cast<std::size_t>(nb) * 208);
    std::vector<std::uint16_t> d(static_cast<std::size_t>(nb));
    for (std::int64_t b = 0; b < nb; ++b) {
      std::memcpy(w.data() + 208 * b, in + 210 * b, 208);
      d[static_cast<std::size_t>(b)] = rd16h(in + 210 * b + 208);
    }
    if (!up(w, &out->w) || !up(d, &out->d6)) {
      stream_release(out);
      return false;
    }
    out->bytes = static_cast<std::uint64_t>(nb) * 210;
  } else if (f == kQ8_0) {  // GGUF Q8_0 (8 blocks of 34 bytes: d, qs[32]) -> qs[256] | d[8] per 256 (272 bytes, 16-byte aligned)
    std::vector<std::uint8_t> w(static_cast<std::size_t>(nb) * 272);
    for (std::int64_t b = 0; b < nb; ++b)
      for (int i = 0; i < 8; ++i) {
        const std::uint8_t* p = in + 272 * b + 34 * i;
        std::memcpy(w.data() + 272 * b + 32 * i, p + 2, 32);
        std::memcpy(w.data() + 272 * b + 256 + 2 * i, p, 2);
      }
    if (!up(w, &out->w)) {
      stream_release(out);
      return false;
    }
    out->bytes = static_cast<std::uint64_t>(nb) * 272;
  } else if (f == kIQ4XS) {  // GGUF IQ4_XS (136 bytes) -> 8 header bytes, 8 zero bytes, qs[128] (144, 16-byte aligned)
    std::vector<std::uint8_t> w(static_cast<std::size_t>(nb) * 144, 0);
    for (std::int64_t b = 0; b < nb; ++b) {
      std::memcpy(w.data() + 144 * b, in + 136 * b, 8);
      std::memcpy(w.data() + 144 * b + 16, in + 136 * b + 8, 128);
    }
    if (!up(w, &out->w)) {
      stream_release(out);
      return false;
    }
    out->bytes = static_cast<std::uint64_t>(nb) * 136;
  } else {  // Q4_TG2 (144), Q4_K (144) and Q5_K (176) as stored
    const std::uint64_t bb = f == kQ5K ? 176 : 144;
    std::vector<std::uint8_t> w(in, in + static_cast<std::size_t>(nb) * bb);
    if (!up(w, &out->w)) {
      stream_release(out);
      return false;
    }
    out->bytes = static_cast<std::uint64_t>(nb) * bb;
  }
  return true;
}

void stream_release(StreamWeights* w) {
  if (!w) return;
  dfree(w->w);
  dfree(w->hdr);
  dfree(w->mask);
  dfree(w->hb);
  dfree(w->d6);
  *w = StreamWeights{};
}

StreamPlan::~StreamPlan() { dfree(dev_); }

bool StreamPlan::build(const StreamJob* jobs, int njobs, const int* stage_end, int nstages, int ngroups) {
  if (dev_) return sfail("StreamPlan::build: already built (make a new plan)");
  if (!jobs || njobs <= 0 || !stage_end || nstages <= 0 || ngroups < 0) return sfail("StreamPlan::build: bad arguments");
  if (stage_end[nstages - 1] != njobs) return sfail("StreamPlan::build: the last stage must end at the last job");
  for (int s = 0; s < nstages; ++s)
    if (stage_end[s] <= (s ? stage_end[s - 1] : 0)) return sfail("StreamPlan::build: every stage needs at least one job, in order");
  if (!grid_ready()) return false;
  std::vector<DJob> dj(static_cast<std::size_t>(njobs));
  std::vector<int> grp(static_cast<std::size_t>(njobs)), dep(static_cast<std::size_t>(njobs));
  std::vector<long long> gtasks(static_cast<std::size_t>(ngroups), 0), send(static_cast<std::size_t>(nstages));
  std::vector<int> first_signal(static_cast<std::size_t>(ngroups), -1), last_signal(static_cast<std::size_t>(ngroups), -1);
  long long t = 0;
  bytes_ = 0;
  int stage = 0;
  for (int j = 0; j < njobs; ++j) {
    const StreamJob& J = jobs[j];
    if (!J.w || !J.w->w || !J.xq || !J.xs || !J.xsum || !J.y) return sfail("StreamPlan::build: job " + std::to_string(j) + " has a null pointer");
    if (!stream_type_ok(J.w->type, J.w->k)) return sfail("StreamPlan::build: job " + std::to_string(j) + ": type or k not readable");
    if (reinterpret_cast<std::uintptr_t>(J.xq) % 16 != 0) return sfail("StreamPlan::build: job " + std::to_string(j) + ": xq must be 16-byte aligned");
    if (J.w->type == static_cast<std::uint32_t>(quant::Type::q7_zt) && (!J.w->hdr || !J.w->mask || !J.w->hb))
      return sfail("StreamPlan::build: job " + std::to_string(j) + ": Q7_ZT arrays missing");
    if (J.w->type == static_cast<std::uint32_t>(quant::Type::q6_k) && !J.w->d6) return sfail("StreamPlan::build: job " + std::to_string(j) + ": Q6_K d missing");
    if (J.signal >= ngroups || J.wait >= ngroups) return sfail("StreamPlan::build: job " + std::to_string(j) + ": group out of range");
    if (J.signal >= 0 && J.signal == J.wait) return sfail("StreamPlan::build: job " + std::to_string(j) + " waits for its own group");
    while (j >= stage_end[stage]) ++stage;
    DJob& D = dj[static_cast<std::size_t>(j)];
    D.fmt = fmt_of(J.w->type);
    D.n = static_cast<int>(J.w->n);
    D.K = static_cast<int>(J.w->k);
    D.lpr = D.K / 128;
    D.task0 = t;
    D.ntask = (J.w->n + (32 / D.lpr) - 1) / (32 / D.lpr);
    D.w = static_cast<const std::uint8_t*>(J.w->w);
    D.hdr = static_cast<const std::uint64_t*>(J.w->hdr);
    D.mask = static_cast<const std::uint8_t*>(J.w->mask);
    D.hb = static_cast<const std::uint32_t*>(J.w->hb);
    D.d6 = static_cast<const std::uint16_t*>(J.w->d6);
    D.xq = J.xq;
    D.xs = J.xs;
    D.xsum = J.xsum;
    D.y = J.y;
    grp[static_cast<std::size_t>(j)] = J.signal;
    dep[static_cast<std::size_t>(j)] = J.wait;
    if (J.signal >= 0) {
      gtasks[static_cast<std::size_t>(J.signal)] += D.ntask;
      if (first_signal[static_cast<std::size_t>(J.signal)] < 0) first_signal[static_cast<std::size_t>(J.signal)] = j;
      last_signal[static_cast<std::size_t>(J.signal)] = j;
    }
    t += D.ntask;
    send[static_cast<std::size_t>(stage)] = t;
    bytes_ += J.w->bytes;
  }
  for (int j = 0; j < njobs; ++j) {  // a waited group is signalled, and only by earlier jobs
    const int g = jobs[j].wait;
    if (g < 0) continue;
    if (first_signal[static_cast<std::size_t>(g)] < 0) return sfail("StreamPlan::build: job " + std::to_string(j) + " waits for a group nobody signals");
    if (last_signal[static_cast<std::size_t>(g)] >= j) return sfail("StreamPlan::build: job " + std::to_string(j) + " waits for a group signalled at or after it");
  }
  njobs_ = njobs;
  nstages_ = nstages;
  ngroups_ = ngroups;
  ntasks_ = t;
  off_jobs_ = 0;
  off_stage_ = align_up(off_jobs_ + sizeof(DJob) * dj.size(), 16);
  off_grp_ = align_up(off_stage_ + 8 * send.size(), 16);
  off_dep_ = align_up(off_grp_ + 4 * grp.size(), 16);
  off_gtasks_ = align_up(off_dep_ + 4 * dep.size(), 16);
  off_done_ = align_up(off_gtasks_ + 8 * gtasks.size(), 16);
  off_gdone_ = align_up(off_done_ + 8 * static_cast<std::uint64_t>(nstages), 16);
  const std::uint64_t total = align_up(off_gdone_ + 8 * static_cast<std::uint64_t>(ngroups > 0 ? ngroups : 1), 16);
  std::vector<std::uint8_t> h(static_cast<std::size_t>(total), 0);
  std::memcpy(h.data() + off_jobs_, dj.data(), sizeof(DJob) * dj.size());
  std::memcpy(h.data() + off_stage_, send.data(), 8 * send.size());
  std::memcpy(h.data() + off_grp_, grp.data(), 4 * grp.size());
  std::memcpy(h.data() + off_dep_, dep.data(), 4 * dep.size());
  if (!gtasks.empty()) std::memcpy(h.data() + off_gtasks_, gtasks.data(), 8 * gtasks.size());
  dev_ = dalloc(total);
  if (!dev_) return sfail("StreamPlan::build: device memory");
  if (!to_device(dev_, h.data(), total)) return false;
  int dev = 0, sms = 0, occ = 0, coop = 0;
  if (!sck(cudaGetDevice(&dev), "cudaGetDevice") || !sck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count") ||
      !sck(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev), "cooperative launch attribute") ||
      !sck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_staged, 256, kLutBytes), "occupancy"))
    return false;
  if (!coop) return sfail("StreamPlan::build: the device has no cooperative launch");
  if (occ < 1) return sfail("StreamPlan::build: the staged kernel does not fit on an SM");
  grid_ = sms * occ;
  epoch_ = 0;
  return true;
}

bool StreamPlan::run(Stream s) {
  if (!dev_) return sfail("StreamPlan::run: not built");
  auto* b = static_cast<std::uint8_t*>(dev_);
  const DJob* jobs = reinterpret_cast<const DJob*>(b + off_jobs_);
  const long long* stage_end = reinterpret_cast<const long long*>(b + off_stage_);
  auto* done = reinterpret_cast<unsigned long long*>(b + off_done_);
  const int* grp = reinterpret_cast<const int*>(b + off_grp_);
  const int* dep = reinterpret_cast<const int*>(b + off_dep_);
  const long long* gtasks = reinterpret_cast<const long long*>(b + off_gtasks_);
  auto* gdone = reinterpret_cast<unsigned long long*>(b + off_gdone_);
  int nj = njobs_, ns = nstages_;
  unsigned long long ep = epoch_ + 1;
  void* args[] = {&jobs, &nj, &stage_end, &ns, &done, &grp, &dep, &gtasks, &gdone, &ep};
  if (!sck(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(k_staged), dim3(static_cast<unsigned>(grid_)), dim3(256), args, kLutBytes, sst(s)),
           "stream: cooperative launch"))
    return false;
  epoch_ = ep;
  return true;
}

bool StreamPlan::run_unstaged(Stream s) {
  if (!dev_) return sfail("StreamPlan::run_unstaged: not built");
  const DJob* jobs = reinterpret_cast<const DJob*>(static_cast<std::uint8_t*>(dev_) + off_jobs_);
  k_stream<<<static_cast<unsigned>(grid_), 256, kLutBytes, sst(s)>>>(jobs, njobs_, ntasks_);
  return sck(cudaGetLastError(), "stream: launch");
}

const char* stream_last_error() { return s_err.c_str(); }

}  // namespace pelias::kern::gpu

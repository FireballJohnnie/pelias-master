// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// r3.cu - R3, the streaming decode kernel (game plan, "Decision: R3 goes with Q4_TG2 option A"; Fable's test 1: one launch
// per matrix reads at 150-352 GB/s, a long read at 494). One kernel walks a table of matrices (jobs) for a single token
// (decode, M = 1) and streams their rows back to back, so the memory pipeline stays full across matrix boundaries.
//   r3 check   : every format's GEMV against a double-precision host reference built from r3_formats.h (which matches
//                Pelias's dequantizers value for value)
//   r3 stream  : one token's weight reads, 40 layers (10 attention + 30 DeltaNet, real shapes) + the output head, timed
//                cold three ways: (a) one launch per matrix, queued back to back (today's kernels); (b) one launch for
//                the whole table (no dependencies: the read ceiling of the streaming kernel); (c) one cooperative launch
//                with a grid-wide sync between dependent stages (4 per layer + the head), the shape a real token needs
// Formats (r3_formats.h): dense Q7_ZT (fold route, dp4a), expert gate/up Q4_TG2 (grid table), expert down Q5_K, output
// head Q6_K. Activations: int8 per 32 with an fp32 scale (the gpu256x8 reference format). Lanes split each row in
// 128-value parts (K / 128 lanes per row), every weight load of a part issued before any arithmetic.
//   v2 step 1 (Fable's review, docs/reviews/FABLE_R3_V1_REVIEW_2026-10-29.md, item 1 + harness notes; kernels unchanged
//   apart from an optional clock record): per-format (b) rows on >= 256 MiB of distinct matrices each, the grid-sync cost
//   alone (161 syncs, no work), (a) replayed from a CUDA graph (GPU-side cost without the host submission path), the
//   registers / local memory of each kernel, and the SM clock measured inside the (b) run (clock64 vs globaltimer).
//   r3 stream --soak S : after the rows, (b) back to back for S seconds (for an outside clock / power log)
//   v2 fix 2 (Fable item 2): the Q4_TG2 grid read from a 32-way replicated table in shared memory (grid path 1), and with the
//   int8 -> float done by the magic number (path 2); path 0 is v1. Every check case runs all three paths; the stream rows add
//   (b) and (c) for the new paths and the Q4_TG2-only rows for paths 1 and 2.
//   v2 fix 6 (Fable item 6): (c) with a per-stage counter waited per warp (c1), and with the next stage's weights prefetched into
//   L2 before the wait (c2), and with the warp's whole next-stage share prefetched (c3); the check runs the counted kernel with
//   the whole-share prefetch over two stages (path 3).
// Build (WSL): nvcc -std=c++17 -O3 -arch=sm_89 r3.cu -o r3     (Windows: the same with -o r3.exe, from a VS x64 prompt)
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "r3_formats.h"

namespace cg = cooperative_groups;

#define CK(x)                                                                                            \
  do {                                                                                                   \
    cudaError_t e_ = (x);                                                                                \
    if (e_ != cudaSuccess) {                                                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);      \
      std::exit(2);                                                                                      \
    }                                                                                                    \
  } while (0)

struct Job {
  int fmt, n, K, lpr;          // format, rows, inputs, lanes per row (K / 128)
  long long task0, ntask;      // first warp task of this job in the table, warp tasks (32 / lpr rows each)
  const uint8_t* w;            // Q7ZT base / Q4TG2, Q5K blocks / Q6K repacked 208-byte blocks
  const uint64_t* hdr;         // Q7ZT
  const uint8_t* mask;         // Q7ZT
  const uint32_t* hb;          // Q7ZT
  const uint16_t* d6;          // Q6K d per 256-block
  const int8_t* xq;            // activation int8, K
  const float* xs;             // activation scale per 32
  const int* xsum;             // activation sum per 32
  float* y;                    // n outputs
};

__constant__ float c_grid[4][16];

__device__ __forceinline__ int4 ld_cs4(const void* p) {
  int4 r;
  asm volatile("ld.global.cs.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ int sx8(const int* xw, int i) {  // int8 value i of a 128-value activation slice in 32 words
  return static_cast<int>(static_cast<int8_t>((xw[i >> 2] >> (8 * (i & 3))) & 0xff));
}

// v2 fix 2 (Fable item 2): the Q4_TG2 grid. Q4V 0 = v1 (__constant__ table, one address per pass, so a warp's 32 different
// addresses serialise); 1 = the 64 grid floats replicated 32 times in shared memory, lane L reads lut[L + 32 * (tag * 16 + code)]
// (bank = L, conflict-free, exact: the same floats); 2 = as 1, plus the int8 activation turned into a float with the magic
// number (2 full-rate ops) instead of a conversion instruction (exact for int8)
extern __shared__ float s_dyn[];  // fix 2: the replicated grid (2,048 floats = 8 KB, dynamic shared memory; paths 1 and 2 only)
constexpr int kLutBytes = 2048 * 4;

__device__ __forceinline__ float i8f(const int* xw, int i) {
  const unsigned b = (static_cast<unsigned>(xw[i >> 2]) >> (8 * (i & 3))) & 0xffu;
  return __int_as_float(static_cast<int>((b ^ 0x80u) | 0x4B000000u)) - 8388736.f;
}

// one lane's part (128 values = 4 sub-blocks of 32) of one row; returns the partial dot product
template <int Q4V>
__device__ float lane_part(const Job& J, int r, int part, const float* lut) {
  (void)lut;  // fix 2 reads s_dyn directly (so the loads are ld.shared); kept for the call shape
  const int B = part >> 1, half = part & 1, nB = J.K / 256;
  const long long rb = static_cast<long long>(r) * nB + B;
  const int jg0 = B * 8 + half * 4;  // first 32-block of the part in the row
  int xw[32];
  {
    const int4* x4 = reinterpret_cast<const int4*>(J.xq + jg0 * 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int4 t = __ldg(x4 + i);
      xw[4 * i] = t.x; xw[4 * i + 1] = t.y; xw[4 * i + 2] = t.z; xw[4 * i + 3] = t.w;
    }
  }
  float acc = 0.f;
  if (J.fmt == r3::kQ7ZT) {
    uint32_t wd[29];
    const uint8_t* bp = J.w + rb * 224 + half * 112;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      const int4 t = ld_cs4(bp + 16 * i);
      wd[4 * i] = t.x; wd[4 * i + 1] = t.y; wd[4 * i + 2] = t.z; wd[4 * i + 3] = t.w;
    }
    wd[28] = 0u;
    const uint64_t h = J.hdr[rb];
    const unsigned m = J.mask[rb];
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h & 0xffffu)));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      int pb = 0;
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        uint32_t packed = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int i = 32 * jj + 4 * g + t, bit = 7 * i;
          packed |= (__funnelshift_r(wd[bit >> 5], wd[(bit >> 5) + 1], bit & 31) & 127u) << (8 * t);
        }
        pb = __dp4a(static_cast<int>(packed), xw[8 * jj + g], pb);
      }
      int hx = 0;
      if ((m >> j) & 1u) {
        const uint32_t H = J.hb[rb * 2 + __popc(m & ((1u << j) - 1u))];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const uint32_t nib = (H >> (4 * g)) & 0xfu;
          hx = __dp4a(static_cast<int>((nib * 0x00204081u) & 0x01010101u), xw[8 * jj + g], hx);
        }
      }
      const int P = 2 * pb - 127 * __ldg(J.xsum + jg) + hx;  // sum (2b + h - 127) x
      const float s = (d * static_cast<float>((h >> (16 + 6 * j)) & 63u)) / 63.f;
      acc += s * __ldg(J.xs + jg) * static_cast<float>(P);
    }
  } else if (J.fmt == r3::kQ4TG2) {
    const uint8_t* bp = J.w + rb * 144;
    const int4 hd = ld_cs4(bp);
    int4 c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = ld_cs4(bp + 16 + half * 64 + 16 * i);
    const uint32_t cw[16] = {static_cast<uint32_t>(c[0].x), static_cast<uint32_t>(c[0].y), static_cast<uint32_t>(c[0].z), static_cast<uint32_t>(c[0].w),
                             static_cast<uint32_t>(c[1].x), static_cast<uint32_t>(c[1].y), static_cast<uint32_t>(c[1].z), static_cast<uint32_t>(c[1].w),
                             static_cast<uint32_t>(c[2].x), static_cast<uint32_t>(c[2].y), static_cast<uint32_t>(c[2].z), static_cast<uint32_t>(c[2].w),
                             static_cast<uint32_t>(c[3].x), static_cast<uint32_t>(c[3].y), static_cast<uint32_t>(c[3].z), static_cast<uint32_t>(c[3].w)};
    const uint32_t h0 = static_cast<uint32_t>(hd.x), h1 = static_cast<uint32_t>(hd.y);
    const uint64_t lv = (static_cast<uint64_t>(h0 >> 16)) | (static_cast<uint64_t>(h1) << 16);  // 6 level bytes at 2..7
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const uint64_t tags = static_cast<uint64_t>(static_cast<uint32_t>(hd.z)) | (static_cast<uint64_t>(static_cast<uint32_t>(hd.w)) << 32);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      const unsigned tb = static_cast<unsigned>((tags >> (8 * j)) & 0xffu);
      float sum = 0.f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const unsigned tg = (tb >> (2 * g)) & 3u;
        const uint32_t word = cw[jj * 4 + g];  // 8 codes = values 32 jj + 8 g .. + 7 of the part
        if (Q4V == 0) {
          const float* gr = c_grid[tg];
#pragma unroll
          for (int k = 0; k < 8; ++k) sum += gr[(word >> (4 * k)) & 15u] * static_cast<float>(sx8(xw, 32 * jj + 8 * g + k));
        } else {
          const float* lr = s_dyn + (threadIdx.x & 31) + 32 * 16 * tg;  // the shared symbol itself, so loads are ld.shared
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const float w = lr[32 * ((word >> (4 * k)) & 15u)];
            const float xv = Q4V == 2 ? i8f(xw, 32 * jj + 8 * g + k) : static_cast<float>(sx8(xw, 32 * jj + 8 * g + k));
            sum += w * xv;
          }
        }
      }
      const float s = (d * static_cast<float>((lv >> (6 * j)) & 63u)) / 63.f;
      acc += s * __ldg(J.xs + jg) * sum;
    }
  } else if (J.fmt == r3::kQ5K) {
    const uint8_t* bp = J.w + rb * 176;
    const int4 hd = ld_cs4(bp);
    const int4 qa = ld_cs4(bp + 16), qb = ld_cs4(bp + 32);
    int4 ql[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) ql[i] = ld_cs4(bp + 48 + half * 64 + 16 * i);
    uint8_t hb[4], qh[32], qlb[64];
    memcpy(hb, &hd, 4);
    memcpy(qh, &qa, 16);
    memcpy(qh + 16, &qb, 16);
    memcpy(qlb, ql, 64);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hb[0] | (hb[1] << 8))));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hb[2] | (hb[3] << 8))));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      // the 12 scale bytes sit in header words y, z, w (bytes 4..15); read by shifts, so nothing goes to local memory
      const uint32_t s0 = static_cast<uint32_t>(hd.y), s1 = static_cast<uint32_t>(hd.z), s2 = static_cast<uint32_t>(hd.w);
      auto qb = [&](int k) { return static_cast<int>(((k < 4 ? s0 : (k < 8 ? s1 : s2)) >> (8 * (k & 3))) & 0xffu); };
      int sc, mn;
      if (j < 4) {
        sc = qb(j) & 63;
        mn = qb(j + 4) & 63;
      } else {
        sc = (qb(j + 4) & 0xF) | ((qb(j - 4) >> 6) << 4);
        mn = (qb(j + 4) >> 4) | ((qb(j) >> 6) << 4);
      }
      const uint8_t* q = qlb + 32 * (jj >> 1);
      int sq = 0;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const int lo = (j & 1) ? (q[l] >> 4) : (q[l] & 0xF);
        sq += (lo + (((qh[l] >> j) & 1) << 4)) * sx8(xw, 32 * jj + l);
      }
      acc += __ldg(J.xs + jg) * (d * static_cast<float>(sc) * static_cast<float>(sq) - dmin * static_cast<float>(mn) * static_cast<float>(__ldg(J.xsum + jg)));
    }
  } else {  // kQ6K
    const uint8_t* bp = J.w + rb * 208;
    int4 l4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) l4[i] = ld_cs4(bp + half * 64 + 16 * i);
    const int4 h4a = ld_cs4(bp + 128 + half * 32), h4b = ld_cs4(bp + 128 + half * 32 + 16);
    const uint2 s2 = *reinterpret_cast<const uint2*>(bp + 192 + half * 8);
    uint8_t ql[64], qh[32];
    int8_t sc[8];
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
        if (l < 16) s0 += v; else s1 += v;
      }
      acc += d * __ldg(J.xs + jg) * static_cast<float>(sc[2 * p] * s0 + sc[2 * p + 1] * s1);
    }
  }
  return acc;
}

template <int Q4V>
__device__ __forceinline__ void run_task(const Job* __restrict__ jobs, int njobs, long long t, const float* lut, int* hint = nullptr) {
  int lo = 0;  // the job whose task range holds t
  if (hint) {  // route 2 (Fable fix-6 review): a warp's t only grows, so walk forward from the last job (<= ~3 L1 probes)
    lo = *hint;
    while (lo + 1 < njobs && jobs[lo + 1].task0 <= t) ++lo;
    *hint = lo;
  } else {
    int hi = njobs - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (jobs[mid].task0 <= t) lo = mid; else hi = mid - 1;
    }
  }
  const Job& J = jobs[lo];
  const int lane = threadIdx.x & 31, rpw = 32 / J.lpr;
  const int r = static_cast<int>((t - J.task0) * rpw) + lane / J.lpr, part = lane % J.lpr;
  float acc = r < J.n ? lane_part<Q4V>(J, r, part, lut) : 0.f;
  for (int o = J.lpr / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (r < J.n && part == 0) J.y[r] = acc;
}

// (a) and (b): every warp walks the task range [t_begin, t_end) of the table
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// clk (optional, v2 step 1): block 0's thread 0 records clock64 and globaltimer (ns) at its start and its end, so the
// SM clock during the run is (c1 - c0) / (g1 - g0) GHz
// the shared grid table of fix 2 (8 KB per block), filled from c_grid at kernel start
#define R3_LUT_SETUP(Q4V)                                                                         \
  const float* s_lut = nullptr;                                                                   \
  if (Q4V) {                                                                                      \
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) s_dyn[i] = c_grid[(i >> 9) & 3][(i >> 5) & 15]; \
    __syncthreads();                                                                              \
  }

template <int Q4V>
__global__ void __launch_bounds__(256) r3_stream(const Job* __restrict__ jobs, int njobs, long long t_begin, long long t_end,
                                                 unsigned long long* clk) {
  R3_LUT_SETUP(Q4V)
  const bool rec = clk != nullptr && blockIdx.x == 0 && threadIdx.x == 0;
  if (rec) { clk[0] = clock64(); clk[1] = gtimer(); }
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  for (long long t = t_begin + (static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32); t < t_end; t += warps)
    run_task<Q4V>(jobs, njobs, t, s_lut);
  if (rec) { clk[2] = clock64(); clk[3] = gtimer(); }
}

// (c): stages in order, a grid-wide sync between them (cooperative launch)
template <int Q4V>
__global__ void __launch_bounds__(256) r3_staged(const Job* __restrict__ jobs, int njobs, const long long* __restrict__ stage_end, int nstages) {
  R3_LUT_SETUP(Q4V)
  cg::grid_group grid = cg::this_grid();
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  const long long w0 = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  long long begin = 0;
  for (int s = 0; s < nstages; ++s) {
    for (long long t = begin + w0; t < stage_end[s]; t += warps) run_task<Q4V>(jobs, njobs, t, s_lut);
    begin = stage_end[s];
    grid.sync();
  }
}

// v2 fix 6 (Fable item 6): (c) with a per-stage counter instead of cg grid.sync, waited on per warp, not per block. A warp that
// has finished its tasks of stage s adds its count to done[s] (after a fence) and, with PF, first issues L2 prefetches of the
// weights of its first task in stage s + 1 (weights never depend on the barrier), then waits until done[s] reaches the stage's
// task count. done[] is never reset: launch number epoch (1-based) waits for epoch x tasks, so no memset sits in the timing.
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// the weight bytes lane `part` of row r will read (the same addresses lane_part loads), prefetched into L2
__device__ __forceinline__ void prefetch_task(const Job* __restrict__ jobs, int njobs, long long t, int* hint = nullptr) {
  int lo = 0;
  if (hint) {
    lo = *hint;
    while (lo + 1 < njobs && jobs[lo + 1].task0 <= t) ++lo;
    *hint = lo;
  } else {
    int hi = njobs - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (jobs[mid].task0 <= t) lo = mid; else hi = mid - 1;
    }
  }
  const Job& J = jobs[lo];
  const int lane = threadIdx.x & 31, rpw = 32 / J.lpr;
  const int r = static_cast<int>((t - J.task0) * rpw) + lane / J.lpr, part = lane % J.lpr;
  if (r >= J.n) return;
  const int B = part >> 1, half = part & 1, nB = J.K / 256;
  const long long rb = static_cast<long long>(r) * nB + B;
  if (J.fmt == r3::kQ7ZT) {
    const uint8_t* bp = J.w + rb * 224 + half * 112;
    pf_l2(bp); pf_l2(bp + 111);
  } else if (J.fmt == r3::kQ4TG2) {
    const uint8_t* bp = J.w + rb * 144;
    pf_l2(bp); pf_l2(bp + 16 + half * 64); pf_l2(bp + 16 + half * 64 + 63);
  } else if (J.fmt == r3::kQ5K) {
    const uint8_t* bp = J.w + rb * 176;
    pf_l2(bp); pf_l2(bp + 47); pf_l2(bp + 48 + half * 64); pf_l2(bp + 48 + half * 64 + 63);
  } else {
    const uint8_t* bp = J.w + rb * 208;
    pf_l2(bp + half * 64); pf_l2(bp + half * 64 + 63); pf_l2(bp + 128 + half * 32); pf_l2(bp + 192 + half * 8);
  }
}

// PF: 0 = no prefetch, 1 = the warp's first next-stage task, 2 = the warp's whole next-stage share (stages are 6-24 MB, the L2
// is 36 MB, so the next stage is mostly in L2 when its loads issue; DRAM keeps streaming through the tail and the wait)
template <int Q4V, int PF>
__global__ void __launch_bounds__(256) r3_counted(const Job* __restrict__ jobs, int njobs, const long long* __restrict__ stage_end, int nstages,
                                                  unsigned long long* done, unsigned long long epoch) {
  R3_LUT_SETUP(Q4V)
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  const long long w0 = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  long long begin = 0;
  for (int s = 0; s < nstages; ++s) {
    const long long end = stage_end[s];
    unsigned long long mine = 0;
    for (long long t = begin + w0; t < end; t += warps) {
      run_task<Q4V>(jobs, njobs, t, s_lut);
      ++mine;
    }
    if (s + 1 == nstages) break;
    __syncwarp();
    if (lane == 0) {
      __threadfence();  // this warp's outputs are visible before its count is
      if (mine) atomicAdd(done + s, mine);
    }
    if (PF == 1) {
      const long long tn = end + w0;
      if (tn < stage_end[s + 1]) prefetch_task(jobs, njobs, tn);
    } else if (PF == 2) {
      for (long long tn = end + w0; tn < stage_end[s + 1]; tn += warps) prefetch_task(jobs, njobs, tn);
    }
    if (lane == 0) {
      const unsigned long long target = epoch * static_cast<unsigned long long>(end - begin);
      while (ld_acquire_u64(done + s) < target) __nanosleep(32);
    }
    __syncwarp();
    begin = end;
  }
}

// Routes 1 + 2 (Fable fix-6 review): M = 1 incremental job lookup; M = 3 incremental lookup + rolling one-task-ahead L2 prefetch
// (each warp prefetches task t + warps before computing t, so twice the bytes are in flight and the drain at a stage end is
// already covered: the prefetch crosses into the next stage before the barrier). r3_stream2 streams the table; r3_counted2 adds
// the per-stage counter barrier of fix 6.
template <int Q4V, int M>
__global__ void __launch_bounds__(256) r3_stream2(const Job* __restrict__ jobs, int njobs, long long t_begin, long long t_end) {
  R3_LUT_SETUP(Q4V)
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  int hint = 0, phint = 0;
  for (long long t = t_begin + (static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32); t < t_end; t += warps) {
    if (M & 2) {
      if (t + warps < t_end) prefetch_task(jobs, njobs, t + warps, &phint);
    }
    run_task<Q4V>(jobs, njobs, t, s_lut, &hint);
  }
}

template <int Q4V, int M>
__global__ void __launch_bounds__(256) r3_counted2(const Job* __restrict__ jobs, int njobs, const long long* __restrict__ stage_end, int nstages,
                                                   unsigned long long* done, unsigned long long epoch) {
  R3_LUT_SETUP(Q4V)
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  const long long w0 = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const long long total = stage_end[nstages - 1];
  const int lane = threadIdx.x & 31;
  int hint = 0, phint = 0;
  long long begin = 0;
  for (int s = 0; s < nstages; ++s) {
    const long long end = stage_end[s];
    unsigned long long mine = 0;
    for (long long t = begin + w0; t < end; t += warps) {
      if (M & 2) {
        if (t + warps < total) prefetch_task(jobs, njobs, t + warps, &phint);  // may reach into stage s + 1: weights only
      }
      run_task<Q4V>(jobs, njobs, t, s_lut, &hint);
      ++mine;
    }
    if (s + 1 == nstages) break;
    __syncwarp();
    if (lane == 0) {
      __threadfence();
      if (mine) atomicAdd(done + s, mine);
      const unsigned long long target = epoch * static_cast<unsigned long long>(end - begin);
      while (ld_acquire_u64(done + s) < target) __nanosleep(32);
    }
    __syncwarp();
    begin = end;
  }
}

// (c0), v2 step 1: the grid syncs alone, no tasks (Fable item 6: what one boundary costs before any work is added)
__global__ void __launch_bounds__(256) r3_sync_only(int nstages) {
  cg::grid_group grid = cg::this_grid();
  for (int s = 0; s < nstages; ++s) grid.sync();
}

// ------------------------------------------------------------------------------------------------ host
static const size_t kFlush = 64ull << 20;

struct Mat {  // one weight matrix on the host and the device
  int fmt = 0, n = 0, K = 0;
  std::vector<uint8_t> w;     // device-layout weight bytes
  std::vector<uint64_t> hdr;  // Q7ZT
  std::vector<uint8_t> mask;  // Q7ZT
  std::vector<uint32_t> hb;   // Q7ZT
  std::vector<uint16_t> d6;   // Q6K
  uint8_t* dw = nullptr;
  uint64_t* dhdr = nullptr;
  uint8_t* dmask = nullptr;
  uint32_t* dhb = nullptr;
  uint16_t* dd6 = nullptr;
  size_t bytes() const { return w.size() + hdr.size() * 8 + mask.size() + hb.size() * 4 + d6.size() * 2; }
};

static uint16_t f2h(float f) {  // round to nearest fp16 (normal range only; the scales used here are normal)
  uint32_t b;
  std::memcpy(&b, &f, 4);
  const uint32_t s = (b >> 16) & 0x8000u;
  int e = static_cast<int>((b >> 23) & 255) - 127 + 15;
  uint32_t m = b & 0x7fffffu;
  if (e <= 0) return static_cast<uint16_t>(s);
  uint32_t h = s | (static_cast<uint32_t>(e) << 10) | (m >> 13);
  if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(h);
}

static void make_mat(Mat& M, int fmt, int n, int K, std::mt19937& rng) {
  M.fmt = fmt;
  M.n = n;
  M.K = K;
  const int nB = K / 256;
  const size_t blocks = static_cast<size_t>(n) * nB;
  std::uniform_int_distribution<int> byte(0, 255), lv(20, 63), pick(0, 7), s6(8, 63), s8(-40, 40);
  std::uniform_real_distribution<float> ud(0.5f, 1.5f);
  if (fmt == r3::kQ7ZT) {
    M.w.resize(blocks * 224);
    M.hdr.resize(blocks);
    M.mask.resize(blocks);
    M.hb.resize(blocks * 2);
    for (size_t b = 0; b < blocks; ++b) {
      uint64_t h = f2h(0.0004f * ud(rng));
      for (int j = 0; j < 8; ++j) h |= static_cast<uint64_t>(lv(rng)) << (16 + 6 * j);
      M.hdr[b] = h;
      int g1 = pick(rng), g2 = pick(rng);
      while (g2 == g1) g2 = pick(rng);
      M.mask[b] = static_cast<uint8_t>((1u << g1) | (1u << g2));
      for (int i = 0; i < 224; ++i) M.w[b * 224 + i] = static_cast<uint8_t>(byte(rng));
      M.hb[b * 2] = static_cast<uint32_t>(rng());
      M.hb[b * 2 + 1] = static_cast<uint32_t>(rng());
    }
  } else if (fmt == r3::kQ4TG2) {
    M.w.resize(blocks * 144);
    for (size_t b = 0; b < blocks; ++b) {
      uint8_t* p = M.w.data() + b * 144;
      const uint16_t d = f2h(0.0004f * ud(rng));
      p[0] = static_cast<uint8_t>(d & 0xff);
      p[1] = static_cast<uint8_t>(d >> 8);
      uint64_t packed = 0;
      for (int j = 0; j < 8; ++j) packed |= static_cast<uint64_t>(lv(rng)) << (6 * j);
      for (int i = 0; i < 6; ++i) p[2 + i] = static_cast<uint8_t>(packed >> (8 * i));
      for (int i = 8; i < 144; ++i) p[i] = static_cast<uint8_t>(byte(rng));
    }
  } else if (fmt == r3::kQ5K) {
    M.w.resize(blocks * 176);
    for (size_t b = 0; b < blocks; ++b) {
      uint8_t* p = M.w.data() + b * 176;
      const uint16_t d = f2h(0.0002f * ud(rng)), dm = f2h(0.0003f * ud(rng));
      p[0] = static_cast<uint8_t>(d & 0xff); p[1] = static_cast<uint8_t>(d >> 8);
      p[2] = static_cast<uint8_t>(dm & 0xff); p[3] = static_cast<uint8_t>(dm >> 8);
      for (int i = 4; i < 176; ++i) p[i] = static_cast<uint8_t>(byte(rng));
    }
  } else {
    M.w.resize(blocks * 208);
    M.d6.resize(blocks);
    for (size_t b = 0; b < blocks; ++b) {
      uint8_t* p = M.w.data() + b * 208;
      for (int i = 0; i < 192; ++i) p[i] = static_cast<uint8_t>(byte(rng));
      for (int i = 0; i < 16; ++i) p[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(s8(rng)));
      M.d6[b] = f2h(0.0002f * ud(rng));  // fp16 normal range (0.00002 rounded to 0 and made every output 0: run 07:35)
    }
  }
}

template <typename T>
static T* up(const std::vector<T>& v) {
  if (v.empty()) return nullptr;
  T* d;
  CK(cudaMalloc(&d, v.size() * sizeof(T)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static void upload(Mat& M) {
  M.dw = up(M.w);
  M.dhdr = up(M.hdr);
  M.dmask = up(M.mask);
  M.dhb = up(M.hb);
  M.dd6 = up(M.d6);
}

struct Act {
  int K = 0;
  std::vector<int> q;
  std::vector<float> s;
  std::vector<int> sum;
  int8_t* dq = nullptr;
  float* ds = nullptr;
  int* dsum = nullptr;
};

static void make_act(Act& A, int K, std::mt19937& rng) {
  A.K = K;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(static_cast<size_t>(K));
  for (int b = 0; b < K / 32; ++b) {
    const float e = std::exp(0.8f * nd(rng));
    for (int i = 0; i < 32; ++i) x[static_cast<size_t>(b * 32 + i)] = nd(rng) * e;
  }
  A.q.assign(static_cast<size_t>(K), 0);
  A.s.assign(static_cast<size_t>(K / 32), 0.f);
  A.sum.assign(static_cast<size_t>(K / 32), 0);
  std::vector<int8_t> q8(static_cast<size_t>(K));
  for (int b = 0; b < K / 32; ++b) {
    float m = 0.f;
    for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(x[static_cast<size_t>(b * 32 + i)]));
    const float s = m > 0.f ? m / 127.f : 1.f;
    A.s[static_cast<size_t>(b)] = s;
    for (int i = 0; i < 32; ++i) {
      const int v = std::max(-127, std::min(127, static_cast<int>(std::nearbyint(x[static_cast<size_t>(b * 32 + i)] / s))));
      A.q[static_cast<size_t>(b * 32 + i)] = v;
      q8[static_cast<size_t>(b * 32 + i)] = static_cast<int8_t>(v);
      A.sum[static_cast<size_t>(b)] += v;
    }
  }
  A.dq = up(q8);
  A.ds = up(A.s);
  A.dsum = up(A.sum);
}

// host reference: sum over 32-blocks of scale x (decoded weights . int8 activation), in double
static std::vector<double> reference(const Mat& M, const Act& A) {
  const int nB = M.K / 256;
  std::vector<double> y(static_cast<size_t>(M.n));
  float w[32];
  for (int r = 0; r < M.n; ++r) {
    double acc = 0;
    for (int B = 0; B < nB; ++B) {
      const long long rb = static_cast<long long>(r) * nB + B;
      for (int j = 0; j < 8; ++j) {
        if (M.fmt == r3::kQ7ZT) r3::q7zt_decode32(M.w.data(), M.hdr.data(), M.mask.data(), M.hb.data(), rb, j, w);
        else if (M.fmt == r3::kQ4TG2) r3::q4tg2_decode32(M.w.data() + rb * 144, j, w);
        else if (M.fmt == r3::kQ5K) r3::q5k_decode32(M.w.data() + rb * 176, j, w);
        else r3::q6k_decode32(M.w.data() + rb * 208, M.d6[static_cast<size_t>(rb)], j, w);
        double p = 0;
        for (int i = 0; i < 32; ++i) p += static_cast<double>(w[i]) * A.q[static_cast<size_t>(B * 256 + j * 32 + i)];
        acc += p * A.s[static_cast<size_t>(B * 8 + j)];
      }
    }
    y[static_cast<size_t>(r)] = acc;
  }
  return y;
}

static Job job_of(const Mat& M, const Act& A, float* y) {
  Job J{};
  J.fmt = M.fmt; J.n = M.n; J.K = M.K; J.lpr = M.K / 128;
  J.ntask = (M.n + 32 / J.lpr - 1) / (32 / J.lpr);
  J.w = M.dw; J.hdr = M.dhdr; J.mask = M.dmask; J.hb = M.dhb; J.d6 = M.dd6;
  J.xq = A.dq; J.xs = A.ds; J.xsum = A.dsum; J.y = y;
  return J;
}

static void set_grids() {
  float g[4][16];
  for (int t = 0; t < 4; ++t)
    for (int i = 0; i < 16; ++i) g[t][i] = r3::tg2_grid(t, i);
  CK(cudaMemcpyToSymbol(c_grid, g, sizeof g));
}

static int sms() {
  int n = 0;
  CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0));
  return n;
}

// ---- check: one matrix of every format and shape, GPU vs the host reference
static int run_check() {
  set_grids();
  std::mt19937 rng(2026);
  const struct { int fmt, n, K; const char* name; } C[] = {
      {r3::kQ7ZT, 2048, 2048, "Q7_ZT 2048 x 2048"}, {r3::kQ7ZT, 512, 4096, "Q7_ZT 512 x 4096"}, {r3::kQ7ZT, 2048, 512, "Q7_ZT 2048 x 512"},
      {r3::kQ4TG2, 512, 2048, "Q4_TG2 512 x 2048 (expert gate/up)"}, {r3::kQ5K, 2048, 512, "Q5_K 2048 x 512 (expert down)"},
      {r3::kQ6K, 4096, 2048, "Q6_K 4096 x 2048 (output head rows)"}};
  Act A512, A2048, A4096;
  make_act(A512, 512, rng);
  make_act(A2048, 2048, rng);
  make_act(A4096, 4096, rng);
  int fails = 0;
  for (const auto& c : C) {
    Mat M;
    make_mat(M, c.fmt, c.n, c.K, rng);
    upload(M);
    const Act& A = c.K == 512 ? A512 : (c.K == 2048 ? A2048 : A4096);
    float* dy;
    CK(cudaMalloc(&dy, c.n * 4));
    Job J = job_of(M, A, dy);
    J.task0 = 0;
    Job* dj = up(std::vector<Job>{J});
    const std::vector<double> yr = reference(M, A);
    for (int v = 0; v < 6; ++v) {  // paths 0-2 (fix 2); 3 = fix 6's counted kernel over 2 stages; 4 = routes 1+2 streaming; 5 = routes 1+2 counted, 2 stages
      CK(cudaMemset(dy, 0, c.n * 4));
      if (v == 4) r3_stream2<2, 3><<<sms() * 4, 256, kLutBytes>>>(dj, 1, 0, J.ntask);
      else if (v == 5) {
        int oc = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oc, r3_counted2<2, 3>, 256, kLutBytes));
        std::vector<long long> se{J.ntask / 2, J.ntask};
        long long* dse = up(se);
        unsigned long long* ddone;
        CK(cudaMalloc(&ddone, 2 * sizeof(unsigned long long)));
        CK(cudaMemset(ddone, 0, 2 * sizeof(unsigned long long)));
        int nj = 1, ns = 2;
        unsigned long long ep = 1;
        void* a5[] = {&dj, &nj, &dse, &ns, &ddone, &ep};
        CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted2<2, 3>), dim3(sms() * oc), dim3(256), a5, kLutBytes, nullptr));
      } else if (v == 0) r3_stream<0><<<sms() * 4, 256>>>(dj, 1, 0, J.ntask, nullptr);
      else if (v == 1) r3_stream<1><<<sms() * 4, 256, kLutBytes>>>(dj, 1, 0, J.ntask, nullptr);
      else if (v == 2) r3_stream<2><<<sms() * 4, 256, kLutBytes>>>(dj, 1, 0, J.ntask, nullptr);
      else {
        int oc = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oc, r3_counted<2, 2>, 256, kLutBytes));
        std::vector<long long> se{J.ntask / 2, J.ntask};
        long long* dse = up(se);
        unsigned long long* ddone;
        CK(cudaMalloc(&ddone, 2 * sizeof(unsigned long long)));
        CK(cudaMemset(ddone, 0, 2 * sizeof(unsigned long long)));
        int nj = 1, ns = 2;
        unsigned long long ep = 1;
        void* a3[] = {&dj, &nj, &dse, &ns, &ddone, &ep};
        CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted<2, 2>), dim3(sms() * oc), dim3(256), a3, kLutBytes, nullptr));
      }
      CK(cudaDeviceSynchronize());
      std::vector<float> yg(static_cast<size_t>(c.n));
      CK(cudaMemcpy(yg.data(), dy, c.n * 4, cudaMemcpyDeviceToHost));
      double worst = 0, scale = 0;
      for (int r = 0; r < c.n; ++r) {
        scale = std::max(scale, std::fabs(yr[static_cast<size_t>(r)]));
        worst = std::max(worst, std::fabs(yg[static_cast<size_t>(r)] - yr[static_cast<size_t>(r)]));
      }
      const bool ok = scale > 0 && worst / scale < 2e-5;  // an all-zero output is a failed test, not a pass
      std::printf("  %-38s path %d  max error %.2e of max |y| %.3g (%s)\n", c.name, v, worst / scale, scale, ok ? "ok" : "FAIL");
      fails += !ok;
    }
  }
  std::printf("%s\n", fails ? "SOME FORMATS FAILED" : "every format matches the host reference");
  return fails ? 1 : 0;
}

// ---- stream: one token's weight reads, three ways
static int run_stream(int soak_s, bool distinct) {
  set_grids();
  std::mt19937 rng(7);
  const int nsm = sms();
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  Act A512, A2048, A4096;
  make_act(A512, 512, rng);
  make_act(A2048, 2048, rng);
  make_act(A4096, 4096, rng);
  // four distinct layers (one attention, three DeltaNet), reused cyclically for 40 (a 4-layer cycle is ~200 MB, far past
  // the 36 MB L2, so every read is cold); 7 hot experts per layer (Q4_TG2 gate and up, Q5_K down), output head Q6_K
  struct L { std::vector<Mat> m; std::vector<int> stage; std::vector<const Act*> a; };
  // --distinct: 40 distinct layers (about 2.27 GB, fits the 12 GB card), so no weight is read twice in a token (no L2 reuse)
  const int nl = distinct ? 40 : 4;
  std::vector<L> layers(static_cast<size_t>(nl));
  auto add = [&](L& l, int fmt, int n, int K, int st) {
    l.m.emplace_back();
    make_mat(l.m.back(), fmt, n, K, rng);
    l.stage.push_back(st);
    l.a.push_back(K == 512 ? &A512 : (K == 2048 ? &A2048 : &A4096));
  };
  for (int li = 0; li < nl; ++li) {
    L& l = layers[static_cast<size_t>(li)];
    if (distinct ? li % 4 == 3 : li == 0) {  // attention: q (with gate), k, v | o
      add(l, r3::kQ7ZT, 8192, 2048, 0); add(l, r3::kQ7ZT, 512, 2048, 0); add(l, r3::kQ7ZT, 512, 2048, 0);
      add(l, r3::kQ7ZT, 2048, 4096, 1);
    } else {        // DeltaNet: qkv, gate | ssm_out
      add(l, r3::kQ7ZT, 8192, 2048, 0); add(l, r3::kQ7ZT, 4096, 2048, 0);
      add(l, r3::kQ7ZT, 2048, 4096, 1);
    }
    add(l, r3::kQ7ZT, 512, 2048, 2); add(l, r3::kQ7ZT, 512, 2048, 2);  // shared expert gate, up
    for (int e = 0; e < 7; ++e) { add(l, r3::kQ4TG2, 512, 2048, 2); add(l, r3::kQ4TG2, 512, 2048, 2); }
    add(l, r3::kQ7ZT, 2048, 512, 3);  // shared expert down
    for (int e = 0; e < 7; ++e) add(l, r3::kQ5K, 2048, 512, 3);
    for (Mat& m : l.m) upload(m);
  }
  Mat head;
  make_mat(head, r3::kQ6K, 248320, 2048, rng);
  upload(head);
  // the token's job table: 40 layers in order, 4 stages each, then the head (stage 160)
  std::vector<Job> jobs;
  std::vector<int> jstage;
  float* dy;
  CK(cudaMalloc(&dy, 248320 * 4));  // every job writes into the same scratch output (timing only)
  size_t bytes = 0;
  for (int L = 0; L < 40; ++L) {
    const auto& l = layers[static_cast<size_t>(distinct ? L : (L % 4 == 3 ? 0 : 1 + L % 3))];  // attention on every 4th layer
    for (int st = 0; st < 4; ++st)
      for (size_t i = 0; i < l.m.size(); ++i)
        if (l.stage[i] == st) {
          jobs.push_back(job_of(l.m[i], *l.a[i], dy));
          jstage.push_back(4 * L + st);
          bytes += l.m[i].bytes();
        }
  }
  jobs.push_back(job_of(head, A2048, dy));
  jstage.push_back(160);
  bytes += head.bytes();
  long long t = 0;
  for (Job& J : jobs) { J.task0 = t; t += J.ntask; }
  const long long total = t;
  const int nstages = 161;
  std::vector<long long> stage_end(nstages, 0);
  for (size_t i = 0; i < jobs.size(); ++i) stage_end[static_cast<size_t>(jstage[i])] = jobs[i].task0 + jobs[i].ntask;
  Job* dj = up(jobs);
  long long* dse = up(stage_end);
  void* flush;
  CK(cudaMalloc(&flush, kFlush));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::map<double, std::pair<double, double>> spread;
  auto timed = [&](auto body) {
    std::vector<float> ts;
    body();
    CK(cudaDeviceSynchronize());
    for (int rep = 0; rep < 21; ++rep) {  // 21 (Fable: >= 20 so outliers from other GPU clients show in min / max)
      CK(cudaMemsetAsync(flush, rep & 0xff, kFlush));
      CK(cudaEventRecord(e0));
      body();
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const double med = static_cast<double>(ts[ts.size() / 2]);
    spread[med] = {static_cast<double>(ts.front()), static_cast<double>(ts.back())};  // min / max beside the median (outliers)
    return med;
  };
  size_t head_bytes = head.bytes();
  std::printf("device %s, %d SMs; one token: %zu jobs (40 layers x [dense Q7_ZT, shared expert, 7 experts Q4_TG2 gate/up + Q5_K down] + "
              "Q6_K head), %.3f GB of weights (head %.3f GB), %lld warp tasks; cold (64 MiB written first), median of 21\n",
              prop.name, nsm, jobs.size(), bytes / 1e9, head_bytes / 1e9, total);
  // (a) one launch per matrix, queued back to back
  auto grid_a = [&](const Job& J) { return std::max(1, static_cast<int>(std::min<long long>((J.ntask + 7) / 8, nsm * 8LL))); };
  const double ta = timed([&] {
    for (size_t i = 0; i < jobs.size(); ++i)
      r3_stream<0><<<grid_a(jobs[i]), 256>>>(dj, static_cast<int>(jobs.size()), jobs[i].task0, jobs[i].task0 + jobs[i].ntask, nullptr);
  });
  // (a') the same 1,091 launches captured once in a CUDA graph and replayed: the GPU-side cost without the per-launch host
  // submission path (under WSL2 that path goes through GPU-PV; Fable harness note)
  cudaStream_t cs;
  CK(cudaStreamCreate(&cs));  // a blocking stream: ordered with the legacy stream's flush and events, so e0/e1 bracket the replay
  cudaGraph_t graph;
  cudaGraphExec_t gexec;
  CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
  for (size_t i = 0; i < jobs.size(); ++i)
    r3_stream<0><<<grid_a(jobs[i]), 256, 0, cs>>>(dj, static_cast<int>(jobs.size()), jobs[i].task0, jobs[i].task0 + jobs[i].ntask, nullptr);
  CK(cudaStreamEndCapture(cs, &graph));
  CK(cudaGraphInstantiate(&gexec, graph, 0));
  const double tg = timed([&] {
    CK(cudaGraphLaunch(gexec, cs));
  });
  // (b) one launch for the whole table; the SM clock is recorded inside the run
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, r3_stream<0>, 256, 0));
  unsigned long long* dclk;
  CK(cudaMalloc(&dclk, 4 * sizeof(unsigned long long)));
  const double tb = timed([&] { r3_stream<0><<<nsm * occ, 256>>>(dj, static_cast<int>(jobs.size()), 0, total, dclk); });
  unsigned long long clk[4];
  CK(cudaMemcpy(clk, dclk, sizeof clk, cudaMemcpyDeviceToHost));
  const double ghz_b = clk[3] > clk[1] ? static_cast<double>(clk[2] - clk[0]) / static_cast<double>(clk[3] - clk[1]) : 0.0;
  // (c) one cooperative launch, a grid-wide sync after every stage
  int occc = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occc, r3_staged<0>, 256, 0));
  int njobs = static_cast<int>(jobs.size()), nst = nstages;
  void* args[] = {&dj, &njobs, &dse, &nst};
  const double tc = timed([&] { CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_staged<0>), dim3(nsm * occc), dim3(256), args, 0, nullptr)); });
  // (c0) the 161 syncs alone, same grid as (c)
  void* args0[] = {&nst};
  const double tc0 = timed([&] { CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_sync_only), dim3(nsm * occc), dim3(256), args0, 0, nullptr)); });
  // (c0') one cooperative launch with a single sync: the launch alone, so (c0 - c0') / 160 is one boundary
  int one = 1;
  void* args1[] = {&one};
  const double tc1 = timed([&] { CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_sync_only), dim3(nsm * occc), dim3(256), args1, 0, nullptr)); });
  // v2 fix 2: the same (b) and (c) with the shared-memory grid (path 1) and with it plus the magic-number int8 -> float (path 2)
  int occ1 = 0, occ2 = 0, occc2 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, r3_stream<1>, 256, kLutBytes));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ2, r3_stream<2>, 256, kLutBytes));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occc2, r3_staged<2>, 256, kLutBytes));
  const double tb1 = timed([&] { r3_stream<1><<<nsm * occ1, 256, kLutBytes>>>(dj, static_cast<int>(jobs.size()), 0, total, nullptr); });
  const double tb2 = timed([&] { r3_stream<2><<<nsm * occ2, 256, kLutBytes>>>(dj, static_cast<int>(jobs.size()), 0, total, nullptr); });
  const double tc2 = timed([&] { CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_staged<2>), dim3(nsm * occc2), dim3(256), args, kLutBytes, nullptr)); });
  // v2 fix 6: the counter barrier (c1) and the counter barrier with the next stage's weights prefetched before the wait (c2)
  int occk1 = 0, occk2 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occk1, r3_counted<2, 0>, 256, kLutBytes));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occk2, r3_counted<2, 1>, 256, kLutBytes));
  unsigned long long* ddone;
  CK(cudaMalloc(&ddone, nstages * sizeof(unsigned long long)));
  unsigned long long ep1 = 0, ep2 = 0;
  CK(cudaMemset(ddone, 0, nstages * sizeof(unsigned long long)));
  void* ak1[] = {&dj, &njobs, &dse, &nst, &ddone, &ep1};
  const double tk1 = timed([&] { ++ep1; CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted<2, 0>), dim3(nsm * occk1), dim3(256), ak1, kLutBytes, nullptr)); });
  CK(cudaDeviceSynchronize());
  CK(cudaMemset(ddone, 0, nstages * sizeof(unsigned long long)));
  void* ak2[] = {&dj, &njobs, &dse, &nst, &ddone, &ep2};
  const double tk2 = timed([&] { ++ep2; CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted<2, 1>), dim3(nsm * occk2), dim3(256), ak2, kLutBytes, nullptr)); });
  CK(cudaDeviceSynchronize());
  int occk3 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occk3, r3_counted<2, 2>, 256, kLutBytes));
  unsigned long long ep3 = 0;
  CK(cudaMemset(ddone, 0, nstages * sizeof(unsigned long long)));
  void* ak3[] = {&dj, &njobs, &dse, &nst, &ddone, &ep3};
  const double tk3 = timed([&] { ++ep3; CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted<2, 2>), dim3(nsm * occk3), dim3(256), ak3, kLutBytes, nullptr)); });
  // routes 1 + 2 (Fable fix-6 review): incremental lookup (M1), + rolling one-task-ahead prefetch (M3); streaming and counted
  int os1 = 0, os3 = 0, oq1 = 0, oq3 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&os1, r3_stream2<2, 1>, 256, kLutBytes));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&os3, r3_stream2<2, 3>, 256, kLutBytes));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oq1, r3_counted2<2, 1>, 256, kLutBytes));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oq3, r3_counted2<2, 3>, 256, kLutBytes));
  const double tm1 = timed([&] { r3_stream2<2, 1><<<nsm * os1, 256, kLutBytes>>>(dj, static_cast<int>(jobs.size()), 0, total); });
  const double tm3 = timed([&] { r3_stream2<2, 3><<<nsm * os3, 256, kLutBytes>>>(dj, static_cast<int>(jobs.size()), 0, total); });
  unsigned long long eq1 = 0, eq3 = 0;
  CK(cudaDeviceSynchronize());
  CK(cudaMemset(ddone, 0, nstages * sizeof(unsigned long long)));
  void* aq1[] = {&dj, &njobs, &dse, &nst, &ddone, &eq1};
  const double tq1 = timed([&] { ++eq1; CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted2<2, 1>), dim3(nsm * oq1), dim3(256), aq1, kLutBytes, nullptr)); });
  CK(cudaDeviceSynchronize());
  CK(cudaMemset(ddone, 0, nstages * sizeof(unsigned long long)));
  void* aq3[] = {&dj, &njobs, &dse, &nst, &ddone, &eq3};
  const double tq3 = timed([&] { ++eq3; CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(r3_counted2<2, 3>), dim3(nsm * oq3), dim3(256), aq3, kLutBytes, nullptr)); });
  CK(cudaGetLastError());
  auto row = [&](const char* name, double ms, size_t by) {
    const auto it = spread.find(ms);
    if (it != spread.end())
      std::printf("  %-62s %8.3f ms  %6.1f GB/s   (min %.3f, max %.3f)\n", name, ms, by / (ms * 1e6), it->second.first, it->second.second);
    else
      std::printf("  %-62s %8.3f ms  %6.1f GB/s\n", name, ms, by / (ms * 1e6));
  };
  row("(a) one launch per matrix (today's kernels), queued", ta, bytes);
  row("(a') the same 1,091 launches replayed from a CUDA graph", tg, bytes);
  row("(b) one streaming launch, whole table", tb, bytes);
  row("(c) one cooperative launch, grid sync after each of 161 stages", tc, bytes);
  std::printf("  blocks: (b) %d per SM, (c) %d per SM; (c) - (b) = %.3f ms over 160 syncs = %.2f us per stage boundary\n", occ, occc, tc - tb,
              (tc - tb) * 1e3 / 160.0);
  std::printf("  (c0) 161 grid syncs, no work: %.3f ms; one sync: %.3f ms; per boundary %.2f us (sync alone)\n", tc0, tc1,
              (tc0 - tc1) * 1e3 / 160.0);
  std::printf("  SM clock inside (b), block 0: %.3f GHz (clock64 / globaltimer over %.3f ms)\n", ghz_b, (clk[3] - clk[1]) / 1e6);
  std::printf("v2 fix 2 (Q4_TG2 grid in shared memory), whole table:\n");
  row("(b) grid path 1: shared-memory grid", tb1, bytes);
  row("(b) grid path 2: shared-memory grid + magic int8 -> float", tb2, bytes);
  row("(c) grid path 2, grid sync after each of 161 stages", tc2, bytes);
  std::printf("  blocks per SM: path 1 %d, path 2 %d, (c) path 2 %d\n", occ1, occ2, occc2);
  std::printf("v2 fix 6 (path 2; per-stage counter waited per warp instead of a grid sync):\n");
  row("(c1) counter barrier", tk1, bytes);
  row("(c2) counter barrier + next stage's weights prefetched before the wait", tk2, bytes);
  row("(c3) counter barrier + the whole next-stage share prefetched", tk3, bytes);
  std::printf("routes 1 + 2 (path 2; incremental job lookup = M1, + rolling one-task-ahead L2 prefetch = M3):\n");
  row("(b) M1 incremental lookup", tm1, bytes);
  row("(b) M3 incremental lookup + rolling prefetch", tm3, bytes);
  row("(c) M1 counter barrier, incremental lookup", tq1, bytes);
  row("(c) M3 counter barrier, incremental lookup + rolling prefetch", tq3, bytes);
  std::printf("  per stage boundary, M3 counted over M3 streaming: %.2f us; blocks per SM %d / %d / %d / %d\n", (tq3 - tm3) * 1e3 / 160.0, os1, os3, oq1, oq3);
  std::printf("  per stage boundary over (b) path 2: grid sync %.2f us, counter %.2f us, + one-task prefetch %.2f us, + whole-share prefetch %.2f us; blocks per SM %d / %d / %d\n",
              (tc2 - tb2) * 1e3 / 160.0, (tk1 - tb2) * 1e3 / 160.0, (tk2 - tb2) * 1e3 / 160.0, (tk3 - tb2) * 1e3 / 160.0, occk1, occk2, occk3);

  // ---- per-format (b) rows (Fable item 1): one format per table, real shapes, >= 256 MiB of distinct matrices (one
  // allocation, jobs are row slices of it; nothing is read twice, so nothing is served from the 36 MB L2)
  std::printf("per format, (b) one streaming launch, cold, median of 21:\n");
  struct PF { int fmt, n, K; const char* name; };
  const PF pf[] = {{r3::kQ7ZT, 8192, 2048, "Q7_ZT only (8192 x 2048 matrices, dense)"},
                   {r3::kQ4TG2, 512, 2048, "Q4_TG2 only (512 x 2048, expert gate/up)"},
                   {r3::kQ5K, 2048, 512, "Q5_K only (2048 x 512, expert down)"}};
  const size_t want = 256ull << 20;
  for (const PF& f : pf) {
    const size_t per = static_cast<size_t>(f.n) * (f.K / 256) * (f.fmt == r3::kQ7ZT ? 241 : (f.fmt == r3::kQ4TG2 ? 144 : 176));
    const int count = static_cast<int>((want + per - 1) / per);
    Mat big;
    make_mat(big, f.fmt, f.n * count, f.K, rng);
    upload(big);
    const Act& A = f.K == 512 ? A512 : A2048;
    std::vector<Job> pj;
    long long t0 = 0;
    const long long nB = f.K / 256;
    for (int c = 0; c < count; ++c) {
      Job J = job_of(big, A, dy);
      const long long rb0 = static_cast<long long>(c) * f.n * nB;  // first 256-block of this slice
      J.n = f.n;
      J.ntask = (f.n + 32 / J.lpr - 1) / (32 / J.lpr);
      if (f.fmt == r3::kQ7ZT) { J.w += rb0 * 224; J.hdr += rb0; J.mask += rb0; J.hb += rb0 * 2; }
      else J.w += rb0 * (f.fmt == r3::kQ4TG2 ? 144 : 176);
      J.task0 = t0;
      t0 += J.ntask;
      pj.push_back(J);
    }
    Job* dpj = up(pj);
    int np = static_cast<int>(pj.size());
    const double tp = timed([&] { r3_stream<0><<<nsm * occ, 256>>>(dpj, np, 0, t0, nullptr); });
    char name[128];
    std::snprintf(name, sizeof name, "%s, %d matrices", f.name, count);
    row(name, tp, big.bytes());
    if (f.fmt != r3::kQ4TG2) {
      const double tpm = timed([&] { r3_stream2<2, 3><<<nsm * os3, 256, kLutBytes>>>(dpj, np, 0, t0); });
      row("  the same, routes 1 + 2 (rolling prefetch)", tpm, big.bytes());
    }
    if (f.fmt == r3::kQ4TG2) {
      const double tp1 = timed([&] { r3_stream<1><<<nsm * occ1, 256, kLutBytes>>>(dpj, np, 0, t0, nullptr); });
      const double tp2 = timed([&] { r3_stream<2><<<nsm * occ2, 256, kLutBytes>>>(dpj, np, 0, t0, nullptr); });
      row("  the same, grid path 1 (shared-memory grid)", tp1, big.bytes());
      row("  the same, grid path 2 (+ magic int8 -> float)", tp2, big.bytes());
      const double tp3 = timed([&] { r3_stream2<2, 3><<<nsm * os3, 256, kLutBytes>>>(dpj, np, 0, t0); });
      row("  the same, path 2 + routes 1 + 2 (rolling prefetch)", tp3, big.bytes());
    }
    CK(cudaFree(dpj));
    CK(cudaFree(big.dw));
    if (big.dhdr) CK(cudaFree(big.dhdr));
    if (big.dmask) CK(cudaFree(big.dmask));
    if (big.dhb) CK(cudaFree(big.dhb));
  }
  {  // Q6_K: the output head alone (one 417 MB job, distinct rows)
    Job J = job_of(head, A2048, dy);
    J.task0 = 0;
    Job* dh = up(std::vector<Job>{J});
    const double th = timed([&] { r3_stream<0><<<nsm * occ, 256>>>(dh, 1, 0, J.ntask, nullptr); });
    row("Q6_K only (the 248,320 x 2048 output head)", th, head.bytes());
    CK(cudaFree(dh));
  }
  if (soak_s > 0) {  // (b) back to back for an outside clock / power log (no flush between runs)
    std::printf("soak: (b) path 2 + routes 1 + 2 back to back for %d s\n", soak_s);
    std::fflush(stdout);
    CK(cudaDeviceSynchronize());
    const auto s0 = std::chrono::steady_clock::now();
    long long runs = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count() < soak_s) {
      for (int k = 0; k < 20; ++k) r3_stream2<2, 3><<<nsm * os3, 256, kLutBytes>>>(dj, static_cast<int>(jobs.size()), 0, total);
      CK(cudaDeviceSynchronize());
      runs += 20;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
    std::printf("  soak: %lld runs in %.2f s = %.3f ms per run, %.1f GB/s\n", runs, sec, sec * 1e3 / runs, bytes * runs / (sec * 1e9));
  }
  CK(cudaFree(flush));
  return 0;
}

static void print_kernels() {
  const struct { const void* f; const char* name; int smem; } K[] = {{reinterpret_cast<const void*>(r3_stream<0>), "r3_stream<0>", 0},
                                                           {reinterpret_cast<const void*>(r3_stream<1>), "r3_stream<1>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_stream<2>), "r3_stream<2>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_staged<0>), "r3_staged<0>", 0},
                                                           {reinterpret_cast<const void*>(r3_staged<2>), "r3_staged<2>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_counted<2, 0>), "r3_counted<2,0>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_counted<2, 1>), "r3_counted<2,1>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_counted<2, 2>), "r3_counted<2,2>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_stream2<2, 3>), "r3_stream2<2,3>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_counted2<2, 3>), "r3_counted2<2,3>", kLutBytes},
                                                           {reinterpret_cast<const void*>(r3_sync_only), "r3_sync_only", 0}};
  for (const auto& k : K) {
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, k.f));
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k.f, 256, k.smem));
    std::printf("kernel %-14s registers %3d, local memory %zu bytes, spill-free %s, blocks of 256 per SM %d\n", k.name, fa.numRegs,
                fa.localSizeBytes, fa.localSizeBytes == 0 ? "yes" : "NO", occ);
  }
  int clk = 0, mclk = 0, bus = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);        // kHz, the board's rated SM clock
  cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);  // kHz
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
  cudaGetLastError();  // an attribute this runtime does not report leaves 0, not an error for the run
  std::printf("device attributes: SM clock %.3f GHz (rated), memory clock %.3f GHz, bus %d bits (peak from these %.1f GB/s; "
              "the memory clock the driver reports may be the stock one, nvidia-smi shows the clock in use)\n",
              clk / 1e6, mclk / 1e6, bus, 2.0 * mclk * 1e3 * (bus / 8) / 1e9);
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "check") return run_check();
  if (cmd == "stream") {
    int soak = 0;
    for (int i = 2; i + 1 < argc; ++i)
      if (std::string(argv[i]) == "--soak") soak = std::atoi(argv[i + 1]);
    bool distinct = false;
    for (int i = 2; i < argc; ++i)
      if (std::string(argv[i]) == "--distinct") distinct = true;
    print_kernels();
    std::printf("layers: %s\n", distinct ? "40 distinct (no weight read twice per token)" : "4 distinct, cycled over 40");
    return run_stream(soak, distinct);
  }
  std::fprintf(stderr, "usage: r3 check | stream [--soak SECONDS] [--distinct]\n");
  return 1;
}

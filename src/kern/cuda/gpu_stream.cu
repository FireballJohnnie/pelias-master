// provenance: original (moved from tools/probes/r3/r3.cu, the measured R3 probe; the device math is the probe's)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/cuda/gpu_stream.cu - the R3 streaming kernel behind kern/gpu_stream.h. lane_part is the probe's lane_part<2> (grid
// path 2), run_task its route-2 form (incremental lookup), k_stream its r3_stream2<2, 1> and k_staged its r3_grouped<2>
// (item 7: per-expert counters inside a stage, a counter barrier per warp between stages). What changed in moving: the
// template parameters fixed at the measured choices, the job struct's pointers typed for the engine, the counters and the
// table in one device buffer per plan, the launch and its checks on the host side.
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kern/cuda/moe_dev.cuh"
#include "kern/gpu_decode.h"
#include "kern/gpu_stream.h"
#include "kern/tg2_planes.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::kern::gpu {
cudaStream_t cur_stream_raw();  // gpu_ops.cu (v26)
// gpu_ops.cu (v46a-1): the reader v3 grouping and combine, a GPU stamp
bool rd3_group_launch(const ExpertPtr* tab, const ExpertPtr* shx, const std::int32_t* sel, int tokens, int nu, const ReaderCache* cache, std::int64_t H,
                      std::int64_t F, int* grp, ExpertPtr* gsrc, ExpertPtr* gfill, Stream s);
bool rd3_combine_launch(const float* yd, const float* wsel, const float* gsh, int nu, std::int64_t tokens, std::int64_t H, float* y,
                        const std::int32_t* sel, Stream s);
void stamp_launch(unsigned long long* p, Stream s);
namespace {

// v46a-1: kQ6KS Q6_K as stored (210-byte blocks, 2-byte aligned: the expert stage reads the file's / bank's bytes); kEpi
// the expert stage's silu(g) x u rounding (not a matrix: one warp task per expert group)
enum Fmt : int { kQ7ZT = 0, kQ4TG2 = 1, kQ5K = 2, kQ6K = 3, kQ8_0 = 4, kQ4K = 5, kIQ4XS = 6, kQ6KS = 7, kEpi = 8, kQ4TG2P = 9, kQ5KP = 10 };  // v54 kQ4TG2P / v54c kQ5KP: Q4_TG2 / Q5_K in bit planes (kern/tg2_planes.h)

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
  int ntok;                 // v26: tokens (1-4); token t reads xq + tokx[t] xqs, xs / xsum + tokx[t] xss, writes y + ypk[t] ys
  long long xqs, xss, ys;
  int tokx[4], ypk[4];      // v46a-1: input row and output row of each token (t and t for a plan's jobs; an expert's picks)
  long long fd;             // v46a-1: 0, or the cache slot's offset from w: the loaded weight bytes are also stored at w + fd
  float* gu;                // v46a-1 kEpi: gate | up rows per pick (2K floats at gu + ypk 2K), K = F
  std::int8_t* hq;          // kEpi: int8 per pick (F at hq + ypk F), scale and sum per 32 (hs / hsum + ypk F / 32)
  float* hs;
  int* hsum;
  int stg;                  // v46a-2: 1 = the warp stages its task's rows contiguously in shared memory first (lane_part M = 3)
  int pb;                   // v54b kQ4TG2P: the code bits read (2-4)
  long long rs;             // kQ4TG2P: w's row bytes (144 nB in RAM; the prefix of pb bits in a cache slot)
  std::uint8_t* fw;         // kQ4TG2P: a miss's slot matrix (its rows packed at frs bytes: the prefix), or null
  long long frs;
  // v55f (the 1-bit route): the inputs' bit planes (act_bits32: 8 words per 32 values, at word (input byte offset) / 4
  // from xb, so they follow xq's layout), or null (the dp4a route); hbo: where a kEpi job writes its picks' planes
  const std::uint32_t* xb;
  std::uint32_t* hbo;
};

__constant__ float c_grid[4][16];  // the Q4_TG2 grids (quant::q4_tg_grid(tag, 2)), uploaded once
__constant__ float c_grid3[4][16];  // v54: the 3-bit and 2-bit base grids (tg2p_base_grid of c_grid), uploaded once
__constant__ float c_grid2[4][16];
__constant__ int c_rx_bits;
int g_stage_bytes = 16 * 272 + 32;  // v55f: a warp's staging buffer: 16 blocks of the largest staged format + 32 (rx_set_stage_bytes)
bool g_lean_dense = false;  // v55e experiment switch (--lean-dense since v55e3): v55d's lean one-format dense kernels; off (v55e2: +1.1 ms per token, MEASURED); removed once understood  // v55b: the one grid the expert stage loads (4, 3 or 2; 0 = 4)
__device__ __forceinline__ unsigned long long gtimer_s() {  // v55b: the GPU's global nanosecond clock (as gpu_ops.cu's gtimer)
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

thread_local std::string s_err;
bool sfail(const std::string& why) {
  s_err = why;
  return false;
}
bool sck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  return sfail(std::string(what) + ": " + cudaGetErrorString(e));
}
cudaStream_t sst(Stream s) { return s ? static_cast<cudaStream_t>(s) : cur_stream_raw(); }  // v26

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

// v26: weight loads stream past the caches for one token (as before); with several tokens they stay cached, since the
// same row is read again for the next token (L1 / L2 instead of DRAM). The values are the same either way.
__device__ __forceinline__ int4 ldw(bool cs, const void* p) { return cs ? ld_cs4(p) : __ldg(reinterpret_cast<const int4*>(p)); }

// v46a-1: the expert stage reads inputs written earlier in the same launch (the int8 silu x up), so its activation loads go
// through L2 (ld.global.cg: no stale L1 line); the dense path (M = 0) keeps __ldg. M: 0 dense, 1 expert, 2 expert + fill.
__device__ __forceinline__ int4 ld_cg4(const void* p) {
  int4 r;
  asm volatile("ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float ld_cgf(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int ld_cgi(const int* p) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ std::uint32_t ld_cs1(const void* p) {
  std::uint32_t v;
  asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
template <int M>
__device__ __forceinline__ int4 ldx4(const int4* p) {
  if constexpr (M == 0) return __ldg(p);
  else return ld_cg4(p);
}
template <int M>
__device__ __forceinline__ float ldxs(const float* p) {
  if constexpr (M == 0) return __ldg(p);
  else return ld_cgf(p);
}
template <int M>
__device__ __forceinline__ int ldxi(const int* p) {
  if constexpr (M == 0) return __ldg(p);
  else return ld_cgi(p);
}
// a weight load; with M = 2 (a cache miss being filled) and st, the 16 bytes also go to the same offset in the slot
// v46a-2: M = 3 reads the warp's staged copy in shared memory (generic loads; the staging copy did any slot fill)
template <int M>
__device__ __forceinline__ int4 ldwf(const DJob& J, bool cs, const std::uint8_t* p, bool st) {
  if constexpr (M == 3) return *reinterpret_cast<const int4*>(p);
  const int4 v = ldw(cs, p);
  if constexpr (M == 2) {
    if (st) *reinterpret_cast<int4*>(const_cast<std::uint8_t*>(p) + J.fd) = v;
  }
  return v;
}
// v46a-1, Q6_K as stored: N words of the 2-byte aligned bytes at p (mis = p & 3, 0 or 2), from aligned 4-byte loads joined
// by a funnel shift; with M = 2 the loaded words go to the slot as well (whole words, the same bytes the source holds there)
template <int M, int N>
__device__ __forceinline__ void q6s_words(const DJob& J, bool cs, const std::uint8_t* p, unsigned mis, std::uint32_t* out) {
  const std::uint32_t* a = reinterpret_cast<const std::uint32_t*>(p - mis);
  std::uint32_t r[N + 1];
  if constexpr (M == 3) {
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = a[i];
    r[N] = mis ? a[N] : 0u;
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = cs ? ld_cs1(a + i) : __ldg(a + i);
    r[N] = mis ? (cs ? ld_cs1(a + N) : __ldg(a + N)) : 0u;  // only when the bytes straddle: never past the block
  }
  if constexpr (M == 2) {
    std::uint32_t* f = reinterpret_cast<std::uint32_t*>(const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(a)) + J.fd);
#pragma unroll
    for (int i = 0; i < N; ++i) f[i] = r[i];
    if (mis) f[N] = r[N];
  }
#pragma unroll
  for (int i = 0; i < N; ++i) out[i] = __funnelshift_r(r[i], r[i + 1], 8u * mis);
}


extern __shared__ float s_lut[];  // the replicated grid: 2,048 floats (8 KB), lane L reads s_lut[L + 32 * (tag * 16 + code)]
constexpr int kLutBytes = 2048 * 4;
constexpr int kLutRxBytes = 2048 * 4;  // v55b: the expert stage's one grid (c_rx_bits); v54 held all three (24 KB, 1 block per SM)
// v54: the 8-code words of one lane part (half `half` of block B of the row at R) from the plane layout:
// header, plane A, and planes B / C when J.pb asks for them (missing planes read as zero bits); returns the header
template <int M>
__device__ __forceinline__ int4 tg2p_words(const DJob& J, bool cs, const std::uint8_t* R, int B, int half, int nB, std::uint32_t* cw) {
  const int4 hd = ldwf<M>(J, cs, R + 16 * B, half == 0);
  const int4 a0 = ldwf<M>(J, cs, R + 16 * nB + 64 * B + 32 * half, true), a1 = ldwf<M>(J, cs, R + 16 * nB + 64 * B + 32 * half + 16, true);
  int4 b4 = make_int4(0, 0, 0, 0), c4 = make_int4(0, 0, 0, 0);
  if (J.pb >= 3) b4 = ldwf<M>(J, cs, R + 80 * nB + 32 * B + 16 * half, true);
  if (J.pb >= 4) c4 = ldwf<M>(J, cs, R + 112 * nB + 32 * B + 16 * half, true);
  const std::uint32_t aw[8] = {static_cast<std::uint32_t>(a0.x), static_cast<std::uint32_t>(a0.y), static_cast<std::uint32_t>(a0.z),
                               static_cast<std::uint32_t>(a0.w), static_cast<std::uint32_t>(a1.x), static_cast<std::uint32_t>(a1.y),
                               static_cast<std::uint32_t>(a1.z), static_cast<std::uint32_t>(a1.w)};
  const std::uint32_t bw[4] = {static_cast<std::uint32_t>(b4.x), static_cast<std::uint32_t>(b4.y), static_cast<std::uint32_t>(b4.z), static_cast<std::uint32_t>(b4.w)};
  const std::uint32_t cv[4] = {static_cast<std::uint32_t>(c4.x), static_cast<std::uint32_t>(c4.y), static_cast<std::uint32_t>(c4.z), static_cast<std::uint32_t>(c4.w)};
#pragma unroll
  for (int jj = 0; jj < 4; ++jj)
#pragma unroll
    for (int g = 0; g < 4; ++g)
      cw[jj * 4 + g] = tg2p_codes8((aw[2 * jj] >> (8 * g)) & 0xffu, (aw[2 * jj + 1] >> (8 * g)) & 0xffu, (bw[jj] >> (8 * g)) & 0xffu,
                                   (cv[jj] >> (8 * g)) & 0xffu);  // v55f: plane A as two 1-bit planes
  return hd;
}
// v54c: one lane part of a Q5_K plane row at R: the 16-byte header, and the half's plane words A-D as J.pb asks (missing
// low bits zero; the caller adds their midpoint through the input sums). v55: the words are returned as loaded and read by
// q5p_word straight into dp4a (v54c unpacked them to the stored bytes first: byte arrays, the suspected decode cost)
template <int M>
__device__ __forceinline__ int4 q5p_part(const DJob& J, bool cs, const std::uint8_t* R, int B, int half, int nB, std::uint32_t* aw,
                                         std::uint32_t* bw, std::uint32_t* cw, std::uint32_t* dw) {
  const int4 hd = ldwf<M>(J, cs, R + 16 * B, half == 0);
  const int4 a0 = ldwf<M>(J, cs, R + 16 * nB + 64 * B + 32 * half, true), a1 = ldwf<M>(J, cs, R + 16 * nB + 64 * B + 32 * half + 16, true);
  int4 b4 = make_int4(0, 0, 0, 0), c4 = make_int4(0, 0, 0, 0), d4 = make_int4(0, 0, 0, 0);
  if (J.pb >= 3) b4 = ldwf<M>(J, cs, R + 80 * nB + 32 * B + 16 * half, true);
  if (J.pb >= 4) c4 = ldwf<M>(J, cs, R + 112 * nB + 32 * B + 16 * half, true);
  if (J.pb >= 5) d4 = ldwf<M>(J, cs, R + 144 * nB + 32 * B + 16 * half, true);
  auto put = [](std::uint32_t* w, int4 v) {
    w[0] = static_cast<std::uint32_t>(v.x);
    w[1] = static_cast<std::uint32_t>(v.y);
    w[2] = static_cast<std::uint32_t>(v.z);
    w[3] = static_cast<std::uint32_t>(v.w);
  };
  put(aw, a0);
  put(aw + 4, a1);
  put(bw, b4);
  put(cw, c4);
  put(dw, d4);
  return hd;
}
// v54: the grid the codes are read through (the stored levels, or the 3-bit / 2-bit base) and the mask that clears the code
// bits not read
// v55b: one grid in shared memory: the dense kernels load the stored grid (lut_setup, they read Q4_TG2 at 4 bits), the expert
// stage the grid of c_rx_bits (lut_setup_rx), which is every Q4_TG2 job's bits there (the engine sets it from the slider)
__device__ __forceinline__ const float* tg2_lutb(int) { return s_lut; }
__device__ __forceinline__ std::uint32_t tg2_mask(int bits) { return bits >= 4 ? 0xffffffffu : bits == 3 ? 0xeeeeeeeeu : 0xccccccccu; }

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

// one lane's part (128 values = 4 sub-blocks of 32) of one row, for NT = 1-4 tokens on one load of the weights (acc[t]: the
// partial dot product of token t). M: 0 dense (__ldg inputs), 1 expert, 2 expert + slot fill, 3 the warp's staged rows.
// v55e (author 04:30, "take a full long look at the kernel and fix it"): one decode for one and several tokens. Before,
// lane_part (one token) and lane_part_n (v47, several) held every format twice; v47 already made each token's arithmetic in
// lane_part_n lane_part's operation for operation (tested bit for bit), so lane_part is now lane_part_n with one token. That
// also drops the 32 activation registers lane_part held through the whole call: each sub-block's 8 activation words are
// loaded where they are used. Q6_K (dense and as stored) moves to dp4a with the scalar loop's integers exactly; Q5_K's plane
// and stored paths are separate branches (before, both word sets were live across the shared sub-block loop).
// ONLY: -1 every format; kOnlyRx the expert stage's formats (rx_fmt: the bank, cache and file never hold Q7_ZT, IQ4_XS or
// the dense Q6_K layout; the plan refuses them); one format id: that format only (v55d's lean dense kernels).
constexpr int kOnlyRx = -2;
template <int ONLY>
__device__ __forceinline__ constexpr bool fm_on(int f) {
  return ONLY == -1 || ONLY == f ||
         (ONLY == kOnlyRx && (f == kQ4TG2 || f == kQ4TG2P || f == kQ5K || f == kQ5KP || f == kQ4K || f == kQ6KS || f == kQ8_0));
}
// the 8 activation words (32 int8) of sub-block jg
template <int M>
__device__ __forceinline__ void ld_x8(const std::int8_t* xq, int jg, int* xj) {
  const int4* x4 = reinterpret_cast<const int4*>(xq + jg * 32);
  const int4 u0 = ldx4<M>(x4), u1 = ldx4<M>(x4 + 1);
  xj[0] = u0.x;
  xj[1] = u0.y;
  xj[2] = u0.z;
  xj[3] = u0.w;
  xj[4] = u1.x;
  xj[5] = u1.y;
  xj[6] = u1.z;
  xj[7] = u1.w;
}
// Q4_K / Q5_K: sub-block j's 6-bit scale and min from the 12 scale bytes in header words y, z, w (bytes 4..15), by shifts
__device__ __forceinline__ void k4_scmn(const int4& hd, int j, int* sc, int* mn) {
  const std::uint32_t s0 = static_cast<std::uint32_t>(hd.y), s1 = static_cast<std::uint32_t>(hd.z), s2 = static_cast<std::uint32_t>(hd.w);
  auto qb = [&](int k) { return static_cast<int>(((k < 4 ? s0 : (k < 8 ? s1 : s2)) >> (8 * (k & 3))) & 0xffu); };
  if (j < 4) {
    *sc = qb(j) & 63;
    *mn = qb(j + 4) & 63;
  } else {
    *sc = (qb(j + 4) & 0xF) | ((qb(j - 4) >> 6) << 4);
    *mn = (qb(j + 4) >> 4) | ((qb(j) >> 6) << 4);
  }
}
// v55e: Q6_K sub-block p (32 values) on dp4a. q = low nibble | 2 high bits << 4 (0..63), 4 codes a word; per 16 values
// sum (q - 32) x = dp4a(q, x) - 32 dp4a(1, x): the same integers as the scalar loop (ql bytes p & 1 ? 32.. : 0.., nibble
// p >> 1; qh bits 2p), then sc[2p] s0 + sc[2p + 1] s1 with the int8 scales of bytes 2p, 2p + 1 (scw0: bytes 0-3, scw1: 4-7)
__device__ __forceinline__ int q6_sub(const std::uint32_t* qlw, const std::uint32_t* qhw, std::uint32_t scw0, std::uint32_t scw1, int p,
                                      const int* xj) {
  int a0 = 0, a1 = 0, x0 = 0, x1 = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    const std::uint32_t lo = (qlw[8 * (p & 1) + w] >> (4 * (p >> 1))) & 0x0f0f0f0fu, hi = ((qhw[w] >> (2 * p)) & 0x03030303u) << 4;
    if (w < 4) {
      a0 = __dp4a(static_cast<int>(lo | hi), xj[w], a0);
      x0 = __dp4a(0x01010101, xj[w], x0);
    } else {
      a1 = __dp4a(static_cast<int>(lo | hi), xj[w], a1);
      x1 = __dp4a(0x01010101, xj[w], x1);
    }
  }
  const std::uint32_t sw = (p < 2 ? scw0 : scw1) >> (16 * (p & 1));
  const int c0 = static_cast<int>(static_cast<std::int8_t>(sw & 0xffu)), c1 = static_cast<int>(static_cast<std::int8_t>((sw >> 8) & 0xffu));
  return c0 * (a0 - 32 * x0) + c1 * (a1 - 32 * x1);
}
template <int M, int NT, int ONLY = -1>
__device__ __forceinline__ void lane_part_n(const DJob& J, int r, int part, const std::int8_t* const* xq, const float* const* xs, const int* const* xsum,
                                            bool cs, float* acc, const std::uint8_t* wb = nullptr, long long rb0 = 0) {
  const int B = part >> 1, half = part & 1, nB = J.K / 256;
  const long long rb = static_cast<long long>(r) * nB + B;
  const std::uint8_t* const W = M == 3 ? wb : J.w;
  const long long rbw = M == 3 ? rb - rb0 : rb;
  const int jg0 = B * 8 + half * 4;  // first 32-block of the part in the row
#pragma unroll
  for (int t = 0; t < NT; ++t) acc[t] = 0.f;
  if (fm_on<ONLY>(kQ7ZT) && J.fmt == kQ7ZT) {
    std::uint32_t wd[29];
    const std::uint8_t* bp = J.w + rb * 224 + half * 112;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      const int4 v = ldw(cs, bp + 16 * i);
      wd[4 * i] = static_cast<std::uint32_t>(v.x);
      wd[4 * i + 1] = static_cast<std::uint32_t>(v.y);
      wd[4 * i + 2] = static_cast<std::uint32_t>(v.z);
      wd[4 * i + 3] = static_cast<std::uint32_t>(v.w);
    }
    wd[28] = 0u;
    const std::uint64_t h = J.hdr[rb];
    const unsigned m = J.mask[rb];
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h & 0xffffu)));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg0 + jj, xj);
        const int j = half * 4 + jj, jg = jg0 + jj;
        int pb = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          std::uint32_t packed = 0;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int i = 32 * jj + 4 * g + b, bit = 7 * i;
            packed |= (__funnelshift_r(wd[bit >> 5], wd[(bit >> 5) + 1], static_cast<unsigned>(bit & 31)) & 127u) << (8 * b);
          }
          pb = __dp4a(static_cast<int>(packed), xj[g], pb);
        }
        int hx = 0;
        if ((m >> j) & 1u) {
          const std::uint32_t H = J.hb[rb * 2 + __popc(m & ((1u << j) - 1u))];
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            const std::uint32_t nib = (H >> (4 * g)) & 0xfu;
            hx = __dp4a(static_cast<int>((nib * 0x00204081u) & 0x01010101u), xj[g], hx);
          }
        }
        const int P = 2 * pb - 127 * ldxi<M>(xsum[t] + jg) + hx;  // sum (2b + h - 127) x
        const float s = (d * static_cast<float>((h >> (16 + 6 * j)) & 63u)) / 63.f;
        acc[t] += s * ldxs<M>(xs[t] + jg) * static_cast<float>(P);
      }
    }
  } else if ((fm_on<ONLY>(kQ4TG2) || fm_on<ONLY>(kQ4TG2P)) && (J.fmt == kQ4TG2 || J.fmt == kQ4TG2P)) {  // v54: stored, or in bit planes (the same codes when all 4 bits are read)
    int4 hd;
    std::uint32_t cw[16];
    if (fm_on<ONLY>(kQ4TG2P) && J.fmt == kQ4TG2P) {  // v54b: the row at stride J.rs (or the warp's staged rows, packed at the prefix)
      const long long r0s = M == 3 ? rb0 / nB : 0;
      const long long stride = M == 3 ? static_cast<long long>(tg2p_prefix(static_cast<std::uint64_t>(nB), J.pb)) : J.rs;
      hd = tg2p_words<M>(J, cs, W + (r - r0s) * stride, B, half, nB, cw);
    } else {
      const std::uint8_t* bp = W + rbw * 144;
      hd = ldwf<M>(J, cs, bp, half == 0);
      int4 c[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = ldwf<M>(J, cs, bp + 16 + half * 64 + 16 * i, true);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        cw[4 * i] = static_cast<std::uint32_t>(c[i].x);
        cw[4 * i + 1] = static_cast<std::uint32_t>(c[i].y);
        cw[4 * i + 2] = static_cast<std::uint32_t>(c[i].z);
        cw[4 * i + 3] = static_cast<std::uint32_t>(c[i].w);
      }
    }
    const int tbits = J.fmt == kQ4TG2P ? J.pb : 4;  // v54b: the slider's bits
    const std::uint32_t mk = tg2_mask(tbits);
    const float* lut0 = tg2_lutb(tbits);
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x), h1 = static_cast<std::uint32_t>(hd.y);
    const std::uint64_t lv = (static_cast<std::uint64_t>(h0 >> 16)) | (static_cast<std::uint64_t>(h1) << 16);  // 6 level bytes at 2..7
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const std::uint64_t tags =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(hd.z)) | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(hd.w)) << 32);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg0 + jj, xj);
        const int j = half * 4 + jj, jg = jg0 + jj;
        const unsigned tb = static_cast<unsigned>((tags >> (8 * j)) & 0xffu);
        float sum = 0.f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const unsigned tg = (tb >> (2 * g)) & 3u;
          const std::uint32_t word = cw[jj * 4 + g] & mk;  // 8 codes = values 32 jj + 8 g .. + 7 of the part
          const float* lr = lut0 + (threadIdx.x & 31) + 32 * 16 * tg;
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const float wv = lr[32 * ((word >> (4 * k)) & 15u)];
            sum += wv * i8f(xj, 8 * g + k);
          }
        }
        const float s = (d * static_cast<float>((lv >> (6 * j)) & 63u)) / 63.f;
        acc[t] += s * ldxs<M>(xs[t] + jg) * sum;
      }
    }
  } else if (fm_on<ONLY>(kQ5KP) && J.fmt == kQ5KP) {  // v54c / v55: Q5_K in bit planes, plane words -> 4 codes a word -> dp4a
    std::uint32_t pa[8], pb[4], pc[4], pd[4];
    const long long r0s = M == 3 ? rb0 / nB : 0;
    const long long stride = M == 3 ? static_cast<long long>(plane_prefix(g_q5_kp, 176ull * static_cast<unsigned long long>(nB), J.pb)) : J.rs;
    const int4 hd = q5p_part<M>(J, cs, W + (r - r0s) * stride, B, half, nB, pa, pb, pc, pd);
    const int kmiss = 5 - J.pb;  // code bits not read (their midpoint is added through the input sums)
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 >> 16)));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      int sc, mn;
      k4_scmn(hd, j, &sc, &mn);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int sq = 0;
        if (J.xb) {  // v55f, the 1-bit route (author's 1-bit decoder): each plane AND + popcount against the input's bit planes
          const int4* u4 = reinterpret_cast<const int4*>(J.xb + ((xq[t] - J.xq) >> 2) + 8 * jg);
          const int4 v0 = ldx4<M>(u4), v1 = ldx4<M>(u4 + 1);
          const std::uint32_t u8[8] = {static_cast<std::uint32_t>(v0.x), static_cast<std::uint32_t>(v0.y), static_cast<std::uint32_t>(v0.z),
                                       static_cast<std::uint32_t>(v0.w), static_cast<std::uint32_t>(v1.x), static_cast<std::uint32_t>(v1.y),
                                       static_cast<std::uint32_t>(v1.z), static_cast<std::uint32_t>(v1.w)};
          sq = (plane_dot32(pa[2 * jj], u8) << 4) + (plane_dot32(pa[2 * jj + 1], u8) << 3);
          if (J.pb >= 3) sq += plane_dot32(pb[jj], u8) << 2;
          if (J.pb >= 4) sq += plane_dot32(pc[jj], u8) << 1;
          if (J.pb >= 5) sq += plane_dot32(pd[jj], u8);
        } else {  // the dp4a route (inputs without planes: the separate expert kernels)
          int xj[8];
          ld_x8<M>(xq[t], jg, xj);
#pragma unroll
          for (int g = 0; g < 8; ++g) sq = __dp4a(static_cast<int>(q5p_word(pa, pb, pc, pd, jj, g)), xj[g], sq);
        }
        const int xsi = ldxi<M>(xsum[t] + jg);
        const float sqf = kmiss > 0 ? 0.5f * static_cast<float>(2 * sq + ((1 << kmiss) - 1) * xsi) : static_cast<float>(sq);
        acc[t] += ldxs<M>(xs[t] + jg) * (d * static_cast<float>(sc) * sqf - dmin * static_cast<float>(mn) * static_cast<float>(xsi));
      }
    }
  } else if (fm_on<ONLY>(kQ5K) && J.fmt == kQ5K) {  // v55d: stored Q5_K on dp4a: 4 codes a word from the low nibbles and high bit j
    const std::uint8_t* bp = W + rbw * 176;
    const int4 hd = ldwf<M>(J, cs, bp, half == 0);
    const int4 qa = ldwf<M>(J, cs, bp + 16, half == 0), qb4 = ldwf<M>(J, cs, bp + 32, half == 0);
    int4 ql[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) ql[i] = ldwf<M>(J, cs, bp + 48 + half * 64 + 16 * i, true);
    const std::uint32_t qhw[8] = {static_cast<std::uint32_t>(qa.x),  static_cast<std::uint32_t>(qa.y),  static_cast<std::uint32_t>(qa.z),
                                  static_cast<std::uint32_t>(qa.w),  static_cast<std::uint32_t>(qb4.x), static_cast<std::uint32_t>(qb4.y),
                                  static_cast<std::uint32_t>(qb4.z), static_cast<std::uint32_t>(qb4.w)};
    const std::uint32_t qlw[16] = {static_cast<std::uint32_t>(ql[0].x), static_cast<std::uint32_t>(ql[0].y), static_cast<std::uint32_t>(ql[0].z),
                                   static_cast<std::uint32_t>(ql[0].w), static_cast<std::uint32_t>(ql[1].x), static_cast<std::uint32_t>(ql[1].y),
                                   static_cast<std::uint32_t>(ql[1].z), static_cast<std::uint32_t>(ql[1].w), static_cast<std::uint32_t>(ql[2].x),
                                   static_cast<std::uint32_t>(ql[2].y), static_cast<std::uint32_t>(ql[2].z), static_cast<std::uint32_t>(ql[2].w),
                                   static_cast<std::uint32_t>(ql[3].x), static_cast<std::uint32_t>(ql[3].y), static_cast<std::uint32_t>(ql[3].z),
                                   static_cast<std::uint32_t>(ql[3].w)};
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 >> 16)));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      int sc, mn;
      k4_scmn(hd, j, &sc, &mn);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg, xj);
        int sq = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const std::uint32_t lo = (qlw[8 * (jj >> 1) + g] >> (4 * (j & 1))) & 0x0f0f0f0fu, hi = ((qhw[g] >> j) & 0x01010101u) << 4;
          sq = __dp4a(static_cast<int>(lo | hi), xj[g], sq);
        }
        const int xsi = ldxi<M>(xsum[t] + jg);
        acc[t] += ldxs<M>(xs[t] + jg) * (d * static_cast<float>(sc) * static_cast<float>(sq) - dmin * static_cast<float>(mn) * static_cast<float>(xsi));
      }
    }
  } else if (fm_on<ONLY>(kQ8_0) && J.fmt == kQ8_0) {  // repacked per 256: qs[256] (int8) | d[8] (fp16); the half's 128 values and 4 d's
    const std::uint8_t* bp = W + rbw * 272;
    const uint2 dd = *reinterpret_cast<const uint2*>(bp + 256 + half * 8);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int jg = jg0 + jj;
      const int4 qa = ldwf<M>(J, cs, bp + half * 128 + 32 * jj, false), qb = ldwf<M>(J, cs, bp + half * 128 + 32 * jj + 16, false);  // once for all tokens
      const std::uint32_t dw = (jj < 2 ? dd.x : dd.y) >> (16 * (jj & 1));
      const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dw & 0xffffu)));
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg, xj);
        int P = 0;
        P = __dp4a(qa.x, xj[0], P);
        P = __dp4a(qa.y, xj[1], P);
        P = __dp4a(qa.z, xj[2], P);
        P = __dp4a(qa.w, xj[3], P);
        P = __dp4a(qb.x, xj[4], P);
        P = __dp4a(qb.y, xj[5], P);
        P = __dp4a(qb.z, xj[6], P);
        P = __dp4a(qb.w, xj[7], P);
        acc[t] += d * ldxs<M>(xs[t] + jg) * static_cast<float>(P);
      }
    }
  } else if (fm_on<ONLY>(kQ4K) && J.fmt == kQ4K) {  // as stored (144 bytes): d | dmin | scales[12] | qs[128]; Q5_K without qh
    const std::uint8_t* bp = W + rbw * 144;
    const int4 hd = ldwf<M>(J, cs, bp, half == 0);
    int4 ql[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) ql[i] = ldwf<M>(J, cs, bp + 16 + half * 64 + 16 * i, true);
    const std::uint32_t qw[16] = {static_cast<std::uint32_t>(ql[0].x), static_cast<std::uint32_t>(ql[0].y), static_cast<std::uint32_t>(ql[0].z),
                                  static_cast<std::uint32_t>(ql[0].w), static_cast<std::uint32_t>(ql[1].x), static_cast<std::uint32_t>(ql[1].y),
                                  static_cast<std::uint32_t>(ql[1].z), static_cast<std::uint32_t>(ql[1].w), static_cast<std::uint32_t>(ql[2].x),
                                  static_cast<std::uint32_t>(ql[2].y), static_cast<std::uint32_t>(ql[2].z), static_cast<std::uint32_t>(ql[2].w),
                                  static_cast<std::uint32_t>(ql[3].x), static_cast<std::uint32_t>(ql[3].y), static_cast<std::uint32_t>(ql[3].z),
                                  static_cast<std::uint32_t>(ql[3].w)};
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 >> 16)));
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = half * 4 + jj, jg = jg0 + jj;
      int sc, mn;
      k4_scmn(hd, j, &sc, &mn);
      // sub-block j: 32 nibbles of bytes 32 (jj / 2) .. +31 of the half, low nibbles for even j, high for odd
      const int sh = 4 * (j & 1);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg, xj);
        int sq = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) sq = __dp4a(static_cast<int>((qw[8 * (jj >> 1) + g] >> sh) & 0x0F0F0F0Fu), xj[g], sq);
        acc[t] += ldxs<M>(xs[t] + jg) *
                  (d * static_cast<float>(sc) * static_cast<float>(sq) - dmin * static_cast<float>(mn) * static_cast<float>(ldxi<M>(xsum[t] + jg)));
      }
    }
  } else if (fm_on<ONLY>(kIQ4XS) && J.fmt == kIQ4XS) {  // repacked per 256 (144 bytes): d | scales_h | scales_l[4] | 8 zero bytes | qs[128]
    const std::uint8_t* bp = W + rbw * 144;
    const int4 hd = ldw(cs, bp);
    const std::uint32_t h0 = static_cast<std::uint32_t>(hd.x), sl = static_cast<std::uint32_t>(hd.y);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0xffffu)));
    const std::uint32_t sh = h0 >> 16;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int ib = half * 4 + jj, jg = jg0 + jj;
      const int ls = static_cast<int>(((sl >> (4 * ib)) & 0xfu) | (((sh >> (2 * ib)) & 3u) << 4));
      const int4 qv = ldw(cs, bp + 16 + 16 * ib);  // 16 bytes: value l (low nibble of byte l) and l + 16 (high nibble); once for all tokens
      const std::uint32_t qw[4] = {static_cast<std::uint32_t>(qv.x), static_cast<std::uint32_t>(qv.y), static_cast<std::uint32_t>(qv.z),
                                   static_cast<std::uint32_t>(qv.w)};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg, xj);
        int P = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          P = __dp4a(iq4_levels(qw[w] & 0x0F0F0F0Fu), xj[w], P);
          P = __dp4a(iq4_levels((qw[w] >> 4) & 0x0F0F0F0Fu), xj[4 + w], P);
        }
        acc[t] += d * static_cast<float>(ls - 32) * ldxs<M>(xs[t] + jg) * static_cast<float>(P);
      }
    }
  } else if (M != 0 && fm_on<ONLY>(kQ6KS) && J.fmt == kQ6KS) {  // v46a-1: as stored: ql[128] | qh[64] | scales[16] | d (210 bytes, 2-byte aligned)
    const std::uint8_t* bp = W + rbw * 210;
    const unsigned mis = static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(bp) & 3u);
    std::uint32_t qlw[16], qhw[8], scw[2];
    q6s_words<M, 16>(J, cs, bp + half * 64, mis, qlw);
    q6s_words<M, 8>(J, cs, bp + 128 + half * 32, mis, qhw);
    q6s_words<M, 2>(J, cs, bp + 192 + half * 8, mis, scw);
    const unsigned short dh = *reinterpret_cast<const unsigned short*>(bp + 208);
    if constexpr (M == 2) {
      if (half == 0) *reinterpret_cast<unsigned short*>(const_cast<std::uint8_t*>(bp) + 208 + J.fd) = dh;
    }
    const float d = __half2float(__ushort_as_half(dh));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int jg = jg0 + p;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg, xj);
        acc[t] += d * ldxs<M>(xs[t] + jg) * static_cast<float>(q6_sub(qlw, qhw, scw[0], scw[1], p, xj));
      }
    }
  } else if (fm_on<ONLY>(kQ6K)) {  // kQ6K (the dense layout: d in J.d6)
    const std::uint8_t* bp = J.w + rb * 208;
    int4 l4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) l4[i] = ldw(cs, bp + half * 64 + 16 * i);
    const int4 h4a = ldw(cs, bp + 128 + half * 32), h4b = ldw(cs, bp + 128 + half * 32 + 16);
    const uint2 s2 = *reinterpret_cast<const uint2*>(bp + 192 + half * 8);
    const std::uint32_t qlw[16] = {static_cast<std::uint32_t>(l4[0].x), static_cast<std::uint32_t>(l4[0].y), static_cast<std::uint32_t>(l4[0].z),
                                   static_cast<std::uint32_t>(l4[0].w), static_cast<std::uint32_t>(l4[1].x), static_cast<std::uint32_t>(l4[1].y),
                                   static_cast<std::uint32_t>(l4[1].z), static_cast<std::uint32_t>(l4[1].w), static_cast<std::uint32_t>(l4[2].x),
                                   static_cast<std::uint32_t>(l4[2].y), static_cast<std::uint32_t>(l4[2].z), static_cast<std::uint32_t>(l4[2].w),
                                   static_cast<std::uint32_t>(l4[3].x), static_cast<std::uint32_t>(l4[3].y), static_cast<std::uint32_t>(l4[3].z),
                                   static_cast<std::uint32_t>(l4[3].w)};
    const std::uint32_t qhw[8] = {static_cast<std::uint32_t>(h4a.x), static_cast<std::uint32_t>(h4a.y), static_cast<std::uint32_t>(h4a.z),
                                  static_cast<std::uint32_t>(h4a.w), static_cast<std::uint32_t>(h4b.x), static_cast<std::uint32_t>(h4b.y),
                                  static_cast<std::uint32_t>(h4b.z), static_cast<std::uint32_t>(h4b.w)};
    const float d = __half2float(__ushort_as_half(J.d6[rb]));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int jg = jg0 + p;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int xj[8];
        ld_x8<M>(xq[t], jg, xj);
        acc[t] += d * ldxs<M>(xs[t] + jg) * static_cast<float>(q6_sub(qlw, qhw, s2.x, s2.y, p, xj));
      }
    }
  }
}
// one token: lane_part_n with NT = 1 (returns the partial dot product)
template <int M, int ONLY = -1>
__device__ __forceinline__ float lane_part(const DJob& J, int r, int part, const std::int8_t* xq, const float* xs, const int* xsum, bool cs,
                                           const std::uint8_t* wb = nullptr, long long rb0 = 0) {
  const std::int8_t* q1[1] = {xq};
  const float* s1[1] = {xs};
  const int* m1[1] = {xsum};
  float a[1];
  lane_part_n<M, 1, ONLY>(J, r, part, q1, s1, m1, cs, a, wb, rb0);
  return a[0];
}

// v47: NT = 2, 3 or 4 tokens on one load of the weights (cs: the weights are read once, so they stream past the caches)
template <int M, int ONLY = -1>
__device__ __forceinline__ void lane_part_d(const DJob& J, int r, int part, const std::int8_t* const* xq, const float* const* xs, const int* const* xsum, int nt,
                                            float* acc, const std::uint8_t* wb = nullptr, long long rb0 = 0) {
  if (nt == 2) lane_part_n<M, 2, ONLY>(J, r, part, xq, xs, xsum, true, acc, wb, rb0);
  else if (nt == 3) lane_part_n<M, 3, ONLY>(J, r, part, xq, xs, xsum, true, acc, wb, rb0);
  else lane_part_n<M, 4, ONLY>(J, r, part, xq, xs, xsum, true, acc, wb, rb0);
}
// v47: the warp's NT accumulators reduced over the row's lanes and stored (row r, token outputs at y + ypk / tk strides)
__device__ __forceinline__ void store_rows(const DJob& J, int r, int part, const float* acc, int nt, const long long* yoff) {
  for (int tk = 0; tk < nt; ++tk) {
    float a = acc[tk];
    for (int o = J.lpr / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (r < J.n && part == 0) J.y[yoff[tk] + r] = a;
  }
}

// route 2: a warp's t only grows, so the job is found by walking forward from the last one
__device__ __forceinline__ int find_job(const DJob* __restrict__ jobs, int njobs, long long t, int* hint) {
  int lo = *hint;
  while (lo + 1 < njobs && jobs[lo + 1].task0 <= t) ++lo;
  *hint = lo;
  return lo;
}

template <bool kMT, int ONLY = -1>
__device__ __forceinline__ void run_task(const DJob* __restrict__ jobs, int njobs, long long t, int* hint) {
  const DJob& J = jobs[find_job(jobs, njobs, t, hint)];
  const int lane = static_cast<int>(threadIdx.x & 31), rpw = 32 / J.lpr;
  const int r = static_cast<int>((t - J.task0) * rpw) + lane / J.lpr, part = lane % J.lpr;
  if constexpr (kMT) {  // v47: several tokens on one load of the weights (the plan's jobs have 2-4 tokens)
    if (J.ntok > 1) {
      const std::int8_t* xq[4];
      const float* xs[4];
      const int* xsum[4];
      long long yo[4];
      for (int tk = 0; tk < 4; ++tk) {
        const int u = tk < J.ntok ? tk : 0;
        xq[tk] = J.xq + u * J.xqs;
        xs[tk] = J.xs + u * J.xss;
        xsum[tk] = J.xsum + u * J.xss;
        yo[tk] = u * J.ys;
      }
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < J.n) lane_part_d<0, ONLY>(J, r, part, xq, xs, xsum, J.ntok, acc);
      store_rows(J, r, part, acc, J.ntok, yo);
      return;
    }
  }
  // v26: each token's arithmetic is the one-token path's (lane_part on that token's inputs), so a row of a several-token
  // job equals the same token alone bit for bit
  for (int tk = 0; tk < J.ntok; ++tk) {
    float acc = r < J.n ? lane_part<0, ONLY>(J, r, part, J.xq + tk * J.xqs, J.xs + tk * J.xss, J.xsum + tk * J.xss, J.ntok == 1) : 0.f;
    for (int o = J.lpr / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (r < J.n && part == 0) J.y[tk * J.ys + r] = acc;
  }
}

__device__ __forceinline__ void lut_setup() {
  for (int i = static_cast<int>(threadIdx.x); i < 2048; i += static_cast<int>(blockDim.x)) s_lut[i] = c_grid[(i >> 9) & 3][(i >> 5) & 15];
  __syncthreads();
}
// v55b: the expert stage's one grid (kLutRxBytes): stored levels, or the 3-bit / 2-bit base, replicated as lut_setup's
__device__ __forceinline__ void lut_setup_rx() {
  const int b = c_rx_bits;
  for (int i = static_cast<int>(threadIdx.x); i < 2048; i += static_cast<int>(blockDim.x)) {
    const int tg = (i >> 9) & 3, c = (i >> 5) & 15;
    s_lut[i] = b == 3 ? c_grid3[tg][c] : b == 2 ? c_grid2[tg][c] : c_grid[tg][c];
  }
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
// v55d: no register cap (v55c's 2-block cap spilled 152-424 B and ran 0.55-0.92 ms slower, MEASURED); instead the dense
// one-token plans whose jobs are all one type run a kernel compiled for that type only (ONLY), which needs far fewer
// registers (the 2026-11-02 texture probe's lean Q8_0 GEMV: 39-64 registers, 6 blocks per SM, 503 GB/s, MEASURED)
template <bool kMT, int ONLY = -1>
__global__ void __launch_bounds__(256) k_stream(const DJob* __restrict__ jobs, int njobs, long long t_end) {
  if constexpr (ONLY < 0 || ONLY == kQ4TG2 || ONLY == kQ4TG2P) lut_setup();  // v55e: a lean kernel of another format reads no grid
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  int hint = 0;
  for (long long t = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; t < t_end; t += warps)
    run_task<kMT, ONLY>(jobs, njobs, t, &hint);
}

// the token (r3_grouped<2>): stages with a counter barrier per warp between them; inside a stage a job may wait for a
// group's tasks (acquire) and its own tasks may signal a group (release). done[] and gdone[] are never reset: run number
// epoch (1-based) waits for epoch x the counts.
template <bool kMT>
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
      run_task<kMT>(jobs, njobs, t, &hint);
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

// ------------------------------------------------------------------------------------------------ v46a-1 expert stage
// the stage's matrix formats by the GPU path's type ids (gpu_decode.h): the bank, cache and file hold experts as stored;
// the shared expert is R3's repacked Q8_0 (g_q8_0r)
__host__ __device__ inline int rx_fmt(std::uint32_t t) {
  switch (t) {
    case g_q4_tg2: return kQ4TG2;
    case g_q4_tg2p: return kQ4TG2P;
    case g_q5_kp: return kQ5KP;
    case g_q5_k: return kQ5K;
    case g_q4_k: return kQ4K;
    case g_q6_k: return kQ6KS;
    case g_q8_0r: return kQ8_0;
    default: return -1;
  }
}
// the plan buffer: header | jobs (4 per expert group: gate, up, silu x up, down) | signal and wait group per job | tasks
// per group | the group counters (zeroed by the planner, so every launch counts from 0: graph replays need no reset)
struct RxHdr {
  int njobs, ngroups;
  long long ntasks;
  int err, pad[3];
};
struct RxLayout {
  std::uint64_t jobs, sig, dep, gtasks, gdone, total;
};
__host__ __device__ inline std::uint64_t rx_al(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }
__host__ __device__ inline RxLayout rx_layout(int maxp) {
  RxLayout L;
  const std::uint64_t nj = 4ull * static_cast<std::uint64_t>(maxp), ng = 2ull * static_cast<std::uint64_t>(maxp);
  L.jobs = 64;
  L.sig = rx_al(L.jobs + sizeof(DJob) * nj, 16);
  L.dep = rx_al(L.sig + 4 * nj, 16);
  L.gtasks = rx_al(L.dep + 4 * nj, 16);
  L.gdone = rx_al(L.gtasks + 8 * ng, 128);
  L.total = rx_al(L.gdone + 8 * ng, 128);
  return L;
}
// one thread per job: the groups of moe_reader3's grouping kernel (grp: count, then per group expert, picks, pick list;
// gsrc / gfill: where each group's rows are read from and, on a cache miss, the slot they are copied into). Every expert
// has the shape (F x H gate and up, H x F down), so each job's first task follows from its index alone.
// v52: the planning body (the whole block; any block size) for k_rx_plan and the fused MoE half
__device__ __forceinline__ void rx_plan_body(const int* __restrict__ grp, const ExpertPtr* __restrict__ gsrc, const ExpertPtr* __restrict__ gfill,
                                             int tokens, int nu, int H, int F, const std::int8_t* xq, const float* xs, const int* xsum, float* gu,
                                             std::int8_t* hq, float* hs, int* hsum, float* yd, std::uint8_t* plan, int maxp, int load_mode,
                                             const std::uint32_t* xb = nullptr, std::uint32_t* hb = nullptr) {
  const RxLayout Lo = rx_layout(maxp);
  auto* hd = reinterpret_cast<RxHdr*>(plan);
  auto* jobs = reinterpret_cast<DJob*>(plan + Lo.jobs);
  auto* sig = reinterpret_cast<int*>(plan + Lo.sig);
  auto* dep = reinterpret_cast<int*>(plan + Lo.dep);
  auto* gt = reinterpret_cast<long long*>(plan + Lo.gtasks);
  auto* gd = reinterpret_cast<unsigned long long*>(plan + Lo.gdone);
  const int U = grp[0], routed = tokens * nu;
  const int lpr_gu = H / 128, lpr_dn = F / 128;
  const long long nt_gu = (F + 32 / lpr_gu - 1) / (32 / lpr_gu), nt_dn = (H + 32 / lpr_dn - 1) / (32 / lpr_dn);
  const int nj = 4 * U;
  __shared__ int s_err;
  if (threadIdx.x == 0) s_err = U > maxp ? 1 : 0;
  __syncthreads();
  if (s_err == 0)
    for (int i = static_cast<int>(threadIdx.x); i < nj; i += static_cast<int>(blockDim.x)) {
      // v55f, pipelined (author: no warp idles while work exists): gate / up of expert s, then silu x up and down of expert
      // s - 1, so a down starts one expert behind its gate / up instead of after every expert's gate / up (v55e3: 20 % of
      // the warps' time waited for a group). Order: GU0 | GU1 EP0 DN0 | GU2 EP1 DN1 | ... | EP(U-1) DN(U-1). A job waits only
      // for jobs before it, so the lowest unfinished task can always run. Same jobs, same outputs; only the order moves.
      const long long t4 = 2 * nt_gu + 1 + nt_dn;  // one expert's tasks
      int kind, u;
      long long task0;
      if (i < 2) {
        kind = i;
        u = 0;
        task0 = i * nt_gu;
      } else if (i < 2 + 4 * (U - 1)) {
        const int sx = 1 + (i - 2) / 4, r = (i - 2) % 4;
        const long long base = 2 * nt_gu + static_cast<long long>(sx - 1) * t4;
        kind = r < 2 ? r : (r == 2 ? 3 : 2);  // 0 gate, 1 up, 2 down, 3 silu x up
        u = r < 2 ? sx : sx - 1;
        task0 = r == 0 ? base : r == 1 ? base + nt_gu : r == 2 ? base + 2 * nt_gu : base + 2 * nt_gu + 1;
      } else {
        const int r = i - (2 + 4 * (U - 1));
        const long long base = 2 * nt_gu + static_cast<long long>(U - 1) * t4;
        kind = r == 0 ? 3 : 2;
        u = U - 1;
        task0 = r == 0 ? base : base + 1;
      }
      const int* G = grp + 1 + 6 * u;
      const int cnt = G[1];
      DJob J{};
      J.task0 = task0;
      J.ntok = cnt;
      // v55e: a fixed trip count, so tokx / ypk are indexed by constants and J stays in registers (with the runtime count the
      // whole DJob went to local memory: k_rx_plan's and k_moe_half's 240 B stack frame, v55d ptxas listing). cnt <= 4: an
      // expert is picked at most once per token, and the shared expert's count is the tokens (1-4)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        if (c < cnt) {
          const int pk = G[2 + c];
          J.ypk[c] = pk;
          J.tokx[c] = kind < 2 ? (pk < routed ? pk / nu : pk - routed) : pk;
        }
      }
      if (kind == 3) {
        J.fmt = kEpi;
        J.K = F;
        J.lpr = 32;
        J.ntask = 1;
        J.gu = gu;
        J.hq = hq;
        J.hs = hs;
        J.hsum = hsum;
        J.hbo = hb;  // v55f
        sig[i] = U + u;
        dep[i] = u;
      } else {
        const ExpertPtr& X = gsrc[u];
        const ExpertPtr& FL = gfill[u];
        J.fmt = rx_fmt(X.type[kind]);
        {  // v54b / v54c: the slider's bits, at most the matrix's own (4 for Q4_TG2, 5 for Q5_K)
          const int top = J.fmt == kQ5KP ? 5 : 4;
          J.pb = X.pbits >= 2 && static_cast<int>(X.pbits) < top ? static_cast<int>(X.pbits) : top;
        }
        J.rs = static_cast<long long>(X.rb[kind]);
        J.fw = J.fmt == kQ4TG2P || J.fmt == kQ5KP ? const_cast<std::uint8_t*>(FL.w[kind]) : nullptr;
        J.frs = static_cast<long long>(FL.rb[kind]);
        if (J.fmt < 0) s_err = 2;  // every writer stores 2; read after the barrier
        J.n = kind < 2 ? F : H;
        J.K = kind < 2 ? H : F;
        J.lpr = J.K / 128;
        J.ntask = kind < 2 ? nt_gu : nt_dn;
        J.w = X.w[kind];
        J.fd = FL.w[kind] ? static_cast<long long>(FL.w[kind] - X.w[kind]) : 0;
        J.stg = load_mode == 2 || (load_mode == 1 && !X.vram);  // v46a-2: all experts, or those read from RAM
        if (J.fmt == kQ4TG2P || J.fmt == kQ5KP) J.fd = 0;  // v54b: a plane matrix's miss fills its slot through fw / frs (packed rows), staged
        if (kind < 2) {
          J.xq = xq;
          J.xs = xs;
          J.xsum = xsum;
          J.xb = xb;  // v55f
          J.xqs = H;
          J.xss = H / 32;
          J.y = gu + static_cast<long long>(kind) * F;
          J.ys = 2LL * F;
          sig[i] = u;
          dep[i] = -1;
        } else {
          J.xq = hq;
          J.xs = hs;
          J.xsum = hsum;
          J.xb = hb;  // v55f
          J.xqs = F;
          J.xss = F / 32;
          J.y = yd;
          J.ys = H;
          sig[i] = -1;
          dep[i] = U + u;
        }
      }
      jobs[i] = J;
    }
  for (int g = static_cast<int>(threadIdx.x); g < 2 * U && g < 2 * maxp; g += static_cast<int>(blockDim.x)) {
    gt[g] = g < U ? 2 * nt_gu : 1;
    gd[g] = 0ull;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    hd->err = s_err;
    hd->njobs = s_err ? 0 : nj;
    hd->ngroups = 2 * U;
    hd->ntasks = s_err ? 0 : 2LL * U * nt_gu + U + static_cast<long long>(U) * nt_dn;
  }
}
__global__ void __launch_bounds__(128) k_rx_plan(const int* __restrict__ grp, const ExpertPtr* __restrict__ gsrc, const ExpertPtr* __restrict__ gfill,
                                                 int tokens, int nu, int H, int F, const std::int8_t* xq, const float* xs, const int* xsum, float* gu,
                                                 std::int8_t* hq, float* hs, int* hsum, float* yd, std::uint8_t* plan, int maxp, int load_mode) {
  rx_plan_body(grp, gsrc, gfill, tokens, nu, H, F, xq, xs, xsum, gu, hq, hs, hsum, yd, plan, maxp, load_mode);
}
// silu(g) x u of each pick of the group, rounded to int8 per 32: the reader's k_rd_silu_round and act_q8x32, operation for
// operation (the down projection's inputs are the same numbers the reader used, as int8 and scale)
__device__ __forceinline__ void rx_epi(const DJob& J) {
  const int lane = static_cast<int>(threadIdx.x & 31), F = J.K, nb = F / 32;
  for (int c = 0; c < J.ntok; ++c) {
    const long long pk = J.ypk[c];
    const float* g = J.gu + pk * 2 * F;
    const float* u = g + F;
    for (int b = lane; b < nb; b += 32) {
      float v[32];
      float a = 0.0f;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        const float x = ld_cgf(g + 32 * b + i);
        v[i] = mul(quo(x, add(1.0f, expf(-x))), ld_cgf(u + 32 * b + i));
        a = fmaxf(a, fabsf(v[i]));
      }
      const float sc = a > 0.0f ? quo(a, 127.0f) : 1.0f;
      const float inv = quo(1.0f, sc);
      int sm = 0;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        const float qv = fminf(127.0f, fmaxf(-127.0f, rintf(mul(v[i], inv))));
        J.hq[pk * F + 32 * b + i] = static_cast<std::int8_t>(qv);
        sm += static_cast<int>(qv);
      }
      J.hs[pk * nb + b] = sc;
      J.hsum[pk * nb + b] = sm;
      if (J.hbo) act_bits32(J.hq + pk * F + 32 * b, J.hbo + (pk * F + 32 * b) / 4);  // v55f: the down inputs' bit planes
    }
  }
}
// one warp task of a matrix job: each pick's inputs (tokx) and output row (ypk); the first pick fills the slot on a miss
template <int M, bool kMT, int ONLY = kOnlyRx>
__device__ __forceinline__ void rx_rows(const DJob& J, long long t) {
  const int lane = static_cast<int>(threadIdx.x & 31), rpw = 32 / J.lpr;
  const int r = static_cast<int>((t - J.task0) * rpw) + lane / J.lpr, part = lane % J.lpr;
  if constexpr (kMT) {  // v47: the group's picks (2-4) on one load of the rows (fill, if any, by that one load)
    if (J.ntok > 1) {
      const std::int8_t* xq[4];
      const float* xs[4];
      const int* xsum[4];
      long long yo[4];
      for (int tk = 0; tk < 4; ++tk) {
        const int u = tk < J.ntok ? tk : 0;
        const long long xo = J.tokx[u];
        xq[tk] = J.xq + xo * J.xqs;
        xs[tk] = J.xs + xo * J.xss;
        xsum[tk] = J.xsum + xo * J.xss;
        yo[tk] = J.ypk[u] * J.ys;
      }
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < J.n) lane_part_d<M, ONLY>(J, r, part, xq, xs, xsum, J.ntok, acc);
      store_rows(J, r, part, acc, J.ntok, yo);
      return;
    }
  }
  for (int tk = 0; tk < J.ntok; ++tk) {
    const long long xo = J.tokx[tk];
    const std::int8_t* xq = J.xq + xo * J.xqs;
    const float* xs = J.xs + xo * J.xss;
    const int* xsum = J.xsum + xo * J.xss;
    float acc = 0.f;
    if (r < J.n)
      acc = (M == 2 && tk == 0) ? lane_part<2, ONLY>(J, r, part, xq, xs, xsum, J.ntok == 1) : lane_part<1, ONLY>(J, r, part, xq, xs, xsum, J.ntok == 1);
    for (int o = J.lpr / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (r < J.n && part == 0) J.y[J.ypk[tk] * J.ys + r] = acc;
  }
}
// v46a-2: bytes per 256-value block of the stage's formats (as read: Q8_0 in R3's repacked layout)
__host__ __device__ inline int rx_blk(int fmt) {
  return fmt == kQ5K || fmt == kQ5KP ? 176 : fmt == kQ6KS ? 210 : fmt == kQ8_0 ? 272 : 144;  // Q4_TG2(P), Q4_K: 144
}
// a warp copies n contiguous bytes from src into its shared buffer, keeping src's 16-byte phase (32 lanes x 16 bytes per
// instruction: every 32-byte sector requested once, as the reader's staging) and, with fill, into the cache slot at the
// same offset; returns where the copy of src[0] is
__device__ __forceinline__ const std::uint8_t* rx_stage(const std::uint8_t* src, long long n, std::uint8_t* buf, std::uint8_t* fill, int lane) {
  const unsigned mis = static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(src) & 15u);
  std::uint8_t* dst = buf + mis;
  long long head = static_cast<long long>((16u - mis) & 15u);
  if (head > n) head = n;
  for (long long i = lane; i < head; i += 32) {
    const std::uint8_t b = src[i];
    dst[i] = b;
    if (fill) fill[i] = b;
  }
  const long long n4 = (n - head) / 16;
  const int4* s4 = reinterpret_cast<const int4*>(src + head);
  int4* d4 = reinterpret_cast<int4*>(dst + head);
  int4* f4 = fill ? reinterpret_cast<int4*>(fill + head) : nullptr;
  for (long long i = lane; i < n4; i += 32) {
    const int4 v = ld_cs4(s4 + i);
    d4[i] = v;
    if (f4) f4[i] = v;
  }
  for (long long i = head + n4 * 16 + lane; i < n; i += 32) {
    const std::uint8_t b = src[i];
    dst[i] = b;
    if (fill) fill[i] = b;
  }
  __syncwarp();
  return dst;
}
// v46a-2: one warp task with its rows staged first (a task's rows are consecutive: 16 blocks of 256 values in one range)
template <bool kMT, int ONLY = kOnlyRx>
__device__ __forceinline__ void rx_rows_staged(const DJob& J, long long t, std::uint8_t* sbuf) {
  const int lane = static_cast<int>(threadIdx.x & 31), rpw = 32 / J.lpr, nB = J.K / 256;
  const int r0 = static_cast<int>((t - J.task0) * rpw);
  const int r = r0 + lane / J.lpr, part = lane % J.lpr;
  const int nr = J.n - r0 < rpw ? J.n - r0 : rpw;
  const long long rowb = static_cast<long long>(nB) * rx_blk(J.fmt);
  const std::uint8_t* src = J.w + static_cast<long long>(r0) * rowb;
  const std::uint8_t* wb;
  if (J.fmt == kQ4TG2P || J.fmt == kQ5KP) {  // v54b / v54c: each row's first J.pb bits (the prefix) from w at stride J.rs,
                                            // packed in the buffer and, on a miss, in the slot (fw at frs)
    const long long pf = static_cast<long long>(plane_prefix(J.fmt == kQ4TG2P ? g_q4_tg2p : g_q5_kp,
                                                             static_cast<std::uint64_t>(nB) * (J.fmt == kQ4TG2P ? 144u : 176u), J.pb));
    wb = nullptr;
    for (int rr = 0; rr < nr; ++rr) {
      const std::uint8_t* d = rx_stage(J.w + (r0 + rr) * J.rs, pf, sbuf + rr * pf, J.fw ? J.fw + (r0 + rr) * J.frs : nullptr, lane);
      if (rr == 0) wb = d;
    }
  } else {
    wb = rx_stage(src, nr * rowb, sbuf, J.fd ? const_cast<std::uint8_t*>(src) + J.fd : nullptr, lane);
  }
  if constexpr (kMT) {  // v47: the group's picks (2-4) on one load of the rows (fill, if any, by that one load)
    if (J.ntok > 1) {
      const std::int8_t* xq[4];
      const float* xs[4];
      const int* xsum[4];
      long long yo[4];
      for (int tk = 0; tk < 4; ++tk) {
        const int u = tk < J.ntok ? tk : 0;
        const long long xo = J.tokx[u];
        xq[tk] = J.xq + xo * J.xqs;
        xs[tk] = J.xs + xo * J.xss;
        xsum[tk] = J.xsum + xo * J.xss;
        yo[tk] = J.ypk[u] * J.ys;
      }
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < J.n) lane_part_d<3, ONLY>(J, r, part, xq, xs, xsum, J.ntok, acc, wb, static_cast<long long>(r0) * nB);
      store_rows(J, r, part, acc, J.ntok, yo);
      __syncwarp();
      return;
    }
  }
  for (int tk = 0; tk < J.ntok; ++tk) {
    const long long xo = J.tokx[tk];
    float acc = 0.f;
    if (r < J.n) acc = lane_part<3, ONLY>(J, r, part, J.xq + xo * J.xqs, J.xs + xo * J.xss, J.xsum + xo * J.xss, false, wb, static_cast<long long>(r0) * nB);
    for (int o = J.lpr / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (r < J.n && part == 0) J.y[J.ypk[tk] * J.ys + r] = acc;
  }
  __syncwarp();  // the buffer is reused by the warp's next task
}
// the layer's experts: every warp walks the tasks in table order (gate and up of all experts first, then the roundings,
// then the downs); a task whose job waits for a group starts once the group's count is complete (acquire), and a task of
// a signalling job adds 1 (release). All blocks are resident (cooperative launch), and a job waits only for groups of
// jobs before it, so the lowest unfinished task can always run: no deadlock.
// v52: the task loop (after lut_setup) for k_rx and the fused MoE half
// v55e3 diagnosis (temporary; kStat, the verbose kernel only): per warp, SM cycles waiting for a group (dep) vs running
// tasks, tasks, the loop's span; per format (J.fmt 0..10) the cycles and tasks; staged and filling tasks (st layout:
// gpu_stream.h kRxStatN)
template <bool kMT, int ONLY = kOnlyRx, bool kStat = false>
__device__ __forceinline__ void rx_body(const std::uint8_t* __restrict__ plan, int maxp, int sbytes, unsigned long long* st = nullptr) {
  std::uint8_t* sbuf = reinterpret_cast<std::uint8_t*>(s_lut) + kLutRxBytes + static_cast<long long>(threadIdx.x / 32) * sbytes;  // v46a-2, v54
  const RxLayout Lo = rx_layout(maxp);
  const auto* hd = reinterpret_cast<const RxHdr*>(plan);
  const auto* jobs = reinterpret_cast<const DJob*>(plan + Lo.jobs);
  const auto* sig = reinterpret_cast<const int*>(plan + Lo.sig);
  const auto* dep = reinterpret_cast<const int*>(plan + Lo.dep);
  const auto* gt = reinterpret_cast<const long long*>(plan + Lo.gtasks);
  auto* gd = reinterpret_cast<unsigned long long*>(const_cast<std::uint8_t*>(plan) + Lo.gdone);
  const int njobs = hd->njobs;
  const long long end = hd->ntasks;
  const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
  const long long w0 = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = static_cast<int>(threadIdx.x & 31);
  int j = 0;
  long long c_wait = 0, c_work = 0, c_beg = 0, ntk = 0;
  if constexpr (kStat) c_beg = clock64();
  for (long long t = w0; t < end; t += warps) {
    long long ca = 0, cb = 0;
    if constexpr (kStat) ca = clock64();
    while (j + 1 < njobs && jobs[j + 1].task0 <= t) ++j;
    const int dg = dep[j];
    if (dg >= 0) {
      if (lane == 0) {
        const unsigned long long target = static_cast<unsigned long long>(gt[dg]);
        while (ld_acquire_u64(gd + dg) < target) __nanosleep(32);
      }
      __syncwarp();
    }
    if constexpr (kStat) cb = clock64();
    const DJob& J = jobs[j];
    if (J.fmt == kEpi) rx_epi(J);
    else if (J.stg) rx_rows_staged<kMT, ONLY>(J, t, sbuf);
    else if (J.fd) rx_rows<2, kMT, ONLY>(J, t);
    else rx_rows<1, kMT, ONLY>(J, t);
    if constexpr (kStat) {
      __syncwarp();
      const long long cc = clock64();
      c_wait += cb - ca;
      c_work += cc - cb;
      ++ntk;
      if (lane == 0) {
        const int f = J.fmt >= 0 && J.fmt <= 10 ? J.fmt : 11;
        atomicAdd(st + 8 + f, static_cast<unsigned long long>(cc - cb));
        atomicAdd(st + 20 + f, 1ull);
        if (J.stg) {  // staged = read from host RAM over PCIe (rx-load miss: the experts not in VRAM)
          atomicAdd(st + 5, 1ull);
          atomicAdd(st + 7, static_cast<unsigned long long>(cc - cb));
        }
        if (J.fd) atomicAdd(st + 6, 1ull);
      }
    }
    const int sg = sig[j];
    if (sg >= 0) {
      __syncwarp();
      if (lane == 0) red_release_add(gd + sg, 1ull);
    }
  }
  if constexpr (kStat) {
    if (lane == 0) {
      atomicAdd(st + 0, static_cast<unsigned long long>(c_wait));
      atomicAdd(st + 1, static_cast<unsigned long long>(c_work));
      atomicAdd(st + 2, static_cast<unsigned long long>(ntk));
      atomicAdd(st + 3, static_cast<unsigned long long>(clock64() - c_beg));
      atomicAdd(st + 4, 1ull);
    }
  }
}
template <bool kMT>
__global__ void __launch_bounds__(256) k_rx(const std::uint8_t* __restrict__ plan, int maxp, int sbytes) {
  lut_setup_rx();
  rx_body<kMT>(plan, maxp, sbytes);
}
// v55e diagnosis (temporary, never launched): the expert stage compiled for one format, so the kernel report and the ptxas
// listing give each format path's registers and stack (the stage's kernel needs the largest of them)
template <int F>
__global__ void __launch_bounds__(256) k_rx_probe(const std::uint8_t* __restrict__ plan, int maxp, int sbytes) {
  lut_setup_rx();
  rx_body<false, F>(plan, maxp, sbytes);
}

// v52 (token kernel 3a): the MoE half of a layer, one cooperative launch (moe_half in gpu_stream.h). Phases:
//  A every block: x + a normalized per row into its shared-memory staging (reads only; block 0 also writes xn and the
//    8-bit rounding), then its warps' share of the router rows and the shared gate row (md::gemv_row, k_gemv_f32's sums)
//  B block 0: each token's picks (md::router_row), the shared gate's sigmoid, the grouping (md::rd3_group_one), the plan
//  C every block: the R3 expert stage (rx_body, k_rx's task loop)
//  D every thread: the combine (md::rd3_combine_one) into y and the residual x = (x + a) + y
// A grid barrier between phases. The values and their order are the separate kernels', so the bits are theirs.
// v55f2 (author 06:17: the v55f fused kernel spilled 152 B at 128 registers while k_rx alone is 108 with none): phases A
// and B run once per layer and carry the heaviest state (the rounding with the bit planes, the router, the warp grouping's
// expert pointers). As separate, non-inlined functions their registers and any spill stay in that setup code, and the
// expert loop (phase C) keeps its own allocation. The kernel's argument is a __grid_constant__, so the functions read it
// in place (no copy). Same code, same order: the bits are unchanged.
#define PG_NOINL __attribute__((noinline))
template <int NJ>
__device__ PG_NOINL void mh_phase_a(const MoeHalfArgs& A, float* stg, double* s_sh) {
  const int H = A.H, n = A.n, nsub = H / 32;
  const int tid = static_cast<int>(threadIdx.x);
  for (int t = 0; t < n; ++t) {
    const float* xt = A.x + static_cast<long long>(t) * H;
    const float* at = A.a + static_cast<long long>(t) * H;
    float* st = stg + t * nsub * md::kGvPad;
    float* xnt = A.xn + static_cast<long long>(t) * H;
    const bool b0 = blockIdx.x == 0;
    md::rms_norm_seg([xt, at](std::int64_t i) { return add(xt[i], at[i]); },
                     [st, xnt, b0](std::int64_t i, float v) {
                       st[(i >> 5) * md::kGvPad + (i & 31)] = v;
                       if (b0) xnt[i] = v;
                     },
                     A.post_w, H, A.eps, 0, s_sh);
  }
  __syncthreads();
  if (blockIdx.x == 0)
    for (int id = tid; id < n * nsub; id += static_cast<int>(blockDim.x)) {
      const int t = id / nsub, b = id - t * nsub;
      float v[32];
      const float* sb = stg + (t * nsub + b) * md::kGvPad;
      for (int i = 0; i < 32; ++i) v[i] = sb[i];
      md::round32<false>(v, t, b, H, n, 0, 0.0f, A.xr, H, A.xq, A.xs, A.xsum);
      if (A.xb) act_bits32(A.xq + t * H + 32 * b, A.xb + (t * H + 32 * b) / 4);  // v55f: the expert inputs' bit planes (this thread's values)
    }
  {
    const long long rows = A.ne + (A.wsh ? 1 : 0);
    const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
    for (long long r = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + tid / 32; r < rows; r += warps) {
      if (r < A.ne) md::gemv_row<NJ>(A.wr, static_cast<std::uint64_t>(H) * 4u, r, stg, nsub, n, A.logits, A.ne);
      else md::gemv_row<NJ>(A.wsh, static_cast<std::uint64_t>(H) * 4u, 0, stg, nsub, n, A.gsh, 1);
    }
  }
}
// block 0 only (every thread of it: the router and the plan synchronise the block)
__device__ PG_NOINL void mh_phase_b(const MoeHalfArgs& A, float* stg, float* s_rsh, float* s_inv, float* s_bv, int* s_bi) {
  const int n = A.n, tid = static_cast<int>(threadIdx.x);
  const md::RouterSmem S{stg, stg + md::kRouterMaxExperts, reinterpret_cast<unsigned char*>(stg + 2 * md::kRouterMaxExperts), s_rsh, s_inv, s_bv, s_bi};
  for (int t = 0; t < n; ++t) {
    md::router_row(A.logits + static_cast<long long>(t) * A.ne, A.ne, A.nu, A.wscale, A.sel + t * A.nu, A.wsel + t * A.nu, A.band, S);
    __syncthreads();
  }
  if (A.wsh && tid < n) A.gsh[tid] = md::sigmoid_d(A.gsh[tid]);  // deltanet_gates(gsh, ...) with one head: beta = sigmoid(beta)
  if (tid < 32) md::rd3_group_one<true>(A.tab, A.wsh ? A.shx : nullptr, A.sel, n, A.nu, A.cache, A.F, A.grp, A.gsrc, A.gfill, nullptr);  // v55e: one warp
  __syncthreads();
  rx_plan_body(A.grp, A.gsrc, A.gfill, n, A.nu, A.H, A.F, A.xq, A.xs, A.xsum, A.gu, A.hq, A.hs, A.hsum, A.yd, static_cast<std::uint8_t*>(A.plan),
               A.max_picks, A.load_mode, A.xb, A.hb);
}
template <bool kMT, int NJ, bool kStat = false>
__global__ void __launch_bounds__(256) k_moe_half(const __grid_constant__ MoeHalfArgs A, int sbytes) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  __shared__ double s_sh[33];
  __shared__ float s_rsh[33];
  __shared__ float s_inv;
  __shared__ float s_bv[8];
  __shared__ int s_bi[8];
  float* stg = s_lut;  // the dynamic shared memory: staging (A), the router's arrays (B), the LUT + warp buffers (C)
  const int H = A.H, n = A.n;
  const int tid = static_cast<int>(threadIdx.x);
  const bool phs = A.ph != nullptr && blockIdx.x == 0 && threadIdx.x == 0;  // v55b: phase stamps
  if (phs) A.ph[0] = gtimer_s();
  // ---- A
  mh_phase_a<NJ>(A, stg, s_sh);
  grid.sync();
  if (phs) A.ph[1] = gtimer_s();
  // ---- B
  if (blockIdx.x == 0) mh_phase_b(A, stg, s_rsh, &s_inv, s_bv, s_bi);
  grid.sync();
  if (phs) A.ph[2] = gtimer_s();
  // ---- C
  lut_setup_rx();
  rx_body<kMT, kOnlyRx, kStat>(static_cast<const std::uint8_t*>(A.plan), A.max_picks, sbytes, A.st);
  grid.sync();
  if (phs) A.ph[3] = gtimer_s();
  // ---- D
  const long long tot = static_cast<long long>(n) * H, step = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long id = static_cast<long long>(blockIdx.x) * blockDim.x + tid; id < tot; id += step) {
    const long long t = id / H, r = id - t * H;
    const float yv = md::rd3_combine_one(A.yd, A.wsel, A.wsh ? A.gsh : nullptr, A.nu, n, H, t, r, A.sel, nullptr);
    const float x1 = add(A.x[id], A.a[id]);  // k_add_rows of the attention's output
    A.y[id] = yv;
    A.x[id] = add(x1, yv);  // k_add_rows of the MoE output
  }
  if (phs) A.ph[4] = gtimer_s();  // block 0's end of D
}

bool grid_ready() {
  static bool ready = false;
  if (ready) return true;
  float g[4][16];
  for (int t = 0; t < 4; ++t)
    for (int i = 0; i < 16; ++i) g[t][i] = quant::q4_tg_grid(t, 2)[i];
  if (!sck(cudaMemcpyToSymbol(c_grid, g, sizeof g), "stream: grid upload")) return false;
  float g3[4][16], g2[4][16];  // v54: the base grids of the bit planes
  for (int t = 0; t < 4; ++t) {
    tg2p_base_grid(g[t], 3, g3[t]);
    tg2p_base_grid(g[t], 2, g2[t]);
  }
  if (!sck(cudaMemcpyToSymbol(c_grid3, g3, sizeof g3), "stream: grid upload (3-bit base)") ||
      !sck(cudaMemcpyToSymbol(c_grid2, g2, sizeof g2), "stream: grid upload (2-bit base)"))
    return false;
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
    if (J.ntok < 1 || J.ntok > 4) return sfail("StreamPlan::build: job " + std::to_string(j) + ": 1-4 tokens");
    if (J.ntok > 1) mt_ = true;  // v47: the multi-token kernel (weights once for all tokens)
    D.ntok = J.ntok;
    D.xqs = J.xq_stride > 0 ? J.xq_stride : J.w->k;
    D.xss = J.xs_stride > 0 ? J.xs_stride : J.w->k / 32;
    D.ys = J.y_stride > 0 ? J.y_stride : J.w->n;
    if (J.ntok > 1 && D.xqs % 16 != 0) return sfail("StreamPlan::build: job " + std::to_string(j) + ": xq stride must keep 16-byte alignment");
    grp[static_cast<std::size_t>(j)] = J.signal;
    dep[static_cast<std::size_t>(j)] = J.wait;
    if (J.signal >= 0) {
      gtasks[static_cast<std::size_t>(J.signal)] += D.ntask;
      if (first_signal[static_cast<std::size_t>(J.signal)] < 0) first_signal[static_cast<std::size_t>(J.signal)] = j;
      last_signal[static_cast<std::size_t>(J.signal)] = j;
    }
    t += D.ntask;
    send[static_cast<std::size_t>(stage)] = t;
    bytes_ += J.w->bytes;  // read once from DRAM whatever ntok
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
      !sck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mt_ ? k_staged<true> : k_staged<false>, 256, kLutBytes), "occupancy"))
    return false;
  if (!coop) return sfail("StreamPlan::build: the device has no cooperative launch");
  if (occ < 1) return sfail("StreamPlan::build: the staged kernel does not fit on an SM");
  grid_ = sms * occ;
  epoch_ = 0;
  // v55d: a one-token plan whose jobs are all Q8_0 (or all Q6_K) runs the kernel compiled for that format alone
  only_ = -1;
  grid_lean_ = 0;
  if (g_lean_dense && !mt_ && !dj.empty()) {
    const int f0 = dj[0].fmt;
    bool same = f0 == kQ8_0 || f0 == kQ6K;
    for (const DJob& D : dj) same = same && D.fmt == f0;
    if (same) {
      const void* fn = f0 == kQ8_0 ? reinterpret_cast<const void*>(k_stream<false, kQ8_0>) : reinterpret_cast<const void*>(k_stream<false, kQ6K>);
      int occl = 0;
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occl, fn, 256, kLutBytes) == cudaSuccess && occl >= 1) {
        only_ = f0;
        // v55e: no more blocks than the plan's tasks fill (8 warps a block): v55d launched every resident block (5 x 46
        // for Q8_0) even for a matrix of a few hundred tasks (v55d A/B: 1.2-1.3 ms per token slower than g55b, MEASURED;
        // the lean kernels are the suspect, --no-lean-dense the control)
        const long long need = (ntasks_ + 7) / 8;
        grid_lean_ = static_cast<int>(std::min<long long>(static_cast<long long>(sms) * occl, need < 1 ? 1 : need));
      }
      (void)cudaGetLastError();
    }
  }
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
  if (!sck(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mt_ ? k_staged<true> : k_staged<false>), dim3(static_cast<unsigned>(grid_)), dim3(256), args, kLutBytes, sst(s)),
           "stream: cooperative launch"))
    return false;
  epoch_ = ep;
  return true;
}

// v26: for CUDA graphs: the counters cleared on the stream before the launch, run number 1 every time (a captured launch
// keeps its arguments, so the host-side epoch cannot advance between replays)
bool StreamPlan::run_reset(Stream s) {
  if (!dev_) return sfail("StreamPlan::run_reset: not built");
  auto* b = static_cast<std::uint8_t*>(dev_);
  const std::uint64_t cbytes = off_gdone_ + 8 * static_cast<std::uint64_t>(ngroups_ > 0 ? ngroups_ : 1) - off_done_;
  if (!sck(cudaMemsetAsync(b + off_done_, 0, cbytes, sst(s)), "stream: counter reset")) return false;
  const DJob* jobs = reinterpret_cast<const DJob*>(b + off_jobs_);
  const long long* stage_end = reinterpret_cast<const long long*>(b + off_stage_);
  auto* done = reinterpret_cast<unsigned long long*>(b + off_done_);
  const int* grp = reinterpret_cast<const int*>(b + off_grp_);
  const int* dep = reinterpret_cast<const int*>(b + off_dep_);
  const long long* gtasks = reinterpret_cast<const long long*>(b + off_gtasks_);
  auto* gdone = reinterpret_cast<unsigned long long*>(b + off_gdone_);
  int nj = njobs_, ns = nstages_;
  unsigned long long ep = 1;
  void* args[] = {&jobs, &nj, &stage_end, &ns, &done, &grp, &dep, &gtasks, &gdone, &ep};
  return sck(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mt_ ? k_staged<true> : k_staged<false>), dim3(static_cast<unsigned>(grid_)), dim3(256), args, kLutBytes, sst(s)),
             "stream: cooperative launch (reset)");
}

bool StreamPlan::run_unstaged(Stream s) {
  if (!dev_) return sfail("StreamPlan::run_unstaged: not built");
  const DJob* jobs = reinterpret_cast<const DJob*>(static_cast<std::uint8_t*>(dev_) + off_jobs_);
  if (mt_) k_stream<true><<<static_cast<unsigned>(grid_), 256, kLutBytes, sst(s)>>>(jobs, njobs_, ntasks_);
  else if (only_ == kQ8_0) k_stream<false, kQ8_0><<<static_cast<unsigned>(grid_lean_), 256, kLutBytes, sst(s)>>>(jobs, njobs_, ntasks_);  // v55d
  else if (only_ == kQ6K) k_stream<false, kQ6K><<<static_cast<unsigned>(grid_lean_), 256, kLutBytes, sst(s)>>>(jobs, njobs_, ntasks_);
  else k_stream<false><<<static_cast<unsigned>(grid_), 256, kLutBytes, sst(s)>>>(jobs, njobs_, ntasks_);
  return sck(cudaGetLastError(), "stream: launch");
}

const char* stream_last_error() { return s_err.c_str(); }

bool rx_type_ok(std::uint32_t type, std::int64_t k) { return rx_fmt(type) >= 0 && (k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096); }

std::uint64_t rx_scratch_bytes(int max_picks) { return max_picks > 0 ? rx_layout(max_picks).total : 0; }

bool moe_r3(const ExpertPtr* tab, const ExpertPtr* shx, const std::int32_t* sel, const float* wsel, const float* gsh, int n_used,
            const std::int8_t* xq, const float* xs, const int* xsum, std::int64_t tokens, std::int64_t H, std::int64_t F, const ReaderCache* cache,
            float* gu, std::int8_t* hq, float* hs, int* hsum, float* yd, int* grp, ExpertPtr* gsrc, ExpertPtr* gfill, void* plan, int max_picks,
            float* y, unsigned long long* ts, Stream s, int load_mode) {
  if (load_mode < 0 || load_mode > 2) return sfail("moe_r3: load_mode 0 (lane), 1 (staged for RAM experts) or 2 (staged for all)");
  if (!tab || !sel || !wsel || !xq || !xs || !xsum || !gu || !hq || !hs || !hsum || !yd || !grp || !gsrc || !gfill || !plan || !y || n_used <= 0 ||
      n_used > 8 || (shx && !gsh))
    return sfail("moe_r3: bad pointer or n_used");
  auto k_ok = [](std::int64_t k) { return k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096; };
  if (!k_ok(H) || !k_ok(F)) return sfail("moe_r3: H and F must be 256, 512, 1024, 2048 or 4096");
  if (tokens <= 0) return true;
  if (tokens > 4) return sfail("moe_r3: at most 4 tokens");
  const std::int64_t routed = tokens * n_used, picks = routed + (shx ? tokens : 0);
  if (picks > max_picks) return sfail("moe_r3: more picks than the plan buffer holds");
  if (reinterpret_cast<std::uintptr_t>(xq) % 16 != 0 || reinterpret_cast<std::uintptr_t>(hq) % 16 != 0) return sfail("moe_r3: xq / hq must be 16-byte aligned");
  if (!grid_ready()) return false;
  // v46a-2: the staging buffer per warp: one task's 16 blocks of the largest format (Q8_0 repacked, 272 bytes) + the 16-byte
  // phase, 8 warps per block: 8 KB grid + 34.3 KB (under the 48 KB a launch gets without opting in)
  const int sbytes = load_mode ? g_stage_bytes : 0;  // v55f
  const int smem = kLutRxBytes + 8 * sbytes;  // v54: 24 KB of grids + the warps' buffers (59 KB: opted in past 48 KB)
  static int grids[2][3] = {{0, 0, 0}, {0, 0, 0}};
  const bool mt = tokens > 1;  // v47: steps of 2-4 tokens run the kernel that loads each row once for all of a group's picks
  int& grid = grids[mt ? 1 : 0][load_mode];
  if (!grid) {
    int dev = 0, sms = 0, occ = 0, coop = 0;
    if (!sck(cudaGetDevice(&dev), "cudaGetDevice") || !sck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count") ||
        !sck(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev), "cooperative launch attribute") ||
        !sck(cudaFuncSetAttribute(reinterpret_cast<const void*>(mt ? k_rx<true> : k_rx<false>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
             "expert stage: shared memory opt-in") ||
        !sck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mt ? k_rx<true> : k_rx<false>, 256, smem), "occupancy (expert stage)"))
      return false;
    if (!coop) return sfail("moe_r3: the device has no cooperative launch");
    if (occ < 1) return sfail("moe_r3: the expert stage kernel does not fit on an SM");
    grid = sms * occ;
  }
  if (ts) stamp_launch(ts + 0, s);
  if (!rd3_group_launch(tab, shx, sel, static_cast<int>(tokens), n_used, cache, H, F, grp, gsrc, gfill, s)) return sfail("moe_r3: grouping launch");
  if (ts) stamp_launch(ts + 1, s);
  auto* pb = static_cast<std::uint8_t*>(plan);
  k_rx_plan<<<1, 128, 0, sst(s)>>>(grp, gsrc, gfill, static_cast<int>(tokens), n_used, static_cast<int>(H), static_cast<int>(F), xq, xs, xsum, gu, hq, hs,
                                   hsum, yd, pb, max_picks, load_mode);
  if (!sck(cudaGetLastError(), "moe_r3: plan launch")) return false;
  if (ts) stamp_launch(ts + 2, s);
  const std::uint8_t* cp = pb;
  int mp = max_picks, sb = sbytes;
  void* args[] = {&cp, &mp, &sb};
  if (!sck(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mt ? k_rx<true> : k_rx<false>), dim3(static_cast<unsigned>(grid)), dim3(256), args, static_cast<std::size_t>(smem), sst(s)),
           "moe_r3: cooperative launch"))
    return false;
  if (ts) {
    stamp_launch(ts + 3, s);
    stamp_launch(ts + 4, s);
  }
  if (!rd3_combine_launch(yd, wsel, shx ? gsh : nullptr, n_used, tokens, H, y, sel, s)) return sfail("moe_r3: combine launch");
  if (ts) stamp_launch(ts + 5, s);
  return true;
}

bool moe_half(const MoeHalfArgs& A, Stream s) {
  if (A.load_mode < 0 || A.load_mode > 2) return sfail("moe_half: load_mode 0, 1 or 2");
  if (!A.x || !A.a || !A.post_w || !A.xn || !A.xr || !A.xq || !A.xs || !A.xsum || !A.wr || !A.logits || !A.sel || !A.wsel || !A.tab || !A.grp ||
      !A.gsrc || !A.gfill || !A.gu || !A.hq || !A.hs || !A.hsum || !A.yd || !A.plan || !A.y || (A.wsh && (!A.gsh || !A.shx)))
    return sfail("moe_half: bad pointer");
  if (A.n <= 0) return true;
  if (A.n > 4) return sfail("moe_half: at most 4 tokens");
  if (A.nu <= 0 || A.nu > 8 || A.ne <= 0 || A.ne > md::kRouterMaxExperts) return sfail("moe_half: n_used 1-8, n_expert 1-1024");
  const int nsub = A.H / 32;
  auto k_ok = [](std::int64_t k) { return k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096; };
  if (!k_ok(A.H) || !k_ok(A.F)) return sfail("moe_half: H and F must be 256, 512, 1024, 2048 or 4096");
  const std::int64_t picks = static_cast<std::int64_t>(A.n) * A.nu + (A.wsh ? A.n : 0);
  if (picks > A.max_picks) return sfail("moe_half: more picks than the plan buffer holds");
  if ((reinterpret_cast<std::uintptr_t>(A.wr) & 15u) || (A.wsh && (reinterpret_cast<std::uintptr_t>(A.wsh) & 15u)) ||
      reinterpret_cast<std::uintptr_t>(A.xq) % 16 != 0 || reinterpret_cast<std::uintptr_t>(A.hq) % 16 != 0)
    return sfail("moe_half: wr / wsh / xq / hq must be 16-byte aligned");
  if (!grid_ready()) return false;
  const int sbytes = A.load_mode ? g_stage_bytes : 0;  // v55f
  // dynamic shared memory: the largest of the phases' uses (at most 4 x 128 x 144 B staging = 72 KB: opt in past 48 KB)
  const int sm_stage = 4 * nsub * md::kGvPad * 4, sm_router = 2 * md::kRouterMaxExperts * 4 + md::kRouterMaxExperts, sm_rx = kLutRxBytes + 8 * sbytes;
  const int smem = std::max(sm_stage, std::max(sm_router, sm_rx));
  const bool mt = A.n > 1;
  const bool stat = !mt && A.st != nullptr;  // v55e3 diagnosis: the counting kernel (verbose runs only)
  const int nj = nsub <= 32 ? 1 : nsub / 32;
  void* fn = nullptr;
  if (stat) fn = nj == 1 ? reinterpret_cast<void*>(k_moe_half<false, 1, true>) : nj == 2 ? reinterpret_cast<void*>(k_moe_half<false, 2, true>)
                                                                                        : reinterpret_cast<void*>(k_moe_half<false, 4, true>);
  else if (nj == 1) fn = mt ? reinterpret_cast<void*>(k_moe_half<true, 1>) : reinterpret_cast<void*>(k_moe_half<false, 1>);
  else if (nj == 2) fn = mt ? reinterpret_cast<void*>(k_moe_half<true, 2>) : reinterpret_cast<void*>(k_moe_half<false, 2>);
  else fn = mt ? reinterpret_cast<void*>(k_moe_half<true, 4>) : reinterpret_cast<void*>(k_moe_half<false, 4>);
  static int grids[2][3][3][2] = {};
  int& grid = grids[mt ? 1 : 0][nj == 1 ? 0 : (nj == 2 ? 1 : 2)][A.load_mode][stat ? 1 : 0];
  if (!grid) {
    int dev = 0, sms = 0, occ = 0, coop = 0;
    if (!sck(cudaGetDevice(&dev), "cudaGetDevice") || !sck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count") ||
        !sck(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev), "cooperative launch attribute") ||
        !sck(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "moe_half: shared memory opt-in") ||
        !sck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, static_cast<std::size_t>(smem)), "occupancy (MoE half)"))
      return false;
    if (!coop) return sfail("moe_half: the device has no cooperative launch");
    if (occ < 1) return sfail("moe_half: the kernel does not fit on an SM");
    grid = sms * occ;
    int occ_rx = 0;  // v52 diagnosis: the separate expert stage's blocks per SM at its own shared memory, for comparison
    (void)cudaFuncSetAttribute(reinterpret_cast<const void*>(mt ? k_rx<true> : k_rx<false>), cudaFuncAttributeMaxDynamicSharedMemorySize, sm_rx);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_rx, mt ? k_rx<true> : k_rx<false>, 256, static_cast<std::size_t>(sm_rx));
    std::fprintf(stderr, "moe_half: %s tokens%s, H %d: grid %d (%d SMs x %d blocks), shared memory %d B; k_rx alone %d blocks per SM\n", mt ? "2-4" : "1",
                 stat ? " (counting kernel)" : "", A.H, grid, sms, occ, smem, occ_rx);
  }
  MoeHalfArgs a = A;
  int sb = sbytes;
  void* args[] = {&a, &sb};
  return sck(cudaLaunchCooperativeKernel(fn, dim3(static_cast<unsigned>(grid)), dim3(256), args, static_cast<std::size_t>(smem), sst(s)),
             "moe_half: cooperative launch");
}

void set_lean_dense(bool on) { g_lean_dense = on; }  // v55e (temporary switch)

// v55f: a warp task stages 16 blocks of 256 values whatever the matrix shape (rows x blocks per row = 4096 / K x K / 256), so
// the buffer is 16 x the largest block a staged matrix has, as staged (a plane matrix: its prefix at the slider's bits)
int rx_stage_block_bytes(std::uint32_t type, int plane_bits) {
  if (type == g_q4_tg2p) return static_cast<int>(tg2p_prefix(1, plane_bits >= 2 ? plane_bits : 4));
  if (type == g_q5_kp) return static_cast<int>(plane_prefix(g_q5_kp, 176, plane_bits >= 2 ? plane_bits : 5));
  const int f = rx_fmt(type);
  return f < 0 ? 272 : rx_blk(f);
}
void rx_set_stage_bytes(int max_block_bytes) {
  const int b = max_block_bytes > 0 && max_block_bytes <= 272 ? max_block_bytes : 272;
  g_stage_bytes = 16 * b + 32;
}

bool rx_set_grid_bits(int bits) {  // v55b
  const int b = bits == 3 || bits == 2 ? bits : 4;
  return sck(cudaMemcpyToSymbol(c_rx_bits, &b, sizeof b), "stream: expert stage grid bits");
}

void stream_kernel_report() {  // v55b: registers, local memory and blocks per SM at the shared memory each kernel is launched with
  struct K {
    const void* f;
    const char* name;
    int smem;
  };
  const int sb = g_stage_bytes;  // v55f
  const int sm_moe = std::max(4 * 64 * md::kGvPad * 4, std::max(2 * md::kRouterMaxExperts * 4 + md::kRouterMaxExperts, kLutRxBytes + 8 * sb));
  const K ks[] = {{reinterpret_cast<const void*>(k_moe_half<false, 1>), "k_moe_half<1 token>", sm_moe},
                  {reinterpret_cast<const void*>(k_moe_half<true, 1>), "k_moe_half<2-4 tokens>", sm_moe},
                  {reinterpret_cast<const void*>(k_moe_half<false, 2, true>), "k_moe_half<1, counting>", sm_moe},
                  {reinterpret_cast<const void*>(k_rx<false>), "k_rx<1 token>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx<true>), "k_rx<2-4 tokens>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_stream<false>), "k_stream<1 token>", kLutBytes},
                  {reinterpret_cast<const void*>(k_stream<true>), "k_stream<2-4 tokens>", kLutBytes},
                  {reinterpret_cast<const void*>(k_stream<false, kQ8_0>), "k_stream<Q8_0 only>", kLutBytes},
                  {reinterpret_cast<const void*>(k_stream<false, kQ6K>), "k_stream<Q6_K only>", kLutBytes},
                  {reinterpret_cast<const void*>(k_staged<false>), "k_staged<1 token>", kLutBytes},
                  // v55e diagnosis (temporary): the planner, and the expert stage compiled for each one format
                  {reinterpret_cast<const void*>(k_rx_plan), "k_rx_plan", 0},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ4TG2P>), "k_rx<Q4_TG2 planes only>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ4TG2>), "k_rx<Q4_TG2 only>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ5KP>), "k_rx<Q5_K planes only>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ5K>), "k_rx<Q5_K only>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ4K>), "k_rx<Q4_K only>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ6KS>), "k_rx<Q6_K only>", kLutRxBytes + 8 * sb},
                  {reinterpret_cast<const void*>(k_rx_probe<kQ8_0>), "k_rx<Q8_0 only>", kLutRxBytes + 8 * sb}};
  for (const K& k : ks) {
    cudaFuncAttributes a{};
    int occ = 0;
    if (cudaFuncGetAttributes(&a, k.f) != cudaSuccess) continue;
    (void)cudaFuncSetAttribute(k.f, cudaFuncAttributeMaxDynamicSharedMemorySize, k.smem);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k.f, 256, static_cast<std::size_t>(k.smem));
    std::fprintf(stderr, "kernel %-26s registers %3d, local %4zu B, static shared %5zu B, dynamic shared %6d B: %d blocks of 256 per SM\n", k.name, a.numRegs,
                 a.localSizeBytes, a.sharedSizeBytes, k.smem, occ);
  }
  (void)cudaGetLastError();
}


}  // namespace pelias::kern::gpu

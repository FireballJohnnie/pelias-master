// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/cuda/moe_dev.cuh - v52 (token kernel 3a): the device bodies of the MoE half of a layer (residual add, post norm,
// the 8-bit rounding, the router and shared-gate F32 products, the router's picks, the shared gate's sigmoid, the expert
// grouping with the reader cache, the combine) in one place, so the separate kernels of gpu_ops.cu and the fused kernel of
// gpu_stream.cu run the same source: the same operations in the same order, the same bits. Each body takes its block or
// element index from the caller; nothing here reads blockIdx or gridDim. Included by .cu files only.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kern/gpu_decode.h"
#include "kern/gpu_ops.h"
#include "kern/tg2_planes.h"

namespace pelias::kern::gpu::md {

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// block-wide double sum (blockDim.x a multiple of 32, at most 1024); every thread gets the result
__device__ __forceinline__ double block_sum_d(double v, double* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) sh[w] = v;
  __syncthreads();
  const int nw = static_cast<int>(blockDim.x) >> 5;
  double r = 0.0;
  if (threadIdx.x < 32) {
    r = static_cast<int>(threadIdx.x) < nw ? sh[threadIdx.x] : 0.0;
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (threadIdx.x == 0) sh[32] = r;
  }
  __syncthreads();
  r = sh[32];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float block_max_f(float v, float* sh) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) sh[w] = v;
  __syncthreads();
  const int nw = static_cast<int>(blockDim.x) >> 5;
  if (threadIdx.x < 32) {
    float r = static_cast<int>(threadIdx.x) < nw ? sh[threadIdx.x] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, o));
    if (threadIdx.x == 0) sh[32] = r;
  }
  __syncthreads();
  const float r = sh[32];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float silu_d(float x) { return quo(x, add(1.0f, expf(-x))); }
__device__ __forceinline__ float sigmoid_d(float x) { return quo(1.0f, add(1.0f, expf(-x))); }

// ------------------------------------------------------------------------------------------------ norm

// one segment of n values, the whole block (k_rms_norm's body): ld(i) gives value i (read twice: the sum, then the scale),
// st(i, v) stores the result
template <class Ld, class St>
__device__ __forceinline__ void rms_norm_seg(Ld ld, St st, const float* __restrict__ w, std::int64_t n, float eps, int l2, double* sh) {
  double part = 0.0;
  for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float xv = ld(i);
    part += static_cast<double>(mul(xv, xv));
  }
  const double sum = block_sum_d(part, sh);
  const float nf = static_cast<float>(n);
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  if (l2) {
    const float scale = quo(1.0f, sqrtf(add(mean, quo(eps, nf))));
    const float post = quo(1.0f, sqrtf(nf));
    for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) st(i, mul(mul(ld(i), scale), post));
  } else {
    const float scale = quo(1.0f, sqrtf(add(mean, eps)));
    for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) st(i, w ? mul(mul(ld(i), scale), w[i]) : mul(ld(i), scale));
  }
}

// ------------------------------------------------------------------------------------------------ 8-bit rounding

// one 32-value block t, b of k_act_q8x32: v holds the 32 inputs (rotated here when rot)
template <bool kHalves>
__device__ __forceinline__ void round32(float* v, std::int64_t t, std::int64_t b, std::int64_t n, std::int64_t rows, int rot, float norm, float* xr,
                                        std::int64_t ldr, std::int8_t* __restrict__ q, float* __restrict__ scale, int* __restrict__ sum) {
  const std::int64_t nb = n / 32;
  if (rot) {
    for (int h = 1; h < 32; h *= 2)
      for (int i = 0; i < 32; i += 2 * h)
        for (int j = i; j < i + h; ++j) {
          const float a = v[j], d = v[j + h];
          v[j] = add(a, d);
          v[j + h] = sub(a, d);
        }
    for (int j = 0; j < 32; ++j) v[j] = mul(v[j], norm);
  }
  float a = 0.0f;
  for (int j = 0; j < 32; ++j) a = fmaxf(a, fabsf(v[j]));
  const float sc = a > 0.0f ? quo(a, 127.0f) : 1.0f;
  const float inv = quo(1.0f, sc);
  int s = 0;
  for (int j = 0; j < 32; ++j) {
    const float qv = fminf(127.0f, fmaxf(-127.0f, rintf(mul(v[j], inv))));
    xr[t * ldr + 32 * b + j] = mul(sc, qv);
    if (q) q[t * n + 32 * b + j] = static_cast<std::int8_t>(qv);
    s += static_cast<int>(qv);
    if constexpr (kHalves) v[j] = sub(v[j], mul(sc, qv));  // the leftover, rounded below at sc / 256
  }
  if (scale) scale[t * nb + b] = sc;
  if (sum) sum[t * nb + b] = s;
  if constexpr (kHalves) {
    const float sl = quo(sc, 256.0f), il = quo(1.0f, sl);
    int s2 = 0;
    for (int j = 0; j < 32; ++j) {
      const float qv = fminf(127.0f, fmaxf(-127.0f, rintf(mul(v[j], il))));
      q[(t + rows) * n + 32 * b + j] = static_cast<std::int8_t>(qv);
      s2 += static_cast<int>(qv);
    }
    scale[(t + rows) * nb + b] = sl;
    sum[(t + rows) * nb + b] = s2;
  }
}

// ------------------------------------------------------------------------------------------------ F32 products (v52)

// rows of 32 padded to 36 floats in shared memory: the lanes' float4 reads hit distinct banks
constexpr int kGvPad = 36;

// x (n_tok rows of 32 x nsub values, row stride ldx, 16-byte aligned) into the padded staging, the whole block
__device__ __forceinline__ void gemv_stage(const float* __restrict__ x, int n_tok, int nsub, std::int64_t ldx, float* xs) {
  for (int idx = static_cast<int>(threadIdx.x); idx < n_tok * nsub * 8; idx += static_cast<int>(blockDim.x)) {
    const int t = idx / (nsub * 8), rem = idx - t * nsub * 8, j = rem >> 3, q4 = rem & 7;
    const float4 v = *reinterpret_cast<const float4*>(x + t * ldx + 32 * j + 4 * q4);
    *reinterpret_cast<float4*>(xs + (t * nsub + j) * kGvPad + 4 * q4) = v;
  }
}

// one warp: output row r of the F32 matrix w (row bytes rb; nsub sub-blocks of 32, at most 32 x NJ) for n_tok (1-4) staged rows: k_matmul's sums in its order (lane
// l: sub-blocks l, l + 32, ...; each a chain over its 32 values; then warp_sum), every weight load issued first
template <int NJ>
__device__ __forceinline__ void gemv_row(const float* __restrict__ w, std::uint64_t rb, std::int64_t r, const float* xs, int nsub, int n_tok,
                                         float* __restrict__ y, std::int64_t ldy) {
  const int lane = static_cast<int>(threadIdx.x & 31);
  const float4* p4 = reinterpret_cast<const float4*>(reinterpret_cast<const std::uint8_t*>(w) + static_cast<std::uint64_t>(r) * rb);
  float4 wv[NJ][8];
#pragma unroll
  for (int jj = 0; jj < NJ; ++jj)
#pragma unroll
    for (int i4 = 0; i4 < 8; ++i4) wv[jj][i4] = lane + 32 * jj < nsub ? p4[8 * (lane + 32 * jj) + i4] : float4{0.0f, 0.0f, 0.0f, 0.0f};
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int jj = 0; jj < NJ; ++jj) {
    const int j = lane + 32 * jj;
    if (j >= nsub) break;  // nsub < 32 (small rows): lanes past the last sub-block add nothing, as in k_matmul
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      if (tt >= n_tok) break;
      const float4* xr = reinterpret_cast<const float4*>(xs + (tt * nsub + j) * kGvPad);
      float s = 0.0f;
#pragma unroll
      for (int i4 = 0; i4 < 8; ++i4) {
        const float4 a = wv[jj][i4], b = xr[i4];
        s += a.x * b.x;
        s += a.y * b.y;
        s += a.z * b.z;
        s += a.w * b.w;
      }
      acc[tt] += s;
    }
  }
#pragma unroll
  for (int tt = 0; tt < 4; ++tt) {
    if (tt >= n_tok) break;
    const float v = warp_sum(acc[tt]);
    if (lane == 0) y[tt * ldy + r] = v;
  }
}

// ------------------------------------------------------------------------------------------------ route

constexpr int kRouterMaxExperts = 1024;
struct RouterSmem {
  float* p;               // kRouterMaxExperts
  float* key;             // kRouterMaxExperts
  unsigned char* taken;   // kRouterMaxExperts
  float* shm;             // 33
  float* inv_s;           // 1
  float* bv;              // 8
  int* bi;                // 8
};

// one token's picks, the whole block (k_router_fast's body, v22 / v24): the softmax element for element the reference's
// (block max, exp of the difference, the sum in double in index order by one thread, one multiply per element); a pick is
// the highest probability (or band key), the lower index on a tie, not picked before; the weights and their sum in pick
// order. sel / wsel point at this token's nu entries.
__device__ __forceinline__ void router_row(const float* __restrict__ l, std::int64_t ne, int nu, float wscale, std::int32_t* __restrict__ sel,
                                           float* __restrict__ wsel, float band, const RouterSmem& S) {
  float* p = S.p;
  float* key = S.key;
  unsigned char* taken = S.taken;
  const int tid = static_cast<int>(threadIdx.x), nth = static_cast<int>(blockDim.x);
  float m = -INFINITY;
  for (std::int64_t i = tid; i < ne; i += nth) m = fmaxf(m, l[i]);
  const float mx = block_max_f(m, S.shm);
  for (std::int64_t i = tid; i < ne; i += nth) {
    const float dl = sub(l[i], mx);
    p[i] = expf(dl);
    key[i] = band > 0.0f ? floorf(quo(dl, band)) : 0.0f;
    taken[i] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < ne; ++i) sum += static_cast<double>(p[i]);
    *S.inv_s = static_cast<float>(1.0 / sum);
  }
  __syncthreads();
  const float inv = *S.inv_s;
  for (std::int64_t i = tid; i < ne; i += nth) p[i] = mul(p[i], inv);
  __syncthreads();
  float tot = 0.0f;
  for (int kk = 0; kk < nu; ++kk) {
    float v = -INFINITY;
    int ix = 0x7fffffff;
    for (std::int64_t i = tid; i < ne; i += nth) {
      const float c = band > 0.0f ? key[i] : p[i];
      if (!taken[i] && (ix == 0x7fffffff || c > v)) {  // ascending i per thread: strict > keeps the lower index on a tie
        v = c;
        ix = static_cast<int>(i);
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
      const int i2 = __shfl_xor_sync(0xffffffffu, ix, o);
      if (i2 != 0x7fffffff && (ix == 0x7fffffff || v2 > v || (v2 == v && i2 < ix))) {
        v = v2;
        ix = i2;
      }
    }
    if ((tid & 31) == 0) {
      S.bv[tid >> 5] = v;
      S.bi[tid >> 5] = ix;
    }
    __syncthreads();
    if (tid == 0) {
      float bvv = S.bv[0];
      int bix = S.bi[0];
      for (int w = 1; w < (nth >> 5); ++w)
        if (S.bi[w] != 0x7fffffff && (bix == 0x7fffffff || S.bv[w] > bvv || (S.bv[w] == bvv && S.bi[w] < bix))) {
          bvv = S.bv[w];
          bix = S.bi[w];
        }
      sel[kk] = bix;
      wsel[kk] = p[bix];
      tot = add(tot, p[bix]);
      taken[bix] = 1;
    }
    __syncthreads();
  }
  if (tid == 0) {
    tot = fmaxf(tot, 6.103515625e-5f);
    for (int kk = 0; kk < nu; ++kk) {
      float w = quo(wsel[kk], tot);
      if (wscale != 0.0f && wscale != 1.0f) w = mul(w, wscale);
      wsel[kk] = w;
    }
  }
}

// ------------------------------------------------------------------------------------------------ v26 reader v3 grouping

// v51: the slot range of the pool expert e is cached in, its slot size and the pool's byte offset (np 0: one pool, v26)
__device__ __forceinline__ void rc_pool(const ReaderCache& c, int e, int* k0, int* k1, std::uint64_t* sb, std::uint64_t* boff) {
  if (c.np <= 0) {
    *k0 = 0;
    *k1 = c.K;
    *sb = c.slot_bytes;
    *boff = 0;
    return;
  }
  const int p = c.pool_of ? static_cast<int>(c.pool_of[e]) : -1;
  if (p < 0 || p >= c.np) {
    *k0 = 0;
    *k1 = 0;
    *sb = 0;
    *boff = 0;
    return;
  }
  *k0 = c.pk0[p];
  *k1 = c.pk0[p + 1];
  *sb = c.pbytes[p];
  *boff = c.pboff[p];
}

// one thread (k_rd3_group's body): group the routed picks by expert (first appearance), then the shared expert as one more
// group over every token; per group the source (bank / cache slot / file) and, on a cache miss, the slot its rows are
// written into
__device__ __forceinline__ void rd3_group_one(const ExpertPtr* __restrict__ tab, const ExpertPtr* __restrict__ shx, const std::int32_t* __restrict__ sel,
                                              int tokens, int nu, const ReaderCache& c, std::int64_t F, int* __restrict__ grp, ExpertPtr* __restrict__ gsrc,
                                              ExpertPtr* __restrict__ gfill, const std::uint8_t* __restrict__ cmask) {
  const int picks = tokens * nu;
  const unsigned stamp = c.K > 0 ? ++(*c.stamp) : 0u;
  int U = 0;
  for (int pi = 0; pi < picks; ++pi) {
    const int e = sel[pi];
    if (cmask && cmask[e]) continue;  // v28: the CPU's pick
    int u = 0;
    while (u < U && grp[1 + 6 * u] != e) ++u;
    if (u == U) {
      grp[1 + 6 * u] = e;
      grp[2 + 6 * u] = 0;
      ExpertPtr src = tab[e], fill;
      // v54b: an expert with plane matrices is read with the slider's bits; its slot holds those bits' prefix of each row
      const bool planed = src.type[0] == g_q4_tg2p || src.type[1] == g_q4_tg2p || src.type[2] == g_q4_tg2p;
      const int want = planed ? c.plane_bits : 4;
      src.pbits = static_cast<std::uint32_t>(want);
      if (c.K > 0 && !src.vram) {
        int hit = -1, victim = -1, k0 = 0, k1 = 0;
        std::uint64_t pslot = 0, poff = 0;
        rc_pool(c, e, &k0, &k1, &pslot, &poff);
        unsigned best = 0xffffffffu;
        for (int k = k0; k < k1; ++k) {
          if (c.tag[k] == e) hit = k;
          if (c.use[k] != stamp && c.use[k] < best) {
            best = c.use[k];
            victim = k;
          }
        }
        const int slot = hit >= 0 ? hit : victim;
        if (slot >= 0) {
          std::uint8_t* sb = c.base + poff + static_cast<std::uint64_t>(slot - k0) * pslot;
          // the slot's row bytes per matrix: a plane matrix's rows hold the first `want` bits only (packed)
          std::uint64_t srb[3];
          for (int pj = 0; pj < 3; ++pj) srb[pj] = src.type[pj] == g_q4_tg2p ? tg2p_prefix(src.rb[pj] / 144, want) : src.rb[pj];
          ExpertPtr at = src;
          at.w[0] = sb;
          at.w[1] = sb + srb[0] * static_cast<std::uint64_t>(F);
          at.w[2] = sb + (srb[0] + srb[1]) * static_cast<std::uint64_t>(F);
          for (int pj = 0; pj < 3; ++pj) at.rb[pj] = srb[pj];
          at.vram = 1;
          c.use[slot] = stamp;
          if (hit >= 0) {
            src = at;
          } else {
            c.tag[slot] = e;
            fill = at;
          }
        }
      }
      gsrc[u] = src;
      gfill[u] = fill;
      ++U;
    }
    const int cn = grp[2 + 6 * u];
    grp[3 + 6 * u + cn] = pi;
    grp[2 + 6 * u] = cn + 1;
  }
  if (shx) {  // the shared expert: picks picks .. picks + tokens - 1
    grp[1 + 6 * U] = -1;
    grp[2 + 6 * U] = tokens;
    for (int t = 0; t < tokens; ++t) grp[3 + 6 * U + t] = picks + t;
    gsrc[U] = *shx;
    gfill[U] = ExpertPtr{};
    ++U;
  }
  grp[0] = U;
}

// one output value t, r of k_rd3_combine: the picks' weighted rows in pick order, then the shared expert's
__device__ __forceinline__ float rd3_combine_one(const float* __restrict__ yd, const float* __restrict__ wsel, const float* __restrict__ gsh, int nu,
                                                 std::int64_t tokens, std::int64_t H, std::int64_t t, std::int64_t r,
                                                 const std::int32_t* __restrict__ sel, const std::uint8_t* __restrict__ cmask) {
  float tot = 0.0f;
  for (int j = 0; j < nu; ++j) {
    if (cmask && cmask[sel[t * nu + j]]) continue;  // v28: the CPU's pick, added by the caller
    tot = add(tot, mul(wsel[t * nu + j], yd[(t * nu + j) * H + r]));
  }
  if (gsh) tot = add(tot, mul(gsh[t], yd[(tokens * nu + t) * H + r]));
  return tot;
}

}  // namespace pelias::kern::gpu::md

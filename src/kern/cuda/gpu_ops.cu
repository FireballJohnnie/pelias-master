// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/cuda/gpu_ops.cu - GP3-10 GPU: the kernels behind kern/gpu_ops.h. Plain, correct first (rung 1); the speed forms (R3's
// streaming kernel, fused norm + int8 quantization, chunked DeltaNet for long prompts) build on these and are checked
// against them. Float expressions that the CPU reference evaluates in a fixed order are written in that order with
// mul / add / sub / quo (never fused), so those values match the reference bit for bit; the exceptions are named where they
// are: sums the GPU splits across threads (dot products, norms, attention) and the library functions expf / logf /
// sinf / cosf, which may differ from the host's by an ulp.
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "kern/cuda/moe_dev.cuh"
#include "kern/gpu_decode.h"
#include "kern/gpu_ops.h"
#include "kern/tg2_planes.h"
#include "quant/iq_tables.h"
#include "quant/quant_cpu.h"

namespace pelias::kern::gpu {
namespace {

thread_local std::string g_err;

bool fail(const std::string& why) {
  g_err = why;
  return false;
}

bool ck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  return fail(std::string(what) + ": " + cudaGetErrorString(e));
}

bool launched(const char* what) { return ck(cudaGetLastError(), what); }

cudaStream_t g_cur_stream = nullptr;  // v26: the stream a null Stream means (set_stream; the CUDA graph captures it)
cudaStream_t st(Stream s) { return s ? static_cast<cudaStream_t>(s) : g_cur_stream; }

unsigned blocks_for(std::int64_t n, int per) { return static_cast<unsigned>((n + per - 1) / per); }

// v52: the shared helpers live in moe_dev.cuh (the fused MoE half runs the same source)
using md::block_max_f;
using md::block_sum_d;
using md::sigmoid_d;
using md::silu_d;
using md::warp_sum;

// ------------------------------------------------------------------------------------------------ gemm_gguf

constexpr int kTT = 8;  // tokens per pass of a decoded sub-block

// one warp per output row; lane l decodes sub-blocks l, l + 32, ... and multiplies each with up to kTT token rows
__global__ void __launch_bounds__(128) k_matmul(std::uint32_t type, const std::uint8_t* __restrict__ w, std::uint64_t rb, std::int64_t n_out,
                                                std::int64_t k, const float* __restrict__ x, std::int64_t n_tok, std::int64_t ldx,
                                                float* __restrict__ y, std::int64_t ldy, const Tables tb) {
  const std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = static_cast<int>(threadIdx.x & 31);
  if (r >= n_out) return;
  const std::uint8_t* row = w + static_cast<std::uint64_t>(r) * rb;
  const std::int64_t nsub = k / 32;
  for (std::int64_t t0 = 0; t0 < n_tok; t0 += kTT) {
    float acc[kTT];
#pragma unroll
    for (int i = 0; i < kTT; ++i) acc[i] = 0.0f;
    for (std::int64_t j = lane; j < nsub; j += 32) {
      float wv[32];
      if (type == g_f32 && (reinterpret_cast<std::uintptr_t>(row) & 15u) == 0) {  // v26: float4 loads (the same floats as decode32)
        const float4* p4 = reinterpret_cast<const float4*>(row) + 8 * j;
#pragma unroll
        for (int i4 = 0; i4 < 8; ++i4) {
          const float4 v4 = p4[i4];
          wv[4 * i4] = v4.x;
          wv[4 * i4 + 1] = v4.y;
          wv[4 * i4 + 2] = v4.z;
          wv[4 * i4 + 3] = v4.w;
        }
      } else {
        decode32(type, row, j, tb, wv);
      }
#pragma unroll
      for (int tt = 0; tt < kTT; ++tt) {
        if (t0 + tt >= n_tok) break;
        const float* xr = x + (t0 + tt) * ldx + 32 * j;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) s += wv[i] * xr[i];
        acc[tt] += s;
      }
    }
#pragma unroll
    for (int tt = 0; tt < kTT; ++tt) {
      const float v = warp_sum(acc[tt]);
      if (lane == 0 && t0 + tt < n_tok) y[(t0 + tt) * ldy + r] = v;
    }
  }
}

// v52 (token kernel 3a, first piece): the F32 product for 1-4 tokens (the router, the shared gate). The same sums as
// k_matmul, in the same order (lane l: sub-blocks l, l + 32, ...; each a chain over its 32 values; then warp_sum), so the
// same bits. What changes is only where the values come from: x staged once per block in shared memory (rows of 32 padded
// to 36 floats: the lanes' float4 reads hit distinct banks), every weight load of the lane issued before the first sum,
// and two rows per block so the 256 router rows spread over 128 SMs.
using md::kGvPad;
template <int NJ>
__global__ void __launch_bounds__(64) k_gemv_f32(const float* __restrict__ w, std::uint64_t rb, std::int64_t n_out, std::int64_t k,
                                                 const float* __restrict__ x, int n_tok, std::int64_t ldx, float* __restrict__ y,
                                                 std::int64_t ldy) {
  extern __shared__ float4 s_gv4[];
  float* xs = reinterpret_cast<float*>(s_gv4);
  const int nsub = static_cast<int>(k / 32);
  md::gemv_stage(x, n_tok, nsub, ldx, xs);
  __syncthreads();
  const std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * 2 + threadIdx.x / 32;
  if (r >= n_out) return;
  md::gemv_row<NJ>(w, rb, r, xs, nsub, n_tok, y, ldy);
}

bool g_fast_f32 = true;  // v52 experiment switch (--no-fast-f32): removed once measured

// grouped experts: blockIdx.y = expert, one warp per output row of it; the expert's slots in passes of kTT
__global__ void __launch_bounds__(128) k_moe_matmul(std::uint32_t type, const std::uint8_t* __restrict__ w, std::uint64_t eb, std::uint64_t rb,
                                                    std::int64_t n_out, std::int64_t k, const int* __restrict__ off,
                                                    const int* __restrict__ src, const int* __restrict__ dst, const float* __restrict__ x,
                                                    std::int64_t ldx, float* __restrict__ y, std::int64_t ldy, const Tables tb) {
  const int e = static_cast<int>(blockIdx.y);
  const int s0 = off[e], s1 = off[e + 1];
  if (s1 <= s0) return;  // an expert nobody picked: no work, no reads
  const std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = static_cast<int>(threadIdx.x & 31);
  if (r >= n_out) return;
  const std::uint8_t* row = w + static_cast<std::uint64_t>(e) * eb + static_cast<std::uint64_t>(r) * rb;
  const std::int64_t nsub = k / 32;
  for (int c0 = s0; c0 < s1; c0 += kTT) {
    float acc[kTT];
#pragma unroll
    for (int i = 0; i < kTT; ++i) acc[i] = 0.0f;
    for (std::int64_t j = lane; j < nsub; j += 32) {
      float wv[32];
      decode32(type, row, j, tb, wv);
#pragma unroll
      for (int tt = 0; tt < kTT; ++tt) {
        if (c0 + tt >= s1) break;
        const float* xr = x + static_cast<std::int64_t>(src[c0 + tt]) * ldx + 32 * j;
        float sm = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) sm += wv[i] * xr[i];
        acc[tt] += sm;
      }
    }
#pragma unroll
    for (int tt = 0; tt < kTT; ++tt) {
      const float v = warp_sum(acc[tt]);
      if (lane == 0 && c0 + tt < s1) y[static_cast<std::int64_t>(dst[c0 + tt]) * ldy + r] = v;
    }
  }
}

__global__ void k_gather(std::uint32_t type, const std::uint8_t* __restrict__ w, std::uint64_t rb, std::int64_t k,
                         const std::int32_t* __restrict__ ids, float* __restrict__ out, const Tables tb) {
  const std::int64_t t = blockIdx.x;
  const std::uint8_t* row = w + static_cast<std::uint64_t>(ids[t]) * rb;
  for (std::int64_t j = threadIdx.x; j < k / 32; j += blockDim.x) {
    float v[32];
    decode32(type, row, j, tb, v);
    for (int i = 0; i < 32; ++i) out[t * k + 32 * j + i] = v[i];
  }
}

// ------------------------------------------------------------------------------------------------ activations

// one thread per 32-value block: the same loops as quant::hadamard32 and model::act_round_q8_256x8. v50 kHalves: also the
// leftover x - sc q rounded at sc / 256 (clamped to +-127) as rows rows .. 2 rows - 1 of q / scale / sum: hi + lo / 256 is an
// int16-level input (the R3 kernels run hi and lo as two rows on one load of the weights; their outputs add)
template <bool kHalves>
__global__ void k_act_q8x32(const float* __restrict__ x, std::int64_t n, std::int64_t rows, std::int64_t ldx, int rot, float norm,
                            float* xr, std::int64_t ldr, std::int8_t* __restrict__ q, float* __restrict__ scale, int* __restrict__ sum) {
  const std::int64_t nb = n / 32;
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= rows * nb) return;
  const std::int64_t t = id / nb, b = id % nb;
  float v[32];
  for (int i = 0; i < 32; ++i) v[i] = x[t * ldx + 32 * b + i];
  md::round32<kHalves>(v, t, b, n, rows, rot, norm, xr, ldr, q, scale, sum);
}

// ------------------------------------------------------------------------------------------------ elementwise

// one block (256 threads) per segment
__global__ void __launch_bounds__(256) k_rms_norm(const float* x, const float* __restrict__ w, float* y, std::int64_t n, std::int64_t n_seg,
                                                  std::int64_t stride, std::int64_t ld, float eps, int l2) {
  __shared__ double sh[33];
  const std::int64_t t = blockIdx.x / n_seg, g = blockIdx.x % n_seg;
  const float* xs = x + t * ld + g * stride;
  float* ys = y + t * ld + g * stride;
  md::rms_norm_seg([xs](std::int64_t i) { return xs[i]; }, [ys](std::int64_t i, float v) { ys[i] = v; }, w, n, eps, l2, sh);
}

// rms_norm + act_q8x32 fused: one block (256 threads) per row. The sum of squares exactly as k_rms_norm (same partial
// order, same block_sum_d); then each thread takes whole 32-value blocks (b = threadIdx.x, + blockDim.x), forms the
// normalized values in registers with k_rms_norm's expression and rounds them with k_act_q8x32's loops.
__global__ void __launch_bounds__(256) k_rms_norm_q8x32(const float* __restrict__ x, const float* __restrict__ w, std::int64_t n, std::int64_t ldx,
                                                        float eps, int rot, float norm, float* __restrict__ y, std::int64_t ldy, float* __restrict__ xr,
                                                        std::int64_t ldr, std::int8_t* __restrict__ q, float* __restrict__ scale,
                                                        int* __restrict__ sum) {
  __shared__ double sh[33];
  const std::int64_t t = blockIdx.x;
  const float* xs = x + t * ldx;
  double part = 0.0;
  for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) part += static_cast<double>(mul(xs[i], xs[i]));
  const double ssum = block_sum_d(part, sh);
  const float mean = static_cast<float>(ssum / static_cast<double>(n));
  const float rs = quo(1.0f, sqrtf(add(mean, eps)));
  const std::int64_t nb = n / 32;
  for (std::int64_t b = threadIdx.x; b < nb; b += blockDim.x) {
    float v[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const std::int64_t c = 32 * b + i;
      v[i] = w ? mul(mul(xs[c], rs), w[c]) : mul(xs[c], rs);
    }
    if (y) {
#pragma unroll
      for (int i = 0; i < 32; ++i) y[t * ldy + 32 * b + i] = v[i];
    }
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
    int si = 0;
    for (int j = 0; j < 32; ++j) {
      const float qv = fminf(127.0f, fmaxf(-127.0f, rintf(mul(v[j], inv))));
      if (xr) xr[t * ldr + 32 * b + j] = mul(sc, qv);
      if (q) q[t * n + 32 * b + j] = static_cast<std::int8_t>(qv);
      si += static_cast<int>(qv);
    }
    if (scale) scale[t * nb + b] = sc;
    if (sum) sum[t * nb + b] = si;
  }
}

// one thread per rotated pair; the angle grows by one multiply per pair, as model::rope_cache does
__global__ void k_rope(float* v, std::int64_t n_heads, std::int64_t hs, std::int64_t ld, std::int64_t rows, std::int64_t pos0_arg, int n_rot,
                       float theta_scale, int s0, int s1, int s2, int s3, int neox, const float* __restrict__ ff, const int* __restrict__ pos_d) {
  const std::int64_t pos0 = pos_d ? static_cast<std::int64_t>(*pos_d) : pos0_arg;  // v26: the position from the device (graph replay)
  const std::int64_t pairs = n_rot / 2;
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= rows * n_heads * pairs) return;
  const std::int64_t p = id % pairs, g = (id / pairs) % n_heads, t = id / (pairs * n_heads);
  const int sect = s0 + s1 + s2 + s3;
  float th = static_cast<float>(pos0 + t), the = 0.0f;
  for (std::int64_t i = 0; i < p; ++i) {
    th = mul(th, theta_scale);
    the = mul(the, theta_scale);
  }
  bool use_e = false;
  if (sect > 0) {
    const int sector = static_cast<int>(p % sect);
    if (sector % 3 == 1 && sector < 3 * s1) use_e = false;
    else if (sector % 3 == 2 && sector < 3 * s2) use_e = false;
    else if (sector % 3 == 0 && sector < 3 * s0) use_e = false;
    else use_e = true;
  }
  float theta = use_e ? the : th;
  if (ff) theta = quo(theta, ff[p]);
  const float c = cosf(theta), sn = sinf(theta);
  float* h = v + t * ld + g * hs;
  const std::int64_t a = neox ? p : 2 * p, b = neox ? p + pairs : 2 * p + 1;
  const float x0 = h[a], x1 = h[b];
  h[a] = sub(mul(x0, c), mul(x1, sn));
  h[b] = add(mul(x0, sn), mul(x1, c));
}

__global__ void k_silu_mul(const float* g, const float* u, float* out, std::int64_t n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = mul(silu_d(g[i]), u[i]);
}

__global__ void k_add_rows(float* y, std::int64_t ldy, const float* x, std::int64_t ldx, const float* a, std::int64_t n, std::int64_t rows) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= n * rows) return;
  const std::int64_t t = id / n, i = id % n;
  const float xv = x[t * ldx + i];
  y[t * ldy + i] = add(y[t * ldy + i], a ? mul(xv, a[t]) : xv);
}

// ------------------------------------------------------------------------------------------------ route

// one block per row; the softmax sum and the picks run in thread 0 in the reference's order (256 experts: a few us)
__global__ void k_router(const float* __restrict__ logits, std::int64_t ne, int nu, float wscale, std::int32_t* __restrict__ sel,
                         float* __restrict__ wsel, float* __restrict__ work) {
  __shared__ float shm[33];
  const std::int64_t t = blockIdx.x;
  const float* l = logits + t * ne;
  float* p = work + t * ne;
  float m = -INFINITY;
  for (std::int64_t i = threadIdx.x; i < ne; i += blockDim.x) m = fmaxf(m, l[i]);
  const float mx = block_max_f(m, shm);
  for (std::int64_t i = threadIdx.x; i < ne; i += blockDim.x) p[i] = expf(sub(l[i], mx));
  __syncthreads();
  if (threadIdx.x != 0) return;
  double sum = 0.0;
  for (std::int64_t i = 0; i < ne; ++i) sum += static_cast<double>(p[i]);
  const float inv = static_cast<float>(1.0 / sum);
  for (std::int64_t i = 0; i < ne; ++i) p[i] = mul(p[i], inv);
  float tot = 0.0f;
  for (int kk = 0; kk < nu; ++kk) {  // the next pick: highest probability, lower index on a tie, not picked before
    std::int32_t best = -1;
    for (std::int64_t i = 0; i < ne; ++i) {
      bool taken = false;
      for (int q = 0; q < kk; ++q) taken = taken || sel[t * nu + q] == static_cast<std::int32_t>(i);
      if (taken) continue;
      if (best < 0 || p[i] > p[best]) best = static_cast<std::int32_t>(i);
    }
    sel[t * nu + kk] = best;
    wsel[t * nu + kk] = p[best];
    tot = add(tot, p[best]);
  }
  tot = fmaxf(tot, 6.103515625e-5f);
  for (int kk = 0; kk < nu; ++kk) {
    float v = quo(wsel[t * nu + kk], tot);
    if (wscale != 0.0f && wscale != 1.0f) v = mul(v, wscale);
    wsel[t * nu + kk] = v;
  }
}

// v22 (the v21 per-operation timeline: router + top-k 0.325 ms per layer at one token, thread 0 walking global memory
// for the sum and 8 x 256 x 8 pick checks): the same arithmetic with the row in shared memory and each pick an argmax over
// the block. The softmax is element for element the reference's (block max, exp of the difference, the sum in double in
// index order by one thread, one multiply per element); a pick is the highest probability, the lower index on a tie, not
// picked before (the serial loop's rule); the weights and their sum in pick order as before. Same values bit for bit.
using md::kRouterMaxExperts;
__global__ void __launch_bounds__(256) k_router_fast(const float* __restrict__ logits, std::int64_t ne, int nu, float wscale,
                                                     std::int32_t* __restrict__ sel, float* __restrict__ wsel, float band) {
  __shared__ float shm[33];
  __shared__ float p[kRouterMaxExperts];
  __shared__ float key[kRouterMaxExperts];  // v24 tie rule: floor((logit - max) / band); unused when band == 0
  __shared__ unsigned char taken[kRouterMaxExperts];
  __shared__ float inv_s;
  __shared__ float bv[8];
  __shared__ int bi[8];
  const std::int64_t t = blockIdx.x;
  const md::RouterSmem S{p, key, taken, shm, &inv_s, bv, bi};
  md::router_row(logits + t * ne, ne, nu, wscale, sel + t * nu, wsel + t * nu, band, S);
}

// ------------------------------------------------------------------------------------------------ attention

// scores[t][h][u] = (q . k_u) * scale for u <= p0 + t; one warp per position
__global__ void __launch_bounds__(256) k_attn_scores(const float* __restrict__ q, std::int64_t qs, std::int64_t qh, const float* __restrict__ K,
                                                     std::int64_t kv_w, std::int64_t nh, std::int64_t group, std::int64_t hd, std::int64_t p0,
                                                     std::int64_t L, float scale, float* __restrict__ sc, const int* __restrict__ pos_d) {
  const std::int64_t h = blockIdx.y, t = blockIdx.z;
  const std::int64_t p0r = pos_d ? static_cast<std::int64_t>(*pos_d) : p0;  // v26
  const int lane = static_cast<int>(threadIdx.x & 31);
  const std::int64_t wstep = static_cast<std::int64_t>(gridDim.x) * (blockDim.x / 32);
  for (std::int64_t u = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; u <= p0r + t; u += wstep) {
    const float* qq = q + t * qs + h * qh;
    const float* kk = K + u * kv_w + (h / group) * hd;
    float d = 0.0f;
    for (std::int64_t i = lane; i < hd; i += 32) d += qq[i] * kk[i];
    d = warp_sum(d);
    if (lane == 0) sc[(t * nh + h) * L + u] = mul(d, scale);
  }
}

// softmax of each (t, h) row of scores over 0 .. p0 + t: max, exp, sum in double, times 1 / sum (model::softmax)
__global__ void __launch_bounds__(256) k_attn_softmax(float* __restrict__ sc, std::int64_t nh, std::int64_t p0, std::int64_t L, const int* __restrict__ pos_d) {
  __shared__ float shm[33];
  __shared__ double shd[33];
  const std::int64_t row = blockIdx.x, t = row / nh, len = (pos_d ? static_cast<std::int64_t>(*pos_d) : p0) + t + 1;
  float* s = sc + row * L;
  float m = -INFINITY;
  for (std::int64_t u = threadIdx.x; u < len; u += blockDim.x) m = fmaxf(m, s[u]);
  const float mx = block_max_f(m, shm);
  double part = 0.0;
  for (std::int64_t u = threadIdx.x; u < len; u += blockDim.x) {
    const float e = expf(sub(s[u], mx));
    s[u] = e;
    part += static_cast<double>(e);
  }
  const double sum = block_sum_d(part, shd);
  const float inv = static_cast<float>(1.0 / sum);
  for (std::int64_t u = threadIdx.x; u < len; u += blockDim.x) s[u] = mul(s[u], inv);
}

// out[t][h][i] = sum_u p_u v_u[i] (u in order), then times sigmoid(gate[i]); one thread per i
__global__ void k_attn_out(const float* __restrict__ sc, const float* __restrict__ V, std::int64_t kv_w, std::int64_t nh, std::int64_t group,
                           std::int64_t hd, std::int64_t p0, std::int64_t L, const float* __restrict__ q, std::int64_t qs, std::int64_t qh,
                           int gate, float* __restrict__ out, const int* __restrict__ pos_d) {
  const std::int64_t row = blockIdx.x, t = row / nh, h = row % nh, len = (pos_d ? static_cast<std::int64_t>(*pos_d) : p0) + t + 1;
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (i >= hd) return;
  const float* s = sc + row * L;
  const float* vv = V + (h / group) * hd + i;
  float o = 0.0f;
  for (std::int64_t u = 0; u < len; ++u) o = add(o, mul(s[u], vv[u * kv_w]));
  if (gate) o = mul(o, sigmoid_d(q[t * qs + h * qh + hd + i]));
  out[t * nh * hd + h * hd + i] = o;
}

// ------------------------------------------------------------------------------------------------ Gated DeltaNet

__global__ void k_dn_conv(const float* __restrict__ mixed, const float* __restrict__ tail, const float* __restrict__ w, std::int64_t n,
                          std::int64_t ch, std::int64_t taps, float* __restrict__ conv) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= n * ch) return;
  const std::int64_t t = id / ch, c = id % ch;
  float sum = 0.0f;
  for (std::int64_t kk = 0; kk < taps; ++kk) {
    const std::int64_t src = t + kk - (taps - 1);
    const float xv = src >= 0 ? mixed[src * ch + c] : (tail ? tail[(taps - 1 + src) * ch + c] : 0.0f);
    sum = add(sum, mul(xv, w[c * taps + kk]));
  }
  conv[id] = silu_d(sum);
}

__global__ void k_dn_tail(const float* __restrict__ mixed, const float* __restrict__ tail, std::int64_t n, std::int64_t ch, std::int64_t taps,
                          float* __restrict__ nt) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= (taps - 1) * ch) return;
  const std::int64_t r = id / ch, c = id % ch, src = n - (taps - 1) + r;
  nt[id] = src >= 0 ? mixed[src * ch + c] : (tail ? tail[(taps - 1 + src) * ch + c] : 0.0f);
}

__global__ void k_dn_gates(float* beta, float* alpha, const float* __restrict__ dt, const float* __restrict__ a, std::int64_t n, std::int64_t nv) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= n * nv) return;
  const std::int64_t h = id % nv;
  beta[id] = sigmoid_d(beta[id]);
  const float x = add(alpha[id], dt[h]);
  const float sp = x > 20.0f ? x : logf(add(1.0f, expf(x)));
  alpha[id] = mul(sp, a[h]);
}

// one block per value head, thread j owns state row j (shared memory, rows padded by one float: conflict-free columns)
__global__ void k_dn_recurrent(const float* __restrict__ conv, std::int64_t ch, const float* __restrict__ beta, const float* __restrict__ alpha,
                               float* __restrict__ state, std::int64_t n, std::int64_t nk, std::int64_t nv, std::int64_t ds, float qscale,
                               float* __restrict__ core) {
  extern __shared__ float smem[];
  const std::int64_t hv = blockIdx.x, hk = hv % nk, j = threadIdx.x, P = ds + 1;
  float* M = smem;
  float* qv = smem + ds * P;
  float* kv = qv + ds;
  float* vv = kv + ds;
  float* gst = state + hv * ds * ds;
  for (std::int64_t i = 0; i < ds; ++i) M[j * P + i] = gst[j * ds + i];
  const std::int64_t d_inner = nv * ds;
  for (std::int64_t t = 0; t < n; ++t) {
    const float* row = conv + t * ch;
    qv[j] = row[hk * ds + j];
    kv[j] = row[nk * ds + hk * ds + j];
    vv[j] = row[2 * nk * ds + hv * ds + j];
    __syncthreads();
    const float b = beta[t * nv + hv];
    const float decay = expf(alpha[t * nv + hv]);
    float sum = 0.0f;
    for (std::int64_t i = 0; i < ds; ++i) {
      const float m = mul(M[j * P + i], decay);
      M[j * P + i] = m;
      sum = add(sum, mul(m, kv[i]));
    }
    const float delta = mul(sub(vv[j], sum), b);
    float o = 0.0f;
    for (std::int64_t i = 0; i < ds; ++i) {
      const float m = add(M[j * P + i], mul(kv[i], delta));
      M[j * P + i] = m;
      o = add(o, mul(m, qv[i]));
    }
    core[t * d_inner + hv * ds + j] = mul(o, qscale);
    __syncthreads();
  }
  for (std::int64_t i = 0; i < ds; ++i) gst[j * ds + i] = M[j * P + i];
}

// chunked DeltaNet (see deltanet_chunked in kern/gpu_ops.h for the math): one block of 256 threads per value head walks
// the chunks in order with the head's state in shared memory; inside a chunk every step is parallel except the forward
// substitution, which runs per state column (column j only needs column j).
constexpr int kDnC = 16;     // tokens per chunk
constexpr int kDnMaxPairs = 16;  // (token, column) pairs per thread kept in registers: C x ds / 256 <= 16, so ds <= 256
__global__ void __launch_bounds__(256) k_dn_chunked(const float* __restrict__ conv, std::int64_t ch, const float* __restrict__ beta,
                                                    const float* __restrict__ alpha, float* __restrict__ state, std::int64_t n,
                                                    std::int64_t nk, std::int64_t nv, std::int64_t ds, float qscale, float* __restrict__ core) {
  extern __shared__ float smem[];
  constexpr int C = kDnC;
  const std::int64_t hv = blockIdx.x, hk = hv % nk, P = ds + 1;
  const int tid = static_cast<int>(threadIdx.x), nth = static_cast<int>(blockDim.x);
  float* S = smem;              // ds x P: S[j * P + i] = state M[j][i]
  float* Kc = S + ds * P;       // C x ds
  float* Qc = Kc + C * ds;      // C x ds (q, then e^l_t S0 q_t)
  float* Uc = Qc + C * ds;      // C x ds (v, then u)
  float* L = Uc + C * ds;       // C x C: beta_t e^(l_t - l_s) k_t . k_s, s < t
  float* Pm = L + C * C;        // C x C: e^(l_t - l_s) q_t . k_s, s <= t
  float* lg = Pm + C * C;       // C: cumulative log decay
  float* bt = lg + C;           // C: beta
  float* gst = state + hv * ds * ds;
  for (std::int64_t x = tid; x < ds * ds; x += nth) S[(x / ds) * P + x % ds] = gst[x];
  const std::int64_t d_inner = nv * ds, cds = C * ds;
  for (std::int64_t c0 = 0; c0 < n; c0 += C) {
    const int m = static_cast<int>(n - c0 < C ? n - c0 : C);
    for (std::int64_t x = tid; x < cds; x += nth) {
      const int t = static_cast<int>(x / ds);
      const std::int64_t i = x % ds;
      if (t < m) {
        const float* row = conv + (c0 + t) * ch;
        Qc[x] = row[hk * ds + i];
        Kc[x] = row[nk * ds + hk * ds + i];
        Uc[x] = row[2 * nk * ds + hv * ds + i];
      } else {
        Qc[x] = 0.0f;
        Kc[x] = 0.0f;
        Uc[x] = 0.0f;
      }
    }
    if (tid == 0) {
      float cum = 0.0f;
      for (int t = 0; t < C; ++t) {
        if (t < m) {
          cum += alpha[(c0 + t) * nv + hv];
          bt[t] = beta[(c0 + t) * nv + hv];
        } else {
          bt[t] = 0.0f;
        }
        lg[t] = cum;
      }
    }
    __syncthreads();
    for (int x = tid; x < C * C; x += nth) {
      const int t = x / C, s = x % C;
      float kk = 0.0f, qk = 0.0f;
      if (t < m && s <= t) {
        for (std::int64_t i = 0; i < ds; ++i) {
          kk += Kc[t * ds + i] * Kc[s * ds + i];
          qk += Qc[t * ds + i] * Kc[s * ds + i];
        }
      }
      const float w = (t < m && s <= t) ? expf(lg[t] - lg[s]) : 0.0f;
      L[x] = s < t ? bt[t] * w * kk : 0.0f;
      Pm[x] = w * qk;
    }
    // S0 k_t into u's right-hand side, S0 q_t kept in registers until every thread has read Qc
    float sq[kDnMaxPairs];
    int np = 0;
    for (std::int64_t x = tid; x < cds; x += nth, ++np) {
      const int t = static_cast<int>(x / ds);
      const std::int64_t j = x % ds;
      float sk = 0.0f, sqv = 0.0f;
      if (t < m) {
        for (std::int64_t i = 0; i < ds; ++i) {
          const float sv = S[j * P + i];
          sk += sv * Kc[t * ds + i];
          sqv += sv * Qc[t * ds + i];
        }
      }
      const float e = expf(lg[t]);
      sq[np] = e * sqv;
      Uc[x] = t < m ? bt[t] * (Uc[x] - e * sk) : 0.0f;
    }
    __syncthreads();
    np = 0;
    for (std::int64_t x = tid; x < cds; x += nth, ++np) Qc[x] = sq[np];
    __syncthreads();
    for (std::int64_t j = tid; j < ds; j += nth)  // forward substitution, one column per thread
      for (int t = 1; t < m; ++t) {
        float acc = Uc[t * ds + j];
        for (int s = 0; s < t; ++s) acc -= L[t * C + s] * Uc[s * ds + j];
        Uc[t * ds + j] = acc;
      }
    __syncthreads();
    for (std::int64_t x = tid; x < cds; x += nth) {  // outputs
      const int t = static_cast<int>(x / ds);
      if (t >= m) continue;
      const std::int64_t j = x % ds;
      float o = Qc[x];
      for (int s = 0; s <= t; ++s) o += Pm[t * C + s] * Uc[s * ds + j];
      core[(c0 + t) * d_inner + hv * ds + j] = o * qscale;
    }
    const float el = expf(lg[m - 1]);
    for (std::int64_t x = tid; x < ds * ds; x += nth) {  // the state at the chunk's end
      const std::int64_t j = x / ds, i = x % ds;
      float v = el * S[j * P + i];
      for (int s = 0; s < m; ++s) v += expf(lg[m - 1] - lg[s]) * Uc[s * ds + j] * Kc[s * ds + i];
      S[j * P + i] = v;
    }
    __syncthreads();
  }
  for (std::int64_t x = tid; x < ds * ds; x += nth) gst[x] = S[(x / ds) * P + x % ds];
}

// variant: the per-token recurrence with the state in registers (C2). Block = 32 state rows x tpr threads (tpr = ds / 32);
// thread l of a row owns columns c * tpr + l (c < 32), so neighbouring lanes read neighbouring q / k words (no bank conflicts).
__global__ void __launch_bounds__(1024) k_dn_rec_reg(const float* __restrict__ conv, std::int64_t ch, const float* __restrict__ beta,
                                                     const float* __restrict__ alpha, float* __restrict__ state, std::int64_t n, std::int64_t nk,
                                                     std::int64_t nv, std::int64_t ds, int tpr, float qscale, float* __restrict__ core) {
  extern __shared__ float smem[];
  float* qv = smem;
  float* kv = smem + ds;
  const std::int64_t hv = blockIdx.x, hk = hv % nk;
  const int tid = static_cast<int>(threadIdx.x), nth = static_cast<int>(blockDim.x);
  const int l = tid % tpr;
  const std::int64_t j = static_cast<std::int64_t>(blockIdx.y) * 32 + tid / tpr;  // state row (value index)
  float* grow = state + hv * ds * ds + j * ds;
  float M[32];
#pragma unroll
  for (int c = 0; c < 32; ++c) M[c] = grow[c * tpr + l];
  const std::int64_t d_inner = nv * ds;
  for (std::int64_t t = 0; t < n; ++t) {
    const float* row = conv + t * ch;
    for (std::int64_t i = tid; i < ds; i += nth) {
      qv[i] = row[hk * ds + i];
      kv[i] = row[nk * ds + hk * ds + i];
    }
    __syncthreads();
    const float decay = expf(alpha[t * nv + hv]), b = beta[t * nv + hv];
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      M[c] *= decay;
      sum += M[c] * kv[c * tpr + l];
    }
    for (int o = tpr / 2; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o, tpr);
    const float delta = (row[2 * nk * ds + hv * ds + j] - sum) * b;
    float out = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      M[c] += kv[c * tpr + l] * delta;
      out += M[c] * qv[c * tpr + l];
    }
    for (int o = tpr / 2; o > 0; o >>= 1) out += __shfl_xor_sync(0xffffffffu, out, o, tpr);
    if (l == 0) core[t * d_inner + hv * ds + j] = out * qscale;
    __syncthreads();
  }
#pragma unroll
  for (int c = 0; c < 32; ++c) grow[c * tpr + l] = M[c];
}

// GP3-14 (author 2026-10-31 03:1x, MTP rollback): k_dn_rec_reg with the state after each token t < n - 1 also written to
// ckpt + t * nv * ds * ds (same layout as the state). The arithmetic is k_dn_rec_reg's, unchanged, so core and the final
// state are bit for bit the same; a verify batch can then keep any prefix of its tokens without running them again.
__global__ void __launch_bounds__(1024) k_dn_rec_reg_ck(const float* __restrict__ conv, std::int64_t ch, const float* __restrict__ beta,
                                                     const float* __restrict__ alpha, float* __restrict__ state, std::int64_t n, std::int64_t nk,
                                                     std::int64_t nv, std::int64_t ds, int tpr, float qscale, float* __restrict__ core,
                                                        float* __restrict__ ckpt) {
  extern __shared__ float smem[];
  float* qv = smem;
  float* kv = smem + ds;
  const std::int64_t hv = blockIdx.x, hk = hv % nk;
  const int tid = static_cast<int>(threadIdx.x), nth = static_cast<int>(blockDim.x);
  const int l = tid % tpr;
  const std::int64_t j = static_cast<std::int64_t>(blockIdx.y) * 32 + tid / tpr;  // state row (value index)
  float* grow = state + hv * ds * ds + j * ds;
  float M[32];
#pragma unroll
  for (int c = 0; c < 32; ++c) M[c] = grow[c * tpr + l];
  const std::int64_t d_inner = nv * ds;
  for (std::int64_t t = 0; t < n; ++t) {
    const float* row = conv + t * ch;
    for (std::int64_t i = tid; i < ds; i += nth) {
      qv[i] = row[hk * ds + i];
      kv[i] = row[nk * ds + hk * ds + i];
    }
    __syncthreads();
    const float decay = expf(alpha[t * nv + hv]), b = beta[t * nv + hv];
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      M[c] *= decay;
      sum += M[c] * kv[c * tpr + l];
    }
    for (int o = tpr / 2; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o, tpr);
    const float delta = (row[2 * nk * ds + hv * ds + j] - sum) * b;
    float out = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      M[c] += kv[c * tpr + l] * delta;
      out += M[c] * qv[c * tpr + l];
    }
    for (int o = tpr / 2; o > 0; o >>= 1) out += __shfl_xor_sync(0xffffffffu, out, o, tpr);
    if (l == 0) core[t * d_inner + hv * ds + j] = out * qscale;
    if (t + 1 < n) {
      float* crow = ckpt + t * nv * ds * ds + hv * ds * ds + j * ds;
#pragma unroll
      for (int c = 0; c < 32; ++c) crow[c * tpr + l] = M[c];
    }
    __syncthreads();
  }
#pragma unroll
  for (int c = 0; c < 32; ++c) grow[c * tpr + l] = M[c];
}

// variant: chunked, each head split over ds / 32 blocks of 32 state rows (C2); decays once per chunk
constexpr int kDnRows = 32;
template <int C>
__global__ void __launch_bounds__(256) k_dn_chunked_rows(const float* __restrict__ conv, std::int64_t ch, const float* __restrict__ beta,
                                                         const float* __restrict__ alpha, float* __restrict__ state, std::int64_t n, std::int64_t nk,
                                                         std::int64_t nv, std::int64_t ds, float qscale, float* __restrict__ core) {
  extern __shared__ float smem[];
  constexpr int R = kDnRows;
  const std::int64_t hv = blockIdx.x, hk = hv % nk, P = ds + 1, j0 = static_cast<std::int64_t>(blockIdx.y) * R;
  const int tid = static_cast<int>(threadIdx.x), nth = static_cast<int>(blockDim.x);
  float* S = smem;            // R x P: rows j0 .. j0 + R - 1 of the head's state
  float* Kc = S + R * P;      // C x ds
  float* Qc = Kc + C * ds;    // C x ds
  float* Uc = Qc + C * ds;    // C x R (v, then u)
  float* Oc = Uc + C * R;     // C x R: e^l_t S0 q_t
  float* L = Oc + C * R;      // C x C
  float* Pm = L + C * C;      // C x C
  float* lg = Pm + C * C;     // C
  float* bt = lg + C;         // C
  float* ew = bt + C;         // C: e^(l_last - l_s)
  float* et = ew + C;         // C: e^l_t
  float* gst = state + hv * ds * ds;
  for (std::int64_t x = tid; x < R * ds; x += nth) S[(x / ds) * P + x % ds] = gst[(j0 + x / ds) * ds + x % ds];
  const std::int64_t d_inner = nv * ds, cds = C * ds;
  for (std::int64_t c0 = 0; c0 < n; c0 += C) {
    const int m = static_cast<int>(n - c0 < C ? n - c0 : C);
    for (std::int64_t x = tid; x < cds; x += nth) {
      const int t = static_cast<int>(x / ds);
      const std::int64_t i = x % ds;
      const float* row = conv + (c0 + t) * ch;
      Qc[x] = t < m ? row[hk * ds + i] : 0.0f;
      Kc[x] = t < m ? row[nk * ds + hk * ds + i] : 0.0f;
    }
    for (int x = tid; x < C * R; x += nth) {
      const int t = x / R, jl = x % R;
      Uc[x] = t < m ? conv[(c0 + t) * ch + 2 * nk * ds + hv * ds + j0 + jl] : 0.0f;
    }
    if (tid == 0) {
      float cum = 0.0f;
      for (int t = 0; t < C; ++t) {
        if (t < m) {
          cum += alpha[(c0 + t) * nv + hv];
          bt[t] = beta[(c0 + t) * nv + hv];
        } else {
          bt[t] = 0.0f;
        }
        lg[t] = cum;
      }
    }
    __syncthreads();
    if (tid < C) {
      ew[tid] = tid < m ? expf(lg[m - 1] - lg[tid]) : 0.0f;
      et[tid] = expf(lg[tid]);
    }
    for (int x = tid; x < C * C; x += nth) {
      const int t = x / C, s = x % C;
      float kk = 0.0f, qk = 0.0f;
      if (t < m && s <= t) {
        for (std::int64_t i = 0; i < ds; ++i) {
          kk += Kc[t * ds + i] * Kc[s * ds + i];
          qk += Qc[t * ds + i] * Kc[s * ds + i];
        }
      }
      const float w = (t < m && s <= t) ? expf(lg[t] - lg[s]) : 0.0f;
      L[x] = s < t ? bt[t] * w * kk : 0.0f;
      Pm[x] = w * qk;
    }
    __syncthreads();
    for (int x = tid; x < C * R; x += nth) {  // S0 k_t, S0 q_t for this block's rows
      const int t = x / R, jl = x % R;
      float sk = 0.0f, sqv = 0.0f;
      if (t < m) {
        for (std::int64_t i = 0; i < ds; ++i) {
          const float sv = S[jl * P + i];
          sk += sv * Kc[t * ds + i];
          sqv += sv * Qc[t * ds + i];
        }
      }
      Oc[x] = et[t] * sqv;
      Uc[x] = t < m ? bt[t] * (Uc[x] - et[t] * sk) : 0.0f;
    }
    __syncthreads();
    if (tid < R)  // forward substitution, one column per thread
      for (int t = 1; t < m; ++t) {
        float acc = Uc[t * R + tid];
        for (int s = 0; s < t; ++s) acc -= L[t * C + s] * Uc[s * R + tid];
        Uc[t * R + tid] = acc;
      }
    __syncthreads();
    for (int x = tid; x < C * R; x += nth) {
      const int t = x / R, jl = x % R;
      if (t >= m) continue;
      float o = Oc[x];
      for (int s = 0; s <= t; ++s) o += Pm[t * C + s] * Uc[s * R + jl];
      core[(c0 + t) * d_inner + hv * ds + j0 + jl] = o * qscale;
    }
    const float el = et[m - 1];
    for (std::int64_t x = tid; x < R * ds; x += nth) {
      const std::int64_t jl = x / ds, i = x % ds;
      float v = el * S[jl * P + i];
      for (int s = 0; s < m; ++s) v += ew[s] * Uc[s * R + jl] * Kc[s * ds + i];
      S[jl * P + i] = v;
    }
    __syncthreads();
  }
  for (std::int64_t x = tid; x < R * ds; x += nth) gst[(j0 + x / ds) * ds + x % ds] = S[(x / ds) * P + x % ds];
}

__global__ void __launch_bounds__(128) k_gated_norm(float* core, const float* __restrict__ z, const float* __restrict__ w, std::int64_t nv,
                                                    std::int64_t ds, float eps) {
  __shared__ double sh[33];
  const std::int64_t seg = blockIdx.x;  // t * nv + h
  float* o = core + seg * ds;
  const float* zz = z + seg * ds;
  double part = 0.0;
  for (std::int64_t i = threadIdx.x; i < ds; i += blockDim.x) part += static_cast<double>(mul(o[i], o[i]));
  const double sum = block_sum_d(part, sh);
  const float mean = static_cast<float>(sum / static_cast<double>(ds));
  const float scale = quo(1.0f, sqrtf(add(mean, eps)));
  for (std::int64_t i = threadIdx.x; i < ds; i += blockDim.x) o[i] = mul(mul(mul(o[i], scale), w[i]), silu_d(zz[i]));
  (void)nv;
}

bool type_ok(std::uint32_t type, std::int64_t k, std::uint64_t row_bytes) {
  if (!type_supported(type)) return fail("type " + std::to_string(type) + " has no GPU decoder (CPU path)");
  const std::uint32_t bv = block_values(type);
  if (k <= 0 || k % 32 != 0 || k % static_cast<std::int64_t>(bv) != 0) return fail("row length " + std::to_string(k) + " is not whole blocks");
  if (row_bytes != static_cast<std::uint64_t>(k / bv) * block_bytes(type)) return fail("row_bytes does not match the type and row length");
  return true;
}

}  // namespace

bool compiled() { return true; }

bool available() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    cudaGetLastError();
    return fail("no CUDA device");
  }
  return true;
}

const char* last_error() { return g_err.c_str(); }

void* dalloc(std::size_t bytes) {
  void* p = nullptr;
  if (!ck(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc")) return nullptr;
  return p;
}
void dfree(void* p) {
  if (p) cudaFree(p);
}
bool to_device(void* dst, const void* src, std::size_t bytes, Stream s) {
  return ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st(s)), "copy to the device") && sync(s);
}
bool to_host(void* dst, const void* src, std::size_t bytes, Stream s) {
  return ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st(s)), "copy to the host") && sync(s);
}
bool dzero(void* p, std::size_t bytes, Stream s) { return ck(cudaMemsetAsync(p, 0, bytes, st(s)), "cudaMemset"); }
bool sync(Stream s) { return ck(cudaStreamSynchronize(st(s)), "synchronize"); }
bool set_host_wait(int mode, std::string* why) {
  const unsigned f = mode == 1 ? cudaDeviceScheduleSpin : mode == 2 ? cudaDeviceScheduleYield : mode == 3 ? cudaDeviceScheduleBlockingSync : cudaDeviceScheduleAuto;
  const cudaError_t e = cudaSetDeviceFlags(f);
  if (e != cudaSuccess) {
    if (why) *why = cudaGetErrorString(e);
    (void)cudaGetLastError();
    return false;
  }
  return true;
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__global__ void k_bell_ring(volatile unsigned* f, unsigned v, volatile unsigned long long* ts) {
  __threadfence_system();
  if (ts) ts[0] = gtimer();
  *f = v;
  __threadfence_system();
}
__global__ void k_bell_wait(const volatile unsigned* f, unsigned v, volatile unsigned long long* ts) {
  if (ts) ts[0] = gtimer();
  unsigned ns = 64;
  while (static_cast<int>(*f - v) < 0) {
    __nanosleep(ns);
    if (ns < 200) ns += 32;
  }
  if (ts) ts[1] = gtimer();
  __threadfence_system();
}
void* host_alloc_mapped(std::size_t bytes, void** dev) {
  void* h = nullptr;
  if (!ck(cudaHostAlloc(&h, bytes ? bytes : 1, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc (mapped)")) return nullptr;
  std::memset(h, 0, bytes ? bytes : 1);
  void* d = nullptr;
  if (!ck(cudaHostGetDevicePointer(&d, h, 0), "cudaHostGetDevicePointer")) {
    cudaFreeHost(h);
    return nullptr;
  }
  if (dev) *dev = d;
  return h;
}
void host_free(void* p) {
  if (p) cudaFreeHost(p);
}
bool bell_ring(unsigned* flag, unsigned value, Stream s, unsigned long long* ts) {
  if (!flag) return fail("bell_ring: no flag");
  k_bell_ring<<<1, 1, 0, st(s)>>>(flag, value, ts);
  return launched("doorbell ring");
}
bool bell_wait(const unsigned* flag, unsigned value, Stream s, unsigned long long* ts) {
  if (!flag) return fail("bell_wait: no flag");
  k_bell_wait<<<1, 1, 0, st(s)>>>(flag, value, ts);
  return launched("doorbell wait");
}
__global__ void k_bell_ring_dev(volatile unsigned* f, const unsigned* gen, unsigned add) {
  __threadfence_system();
  *f = *gen + add;
  __threadfence_system();
}
__global__ void k_bell_wait_dev(const volatile unsigned* f, const unsigned* gen, unsigned add) {
  const unsigned v = *gen + add;
  unsigned ns = 64;
  while (static_cast<int>(*f - v) < 0) {
    __nanosleep(ns);
    if (ns < 200) ns += 32;
  }
  __threadfence_system();
}
bool bell_ring_dev(unsigned* flag, const unsigned* gen, unsigned add, Stream s) {
  if (!flag || !gen) return fail("bell_ring_dev: no flag or generation");
  k_bell_ring_dev<<<1, 1, 0, st(s)>>>(flag, gen, add);
  return launched("doorbell ring (device generation)");
}
bool bell_wait_dev(const unsigned* flag, const unsigned* gen, unsigned add, Stream s) {
  if (!flag || !gen) return fail("bell_wait_dev: no flag or generation");
  k_bell_wait_dev<<<1, 1, 0, st(s)>>>(flag, gen, add);
  return launched("doorbell wait (device generation)");
}
__global__ void k_stamp(volatile unsigned long long* p) { *p = gtimer(); }
// v23 pin probe: register existing host memory (e.g. a read-only file mapping) for GPU access, and a read-rate kernel
bool host_register(void* p, std::size_t bytes, void** dev, std::string* why) {
  cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
  if (e != cudaSuccess) {
    if (why) *why = std::string("cudaHostRegister (mapped, read-only): ") + cudaGetErrorString(e);
    cudaGetLastError();
    e = cudaHostRegister(p, bytes, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
      if (why) *why += std::string("; cudaHostRegister (mapped): ") + cudaGetErrorString(e);
      cudaGetLastError();
      return false;
    }
  }
  void* d = nullptr;
  if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
    if (why) *why += "; cudaHostGetDevicePointer failed";
    cudaGetLastError();
    cudaHostUnregister(p);
    return false;
  }
  if (dev) *dev = d;
  return true;
}
void host_unregister(void* p) {
  if (p) cudaHostUnregister(p);
  cudaGetLastError();
}
__global__ void k_read_sum(const uint4* __restrict__ p, std::size_t n, unsigned* out) {
  unsigned acc = 0;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1u);  // keeps the reads
}
// v24 scattered reads: block b reads chunk b (`chunk` bytes at off[b] from base), as one routed expert per block
__global__ void k_read_chunks(const std::uint8_t* __restrict__ base, const std::uint64_t* __restrict__ off, std::size_t chunk, unsigned* out) {
  const uint4* p = reinterpret_cast<const uint4*>(base + off[blockIdx.x]);
  unsigned acc = 0;
  for (std::size_t i = threadIdx.x; i < chunk / 16; i += blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1u);
}
bool read_chunks_rate(const void* dev_base, const std::uint64_t* offsets, int n, std::size_t chunk, double* seconds) {
  std::uint64_t* doff = static_cast<std::uint64_t*>(dalloc(static_cast<std::size_t>(n) * 8));
  unsigned* o = static_cast<unsigned*>(dalloc(sizeof(unsigned)));
  if (!doff || !o) return false;
  bool ok = to_device(doff, offsets, static_cast<std::size_t>(n) * 8) && sync();
  const auto t0 = std::chrono::steady_clock::now();
  k_read_chunks<<<static_cast<unsigned>(n), 256>>>(static_cast<const std::uint8_t*>(dev_base), doff, chunk, o);
  ok = ok && launched("read chunks") && sync();
  *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  dfree(doff);
  dfree(o);
  return ok;
}

// ------------------------------------------------------------------------------------------------ expert reader helpers
// v58 (bloat audit): the v24 / v25 readers (moe_reader_v24, moe_reader) removed; reader v3 below is the engine's
bool device_tables(Tables* out);  // defined below
// silu x up rounded per 32 (the CPU cold path's rule), one thread per 32 values
__global__ void k_rd_silu_round(const float* __restrict__ gu, std::int64_t picks, std::int64_t F, float* __restrict__ h) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x, nb = F / 32;
  if (id >= picks * nb) return;
  const std::int64_t pi = id / nb, b = id % nb;
  const float* g = gu + pi * 2 * F + 32 * b;
  const float* u = gu + (pi * 2 + 1) * F + 32 * b;
  float v[32];
  float a = 0.0f;
  for (int i = 0; i < 32; ++i) {
    v[i] = mul(silu_d(g[i]), u[i]);
    a = fmaxf(a, fabsf(v[i]));
  }
  const float sc = a > 0.0f ? quo(a, 127.0f) : 1.0f;
  const float inv = quo(1.0f, sc);
  for (int i = 0; i < 32; ++i) h[pi * F + 32 * b + i] = mul(sc, fminf(127.0f, fmaxf(-127.0f, rintf(mul(v[i], inv)))));
}
constexpr int kRdMaxTok = 4;

// ------------------------------------------------------------------------------------------------ v26 reader v3
using md::rc_pool;
__global__ void k_rd3_group(const ExpertPtr* __restrict__ tab, const ExpertPtr* __restrict__ shx, const std::int32_t* __restrict__ sel, int tokens,
                            int nu, ReaderCache c, std::int64_t H, std::int64_t F, int* __restrict__ grp, ExpertPtr* __restrict__ gsrc,
                            ExpertPtr* __restrict__ gfill, const std::uint8_t* __restrict__ cmask) {
  if (blockIdx.x != 0) return;
  md::rd3_group_one<true>(tab, shx, sel, tokens, nu, c, F, grp, gsrc, gfill, cmask);  // v55e: the launch's one warp (<<<1, 32>>>)
  (void)H;
}
// a group of `glanes` lanes (lane index gl within it) copies n bytes from src into a 16-byte aligned buffer (alignment of
// src kept, the middle as uint4) and, if fill, into fill; returns the row in the buffer
__device__ __forceinline__ const std::uint8_t* rd3_stage(const std::uint8_t* src, std::uint64_t n, std::uint8_t* buf, std::uint8_t* fill, int gl,
                                                         int glanes) {
  const std::uint64_t mis = reinterpret_cast<std::uintptr_t>(src) & 15u;
  std::uint8_t* dst = buf + mis;
  std::uint64_t head = (16u - mis) & 15u;
  if (head > n) head = n;
  const std::uint64_t L = static_cast<std::uint64_t>(glanes), l0 = static_cast<std::uint64_t>(gl);
  for (std::uint64_t i = l0; i < head; i += L) dst[i] = src[i];
  const std::uint64_t n4 = (n - head) / 16;
  const uint4* s4 = reinterpret_cast<const uint4*>(src + head);
  uint4* d4 = reinterpret_cast<uint4*>(dst + head);
  for (std::uint64_t i = l0; i < n4; i += L) d4[i] = s4[i];
  for (std::uint64_t i = head + n4 * 16 + l0; i < n; i += L) dst[i] = src[i];
  __syncwarp();
  if (fill)
    for (std::uint64_t i = l0; i < n; i += L) fill[i] = dst[i];
  return dst;
}
// 32 values of block j of a staged row: Q4_TG2 from the lane's column of the replicated grid (decode32's floats), else decode32
__device__ __forceinline__ void rd3_decode(std::uint32_t type, const std::uint8_t* row, std::int64_t j, const Tables& tb, const float* lut, float* wv) {
  if (type == g_q4_tg2) {
    const std::uint8_t* b = row + 144 * (j / 8);
    const int js = static_cast<int>(j % 8);
    const float d = half_float(rd16(b));
    std::uint64_t packed = 0;
    for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
    const float s = quo(mul(d, static_cast<float>((packed >> (6 * js)) & 63u)), 63.0f);
    const unsigned tg = b[8 + js];
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const float* lr = lut + (threadIdx.x & 31) + 32 * 16 * ((tg >> (2 * g)) & 3u);  // the lane's own column: no bank conflicts
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int idx = 32 * js + 8 * g + k;
        wv[8 * g + k] = mul(s, lr[32 * ((b[16 + idx / 2] >> (4 * (idx & 1))) & 15)]);
      }
    }
    return;
  }
  decode32(type, row, j, tb, wv);
}
__device__ __forceinline__ void rd3_lut(const Tables& tb, float* lut, float* lut2) {
  // the grid per lane (bank = lane); lut2 (v27 plane test): each code's value replaced by the mean of the four codes that
  // share its top two bits (a 2-bit base plane of the grid)
  for (int i = static_cast<int>(threadIdx.x); i < 2048; i += static_cast<int>(blockDim.x)) {
    lut[i] = tb.tg2[i >> 5];
    if (lut2) {
      const int en = i >> 5, base = en & ~3;
      lut2[i] = (tb.tg2[base] + tb.tg2[base + 1] + tb.tg2[base + 2] + tb.tg2[base + 3]) * 0.25f;
    }
  }
  __syncthreads();
}
// v27: the reader v3 kernels loop over row pairs (grid-stride), so each block builds its shared grid once for many rows
// (v26 built it per 8 rows: ~28 MB of shared stores per layer), and only when the group's matrices are Q4_TG2. tail > 0
// (the plane test): routed picks of rank >= n_used - tail use the 2-bit base plane of Q4_TG2 rows (lut2).
__device__ __forceinline__ bool rd3_tail_pick(int pk, int picks_routed, int nu, int tail) { return tail > 0 && pk < picks_routed && pk % nu >= nu - tail; }
// gate and up: a warp takes two rows (r0, r0 + 1 of the 2F gate|up rows) of one group, stages both, then decodes.
// v46 step 1 (v29 MEASURED: the plane-test code in these loops cost ~1.3 ms per token even when off): kTail is a template
// parameter, so the plain instantiation carries no second decode buffer, rank check or second grid.
template <bool kTail>
__global__ void __launch_bounds__(128) k_rd3_gate_up(const ExpertPtr* __restrict__ gsrc, const ExpertPtr* __restrict__ gfill, const int* __restrict__ grp,
                                                     int nu, int picks_routed, const float* __restrict__ xr, std::int64_t H, std::int64_t F,
                                                     std::uint64_t slot, float* __restrict__ gu, const Tables tb, int tail, int lut_bytes, int rpw) {
  extern __shared__ __align__(16) std::uint8_t rd3_sh[];
  const int u = static_cast<int>(blockIdx.y);
  if (u >= grp[0]) return;
  const int* G = grp + 1 + 6 * u;
  const int cnt = G[1];
  const ExpertPtr& X = gsrc[u];
  const ExpertPtr& FL = gfill[u];
  bool any_tail = false;
  if (kTail)
    for (int c = 0; c < cnt; ++c) any_tail = any_tail || rd3_tail_pick(G[2 + c], picks_routed, nu, tail);
  const bool tg = X.type[0] == g_q4_tg2 || X.type[1] == g_q4_tg2;
  float* lut = reinterpret_cast<float*>(rd3_sh);
  float* lut2 = any_tail ? lut + 2048 : nullptr;
  if (tg) rd3_lut(tb, lut, lut2);
  std::uint8_t* stage0 = rd3_sh + lut_bytes;
  const int warp = static_cast<int>(threadIdx.x >> 5), lane = static_cast<int>(threadIdx.x & 31);
  std::uint8_t* buf = stage0 + slot * static_cast<std::uint64_t>(2 * warp);
  for (std::int64_t chunk = blockIdx.x; chunk * 4 * rpw < 2 * F; chunk += gridDim.x) {
    const std::int64_t r0 = (chunk * 4 + warp) * rpw;
    if (r0 >= 2 * F) continue;
    const std::uint8_t* rows[2] = {nullptr, nullptr};
    int pjs[2] = {0, 0};
    std::int64_t rrs[2] = {0, 0};
    for (int q = 0; q < rpw; ++q) {
      const std::int64_t r = r0 + q;
      if (r >= 2 * F) break;
      const int pj = r < F ? 0 : 1;
      const std::int64_t rr = r < F ? r : r - F;
      pjs[q] = pj;
      rrs[q] = rr;
      const std::uint64_t off = static_cast<std::uint64_t>(rr) * X.rb[pj];
      rows[q] = rd3_stage(X.w[pj] + off, X.rb[pj], buf + slot * static_cast<std::uint64_t>(q), FL.w[pj] ? const_cast<std::uint8_t*>(FL.w[pj]) + off : nullptr,
                          lane, 32);
    }
    for (int q = 0; q < rpw; ++q) {
      if (!rows[q]) break;
      const int pj = pjs[q];
      const bool shadow = kTail && any_tail && X.type[pj] == g_q4_tg2;
      float acc[kRdMaxTok] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (std::int64_t j = lane; j < H / 32; j += 32) {
        float wv[32], wv2[32];
        rd3_decode(X.type[pj], rows[q], j, tb, lut, wv);
        if (shadow) rd3_decode(X.type[pj], rows[q], j, tb, lut2, wv2);
        for (int c = 0; c < cnt; ++c) {
          const int pk = G[2 + c];
          const int tok = pk < picks_routed ? pk / nu : pk - picks_routed;
          const float* x = xr + tok * H + 32 * j;
          const float* w = (shadow && rd3_tail_pick(pk, picks_routed, nu, tail)) ? wv2 : wv;
          float s = 0.0f;
#pragma unroll
          for (int i = 0; i < 32; ++i) s += w[i] * x[i];
          acc[c] += s;
        }
      }
      for (int c = 0; c < cnt; ++c) {
        const float a = warp_sum(acc[c]);
        if (lane == 0) gu[(static_cast<std::int64_t>(G[2 + c]) * 2 + pj) * F + rrs[q]] = a;
      }
    }
    __syncwarp();
  }
}
// down: rows of F values; with F / 32 <= 16 a half-warp per row (two rows per warp), else a warp per row (two in turn)
template <bool kTail>
__global__ void __launch_bounds__(128) k_rd3_down(const ExpertPtr* __restrict__ gsrc, const ExpertPtr* __restrict__ gfill, const int* __restrict__ grp,
                                                  int nu, int picks_routed, const float* __restrict__ h, std::int64_t H, std::int64_t F, std::uint64_t slot,
                                                  float* __restrict__ yd, const Tables tb, int tail, int lut_bytes) {
  extern __shared__ __align__(16) std::uint8_t rd3_sh[];
  const int u = static_cast<int>(blockIdx.y);
  if (u >= grp[0]) return;
  const int* G = grp + 1 + 6 * u;
  const int cnt = G[1];
  const ExpertPtr& X = gsrc[u];
  const ExpertPtr& FL = gfill[u];
  bool any_tail = false;
  if (kTail)
    for (int c = 0; c < cnt; ++c) any_tail = any_tail || rd3_tail_pick(G[2 + c], picks_routed, nu, tail);
  const bool tg = X.type[2] == g_q4_tg2;
  const bool shadow = kTail && any_tail && tg;
  float* lut = reinterpret_cast<float*>(rd3_sh);
  float* lut2 = shadow ? lut + 2048 : nullptr;
  if (tg) rd3_lut(tb, lut, lut2);
  std::uint8_t* stage0 = rd3_sh + lut_bytes;
  const int warp = static_cast<int>(threadIdx.x >> 5), lane = static_cast<int>(threadIdx.x & 31);
  std::uint8_t* buf = stage0 + slot * static_cast<std::uint64_t>(2 * warp);
  const std::int64_t nb = F / 32;
  const bool halves = nb <= 16;
  for (std::int64_t chunk = blockIdx.x; chunk * 8 < H; chunk += gridDim.x) {
    const std::int64_t r0 = (chunk * 4 + warp) * 2;
    if (r0 >= H) continue;
    if (halves) {  // lanes 0-15 row r0, lanes 16-31 row r0 + 1
      const int hq = lane >> 4, hl = lane & 15;
      const std::int64_t r = r0 + hq;
      const bool live = r < H;
      const std::uint64_t off = static_cast<std::uint64_t>(live ? r : r0) * X.rb[2];
      const std::uint8_t* row = rd3_stage(X.w[2] + off, live ? X.rb[2] : 0, buf + slot * static_cast<std::uint64_t>(hq),
                                          (live && FL.w[2]) ? const_cast<std::uint8_t*>(FL.w[2]) + off : nullptr, hl, 16);
      float acc[kRdMaxTok] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (live)
        for (std::int64_t q = hl; q < nb; q += 16) {
          float wv[32], wv2[32];
          rd3_decode(X.type[2], row, q, tb, lut, wv);
          if (shadow) rd3_decode(X.type[2], row, q, tb, lut2, wv2);
          for (int c = 0; c < cnt; ++c) {
            const int pk = G[2 + c];
            const float* hv = h + static_cast<std::int64_t>(pk) * F + 32 * q;
            const float* w = (shadow && rd3_tail_pick(pk, picks_routed, nu, tail)) ? wv2 : wv;
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) s += w[i] * hv[i];
            acc[c] += s;
          }
        }
      for (int c = 0; c < cnt; ++c) {
        float a = acc[c];
        for (int o = 8; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (live && hl == 0) yd[static_cast<std::int64_t>(G[2 + c]) * H + r] = a;
      }
      __syncwarp();
      continue;
    }
    for (int qq = 0; qq < 2; ++qq) {
      const std::int64_t r = r0 + qq;
      if (r >= H) break;
      const std::uint64_t off = static_cast<std::uint64_t>(r) * X.rb[2];
      const std::uint8_t* row = rd3_stage(X.w[2] + off, X.rb[2], buf, FL.w[2] ? const_cast<std::uint8_t*>(FL.w[2]) + off : nullptr, lane, 32);
      float acc[kRdMaxTok] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (std::int64_t q = lane; q < nb; q += 32) {
        float wv[32], wv2[32];
        rd3_decode(X.type[2], row, q, tb, lut, wv);
        if (shadow) rd3_decode(X.type[2], row, q, tb, lut2, wv2);
        for (int c = 0; c < cnt; ++c) {
          const int pk = G[2 + c];
          const float* hv = h + static_cast<std::int64_t>(pk) * F + 32 * q;
          const float* w = (shadow && rd3_tail_pick(pk, picks_routed, nu, tail)) ? wv2 : wv;
          float s = 0.0f;
#pragma unroll
          for (int i = 0; i < 32; ++i) s += w[i] * hv[i];
          acc[c] += s;
        }
      }
      for (int c = 0; c < cnt; ++c) {
        const float a = warp_sum(acc[c]);
        if (lane == 0) yd[static_cast<std::int64_t>(G[2 + c]) * H + r] = a;
      }
      __syncwarp();
    }
  }
}
__global__ void k_rd3_combine(const float* __restrict__ yd, const float* __restrict__ wsel, const float* __restrict__ gsh, int nu, std::int64_t tokens,
                              std::int64_t H, float* __restrict__ y, const std::int32_t* __restrict__ sel, const std::uint8_t* __restrict__ cmask) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= tokens * H) return;
  y[id] = md::rd3_combine_one(yd, wsel, gsh, nu, tokens, H, id / H, id % H, sel, cmask);
}
bool moe_reader3(const ExpertPtr* tab, const ExpertPtr* shx, const std::int32_t* sel, const float* wsel, const float* gsh, int n_used, const float* xr,
                 std::int64_t tokens, std::int64_t H, std::int64_t F, std::uint64_t row_max, const ReaderCache* cache, float* gu, float* h, float* yd,
                 int* grp, ExpertPtr* gsrc, ExpertPtr* gfill, float* y, unsigned long long* ts, Stream s, int plane_tail,
                 const std::uint8_t* cpu_mask, int rows_per_warp) {
  if (plane_tail < 0 || plane_tail > n_used) return fail("moe_reader3: plane_tail 0 .. n_used");
  if (rows_per_warp != 1 && rows_per_warp != 2) return fail("moe_reader3: rows_per_warp 1 or 2");
  if (!tab || !sel || !wsel || !xr || !gu || !h || !yd || !grp || !gsrc || !gfill || !y || n_used <= 0 || n_used > 8 || H % 32 != 0 || F % 32 != 0 ||
      row_max == 0 || (shx && !gsh))
    return fail("moe_reader3: bad shape or pointer");
  if (tokens <= 0) return true;
  if (tokens > kRdMaxTok) return fail("moe_reader3: at most 4 tokens");
  Tables tb;
  if (!device_tables(&tb)) return false;
  ReaderCache c;
  if (cache && cache->K > 0) {
    c = *cache;
    if (!c.tag || !c.use || !c.stamp || !c.base || c.slot_bytes == 0) return fail("moe_reader3: incomplete cache");
  }
  const std::int64_t routed = tokens * n_used, picks = routed + (shx ? tokens : 0);
  const std::uint64_t slot = (row_max + 32 + 15) & ~std::uint64_t{15};
  // v28: the plane grid's 8 KB only when the plane test is on (v26's 8 KB + staging otherwise); the grid covers every row
  // (v26's shape: v27's cap of 16 blocks per group with a per-block row loop cut occupancy, 24.0 -> 29.6 ms)
  const int lut_bytes = 8192 * (plane_tail > 0 ? 2 : 1);
  const std::size_t shm = static_cast<std::size_t>(lut_bytes) + static_cast<std::size_t>(8 * slot);
  if (shm > 48 * 1024) return fail("moe_reader3: rows too long for the shared staging");
  const std::int64_t rows_up = 4 * static_cast<std::int64_t>(rows_per_warp);
  const unsigned gx_up = static_cast<unsigned>((2 * F + rows_up - 1) / rows_up), gx_dn = static_cast<unsigned>((H + 7) / 8);
  if (ts) k_stamp<<<1, 1, 0, st(s)>>>(ts + 0);
  k_rd3_group<<<1, 32, 0, st(s)>>>(tab, shx, sel, static_cast<int>(tokens), n_used, c, H, F, grp, gsrc, gfill, cpu_mask);
  if (ts) k_stamp<<<1, 1, 0, st(s)>>>(ts + 1);
  if (plane_tail > 0)
    k_rd3_gate_up<true><<<dim3(gx_up, static_cast<unsigned>(picks)), 128, shm, st(s)>>>(gsrc, gfill, grp, n_used, static_cast<int>(routed), xr, H, F, slot,
                                                                                      gu, tb, plane_tail, lut_bytes, rows_per_warp);
  else
    k_rd3_gate_up<false><<<dim3(gx_up, static_cast<unsigned>(picks)), 128, shm, st(s)>>>(gsrc, gfill, grp, n_used, static_cast<int>(routed), xr, H, F, slot,
                                                                                       gu, tb, 0, lut_bytes, rows_per_warp);
  if (ts) k_stamp<<<1, 1, 0, st(s)>>>(ts + 2);
  k_rd_silu_round<<<blocks_for(picks * (F / 32), 128), 128, 0, st(s)>>>(gu, picks, F, h);
  if (ts) k_stamp<<<1, 1, 0, st(s)>>>(ts + 3);
  if (plane_tail > 0)
    k_rd3_down<true><<<dim3(gx_dn, static_cast<unsigned>(picks)), 128, shm, st(s)>>>(gsrc, gfill, grp, n_used, static_cast<int>(routed), h, H, F, slot, yd,
                                                                                  tb, plane_tail, lut_bytes);
  else
    k_rd3_down<false><<<dim3(gx_dn, static_cast<unsigned>(picks)), 128, shm, st(s)>>>(gsrc, gfill, grp, n_used, static_cast<int>(routed), h, H, F, slot, yd,
                                                                                   tb, 0, lut_bytes);
  if (ts) k_stamp<<<1, 1, 0, st(s)>>>(ts + 4);
  k_rd3_combine<<<blocks_for(tokens * H, 256), 256, 0, st(s)>>>(yd, wsel, shx ? gsh : nullptr, n_used, tokens, H, y, sel, cpu_mask);
  if (ts) k_stamp<<<1, 1, 0, st(s)>>>(ts + 5);
  return launched("moe_reader3");
}

// v46a-1: for kern/gpu_stream's expert stage (moe_r3): the reader v3 grouping (cache lookup and slot choice included), its
// combine, and a GPU stamp, as moe_reader3 launches them
bool rd3_group_launch(const ExpertPtr* tab, const ExpertPtr* shx, const std::int32_t* sel, int tokens, int nu, const ReaderCache* cache, std::int64_t H,
                      std::int64_t F, int* grp, ExpertPtr* gsrc, ExpertPtr* gfill, Stream s) {
  ReaderCache c;
  if (cache) {  // v54b: also with no slots (K 0): it carries the bit-plane slider
    c = *cache;
    if (c.K > 0 && (!c.tag || !c.use || !c.stamp || !c.base || c.slot_bytes == 0)) return fail("rd3_group_launch: incomplete cache");
  }
  k_rd3_group<<<1, 32, 0, st(s)>>>(tab, shx, sel, tokens, nu, c, H, F, grp, gsrc, gfill, nullptr);
  return launched("rd3 group");
}
bool rd3_combine_launch(const float* yd, const float* wsel, const float* gsh, int nu, std::int64_t tokens, std::int64_t H, float* y,
                        const std::int32_t* sel, Stream s) {
  k_rd3_combine<<<blocks_for(tokens * H, 256), 256, 0, st(s)>>>(yd, wsel, gsh, nu, tokens, H, y, sel, nullptr);
  return launched("rd3 combine");
}
void stamp_launch(unsigned long long* p, Stream s) { k_stamp<<<1, 1, 0, st(s)>>>(p); }

bool read_rate(const void* dev, std::size_t bytes, int blocks, double* seconds) {
  unsigned* o = static_cast<unsigned*>(dalloc(sizeof(unsigned)));
  if (!o) return false;
  if (!sync()) return false;
  const auto t0 = std::chrono::steady_clock::now();
  k_read_sum<<<static_cast<unsigned>(blocks), 256>>>(static_cast<const uint4*>(dev), bytes / 16, o);
  const bool ok = launched("read rate") && sync();
  *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  dfree(o);
  return ok;
}
bool stamp(unsigned long long* p, Stream s) {
  if (!p) return fail("stamp: no address");
  k_stamp<<<1, 1, 0, st(s)>>>(p);
  return launched("stamp");
}
bool to_host_async(void* dst, const void* src, std::size_t bytes, Stream s) {
  return ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st(s)), "copy to the host (async)");
}
bool dcopy(void* dst, const void* src, std::size_t bytes, Stream s) {
  return ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st(s)), "device copy");
}
// v60 C3: the SM-load probe over a mapped host range (the path the expert stage's loads take; v60e MEASURED: the copy
// engine read 26 GB/s in an arm whose SM loads ran 2.8x slow). mode 0: streaming, every thread one 16 B word per step,
// coalesced; mode 1: one 128 B line per 4 KB page (8 lanes per line, the next line 4 KB on): the translation-bound pattern.
// The xor keeps the loads live; *out is written only for an unreachable value.
__global__ void k_probe_read(const uint4* __restrict__ p, std::size_t n16, int mode, unsigned* out) {
  unsigned acc = 0;
  const std::size_t total = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (mode == 0) {
    for (std::size_t i = tid; i < n16; i += total) {
      const uint4 v = p[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  } else {
    const std::size_t pages = n16 / 256, lines = total / 8;
    for (std::size_t g = tid / 8; g < pages; g += lines) {
      const uint4 v = p[g * 256 + (tid & 7)];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1u);
}
bool probe_read_sm(const void* src, std::size_t bytes, int mode, unsigned* out, Stream s) {
  if (!src || !out || bytes < 4096) return fail("probe_read_sm: bad range");
  k_probe_read<<<46 * 8, 256, 0, st(s)>>>(static_cast<const uint4*>(src), bytes / 16, mode, out);
  return launched("probe_read_sm");
}
bool dcopy_any(void* dst, const void* src, std::size_t bytes, Stream s) {  // v60 C2
  return ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, st(s)), "copy (default kind)");
}
bool dcopy2d(void* dst, std::size_t dpitch, const void* src, std::size_t spitch, std::size_t width, std::size_t rows, Stream s) {  // v60 C
  if (rows == 0 || width == 0) return true;
  return ck(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, rows, cudaMemcpyDeviceToDevice, st(s)), "device copy (2d)");
}
bool mem_info(std::size_t* free_bytes, std::size_t* total_bytes) {
  std::size_t f = 0, t = 0;
  if (!ck(cudaMemGetInfo(&f, &t), "cudaMemGetInfo")) return false;
  if (free_bytes) *free_bytes = f;
  if (total_bytes) *total_bytes = t;
  return true;
}

// the format tables of gpu_decode.h on the device, uploaded once (34,432 bytes: the Q4_TG grids, the IQ grids, the signs)
bool device_tables(Tables* out) {
  static Tables t;
  static bool ready = false;
  if (!ready) {
    constexpr std::size_t kTg = 64 * 4, k2xxs = 256 * 8, k2xs = 512 * 8, k2s = 1024 * 8, k3xxs = 256 * 4, k3s = 512 * 4,
                          k1s = 2048 * 8, kSg = 128;
    constexpr std::size_t kAll = 2 * kTg + k2xxs + k2xs + k2s + k3xxs + k3s + k1s + kSg;
    static unsigned char h[kAll];
    std::size_t o = 0;
    float g1[64], g2[64];
    for (int tag = 0; tag < 4; ++tag)
      for (int i = 0; i < 16; ++i) {
        g1[16 * tag + i] = quant::q4_tg_grid(tag, 1)[i];
        g2[16 * tag + i] = quant::q4_tg_grid(tag, 2)[i];
      }
    std::size_t off[9];
    const void* src[9] = {g1, g2, quant::iq::kIq2xxsGrid, quant::iq::kIq2xsGrid, quant::iq::kIq2sGrid, quant::iq::kIq3xxsGrid,
                          quant::iq::kIq3sGrid, quant::iq::kIq1sGrid, quant::iq::kSigns};
    const std::size_t len[9] = {kTg, kTg, k2xxs, k2xs, k2s, k3xxs, k3s, k1s, kSg};
    for (int i = 0; i < 9; ++i) {  // every piece starts on 8 bytes (the grids are read as 8-byte words)
      off[i] = o;
      std::memcpy(h + o, src[i], len[i]);
      o += len[i];
    }
    auto* d = static_cast<unsigned char*>(dalloc(kAll));
    if (!d || !to_device(d, h, kAll)) return fail("device_tables: upload failed");
    t.tg1 = reinterpret_cast<const float*>(d + off[0]);
    t.tg2 = reinterpret_cast<const float*>(d + off[1]);
    t.iq2xxs = reinterpret_cast<const std::uint64_t*>(d + off[2]);
    t.iq2xs = reinterpret_cast<const std::uint64_t*>(d + off[3]);
    t.iq2s = reinterpret_cast<const std::uint64_t*>(d + off[4]);
    t.iq3xxs = reinterpret_cast<const std::uint32_t*>(d + off[5]);
    t.iq3s = reinterpret_cast<const std::uint32_t*>(d + off[6]);
    t.iq1s = reinterpret_cast<const std::uint64_t*>(d + off[7]);
    t.signs = d + off[8];
    ready = true;
  }
  *out = t;
  return true;
}

bool matmul(std::uint32_t type, const void* w, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k, const float* x,
            std::int64_t n_tok, std::int64_t ldx, float* y, std::int64_t ldy, Stream s) {
  if (!type_ok(type, k, row_bytes)) return false;
  if (n_out <= 0 || n_tok < 0 || ldx < k || ldy < n_out || !w || !x || !y) return fail("matmul: bad shape or pointer");
  if (n_tok == 0) return true;
  Tables tb;
  if (!device_tables(&tb)) return false;
  const int nsub = static_cast<int>(k / 32);
  if (g_fast_f32 && type == g_f32 && n_tok <= 4 && (nsub == 8 || nsub == 16 || nsub == 32 || nsub == 64 || nsub == 128) && n_tok * nsub * kGvPad * 4 <= 48 * 1024 && row_bytes % 16 == 0 &&
      ldx % 4 == 0 && (reinterpret_cast<std::uintptr_t>(w) & 15u) == 0 && (reinterpret_cast<std::uintptr_t>(x) & 15u) == 0) {
    const std::size_t sm = static_cast<std::size_t>(n_tok) * nsub * kGvPad * sizeof(float);  // at most 48 KB (the default limit)
    const float* wf = static_cast<const float*>(w);
    const unsigned nb = blocks_for(n_out, 2);
    if (nsub <= 32) k_gemv_f32<1><<<nb, 64, sm, st(s)>>>(wf, row_bytes, n_out, k, x, static_cast<int>(n_tok), ldx, y, ldy);
    else if (nsub == 64) k_gemv_f32<2><<<nb, 64, sm, st(s)>>>(wf, row_bytes, n_out, k, x, static_cast<int>(n_tok), ldx, y, ldy);
    else k_gemv_f32<4><<<nb, 64, sm, st(s)>>>(wf, row_bytes, n_out, k, x, static_cast<int>(n_tok), ldx, y, ldy);
    return launched("matmul f32");
  }
  k_matmul<<<blocks_for(n_out, 4), 128, 0, st(s)>>>(type, static_cast<const std::uint8_t*>(w), row_bytes, n_out, k, x, n_tok, ldx, y, ldy, tb);
  return launched("matmul");
}

void set_fast_f32(bool on) { g_fast_f32 = on; }

bool moe_matmul(std::uint32_t type, const void* w, std::uint64_t expert_bytes, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k,
                int n_expert, const int* slot_off, const int* src, const int* dst, const float* x, std::int64_t ldx, float* y,
                std::int64_t ldy, int max_slots_per_expert, Stream s) {
  if (!type_ok(type, k, row_bytes)) return false;
  if (n_expert <= 0 || n_out <= 0 || expert_bytes < static_cast<std::uint64_t>(n_out) * row_bytes || !slot_off || !w || !x || !y)
    return fail("moe_matmul: bad shape or pointer");
  if (max_slots_per_expert == 0) return true;
  Tables tb;
  if (!device_tables(&tb)) return false;
  const dim3 grd(blocks_for(n_out, 4), static_cast<unsigned>(n_expert));
  k_moe_matmul<<<grd, 128, 0, st(s)>>>(type, static_cast<const std::uint8_t*>(w), expert_bytes, row_bytes, n_out, k, slot_off, src, dst, x, ldx, y,
                                       ldy, tb);
  return launched("moe_matmul");
}

bool gather_rows(std::uint32_t type, const void* w, std::uint64_t row_bytes, std::int64_t k, const std::int32_t* ids, std::int64_t n, float* out,
                 Stream s) {
  if (!type_ok(type, k, row_bytes)) return false;
  if (n <= 0) return true;
  Tables tb;
  if (!device_tables(&tb)) return false;
  k_gather<<<static_cast<unsigned>(n), 128, 0, st(s)>>>(type, static_cast<const std::uint8_t*>(w), row_bytes, k, ids, out, tb);
  return launched("gather_rows");
}

bool act_q8x32(const float* x, std::int64_t n, std::int64_t rows, std::int64_t ldx, bool rot32, float* xr, std::int64_t ldr, std::int8_t* q,
               float* scale, int* sum, Stream s) {
  if (n <= 0 || n % 32 != 0 || ldx < n || ldr < n || !x || !xr) return fail("act_q8x32: the row length must be whole 32-blocks");
  if (rows <= 0) return true;
  const float norm = 1.0f / std::sqrt(32.0f);  // as quant::hadamard32 computes it
  k_act_q8x32<false><<<blocks_for(rows * (n / 32), 128), 128, 0, st(s)>>>(x, n, rows, ldx, rot32 ? 1 : 0, norm, xr, ldr, q, scale, sum);
  return launched("act_q8x32");
}

bool rms_norm_q8x32(const float* x, const float* w, std::int64_t n, std::int64_t ldx, std::int64_t rows, float eps, bool rot32, float* y,
                    std::int64_t ldy, float* xr, std::int64_t ldr, std::int8_t* q, float* scale, int* sum, Stream s) {
  if (n <= 0 || n % 32 != 0 || ldx < n || !x) return fail("rms_norm_q8x32: the row length must be whole 32-blocks");
  if ((y && ldy < n) || (xr && ldr < n)) return fail("rms_norm_q8x32: an output stride is shorter than the row");
  if (rows <= 0) return true;
  const float norm = 1.0f / std::sqrt(32.0f);  // as act_q8x32
  k_rms_norm_q8x32<<<static_cast<unsigned>(rows), 256, 0, st(s)>>>(x, w, n, ldx, eps, rot32 ? 1 : 0, norm, y, ldy, xr, ldr, q, scale, sum);
  return launched("rms_norm_q8x32");
}

bool rms_norm(const float* x, const float* w, float* y, std::int64_t n, std::int64_t n_seg, std::int64_t seg_stride, std::int64_t ld,
              std::int64_t rows, float eps, Stream s) {
  if (n <= 0 || n_seg <= 0 || !x || !y) return fail("rms_norm: bad shape");
  if (rows <= 0) return true;
  k_rms_norm<<<static_cast<unsigned>(rows * n_seg), 256, 0, st(s)>>>(x, w, y, n, n_seg, seg_stride, ld, eps, 0);
  return launched("rms_norm");
}

bool l2_norm(float* x, std::int64_t n, std::int64_t n_seg, std::int64_t seg_stride, std::int64_t ld, std::int64_t rows, float eps, Stream s) {
  if (n <= 0 || n_seg <= 0 || !x) return fail("l2_norm: bad shape");
  if (rows <= 0) return true;
  k_rms_norm<<<static_cast<unsigned>(rows * n_seg), 256, 0, st(s)>>>(x, nullptr, x, n, n_seg, seg_stride, ld, eps, 1);
  return launched("l2_norm");
}

bool rope(float* v, std::int64_t n_heads, std::int64_t head_stride, std::int64_t ld, std::int64_t rows, std::int64_t pos0, int n_rot, float freq_base,
          const int sections[4], bool neox, const float* freq_factors, Stream s, const int* pos_d) {
  if (n_rot <= 0 || n_rot % 2 != 0 || n_heads <= 0 || !v) return fail("rope: bad shape");
  if (rows <= 0) return true;
  const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));  // as model::rope_cache, on the host
  const int s0 = sections ? sections[0] : 0, s1 = sections ? sections[1] : 0, s2 = sections ? sections[2] : 0, s3 = sections ? sections[3] : 0;
  const std::int64_t total = rows * n_heads * (n_rot / 2);
  k_rope<<<blocks_for(total, 128), 128, 0, st(s)>>>(v, n_heads, head_stride, ld, rows, pos0, n_rot, theta_scale, s0, s1, s2, s3, neox ? 1 : 0, freq_factors,
                                                    pos_d);
  return launched("rope");
}

bool silu_mul(const float* g, const float* u, float* out, std::int64_t n, Stream s) {
  if (n < 0 || !g || !u || !out) return fail("silu_mul: bad pointer");
  if (n == 0) return true;
  k_silu_mul<<<blocks_for(n, 256), 256, 0, st(s)>>>(g, u, out, n);
  return launched("silu_mul");
}

bool add_rows(float* y, std::int64_t ldy, const float* x, std::int64_t ldx, const float* a, std::int64_t n, std::int64_t rows, Stream s) {
  if (n <= 0 || !y || !x) return fail("add_rows: bad shape");
  if (rows <= 0) return true;
  k_add_rows<<<blocks_for(n * rows, 256), 256, 0, st(s)>>>(y, ldy, x, ldx, a, n, rows);
  return launched("add_rows");
}

bool g_router_serial = false;  // tests only: the serial kernel (the v22 kernel is checked against it bit for bit)
void router_topk_serial_for_tests(bool on) { g_router_serial = on; }
bool router_fast_for(std::int64_t n_expert, float tie_band) { return n_expert <= kRouterMaxExperts && (!g_router_serial || tie_band > 0.0f); }
bool router_topk(const float* logits, std::int64_t n_expert, std::int64_t rows, int n_used, float weights_scale, std::int32_t* sel, float* wsel,
                 Stream s, float tie_band) {
  if (n_expert <= 0 || n_used <= 0 || n_used > n_expert || !logits || !sel || !wsel) return fail("router_topk: bad shape");
  if (rows <= 0) return true;
  if (tie_band < 0.0f) return fail("router_topk: a negative tie band");
  if (n_expert <= kRouterMaxExperts && (!g_router_serial || tie_band > 0.0f)) {  // v22: shared-memory picks, same values
    k_router_fast<<<static_cast<unsigned>(rows), 256, 0, st(s)>>>(logits, n_expert, n_used, weights_scale, sel, wsel, tie_band);
    return launched("router_topk");
  }
  if (tie_band > 0.0f) return fail("router_topk: the tie band needs at most 1024 experts");
  // the probabilities need a work row per token; kept in a buffer that grows (a model's shape is fixed after load, so this
  // allocates on the first calls only)
  static float* work = nullptr;
  static std::int64_t cap = 0;
  if (rows * n_expert > cap) {
    dfree(work);
    work = static_cast<float*>(dalloc(static_cast<std::size_t>(rows * n_expert) * sizeof(float)));
    if (!work) return false;
    cap = rows * n_expert;
  }
  k_router<<<static_cast<unsigned>(rows), 256, 0, st(s)>>>(logits, n_expert, n_used, weights_scale, sel, wsel, work);
  return launched("router_topk");
}

bool attention(const float* q, std::int64_t q_stride, std::int64_t q_head, const float* K, const float* V, std::int64_t kv_w, std::int64_t n_head,
               std::int64_t n_head_kv, std::int64_t head_dim, std::int64_t n_tok, std::int64_t p0, float scale, bool gate, float* out, float* scratch,
               Stream s, const int* pos_d, std::int64_t l_max) {
  if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > 1024 || kv_w < n_head_kv * head_dim || p0 < 0 ||
      q_head < (gate ? 2 : 1) * head_dim || !q || !K || !V || !out || !scratch)
    return fail("attention: bad shape or pointer");
  if (n_tok <= 0) return true;
  if (n_tok > 65535 || n_head > 65535) return fail("attention: at most 65535 tokens and heads per call (grid limit; split a longer prompt)");
  // v26: with pos_d the position is read on the device (graph replay): the score rows have stride l_max and the key blocks
  // loop (64 per row-head); the per-score arithmetic is the same either way
  if (pos_d && l_max < p0 + n_tok) return fail("attention: l_max below the position");
  const std::int64_t L = pos_d ? l_max : p0 + n_tok, group = n_head / n_head_kv;
  const dim3 g1(pos_d ? 64u : blocks_for(L, 8), static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok));
  k_attn_scores<<<g1, 256, 0, st(s)>>>(q, q_stride, q_head, K, kv_w, n_head, group, head_dim, p0, L, scale, scratch, pos_d);
  if (!launched("attention scores")) return false;
  k_attn_softmax<<<static_cast<unsigned>(n_tok * n_head), 256, 0, st(s)>>>(scratch, n_head, p0, L, pos_d);
  if (!launched("attention softmax")) return false;
  const int tpb = head_dim >= 256 ? 256 : 128;
  const dim3 g3(static_cast<unsigned>(n_tok * n_head), blocks_for(head_dim, tpb));
  k_attn_out<<<g3, tpb, 0, st(s)>>>(scratch, V, kv_w, n_head, group, head_dim, p0, L, q, q_stride, q_head, gate ? 1 : 0, out, pos_d);
  return launched("attention output");
}

bool deltanet_conv(const float* mixed, const float* tail, const float* w, std::int64_t n, std::int64_t ch, std::int64_t taps, float* conv,
                   float* new_tail, Stream s) {
  if (ch <= 0 || taps <= 0 || !mixed || !w || !conv || (new_tail && new_tail == tail)) return fail("deltanet_conv: bad shape or aliased tail");
  if (n <= 0) return true;
  k_dn_conv<<<blocks_for(n * ch, 256), 256, 0, st(s)>>>(mixed, tail, w, n, ch, taps, conv);
  if (!launched("deltanet conv")) return false;
  if (new_tail && taps > 1) {
    k_dn_tail<<<blocks_for((taps - 1) * ch, 256), 256, 0, st(s)>>>(mixed, tail, n, ch, taps, new_tail);
    if (!launched("deltanet conv tail")) return false;
  }
  return true;
}

// v55g (L2, the author's order 03:13: products that share an input in one launch): a DeltaNet layer's beta and alpha
// products (F32, nv rows each, the same input) and the gates in one launch: block h computes head h's beta row (warp 0)
// and alpha row (warp 1) with k_gemv_f32's sums (md::gemv_row on the same staged input, so the same bits as the two
// separate products), then the gates of that head for each token as k_dn_gates. Three launches per layer become one.
template <int NJ>
__global__ void __launch_bounds__(64) k_dn_beta_alpha_gates(const float* __restrict__ wb, std::uint64_t rbb, const float* __restrict__ wa,
                                                            std::uint64_t rba, std::int64_t nv, std::int64_t k, const float* __restrict__ x, int n_tok,
                                                            std::int64_t ldx, float* beta, float* alpha, const float* __restrict__ dt,
                                                            const float* __restrict__ a) {
  extern __shared__ float4 s_gv4[];
  float* xs = reinterpret_cast<float*>(s_gv4);
  const int nsub = static_cast<int>(k / 32);
  md::gemv_stage(x, n_tok, nsub, ldx, xs);
  __syncthreads();
  const std::int64_t h = blockIdx.x;
  if (threadIdx.x < 32) md::gemv_row<NJ>(wb, rbb, h, xs, nsub, n_tok, beta, nv);
  else md::gemv_row<NJ>(wa, rba, h, xs, nsub, n_tok, alpha, nv);
  __syncthreads();
  for (int t = static_cast<int>(threadIdx.x); t < n_tok; t += static_cast<int>(blockDim.x)) {
    const std::int64_t id = t * nv + h;
    beta[id] = sigmoid_d(beta[id]);
    const float xv = add(alpha[id], dt[h]);
    const float sp = xv > 20.0f ? xv : logf(add(1.0f, expf(xv)));
    alpha[id] = mul(sp, a[h]);
  }
}
// the fused form only where both products would take k_gemv_f32 (matmul's fast F32 path), so the sums are the same
bool deltanet_ba_fast(std::uint32_t type_b, const void* wb, std::uint64_t rbb, std::uint32_t type_a, const void* wa, std::uint64_t rba, std::int64_t nv,
                      std::int64_t k, const float* x, std::int64_t n_tok, std::int64_t ldx) {
  const int nsub = static_cast<int>(k / 32);
  return g_fast_f32 && type_b == g_f32 && type_a == g_f32 && n_tok >= 1 && n_tok <= 4 && k % 32 == 0 &&
         (nsub == 8 || nsub == 16 || nsub == 32 || nsub == 64 || nsub == 128) && n_tok * nsub * kGvPad * 4 <= 48 * 1024 && rbb % 16 == 0 &&
         rba % 16 == 0 && ldx % 4 == 0 && (reinterpret_cast<std::uintptr_t>(wb) & 15u) == 0 && (reinterpret_cast<std::uintptr_t>(wa) & 15u) == 0 &&
         (reinterpret_cast<std::uintptr_t>(x) & 15u) == 0 && nv > 0;
}
bool deltanet_beta_alpha_gates(std::uint32_t type_b, const void* wb, std::uint64_t rbb, std::uint32_t type_a, const void* wa, std::uint64_t rba,
                               std::int64_t nv, std::int64_t k, const float* x, std::int64_t n_tok, std::int64_t ldx, float* beta, float* alpha,
                               const float* dt, const float* a, Stream s) {
  const int nsub = static_cast<int>(k / 32);
  if (!deltanet_ba_fast(type_b, wb, rbb, type_a, wa, rba, nv, k, x, n_tok, ldx) || !beta || !alpha || !dt || !a)
    return false;  // not a failure: the caller runs the separate products and gates
  const std::size_t sm = static_cast<std::size_t>(n_tok) * nsub * kGvPad * sizeof(float);
  const float* fb = static_cast<const float*>(wb);
  const float* fa = static_cast<const float*>(wa);
  const unsigned nb = static_cast<unsigned>(nv);
  if (nsub <= 32) k_dn_beta_alpha_gates<1><<<nb, 64, sm, st(s)>>>(fb, rbb, fa, rba, nv, k, x, static_cast<int>(n_tok), ldx, beta, alpha, dt, a);
  else if (nsub == 64) k_dn_beta_alpha_gates<2><<<nb, 64, sm, st(s)>>>(fb, rbb, fa, rba, nv, k, x, static_cast<int>(n_tok), ldx, beta, alpha, dt, a);
  else k_dn_beta_alpha_gates<4><<<nb, 64, sm, st(s)>>>(fb, rbb, fa, rba, nv, k, x, static_cast<int>(n_tok), ldx, beta, alpha, dt, a);
  return launched("deltanet beta / alpha / gates");
}

// v56 (MTP and greedy decode): each row's argmax on the device (the first index of the largest value, as the host's strict
// > scan from index 0), so a step copies one int per row to the host instead of the row's n_vocab floats
__global__ void __launch_bounds__(1024) k_argmax_rows(const float* __restrict__ x, std::int64_t n, std::int64_t ld, std::int32_t* __restrict__ out) {
  __shared__ float sv[32];
  __shared__ long long si[32];
  const float* r = x + static_cast<std::int64_t>(blockIdx.x) * ld;
  const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, w = tid >> 5;
  float bv = 0.0f;
  long long bi = -1;
  for (std::int64_t i = tid; i < n; i += blockDim.x) {
    const float v = r[i];
    if (bi < 0 || v > bv) {
      bv = v;
      bi = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const long long oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
      bv = ov;
      bi = oi;
    }
  }
  if (lane == 0) {
    sv[w] = bv;
    si[w] = bi;
  }
  __syncthreads();
  if (w == 0) {
    const int nw = static_cast<int>(blockDim.x) >> 5;
    bv = lane < nw ? sv[lane] : 0.0f;
    bi = lane < nw ? si[lane] : -1;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const long long oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) out[blockIdx.x] = static_cast<std::int32_t>(bi < 0 ? 0 : bi);
  }
}
bool argmax_rows(const float* x, std::int64_t n, std::int64_t rows, std::int64_t ld, std::int32_t* out, Stream s) {
  if (n <= 0 || ld < n || !x || !out) return fail("argmax_rows: bad shape");
  if (rows <= 0) return true;
  k_argmax_rows<<<static_cast<unsigned>(rows), 1024, 0, st(s)>>>(x, n, ld, out);
  return launched("argmax rows");
}

// ------------------------------------------------------------------------------------------------ head sketch (v60 E)
// The MTP draft's head through the Q6_K head's top two code planes (kern/tg2_planes.h g_q6_ksk): built once at open, read by
// the 1-bit route per draft, its top k rows re-scored by the full product's own kernel. Nothing here is on the verify path.

// one thread per 256-value block: 208 R3 bytes -> 16 plane words + 16 scale bytes in the row layout (nbr blocks per row)
__global__ void k_head_sketch_build(const std::uint8_t* __restrict__ w208, std::int64_t nblk, int nbr, std::uint8_t* __restrict__ out) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= nblk) return;
  const std::int64_t r = id / nbr, b = id % nbr;
  std::uint32_t pl[16];
  std::uint8_t sc[16];
  q6k_sketch_block(w208 + 208 * id, pl, sc);
  const std::uint64_t rb = static_cast<std::uint64_t>(nbr) * 80;  // 10 k / 32 with k = 256 nbr
  auto* pw = reinterpret_cast<std::uint32_t*>(out + static_cast<std::uint64_t>(r) * rb) + 16 * b;
  for (int i = 0; i < 16; ++i) pw[i] = pl[i];
  std::uint8_t* ps = out + static_cast<std::uint64_t>(r) * rb + static_cast<std::uint64_t>(nbr) * 64 + 16 * b;
  for (int i = 0; i < 16; ++i) ps[i] = sc[i];
}

// one warp per row, 8 rows per block; the token's int8 activations as words and the per-half sums of x (dp4a against
// 0x01010101) built once per block in shared memory; lane l takes sub-blocks l, l + 32, ... (8 B of planes + 2 scale bytes
// each) and forms the sketch term with the shared tg2_planes.h helpers in their dp4a form (v60 E3; the popcount form read at
// 318 GB/s, MEASURED v60j: popc is a quarter-rate instruction on sm_89); warp_sum, as the R3 kernel's lanes
constexpr int kSkSubMax = 128;  // k <= 4096
__global__ void __launch_bounds__(256) k_head_sketch_dot(const std::uint8_t* __restrict__ sk, const std::uint16_t* __restrict__ d6, std::int64_t rows,
                                                          int nsub, const std::int8_t* __restrict__ q, const float* __restrict__ xs,
                                                          float* __restrict__ y) {
  __shared__ std::int32_t xw[kSkSubMax][9];  // 9: lanes read s = lane -> distinct banks
  __shared__ int xh[kSkSubMax][2];
  __shared__ float sxs[kSkSubMax];
  for (int s = static_cast<int>(threadIdx.x); s < nsub; s += static_cast<int>(blockDim.x)) {
    const auto* qw = reinterpret_cast<const std::int32_t*>(q + 32 * s);
    int h0 = 0, h1 = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      xw[s][g] = qw[g];
      xw[s][4 + g] = qw[4 + g];
      h0 = dp4a_i32(qw[g], 0x01010101, h0);
      h1 = dp4a_i32(qw[4 + g], 0x01010101, h1);
    }
    xh[s][0] = h0;
    xh[s][1] = h1;
    sxs[s] = xs[s];
  }
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * 8 + warp;
  if (r >= rows) return;
  const std::uint64_t rb = static_cast<std::uint64_t>(nsub) * 10;
  const std::uint8_t* row = sk + static_cast<std::uint64_t>(r) * rb;
  const auto* pw = reinterpret_cast<const std::uint32_t*>(row);
  const std::uint8_t* sc = row + static_cast<std::uint64_t>(nsub) * 8;
  const std::uint16_t* dr = d6 + r * (nsub >> 3);
  float acc = 0.0f;
  for (int s = lane; s < nsub; s += 32) {
    const uint2 w2 = *reinterpret_cast<const uint2*>(pw + 2 * s);
    const int sc0 = static_cast<int>(static_cast<std::int8_t>(sc[2 * s])), sc1 = static_cast<int>(static_cast<std::int8_t>(sc[2 * s + 1]));
    const float d = half_float(dr[s >> 3]);  // gpu_decode.h: exact, as quant::fp16_to_fp32 (the host test uses that)
    const int t0 = q6k_sketch_half2_dp(w2.x, w2.y, xw[s], 0, xh[s][0]);
    const int t1 = q6k_sketch_half2_dp(w2.x, w2.y, xw[s], 1, xh[s][1]);
    acc += q6k_sketch_sub(d, sxs[s], sc0, sc1, t0, t1);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[r] = acc;
}

// the k largest of n floats, one block of 1024: radix select on the order-preserving key of the float (4 passes of 8 bits from
// the top: a shared histogram of the candidates' digit, the digit holding the k-th largest chosen), then the ids with a key
// above the k-th key, then as many ids with the k-th key as are still needed
__device__ __forceinline__ unsigned f_key(float f) {
  const unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void __launch_bounds__(1024) k_topk_select(const float* __restrict__ x, std::int64_t n, int k, std::int32_t* __restrict__ ids) {
  __shared__ unsigned hist[256];
  __shared__ unsigned s_pre, s_mask, s_need, s_cnt_gt, s_cnt_eq;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    s_pre = 0;
    s_mask = 0;
    s_need = static_cast<unsigned>(k);
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    const unsigned pre = s_pre, mask = s_mask;
    // the plain per-element shared atomicAdd (v60 E; 0.105 ms per 248k on this card, MEASURED v60j). v60 E3 tried a warp
    // aggregation first (__ballot_sync + __match_any_sync, one add per distinct digit per warp, the loop bounded on base):
    // 0.26 ms, MEASURED v60k - the match costs more than the same-bin atomics it saves on sm_89; reverted in E3b. The next
    // step for this 0.1 ms, if it ever matters, is a grid-wide radix select (per-block histograms into a global one)
    for (std::int64_t i = tid; i < n; i += 1024) {
      const unsigned ky = f_key(x[i]);
      if ((ky & mask) == pre) atomicAdd(&hist[(ky >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned cum = 0, need = s_need, chosen = 0;
      for (int d = 255; d >= 0; --d) {
        cum += hist[d];
        if (cum >= need) {
          chosen = static_cast<unsigned>(d);
          need -= cum - hist[d];
          break;
        }
      }
      s_pre = pre | (chosen << shift);
      s_mask = mask | (255u << shift);
      s_need = need;
      s_cnt_gt = 0;
      s_cnt_eq = 0;
    }
    __syncthreads();
  }
  const unsigned kth = s_pre, need = s_need;
  const unsigned n_gt = static_cast<unsigned>(k) - need;
  for (std::int64_t i = tid; i < n; i += 1024) {
    const unsigned ky = f_key(x[i]);
    if (ky > kth) {
      const unsigned at = atomicAdd(&s_cnt_gt, 1u);
      if (at < n_gt) ids[at] = static_cast<std::int32_t>(i);
    } else if (ky == kth) {
      const unsigned at = atomicAdd(&s_cnt_eq, 1u);
      if (at < need) ids[n_gt + at] = static_cast<std::int32_t>(i);
    }
  }
}

// n rows of 208-byte blocks (13 int4 each) and their d6 gathered by id; one block per row
__global__ void __launch_bounds__(256) k_gather_q6k_rows(const int4* __restrict__ w, const std::uint16_t* __restrict__ d6, int nbr, const std::int32_t* __restrict__ ids,
                                                          int4* __restrict__ wo, std::uint16_t* __restrict__ d6o) {
  const int r = static_cast<int>(blockIdx.x);
  const std::int64_t src = static_cast<std::int64_t>(ids[r]) * nbr;
  const int nw = nbr * 13;
  for (int i = static_cast<int>(threadIdx.x); i < nw; i += static_cast<int>(blockDim.x)) wo[static_cast<std::int64_t>(r) * nw + i] = w[src * 13 + i];
  if (threadIdx.x < static_cast<unsigned>(nbr)) d6o[r * nbr + static_cast<int>(threadIdx.x)] = d6[src + static_cast<int>(threadIdx.x)];
}
__global__ void k_pick_id(const std::int32_t* __restrict__ ids, const std::int32_t* __restrict__ idx, std::int32_t* __restrict__ out) { out[0] = ids[idx[0]]; }
__global__ void k_recall_count(const std::int32_t* __restrict__ ids, int n, const std::int32_t* __restrict__ truth, unsigned* hit, unsigned* total) {
  __shared__ int found;
  if (threadIdx.x == 0) found = 0;
  __syncthreads();
  const std::int32_t t = truth[0];
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x))
    if (ids[i] == t) found = 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    ++*total;
    if (found) ++*hit;
  }
}

bool head_sketch_build(const void* w208, std::int64_t rows, std::int64_t k, std::uint8_t* out, Stream s) {
  if (!w208 || !out || rows <= 0 || k < 256 || k > 4096 || k % 256) return fail("head_sketch_build: bad shape");
  const std::int64_t nblk = rows * (k / 256);
  k_head_sketch_build<<<blocks_for(nblk, 256), 256, 0, st(s)>>>(static_cast<const std::uint8_t*>(w208), nblk, static_cast<int>(k / 256), out);
  return launched("head sketch build");
}
bool head_sketch_dot(const std::uint8_t* sk, const std::uint16_t* d6, std::int64_t rows, std::int64_t k, const std::int8_t* q, const float* xs, float* y,
                     Stream s) {
  if (!sk || !d6 || !q || !xs || !y || rows <= 0 || k < 256 || k > 4096 || k % 256) return fail("head_sketch_dot: bad shape");
  k_head_sketch_dot<<<blocks_for(rows, 8), 256, 0, st(s)>>>(sk, d6, rows, static_cast<int>(k / 32), q, xs, y);
  return launched("head sketch dot");
}
bool topk_select(const float* x, std::int64_t n, int k, std::int32_t* ids, Stream s) {
  if (!x || !ids || k <= 0 || k > 1024 || n < k) return fail("topk_select: bad shape");
  k_topk_select<<<1, 1024, 0, st(s)>>>(x, n, k, ids);
  return launched("top-k select");
}
bool gather_q6k_rows(const void* w208, const std::uint16_t* d6, std::int64_t k, const std::int32_t* ids, int n, void* w_out, std::uint16_t* d6_out, Stream s) {
  if (!w208 || !d6 || !ids || !w_out || !d6_out || n <= 0 || k < 256 || k > 4096 || k % 256) return fail("gather_q6k_rows: bad shape");
  k_gather_q6k_rows<<<static_cast<unsigned>(n), 256, 0, st(s)>>>(static_cast<const int4*>(w208), d6, static_cast<int>(k / 256), ids, static_cast<int4*>(w_out), d6_out);
  return launched("gather Q6_K rows");
}
bool pick_id(const std::int32_t* ids, const std::int32_t* idx, std::int32_t* out, Stream s) {
  if (!ids || !idx || !out) return fail("pick_id: no address");
  k_pick_id<<<1, 1, 0, st(s)>>>(ids, idx, out);
  return launched("pick id");
}
bool recall_count(const std::int32_t* ids, int n, const std::int32_t* truth, unsigned* hit, unsigned* total, Stream s) {
  if (!ids || !truth || !hit || !total || n <= 0) return fail("recall_count: bad arguments");
  k_recall_count<<<1, 256, 0, st(s)>>>(ids, n, truth, hit, total);
  return launched("recall count");
}

bool deltanet_gates(float* beta, float* alpha, const float* dt, const float* a, std::int64_t n, std::int64_t nv, Stream s) {
  if (nv <= 0 || !beta || !alpha || !dt || !a) return fail("deltanet_gates: bad pointer");
  if (n <= 0) return true;
  k_dn_gates<<<blocks_for(n * nv, 256), 256, 0, st(s)>>>(beta, alpha, dt, a, n, nv);
  return launched("deltanet gates");
}

// The shared-memory opt-in of a kernel, done once per kernel and size; the card's limit read once (C2: the per-call device
// query and opt-in were part of every decode step's host time).
bool smem_ready(const void* kernel, std::size_t smem, const char* what) {
  static std::mutex mu;
  static int optin = -1;
  static std::vector<std::pair<const void*, std::size_t>> done;
  std::lock_guard<std::mutex> lock(mu);
  if (optin < 0) {
    int dev = 0, v = 0;
    if (!ck(cudaGetDevice(&dev), "cudaGetDevice") || !ck(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "shared memory limit"))
      return false;
    optin = v;
  }
  if (smem > static_cast<std::size_t>(optin))
    return fail(std::string(what) + ": needs " + std::to_string(smem) + " bytes of shared memory, this card offers " + std::to_string(optin));
  for (const auto& d : done)
    if (d.first == kernel && d.second >= smem) return true;
  if (!ck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "shared memory opt-in")) return false;
  for (auto& d : done)
    if (d.first == kernel) {
      d.second = smem;
      return true;
    }
  done.emplace_back(kernel, smem);
  return true;
}

bool deltanet_recurrent(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n, std::int64_t nk,
                        std::int64_t nv, std::int64_t ds, float* core, Stream s) {
  if (nk <= 0 || nv <= 0 || nv % nk != 0 || ds <= 0 || ds > 1024 || ds % 32 != 0 || ch < (2 * nk + nv) * ds || !conv || !beta || !alpha || !state || !core)
    return fail("deltanet_recurrent: bad shape or pointer");
  if (n <= 0) return true;
  const std::size_t smem = static_cast<std::size_t>(ds * (ds + 1) + 3 * ds) * sizeof(float);
  if (!smem_ready(reinterpret_cast<const void*>(k_dn_recurrent), smem, "deltanet_recurrent")) return false;
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));  // as the reference, on the host
  k_dn_recurrent<<<static_cast<unsigned>(nv), static_cast<unsigned>(ds), smem, st(s)>>>(conv, ch, beta, alpha, state, n, nk, nv, ds, qscale, core);
  return launched("deltanet recurrent");
}

bool deltanet_chunked(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n, std::int64_t nk,
                      std::int64_t nv, std::int64_t ds, float* core, Stream s) {
  if (nk <= 0 || nv <= 0 || nv % nk != 0 || ds <= 0 || ds > 256 || ds % 32 != 0 || ch < (2 * nk + nv) * ds || !conv || !beta || !alpha || !state || !core)
    return fail("deltanet_chunked: bad shape or pointer (d_state a multiple of 32, at most 256)");
  if (n <= 0) return true;
  const std::size_t smem = static_cast<std::size_t>(ds * (ds + 1) + 3 * kDnC * ds + 2 * kDnC * kDnC + 2 * kDnC) * sizeof(float);
  if (!smem_ready(reinterpret_cast<const void*>(k_dn_chunked), smem, "deltanet_chunked")) return false;
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));  // as the reference, on the host
  k_dn_chunked<<<static_cast<unsigned>(nv), 256, smem, st(s)>>>(conv, ch, beta, alpha, state, n, nk, nv, ds, qscale, core);
  return launched("deltanet chunked");
}

bool deltanet_recurrent_reg(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                            std::int64_t nk, std::int64_t nv, std::int64_t ds, float* core, Stream s) {
  const bool pow2 = ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0;
  if (nk <= 0 || nv <= 0 || nv % nk != 0 || !pow2 || ch < (2 * nk + nv) * ds || !conv || !beta || !alpha || !state || !core)
    return fail("deltanet_recurrent_reg: bad shape or pointer (d_state a power of two, 32 to 1024)");
  if (n <= 0) return true;
  if (nv > 2147483647) return fail("deltanet_recurrent_reg: too many heads");
  const int tpr = static_cast<int>(ds / 32);
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));
  const dim3 grid(static_cast<unsigned>(nv), static_cast<unsigned>(ds / 32));
  k_dn_rec_reg<<<grid, static_cast<unsigned>(32 * tpr), static_cast<std::size_t>(2 * ds) * sizeof(float), st(s)>>>(conv, ch, beta, alpha, state, n, nk, nv, ds,
                                                                                                                   tpr, qscale, core);
  return launched("deltanet recurrent (registers)");
}

bool deltanet_recurrent_reg_ckpt(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                                 std::int64_t nk, std::int64_t nv, std::int64_t ds, float* core, float* ckpt, Stream s) {
  const bool pow2 = ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0;
  if (nk <= 0 || nv <= 0 || nv % nk != 0 || !pow2 || ch < (2 * nk + nv) * ds || !conv || !beta || !alpha || !state || !core || (n > 1 && !ckpt))
    return fail("deltanet_recurrent_reg_ckpt: bad shape or pointer (d_state a power of two, 32 to 1024; ckpt for n > 1)");
  if (n <= 0) return true;
  if (nv > 2147483647) return fail("deltanet_recurrent_reg_ckpt: too many heads");
  const int tpr = static_cast<int>(ds / 32);
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));
  const dim3 grid(static_cast<unsigned>(nv), static_cast<unsigned>(ds / 32));
  k_dn_rec_reg_ck<<<grid, static_cast<unsigned>(32 * tpr), static_cast<std::size_t>(2 * ds) * sizeof(float), st(s)>>>(conv, ch, beta, alpha, state, n, nk, nv,
                                                                                                                      ds, tpr, qscale, core, ckpt);
  return launched("deltanet recurrent (registers, checkpoints)");
}

template <int C>
bool launch_chunked_rows(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n, std::int64_t nk,
                         std::int64_t nv, std::int64_t ds, float* core, Stream s) {
  const std::size_t smem =
      static_cast<std::size_t>(kDnRows * (ds + 1) + 2 * C * ds + 2 * C * kDnRows + 2 * C * C + 4 * C) * sizeof(float);
  if (!smem_ready(reinterpret_cast<const void*>(k_dn_chunked_rows<C>), smem, "deltanet_chunked_rows")) return false;
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));
  const dim3 grid(static_cast<unsigned>(nv), static_cast<unsigned>(ds / kDnRows));
  k_dn_chunked_rows<C><<<grid, 256, smem, st(s)>>>(conv, ch, beta, alpha, state, n, nk, nv, ds, qscale, core);
  return launched("deltanet chunked (rows)");
}

bool deltanet_chunked_rows(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                           std::int64_t nk, std::int64_t nv, std::int64_t ds, int chunk, float* core, Stream s) {
  if (nk <= 0 || nv <= 0 || nv % nk != 0 || ds < 32 || ds > 256 || ds % 32 != 0 || ch < (2 * nk + nv) * ds || !conv || !beta || !alpha || !state || !core)
    return fail("deltanet_chunked_rows: bad shape or pointer (d_state a multiple of 32, at most 256)");
  if (chunk != 16 && chunk != 32) return fail("deltanet_chunked_rows: chunk must be 16 or 32");
  if (n <= 0) return true;
  return chunk == 16 ? launch_chunked_rows<16>(conv, ch, beta, alpha, state, n, nk, nv, ds, core, s)
                     : launch_chunked_rows<32>(conv, ch, beta, alpha, state, n, nk, nv, ds, core, s);
}

bool gated_norm(float* core, const float* z, const float* w, std::int64_t n, std::int64_t nv, std::int64_t ds, float eps, Stream s) {
  if (nv <= 0 || ds <= 0 || !core || !z || !w) return fail("gated_norm: bad pointer");
  if (n <= 0) return true;
  k_gated_norm<<<static_cast<unsigned>(n * nv), 128, 0, st(s)>>>(core, z, w, nv, ds, eps);
  return launched("gated_norm");
}


// ------------------------------------------------------------------------------------------------ v26 streams, graphs
__global__ void k_kv_store(float* __restrict__ K, float* __restrict__ V, const float* __restrict__ kn, const float* __restrict__ vn, std::int64_t kv_w,
                           std::int64_t n, const int* __restrict__ pos_d) {
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= n * kv_w) return;
  const std::int64_t at = static_cast<std::int64_t>(*pos_d) * kv_w + id;
  K[at] = kn[id];
  V[at] = vn[id];
}
bool kv_store(float* K, float* V, const float* kn, const float* vn, std::int64_t kv_w, std::int64_t n, const int* pos_d, Stream s) {
  if (!K || !V || !kn || !vn || !pos_d || kv_w <= 0) return fail("kv_store: bad pointer");
  if (n <= 0) return true;
  k_kv_store<<<blocks_for(n * kv_w, 256), 256, 0, st(s)>>>(K, V, kn, vn, kv_w, n, pos_d);
  return launched("kv_store");
}
cudaStream_t cur_stream_raw() { return g_cur_stream; }
void set_stream(Stream s) { g_cur_stream = static_cast<cudaStream_t>(s); }
Stream stream_new() {
  cudaStream_t s = nullptr;
  if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate")) return nullptr;
  return s;
}
void stream_free(Stream s) {
  if (s) cudaStreamDestroy(static_cast<cudaStream_t>(s));
}
bool h2d_async(void* dst, const void* src, std::size_t bytes, Stream s) {
  return ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st(s)), "copy to the device (async)");
}
bool capture_begin(Stream s) { return ck(cudaStreamBeginCapture(st(s), cudaStreamCaptureModeRelaxed), "graph capture begin"); }
bool capture_end(Stream s, void** exec) {
  cudaGraph_t gr = nullptr;
  *exec = nullptr;
  if (!ck(cudaStreamEndCapture(st(s), &gr), "graph capture end")) return false;
  cudaGraphExec_t ex = nullptr;
  const bool ok = ck(cudaGraphInstantiate(&ex, gr, 0ull), "graph instantiate");
  cudaGraphDestroy(gr);
  if (!ok) return false;
  *exec = ex;
  return true;
}
bool graph_launch(void* exec, Stream s) { return ck(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), st(s)), "graph launch"); }
void graph_free(void* exec) {
  if (exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec));
}

// ------------------------------------------------------------------------------------------------ v26 reader prefetch
// One thread: the step's predicted picks of the next layer (sel_pred, the next layer's router on this layer's output);
// each predicted file expert not yet cached gets the least recently used slot, marked as used by the coming step (so that
// step's own misses do not evict it). The list of (expert, slot) goes to `job` ([0] = count).
__global__ void k_pf_assign(const ExpertPtr* __restrict__ tab, const std::int32_t* __restrict__ sel_pred, int npicks, ReaderCache c,
                            int* __restrict__ job) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const unsigned next = *c.stamp + 1u;
  int nj = 0;
  for (int pi = 0; pi < npicks; ++pi) {
    const int e = sel_pred[pi];
    if (tab[e].vram) continue;
    bool have = false;
    int k0 = 0, k1 = 0;
    std::uint64_t pslot = 0, poff = 0;
    rc_pool(c, e, &k0, &k1, &pslot, &poff);  // v51
    for (int k = k0; k < k1 && !have; ++k)
      if (c.tag[k] == e) {
        have = true;
        c.use[k] = next;
      }
    if (have) continue;
    int victim = -1;
    unsigned best = 0xffffffffu;
    for (int k = k0; k < k1; ++k)
      if (c.use[k] != next && c.use[k] < best) {
        best = c.use[k];
        victim = k;
      }
    if (victim < 0) continue;  // v51: this expert's pool is full for the coming step (others may still fit theirs)
    c.tag[victim] = e;
    c.use[victim] = next;
    job[1 + 2 * nj] = e;
    job[2 + 2 * nj] = victim;
    ++nj;
  }
  job[0] = nj;
}
// copy the listed experts' three matrices into their slots (blockIdx.y = item; blocks stride over the bytes)
__global__ void k_pf_copy(const ExpertPtr* __restrict__ tab, const int* __restrict__ job, ReaderCache c, std::int64_t H, std::int64_t F) {
  const int it = static_cast<int>(blockIdx.y);
  if (it >= job[0]) return;
  const ExpertPtr& X = tab[job[1 + 2 * it]];
  int k0 = 0, k1 = 0;
  std::uint64_t pslot = 0, poff = 0;
  rc_pool(c, job[1 + 2 * it], &k0, &k1, &pslot, &poff);  // v51
  std::uint8_t* sb = c.base + poff + static_cast<std::uint64_t>(job[2 + 2 * it] - k0) * pslot;
  const std::uint64_t len[3] = {X.rb[0] * static_cast<std::uint64_t>(F), X.rb[1] * static_cast<std::uint64_t>(F), X.rb[2] * static_cast<std::uint64_t>(H)};
  std::uint64_t off = 0;
  const std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x, nth = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (int pj = 0; pj < 3; ++pj) {
    const std::uint8_t* src = X.w[pj];
    std::uint8_t* dst = sb + off;
    const std::uint64_t n = len[pj];
    if (((reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst) | n) & 15u) == 0) {
      const uint4* s4 = reinterpret_cast<const uint4*>(src);
      uint4* d4 = reinterpret_cast<uint4*>(dst);
      for (std::uint64_t i = tid; i < n / 16; i += nth) d4[i] = s4[i];
    } else {
      for (std::uint64_t i = tid; i < n; i += nth) dst[i] = src[i];
    }
    off += n;
  }
}
bool reader_prefetch(const ExpertPtr* tab, const std::int32_t* sel_pred, int npicks, const ReaderCache* cache, std::int64_t H, std::int64_t F, int* job,
                     Stream s) {
  if (!tab || !sel_pred || !cache || cache->K <= 0 || !job || npicks <= 0 || npicks > 64) return fail("reader_prefetch: bad arguments");
  k_pf_assign<<<1, 32, 0, st(s)>>>(tab, sel_pred, npicks, *cache, job);
  k_pf_copy<<<dim3(32u, static_cast<unsigned>(npicks)), 256, 0, st(s)>>>(tab, job, *cache, H, F);
  return launched("reader_prefetch");
}
void* event_new() {
  cudaEvent_t e = nullptr;
  if (!ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate")) return nullptr;
  return e;
}
void event_free(void* e) {
  if (e) cudaEventDestroy(static_cast<cudaEvent_t>(e));
}
bool event_record(void* e, Stream s) { return ck(cudaEventRecord(static_cast<cudaEvent_t>(e), st(s)), "cudaEventRecord"); }
bool stream_wait(Stream s, void* e) { return ck(cudaStreamWaitEvent(st(s), static_cast<cudaEvent_t>(e), 0), "cudaStreamWaitEvent"); }
}  // namespace pelias::kern::gpu

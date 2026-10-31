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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "kern/gpu_decode.h"
#include "kern/gpu_ops.h"
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

cudaStream_t st(Stream s) { return static_cast<cudaStream_t>(s); }

unsigned blocks_for(std::int64_t n, int per) { return static_cast<unsigned>((n + per - 1) / per); }

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// block-wide double sum (blockDim.x a multiple of 32, at most 1024); every thread gets the result
__device__ double block_sum_d(double v, double* sh) {
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

__device__ float block_max_f(float v, float* sh) {
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
      decode32(type, row, j, tb, wv);
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

// one thread per 32-value block: the same loops as quant::hadamard32 and model::act_round_q8_256x8
__global__ void k_act_q8x32(const float* __restrict__ x, std::int64_t n, std::int64_t rows, std::int64_t ldx, int rot, float norm,
                            float* xr, std::int64_t ldr, std::int8_t* __restrict__ q, float* __restrict__ scale, int* __restrict__ sum) {
  const std::int64_t nb = n / 32;
  const std::int64_t id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= rows * nb) return;
  const std::int64_t t = id / nb, b = id % nb;
  float v[32];
  for (int i = 0; i < 32; ++i) v[i] = x[t * ldx + 32 * b + i];
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
  }
  if (scale) scale[t * nb + b] = sc;
  if (sum) sum[t * nb + b] = s;
}

// ------------------------------------------------------------------------------------------------ elementwise

// one block (256 threads) per segment
__global__ void __launch_bounds__(256) k_rms_norm(const float* x, const float* __restrict__ w, float* y, std::int64_t n, std::int64_t n_seg,
                                                  std::int64_t stride, std::int64_t ld, float eps, int l2) {
  __shared__ double sh[33];
  const std::int64_t t = blockIdx.x / n_seg, g = blockIdx.x % n_seg;
  const float* xs = x + t * ld + g * stride;
  float* ys = y + t * ld + g * stride;
  double part = 0.0;
  for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) part += static_cast<double>(mul(xs[i], xs[i]));
  const double sum = block_sum_d(part, sh);
  const float nf = static_cast<float>(n);
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  if (l2) {
    const float scale = quo(1.0f, sqrtf(add(mean, quo(eps, nf))));
    const float post = quo(1.0f, sqrtf(nf));
    for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) ys[i] = mul(mul(xs[i], scale), post);
  } else {
    const float scale = quo(1.0f, sqrtf(add(mean, eps)));
    for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) ys[i] = w ? mul(mul(xs[i], scale), w[i]) : mul(xs[i], scale);
  }
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
__global__ void k_rope(float* v, std::int64_t n_heads, std::int64_t hs, std::int64_t ld, std::int64_t rows, std::int64_t pos0, int n_rot,
                       float theta_scale, int s0, int s1, int s2, int s3, int neox, const float* __restrict__ ff) {
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

// ------------------------------------------------------------------------------------------------ attention

// scores[t][h][u] = (q . k_u) * scale for u <= p0 + t; one warp per position
__global__ void __launch_bounds__(256) k_attn_scores(const float* __restrict__ q, std::int64_t qs, std::int64_t qh, const float* __restrict__ K,
                                                     std::int64_t kv_w, std::int64_t nh, std::int64_t group, std::int64_t hd, std::int64_t p0,
                                                     std::int64_t L, float scale, float* __restrict__ sc) {
  const std::int64_t h = blockIdx.y, t = blockIdx.z;
  const std::int64_t u = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = static_cast<int>(threadIdx.x & 31);
  if (u > p0 + t) return;
  const float* qq = q + t * qs + h * qh;
  const float* kk = K + u * kv_w + (h / group) * hd;
  float d = 0.0f;
  for (std::int64_t i = lane; i < hd; i += 32) d += qq[i] * kk[i];
  d = warp_sum(d);
  if (lane == 0) sc[(t * nh + h) * L + u] = mul(d, scale);
}

// softmax of each (t, h) row of scores over 0 .. p0 + t: max, exp, sum in double, times 1 / sum (model::softmax)
__global__ void __launch_bounds__(256) k_attn_softmax(float* __restrict__ sc, std::int64_t nh, std::int64_t p0, std::int64_t L) {
  __shared__ float shm[33];
  __shared__ double shd[33];
  const std::int64_t row = blockIdx.x, t = row / nh, len = p0 + t + 1;
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
                           int gate, float* __restrict__ out) {
  const std::int64_t row = blockIdx.x, t = row / nh, h = row % nh, len = p0 + t + 1;
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
__global__ void k_stamp(volatile unsigned long long* p) { *p = gtimer(); }
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
  k_matmul<<<blocks_for(n_out, 4), 128, 0, st(s)>>>(type, static_cast<const std::uint8_t*>(w), row_bytes, n_out, k, x, n_tok, ldx, y, ldy, tb);
  return launched("matmul");
}

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
  k_act_q8x32<<<blocks_for(rows * (n / 32), 128), 128, 0, st(s)>>>(x, n, rows, ldx, rot32 ? 1 : 0, norm, xr, ldr, q, scale, sum);
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
          const int sections[4], bool neox, const float* freq_factors, Stream s) {
  if (n_rot <= 0 || n_rot % 2 != 0 || n_heads <= 0 || !v) return fail("rope: bad shape");
  if (rows <= 0) return true;
  const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));  // as model::rope_cache, on the host
  const int s0 = sections ? sections[0] : 0, s1 = sections ? sections[1] : 0, s2 = sections ? sections[2] : 0, s3 = sections ? sections[3] : 0;
  const std::int64_t total = rows * n_heads * (n_rot / 2);
  k_rope<<<blocks_for(total, 128), 128, 0, st(s)>>>(v, n_heads, head_stride, ld, rows, pos0, n_rot, theta_scale, s0, s1, s2, s3, neox ? 1 : 0, freq_factors);
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

bool router_topk(const float* logits, std::int64_t n_expert, std::int64_t rows, int n_used, float weights_scale, std::int32_t* sel, float* wsel,
                 Stream s) {
  if (n_expert <= 0 || n_used <= 0 || n_used > n_expert || !logits || !sel || !wsel) return fail("router_topk: bad shape");
  if (rows <= 0) return true;
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
               Stream s) {
  if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > 1024 || kv_w < n_head_kv * head_dim || p0 < 0 ||
      q_head < (gate ? 2 : 1) * head_dim || !q || !K || !V || !out || !scratch)
    return fail("attention: bad shape or pointer");
  if (n_tok <= 0) return true;
  if (n_tok > 65535 || n_head > 65535) return fail("attention: at most 65535 tokens and heads per call (grid limit; split a longer prompt)");
  const std::int64_t L = p0 + n_tok, group = n_head / n_head_kv;
  const dim3 g1(blocks_for(L, 8), static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok));
  k_attn_scores<<<g1, 256, 0, st(s)>>>(q, q_stride, q_head, K, kv_w, n_head, group, head_dim, p0, L, scale, scratch);
  if (!launched("attention scores")) return false;
  k_attn_softmax<<<static_cast<unsigned>(n_tok * n_head), 256, 0, st(s)>>>(scratch, n_head, p0, L);
  if (!launched("attention softmax")) return false;
  const int tpb = head_dim >= 256 ? 256 : 128;
  const dim3 g3(static_cast<unsigned>(n_tok * n_head), blocks_for(head_dim, tpb));
  k_attn_out<<<g3, tpb, 0, st(s)>>>(scratch, V, kv_w, n_head, group, head_dim, p0, L, q, q_stride, q_head, gate ? 1 : 0, out);
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

}  // namespace pelias::kern::gpu

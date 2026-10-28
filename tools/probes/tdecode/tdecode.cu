// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// tdecode.cu - the 1-bit decoder (T-decode) on the GPU (docs/PELIAS_7_T_H_PLAN.md, section 6, steps 4-5; target sm_89).
//   tdecode mma    : rate of the 1-bit tensor path (mma m16n8k256 b1 .and.popc) against the int8 one (m16n8k32 s8)
//   tdecode gemv   : single-token GEMV (decode, memory-bound) on real dense shapes, every route checked against a
//                    double-precision host reference and timed:
//     Q8Z        Q8_Z values (int8, 8.25 bpw): the baseline
//     ZT-fold    Q7_ZT, the 8th bit folded into the int8 value (q = 2b + h - 127), dp4a
//     ZT-1bit    Q7_ZT, base by dp4a + the 8th bit by AND + popcount against the activation's bit planes
//     ZH-live f  Q8_Z stored split (7-bit base + H plane grouped by input block); the 8th bit read only on the fraction
//                f of input blocks chosen once per token by activation energy (shared by every row), the rest filled
//                with 0.5 through the activation block sum; f = 0, 0.25, 1
//   tdecode prep   : activation quantization per 32 + the 8 bit planes by warp ballot, checked against the host
//   tdecode gemv2  : version 2 kernels (lanes split each row), cold L2 (64 MiB written between launches), median of 50
// Device layouts are the engine's (repacked at load from the GGUF blocks): headers and H bits in their own arrays.
// Build (WSL): nvcc -std=c++17 -O3 -arch=sm_89 tdecode.cu -o tdecode
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

// ------------------------------------------------------------------------------------------------ mma rates
__global__ void mma_b1_loop(int iters, int* sink) {
  uint32_t a[4][4], b[4][2];
  int c[4][4];
  for (int s = 0; s < 4; ++s) {
    for (int i = 0; i < 4; ++i) a[s][i] = (threadIdx.x + 1u) * 2654435761u ^ (s * 977u + i * 131u);
    for (int i = 0; i < 2; ++i) b[s][i] = (threadIdx.x + 7u) * 40503u ^ (s * 31u + i);
    for (int i = 0; i < 4; ++i) c[s][i] = 0;
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int s = 0; s < 4; ++s)
      asm volatile(
          "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+r"(c[s][0]), "+r"(c[s][1]), "+r"(c[s][2]), "+r"(c[s][3])
          : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]), "r"(b[s][0]), "r"(b[s][1]));
  }
  int t = 0;
  for (int s = 0; s < 4; ++s) t += c[s][0] + c[s][1] + c[s][2] + c[s][3];
  if (t == 0x7fffffff) sink[0] = t;
}

__global__ void mma_s8_loop(int iters, int* sink) {
  uint32_t a[4][4], b[4][2];
  int c[4][4];
  for (int s = 0; s < 4; ++s) {
    for (int i = 0; i < 4; ++i) a[s][i] = (threadIdx.x + 1u) * 2654435761u ^ (s * 977u + i * 131u);
    for (int i = 0; i < 2; ++i) b[s][i] = (threadIdx.x + 7u) * 40503u ^ (s * 31u + i);
    for (int i = 0; i < 4; ++i) c[s][i] = 0;
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int s = 0; s < 4; ++s)
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+r"(c[s][0]), "+r"(c[s][1]), "+r"(c[s][2]), "+r"(c[s][3])
          : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]), "r"(b[s][0]), "r"(b[s][1]));
  }
  int t = 0;
  for (int s = 0; s < 4; ++s) t += c[s][0] + c[s][1] + c[s][2] + c[s][3];
  if (t == 0x7fffffff) sink[0] = t;
}

static int run_mma() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  std::printf("device %s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
  int* sink;
  CK(cudaMalloc(&sink, 4));
  const int blocks = p.multiProcessorCount * 4, threads = 256, iters = 1 << 14;
  const double warps = blocks * (threads / 32.0), mmas = warps * iters * 4.0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_b1 = 0, ms_s8 = 0;
  for (int rep = 0; rep < 2; ++rep) {  // the first pass warms up
    CK(cudaEventRecord(e0));
    mma_b1_loop<<<blocks, threads>>>(iters, sink);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms_b1, e0, e1));
    CK(cudaEventRecord(e0));
    mma_s8_loop<<<blocks, threads>>>(iters, sink);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms_s8, e0, e1));
  }
  CK(cudaGetLastError());
  const double b1_macs = mmas * 16 * 8 * 256 / (ms_b1 * 1e-3), s8_macs = mmas * 16 * 8 * 32 / (ms_s8 * 1e-3);
  std::printf("mma b1 m16n8k256 .and.popc: %.2f ms, %.1f T bit-MAC/s (%.0f mma per us)\n", ms_b1, b1_macs / 1e12, mmas / (ms_b1 * 1e3));
  std::printf("mma s8 m16n8k32:            %.2f ms, %.1f T int8-MAC/s (%.0f mma per us)\n", ms_s8, s8_macs / 1e12, mmas / (ms_s8 * 1e3));
  std::printf("bit-MAC per int8-MAC: %.2fx (one 8-bit x 8-bit product = 64 bit-products, so the 1-bit path pays off above 64x... per weight plane x activation planes: 8 activation planes per weight plane -> %.2fx an int8 MAC per weight bit)\n",
              b1_macs / s8_macs, b1_macs / s8_macs / 8.0);
  CK(cudaFree(sink));
  return 0;
}

// ------------------------------------------------------------------------------------------------ activations
struct ActDev {
  int8_t* q;       // k
  float* scale;    // k / 32
  int* sum;        // k / 32
  uint32_t* plane; // k / 32 * 8
};

// one warp per 32-value block: amax, int8, sum, and the 8 bit planes of u = q + 128 by warp ballot
__global__ void act_prep(const float* x, int nblk, int8_t* q, float* scale, int* sum, uint32_t* plane) {
  const int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (w >= nblk) return;
  const float v = x[w * 32 + lane];
  float m = fabsf(v);
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float s = m > 0.f ? m / 127.f : 1.f;
  int qi = __float2int_rn(v * (1.f / s));
  qi = max(-127, min(127, qi));
  q[w * 32 + lane] = static_cast<int8_t>(qi);
  int t = qi;
  for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  const unsigned u = static_cast<unsigned>(qi + 128);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned bits = __ballot_sync(0xffffffffu, (u >> k) & 1u);
    if (lane == 0) plane[w * 8 + k] = bits;
  }
  if (lane == 0) {
    scale[w] = s;
    sum[w] = t;
  }
}

// ------------------------------------------------------------------------------------------------ GEMV kernels
// Mapping: one thread per output row (lane = row inside a warp), every thread walks the whole input; the activation and
// its planes sit in shared memory (the same for every row). hdr per row and 256-block: fp16 d (bits 0-15) + eight 6-bit
// levels (bits 16-63); scale of 32-block j = (d * lv_j) / 63.
__device__ __forceinline__ float zscale(uint64_t h, int j) {
  const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h & 0xffffu)));
  return (d * static_cast<float>((h >> (16 + 6 * j)) & 63u)) / 63.f;
}

// 32 seven-bit values of one 32-block (28 bytes, value i at bits 7i..7i+6) -> eight int8x4 words of (2b + h_i - 127),
// h_i = bit i of H (0 for the base alone). Each byte is built whole: adding h to packed bytes would carry out of a byte
// holding -1 (0xff).
__device__ __forceinline__ void unpack7(const uint32_t* wd, uint32_t H, int out[8]) {
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    int packed = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int i = 4 * g + t, bit = 7 * i, w = bit >> 5, sh = bit & 31;
      uint32_t v = wd[w] >> sh;
      if (sh > 25) v |= wd[w + 1] << (32 - sh);
      const int b = static_cast<int>(v & 127u);
      packed |= ((2 * b + static_cast<int>((H >> i) & 1u) - 127) & 0xff) << (8 * t);
    }
    out[g] = packed;
  }
}

template <int K>
__device__ __forceinline__ void load_act(const int8_t* q, const float* sc, const int* sm, const uint32_t* pl, int8_t* sq, float* ssc,
                                         int* ssm, uint32_t* spl) {
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) reinterpret_cast<int*>(sq)[i] = reinterpret_cast<const int*>(q)[i];
  for (int i = threadIdx.x; i < K / 32; i += blockDim.x) {
    ssc[i] = sc[i];
    ssm[i] = sm[i];
  }
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) spl[i] = pl[i];  // K/32 blocks x 8 planes = K/4 words
  __syncthreads();
}

template <int K>
__global__ void gemv_q8z(const int8_t* __restrict__ v, const uint64_t* __restrict__ hdr, int n, const int8_t* q, const float* sc,
                         const int* sm, const uint32_t* pl, float* y) {
  __shared__ __align__(16) int8_t sq[K];
  __shared__ float ssc[K / 32];
  __shared__ int ssm[K / 32];
  __shared__ uint32_t spl[K / 4];
  load_act<K>(q, sc, sm, pl, sq, ssc, ssm, spl);
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  float acc = 0.f;
  for (int B = 0; B < K / 256; ++B) {
    const uint64_t h = hdr[static_cast<size_t>(r) * (K / 256) + B];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int4* wv = reinterpret_cast<const int4*>(v + static_cast<size_t>(r) * K + B * 256 + j * 32);
      const int4 w0 = wv[0], w1 = wv[1];
      const int* xv = reinterpret_cast<const int*>(sq + B * 256 + j * 32);
      int p = 0;
      p = __dp4a(w0.x, xv[0], p); p = __dp4a(w0.y, xv[1], p); p = __dp4a(w0.z, xv[2], p); p = __dp4a(w0.w, xv[3], p);
      p = __dp4a(w1.x, xv[4], p); p = __dp4a(w1.y, xv[5], p); p = __dp4a(w1.z, xv[6], p); p = __dp4a(w1.w, xv[7], p);
      acc += zscale(h, j) * ssc[B * 8 + j] * static_cast<float>(p);
    }
  }
  y[r] = acc;
}

// Q7_ZT: base (224 bytes per 256-block), hdr, mask (1 byte per 256-block, two bits set), H (two words per 256-block)
template <int K, bool ONE_BIT>
__global__ void gemv_q7zt(const uint8_t* __restrict__ base, const uint64_t* __restrict__ hdr, const uint8_t* __restrict__ mask,
                          const uint32_t* __restrict__ hb, int n, const int8_t* q, const float* sc, const int* sm, const uint32_t* pl,
                          float* y) {
  __shared__ __align__(16) int8_t sq[K];
  __shared__ float ssc[K / 32];
  __shared__ int ssm[K / 32];
  __shared__ uint32_t spl[K / 4];
  load_act<K>(q, sc, sm, pl, sq, ssc, ssm, spl);
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  float acc = 0.f;
  for (int B = 0; B < K / 256; ++B) {
    const size_t rb = static_cast<size_t>(r) * (K / 256) + B;
    const uint64_t h = hdr[rb];
    const unsigned m = mask[rb];
    int slot = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t* wd = reinterpret_cast<const uint32_t*>(base + rb * 224 + j * 28);
      uint32_t w7[8];
#pragma unroll
      for (int i = 0; i < 7; ++i) w7[i] = wd[i];
      w7[7] = 0;
      const bool gated = (m >> j) & 1u;
      uint32_t H = 0;
      if (gated) H = hb[rb * 2 + slot++];
      int bw[8];
      unpack7(w7, ONE_BIT ? 0u : H, bw);  // fold route: the 8th bit goes into the int8 value here
      const int* xv = reinterpret_cast<const int*>(sq + B * 256 + j * 32);
      int p = 0;
      {
#pragma unroll
        for (int g = 0; g < 8; ++g) p = __dp4a(bw[g], xv[g], p);
        if (ONE_BIT && gated) {
          const uint32_t* pp = spl + (B * 8 + j) * 8;
          int hc = -128 * __popc(H);
#pragma unroll
          for (int k = 0; k < 8; ++k) hc += __popc(H & pp[k]) << k;
          p += hc;
        }
      }
      acc += zscale(h, j) * ssc[B * 8 + j] * static_cast<float>(p);
    }
  }
  y[r] = acc;
}

// ZH-live: Q8_Z values stored as a 7-bit base (b = (q + 127) >> 1, 224 bytes per 256) + an H plane grouped by input
// block (hp[jglobal * n + r], one word per row: coalesced across rows); sel[jglobal] = 1 reads H (1-bit route), else
// the 0.5 fill: 2P = 2 * base_dot + sum(x)
template <int K>
__global__ void gemv_zh(const uint8_t* __restrict__ base, const uint64_t* __restrict__ hdr, const uint32_t* __restrict__ hp,
                        const uint8_t* __restrict__ sel, int n, const int8_t* q, const float* sc, const int* sm, const uint32_t* pl,
                        float* y) {
  __shared__ __align__(16) int8_t sq[K];
  __shared__ float ssc[K / 32];
  __shared__ int ssm[K / 32];
  __shared__ uint32_t spl[K / 4];
  __shared__ uint8_t ssel[K / 32];
  for (int i = threadIdx.x; i < K / 32; i += blockDim.x) ssel[i] = sel[i];
  load_act<K>(q, sc, sm, pl, sq, ssc, ssm, spl);
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  float acc = 0.f;
  for (int B = 0; B < K / 256; ++B) {
    const size_t rb = static_cast<size_t>(r) * (K / 256) + B;
    const uint64_t h = hdr[rb];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int jg = B * 8 + j;
      const uint32_t* wd = reinterpret_cast<const uint32_t*>(base + rb * 224 + j * 28);
      uint32_t w7[8];
#pragma unroll
      for (int i = 0; i < 7; ++i) w7[i] = wd[i];
      w7[7] = 0;
      int bw[8];
      unpack7(w7, 0u, bw);
      const int* xv = reinterpret_cast<const int*>(sq + jg * 32);
      int p = 0;
#pragma unroll
      for (int g = 0; g < 8; ++g) p = __dp4a(bw[g], xv[g], p);
      float pf;
      if (ssel[jg]) {
        const uint32_t H = hp[static_cast<size_t>(jg) * n + r];
        const uint32_t* pp = spl + jg * 8;
        int hc = -128 * __popc(H);
#pragma unroll
        for (int k = 0; k < 8; ++k) hc += __popc(H & pp[k]) << k;
        pf = static_cast<float>(p + hc);
      } else {
        pf = static_cast<float>(p) + 0.5f * static_cast<float>(ssm[jg]);
      }
      acc += zscale(h, j) * ssc[jg] * pf;
    }
  }
  y[r] = acc;
}

// ------------------------------------------------------------------------------------------------ v2: lanes split K
// Version 2 (2026-10-28 17:3x). The first run gave one thread per output row (~1.4 warps per scheduler, a latency-bound
// chain; Fable's read-only review agreed) and timed L2-warm. Here each lane takes 4 consecutive 32-blocks (128 values)
// of one row, K / 128 lanes per row, shuffle reduction; activations read straight from global (shared through L1 by
// every row on the SM). Same arrays as version 1, the ZH H plane kept block-major (hp[block][row]: a 32-byte sector
// holds one block for 8 rows, so a gate fetches exactly the selected sectors). Identity:
// sum (2b - 127) x = 2 dp4a(b, x) - 127 sum(x); b <= 127 fits a signed byte.
// ROUTE 0: Q8_Z. 1: Q7_ZT, H as bytes into dp4a (int8 route). 2: Q7_ZT, H by AND + popcount (T-decode).
// 3: ZH arrays, H read where sel[block] (1-bit), else the 0.5 fill.
template <int K, int ROUTE>
__global__ void gemv_w(const int8_t* __restrict__ v8, const uint8_t* __restrict__ base, const uint64_t* __restrict__ hdr,
                       const uint8_t* __restrict__ mask, const uint32_t* __restrict__ hb, const uint32_t* __restrict__ hp,
                       const uint8_t* __restrict__ sel, int n, const int8_t* __restrict__ q, const float* __restrict__ sc,
                       const int* __restrict__ sm, const uint32_t* __restrict__ pl, float* __restrict__ y) {
  constexpr int LPR = K / 128;
  static_assert(LPR >= 1 && LPR <= 32 && (32 % LPR) == 0, "K / 128 lanes per row must divide a warp");
  const int gt = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = gt / LPR, part = gt % LPR;
  float acc = 0.f;
  if (r < n) {
    const int j0 = part * 4;  // first 32-block of this lane (row-global index)
    int xw[32];
    {
      const int4* x4 = reinterpret_cast<const int4*>(q + j0 * 32);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int4 t = __ldg(x4 + i);
        xw[4 * i] = t.x; xw[4 * i + 1] = t.y; xw[4 * i + 2] = t.z; xw[4 * i + 3] = t.w;
      }
    }
    if (ROUTE == 0) {
      const int4* w4 = reinterpret_cast<const int4*>(v8 + static_cast<size_t>(r) * K + j0 * 32);
      int4 w[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) w[i] = w4[i];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int jg = j0 + jj;
        int p = 0;
        p = __dp4a(w[2 * jj].x, xw[8 * jj], p); p = __dp4a(w[2 * jj].y, xw[8 * jj + 1], p);
        p = __dp4a(w[2 * jj].z, xw[8 * jj + 2], p); p = __dp4a(w[2 * jj].w, xw[8 * jj + 3], p);
        p = __dp4a(w[2 * jj + 1].x, xw[8 * jj + 4], p); p = __dp4a(w[2 * jj + 1].y, xw[8 * jj + 5], p);
        p = __dp4a(w[2 * jj + 1].z, xw[8 * jj + 6], p); p = __dp4a(w[2 * jj + 1].w, xw[8 * jj + 7], p);
        acc += zscale(hdr[static_cast<size_t>(r) * (K / 256) + jg / 8], jg % 8) * __ldg(sc + jg) * static_cast<float>(p);
      }
    } else {
      uint32_t wd[29];
      {
        const int4* b4 = reinterpret_cast<const int4*>(base + static_cast<size_t>(r) * (K / 256) * 224 + part * 112);
#pragma unroll
        for (int i = 0; i < 7; ++i) {
          const int4 t = b4[i];
          wd[4 * i] = static_cast<uint32_t>(t.x); wd[4 * i + 1] = static_cast<uint32_t>(t.y);
          wd[4 * i + 2] = static_cast<uint32_t>(t.z); wd[4 * i + 3] = static_cast<uint32_t>(t.w);
        }
        wd[28] = 0u;
      }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int jg = j0 + jj, B = jg / 8, j = jg % 8;
        const size_t rb = static_cast<size_t>(r) * (K / 256) + B;
        int pb = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          uint32_t packed = 0;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const int i = 32 * jj + 4 * g + t, bit = 7 * i;
            const uint32_t v = __funnelshift_r(wd[bit >> 5], wd[(bit >> 5) + 1], bit & 31) & 127u;
            packed |= v << (8 * t);
          }
          pb = __dp4a(static_cast<int>(packed), xw[8 * jj + g], pb);
        }
        const int sx = __ldg(sm + jg);
        bool readH = false;
        uint32_t H = 0;
        if (ROUTE == 1 || ROUTE == 2) {
          const unsigned m = mask[rb];
          if ((m >> j) & 1u) {
            readH = true;
            H = hb[rb * 2 + __popc(m & ((1u << j) - 1u))];
          }
        } else if (__ldg(sel + jg)) {
          readH = true;
          H = hp[static_cast<size_t>(jg) * n + r];
        }
        int hx = 0;
        if (readH) {
          if (ROUTE == 1) {
#pragma unroll
            for (int g = 0; g < 8; ++g) {
              const uint32_t nib = (H >> (4 * g)) & 0xfu;
              hx = __dp4a(static_cast<int>((nib * 0x00204081u) & 0x01010101u), xw[8 * jj + g], hx);
            }
          } else {
            const uint32_t* pp = pl + jg * 8;
            hx = -128 * __popc(H);
#pragma unroll
            for (int k = 0; k < 8; ++k) hx += __popc(H & __ldg(pp + k)) << k;
          }
        }
        const int P = 2 * pb - 127 * sx;
        const int P2 = (ROUTE == 3 && !readH) ? 2 * P + sx : 2 * (P + hx);  // twice the partial sum; 0.5 fill when H is not read
        acc += zscale(hdr[rb], j) * __ldg(sc + jg) * (0.5f * static_cast<float>(P2));
      }
    }
  }
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (r < n && part == 0) y[r] = acc;
}

// ------------------------------------------------------------------------------------------------ host side
static bool g_cold = false, g_v1 = true, g_v2 = false;
static void* g_flush = nullptr;
static const size_t kFlushBytes = 64ull << 20;

static uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
static float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }

struct Host {
  int n = 0, k = 0;
  std::vector<uint64_t> hdr;             // n * k/256
  std::vector<int> q8;                   // n * k   (Q8_Z / ZH values, -127..127)
  std::vector<int> q7;                   // n * k   (Q7_ZT values: gated -127..127, ungated odd)
  std::vector<uint8_t> mask;             // n * k/256
  std::vector<float> x;                  // k
};

static double zscale_h(uint64_t h, int j) {
  const float d = h2f(static_cast<uint16_t>(h & 0xffffu));
  return static_cast<double>((d * static_cast<float>((h >> (16 + 6 * j)) & 63u)) / 63.f);
}

static void make_host(Host& H, int n, int k, std::mt19937& rng) {
  H.n = n;
  H.k = k;
  const int nB = k / 256;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> lvd(20, 63), bit(0, 1), pick(0, 7);
  H.hdr.resize(static_cast<size_t>(n) * nB);
  H.q8.resize(static_cast<size_t>(n) * k);
  H.q7.resize(static_cast<size_t>(n) * k);
  H.mask.resize(static_cast<size_t>(n) * nB);
  for (int r = 0; r < n; ++r)
    for (int B = 0; B < nB; ++B) {
      uint64_t h = f2h(0.0004f * (1.f + 0.2f * std::fabs(nd(rng))));
      for (int j = 0; j < 8; ++j) h |= static_cast<uint64_t>(lvd(rng)) << (16 + 6 * j);
      H.hdr[static_cast<size_t>(r) * nB + B] = h;
      int g1 = pick(rng), g2 = pick(rng);
      while (g2 == g1) g2 = pick(rng);
      H.mask[static_cast<size_t>(r) * nB + B] = static_cast<uint8_t>((1u << g1) | (1u << g2));
      for (int i = 0; i < 256; ++i) {
        const int j = i / 32;
        int v = static_cast<int>(std::lrint(nd(rng) * 40.f));
        v = std::max(-127, std::min(127, v));
        const size_t at = static_cast<size_t>(r) * k + B * 256 + i;
        H.q8[at] = v;
        const bool gated = (H.mask[static_cast<size_t>(r) * nB + B] >> j) & 1u;
        H.q7[at] = gated ? v : (2 * ((v + 127) >> 1) - 127);
      }
    }
  H.x.resize(static_cast<size_t>(k));
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  for (int i = 0; i < k; ++i) H.x[static_cast<size_t>(i)] = nd(rng) * std::exp(0.8f * nd(rng) * (i % 32 == 0 ? 1.f : 0.f));
  for (int b = 0; b < k / 32; ++b) {  // uneven energy per 32-block (heavy-tailed), as in the sandbox check
    const float e = std::exp(0.8f * nd(rng));
    for (int i = 0; i < 32; ++i) H.x[static_cast<size_t>(b * 32 + i)] *= e;
  }
}

// host activation quantization (same arithmetic as act_prep)
static void act_host(const std::vector<float>& x, std::vector<int>& q, std::vector<float>& sc, std::vector<int>& sm, std::vector<uint32_t>& pl) {
  const int k = static_cast<int>(x.size());
  q.assign(static_cast<size_t>(k), 0);
  sc.assign(static_cast<size_t>(k / 32), 0.f);
  sm.assign(static_cast<size_t>(k / 32), 0);
  pl.assign(static_cast<size_t>(k / 4), 0u);
  for (int b = 0; b < k / 32; ++b) {
    float m = 0.f;
    for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(x[static_cast<size_t>(b * 32 + i)]));
    const float s = m > 0.f ? m / 127.f : 1.f;
    sc[static_cast<size_t>(b)] = s;
    for (int i = 0; i < 32; ++i) {
      int v = static_cast<int>(std::nearbyint(x[static_cast<size_t>(b * 32 + i)] * (1.f / s)));
      v = std::max(-127, std::min(127, v));
      q[static_cast<size_t>(b * 32 + i)] = v;
      sm[static_cast<size_t>(b)] += v;
      const unsigned u = static_cast<unsigned>(v + 128);
      for (int kk = 0; kk < 8; ++kk) pl[static_cast<size_t>(b * 8 + kk)] |= ((u >> kk) & 1u) << i;
    }
  }
}

template <typename T>
static T* up(const std::vector<T>& v) {
  T* d;
  CK(cudaMalloc(&d, v.size() * sizeof(T)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static std::vector<uint8_t> pack7(const std::vector<int>& vals, int n, int k, bool from_q8) {
  // base bytes: per row and 256-block, 224 bytes; value i of the block at bits 7i..7i+6; b = (q + 127) >> 1
  std::vector<uint8_t> out(static_cast<size_t>(n) * (k / 256) * 224, 0);
  for (int r = 0; r < n; ++r)
    for (int B = 0; B < k / 256; ++B) {
      uint8_t* o = out.data() + (static_cast<size_t>(r) * (k / 256) + B) * 224;
      for (int i = 0; i < 256; ++i) {
        const int q = vals[static_cast<size_t>(r) * k + B * 256 + i];
        const unsigned b = static_cast<unsigned>(from_q8 ? (q + 127) >> 1 : (q + 127) >> 1);
        const int bitpos = 7 * i;
        for (int t = 0; t < 7; ++t)
          if ((b >> t) & 1u) o[(bitpos + t) >> 3] |= static_cast<uint8_t>(1u << ((bitpos + t) & 7));
      }
    }
  return out;
}

template <int K>
static int run_shape(int n, std::mt19937& rng) {
  Host H;
  make_host(H, n, K, rng);
  const int nB = K / 256, nb = K / 32;
  std::vector<int> aq, asm_;
  std::vector<float> asc;
  std::vector<uint32_t> apl;
  act_host(H.x, aq, asc, asm_, apl);
  // device activation via act_prep, checked against the host
  float* dx = up(H.x);
  int8_t* dq;
  float* dsc;
  int* dsm;
  uint32_t* dpl;
  CK(cudaMalloc(&dq, K));
  CK(cudaMalloc(&dsc, nb * 4));
  CK(cudaMalloc(&dsm, nb * 4));
  CK(cudaMalloc(&dpl, nb * 32));
  act_prep<<<(nb + 7) / 8, 256>>>(dx, nb, dq, dsc, dsm, dpl);
  CK(cudaDeviceSynchronize());
  {
    std::vector<int8_t> gq(K);
    std::vector<uint32_t> gpl(static_cast<size_t>(nb) * 8);
    std::vector<int> gsm(static_cast<size_t>(nb));
    CK(cudaMemcpy(gq.data(), dq, K, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gpl.data(), dpl, nb * 32, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gsm.data(), dsm, nb * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < K; ++i) bad += gq[static_cast<size_t>(i)] != aq[static_cast<size_t>(i)];
    for (int i = 0; i < nb * 8; ++i) bad += gpl[static_cast<size_t>(i)] != apl[static_cast<size_t>(i)];
    for (int i = 0; i < nb; ++i) bad += gsm[static_cast<size_t>(i)] != asm_[static_cast<size_t>(i)];
    std::printf("  activation prep (warp ballot planes) vs host: %s\n", bad ? "MISMATCH" : "identical");
    if (bad) return 1;
  }
  // weights on the device
  std::vector<int8_t> v8(H.q8.size());
  for (size_t i = 0; i < v8.size(); ++i) v8[i] = static_cast<int8_t>(H.q8[i]);
  int8_t* d_v8 = up(v8);
  uint64_t* d_hdr = up(H.hdr);
  std::vector<uint8_t> base7 = pack7(H.q7, n, K, false), basez = pack7(H.q8, n, K, true);
  uint8_t* d_b7 = up(base7);
  uint8_t* d_bz = up(basez);
  uint8_t* d_mask = up(H.mask);
  std::vector<uint32_t> hb(static_cast<size_t>(n) * nB * 2, 0u), hp(static_cast<size_t>(nb) * n, 0u);
  for (int r = 0; r < n; ++r)
    for (int B = 0; B < nB; ++B) {
      int slot = 0;
      const unsigned m = H.mask[static_cast<size_t>(r) * nB + B];
      for (int j = 0; j < 8; ++j) {
        uint32_t w7 = 0, wz = 0;
        for (int i = 0; i < 32; ++i) {
          const size_t at = static_cast<size_t>(r) * K + B * 256 + j * 32 + i;
          w7 |= static_cast<uint32_t>((H.q7[at] + 127) & 1) << i;
          wz |= static_cast<uint32_t>((H.q8[at] + 127) & 1) << i;
        }
        if ((m >> j) & 1u) hb[(static_cast<size_t>(r) * nB + B) * 2 + slot++] = w7;
        hp[static_cast<size_t>(B * 8 + j) * n + r] = wz;
      }
    }
  uint32_t* d_hb = up(hb);
  uint32_t* d_hp = up(hp);
  // the shared live gate: input blocks by activation energy, top 25 %
  std::vector<double> en(static_cast<size_t>(nb), 0.0);
  for (int b = 0; b < nb; ++b)
    for (int i = 0; i < 32; ++i) {
      const double xv = aq[static_cast<size_t>(b * 32 + i)] * static_cast<double>(asc[static_cast<size_t>(b)]);
      en[static_cast<size_t>(b)] += xv * xv;
    }
  std::vector<int> order(static_cast<size_t>(nb));
  for (int i = 0; i < nb; ++i) order[static_cast<size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return en[static_cast<size_t>(a)] > en[static_cast<size_t>(b)]; });
  std::vector<uint8_t> sel0(static_cast<size_t>(nb), 0), sel25(static_cast<size_t>(nb), 0), sel100(static_cast<size_t>(nb), 1);
  for (int i = 0; i < nb / 4; ++i) sel25[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
  uint8_t* d_s0 = up(sel0);
  uint8_t* d_s25 = up(sel25);
  uint8_t* d_s100 = up(sel100);
  float* d_y;
  CK(cudaMalloc(&d_y, n * 4));
  // host references (double), per route
  auto ref = [&](int route, const std::vector<uint8_t>* sel) {
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int r = 0; r < n; ++r) {
      double acc = 0.0;
      for (int jg = 0; jg < nb; ++jg) {
        const int B = jg / 8, j = jg % 8;
        long long p2 = 0;
        for (int i = 0; i < 32; ++i) {
          const size_t at = static_cast<size_t>(r) * K + jg * 32 + i;
          const long long xq = aq[static_cast<size_t>(jg * 32 + i)];
          if (route == 0) p2 += 2LL * H.q8[at] * xq;
          else if (route == 1) p2 += 2LL * H.q7[at] * xq;
          else {
            const int q = H.q8[at], b = (q + 127) >> 1;
            p2 += (*sel)[static_cast<size_t>(jg)] ? 2LL * q * xq : (2LL * (2 * b - 127) + 1) * xq;  // 0.5 fill
          }
        }
        acc += zscale_h(H.hdr[static_cast<size_t>(r) * nB + B], j) * asc[static_cast<size_t>(jg)] * 0.5 * static_cast<double>(p2);
      }
      y[static_cast<size_t>(r)] = acc;
    }
    return y;
  };
  auto check_cold = [&](const char* name, const std::vector<double>& yr, double bytes, auto launch) {
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> yg(static_cast<size_t>(n));
    CK(cudaMemcpy(yg.data(), d_y, n * 4, cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    for (int r = 0; r < n; ++r) scale = std::max(scale, std::fabs(yr[static_cast<size_t>(r)]));
    for (int r = 0; r < n; ++r) worst = std::max(worst, std::fabs(yg[static_cast<size_t>(r)] - yr[static_cast<size_t>(r)]));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int reps = 50;
    std::vector<float> t(reps);
    for (int i = 0; i < reps; ++i) {
      CK(cudaMemsetAsync(g_flush, i & 0xff, kFlushBytes));  // 64 MiB written between launches: the 36 MB L2 holds none of the weights
      CK(cudaEventRecord(e0));
      launch();
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      CK(cudaEventElapsedTime(&t[static_cast<size_t>(i)], e0, e1));
    }
    std::sort(t.begin(), t.end());
    const double us = t[reps / 2] * 1e3;  // median
    std::printf("  %-30s %8.2f us cold  %7.1f GB/s  bytes %.2f MB (%.3f bits per weight)  max err %.2e of max |y| (%s)\n", name, us,
                bytes / (us * 1e3), bytes / 1e6, bytes * 8.0 / (static_cast<double>(n) * K), worst / scale, worst / scale < 1e-5 ? "ok" : "FAIL");
    return worst / scale < 1e-5 ? 0 : 1;
  };
  auto check = [&](const char* name, const std::vector<double>& yr, double bytes, auto launch) {
    if (g_cold) return check_cold(name, yr, bytes, launch);
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> yg(static_cast<size_t>(n));
    CK(cudaMemcpy(yg.data(), d_y, n * 4, cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    for (int r = 0; r < n; ++r) scale = std::max(scale, std::fabs(yr[static_cast<size_t>(r)]));
    for (int r = 0; r < n; ++r) worst = std::max(worst, std::fabs(yg[static_cast<size_t>(r)] - yr[static_cast<size_t>(r)]));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int reps = 200;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) launch();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = ms * 1e3 / reps;
    std::printf("  %-26s %8.2f us  %7.1f GB/s  bytes %.2f MB (%.3f bits per weight)  max err %.2e of max |y| (%s)\n", name, us, bytes / (us * 1e3),
                bytes / 1e6, bytes * 8.0 / (static_cast<double>(n) * K), worst / scale, worst / scale < 1e-5 ? "ok" : "FAIL");
    return worst / scale < 1e-5 ? 0 : 1;
  };
  const int T = 128, G = (n + T - 1) / T;
  std::printf("shape: %d outputs x %d inputs\n", n, K);
  int fails = 0;
  const double hdrB = static_cast<double>(n) * nB * 8;
  const double ztB = static_cast<double>(n) * nB * (224 + 1 + 8) + hdrB;
  const double zbB = static_cast<double>(n) * nB * 224 + hdrB;
  const std::vector<double> r8 = ref(0, nullptr), r7 = ref(1, nullptr), z0 = ref(2, &sel0), z25 = ref(2, &sel25), z100 = ref(2, &sel100);
  if (g_v1) {
  fails += check("Q8_Z (int8, baseline)", r8, static_cast<double>(n) * K + hdrB,
                 [&] { gemv_q8z<K><<<G, T>>>(d_v8, d_hdr, n, dq, dsc, dsm, dpl, d_y); });
  fails += check("Q7_ZT fold (int8 route)", r7, ztB,
                 [&] { gemv_q7zt<K, false><<<G, T>>>(d_b7, d_hdr, d_mask, d_hb, n, dq, dsc, dsm, dpl, d_y); });
  fails += check("Q7_ZT 1-bit (T-decode)", r7, ztB,
                 [&] { gemv_q7zt<K, true><<<G, T>>>(d_b7, d_hdr, d_mask, d_hb, n, dq, dsc, dsm, dpl, d_y); });
  fails += check("ZH live, H on 0 %", z0, zbB, [&] { gemv_zh<K><<<G, T>>>(d_bz, d_hdr, d_hp, d_s0, n, dq, dsc, dsm, dpl, d_y); });
  fails += check("ZH live, H on 25 % (shared)", z25, zbB + static_cast<double>(n) * (nb / 4) * 4,
                 [&] { gemv_zh<K><<<G, T>>>(d_bz, d_hdr, d_hp, d_s25, n, dq, dsc, dsm, dpl, d_y); });
  fails += check("ZH live, H on 100 %", z100, zbB + static_cast<double>(n) * nb * 4,
                 [&] { gemv_zh<K><<<G, T>>>(d_bz, d_hdr, d_hp, d_s100, n, dq, dsc, dsm, dpl, d_y); });
  }
  if (g_v2) {
    const int T2 = 128, G2 = (n * (K / 128) + T2 - 1) / T2;
    std::printf("  -- version 2 (lanes split each row, %d lanes per row)\n", K / 128);
    fails += check("v2 Q8_Z (int8, baseline)", r8, static_cast<double>(n) * K + hdrB,
                   [&] { gemv_w<K, 0><<<G2, T2>>>(d_v8, nullptr, d_hdr, nullptr, nullptr, nullptr, nullptr, n, dq, dsc, dsm, dpl, d_y); });
    fails += check("v2 Q7_ZT, H as int8 bytes", r7, ztB,
                   [&] { gemv_w<K, 1><<<G2, T2>>>(nullptr, d_b7, d_hdr, d_mask, d_hb, nullptr, nullptr, n, dq, dsc, dsm, dpl, d_y); });
    fails += check("v2 Q7_ZT, H by 1-bit decode", r7, ztB,
                   [&] { gemv_w<K, 2><<<G2, T2>>>(nullptr, d_b7, d_hdr, d_mask, d_hb, nullptr, nullptr, n, dq, dsc, dsm, dpl, d_y); });
    fails += check("v2 ZH arrays, H on 0 %", z0, zbB,
                   [&] { gemv_w<K, 3><<<G2, T2>>>(nullptr, d_bz, d_hdr, nullptr, nullptr, d_hp, d_s0, n, dq, dsc, dsm, dpl, d_y); });
    fails += check("v2 ZH arrays, H on 25 %", z25, zbB + static_cast<double>(n) * (nb / 4) * 4,
                   [&] { gemv_w<K, 3><<<G2, T2>>>(nullptr, d_bz, d_hdr, nullptr, nullptr, d_hp, d_s25, n, dq, dsc, dsm, dpl, d_y); });
    fails += check("v2 ZH arrays, H on 100 %", z100, zbB + static_cast<double>(n) * nb * 4,
                   [&] { gemv_w<K, 3><<<G2, T2>>>(nullptr, d_bz, d_hdr, nullptr, nullptr, d_hp, d_s100, n, dq, dsc, dsm, dpl, d_y); });
  }
  CK(cudaGetLastError());
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dq), static_cast<void*>(dsc), static_cast<void*>(dsm), static_cast<void*>(dpl),
                  static_cast<void*>(d_v8), static_cast<void*>(d_hdr), static_cast<void*>(d_b7), static_cast<void*>(d_bz), static_cast<void*>(d_mask),
                  static_cast<void*>(d_hb), static_cast<void*>(d_hp), static_cast<void*>(d_s0), static_cast<void*>(d_s25), static_cast<void*>(d_s100),
                  static_cast<void*>(d_y)})
    CK(cudaFree(p));
  return fails;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "mma") return run_mma();
  if (cmd == "gemv" || cmd == "prep" || cmd == "gemv2") {
    if (cmd == "gemv2") {  // version 2 only, cold L2, median of 50
      g_v1 = false;
      g_v2 = true;
      g_cold = true;
      CK(cudaMalloc(&g_flush, kFlushBytes));
    }
    std::mt19937 rng(12345);
    int fails = 0;
    fails += run_shape<2048>(8192, rng);  // attn_qkv-like: 2048 inputs, 8192 outputs
    fails += run_shape<4096>(2048, rng);  // 4096 inputs
    fails += run_shape<2048>(4096, rng);
    fails += run_shape<2048>(512, rng);   // shared-expert gate / up
    fails += run_shape<512>(2048, rng);   // shared-expert down
    std::printf("%s\n", fails ? "SOME ROUTES FAILED" : "all routes match the host reference");
    return fails ? 1 : 0;
  }
  std::fprintf(stderr, "usage: tdecode mma | gemv | gemv2\n");
  return 1;
}

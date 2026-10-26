// kernels.cu - the only file that needs nvcc. Target: sm_89 (Ada). PTX written by hand for mma.sync.
#include "kernels.h"

#include <cstdint>

namespace pk {
namespace {

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

// ---- operand generators: finite values with realistic bit activity (never NaN/Inf)
__device__ __forceinline__ uint32_t bf16_one(uint32_t r) {
  return (r & 0x8000u) | ((120u + ((r >> 7) % 7u)) << 7) | (r & 0x7Fu);  // |v| in [2^-7, 2^-1)
}
__device__ __forceinline__ uint32_t gen_bf16x2(uint32_t r) { return bf16_one(r) | (bf16_one(r >> 16) << 16); }
__device__ __forceinline__ uint32_t fp16_one(uint32_t r) {
  return (r & 0x8000u) | ((10u + ((r >> 10) % 5u)) << 10) | (r & 0x3FFu);  // |v| in [2^-5, 2^-1)
}
__device__ __forceinline__ uint32_t gen_fp16x2(uint32_t r) { return fp16_one(r) | (fp16_one(r >> 16) << 16); }
__device__ __forceinline__ uint32_t fp8_one(uint32_t b) {
  return (b & 0x80u) | ((4u + ((b >> 3) % 5u)) << 3) | (b & 7u);  // exponent 4..8: never the NaN code
}
__device__ __forceinline__ uint32_t gen_fp8x4(uint32_t r) {
  return fp8_one(r) | (fp8_one(r >> 8) << 8) | (fp8_one(r >> 16) << 16) | (fp8_one(r >> 24) << 24);
}

template <int KIND>
__device__ __forceinline__ uint32_t gen_operand(uint32_t r) {
  if constexpr (KIND == 0) return gen_bf16x2(r);
  else if constexpr (KIND == 1) return gen_fp16x2(r);
  else if constexpr (KIND == 2) return r;
  else return gen_fp8x4(r);
}

// ---- warp-level tensor instructions (one 16x8 tile per instruction)
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_fp16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)d; (void)a; (void)b;
  __trap();
#endif
}

// Eight independent accumulator chains per warp, operands held in registers: measures the tensor pipe, not memory.
template <int KIND>
__global__ void __launch_bounds__(kMmaThreads) k_mma(unsigned iters, uint32_t seed, float* sink) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[4];
  uint32_t b[kMmaChains][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = gen_operand<KIND>(mix32(seed + tid * 7919u + static_cast<uint32_t>(i)));
#pragma unroll
  for (int c = 0; c < kMmaChains; ++c) {
#pragma unroll
    for (int j = 0; j < 2; ++j)
      b[c][j] = gen_operand<KIND>(mix32(seed ^ (tid * 104729u + static_cast<uint32_t>(c * 31 + j + 17))));
  }
  if constexpr (KIND == 2) {
    int acc[kMmaChains][4];
#pragma unroll
    for (int c = 0; c < kMmaChains; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[c][j] = 0;
#pragma unroll 1
    for (unsigned it = 0; it < iters; ++it) {
#pragma unroll
      for (int c = 0; c < kMmaChains; ++c) mma_s8(acc[c], a, b[c]);
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < kMmaChains; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) s += acc[c][j];
    sink[tid] = __int_as_float(s);
  } else {
    float acc[kMmaChains][4];
#pragma unroll
    for (int c = 0; c < kMmaChains; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[c][j] = 0.f;
#pragma unroll 1
    for (unsigned it = 0; it < iters; ++it) {
#pragma unroll
      for (int c = 0; c < kMmaChains; ++c) {
        if constexpr (KIND == 0) mma_bf16(acc[c], a, b[c]);
        else if constexpr (KIND == 1) mma_fp16(acc[c], a, b[c]);
        else mma_e4m3(acc[c], a, b[c]);
      }
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kMmaChains; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) s += acc[c][j];
    sink[tid] = s;
  }
}

// ---- streaming kernels
__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned long long* sink) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  unsigned long long acc = 0;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = p[i];
    const uint4 b = p[i + stride];
    const uint4 c = p[i + 2 * stride];
    const uint4 d = p[i + 3 * stride];
    acc ^= (static_cast<unsigned long long>(a.x ^ b.x ^ c.x ^ d.x) << 32) | (a.y ^ b.y ^ c.y ^ d.y);
    acc += (static_cast<unsigned long long>(a.z ^ b.z ^ c.z ^ d.z) << 32) | (a.w ^ b.w ^ c.w ^ d.w);
  }
  for (; i < n; i += stride) {
    const uint4 a = p[i];
    acc ^= (static_cast<unsigned long long>(a.x ^ a.z) << 32) | (a.y ^ a.w);
  }
  if (acc == 0x9E3779B97F4A7C15ULL) *sink = acc;  // practically never true; keeps the loads alive
}

__global__ void k_write(uint4* __restrict__ p, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const uint4 v = make_uint4(0xA5A5A5A5u, 0x5A5A5A5Au, 0x0F0F0F0Fu, 0xF0F0F0F0u);
  for (; i < n; i += stride) p[i] = v;
}

__global__ void k_copy(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

// ---- canary GEMM (bf16 inputs, fp32 accumulate), one warp per 16x8 output tile
__device__ __forceinline__ uint32_t ld32(const uint16_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

__global__ void k_fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    const uint32_t r = mix32(seed + static_cast<uint32_t>(i) * 2654435761u);
    p[i] = static_cast<uint16_t>(bf16_one(r));
  }
}

__global__ void k_gemm_bf16(const uint16_t* __restrict__ A, const uint16_t* __restrict__ Bt, float* __restrict__ C,
                            int M, int N, int K) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane = threadIdx.x & 31;
  const int tilesN = N / 8;
  const int tm = warp / tilesN;
  const int tn = warp % tilesN;
  if (tm >= M / 16) return;
  const int g = lane >> 2;
  const int t = lane & 3;
  const uint16_t* a0 = A + static_cast<size_t>(tm * 16 + g) * K;
  const uint16_t* a1 = A + static_cast<size_t>(tm * 16 + g + 8) * K;
  const uint16_t* bp = Bt + static_cast<size_t>(tn * 8 + g) * K;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < K; k += 16) {
    uint32_t a[4];
    uint32_t b[2];
    a[0] = ld32(a0 + k + 2 * t);
    a[1] = ld32(a1 + k + 2 * t);
    a[2] = ld32(a0 + k + 2 * t + 8);
    a[3] = ld32(a1 + k + 2 * t + 8);
    b[0] = ld32(bp + k + 2 * t);
    b[1] = ld32(bp + k + 2 * t + 8);
    mma_bf16(c, a, b);
  }
  float* r0 = C + static_cast<size_t>(tm * 16 + g) * N + tn * 8 + 2 * t;
  float* r1 = C + static_cast<size_t>(tm * 16 + g + 8) * N + tn * 8 + 2 * t;
  r0[0] = c[0];
  r0[1] = c[1];
  r1[0] = c[2];
  r1[1] = c[3];
}

__global__ void __launch_bounds__(kHashThreads) k_hash(const float* __restrict__ C, size_t n,
                                                       unsigned long long* acc, float* partial) {
  __shared__ float sf[kHashThreads];
  __shared__ unsigned long long sw[kHashThreads / 32];
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float s = 0.f;
  unsigned long long h = 0;
  for (; i < n; i += stride) {
    const float v = C[i];
    s += v;  // fixed order per thread
    h += mix64(static_cast<unsigned long long>(i) * 0x9E3779B97F4A7C15ULL ^ __float_as_uint(v));
  }
  for (int off = 16; off > 0; off >>= 1) h += __shfl_down_sync(0xffffffffu, h, off);
  if ((threadIdx.x & 31) == 0) sw[threadIdx.x >> 5] = h;
  sf[threadIdx.x] = s;
  __syncthreads();
  for (int st = kHashThreads / 2; st > 0; st >>= 1) {  // fixed-order tree
    if (static_cast<int>(threadIdx.x) < st) sf[threadIdx.x] += sf[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    unsigned long long tot = 0;
    for (int w = 0; w < kHashThreads / 32; ++w) tot += sw[w];
    atomicAdd(acc, tot);  // integer addition: order does not matter
    partial[blockIdx.x] = sf[0];
  }
}

__global__ void __launch_bounds__(kHashBlocks) k_final(const float* __restrict__ partial,
                                                       const unsigned long long* __restrict__ acc,
                                                       unsigned long long* hash_out, float* sum_out, int slot) {
  __shared__ float sf[kHashBlocks];
  sf[threadIdx.x] = partial[threadIdx.x];
  __syncthreads();
  for (int st = kHashBlocks / 2; st > 0; st >>= 1) {
    if (static_cast<int>(threadIdx.x) < st) sf[threadIdx.x] += sf[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    hash_out[slot] = *acc;
    sum_out[slot] = sf[0];
  }
}

// ---- misc
__device__ __forceinline__ unsigned long long global_timer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void k_spin(unsigned long long ns, unsigned long long* sink) {
  const unsigned long long t0 = global_timer();
  const long long c0 = clock64();
  const long long cycle_cap = static_cast<long long>(ns) * 5;  // > any real clock in GHz: stops a stuck timer
  unsigned long long t = t0;
  do {
    t = global_timer();
  } while (t - t0 < ns && (clock64() - c0) < cycle_cap);
  if (t == 0xFFFFFFFFFFFFFFFFULL) *sink = t;
}

__global__ void k_bump(unsigned long long* p, unsigned inc) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *p += inc;
}
__global__ void k_empty() {}

__global__ void k_smem(int* out, int bytes) {
  extern __shared__ unsigned char sm[];
  if (threadIdx.x == 0) {
    sm[0] = 1;
    sm[bytes - 1] = 2;
    out[0] = sm[0] + sm[bytes - 1];
  }
}

}  // namespace

// ------------------------------------------------------------------ host wrappers
cudaError_t launch_mma(Mma k, int blocks, unsigned iters, float* sink, cudaStream_t s) {
  switch (k) {
    case Mma::BF16: k_mma<0><<<blocks, kMmaThreads, 0, s>>>(iters, 1u, sink); break;
    case Mma::FP16: k_mma<1><<<blocks, kMmaThreads, 0, s>>>(iters, 2u, sink); break;
    case Mma::INT8: k_mma<2><<<blocks, kMmaThreads, 0, s>>>(iters, 3u, sink); break;
    case Mma::FP8: k_mma<3><<<blocks, kMmaThreads, 0, s>>>(iters, 4u, sink); break;
  }
  return cudaGetLastError();
}

cudaError_t launch_read(const void* p, size_t n, unsigned long long* sink, int blocks, int threads, cudaStream_t s) {
  k_read<<<blocks, threads, 0, s>>>(static_cast<const uint4*>(p), n, sink);
  return cudaGetLastError();
}
cudaError_t launch_write(void* p, size_t n, int blocks, int threads, cudaStream_t s) {
  k_write<<<blocks, threads, 0, s>>>(static_cast<uint4*>(p), n);
  return cudaGetLastError();
}
cudaError_t launch_copy(void* dst, const void* src, size_t n, int blocks, int threads, cudaStream_t s) {
  k_copy<<<blocks, threads, 0, s>>>(static_cast<uint4*>(dst), static_cast<const uint4*>(src), n);
  return cudaGetLastError();
}

cudaError_t launch_fill_bf16(uint16_t* p, size_t n, uint32_t seed, cudaStream_t s) {
  k_fill_bf16<<<1024, 256, 0, s>>>(p, n, seed);
  return cudaGetLastError();
}
cudaError_t launch_gemm_bf16(const uint16_t* A, const uint16_t* Bt, float* C, int M, int N, int K, cudaStream_t s) {
  const int warps = (M / 16) * (N / 8);
  const int blocks = (warps * 32 + 127) / 128;
  k_gemm_bf16<<<blocks, 128, 0, s>>>(A, Bt, C, M, N, K);
  return cudaGetLastError();
}
cudaError_t launch_hash(const float* C, size_t n, unsigned long long* acc, float* partial, cudaStream_t s) {
  k_hash<<<kHashBlocks, kHashThreads, 0, s>>>(C, n, acc, partial);
  return cudaGetLastError();
}
cudaError_t launch_final(const float* partial, const unsigned long long* acc, unsigned long long* hash_out,
                         float* sum_out, int slot, cudaStream_t s) {
  k_final<<<1, kHashBlocks, 0, s>>>(partial, acc, hash_out, sum_out, slot);
  return cudaGetLastError();
}

cudaError_t launch_spin(unsigned long long ns, unsigned long long* sink, int blocks, cudaStream_t s) {
  k_spin<<<blocks, 128, 0, s>>>(ns, sink);
  return cudaGetLastError();
}
cudaError_t launch_bump(unsigned long long* p, unsigned inc, cudaStream_t s) {
  k_bump<<<1, 32, 0, s>>>(p, inc);
  return cudaGetLastError();
}
cudaError_t launch_empty(cudaStream_t s) {
  k_empty<<<1, 32, 0, s>>>();
  return cudaGetLastError();
}

cudaError_t smem_probe(int bytes, int* d_out) {
  cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(&k_smem), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    return e;
  }
  k_smem<<<1, 32, static_cast<size_t>(bytes)>>>(d_out, bytes);
  e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  return cudaDeviceSynchronize();
}

cudaError_t mma_func_info(cudaFuncAttributes* fa, int* occ_blocks) {
  const void* f = reinterpret_cast<const void*>(&k_mma<0>);
  cudaError_t e = cudaFuncGetAttributes(fa, f);
  if (e != cudaSuccess) return e;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ_blocks, f, kMmaThreads, 0);
}

}  // namespace pk

// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
// Confidential, not for distribution.
//
// tools/probes/tex/tex_probe.cu (2026-11-02, author 03:37 "probe it man probe it": the texture units on the dense decode).
// A one-token GEMV over 0.95 GiB of int8 weights (500,000 rows of 2,048, Q8_0-class; 27x the L2, so nothing is read twice), four
// ways, each the median of 15 timed launches after 3 warm ones:
//   V0 global 16-byte loads, int8 -> float on the SMs, FMA with float x (shared memory)
//   V1 global 16-byte loads, dp4a with int8 x (the engine's R3 Q8_0 arithmetic)
//   V2 texture: char4 elements read as normalized floats (the texture unit converts: q / 127), FMA with float x
//   V3 texture: 16-byte elements read raw (int4), dp4a with int8 x
// Checks: V1 = V3 exactly; V2 = V0 / 127 within float rounding (values -128 clamp to -1 in V2: counted). Prints GB/s.
// Build: nvcc -O3 -std=c++17 -arch=sm_89 tex_probe.cu -o tex_probe. Nothing is written to disk.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x)                                                                                                   \
  do {                                                                                                          \
    cudaError_t e_ = (x);                                                                                       \
    if (e_ != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return 1;                                                                                                 \
    }                                                                                                           \
  } while (0)

constexpr int K = 2048;          // values per row
constexpr long long ROWS = 500000;  // 0.95 GiB of int8 (char4 elements under the 2^28 1D linear texture limit)
constexpr int TPB = 256;

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ int warp_sumi(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// V0: a warp per row; lane loads 4 x 16 bytes (64 values), converts on the SMs
__global__ void __launch_bounds__(TPB) v0_global_float(const int4* __restrict__ w, const float* __restrict__ xg, float* __restrict__ y) {
  __shared__ float xs[K];
  for (int i = threadIdx.x; i < K; i += TPB) xs[i] = xg[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const long long warps = static_cast<long long>(gridDim.x) * (TPB / 32);
  for (long long r = blockIdx.x * (TPB / 32) + threadIdx.x / 32; r < ROWS; r += warps) {
    const int4* row = w + r * (K / 16);
    float a = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int4 v = __ldg(row + lane + 32 * c);
      const int base = 16 * (lane + 32 * c);
      const int vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int b = 0; b < 4; ++b) a += static_cast<float>(static_cast<std::int8_t>((vv[q] >> (8 * b)) & 0xff)) * xs[base + 4 * q + b];
    }
    a = warp_sum(a);
    if (lane == 0) y[r] = a;
  }
}
// V1: dp4a with int8 x (as the engine's Q8_0)
__global__ void __launch_bounds__(TPB) v1_global_dp4a(const int4* __restrict__ w, const int* __restrict__ xq, int* __restrict__ y) {
  __shared__ int xs[K / 4];
  for (int i = threadIdx.x; i < K / 4; i += TPB) xs[i] = xq[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const long long warps = static_cast<long long>(gridDim.x) * (TPB / 32);
  for (long long r = blockIdx.x * (TPB / 32) + threadIdx.x / 32; r < ROWS; r += warps) {
    const int4* row = w + r * (K / 16);
    int a = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int4 v = __ldg(row + lane + 32 * c);
      const int base = 4 * (lane + 32 * c);
      a = __dp4a(v.x, xs[base], a);
      a = __dp4a(v.y, xs[base + 1], a);
      a = __dp4a(v.z, xs[base + 2], a);
      a = __dp4a(v.w, xs[base + 3], a);
    }
    a = warp_sumi(a);
    if (lane == 0) y[r] = a;
  }
}
// V2: the texture unit returns 4 normalized floats per char4 fetch; a lane takes 16 fetches (64 values) per row
__global__ void __launch_bounds__(TPB) v2_tex_snorm(cudaTextureObject_t t, const float* __restrict__ xg, float* __restrict__ y) {
  __shared__ float xs[K];
  for (int i = threadIdx.x; i < K; i += TPB) xs[i] = xg[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const long long warps = static_cast<long long>(gridDim.x) * (TPB / 32);
  for (long long r = blockIdx.x * (TPB / 32) + threadIdx.x / 32; r < ROWS; r += warps) {
    const long long e0 = r * (K / 4);  // char4 elements
    float a = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
      const int ei = lane + 32 * c;
      const float4 v = tex1Dfetch<float4>(t, static_cast<int>(e0 + ei));
      const int base = 4 * ei;
      a += v.x * xs[base] + v.y * xs[base + 1] + v.z * xs[base + 2] + v.w * xs[base + 3];
    }
    a = warp_sum(a);
    if (lane == 0) y[r] = a;
  }
}
// V3: 16-byte raw texture elements, dp4a
__global__ void __launch_bounds__(TPB) v3_tex_raw_dp4a(cudaTextureObject_t t, const int* __restrict__ xq, int* __restrict__ y) {
  __shared__ int xs[K / 4];
  for (int i = threadIdx.x; i < K / 4; i += TPB) xs[i] = xq[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const long long warps = static_cast<long long>(gridDim.x) * (TPB / 32);
  for (long long r = blockIdx.x * (TPB / 32) + threadIdx.x / 32; r < ROWS; r += warps) {
    const long long e0 = r * (K / 16);
    int a = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int4 v = tex1Dfetch<int4>(t, static_cast<int>(e0 + lane + 32 * c));
      const int base = 4 * (lane + 32 * c);
      a = __dp4a(v.x, xs[base], a);
      a = __dp4a(v.y, xs[base + 1], a);
      a = __dp4a(v.z, xs[base + 2], a);
      a = __dp4a(v.w, xs[base + 3], a);
    }
    a = warp_sumi(a);
    if (lane == 0) y[r] = a;
  }
}

template <class F>
static float median_ms(F launch) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 3; ++i) launch();
  std::vector<float> ms;
  for (int i = 0; i < 15; ++i) {
    cudaEventRecord(a);
    launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float m = 0.f;
    cudaEventElapsedTime(&m, a, b);
    ms.push_back(m);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

int main() {
  const std::size_t bytes = static_cast<std::size_t>(ROWS) * K;
  cudaDeviceProp pr{};
  CK(cudaGetDeviceProperties(&pr, 0));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, v0_global_float, TPB, 0));
  const int grid = pr.multiProcessorCount * std::max(occ, 1);
  int texmax = 0, engines = 0;
  CK(cudaDeviceGetAttribute(&texmax, cudaDevAttrMaxTexture1DLinearWidth, 0));
  CK(cudaDeviceGetAttribute(&engines, cudaDevAttrAsyncEngineCount, 0));
  std::printf("tex_probe: %s, %d SMs, L2 %d MB, async copy engines %d, 1D linear texture max %d elements; grid %d x %d, %.2f GiB of int8\n", pr.name,
              pr.multiProcessorCount, pr.l2CacheSize >> 20, engines, texmax, grid, TPB, bytes / 1073741824.0);
  // weights: a fixed pseudo-random int8 pattern in -127..127 (no -128, so V2's normalization is exact up to rounding)
  std::vector<std::int8_t> h(bytes);
  std::uint32_t s = 12345u;
  for (std::size_t i = 0; i < bytes; ++i) {
    s = s * 1664525u + 1013904223u;
    h[i] = static_cast<std::int8_t>(static_cast<int>((s >> 24) % 255u) - 127);
  }
  std::vector<float> xf(K);
  std::vector<std::int8_t> x8(K);
  for (int i = 0; i < K; ++i) {
    s = s * 1664525u + 1013904223u;
    x8[i] = static_cast<std::int8_t>(static_cast<int>((s >> 24) % 255u) - 127);
    xf[i] = static_cast<float>(x8[i]) / 127.f;
  }
  std::int8_t* dw = nullptr;
  float *dxf = nullptr, *y0 = nullptr, *y2 = nullptr;
  int *dxq = nullptr, *y1 = nullptr, *y3 = nullptr;
  CK(cudaMalloc(&dw, bytes));
  CK(cudaMemcpy(dw, h.data(), bytes, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dxf, K * sizeof(float)));
  CK(cudaMemcpy(dxf, xf.data(), K * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dxq, K));
  CK(cudaMemcpy(dxq, x8.data(), K, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y0, ROWS * sizeof(float)));
  CK(cudaMalloc(&y2, ROWS * sizeof(float)));
  CK(cudaMalloc(&y1, ROWS * sizeof(int)));
  CK(cudaMalloc(&y3, ROWS * sizeof(int)));
  // textures over the same bytes
  cudaTextureObject_t tn = 0, tr = 0;
  {
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = dw;
    rd.res.linear.desc = cudaCreateChannelDesc<char4>();
    rd.res.linear.sizeInBytes = bytes;
    cudaTextureDesc td{};
    td.readMode = cudaReadModeNormalizedFloat;
    CK(cudaCreateTextureObject(&tn, &rd, &td, nullptr));
    cudaResourceDesc rr = rd;
    rr.res.linear.desc = cudaCreateChannelDesc<int4>();
    cudaTextureDesc tdr{};
    tdr.readMode = cudaReadModeElementType;
    CK(cudaCreateTextureObject(&tr, &rr, &tdr, nullptr));
  }
  const int4* w4 = reinterpret_cast<const int4*>(dw);
  const float t0 = median_ms([&] { v0_global_float<<<grid, TPB>>>(w4, dxf, y0); });
  CK(cudaGetLastError());
  const float t1 = median_ms([&] { v1_global_dp4a<<<grid, TPB>>>(w4, dxq, y1); });
  CK(cudaGetLastError());
  const float t2 = median_ms([&] { v2_tex_snorm<<<grid, TPB>>>(tn, dxf, y2); });
  CK(cudaGetLastError());
  const float t3 = median_ms([&] { v3_tex_raw_dp4a<<<grid, TPB>>>(tr, dxq, y3); });
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> h0(ROWS), h2(ROWS);
  std::vector<int> h1(ROWS), h3(ROWS);
  CK(cudaMemcpy(h0.data(), y0, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h2.data(), y2, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h1.data(), y1, ROWS * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h3.data(), y3, ROWS * sizeof(int), cudaMemcpyDeviceToHost));
  long long diff13 = 0;
  double worst02 = 0.0;
  for (long long r = 0; r < ROWS; ++r) {
    diff13 += h1[r] != h3[r];
    const double ref = static_cast<double>(h0[r]) / 127.0, d = std::fabs(ref - h2[r]) / (std::fabs(ref) + 1e-3);
    worst02 = std::max(worst02, d);
  }
  auto gbs = [&](float ms) { return static_cast<double>(bytes) / (ms * 1e-3) / 1e9; };
  std::printf("V0 global, int8 -> float on the SMs, FMA   %8.3f ms  %7.1f GB/s\n", t0, gbs(t0));
  std::printf("V1 global, dp4a (the engine's Q8_0)        %8.3f ms  %7.1f GB/s\n", t1, gbs(t1));
  std::printf("V2 texture, normalized float (unit converts) %6.3f ms  %7.1f GB/s\n", t2, gbs(t2));
  std::printf("V3 texture, raw 16-byte elements, dp4a     %8.3f ms  %7.1f GB/s\n", t3, gbs(t3));
  std::printf("checks: V1 vs V3 rows that differ %lld of %lld (must be 0); V2 vs V0/127 worst relative difference %.3e\n", diff13, ROWS, worst02);
  cudaDestroyTextureObject(tn);
  cudaDestroyTextureObject(tr);
  return diff13 == 0 ? 0 : 2;
}

// TEST-ONLY stand-in for the CUDA toolkit's cuda_fp16.h (tools/cuda_check: type-checking .cu files with clang where no
// toolkit is installed). Declares only what Pelias's kernels use; never used by a real build.
#pragma once
#include "cuda_runtime.h"
struct __half {
  unsigned short x;
};
__device__ inline __half __ushort_as_half(unsigned short v) {
  __half h;
  h.x = v;
  return h;
}
__host__ __device__ float __half2float(__half h);
__host__ __device__ __half __float2half_rn(float);
struct __attribute__((aligned(4))) __half2 {
  __half x, y;
};
__device__ __half2 __hsub2(__half2, __half2);
__device__ __half2 __float2half2_rn(float);

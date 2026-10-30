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
__device__ float __half2float(__half h);

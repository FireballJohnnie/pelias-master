// TEST-ONLY shim for tools/cuda_check/check.sh (clang -fsyntax-only, no CUDA toolkit): the declarations the kernels use.
#pragma once
#include "cuda_runtime.h"
namespace cooperative_groups {
struct grid_group {
  __device__ void sync();
};
__device__ grid_group this_grid();
}  // namespace cooperative_groups

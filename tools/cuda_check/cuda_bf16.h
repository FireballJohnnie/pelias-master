// TEST-ONLY stand-in (sandbox type-check, no CUDA toolkit): the bf16 names gpu_prompt.cu and the batch_c probe use.
#pragma once
#include "cuda_runtime.h"
#include "cuda_fp16.h"
struct __nv_bfloat16 { unsigned short x; };
__device__ __nv_bfloat16 __float2bfloat16_rn(float);
__device__ unsigned short __bfloat16_as_ushort(__nv_bfloat16);
__device__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short);
__device__ float __bfloat162float(__nv_bfloat16);
__device__ __half __float2half_rn(float);
__host__ __device__ inline float2 make_float2(float a, float b) { float2 r; r.x = a; r.y = b; return r; }
__device__ unsigned short __half_as_ushort(__half);

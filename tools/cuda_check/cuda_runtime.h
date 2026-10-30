// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// tools/cuda_check/cuda_runtime.h - TEST-ONLY, never part of a build: a stand-in for the CUDA SDK's cuda_runtime.h so that a
// machine without the CUDA toolkit (the sandbox) can type-check Pelias's .cu files with clang (tools/cuda_check/check.sh:
// clang -x cuda -fsyntax-only, host and device sides, sm_89). It declares only what Pelias's kernels call, on top of clang's
// own CUDA wrapper headers. The real compile is nvcc's on the PC; this only catches errors before PC time is spent.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <climits>

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#ifndef CUDA_VERSION
#define CUDA_VERSION 12080
#endif
#define __CUDACC__ 1

struct uint3 { unsigned x, y, z; };
// the SDK vector types clang's __ldg overloads name
struct __attribute__((aligned(2))) char2 { char x, y; };
struct __attribute__((aligned(4))) char4 { char x, y, z, w; };
struct __attribute__((aligned(4))) short2 { short x, y; };
struct __attribute__((aligned(8))) short4 { short x, y, z, w; };
struct __attribute__((aligned(8))) int2 { int x, y; };
struct __attribute__((aligned(16))) int4 { int x, y, z, w; };
struct __attribute__((aligned(16))) longlong2 { long long x, y; };
struct __attribute__((aligned(2))) uchar2 { unsigned char x, y; };
struct __attribute__((aligned(4))) uchar4 { unsigned char x, y, z, w; };
struct __attribute__((aligned(4))) ushort2 { unsigned short x, y; };
struct __attribute__((aligned(8))) ushort4 { unsigned short x, y, z, w; };
struct __attribute__((aligned(8))) uint2 { unsigned int x, y; };
struct __attribute__((aligned(16))) uint4 { unsigned int x, y, z, w; };
struct __attribute__((aligned(16))) ulonglong2 { unsigned long long x, y; };
struct __attribute__((aligned(8))) float2 { float x, y; };
struct __attribute__((aligned(16))) float4 { float x, y, z, w; };
struct __attribute__((aligned(16))) double2 { double x, y; };
struct __attribute__((aligned(16))) long2 { long x, y; };
struct __attribute__((aligned(16))) ulong2 { unsigned long x, y; };
struct float3 { float x, y, z; };
typedef std::uint32_t cuuint32_t;
__device__ unsigned int __isShared(const void*);
__device__ std::size_t __cvta_generic_to_shared(const void*);
__host__ __device__ inline uint2 make_uint2(unsigned int x, unsigned int y) { return uint2{x, y}; }
struct dim3 {
  unsigned x, y, z;
  __host__ __device__ constexpr dim3(unsigned vx = 1, unsigned vy = 1, unsigned vz = 1) : x(vx), y(vy), z(vz) {}
  __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
};
#include <__clang_cuda_builtin_vars.h>
#pragma push_macro("__DEVICE__")
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_math.h>
#include <__clang_cuda_intrinsics.h>
#pragma pop_macro("__DEVICE__")

typedef enum cudaError { cudaSuccess = 0, cudaErrorUnknown = 999 } cudaError_t;
typedef struct CUstream_st* cudaStream_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount = 16, cudaDevAttrCooperativeLaunch = 95, cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t cudaFuncSetAttributeImpl(const void*, cudaFuncAttribute, int);
template <class T>
cudaError_t cudaFuncSetAttribute(T* f, cudaFuncAttribute a, int v) { return cudaFuncSetAttributeImpl(reinterpret_cast<const void*>(f), a, v); }
cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shmem = 0, cudaStream_t stream = nullptr);
extern "C" unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shmem = 0, void* stream = nullptr);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3* grid, dim3* block, size_t* shmem, void* stream);
extern "C" cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
cudaError_t cudaLaunchCooperativeKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorImpl(int*, const void*, int, size_t);
template <class T>
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, T* f, int threads, size_t smem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorImpl(n, reinterpret_cast<const void*>(f), threads, smem);
}
cudaError_t cudaMemcpyToSymbolImpl(const void*, const void*, size_t);
template <class T>
cudaError_t cudaMemcpyToSymbol(const T& sym, const void* src, size_t n) { return cudaMemcpyToSymbolImpl(&sym, src, n); }
__device__ int __dp4a(int, int, int);
__device__ void __nanosleep(unsigned);
__device__ unsigned long long atomicAdd(unsigned long long*, unsigned long long);

// kernels.h - host-callable wrappers around the probe kernels (kernels.cu is the only file nvcc must compile).
// Every wrapper returns cudaGetLastError() right after the launch; nothing here synchronizes.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace pk {

enum class Mma : int { BF16 = 0, FP16 = 1, INT8 = 2, FP8 = 3 };

inline const char* mma_name(Mma k) {
  switch (k) {
    case Mma::BF16: return "bf16_fp32acc_m16n8k16";
    case Mma::FP16: return "fp16_fp32acc_m16n8k16";
    case Mma::INT8: return "int8_int32acc_m16n8k32";
    case Mma::FP8: return "fp8_e4m3_fp32acc_m16n8k32";
  }
  return "?";
}
// FLOP (or OP) done by one warp-level mma instruction: 2*M*N*K.
inline double mma_flop(Mma k) { return (k == Mma::BF16 || k == Mma::FP16) ? 2.0 * 16 * 8 * 16 : 2.0 * 16 * 8 * 32; }

inline constexpr int kMmaThreads = 128;  // threads per block of the tensor microbenchmark
inline constexpr int kMmaChains = 8;     // independent accumulator chains per warp
inline constexpr int kHashBlocks = 256;  // fixed so the float reduction order is fixed
inline constexpr int kHashThreads = 256;

// Total FLOP of one launch_mma call.
inline double mma_launch_flop(Mma k, int blocks, unsigned iters) {
  return static_cast<double>(blocks) * (kMmaThreads / 32) * static_cast<double>(iters) * kMmaChains * mma_flop(k);
}

// Tensor-core microbenchmark. sink needs blocks*kMmaThreads floats (keeps the work alive).
cudaError_t launch_mma(Mma k, int blocks, unsigned iters, float* sink, cudaStream_t s);

// Streaming kernels over uint4 (16-byte) elements.
cudaError_t launch_read(const void* p, size_t n_uint4, unsigned long long* sink, int blocks, int threads, cudaStream_t s);
cudaError_t launch_write(void* p, size_t n_uint4, int blocks, int threads, cudaStream_t s);
cudaError_t launch_copy(void* dst, const void* src, size_t n_uint4, int blocks, int threads, cudaStream_t s);

// Canary GEMM: C[M,N] = A[M,K] * Bt[N,K]^T in bf16 with fp32 accumulate. M%16==0, N%8==0, K%16==0.
cudaError_t launch_fill_bf16(uint16_t* p, size_t n, uint32_t seed, cudaStream_t s);
cudaError_t launch_gemm_bf16(const uint16_t* A, const uint16_t* Bt, float* C, int M, int N, int K, cudaStream_t s);
// Order-independent 64-bit hash of every element of C and a fixed-order float sum.
// acc: one u64 (zero it first). partial: kHashBlocks floats.
cudaError_t launch_hash(const float* C, size_t n, unsigned long long* acc, float* partial, cudaStream_t s);
cudaError_t launch_final(const float* partial, const unsigned long long* acc, unsigned long long* hash_out,
                         float* sum_out, int slot, cudaStream_t s);

// Time-based spin (about `ns` nanoseconds on every SM): used for the display-timeout test.
cudaError_t launch_spin(unsigned long long ns, unsigned long long* sink, int blocks, cudaStream_t s);
// Tiny kernels for the graph test.
cudaError_t launch_bump(unsigned long long* p, unsigned inc, cudaStream_t s);
cudaError_t launch_empty(cudaStream_t s);

// Shared-memory opt-in ramp: set the attribute to `bytes` and run a block with that much dynamic shared memory.
cudaError_t smem_probe(int bytes, int* d_out);
// Register / occupancy facts about the tensor kernel.
cudaError_t mma_func_info(cudaFuncAttributes* fa, int* occ_blocks);

}  // namespace pk

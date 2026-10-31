// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kernels_c.h - Batch C (P-K1, P-K2, P-K3, P-Q10): host-callable wrappers around the kernels in kernels_c.cu.
// Plain C++ (no kernel syntax). Every wrapper returns cudaGetLastError() right after its launches; nothing synchronizes.
//
// GGUF block layouts are read from llama.cpp ggml/src/ggml-common.h and ggml-quants.c (READ-SRC; the sandbox copy is
// commit 9bc6578, the project's pinned commit is 2ca15f5; both define the same blocks):
//   block_q4_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[128]; }   144 bytes, 256 weights
//   block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d; }      210 bytes, 256 weights
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pkc {

inline constexpr int kQK = 256;          // weights per super-block (QK_K)
inline constexpr int kQ4KBytes = 144;    // sizeof(block_q4_K)
inline constexpr int kQ6KBytes = 210;    // sizeof(block_q6_K)

struct BlockQ4K {
  uint16_t d, dmin;  // fp16 bits
  uint8_t scales[12];
  uint8_t qs[128];
};
struct BlockQ6K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;  // fp16 bits
};
static_assert(sizeof(BlockQ4K) == kQ4KBytes, "block_q4_K must be 144 bytes");
static_assert(sizeof(BlockQ6K) == kQ6KBytes, "block_q6_K must be 210 bytes");

// get_scale_min_k4 from ggml-quants.c, word for word in meaning.
inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// IEEE half -> float on the host (no CUDA headers needed).
inline float half_to_float(uint16_t h) {
  const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
  if (e == 0) {
    if (m == 0) bits = s;
    else {  // subnormal
      e = 127 - 15 + 1;
      while ((m & 0x400u) == 0) { m <<= 1; --e; }
      m &= 0x3FFu;
      bits = s | (e << 23) | (m << 13);
    }
  } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
  else bits = s | ((e + 127 - 15) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Host reference dequantizers (dequantize_row_q4_K / dequantize_row_q6_K of ggml-quants.c), 256 floats out.
inline void dequant_q4k_host(const BlockQ4K& b, float* y) {
  const float d = half_to_float(b.d), mn = half_to_float(b.dmin);
  const uint8_t* q = b.qs;
  int is = 0;
  for (int j = 0; j < kQK; j += 64) {
    int sc, m;
    scale_min_k4(is + 0, b.scales, sc, m);
    const float d1 = d * sc, m1 = mn * m;
    scale_min_k4(is + 1, b.scales, sc, m);
    const float d2 = d * sc, m2 = mn * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
    q += 32;
    is += 2;
  }
}
inline void dequant_q6k_host(const BlockQ6K& b, float* y) {
  const float d = half_to_float(b.d);
  const uint8_t* ql = b.ql;
  const uint8_t* qh = b.qh;
  const int8_t* sc = b.scales;
  for (int n = 0; n < kQK; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

// ---- data generation (deterministic from the seed)
cudaError_t fill_q4k(BlockQ4K* w, size_t nblocks, uint32_t seed, cudaStream_t s);
cudaError_t fill_q6k(BlockQ6K* w, size_t nblocks, uint32_t seed, cudaStream_t s);
// bf16 activations with |x| in a realistic range, sign random.
cudaError_t fill_bf16(uint16_t* p, size_t n, uint32_t seed, cudaStream_t s);
// Gaussian (dist 0) or heavy-tailed (dist 1: gaussian * exp(gaussian)) fp32 values times `scale`.
cudaError_t fill_f32(float* p, size_t n, uint32_t seed, int dist, float scale, cudaStream_t s);

// ---- P-K1 (bf16 path): dequantize Q4_K weights to a bf16 scratch, then a tiled mma.sync GEMM.
// W: N rows of K/256 blocks. Wout: N x K bf16 row-major.
cudaError_t dequant_q4k_to_bf16(const BlockQ4K* w, uint16_t* wout, int N, int K, cudaStream_t s);
// C[M,N] (fp32, row-major) = A[M,K] (bf16 row-major) * B[N,K]^T (bf16 row-major). K % 32 == 0.
// cfg: 0 = 128x128 tile (M, N % 128), 1 = 128x64 (M % 128, N % 64), 2 = 64x64 (M, N % 64).
// raster: 0 = output tiles row by row (N fastest), 1 = grouped by 8 tiles along M (weight tiles reused in L2).
inline constexpr int kGemmCfgs = 3;
inline const char* gemm_cfg_name(int cfg) { return cfg == 0 ? "128x128" : (cfg == 1 ? "128x64" : "64x64"); }
cudaError_t gemm_bf16_tiled(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K, int cfg, int raster, cudaStream_t s);

// ---- P-K2 (int8 path): activations quantized per row per 32 (int8 + fp32 scale + int sum); Q4_K weights read raw and
// unpacked to int8 in registers; mma.sync s8 with a per-sub-block fp32 epilogue.
// a8: M x K int8; sa: M x K/32 floats; s8: M x K/32 ints.
// sas: M x K/32 floats, scale * sum (the activation side of the Q4_K min term, used by the v4 kernel).
cudaError_t quantize_rows_q8(const uint16_t* A, int8_t* a8, float* sa, int* s8, float* sas, int M, int K, cudaStream_t s);
cudaError_t gemm_q8_q4k(const int8_t* a8, const float* sa, const int* s8, const BlockQ4K* w, float* C, int M, int N, int K,
                        int raster, cudaStream_t s);
// v4 (revision 4): same result; the 16 products of a sub-block are issued before the scale epilogue; exact magic-number
// int->float, precomputed min-term product, B words loaded once per K step, ldmatrix for A (v3 had all but the first).
cudaError_t gemm_q8_q4k_v4(const int8_t* a8, const float* sa, const float* sas, const BlockQ4K* w, float* C, int M, int N, int K,
                           int raster, cudaStream_t s);
// v5 (revision 5): registers capped for two blocks per SM; weight scales d*sc and dmin*m prepared once per weight into
// dsc / dmn (N x K/32 floats each, by prep_q4k_scales) and streamed like the activation scales.
cudaError_t prep_q4k_scales(const BlockQ4K* w, float* dsc, float* dmn, int N, int K, cudaStream_t s);
cudaError_t gemm_q8_q4k_v5(const int8_t* a8, const float* sa, const float* sas, const BlockQ4K* w, const float* dsc, const float* dmn,
                           float* C, int M, int N, int K, int raster, cudaStream_t s);
// Registers, local (spill) bytes, static shared memory and resident blocks per SM at 256 threads, for:
// 0..2 = bf16 tiles 128x128 / 128x64 / 64x64, 3 = int8 v2, 4 = int8 v4, 5 = int8 v5, 6..8 = int8 v6 variants 0..2,
// 9 = int8 v7, 10 / 11 = int8 v8 with 3 / 4 stages (shared bytes reported as static plus dynamic), 12 / 13 = int8 v9
// with 2 / 3 stages (512 threads; blocks per SM at 512 threads), 14 / 15 = int8 v10 with 2 / 3 stages,
// 16 / 17 = int8 v11 with 2 / 3 stages (512 threads), 18..21 = int8 v12 variants 0..3.
// v6 (revision 6): v5's arithmetic with smaller warp tiles and no register cap. variant 0: 128x64 tile, warp 32x32;
// 1: 128x64, warp 64x16; 2: 64x64, warp 32x16.
inline constexpr int kV6Variants = 3;
inline const char* v6_name(int v) { return v == 0 ? "v6 128x64 w32x32" : (v == 1 ? "v6 128x64 w64x16" : "v6 64x64 w32x16"); }
cudaError_t gemm_q8_q4k_v6(const int8_t* a8, const float* sa, const float* sas, const BlockQ4K* w, const float* dsc, const float* dmn,
                           float* C, int M, int N, int K, int variant, int raster, cudaStream_t s);
// v7 (revision 7): activations with one scale per 256 (sa256), per-32 integer sums s32 and x32 = sa256 * s32; the weight
// sub-block scale applied in integers (one IMAD per accumulator per sub-block), d * sa256 once per super-block.
cudaError_t quantize_rows_q8k(const uint16_t* A, int8_t* a8, float* sa256, int* s32, float* x32, int M, int K, cudaStream_t s);
cudaError_t gemm_q8k_q4k_v7(const int8_t* a8, const float* sa256, const float* x32, const BlockQ4K* w, const float* dmn, float* C,
                            int M, int N, int K, int raster, cudaStream_t s);
// v8 (revision 8): v7's arithmetic, 3 or 4 cp.async stages in dynamic shared memory, one barrier per step, and a 16-byte
// record per weight row and super-block (d, eight 6-bit scales) prepared once per weight by prep_q4k_rec (16 bytes per
// Q4_K block). variant 0 = 3 stages, 1 = 4 stages.
inline constexpr int kV8Variants = 2;
inline const char* v8_name(int v) { return v == 0 ? "v8 3-stage" : "v8 4-stage"; }
int v8_stages(int variant);
int v8_smem_bytes(int variant);
cudaError_t prep_q4k_rec(const BlockQ4K* w, void* rec, int N, int K, cudaStream_t s);
cudaError_t gemm_q8k_q4k_v8(const int8_t* a8, const float* sa256, const float* x32, const BlockQ4K* w, const void* rec, const float* dmn,
                            float* C, int M, int N, int K, int variant, int raster, cudaStream_t s);
// v9 (revision 9): v8's arithmetic with 16 warps per 128x128 tile (512 threads, registers capped at 128), weight scales by
// column prepared once per weight by prep_q4k_v9: scT and dmnT N * K / 32 each (int / float), dT N * K / 256 floats.
// variant 0 = 2 stages, 1 = 3 stages.
inline constexpr int kV9Variants = 2;
inline const char* v9_name(int v) { return v == 0 ? "v9 2-stage" : "v9 3-stage"; }
int v9_stages(int variant);
int v9_smem_bytes(int variant);
cudaError_t prep_q4k_v9(const BlockQ4K* w, int* scT, float* dmnT, float* dT, int N, int K, cudaStream_t s);
cudaError_t gemm_q8k_q4k_v9(const int8_t* a8, const float* sa256, const float* x32, const BlockQ4K* w, const int* scT, const float* dmnT,
                            const float* dT, float* C, int M, int N, int K, int variant, int raster, cudaStream_t s);
// v10 (revision 10): v8's integer main term; the min term once per super-block as one exact bf16 tensor-core product of
// digits. Activation digits from quantize_rows_q8k_dig (M * K / 256 * 16 bf16), weight digits from prep_q4k_dig
// (N * K / 256 * 16 bf16), and prep_q4k_rec's record (its 4th word holds dmin). variant 0 = 2 stages, 1 = 3 stages.
inline constexpr int kV10Variants = 2;
inline const char* v10_name(int v) { return v == 0 ? "v10 2-stage" : "v10 3-stage"; }
int v10_stages(int variant);
int v10_smem_bytes(int variant);
cudaError_t quantize_rows_q8k_dig(const uint16_t* A, int8_t* a8, float* sa256, int* s32, float* x32, uint16_t* dig, int M, int K, cudaStream_t s);
cudaError_t prep_q4k_dig(const BlockQ4K* w, uint16_t* wd, int N, int K, cudaStream_t s);
cudaError_t gemm_q8k_q4k_v10(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             float* C, int M, int N, int K, int variant, int raster, cudaStream_t s);
// RMSNorm (eps 1e-6, gamma bf16) producing bf16 (the bf16 path) or, fused, v10's int8 inputs directly (a8, sa256, s32,
// x32, digits). K a multiple of 256. Revision 11.
cudaError_t rmsnorm_bf16(const uint16_t* X, const uint16_t* gamma, uint16_t* Y, int M, int K, cudaStream_t s);
cudaError_t rmsnorm_q8k(const uint16_t* X, const uint16_t* gamma, int8_t* a8, float* sa256, int* s32, float* x32, uint16_t* dig, int M, int K,
                        cudaStream_t s);
// v11 (revision 11): v10's arithmetic and inputs with 16 warps per 128x128 tile (512 threads, warp tile 32x32, registers
// capped at 128). variant 0 = 2 stages, 1 = 3 stages.
inline constexpr int kV11Variants = 2;
inline const char* v11_name(int v) { return v == 0 ? "v11 2-stage" : "v11 3-stage"; }
int v11_smem_bytes(int variant);
cudaError_t gemm_q8k_q4k_v11(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             float* C, int M, int N, int K, int variant, int raster, cudaStream_t s);
// v12 (revision 12): v10's arithmetic and inputs with split-K, grouped problems (batch: problem z has its own M rows of
// A / scales / digits, its own N weight rows and records and digits, its own M x N of C), a 64-row tile option and bf16
// output. variants: 0 = 128x128 2-stage, 1 = 128x128 3-stage, 2 = 64x128 2-stage, 3 = 64x128 3-stage. splits must divide
// K / 256; with splits > 1, `partial` must hold splitk_partial_bytes(M, N, batch, splits) and a reduction kernel follows.
inline constexpr int kV12Variants = 4;
const char* v12_name(int variant);
int v12_tile_m(int variant);
size_t splitk_partial_bytes(int M, int N, int batch, int splits);
cudaError_t gemm_q8k_q4k_v12(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             void* C, int M, int N, int K, int batch, int splits, int out_bf16, float* partial, int variant, int raster, cudaStream_t s);
// v13 (O13, 2026-10-30): v12's arithmetic made persistent: at most (SMs x resident blocks per SM, capped by
// blocks_per_sm_cap when > 0) blocks, each walking tiles blockIdx.x + i * gridDim.x over all problems with one cp.async
// pipeline across tile boundaries (the next tile loads while the current one finishes and stores). Same variants as v12,
// no split-K. grid_out (optional) receives the launched block count. Results equal v12's (splits 1) bit for bit.
const char* v13_name(int variant);
cudaError_t gemm_q8k_q4k_v13(const int8_t* a8, const float* sa256, const uint16_t* adig, const BlockQ4K* w, const void* rec, const uint16_t* wdig,
                             void* C, int M, int N, int K, int batch, int out_bf16, int variant, int raster, int blocks_per_sm_cap, int* grid_out,
                             cudaStream_t s);
// The bf16 GEMM on `batch` problems at once (A: batch x M x K, B: batch x N x K, C: batch x M x N), same tiles as gemm_bf16_tiled.
cudaError_t gemm_bf16_batched(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K, int batch, int cfg, int raster, cudaStream_t s);
// fp32 reference for v7's inputs (activation scale per 256).
cudaError_t ref_q8k_q4k_rows(const int8_t* a8, const float* sa256, const BlockQ4K* w, float* C, int rows, int ncols, int K, cudaStream_t s);
cudaError_t kernel_info(int which, int* regs, int* local_bytes, int* static_smem, int* blocks_per_sm);

// fp32 reference for rows [0, rows) and columns [0, ncols) of the int8 path, from the same dequantized activations and
// weights in fp32. C is rows x ncols.
cudaError_t ref_q8_q4k_rows(const int8_t* a8, const float* sa, const BlockQ4K* w, float* C, int rows, int ncols, int K,
                            cudaStream_t s);

// ---- P-K3: stream Q4_K / Q6_K blocks from VRAM and dequantize into registers (no mma). One warp per 32 blocks.
cudaError_t stream_dequant_q4k(const BlockQ4K* w, size_t nblocks, float* sink, int blocks, cudaStream_t s);
cudaError_t stream_dequant_q6k(const BlockQ6K* w, size_t nblocks, float* sink, int blocks, cudaStream_t s);
// Dequantize a few blocks on the GPU into fp32 (for the host comparison).
cudaError_t dequant_q4k_f32(const BlockQ4K* w, float* out, size_t nblocks, cudaStream_t s);
cudaError_t dequant_q6k_f32(const BlockQ6K* w, float* out, size_t nblocks, cudaStream_t s);

// ---- P-Q10 (FP8 numerics for the data gradient): dX[M,K] = dY[M,N] * W[N,K].
// Round dY to fp8 per row (kind 0 = e4m3, scale = amax/448; kind 1 = e5m2, scale = amax/57344) and back to fp32.
cudaError_t fp8_roundtrip_rows(const float* x, float* y, int M, int N, int kind, cudaStream_t s);
// Plain tiled fp32 GEMM: C[M,K] = A[M,N] * B[N,K] (all row-major fp32). M,N,K % 32 == 0.
cudaError_t gemm_f32_nn(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
// bf16 -> fp32 copy.
cudaError_t bf16_to_f32(const uint16_t* in, float* out, size_t n, cudaStream_t s);

// ---- comparisons
// out[0] = max |a-b|, out[1] = max |b| (as float bits in unsigned), over n elements. Zero `out` first.
cudaError_t max_abs_diff(const float* a, const float* b, size_t n, unsigned* out, cudaStream_t s);
// Cosine pieces: out[0] += sum a*b, out[1] += sum a*a, out[2] += sum b*b (double). Zero `out` first.
cudaError_t dot3(const float* a, const float* b, size_t n, double* out, cudaStream_t s);

}  // namespace pkc

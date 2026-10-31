// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kernels_tg2.h - TG2 probe (2026-10-30 22:2x, author 22:17 "go ahead with the five-point probe"): the int8 prompt path
// for Q4_TG2 weights (grids not int8-exact), five ways, on 64-row tiles (column tile 128 in blockIdx.x, units table in
// blockIdx.y as kern/gpu_prompt's tile kernel):
//   kF32Lut   exact: weights decoded through an fp32 table into shared memory, fp32 FMAs on the CUDA cores
//   kF16      one fp16 digit per grid value, fp16 mma (fp32 accumulate)
//   kF16HiLo  two fp16 digits (hi + lo), two fp16 mmas
//   kI8x2     16-bit fixed point (s8 high byte, u8 low byte), two int8 mmas, combined in int32
//   kI8x3     24-bit fixed point (s8, u8, u8), three int8 mmas, combined in fp32
// plus the warp-vector path for 1-8 rows (exact fp32 table) next to kern/gpu_prompt's Q4_K vector kernel, and the Q4_K
// tile kernel of kern/gpu_prompt with its min term as the v10 bf16 two-digit mma or the v10b fp32 FMAs (timing sweep).
// Host API only.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ptg {

constexpr int kQK = 256, kTile = 64, kBN = 128, kVecMax = 8;

enum Var : int { kF32Lut = 0, kF16 = 1, kF16HiLo = 2, kI8x2 = 3, kI8x3 = 4, kTg2Vars = 5 };
const char* var_name(int v);

// Q4_TG2 v2 grid values (as quant::q4_tg_grid(tag, 2)), [4][16]
void tg2_grids(float g[4][16]);
// the 64-word table of variant v for the grids (host side; upload to the device): fp32 bits / fp16 / fp16 hi|lo<<16 /
// int8 bytes (lo, hi) / (lo, mid, hi). Fixed-point scales: kI8x2 32767, kI8x3 8388607.
void tg2_table(int v, std::uint32_t out[64]);
float tg2_fixed_scale(int v);

// per block: the eight sub-block scales s_j = fl(fl(d x lv_j) / 63) (fp32), [blocks][8]
cudaError_t tg2_prep(const void* w, float* sc, std::size_t nblocks, cudaStream_t s);

// units: int4 (expert, first row, valid rows, 0), device; C rows x N (row = the unit's rows)
cudaError_t tg2_tile(int v, const std::int8_t* a8, const float* sa32, const void* w, const float* sc, const std::uint32_t* lut, const int4* units, int nunits,
                     int N, int K, float* C, cudaStream_t s);
cudaError_t tg2_vec(const std::int8_t* a8, const float* sa32, const void* w, const float* sc, const std::uint32_t* lut, const int4* units, int nunits, int N,
                    int K, float* C, cudaStream_t s);
cudaError_t tg2_tile_info(int v, int* regs, int* spill, int* smem, int* blocks_per_sm);

// Q4_K side (kern/gpu_prompt's kernels, plain row store)
cudaError_t q4k_prep(const void* w, uint4* rec, void* wmin_bf16, float* wmin_f32, std::size_t nblocks, cudaStream_t s);
cudaError_t q4k_act_min(const float* scale, const int* sum, std::size_t n32, void* amin_bf16, float* amin_f32, cudaStream_t s);
// f32min: false = v10 (bf16 two-digit mma; wmin / amin bf16), true = v10b (fp32 FMAs; wmin / amin fp32)
cudaError_t q4k_tile(bool f32min, const std::int8_t* a8, const float* sa32, const void* amin, const void* w, const uint4* rec, const void* wmin, const int4* units,
                     int nunits, int N, int K, float* C, cudaStream_t s);
cudaError_t q4k_vec(const std::int8_t* a8, const float* sa32, const void* w, const int4* units, int nunits, int N, int K, float* C, cudaStream_t s);
cudaError_t q4k_tile_info(bool f32min, int* regs, int* spill, int* smem, int* blocks_per_sm);

}  // namespace ptg

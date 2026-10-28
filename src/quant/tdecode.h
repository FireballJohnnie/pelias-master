// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
#pragma once
// T-decode (docs/PELIAS_GAME_PLAN.md, "Q7_ZH and the binary gate"; author 2026-10-28 14:39-14:45): the dot product of a
// Q7_ZT row with an activation in the per-32 int8 format, with the 8th bit (H) taken one of three ways:
//   fold     h folded into the int8 value (q = 2b + h - 127) and one int8 dot per 32-block: what an int8 kernel sees
//   one_bit  the base dot sum (2b - 127) x in int8, plus the H bits by AND + popcount against the activation's 8 bit
//            planes: sum h x = sum_k 2^k popcount(H & plane_k) - 128 popcount(H)   (x = xu - 128, xu = x + 128)
//   missing  H not read (skipped or late): filled with 0.5 through the block sum, 0.5 * sum(x) (never an integer fill)
// fold and one_bit give identical integer partial sums (exact); the base and the H bits can live in separate arrays
// (q7_zt_split / q7_zt_join), so the H bits can sit somewhere else than the base (RAM vs VRAM, CPU vs GPU).
#include <cstdint>

namespace pelias::quant {

// one 32-value activation block: fp32 scale (amax / 127), int8 values, their sum, and 8 bit planes of xu = x + 128
struct ActBlock32 {
  float scale = 0.0f;
  std::int32_t sum = 0;
  std::uint32_t plane[8] = {};
  std::int8_t q[32] = {};
};

void act_planes_32(const std::int8_t* q, std::uint32_t plane[8]);       // portable
bool act_planes_32_fast(const std::int8_t* q, std::uint32_t plane[8]);  // AVX2 (pmovmskb), false if not available
void act_quantize_32(const float* x, std::int64_t n, ActBlock32* out);  // n a multiple of 32

constexpr std::int64_t kQ7ZTBaseBytes = 233;  // per 256: fp16 d, 6 level bytes, gate mask, 224 bytes of 7-bit base
constexpr std::int64_t kQ7ZTHBytes = 8;       // per 256: the H bits of the two gated 32-blocks
void q7_zt_split(const std::uint8_t* in, std::int64_t n, std::uint8_t* base, std::uint8_t* hbits);
void q7_zt_join(const std::uint8_t* base, const std::uint8_t* hbits, std::int64_t n, std::uint8_t* out);

enum class HRoute { fold, one_bit, missing };
// n values; base / hbits as split (hbits may be null only with HRoute::missing). part2, if given, receives per 32-block
// twice the integer partial sum (2P, so the 0.5 fill stays an integer). Returns sum_j s_j * a_j * P_j in fp32.
float q7_zt_dot(const std::uint8_t* base, const std::uint8_t* hbits, const ActBlock32* x, std::int64_t n, HRoute route,
                std::int64_t* part2 = nullptr);

}  // namespace pelias::quant

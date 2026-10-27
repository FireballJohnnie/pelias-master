// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// quant_golden.h - one fixed, platform-independent input for the quantizers (integer generator, exact integer-to-float
// steps only, so every compiler and standard library produces the same floats), and the SHA-256 of each type's output
// on it. The hashes were produced by the reference library (tests/oracle/quant_oracle --golden prints them from the
// reference's own output); the unit test checks Pelias's output against them on every platform, so a Windows (MSVC)
// build is checked bit for bit without the oracle.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace ptest {

// 64 super-blocks of 256 values = 16,384 floats: weight-like values, rows rounded to bf16, ties, zero blocks, constant
// blocks, a spike, tiny values and large values. Each value is one rounding at most (one multiply of an exact value),
// so no compiler can produce a different input by fusing operations.
inline std::vector<float> golden_input() {
  std::uint64_t s = 0x9E3779B97F4A7C15ull;
  auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  auto unit = [&next]() {  // a 24-bit integer times 2^-23, minus 1: exact, in [-1, 1)
    return static_cast<float>(static_cast<std::int32_t>(next() >> 40)) * (1.0f / 8388608.0f) - 1.0f;
  };
  auto to_bf16 = [](float f) {  // round to bf16 by the same integer rule the format uses
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + (0x7FFFu + ((u >> 16) & 1u))) & 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
  };
  std::vector<float> x(64 * 256);
  for (int b = 0; b < 64; ++b) {
    float* p = x.data() + b * 256;
    const int kind = b % 8;
    for (int i = 0; i < 256; ++i) {
      const float u = unit();
      switch (kind) {
        case 0: p[i] = to_bf16(u * 0.03f); break;                                        // weights, bf16 values
        case 1: p[i] = u * (i % 37 == 0 ? 2.5f : 0.05f); break;                          // outliers
        case 2: p[i] = static_cast<float>(static_cast<int>(u * 64.0f)) * 0.5f; break;    // ties
        case 3: p[i] = (b / 8) % 2 ? 0.0f : u * 1e-6f; break;                            // zero or tiny block
        case 4: p[i] = 0.75f - static_cast<float>(b) * 0.0078125f; break;                // constant block (exact steps)
        case 5: p[i] = i == 100 ? 40.0f : u * 0.001f; break;                             // one spike
        case 6: p[i] = u * 3e-5f; break;                                                 // below fp16 normal range
        default: p[i] = u * 2000.0f; break;                                              // large
      }
    }
  }
  return x;
}

// SHA-256 of each type's output on golden_input(), from the reference library at commit 2ca15f5 (MEASURED in the
// sandbox 2026-10-26 by tests/oracle/quant_oracle; Pelias's output matched each one there).
struct GoldenHash {
  const char* type;
  const char* sha256;
};
inline const GoldenHash kGoldenHashes[] = {
    {"Q8_0", "ee3385f3a41eff154cf8b0a181bfc34b450d046fc615e0f003b31a1d602081b9"},
    {"Q4_K", "a3b3345b91da0a2c2705d4d45720880e81573c60d33af1776966dd2008fea56a"},
    {"Q6_K", "48b498d48f435df1210c3e9e7e411d85ea31c868f9036fe69d77c1d5d48d2686"},
    {"F16", "6a597b6e8d805d1423010d0ef1aae0b8ceaa92b9874544e6869cf62930069c54"},
    {"BF16", "6c73c994908a4895b70e32886e5bedb93dd5eae94d1f227e3d8e2844868d5d6b"},
    {"Q8_1", "691bab17fb59391285f8b170d69013b3a22dfcfb5c962b42a3e529c224620ba3"},
    // Q8_K: an all-zero block's sums are left unwritten by the reference; both buffers start zeroed for this hash
    {"Q8_K", "1b23ab585ba0b4058f6dcae1ac667f3c90771ec4c8b48af451a60562b8b56a4b"},
};
// The read-only types (dequantize only): 64 blocks of fixed pseudo-random bytes with finite fp16 scales written at the
// block's scale offsets; the hash is of the dequantized floats' bytes (little-endian), as the reference produced them.
struct DequantGolden {
  const char* type;
  std::uint32_t type_id;      // quant::Type
  std::uint32_t block_bytes;
  int d_at, dmin_at;          // byte offsets of the fp16 scale and min scale (-1: none)
  const char* sha256;
  int scale_kind = 0;         // 0: fp16 at d_at; 1: IQ1_M (the fp16 scale in the top nibbles of 4 u16 at 48); 2: E8M0 byte at 0
};
inline std::vector<std::uint8_t> dequant_golden_input(const DequantGolden& g) {
  std::uint64_t s = 0xD1B54A32D192ED03ull ^ g.type_id;
  std::vector<std::uint8_t> b(static_cast<std::size_t>(64) * g.block_bytes);
  for (auto& v : b) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    v = static_cast<std::uint8_t>(s >> 56);
  }
  for (std::size_t k = 0; k < 64; ++k) {
    // fp16 0x2000 + k*64: 2^-7 up to about 0.13, exact and finite; the min scale a little smaller
    const std::uint16_t d = static_cast<std::uint16_t>(0x2000u + k * 64u), m = static_cast<std::uint16_t>(0x1C00u + k * 32u);
    std::uint8_t* blk = b.data() + k * g.block_bytes;
    if (g.scale_kind == 1) {
      for (int nib = 0; nib < 4; ++nib)
        blk[48 + 2 * nib + 1] = static_cast<std::uint8_t>((blk[48 + 2 * nib + 1] & 0x0F) | (((d >> (4 * nib)) & 0xF) << 4));
      continue;
    }
    if (g.scale_kind == 2) {
      blk[0] = static_cast<std::uint8_t>(112 + k % 24);  // 2^-16 .. 2^7 (times the half)
      continue;
    }
    blk[g.d_at] = static_cast<std::uint8_t>(d & 0xFF);
    blk[g.d_at + 1] = static_cast<std::uint8_t>(d >> 8);
    if (g.dmin_at >= 0) {
      blk[g.dmin_at] = static_cast<std::uint8_t>(m & 0xFF);
      blk[g.dmin_at + 1] = static_cast<std::uint8_t>(m >> 8);
    }
  }
  return b;
}
inline const DequantGolden kDequantGolden[] = {
    {"Q2_K", 10, 84, 80, 82, "03e1198dbf7febb1f3f247e07de1f7970941dc81bbd3d89ccb7b20762ce093c0"},
    {"Q3_K", 11, 110, 108, -1, "85ab74478dd8c3fdd1e05972c23e795cf40f6909e86a958d334f0438898ba525"},
    {"Q5_K", 13, 176, 0, 2, "9f8c64c3885b464a2b59471f14866ae2c54ac8d7c826210a006a2c4801737d46"},
    {"IQ4_XS", 23, 136, 0, -1, "56b6e3e3bb5f330aabc6780a18e7bc1cc5aa59c0850adbab4a05af12aef3f04f"},
    {"Q4_0", 2, 18, 0, -1, "9638f7c7d66529aad19837440bf3fb84eae5430872b77902c87cc5e3b745a9d7"},
    {"Q4_1", 3, 20, 0, 2, "124484f8073fa13b783b0270d335734b48b4792a235e042701d52f87e20a820b"},
    {"Q5_0", 6, 22, 0, -1, "104da7851b20e69bcd5b304739a08e7c2b4a4127248d57914ee6d54c1099b208"},
    {"Q5_1", 7, 24, 0, 2, "856958866b778ef5103e4cc38970da2b61f930e0f6f5e361c7b8d808a0fc452c"},
    {"IQ4_NL", 20, 18, 0, -1, "fa976eef0df72e975f276802eda56aa6b4534687c249cbae20d295e8f90bc379"},
    {"MXFP4", 39, 17, 0, -1, "ea0fdcf059fcabc050576d30fb98733eeee17835b2928318ed0b51b0990fcd5a", 2},
    {"TQ1_0", 34, 54, 52, -1, "944b1634471b2f98073b7a925c052a6d16959ece2a58019b3922dbd26a44bfc4"},
    {"TQ2_0", 35, 66, 64, -1, "109e0d087352383a00e375b4ac3f78312589426b89a7c32223af45c798366d2f"},
    {"IQ2_XXS", 16, 66, 0, -1, "866545bb07a3235de1b1f02152dfbac2ef2205837cfb7b7c7af5c25d52e838c0"},
    {"IQ2_XS", 17, 74, 0, -1, "4b19a6a52fe9450d4007edaaed046ab9be4749c38f4e274ade3969e6359ddf55"},
    {"IQ2_S", 22, 82, 0, -1, "c654c67b358b06b7316c0289f233507e3bf025c572a4ba796cd7013b18e0eb3d"},
    {"IQ3_XXS", 18, 98, 0, -1, "0251e05c3fce76a4f8e2df61c710c6718638308b10bb04f9af790c0cdb3c80d2"},
    {"IQ3_S", 21, 110, 0, -1, "9f7e3ec676f01fdc71262820c0613346c1687b2c6a8ba3b7de9b45b219d18df1"},
    {"IQ1_S", 19, 50, 0, -1, "7bbf53e28c44b80a657af66ae861cd10683a70357dfd324e6f40f3b3a7acfa95"},
    {"IQ1_M", 29, 56, 0, -1, "ed685dbfaa8a7536d3711c561efb4053abab5260d736c44b1b243e465ca19879", 1},
};
}  // namespace ptest

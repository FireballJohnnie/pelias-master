// provenance: re-expressed (not copied) from ggml ggml/src/ggml-quants.c (dequantize_row_*) and ggml/src/ggml-impl.h
//   (ggml_e8m0_to_fp32_half) at llama.cpp commit 2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md),
//   read as a specification so the floats match bit for bit; the IQ grids are format data in quant/iq_tables.cpp
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// The read-only weight types beyond the K-quants: the legacy 32-element blocks (Q4_0, Q4_1, Q5_0, Q5_1), the
// non-linear and microscaling 4-bit types (IQ4_NL, MXFP4), the ternary types (TQ1_0, TQ2_0) and the grid types
// (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M). Every float expression keeps the reference's operand order and
// its integer-to-float conversions. Blocks are read field by field at fixed byte offsets (little-endian).
#include <cstring>

#include "quant/iq_tables.h"
#include "quant/quant_cpu.h"

namespace pelias::quant {

namespace {

constexpr int kQK32 = 32;
constexpr int kQKK = 256;

inline std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
inline std::uint32_t u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) | (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}
inline float h(const std::uint8_t* p) { return fp16_to_fp32(u16(p)); }
inline std::uint8_t byte_of(std::uint64_t g, int j) { return static_cast<std::uint8_t>(g >> (8 * j)); }
inline float sgn(std::uint8_t signs, int j) { return (signs & iq::kMask[j]) ? -1.f : 1.f; }

// the IQ4_NL / IQ4_XS levels and the E2M1 levels times two (MXFP4)
constexpr std::int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr std::int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constexpr float kIq1sDelta = 0.125f;

// E8M0 exponent byte to 2^(x-127) * 0.5 as a float (subnormal for x < 2)
inline float e8m0_half(std::uint8_t x) {
  const std::uint32_t bits = x < 2 ? (0x00200000u << x) : (static_cast<std::uint32_t>(x - 1) << 23);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

}  // namespace

// ---- legacy 32-element blocks

void dequantize_q4_0(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs[16]
  for (std::int64_t i = 0; i < n / kQK32; ++i, in += 18, y += kQK32) {
    const float d = h(in);
    for (int j = 0; j < kQK32 / 2; ++j) {
      const int x0 = (in[2 + j] & 0x0F) - 8, x1 = (in[2 + j] >> 4) - 8;
      y[j] = static_cast<float>(x0) * d;
      y[j + kQK32 / 2] = static_cast<float>(x1) * d;
    }
  }
}

void dequantize_q4_1(const std::uint8_t* in, float* y, std::int64_t n) {  // d, m, qs[16]
  for (std::int64_t i = 0; i < n / kQK32; ++i, in += 20, y += kQK32) {
    const float d = h(in), m = h(in + 2);
    for (int j = 0; j < kQK32 / 2; ++j) {
      const int x0 = in[4 + j] & 0x0F, x1 = in[4 + j] >> 4;
      y[j] = static_cast<float>(x0) * d + m;
      y[j + kQK32 / 2] = static_cast<float>(x1) * d + m;
    }
  }
}

void dequantize_q5_0(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qh[4], qs[16]
  for (std::int64_t i = 0; i < n / kQK32; ++i, in += 22, y += kQK32) {
    const float d = h(in);
    const std::uint32_t qh = u32(in + 2);
    for (int j = 0; j < kQK32 / 2; ++j) {
      const std::uint8_t xh0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const std::uint8_t xh1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
      const std::int32_t x0 = ((in[6 + j] & 0x0F) | xh0) - 16, x1 = ((in[6 + j] >> 4) | xh1) - 16;
      y[j] = static_cast<float>(x0) * d;
      y[j + kQK32 / 2] = static_cast<float>(x1) * d;
    }
  }
}

void dequantize_q5_1(const std::uint8_t* in, float* y, std::int64_t n) {  // d, m, qh[4], qs[16]
  for (std::int64_t i = 0; i < n / kQK32; ++i, in += 24, y += kQK32) {
    const float d = h(in), m = h(in + 2);
    const std::uint32_t qh = u32(in + 4);
    for (int j = 0; j < kQK32 / 2; ++j) {
      const std::uint8_t xh0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const std::uint8_t xh1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
      const int x0 = (in[8 + j] & 0x0F) | xh0, x1 = (in[8 + j] >> 4) | xh1;
      y[j] = static_cast<float>(x0) * d + m;
      y[j + kQK32 / 2] = static_cast<float>(x1) * d + m;
    }
  }
}

// ---- non-linear and microscaling 4-bit

void dequantize_iq4_nl(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs[16]
  for (std::int64_t i = 0; i < n / kQK32; ++i, in += 18, y += kQK32) {
    const float d = h(in);
    for (int j = 0; j < kQK32 / 2; ++j) {
      y[j] = d * static_cast<float>(kIq4nl[in[2 + j] & 0xf]);
      y[j + kQK32 / 2] = d * static_cast<float>(kIq4nl[in[2 + j] >> 4]);
    }
  }
}

void dequantize_mxfp4(const std::uint8_t* in, float* y, std::int64_t n) {  // e (E8M0), qs[16]
  for (std::int64_t i = 0; i < n / kQK32; ++i, in += 17, y += kQK32) {
    const float d = e8m0_half(in[0]);
    for (int j = 0; j < kQK32 / 2; ++j) {
      const std::int8_t x0 = kFp4[in[1 + j] & 0x0F], x1 = kFp4[in[1 + j] >> 4];
      y[j] = static_cast<float>(x0) * d;
      y[j + kQK32 / 2] = static_cast<float>(x1) * d;
    }
  }
}

// ---- ternary

void dequantize_tq1_0(const std::uint8_t* in, float* y, std::int64_t n) {  // qs[48] (5 trits per byte), qh[4] (4 per byte), d
  constexpr std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
  auto trit = [](std::uint8_t q) { return static_cast<std::int16_t>((static_cast<std::uint16_t>(q) * 3) >> 8); };
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 54) {
    const std::uint8_t* qs = in;
    const std::uint8_t* qh = in + 48;
    const float d = h(in + 52);
    for (int nn = 0; nn < 5; ++nn)
      for (int m = 0; m < 32; ++m) *y++ = static_cast<float>(trit(static_cast<std::uint8_t>(qs[m] * pow3[nn])) - 1) * d;
    for (int nn = 0; nn < 5; ++nn)
      for (int m = 0; m < 16; ++m) *y++ = static_cast<float>(trit(static_cast<std::uint8_t>(qs[32 + m] * pow3[nn])) - 1) * d;
    for (int nn = 0; nn < 4; ++nn)
      for (int j = 0; j < 4; ++j) *y++ = static_cast<float>(trit(static_cast<std::uint8_t>(qh[j] * pow3[nn])) - 1) * d;
  }
}

void dequantize_i2_s(const std::uint8_t* in, float* y, std::int64_t n) {  // 32 bytes per 128 values, 2-bit codes, no scale
  for (std::int64_t i = 0; i < n / 128; ++i, in += 32, y += 128)
    for (int g = 0; g < 4; ++g)
      for (int j = 0; j < 32; ++j) y[32 * g + j] = static_cast<float>(static_cast<int>((in[j] >> (6 - 2 * g)) & 3) - 1);
}

void dequantize_tq2_0(const std::uint8_t* in, float* y, std::int64_t n) {  // qs[64] (2 bits each), d
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 66) {
    const float d = h(in + 64);
    for (int j = 0; j < 64; j += 32)
      for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 32; ++m) {
          const std::int8_t q = static_cast<std::int8_t>((in[j + m] >> (l * 2)) & 3);
          *y++ = static_cast<float>(q - 1) * d;
        }
  }
}

// ---- grid types

void dequantize_iq2_xxs(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs: 8 bytes per 32 values
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 66) {
    const float d = h(in);
    for (int ib32 = 0; ib32 < kQKK / 32; ++ib32) {
      const std::uint8_t* q = in + 2 + 8 * ib32;
      const std::uint32_t aux1 = u32(q + 4);
      const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const std::uint64_t g = iq::kIq2xxsGrid[q[l]];
        const std::uint8_t signs = iq::kSigns[(aux1 >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j) y[j] = db * static_cast<float>(byte_of(g, j)) * sgn(signs, j);
        y += 8;
      }
    }
  }
}

void dequantize_iq2_xs(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs u16[32], scales[8]
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 74) {
    const float d = h(in);
    const std::uint8_t* scales = in + 66;
    for (int ib32 = 0; ib32 < kQKK / 32; ++ib32) {
      const float db[2] = {d * (0.5f + static_cast<float>(scales[ib32] & 0xf)) * 0.25f, d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f};
      for (int l = 0; l < 4; ++l) {
        const std::uint16_t qv = u16(in + 2 + 2 * (4 * ib32 + l));
        const std::uint64_t g = iq::kIq2xsGrid[qv & 511];
        const std::uint8_t signs = iq::kSigns[qv >> 9];
        for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * static_cast<float>(byte_of(g, j)) * sgn(signs, j);
        y += 8;
      }
    }
  }
}

void dequantize_iq2_s(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs[64] (32 indices + 32 sign bytes), qh[8], scales[8]
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 82) {
    const float d = h(in);
    const std::uint8_t* qs = in + 2;
    const std::uint8_t* qh = in + 66;
    const std::uint8_t* scales = in + 74;
    const std::uint8_t* signs = qs + kQKK / 8;
    for (int ib32 = 0; ib32 < kQKK / 32; ++ib32) {
      const float db[2] = {d * (0.5f + static_cast<float>(scales[ib32] & 0xf)) * 0.25f, d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f};
      for (int l = 0; l < 4; ++l) {
        const float dl = db[l / 2];
        const std::uint64_t g = iq::kIq2sGrid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
        for (int j = 0; j < 8; ++j) y[j] = dl * static_cast<float>(byte_of(g, j)) * sgn(signs[l], j);
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
}

void dequantize_iq3_xxs(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs[64] grid indices, then 32 bytes of scales and signs
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 98) {
    const float d = h(in);
    const std::uint8_t* qs = in + 2;
    const std::uint8_t* scales_and_signs = qs + kQKK / 4;
    for (int ib32 = 0; ib32 < kQKK / 32; ++ib32) {
      const std::uint32_t aux = u32(scales_and_signs + 4 * ib32);
      const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const std::uint8_t signs = iq::kSigns[(aux >> 7 * l) & 127];
        const std::uint32_t g1 = iq::kIq3xxsGrid[qs[2 * l + 0]], g2 = iq::kIq3xxsGrid[qs[2 * l + 1]];
        for (int j = 0; j < 4; ++j) {
          y[j + 0] = db * static_cast<float>(byte_of(g1, j)) * sgn(signs, j + 0);
          y[j + 4] = db * static_cast<float>(byte_of(g2, j)) * sgn(signs, j + 4);
        }
        y += 8;
      }
      qs += 8;
    }
  }
}

void dequantize_iq3_s(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs[64], qh[8], signs[32], scales[4]
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 110) {
    const float d = h(in);
    const std::uint8_t* qs = in + 2;
    const std::uint8_t* qh = in + 66;
    const std::uint8_t* signs = in + 74;
    const std::uint8_t* scales = in + 106;
    for (int ib32 = 0; ib32 < kQKK / 32; ib32 += 2) {
      const float db1 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] & 0xf));
      const float db2 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] >> 4));
      for (int half = 0; half < 2; ++half) {
        const float db = half == 0 ? db1 : db2;
        for (int l = 0; l < 4; ++l) {
          const std::uint32_t g1 = iq::kIq3sGrid[qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256)];
          const std::uint32_t g2 = iq::kIq3sGrid[qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)];
          for (int j = 0; j < 4; ++j) {
            y[j + 0] = db * static_cast<float>(byte_of(g1, j)) * sgn(signs[l], j + 0);
            y[j + 4] = db * static_cast<float>(byte_of(g2, j)) * sgn(signs[l], j + 4);
          }
          y += 8;
        }
        qs += 8;
        signs += 4;
      }
      qh += 2;
    }
  }
}

void dequantize_iq1_s(const std::uint8_t* in, float* y, std::int64_t n) {  // d, qs[32], qh u16[8]
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 50) {
    const float d = h(in);
    const std::uint8_t* qs = in + 2;
    for (int ib = 0; ib < kQKK / 32; ++ib) {
      const std::uint16_t qh = u16(in + 34 + 2 * ib);
      const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
      const float delta = (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
      for (int l = 0; l < 4; ++l) {
        const std::uint64_t g = iq::kIq1sGrid[qs[l] | (((qh >> 3 * l) & 7) << 8)];
        for (int j = 0; j < 8; ++j) y[j] = dl * (static_cast<float>(static_cast<std::int8_t>(byte_of(g, j))) + delta);
        y += 8;
      }
      qs += 4;
    }
  }
}

void dequantize_iq1_m(const std::uint8_t* in, float* y, std::int64_t n) {  // qs[32], qh[16], scales[8] (the fp16 scale in their top nibbles)
  for (std::int64_t i = 0; i < n / kQKK; ++i, in += 56) {
    const std::uint8_t* qs = in;
    const std::uint8_t* qh = in + 32;
    std::uint16_t sc[4];
    for (int k = 0; k < 4; ++k) sc[k] = u16(in + 48 + 2 * k);
    const std::uint16_t scale16 =
        static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = fp16_to_fp32(scale16);
    for (int ib = 0; ib < kQKK / 32; ++ib) {
      const float dl1 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
      const float dl2 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
      const std::uint16_t idx[4] = {static_cast<std::uint16_t>(qs[0] | ((qh[0] << 8) & 0x700)), static_cast<std::uint16_t>(qs[1] | ((qh[0] << 4) & 0x700)),
                                    static_cast<std::uint16_t>(qs[2] | ((qh[1] << 8) & 0x700)), static_cast<std::uint16_t>(qs[3] | ((qh[1] << 4) & 0x700))};
      const float delta[4] = {(qh[0] & 0x08) ? -kIq1sDelta : kIq1sDelta, (qh[0] & 0x80) ? -kIq1sDelta : kIq1sDelta,
                              (qh[1] & 0x08) ? -kIq1sDelta : kIq1sDelta, (qh[1] & 0x80) ? -kIq1sDelta : kIq1sDelta};
      for (int l = 0; l < 4; ++l) {
        const float dl = l < 2 ? dl1 : dl2;
        const std::uint64_t g = iq::kIq1sGrid[idx[l]];
        for (int j = 0; j < 8; ++j) y[j] = dl * (static_cast<float>(static_cast<std::int8_t>(byte_of(g, j))) + delta[l]);
        y += 8;
      }
      qs += 4;
      qh += 2;
    }
  }
}

}  // namespace pelias::quant

// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_decode.h - GP3-10 GPU: one 32-value sub-block of a weight row, decoded straight from the GGUF bytes as stored.
// The same functions compile for the GPU (nvcc, __host__ __device__) and for the host, so the unit test (test_gpu_decode)
// proves in any build, without a GPU, that each decoder gives exactly the floats of quant::dequantize_row. The float
// expressions follow quant_cpu.cpp operation for operation; on the device every multiply and add goes through mul / add
// (__fmul_rn / __fadd_rn), which nvcc never fuses, so the device values are the reference's bit for bit as well.
// Byte reads only: the stored blocks are 17 to 264 bytes, most of them odd sizes, so nothing here assumes alignment.
// Every type quant::dequantize_row reads is here (32: the float types, the legacy 32-blocks, the K-quants, the IQ grids,
// the ternary BitNet types TQ1_0 / TQ2_0 / I2_S, MXFP4, NVFP4 and the Pelias types). I2_S codes carry no scale (the
// tensor's scale is applied after the product, as on the CPU path).
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define PG_HD __host__ __device__ __forceinline__
#else
#define PG_HD inline
#endif

namespace pelias::kern::gpu {

// The types the GPU path reads (quant::Type ids): every type quant::dequantize_row reads
enum GType : std::uint32_t {
  g_f32 = 0, g_f16 = 1, g_q4_0 = 2, g_q4_1 = 3, g_q5_0 = 6, g_q5_1 = 7, g_q8_0 = 8, g_q2_k = 10, g_q3_k = 11, g_q4_k = 12,
  g_q5_k = 13, g_q6_k = 14, g_iq2_xxs = 16, g_iq2_xs = 17, g_iq3_xxs = 18, g_iq1_s = 19, g_iq4_nl = 20, g_iq3_s = 21,
  g_iq2_s = 22, g_iq4_xs = 23, g_iq1_m = 29, g_bf16 = 30, g_tq1_0 = 34, g_tq2_0 = 35, g_i2_s = 36, g_mxfp4 = 39, g_nvfp4 = 40,
  g_q8_z = 200, g_q7_z = 201, g_q7_zt = 202, g_q4_tg = 203, g_q4_tg2 = 204
};

// The format tables a decoder reads (host or device copies; the GPU's are uploaded once by kern/cuda/gpu_ops.cu):
// the Q4_TG grids (4 tags x 16 levels; quant::q4_tg_grid(tag, 1) and (tag, 2)), the IQ grids and the IQ2 / IQ3 sign
// patterns (quant/iq_tables.h). A type that needs none of them reads none.
struct Tables {
  const float* tg1 = nullptr;
  const float* tg2 = nullptr;
  const std::uint64_t* iq2xxs = nullptr;  // 256
  const std::uint64_t* iq2xs = nullptr;   // 512
  const std::uint64_t* iq2s = nullptr;    // 1024
  const std::uint32_t* iq3xxs = nullptr;  // 256
  const std::uint32_t* iq3s = nullptr;    // 512
  const std::uint64_t* iq1s = nullptr;    // 2048
  const std::uint8_t* signs = nullptr;    // 128
};

// values per block and bytes per block (0 for a type the GPU path does not read)
PG_HD std::uint32_t block_values(std::uint32_t t) {
  switch (t) {
    case g_f32: case g_f16: case g_bf16: return 1;
    case g_q4_0: case g_q4_1: case g_q5_0: case g_q5_1: case g_q8_0: case g_iq4_nl: case g_mxfp4: return 32;
    case g_nvfp4: return 64;
    case g_i2_s: return 128;
    case g_q2_k: case g_q3_k: case g_q4_k: case g_q5_k: case g_q6_k: case g_iq2_xxs: case g_iq2_xs: case g_iq3_xxs:
    case g_iq1_s: case g_iq3_s: case g_iq2_s: case g_iq4_xs: case g_iq1_m: case g_tq1_0: case g_tq2_0: case g_q8_z:
    case g_q7_z: case g_q7_zt: case g_q4_tg: case g_q4_tg2:
      return 256;
    default: return 0;
  }
}
PG_HD std::uint32_t block_bytes(std::uint32_t t) {
  switch (t) {
    case g_f32: return 4;
    case g_f16: case g_bf16: return 2;
    case g_q4_0: return 18;
    case g_q4_1: return 20;
    case g_q5_0: return 22;
    case g_q5_1: return 24;
    case g_q8_0: return 34;
    case g_iq4_nl: return 18;
    case g_mxfp4: return 17;
    case g_nvfp4: return 36;
    case g_i2_s: return 32;
    case g_q2_k: return 84;
    case g_q3_k: return 110;
    case g_q4_k: return 144;
    case g_q5_k: return 176;
    case g_q6_k: return 210;
    case g_iq2_xxs: return 66;
    case g_iq2_xs: return 74;
    case g_iq2_s: return 82;
    case g_iq3_xxs: return 98;
    case g_iq3_s: return 110;
    case g_iq1_s: return 50;
    case g_iq1_m: return 56;
    case g_iq4_xs: return 136;
    case g_tq1_0: return 54;
    case g_tq2_0: return 66;
    case g_q8_z: return 264;
    case g_q7_z: return 232;
    case g_q7_zt: return 241;
    case g_q4_tg: case g_q4_tg2: return 144;
    default: return 0;
  }
}
PG_HD bool type_supported(std::uint32_t t) { return block_values(t) != 0; }

PG_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
PG_HD float add(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
PG_HD float sub(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}
PG_HD float quo(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

PG_HD float bits_float(std::uint32_t b) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(b);
#else
  float f;
  std::memcpy(&f, &b, 4);
  return f;
#endif
}

// IEEE half to float by bits (exact for every half, subnormals included), as quant::fp16_to_fp32
PG_HD float half_float(std::uint16_t h) {
  const std::uint32_t s = static_cast<std::uint32_t>(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
  std::uint32_t bits;
  if (e == 0) {
    if (m == 0) {
      bits = s;
    } else {
      int ee = -1;
      std::uint32_t mm = m;
      do {
        ++ee;
        mm <<= 1;
      } while ((mm & 1024u) == 0);
      bits = s | static_cast<std::uint32_t>(127 - 15 - ee) << 23 | (mm & 1023u) << 13;
    }
  } else if (e == 31) {
    bits = s | 0x7f800000u | m << 13;
  } else {
    bits = s | (e + 112u) << 23 | m << 13;
  }
  return bits_float(bits);
}

PG_HD std::uint16_t rd16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
PG_HD std::uint32_t rd32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 | static_cast<std::uint32_t>(p[2]) << 16 |
         static_cast<std::uint32_t>(p[3]) << 24;
}

// the K-quant 6-bit scale and min of sub-block j (quant::detail::get_scale_min_k4)
PG_HD void scale_min_k4(int j, const std::uint8_t* q, int* sc, int* mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

PG_HD float iq4nl_value(int i) {
  // the IQ4_NL / IQ4_XS level table (quant_cpu.cpp kValuesIq4nl)
  constexpr signed char v[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  return static_cast<float>(v[i & 15]);
}

// E2M1 levels times two (MXFP4, NVFP4; dequant_more.cpp kFp4)
PG_HD float fp4_value(int i) {
  constexpr signed char v[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return static_cast<float>(v[i & 15]);
}
// E8M0 exponent byte to 2^(x-127) * 0.5 (dequant_more.cpp e8m0_half)
PG_HD float e8m0_half(std::uint8_t x) {
  return bits_float(x < 2 ? (0x00200000u << x) : (static_cast<std::uint32_t>(x - 1) << 23));
}
// unsigned E4M3 halved (quant::ue4m3_to_fp32); every step is exact (a power-of-two scale of a 4-bit mantissa)
PG_HD float ue4m3_half(std::uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  const int ex = (x >> 3) & 0xF, man = x & 0x7;
  const float raw = ex == 0 ? mul(static_cast<float>(man), bits_float(static_cast<std::uint32_t>(127 - 9) << 23))
                            : mul(add(1.0f, quo(static_cast<float>(man), 8.0f)), bits_float(static_cast<std::uint32_t>(127 + ex - 7) << 23));
  return mul(raw, 0.5f);
}
PG_HD float sign_of(std::uint8_t signs, int j) { return ((static_cast<unsigned>(signs) >> j) & 1u) ? -1.0f : 1.0f; }
PG_HD float grid_byte(std::uint64_t g, int j) { return static_cast<float>(static_cast<std::uint8_t>(g >> (8 * j))); }
// the ternary code of one TQ1_0 byte position (dequant_more.cpp: trit of (q * 3^n) mod 256)
PG_HD int tq1_trit(std::uint8_t q, int nn) {
  constexpr std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
  const std::uint8_t v = static_cast<std::uint8_t>(q * pow3[nn]);
  return static_cast<int>((static_cast<std::uint16_t>(v) * 3) >> 8);
}

// Sub-block j (32 values starting at value 32 j of the row) of a row of type t into y[0..31]. `row` points at the row's
// first byte. tb: the format tables (Q4_TG / Q4_TG2 grids, IQ grids and sign patterns).
PG_HD void decode32(std::uint32_t t, const std::uint8_t* row, std::int64_t j, const Tables& tb, float* y) {
  switch (t) {
    case g_f32:
      for (int i = 0; i < 32; ++i) y[i] = bits_float(rd32(row + 4 * (32 * j + i)));
      return;
    case g_f16:
      for (int i = 0; i < 32; ++i) y[i] = half_float(rd16(row + 2 * (32 * j + i)));
      return;
    case g_bf16:
      for (int i = 0; i < 32; ++i) y[i] = bits_float(static_cast<std::uint32_t>(rd16(row + 2 * (32 * j + i))) << 16);
      return;
    case g_q8_0: {
      const std::uint8_t* b = row + 34 * j;
      const float d = half_float(rd16(b));
      for (int i = 0; i < 32; ++i) y[i] = mul(static_cast<float>(static_cast<signed char>(b[2 + i])), d);
      return;
    }
    case g_q4_k: {
      const std::uint8_t* b = row + 144 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b)), mn = half_float(rd16(b + 2));
      int sc, m;
      scale_min_k4(js, b + 4, &sc, &m);
      const float d1 = mul(d, static_cast<float>(sc)), m1 = mul(mn, static_cast<float>(m));
      const std::uint8_t* q = b + 16 + 32 * (js / 2);
      for (int l = 0; l < 32; ++l) {
        const int v = (js & 1) ? (q[l] >> 4) : (q[l] & 0xF);
        y[l] = sub(mul(d1, static_cast<float>(v)), m1);
      }
      return;
    }
    case g_q5_k: {
      const std::uint8_t* b = row + 176 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b)), mn = half_float(rd16(b + 2));
      int sc, m;
      scale_min_k4(js, b + 4, &sc, &m);
      const float d1 = mul(d, static_cast<float>(sc)), m1 = mul(mn, static_cast<float>(m));
      const std::uint8_t* qh = b + 16;
      const std::uint8_t* ql = b + 48 + 32 * (js / 2);
      for (int l = 0; l < 32; ++l) {
        const int lo = (js & 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
        const int v = lo + (((qh[l] >> js) & 1) ? 16 : 0);
        y[l] = sub(mul(d1, static_cast<float>(v)), m1);
      }
      return;
    }
    case g_q6_k: {
      const std::uint8_t* b = row + 210 * (j / 8);
      const int js = static_cast<int>(j % 8), half = js / 4, part = js % 4;
      const float d = half_float(rd16(b + 208));
      const std::uint8_t* ql = b + 64 * half;
      const std::uint8_t* qh = b + 128 + 32 * half;
      const std::uint8_t* sc = b + 192 + 8 * half;
      for (int l = 0; l < 32; ++l) {
        int q;
        if (part == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
        else if (part == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
        else if (part == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
        else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        const float s = static_cast<float>(static_cast<signed char>(sc[2 * part + l / 16]));
        y[l] = mul(mul(d, s), static_cast<float>(q - 32));
      }
      return;
    }
    case g_iq4_xs: {
      const std::uint8_t* b = row + 136 * (j / 8);
      const int ib = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint16_t sh = rd16(b + 2);
      const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
      const float dl = mul(d, static_cast<float>(ls - 32));
      const std::uint8_t* qs = b + 8 + 16 * ib;
      for (int l = 0; l < 16; ++l) {
        y[l] = mul(dl, iq4nl_value(qs[l] & 0xf));
        y[l + 16] = mul(dl, iq4nl_value(qs[l] >> 4));
      }
      return;
    }
    case g_q8_z: {
      const std::uint8_t* b = row + 264 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      std::uint64_t packed = 0;
      for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
      const float s = quo(mul(d, static_cast<float>((packed >> (6 * js)) & 63u)), 63.0f);
      for (int i = 0; i < 32; ++i) y[i] = mul(static_cast<float>(static_cast<signed char>(b[8 + 32 * js + i])), s);
      return;
    }
    case g_q7_zt: {
      const std::uint8_t* b = row + 241 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      std::uint64_t packed = 0;
      for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
      const float s = quo(mul(d, static_cast<float>((packed >> (6 * js)) & 63u)), 63.0f);
      const unsigned mask = b[8];
      int slot = 0;  // gated blocks before js (the format has two; a corrupt mask reads no H word past them)
      for (int t2 = 0; t2 < js; ++t2) slot += static_cast<int>((mask >> t2) & 1u);
      const bool gated = ((mask >> js) & 1u) && slot < 2;
      const std::uint32_t hb = gated ? rd32(b + 233 + 4 * slot) : 0u;
      for (int g = 0; g < 4; ++g) {
        std::uint64_t wv = 0;
        for (int k = 0; k < 7; ++k) wv |= static_cast<std::uint64_t>(b[9 + 7 * (4 * js + g) + k]) << (8 * k);
        for (int k = 0; k < 8; ++k) {
          const int bb = static_cast<int>((wv >> (7 * k)) & 127u), idx = 8 * g + k;
          const int q = gated ? 2 * bb + static_cast<int>((hb >> idx) & 1u) - 127 : 2 * bb - 127;
          y[idx] = mul(static_cast<float>(q), s);
        }
      }
      return;
    }
    case g_q4_tg2: {
      const std::uint8_t* b = row + 144 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      std::uint64_t packed = 0;
      for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
      const float s = quo(mul(d, static_cast<float>((packed >> (6 * js)) & 63u)), 63.0f);
      const unsigned tg = b[8 + js];
      for (int g = 0; g < 4; ++g) {
        const float* gr = tb.tg2 + 16 * ((tg >> (2 * g)) & 3u);
        for (int k = 0; k < 8; ++k) {
          const int idx = 32 * js + 8 * g + k;
          y[8 * g + k] = mul(s, gr[(b[16 + idx / 2] >> (4 * (idx & 1))) & 15]);
        }
      }
      return;
    }
    case g_q4_tg: {
      const std::uint8_t* b = row + 144 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      std::uint64_t packed = 0;
      for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
      const float s = quo(mul(d, static_cast<float>((packed >> (6 * js)) & 63u)), 63.0f);
      const unsigned tg = b[8 + js];
      for (int g = 0; g < 4; ++g) {
        const float* gr = tb.tg1 + 16 * ((tg >> (2 * g)) & 3u);
        for (int k = 0; k < 8; ++k) {
          const int idx = 32 * js + 8 * g + k;
          y[8 * g + k] = mul(s, gr[(b[16 + idx / 2] >> (4 * (idx & 1))) & 15]);
        }
      }
      return;
    }
    case g_q7_z: {
      const std::uint8_t* b = row + 232 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      std::uint64_t packed = 0;
      for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(b[2 + k]) << (8 * k);
      const float s = quo(mul(d, static_cast<float>((packed >> (6 * js)) & 63u)), 63.0f);
      for (int g = 0; g < 4; ++g) {
        std::uint64_t wv = 0;
        for (int k = 0; k < 7; ++k) wv |= static_cast<std::uint64_t>(b[8 + 7 * (4 * js + g) + k]) << (8 * k);
        for (int k = 0; k < 8; ++k) y[8 * g + k] = mul(static_cast<float>(static_cast<int>((wv >> (7 * k)) & 127u) - 64), s);
      }
      return;
    }
    case g_q4_0: {
      const std::uint8_t* b = row + 18 * j;
      const float d = half_float(rd16(b));
      for (int l = 0; l < 16; ++l) {
        y[l] = mul(static_cast<float>((b[2 + l] & 0x0F) - 8), d);
        y[l + 16] = mul(static_cast<float>((b[2 + l] >> 4) - 8), d);
      }
      return;
    }
    case g_q4_1: {
      const std::uint8_t* b = row + 20 * j;
      const float d = half_float(rd16(b)), m = half_float(rd16(b + 2));
      for (int l = 0; l < 16; ++l) {
        y[l] = add(mul(static_cast<float>(b[4 + l] & 0x0F), d), m);
        y[l + 16] = add(mul(static_cast<float>(b[4 + l] >> 4), d), m);
      }
      return;
    }
    case g_q5_0: {
      const std::uint8_t* b = row + 22 * j;
      const float d = half_float(rd16(b));
      const std::uint32_t qh = rd32(b + 2);
      for (int l = 0; l < 16; ++l) {
        const int xh0 = static_cast<int>(((qh >> l) << 4) & 0x10), xh1 = static_cast<int>((qh >> (l + 12)) & 0x10);
        y[l] = mul(static_cast<float>(((b[6 + l] & 0x0F) | xh0) - 16), d);
        y[l + 16] = mul(static_cast<float>(((b[6 + l] >> 4) | xh1) - 16), d);
      }
      return;
    }
    case g_q5_1: {
      const std::uint8_t* b = row + 24 * j;
      const float d = half_float(rd16(b)), m = half_float(rd16(b + 2));
      const std::uint32_t qh = rd32(b + 4);
      for (int l = 0; l < 16; ++l) {
        const int xh0 = static_cast<int>(((qh >> l) << 4) & 0x10), xh1 = static_cast<int>((qh >> (l + 12)) & 0x10);
        y[l] = add(mul(static_cast<float>((b[8 + l] & 0x0F) | xh0), d), m);
        y[l + 16] = add(mul(static_cast<float>((b[8 + l] >> 4) | xh1), d), m);
      }
      return;
    }
    case g_iq4_nl: {
      const std::uint8_t* b = row + 18 * j;
      const float d = half_float(rd16(b));
      for (int l = 0; l < 16; ++l) {
        y[l] = mul(d, iq4nl_value(b[2 + l] & 0xf));
        y[l + 16] = mul(d, iq4nl_value(b[2 + l] >> 4));
      }
      return;
    }
    case g_mxfp4: {
      const std::uint8_t* b = row + 17 * j;
      const float d = e8m0_half(b[0]);
      for (int l = 0; l < 16; ++l) {
        y[l] = mul(fp4_value(b[1 + l] & 0x0F), d);
        y[l + 16] = mul(fp4_value(b[1 + l] >> 4), d);
      }
      return;
    }
    case g_nvfp4: {
      const std::uint8_t* b = row + 36 * (j / 2);
      for (int h = 0; h < 2; ++h) {
        const int sidx = 2 * static_cast<int>(j % 2) + h;
        const float d = ue4m3_half(b[sidx]);
        for (int l = 0; l < 8; ++l) {
          const std::uint8_t q = b[4 + 8 * sidx + l];
          y[16 * h + l] = mul(fp4_value(q & 0x0F), d);
          y[16 * h + l + 8] = mul(fp4_value(q >> 4), d);
        }
      }
      return;
    }
    case g_i2_s: {
      const std::uint8_t* b = row + 32 * (j / 4);
      const int g = static_cast<int>(j % 4);
      for (int l = 0; l < 32; ++l) y[l] = static_cast<float>(static_cast<int>((b[l] >> (6 - 2 * g)) & 3) - 1);
      return;
    }
    case g_tq2_0: {
      const std::uint8_t* b = row + 66 * (j / 8);
      const int js = static_cast<int>(j % 8), half = js / 4, l2 = js % 4;
      const float d = half_float(rd16(b + 64));
      for (int m = 0; m < 32; ++m) y[m] = mul(static_cast<float>(static_cast<int>((b[32 * half + m] >> (2 * l2)) & 3) - 1), d);
      return;
    }
    case g_tq1_0: {
      const std::uint8_t* b = row + 54 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b + 52));
      for (int i = 0; i < 32; ++i) {
        const int v = 32 * js + i;  // value in the block: 160 from qs[0..31], 80 from qs[32..47], 16 from qh[0..3]
        int q;
        if (v < 160) q = tq1_trit(b[v % 32], v / 32);
        else if (v < 240) q = tq1_trit(b[32 + (v - 160) % 16], (v - 160) / 16);
        else q = tq1_trit(b[48 + (v - 240) % 4], (v - 240) / 4);
        y[i] = mul(static_cast<float>(q - 1), d);
      }
      return;
    }
    case g_q2_k: {
      const std::uint8_t* b = row + 84 * (j / 8);
      const int js = static_cast<int>(j % 8), half = js / 4, jq = js % 4, shift = 2 * jq;
      const float d = half_float(rd16(b + 80)), mn = half_float(rd16(b + 82));
      const std::uint8_t* q = b + 16 + 32 * half;
      for (int p = 0; p < 2; ++p) {
        const std::uint8_t sc = b[8 * half + 2 * jq + p];
        const float dl = mul(d, static_cast<float>(sc & 0xF)), ml = mul(mn, static_cast<float>(sc >> 4));
        for (int l = 0; l < 16; ++l) y[16 * p + l] = sub(mul(dl, static_cast<float>((q[16 * p + l] >> shift) & 3)), ml);
      }
      return;
    }
    case g_q3_k: {
      const std::uint8_t* b = row + 110 * (j / 8);
      const int js = static_cast<int>(j % 8), half = js / 4, jq = js % 4, shift = 2 * jq;
      const float d_all = half_float(rd16(b + 108));
      constexpr std::uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
      const std::uint32_t a0 = rd32(b + 96), a1 = rd32(b + 100), tmp = rd32(b + 104);
      std::uint32_t aux[4];
      aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
      aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
      aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
      aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      const std::uint8_t* hm = b;
      const std::uint8_t* q = b + 32 + 32 * half;
      const unsigned m = 1u << js;
      for (int p = 0; p < 2; ++p) {
        const int is = 8 * half + 2 * jq + p;
        const int sc = static_cast<signed char>(static_cast<std::uint8_t>(aux[is / 4] >> (8 * (is % 4))));
        const float dl = mul(d_all, static_cast<float>(sc - 32));
        for (int l = 0; l < 16; ++l) {
          const int k = 16 * p + l;
          y[k] = mul(dl, static_cast<float>(static_cast<int>((q[k] >> shift) & 3) - ((hm[k] & m) ? 0 : 4)));
        }
      }
      return;
    }
    case g_iq2_xxs: {
      const std::uint8_t* b = row + 66 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint8_t* q = b + 2 + 8 * js;
      const std::uint32_t aux1 = rd32(q + 4);
      const float db = mul(mul(d, add(0.5f, static_cast<float>(aux1 >> 28))), 0.25f);
      for (int l = 0; l < 4; ++l) {
        const std::uint64_t g = tb.iq2xxs[q[l]];
        const std::uint8_t signs = tb.signs[(aux1 >> (7 * l)) & 127];
        for (int k = 0; k < 8; ++k) y[8 * l + k] = mul(mul(db, grid_byte(g, k)), sign_of(signs, k));
      }
      return;
    }
    case g_iq2_xs: {
      const std::uint8_t* b = row + 74 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint8_t sc = b[66 + js];
      const float db0 = mul(mul(d, add(0.5f, static_cast<float>(sc & 0xf))), 0.25f);
      const float db1 = mul(mul(d, add(0.5f, static_cast<float>(sc >> 4))), 0.25f);
      for (int l = 0; l < 4; ++l) {
        const std::uint16_t qv = rd16(b + 2 + 2 * (4 * js + l));
        const std::uint64_t g = tb.iq2xs[qv & 511];
        const std::uint8_t signs = tb.signs[qv >> 9];
        const float dl = l < 2 ? db0 : db1;
        for (int k = 0; k < 8; ++k) y[8 * l + k] = mul(mul(dl, grid_byte(g, k)), sign_of(signs, k));
      }
      return;
    }
    case g_iq2_s: {
      const std::uint8_t* b = row + 82 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint8_t* qs = b + 2 + 4 * js;
      const std::uint8_t* sg = b + 2 + 32 + 4 * js;
      const int qh = b[66 + js];
      const std::uint8_t sc = b[74 + js];
      const float db0 = mul(mul(d, add(0.5f, static_cast<float>(sc & 0xf))), 0.25f);
      const float db1 = mul(mul(d, add(0.5f, static_cast<float>(sc >> 4))), 0.25f);
      for (int l = 0; l < 4; ++l) {
        const float dl = l < 2 ? db0 : db1;
        const std::uint64_t g = tb.iq2s[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
        for (int k = 0; k < 8; ++k) y[8 * l + k] = mul(mul(dl, grid_byte(g, k)), sign_of(sg[l], k));
      }
      return;
    }
    case g_iq3_xxs: {
      const std::uint8_t* b = row + 98 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint8_t* qs = b + 2 + 8 * js;
      const std::uint32_t aux = rd32(b + 2 + 64 + 4 * js);
      const float db = mul(mul(d, add(0.5f, static_cast<float>(aux >> 28))), 0.5f);
      for (int l = 0; l < 4; ++l) {
        const std::uint8_t signs = tb.signs[(aux >> (7 * l)) & 127];
        const std::uint32_t g1 = tb.iq3xxs[qs[2 * l]], g2 = tb.iq3xxs[qs[2 * l + 1]];
        for (int k = 0; k < 4; ++k) {
          y[8 * l + k] = mul(mul(db, grid_byte(g1, k)), sign_of(signs, k));
          y[8 * l + k + 4] = mul(mul(db, grid_byte(g2, k)), sign_of(signs, k + 4));
        }
      }
      return;
    }
    case g_iq3_s: {
      const std::uint8_t* b = row + 110 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint8_t* qs = b + 2 + 8 * js;
      const int qh = b[66 + js];
      const std::uint8_t* sg = b + 74 + 4 * js;
      const std::uint8_t sc = b[106 + js / 2];
      const float db = mul(d, static_cast<float>(1 + 2 * ((js % 2 == 0) ? (sc & 0xf) : (sc >> 4))));
      for (int l = 0; l < 4; ++l) {
        const std::uint32_t g1 = tb.iq3s[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
        const std::uint32_t g2 = tb.iq3s[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
        for (int k = 0; k < 4; ++k) {
          y[8 * l + k] = mul(mul(db, grid_byte(g1, k)), sign_of(sg[l], k));
          y[8 * l + k + 4] = mul(mul(db, grid_byte(g2, k)), sign_of(sg[l], k + 4));
        }
      }
      return;
    }
    case g_iq1_s: {
      const std::uint8_t* b = row + 50 * (j / 8);
      const int js = static_cast<int>(j % 8);
      const float d = half_float(rd16(b));
      const std::uint8_t* qs = b + 2 + 4 * js;
      const std::uint16_t qh = rd16(b + 34 + 2 * js);
      const float dl = mul(d, static_cast<float>(2 * ((qh >> 12) & 7) + 1));
      const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
      for (int l = 0; l < 4; ++l) {
        const std::uint64_t g = tb.iq1s[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
        for (int k = 0; k < 8; ++k)
          y[8 * l + k] = mul(dl, add(static_cast<float>(static_cast<signed char>(static_cast<std::uint8_t>(g >> (8 * k)))), delta));
      }
      return;
    }
    case g_iq1_m: {
      const std::uint8_t* b = row + 56 * (j / 8);
      const int ib = static_cast<int>(j % 8);
      std::uint16_t sc[4];
      for (int k = 0; k < 4; ++k) sc[k] = rd16(b + 48 + 2 * k);
      const std::uint16_t scale16 =
          static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
      const float d = half_float(scale16);
      const float dl1 = mul(d, static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1));
      const float dl2 = mul(d, static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1));
      const std::uint8_t* qs = b + 4 * ib;
      const std::uint8_t* qh = b + 32 + 2 * ib;
      const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700),
                          qs[3] | ((qh[1] << 4) & 0x700)};
      const float delta[4] = {(qh[0] & 0x08) ? -0.125f : 0.125f, (qh[0] & 0x80) ? -0.125f : 0.125f, (qh[1] & 0x08) ? -0.125f : 0.125f,
                              (qh[1] & 0x80) ? -0.125f : 0.125f};
      for (int l = 0; l < 4; ++l) {
        const float dl = l < 2 ? dl1 : dl2;
        const std::uint64_t g = tb.iq1s[idx[l]];
        for (int k = 0; k < 8; ++k)
          y[8 * l + k] = mul(dl, add(static_cast<float>(static_cast<signed char>(static_cast<std::uint8_t>(g >> (8 * k)))), delta[l]));
      }
      return;
    }
    default:
      for (int i = 0; i < 32; ++i) y[i] = 0.0f;
      return;
  }
}

}  // namespace pelias::kern::gpu

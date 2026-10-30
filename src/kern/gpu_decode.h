// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_decode.h - GP3-10 GPU: one 32-value sub-block of a weight row, decoded straight from the GGUF bytes as stored.
// The same functions compile for the GPU (nvcc, __host__ __device__) and for the host, so the unit test (test_gpu_decode)
// proves in any build, without a GPU, that each decoder gives exactly the floats of quant::dequantize_row. The float
// expressions follow quant_cpu.cpp operation for operation; on the device every multiply and add goes through mul / add
// (__fmul_rn / __fadd_rn), which nvcc never fuses, so the device values are the reference's bit for bit as well.
// Byte reads only: the stored blocks are 34, 136, 144, 176, 210, 241 or 264 bytes, so nothing here assumes alignment.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define PG_HD __host__ __device__ __forceinline__
#else
#define PG_HD inline
#endif

namespace pelias::kern::gpu {

// The types the GPU path reads (quant::Type ids; the rest stay on the CPU path until they are added here)
enum GType : std::uint32_t {
  g_f32 = 0, g_f16 = 1, g_q8_0 = 8, g_q4_k = 12, g_q5_k = 13, g_q6_k = 14, g_iq4_xs = 23, g_bf16 = 30,
  g_q8_z = 200, g_q7_zt = 202, g_q4_tg2 = 204
};

PG_HD bool type_supported(std::uint32_t t) {
  switch (t) {
    case g_f32: case g_f16: case g_bf16: case g_q8_0: case g_q4_k: case g_q5_k: case g_q6_k: case g_iq4_xs: case g_q8_z:
    case g_q7_zt: case g_q4_tg2:
      return true;
    default:
      return false;
  }
}

// values per block and bytes per block of the supported types (0 for the rest)
PG_HD std::uint32_t block_values(std::uint32_t t) {
  switch (t) {
    case g_f32: case g_f16: case g_bf16: return 1;
    case g_q8_0: return 32;
    case g_q4_k: case g_q5_k: case g_q6_k: case g_iq4_xs: case g_q8_z: case g_q7_zt: case g_q4_tg2: return 256;
    default: return 0;
  }
}
PG_HD std::uint32_t block_bytes(std::uint32_t t) {
  switch (t) {
    case g_f32: return 4;
    case g_f16: case g_bf16: return 2;
    case g_q8_0: return 34;
    case g_q4_k: return 144;
    case g_q5_k: return 176;
    case g_q6_k: return 210;
    case g_iq4_xs: return 136;
    case g_q8_z: return 264;
    case g_q7_zt: return 241;
    case g_q4_tg2: return 144;
    default: return 0;
  }
}

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

// Sub-block j (32 values starting at value 32 j of the row) of a row of type t into y[0..31]. `row` points at the row's
// first byte. grid: the 64 Q4_TG2 grid floats (4 tags x 16 levels; quant::q4_tg_grid(tag, 2)), used by Q4_TG2 only.
PG_HD void decode32(std::uint32_t t, const std::uint8_t* row, std::int64_t j, const float* grid, float* y) {
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
      const unsigned tb = b[8 + js];
      for (int g = 0; g < 4; ++g) {
        const float* gr = grid + 16 * ((tb >> (2 * g)) & 3u);
        for (int k = 0; k < 8; ++k) {
          const int idx = 32 * js + 8 * g + k;
          y[8 * g + k] = mul(s, gr[(b[16 + idx / 2] >> (4 * (idx & 1))) & 15]);
        }
      }
      return;
    }
    default:
      for (int i = 0; i < 32; ++i) y[i] = 0.0f;
      return;
  }
}

}  // namespace pelias::kern::gpu

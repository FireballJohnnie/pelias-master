// provenance: re-expressed (not copied) from ggml ggml/src/ggml-quants.c and ggml/src/ggml-impl.h at llama.cpp commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md), read as a specification so the bytes match
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// Every float expression below keeps the reference's operand order and grouping, because float arithmetic is not
// associative: (w*l)*l and w*(l*l) can round differently, and the bytes must match. Integer-to-float conversions
// happen where the reference has them. Block layouts are written field by field with memcpy, so no struct padding or
// alignment assumption reaches the output.
#include "quant/quant_cpu.h"
#include "quant/quant_internal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace pelias::quant {
namespace {

constexpr int kQK8 = 32;   // elements per Q8_0 / Q8_1 block
constexpr int kQKK = 256;  // elements per K-quant super-block
constexpr float kGroupMaxEps = 1e-15f;

// ---- block layouts (byte offsets inside one block; sizes checked against block_types)
// Q8_0: d (fp16) | qs[32] (int8)                                         34 bytes
// Q8_1: d (fp16) | s (fp16) | qs[32]                                     36 bytes
// Q4_K: d (fp16) | dmin (fp16) | scales[12] | qs[128]                    144 bytes
// Q6_K: ql[128] | qh[64] | scales[16] (int8) | d (fp16)                  210 bytes
// Q8_K: d (fp32) | qs[256] (int8) | bsums[16] (int16)                    292 bytes
constexpr std::size_t kQ8_0Bytes = 34, kQ8_1Bytes = 36, kQ4KBytes = 144, kQ6KBytes = 210, kQ8KBytes = 292, kQ8ZBytes = 264, kQ7ZBytes = 232, kQ7ZTBytes = 241;
constexpr std::size_t kQ2KBytes = 84, kQ3KBytes = 110, kQ5KBytes = 176, kIQ4XSBytes = 136;
// the IQ4_NL / IQ4_XS non-linear 4-bit levels (ggml-common.h kvalues_iq4nl)
constexpr std::int8_t kValuesIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

}  // namespace

// helpers shared with quant_weighted.cpp (quant_internal.h)
namespace detail {

void put_u16(std::uint8_t* p, std::uint16_t v) { std::memcpy(p, &v, 2); }
std::uint16_t get_u16(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

// The reference's make_qkx2_quants: an asymmetric (scale, min) search over `nstep` + 1 candidate scales.
float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, std::uint8_t* L, float* the_min,
                       std::uint8_t* Laux, float rmin, float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    const float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = static_cast<float>(nmax) / (max - min);
  float scale = 1 / iscale;
  float best_error = 0;
  for (int i = 0; i < n; ++i) {
    const int l = nearest_int(iscale * (x[i] - min));
    L[i] = static_cast<std::uint8_t>(std::max(0, std::min(nmax, l)));
    float diff = scale * static_cast<float>(L[i]) + min - x[i];
    diff = use_mad ? std::fabs(diff) : diff * diff;
    const float w = weights[i];
    best_error += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * static_cast<float>(is) + static_cast<float>(nmax)) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = std::max(0, std::min(nmax, l));
      Laux[i] = static_cast<std::uint8_t>(l);
      const float w = weights[i];
      const float lf = static_cast<float>(l);
      sum_l += w * lf;
      sum_l2 += w * lf * lf;
      sum_xl += w * lf * x[i];
    }
    const float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float cur_error = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * static_cast<float>(Laux[i]) + this_min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        const float w = weights[i];
        cur_error += w * diff;
      }
      if (cur_error < best_error) {
        for (int i = 0; i < n; ++i) L[i] = Laux[i];
        best_error = cur_error;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

// The reference's make_qx_quants: a symmetric scale search, levels stored offset by nmax; `qw` (importance weights,
// may be null) replaces the rmse_type weighting when given.
float make_qx_quants(int n, int nmax, const float* x, std::int8_t* L, int rmse_type, const float* qw) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    const float ax = std::fabs(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < kGroupMaxEps) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  float iscale = static_cast<float>(-nmax) / max;
  if (rmse_type == 0) {
    for (int i = 0; i < n; ++i) {
      const int l = nearest_int(iscale * x[i]);
      L[i] = static_cast<std::int8_t>(nmax + std::max(-nmax, std::min(nmax - 1, l)));
    }
    return 1 / iscale;
  }
  bool return_early = false;
  if (rmse_type < 0) {
    rmse_type = -rmse_type;
    return_early = true;
  }
  auto weight_at = [&](int i) {
    const float xi = x[i];
    return qw ? qw[i] : rmse_type == 1 ? xi * xi : rmse_type == 2 ? 1.0f : rmse_type == 3 ? std::fabs(xi) : std::sqrt(std::fabs(xi));
  };
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = std::max(-nmax, std::min(nmax - 1, l));
    L[i] = static_cast<std::int8_t>(l + nmax);
    const float w = weight_at(i);
    const float lf = static_cast<float>(l);
    sumlx += w * x[i] * lf;
    suml2 += w * lf * lf;
  }
  float scale = suml2 != 0.0f ? sumlx / suml2 : 0.0f;
  if (return_early) return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is) {
    if (is == 0) continue;
    iscale = -(static_cast<float>(nmax) + 0.1f * static_cast<float>(is)) / max;
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = std::max(-nmax, std::min(nmax - 1, l));
      const float w = weight_at(i);
      const float lf = static_cast<float>(l);
      sumlx += w * x[i] * lf;
      suml2 += w * lf * lf;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2) {
      for (int i = 0; i < n; ++i) {
        const int l = nearest_int(iscale * x[i]);
        L[i] = static_cast<std::int8_t>(nmax + std::max(-nmax, std::min(nmax - 1, l)));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

// The 12-byte scale and min field of Q4_K: six bits each for eight sub-blocks.
void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

}  // namespace detail

namespace {
using namespace detail;
}  // namespace

// ---- scalar conversions

float fp16_to_fp32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t man = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;  // signed zero
    } else {           // subnormal half: normalize into a normal float
      std::uint32_t e = 113;  // 127 - 15 + 1
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --e;
      }
      man &= 0x3FFu;
      bits = sign | (e << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
    if (man != 0) bits |= 0x00400000u;  // NaN comes out quiet, as the reference's arithmetic path leaves it
  } else {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  }
  return std::bit_cast<float>(bits);
}

std::uint16_t fp32_to_fp16(float f) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t abs = bits & 0x7FFFFFFFu;
  if (abs > 0x7F800000u) return static_cast<std::uint16_t>(sign | 0x7E00u);  // NaN: the quiet NaN of the reference
  if (abs >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);  // >= 65520 (and inf) rounds to inf
  const std::uint32_t e = abs >> 23;
  if (abs < 0x38800000u) {  // below the smallest normal half (2^-14): a subnormal half or zero
    const std::uint32_t m = e == 0 ? (abs & 0x7FFFFFu) : ((abs & 0x7FFFFFu) | 0x800000u);
    const std::uint32_t s = e == 0 ? 126u : 126u - e;  // value = m * 2^(e-150); unit of a subnormal half = 2^-24
    if (s >= 32) return sign;
    std::uint32_t q = m >> s;
    const std::uint32_t rem = m & ((1u << s) - 1u);
    const std::uint32_t half = 1u << (s - 1);
    if (rem > half || (rem == half && (q & 1u))) ++q;
    return static_cast<std::uint16_t>(sign | q);
  }
  std::uint32_t q = ((e - 112u) << 10) | ((abs & 0x7FFFFFu) >> 13);
  const std::uint32_t rem = abs & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) ++q;  // a carry moves into the exponent, as it should
  return static_cast<std::uint16_t>(sign | q);
}

float bf16_to_fp32(std::uint16_t h) { return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16); }

std::uint16_t fp32_to_bf16(float f) {
  const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<std::uint16_t>((u >> 16) | 64u);  // NaN, made quiet
  return static_cast<std::uint16_t>((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

int nearest_int(float x) {
  // Adding 1.5 * 2^23 moves the integer part into the low mantissa bits; the hardware's round-to-nearest-even does
  // the rounding. Identical to the reference for |x| <= 4194303 (its stated range), and identical outside it too.
  const float v = x + 12582912.f;
  const auto i = std::bit_cast<std::int32_t>(v);
  return (i & 0x007FFFFF) - 0x00400000;
}

// ---- Q8_0 and Q8_1

void quantize_q8_0(const float* x, std::uint8_t* out, std::int64_t n) {
  const std::int64_t nb = n / kQK8;
  for (std::int64_t i = 0; i < nb; ++i, x += kQK8, out += kQ8_0Bytes) {
    float amax = 0.0f;
    for (int j = 0; j < kQK8; ++j) amax = std::max(amax, std::fabs(x[j]));  // MAX(a, b) = a > b ? a : b
    const float d = amax / static_cast<float>((1 << 7) - 1);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_u16(out, fp32_to_fp16(d));
    for (int j = 0; j < kQK8; ++j) {
      const float x0 = x[j] * id;
      const auto q = static_cast<std::int8_t>(std::round(x0));
      std::memcpy(out + 2 + j, &q, 1);
    }
  }
}

void quantize_q8_1(const float* x, std::uint8_t* out, std::int64_t n) {
  const std::int64_t nb = n / kQK8;
  for (std::int64_t i = 0; i < nb; ++i, x += kQK8, out += kQ8_1Bytes) {
    float amax = 0.0f;
    for (int j = 0; j < kQK8; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float d = amax / static_cast<float>((1 << 7) - 1);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_u16(out, fp32_to_fp16(d));
    int sum = 0;
    for (int j = 0; j < kQK8 / 2; ++j) {
      const float v0 = x[j] * id;
      const float v1 = x[kQK8 / 2 + j] * id;
      const auto q0 = static_cast<std::int8_t>(std::round(v0));
      const auto q1 = static_cast<std::int8_t>(std::round(v1));
      std::memcpy(out + 4 + j, &q0, 1);
      std::memcpy(out + 4 + kQK8 / 2 + j, &q1, 1);
      sum += q0;
      sum += q1;
    }
    put_u16(out + 2, fp32_to_fp16(static_cast<float>(sum) * d));
  }
}

// ---- Q8_Z (Pelias-only; provenance: original)
float q8_z_scale(float d, int lv) { return (d * static_cast<float>(lv)) / 63.0f; }

void quantize_q8_z(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / 256; ++i, x += 256, out += kQ8ZBytes) {
    float amax[8], top = 0.0f;
    for (int j = 0; j < 8; ++j) {
      amax[j] = 0.0f;
      for (int k = 0; k < 32; ++k) amax[j] = std::max(amax[j], std::fabs(x[32 * j + k]));
      top = std::max(top, amax[j]);
    }
    const std::uint16_t dh = fp32_to_fp16(top / 127.0f);
    const float d = fp16_to_fp32(dh);
    put_u16(out, dh);
    std::uint64_t packed = 0;
    for (int j = 0; j < 8; ++j) {
      const float* xb = x + 32 * j;
      int best = 63;
      if (d > 0.0f && amax[j] > 0.0f) {
        // the level that just covers the block's largest value, and every level down to 70 % of it
        const float ideal = amax[j] / 127.0f / d * 63.0f;
        const int hi = std::min(63, std::max(1, static_cast<int>(std::ceil(ideal))));
        const int lo = std::max(1, static_cast<int>(std::floor(0.70f * ideal)));
        float best_err = -1.0f;
        for (int lv = hi; lv >= lo; --lv) {
          const float s = q8_z_scale(d, lv), inv = 1.0f / s;
          float err = 0.0f;
          for (int k = 0; k < 32; ++k) {
            const float q = std::min(127.0f, std::max(-127.0f, std::nearbyint(xb[k] * inv)));
            const float e = xb[k] - q * s;
            err += e * e;
          }
          if (best_err < 0.0f || err < best_err) { best_err = err; best = lv; }
        }
      }
      packed |= static_cast<std::uint64_t>(best) << (6 * j);
      const float s = q8_z_scale(d, best), inv = s > 0.0f ? 1.0f / s : 0.0f;
      for (int k = 0; k < 32; ++k) {
        const auto q = static_cast<std::int8_t>(std::min(127.0f, std::max(-127.0f, std::nearbyint(xb[k] * inv))));
        std::memcpy(out + 8 + 32 * j + k, &q, 1);
      }
    }
    for (int b = 0; b < 6; ++b) out[2 + b] = static_cast<std::uint8_t>(packed >> (8 * b));
  }
}

void dequantize_q8_z(const std::uint8_t* in, float* y, std::int64_t n) {
  for (std::int64_t i = 0; i < n / 256; ++i, in += kQ8ZBytes, y += 256) {
    const float d = fp16_to_fp32(get_u16(in));
    std::uint64_t packed = 0;
    for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(in[2 + b]) << (8 * b);
    for (int j = 0; j < 8; ++j) {
      const float s = q8_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u));
      for (int k = 0; k < 32; ++k) {
        std::int8_t q;
        std::memcpy(&q, in + 8 + 32 * j + k, 1);
        y[32 * j + k] = static_cast<float>(q) * s;
      }
    }
  }
}

// ---- Q7_Z (Pelias-only; provenance: original): Q8_Z's scales with 7-bit values (-63..63), eight values per 7 bytes
float q7_z_scale(float d, int lv) { return (d * static_cast<float>(lv)) / 63.0f; }

void quantize_q7_z(const float* x, std::uint8_t* out, std::int64_t n) {
  constexpr float kMax = 63.0f;
  for (std::int64_t i = 0; i < n / 256; ++i, x += 256, out += kQ7ZBytes) {
    float amax[8], top = 0.0f;
    for (int j = 0; j < 8; ++j) {
      amax[j] = 0.0f;
      for (int k = 0; k < 32; ++k) amax[j] = std::max(amax[j], std::fabs(x[32 * j + k]));
      top = std::max(top, amax[j]);
    }
    const std::uint16_t dh = fp32_to_fp16(top / kMax);
    const float d = fp16_to_fp32(dh);
    put_u16(out, dh);
    std::uint64_t packed = 0;
    int qv[256];
    for (int j = 0; j < 8; ++j) {
      const float* xb = x + 32 * j;
      int best = 63;
      if (d > 0.0f && amax[j] > 0.0f) {
        const float ideal = amax[j] / kMax / d * 63.0f;
        const int hi = std::min(63, std::max(1, static_cast<int>(std::ceil(ideal))));
        const int lo = std::max(1, static_cast<int>(std::floor(0.70f * ideal)));
        float best_err = -1.0f;
        for (int lv = hi; lv >= lo; --lv) {
          const float s = q7_z_scale(d, lv), inv = 1.0f / s;
          float err = 0.0f;
          for (int k = 0; k < 32; ++k) {
            const float q = std::min(kMax, std::max(-kMax, std::nearbyint(xb[k] * inv)));
            const float e = xb[k] - q * s;
            err += e * e;
          }
          if (best_err < 0.0f || err < best_err) { best_err = err; best = lv; }
        }
      }
      packed |= static_cast<std::uint64_t>(best) << (6 * j);
      const float s = q7_z_scale(d, best), inv = s > 0.0f ? 1.0f / s : 0.0f;
      for (int k = 0; k < 32; ++k)
        qv[32 * j + k] = static_cast<int>(std::min(kMax, std::max(-kMax, std::nearbyint(xb[k] * inv))));
    }
    for (int b = 0; b < 6; ++b) out[2 + b] = static_cast<std::uint8_t>(packed >> (8 * b));
    for (int g = 0; g < 32; ++g) {  // eight values (offset by 64, 1..127) per 7 bytes, little-endian
      std::uint64_t w = 0;
      for (int k = 0; k < 8; ++k) w |= static_cast<std::uint64_t>(qv[8 * g + k] + 64) << (7 * k);
      for (int b = 0; b < 7; ++b) out[8 + 7 * g + b] = static_cast<std::uint8_t>(w >> (8 * b));
    }
  }
}

void dequantize_q7_z(const std::uint8_t* in, float* y, std::int64_t n) {
  for (std::int64_t i = 0; i < n / 256; ++i, in += kQ7ZBytes, y += 256) {
    const float d = fp16_to_fp32(get_u16(in));
    std::uint64_t packed = 0;
    for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(in[2 + b]) << (8 * b);
    for (int g = 0; g < 32; ++g) {
      std::uint64_t w = 0;
      for (int b = 0; b < 7; ++b) w |= static_cast<std::uint64_t>(in[8 + 7 * g + b]) << (8 * b);
      const float s = q7_z_scale(d, static_cast<int>((packed >> (6 * (g / 4))) & 63u));
      for (int k = 0; k < 8; ++k) y[8 * g + k] = static_cast<float>(static_cast<int>((w >> (7 * k)) & 127u) - 64) * s;
    }
  }
}

// ---- Q7_ZT (Pelias-only; provenance: original): a 7-bit base plane everywhere plus the H bit (the 8th bit) on exactly
// two of each 256-block's eight 32-blocks, the two where H removes the most (importance-weighted) error ("H as a gate").
namespace {
float q7zt_err(const float* x, const float* qw, float s, bool full, int* q) {
  float err = 0.0f;
  const float inv = 1.0f / s;
  for (int k = 0; k < 32; ++k) {
    float v;
    if (full) v = std::min(127.0f, std::max(-127.0f, std::nearbyint(x[k] * inv)));
    else v = std::min(127.0f, std::max(-127.0f, 2.0f * std::floor(x[k] * inv * 0.5f) + 1.0f));  // nearest odd integer
    if (q) q[k] = static_cast<int>(v);
    const float e = x[k] - v * s;
    err += (qw ? qw[k] : 1.0f) * e * e;
  }
  return err;
}
}  // namespace

void q7_zt_block(const float* x, std::uint8_t* out, const float* qw) {
  float amax[8], top = 0.0f;
  for (int j = 0; j < 8; ++j) {
    amax[j] = 0.0f;
    for (int k = 0; k < 32; ++k) amax[j] = std::max(amax[j], std::fabs(x[32 * j + k]));
    top = std::max(top, amax[j]);
  }
  const std::uint16_t dh = fp32_to_fp16(top / 127.0f);
  const float d = fp16_to_fp32(dh);
  put_u16(out, dh);
  int lv_full[8], lv_odd[8];
  float gain[8];
  for (int j = 0; j < 8; ++j) {
    const float* xb = x + 32 * j;
    const float* wb = qw ? qw + 32 * j : nullptr;
    lv_full[j] = lv_odd[j] = 63;
    gain[j] = 0.0f;
    if (!(d > 0.0f && amax[j] > 0.0f)) continue;
    const float ideal = amax[j] / 127.0f / d * 63.0f;
    const int lo = std::max(1, static_cast<int>(std::floor(0.70f * ideal)));
    const int hi = std::min(63, std::max(1, static_cast<int>(std::ceil(1.05f * ideal))));
    // levels by plain error (as Q8_Z and Q7_Z): a level search weighted by the importance matrix over-fits the
    // calibration text (urm_ZT, 2026-10-28: chunks 23-24 at 2-3x Q7_Z's KL); the importance only ranks the gates,
    // tempered to its square root so no input column is starved
    float bf = -1.0f, bo = -1.0f;
    for (int lv = hi; lv >= lo; --lv) {
      const float s = q8_z_scale(d, lv);
      const float ef = q7zt_err(xb, nullptr, s, true, nullptr), eo = q7zt_err(xb, nullptr, s, false, nullptr);
      if (bf < 0.0f || ef < bf) { bf = ef; lv_full[j] = lv; }
      if (bo < 0.0f || eo < bo) { bo = eo; lv_odd[j] = lv; }
    }
    if (wb) {
      float tw[32];
      for (int k = 0; k < 32; ++k) tw[k] = std::sqrt(std::max(0.0f, wb[k]));
      gain[j] = q7zt_err(xb, tw, q8_z_scale(d, lv_odd[j]), false, nullptr) - q7zt_err(xb, tw, q8_z_scale(d, lv_full[j]), true, nullptr);
    } else {
      gain[j] = bo - bf;
    }
  }
  int g1 = 0;
  for (int j = 1; j < 8; ++j) if (gain[j] > gain[g1]) g1 = j;
  int g2 = g1 == 0 ? 1 : 0;
  for (int j = 0; j < 8; ++j) if (j != g1 && gain[j] > gain[g2]) g2 = j;
  const std::uint8_t mask = static_cast<std::uint8_t>((1u << g1) | (1u << g2));
  std::uint64_t packed = 0;
  int u[256];
  std::uint32_t hbits[2] = {0, 0};
  int hslot = 0;
  for (int j = 0; j < 8; ++j) {
    const bool gated = (mask >> j) & 1u;
    const int lv = gated ? lv_full[j] : lv_odd[j];
    packed |= static_cast<std::uint64_t>(lv) << (6 * j);
    int q[32];
    q7zt_err(x + 32 * j, nullptr, q8_z_scale(d, lv), gated, q);
    for (int k = 0; k < 32; ++k) {
      const int v = q[k] + 127;  // gated: 0..254 = 2b + h; ungated (odd q): 0..254 even, = 2b
      u[32 * j + k] = v >> 1;
      if (gated) hbits[hslot] |= static_cast<std::uint32_t>(v & 1) << k;
    }
    if (gated) ++hslot;
  }
  for (int b = 0; b < 6; ++b) out[2 + b] = static_cast<std::uint8_t>(packed >> (8 * b));
  out[8] = mask;
  for (int g = 0; g < 32; ++g) {
    std::uint64_t wv = 0;
    for (int k = 0; k < 8; ++k) wv |= static_cast<std::uint64_t>(u[8 * g + k]) << (7 * k);
    for (int b = 0; b < 7; ++b) out[9 + 7 * g + b] = static_cast<std::uint8_t>(wv >> (8 * b));
  }
  for (int h = 0; h < 2; ++h)
    for (int b = 0; b < 4; ++b) out[233 + 4 * h + b] = static_cast<std::uint8_t>(hbits[h] >> (8 * b));
}

// ---- Q4_TG (Pelias-only; provenance: original; author's bit-tag matrix at 4 bits, C4, run E)
namespace {
struct Q4TGGrids {
  float g[4][16];
  Q4TGGrids() {
    for (int i = 0; i < 16; ++i) g[0][i] = (static_cast<float>(i) - 7.5f) / 7.5f;
    const double ratios[2] = {1.3, 1.45};
    for (int t = 0; t < 2; ++t) {
      const double top = std::pow(ratios[t], 7.0);
      for (int m = 0; m < 8; ++m) {
        const float v = static_cast<float>(std::pow(ratios[t], static_cast<double>(m)) / top);
        g[1 + t][8 + m] = v;
        g[1 + t][7 - m] = -v;
      }
    }
    for (int i = 0; i < 16; ++i) g[3][i] = 0.6f * g[0][i];
    // v2 (Q4_TG2; author's 0-weight idea, sandbox q4tg_zero_level_check.py "uniform + log with 0"): grid 0 = (i - 7) / 8,
    // levels -7/8..1 with an exact 0; grid 1 = the 1.3 log grid with its smallest positive level replaced by 0; grid 2 as
    // v1; grid 3 = 0.6 x grid 0
    for (int i = 0; i < 16; ++i) h[0][i] = (static_cast<float>(i) - 7.0f) / 8.0f;
    for (int i = 0; i < 16; ++i) h[1][i] = g[1][i];
    h[1][8] = 0.0f;
    for (int i = 0; i < 16; ++i) h[2][i] = g[2][i];
    for (int i = 0; i < 16; ++i) h[3][i] = 0.6f * h[0][i];
  }
  float h[4][16];
};
const Q4TGGrids kQ4TG;
inline int q4tg_nearest(const float* g, float v) {  // g ascending, 16 levels
  int lo = 0, hi = 15;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (g[mid] <= v) lo = mid; else hi = mid;
  }
  return (v - g[lo] <= g[hi] - v) ? lo : hi;
}
// one group of 8 values at scale s: the best tag, its codes and its error
float q4tg_group(const float* x, const float* qw, float s, int* tag, std::uint8_t* code, const float (*grids)[16]) {
  float best = std::numeric_limits<float>::infinity();
  const float inv = s > 0.0f ? 1.0f / s : 0.0f;
  for (int t = 0; t < 4; ++t) {
    float e = 0.0f;
    std::uint8_t c[8];
    for (int k = 0; k < 8; ++k) {
      c[k] = static_cast<std::uint8_t>(q4tg_nearest(grids[t], x[k] * inv));
      const float d = x[k] - s * grids[t][c[k]];
      e += (qw ? qw[k] : 1.0f) * d * d;
    }
    if (e < best) {
      best = e;
      if (tag) *tag = t;
      if (code) std::memcpy(code, c, 8);
    }
  }
  return best;
}
}  // namespace

const float* q4_tg_grid(int tag, int version) { return version == 2 ? kQ4TG.h[tag & 3] : kQ4TG.g[tag & 3]; }

void q4_tg_block(const float* x, std::uint8_t* out, const float* qw, int version) {
  const float (*grids)[16] = version == 2 ? kQ4TG.h : kQ4TG.g;
  static const float kF[19] = {0.35f, 0.4f, 0.45f, 0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f, 1.0f, 1.05f, 1.1f, 1.15f, 1.2f, 1.25f};
  float sbest[8], top = 0.0f;
  for (int j = 0; j < 8; ++j) {
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(x[32 * j + k]));
    float be = std::numeric_limits<float>::infinity();
    sbest[j] = 0.0f;
    if (amax > 0.0f)
      for (float f : kF) {
        const float s = amax * f;
        float e = 0.0f;
        for (int g = 0; g < 4; ++g) e += q4tg_group(x + 32 * j + 8 * g, qw ? qw + 32 * j + 8 * g : nullptr, s, nullptr, nullptr, grids);
        if (e < be) { be = e; sbest[j] = s; }
      }
    top = std::max(top, sbest[j]);
  }
  const std::uint16_t dh = fp32_to_fp16(top);
  const float d = fp16_to_fp32(dh);
  std::memset(out, 0, 144);
  out[0] = static_cast<std::uint8_t>(dh & 0xff);
  out[1] = static_cast<std::uint8_t>(dh >> 8);
  std::uint64_t packed = 0;
  for (int j = 0; j < 8; ++j) {
    int lv = d > 0.0f ? static_cast<int>(std::lrint(63.0f * sbest[j] / d)) : 1;
    lv = std::max(1, std::min(63, lv));
    packed |= static_cast<std::uint64_t>(lv) << (6 * j);
    const float s = q8_z_scale(d, lv);
    std::uint8_t tagbyte = 0;
    for (int g = 0; g < 4; ++g) {
      int t = 0;
      std::uint8_t c[8];
      q4tg_group(x + 32 * j + 8 * g, qw ? qw + 32 * j + 8 * g : nullptr, s, &t, c, grids);
      tagbyte = static_cast<std::uint8_t>(tagbyte | (t << (2 * g)));
      for (int k = 0; k < 8; ++k) {
        const int i = 32 * j + 8 * g + k;
        out[16 + i / 2] = static_cast<std::uint8_t>(out[16 + i / 2] | (c[k] << (4 * (i & 1))));
      }
    }
    out[8 + j] = tagbyte;
  }
  for (int b = 0; b < 6; ++b) out[2 + b] = static_cast<std::uint8_t>(packed >> (8 * b));
}

void quantize_q4_tg(const float* x, std::uint8_t* out, std::int64_t n, int version) {
  for (std::int64_t i = 0; i < n / 256; ++i, x += 256, out += 144) q4_tg_block(x, out, nullptr, version);
}

void dequantize_q4_tg(const std::uint8_t* in, float* y, std::int64_t n, int version) {
  const float (*grids)[16] = version == 2 ? kQ4TG.h : kQ4TG.g;
  for (std::int64_t i = 0; i < n / 256; ++i, in += 144, y += 256) {
    const float d = fp16_to_fp32(get_u16(in));
    std::uint64_t packed = 0;
    for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(in[2 + b]) << (8 * b);
    for (int j = 0; j < 8; ++j) {
      const float s = q8_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u));
      for (int g = 0; g < 4; ++g) {
        const float* gr = grids[(in[8 + j] >> (2 * g)) & 3];
        for (int k = 0; k < 8; ++k) {
          const int idx = 32 * j + 8 * g + k;
          y[idx] = s * gr[(in[16 + idx / 2] >> (4 * (idx & 1))) & 15];
        }
      }
    }
  }
}

void quantize_q7_zt(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / 256; ++i, x += 256, out += kQ7ZTBytes) q7_zt_block(x, out, nullptr);
}

void dequantize_q7_zt(const std::uint8_t* in, float* y, std::int64_t n) {
  for (std::int64_t i = 0; i < n / 256; ++i, in += kQ7ZTBytes, y += 256) {
    const float d = fp16_to_fp32(get_u16(in));
    std::uint64_t packed = 0;
    for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(in[2 + b]) << (8 * b);
    const std::uint8_t mask = in[8];
    int hslot = 0;
    for (int j = 0; j < 8; ++j) {
      const float s = q8_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u));
      // the format has exactly two gated blocks; a corrupt mask with more bits set reads no H word past the block's two
      const bool gated = ((mask >> j) & 1u) && hslot < 2;
      std::uint32_t hb = 0;
      if (gated) {
        for (int b = 0; b < 4; ++b) hb |= static_cast<std::uint32_t>(in[233 + 4 * hslot + b]) << (8 * b);
        ++hslot;
      }
      for (int g = 0; g < 4; ++g) {
        std::uint64_t wv = 0;
        for (int b = 0; b < 7; ++b) wv |= static_cast<std::uint64_t>(in[9 + 7 * (4 * j + g) + b]) << (8 * b);
        for (int k = 0; k < 8; ++k) {
          const int bb = static_cast<int>((wv >> (7 * k)) & 127u), idx = 8 * g + k;
          const int q = gated ? 2 * bb + static_cast<int>((hb >> idx) & 1u) - 127 : 2 * bb - 127;
          y[32 * j + idx] = static_cast<float>(q) * s;
        }
      }
    }
  }
}

void dequantize_q8_0(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQK8;
  for (std::int64_t i = 0; i < nb; ++i, in += kQ8_0Bytes, y += kQK8) {
    const float d = fp16_to_fp32(get_u16(in));
    for (int j = 0; j < kQK8; ++j) {
      std::int8_t q;
      std::memcpy(&q, in + 2 + j, 1);
      y[j] = static_cast<float>(q) * d;
    }
  }
}

// ---- Q4_K

void quantize_q4_k(const float* x, std::uint8_t* out, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  std::uint8_t L[kQKK];
  std::uint8_t Laux[32];
  float weights[32];
  float mins[kQKK / 32];
  float scales[kQKK / 32];
  for (std::int64_t i = 0; i < nb; ++i, x += kQKK, out += kQ4KBytes) {
    std::uint8_t sc12[12] = {};
    float max_scale = 0;  // the min is subtracted, so scales are never negative
    float max_min = 0;
    for (int j = 0; j < kQKK / 32; ++j) {
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
      const float av_x = std::sqrt(sum_x2 / 32);
      for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
      const float scale = scales[j];
      if (scale > max_scale) max_scale = scale;
      const float min = mins[j];
      if (min > max_min) max_min = min;
    }
    const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < kQKK / 32; ++j) {
      auto ls = static_cast<std::uint8_t>(nearest_int(inv_scale * scales[j]));  // the reference narrows, then clamps
      auto lm = static_cast<std::uint8_t>(nearest_int(inv_min * mins[j]));
      ls = std::min<std::uint8_t>(63, ls);
      lm = std::min<std::uint8_t>(63, lm);
      if (j < 4) {
        sc12[j] = ls;
        sc12[j + 4] = lm;
      } else {
        sc12[j + 4] = static_cast<std::uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
        sc12[j - 4] = static_cast<std::uint8_t>(sc12[j - 4] | ((ls >> 4) << 6));
        sc12[j - 0] = static_cast<std::uint8_t>(sc12[j - 0] | ((lm >> 4) << 6));
      }
    }
    const std::uint16_t d16 = fp32_to_fp16(max_scale / 63.f);
    const std::uint16_t dmin16 = fp32_to_fp16(max_min / 63.f);
    for (int j = 0; j < kQKK / 32; ++j) {
      std::uint8_t sc, m;
      get_scale_min_k4(j, sc12, &sc, &m);
      const float d = fp16_to_fp32(d16) * static_cast<float>(sc);
      if (d == 0.0f) continue;
      const float dm = fp16_to_fp32(dmin16) * static_cast<float>(m);
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = std::max(0, std::min(15, l));
        L[32 * j + ii] = static_cast<std::uint8_t>(l);
      }
    }
    put_u16(out, d16);
    put_u16(out + 2, dmin16);
    std::memcpy(out + 4, sc12, 12);
    std::uint8_t* q = out + 16;
    for (int j = 0; j < kQKK; j += 64) {
      for (int l = 0; l < 32; ++l) q[l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 4));
      q += 32;
    }
  }
}

void dequantize_q4_k(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  for (std::int64_t i = 0; i < nb; ++i, in += kQ4KBytes) {
    const float d = fp16_to_fp32(get_u16(in));
    const float min = fp16_to_fp32(get_u16(in + 2));
    const std::uint8_t* scales = in + 4;
    const std::uint8_t* q = in + 16;
    int is = 0;
    std::uint8_t sc, m;
    for (int j = 0; j < kQKK; j += 64) {
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min * static_cast<float>(m);
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = min * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// ---- Q6_K

void quantize_q6_k(const float* x, std::uint8_t* out, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  std::int8_t L[kQKK];
  float scales[kQKK / 16];
  for (std::int64_t i = 0; i < nb; ++i, x += kQKK, out += kQ6KBytes) {
    float max_scale = 0;
    float max_abs_scale = 0;
    for (int ib = 0; ib < kQKK / 16; ++ib) {
      const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
      scales[ib] = scale;
      const float abs_scale = std::fabs(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }
    if (max_abs_scale < kGroupMaxEps) {
      std::memset(out, 0, kQ6KBytes);
      put_u16(out + 208, fp32_to_fp16(0.f));
      continue;
    }
    std::int8_t sc16[16];
    const float iscale = -128.f / max_scale;
    const std::uint16_t d16 = fp32_to_fp16(1 / iscale);
    for (int ib = 0; ib < kQKK / 16; ++ib) sc16[ib] = static_cast<std::int8_t>(std::min(127, nearest_int(iscale * scales[ib])));
    for (int j = 0; j < kQKK / 16; ++j) {
      const float d = fp16_to_fp32(d16) * static_cast<float>(sc16[j]);
      if (d == 0.0f) continue;
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = std::max(-32, std::min(31, l));
        L[16 * j + ii] = static_cast<std::int8_t>(l + 32);
      }
    }
    std::uint8_t* ql = out;
    std::uint8_t* qh = out + 128;
    for (int j = 0; j < kQKK; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const auto u = [&](int k) { return static_cast<std::uint8_t>(L[j + l + k]); };
        const auto q1 = static_cast<std::uint8_t>(u(0) & 0xF);
        const auto q2 = static_cast<std::uint8_t>(u(32) & 0xF);
        const auto q3 = static_cast<std::uint8_t>(u(64) & 0xF);
        const auto q4 = static_cast<std::uint8_t>(u(96) & 0xF);
        ql[l + 0] = static_cast<std::uint8_t>(q1 | (q3 << 4));
        ql[l + 32] = static_cast<std::uint8_t>(q2 | (q4 << 4));
        qh[l] = static_cast<std::uint8_t>((u(0) >> 4) | ((u(32) >> 4) << 2) | ((u(64) >> 4) << 4) | ((u(96) >> 4) << 6));
      }
      ql += 64;
      qh += 32;
    }
    std::memcpy(out + 192, sc16, 16);
    put_u16(out + 208, d16);
  }
}

void dequantize_q6_k(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  for (std::int64_t i = 0; i < nb; ++i, in += kQ6KBytes) {
    const float d = fp16_to_fp32(get_u16(in + 208));
    const std::uint8_t* ql = in;
    const std::uint8_t* qh = in + 128;
    std::int8_t scv[16];
    std::memcpy(scv, in + 192, 16);
    const std::int8_t* sc = scv;
    for (int k = 0; k < kQKK; k += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const auto q1 = static_cast<std::int8_t>(((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        const auto q2 = static_cast<std::int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        const auto q3 = static_cast<std::int8_t>(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        const auto q4 = static_cast<std::int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// ---- Q8_K (activations on the CPU path)

void quantize_q8_k(const float* x, std::uint8_t* out, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  for (std::int64_t i = 0; i < nb; ++i, x += kQKK, out += kQ8KBytes) {
    float max = 0;
    float amax = 0;
    for (int j = 0; j < kQKK; ++j) {
      const float ax = std::fabs(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (amax == 0.0f) {
      std::memset(out, 0, kQ8KBytes);  // d = 0, qs = 0, and the sums of zeros are zero
      continue;
    }
    const float iscale = -127.f / max;
    std::int8_t qs[kQKK];
    for (int j = 0; j < kQKK; ++j) {
      const int v = nearest_int(iscale * x[j]);
      qs[j] = static_cast<std::int8_t>(std::min(127, v));
    }
    std::int16_t bsums[kQKK / 16];
    for (int j = 0; j < kQKK / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
      bsums[j] = static_cast<std::int16_t>(sum);
    }
    const float d = 1 / iscale;
    std::memcpy(out, &d, 4);
    std::memcpy(out + 4, qs, kQKK);
    std::memcpy(out + 4 + kQKK, bsums, sizeof bsums);
  }
}

// ---- dispatch

// ---- read-only types (no quantizer yet): Q2_K, Q3_K, Q5_K, IQ4_XS

// Q2_K block (84 bytes): scales[16] (low nibble scale, high nibble min), qs[64] (2-bit quants), d, dmin (fp16)
void dequantize_q2_k(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  for (std::int64_t i = 0; i < nb; ++i, in += kQ2KBytes) {
    const std::uint8_t* scales = in;
    const std::uint8_t* q = in + 16;
    const float d = fp16_to_fp32(get_u16(in + 80));
    const float min = fp16_to_fp32(get_u16(in + 82));
    int is = 0;
    for (int n0 = 0; n0 < kQKK; n0 += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        std::uint8_t sc = scales[is++];
        float dl = d * static_cast<float>(sc & 0xF);
        float ml = min * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
        sc = scales[is++];
        dl = d * static_cast<float>(sc & 0xF);
        ml = min * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

// Q3_K block (110 bytes): hmask[32] (high bit), qs[64] (low 2 bits), scales[12] (sixteen 6-bit scales, offset 32), d
void dequantize_q3_k(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  constexpr std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  for (std::int64_t i = 0; i < nb; ++i, in += kQ3KBytes) {
    const std::uint8_t* hm = in;
    const std::uint8_t* q = in + 32;
    const float d_all = fp16_to_fp32(get_u16(in + 108));
    std::uint32_t aux[4];
    std::memcpy(aux, in + 96, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::int8_t scales[16];
    std::memcpy(scales, aux, 16);  // the reference reads the words as bytes in memory order (little-endian)
    int is = 0;
    std::uint8_t m = 1;
    for (int n0 = 0; n0 < kQKK; n0 += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<float>(static_cast<int>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<float>(static_cast<int>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        shift += 2;
        m = static_cast<std::uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

// Q5_K block (176 bytes): d, dmin (fp16), scales[12] (as Q4_K), qh[32] (fifth bit), qs[128] (low 4 bits)
void dequantize_q5_k(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  for (std::int64_t i = 0; i < nb; ++i, in += kQ5KBytes) {
    const float d = fp16_to_fp32(get_u16(in));
    const float min = fp16_to_fp32(get_u16(in + 2));
    const std::uint8_t* scales = in + 4;
    const std::uint8_t* qh = in + 16;
    const std::uint8_t* ql = in + 48;
    int is = 0;
    std::uint8_t sc, m;
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < kQKK; j += 64) {
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min * static_cast<float>(m);
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = min * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 = static_cast<std::uint8_t>(u1 << 2);
      u2 = static_cast<std::uint8_t>(u2 << 2);
    }
  }
}

// IQ4_XS block (136 bytes): d (fp16), scales_h (u16), scales_l[4], qs[128]; eight 32-element groups, 6-bit scales
// offset 32, levels from the non-linear table
void dequantize_iq4_xs(const std::uint8_t* in, float* y, std::int64_t n) {
  const std::int64_t nb = n / kQKK;
  for (std::int64_t i = 0; i < nb; ++i, in += kIQ4XSBytes) {
    const float d = fp16_to_fp32(get_u16(in));
    const std::uint16_t scales_h = get_u16(in + 2);
    const std::uint8_t* scales_l = in + 4;
    const std::uint8_t* qs = in + 8;
    for (int ib = 0; ib < kQKK / 32; ++ib) {
      const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
      const float dl = d * static_cast<float>(ls - 32);
      for (int j = 0; j < 16; ++j) {
        y[j + 0] = dl * static_cast<float>(kValuesIq4nl[qs[j] & 0xf]);
        y[j + 16] = dl * static_cast<float>(kValuesIq4nl[qs[j] >> 4]);
      }
      y += 32;
      qs += 16;
    }
  }
}

bool can_quantize(Type t) {
  switch (t) {
    case Type::f32: case Type::f16: case Type::bf16: case Type::q8_0: case Type::q8_1: case Type::q4_k: case Type::q6_k:
    case Type::q8_k: case Type::q2_k: case Type::q3_k: case Type::q5_k: case Type::iq4_nl: case Type::iq4_xs: case Type::q8_z:
    case Type::q7_z: case Type::q7_zt: case Type::q4_tg: case Type::q4_tg2:
      return true;
    default:
      return false;
  }
}

bool can_dequantize(Type t) {
  switch (t) {
    case Type::f32: case Type::f16: case Type::bf16: case Type::q8_0: case Type::q4_k: case Type::q6_k:
    case Type::q2_k: case Type::q3_k: case Type::q5_k: case Type::iq4_xs:
    case Type::q4_0: case Type::q4_1: case Type::q5_0: case Type::q5_1: case Type::iq4_nl: case Type::mxfp4:
    case Type::tq1_0: case Type::tq2_0: case Type::iq2_xxs: case Type::iq2_xs: case Type::iq2_s: case Type::iq3_xxs:
    case Type::iq3_s: case Type::iq1_s: case Type::iq1_m: case Type::q8_z: case Type::q7_z: case Type::q7_zt: case Type::q4_tg: case Type::q4_tg2:
    case Type::i2_s: case Type::nvfp4:
      return true;
    default:
      return false;
  }
}

namespace {
Status check_row(Type t, std::int64_t n, bool quantize) {
  const TypeInfo ti = type_info(t);
  if (!(quantize ? can_quantize(t) : can_dequantize(t)))
    return Status::fail(Code::unsupported, std::string(quantize ? "quantizing to " : "reading ") + ti.name + " is not built yet");
  if (n < 0 || n % ti.block_elements != 0)
    return Status::fail(Code::invalid_argument, "row of " + std::to_string(n) + " elements is not a whole number of " + ti.name + " blocks");
  return Status::good();
}
}  // namespace

void hadamard32(float* x, std::int64_t n) {
  const float norm = 1.0f / std::sqrt(32.0f);
  for (std::int64_t b = 0; b + 32 <= n; b += 32) {
    float* v = x + b;
    for (int h = 1; h < 32; h *= 2)
      for (int i = 0; i < 32; i += 2 * h)
        for (int j = i; j < i + h; ++j) {
          const float a = v[j], d = v[j + h];
          v[j] = a + d;
          v[j + h] = a - d;
        }
    for (int j = 0; j < 32; ++j) v[j] *= norm;
  }
}

Status quantize_row(Type t, const float* x, std::uint8_t* out, std::int64_t n) {
  const Status s = check_row(t, n, true);
  if (!s) return s;
  switch (t) {
    case Type::f32: std::memcpy(out, x, static_cast<std::size_t>(n) * 4); break;
    case Type::f16:
      for (std::int64_t i = 0; i < n; ++i) put_u16(out + 2 * i, fp32_to_fp16(x[i]));
      break;
    case Type::bf16:
      for (std::int64_t i = 0; i < n; ++i) put_u16(out + 2 * i, fp32_to_bf16(x[i]));
      break;
    case Type::q8_0: quantize_q8_0(x, out, n); break;
    case Type::q8_1: quantize_q8_1(x, out, n); break;
    case Type::q4_k: quantize_q4_k(x, out, n); break;
    case Type::q6_k: quantize_q6_k(x, out, n); break;
    case Type::q8_k: quantize_q8_k(x, out, n); break;
    case Type::q2_k: quantize_q2_k(x, out, n); break;
    case Type::q3_k: quantize_q3_k(x, out, n); break;
    case Type::q5_k: quantize_q5_k(x, out, n); break;
    case Type::iq4_nl: quantize_iq4_nl(x, out, n); break;
    case Type::iq4_xs: quantize_iq4_xs(x, out, n); break;
    case Type::q8_z: quantize_q8_z(x, out, n); break;
    case Type::q7_z: quantize_q7_z(x, out, n); break;
    case Type::q7_zt: quantize_q7_zt(x, out, n); break;
    case Type::q4_tg: quantize_q4_tg(x, out, n, 1); break;
    case Type::q4_tg2: quantize_q4_tg(x, out, n, 2); break;
    default: return Status::fail(Code::internal, "quantize dispatch");
  }
  return Status::good();
}

Status dequantize_row(Type t, const std::uint8_t* in, float* y, std::int64_t n) {
  const Status s = check_row(t, n, false);
  if (!s) return s;
  switch (t) {
    case Type::f32: std::memcpy(y, in, static_cast<std::size_t>(n) * 4); break;
    case Type::f16:
      for (std::int64_t i = 0; i < n; ++i) y[i] = fp16_to_fp32(get_u16(in + 2 * i));
      break;
    case Type::bf16:
      for (std::int64_t i = 0; i < n; ++i) y[i] = bf16_to_fp32(get_u16(in + 2 * i));
      break;
    case Type::q8_0: dequantize_q8_0(in, y, n); break;
    case Type::q4_k: dequantize_q4_k(in, y, n); break;
    case Type::q6_k: dequantize_q6_k(in, y, n); break;
    case Type::q2_k: dequantize_q2_k(in, y, n); break;
    case Type::q3_k: dequantize_q3_k(in, y, n); break;
    case Type::q5_k: dequantize_q5_k(in, y, n); break;
    case Type::iq4_xs: dequantize_iq4_xs(in, y, n); break;
    case Type::q8_z: dequantize_q8_z(in, y, n); break;
    case Type::q7_z: dequantize_q7_z(in, y, n); break;
    case Type::q7_zt: dequantize_q7_zt(in, y, n); break;
    case Type::q4_tg: dequantize_q4_tg(in, y, n, 1); break;
    case Type::q4_tg2: dequantize_q4_tg(in, y, n, 2); break;
    case Type::q4_0: dequantize_q4_0(in, y, n); break;
    case Type::q4_1: dequantize_q4_1(in, y, n); break;
    case Type::q5_0: dequantize_q5_0(in, y, n); break;
    case Type::q5_1: dequantize_q5_1(in, y, n); break;
    case Type::iq4_nl: dequantize_iq4_nl(in, y, n); break;
    case Type::mxfp4: dequantize_mxfp4(in, y, n); break;
    case Type::tq1_0: dequantize_tq1_0(in, y, n); break;
    case Type::tq2_0: dequantize_tq2_0(in, y, n); break;
    case Type::i2_s: dequantize_i2_s(in, y, n); break;
    case Type::nvfp4: dequantize_nvfp4(in, y, n); break;
    case Type::iq2_xxs: dequantize_iq2_xxs(in, y, n); break;
    case Type::iq2_xs: dequantize_iq2_xs(in, y, n); break;
    case Type::iq2_s: dequantize_iq2_s(in, y, n); break;
    case Type::iq3_xxs: dequantize_iq3_xxs(in, y, n); break;
    case Type::iq3_s: dequantize_iq3_s(in, y, n); break;
    case Type::iq1_s: dequantize_iq1_s(in, y, n); break;
    case Type::iq1_m: dequantize_iq1_m(in, y, n); break;
    default: return Status::fail(Code::internal, "dequantize dispatch");
  }
  return Status::good();
}

}  // namespace pelias::quant
